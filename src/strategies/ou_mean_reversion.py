"""OU Mean Reversion: VWAP mean reversion + vol-targeted dynamic leverage.

This strategy pairs the validated MR Turbo / MR Macro signal base (session
filter + fixed SL/TP + EOD/td exit) with a dynamic leverage overlay that
targets a given volatility level. The leverage per bar is:

    leverage(t) = min(sigma_target / rolling_vol(t), max_leverage)

The earlier signal-function-based design (with VWAP exit and no session
filter) has been **deprecated** because it generated ~13k trades with
negative edge on EUR-USD. This rewrite delegates the entry/exit logic
to the same pre-computed Bollinger-band signals used by
``backtest_mr_turbo`` and only keeps vol-targeted leverage as the
distinguishing feature.

See also
--------
- :func:`strategies.mr_turbo.backtest_mr_turbo` — base signal (fixed leverage=1)
- :func:`strategies.mr_macro.backtest_mr_macro` — adds macro regime filter
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import vectorbtpro as vbt
from numba import njit

from framework.spec import (
    IndicatorSpec,
    OverlayLine,
    ParamDef,
    PlotConfig,
    PortfolioConfig,
    StrategySpec,
)
from utils import (
    compute_daily_rolling_volatility_nb,
    compute_leverage_nb,
    compute_mr_bands_nb,
    mr_band_signal_nb,
    prepare_mr,
)


# ═══════════════════════════════════════════════════════════════════════
# STANDALONE BACKTEST — mr_turbo signals + vol-targeted leverage overlay
# ═══════════════════════════════════════════════════════════════════════


def backtest_ou_mr(
    data: vbt.Data,
    bb_window: int = 80,
    bb_alpha: float = 5.0,
    sl_stop: float = 0.005,
    tp_stop: float = 0.006,
    session_start: int = 6,
    session_end: int = 14,
    dt_stop: str = "21:00",
    td_stop: str = "6h",
    vol_window: int = 20,
    sigma_target: float = 0.10,
    max_leverage: float = 3.0,
    leverage_mult: float = 1.0,
    init_cash: float = 1_000_000.0,
    slippage: float = 0.00015,
) -> vbt.Portfolio:
    """VWAP mean reversion with vol-targeted dynamic leverage.

    Pipeline:
    1. Native VBT VWAP (daily anchored)
    2. Bollinger bands on close - VWAP deviation (bb_window, bb_alpha)
    3. Session filter (only trade between session_start and session_end)
    4. Pre-computed boolean entries / short_entries
    5. Fixed SL / TP / EOD / td_stop
    6. Vol-targeted leverage array based on 20-day close-to-close vol

    Parameters
    ----------
    sigma_target
        Target annualized volatility (e.g. 0.10 = 10%/y). The rolling
        leverage is ``min(sigma_target / realized_vol, max_leverage)``.
    max_leverage
        Cap on the dynamic leverage multiplier.
    leverage_mult
        Scalar multiplier applied on top of the dynamic leverage array —
        allows overriding the overall exposure for sweeps.
    """
    close = data.close

    # Native VBT VWAP (daily session anchor)
    vwap = vbt.VWAP.run(data.high, data.low, close, data.volume, anchor="D").vwap

    # Bollinger bands on close - VWAP deviation
    deviation = close - vwap
    bb = vbt.BBANDS.run(deviation, window=bb_window, alpha=bb_alpha)
    upper = vwap + bb.upper
    lower = vwap + bb.lower

    # Session filter
    hours = close.index.hour
    session = (hours >= session_start) & (hours < session_end)

    # Pre-computed boolean entries
    entries = (close < lower) & session
    short_entries = (close > upper) & session

    # Vol-targeted dynamic leverage array (Numba-compiled helpers)
    index_ns = vbt.dt.to_ns(close.index)
    close_vals = np.ascontiguousarray(close.values, dtype=np.float64)
    rolling_vol = compute_daily_rolling_volatility_nb(index_ns, close_vals, vol_window)
    leverage_arr = compute_leverage_nb(rolling_vol, sigma_target, max_leverage)
    leverage_arr = leverage_arr * float(leverage_mult)
    leverage_arr = np.where(np.isnan(leverage_arr), 1.0, leverage_arr)

    return vbt.Portfolio.from_signals(
        data,
        entries=entries,
        exits=False,
        short_entries=short_entries,
        short_exits=False,
        sl_stop=sl_stop,
        tp_stop=tp_stop,
        dt_stop=dt_stop,
        td_stop=td_stop,
        slippage=slippage,
        init_cash=init_cash,
        leverage=leverage_arr,
        freq="1min",
    )


# ═══════════════════════════════════════════════════════════════════════
# MULTI-PARAM VARIANT (Numba-parallel via VBT native broadcasting)
# ═══════════════════════════════════════════════════════════════════════


def backtest_ou_mr_multi(
    data: vbt.Data,
    bb_window: list | int = 80,
    bb_alpha: list | float = 5.0,
    sl_stop: float = 0.005,
    tp_stop: float = 0.006,
    session_start: int = 6,
    session_end: int = 14,
    dt_stop: str = "21:00",
    td_stop: str = "6h",
    vol_window: int = 20,
    sigma_target: float = 0.10,
    max_leverage: float = 3.0,
    leverage_mult: float = 1.0,
    init_cash: float = 1_000_000.0,
    slippage: float = 0.00015,
    param_product: bool = True,
) -> vbt.Portfolio:
    """Multi-column vol-targeted MR — fully Numba-parallel.

    Multi-column variant of :func:`backtest_ou_mr` accepting lists
    for ``bb_window`` / ``bb_alpha``. Shares the leverage array
    (1D, N_bars) across all columns — VBT broadcasts it under the
    hood.
    """
    close = data.close

    vwap = vbt.VWAP.run(data.high, data.low, close, data.volume, anchor="D").vwap
    deviation = close - vwap

    bb = vbt.BBANDS.run(
        deviation,
        window=bb_window,
        alpha=bb_alpha,
        param_product=param_product,
    )
    upper = vwap.values[:, None] + bb.upper.values
    lower = vwap.values[:, None] + bb.lower.values

    hours = close.index.hour.values
    session_1d = (hours >= session_start) & (hours < session_end)
    session = np.broadcast_to(session_1d[:, None], upper.shape)

    close_2d = np.broadcast_to(close.values[:, None], upper.shape)
    entries = pd.DataFrame(
        (close_2d < lower) & session,
        index=close.index,
        columns=bb.upper.columns,
    )
    short_entries = pd.DataFrame(
        (close_2d > upper) & session,
        index=close.index,
        columns=bb.upper.columns,
    )

    # Leverage array (1D, shared across combos)
    index_ns = vbt.dt.to_ns(close.index)
    close_vals = np.ascontiguousarray(close.values, dtype=np.float64)
    rolling_vol = compute_daily_rolling_volatility_nb(index_ns, close_vals, vol_window)
    leverage_arr = compute_leverage_nb(rolling_vol, sigma_target, max_leverage)
    leverage_arr = leverage_arr * float(leverage_mult)
    leverage_arr = np.where(np.isnan(leverage_arr), 1.0, leverage_arr)

    return vbt.Portfolio.from_signals(
        data,
        entries=entries,
        exits=False,
        short_entries=short_entries,
        short_exits=False,
        sl_stop=sl_stop,
        tp_stop=tp_stop,
        dt_stop=dt_stop,
        td_stop=td_stop,
        slippage=slippage,
        init_cash=init_cash,
        leverage=leverage_arr,
        freq="1min",
        chunked="threadpool",
    )


# ═══════════════════════════════════════════════════════════════════════
# INDICATOR KERNEL (simplified — VWAP & ADX pre-computed natively)
# ═══════════════════════════════════════════════════════════════════════


@njit(nogil=True)
def compute_ou_indicators_nb(
    index_ns: np.ndarray,
    close: np.ndarray,
    vwap: np.ndarray,
    lookback: int,
    band_width: float,
    vol_window: int,
    sigma_target: float,
    max_leverage: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Bands around VWAP + vol-targeted leverage."""
    zscore, upper_band, lower_band = compute_mr_bands_nb(
        index_ns, close, vwap, lookback, band_width
    )
    rolling_vol = compute_daily_rolling_volatility_nb(index_ns, close, vol_window)
    leverage = compute_leverage_nb(rolling_vol, sigma_target, max_leverage)
    return zscore, upper_band, lower_band, leverage


# ═══════════════════════════════════════════════════════════════════════
# PREPARE FUNCTION (native VBT: VWAP + ADX)
# ═══════════════════════════════════════════════════════════════════════


def prepare_ou_mr(raw, data):
    return prepare_mr(raw, data, adx_period=14, adx_threshold=30.0)


# ═══════════════════════════════════════════════════════════════════════
# STRATEGY SPECIFICATION
# ═══════════════════════════════════════════════════════════════════════

spec = StrategySpec(
    name="OU Mean Reversion (Vol-Targeted)",
    indicator=IndicatorSpec(
        class_name="IntradayMR",
        short_name="imr",
        input_names=("index_ns", "close_minute", "vwap"),
        param_names=(
            "lookback",
            "band_width",
            "vol_window",
            "sigma_target",
            "max_leverage",
        ),
        output_names=("zscore", "upper_band", "lower_band", "leverage"),
        kernel_func=compute_ou_indicators_nb,
    ),
    signal_func=mr_band_signal_nb,
    signal_args_map=(
        ("close_arr", "data.close"),
        ("upper_arr", "ind.upper_band"),
        ("lower_arr", "ind.lower_band"),
        ("twap_arr", "pre.vwap"),
        ("regime_ok_arr", "pre.regime_ok"),
        ("index_arr", "extra.index_ns"),
        ("eod_hour", "param.eod_hour"),
        ("eod_minute", "param.eod_minute"),
        ("eval_freq", "param.eval_freq"),
    ),
    params={
        "lookback": ParamDef(60, sweep=[20, 40, 60, 120, 240]),
        "band_width": ParamDef(2.0, sweep=[1.5, 2.0, 2.5, 3.0, 3.5]),
        "vol_window": ParamDef(20),
        "sigma_target": ParamDef(0.01),
        "max_leverage": ParamDef(3.0),
        "eod_hour": ParamDef(21),
        "eod_minute": ParamDef(0),
        "eval_freq": ParamDef(5),
        "sl_stop": ParamDef(0.005, sweep=[0.002, 0.003, 0.005]),
    },
    portfolio_config=PortfolioConfig(leverage="ind.leverage", sl_stop=0.005),
    plot_config=PlotConfig(
        overlays=(
            OverlayLine("pre.vwap", "VWAP", color="#FF9800", dash="dash"),
            OverlayLine("ind.upper_band", "Upper Band", color="#E91E63", dash="dot"),
            OverlayLine("ind.lower_band", "Lower Band", color="#E91E63", dash="dot"),
        ),
    ),
    prepare_fn=prepare_ou_mr,
)


if __name__ == "__main__":
    import argparse
    import sys
    from pathlib import Path as _Path

    # Allow direct execution: `python src/strategies/ou_mean_reversion.py`
    _SRC = _Path(__file__).resolve().parent.parent
    if str(_SRC) not in sys.path:
        sys.path.insert(0, str(_SRC))

    from framework.plotting import generate_standalone_report
    from utils import apply_vbt_settings, load_fx_data

    ap = argparse.ArgumentParser(
        description="OU Mean Reversion — VWAP MR + vol-targeted leverage"
    )
    ap.add_argument("--data", default="data/EUR-USD_minute.parquet")
    ap.add_argument("--sigma-target", type=float, default=0.10,
                    help="Target annualized volatility (default: 10%)")
    ap.add_argument("--max-leverage", type=float, default=3.0)
    ap.add_argument("--leverage-mult", type=float, default=1.0,
                    help="Scalar multiplier on the vol-targeted leverage array")
    ap.add_argument("--no-grid", action="store_true")
    ap.add_argument("--no-show", action="store_true")
    ap.add_argument("--output-dir", default="results/ou_standalone")
    args = ap.parse_args()

    apply_vbt_settings()
    print("Loading data ...")
    raw, data = load_fx_data(args.data)

    param_grid = None if args.no_grid else {
        "bb_window": [60, 80, 120],
        "bb_alpha": [4.0, 5.0, 6.0],
        "sigma_target": [0.05, 0.10, 0.20],
    }

    fixed_params = {
        "sigma_target": args.sigma_target,
        "max_leverage": args.max_leverage,
        "leverage_mult": args.leverage_mult,
    }

    generate_standalone_report(
        backtest_fn=backtest_ou_mr,
        backtest_multi_fn=backtest_ou_mr_multi,  # Numba-parallel path
        data=data,
        name="OU Mean Reversion",
        param_grid=param_grid,
        fixed_params=fixed_params if args.no_grid else {
            "max_leverage": args.max_leverage,
            "leverage_mult": args.leverage_mult,
        },
        output_dir=args.output_dir,
        show=not args.no_show,
    )
    print("\nDone.")
