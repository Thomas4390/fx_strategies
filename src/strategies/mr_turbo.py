"""MR Turbo: Validated intraday VWAP mean reversion.

100% VBT Pro native approach — pre-computed boolean signals, no signal_func_nb.
Validated OOS on 2025 data with walk-forward across 2021-2025 (4/5 years positive).

Strategy logic:
- Entry: Bollinger Bands on close-VWAP deviation breach (BB 60, alpha 5.0)
- Session filter: 6-14 UTC (pre-London + London AM)
- Exit: Fixed SL=0.5%, TP=0.6%, EOD 21:00 UTC, max hold 4h
- No signal-based exit — pure stop/time management

Research findings (walk-forward 2021-2025):
  Avg Sharpe: 0.23 | 4/5 years positive | OOS 2025: +0.47
  ~35 trades/year | 52-55% win rate | PF 1.04-1.08 | Max DD ~5%
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import vectorbtpro as vbt

from framework.spec import (
    IndicatorSpec,
    OverlayLine,
    ParamDef,
    PlotConfig,
    PortfolioConfig,
    StrategySpec,
)


# ═══════════════════════════════════════════════════════════════════════
# STANDALONE BACKTEST (100% VBT native, no framework dependency)
# ═══════════════════════════════════════════════════════════════════════


def backtest_mr_turbo(
    data: vbt.Data,
    bb_window: int = 80,
    bb_alpha: float = 5.0,
    sl_stop: float = 0.005,
    tp_stop: float = 0.006,
    session_start: int = 6,
    session_end: int = 14,
    dt_stop: str = "21:00",
    td_stop: str = "6h",
    init_cash: float = 1_000_000,
    slippage: float = 0.00015,
    leverage: float = 1.0,
) -> vbt.Portfolio:
    """Run MR Turbo backtest using 100% VBT native functions.

    Can be used standalone (no StrategySpec/Runner needed).
    """
    close = data.close

    # Native VWAP (session-anchored, daily reset)
    vwap = vbt.VWAP.run(data.high, data.low, close, data.volume, anchor="D").vwap

    # Bollinger Bands on close-VWAP deviation
    deviation = close - vwap
    bb = vbt.BBANDS.run(deviation, window=bb_window, alpha=bb_alpha)
    upper = vwap + bb.upper
    lower = vwap + bb.lower

    # Session filter
    hours = close.index.hour
    session = (hours >= session_start) & (hours < session_end)

    # Pre-computed boolean entry signals
    entries = (close < lower) & session
    short_entries = (close > upper) & session

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
        leverage=leverage,
        freq="1min",
    )


# ═══════════════════════════════════════════════════════════════════════
# MULTI-PARAM VARIANT (fully Numba-parallel via VBT native broadcasting)
# ═══════════════════════════════════════════════════════════════════════


def backtest_mr_turbo_multi(
    data: vbt.Data,
    bb_window: list | int = 80,
    bb_alpha: list | float = 5.0,
    sl_stop: float = 0.005,
    tp_stop: float = 0.006,
    session_start: int = 6,
    session_end: int = 14,
    dt_stop: str = "21:00",
    td_stop: str = "6h",
    init_cash: float = 1_000_000,
    slippage: float = 0.00015,
    leverage: float = 1.0,
    param_product: bool = True,
) -> vbt.Portfolio:
    """Multi-column MR Turbo — one broadcasted portfolio for the full grid.

    Accepts lists for ``bb_window`` and ``bb_alpha`` and builds a
    multi-column portfolio via ``vbt.BBANDS.run(..., param_product=True)``
    + ``Portfolio.from_signals(chunked="threadpool")``. The entire
    pipeline runs inside Numba kernels in parallel across combinations —
    no Python loop, no GIL contention.

    This is the preferred path for parameter sweeps / CV sweeps where
    throughput matters. Returns a multi-column portfolio whose
    ``sharpe_ratio``, ``annualized_return`` etc. are ``pd.Series``
    indexed by the parameter combinations.
    """
    import numpy as np

    close = data.close

    # Native VWAP — single column (shared across all param combos)
    vwap = vbt.VWAP.run(data.high, data.low, close, data.volume, anchor="D").vwap
    deviation = close - vwap

    # Multi-column BBANDS via native param broadcasting
    bb = vbt.BBANDS.run(
        deviation,
        window=bb_window,
        alpha=bb_alpha,
        param_product=param_product,
    )
    # upper/lower are DataFrames (N_bars × n_combos)
    upper = vwap.values[:, None] + bb.upper.values
    lower = vwap.values[:, None] + bb.lower.values

    # Session filter broadcast to match column count
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

    # Pass the vbt.Data object so Portfolio.from_signals broadcasts OHLC
    # to match the multi-column signal shape (essential for correct
    # intrabar SL/TP evaluation). VBT auto-aligns on entries.columns.
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
        leverage=leverage,
        freq="1min",
        chunked="threadpool",
    )


# ═══════════════════════════════════════════════════════════════════════
# FRAMEWORK INTEGRATION (StrategySpec-compatible)
# ═══════════════════════════════════════════════════════════════════════

# Minimal kernel — computes only what the framework needs for plotting/indicators
from numba import njit
from utils import compute_deviation_nb


@njit(nogil=True)
def compute_turbo_indicators_nb(
    index_ns: np.ndarray,
    close: np.ndarray,
    vwap: np.ndarray,
    bb_window: int,
    bb_alpha: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute BB bands around VWAP for plotting overlay.

    Note: The actual backtest uses VBT native BBANDS, not this kernel.
    This kernel exists only for the StrategySpec indicator output contract.
    """
    deviation = compute_deviation_nb(close, vwap)
    n = len(close)

    # Simple rolling std + bands (for indicator output)
    from utils import compute_intraday_rolling_std_nb

    rstd = compute_intraday_rolling_std_nb(index_ns, deviation, bb_window)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    for i in range(n):
        s = rstd[i]
        if not np.isnan(s) and s > 1e-10 and not np.isnan(vwap[i]):
            upper[i] = vwap[i] + bb_alpha * s
            lower[i] = vwap[i] - bb_alpha * s

    return deviation, upper, lower


def prepare_turbo(
    raw: pd.DataFrame,
    data: vbt.Data | None,
) -> dict[str, np.ndarray]:
    """Pre-compute VWAP natively."""
    vwap_ind = vbt.VWAP.run(
        raw["high"], raw["low"], raw["close"], raw["volume"], anchor="D"
    )
    return {"vwap": vwap_ind.vwap.values}


# ═══════════════════════════════════════════════════════════════════════
# STRATEGY SPECIFICATION (for framework runner compatibility)
# ═══════════════════════════════════════════════════════════════════════

# Note: The framework runner uses signal_func_nb, but the recommended way
# to run MR Turbo is via backtest_mr_turbo() which uses pure pre-computed
# signals. The spec below is for registry/plotting compatibility.

from utils import mr_band_signal_nb

spec = StrategySpec(
    name="MR Turbo",
    indicator=IndicatorSpec(
        class_name="MRTurbo",
        short_name="mrt",
        input_names=("index_ns", "close_minute", "vwap"),
        param_names=("bb_window", "bb_alpha"),
        output_names=("deviation", "upper_band", "lower_band"),
        kernel_func=compute_turbo_indicators_nb,
    ),
    signal_func=mr_band_signal_nb,
    signal_args_map=(
        ("close_arr", "data.close"),
        ("upper_arr", "ind.upper_band"),
        ("lower_arr", "ind.lower_band"),
        ("twap_arr", "pre.vwap"),
        ("regime_ok_arr", "pre.vwap"),  # dummy — no regime filter
        ("index_arr", "extra.index_ns"),
        ("eod_hour", "param.eod_hour"),
        ("eod_minute", "param.eod_minute"),
        ("eval_freq", "param.eval_freq"),
    ),
    params={
        "bb_window": ParamDef(60),
        "bb_alpha": ParamDef(5.0),
        "eod_hour": ParamDef(21),
        "eod_minute": ParamDef(0),
        "eval_freq": ParamDef(5),
    },
    portfolio_config=PortfolioConfig(
        sl_stop=0.005,
        extra_kwargs={
            "tp_stop": 0.006,
            "dt_stop": "21:00",
            "td_stop": "4h",
        },
    ),
    plot_config=PlotConfig(
        overlays=(
            OverlayLine("pre.vwap", "VWAP", color="#FF9800", dash="dash"),
            OverlayLine("ind.upper_band", "Upper Band", color="#E91E63", dash="dot"),
            OverlayLine("ind.lower_band", "Lower Band", color="#E91E63", dash="dot"),
        ),
    ),
    prepare_fn=prepare_turbo,
)


# ═══════════════════════════════════════════════════════════════════════
# CLI ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════

def _walk_forward_report(data: vbt.Data) -> None:
    """Per-year walk-forward diagnostic table."""
    print(f"\n{'=' * 60}\nWalk-Forward Validation (per-year)\n{'=' * 60}")
    for year in range(2021, 2027):
        d_yr = data.loc[f"{year}-01-01":f"{year}-12-31"]
        if d_yr.shape[0] < 1000:
            continue
        pf_yr = backtest_mr_turbo(d_yr)
        tc = pf_yr.trades.count()
        sr = pf_yr.sharpe_ratio if tc > 0 else 0
        ret = pf_yr.total_return * 100 if tc > 0 else 0
        wr = pf_yr.trades.win_rate * 100 if tc > 0 else 0
        print(f"  {year}: Sharpe={sr:>7.3f}  Ret={ret:>6.2f}%  Trades={tc}  WR={wr:.1f}%")


if __name__ == "__main__":
    import argparse
    import sys
    from pathlib import Path as _Path

    # Allow running directly: `python src/strategies/mr_turbo.py`
    _SRC = _Path(__file__).resolve().parent.parent
    if str(_SRC) not in sys.path:
        sys.path.insert(0, str(_SRC))

    from framework.plotting import generate_standalone_report
    from utils import apply_vbt_settings, load_fx_data

    ap = argparse.ArgumentParser(description="MR Turbo standalone report")
    ap.add_argument("--data", default="data/EUR-USD_minute.parquet")
    ap.add_argument("--leverage", type=float, default=1.0,
                    help="Fixed leverage for the single run")
    ap.add_argument("--no-grid", action="store_true",
                    help="Skip parameter grid sweep (faster)")
    ap.add_argument("--no-show", action="store_true",
                    help="Do not open plots in the browser")
    ap.add_argument("--output-dir", default="results/mrt_standalone",
                    help="Directory for fullscreen HTML plots")
    args = ap.parse_args()

    apply_vbt_settings()
    print("Loading data...")
    raw, data = load_fx_data(args.data)

    # Annual walk-forward diagnostic still printed to console
    _walk_forward_report(data)

    # Multi-broadcast-native params only (bb_window × bb_alpha) →
    # fully Numba-parallel via backtest_mr_turbo_multi.
    param_grid = None if args.no_grid else {
        "bb_window": [40, 60, 80, 120],
        "bb_alpha": [3.5, 4.0, 4.5, 5.0, 5.5, 6.0],
    }

    generate_standalone_report(
        backtest_fn=backtest_mr_turbo,
        backtest_multi_fn=backtest_mr_turbo_multi,  # Numba-parallel path
        data=data,
        name="MR Turbo",
        param_grid=param_grid,
        fixed_params={"leverage": args.leverage},
        output_dir=args.output_dir,
        show=not args.no_show,
    )
    print("\nDone.")
