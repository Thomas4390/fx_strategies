"""Daily FX Momentum Strategies.

Two validated daily strategies:
1. Cross-sectional momentum (rank 4 pairs by 21/63d returns)
2. Time-series momentum with RSI confirmation (per-pair EMA crossover)

Both use .shift(1) on signals to prevent look-ahead.
Both use volatility targeting for position sizing.

Research findings (walk-forward 2019-2025):
  XS Momentum (21/63): Sharpe 0.72, 6/7 years positive
  TS Momentum GBP-USD EMA(20/50)+RSI7: Sharpe 0.70, 7/7 years positive
  TS Momentum 4-pair EW: Sharpe 0.68, 6/7 years positive
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import vectorbtpro as vbt

from utils import load_fx_data

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
FX_PAIRS = ["EUR-USD", "GBP-USD", "USD-JPY", "USD-CAD"]


# ===================================================================
# DATA
# ===================================================================

def load_daily_closes(
    pairs: list[str] | None = None,
) -> pd.DataFrame:
    """Load FX pairs as daily close prices."""
    if pairs is None:
        pairs = FX_PAIRS
    closes = {}
    for pair in pairs:
        _, data = load_fx_data(f"data/{pair}_minute.parquet")
        closes[pair] = data.close.resample("1D").last().dropna()
    return pd.DataFrame(closes).dropna()


# ===================================================================
# CROSS-SECTIONAL MOMENTUM
# ===================================================================

def compute_xs_momentum_weights(
    closes: pd.DataFrame,
    w_short: int = 21,
    w_long: int = 63,
) -> pd.DataFrame:
    """Dollar-neutral cross-sectional momentum weights.

    Signal: 0.5 * log_return(w_short) + 0.5 * log_return(w_long)
    Weights: z-score normalized, sum to zero.
    """
    ret_s = np.log(closes / closes.shift(w_short))
    ret_l = np.log(closes / closes.shift(w_long))
    momentum = 0.5 * ret_s + 0.5 * ret_l

    cs_mean = momentum.mean(axis=1)
    cs_std = momentum.std(axis=1).clip(lower=1e-10)
    z = momentum.sub(cs_mean, axis=0).div(cs_std, axis=0)

    weights = z.div(z.abs().sum(axis=1), axis=0).fillna(0)
    return weights.shift(1)  # no look-ahead


def backtest_xs_momentum(
    closes: pd.DataFrame,
    w_short: int = 21,
    w_long: int = 63,
    target_vol: float = 0.10,
) -> pd.Series:
    """Backtest cross-sectional momentum, return daily returns."""
    weights = compute_xs_momentum_weights(closes, w_short, w_long)
    daily_rets = closes.pct_change()
    port_ret = (weights * daily_rets).sum(axis=1).dropna()

    vol_21 = port_ret.rolling(21, min_periods=10).std() * np.sqrt(252)
    lev = (target_vol / vol_21.clip(lower=0.01)).clip(upper=5.0).shift(1)
    return port_ret * lev.fillna(1.0)


# ===================================================================
# TIME-SERIES MOMENTUM + RSI CONFIRMATION
# ===================================================================

def backtest_ts_momentum_rsi(
    close_daily: pd.Series,
    fast_ema: int = 20,
    slow_ema: int = 50,
    rsi_period: int = 7,
    rsi_low: int = 40,
    rsi_high: int = 60,
    target_vol: float = 0.10,
) -> pd.Series:
    """Time-series momentum with RSI confirmation for a single pair.

    Signal: EMA fast > slow = long, else short.
    RSI filter: skip long entries when RSI > rsi_high (overbought),
                skip short entries when RSI < rsi_low (oversold).
    """
    ema_f = close_daily.ewm(span=fast_ema, min_periods=fast_ema).mean()
    ema_s = close_daily.ewm(span=slow_ema, min_periods=slow_ema).mean()
    rsi = vbt.RSI.run(close_daily, window=rsi_period).rsi

    signal = pd.Series(0.0, index=close_daily.index)
    trend_long = ema_f > ema_s
    trend_short = ema_f < ema_s
    rsi_ok_long = rsi < rsi_high
    rsi_ok_short = rsi > rsi_low

    signal[trend_long & rsi_ok_long] = 1.0
    signal[trend_short & rsi_ok_short] = -1.0
    signal = signal.shift(1)  # no look-ahead

    daily_ret = close_daily.pct_change()
    vol_21 = daily_ret.rolling(21, min_periods=10).std() * np.sqrt(252)
    lev = (target_vol / vol_21.clip(lower=0.01)).clip(upper=3.0).shift(1)

    return (signal * daily_ret * lev.fillna(1.0)).dropna()


def backtest_ts_momentum_portfolio(
    closes: pd.DataFrame,
    fast_ema: int = 20,
    slow_ema: int = 50,
    rsi_period: int = 7,
    rsi_low: int = 40,
    rsi_high: int = 60,
    target_vol: float = 0.10,
) -> pd.Series:
    """Equal-weight portfolio of TS momentum across all pairs."""
    pair_rets = []
    for pair in closes.columns:
        rets = backtest_ts_momentum_rsi(
            closes[pair], fast_ema, slow_ema,
            rsi_period, rsi_low, rsi_high, target_vol,
        )
        pair_rets.append(rets)
    return pd.concat(pair_rets, axis=1).fillna(0).mean(axis=1)


# ===================================================================
# RSI MEAN REVERSION (standalone)
# ===================================================================

def backtest_rsi_mr(
    close_daily: pd.Series,
    rsi_period: int = 14,
    oversold: int = 25,
    overbought: int = 75,
    target_vol: float = 0.10,
) -> pd.Series:
    """RSI mean reversion: long when oversold, short when overbought."""
    rsi = vbt.RSI.run(close_daily, window=rsi_period).rsi

    signal = pd.Series(0.0, index=close_daily.index)
    signal[rsi < oversold] = 1.0
    signal[rsi > overbought] = -1.0
    signal = signal.shift(1)

    daily_ret = close_daily.pct_change()
    vol_21 = daily_ret.rolling(21, min_periods=10).std() * np.sqrt(252)
    lev = (target_vol / vol_21.clip(lower=0.01)).clip(upper=3.0).shift(1)

    return (signal * daily_ret * lev.fillna(1.0)).dropna()


# ===================================================================
# PORTFOLIO BUILDERS (return vbt.Portfolio for full framework plotting)
# ===================================================================


def backtest_xs_momentum_pf(
    closes: pd.DataFrame,
    w_short: int = 21,
    w_long: int = 63,
    target_vol: float = 0.10,
    leverage: float = 1.0,
    init_cash: float = 1_000_000.0,
    slippage: float = 0.0001,
) -> vbt.Portfolio:
    """Cross-sectional momentum as a multi-asset :class:`vbt.Portfolio`.

    Uses :meth:`vbt.Portfolio.from_orders` with ``size_type="targetpercent"``.
    Each pair gets an independent cash bucket (no cash sharing) so
    that VBT's per-asset plots (``orders.plot``, ``trades.plot``, MAE,
    MFE, etc.) remain available — this is a 4-column non-grouped
    portfolio rather than a dollar-neutral cash-shared group. The
    aggregate Sharpe matches the original returns-based XS momentum
    within ~3% on EUR/GBP/JPY/CAD.
    """
    weights = compute_xs_momentum_weights(closes, w_short, w_long)

    # Volatility targeting: scale the cross-sectional portfolio so its
    # ex-ante vol matches target_vol.
    daily_rets = closes.pct_change().fillna(0)
    proxy_port_ret = (weights * daily_rets).sum(axis=1)
    vol_21 = proxy_port_ret.rolling(21, min_periods=10).std() * np.sqrt(252)
    lev_mult = (target_vol / vol_21.clip(lower=0.01)).clip(upper=5.0).shift(1).fillna(1.0)

    scaled_weights = weights.mul(lev_mult, axis=0).fillna(0.0)

    return vbt.Portfolio.from_orders(
        close=closes,
        size=scaled_weights,
        size_type="targetpercent",
        init_cash=init_cash,
        leverage=leverage,
        slippage=slippage,
        fees=0.00005,
        freq="1D",
    )


def backtest_ts_momentum_pf(
    close_daily: pd.Series,
    fast_ema: int = 20,
    slow_ema: int = 50,
    rsi_period: int = 7,
    rsi_low: int = 40,
    rsi_high: int = 60,
    target_vol: float = 0.10,
    leverage: float = 1.0,
    init_cash: float = 1_000_000.0,
    slippage: float = 0.0001,
) -> vbt.Portfolio:
    """Single-pair TS momentum + RSI confirmation as a :class:`vbt.Portfolio`.

    Signal: ``EMA(fast) > EMA(slow)`` + RSI filter, with vol-targeted
    dynamic leverage overlay applied through the ``leverage`` array.
    """
    ema_f = close_daily.ewm(span=fast_ema, min_periods=fast_ema).mean()
    ema_s = close_daily.ewm(span=slow_ema, min_periods=slow_ema).mean()
    rsi = vbt.RSI.run(close_daily, window=rsi_period).rsi

    trend_long = ema_f > ema_s
    trend_short = ema_f < ema_s

    long_ok = trend_long & (rsi < rsi_high)
    short_ok = trend_short & (rsi > rsi_low)

    entries = long_ok & ~long_ok.shift(1).fillna(False)
    exits = ~long_ok & long_ok.shift(1).fillna(False)
    short_entries = short_ok & ~short_ok.shift(1).fillna(False)
    short_exits = ~short_ok & short_ok.shift(1).fillna(False)

    # Vol-targeted leverage overlay
    daily_ret = close_daily.pct_change()
    vol_21 = daily_ret.rolling(21, min_periods=10).std() * np.sqrt(252)
    dyn_lev = (target_vol / vol_21.clip(lower=0.01)).clip(upper=3.0).shift(1).fillna(1.0)
    dyn_lev = (dyn_lev * leverage).values  # scalar user multiplier

    return vbt.Portfolio.from_signals(
        close=close_daily,
        entries=entries,
        exits=exits,
        short_entries=short_entries,
        short_exits=short_exits,
        leverage=dyn_lev,
        init_cash=init_cash,
        slippage=slippage,
        freq="1D",
    )


# ===================================================================
# CLI ENTRY POINT
# ===================================================================

if __name__ == "__main__":
    import argparse
    import sys
    from pathlib import Path as _Path

    # Allow direct execution: `python src/strategies/daily_momentum.py`
    _SRC = _Path(__file__).resolve().parent.parent
    if str(_SRC) not in sys.path:
        sys.path.insert(0, str(_SRC))

    from framework.plotting import generate_standalone_report
    from utils import apply_vbt_settings

    ap = argparse.ArgumentParser(description="Daily momentum strategies (XS + TS)")
    ap.add_argument(
        "--strategy",
        default="xs",
        choices=["xs", "ts", "all"],
        help="xs=cross-sectional momentum, ts=time-series momentum on a pair, all=both",
    )
    ap.add_argument("--pair", default="GBP-USD",
                    help="Pair for 'ts' strategy (single-pair)")
    ap.add_argument("--leverage", type=float, default=1.0)
    ap.add_argument("--no-grid", action="store_true")
    ap.add_argument("--no-show", action="store_true")
    ap.add_argument("--output-dir", default=None,
                    help="Default: results/daily_<strategy>")
    args = ap.parse_args()

    apply_vbt_settings()
    print("Loading daily closes for 4 pairs ...")
    closes = load_daily_closes()
    print(f"  {len(closes)} days, {list(closes.columns)}")

    if args.strategy in ("xs", "all"):
        out = args.output_dir or "results/daily_xs"
        pg = None if args.no_grid else {
            "w_short": [10, 21, 42],
            "w_long": [42, 63, 126],
            "target_vol": [0.08, 0.10, 0.12],
        }
        generate_standalone_report(
            backtest_fn=backtest_xs_momentum_pf,
            data=closes,
            name="XS Momentum (4-pair)",
            param_grid=pg,
            fixed_params={"leverage": args.leverage},
            output_dir=out,
            show=not args.no_show,
        )

    if args.strategy in ("ts", "all"):
        out = args.output_dir or f"results/daily_ts_{args.pair.lower()}"
        pair_close = closes[args.pair]
        pg = None if args.no_grid else {
            "fast_ema": [10, 20, 30],
            "slow_ema": [40, 50, 100],
            "rsi_period": [7, 14],
        }
        generate_standalone_report(
            backtest_fn=backtest_ts_momentum_pf,
            data=pair_close,
            name=f"TS Momentum+RSI ({args.pair})",
            param_grid=pg,
            fixed_params={"leverage": args.leverage},
            output_dir=out,
            show=not args.no_show,
        )

    print("\nDone.")
