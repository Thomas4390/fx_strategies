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
# CLI ENTRY POINT
# ===================================================================

if __name__ == "__main__":
    from utils import apply_vbt_settings

    apply_vbt_settings()

    print("Loading daily closes...")
    closes = load_daily_closes()
    print(f"  {len(closes)} days, {list(closes.columns)}")

    # XS Momentum
    print("\n=== Cross-Sectional Momentum (21/63) ===")
    xs_rets = backtest_xs_momentum(closes)
    sr = xs_rets.mean() / xs_rets.std() * np.sqrt(252)
    print(f"  Sharpe: {sr:.2f}")
    print(f"  Annual return: {xs_rets.mean() * 252 * 100:.2f}%")

    # TS Momentum + RSI (GBP-USD best pair)
    print("\n=== TS Momentum + RSI7 (GBP-USD) ===")
    ts_rets = backtest_ts_momentum_rsi(closes["GBP-USD"])
    sr = ts_rets.mean() / ts_rets.std() * np.sqrt(252)
    print(f"  Sharpe: {sr:.2f}")

    # TS Portfolio
    print("\n=== TS Momentum Portfolio (4 pairs EW) ===")
    port_rets = backtest_ts_momentum_portfolio(closes)
    sr = port_rets.mean() / port_rets.std() * np.sqrt(252)
    print(f"  Sharpe: {sr:.2f}")

    # RSI MR (EUR-USD)
    print("\n=== RSI MR (EUR-USD p=14 25/75) ===")
    rsi_rets = backtest_rsi_mr(closes["EUR-USD"])
    sr = rsi_rets.mean() / rsi_rets.std() * np.sqrt(252)
    print(f"  Sharpe: {sr:.2f}")

    # Walk-forward per year
    print("\n=== Walk-Forward XS Momentum ===")
    for year in range(2019, 2027):
        yr = xs_rets.loc[f"{year}"]
        if len(yr) > 20:
            yr_sr = yr.mean() / yr.std() * np.sqrt(252) if yr.std() > 0 else 0
            print(f"  {year}: Sharpe={yr_sr:.2f}")
