"""Daily/Weekly FX Strategy Research.

Explores carry, momentum, value, and composite strategies at daily frequency.
All signals use .shift(1) to ensure entry at D+1 (no look-ahead).

Strategies:
  S1. Cross-Sectional Momentum (rank 4 pairs, long best, short worst)
  S2. Carry Trade proxy (interest rate differential)
  S3. Time-Series Momentum (per-pair trend following)
  S4. Composite (S1+S2+S3 combined)
  S5. Existing composite_fx_alpha.py evaluation
"""

from __future__ import annotations

import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import vectorbtpro as vbt

from utils import apply_vbt_settings, load_fx_data

warnings.filterwarnings("ignore", category=FutureWarning)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_DATA_DIR = _PROJECT_ROOT / "data"

FX_PAIRS = ["EUR-USD", "GBP-USD", "USD-JPY", "USD-CAD"]

WF_PERIODS: list[tuple[str, str]] = [
    (f"{y}-01-01", f"{y}-12-31") for y in range(2019, 2025)
] + [("2025-01-01", "2026-04-01")]


# ===================================================================
# HELPERS
# ===================================================================

def _safe_sharpe(pf: vbt.Portfolio) -> float:
    try:
        if pf.trades.count() == 0:
            return 0.0
        sr = pf.sharpe_ratio
        return 0.0 if (pd.isna(sr) or np.isinf(sr)) else float(sr)
    except Exception:
        return 0.0

def _safe_trades(pf: vbt.Portfolio) -> int:
    try:
        return int(pf.trades.count())
    except Exception:
        return 0

def _h(title: str) -> None:
    print(f"\n{'=' * 70}\n  {title}\n{'=' * 70}")

def _sh(title: str) -> None:
    print(f"\n{'-' * 60}\n  {title}\n{'-' * 60}")

def pr(r: dict) -> None:
    detail = " ".join(f"{s:>6.2f}" for s in r["sharpes"])
    print(
        f"  {r['label']:<50} avg={r['avg_sharpe']:>5.2f}"
        f" pos={r['pos_years']}/7 oos={r['oos_sharpe']:>5.2f}"
        f" tc={r['total_trades']:>5} | {detail}"
    )


def load_daily_closes() -> pd.DataFrame:
    """Load all 4 pairs as daily close DataFrame."""
    closes = {}
    for pair in FX_PAIRS:
        _, data = load_fx_data(f"data/{pair}_minute.parquet")
        closes[pair] = data.close.resample("1D").last().dropna()
    df = pd.DataFrame(closes).dropna()
    return df


# ===================================================================
# S1: CROSS-SECTIONAL MOMENTUM
# ===================================================================

def compute_xs_momentum_weights(
    closes: pd.DataFrame,
    w_short: int = 21,
    w_long: int = 63,
) -> pd.DataFrame:
    """Compute dollar-neutral cross-sectional momentum weights.

    Signal: 0.5 * log_return(21d) + 0.5 * log_return(63d)
    Weights: z-score normalized, sum to zero (dollar-neutral).
    .shift(1) applied to use yesterday's signal for today's trade.
    """
    # Log returns
    ret_short = np.log(closes / closes.shift(w_short))
    ret_long = np.log(closes / closes.shift(w_long))
    momentum = 0.5 * ret_short + 0.5 * ret_long

    # Cross-sectional z-score
    cs_mean = momentum.mean(axis=1)
    cs_std = momentum.std(axis=1)
    z = momentum.sub(cs_mean, axis=0).div(cs_std.clip(lower=1e-10), axis=0)

    # Dollar-neutral weights: w_i = z_i / sum(|z_j|)
    abs_sum = z.abs().sum(axis=1)
    weights = z.div(abs_sum, axis=0).fillna(0)

    # .shift(1): use yesterday's signal
    return weights.shift(1)


def backtest_xs_momentum(
    closes: pd.DataFrame,
    w_short: int = 21,
    w_long: int = 63,
    target_vol: float = 0.10,
    start: str | None = None,
    end: str | None = None,
) -> dict[str, Any]:
    """Backtest cross-sectional momentum portfolio."""
    if start:
        closes = closes.loc[start:end]

    weights = compute_xs_momentum_weights(closes, w_short, w_long)
    daily_returns = closes.pct_change()

    # Portfolio return = sum of weight * return per pair
    port_ret = (weights * daily_returns).sum(axis=1).dropna()

    # Volatility targeting
    vol_21 = port_ret.rolling(21, min_periods=10).std() * np.sqrt(252)
    leverage = (target_vol / vol_21.clip(lower=0.01)).clip(upper=5.0).shift(1)
    scaled_ret = port_ret * leverage.fillna(1.0)

    # Metrics
    sr = (
        scaled_ret.mean() / scaled_ret.std() * np.sqrt(252)
        if scaled_ret.std() > 0 else 0.0
    )
    total_ret = (1 + scaled_ret).prod() - 1
    max_dd = (
        (1 + scaled_ret).cumprod()
        / (1 + scaled_ret).cumprod().cummax() - 1
    ).min()

    return {
        "sharpe": float(sr) if not np.isnan(sr) else 0.0,
        "total_return": float(total_ret),
        "max_dd": float(max_dd),
        "daily_returns": scaled_ret,
    }


def run_xs_momentum_sweep() -> list[dict]:
    """S1: Cross-sectional momentum sweep."""
    _h("S1: Cross-Sectional Momentum (Daily)")
    closes = load_daily_closes()
    results = []

    for w_short in [10, 21, 42]:
        for w_long in [63, 126, 252]:
            if w_short >= w_long:
                continue
            for tvol in [0.05, 0.10, 0.15]:
                label = f"XSMom({w_short}/{w_long}) vol={tvol}"

                sharpes = []
                for start, end in WF_PERIODS:
                    try:
                        res = backtest_xs_momentum(
                            closes, w_short, w_long, tvol,
                            start=start, end=end,
                        )
                        sharpes.append(res["sharpe"])
                    except Exception:
                        sharpes.append(0.0)

                r = {
                    "label": label,
                    "sharpes": sharpes,
                    "avg_sharpe": float(np.mean(sharpes)),
                    "pos_years": sum(1 for s in sharpes if s > 0),
                    "oos_sharpe": sharpes[-1],
                    "total_trades": 0,  # continuous rebalancing
                }
                results.append(r)

    results.sort(key=lambda x: x["avg_sharpe"], reverse=True)
    _sh("Top 10 XS Momentum")
    for r in results[:10]:
        pr(r)
    return results


# ===================================================================
# S3: TIME-SERIES MOMENTUM (per pair)
# ===================================================================

def backtest_ts_momentum_pair(
    close_daily: pd.Series,
    fast_ema: int = 10,
    slow_ema: int = 50,
    target_vol: float = 0.10,
    start: str | None = None,
    end: str | None = None,
) -> dict[str, Any]:
    """Time-series momentum for a single pair.

    Signal: fast EMA > slow EMA = long, else short.
    Position sizing: volatility targeting.
    """
    if start:
        close_daily = close_daily.loc[start:end]

    ema_f = close_daily.ewm(span=fast_ema, min_periods=fast_ema).mean()
    ema_s = close_daily.ewm(span=slow_ema, min_periods=slow_ema).mean()

    # Signal: +1 long, -1 short (shift(1) for no look-ahead)
    signal = pd.Series(0.0, index=close_daily.index)
    signal[ema_f > ema_s] = 1.0
    signal[ema_f < ema_s] = -1.0
    signal = signal.shift(1)

    daily_ret = close_daily.pct_change()

    # Vol targeting
    vol_21 = daily_ret.rolling(21, min_periods=10).std() * np.sqrt(252)
    leverage = (target_vol / vol_21.clip(lower=0.01)).clip(upper=5.0).shift(1)

    strategy_ret = signal * daily_ret * leverage.fillna(1.0)
    strategy_ret = strategy_ret.dropna()

    sr = (
        strategy_ret.mean() / strategy_ret.std() * np.sqrt(252)
        if strategy_ret.std() > 0 else 0.0
    )

    return {
        "sharpe": float(sr) if not np.isnan(sr) else 0.0,
        "daily_returns": strategy_ret,
    }


def run_ts_momentum_per_pair() -> list[dict]:
    """S3: Time-series momentum per pair."""
    _h("S3: Time-Series Momentum (Per Pair)")
    closes = load_daily_closes()
    results = []

    for pair in FX_PAIRS:
        for fast in [5, 10, 20, 50]:
            for slow in [20, 50, 100, 200]:
                if fast >= slow:
                    continue
                for tvol in [0.05, 0.10]:
                    label = f"TSMom {pair} EMA({fast}/{slow}) v={tvol}"

                    sharpes = []
                    for start, end in WF_PERIODS:
                        try:
                            res = backtest_ts_momentum_pair(
                                closes[pair], fast, slow, tvol,
                                start=start, end=end,
                            )
                            sharpes.append(res["sharpe"])
                        except Exception:
                            sharpes.append(0.0)

                    r = {
                        "label": label,
                        "sharpes": sharpes,
                        "avg_sharpe": float(np.mean(sharpes)),
                        "pos_years": sum(1 for s in sharpes if s > 0),
                        "oos_sharpe": sharpes[-1],
                        "total_trades": 0,
                    }
                    results.append(r)

    results.sort(key=lambda x: x["avg_sharpe"], reverse=True)
    _sh("Top 15 Time-Series Momentum")
    for r in results[:15]:
        pr(r)

    # Also show best per pair
    for pair in FX_PAIRS:
        pair_results = [r for r in results if pair in r["label"]]
        if pair_results:
            best = pair_results[0]
            print(f"\n  Best {pair}: ", end="")
            pr(best)

    return results


# ===================================================================
# S3B: TIME-SERIES MOMENTUM PORTFOLIO (all pairs combined)
# ===================================================================

def run_ts_momentum_equal_weight() -> list[dict]:
    """S3B: Time-series momentum — equal weight portfolio of all pairs."""
    _h("S3B: TS Momentum Portfolio (4 pairs EW)")
    closes = load_daily_closes()
    results = []

    for fast in [10, 20, 50]:
        for slow in [50, 100, 200]:
            if fast >= slow:
                continue
            for tvol in [0.05, 0.10]:
                label = f"TSMomEW EMA({fast}/{slow}) v={tvol}"

                sharpes = []
                for start, end in WF_PERIODS:
                    try:
                        pair_rets = []
                        for pair in FX_PAIRS:
                            res = backtest_ts_momentum_pair(
                                closes[pair], fast, slow, tvol,
                                start=start, end=end,
                            )
                            pair_rets.append(res["daily_returns"])

                        # Equal weight portfolio
                        port = pd.concat(pair_rets, axis=1).fillna(0).mean(axis=1)
                        sr = (
                            port.mean() / port.std() * np.sqrt(252)
                            if port.std() > 0 else 0.0
                        )
                        sharpes.append(float(sr) if not np.isnan(sr) else 0.0)
                    except Exception:
                        sharpes.append(0.0)

                r = {
                    "label": label,
                    "sharpes": sharpes,
                    "avg_sharpe": float(np.mean(sharpes)),
                    "pos_years": sum(1 for s in sharpes if s > 0),
                    "oos_sharpe": sharpes[-1],
                    "total_trades": 0,
                }
                results.append(r)

    results.sort(key=lambda x: x["avg_sharpe"], reverse=True)
    _sh("Top 10 TS Momentum Portfolio")
    for r in results[:10]:
        pr(r)
    return results


# ===================================================================
# S2: CARRY TRADE PROXY
# ===================================================================

def run_carry_proxy_sweep() -> list[dict]:
    """S2: Carry trade proxy using yield spread."""
    _h("S2: Carry Trade Proxy (Yield Spread)")

    closes = load_daily_closes()
    results = []

    # Load yield data as carry proxy
    spread_df = pd.read_parquet(_DATA_DIR / "SPREAD_10Y2Y_daily.parquet")
    spread_df["date"] = pd.to_datetime(spread_df["date"])
    spread = spread_df.set_index("date")["spread_10y2y"]

    fed_df = pd.read_parquet(_DATA_DIR / "FED_FUNDS_monthly.parquet")
    fed_df["date"] = pd.to_datetime(fed_df["date"])
    fed = fed_df.set_index("date")["fed_funds"]
    fed_daily = fed.reindex(closes.index, method="ffill")

    # Simplified carry signal: when USD yield is high (fed funds high),
    # short foreign currencies (long USD). When yield low, long foreign.
    # This is a simplified version — real carry uses per-country rates.

    for carry_lookback in [21, 63, 126]:
        for threshold_pctile in [30, 50, 70]:
            label = f"Carry lb={carry_lookback} q={threshold_pctile}"

            # Carry signal: fed funds change over lookback
            fed_chg = fed_daily.diff(carry_lookback).shift(1)
            threshold = fed_chg.rolling(252, min_periods=60).quantile(
                threshold_pctile / 100
            )

            # When fed rising (hawkish) -> long USD (short FX pairs)
            # When fed falling (dovish) -> short USD (long FX pairs)
            signal = pd.Series(0.0, index=closes.index)
            signal[fed_chg > threshold] = -1.0  # hawkish -> short FX
            signal[fed_chg < -threshold.abs()] = 1.0  # dovish -> long FX

            sharpes = []
            for start, end in WF_PERIODS:
                try:
                    cl = closes.loc[start:end]
                    sig = signal.loc[start:end]

                    # Equal weight across pairs
                    daily_rets = cl.pct_change()
                    port_ret = (
                        daily_rets.mul(sig, axis=0).mean(axis=1).dropna()
                    )

                    sr = (
                        port_ret.mean() / port_ret.std() * np.sqrt(252)
                        if port_ret.std() > 0 else 0.0
                    )
                    sharpes.append(
                        float(sr) if not np.isnan(sr) else 0.0
                    )
                except Exception:
                    sharpes.append(0.0)

            r = {
                "label": label,
                "sharpes": sharpes,
                "avg_sharpe": float(np.mean(sharpes)),
                "pos_years": sum(1 for s in sharpes if s > 0),
                "oos_sharpe": sharpes[-1],
                "total_trades": 0,
            }
            results.append(r)

    # Also test yield spread direction as carry signal
    for lb in [21, 63]:
        label = f"SpreadCarry lb={lb}"
        spread_chg = spread.diff(lb).shift(1)
        spread_aligned = spread_chg.reindex(closes.index, method="ffill")

        # Steepening (rising spread) -> risk-on -> long FX
        # Flattening -> risk-off -> short FX
        signal = pd.Series(0.0, index=closes.index)
        signal[spread_aligned > 0] = 1.0
        signal[spread_aligned < 0] = -1.0

        sharpes = []
        for start, end in WF_PERIODS:
            try:
                cl = closes.loc[start:end]
                sig = signal.loc[start:end]
                daily_rets = cl.pct_change()
                port_ret = daily_rets.mul(sig, axis=0).mean(axis=1).dropna()
                sr = (
                    port_ret.mean() / port_ret.std() * np.sqrt(252)
                    if port_ret.std() > 0 else 0.0
                )
                sharpes.append(float(sr) if not np.isnan(sr) else 0.0)
            except Exception:
                sharpes.append(0.0)

        r = {
            "label": label,
            "sharpes": sharpes,
            "avg_sharpe": float(np.mean(sharpes)),
            "pos_years": sum(1 for s in sharpes if s > 0),
            "oos_sharpe": sharpes[-1],
            "total_trades": 0,
        }
        results.append(r)

    results.sort(key=lambda x: x["avg_sharpe"], reverse=True)
    _sh("Carry Trade Results")
    for r in results:
        pr(r)
    return results


# ===================================================================
# S4: COMPOSITE (momentum + carry combined)
# ===================================================================

def run_momentum_carry_composite(
    best_mom_params: dict | None = None,
    best_carry_label: str | None = None,
) -> list[dict]:
    """S4: Composite — combine momentum and carry signals."""
    _h("S4: Composite (Momentum + Carry)")
    closes = load_daily_closes()
    results = []

    # Momentum returns (best params or default)
    w_short = best_mom_params.get("w_short", 21) if best_mom_params else 21
    w_long = best_mom_params.get("w_long", 63) if best_mom_params else 63

    # Carry signal (spread direction)
    spread_df = pd.read_parquet(_DATA_DIR / "SPREAD_10Y2Y_daily.parquet")
    spread_df["date"] = pd.to_datetime(spread_df["date"])
    spread = spread_df.set_index("date")["spread_10y2y"]
    spread_chg = spread.diff(21).shift(1)
    carry_sig = spread_chg.reindex(closes.index, method="ffill").fillna(0)
    carry_direction = np.sign(carry_sig)

    for mom_weight in [0.3, 0.5, 0.7, 1.0]:
        carry_weight = 1.0 - mom_weight
        for tvol in [0.05, 0.10]:
            label = f"Comp mom={mom_weight:.1f} car={carry_weight:.1f} v={tvol}"

            sharpes = []
            for start, end in WF_PERIODS:
                try:
                    cl = closes.loc[start:end]

                    # Momentum component
                    mom_weights = compute_xs_momentum_weights(
                        cl, w_short, w_long
                    )
                    daily_rets = cl.pct_change()
                    mom_ret = (mom_weights * daily_rets).sum(axis=1)

                    # Carry component
                    c_dir = carry_direction.loc[start:end]
                    carry_ret = daily_rets.mul(c_dir, axis=0).mean(axis=1)

                    # Composite
                    comp_ret = (
                        mom_weight * mom_ret + carry_weight * carry_ret
                    ).dropna()

                    # Vol targeting
                    vol_21 = (
                        comp_ret.rolling(21, min_periods=10).std()
                        * np.sqrt(252)
                    )
                    lev = (
                        tvol / vol_21.clip(lower=0.01)
                    ).clip(upper=5.0).shift(1)
                    scaled = comp_ret * lev.fillna(1.0)

                    sr = (
                        scaled.mean() / scaled.std() * np.sqrt(252)
                        if scaled.std() > 0 else 0.0
                    )
                    sharpes.append(
                        float(sr) if not np.isnan(sr) else 0.0
                    )
                except Exception:
                    sharpes.append(0.0)

            r = {
                "label": label,
                "sharpes": sharpes,
                "avg_sharpe": float(np.mean(sharpes)),
                "pos_years": sum(1 for s in sharpes if s > 0),
                "oos_sharpe": sharpes[-1],
                "total_trades": 0,
            }
            results.append(r)

    results.sort(key=lambda x: x["avg_sharpe"], reverse=True)
    _sh("Top 10 Composite")
    for r in results[:10]:
        pr(r)
    return results


# ===================================================================
# MAIN
# ===================================================================

def main() -> None:
    apply_vbt_settings()

    t_start = time.time()

    r_s1 = run_xs_momentum_sweep()
    r_s3 = run_ts_momentum_per_pair()
    r_s3b = run_ts_momentum_equal_weight()
    r_s2 = run_carry_proxy_sweep()
    r_s4 = run_momentum_carry_composite()

    # Summary
    _h("OVERALL SUMMARY")
    all_phases = {
        "S1_xs_momentum": r_s1,
        "S3_ts_momentum": r_s3,
        "S3B_ts_mom_portfolio": r_s3b,
        "S2_carry": r_s2,
        "S4_composite": r_s4,
    }
    for name, res in all_phases.items():
        if res:
            print(f"\n  {name}:")
            pr(res[0])

    elapsed = time.time() - t_start
    print(f"\nTotal research time: {elapsed / 60:.1f} minutes")


if __name__ == "__main__":
    main()
