"""Combined portfolio research + RSI daily exploration.

Part 1: Combine intraday MR + daily momentum into a multi-horizon portfolio
Part 2: RSI-based strategies on daily timeframe across all pairs
Part 3: RSI combined with momentum and MR signals
"""

from __future__ import annotations

import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import vectorbtpro as vbt

from strategies.mr_macro import backtest_mr_macro, load_macro_filters
from utils import apply_vbt_settings, load_fx_data

warnings.filterwarnings("ignore", category=FutureWarning)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_DATA_DIR = _PROJECT_ROOT / "data"

FX_PAIRS = ["EUR-USD", "GBP-USD", "USD-JPY", "USD-CAD"]
WF_PERIODS = [
    (f"{y}-01-01", f"{y}-12-31") for y in range(2019, 2025)
] + [("2025-01-01", "2026-04-01")]


def _safe_sharpe(pf: vbt.Portfolio) -> float:
    try:
        if pf.trades.count() == 0:
            return 0.0
        sr = pf.sharpe_ratio
        return 0.0 if (pd.isna(sr) or np.isinf(sr)) else float(sr)
    except Exception:
        return 0.0

def _h(t: str) -> None:
    print(f"\n{'=' * 70}\n  {t}\n{'=' * 70}")

def _sh(t: str) -> None:
    print(f"\n{'-' * 60}\n  {t}\n{'-' * 60}")

def pr(r: dict) -> None:
    detail = " ".join(f"{s:>6.2f}" for s in r["sharpes"])
    print(
        f"  {r['label']:<50} avg={r['avg_sharpe']:>5.2f}"
        f" pos={r['pos_years']}/7 oos={r['oos_sharpe']:>5.2f} | {detail}"
    )


def load_daily_closes() -> pd.DataFrame:
    closes = {}
    for pair in FX_PAIRS:
        _, data = load_fx_data(f"data/{pair}_minute.parquet")
        closes[pair] = data.close.resample("1D").last().dropna()
    return pd.DataFrame(closes).dropna()


# ===================================================================
# PART 1: COMBINED PORTFOLIO (Intraday MR + Daily Momentum)
# ===================================================================

def get_intraday_mr_daily_returns() -> pd.Series:
    """Get daily returns from the intraday MR macro strategy on EUR-USD."""
    _, data = load_fx_data()
    pf = backtest_mr_macro(data)
    return pf.daily_returns


def get_xs_momentum_daily_returns(
    closes: pd.DataFrame,
    w_short: int = 21,
    w_long: int = 63,
    target_vol: float = 0.10,
) -> pd.Series:
    """Get daily returns from cross-sectional momentum."""
    ret_s = np.log(closes / closes.shift(w_short))
    ret_l = np.log(closes / closes.shift(w_long))
    momentum = 0.5 * ret_s + 0.5 * ret_l

    cs_mean = momentum.mean(axis=1)
    cs_std = momentum.std(axis=1).clip(lower=1e-10)
    z = momentum.sub(cs_mean, axis=0).div(cs_std, axis=0)
    weights = z.div(z.abs().sum(axis=1), axis=0).fillna(0).shift(1)

    daily_rets = closes.pct_change()
    port_ret = (weights * daily_rets).sum(axis=1).dropna()

    vol_21 = port_ret.rolling(21, min_periods=10).std() * np.sqrt(252)
    lev = (target_vol / vol_21.clip(lower=0.01)).clip(upper=5.0).shift(1)
    return port_ret * lev.fillna(1.0)


def get_ts_momentum_daily_returns(
    closes: pd.DataFrame,
    fast: int = 20,
    slow: int = 50,
    target_vol: float = 0.10,
) -> pd.Series:
    """Get daily returns from time-series momentum EW portfolio."""
    pair_rets = []
    for pair in FX_PAIRS:
        c = closes[pair]
        ef = c.ewm(span=fast, min_periods=fast).mean()
        es = c.ewm(span=slow, min_periods=slow).mean()
        signal = pd.Series(0.0, index=c.index)
        signal[ef > es] = 1.0
        signal[ef < es] = -1.0
        signal = signal.shift(1)

        dr = c.pct_change()
        vol_21 = dr.rolling(21, min_periods=10).std() * np.sqrt(252)
        lev = (target_vol / vol_21.clip(lower=0.01)).clip(upper=5.0).shift(1)
        pair_rets.append(signal * dr * lev.fillna(1.0))

    return pd.concat(pair_rets, axis=1).fillna(0).mean(axis=1)


def part1_combined_portfolio() -> None:
    """Test combined intraday MR + daily momentum portfolio."""
    _h("PART 1: Combined Portfolio (Intraday MR + Daily Momentum)")

    closes = load_daily_closes()

    print("Computing strategy returns...")
    mr_rets = get_intraday_mr_daily_returns()
    xs_rets = get_xs_momentum_daily_returns(closes)
    ts_rets = get_ts_momentum_daily_returns(closes)

    # Align all to common index
    common = mr_rets.index.intersection(xs_rets.index).intersection(ts_rets.index)
    mr = mr_rets.reindex(common).fillna(0)
    xs = xs_rets.reindex(common).fillna(0)
    ts = ts_rets.reindex(common).fillna(0)

    # Individual strategy stats
    _sh("Individual Strategy Performance (full period)")
    for name, rets in [("Intraday MR", mr), ("XS Momentum", xs), ("TS Momentum EW", ts)]:
        sr = rets.mean() / rets.std() * np.sqrt(252) if rets.std() > 0 else 0
        ann_ret = rets.mean() * 252 * 100
        dd = ((1 + rets).cumprod() / (1 + rets).cumprod().cummax() - 1).min() * 100
        print(f"  {name:<25} Sharpe={sr:>5.2f}  AnnRet={ann_ret:>5.2f}%  MaxDD={dd:>6.2f}%")

    # Correlation matrix
    _sh("Return Correlations (daily)")
    corr_df = pd.DataFrame({
        "MR_intraday": mr, "XS_momentum": xs, "TS_momentum": ts,
    })
    print(corr_df.corr().round(3).to_string())

    # Portfolio combinations
    _sh("Portfolio Combinations")
    combos = [
        ("MR only", {"mr": 1.0, "xs": 0.0, "ts": 0.0}),
        ("XS only", {"mr": 0.0, "xs": 1.0, "ts": 0.0}),
        ("TS only", {"mr": 0.0, "xs": 0.0, "ts": 1.0}),
        ("MR + XS (50/50)", {"mr": 0.5, "xs": 0.5, "ts": 0.0}),
        ("MR + TS (50/50)", {"mr": 0.5, "xs": 0.0, "ts": 0.5}),
        ("XS + TS (50/50)", {"mr": 0.0, "xs": 0.5, "ts": 0.5}),
        ("MR + XS + TS (33/33/33)", {"mr": 0.33, "xs": 0.33, "ts": 0.34}),
        ("MR 50 + XS 25 + TS 25", {"mr": 0.50, "xs": 0.25, "ts": 0.25}),
        ("MR 25 + XS 50 + TS 25", {"mr": 0.25, "xs": 0.50, "ts": 0.25}),
    ]

    results = []
    for name, w in combos:
        port = w["mr"] * mr + w["xs"] * xs + w["ts"] * ts

        # Walk-forward
        sharpes = []
        for start, end in WF_PERIODS:
            p = port.loc[start:end]
            if len(p) < 20:
                sharpes.append(0.0)
                continue
            sr = p.mean() / p.std() * np.sqrt(252) if p.std() > 0 else 0
            sharpes.append(float(sr) if not np.isnan(sr) else 0.0)

        r = {
            "label": name,
            "sharpes": sharpes,
            "avg_sharpe": float(np.mean(sharpes)),
            "pos_years": sum(1 for s in sharpes if s > 0),
            "oos_sharpe": sharpes[-1],
        }
        results.append(r)

    results.sort(key=lambda x: x["avg_sharpe"], reverse=True)
    for r in results:
        detail = " ".join(f"{s:>6.2f}" for s in r["sharpes"])
        print(
            f"  {r['label']:<35} avg={r['avg_sharpe']:>5.2f}"
            f" pos={r['pos_years']}/7 oos={r['oos_sharpe']:>5.2f} | {detail}"
        )

    # Risk parity
    _sh("Risk Parity Portfolio")
    vols = pd.Series({
        "mr": mr.std(), "xs": xs.std(), "ts": ts.std(),
    })
    inv_vol = 1.0 / vols
    rp_w = inv_vol / inv_vol.sum()
    print(f"  Weights: MR={rp_w['mr']:.2f} XS={rp_w['xs']:.2f} TS={rp_w['ts']:.2f}")

    port_rp = rp_w["mr"] * mr + rp_w["xs"] * xs + rp_w["ts"] * ts
    sharpes_rp = []
    for start, end in WF_PERIODS:
        p = port_rp.loc[start:end]
        sr = p.mean() / p.std() * np.sqrt(252) if p.std() > 0 else 0
        sharpes_rp.append(float(sr) if not np.isnan(sr) else 0.0)

    avg_rp = float(np.mean(sharpes_rp))
    pos_rp = sum(1 for s in sharpes_rp if s > 0)
    detail_rp = " ".join(f"{s:>6.2f}" for s in sharpes_rp)
    print(f"  Risk Parity                       avg={avg_rp:>5.2f} pos={pos_rp}/7 | {detail_rp}")


# ===================================================================
# PART 2: RSI DAILY STRATEGIES
# ===================================================================

def backtest_rsi_daily_mr(
    close_daily: pd.Series,
    rsi_period: int = 14,
    oversold: int = 30,
    overbought: int = 70,
    target_vol: float = 0.10,
) -> pd.Series:
    """RSI mean reversion on daily data.

    Long when RSI < oversold, short when RSI > overbought.
    Exit when RSI crosses 50.
    """
    rsi = vbt.RSI.run(close_daily, window=rsi_period).rsi

    # Signal: .shift(1) for no look-ahead
    signal = pd.Series(0.0, index=close_daily.index)
    signal[rsi < oversold] = 1.0
    signal[rsi > overbought] = -1.0
    signal = signal.shift(1)

    # Hold position until RSI crosses 50 (simplified: just use signal)
    # For simplicity, use the direct signal (enter/exit each day)
    daily_ret = close_daily.pct_change()

    vol_21 = daily_ret.rolling(21, min_periods=10).std() * np.sqrt(252)
    lev = (target_vol / vol_21.clip(lower=0.01)).clip(upper=3.0).shift(1)

    return (signal * daily_ret * lev.fillna(1.0)).dropna()


def backtest_rsi_daily_momentum(
    close_daily: pd.Series,
    rsi_period: int = 14,
    threshold: int = 50,
    target_vol: float = 0.10,
) -> pd.Series:
    """RSI momentum on daily data.

    Long when RSI > threshold (bullish momentum), short when RSI < threshold.
    This is the OPPOSITE of mean reversion RSI.
    """
    rsi = vbt.RSI.run(close_daily, window=rsi_period).rsi

    signal = pd.Series(0.0, index=close_daily.index)
    signal[rsi > threshold] = 1.0
    signal[rsi < (100 - threshold)] = -1.0
    signal = signal.shift(1)

    daily_ret = close_daily.pct_change()
    vol_21 = daily_ret.rolling(21, min_periods=10).std() * np.sqrt(252)
    lev = (target_vol / vol_21.clip(lower=0.01)).clip(upper=3.0).shift(1)

    return (signal * daily_ret * lev.fillna(1.0)).dropna()


def part2_rsi_exploration() -> None:
    """RSI strategies on daily timeframe across all pairs."""
    _h("PART 2: RSI Daily Exploration")

    closes = load_daily_closes()
    results = []

    # RSI Mean Reversion
    _sh("RSI Mean Reversion (buy oversold, sell overbought)")
    for pair in FX_PAIRS:
        for period in [7, 14, 21]:
            for os_ob in [(20, 80), (25, 75), (30, 70), (35, 65)]:
                label = f"RSI_MR {pair} p={period} {os_ob[0]}/{os_ob[1]}"

                sharpes = []
                for start, end in WF_PERIODS:
                    try:
                        rets = backtest_rsi_daily_mr(
                            closes[pair].loc[start:end],
                            rsi_period=period,
                            oversold=os_ob[0],
                            overbought=os_ob[1],
                        )
                        sr = (
                            rets.mean() / rets.std() * np.sqrt(252)
                            if rets.std() > 0 else 0.0
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
                }
                results.append(r)

    # RSI Momentum
    _sh("RSI Momentum (buy high RSI, sell low RSI)")
    results_mom = []
    for pair in FX_PAIRS:
        for period in [7, 14, 21]:
            for thresh in [45, 50, 55, 60]:
                label = f"RSI_Mom {pair} p={period} th={thresh}"

                sharpes = []
                for start, end in WF_PERIODS:
                    try:
                        rets = backtest_rsi_daily_momentum(
                            closes[pair].loc[start:end],
                            rsi_period=period,
                            threshold=thresh,
                        )
                        sr = (
                            rets.mean() / rets.std() * np.sqrt(252)
                            if rets.std() > 0 else 0.0
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
                }
                results_mom.append(r)

    # Show top results
    all_rsi = results + results_mom
    all_rsi.sort(key=lambda x: x["avg_sharpe"], reverse=True)
    _sh("Top 15 RSI Strategies (all pairs, all modes)")
    for r in all_rsi[:15]:
        pr(r)

    # Best per pair
    _sh("Best RSI per pair")
    for pair in FX_PAIRS:
        pair_r = [r for r in all_rsi if pair in r["label"]]
        if pair_r:
            print(f"  {pair}:")
            pr(pair_r[0])

    return all_rsi


# ===================================================================
# PART 3: RSI COMBINED WITH MOMENTUM
# ===================================================================

def part3_rsi_combinations() -> None:
    """Combine RSI with momentum signals."""
    _h("PART 3: RSI + Momentum Combinations")

    closes = load_daily_closes()
    results = []

    # RSI filter ON momentum: only follow momentum when RSI confirms
    _sh("RSI as Momentum Filter (trade trend only when RSI confirms)")

    for pair in FX_PAIRS:
        c = closes[pair]

        for rsi_p in [7, 14]:
            rsi = vbt.RSI.run(c, window=rsi_p).rsi.shift(1)

            for fast, slow in [(10, 50), (20, 50), (20, 100)]:
                ef = c.ewm(span=fast, min_periods=fast).mean()
                es = c.ewm(span=slow, min_periods=slow).mean()

                # Pure trend signal
                trend = pd.Series(0.0, index=c.index)
                trend[ef > es] = 1.0
                trend[ef < es] = -1.0
                trend = trend.shift(1)

                # RSI confirmation: trend long + RSI not overbought
                for rsi_filter in [(30, 70), (40, 60), (25, 75)]:
                    rsi_ok_long = rsi < rsi_filter[1]  # not overbought
                    rsi_ok_short = rsi > rsi_filter[0]  # not oversold

                    signal = pd.Series(0.0, index=c.index)
                    signal[(trend > 0) & rsi_ok_long] = 1.0
                    signal[(trend < 0) & rsi_ok_short] = -1.0

                    daily_ret = c.pct_change()
                    vol_21 = (
                        daily_ret.rolling(21, min_periods=10).std()
                        * np.sqrt(252)
                    )
                    lev = (
                        0.10 / vol_21.clip(lower=0.01)
                    ).clip(upper=3.0).shift(1)
                    strat_ret = (signal * daily_ret * lev.fillna(1.0)).dropna()

                    label = (
                        f"Trend+RSI {pair} EMA({fast}/{slow})"
                        f" RSI{rsi_p} {rsi_filter[0]}/{rsi_filter[1]}"
                    )

                    sharpes = []
                    for start, end in WF_PERIODS:
                        sr_slice = strat_ret.loc[start:end]
                        if len(sr_slice) < 20:
                            sharpes.append(0.0)
                            continue
                        sr = (
                            sr_slice.mean() / sr_slice.std() * np.sqrt(252)
                            if sr_slice.std() > 0 else 0.0
                        )
                        sharpes.append(
                            float(sr) if not np.isnan(sr) else 0.0
                        )

                    r = {
                        "label": label,
                        "sharpes": sharpes,
                        "avg_sharpe": float(np.mean(sharpes)),
                        "pos_years": sum(1 for s in sharpes if s > 0),
                        "oos_sharpe": sharpes[-1],
                    }
                    results.append(r)

    results.sort(key=lambda x: x["avg_sharpe"], reverse=True)
    _sh("Top 15 RSI + Momentum Combinations")
    for r in results[:15]:
        pr(r)

    # Compare best combo vs pure trend
    _sh("Best per pair: RSI+Trend vs Pure Trend")
    for pair in FX_PAIRS:
        combo = [r for r in results if pair in r["label"]]
        if combo:
            print(f"  {pair} best RSI+Trend:")
            pr(combo[0])

    return results


# ===================================================================
# PART 4: RSI CROSS-SECTIONAL
# ===================================================================

def part4_rsi_cross_sectional() -> None:
    """Cross-sectional RSI: rank pairs by RSI, long lowest, short highest."""
    _h("PART 4: RSI Cross-Sectional (rank by RSI across pairs)")

    closes = load_daily_closes()
    results = []

    for rsi_p in [7, 14, 21]:
        # Compute RSI for all pairs
        rsi_df = pd.DataFrame({
            pair: vbt.RSI.run(closes[pair], window=rsi_p).rsi
            for pair in FX_PAIRS
        }).shift(1)  # no look-ahead

        # Mode 1: MR — long lowest RSI (oversold), short highest (overbought)
        for mode, mode_name in [(-1.0, "MR"), (1.0, "Mom")]:
            ranks = rsi_df.rank(axis=1) if mode > 0 else (-rsi_df).rank(axis=1)
            n = len(FX_PAIRS)
            weights = ranks.sub(ranks.mean(axis=1), axis=0)
            weights = weights.div(weights.abs().sum(axis=1), axis=0).fillna(0)

            daily_rets = closes.pct_change()
            port_ret = (weights * daily_rets).sum(axis=1).dropna()

            # Vol target
            vol_21 = port_ret.rolling(21, min_periods=10).std() * np.sqrt(252)
            lev = (0.10 / vol_21.clip(lower=0.01)).clip(upper=5.0).shift(1)
            scaled = port_ret * lev.fillna(1.0)

            label = f"RSI_XS_{mode_name} p={rsi_p}"

            sharpes = []
            for start, end in WF_PERIODS:
                s = scaled.loc[start:end]
                if len(s) < 20:
                    sharpes.append(0.0)
                    continue
                sr = s.mean() / s.std() * np.sqrt(252) if s.std() > 0 else 0
                sharpes.append(float(sr) if not np.isnan(sr) else 0.0)

            r = {
                "label": label,
                "sharpes": sharpes,
                "avg_sharpe": float(np.mean(sharpes)),
                "pos_years": sum(1 for s in sharpes if s > 0),
                "oos_sharpe": sharpes[-1],
            }
            results.append(r)
            pr(r)

    return results


# ===================================================================
# MAIN
# ===================================================================

def main() -> None:
    apply_vbt_settings()
    t_start = time.time()

    part1_combined_portfolio()
    rsi_results = part2_rsi_exploration()
    part3_rsi_combinations()
    part4_rsi_cross_sectional()

    print(f"\nTotal time: {(time.time() - t_start) / 60:.1f} min")


if __name__ == "__main__":
    main()
