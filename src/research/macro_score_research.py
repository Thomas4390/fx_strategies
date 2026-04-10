"""Phase 1: Macro Deep-Dive Research.

Systematic exploration of macro regime filtering for EUR-USD BB MR strategy.
Tests dynamic thresholds, composite scores, lead/lag effects, and regime clustering.

Key insight from prior research: macro filter IS the alpha (>90% of returns).
Without filter: Sharpe 0.08. With spread<0.5 + unemp stable: Sharpe 0.94.
"""

from __future__ import annotations

import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import vectorbtpro as vbt

from strategies.mr_turbo import backtest_mr_turbo
from utils import apply_vbt_settings, load_fx_data

warnings.filterwarnings("ignore", category=FutureWarning)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_DATA_DIR = _PROJECT_ROOT / "data"

# Walk-forward periods (7 annual periods 2019-2025)
WF_PERIODS: list[tuple[str, str]] = [
    (f"{y}-01-01", f"{y}-12-31") for y in range(2019, 2025)
] + [("2025-01-01", "2026-04-01")]
YEAR_LABELS = [str(y) for y in range(2019, 2025)] + ["2025*"]


# ===================================================================
# HELPERS
# ===================================================================


def _safe_sharpe(pf: vbt.Portfolio) -> float:
    try:
        tc = pf.trades.count()
        if tc == 0:
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


def _safe_total_return(pf: vbt.Portfolio) -> float:
    try:
        tr = pf.total_return
        return 0.0 if (pd.isna(tr) or np.isinf(tr)) else float(tr)
    except Exception:
        return 0.0


def _safe_win_rate(pf: vbt.Portfolio) -> float:
    try:
        tc = pf.trades.count()
        if tc == 0:
            return 0.0
        wr = pf.trades.win_rate
        return 0.0 if (pd.isna(wr) or np.isinf(wr)) else float(wr)
    except Exception:
        return 0.0


def _print_header(title: str) -> None:
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")


def _print_subheader(title: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")


# ===================================================================
# MACRO DATA LOADING (all 9 datasets + derived features)
# ===================================================================


def load_all_macro(data_dir: Path | None = None) -> pd.DataFrame:
    """Load all 9 macro datasets into a single daily DataFrame.

    Returns DataFrame indexed by date with columns for each raw variable
    plus derived features (momentum, diffs, YoY changes).
    """
    if data_dir is None:
        data_dir = _DATA_DIR

    # Daily data
    spread = pd.read_parquet(data_dir / "SPREAD_10Y2Y_daily.parquet")
    spread["date"] = pd.to_datetime(spread["date"])
    spread = spread.set_index("date")

    dgs10 = pd.read_parquet(data_dir / "DGS10_daily.parquet")
    dgs10["date"] = pd.to_datetime(dgs10["date"])
    dgs10 = dgs10.set_index("date")

    dgs2 = pd.read_parquet(data_dir / "DGS2_daily.parquet")
    dgs2["date"] = pd.to_datetime(dgs2["date"])
    dgs2 = dgs2.set_index("date")

    # Monthly data -- resample to daily via forward-fill
    monthly_files = {
        "unemployment": ("UNEMPLOYMENT_monthly.parquet", "unemployment"),
        "fed_funds": ("FED_FUNDS_monthly.parquet", "fed_funds"),
        "cpi": ("CPI_monthly.parquet", "cpi"),
        "core_cpi": ("CPI_CORE_monthly.parquet", "core_cpi"),
        "nfp": ("NFP_monthly.parquet", "nfp"),
        "pce": ("PCE_monthly.parquet", "pce"),
    }

    monthly_dfs = {}
    for key, (filename, col) in monthly_files.items():
        df = pd.read_parquet(data_dir / filename)
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
        monthly_dfs[key] = df[col]

    # Build unified daily DataFrame
    daily_index = spread.index
    macro = pd.DataFrame(index=daily_index)

    # Raw daily
    macro["spread"] = spread["spread_10y2y"]
    macro["dgs10"] = dgs10["dgs10"].reindex(daily_index, method="ffill")
    macro["dgs2"] = dgs2["dgs2"].reindex(daily_index, method="ffill")

    # Raw monthly (forward-filled to daily)
    for key, series in monthly_dfs.items():
        macro[key] = series.reindex(daily_index, method="ffill")

    # Derived features
    macro["spread_mom20"] = macro["spread"].diff(20)
    macro["dgs10_mom20"] = macro["dgs10"].diff(20)
    macro["dgs2_mom20"] = macro["dgs2"].diff(20)
    macro["unemp_diff3m"] = macro["unemployment"].diff(63)  # ~3 months trading days
    macro["unemp_diff6m"] = macro["unemployment"].diff(126)
    macro["fed_diff3m"] = macro["fed_funds"].diff(63)
    macro["cpi_yoy"] = macro["cpi"].pct_change(252) * 100  # ~1 year
    macro["core_cpi_yoy"] = macro["core_cpi"].pct_change(252) * 100
    macro["nfp_3m_chg"] = macro["nfp"].diff(63)
    macro["pce_yoy"] = macro["pce"].pct_change(252) * 100

    return macro


def align_macro_to_minute(
    macro_daily: pd.DataFrame | pd.Series,
    minute_index: pd.DatetimeIndex,
) -> pd.DataFrame | pd.Series:
    """Forward-fill daily macro data to minute frequency."""
    return macro_daily.reindex(minute_index, method="ffill")


# ===================================================================
# BACKTEST WITH CUSTOM FILTER
# ===================================================================


def backtest_mr_with_filter(
    data: vbt.Data,
    macro_filter: pd.Series,
    bb_window: int = 80,
    bb_alpha: float = 5.0,
    sl_stop: float = 0.005,
    tp_stop: float = 0.006,
    session_start: int = 6,
    session_end: int = 14,
    dt_stop: str = "21:00",
    td_stop: str = "6h",
    slippage: float = 0.00015,
) -> vbt.Portfolio:
    """Run MR backtest with a pre-computed boolean macro filter.

    This is the core function: identical to backtest_mr_macro but accepts
    any boolean Series as filter instead of hardcoding load_macro_filters().
    """
    close = data.close

    # Native VWAP
    vwap = vbt.VWAP.run(data.high, data.low, close, data.volume, anchor="D").vwap

    # Bollinger Bands on deviation
    deviation = close - vwap
    bb = vbt.BBANDS.run(deviation, window=bb_window, alpha=bb_alpha)
    upper = vwap + bb.upper
    lower = vwap + bb.lower

    # Session filter
    hours = close.index.hour
    session = (hours >= session_start) & (hours < session_end)

    # Align filter to data index
    filt = macro_filter.reindex(close.index, method="ffill").fillna(False)

    # Entries
    entries = (close < lower) & session & filt
    short_entries = (close > upper) & session & filt

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
        init_cash=1_000_000,
        freq="1min",
    )


# ===================================================================
# WALK-FORWARD EVALUATION
# ===================================================================


def walk_forward_eval(
    data: vbt.Data,
    macro_filter: pd.Series,
    label: str = "",
    **backtest_kwargs: Any,
) -> dict[str, Any]:
    """Run walk-forward across 7 annual periods.

    Returns dict with sharpes, avg_sharpe, pos_years, oos_sharpe, total_trades.
    """
    sharpes = []
    trades_list = []
    returns_list = []

    for start, end in WF_PERIODS:
        try:
            d_yr = data.loc[start:end]
            if d_yr.shape[0] < 1000:
                sharpes.append(0.0)
                trades_list.append(0)
                returns_list.append(0.0)
                continue
            pf = backtest_mr_with_filter(d_yr, macro_filter, **backtest_kwargs)
            sharpes.append(_safe_sharpe(pf))
            trades_list.append(_safe_trades(pf))
            returns_list.append(_safe_total_return(pf))
        except Exception:
            sharpes.append(0.0)
            trades_list.append(0)
            returns_list.append(0.0)

    avg_sharpe = float(np.mean(sharpes))
    pos_years = sum(1 for s in sharpes if s > 0)
    oos_sharpe = sharpes[-1]  # 2025 OOS

    return {
        "label": label,
        "sharpes": sharpes,
        "avg_sharpe": avg_sharpe,
        "pos_years": pos_years,
        "oos_sharpe": oos_sharpe,
        "total_trades": sum(trades_list),
        "returns": returns_list,
    }


def print_wf_result(result: dict[str, Any]) -> None:
    """Print a walk-forward result in table format."""
    label = result["label"]
    sharpes = result["sharpes"]
    avg = result["avg_sharpe"]
    pos = result["pos_years"]
    oos = result["oos_sharpe"]
    tc = result["total_trades"]

    detail = " ".join(f"{s:>6.2f}" for s in sharpes)
    print(
        f"  {label:<40} avg={avg:>5.2f} pos={pos}/7"
        f" oos={oos:>5.2f} tc={tc:>4} | {detail}"
    )


# ===================================================================
# 1A: DYNAMIC THRESHOLDS (rolling percentile)
# ===================================================================


def compute_dynamic_filter(
    macro_daily: pd.DataFrame,
    spread_window: int = 252,
    spread_pctile: int = 30,
    unemp_window: int = 252,
    unemp_pctile: int = 50,
) -> pd.Series:
    """Build dynamic macro filter using rolling percentile thresholds.

    Instead of fixed spread < 0.5, uses spread < rolling_quantile(spread, q).
    Similarly for unemployment diff.
    """
    spread = macro_daily["spread"]
    unemp_diff = macro_daily["unemp_diff3m"]

    # Rolling percentile thresholds
    spread_thresh = spread.rolling(spread_window, min_periods=60).quantile(
        spread_pctile / 100
    )
    unemp_thresh = unemp_diff.rolling(unemp_window, min_periods=60).quantile(
        unemp_pctile / 100
    )

    # Filter: spread below its q-th percentile AND unemp diff below its q-th pctile
    spread_ok = spread < spread_thresh
    unemp_ok = unemp_diff < unemp_thresh

    return (spread_ok & unemp_ok).fillna(False)


def phase1a_dynamic_thresholds(
    data: vbt.Data,
    macro_daily: pd.DataFrame,
) -> list[dict[str, Any]]:
    """Test 1A: Dynamic thresholds via rolling percentile.

    Hypothesis: Fixed spread<0.5 is suboptimal because spread range varies
    enormously (from -0.5 to +2.5). Rolling percentile adapts to current regime.
    """
    _print_header("PHASE 1A: Dynamic Thresholds (Rolling Percentile)")

    results = []

    # Baseline: current fixed filter
    from strategies.mr_macro import load_macro_filters
    baseline_filter = load_macro_filters(data.close.index, spread_threshold=0.5)
    baseline = walk_forward_eval(data, baseline_filter, label="BASELINE sp<0.5+unemp")
    print_wf_result(baseline)
    results.append(baseline)

    # No filter baseline
    no_filter = pd.Series(True, index=data.close.index)
    nf_result = walk_forward_eval(data, no_filter, label="NO FILTER")
    print_wf_result(nf_result)
    results.append(nf_result)

    # Dynamic threshold sweep
    for sp_window in [126, 252, 504]:
        for sp_pctile in [20, 30, 40, 50]:
            for un_pctile in [30, 40, 50, 60]:
                dyn_filter = compute_dynamic_filter(
                    macro_daily,
                    spread_window=sp_window,
                    spread_pctile=sp_pctile,
                    unemp_pctile=un_pctile,
                )
                dyn_filter_min = align_macro_to_minute(
                    dyn_filter, data.close.index
                )

                label = f"dyn sp_w={sp_window} sp_q={sp_pctile} un_q={un_pctile}"
                result = walk_forward_eval(data, dyn_filter_min, label=label)
                results.append(result)

    # Sort by avg Sharpe and show top 10
    results_sorted = sorted(
        results, key=lambda r: r["avg_sharpe"], reverse=True
    )
    _print_subheader("Top 10 Configurations")
    for r in results_sorted[:10]:
        print_wf_result(r)

    return results_sorted


# ===================================================================
# 1B: COMPOSITE MACRO SCORE
# ===================================================================


def compute_composite_score(
    macro_daily: pd.DataFrame,
    weights: dict[str, float],
    zscore_window: int = 252,
) -> pd.Series:
    """Build continuous composite macro score from weighted z-scored features.

    Lower score = more favorable for MR trading.
    Uses expanding window for first year, then rolling.
    """
    available = [f for f in weights if f in macro_daily.columns]
    if not available:
        return pd.Series(0.0, index=macro_daily.index)

    score = pd.Series(0.0, index=macro_daily.index)

    for feat in available:
        series = macro_daily[feat]
        # Rolling z-score
        roll_mean = series.rolling(zscore_window, min_periods=60).mean()
        roll_std = series.rolling(zscore_window, min_periods=60).std()
        z = (series - roll_mean) / roll_std.clip(lower=1e-10)

        score = score + weights[feat] * z.fillna(0)

    return score


def phase1b_composite_score(
    data: vbt.Data,
    macro_daily: pd.DataFrame,
) -> list[dict[str, Any]]:
    """Test 1B: Composite macro score as continuous filter.

    Hypothesis: A weighted combination of all macro variables captures more
    information than binary AND of two variables.

    Economic rationale: financial markets respond to a constellation of
    macro conditions, not just yield curve + unemployment. A composite score
    captures the joint state of the economy.
    """
    _print_header("PHASE 1B: Composite Macro Score")

    results = []

    # Weight configurations to test
    weight_configs = {
        "spread_heavy": {
            "spread": 0.40, "unemp_diff3m": 0.25, "fed_diff3m": 0.15,
            "cpi_yoy": 0.10, "dgs10_mom20": 0.05, "pce_yoy": 0.05,
        },
        "balanced": {
            "spread": 0.25, "unemp_diff3m": 0.25, "fed_diff3m": 0.20,
            "cpi_yoy": 0.10, "core_cpi_yoy": 0.05, "dgs10_mom20": 0.05,
            "nfp_3m_chg": 0.05, "pce_yoy": 0.05,
        },
        "unemp_heavy": {
            "spread": 0.20, "unemp_diff3m": 0.35, "unemp_diff6m": 0.10,
            "fed_diff3m": 0.15, "cpi_yoy": 0.10, "nfp_3m_chg": 0.10,
        },
        "rates_focused": {
            "spread": 0.25, "dgs10": 0.15, "dgs2": 0.15,
            "dgs10_mom20": 0.15, "fed_diff3m": 0.15, "unemp_diff3m": 0.15,
        },
        "inflation_focused": {
            "cpi_yoy": 0.25, "core_cpi_yoy": 0.25, "pce_yoy": 0.20,
            "spread": 0.15, "unemp_diff3m": 0.15,
        },
    }

    for config_name, weights in weight_configs.items():
        score = compute_composite_score(macro_daily, weights, zscore_window=252)

        # Threshold sweep: trade when score is LOW (favorable macro)
        for q in [20, 30, 40, 50]:
            threshold = score.rolling(252, min_periods=60).quantile(q / 100)
            filt = (score < threshold).fillna(False)
            filt_min = align_macro_to_minute(filt, data.close.index)

            label = f"{config_name} q<{q}"
            result = walk_forward_eval(data, filt_min, label=label)
            results.append(result)

    results_sorted = sorted(
        results, key=lambda r: r["avg_sharpe"], reverse=True
    )
    _print_subheader("Top 10 Composite Score Configs")
    for r in results_sorted[:10]:
        print_wf_result(r)

    return results_sorted


# ===================================================================
# 1C: LEAD/LAG MACRO EFFECTS
# ===================================================================


def phase1c_lead_lag(
    data: vbt.Data,
    macro_daily: pd.DataFrame,
) -> list[dict[str, Any]]:
    """Test 1C: Lead/lag effects on macro variables.

    Hypothesis: Macro data is published with delays (NFP first Friday,
    CPI ~15th). Using lagged values may be more realistic AND may reveal
    that the market responds with a delay to macro changes.

    Economic rationale: Markets price in macro changes gradually. A spread
    inversion doesn't instantly change FX dynamics -- it takes weeks for the
    effects to propagate through carry trades and fund flows.
    """
    _print_header("PHASE 1C: Lead/Lag Macro Effects")

    results = []

    # Test lagged versions of the best individual filters
    key_variables = ["spread", "unemp_diff3m", "fed_diff3m", "cpi_yoy"]
    lags = [0, 5, 10, 20]

    for var in key_variables:
        if var not in macro_daily.columns:
            continue
        series = macro_daily[var]

        for lag in lags:
            # Apply lag: shift the series forward (use older data)
            lagged = series.shift(lag) if lag > 0 else series

            # Build a simple filter: variable below its rolling median
            median = lagged.rolling(252, min_periods=60).median()
            filt = (lagged < median).fillna(False)
            filt_min = align_macro_to_minute(filt, data.close.index)

            label = f"{var} lag={lag}d"
            result = walk_forward_eval(data, filt_min, label=label)
            results.append(result)

    # Also test lagged version of the BEST filter (spread<0.5 + unemp)
    for lag in [5, 10, 20]:
        spread_lagged = macro_daily["spread"].shift(lag)
        unemp_diff_lagged = macro_daily["unemp_diff3m"].shift(lag)

        filt = ((spread_lagged < 0.5) & (unemp_diff_lagged < 0)).fillna(False)
        filt_min = align_macro_to_minute(filt, data.close.index)

        label = f"BEST_COMBO lag={lag}d"
        result = walk_forward_eval(data, filt_min, label=label)
        results.append(result)

    results_sorted = sorted(
        results, key=lambda r: r["avg_sharpe"], reverse=True
    )
    _print_subheader("Lead/Lag Results (sorted)")
    for r in results_sorted:
        print_wf_result(r)

    return results_sorted


# ===================================================================
# 1D: REGIME CLUSTERING (K-means)
# ===================================================================


def phase1d_regime_clustering(
    data: vbt.Data,
    macro_daily: pd.DataFrame,
) -> list[dict[str, Any]]:
    """Test 1D: K-means regime clustering on macro features.

    Hypothesis: Natural regimes in the macro data may not align with
    hand-crafted binary thresholds. K-means discovers data-driven regimes.

    Economic rationale: The economy operates in distinct regimes (expansion,
    contraction, transition) that are multivariate -- a single threshold
    on one variable cannot capture the full picture.
    """
    _print_header("PHASE 1D: Regime Clustering (K-Means)")

    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    results = []

    # Features for clustering
    cluster_features = [
        "spread", "dgs10", "dgs2", "spread_mom20",
        "dgs10_mom20", "dgs2_mom20",
        "unemployment", "fed_funds",
        "unemp_diff3m", "fed_diff3m",
    ]
    available_feats = [f for f in cluster_features if f in macro_daily.columns]

    # Prepare feature matrix (drop NaN rows)
    feat_df = macro_daily[available_feats].dropna()

    if len(feat_df) < 252:
        print("  Not enough data for clustering. Skipping.")
        return results

    # Walk-forward clustering: train before year Y, predict year Y
    for n_clusters in [2, 3, 4]:
        _print_subheader(f"K-Means k={n_clusters}")

        daily_filter = pd.Series(False, index=macro_daily.index)

        for start, end in WF_PERIODS:
            train_end = pd.Timestamp(start) - pd.Timedelta(days=1)

            train_data = feat_df.loc[:train_end]
            test_data = feat_df.loc[start:end]

            if len(train_data) < 100 or len(test_data) < 10:
                continue

            scaler = StandardScaler()
            x_train = scaler.fit_transform(train_data.values)
            x_test = scaler.transform(test_data.values)

            km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            km.fit(x_train)

            train_labels = km.labels_
            test_labels = km.predict(x_test)

            # Get unfiltered strategy daily returns on training period
            try:
                d_train = data.loc[:str(train_end)]
                pf_train = backtest_mr_turbo(d_train)
                train_returns = pf_train.daily_returns
            except Exception:
                continue

            # Align returns with training feature dates
            train_rets_aligned = train_returns.reindex(
                train_data.index, method="ffill"
            ).fillna(0)

            # Identify "good" clusters (positive mean return)
            good_clusters = set()
            for c in range(n_clusters):
                mask = train_labels == c
                if mask.sum() > 0:
                    mean_ret = float(train_rets_aligned.iloc[mask].mean())
                    if mean_ret > 0:
                        good_clusters.add(c)

            # Allow trading in test period only in good clusters
            for i, idx in enumerate(test_data.index):
                if test_labels[i] in good_clusters:
                    daily_filter.loc[idx] = True

        filt_min = align_macro_to_minute(daily_filter, data.close.index)

        label = f"KMeans k={n_clusters}"
        result = walk_forward_eval(data, filt_min, label=label)
        print_wf_result(result)
        results.append(result)

        pct_on = daily_filter.sum() / max(len(daily_filter), 1) * 100
        print(f"    Filter active: {pct_on:.1f}% of days")

    return results


# ===================================================================
# 1E: INDIVIDUAL MACRO VARIABLE SWEEP
# ===================================================================


def phase1e_individual_variables(
    data: vbt.Data,
    macro_daily: pd.DataFrame,
) -> list[dict[str, Any]]:
    """Test each macro variable individually with direction analysis.

    This provides granular understanding of which variables have
    predictive power and in which direction.
    """
    _print_header("PHASE 1E: Individual Macro Variable Analysis")

    results = []

    variables = [
        "spread", "dgs10", "dgs2", "unemployment", "fed_funds",
        "cpi_yoy", "core_cpi_yoy", "pce_yoy",
        "spread_mom20", "dgs10_mom20", "dgs2_mom20",
        "unemp_diff3m", "fed_diff3m", "nfp_3m_chg",
    ]

    for var in variables:
        if var not in macro_daily.columns:
            continue

        series = macro_daily[var].dropna()
        if len(series) < 252:
            continue

        # Trade when variable is LOW (below rolling median)
        median = series.rolling(252, min_periods=60).median()
        filt_low = (series < median).fillna(False)
        filt_low_min = align_macro_to_minute(filt_low, data.close.index)

        result = walk_forward_eval(data, filt_low_min, label=f"{var} < median")
        results.append(result)

        # Trade when variable is HIGH (above rolling median)
        filt_high = (series >= median).fillna(False)
        filt_high_min = align_macro_to_minute(filt_high, data.close.index)

        result = walk_forward_eval(
            data, filt_high_min, label=f"{var} >= median"
        )
        results.append(result)

    results_sorted = sorted(
        results, key=lambda r: r["avg_sharpe"], reverse=True
    )
    _print_subheader("Individual Variables (sorted by Sharpe)")
    for r in results_sorted:
        print_wf_result(r)

    return results_sorted


# ===================================================================
# MAIN ENTRY POINT
# ===================================================================


def main() -> None:
    """Run all Phase 1 experiments."""
    apply_vbt_settings()
    vbt.settings.returns.year_freq = pd.Timedelta(hours=24) * 252

    print("Loading EUR-USD minute data...")
    t0 = time.time()
    _, data = load_fx_data()
    print(f"  Loaded in {time.time() - t0:.1f}s ({data.shape[0]:,} bars)")

    print("Loading macro data...")
    macro_daily = load_all_macro()
    print(f"  {len(macro_daily)} daily rows, {len(macro_daily.columns)} features")
    print(f"  Features: {list(macro_daily.columns)}")

    all_results: dict[str, list[dict[str, Any]]] = {}

    t_start = time.time()

    all_results["1e_individual"] = phase1e_individual_variables(data, macro_daily)
    all_results["1a_dynamic"] = phase1a_dynamic_thresholds(data, macro_daily)
    all_results["1b_composite"] = phase1b_composite_score(data, macro_daily)
    all_results["1c_lead_lag"] = phase1c_lead_lag(data, macro_daily)
    all_results["1d_clustering"] = phase1d_regime_clustering(data, macro_daily)

    # SUMMARY
    _print_header("PHASE 1 SUMMARY -- Best Configuration per Experiment")

    for exp_name, exp_results in all_results.items():
        if exp_results:
            best = exp_results[0]  # Already sorted
            print(f"\n  {exp_name}:")
            print_wf_result(best)

    # Overall best across all experiments
    all_flat = []
    for exp_results in all_results.values():
        all_flat.extend(exp_results)
    all_flat.sort(key=lambda r: r["avg_sharpe"], reverse=True)

    _print_header("OVERALL TOP 15 CONFIGURATIONS")
    for r in all_flat[:15]:
        print_wf_result(r)

    elapsed = time.time() - t_start
    print(f"\nTotal research time: {elapsed / 60:.1f} minutes")


if __name__ == "__main__":
    main()
