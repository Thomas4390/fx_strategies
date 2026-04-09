"""Multi-pair, multi-strategy FX research pipeline.

Systematic walk-forward evaluation across 4 FX pairs and multiple strategy
families (MR, SuperTrend, MACD), with macro regime optimization and
portfolio construction.

Phases:
    1. Per-pair baseline classification (MR-responsive vs not)
    2. Parameter sweep for MR-responsive pairs
    3. Alternative strategies (SuperTrend, MACD) for non-MR pairs
    4. Portfolio construction with equal-weight combination
    5. Macro regime optimization across filter combos
"""

from __future__ import annotations

import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import vectorbtpro as vbt

from framework.plotting import (
    plot_monthly_heatmap,
    plot_portfolio_summary,
    show_browser,
)
from strategies.mr_macro import backtest_mr_macro, load_macro_filters
from strategies.mr_turbo import backtest_mr_turbo
from utils import apply_vbt_settings, load_fx_data

# Suppress expected warnings from VBT internals
warnings.filterwarnings("ignore", category=FutureWarning)

# ═══════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

FX_PAIRS = ["EUR-USD", "USD-JPY", "GBP-USD", "USD-CAD"]

WF_PERIODS: list[tuple[str, str]] = [
    (f"{y}-01-01", f"{y}-12-31") for y in range(2019, 2025)
] + [("2025-01-01", "2026-04-01")]

YEAR_LABELS = [str(y) for y in range(2019, 2025)] + ["2025*"]

BB_WINDOWS = [40, 60, 80, 100]
BB_ALPHAS = [4, 5, 6]
SESSIONS = [(6, 14), (7, 15), (8, 16)]
SL_TP_COMBOS = [
    (0.004, 0.006),
    (0.005, 0.006),
    (0.005, 0.008),
    (0.006, 0.008),
]
MACRO_THRESHOLDS = [0.3, 0.5, None]  # None = no macro filter


# ═══════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════


def _safe_sharpe(pf: vbt.Portfolio) -> float:
    """Extract Sharpe ratio, returning 0.0 on failure or no trades."""
    try:
        tc = pf.trades.count()
        if tc == 0:
            return 0.0
        sr = pf.sharpe_ratio
        if pd.isna(sr) or np.isinf(sr):
            return 0.0
        return float(sr)
    except Exception:
        return 0.0


def _safe_total_return(pf: vbt.Portfolio) -> float:
    """Extract total return, returning 0.0 on failure."""
    try:
        tr = pf.total_return
        if pd.isna(tr) or np.isinf(tr):
            return 0.0
        return float(tr)
    except Exception:
        return 0.0


def _safe_trades(pf: vbt.Portfolio) -> int:
    """Extract trade count, returning 0 on failure."""
    try:
        return int(pf.trades.count())
    except Exception:
        return 0


def _load_pair(pair: str) -> tuple[pd.DataFrame, vbt.Data]:
    """Load a single FX pair."""
    path = f"data/{pair}_minute.parquet"
    return load_fx_data(path)


def _slice_data(
    data: vbt.Data,
    start: str,
    end: str,
) -> vbt.Data:
    """Slice VBT Data to a date range."""
    return data.loc[start:end]


def _print_header(title: str) -> None:
    """Print a formatted section header."""
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")


def _print_subheader(title: str) -> None:
    """Print a formatted subsection header."""
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")


# ═══════════════════════════════════════════════════════════════════════
# PHASE 1: PER-PAIR BASELINE
# ═══════════════════════════════════════════════════════════════════════


def phase1_baseline(
    pair_data: dict[str, tuple[pd.DataFrame, vbt.Data]],
) -> tuple[list[str], list[str], dict[str, pd.DataFrame]]:
    """Run per-pair baseline: MR Turbo + MR Macro per year.

    Returns (mr_pairs, non_mr_pairs, results_dict).
    MR-responsive = avg Sharpe > 0 AND >= 4/7 years positive for either strategy.
    """
    _print_header("PHASE 1: Per-Pair Baseline Classification")

    all_results: dict[str, pd.DataFrame] = {}
    mr_pairs: list[str] = []
    non_mr_pairs: list[str] = []

    print(f"\n{'Pair':<12} {'Strategy':<15} {'AvgSR':>7} {'Pos':>5} {'MR?':>5}")
    print("-" * 50)

    for pair in FX_PAIRS:
        _, data = pair_data[pair]
        rows = []

        for strat_name, backtest_fn in [
            ("MR Turbo", backtest_mr_turbo),
            ("MR Macro", backtest_mr_macro),
        ]:
            sharpes = []
            for start, end in WF_PERIODS:
                try:
                    d_yr = _slice_data(data, start, end)
                    if d_yr.shape[0] < 1000:
                        sharpes.append(0.0)
                        continue
                    pf = backtest_fn(d_yr)
                    sharpes.append(_safe_sharpe(pf))
                except Exception:
                    sharpes.append(0.0)

            avg_sr = np.mean(sharpes)
            pos_years = sum(1 for s in sharpes if s > 0)
            rows.append({
                "strategy": strat_name,
                "avg_sharpe": avg_sr,
                "pos_years": pos_years,
                "sharpes": sharpes,
            })
            print(
                f"{pair:<12} {strat_name:<15} {avg_sr:>7.3f} "
                f"{pos_years:>3}/7 {'  Y' if avg_sr > 0 and pos_years >= 4 else '  N'}"
            )

        all_results[pair] = pd.DataFrame(rows)

        # Classify: MR-responsive if EITHER strategy qualifies
        is_mr = any(
            r["avg_sharpe"] > 0 and r["pos_years"] >= 4 for r in rows
        )
        if is_mr:
            mr_pairs.append(pair)
        else:
            non_mr_pairs.append(pair)

    # Print per-year detail
    _print_subheader("Per-Year Sharpe Detail")
    header = f"{'Pair':<12} {'Strategy':<12}"
    for label in YEAR_LABELS:
        header += f" {label:>7}"
    print(header)
    print("-" * (24 + 8 * len(YEAR_LABELS)))

    for pair in FX_PAIRS:
        df = all_results[pair]
        for _, row in df.iterrows():
            line = f"{pair:<12} {row['strategy']:<12}"
            for s in row["sharpes"]:
                line += f" {s:>7.3f}"
            print(line)

    print(f"\nMR-responsive pairs: {mr_pairs}")
    print(f"Non-MR pairs:        {non_mr_pairs}")

    return mr_pairs, non_mr_pairs, all_results


# ═══════════════════════════════════════════════════════════════════════
# PHASE 2: PARAMETER SWEEP FOR MR PAIRS
# ═══════════════════════════════════════════════════════════════════════


def _run_mr_sweep_single(
    data: vbt.Data,
    raw: pd.DataFrame,
    bb_window: int,
    bb_alpha: float,
    session_start: int,
    session_end: int,
    sl_stop: float,
    tp_stop: float,
    macro_th: float | None,
) -> vbt.Portfolio:
    """Run a single MR backtest with given parameters."""
    close = data.close
    vwap = vbt.VWAP.run(
        data.high, data.low, close, data.volume, anchor="D"
    ).vwap

    deviation = close - vwap
    bb = vbt.BBANDS.run(deviation, window=bb_window, alpha=bb_alpha)
    upper = vwap + bb.upper
    lower = vwap + bb.lower

    hours = np.array(close.index.hour)
    session = (hours >= session_start) & (hours < session_end)

    entries = (close < lower) & session
    short_entries = (close > upper) & session

    if macro_th is not None:
        macro_ok = load_macro_filters(close.index, spread_threshold=macro_th)
        entries = entries & macro_ok
        short_entries = short_entries & macro_ok

    return vbt.Portfolio.from_signals(
        data,
        entries=entries,
        exits=False,
        short_entries=short_entries,
        short_exits=False,
        sl_stop=sl_stop,
        tp_stop=tp_stop,
        dt_stop="21:00",
        td_stop="4h",
        slippage=0.00015,
        init_cash=1_000_000,
        freq="1min",
    )


def phase2_mr_sweep(
    pair_data: dict[str, tuple[pd.DataFrame, vbt.Data]],
    mr_pairs: list[str],
) -> dict[str, dict[str, Any]]:
    """Parameter sweep for MR-responsive pairs.

    Sweeps: BB window, BB alpha, session, SL/TP, macro threshold.
    Walk-forward evaluation on top 3 combos per pair.

    Returns dict of {pair: {best_params, wf_sharpes, ...}}.
    """
    _print_header("PHASE 2: Parameter Sweep for MR-Responsive Pairs")

    if not mr_pairs:
        print("  No MR-responsive pairs found. Skipping Phase 2.")
        return {}

    results: dict[str, dict[str, Any]] = {}

    for pair in mr_pairs:
        _print_subheader(f"Sweeping {pair}")
        raw, data = pair_data[pair]
        t0 = time.time()

        sweep_results: list[dict[str, Any]] = []

        for sl, tp in SL_TP_COMBOS:
            for sess_start, sess_end in SESSIONS:
                for macro_th in MACRO_THRESHOLDS:
                    macro_label = (
                        f"sp<{macro_th}" if macro_th is not None else "no_macro"
                    )
                    try:
                        pf = _run_mr_sweep_single(
                            data, raw,
                            bb_window=60,
                            bb_alpha=5.0,
                            session_start=sess_start,
                            session_end=sess_end,
                            sl_stop=sl,
                            tp_stop=tp,
                            macro_th=macro_th,
                        )
                        sr = _safe_sharpe(pf)
                        tc = _safe_trades(pf)
                        sweep_results.append({
                            "bb_window": 60,
                            "bb_alpha": 5.0,
                            "session": f"{sess_start}-{sess_end}",
                            "sl": sl,
                            "tp": tp,
                            "macro": macro_label,
                            "sharpe": sr,
                            "trades": tc,
                        })
                    except Exception as e:
                        sweep_results.append({
                            "bb_window": 60,
                            "bb_alpha": 5.0,
                            "session": f"{sess_start}-{sess_end}",
                            "sl": sl,
                            "tp": tp,
                            "macro": macro_label,
                            "sharpe": 0.0,
                            "trades": 0,
                            "error": str(e),
                        })

        # Now sweep BB params with best session/SL/TP from above
        if sweep_results:
            best_base = max(sweep_results, key=lambda x: x["sharpe"])
            best_sess = best_base["session"]
            best_sl = best_base["sl"]
            best_tp = best_base["tp"]
            best_macro_label = best_base["macro"]
            best_macro_th: float | None = None
            if best_macro_label.startswith("sp<"):
                best_macro_th = float(best_macro_label.replace("sp<", ""))

            sess_parts = best_sess.split("-")
            best_sess_start = int(sess_parts[0])
            best_sess_end = int(sess_parts[1])

            for bbw in BB_WINDOWS:
                for bba in BB_ALPHAS:
                    if bbw == 60 and bba == 5.0:
                        continue  # Already tested
                    try:
                        pf = _run_mr_sweep_single(
                            data, raw,
                            bb_window=bbw,
                            bb_alpha=float(bba),
                            session_start=best_sess_start,
                            session_end=best_sess_end,
                            sl_stop=best_sl,
                            tp_stop=best_tp,
                            macro_th=best_macro_th,
                        )
                        sr = _safe_sharpe(pf)
                        tc = _safe_trades(pf)
                        sweep_results.append({
                            "bb_window": bbw,
                            "bb_alpha": float(bba),
                            "session": best_sess,
                            "sl": best_sl,
                            "tp": best_tp,
                            "macro": best_macro_label,
                            "sharpe": sr,
                            "trades": tc,
                        })
                    except Exception:
                        pass

        # Sort and print top 20
        sweep_results.sort(key=lambda x: x["sharpe"], reverse=True)
        print(f"\n  Sweep completed in {time.time() - t0:.1f}s "
              f"({len(sweep_results)} combos)")

        print(f"\n  {'Rank':>4} {'BBw':>4} {'BBa':>4} {'Sess':>6} "
              f"{'SL':>6} {'TP':>6} {'Macro':>10} {'Sharpe':>8} {'Trades':>7}")
        print(f"  {'-' * 60}")
        for i, r in enumerate(sweep_results[:20]):
            print(
                f"  {i + 1:>4} {r['bb_window']:>4} {r['bb_alpha']:>4.0f} "
                f"{r['session']:>6} {r['sl']:>6.3f} {r['tp']:>6.3f} "
                f"{r['macro']:>10} {r['sharpe']:>8.3f} {r['trades']:>7}"
            )

        # Walk-forward top 3
        _print_subheader(f"Walk-Forward Top 3 for {pair}")
        top3 = sweep_results[:3]

        for rank, combo in enumerate(top3, 1):
            sess_parts = combo["session"].split("-")
            ss, se = int(sess_parts[0]), int(sess_parts[1])
            macro_th_val: float | None = None
            if combo["macro"].startswith("sp<"):
                macro_th_val = float(combo["macro"].replace("sp<", ""))

            wf_sharpes = []
            line = f"  #{rank} (BBw={combo['bb_window']}, "
            line += f"BBa={combo['bb_alpha']:.0f}, "
            line += f"sess={combo['session']}, "
            line += f"SL={combo['sl']}, TP={combo['tp']}, "
            line += f"macro={combo['macro']})"
            print(line)

            yr_line = "     "
            for start, end in WF_PERIODS:
                try:
                    d_yr = _slice_data(data, start, end)
                    if d_yr.shape[0] < 1000:
                        wf_sharpes.append(0.0)
                        yr_line += f" {0.0:>7.3f}"
                        continue
                    pf_yr = _run_mr_sweep_single(
                        d_yr, raw.loc[start:end],
                        bb_window=combo["bb_window"],
                        bb_alpha=combo["bb_alpha"],
                        session_start=ss,
                        session_end=se,
                        sl_stop=combo["sl"],
                        tp_stop=combo["tp"],
                        macro_th=macro_th_val,
                    )
                    sr = _safe_sharpe(pf_yr)
                    wf_sharpes.append(sr)
                    yr_line += f" {sr:>7.3f}"
                except Exception:
                    wf_sharpes.append(0.0)
                    yr_line += f" {0.0:>7.3f}"

            avg_wf = np.mean(wf_sharpes)
            pos_wf = sum(1 for s in wf_sharpes if s > 0)
            print(yr_line + f"  avg={avg_wf:.3f} pos={pos_wf}/7")

        # Store best combo for this pair
        best = sweep_results[0] if sweep_results else None
        results[pair] = {
            "best_params": best,
            "all_results": sweep_results[:20],
        }

    return results


# ═══════════════════════════════════════════════════════════════════════
# PHASE 3: ALTERNATIVE STRATEGIES FOR NON-MR PAIRS
# ═══════════════════════════════════════════════════════════════════════


def _run_supertrend(
    data_1h: vbt.Data,
) -> dict[str, vbt.Portfolio]:
    """Run SuperTrend sweep on 1h data."""
    close_1h = data_1h.close
    high_1h = data_1h.high
    low_1h = data_1h.low

    st = vbt.SUPERTREND.run(
        high_1h,
        low_1h,
        close_1h,
        period=vbt.Param([7, 10, 14]),
        multiplier=vbt.Param([2.0, 3.0, 4.0]),
        param_product=True,
    )

    long_e = ((st.direction == 1) & (st.direction.shift(1) != 1)).fillna(False)
    short_e = ((st.direction == -1) & (st.direction.shift(1) != -1)).fillna(False)
    long_x = (st.direction == -1).fillna(False)
    short_x = (st.direction == 1).fillna(False)

    pf = vbt.Portfolio.from_signals(
        data_1h,
        entries=long_e,
        exits=long_x,
        short_entries=short_e,
        short_exits=short_x,
        sl_stop=0.012,
        slippage=0.0003,
        init_cash=250_000,
        freq="1h",
        chunked="threadpool",
    )
    return {"pf": pf, "st": st}


def _run_macd(
    data_1h: vbt.Data,
) -> dict[str, vbt.Portfolio]:
    """Run MACD sweep on 1h data."""
    close_1h = data_1h.close

    macd = vbt.MACD.run(
        close_1h,
        fast_window=vbt.Param([8, 12]),
        slow_window=vbt.Param([21, 26]),
        signal_window=9,
        param_product=True,
    )

    entries = macd.macd_crossed_above(macd.signal)
    exits = macd.macd_crossed_below(macd.signal)

    pf = vbt.Portfolio.from_signals(
        data_1h,
        entries=entries,
        exits=exits,
        short_entries=exits,
        short_exits=entries,
        sl_stop=0.01,
        slippage=0.0003,
        init_cash=250_000,
        freq="1h",
        chunked="threadpool",
    )
    return {"pf": pf, "macd": macd}


def phase3_alternatives(
    pair_data: dict[str, tuple[pd.DataFrame, vbt.Data]],
    non_mr_pairs: list[str],
) -> dict[str, dict[str, Any]]:
    """Test SuperTrend and MACD on non-MR pairs using 1h data.

    Walk-forward per year, pick best combo per pair.
    Returns dict of {pair: {strategy, best_params, wf_sharpes}}.
    """
    _print_header("PHASE 3: Alternative Strategies for Non-MR Pairs")

    if not non_mr_pairs:
        print("  All pairs are MR-responsive. Skipping Phase 3.")
        return {}

    results: dict[str, dict[str, Any]] = {}

    for pair in non_mr_pairs:
        _print_subheader(f"Testing alternatives for {pair}")
        _, data = pair_data[pair]

        # Resample to 1h
        data_1h = data.resample("1h")

        strategy_results: list[dict[str, Any]] = []

        # SuperTrend
        print("  Running SuperTrend sweep...")
        t0 = time.time()
        try:
            st_result = _run_supertrend(data_1h)
            pf_st = st_result["pf"]

            # Extract per-column Sharpe
            sharpes_st = pf_st.sharpe_ratio
            if isinstance(sharpes_st, pd.Series):
                for col_idx in sharpes_st.index:
                    sr_val = sharpes_st.loc[col_idx]
                    if pd.isna(sr_val) or np.isinf(sr_val):
                        sr_val = 0.0
                    strategy_results.append({
                        "strategy": "SuperTrend",
                        "params": str(col_idx),
                        "sharpe": float(sr_val),
                        "trades": int(pf_st[col_idx].trades.count())
                        if hasattr(pf_st[col_idx], "trades")
                        else 0,
                    })
            else:
                strategy_results.append({
                    "strategy": "SuperTrend",
                    "params": "single",
                    "sharpe": _safe_sharpe(pf_st),
                    "trades": _safe_trades(pf_st),
                })
            print(f"    SuperTrend done in {time.time() - t0:.1f}s")
        except Exception as e:
            print(f"    SuperTrend failed: {e}")

        # MACD
        print("  Running MACD sweep...")
        t0 = time.time()
        try:
            macd_result = _run_macd(data_1h)
            pf_macd = macd_result["pf"]

            sharpes_macd = pf_macd.sharpe_ratio
            if isinstance(sharpes_macd, pd.Series):
                for col_idx in sharpes_macd.index:
                    sr_val = sharpes_macd.loc[col_idx]
                    if pd.isna(sr_val) or np.isinf(sr_val):
                        sr_val = 0.0
                    strategy_results.append({
                        "strategy": "MACD",
                        "params": str(col_idx),
                        "sharpe": float(sr_val),
                        "trades": int(pf_macd[col_idx].trades.count())
                        if hasattr(pf_macd[col_idx], "trades")
                        else 0,
                    })
            else:
                strategy_results.append({
                    "strategy": "MACD",
                    "params": "single",
                    "sharpe": _safe_sharpe(pf_macd),
                    "trades": _safe_trades(pf_macd),
                })
            print(f"    MACD done in {time.time() - t0:.1f}s")
        except Exception as e:
            print(f"    MACD failed: {e}")

        # Sort and print top results
        strategy_results.sort(key=lambda x: x["sharpe"], reverse=True)

        print(f"\n  {'Rank':>4} {'Strategy':>12} {'Params':>30} "
              f"{'Sharpe':>8} {'Trades':>7}")
        print(f"  {'-' * 65}")
        for i, r in enumerate(strategy_results[:10]):
            print(
                f"  {i + 1:>4} {r['strategy']:>12} {r['params']:>30} "
                f"{r['sharpe']:>8.3f} {r['trades']:>7}"
            )

        # Walk-forward best combo
        if strategy_results:
            best = strategy_results[0]
            _print_subheader(f"Walk-Forward Best {best['strategy']} for {pair}")

            wf_sharpes = []
            yr_line = "     "
            for start, end in WF_PERIODS:
                try:
                    d_yr_1h = data_1h.loc[start:end]
                    if d_yr_1h.shape[0] < 100:
                        wf_sharpes.append(0.0)
                        yr_line += f" {0.0:>7.3f}"
                        continue

                    if best["strategy"] == "SuperTrend":
                        res = _run_supertrend(d_yr_1h)
                    else:
                        res = _run_macd(d_yr_1h)

                    pf_yr = res["pf"]
                    sharpes_yr = pf_yr.sharpe_ratio
                    if isinstance(sharpes_yr, pd.Series):
                        best_sr = sharpes_yr.max()
                        if pd.isna(best_sr):
                            best_sr = 0.0
                    else:
                        best_sr = _safe_sharpe(pf_yr)
                    wf_sharpes.append(float(best_sr))
                    yr_line += f" {best_sr:>7.3f}"
                except Exception:
                    wf_sharpes.append(0.0)
                    yr_line += f" {0.0:>7.3f}"

            avg_wf = np.mean(wf_sharpes)
            pos_wf = sum(1 for s in wf_sharpes if s > 0)
            print(yr_line + f"  avg={avg_wf:.3f} pos={pos_wf}/7")

            results[pair] = {
                "strategy": best["strategy"],
                "best_params": best,
                "wf_sharpes": wf_sharpes,
            }

    return results


# ═══════════════════════════════════════════════════════════════════════
# PHASE 4: PORTFOLIO CONSTRUCTION
# ═══════════════════════════════════════════════════════════════════════


def _get_best_backtest_fn(
    pair: str,
    mr_pairs: list[str],
    mr_results: dict[str, dict[str, Any]],
    alt_results: dict[str, dict[str, Any]],
) -> tuple[str, Any]:
    """Return (strategy_name, backtest_callable) for a pair.

    The callable takes (data_minute, data_1h) and returns a Portfolio.
    """
    if pair in mr_pairs:
        params = mr_results.get(pair, {}).get("best_params", None)
        if params is not None:
            sess = params["session"].split("-")
            ss, se = int(sess[0]), int(sess[1])
            macro_th: float | None = None
            if params["macro"].startswith("sp<"):
                macro_th = float(params["macro"].replace("sp<", ""))

            def run_mr(data_min: vbt.Data, _data_1h: vbt.Data) -> vbt.Portfolio:
                return _run_mr_sweep_single(
                    data_min,
                    pd.DataFrame(),  # raw not needed for signal generation
                    bb_window=params["bb_window"],
                    bb_alpha=params["bb_alpha"],
                    session_start=ss,
                    session_end=se,
                    sl_stop=params["sl"],
                    tp_stop=params["tp"],
                    macro_th=macro_th,
                )

            return ("MR", run_mr)

        # Fallback to default MR Macro
        def run_mr_default(data_min: vbt.Data, _: vbt.Data) -> vbt.Portfolio:
            return backtest_mr_macro(data_min)

        return ("MR Macro", run_mr_default)

    # Non-MR pair: use alternative strategy
    alt = alt_results.get(pair, {})
    strat = alt.get("strategy", "SuperTrend")

    if strat == "SuperTrend":

        def run_st(_: vbt.Data, data_1h: vbt.Data) -> vbt.Portfolio:
            res = _run_supertrend(data_1h)
            pf = res["pf"]
            sharpes = pf.sharpe_ratio
            if isinstance(sharpes, pd.Series):
                best_idx = sharpes.idxmax()
                return pf[best_idx]
            return pf

        return ("SuperTrend", run_st)
    else:

        def run_macd(_: vbt.Data, data_1h: vbt.Data) -> vbt.Portfolio:
            res = _run_macd(data_1h)
            pf = res["pf"]
            sharpes = pf.sharpe_ratio
            if isinstance(sharpes, pd.Series):
                best_idx = sharpes.idxmax()
                return pf[best_idx]
            return pf

        return ("MACD", run_macd)


def phase4_portfolio(
    pair_data: dict[str, tuple[pd.DataFrame, vbt.Data]],
    mr_pairs: list[str],
    mr_results: dict[str, dict[str, Any]],
    alt_results: dict[str, dict[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Construct equal-weight portfolio from best strategy per pair.

    For each year, run each pair's best strategy, collect daily returns,
    and compute equal-weight portfolio.

    Returns (year_sharpes_df, portfolio_returns_series).
    """
    _print_header("PHASE 4: Portfolio Construction")

    year_data: list[dict[str, Any]] = []

    # Collect daily returns per pair per year
    all_daily_returns: dict[str, list[pd.Series]] = {p: [] for p in FX_PAIRS}

    for period_idx, (start, end) in enumerate(WF_PERIODS):
        label = YEAR_LABELS[period_idx]
        pair_sharpes: dict[str, float] = {}
        pair_returns: dict[str, pd.Series] = {}

        for pair in FX_PAIRS:
            _, data = pair_data[pair]
            strat_name, backtest_fn = _get_best_backtest_fn(
                pair, mr_pairs, mr_results, alt_results
            )

            try:
                d_yr = _slice_data(data, start, end)
                if d_yr.shape[0] < 1000:
                    pair_sharpes[pair] = 0.0
                    continue

                d_yr_1h = d_yr.resample("1h")
                pf = backtest_fn(d_yr, d_yr_1h)
                sr = _safe_sharpe(pf)
                pair_sharpes[pair] = sr

                # Daily returns for portfolio
                pf_daily = pf.resample("1D")
                daily_ret = pf_daily.returns
                pair_returns[pair] = daily_ret
            except Exception as e:
                pair_sharpes[pair] = 0.0
                print(f"    Warning: {pair} {label} failed: {e}")

        # Equal-weight portfolio
        if pair_returns:
            ret_df = pd.DataFrame(pair_returns)
            portfolio_ret = ret_df.mean(axis=1)

            # Portfolio Sharpe
            if portfolio_ret.std() > 0:
                portfolio_sr = (
                    portfolio_ret.mean()
                    / portfolio_ret.std()
                    * np.sqrt(252)
                )
            else:
                portfolio_sr = 0.0
        else:
            portfolio_sr = 0.0

        row = {"period": label, "portfolio_sharpe": portfolio_sr}
        for pair in FX_PAIRS:
            row[pair] = pair_sharpes.get(pair, 0.0)
        year_data.append(row)

    year_df = pd.DataFrame(year_data)

    # Print comparison table
    header = f"{'Period':<10} "
    for pair in FX_PAIRS:
        header += f"{pair:>10} "
    header += f"{'Portfolio':>10}"
    print(f"\n{header}")
    print("-" * (10 + 11 * (len(FX_PAIRS) + 1)))

    for _, row in year_df.iterrows():
        line = f"{row['period']:<10} "
        for pair in FX_PAIRS:
            line += f"{row[pair]:>10.3f} "
        line += f"{row['portfolio_sharpe']:>10.3f}"
        print(line)

    # Averages
    line = f"{'Average':<10} "
    for pair in FX_PAIRS:
        line += f"{year_df[pair].mean():>10.3f} "
    line += f"{year_df['portfolio_sharpe'].mean():>10.3f}"
    print(line)

    # Build full portfolio returns for final plots
    full_pair_returns: dict[str, pd.Series] = {}
    for pair in FX_PAIRS:
        _, data = pair_data[pair]
        strat_name, backtest_fn = _get_best_backtest_fn(
            pair, mr_pairs, mr_results, alt_results
        )
        try:
            d_1h = data.resample("1h")
            pf = backtest_fn(data, d_1h)
            pf_daily = pf.resample("1D")
            full_pair_returns[pair] = pf_daily.returns
        except Exception as e:
            print(f"  Warning: Full run for {pair} failed: {e}")

    if full_pair_returns:
        full_ret_df = pd.DataFrame(full_pair_returns)
        portfolio_full = full_ret_df.mean(axis=1)
    else:
        portfolio_full = pd.Series(dtype=float)

    return year_df, portfolio_full


# ═══════════════════════════════════════════════════════════════════════
# PHASE 5: MACRO REGIME OPTIMIZATION
# ═══════════════════════════════════════════════════════════════════════


def _build_macro_filter_combo(
    minute_index: pd.DatetimeIndex,
    spread_threshold: float,
    unemp_filter: bool,
    fed_filter: bool,
    data_dir: Path | None = None,
) -> pd.Series:
    """Build combined macro filter from multiple conditions.

    Parameters
    ----------
    minute_index : DatetimeIndex
    spread_threshold : float
    unemp_filter : bool
        If True, require unemployment not rising (3m).
    fed_filter : bool
        If True, require Fed Funds rate not rising (3m).
    data_dir : Path, optional
    """
    if data_dir is None:
        data_dir = _PROJECT_ROOT / "data"

    # Yield spread (always applied)
    spread_df = pd.read_parquet(data_dir / "SPREAD_10Y2Y_daily.parquet")
    spread_df["date"] = pd.to_datetime(spread_df["date"])
    spread = spread_df.set_index("date")["spread_10y2y"]
    spread_min = spread.resample("1D").ffill().reindex(
        minute_index, method="ffill"
    )
    macro_ok = spread_min < spread_threshold

    # Unemployment filter
    if unemp_filter:
        unemp_df = pd.read_parquet(data_dir / "UNEMPLOYMENT_monthly.parquet")
        unemp_df["date"] = pd.to_datetime(unemp_df["date"])
        unemp = unemp_df.set_index("date")["unemployment"]
        unemp_rising = unemp.diff(3) > 0
        unemp_min = (
            unemp_rising.resample("1D")
            .ffill()
            .reindex(minute_index, method="ffill")
        )
        macro_ok = macro_ok & (~unemp_min.fillna(False).astype(bool))

    # Fed Funds filter
    if fed_filter:
        fed_df = pd.read_parquet(data_dir / "FED_FUNDS_monthly.parquet")
        fed_df["date"] = pd.to_datetime(fed_df["date"])
        fed = fed_df.set_index("date")["fed_funds"]
        fed_rising = fed.diff(3) > 0
        fed_min = (
            fed_rising.resample("1D")
            .ffill()
            .reindex(minute_index, method="ffill")
        )
        macro_ok = macro_ok & (~fed_min.fillna(False).astype(bool))

    return macro_ok.fillna(False)


def phase5_macro_optimization(
    pair_data: dict[str, tuple[pd.DataFrame, vbt.Data]],
    mr_pairs: list[str],
    mr_results: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    """Sweep macro regime filter combos across MR pairs.

    5 spread thresholds x 2 unemp x 2 fed = 20 combos.
    Re-runs portfolio with macro-gated entries.

    Returns DataFrame of ranked results.
    """
    _print_header("PHASE 5: Macro Regime Optimization")

    spread_thresholds = [0.0, 0.1, 0.3, 0.5, 1.0]
    unemp_options = [True, False]
    fed_options = [True, False]

    active_pairs = mr_pairs if mr_pairs else FX_PAIRS[:1]

    combo_results: list[dict[str, Any]] = []
    total = len(spread_thresholds) * len(unemp_options) * len(fed_options)
    print(f"  Testing {total} macro filter combos on {len(active_pairs)} pairs...")

    combo_idx = 0
    for sp_th in spread_thresholds:
        for unemp in unemp_options:
            for fed in fed_options:
                combo_idx += 1
                label = (
                    f"sp<{sp_th:.1f}_"
                    f"unemp={'Y' if unemp else 'N'}_"
                    f"fed={'Y' if fed else 'N'}"
                )

                pair_sharpes: dict[str, float] = {}
                pair_returns: dict[str, pd.Series] = {}

                for pair in active_pairs:
                    _, data = pair_data[pair]
                    close = data.close

                    try:
                        macro_ok = _build_macro_filter_combo(
                            close.index,
                            spread_threshold=sp_th,
                            unemp_filter=unemp,
                            fed_filter=fed,
                        )

                        # Get best MR params or defaults
                        params = mr_results.get(pair, {}).get(
                            "best_params", None
                        )
                        if params is not None:
                            sess = params["session"].split("-")
                            ss, se = int(sess[0]), int(sess[1])
                            bbw = params["bb_window"]
                            bba = params["bb_alpha"]
                            sl = params["sl"]
                            tp = params["tp"]
                        else:
                            ss, se = 6, 14
                            bbw, bba = 60, 5.0
                            sl, tp = 0.005, 0.006

                        vwap = vbt.VWAP.run(
                            data.high, data.low, close, data.volume,
                            anchor="D",
                        ).vwap
                        deviation = close - vwap
                        bb = vbt.BBANDS.run(
                            deviation, window=bbw, alpha=bba,
                        )
                        upper = vwap + bb.upper
                        lower = vwap + bb.lower

                        hours = np.array(close.index.hour)
                        session = (hours >= ss) & (hours < se)

                        entries = (close < lower) & session & macro_ok
                        short_entries = (close > upper) & session & macro_ok

                        pf = vbt.Portfolio.from_signals(
                            data,
                            entries=entries,
                            exits=False,
                            short_entries=short_entries,
                            short_exits=False,
                            sl_stop=sl,
                            tp_stop=tp,
                            dt_stop="21:00",
                            td_stop="4h",
                            slippage=0.00015,
                            init_cash=1_000_000,
                            freq="1min",
                        )

                        pair_sharpes[pair] = _safe_sharpe(pf)

                        pf_daily = pf.resample("1D")
                        pair_returns[pair] = pf_daily.returns
                    except Exception:
                        pair_sharpes[pair] = 0.0

                # Portfolio-level metrics
                if pair_returns:
                    ret_df = pd.DataFrame(pair_returns)
                    port_ret = ret_df.mean(axis=1)
                    if port_ret.std() > 0:
                        port_sr = (
                            port_ret.mean()
                            / port_ret.std()
                            * np.sqrt(252)
                        )
                    else:
                        port_sr = 0.0
                else:
                    port_sr = 0.0

                combo_results.append({
                    "combo": label,
                    "spread_th": sp_th,
                    "unemp": unemp,
                    "fed": fed,
                    "portfolio_sharpe": port_sr,
                    **pair_sharpes,
                })

                if combo_idx % 5 == 0:
                    print(f"    {combo_idx}/{total} combos done...")

    # Sort and display
    results_df = pd.DataFrame(combo_results).sort_values(
        "portfolio_sharpe", ascending=False
    )

    print(f"\n  {'Rank':>4} {'Combo':<30} {'PortSR':>8}", end="")
    for pair in active_pairs:
        print(f" {pair:>10}", end="")
    print()
    print(f"  {'-' * (45 + 11 * len(active_pairs))}")

    for i, (_, row) in enumerate(results_df.head(20).iterrows()):
        print(f"  {i + 1:>4} {row['combo']:<30} "
              f"{row['portfolio_sharpe']:>8.3f}", end="")
        for pair in active_pairs:
            val = row.get(pair, 0.0)
            print(f" {val:>10.3f}", end="")
        print()

    return results_df


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════


def main() -> None:
    """Orchestrate all 5 phases of the multi-pair research pipeline."""
    t_start = time.time()

    _print_header("MULTI-PAIR FX RESEARCH PIPELINE")
    print(f"  Pairs: {FX_PAIRS}")
    print(f"  Walk-forward periods: {len(WF_PERIODS)}")
    print(f"  Date range: {WF_PERIODS[0][0]} to {WF_PERIODS[-1][1]}")

    # Apply VBT settings
    apply_vbt_settings()

    # Load all pair data
    _print_subheader("Loading FX Data")
    pair_data: dict[str, tuple[pd.DataFrame, vbt.Data]] = {}
    for pair in FX_PAIRS:
        t0 = time.time()
        raw, data = _load_pair(pair)
        print(f"  {pair}: {data.shape[0]:,} bars ({time.time() - t0:.1f}s)")
        pair_data[pair] = (raw, data)

    # Phase 1
    t_p1 = time.time()
    mr_pairs, non_mr_pairs, baseline_results = phase1_baseline(pair_data)
    print(f"\n  Phase 1 completed in {time.time() - t_p1:.1f}s")

    # Phase 2
    t_p2 = time.time()
    mr_results = phase2_mr_sweep(pair_data, mr_pairs)
    print(f"\n  Phase 2 completed in {time.time() - t_p2:.1f}s")

    # Phase 3
    t_p3 = time.time()
    alt_results = phase3_alternatives(pair_data, non_mr_pairs)
    print(f"\n  Phase 3 completed in {time.time() - t_p3:.1f}s")

    # Phase 4
    t_p4 = time.time()
    year_df, portfolio_returns = phase4_portfolio(
        pair_data, mr_pairs, mr_results, alt_results
    )
    print(f"\n  Phase 4 completed in {time.time() - t_p4:.1f}s")

    # Phase 5
    t_p5 = time.time()
    macro_df = phase5_macro_optimization(pair_data, mr_pairs, mr_results)
    print(f"\n  Phase 5 completed in {time.time() - t_p5:.1f}s")

    # Summary
    _print_header("PIPELINE SUMMARY")
    print(f"  Total runtime: {time.time() - t_start:.1f}s")
    print(f"  MR-responsive pairs: {mr_pairs}")
    print(f"  Non-MR pairs: {non_mr_pairs}")

    if not year_df.empty:
        print(f"\n  Portfolio Sharpe by year:")
        for _, row in year_df.iterrows():
            print(f"    {row['period']:<10} {row['portfolio_sharpe']:>8.3f}")
        print(
            f"    {'Average':<10} "
            f"{year_df['portfolio_sharpe'].mean():>8.3f}"
        )

    if not macro_df.empty:
        best_macro = macro_df.iloc[0]
        print(f"\n  Best macro filter: {best_macro['combo']}")
        print(f"  Best macro portfolio Sharpe: {best_macro['portfolio_sharpe']:.3f}")

    # Final plots
    _print_subheader("Generating Final Plots")

    if len(portfolio_returns) > 0:
        # Build portfolio equity curve
        equity = (1 + portfolio_returns).cumprod()
        fig_equity = equity.vbt.plot()
        fig_equity.update_layout(
            title="Multi-Pair Portfolio Equity (Equal Weight)",
            height=500,
            yaxis_title="Equity (Normalized)",
        )
        show_browser(fig_equity)

        # Monthly heatmap via manual construction
        mo_rets = portfolio_returns.resample("ME").apply(
            lambda x: (1 + x).prod() - 1
        )
        if len(mo_rets) > 0:
            mo_matrix = pd.Series(
                mo_rets.values,
                index=pd.MultiIndex.from_arrays(
                    [mo_rets.index.year, mo_rets.index.month],
                    names=["year", "month"],
                ),
            ).unstack("month")

            import calendar

            mo_matrix.columns = [
                calendar.month_abbr[m] for m in mo_matrix.columns
            ]
            fig_heatmap = mo_matrix.vbt.heatmap(
                is_x_category=True,
                trace_kwargs={
                    "zmid": 0,
                    "colorscale": "RdYlGn",
                    "text": np.round(mo_matrix.values * 100, 1),
                    "texttemplate": "%{text}%",
                },
            )
            fig_heatmap.update_layout(
                title="Multi-Pair Portfolio Monthly Returns (%)",
                height=400,
            )
            show_browser(fig_heatmap)

    print("\nPipeline complete.")


if __name__ == "__main__":
    main()
