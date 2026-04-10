"""Phase 2 and 3: Multi-Timeframe Signals + Entry Quality Research.

Phase 2: Test BB on higher timeframes (5min, 15min, 1H) to reduce noise.
Phase 3: Day-of-week analysis, VWAP anchor, velocity filter.
"""

from __future__ import annotations

import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import vectorbtpro as vbt

from strategies.mr_macro import load_macro_filters
from utils import apply_vbt_settings, load_fx_data

warnings.filterwarnings("ignore", category=FutureWarning)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

WF_PERIODS: list[tuple[str, str]] = [
    (f"{y}-01-01", f"{y}-12-31") for y in range(2019, 2025)
] + [("2025-01-01", "2026-04-01")]
YEAR_LABELS = [str(y) for y in range(2019, 2025)] + ["2025*"]


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


def _print_header(title: str) -> None:
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")


def _print_subheader(title: str) -> None:
    print(f"\n{'-' * 60}")
    print(f"  {title}")
    print(f"{'-' * 60}")


def print_wf_result(result: dict[str, Any]) -> None:
    label = result["label"]
    sharpes = result["sharpes"]
    avg = result["avg_sharpe"]
    pos = result["pos_years"]
    oos = result["oos_sharpe"]
    tc = result["total_trades"]
    detail = " ".join(f"{s:>6.2f}" for s in sharpes)
    print(
        f"  {label:<45} avg={avg:>5.2f} pos={pos}/7"
        f" oos={oos:>5.2f} tc={tc:>4} | {detail}"
    )


# =================================================================
# MULTI-TF BACKTEST
# =================================================================


def backtest_mr_multitf(
    data: vbt.Data,
    macro_filter: pd.Series,
    bb_timeframe: str = "5min",
    bb_window: int = 24,
    bb_alpha: float = 5.0,
    sl_stop: float = 0.005,
    tp_stop: float = 0.006,
    session_start: int = 6,
    session_end: int = 14,
    dt_stop: str = "21:00",
    td_stop: str = "6h",
    slippage: float = 0.00015,
) -> vbt.Portfolio:
    """MR backtest with BB computed on higher timeframe."""
    close = data.close
    minute_index = close.index

    close_tf = close.resample(bb_timeframe).last().dropna()
    high_tf = data.high.resample(bb_timeframe).max().dropna()
    low_tf = data.low.resample(bb_timeframe).min().dropna()
    vol_tf = data.volume.resample(bb_timeframe).sum().dropna()

    common_idx = close_tf.index.intersection(high_tf.index)
    close_tf = close_tf.loc[common_idx]
    high_tf = high_tf.loc[common_idx]
    low_tf = low_tf.loc[common_idx]
    vol_tf = vol_tf.loc[common_idx]

    vwap_tf = vbt.VWAP.run(high_tf, low_tf, close_tf, vol_tf, anchor="D").vwap

    deviation_tf = close_tf - vwap_tf
    bb = vbt.BBANDS.run(deviation_tf, window=bb_window, alpha=bb_alpha)

    upper_tf = vwap_tf + bb.upper
    lower_tf = vwap_tf + bb.lower

    upper_1m = upper_tf.reindex(minute_index, method="ffill")
    lower_1m = lower_tf.reindex(minute_index, method="ffill")

    hours = close.index.hour
    session = (hours >= session_start) & (hours < session_end)
    filt = macro_filter.reindex(minute_index, method="ffill").fillna(False)

    entries = (close < lower_1m) & session & filt
    short_entries = (close > upper_1m) & session & filt

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


def walk_forward_multitf(
    data: vbt.Data,
    macro_filter: pd.Series,
    label: str = "",
    **kwargs: Any,
) -> dict[str, Any]:
    sharpes = []
    trades_list = []

    for start, end in WF_PERIODS:
        try:
            d_yr = data.loc[start:end]
            if d_yr.shape[0] < 1000:
                sharpes.append(0.0)
                trades_list.append(0)
                continue
            pf = backtest_mr_multitf(d_yr, macro_filter, **kwargs)
            sharpes.append(_safe_sharpe(pf))
            trades_list.append(_safe_trades(pf))
        except Exception:
            sharpes.append(0.0)
            trades_list.append(0)

    return {
        "label": label,
        "sharpes": sharpes,
        "avg_sharpe": float(np.mean(sharpes)),
        "pos_years": sum(1 for s in sharpes if s > 0),
        "oos_sharpe": sharpes[-1],
        "total_trades": sum(trades_list),
    }


# =================================================================
# PHASE 2A: BB on Higher Timeframes
# =================================================================


def phase2a_multitf_bb(
    data: vbt.Data,
    macro_filter: pd.Series,
) -> list[dict[str, Any]]:
    """Test BB on 5min, 15min, 1H timeframes.

    Hypothesis: 1-minute BB is too noisy. Higher TF BB produces fewer,
    higher-quality signals.

    Economic rationale: Intraday mean reversion operates on institutional
    order flow timescale (minutes to hours), not tick-by-tick noise.
    """
    _print_header("PHASE 2A: Multi-Timeframe BB")

    results = []

    from research.macro_score_research import walk_forward_eval
    baseline = walk_forward_eval(
        data, macro_filter,
        label="BASELINE 1min BB(80,5.0)",
    )
    print_wf_result(baseline)
    results.append(baseline)

    tf_configs = {
        "5min": {"windows": [16, 24, 32, 48], "alphas": [4.0, 5.0, 6.0]},
        "15min": {"windows": [8, 12, 16, 24], "alphas": [4.0, 5.0, 6.0]},
        "1h": {"windows": [4, 6, 8, 12], "alphas": [3.0, 4.0, 5.0]},
    }

    for tf, config in tf_configs.items():
        _print_subheader(f"Timeframe: {tf}")
        for window in config["windows"]:
            for alpha in config["alphas"]:
                label = f"{tf} BB({window},{alpha})"
                result = walk_forward_multitf(
                    data, macro_filter,
                    label=label,
                    bb_timeframe=tf,
                    bb_window=window,
                    bb_alpha=alpha,
                )
                results.append(result)

    results_sorted = sorted(
        results, key=lambda r: r["avg_sharpe"], reverse=True
    )
    _print_subheader("Top 15 Multi-TF Configurations")
    for r in results_sorted[:15]:
        print_wf_result(r)

    return results_sorted


# =================================================================
# PHASE 2B: HTF Confirmation (RSI on 1H)
# =================================================================


def backtest_mr_htf_confirmation(
    data: vbt.Data,
    macro_filter: pd.Series,
    rsi_period: int = 14,
    rsi_oversold: int = 30,
    rsi_overbought: int = 70,
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
    """MR backtest with RSI confirmation on 1H timeframe."""
    close = data.close
    minute_index = close.index

    vwap = vbt.VWAP.run(data.high, data.low, close, data.volume, anchor="D").vwap
    deviation = close - vwap
    bb = vbt.BBANDS.run(deviation, window=bb_window, alpha=bb_alpha)
    upper = vwap + bb.upper
    lower = vwap + bb.lower

    close_1h = close.resample("1h").last().dropna()
    rsi_1h = vbt.RSI.run(close_1h, window=rsi_period).rsi
    rsi_1m = rsi_1h.reindex(minute_index, method="ffill")

    hours = close.index.hour
    session = (hours >= session_start) & (hours < session_end)
    filt = macro_filter.reindex(minute_index, method="ffill").fillna(False)

    rsi_oversold_mask = rsi_1m < rsi_oversold
    rsi_overbought_mask = rsi_1m > rsi_overbought

    entries = (close < lower) & session & filt & rsi_oversold_mask
    short_entries = (close > upper) & session & filt & rsi_overbought_mask

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


def phase2b_htf_confirmation(
    data: vbt.Data,
    macro_filter: pd.Series,
) -> list[dict[str, Any]]:
    """Test RSI on 1H as entry confirmation.

    Hypothesis: RSI on 1-minute killed performance (too noisy), but RSI
    on 1H captures genuine oversold/overbought conditions.
    """
    _print_header("PHASE 2B: HTF RSI Confirmation (1H)")

    results = []

    configs = [
        {"rsi_period": 14, "rsi_oversold": 25, "rsi_overbought": 75},
        {"rsi_period": 14, "rsi_oversold": 30, "rsi_overbought": 70},
        {"rsi_period": 14, "rsi_oversold": 35, "rsi_overbought": 65},
        {"rsi_period": 14, "rsi_oversold": 40, "rsi_overbought": 60},
        {"rsi_period": 7, "rsi_oversold": 30, "rsi_overbought": 70},
        {"rsi_period": 7, "rsi_oversold": 35, "rsi_overbought": 65},
        {"rsi_period": 21, "rsi_oversold": 30, "rsi_overbought": 70},
        {"rsi_period": 21, "rsi_oversold": 35, "rsi_overbought": 65},
    ]

    for cfg in configs:
        label = (
            f"RSI({cfg['rsi_period']}) "
            f"{cfg['rsi_oversold']}/{cfg['rsi_overbought']}"
        )

        sharpes = []
        trades_list = []
        for start, end in WF_PERIODS:
            try:
                d_yr = data.loc[start:end]
                if d_yr.shape[0] < 1000:
                    sharpes.append(0.0)
                    trades_list.append(0)
                    continue
                pf = backtest_mr_htf_confirmation(d_yr, macro_filter, **cfg)
                sharpes.append(_safe_sharpe(pf))
                trades_list.append(_safe_trades(pf))
            except Exception:
                sharpes.append(0.0)
                trades_list.append(0)

        result = {
            "label": label,
            "sharpes": sharpes,
            "avg_sharpe": float(np.mean(sharpes)),
            "pos_years": sum(1 for s in sharpes if s > 0),
            "oos_sharpe": sharpes[-1],
            "total_trades": sum(trades_list),
        }
        results.append(result)

    results_sorted = sorted(
        results, key=lambda r: r["avg_sharpe"], reverse=True
    )
    _print_subheader("HTF RSI Results (sorted)")
    for r in results_sorted:
        print_wf_result(r)

    return results_sorted


# =================================================================
# PHASE 3A: Day-of-Week Analysis
# =================================================================


def phase3a_day_of_week(
    data: vbt.Data,
    macro_filter: pd.Series,
) -> list[dict[str, Any]]:
    """Analyze strategy performance by day of week.

    Hypothesis: Some weekdays are structurally better for MR due to
    institutional flow patterns.

    Economic rationale: Monday = continuation, Wed/Thu = reversals
    around data releases, Friday = reduced liquidity.
    """
    _print_header("PHASE 3A: Day-of-Week Analysis")

    from research.macro_score_research import (
        backtest_mr_with_filter, walk_forward_eval,
    )

    results = []
    close = data.close
    day_names = ["Mon", "Tue", "Wed", "Thu", "Fri"]

    for day_idx in range(5):
        day_mask = pd.Series(
            close.index.dayofweek == day_idx,
            index=close.index,
        )
        combined_filter = macro_filter & day_mask

        sharpes = []
        trades_list = []
        for start, end in WF_PERIODS:
            try:
                d_yr = data.loc[start:end]
                if d_yr.shape[0] < 1000:
                    sharpes.append(0.0)
                    trades_list.append(0)
                    continue
                pf = backtest_mr_with_filter(d_yr, combined_filter)
                sharpes.append(_safe_sharpe(pf))
                trades_list.append(_safe_trades(pf))
            except Exception:
                sharpes.append(0.0)
                trades_list.append(0)

        result = {
            "label": f"{day_names[day_idx]} only",
            "sharpes": sharpes,
            "avg_sharpe": float(np.mean(sharpes)),
            "pos_years": sum(1 for s in sharpes if s > 0),
            "oos_sharpe": sharpes[-1],
            "total_trades": sum(trades_list),
        }
        results.append(result)

    _print_subheader("Per-Day Performance")
    for r in results:
        print_wf_result(r)

    day_sharpes = {
        r["label"].split()[0]: r["avg_sharpe"] for r in results
    }
    worst_day = min(day_sharpes, key=day_sharpes.get)
    worst_day_idx = day_names.index(worst_day)

    _print_subheader("Day Exclusion Tests")
    print(f"  Worst day: {worst_day} (Sharpe {day_sharpes[worst_day]:.2f})")

    excl_mask = pd.Series(
        close.index.dayofweek != worst_day_idx,
        index=close.index,
    )
    result_excl = walk_forward_eval(
        data, macro_filter & excl_mask,
        label=f"Exclude {worst_day}",
    )
    print_wf_result(result_excl)
    results.append(result_excl)

    sorted_days = sorted(day_sharpes.items(), key=lambda x: x[1])
    worst_two = [day_names.index(d) for d, _ in sorted_days[:2]]
    excl2_mask = pd.Series(
        ~close.index.dayofweek.isin(worst_two),
        index=close.index,
    )
    result_excl2 = walk_forward_eval(
        data, macro_filter & excl2_mask,
        label=f"Exclude {sorted_days[0][0]}+{sorted_days[1][0]}",
    )
    print_wf_result(result_excl2)
    results.append(result_excl2)

    return results


# =================================================================
# PHASE 3B: VWAP Anchor at NY Session
# =================================================================


def phase3b_vwap_anchor(
    data: vbt.Data,
    macro_filter: pd.Series,
) -> list[dict[str, Any]]:
    """Test VWAP anchor at NY session open (22:00 UTC) vs midnight.

    Hypothesis: FX day begins at 5pm ET = 22:00 UTC. VWAP anchored to
    this boundary is more meaningful than midnight UTC.
    """
    _print_header("PHASE 3B: VWAP Anchor Test")

    from research.macro_score_research import walk_forward_eval

    results = []

    baseline = walk_forward_eval(
        data, macro_filter,
        label="BASELINE anchor=D (midnight)",
    )
    print_wf_result(baseline)
    results.append(baseline)

    for shift_hours in [2, 3, 5]:
        anchor_hour = 24 - shift_hours
        label = f"anchor ~{anchor_hour}:00 UTC (shift +{shift_hours}h)"

        sharpes = []
        trades_list = []

        for start, end in WF_PERIODS:
            try:
                d_yr = data.loc[start:end]
                if d_yr.shape[0] < 1000:
                    sharpes.append(0.0)
                    trades_list.append(0)
                    continue

                close = d_yr.close
                shifted_idx = close.index + pd.Timedelta(hours=shift_hours)

                close_s = close.copy()
                close_s.index = shifted_idx
                high_s = d_yr.high.copy()
                high_s.index = shifted_idx
                low_s = d_yr.low.copy()
                low_s.index = shifted_idx
                vol_s = d_yr.volume.copy()
                vol_s.index = shifted_idx

                vwap_s = vbt.VWAP.run(
                    high_s, low_s, close_s, vol_s, anchor="D",
                ).vwap
                vwap_s.index = close.index

                deviation = close - vwap_s
                bb = vbt.BBANDS.run(deviation, window=80, alpha=5.0)
                upper = vwap_s + bb.upper
                lower = vwap_s + bb.lower

                hours_arr = close.index.hour
                session = (hours_arr >= 6) & (hours_arr < 14)
                filt = macro_filter.reindex(
                    close.index, method="ffill"
                ).fillna(False)

                entries = (close < lower) & session & filt
                short_entries = (close > upper) & session & filt

                pf = vbt.Portfolio.from_signals(
                    d_yr,
                    entries=entries,
                    exits=False,
                    short_entries=short_entries,
                    short_exits=False,
                    sl_stop=0.005,
                    tp_stop=0.006,
                    dt_stop="21:00",
                    td_stop="6h",
                    slippage=0.00015,
                    init_cash=1_000_000,
                    freq="1min",
                )
                sharpes.append(_safe_sharpe(pf))
                trades_list.append(_safe_trades(pf))
            except Exception:
                sharpes.append(0.0)
                trades_list.append(0)

        result = {
            "label": label,
            "sharpes": sharpes,
            "avg_sharpe": float(np.mean(sharpes)),
            "pos_years": sum(1 for s in sharpes if s > 0),
            "oos_sharpe": sharpes[-1],
            "total_trades": sum(trades_list),
        }
        results.append(result)
        print_wf_result(result)

    return results


# =================================================================
# PHASE 3C: Deviation Velocity Filter
# =================================================================


def phase3c_velocity_filter(
    data: vbt.Data,
    macro_filter: pd.Series,
) -> list[dict[str, Any]]:
    """Test deviation velocity filter to avoid catching falling knives.

    Hypothesis: Entering when deviation is decelerating produces better
    entries than entering at any BB breach.
    """
    _print_header("PHASE 3C: Deviation Velocity Filter")

    results = []

    for velocity_lookback in [5, 10, 20]:
        for vel_threshold in [0.0, 0.00005, 0.0001, 0.0002]:
            label = f"vel_lb={velocity_lookback} thr={vel_threshold:.5f}"

            sharpes = []
            trades_list = []

            for start, end in WF_PERIODS:
                try:
                    d_yr = data.loc[start:end]
                    if d_yr.shape[0] < 1000:
                        sharpes.append(0.0)
                        trades_list.append(0)
                        continue

                    close = d_yr.close
                    vwap = vbt.VWAP.run(
                        d_yr.high, d_yr.low, close,
                        d_yr.volume, anchor="D",
                    ).vwap
                    deviation = close - vwap
                    bb = vbt.BBANDS.run(deviation, window=80, alpha=5.0)
                    upper = vwap + bb.upper
                    lower = vwap + bb.lower

                    dev_velocity = (
                        deviation.diff(velocity_lookback) / velocity_lookback
                    )

                    hours = close.index.hour
                    session = (hours >= 6) & (hours < 14)
                    filt = macro_filter.reindex(
                        close.index, method="ffill"
                    ).fillna(False)

                    long_vel_ok = dev_velocity > -vel_threshold
                    short_vel_ok = dev_velocity < vel_threshold

                    entries = (
                        (close < lower) & session & filt & long_vel_ok
                    )
                    short_entries = (
                        (close > upper) & session & filt & short_vel_ok
                    )

                    pf = vbt.Portfolio.from_signals(
                        d_yr,
                        entries=entries,
                        exits=False,
                        short_entries=short_entries,
                        short_exits=False,
                        sl_stop=0.005,
                        tp_stop=0.006,
                        dt_stop="21:00",
                        td_stop="6h",
                        slippage=0.00015,
                        init_cash=1_000_000,
                        freq="1min",
                    )
                    sharpes.append(_safe_sharpe(pf))
                    trades_list.append(_safe_trades(pf))
                except Exception:
                    sharpes.append(0.0)
                    trades_list.append(0)

            result = {
                "label": label,
                "sharpes": sharpes,
                "avg_sharpe": float(np.mean(sharpes)),
                "pos_years": sum(1 for s in sharpes if s > 0),
                "oos_sharpe": sharpes[-1],
                "total_trades": sum(trades_list),
            }
            results.append(result)

    results_sorted = sorted(
        results, key=lambda r: r["avg_sharpe"], reverse=True
    )
    _print_subheader("Velocity Filter Results (top 10)")
    for r in results_sorted[:10]:
        print_wf_result(r)

    return results_sorted


# =================================================================
# MAIN
# =================================================================


def main() -> None:
    apply_vbt_settings()
    vbt.settings.returns.year_freq = pd.Timedelta(hours=24) * 252

    print("Loading data...")
    _, data = load_fx_data()
    print(f"  {data.shape[0]:,} bars")

    macro_filter = load_macro_filters(data.close.index, spread_threshold=0.5)

    t_start = time.time()

    phase2a_multitf_bb(data, macro_filter)
    phase2b_htf_confirmation(data, macro_filter)
    phase3a_day_of_week(data, macro_filter)
    phase3b_vwap_anchor(data, macro_filter)
    phase3c_velocity_filter(data, macro_filter)

    elapsed = time.time() - t_start
    print(f"\nTotal Phase 2+3 time: {elapsed / 60:.1f} minutes")


if __name__ == "__main__":
    main()
