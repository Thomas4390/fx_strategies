#!/usr/bin/env python
"""
Intraday Donchian Channel Breakout — EUR/USD Minute Data

Full-Numba kernels + VBT PRO pipeline:
- Rolling max/min channels for breakout detection
- Separate entry/exit channel periods
- VWAP confirmation filter
- EOD forced exit
"""

import os
from typing import Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import vectorbtpro as vbt
from numba import njit


# ═══════════════════════════════════════════════════════════════════════
# 0. SETTINGS
# ═══════════════════════════════════════════════════════════════════════

def configure_figure_for_fullscreen(fig):
    fig.update_layout(
        width=None, height=None, autosize=True,
        margin=dict(l=30, r=30, t=60, b=30),
        title=dict(font=dict(size=20), x=0.5, xanchor="center"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
    )
    return fig

vbt.settings.set("plotting.pre_show_func", configure_figure_for_fullscreen)
vbt.settings.returns.year_freq = pd.Timedelta(hours=24) * 252


# ═══════════════════════════════════════════════════════════════════════
# 1. NUMBA KERNELS
# ═══════════════════════════════════════════════════════════════════════

@njit
def find_day_boundaries_nb(index_ns: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int]:
    n = len(index_ns)
    start_idx = np.empty(n, dtype=np.int64)
    end_idx = np.empty(n, dtype=np.int64)
    if n == 0:
        return start_idx, end_idx, 0
    day_number = vbt.dt_nb.days_nb(ts=index_ns)
    current_day = day_number[0]
    day_counter = 0
    current_start = 0
    for i in range(1, n):
        if day_number[i] != current_day:
            start_idx[day_counter] = current_start
            end_idx[day_counter] = i
            day_counter += 1
            current_day = day_number[i]
            current_start = i
    start_idx[day_counter] = current_start
    end_idx[day_counter] = n
    day_counter += 1
    return start_idx, end_idx, day_counter


@njit
def rolling_max_nb(data: np.ndarray, period: int) -> np.ndarray:
    """Rolling max over [i-period, i-1] (lagged by 1 bar)."""
    n = len(data)
    out = np.full(n, np.nan)
    for i in range(period + 1, n):
        mx = data[i - period]
        for j in range(i - period + 1, i):
            if data[j] > mx:
                mx = data[j]
        out[i] = mx
    return out


@njit
def rolling_min_nb(data: np.ndarray, period: int) -> np.ndarray:
    """Rolling min over [i-period, i-1] (lagged by 1 bar)."""
    n = len(data)
    out = np.full(n, np.nan)
    for i in range(period + 1, n):
        mn = data[i - period]
        for j in range(i - period + 1, i):
            if data[j] < mn:
                mn = data[j]
        out[i] = mn
    return out


@njit
def compute_intraday_donchian_nb(
    index_ns: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    open_: np.ndarray,
    volume: np.ndarray,
    entry_period: int,
    exit_period: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Entry channels
    upper = rolling_max_nb(high, entry_period)
    lower = rolling_min_nb(low, entry_period)

    # Exit channels (shorter period)
    exit_upper = rolling_max_nb(high, exit_period)
    exit_lower = rolling_min_nb(low, exit_period)

    # VWAP
    start_arr, end_arr, n_days = find_day_boundaries_nb(index_ns)
    group_lens = end_arr[:n_days] - start_arr[:n_days]
    vwap = vbt.indicators.nb.vwap_1d_nb(high, low, close, volume, group_lens)

    return upper, lower, exit_upper, exit_lower, vwap


# ═══════════════════════════════════════════════════════════════════════
# 2. SIGNAL FUNCTION
# ═══════════════════════════════════════════════════════════════════════

@njit
def intraday_donchian_signal_nb(
    c,
    close_arr: np.ndarray,
    upper_arr: np.ndarray,
    lower_arr: np.ndarray,
    exit_upper_arr: np.ndarray,
    exit_lower_arr: np.ndarray,
    vwap_arr: np.ndarray,
    index_ns_arr: np.ndarray,
    eod_hour_arr: np.ndarray,
    eod_minute_arr: np.ndarray,
):
    ts_ns = index_ns_arr[c.i]
    cur_hour = vbt.dt_nb.hour_nb(ts_ns)
    cur_minute = vbt.dt_nb.minute_nb(ts_ns)

    eod_hour = vbt.pf_nb.select_nb(c, eod_hour_arr)
    eod_minute = vbt.pf_nb.select_nb(c, eod_minute_arr)

    # EOD forced exit
    is_eod = (cur_hour > eod_hour) or (cur_hour == eod_hour and cur_minute >= eod_minute)
    if is_eod:
        el = vbt.pf_nb.ctx_helpers.in_long_position_nb(c)
        es = vbt.pf_nb.ctx_helpers.in_short_position_nb(c)
        return False, el, False, es

    # Evaluate every 15 minutes
    if cur_minute % 15 != 0:
        return False, False, False, False

    px = vbt.pf_nb.select_nb(c, close_arr)
    ub = vbt.pf_nb.select_nb(c, upper_arr)
    lb = vbt.pf_nb.select_nb(c, lower_arr)
    exu = vbt.pf_nb.select_nb(c, exit_upper_arr)
    exl = vbt.pf_nb.select_nb(c, exit_lower_arr)
    vw = vbt.pf_nb.select_nb(c, vwap_arr)

    if np.isnan(px) or np.isnan(ub) or np.isnan(lb) or np.isnan(vw):
        return False, False, False, False

    in_long = vbt.pf_nb.ctx_helpers.in_long_position_nb(c)
    in_short = vbt.pf_nb.ctx_helpers.in_short_position_nb(c)

    if not in_long and not in_short:
        # Breakout entries
        if px > ub and px > vw:
            return True, False, False, False
        elif px < lb and px < vw:
            return False, False, True, False
    elif in_long:
        # Exit on shorter channel break
        if not np.isnan(exl) and px < exl:
            return False, True, False, False
    elif in_short:
        if not np.isnan(exu) and px > exu:
            return False, False, False, True

    return False, False, False, False


# ═══════════════════════════════════════════════════════════════════════
# 3. STANDARD BACKTEST
# ═══════════════════════════════════════════════════════════════════════

def run_standard_backtest(raw, index_ns, entry_period=240, exit_period=60,
                          eod_hour=21, eod_minute=0):
    IDB = vbt.IF(
        class_name="IntradayDonchian", short_name="idb",
        input_names=["index_ns", "high_minute", "low_minute", "close_minute", "open_minute", "volume_minute"],
        param_names=["entry_period", "exit_period"],
        output_names=["upper_channel", "lower_channel", "exit_upper", "exit_lower", "vwap"],
    ).with_apply_func(
        compute_intraday_donchian_nb, takes_1d=True,
        entry_period=entry_period, exit_period=exit_period,
    )

    idb = IDB.run(
        index_ns=index_ns,
        high_minute=raw["high"], low_minute=raw["low"],
        close_minute=raw["close"], open_minute=raw["open"], volume_minute=raw["volume"],
        entry_period=entry_period, exit_period=exit_period,
        jitted_loop=True, jitted_warmup=True,
        execute_kwargs=dict(engine="threadpool", n_chunks="auto"),
    )

    pf = vbt.Portfolio.from_signals(
        raw["close"],
        signal_func_nb=intraday_donchian_signal_nb,
        signal_args=(
            vbt.Rep("close_arr"), vbt.Rep("upper_arr"), vbt.Rep("lower_arr"),
            vbt.Rep("exit_upper_arr"), vbt.Rep("exit_lower_arr"), vbt.Rep("vwap_arr"),
            vbt.Rep("index_arr"), vbt.Rep("eod_hour"), vbt.Rep("eod_minute"),
        ),
        broadcast_named_args=dict(
            close_arr=raw["close"],
            upper_arr=idb.upper_channel.values,
            lower_arr=idb.lower_channel.values,
            exit_upper_arr=idb.exit_upper.values,
            exit_lower_arr=idb.exit_lower.values,
            vwap_arr=idb.vwap.values,
            index_arr=index_ns,
            eod_hour=eod_hour, eod_minute=eod_minute,
        ),
        fixed_fees=0.0035, init_cash=1_000_000, freq="1T",
    )
    return pf, idb


# ═══════════════════════════════════════════════════════════════════════
# 4. CV PIPELINE
# ═══════════════════════════════════════════════════════════════════════

def create_cv_pipeline(splitter):
    def _run(high_arr, low_arr, close_arr, open_arr, volume_arr, idx_ns,
             entry_period, exit_period, eod_hour=21, eod_minute=0,
             metric="sharpe_ratio"):
        close_s = pd.Series(close_arr[:, 0]) if close_arr.ndim > 1 else pd.Series(close_arr)
        high_s = pd.Series(high_arr[:, 0]) if high_arr.ndim > 1 else pd.Series(high_arr)
        low_s = pd.Series(low_arr[:, 0]) if low_arr.ndim > 1 else pd.Series(low_arr)
        open_s = pd.Series(open_arr[:, 0]) if open_arr.ndim > 1 else pd.Series(open_arr)
        vol_s = pd.Series(volume_arr[:, 0]) if volume_arr.ndim > 1 else pd.Series(volume_arr)

        IDB = vbt.IF(
            class_name="IntradayDonchian", short_name="idb",
            input_names=["index_ns", "high_minute", "low_minute", "close_minute", "open_minute", "volume_minute"],
            param_names=["entry_period", "exit_period"],
            output_names=["upper_channel", "lower_channel", "exit_upper", "exit_lower", "vwap"],
        ).with_apply_func(
            compute_intraday_donchian_nb, takes_1d=True,
            entry_period=entry_period, exit_period=exit_period,
        )

        idb = IDB.run(
            index_ns=idx_ns, high_minute=high_s, low_minute=low_s,
            close_minute=close_s, open_minute=open_s, volume_minute=vol_s,
            entry_period=entry_period, exit_period=exit_period,
        )

        pf = vbt.Portfolio.from_signals(
            close_s,
            signal_func_nb=intraday_donchian_signal_nb,
            signal_args=(
                vbt.Rep("close_arr"), vbt.Rep("upper_arr"), vbt.Rep("lower_arr"),
                vbt.Rep("exit_upper_arr"), vbt.Rep("exit_lower_arr"), vbt.Rep("vwap_arr"),
                vbt.Rep("index_arr"), vbt.Rep("eod_hour"), vbt.Rep("eod_minute"),
            ),
            broadcast_named_args=dict(
                close_arr=close_s, upper_arr=idb.upper_channel.values,
                lower_arr=idb.lower_channel.values, exit_upper_arr=idb.exit_upper.values,
                exit_lower_arr=idb.exit_lower.values, vwap_arr=idb.vwap.values,
                index_arr=idx_ns, eod_hour=eod_hour, eod_minute=eod_minute,
            ),
            fixed_fees=0.0035, init_cash=1_000_000, freq="1T",
        )
        return pf.deep_getattr(metric)

    return vbt.cv_split(
        _run, splitter=splitter,
        takeable_args=["high_arr", "low_arr", "close_arr", "open_arr", "volume_arr", "idx_ns"],
        parameterized_kwargs=dict(engine="threadpool", chunk_len="auto"),
        merge_func="concat",
    )


# ═══════════════════════════════════════════════════════════════════════
# 5. PLOTTING
# ═══════════════════════════════════════════════════════════════════════

def plot_monthly_heatmap(pf: vbt.Portfolio) -> go.Figure:
    rets = pf.returns
    monthly = rets.resample("ME").apply(lambda x: (1 + x).prod() - 1)
    df = pd.DataFrame({"year": monthly.index.year, "month": monthly.index.month, "ret": monthly.values})
    pivot = df.pivot(index="year", columns="month", values="ret")
    pivot.columns = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                     "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    fig = go.Figure(data=go.Heatmap(
        z=pivot.values, x=pivot.columns, y=pivot.index.astype(str),
        colorscale="RdYlGn", zmid=0,
        text=np.round(pivot.values * 100, 1), texttemplate="%{text}%",
    ))
    fig.update_layout(title="Donchian Breakout — Monthly Returns (%)", height=400)
    return fig


# ═══════════════════════════════════════════════════════════════════════
# 6. MAIN
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    results_dir = "results/intraday_donchian"
    os.makedirs(results_dir, exist_ok=True)

    print("Loading EUR-USD minute data...")
    raw = pd.read_parquet("data/EUR-USD.parquet")
    raw["date"] = pd.to_datetime(raw["date"])
    raw = raw.set_index("date").sort_index()
    raw["volume"] = 1.0
    index_ns = vbt.dt.to_ns(raw.index)
    print(f"  {len(raw)} bars: {raw.index[0]} → {raw.index[-1]}")

    # ── Standard backtest ──────────────────────────────────────────
    print("\nRunning standard backtest (entry=240, exit=60)...")
    pf, idb = run_standard_backtest(raw, index_ns, entry_period=240, exit_period=60)

    print("\n" + "=" * 60)
    print("INTRADAY DONCHIAN BREAKOUT — STANDARD")
    print("=" * 60)
    print(pf.stats().to_string())
    print("=" * 60)

    fig_pf = pf.plot(subplots=["cumulative_returns", "drawdowns", "underwater"])
    fig_pf.update_layout(title="Donchian Breakout — Portfolio Overview", height=900)
    fig_pf.write_html(f"{results_dir}/portfolio_overview.html")

    fig_monthly = plot_monthly_heatmap(pf)
    fig_monthly.write_html(f"{results_dir}/monthly_returns.html")
    print(f"\nPlots saved to {results_dir}/")

    # ── Cross-validation ───────────────────────────────────────────
    print("\nPreparing CV...")
    high_arr = vbt.to_2d_array(raw["high"])
    low_arr = vbt.to_2d_array(raw["low"])
    close_arr = vbt.to_2d_array(raw["close"])
    open_arr = vbt.to_2d_array(raw["open"])
    volume_arr = vbt.to_2d_array(raw["volume"])

    splitter = vbt.Splitter.from_n_rolling(
        raw.index, n=5, length=1_500_000, split=0.5, set_labels=["train", "test"],
    )
    cv = create_cv_pipeline(splitter)

    n_combos = 4 * 3
    print(f"Running CV: {n_combos} combos x 5 splits = {n_combos * 5} backtests...")

    grid_perf, best_perf = cv(
        high_arr=high_arr, low_arr=low_arr, close_arr=close_arr,
        open_arr=open_arr, volume_arr=volume_arr, idx_ns=index_ns,
        entry_period=vbt.Param([120, 240, 480, 720]),
        exit_period=vbt.Param([30, 60, 120]),
        eod_hour=21, eod_minute=0, metric="sharpe_ratio",
        _return_grid="all", _index=raw.index,
    )

    print(f"  Grid shape: {grid_perf.shape}")
    print(f"  Best perf shape: {best_perf.shape}")

    fig_cv = grid_perf.vbt.heatmap(
        x_level="entry_period", y_level="exit_period", slider_level="split",
    )
    fig_cv.write_html(f"{results_dir}/cv_heatmap.html")

    if isinstance(best_perf.index, pd.MultiIndex):
        print(f"\nBest params: {best_perf.idxmax()}")
        print(f"Best Sharpe: {best_perf.max():.4f}")

    print(f"\nAll results saved to {results_dir}/")
