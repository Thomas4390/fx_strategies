#!/usr/bin/env python
"""
Multi-Strategy Walk-Forward CV — True Multicore via Numba prange

- prange parallel loops for real 32-core utilization
- @vbt.cv_split with attach_bounds="index" for date-labeled splits
- slider_level="start" for temporal heatmap navigation
- Fullscreen browser plots
"""

import os
import sys
import warnings

import numba
import numpy as np
import pandas as pd
import vectorbtpro as vbt
from numba import njit, prange

warnings.filterwarnings("ignore")
os.environ["NUMBA_DISABLE_PERFORMANCE_WARNINGS"] = "1"

sys.path.insert(0, os.path.dirname(__file__))
from utils import load_fx_data

RESULTS_DIR = "results/exploration/cv"
SLIPPAGE = 0.00008
FEES = 0.0001
INIT_CASH = 1_000_000


def fullscreen(fig, title="", height=900):
    fig.update_layout(
        width=None, height=height, autosize=True,
        title={"text": title, "font": {"size": 22}, "x": 0.5, "xanchor": "center"},
        margin={"l": 40, "r": 40, "t": 80, "b": 40},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0.5, "xanchor": "center"},
        template="plotly_white",
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════
# NUMBA KERNELS
# ═══════════════════════════════════════════════════════════════════════


@njit(nogil=True)
def rsi_signal_nb(c, rsi_arr, entry_th_arr, exit_th_arr):
    rsi_val = vbt.pf_nb.select_nb(c, rsi_arr)
    entry_th = vbt.pf_nb.select_nb(c, entry_th_arr)
    exit_th = vbt.pf_nb.select_nb(c, exit_th_arr)
    if np.isnan(rsi_val) or c.i == 0:
        return False, False, False, False
    prev = rsi_arr[c.i - 1, c.col] if rsi_arr.ndim > 1 else rsi_arr[c.i - 1]
    if np.isnan(prev):
        return False, False, False, False
    return (prev >= entry_th and rsi_val < entry_th,
            prev <= 50.0 and rsi_val > 50.0,
            prev <= exit_th and rsi_val > exit_th,
            prev >= 50.0 and rsi_val < 50.0)


@njit(nogil=True)
def bbands_signal_nb(c, close_arr, upper_arr, middle_arr, lower_arr):
    px = vbt.pf_nb.select_nb(c, close_arr)
    up = vbt.pf_nb.select_nb(c, upper_arr)
    mid = vbt.pf_nb.select_nb(c, middle_arr)
    lo = vbt.pf_nb.select_nb(c, lower_arr)
    if np.isnan(px) or np.isnan(up) or c.i == 0:
        return False, False, False, False
    pp = close_arr[c.i-1, c.col] if close_arr.ndim > 1 else close_arr[c.i-1]
    plo = lower_arr[c.i-1, c.col] if lower_arr.ndim > 1 else lower_arr[c.i-1]
    pup = upper_arr[c.i-1, c.col] if upper_arr.ndim > 1 else upper_arr[c.i-1]
    pmid = middle_arr[c.i-1, c.col] if middle_arr.ndim > 1 else middle_arr[c.i-1]
    return (pp >= plo and px < lo, pp <= pmid and px > mid,
            pp <= pup and px > up, pp >= pmid and px < mid)


@njit(nogil=True)
def _run_rsi_sim(close_arr, rsi_window, entry_th, fees, slippage):
    """Single RSI backtest, returns Sharpe scalar."""
    ts = close_arr.shape
    rsi = vbt.indicators.nb.rsi_1d_nb(close_arr[:, 0], window=rsi_window)
    eth_arr = np.full(ts, entry_th)
    exth_arr = np.full(ts, 100.0 - entry_th)
    gl = np.full(ts[1], 1)
    so = vbt.pf_nb.from_signal_func_nb(
        target_shape=ts, group_lens=gl, init_cash=1000000.0,
        cash_sharing=False, close=close_arr,
        signal_func_nb=rsi_signal_nb,
        signal_args=(rsi.reshape(-1, 1), eth_arr, exth_arr),
        slippage=np.full(ts, slippage), fees=np.full(ts, fees),
        post_segment_func_nb=vbt.pf_nb.save_post_segment_func_nb,
        in_outputs=vbt.pf_nb.init_FSInOutputs_nb(ts, gl, cash_sharing=False),
    )
    return vbt.ret_nb.sharpe_ratio_nb(returns=so.in_outputs.returns, ann_factor=252.0)[0]


@njit(nogil=True)
def _run_bb_sim(close_arr, bb_window, bb_alpha, fees, slippage):
    """Single BBands backtest, returns Sharpe scalar."""
    ts = close_arr.shape
    upper, middle, lower = vbt.indicators.nb.bbands_1d_nb(close_arr[:, 0], window=bb_window, alpha=bb_alpha)
    gl = np.full(ts[1], 1)
    so = vbt.pf_nb.from_signal_func_nb(
        target_shape=ts, group_lens=gl, init_cash=1000000.0,
        cash_sharing=False, close=close_arr,
        signal_func_nb=bbands_signal_nb,
        signal_args=(close_arr, upper.reshape(-1,1), middle.reshape(-1,1), lower.reshape(-1,1)),
        slippage=np.full(ts, slippage), fees=np.full(ts, fees),
        post_segment_func_nb=vbt.pf_nb.save_post_segment_func_nb,
        in_outputs=vbt.pf_nb.init_FSInOutputs_nb(ts, gl, cash_sharing=False),
    )
    return vbt.ret_nb.sharpe_ratio_nb(returns=so.in_outputs.returns, ann_factor=252.0)[0]


# ═══════════════════════════════════════════════════════════════════════
# PRANGE BATCH SWEEPS (true multicore)
# ═══════════════════════════════════════════════════════════════════════


@njit(parallel=True, nogil=True)
def rsi_batch(close_arr, rsi_windows, entry_ths, fees, slippage):
    """Parallel RSI sweep using all CPU cores via prange."""
    n_rw, n_eth = len(rsi_windows), len(entry_ths)
    results = np.empty(n_rw * n_eth)
    for idx in prange(n_rw * n_eth):
        results[idx] = _run_rsi_sim(
            close_arr, rsi_windows[idx // n_eth], entry_ths[idx % n_eth], fees, slippage,
        )
    return results


@njit(parallel=True, nogil=True)
def bbands_batch(close_arr, bb_windows, bb_alphas, fees, slippage):
    """Parallel BBands sweep using all CPU cores via prange."""
    n_w, n_a = len(bb_windows), len(bb_alphas)
    results = np.empty(n_w * n_a)
    for idx in prange(n_w * n_a):
        results[idx] = _run_bb_sim(
            close_arr, bb_windows[idx // n_a], bb_alphas[idx % n_a], fees, slippage,
        )
    return results


def batch_to_series(results, param1_name, param1_vals, param2_name, param2_vals):
    """Convert flat batch results to MultiIndex Series for VBT heatmaps."""
    idx = pd.MultiIndex.from_product(
        [param1_vals, param2_vals], names=[param1_name, param2_name]
    )
    return pd.Series(results, index=idx)


# ═══════════════════════════════════════════════════════════════════════
# CV PIPELINE (with date-labeled splits)
# ═══════════════════════════════════════════════════════════════════════


def build_rsi_cv(n_splits, window_length):
    @vbt.cv_split(
        splitter="from_n_rolling",
        splitter_kwargs={"n": n_splits, "length": window_length, "split": 0.5, "set_labels": ["train", "test"]},
        takeable_args=["close_arr"],
        parameterized_kwargs={"execute_kwargs": {"chunk_len": "auto", "engine": "threadpool"}, "merge_func": "concat"},
        merge_func="concat",
        attach_bounds="index",
    )
    @njit(nogil=True)
    def cv_fn(close_arr, rsi_window, entry_th, ann_factor: float = 252.0):
        return _run_rsi_sim(close_arr, rsi_window, entry_th, FEES, SLIPPAGE)
    return cv_fn


def build_bb_cv(n_splits, window_length):
    @vbt.cv_split(
        splitter="from_n_rolling",
        splitter_kwargs={"n": n_splits, "length": window_length, "split": 0.5, "set_labels": ["train", "test"]},
        takeable_args=["close_arr"],
        parameterized_kwargs={"execute_kwargs": {"chunk_len": "auto", "engine": "threadpool"}, "merge_func": "concat"},
        merge_func="concat",
        attach_bounds="index",
    )
    @njit(nogil=True)
    def cv_fn(close_arr, bb_window, bb_alpha, ann_factor: float = 252.0):
        return _run_bb_sim(close_arr, bb_window, bb_alpha, FEES, SLIPPAGE)
    return cv_fn


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("MULTI-STRATEGY CV — True Multicore (prange)")
    print(f"Numba threads: {numba.config.NUMBA_NUM_THREADS}")
    print("=" * 60)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    _, data = load_fx_data()
    close_d = data.resample("1d").close.dropna()
    close_4h = data.resample("4h").close.dropna()
    arr_d = vbt.to_2d_array(close_d)
    arr_4h = vbt.to_2d_array(close_4h)
    print(f"Daily: {len(close_d)} | 4H: {len(close_4h)}")

    rsi_ws = np.array(list(range(3, 35, 2)), dtype=np.int64)
    entry_ts = np.array([float(x) for x in range(15, 46, 2)])
    bb_ws = np.array(list(range(5, 45, 3)), dtype=np.int64)
    bb_as = np.array([x / 10.0 for x in range(10, 35, 2)])

    # ── A. RSI FULL SWEEP (prange multicore) ─────────────────────
    print("\n" + "=" * 60)
    print("A. RSI DAILY — prange sweep")
    print("=" * 60)
    import time

    # Warmup
    _run_rsi_sim(arr_d, 14, 30.0, FEES, SLIPPAGE)
    rsi_batch(arr_d, rsi_ws[:1], entry_ts[:1], FEES, SLIPPAGE)

    t0 = time.time()
    rsi_res = rsi_batch(arr_d, rsi_ws, entry_ts, FEES, SLIPPAGE)
    t_rsi = time.time() - t0
    rsi_sr = batch_to_series(rsi_res, "rsi_window", rsi_ws.tolist(), "entry_th", entry_ts.tolist())
    best = rsi_sr.idxmax()
    print(f"  {len(rsi_res)} combos in {t_rsi:.3f}s → Best RSI({best[0]}) {best[1]}/{100-best[1]} Sharpe={rsi_sr.max():.4f}")

    # ── B. BBANDS FULL SWEEP ─────────────────────────────────────
    print("\n" + "=" * 60)
    print("B. BBANDS DAILY — prange sweep")
    print("=" * 60)
    _run_bb_sim(arr_d, 20, 2.0, FEES, SLIPPAGE)
    bbands_batch(arr_d, bb_ws[:1], bb_as[:1], FEES, SLIPPAGE)

    t0 = time.time()
    bb_res = bbands_batch(arr_d, bb_ws, bb_as, FEES, SLIPPAGE)
    t_bb = time.time() - t0
    bb_sr = batch_to_series(bb_res, "bb_window", bb_ws.tolist(), "bb_alpha", bb_as.tolist())
    best_bb = bb_sr.idxmax()
    print(f"  {len(bb_res)} combos in {t_bb:.3f}s → Best BB({best_bb[0]}, {best_bb[1]}) Sharpe={bb_sr.max():.4f}")

    # ── C. RSI 4H SWEEP ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("C. RSI 4H — prange sweep")
    print("=" * 60)
    rsi_batch(arr_4h, rsi_ws[:1], entry_ts[:1], FEES, SLIPPAGE)
    t0 = time.time()
    rsi_4h_res = rsi_batch(arr_4h, rsi_ws, entry_ts, FEES, SLIPPAGE)
    t_4h = time.time() - t0
    rsi_4h_sr = batch_to_series(rsi_4h_res, "rsi_window", rsi_ws.tolist(), "entry_th", entry_ts.tolist())
    best_4h = rsi_4h_sr.idxmax()
    print(f"  {len(rsi_4h_res)} combos in {t_4h:.3f}s → Best RSI({best_4h[0]}) {best_4h[1]} Sharpe={rsi_4h_sr.max():.4f}")

    # ── D. CV WALK-FORWARD (8 splits, date labels) ───────────────
    print("\n" + "=" * 60)
    print("D. RSI CV (8 splits, dates)")
    print("=" * 60)
    cv_rsi = build_rsi_cv(n_splits=8, window_length=500)
    grid_rsi, best_rsi_cv = cv_rsi(
        close_arr=arr_d,
        rsi_window=vbt.Param(rsi_ws.tolist()),
        entry_th=vbt.Param(entry_ts.tolist()),
        _return_grid="all", _index=close_d.index,
    )
    for s in ["train", "test"]:
        if "set" in best_rsi_cv.index.names:
            v = best_rsi_cv.xs(s, level="set")
            print(f"  {s}: mean={v.mean():.4f} min={v.min():.4f} max={v.max():.4f}")

    # ── E. BBANDS CV ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("E. BBANDS CV (8 splits, dates)")
    print("=" * 60)
    cv_bb = build_bb_cv(n_splits=8, window_length=500)
    grid_bb, best_bb_cv = cv_bb(
        close_arr=arr_d,
        bb_window=vbt.Param(bb_ws.tolist()),
        bb_alpha=vbt.Param(bb_as.tolist()),
        _return_grid="all", _index=close_d.index,
    )
    for s in ["train", "test"]:
        if "set" in best_bb_cv.index.names:
            v = best_bb_cv.xs(s, level="set")
            print(f"  {s}: mean={v.mean():.4f} min={v.min():.4f} max={v.max():.4f}")

    # ── F. HEATMAPS → BROWSER ────────────────────────────────────
    print("\n" + "=" * 60)
    print("F. HEATMAPS → browser")
    print("=" * 60)

    # 1. RSI 2D full sample
    fig = fullscreen(rsi_sr.vbt.heatmap(x_level="rsi_window", y_level="entry_th",
        trace_kwargs={"colorscale": "RdYlGn", "zmid": 0}),
        "RSI Daily — Sharpe (prange multicore)", height=800)
    fig.write_html(f"{RESULTS_DIR}/rsi_heatmap_2d.html")
    fig.show(renderer="browser")

    # 2. BBands 2D full sample
    fig = fullscreen(bb_sr.vbt.heatmap(x_level="bb_window", y_level="bb_alpha",
        trace_kwargs={"colorscale": "RdYlGn", "zmid": 0}),
        "BBands Daily — Sharpe (prange multicore)", height=800)
    fig.write_html(f"{RESULTS_DIR}/bb_heatmap_2d.html")
    fig.show(renderer="browser")

    # 3. RSI 4H 2D
    fig = fullscreen(rsi_4h_sr.vbt.heatmap(x_level="rsi_window", y_level="entry_th",
        trace_kwargs={"colorscale": "RdYlGn", "zmid": 0}),
        "RSI 4H — Sharpe (prange multicore)", height=800)
    fig.write_html(f"{RESULTS_DIR}/rsi_4h_heatmap_2d.html")
    fig.show(renderer="browser")

    # 4. RSI CV with date slider (train only)
    try:
        train_grid = grid_rsi.xs("train", level="set")
        fig = fullscreen(train_grid.vbt.heatmap(x_level="rsi_window", y_level="entry_th",
            slider_level="start", trace_kwargs={"colorscale": "RdYlGn", "zmid": 0}),
            "RSI CV Train — Sharpe per split (date slider)", height=800)
        fig.write_html(f"{RESULTS_DIR}/rsi_cv_train_dates.html")
        fig.show(renderer="browser")
    except Exception as e:
        print(f"  RSI CV train heatmap error: {e}")

    # 5. RSI CV test with date slider
    try:
        test_grid = grid_rsi.xs("test", level="set")
        fig = fullscreen(test_grid.vbt.heatmap(x_level="rsi_window", y_level="entry_th",
            slider_level="start", trace_kwargs={"colorscale": "RdYlGn", "zmid": 0}),
            "RSI CV Test — Sharpe per split (date slider)", height=800)
        fig.write_html(f"{RESULTS_DIR}/rsi_cv_test_dates.html")
        fig.show(renderer="browser")
    except Exception as e:
        print(f"  RSI CV test heatmap error: {e}")

    # 6. BBands CV with date slider
    try:
        train_bb = grid_bb.xs("train", level="set")
        fig = fullscreen(train_bb.vbt.heatmap(x_level="bb_window", y_level="bb_alpha",
            slider_level="start", trace_kwargs={"colorscale": "RdYlGn", "zmid": 0}),
            "BBands CV Train — Sharpe per split (date slider)", height=800)
        fig.write_html(f"{RESULTS_DIR}/bb_cv_train_dates.html")
        fig.show(renderer="browser")
    except Exception as e:
        print(f"  BBands CV heatmap error: {e}")

    # 7. 3D: RSI window × entry_th with temporal split slider
    try:
        fig = fullscreen(grid_rsi.vbt.volume(
            x_level="rsi_window", y_level="entry_th", z_level="start",
            trace_kwargs={"colorscale": "RdYlGn"}),
            "RSI 3D — Sharpe: window × entry × split_date", height=800)
        fig.write_html(f"{RESULTS_DIR}/rsi_3d_temporal.html")
        fig.show(renderer="browser")
    except Exception as e:
        print(f"  RSI 3D temporal error: {e}")

    # ── G. INSPECT BEST ───────────────────────────────────────────
    print("\n" + "=" * 60)
    rw, eth = int(best[0]), int(best[1])
    print(f"G. INSPECT BEST: RSI({rw}) {eth}/{100-eth} Daily")
    print("=" * 60)

    rsi_ind = vbt.RSI.run(close_d, window=rw)
    pf = vbt.PF.from_signals(
        close_d, long_entries=rsi_ind.rsi_crossed_below(eth),
        long_exits=rsi_ind.rsi_crossed_above(50),
        short_entries=rsi_ind.rsi_crossed_above(100-eth),
        short_exits=rsi_ind.rsi_crossed_below(50),
        slippage=SLIPPAGE, fees=FEES, init_cash=INIT_CASH, freq="1d")
    print(pf.stats().to_string())

    trades = pf.trades.records_readable
    if len(trades) > 0:
        pnl, ret, w = trades["PnL"], trades["Return"], trades["PnL"] > 0
        print(f"\n  Trades: {len(trades)}, Win: {w.sum()} ({w.mean():.1%})")
        print(f"  Avg win: {ret[w].mean():.4%}, Avg loss: {ret[~w].mean():.4%}")
        t = trades.copy()
        t["year"] = pd.to_datetime(t["Entry Index"]).dt.year
        print(t.groupby("year").agg(n=("PnL","count"), pnl=("PnL","sum"),
            wr=("PnL", lambda x: (x>0).mean())).round(4).to_string())

    fig = fullscreen(pf.plot(subplots=["cumulative_returns", "drawdowns", "underwater", "trade_pnl"]),
        f"RSI({rw}) {eth}/{100-eth} — Best Portfolio", height=1000)
    fig.write_html(f"{RESULTS_DIR}/best_portfolio.html")
    fig.show(renderer="browser")

    print(f"\n{'='*60}")
    print(f"DONE — {RESULTS_DIR}/")
    print(f"{'='*60}")
