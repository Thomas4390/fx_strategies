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


@njit(nogil=True)
def rsi_exit_signal_nb(c, rsi_arr, entry_th_arr, exit_th_arr, exit_l_arr, exit_s_arr):
    """RSI signal with variable exit level."""
    rv = vbt.pf_nb.select_nb(c, rsi_arr)
    et = vbt.pf_nb.select_nb(c, entry_th_arr)
    ext = vbt.pf_nb.select_nb(c, exit_th_arr)
    xl = vbt.pf_nb.select_nb(c, exit_l_arr)
    xs = vbt.pf_nb.select_nb(c, exit_s_arr)
    if np.isnan(rv) or c.i == 0:
        return False, False, False, False
    p = rsi_arr[c.i - 1, c.col] if rsi_arr.ndim > 1 else rsi_arr[c.i - 1]
    if np.isnan(p):
        return False, False, False, False
    return (p >= et and rv < et, p <= xl and rv > xl,
            p <= ext and rv > ext, p >= xs and rv < xs)


@njit(nogil=True)
def _run_rsi_sim_exit(close_arr, rsi_window, entry_th, exit_level, fees, slippage):
    """RSI backtest with variable exit level (not fixed at 50)."""
    ts = close_arr.shape
    rsi = vbt.indicators.nb.rsi_1d_nb(close_arr[:, 0], window=rsi_window)
    rsi_2d = rsi.reshape(-1, 1)
    eth_arr = np.full(ts, entry_th)
    exth_arr = np.full(ts, 100.0 - entry_th)
    exit_l_arr = np.full(ts, exit_level)
    exit_s_arr = np.full(ts, 100.0 - exit_level)
    gl = np.full(ts[1], 1)
    so = vbt.pf_nb.from_signal_func_nb(
        target_shape=ts, group_lens=gl, init_cash=1000000.0,
        cash_sharing=False, close=close_arr,
        signal_func_nb=rsi_exit_signal_nb,
        signal_args=(rsi_2d, eth_arr, exth_arr, exit_l_arr, exit_s_arr),
        slippage=np.full(ts, slippage), fees=np.full(ts, fees),
        post_segment_func_nb=vbt.pf_nb.save_post_segment_func_nb,
        in_outputs=vbt.pf_nb.init_FSInOutputs_nb(ts, gl, cash_sharing=False),
    )
    return vbt.ret_nb.sharpe_ratio_nb(returns=so.in_outputs.returns, ann_factor=252.0)[0]


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
def rsi_batch_3d(close_arr, rsi_windows, entry_ths, exit_levels, fees, slippage):
    """3-param RSI sweep (window × entry × exit_level) via prange."""
    n_rw = len(rsi_windows)
    n_eth = len(entry_ths)
    n_ex = len(exit_levels)
    total = n_rw * n_eth * n_ex
    results = np.empty(total)
    for idx in prange(total):
        rw_i = idx // (n_eth * n_ex)
        eth_i = (idx // n_ex) % n_eth
        ex_i = idx % n_ex
        results[idx] = _run_rsi_sim_exit(
            close_arr, rsi_windows[rw_i], entry_ths[eth_i], exit_levels[ex_i], fees, slippage,
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


def batch_to_series(results, names_vals):
    """Convert flat batch results to MultiIndex Series for VBT heatmaps.

    names_vals: list of (name, values) tuples.
    """
    names = [nv[0] for nv in names_vals]
    vals = [nv[1] for nv in names_vals]
    idx = pd.MultiIndex.from_product(vals, names=names)
    return pd.Series(results, index=idx)


# ═══════════════════════════════════════════════════════════════════════
# CV VIA MANUAL SPLITTER + PRANGE (true multicore on every split)
# ═══════════════════════════════════════════════════════════════════════


def _run_cv_prange(close_series, batch_fn, param_arrays, param_names, n_splits=8, window_length=500):
    """Generic walk-forward CV using vbt.Splitter + prange batch function.

    batch_fn: a prange-parallel function like rsi_batch(close_arr, *param_arrays, fees, slippage)
    param_arrays: list of numpy arrays for each parameter dimension
    param_names: list of str names for each parameter
    """
    splitter = vbt.Splitter.from_n_rolling(
        close_series.index, n=n_splits, length=window_length,
        split=0.5, set_labels=["train", "test"],
    )
    taken = splitter.take(close_series)

    all_rows = []
    for split_i in range(n_splits):
        for set_label in ["train", "test"]:
            subset = taken[(split_i, set_label)]
            arr = vbt.to_2d_array(subset)

            results = batch_fn(arr, *param_arrays, FEES, SLIPPAGE)

            start_str = str(subset.index[0].date())
            end_str = str(subset.index[-1].date())

            # Decode flat index back to param indices
            sizes = [len(p) for p in param_arrays]
            for flat_idx in range(len(results)):
                param_vals = []
                remainder = flat_idx
                for dim in range(len(sizes) - 1, -1, -1):
                    param_vals.insert(0, param_arrays[dim][remainder % sizes[dim]])
                    remainder //= sizes[dim]

                row = [split_i, set_label, start_str, end_str] + [_convert(v) for v in param_vals]
                all_rows.append((*row, results[flat_idx]))

    level_names = ["split", "set", "start", "end"] + param_names
    idx = pd.MultiIndex.from_tuples(
        [r[:-1] for r in all_rows], names=level_names,
    )
    grid = pd.Series([r[-1] for r in all_rows], index=idx)

    # Best per split: find best train params, report test
    best_rows = []
    for split_i in range(n_splits):
        train = grid.xs((split_i, "train"), level=("split", "set"))
        best_idx = train.idxmax()
        train_sharpe = train.loc[best_idx]
        test = grid.xs((split_i, "test"), level=("split", "set"))
        # Extract param values from best_idx (skip start/end)
        param_vals = best_idx[2:]  # after start, end
        test_match = test.xs(param_vals, level=param_names)
        test_sharpe = test_match.iloc[0] if len(test_match) > 0 else np.nan
        row = {"split": split_i, "train_sharpe": train_sharpe, "test_sharpe": test_sharpe}
        for pn, pv in zip(param_names, param_vals):
            row[pn] = pv
        best_rows.append(row)

    return grid, pd.DataFrame(best_rows)


def _convert(v):
    """Convert numpy scalar to Python native for MultiIndex."""
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    return v


def run_rsi_cv(close_series, rsi_windows, entry_ths, n_splits=8, window_length=500):
    return _run_cv_prange(
        close_series, rsi_batch,
        [rsi_windows, entry_ths], ["rsi_window", "entry_th"],
        n_splits, window_length,
    )


def run_bb_cv(close_series, bb_windows, bb_alphas, n_splits=8, window_length=500):
    return _run_cv_prange(
        close_series, bbands_batch,
        [bb_windows, bb_alphas], ["bb_window", "bb_alpha"],
        n_splits, window_length,
    )


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

    import time

    rsi_ws = np.array(list(range(3, 35, 2)), dtype=np.int64)
    entry_ts = np.array([float(x) for x in range(15, 46, 2)])
    exit_lvls = np.array([40.0, 45.0, 50.0, 55.0, 60.0])
    bb_ws = np.array(list(range(5, 45, 3)), dtype=np.int64)
    bb_as = np.array([x / 10.0 for x in range(10, 35, 2)])

    # ── A. RSI FULL SWEEP (prange multicore) ─────────────────────
    print("\n" + "=" * 60)
    print("A. RSI DAILY — prange sweep")
    print("=" * 60)
    _run_rsi_sim(arr_d, 14, 30.0, FEES, SLIPPAGE)
    rsi_batch(arr_d, rsi_ws[:1], entry_ts[:1], FEES, SLIPPAGE)

    t0 = time.time()
    rsi_res = rsi_batch(arr_d, rsi_ws, entry_ts, FEES, SLIPPAGE)
    t_rsi = time.time() - t0
    rsi_sr = batch_to_series(rsi_res, [("rsi_window", rsi_ws.tolist()), ("entry_th", entry_ts.tolist())])
    best = rsi_sr.idxmax()
    print(f"  {len(rsi_res)} combos in {t_rsi:.3f}s → Best RSI({best[0]}) {best[1]}/{100-best[1]} Sharpe={rsi_sr.max():.4f}")

    # ── B. RSI 3D SWEEP (window × entry × exit_level) ────────────
    print("\n" + "=" * 60)
    print("B. RSI 3D — prange sweep (window × entry × exit_level)")
    print("=" * 60)
    rsi_ws_3d = np.array([5, 7, 10, 14, 21, 28], dtype=np.int64)
    entry_ts_3d = np.array([15.0, 20.0, 25.0, 30.0, 35.0, 40.0])
    _run_rsi_sim_exit(arr_d, 14, 30.0, 50.0, FEES, SLIPPAGE)
    rsi_batch_3d(arr_d, rsi_ws_3d[:1], entry_ts_3d[:1], exit_lvls[:1], FEES, SLIPPAGE)

    t0 = time.time()
    rsi_3d_res = rsi_batch_3d(arr_d, rsi_ws_3d, entry_ts_3d, exit_lvls, FEES, SLIPPAGE)
    t_3d = time.time() - t0
    rsi_3d_sr = batch_to_series(rsi_3d_res, [
        ("rsi_window", rsi_ws_3d.tolist()),
        ("entry_th", entry_ts_3d.tolist()),
        ("exit_level", exit_lvls.tolist()),
    ])
    print(f"  {len(rsi_3d_res)} combos in {t_3d:.3f}s")

    # ── C. BBANDS FULL SWEEP ─────────────────────────────────────
    print("\n" + "=" * 60)
    print("C. BBANDS DAILY — prange sweep")
    print("=" * 60)
    _run_bb_sim(arr_d, 20, 2.0, FEES, SLIPPAGE)
    bbands_batch(arr_d, bb_ws[:1], bb_as[:1], FEES, SLIPPAGE)

    t0 = time.time()
    bb_res = bbands_batch(arr_d, bb_ws, bb_as, FEES, SLIPPAGE)
    t_bb = time.time() - t0
    bb_sr = batch_to_series(bb_res, [("bb_window", bb_ws.tolist()), ("bb_alpha", bb_as.tolist())])
    best_bb = bb_sr.idxmax()
    print(f"  {len(bb_res)} combos in {t_bb:.3f}s → Best BB({best_bb[0]}, {best_bb[1]}) Sharpe={bb_sr.max():.4f}")

    # ── D. RSI 4H SWEEP ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("D. RSI 4H — prange sweep")
    print("=" * 60)
    rsi_batch(arr_4h, rsi_ws[:1], entry_ts[:1], FEES, SLIPPAGE)
    t0 = time.time()
    rsi_4h_res = rsi_batch(arr_4h, rsi_ws, entry_ts, FEES, SLIPPAGE)
    t_4h = time.time() - t0
    rsi_4h_sr = batch_to_series(rsi_4h_res, [("rsi_window", rsi_ws.tolist()), ("entry_th", entry_ts.tolist())])
    best_4h = rsi_4h_sr.idxmax()
    print(f"  {len(rsi_4h_res)} combos in {t_4h:.3f}s → Best RSI({best_4h[0]}) {best_4h[1]} Sharpe={rsi_4h_sr.max():.4f}")

    # ── E. CV WALK-FORWARD (8 splits, prange multicore per split) ─
    print("\n" + "=" * 60)
    print("E. RSI CV (8 splits, prange per split)")
    print("=" * 60)
    t0 = time.time()
    grid_rsi, best_rsi_cv = run_rsi_cv(close_d, rsi_ws, entry_ts, n_splits=8, window_length=500)
    print(f"  {len(grid_rsi)} results in {time.time()-t0:.3f}s")
    print(f"  Train Sharpe: mean={best_rsi_cv['train_sharpe'].mean():.4f} min={best_rsi_cv['train_sharpe'].min():.4f}")
    print(f"  Test Sharpe:  mean={best_rsi_cv['test_sharpe'].mean():.4f} min={best_rsi_cv['test_sharpe'].min():.4f}")
    print(best_rsi_cv.to_string(index=False))

    # ── F. BBANDS CV ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("F. BBANDS CV (8 splits, prange per split)")
    print("=" * 60)
    t0 = time.time()
    grid_bb, best_bb_cv = run_bb_cv(close_d, bb_ws, bb_as, n_splits=8, window_length=500)
    print(f"  {len(grid_bb)} results in {time.time()-t0:.3f}s")
    print(f"  Train Sharpe: mean={best_bb_cv['train_sharpe'].mean():.4f} min={best_bb_cv['train_sharpe'].min():.4f}")
    print(f"  Test Sharpe:  mean={best_bb_cv['test_sharpe'].mean():.4f} min={best_bb_cv['test_sharpe'].min():.4f}")
    print(best_bb_cv.to_string(index=False))

    # ── G. HEATMAPS → BROWSER ────────────────────────────────────
    print("\n" + "=" * 60)
    print("G. HEATMAPS → browser")
    print("=" * 60)

    def show_save(fig, name, title, h=800):
        fig = fullscreen(fig, title, h)
        fig.write_html(f"{RESULTS_DIR}/{name}.html")
        fig.show(renderer="browser")
        print(f"  {name} → browser")

    # 1. RSI 2D full sample
    show_save(rsi_sr.vbt.heatmap(x_level="rsi_window", y_level="entry_th",
        trace_kwargs={"colorscale": "RdYlGn", "zmid": 0}),
        "rsi_heatmap_2d", "RSI Daily — Sharpe (32-core prange)")

    # 2. RSI 3D volume: window × entry_th × exit_level
    show_save(rsi_3d_sr.vbt.volume(
        x_level="rsi_window", y_level="entry_th", z_level="exit_level",
        trace_kwargs={"colorscale": "RdYlGn"}),
        "rsi_3d_volume", "RSI 3D — Sharpe: window × entry × exit_level")

    # 3. RSI 3D as heatmap with exit_level slider
    show_save(rsi_3d_sr.vbt.heatmap(
        x_level="rsi_window", y_level="entry_th", slider_level="exit_level",
        trace_kwargs={"colorscale": "RdYlGn", "zmid": 0}),
        "rsi_3d_slider", "RSI Sharpe — Slider by exit_level")

    # 4. BBands 2D
    show_save(bb_sr.vbt.heatmap(x_level="bb_window", y_level="bb_alpha",
        trace_kwargs={"colorscale": "RdYlGn", "zmid": 0}),
        "bb_heatmap_2d", "BBands Daily — Sharpe")

    # 5. RSI 4H 2D
    show_save(rsi_4h_sr.vbt.heatmap(x_level="rsi_window", y_level="entry_th",
        trace_kwargs={"colorscale": "RdYlGn", "zmid": 0}),
        "rsi_4h_heatmap_2d", "RSI 4H — Sharpe")

    # 6. RSI CV train with date slider
    try:
        train_grid = grid_rsi.xs("train", level="set")
        show_save(train_grid.vbt.heatmap(x_level="rsi_window", y_level="entry_th",
            slider_level="start", trace_kwargs={"colorscale": "RdYlGn", "zmid": 0}),
            "rsi_cv_train", "RSI CV Train — date slider")
    except Exception as e:
        print(f"  RSI CV train error: {e}")

    # 7. RSI CV test with date slider
    try:
        test_grid = grid_rsi.xs("test", level="set")
        show_save(test_grid.vbt.heatmap(x_level="rsi_window", y_level="entry_th",
            slider_level="start", trace_kwargs={"colorscale": "RdYlGn", "zmid": 0}),
            "rsi_cv_test", "RSI CV Test — date slider")
    except Exception as e:
        print(f"  RSI CV test error: {e}")

    # 8. BBands CV train with date slider
    try:
        train_bb = grid_bb.xs("train", level="set")
        show_save(train_bb.vbt.heatmap(x_level="bb_window", y_level="bb_alpha",
            slider_level="start", trace_kwargs={"colorscale": "RdYlGn", "zmid": 0}),
            "bb_cv_train", "BBands CV Train — date slider")
    except Exception as e:
        print(f"  BBands CV train error: {e}")

    # 9. RSI CV 3D volume: window × entry × split_start (temporal z-axis)
    #    with set (train/test) as slider
    try:
        show_save(grid_rsi.vbt.volume(
            x_level="rsi_window", y_level="entry_th", z_level="split",
            slider_level="set",
            trace_kwargs={"colorscale": "RdYlGn"}),
            "rsi_cv_3d", "RSI CV 3D — window × entry × split (train/test slider)")
    except Exception as e:
        print(f"  RSI CV 3D error: {e}")

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
