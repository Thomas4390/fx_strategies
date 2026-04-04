#!/usr/bin/env python
"""Multi-factor sweep (RSI + BBands + Momentum) with prange multicore."""
import os
import sys
import warnings

import numpy as np
import pandas as pd
import vectorbtpro as vbt
from numba import njit, prange

warnings.filterwarnings("ignore")
os.environ["NUMBA_DISABLE_PERFORMANCE_WARNINGS"] = "1"
sys.path.insert(0, os.path.dirname(__file__))
from utils import load_fx_data

RESULTS_DIR = "results/exploration/parallel"
FEES = 0.0001
SLIPPAGE = 0.00008


@njit(nogil=True)
def multifactor_signal_nb(c, rsi_arr, close_arr, bb_lower_arr, bb_upper_arr, bb_mid_arr, mom_arr):
    """2-of-3 scoring: RSI + BBands + Momentum."""
    rsi = vbt.pf_nb.select_nb(c, rsi_arr)
    px = vbt.pf_nb.select_nb(c, close_arr)
    lo = vbt.pf_nb.select_nb(c, bb_lower_arr)
    up = vbt.pf_nb.select_nb(c, bb_upper_arr)
    mid = vbt.pf_nb.select_nb(c, bb_mid_arr)
    mom = vbt.pf_nb.select_nb(c, mom_arr)

    if np.isnan(rsi) or np.isnan(px) or np.isnan(lo) or np.isnan(mom) or c.i == 0:
        return False, False, False, False

    prev_rsi = rsi_arr[c.i-1, c.col] if rsi_arr.ndim > 1 else rsi_arr[c.i-1]
    if np.isnan(prev_rsi):
        return False, False, False, False

    # Scoring
    score_long = (1 if rsi < 35.0 else 0) + (1 if px < lo else 0) + (1 if mom > 0 else 0)
    score_short = (1 if rsi > 65.0 else 0) + (1 if px > up else 0) + (1 if mom < 0 else 0)
    prev_score_l = (1 if prev_rsi < 35.0 else 0)  # simplified cross detection
    prev_score_s = (1 if prev_rsi > 65.0 else 0)

    entry_long = score_long >= 2 and prev_score_l < 1
    entry_short = score_short >= 2 and prev_score_s < 1
    exit_long = prev_rsi <= 50.0 and rsi > 50.0
    exit_short = prev_rsi >= 50.0 and rsi < 50.0

    return entry_long, exit_long, entry_short, exit_short


@njit(nogil=True)
def _run_mf_sim(close_arr, rsi_w, bb_w, bb_alpha, mom_lb, fees, slippage):
    """Single multi-factor backtest."""
    ts = close_arr.shape
    col = close_arr[:, 0]
    rsi = vbt.indicators.nb.rsi_1d_nb(col, window=rsi_w)
    upper, middle, lower = vbt.indicators.nb.bbands_1d_nb(col, window=bb_w, alpha=bb_alpha)

    # Momentum
    mom = np.full(len(col), np.nan)
    for i in range(mom_lb, len(col)):
        if col[i - mom_lb] != 0:
            mom[i] = col[i] / col[i - mom_lb] - 1.0

    gl = np.full(ts[1], 1)
    so = vbt.pf_nb.from_signal_func_nb(
        target_shape=ts, group_lens=gl, init_cash=1000000.0,
        cash_sharing=False, close=close_arr,
        signal_func_nb=multifactor_signal_nb,
        signal_args=(rsi.reshape(-1,1), close_arr, lower.reshape(-1,1),
                     upper.reshape(-1,1), middle.reshape(-1,1), mom.reshape(-1,1)),
        slippage=np.full(ts, slippage), fees=np.full(ts, fees),
        post_segment_func_nb=vbt.pf_nb.save_post_segment_func_nb,
        in_outputs=vbt.pf_nb.init_FSInOutputs_nb(ts, gl, cash_sharing=False),
    )
    return vbt.ret_nb.sharpe_ratio_nb(returns=so.in_outputs.returns, ann_factor=252.0)[0]


@njit(parallel=True, nogil=True)
def mf_batch(close_arr, rsi_ws, bb_ws, bb_alphas, mom_lbs, fees, slippage):
    """Parallel multi-factor sweep."""
    n1, n2, n3, n4 = len(rsi_ws), len(bb_ws), len(bb_alphas), len(mom_lbs)
    total = n1 * n2 * n3 * n4
    results = np.empty(total)
    for idx in prange(total):
        i1 = idx // (n2 * n3 * n4)
        i2 = (idx // (n3 * n4)) % n2
        i3 = (idx // n4) % n3
        i4 = idx % n4
        results[idx] = _run_mf_sim(close_arr, rsi_ws[i1], bb_ws[i2], bb_alphas[i3], mom_lbs[i4], fees, slippage)
    return results


if __name__ == "__main__":
    print("=== Multi-Factor Deep Sweep (prange multicore) ===")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    _, data = load_fx_data()
    results_all = []

    for tf in ["4h", "1d"]:
        close = data.resample(tf).close.dropna()
        split = int(len(close) * 0.7)
        train_arr = vbt.to_2d_array(close.iloc[:split])
        test_arr = vbt.to_2d_array(close.iloc[split:])
        print(f"\n{tf}: train={train_arr.shape[0]} test={test_arr.shape[0]}")

        rsi_ws = np.array([7, 14, 21], dtype=np.int64)
        bb_ws = np.array([10, 20, 30], dtype=np.int64)
        bb_as = np.array([2.0, 2.5])
        mom_lbs = np.array([10, 21, 42], dtype=np.int64)

        # Warmup
        _run_mf_sim(train_arr, 14, 20, 2.0, 21, FEES, SLIPPAGE)
        mf_batch(train_arr, rsi_ws[:1], bb_ws[:1], bb_as[:1], mom_lbs[:1], FEES, SLIPPAGE)

        import time
        t0 = time.time()
        train_res = mf_batch(train_arr, rsi_ws, bb_ws, bb_as, mom_lbs, FEES, SLIPPAGE)
        test_res = mf_batch(test_arr, rsi_ws, bb_ws, bb_as, mom_lbs, FEES, SLIPPAGE)
        print(f"  {len(train_res)} combos in {time.time()-t0:.3f}s")

        n1, n2, n3, n4 = len(rsi_ws), len(bb_ws), len(bb_as), len(mom_lbs)
        for idx in range(len(train_res)):
            i1 = idx // (n2 * n3 * n4)
            i2 = (idx // (n3 * n4)) % n2
            i3 = (idx // n4) % n3
            i4 = idx % n4
            results_all.append({
                "tf": tf, "rsi_w": int(rsi_ws[i1]), "bb_w": int(bb_ws[i2]),
                "bb_alpha": bb_as[i3], "mom_lb": int(mom_lbs[i4]),
                "sharpe_train": train_res[idx], "sharpe_test": test_res[idx],
            })

    df = pd.DataFrame(results_all)
    valid = df[(df["sharpe_train"] > 0) & (df["sharpe_test"] > 0)]
    valid = valid.sort_values("sharpe_test", ascending=False)

    print("\n=== MULTI-FACTOR PROFITABLE IN BOTH TRAIN AND TEST ===")
    print(valid.head(15).to_string(index=False))

    df.to_csv(f"{RESULTS_DIR}/multifactor_deep.csv", index=False)
    valid.to_csv(f"{RESULTS_DIR}/multifactor_validated.csv", index=False)
    print(f"\nSaved to {RESULTS_DIR}/")
