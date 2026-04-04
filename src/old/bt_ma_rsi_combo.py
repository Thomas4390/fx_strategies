#!/usr/bin/env python
"""MA trend + RSI pullback sweep with Numba prange multicore."""
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
from cv_rsi_validated import ma_rsi_signal_nb

RESULTS_DIR = "results/exploration/parallel"
FEES = 0.0001
SLIPPAGE = 0.00008


@njit(nogil=True)
def _run_marsi_sim(close_arr, ma_window, rsi_window, entry_th, fees, slippage):
    """Single MA+RSI backtest: EMA(w, 3w) + RSI pullback."""
    ts = close_arr.shape
    col = close_arr[:, 0]
    fast_ma = vbt.generic.nb.ewm_mean_1d_nb(col, span=ma_window, minp=ma_window, adjust=False)
    slow_ma = vbt.generic.nb.ewm_mean_1d_nb(col, span=ma_window * 3, minp=ma_window * 3, adjust=False)
    rsi = vbt.indicators.nb.rsi_1d_nb(col, window=rsi_window)
    entry_th_arr = np.full(ts, entry_th)
    gl = np.full(ts[1], 1)
    so = vbt.pf_nb.from_signal_func_nb(
        target_shape=ts, group_lens=gl, init_cash=1000000.0,
        cash_sharing=False, close=close_arr,
        signal_func_nb=ma_rsi_signal_nb,
        signal_args=(fast_ma.reshape(-1,1), slow_ma.reshape(-1,1), rsi.reshape(-1,1), entry_th_arr),
        slippage=np.full(ts, slippage), fees=np.full(ts, fees),
        post_segment_func_nb=vbt.pf_nb.save_post_segment_func_nb,
        in_outputs=vbt.pf_nb.init_FSInOutputs_nb(ts, gl, cash_sharing=False),
    )
    return vbt.ret_nb.sharpe_ratio_nb(returns=so.in_outputs.returns, ann_factor=252.0)[0]


@njit(parallel=True, nogil=True)
def marsi_batch(close_arr, ma_ws, rsi_ws, entry_ths, fees, slippage):
    """Parallel MA+RSI sweep."""
    n1, n2, n3 = len(ma_ws), len(rsi_ws), len(entry_ths)
    total = n1 * n2 * n3
    results = np.empty(total)
    for idx in prange(total):
        i1 = idx // (n2 * n3)
        i2 = (idx // n3) % n2
        i3 = idx % n3
        results[idx] = _run_marsi_sim(close_arr, ma_ws[i1], rsi_ws[i2], entry_ths[i3], fees, slippage)
    return results


if __name__ == "__main__":
    print("=== MA + RSI Combo Sweep (prange multicore) ===")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    _, data = load_fx_data()
    results_all = []

    for tf in ["4h", "1d"]:
        close = data.resample(tf).close.dropna()
        split = int(len(close) * 0.7)
        train_arr = vbt.to_2d_array(close.iloc[:split])
        test_arr = vbt.to_2d_array(close.iloc[split:])
        print(f"\n{tf}: train={train_arr.shape[0]} test={test_arr.shape[0]}")

        ma_ws = np.array([10, 15, 20, 30, 50], dtype=np.int64)
        rsi_ws = np.array([5, 7, 10, 14, 21], dtype=np.int64)
        entry_ths = np.array([20.0, 25.0, 30.0, 35.0, 40.0])

        # Warmup
        _run_marsi_sim(train_arr, 20, 14, 30.0, FEES, SLIPPAGE)
        marsi_batch(train_arr, ma_ws[:1], rsi_ws[:1], entry_ths[:1], FEES, SLIPPAGE)

        import time
        t0 = time.time()
        train_res = marsi_batch(train_arr, ma_ws, rsi_ws, entry_ths, FEES, SLIPPAGE)
        test_res = marsi_batch(test_arr, ma_ws, rsi_ws, entry_ths, FEES, SLIPPAGE)
        print(f"  {len(train_res)} combos in {time.time()-t0:.3f}s")

        n1, n2, n3 = len(ma_ws), len(rsi_ws), len(entry_ths)
        for idx in range(len(train_res)):
            i1 = idx // (n2 * n3)
            i2 = (idx // n3) % n2
            i3 = idx % n3
            results_all.append({
                "tf": tf, "ma_w": int(ma_ws[i1]), "rsi_w": int(rsi_ws[i2]),
                "entry_th": entry_ths[i3],
                "sharpe_train": train_res[idx], "sharpe_test": test_res[idx],
            })

    df = pd.DataFrame(results_all)
    valid = df[(df["sharpe_train"] > 0) & (df["sharpe_test"] > 0)]
    valid = valid.sort_values("sharpe_test", ascending=False)

    print("\n=== MA+RSI PROFITABLE IN BOTH TRAIN AND TEST ===")
    print(valid.head(15).to_string(index=False))

    df.to_csv(f"{RESULTS_DIR}/ma_rsi_combo.csv", index=False)
    valid.to_csv(f"{RESULTS_DIR}/ma_rsi_validated.csv", index=False)
    print(f"\nSaved to {RESULTS_DIR}/")
