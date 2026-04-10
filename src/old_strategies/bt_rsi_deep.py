#!/usr/bin/env python
"""Deep RSI sweep with Numba prange multicore + train/test validation."""

import os
import sys
import warnings

import numpy as np
import pandas as pd
import vectorbtpro as vbt

warnings.filterwarnings("ignore")
os.environ["NUMBA_DISABLE_PERFORMANCE_WARNINGS"] = "1"
sys.path.insert(0, os.path.dirname(__file__))
from cv_rsi_validated import _run_rsi_sim, rsi_batch

from utils import load_fx_data

RESULTS_DIR = "results/exploration/parallel"
FEES = 0.0001
SLIPPAGE = 0.00008

if __name__ == "__main__":
    print("=== RSI Deep Sweep (prange multicore) ===")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    _, data = load_fx_data()
    results_all = []

    for tf in ["4h", "1d"]:
        close = data.resample(tf).close.dropna()
        close_arr = vbt.to_2d_array(close)
        split = int(len(close) * 0.7)
        train_arr = vbt.to_2d_array(close.iloc[:split])
        test_arr = vbt.to_2d_array(close.iloc[split:])
        print(f"\n{tf}: train={train_arr.shape[0]} test={test_arr.shape[0]}")

        rsi_ws = np.array([5, 7, 10, 14, 21, 28], dtype=np.int64)
        entry_ts = np.array([20.0, 25.0, 30.0, 35.0, 40.0])

        # Warmup
        _run_rsi_sim(train_arr, 14, 30.0, FEES, SLIPPAGE)
        rsi_batch(train_arr, rsi_ws[:1], entry_ts[:1], FEES, SLIPPAGE)

        # Parallel sweep on train
        train_res = rsi_batch(train_arr, rsi_ws, entry_ts, FEES, SLIPPAGE)
        test_res = rsi_batch(test_arr, rsi_ws, entry_ts, FEES, SLIPPAGE)

        for idx in range(len(train_res)):
            rw_i = idx // len(entry_ts)
            eth_i = idx % len(entry_ts)
            results_all.append(
                {
                    "tf": tf,
                    "window": int(rsi_ws[rw_i]),
                    "lo": int(entry_ts[eth_i]),
                    "hi": int(100 - entry_ts[eth_i]),
                    "sharpe_train": train_res[idx],
                    "sharpe_test": test_res[idx],
                }
            )

    df = pd.DataFrame(results_all)
    valid = df[(df["sharpe_train"] > 0) & (df["sharpe_test"] > 0)]
    valid = valid.sort_values("sharpe_test", ascending=False)

    print("\n=== CONFIGS PROFITABLE IN BOTH TRAIN AND TEST ===")
    print(valid.head(15).to_string(index=False))

    df.to_csv(f"{RESULTS_DIR}/rsi_deep.csv", index=False)
    valid.to_csv(f"{RESULTS_DIR}/rsi_validated.csv", index=False)
    print(f"\nSaved to {RESULTS_DIR}/")
