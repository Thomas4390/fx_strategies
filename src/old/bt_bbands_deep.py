#!/usr/bin/env python
"""Deep BBands sweep with Numba prange multicore + train/test + alignment."""

import os
import sys
import warnings

import numpy as np
import pandas as pd
import vectorbtpro as vbt

warnings.filterwarnings("ignore")
os.environ["NUMBA_DISABLE_PERFORMANCE_WARNINGS"] = "1"
sys.path.insert(0, os.path.dirname(__file__))
from cv_rsi_validated import _run_bb_sim, bbands_batch

from utils import load_fx_data

RESULTS_DIR = "results/exploration/parallel"
FEES = 0.0001
SLIPPAGE = 0.00008

if __name__ == "__main__":
    print("=== BBands Deep Sweep (prange multicore) ===")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    _, data = load_fx_data()
    results_all = []

    # Part 1: Standard sweep on clean data
    for tf in ["4h", "1d"]:
        close = data.resample(tf).close.dropna()
        close_arr = vbt.to_2d_array(close)
        split = int(len(close) * 0.7)
        train_arr = vbt.to_2d_array(close.iloc[:split])
        test_arr = vbt.to_2d_array(close.iloc[split:])
        print(f"\n{tf}: train={train_arr.shape[0]} test={test_arr.shape[0]}")

        bb_ws = np.array([5, 10, 15, 20, 30, 40], dtype=np.int64)
        bb_as = np.array([1.0, 1.5, 2.0, 2.5, 3.0])

        _run_bb_sim(train_arr, 20, 2.0, FEES, SLIPPAGE)
        bbands_batch(train_arr, bb_ws[:1], bb_as[:1], FEES, SLIPPAGE)

        train_res = bbands_batch(train_arr, bb_ws, bb_as, FEES, SLIPPAGE)
        test_res = bbands_batch(test_arr, bb_ws, bb_as, FEES, SLIPPAGE)

        for idx in range(len(train_res)):
            w_i = idx // len(bb_as)
            a_i = idx % len(bb_as)
            results_all.append(
                {
                    "tf": tf,
                    "window": int(bb_ws[w_i]),
                    "alpha": bb_as[a_i],
                    "shift": 0,
                    "subset": "standard",
                    "sharpe_train": train_res[idx],
                    "sharpe_test": test_res[idx],
                }
            )

    # Part 2: 4H alignment robustness
    print("\n=== 4H Alignment Robustness ===")
    bb_ws_align = np.array([10, 20, 30], dtype=np.int64)
    bb_as_align = np.array([2.0, 2.5, 3.0])

    for shift_h in range(4):
        raw = pd.read_parquet("data/EUR-USD.parquet")
        raw = raw.set_index("date").sort_index()
        raw.index = raw.index + pd.Timedelta(hours=shift_h)
        raw.columns = [c.capitalize() for c in raw.columns]
        d = vbt.Data.from_data({"EUR-USD": raw}, tz_localize=False, tz_convert=False)
        close = d.resample("4h").close.dropna()
        split = int(len(close) * 0.7)
        test_arr = vbt.to_2d_array(close.iloc[split:])

        test_res = bbands_batch(test_arr, bb_ws_align, bb_as_align, FEES, SLIPPAGE)
        for idx in range(len(test_res)):
            w_i = idx // len(bb_as_align)
            a_i = idx % len(bb_as_align)
            results_all.append(
                {
                    "tf": "4h",
                    "window": int(bb_ws_align[w_i]),
                    "alpha": bb_as_align[a_i],
                    "shift": shift_h,
                    "subset": f"test_shift{shift_h}",
                    "sharpe_train": np.nan,
                    "sharpe_test": test_res[idx],
                }
            )
            print(
                f"  shift={shift_h} BB({bb_ws_align[w_i]},{bb_as_align[a_i]}): Sharpe={test_res[idx]:.4f}"
            )

    df = pd.DataFrame(results_all)
    standard = df[df["subset"] == "standard"]
    valid = standard[(standard["sharpe_train"] > 0) & (standard["sharpe_test"] > 0)]
    valid = valid.sort_values("sharpe_test", ascending=False)

    print("\n=== BBANDS PROFITABLE IN BOTH TRAIN AND TEST ===")
    print(valid.head(15).to_string(index=False))

    # Alignment robustness
    align = df[df["subset"].str.startswith("test_shift")]
    if len(align) > 0:
        pivot = align.pivot_table(
            index=["tf", "window", "alpha"], columns="subset", values="sharpe_test"
        )
        pivot["worst"] = pivot.min(axis=1)
        pivot["best"] = pivot.max(axis=1)
        print("\n=== 4H ALIGNMENT ROBUSTNESS ===")
        print(pivot.sort_values("worst", ascending=False).head(10).to_string())

    df.to_csv(f"{RESULTS_DIR}/bbands_deep.csv", index=False)
    print(f"\nSaved to {RESULTS_DIR}/")
