#!/usr/bin/env python
"""
Run all MR strategy variations and produce a comparative report.

Usage:
    python src/run_all_variations.py          # full CV + hold-out (slow)
    python src/run_all_variations.py --quick   # 200k bars, small grid (fast sanity check)
"""

import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning, module="numba")

import vectorbtpro as vbt

sys.path.insert(0, os.path.dirname(__file__))

from utils import apply_vbt_settings, compute_ann_factor, load_fx_data

apply_vbt_settings()


def load_data(n_bars: int = 0) -> tuple[pd.DataFrame, np.ndarray, float]:
    raw, data = load_fx_data()
    if n_bars > 0:
        raw = raw.iloc[:n_bars]
    index_ns = vbt.dt.to_ns(raw.index)
    ann_factor = compute_ann_factor(raw.index)
    return raw, index_ns, ann_factor


def holdout_split(
    raw: pd.DataFrame, ratio: float = 0.8
) -> tuple[pd.DataFrame, pd.DataFrame]:
    split_idx = int(len(raw) * ratio)
    holdout_date = raw.index[split_idx]
    return raw.loc[:holdout_date], raw.loc[holdout_date:]


def run_variation(name, run_fn, raw, index_ns, ann_factor, params):
    """Run a single variation and return stats dict."""
    raw_train, raw_test = holdout_split(raw)
    ns_train = vbt.dt.to_ns(raw_train.index)
    ns_test = vbt.dt.to_ns(raw_test.index)

    print(f"\n{'=' * 60}")
    print(f"  {name}")
    print(f"{'=' * 60}")

    # Train
    t0 = time.perf_counter()
    pf_train, _ = run_fn(raw_train, ns_train, ann_factor, **params)
    t_train = time.perf_counter() - t0

    # Hold-out
    pf_test, _ = run_fn(raw_test, ns_test, ann_factor, **params)

    train_stats = pf_train.stats()
    test_stats = pf_test.stats()

    print(
        f"  Train: Sharpe={train_stats.get('Sharpe Ratio', 'N/A'):.4f}, "
        f"Return={train_stats.get('Total Return [%]', 'N/A'):.2f}%, "
        f"MaxDD={train_stats.get('Max Drawdown [%]', 'N/A'):.2f}%, "
        f"Trades={pf_train.trades.count()}"
    )
    print(
        f"  Test:  Sharpe={test_stats.get('Sharpe Ratio', 'N/A'):.4f}, "
        f"Return={test_stats.get('Total Return [%]', 'N/A'):.2f}%, "
        f"MaxDD={test_stats.get('Max Drawdown [%]', 'N/A'):.2f}%, "
        f"Trades={pf_test.trades.count()}"
    )
    print(f"  Time: {t_train:.1f}s")

    return {
        "name": name,
        "train_sharpe": train_stats.get("Sharpe Ratio", np.nan),
        "train_return": train_stats.get("Total Return [%]", np.nan),
        "train_maxdd": train_stats.get("Max Drawdown [%]", np.nan),
        "train_trades": pf_train.trades.count(),
        "train_winrate": train_stats.get("Win Rate [%]", np.nan),
        "train_profit_factor": train_stats.get("Profit Factor", np.nan),
        "test_sharpe": test_stats.get("Sharpe Ratio", np.nan),
        "test_return": test_stats.get("Total Return [%]", np.nan),
        "test_maxdd": test_stats.get("Max Drawdown [%]", np.nan),
        "test_trades": pf_test.trades.count(),
        "test_winrate": test_stats.get("Win Rate [%]", np.nan),
        "test_profit_factor": test_stats.get("Profit Factor", np.nan),
    }


if __name__ == "__main__":
    quick = "--quick" in sys.argv
    n_bars = 200_000 if quick else 0
    mode = "QUICK" if quick else "FULL"

    print(f"=== COMPARATIVE REPORT — {mode} MODE ===\n")

    raw, index_ns, ann_factor = load_data(n_bars)
    print(f"Data: {len(raw)} bars ({raw.index[0]} -> {raw.index[-1]})")
    print(f"Ann factor: {ann_factor:.0f}")

    results = []

    # V1: No leverage
    try:
        from mr_v1_no_leverage import run_backtest as run_v1

        results.append(
            run_variation(
                "V1: No Leverage, SL catastrophe",
                run_v1,
                raw,
                index_ns,
                ann_factor,
                {"lookback": 60, "band_width": 2.0, "sl_stop": 0.005},
            )
        )
    except Exception as e:
        print(f"  V1 FAILED: {e}")

    # V2: Z-score exit
    try:
        from mr_v2_zscore_exit import run_backtest as run_v2

        results.append(
            run_variation(
                "V2: Z-Score Exit + SL catastrophe",
                run_v2,
                raw,
                index_ns,
                ann_factor,
                {"lookback": 60, "entry_z": 2.0, "exit_z": 0.5, "sl_stop": 0.005},
            )
        )
    except Exception as e:
        print(f"  V2 FAILED: {e}")

    # V3: Session filter
    try:
        from mr_v3_session_filter import run_backtest as run_v3

        results.append(
            run_variation(
                "V3: Session Filter (8h-16h UTC)",
                run_v3,
                raw,
                index_ns,
                ann_factor,
                {
                    "lookback": 60,
                    "band_width": 2.0,
                    "sl_stop": 0.005,
                    "session_start": 8,
                    "session_end": 16,
                },
            )
        )
    except Exception as e:
        print(f"  V3 FAILED: {e}")

    # V4: Adaptive EWM bands
    try:
        from mr_v4_adaptive_bands import run_backtest as run_v4

        results.append(
            run_variation(
                "V4: Adaptive EWM Bands",
                run_v4,
                raw,
                index_ns,
                ann_factor,
                {"ewm_span": 60, "band_width": 2.0, "sl_stop": 0.005},
            )
        )
    except Exception as e:
        print(f"  V4 FAILED: {e}")

    # Summary
    if results:
        print("\n" + "=" * 80)
        print("COMPARATIVE SUMMARY")
        print("=" * 80)
        df = pd.DataFrame(results).set_index("name")
        print(df.to_string())

        # Save
        os.makedirs("results/comparison", exist_ok=True)
        df.to_csv("results/comparison/variations_summary.csv")
        print("\nSaved to results/comparison/variations_summary.csv")

        # Best variation
        viable = df[df["test_sharpe"] > 0]
        if not viable.empty:
            best = viable["test_sharpe"].idxmax()
            print(f"\n*** BEST VIABLE VARIATION: {best} ***")
            print(f"    Test Sharpe: {viable.loc[best, 'test_sharpe']:.4f}")
            print(f"    Test Return: {viable.loc[best, 'test_return']:.2f}%")
        else:
            print("\n*** NO VIABLE VARIATION FOUND (all test Sharpe <= 0) ***")
            print("    Ranking by least negative test Sharpe:")
            ranking = df.sort_values("test_sharpe", ascending=False)
            for i, (name, row) in enumerate(ranking.iterrows()):
                print(f"    {i + 1}. {name}: Sharpe={row['test_sharpe']:.4f}")
    else:
        print("\nNo variations ran successfully.")
