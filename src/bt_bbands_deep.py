#!/usr/bin/env python
"""Deep BBands sweep on 1D and 4H with train/test + alignment robustness."""
import os, sys, warnings
import numpy as np, pandas as pd, vectorbtpro as vbt
warnings.filterwarnings("ignore"); os.environ["NUMBA_DISABLE_PERFORMANCE_WARNINGS"]="1"
sys.path.insert(0, os.path.dirname(__file__))
from utils import load_fx_data

COSTS = {"slippage": 0.00008, "fees": 0.0001}
INIT_CASH = 1_000_000

def run_bb(close, window, mult, tf):
    bb = vbt.BBANDS.run(close, window=window, alpha=mult)
    el = close.vbt.crossed_below(bb.lower); xl = close.vbt.crossed_above(bb.middle)
    es = close.vbt.crossed_above(bb.upper); xs = close.vbt.crossed_below(bb.middle)
    return vbt.PF.from_signals(close=close, long_entries=el, long_exits=xl,
        short_entries=es, short_exits=xs, slippage=COSTS["slippage"],
        fees=COSTS["fees"], init_cash=INIT_CASH, freq=tf)

if __name__ == "__main__":
    print("=== BBands Deep Sweep ===")

    results = []

    # Part 1: Standard sweep on clean data (no shift)
    _, data = load_fx_data()
    for tf in ["4h", "1d"]:
        close_full = data.resample(tf).close.dropna()
        split = int(len(close_full) * 0.7)
        train, test = close_full.iloc[:split], close_full.iloc[split:]
        print(f"\n{tf}: train={len(train)} test={len(test)}")

        for w in [5, 10, 15, 20, 30, 40]:
            for mult in [1.0, 1.5, 2.0, 2.5, 3.0]:
                for subset_name, subset in [("train", train), ("test", test)]:
                    if w >= len(subset) // 2:
                        continue
                    pf = run_bb(subset, w, mult, tf)
                    stats = pf.stats()
                    results.append({
                        "tf": tf, "window": w, "mult": mult, "shift": 0,
                        "subset": subset_name,
                        "sharpe": pf.sharpe_ratio,
                        "return": pf.total_return,
                        "pf": stats.get("Profit Factor", np.nan),
                        "trades": stats.get("Total Trades", 0),
                    })

    # Part 2: 4H alignment robustness for top BBands configs
    print("\n=== 4H Alignment Robustness ===")
    for shift_h in range(4):
        raw = pd.read_parquet("data/EUR-USD.parquet")
        raw = raw.set_index("date").sort_index()
        raw.index = raw.index + pd.Timedelta(hours=shift_h)
        raw.columns = [c.capitalize() for c in raw.columns]
        d = vbt.Data.from_data({"EUR-USD": raw}, tz_localize=False, tz_convert=False)
        close = d.resample("4h").close.dropna()
        split = int(len(close) * 0.7)
        test = close.iloc[split:]

        for w in [10, 20, 30]:
            for mult in [2.0, 2.5, 3.0]:
                pf = run_bb(test, w, mult, "4h")
                stats = pf.stats()
                results.append({
                    "tf": "4h", "window": w, "mult": mult, "shift": shift_h,
                    "subset": f"test_shift{shift_h}",
                    "sharpe": pf.sharpe_ratio,
                    "return": pf.total_return,
                    "pf": stats.get("Profit Factor", np.nan),
                    "trades": stats.get("Total Trades", 0),
                })
                print(f"  shift={shift_h} BB({w},{mult}): Sharpe={pf.sharpe_ratio:.4f}")

    df = pd.DataFrame(results)

    # Find configs robust across alignments
    train_df = df[df["subset"]=="train"].set_index(["tf","window","mult"])
    test_df = df[df["subset"]=="test"].set_index(["tf","window","mult"])

    merged = train_df[["sharpe","return","pf","trades"]].join(
        test_df[["sharpe","return","pf","trades"]], lsuffix="_train", rsuffix="_test"
    )
    merged = merged[(merged["sharpe_train"]>0) & (merged["sharpe_test"]>0) & (merged["trades_train"]>=15)]
    merged = merged.sort_values("sharpe_test", ascending=False)

    print("\n=== BBANDS PROFITABLE IN BOTH TRAIN AND TEST ===")
    print(merged.head(20).to_string())

    # Alignment robustness
    align_df = df[df["subset"].str.startswith("test_shift")]
    if len(align_df) > 0:
        align_pivot = align_df.pivot_table(
            index=["tf","window","mult"], columns="subset", values="sharpe"
        )
        align_pivot["worst"] = align_pivot.min(axis=1)
        align_pivot["best"] = align_pivot.max(axis=1)
        align_pivot["range"] = align_pivot["best"] - align_pivot["worst"]
        print("\n=== 4H ALIGNMENT ROBUSTNESS (test set) ===")
        print(align_pivot.sort_values("worst", ascending=False).head(15).to_string())

    out = "results/exploration/parallel/bbands_deep.csv"
    df.to_csv(out, index=False)
    print(f"\nSaved to {out}")
