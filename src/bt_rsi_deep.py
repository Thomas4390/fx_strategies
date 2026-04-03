#!/usr/bin/env python
"""Deep RSI parameter sweep on 1D and 4H with train/test validation."""
import os, sys, warnings, json
import numpy as np, pandas as pd, vectorbtpro as vbt
warnings.filterwarnings("ignore"); os.environ["NUMBA_DISABLE_PERFORMANCE_WARNINGS"]="1"
sys.path.insert(0, os.path.dirname(__file__))
from utils import load_fx_data

COSTS = {"slippage": 0.00008, "fees": 0.0001}
INIT_CASH = 1_000_000

def run_rsi(close, window, lo, hi, tf):
    rsi = vbt.RSI.run(close, window=window)
    el = rsi.rsi_crossed_below(lo); xl = rsi.rsi_crossed_above(50)
    es = rsi.rsi_crossed_above(hi); xs = rsi.rsi_crossed_below(50)
    return vbt.PF.from_signals(close=close, long_entries=el, long_exits=xl,
        short_entries=es, short_exits=xs, slippage=COSTS["slippage"],
        fees=COSTS["fees"], init_cash=INIT_CASH, freq=tf)

if __name__ == "__main__":
    print("=== RSI Deep Sweep ===")
    _, data = load_fx_data()
    results = []

    for tf in ["4h", "1d"]:
        close_full = data.resample(tf).close.dropna()
        split = int(len(close_full) * 0.7)
        train, test = close_full.iloc[:split], close_full.iloc[split:]
        print(f"\n{tf}: train={len(train)} test={len(test)}")

        for w in [5, 7, 10, 14, 21, 28]:
            for lo in [20, 25, 30, 35, 40]:
                hi = 100 - lo  # symmetric
                for subset_name, subset in [("train", train), ("test", test)]:
                    if w >= len(subset) // 2:
                        continue
                    pf = run_rsi(subset, w, lo, hi, tf)
                    stats = pf.stats()
                    results.append({
                        "tf": tf, "window": w, "lo": lo, "hi": hi,
                        "subset": subset_name,
                        "sharpe": pf.sharpe_ratio,
                        "return": pf.total_return,
                        "pf": stats.get("Profit Factor", np.nan),
                        "trades": stats.get("Total Trades", 0),
                        "win_rate": stats.get("Win Rate [%]", np.nan),
                    })

    df = pd.DataFrame(results)

    # Show best configs where BOTH train and test are positive
    train_df = df[df["subset"]=="train"].set_index(["tf","window","lo","hi"])
    test_df = df[df["subset"]=="test"].set_index(["tf","window","lo","hi"])

    merged = train_df[["sharpe","return","pf","trades"]].join(
        test_df[["sharpe","return","pf","trades"]], lsuffix="_train", rsuffix="_test"
    )
    merged = merged[(merged["sharpe_train"]>0) & (merged["sharpe_test"]>0) & (merged["trades_train"]>=20)]
    merged = merged.sort_values("sharpe_test", ascending=False)

    print("\n=== CONFIGS PROFITABLE IN BOTH TRAIN AND TEST ===")
    print(merged.head(20).to_string())

    out = "results/exploration/parallel/rsi_deep.csv"
    df.to_csv(out, index=False)
    merged.to_csv("results/exploration/parallel/rsi_validated.csv")
    print(f"\nSaved to {out}")
