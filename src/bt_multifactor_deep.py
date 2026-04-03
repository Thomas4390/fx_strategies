#!/usr/bin/env python
"""Deep multi-factor sweep: RSI + BBands + MACD + Momentum combinations."""
import os, sys, warnings
import numpy as np, pandas as pd, vectorbtpro as vbt
warnings.filterwarnings("ignore"); os.environ["NUMBA_DISABLE_PERFORMANCE_WARNINGS"]="1"
sys.path.insert(0, os.path.dirname(__file__))
from utils import load_fx_data

COSTS = {"slippage": 0.00008, "fees": 0.0001}
INIT_CASH = 1_000_000

if __name__ == "__main__":
    print("=== Multi-Factor Deep Sweep ===")
    _, data = load_fx_data()
    results = []

    for tf in ["4h", "1d"]:
        close_full = data.resample(tf).close.dropna()
        split = int(len(close_full) * 0.7)
        train, test = close_full.iloc[:split], close_full.iloc[split:]
        print(f"\n{tf}: train={len(train)} test={len(test)}")

        for rsi_w in [7, 14, 21]:
            for bb_w in [10, 20, 30]:
                for bb_mult in [2.0, 2.5]:
                    for mom_lb in [10, 21, 42]:
                        rsi = vbt.RSI.run(close_full, window=rsi_w)
                        bb = vbt.BBANDS.run(close_full, window=bb_w, alpha=bb_mult)
                        mom = close_full / close_full.shift(mom_lb) - 1

                        # Combo A: RSI + BB (mean reversion)
                        el_a = rsi.rsi_crossed_below(30) & (close_full < bb.lower)
                        xl_a = close_full.vbt.crossed_above(bb.middle)
                        es_a = rsi.rsi_crossed_above(70) & (close_full > bb.upper)
                        xs_a = close_full.vbt.crossed_below(bb.middle)

                        # Combo B: RSI + BB + Mom confirmation
                        el_b = rsi.rsi_crossed_below(30) & (close_full < bb.lower) & (mom > 0)
                        xl_b = close_full.vbt.crossed_above(bb.middle)
                        es_b = rsi.rsi_crossed_above(70) & (close_full > bb.upper) & (mom < 0)
                        xs_b = close_full.vbt.crossed_below(bb.middle)

                        # Combo C: 2-of-3 scoring
                        sl = rsi.rsi_below(35).astype(int) + (close_full < bb.lower).astype(int) + (mom > 0).astype(int)
                        ss = rsi.rsi_above(65).astype(int) + (close_full > bb.upper).astype(int) + (mom < 0).astype(int)
                        el_c = sl.vbt.crossed_above(1); es_c = ss.vbt.crossed_above(1)
                        xl_c = rsi.rsi_crossed_above(50); xs_c = rsi.rsi_crossed_below(50)

                        for combo_name, el, xl, es, xs in [
                            ("RSI+BB", el_a, xl_a, es_a, xs_a),
                            ("RSI+BB+Mom", el_b, xl_b, es_b, xs_b),
                            ("2of3", el_c, xl_c, es_c, xs_c),
                        ]:
                            for sname, subset in [("train", train), ("test", test)]:
                                el_s = el.reindex(subset.index, fill_value=False)
                                xl_s = xl.reindex(subset.index, fill_value=False)
                                es_s = es.reindex(subset.index, fill_value=False)
                                xs_s = xs.reindex(subset.index, fill_value=False)

                                pf = vbt.PF.from_signals(
                                    close=subset, long_entries=el_s, long_exits=xl_s,
                                    short_entries=es_s, short_exits=xs_s,
                                    slippage=COSTS["slippage"], fees=COSTS["fees"],
                                    init_cash=INIT_CASH, freq=tf,
                                )
                                stats = pf.stats()
                                results.append({
                                    "tf": tf, "combo": combo_name,
                                    "rsi_w": rsi_w, "bb_w": bb_w, "bb_mult": bb_mult,
                                    "mom_lb": mom_lb, "subset": sname,
                                    "sharpe": pf.sharpe_ratio,
                                    "return": pf.total_return,
                                    "pf": stats.get("Profit Factor", np.nan),
                                    "trades": stats.get("Total Trades", 0),
                                })

            # Print progress
            print(f"  {tf} RSI({rsi_w}) done")

    df = pd.DataFrame(results)
    train_df = df[df["subset"]=="train"]
    test_df = df[df["subset"]=="test"]

    # Merge train/test
    key_cols = ["tf","combo","rsi_w","bb_w","bb_mult","mom_lb"]
    merged = train_df.set_index(key_cols)[["sharpe","return","pf","trades"]].join(
        test_df.set_index(key_cols)[["sharpe","return","pf","trades"]],
        lsuffix="_train", rsuffix="_test"
    )
    merged = merged[(merged["sharpe_train"]>0) & (merged["sharpe_test"]>0) & (merged["trades_train"]>=15)]
    merged = merged.sort_values("sharpe_test", ascending=False)

    print("\n=== MULTI-FACTOR PROFITABLE IN BOTH TRAIN AND TEST ===")
    print(merged.head(20).to_string())

    out = "results/exploration/parallel/multifactor_deep.csv"
    df.to_csv(out, index=False)
    merged.to_csv("results/exploration/parallel/multifactor_validated.csv")
    print(f"\nSaved to {out}")
