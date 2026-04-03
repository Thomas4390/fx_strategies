#!/usr/bin/env python
"""MA trend + RSI filter: trend-following with mean reversion timing."""
import os, sys, warnings
import numpy as np, pandas as pd, vectorbtpro as vbt
warnings.filterwarnings("ignore"); os.environ["NUMBA_DISABLE_PERFORMANCE_WARNINGS"]="1"
sys.path.insert(0, os.path.dirname(__file__))
from utils import load_fx_data

COSTS = {"slippage": 0.00008, "fees": 0.0001}
INIT_CASH = 1_000_000

if __name__ == "__main__":
    print("=== MA + RSI Combo Sweep ===")
    _, data = load_fx_data()
    results = []

    for tf in ["4h", "1d"]:
        close_full = data.resample(tf).close.dropna()
        split = int(len(close_full) * 0.7)
        train, test = close_full.iloc[:split], close_full.iloc[split:]
        print(f"\n{tf}: train={len(train)} test={len(test)}")

        for ma_fast, ma_slow in [(10, 50), (20, 50), (20, 100), (50, 200)]:
            for ma_type in ["simple", "exp"]:
                for rsi_w in [7, 14, 21]:
                    for rsi_lo in [25, 30, 35, 40]:
                        rsi_hi = 100 - rsi_lo
                        if ma_slow >= len(close_full) // 2:
                            continue

                        fast = vbt.MA.run(close_full, window=ma_fast, wtype=ma_type)
                        slow = vbt.MA.run(close_full, window=ma_slow, wtype=ma_type)
                        rsi = vbt.RSI.run(close_full, window=rsi_w)

                        # Strategy: Enter when MA trend + RSI pullback
                        # Long: uptrend (fast > slow) + RSI oversold
                        trend_up = fast.ma > slow.ma
                        trend_down = fast.ma < slow.ma

                        el = trend_up & rsi.rsi_crossed_below(rsi_lo)
                        xl = rsi.rsi_crossed_above(rsi_hi) | (fast.ma_crossed_below(slow))
                        es = trend_down & rsi.rsi_crossed_above(rsi_hi)
                        xs = rsi.rsi_crossed_below(rsi_lo) | (fast.ma_crossed_above(slow))

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
                                "tf": tf, "ma_type": ma_type,
                                "ma_fast": ma_fast, "ma_slow": ma_slow,
                                "rsi_w": rsi_w, "rsi_lo": rsi_lo,
                                "subset": sname,
                                "sharpe": pf.sharpe_ratio,
                                "return": pf.total_return,
                                "pf": stats.get("Profit Factor", np.nan),
                                "trades": stats.get("Total Trades", 0),
                            })

            print(f"  {tf} MA({ma_fast},{ma_slow}) {ma_type} done")

    df = pd.DataFrame(results)
    key_cols = ["tf","ma_type","ma_fast","ma_slow","rsi_w","rsi_lo"]
    train_df = df[df["subset"]=="train"].set_index(key_cols)
    test_df = df[df["subset"]=="test"].set_index(key_cols)

    merged = train_df[["sharpe","return","pf","trades"]].join(
        test_df[["sharpe","return","pf","trades"]], lsuffix="_train", rsuffix="_test"
    )
    merged = merged[(merged["sharpe_train"]>0) & (merged["sharpe_test"]>0) & (merged["trades_train"]>=15)]
    merged = merged.sort_values("sharpe_test", ascending=False)

    print("\n=== MA+RSI PROFITABLE IN BOTH TRAIN AND TEST ===")
    print(merged.head(20).to_string())

    out = "results/exploration/parallel/ma_rsi_combo.csv"
    df.to_csv(out, index=False)
    merged.to_csv("results/exploration/parallel/ma_rsi_validated.csv")
    print(f"\nSaved to {out}")
