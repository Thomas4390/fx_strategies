#!/usr/bin/env python3
"""phase3_incremental — test l'effet d'ajouter les paires retenues au
portfolio combiné prod (alloc 80/10/10, vol-targeting normal, DDCap 0.30).

Variants :
  V0  Baseline 4 paires (config prod actuelle)
  V1  + AUDUSD au MR Macro (5 paires MR)
  V2  + NZDUSD au MR Macro (5 paires MR)
  V3  + AUDUSD + NZDUSD au MR Macro (6 paires MR)
  V4  + EURJPY au TS Momentum (4 paires TS au lieu de 3)
  V5  Combiné : MR=6 (AUDUSD+NZDUSD), TS=4 (EURJPY)

Métrique : ΔCAGR, ΔSharpe, ΔDD vs V0. Critère retenu : ΔSharpe ≥ +0.03.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
WRAPPER = ROOT / "src/mt5/bridge/run_backtest_cli.py"
OUT_DIR = ROOT / "reports/optimization/expansion_pairs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

VARIANTS = [
    {
        "id": "V0_baseline_4pair",
        "label": "Baseline 4-pair (prod default)",
        "inputs": [],  # défauts compilés
    },
    {
        "id": "V1_MR_plus_AUDUSD",
        "label": "MR Macro = 5 paires (+AUDUSD)",
        "inputs": ["Inp_MR_Pairs=EURUSD,GBPUSD,USDJPY,USDCAD,AUDUSD"],
    },
    {
        "id": "V2_MR_plus_NZDUSD",
        "label": "MR Macro = 5 paires (+NZDUSD)",
        "inputs": ["Inp_MR_Pairs=EURUSD,GBPUSD,USDJPY,USDCAD,NZDUSD"],
    },
    {
        "id": "V3_MR_plus_AUDNZD",
        "label": "MR Macro = 6 paires (+AUDUSD +NZDUSD)",
        "inputs": ["Inp_MR_Pairs=EURUSD,GBPUSD,USDJPY,USDCAD,AUDUSD,NZDUSD"],
    },
    {
        "id": "V4_TS_plus_EURJPY",
        "label": "TS Momentum = 4 paires (+EURJPY)",
        "inputs": ["Inp_TS_Pairs=EURUSD,GBPUSD,USDJPY,EURJPY"],
    },
    {
        "id": "V5_combined",
        "label": "Combiné : MR=6 + TS=4 (AUDUSD+NZDUSD+EURJPY)",
        "inputs": [
            "Inp_MR_Pairs=EURUSD,GBPUSD,USDJPY,USDCAD,AUDUSD,NZDUSD",
            "Inp_TS_Pairs=EURUSD,GBPUSD,USDJPY,EURJPY",
        ],
    },
]


def num(s):
    if not s:
        return None
    cleaned = re.sub(r"[^\d.,\-+]", "", s.split("(")[0]).replace(",", ".")
    try:
        return float(cleaned)
    except ValueError:
        return None


def pct(s):
    if not s:
        return None
    m = re.search(r"\(([-+]?\d+(?:[.,]\d+)?)\s*%\)", s)
    if m:
        return float(m.group(1).replace(",", "."))
    m = re.search(r"([-+]?\d+(?:[.,]\d+)?)\s*%", s)
    return float(m.group(1).replace(",", ".")) if m else None


def run_variant(v: dict) -> dict:
    print(f"\n  [{v['id']}] {v['label']}", flush=True)
    cmd = [
        "python3", str(WRAPPER),
        "--report-name", f"p3_{v['id']}",
        "--ini-name", f"p3_{v['id']}.ini",
    ]
    for inp in v["inputs"]:
        cmd.extend(["--input", inp])

    result = subprocess.run(cmd, cwd=str(ROOT), capture_output=True,
                            text=True, timeout=300)
    if result.returncode != 0:
        return {"id": v["id"], "error": f"exit={result.returncode}"}

    json_files = sorted((ROOT / "reports/mt5").glob("run_*.json"),
                        key=lambda p: p.stat().st_mtime)
    if not json_files:
        return {"id": v["id"], "error": "no JSON"}
    payload = json.loads(json_files[-1].read_text())
    m = payload.get("metrics", {})

    net_profit = num(m.get("total_net_profit"))
    deposit = 10000.0
    years = 5.43
    cagr = ((deposit + (net_profit or 0)) / deposit) ** (1.0/years) - 1.0 if net_profit is not None else None

    return {
        "id": v["id"],
        "label": v["label"],
        "sharpe": num(m.get("sharpe_ratio")),
        "cagr_pct": cagr * 100 if cagr is not None else None,
        "net_profit": net_profit,
        "trades": int(num(m.get("total_trades")) or 0),
        "equity_dd_pct": pct(m.get("equity_dd_max")),
        "profit_factor": num(m.get("profit_factor")),
        "recovery_factor": num(m.get("recovery_factor")),
    }


def main() -> int:
    print(f"=== Phase 3 — test incrémental ajout paires ===")
    print(f"  {len(VARIANTS)} variants, ~22s chacun = ~3 min total\n")

    rows = [run_variant(v) for v in VARIANTS]

    df = pd.DataFrame(rows)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    csv_path = OUT_DIR / f"phase3_incremental_{ts}.csv"
    df.to_csv(csv_path, index=False)

    # Calcul deltas vs baseline V0
    baseline = next(r for r in rows if r["id"] == "V0_baseline_4pair")

    print(f"\n{'='*90}\n  RESULTATS\n{'='*90}")
    print(f"  {'Variant':<32} {'CAGR':>7} {'ΔCAGR':>7} {'Sharpe':>7} "
          f"{'ΔSharpe':>8} {'DD':>7} {'Trades':>7}")
    print("  " + "-" * 88)
    for r in rows:
        if "error" in r:
            print(f"  {r['id']:<32} ERROR: {r['error']}")
            continue
        d_cagr = r["cagr_pct"] - baseline["cagr_pct"] if r["cagr_pct"] is not None else None
        d_sharpe = r["sharpe"] - baseline["sharpe"] if r["sharpe"] is not None else None
        marker = "✓" if (d_sharpe or 0) >= 0.03 else " "
        print(f" {marker}{r['id']:<32} "
              f"{r['cagr_pct']:>+6.2f}% "
              f"{(d_cagr or 0):>+6.2f}% "
              f"{r['sharpe']:>+7.2f} "
              f"{(d_sharpe or 0):>+8.2f} "
              f"{r['equity_dd_pct']:>+6.2f}% "
              f"{r['trades']:>7d}")

    print(f"\n  → {csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
