#!/usr/bin/env python3
"""rsi_mt5_variants — lance 3 variants MT5 pour isoler H1/H2/H5.

Variants :
  A — Sleeve 3 isolé (alloc 0/0/1), settings par défaut
  B — Variant A + vol-targeting neutralisé (forces leverage=1.0)
  C — Variant B + slippage RSI à 0 bps

Sortie : tableau comparatif vs baseline VBT (47 trades, Sharpe +0.10).
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WRAPPER = ROOT / "src/mt5/bridge/run_backtest_cli.py"
OUTPUT_DIR = ROOT / "reports/investigations/rsi_daily"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

VARIANTS = [
    {
        "id": "A_isolated",
        "label": "Sleeve 3 seul, settings défaut",
        "extra_inputs": [
            "Inp_AllocMRMacro=0",
            "Inp_AllocTSMomentum=0",
            "Inp_AllocRSIDaily=1.0",
        ],
    },
    {
        "id": "B_no_voltarget",
        "label": "Sleeve 3 seul + vol-target neutralisé (lev=1)",
        "extra_inputs": [
            "Inp_AllocMRMacro=0",
            "Inp_AllocTSMomentum=0",
            "Inp_AllocRSIDaily=1.0",
            "Inp_GlobalTargetVol=1.0",
            "Inp_GlobalMaxLeverage=1.0",
        ],
    },
    {
        "id": "C_no_slippage",
        "label": "Sleeve 3 seul + lev=1 + slippage 0 bps",
        "extra_inputs": [
            "Inp_AllocMRMacro=0",
            "Inp_AllocTSMomentum=0",
            "Inp_AllocRSIDaily=1.0",
            "Inp_GlobalTargetVol=1.0",
            "Inp_GlobalMaxLeverage=1.0",
            "Inp_RSI_SlippageBps=0",
        ],
    },
]


def run_variant(v: dict) -> dict:
    print(f"\n{'='*70}\n  VARIANT {v['id']} — {v['label']}\n{'='*70}", flush=True)
    cmd = [
        "python3", str(WRAPPER),
        "--report-name", f"fx_rsi_{v['id']}",
        "--ini-name", f"fx_rsi_{v['id']}.ini",
    ]
    for inp in v["extra_inputs"]:
        cmd.extend(["--input", inp])

    result = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True,
                            timeout=600)
    print(result.stdout[-2000:], flush=True)
    if result.returncode != 0:
        print(f"  ERREUR exit={result.returncode}", flush=True)
        if result.stderr:
            print(result.stderr[-500:], flush=True)
        return {"id": v["id"], "error": True}

    # Le wrapper écrit le JSON le plus récent dans reports/mt5/
    json_files = sorted((ROOT / "reports/mt5").glob("run_*.json"),
                        key=lambda p: p.stat().st_mtime)
    if not json_files:
        return {"id": v["id"], "error": "no JSON"}
    latest = json_files[-1]
    payload = json.loads(latest.read_text())
    metrics = payload.get("metrics", {})
    return {
        "id": v["id"],
        "label": v["label"],
        "json": str(latest.name),
        "sharpe": metrics.get("sharpe_ratio"),
        "trades": metrics.get("total_trades"),
        "net_profit": metrics.get("total_net_profit"),
        "profit_factor": metrics.get("profit_factor"),
        "equity_dd_max": metrics.get("equity_dd_max"),
    }


def main() -> int:
    rows = [run_variant(v) for v in VARIANTS]

    print(f"\n{'='*70}\n  SYNTHÈSE — variants vs baselines\n{'='*70}\n", flush=True)
    print(f"{'Variant':<20} {'Sharpe':>8} {'Trades':>7} {'NetProfit':>12} {'PF':>6} {'DDMax':>10}")
    print("-" * 70)
    print(f"{'VBT baseline (4p)':<20} {'+0.104':>8} {'47':>7} {'?':>12} {'?':>6} {'-5.24%':>10}")
    print(f"{'MT5 full (référence)':<20} {'+1.150':>8} {'835':>7} {'+4615':>12} {'1.38':>6} {'-7.21%':>10}")
    print("-" * 70)
    for r in rows:
        if "error" in r:
            print(f"{r['id']:<20} ERROR: {r['error']}")
            continue
        print(f"{r['id']:<20} {r['sharpe']:>8} {r['trades']:>7} "
              f"{(r['net_profit'] or 'n/a')[:12]:>12} {r['profit_factor']:>6} "
              f"{(r['equity_dd_max'] or 'n/a')[:10]:>10}")

    out_csv = OUTPUT_DIR / "variants_mt5.csv"
    import csv
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n→ {out_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
