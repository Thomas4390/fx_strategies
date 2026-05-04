#!/usr/bin/env python3
"""eval_new_pairs — Phase 2 plan paires : Sharpe standalone par paire par sleeve.

Pour chaque paire candidate, lance 2 backtests MT5 isolés :
  - TS Momentum seul sur cette paire (alloc 0/1/0, Inp_TS_Pairs=NEWPAIR)
  - RSI Daily seul sur cette paire (alloc 0/0/1, Inp_RSI_Pairs=NEWPAIR)

MR Macro skipped : nécessite M1 sur 5.4 ans, broker limite à 3 mois.

Output : reports/optimization/expansion_pairs/standalone_<ts>.csv
"""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
WRAPPER = ROOT / "src/mt5/bridge/run_backtest_cli.py"
OUT_DIR = ROOT / "reports/optimization/expansion_pairs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Paires candidates ordonnées par hypothèse
NEW_PAIRS = ["USDCHF", "AUDUSD", "NZDUSD", "EURGBP", "EURJPY", "GBPJPY"]

# Limites broker D1 :
#   USDCHF, AUDUSD, NZDUSD, GBPJPY : 2020-11-22 (5.4 ans)
#   EURGBP, EURJPY                 : 2022-11-04 (3.5 ans)
PAIR_FROM = {
    "USDCHF": "2020.11.23", "AUDUSD": "2020.11.23",
    "NZDUSD": "2020.11.23", "GBPJPY": "2020.11.23",
    "EURGBP": "2022.11.05", "EURJPY": "2022.11.05",
}

WINDOW_END = "2026.04.30"


def run_isolated(pair: str, sleeve: str) -> dict:
    """Lance backtest avec un seul sleeve actif sur une seule paire.

    DDCap désactivé pour ne pas freiner artificiellement (alloc 100% +
    1 paire amplifie le DD nominal). Vol-targeting désactivé (lev=1)
    pour mesurer l'edge brut sans amplification.
    """
    common = [
        "Inp_EnableDDCap=false",       # pas de circuit-breaker
        "Inp_GlobalTargetVol=1.0",     # neutralise vol-target
        "Inp_GlobalMaxLeverage=1.0",   # cap levier à 1 → effectif=1
        "Inp_RSI_SlippageBps=10",      # défaut conservateur
    ]
    if sleeve == "MR":
        inputs = common + [
            "Inp_AllocMRMacro=1.0",
            "Inp_AllocTSMomentum=0",
            "Inp_AllocRSIDaily=0",
            f"Inp_MR_Pairs={pair}",
        ]
    elif sleeve == "TS":
        inputs = common + [
            "Inp_AllocMRMacro=0",
            "Inp_AllocTSMomentum=1.0",
            "Inp_AllocRSIDaily=0",
            f"Inp_TS_Pairs={pair}",
        ]
    elif sleeve == "RSI":
        inputs = common + [
            "Inp_AllocMRMacro=0",
            "Inp_AllocTSMomentum=0",
            "Inp_AllocRSIDaily=1.0",
            f"Inp_RSI_Pairs={pair}",
        ]
    else:
        raise ValueError(sleeve)

    pair_id = pair.replace(".", "")
    report_name = f"pair_{sleeve}_{pair_id}"
    cmd = [
        "python3", str(WRAPPER),
        "--from", PAIR_FROM[pair],
        "--to",   WINDOW_END,
        "--report-name", report_name,
        "--ini-name", f"{report_name}.ini",
    ]
    for inp in inputs:
        cmd.extend(["--input", inp])

    print(f"  [{pair}/{sleeve}] running...", flush=True)
    result = subprocess.run(cmd, cwd=str(ROOT), capture_output=True,
                            text=True, timeout=300)
    if result.returncode != 0:
        return {"pair": pair, "sleeve": sleeve, "error": f"exit={result.returncode}"}

    json_files = sorted((ROOT / "reports/mt5").glob("run_*.json"),
                        key=lambda p: p.stat().st_mtime)
    if not json_files:
        return {"pair": pair, "sleeve": sleeve, "error": "no JSON"}
    payload = json.loads(json_files[-1].read_text())
    m = payload.get("metrics", {})

    def num(s):
        if not s:
            return None
        import re
        cleaned = re.sub(r"[^\d.,\-+]", "", s.split("(")[0]).replace(",", ".")
        try:
            return float(cleaned)
        except ValueError:
            return None

    def pct(s):
        if not s:
            return None
        import re
        m_match = re.search(r"\(([-+]?\d+(?:[.,]\d+)?)\s*%\)", s)
        if m_match:
            return float(m_match.group(1).replace(",", "."))
        m_match = re.search(r"([-+]?\d+(?:[.,]\d+)?)\s*%", s)
        return float(m_match.group(1).replace(",", ".")) if m_match else None

    return {
        "pair": pair,
        "sleeve": sleeve,
        "from_date": PAIR_FROM[pair],
        "to_date": WINDOW_END,
        "sharpe": num(m.get("sharpe_ratio")),
        "net_profit": num(m.get("total_net_profit")),
        "trades": int(num(m.get("total_trades")) or 0),
        "equity_dd_pct": pct(m.get("equity_dd_max")),
        "profit_factor": num(m.get("profit_factor")),
        "recovery_factor": num(m.get("recovery_factor")),
    }


def main() -> int:
    print(f"=== Phase 2 — Sharpe standalone par paire par sleeve ===")
    print(f"  {len(NEW_PAIRS)} paires × 2 sleeves = {len(NEW_PAIRS)*2} backtests")
    print(f"  Estimé ~5-7 min séquentiel\n")

    rows = []
    for pair in NEW_PAIRS:
        for sleeve in ["MR", "TS", "RSI"]:
            r = run_isolated(pair, sleeve)
            rows.append(r)
            if "error" in r:
                print(f"    ✗ {r['error']}")
            else:
                print(f"    Sharpe={r['sharpe']:+.2f} Trades={r['trades']:>3d} "
                      f"DD={r['equity_dd_pct']:+.2f}% Net={r['net_profit']:+.2f}")

    df = pd.DataFrame(rows)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    csv_path = OUT_DIR / f"standalone_{ts}.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n=== {len(df)} rows → {csv_path} ===")

    # Filtre : Sharpe ≥ 0.30 sur au moins un sleeve
    print(f"\n=== FILTRE QUALITY GATE (Sharpe ≥ 0.30 sur ≥1 sleeve) ===")
    df_valid = df.dropna(subset=["sharpe"])
    by_pair = df_valid.groupby("pair")
    keep = []
    for pair, group in by_pair:
        max_sharpe = group["sharpe"].max()
        sleeves_pass = group[group["sharpe"] >= 0.30]
        status = "RETAIN" if len(sleeves_pass) > 0 else "SKIP"
        if status == "RETAIN":
            keep.append(pair)
        print(f"  {pair}: max Sharpe = {max_sharpe:+.2f}  → {status}")
        for _, r in group.iterrows():
            mark = "✓" if (r['sharpe'] or 0) >= 0.30 else "✗"
            print(f"    {mark} {r['sleeve']:<4}: Sharpe={r['sharpe']:+.2f} "
                  f"Trades={r['trades']} DD={r['equity_dd_pct']:+.2f}%")

    print(f"\n=== {len(keep)}/{len(NEW_PAIRS)} paires retenues : {keep} ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
