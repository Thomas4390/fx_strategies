#!/usr/bin/env python3
"""walkforward_n5 — 5 fenêtres glissantes IS/OOS pour valider plafond OOS.

Objectif : tester si le plafond CAGR_OOS=+9.54% observé sur 2024-11→2026-04
est constant sur d'autres fenêtres OOS, ou un artefact de cette fenêtre
spécifique.

Fenêtres (sliding 1-an OOS, IS=tout l'historique disponible avant) :

  Fold 1  IS 2020-11→2021-10  (1.0a)  OOS 2021-11→2022-10  (1.0a)
  Fold 2  IS 2020-11→2022-10  (2.0a)  OOS 2022-11→2023-10  (1.0a)
  Fold 3  IS 2020-11→2023-10  (3.0a)  OOS 2023-11→2024-10  (1.0a)
  Fold 4  IS 2020-11→2024-10  (4.0a)  OOS 2024-11→2025-10  (1.0a)
  Fold 5  IS 2020-11→2025-10  (5.0a)  OOS 2025-11→2026-04  (0.5a)

Pour chaque fold : sweep modéré (ne pas tout retester, focus sur les
zones d'intérêt identifiées dans walkforward_3d).

Bornes resserrées (zone optimum trouvée précédemment) :
  vt = 0.20, 0.30, 0.40, 0.50
  lev = 12, 24, 48
  vfloor = 0.02, 0.04, 0.08

= 36 combos × 5 folds × 2 (IS+OOS) = 360 backtests max, ~2-3 min sur 32 cores.
"""
from __future__ import annotations

import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
WRAPPER = ROOT / "scripts/optimization/run_mt5_optimization.py"
OUT_DIR = ROOT / "reports/optimization/walkforward_n5"
OUT_DIR.mkdir(parents=True, exist_ok=True)

VT_GRID = "0.20,0.50,0.10"   # 4 valeurs : 0.20, 0.30, 0.40, 0.50
LEV_GRID = "12,48,12"         # 4 valeurs : 12, 24, 36, 48
VFLOOR_GRID = "0.02,0.04,0.08"  # 3 valeurs

FOLDS = [
    ("fold1_is",  "2020.11.23", "2021.10.31"),
    ("fold1_oos", "2021.11.01", "2022.10.31"),
    ("fold2_is",  "2020.11.23", "2022.10.31"),
    ("fold2_oos", "2022.11.01", "2023.10.31"),
    ("fold3_is",  "2020.11.23", "2023.10.31"),
    ("fold3_oos", "2023.11.01", "2024.10.31"),
    ("fold4_is",  "2020.11.23", "2024.10.31"),
    ("fold4_oos", "2024.11.01", "2025.10.31"),
    ("fold5_is",  "2020.11.23", "2025.10.31"),
    ("fold5_oos", "2025.11.01", "2026.04.30"),
]


def run_sweep(name: str, from_date: str, to_date: str,
              timeout: int = 600) -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    prefix = f"n5_{name}_{ts}"
    cmd = [
        "python3", str(WRAPPER),
        "--vt-start", VT_GRID.split(",")[0],
        "--vt-stop",  VT_GRID.split(",")[1],
        "--vt-step",  VT_GRID.split(",")[2],
        "--lev-start", LEV_GRID.split(",")[0],
        "--lev-stop",  LEV_GRID.split(",")[1],
        "--lev-step",  LEV_GRID.split(",")[2],
        "--vfloor-grid", VFLOOR_GRID,
        "--from-date", from_date,
        "--to-date",   to_date,
        "--out-prefix", prefix,
        "--timeout",   str(timeout),
        # DDCap relâché à 0.30 pour ne pas distort l'analyse
        "--fixed-input", "Inp_DDCap=0.30",
    ]
    print(f"\n--- {name} {from_date}→{to_date} ---", flush=True)
    result = subprocess.run(cmd, cwd=str(ROOT), check=False)
    if result.returncode != 0:
        return None
    csv_path = ROOT / f"reports/optimization/{prefix}.csv"
    return csv_path if csv_path.exists() else None


def load(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["vt_r"] = df["target_vol"].round(3)
    df["lev_r"] = df["max_lev"].round(2)
    df["vfl_r"] = df["vol_floor"].round(3)
    df["calmar"] = df["cagr_pct"] / df["equity_dd_pct"].abs().replace(0, np.nan)
    return df


def main() -> int:
    print("=== Walk-forward N=5 — détection plafond OOS constant ===\n")

    csvs = {}
    for name, from_d, to_d in FOLDS:
        csv = run_sweep(name, from_d, to_d)
        if csv is None:
            print(f"  ✗ {name} failed")
            return 1
        csvs[name] = csv

    # Aggregate stats per fold
    print(f"\n{'='*78}\n  RESULTATS PAR FOLD\n{'='*78}")
    print(f"  {'Fold':<12} {'CAGR_max':>10} {'CAGR_med':>10} {'Sharpe_max':>11} "
          f"{'DD_max':>9}")
    print("  " + "-" * 60)
    rows = []
    for name, _, _ in FOLDS:
        df = load(csvs[name])
        cagr_max = df['cagr_pct'].max()
        cagr_med = df['cagr_pct'].median()
        sharpe_max = df['sharpe'].max()
        dd_max = df['equity_dd_pct'].max()
        print(f"  {name:<12} {cagr_max:>+10.2f} {cagr_med:>+10.2f} "
              f"{sharpe_max:>+11.2f} {dd_max:>+9.2f}")
        rows.append({
            "fold": name,
            "cagr_max": cagr_max,
            "cagr_median": cagr_med,
            "sharpe_max": sharpe_max,
            "dd_max": dd_max,
            "n_configs": len(df),
        })
    summary_df = pd.DataFrame(rows)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    summary_df.to_csv(OUT_DIR / f"summary_{ts}.csv", index=False)

    # Per-fold IS↔OOS Spearman ρ on ranks
    print(f"\n{'='*78}\n  STABILITE IS↔OOS PAR FOLD\n{'='*78}")
    print(f"  {'Fold':<8} {'configs':>8} {'ρ_CAGR':>9} {'ρ_Sharpe':>10}")
    print("  " + "-" * 38)
    folds_rho = []
    for i in range(1, 6):
        is_csv = csvs.get(f"fold{i}_is")
        oos_csv = csvs.get(f"fold{i}_oos")
        if not is_csv or not oos_csv:
            continue
        df_is = load(is_csv)
        df_oos = load(oos_csv)
        merged = df_is.merge(df_oos, on=["vt_r", "lev_r", "vfl_r"],
                             suffixes=("_is", "_oos"))
        if len(merged) < 5:
            continue
        rho_c, _ = spearmanr(merged["cagr_pct_is"], merged["cagr_pct_oos"])
        rho_s, _ = spearmanr(merged["sharpe_is"], merged["sharpe_oos"])
        print(f"  fold{i}   {len(merged):>8} {rho_c:>+9.3f} {rho_s:>+10.3f}")
        folds_rho.append({
            "fold": i,
            "n_configs": len(merged),
            "rho_cagr": rho_c,
            "rho_sharpe": rho_s,
        })
    pd.DataFrame(folds_rho).to_csv(
        OUT_DIR / f"rho_per_fold_{ts}.csv", index=False)

    # Detection : plafond CAGR_OOS constant ?
    oos_caps = [r["cagr_max"] for r in rows if r["fold"].endswith("_oos")]
    if oos_caps:
        avg_cap = np.mean(oos_caps)
        std_cap = np.std(oos_caps)
        print(f"\n{'='*78}\n  PLAFOND OOS CAGR — STABILITE\n{'='*78}")
        print(f"  CAGR_max OOS par fold : {[round(x, 2) for x in oos_caps]}")
        print(f"  Moyenne : {avg_cap:+.2f}%")
        print(f"  Écart-type : {std_cap:.2f}%")
        print(f"  Range : [{min(oos_caps):+.2f}, {max(oos_caps):+.2f}]")
        if std_cap < 2.0:
            print(f"  → VERDICT : plafond OOS STABLE — limite physique d'edge")
        else:
            print(f"  → VERDICT : plafond OOS VARIABLE — dépend de la fenêtre")

    print(f"\n  → {OUT_DIR / f'summary_{ts}.csv'}")
    print(f"  → {OUT_DIR / f'rho_per_fold_{ts}.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
