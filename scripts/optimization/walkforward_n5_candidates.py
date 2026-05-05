#!/usr/bin/env python3
"""walkforward_n5_candidates — N=5 walk-forward sur top-3 candidats sweep noddcap.

Candidats (issus de reports/optimization/walkforward_aggressive/merged_20260505T161927Z.csv):
  C1 conservateur : vt=0.75 lev=64 vfl=0.02   CAGR_avg=15.4% DD_max=12.9%
  C2 médian       : vt=1.00 lev=64 vfl=0.02   CAGR_avg=18.0% DD_max=14.8%
  C3 agressif     : vt=1.50 lev=64 vfl=0.01   CAGR_avg=21.7% DD_max=17.8%

Grid couvre les 3 simultanément (DDCap désactivé pour révéler vrai DD):
  vt = 0.75, 1.00, 1.50
  lev = 64
  vfl = 0.01, 0.02

= 6 combos par fold × 10 fenêtres = 60 backtests.
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
OUT_DIR = ROOT / "reports/optimization/walkforward_n5_candidates"
OUT_DIR.mkdir(parents=True, exist_ok=True)
MT5_CACHE_DIR = Path("/home/thomas/.mt5/drive_c/Program Files/MetaTrader 5/Tester/cache")


def clear_mt5_opt_cache() -> int:
    """Clear MT5 optimization cache files (.opt) to force re-run.

    MT5 caches optimization results by hash of (params, symbol, period, dates).
    If hash matches a previous run, MT5 returns cached results instantly without
    invoking OnTester(), so optim_results.csv is never written.
    """
    if not MT5_CACHE_DIR.exists():
        return 0
    n = 0
    for f in MT5_CACHE_DIR.glob("*.opt"):
        try:
            f.unlink()
            n += 1
        except OSError:
            pass
    return n

VT_GRID = ("0.75", "1.50", "0.25")
LEV_GRID = ("48", "80", "16")
VFLOOR_GRID = "0.01,0.02"

CANDIDATES = [
    ("C1_conservateur", 0.75, 64, 0.02),
    ("C2_median",       1.00, 64, 0.02),
    ("C3_agressif",     1.50, 64, 0.01),
]

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
              timeout: int = 600) -> Path | None:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    prefix = f"n5cand_{name}_{ts}"
    cmd = [
        "python3", str(WRAPPER),
        "--vt-start", VT_GRID[0],
        "--vt-stop",  VT_GRID[1],
        "--vt-step",  VT_GRID[2],
        "--lev-start", LEV_GRID[0],
        "--lev-stop",  LEV_GRID[1],
        "--lev-step",  LEV_GRID[2],
        "--vfloor-grid", VFLOOR_GRID,
        "--from-date", from_date,
        "--to-date",   to_date,
        "--out-prefix", prefix,
        "--timeout",   str(timeout),
        "--fixed-input", "Inp_EnableDDCap=false",
    ]
    n_cleared = clear_mt5_opt_cache()
    print(f"\n--- {name} {from_date}→{to_date} (cleared {n_cleared} .opt) ---",
          flush=True)
    result = subprocess.run(cmd, cwd=str(ROOT), check=False)
    if result.returncode != 0:
        return None
    csv_path = ROOT / f"reports/optimization/{prefix}.csv"
    return csv_path if csv_path.exists() else None


def load(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["vt_r"] = df["target_vol"].round(3)
    df["lev_r"] = df["max_lev"].round(0).astype(int)
    df["vfl_r"] = df["vol_floor"].round(3)
    return df


def extract_candidate(df: pd.DataFrame, vt: float, lev: int, vfl: float) -> dict:
    """Extract row matching candidate params, or empty dict if absent."""
    mask = ((df["vt_r"] == vt) & (df["lev_r"] == lev) & (df["vfl_r"] == vfl))
    matches = df[mask]
    if matches.empty:
        return {}
    r = matches.iloc[0]
    return {
        "cagr_pct": float(r["cagr_pct"]),
        "sharpe": float(r["sharpe"]),
        "equity_dd_pct": float(r["equity_dd_pct"]),
        "profit_factor": float(r.get("profit_factor", 0)),
        "trades": int(r.get("trades", 0)),
        "net_profit": float(r.get("net_profit", 0)),
    }


def main() -> int:
    print("=== Walk-forward N=5 sur 3 candidats (DDCap=OFF) ===\n")

    csvs = {}
    for name, from_d, to_d in FOLDS:
        csv = run_sweep(name, from_d, to_d)
        if csv is None:
            print(f"  ✗ {name} failed — STOP")
            return 1
        csvs[name] = csv

    # Per-candidate per-fold extraction
    print(f"\n{'='*78}\n  RESULTATS PAR CANDIDAT × FOLD\n{'='*78}")

    all_rows = []
    for cand_name, vt, lev, vfl in CANDIDATES:
        print(f"\n  --- {cand_name} (vt={vt} lev={lev} vfl={vfl}) ---")
        print(f"  {'Fold':<12} {'CAGR%':>8} {'Sharpe':>8} {'DD%':>7} {'PF':>6} {'Trades':>7}")
        print("  " + "-" * 50)
        for fold_name, from_d, to_d in FOLDS:
            df = load(csvs[fold_name])
            row = extract_candidate(df, vt, lev, vfl)
            if not row:
                print(f"  {fold_name:<12} {'MISS':>8}")
                continue
            print(f"  {fold_name:<12} {row['cagr_pct']:>+8.2f} "
                  f"{row['sharpe']:>+8.2f} {row['equity_dd_pct']:>+7.2f} "
                  f"{row['profit_factor']:>+6.2f} {row['trades']:>7d}")
            all_rows.append({
                "candidate": cand_name,
                "vt": vt, "lev": lev, "vfl": vfl,
                "fold": fold_name,
                "is_oos": "oos" if "oos" in fold_name else "is",
                **row,
            })

    df_all = pd.DataFrame(all_rows)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    df_all.to_csv(OUT_DIR / f"per_fold_{ts}.csv", index=False)

    # Summary stats per candidate
    print(f"\n{'='*78}\n  SYNTHESE OOS PAR CANDIDAT (5 folds)\n{'='*78}")
    print(f"  {'Candidat':<18} {'CAGR_med':>9} {'CAGR_avg':>9} {'Shr_med':>9} "
          f"{'Shr_min':>9} {'DD_max':>8} {'4/5≥1?':>7}")
    print("  " + "-" * 72)
    summary = []
    for cand_name, vt, lev, vfl in CANDIDATES:
        oos = df_all[(df_all["candidate"] == cand_name) & (df_all["is_oos"] == "oos")]
        if oos.empty:
            continue
        cagr_med = oos["cagr_pct"].median()
        cagr_avg = oos["cagr_pct"].mean()
        shr_med = oos["sharpe"].median()
        shr_min = oos["sharpe"].min()
        dd_max = oos["equity_dd_pct"].max()
        sharpe_pass = (oos["sharpe"] >= 1.0).sum()
        n_oos = len(oos)
        gate = f"{sharpe_pass}/{n_oos}"
        print(f"  {cand_name:<18} {cagr_med:>+9.2f} {cagr_avg:>+9.2f} "
              f"{shr_med:>+9.2f} {shr_min:>+9.2f} {dd_max:>+8.2f} {gate:>7}")
        summary.append({
            "candidate": cand_name,
            "vt": vt, "lev": lev, "vfl": vfl,
            "cagr_oos_median": cagr_med,
            "cagr_oos_mean": cagr_avg,
            "sharpe_oos_median": shr_med,
            "sharpe_oos_min": shr_min,
            "dd_oos_max": dd_max,
            "sharpe_pass_count": int(sharpe_pass),
            "n_folds": n_oos,
        })

    pd.DataFrame(summary).to_csv(OUT_DIR / f"summary_{ts}.csv", index=False)

    # IS↔OOS Spearman per candidate
    print(f"\n{'='*78}\n  STABILITE IS↔OOS PAR CANDIDAT\n{'='*78}")
    print(f"  {'Candidat':<18} {'ρ_CAGR':>10} {'ρ_Sharpe':>10} {'OOS/IS Shr':>12}")
    print("  " + "-" * 54)
    for cand_name, vt, lev, vfl in CANDIDATES:
        cand_df = df_all[df_all["candidate"] == cand_name]
        ix = cand_df[cand_df["is_oos"] == "is"].set_index("fold")
        ox = cand_df[cand_df["is_oos"] == "oos"].set_index("fold")
        # Pair fold_is with fold_oos
        is_cagrs, oos_cagrs, is_shrs, oos_shrs = [], [], [], []
        for i in range(1, 6):
            is_row = ix[ix.index == f"fold{i}_is"]
            oos_row = ox[ox.index == f"fold{i}_oos"]
            if is_row.empty or oos_row.empty:
                continue
            is_cagrs.append(is_row["cagr_pct"].iloc[0])
            oos_cagrs.append(oos_row["cagr_pct"].iloc[0])
            is_shrs.append(is_row["sharpe"].iloc[0])
            oos_shrs.append(oos_row["sharpe"].iloc[0])
        if len(is_cagrs) < 3:
            continue
        rho_c, _ = spearmanr(is_cagrs, oos_cagrs)
        rho_s, _ = spearmanr(is_shrs, oos_shrs)
        ratio = np.mean(oos_shrs) / np.mean(is_shrs) if np.mean(is_shrs) > 0 else 0
        print(f"  {cand_name:<18} {rho_c:>+10.3f} {rho_s:>+10.3f} {ratio:>+12.3f}")

    print(f"\n  → {OUT_DIR / f'per_fold_{ts}.csv'}")
    print(f"  → {OUT_DIR / f'summary_{ts}.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
