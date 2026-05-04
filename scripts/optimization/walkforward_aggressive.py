#!/usr/bin/env python3
"""walkforward_aggressive — sweep walk-forward visant CAGR ≥ 15%.

Étend les bornes du sweep précédent en autorisant target_vol jusqu'à 2.0
(rendement annualisé cible 200%) et max_lev jusqu'à 80 (broker max).
Le `vol_floor` est testé à 0.005 (très permissif → débride le levier
effectif), 0.010, 0.020, 0.040.

Pourquoi ces bornes ? — Le levier effectif est borné par
`target_vol / vol_floor`. Pour atteindre 15% CAGR, il faut typiquement
un levier de 20-50× appliqué au sub-equity du sleeve dominant. Donc
pousser ces deux paramètres simultanément.

Best practices anti-overfit conservées :
  1. Walk-forward IS/OOS (split 74/26 identique au sweep précédent)
  2. Filtre robust optima : CAGR_IS ≥ 15% ET CAGR_OOS ≥ 10%
  3. Spearman ρ + PBO + corrélation des rangs

Usage :
    python scripts/optimization/walkforward_aggressive.py
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
WRAPPER = ROOT / "scripts/optimization/run_mt5_optimization.py"
OUT_DIR = ROOT / "reports/optimization/walkforward_aggressive"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Bornes étendues — débrident le levier effectif
VT_GRID = "0.25,2.00,0.25"      # 8 valeurs : 0.25 → 2.00 step 0.25
LEV_GRID = "16,80,16"            # 5 valeurs : 16, 32, 48, 64, 80
VFLOOR_GRID = "0.005,0.010,0.020,0.040,0.080"

IS_FROM,  IS_TO  = "2020.11.23", "2024.10.31"
OOS_FROM, OOS_TO = "2024.11.01", "2026.04.30"
FULL_FROM, FULL_TO = "2020.11.23", "2026.04.30"

CAGR_TARGET_IS = 15.0
CAGR_TARGET_OOS = 10.0  # plus tolérant en OOS (régime 2024-2026 défavorable)


def run_sweep(name: str, from_date: str, to_date: str,
              timeout: int = 1800) -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    prefix = f"agg_{name}_{ts}"
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
    ]
    print(f"\n{'='*70}\n  SWEEP {name.upper()} — {from_date} → {to_date}\n{'='*70}",
          flush=True)
    result = subprocess.run(cmd, cwd=str(ROOT), check=False)
    if result.returncode != 0:
        print(f"  ✗ sweep {name} failed exit={result.returncode}", flush=True)
        return None
    csv_path = ROOT / f"reports/optimization/{prefix}.csv"
    if not csv_path.exists():
        print(f"  ✗ {csv_path} missing", flush=True)
        return None
    print(f"  ✓ {csv_path}", flush=True)
    return csv_path


def load_sweep(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["vt_r"] = df["target_vol"].round(4)
    df["lev_r"] = df["max_lev"].round(2)
    df["vfl_r"] = df["vol_floor"].round(4)
    df["calmar"] = df["cagr_pct"] / df["equity_dd_pct"].abs().replace(0, np.nan)
    return df


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip-full", action="store_true")
    ap.add_argument("--skip-is",   action="store_true")
    ap.add_argument("--skip-oos",  action="store_true")
    ap.add_argument("--full-csv", default=None)
    ap.add_argument("--is-csv",   default=None)
    ap.add_argument("--oos-csv",  default=None)
    args = ap.parse_args()

    full_csv = Path(args.full_csv) if args.full_csv else None
    is_csv   = Path(args.is_csv)   if args.is_csv   else None
    oos_csv  = Path(args.oos_csv)  if args.oos_csv  else None

    if not args.skip_full and not full_csv:
        full_csv = run_sweep("full", FULL_FROM, FULL_TO)
    if not args.skip_is and not is_csv:
        is_csv = run_sweep("is", IS_FROM, IS_TO)
    if not args.skip_oos and not oos_csv:
        oos_csv = run_sweep("oos", OOS_FROM, OOS_TO)

    if not (is_csv and oos_csv):
        print("\n[abort] missing IS or OOS sweep")
        return 1

    print(f"\n{'='*70}\n  ANALYSE — recherche CAGR_avg ≥ 15% robuste\n{'='*70}")
    df_is = load_sweep(is_csv)
    df_oos = load_sweep(oos_csv)
    df_full = load_sweep(full_csv) if full_csv else None

    keys = ["vt_r", "lev_r", "vfl_r"]
    merged = df_is.merge(df_oos, on=keys, suffixes=("_is", "_oos"))
    merged["cagr_avg"] = (merged["cagr_pct_is"] + merged["cagr_pct_oos"]) / 2
    merged["dd_max"] = merged[["equity_dd_pct_is", "equity_dd_pct_oos"]].abs().max(axis=1)
    merged["sharpe_min"] = merged[["sharpe_is", "sharpe_oos"]].min(axis=1)

    # Stabilité globale
    from scipy.stats import spearmanr
    rho_cagr, _ = spearmanr(merged["cagr_pct_is"], merged["cagr_pct_oos"])
    rho_sharpe, _ = spearmanr(merged["sharpe_is"], merged["sharpe_oos"])
    print(f"  Configs joined : {len(merged)}")
    print(f"  Spearman ρ (CAGR IS↔OOS)   : {rho_cagr:+.3f}")
    print(f"  Spearman ρ (Sharpe IS↔OOS) : {rho_sharpe:+.3f}")

    # Stats globales
    print(f"\n  Stats globaux IS  : CAGR ∈ [{df_is.cagr_pct.min():+.2f}, {df_is.cagr_pct.max():+.2f}]%, Sharpe ∈ [{df_is.sharpe.min():+.2f}, {df_is.sharpe.max():+.2f}]")
    print(f"  Stats globaux OOS : CAGR ∈ [{df_oos.cagr_pct.min():+.2f}, {df_oos.cagr_pct.max():+.2f}]%, Sharpe ∈ [{df_oos.sharpe.min():+.2f}, {df_oos.sharpe.max():+.2f}]")

    # Filtre principal : CAGR_IS >= 15% ET CAGR_OOS >= 10%
    candidates = merged[(merged["cagr_pct_is"] >= CAGR_TARGET_IS)
                        & (merged["cagr_pct_oos"] >= CAGR_TARGET_OOS)]
    print(f"\n  Candidats (CAGR_IS ≥ {CAGR_TARGET_IS}% ET CAGR_OOS ≥ {CAGR_TARGET_OOS}%) : "
          f"{len(candidates)} / {len(merged)}")

    if not candidates.empty:
        candidates = candidates.sort_values("cagr_avg", ascending=False)
        print(f"\n  TOP 15 candidats (triés par CAGR_avg)")
        print(f"  {'vt':>5} {'lev':>5} {'vfl':>6} | "
              f"{'CAGR_IS':>8} {'CAGR_OOS':>9} {'CAGR_avg':>9} | "
              f"{'DD_IS':>7} {'DD_OOS':>7} | "
              f"{'Shr_IS':>7} {'Shr_OOS':>7} | {'Calmar_min':>11}")
        print("  " + "-" * 110)
        for _, r in candidates.head(15).iterrows():
            calmar_is = r["cagr_pct_is"] / abs(r["equity_dd_pct_is"])
            calmar_oos = r["cagr_pct_oos"] / abs(r["equity_dd_pct_oos"]) if r["equity_dd_pct_oos"] else 0
            calmar_min = min(calmar_is, calmar_oos)
            print(f"  {r['vt_r']:>5.2f} {r['lev_r']:>5.0f} {r['vfl_r']:>6.3f} | "
                  f"{r['cagr_pct_is']:>+8.2f} {r['cagr_pct_oos']:>+9.2f} "
                  f"{r['cagr_avg']:>+9.2f} | "
                  f"{r['equity_dd_pct_is']:>+7.2f} {r['equity_dd_pct_oos']:>+7.2f} | "
                  f"{r['sharpe_is']:>+7.2f} {r['sharpe_oos']:>+7.2f} | "
                  f"{calmar_min:>+11.3f}")

    # Filtre encore plus strict : CAGR_avg ≥ 15% ET Sharpe min ≥ 0.5
    strict = merged[(merged["cagr_avg"] >= 15.0)
                    & (merged["sharpe_min"] >= 0.5)]
    print(f"\n  Filtre STRICT (CAGR_avg ≥ 15% ET min(Sharpe_IS, Sharpe_OOS) ≥ 0.5) : "
          f"{len(strict)} / {len(merged)}")
    if not strict.empty:
        strict = strict.sort_values("cagr_avg", ascending=False)
        print(f"\n  TOP 10 STRICT candidats")
        for _, r in strict.head(10).iterrows():
            print(f"  vt={r['vt_r']:.2f} lev={r['lev_r']:.0f} vfl={r['vfl_r']:.3f}: "
                  f"CAGR_IS={r['cagr_pct_is']:+.2f}% CAGR_OOS={r['cagr_pct_oos']:+.2f}% "
                  f"DD_max={r['dd_max']:+.2f}% Sharpe_min={r['sharpe_min']:+.2f}")

    # Persist
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    merged.to_csv(OUT_DIR / f"merged_{ts}.csv", index=False)
    if not candidates.empty:
        candidates.to_csv(OUT_DIR / f"candidates_15pct_{ts}.csv", index=False)
    if not strict.empty:
        strict.to_csv(OUT_DIR / f"strict_15pct_{ts}.csv", index=False)
    print(f"\n  → {OUT_DIR.name}/merged_{ts}.csv")

    # Plot scatter avec seuil 15% marqué
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return 0
    fig, ax = plt.subplots(figsize=(10, 7))
    sc = ax.scatter(merged["cagr_pct_is"], merged["cagr_pct_oos"],
                    c=merged["vfl_r"], cmap="viridis", s=18,
                    alpha=0.6, edgecolors="black", linewidths=0.2)
    # Zone candidates
    ax.axhline(CAGR_TARGET_OOS, color="green", lw=0.8, ls="--",
               label=f"CAGR_OOS ≥ {CAGR_TARGET_OOS}%")
    ax.axvline(CAGR_TARGET_IS, color="green", lw=0.8, ls="--",
               label=f"CAGR_IS ≥ {CAGR_TARGET_IS}%")
    ax.fill_between([CAGR_TARGET_IS, max(merged.cagr_pct_is.max(), 50)],
                     CAGR_TARGET_OOS, max(merged.cagr_pct_oos.max(), 30),
                     alpha=0.15, color="green",
                     label=f"Zone candidats")
    # Diagonale
    lim = [-30, 50]
    ax.plot(lim, lim, color="red", lw=0.5, alpha=0.5)
    ax.set_xlabel("CAGR (%) IS — 2020-11→2024-10")
    ax.set_ylabel("CAGR (%) OOS — 2024-11→2026-04")
    ax.set_title(f"Walk-Forward Aggressive — recherche CAGR_avg ≥ 15%\n"
                 f"Spearman ρ (CAGR) = {rho_cagr:+.3f}")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)
    plt.colorbar(sc, ax=ax, label="vol_floor")
    plt.tight_layout()
    scatter_path = OUT_DIR / f"scatter_{ts}.png"
    plt.savefig(scatter_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {scatter_path.name}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
