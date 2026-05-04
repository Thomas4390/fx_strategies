#!/usr/bin/env python3
"""walkforward_3d — orchestre 3 sweeps MT5 (full / IS / OOS) + analyse robuste.

Best practices anti-overfit appliquées :
  1. Walk-forward IS/OOS — split de la fenêtre 5.4 ans en :
       IS  = 2020-11-23 → 2024-10-31  (4.0 ans, ~74 % de l'historique)
       OOS = 2024-11-01 → 2026-04-30  (1.5 ans, ~26 %)
  2. Identification de plateau — pour chaque config, score "robust" =
     mean(8 voisins immédiats sur la grille 3D).
  3. Stabilité du rang IS↔OOS — corrélation Spearman + PBO approché.
  4. Filtre robust optima — top quartile dans IS ET OOS simultanément.

Usage :
    python scripts/optimization/walkforward_3d.py
    python scripts/optimization/walkforward_3d.py --skip-full --skip-is
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
OUT_DIR = ROOT / "reports/optimization/walkforward_3d"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Grille — élargie pour couvrir le levier broker (jusqu'à 80)
VT_GRID = "0.10,0.50,0.05"      # start, stop, step → 9 valeurs
LEV_GRID = "8,80,8"              # 10 valeurs (8 → 80)
VFLOOR_GRID = "0.01,0.02,0.04,0.08"

# Walk-forward : 74% / 26% split
IS_FROM,  IS_TO  = "2020.11.23", "2024.10.31"
OOS_FROM, OOS_TO = "2024.11.01", "2026.04.30"
FULL_FROM, FULL_TO = "2020.11.23", "2026.04.30"


def run_sweep(name: str, from_date: str, to_date: str,
              timeout: int = 1200) -> Path:
    """Lance un sweep et renvoie le chemin du CSV produit."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    prefix = f"wf_{name}_{ts}"
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
    print(f"  cmd: {' '.join(cmd[2:])}", flush=True)
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
    # Round to handle floating-point drift
    df["vt_r"] = df["target_vol"].round(4)
    df["lev_r"] = df["max_lev"].round(2)
    df["vfl_r"] = df["vol_floor"].round(4)
    df["calmar"] = df["cagr_pct"] / df["equity_dd_pct"].abs().replace(0, np.nan)
    return df


def compute_robust_score(df: pd.DataFrame, metric: str) -> pd.Series:
    """Pour chaque cellule, score robust = mean(metric) sur la cellule + voisins
    sur (vt±1step, lev±1step) en gardant vfloor fixe."""
    scores = []
    pivots = {vfl: g.pivot(index="vt_r", columns="lev_r", values=metric)
              for vfl, g in df.groupby("vfl_r")}
    for _, row in df.iterrows():
        pv = pivots.get(row["vfl_r"])
        if pv is None or pv.empty:
            scores.append(np.nan)
            continue
        vt_idx = list(pv.index)
        lev_idx = list(pv.columns)
        try:
            i = vt_idx.index(row["vt_r"])
            j = lev_idx.index(row["lev_r"])
        except ValueError:
            scores.append(np.nan)
            continue
        i0, i1 = max(0, i-1), min(len(vt_idx), i+2)
        j0, j1 = max(0, j-1), min(len(lev_idx), j+2)
        block = pv.values[i0:i1, j0:j1]
        scores.append(float(np.nanmean(block)))
    return pd.Series(scores, index=df.index, name=f"{metric}_robust")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip-full", action="store_true")
    ap.add_argument("--skip-is",   action="store_true")
    ap.add_argument("--skip-oos",  action="store_true")
    ap.add_argument("--skip-analysis", action="store_true")
    ap.add_argument("--full-csv", default=None, help="reuse existing full CSV")
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

    if args.skip_analysis or not (is_csv and oos_csv):
        return 0

    print(f"\n{'='*70}\n  ANALYSE WALK-FORWARD IS/OOS\n{'='*70}")
    df_is = load_sweep(is_csv)
    df_oos = load_sweep(oos_csv)
    print(f"  IS  : {len(df_is)} configs ({IS_FROM} → {IS_TO})")
    print(f"  OOS : {len(df_oos)} configs ({OOS_FROM} → {OOS_TO})")

    # 1) Robust score = mean voisins
    df_is["cagr_robust"] = compute_robust_score(df_is, "cagr_pct")
    df_is["sharpe_robust"] = compute_robust_score(df_is, "sharpe")

    # 2) Merge IS/OOS sur (vt, lev, vfloor)
    keys = ["vt_r", "lev_r", "vfl_r"]
    merged = df_is.merge(df_oos, on=keys, suffixes=("_is", "_oos"))
    print(f"  Joined : {len(merged)} configs sur (vt, lev, vfloor)")

    # 3) Stabilité du rang : Spearman sur CAGR et Sharpe
    from scipy.stats import spearmanr
    rho_cagr, p_cagr = spearmanr(merged["cagr_pct_is"], merged["cagr_pct_oos"])
    rho_sharpe, p_sharpe = spearmanr(merged["sharpe_is"], merged["sharpe_oos"])
    print(f"\n  Spearman ρ (CAGR IS↔OOS)   : {rho_cagr:+.3f} (p={p_cagr:.3e})")
    print(f"  Spearman ρ (Sharpe IS↔OOS) : {rho_sharpe:+.3f} (p={p_sharpe:.3e})")
    print("  ↗ ρ → 1.0 = très stable  |  ρ ≈ 0 = sélection IS = pile/face en OOS")

    # 4) PBO approché : pour chaque config "best-IS", proba qu'elle reste
    #    au-dessus de la médiane OOS. PBO = 1 - frequency_above_median_OOS.
    median_oos = merged["cagr_pct_oos"].median()
    top_decile_is = merged.nlargest(max(1, len(merged) // 10), "cagr_pct_is")
    above_median_oos = (top_decile_is["cagr_pct_oos"] > median_oos).mean()
    pbo = 1.0 - above_median_oos
    print(f"\n  Top-10% IS au-dessus de la médiane OOS : {above_median_oos*100:.0f}%")
    print(f"  PBO approximé : {pbo*100:.0f}%  (< 50% = OK, > 50% = overfit suspect)")

    # 5) Robust optima : top quartile IS ET OOS
    q_is = merged["cagr_pct_is"].quantile(0.75)
    q_oos = merged["cagr_pct_oos"].quantile(0.75)
    robust = merged[(merged["cagr_pct_is"] >= q_is)
                    & (merged["cagr_pct_oos"] >= q_oos)].copy()
    robust["cagr_avg"] = (robust["cagr_pct_is"] + robust["cagr_pct_oos"]) / 2
    robust = robust.sort_values("cagr_avg", ascending=False)

    print(f"\n  Robust optima (top quartile IS ET OOS) : {len(robust)} / {len(merged)}")
    if not robust.empty:
        print("\n  TOP 10 configurations robustes (triées par CAGR moyen IS/OOS)")
        print(f"  {'vt':>5} {'lev':>4} {'vfl':>5} | "
              f"{'CAGR_IS':>8} {'CAGR_OOS':>9} {'CAGR_avg':>9} | "
              f"{'DD_IS':>7} {'DD_OOS':>7} | "
              f"{'Shr_IS':>7} {'Shr_OOS':>7}")
        print("  " + "-" * 92)
        for _, r in robust.head(10).iterrows():
            print(f"  {r['vt_r']:>5.2f} {r['lev_r']:>4.0f} {r['vfl_r']:>5.3f} | "
                  f"{r['cagr_pct_is']:>+8.2f} {r['cagr_pct_oos']:>+9.2f} "
                  f"{r['cagr_avg']:>+9.2f} | "
                  f"{r['equity_dd_pct_is']:>+7.2f} {r['equity_dd_pct_oos']:>+7.2f} | "
                  f"{r['sharpe_is']:>+7.2f} {r['sharpe_oos']:>+7.2f}")

    # 6) Persist analysis
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    merged_path = OUT_DIR / f"merged_is_oos_{ts}.csv"
    merged.to_csv(merged_path, index=False)
    if not robust.empty:
        robust_path = OUT_DIR / f"robust_optima_{ts}.csv"
        robust.to_csv(robust_path, index=False)
        print(f"\n  → {robust_path.name}  ({len(robust)} robust configs)")
    print(f"  → {merged_path.name}  ({len(merged)} merged rows)")

    # 7) Plot scatter IS vs OOS CAGR
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return 0
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, metric, label in [
        (axes[0], "cagr_pct", "CAGR (%)"),
        (axes[1], "sharpe", "Sharpe Ratio"),
    ]:
        x = merged[f"{metric}_is"]
        y = merged[f"{metric}_oos"]
        sc = ax.scatter(x, y, c=merged["vfl_r"], cmap="viridis",
                        s=20, alpha=0.7, edgecolors="black", linewidths=0.3)
        ax.axhline(y.median(), color="grey", lw=0.5, ls=":")
        ax.axvline(x.median(), color="grey", lw=0.5, ls=":")
        # Diagonale
        lim = [min(x.min(), y.min()), max(x.max(), y.max())]
        ax.plot(lim, lim, color="red", lw=0.7, alpha=0.5)
        ax.set_xlabel(f"{label} IS (2020-11→2024-10)")
        ax.set_ylabel(f"{label} OOS (2024-11→2026-04)")
        rho, _ = spearmanr(x, y)
        ax.set_title(f"{label} IS↔OOS (Spearman ρ = {rho:+.3f})")
        plt.colorbar(sc, ax=ax, label="vol_floor")
    plt.suptitle("Walk-Forward IS↔OOS — FxMultiSleeve combined portfolio",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    scatter_path = OUT_DIR / f"is_oos_scatter_{ts}.png"
    plt.savefig(scatter_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {scatter_path.name}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
