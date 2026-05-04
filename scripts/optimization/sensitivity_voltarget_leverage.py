#!/usr/bin/env python3
"""sensitivity_voltarget_leverage — analyse de sensibilité CAGR du portfolio combiné.

Fait varier (Inp_GlobalTargetVol, Inp_GlobalMaxLeverage) sur une grille,
lance un backtest MT5 par combo via le wrapper CLI existant, et agrège
les métriques (CAGR, Max DD, Sharpe, Calmar) en CSV + heatmaps.

Usage :
    python scripts/optimization/sensitivity_voltarget_leverage.py
    python scripts/optimization/sensitivity_voltarget_leverage.py \\
        --voltarget-grid 0.10,0.20,0.28,0.40 --maxlev-grid 4,8,12,16,20

Output : reports/optimization/voltarget_leverage_sensitivity_<ts>.{csv,png}
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
WRAPPER = ROOT / "src/mt5/bridge/run_backtest_cli.py"
OUT_DIR = ROOT / "reports/optimization"
OUT_DIR.mkdir(parents=True, exist_ok=True)

INITIAL_DEPOSIT = 10_000.0
WINDOW_YEARS = 5.43  # 2020-11-23 → 2026-04-30


def parse_pct(value: str | None) -> float | None:
    """Parse '−7.21 (927.43)' or '7.21%' or '7.21' → 7.21 (en %)."""
    if not value:
        return None
    m = re.search(r"(-?\d+(?:[.,]\d+)?)\s*%", value)
    if m:
        return float(m.group(1).replace(",", "."))
    m = re.search(r"\(([-+]?\d+(?:[.,]\d+)?)\s*%\)", value)
    if m:
        return float(m.group(1).replace(",", "."))
    try:
        return float(value.replace(" ", "").replace(",", "."))
    except (ValueError, AttributeError):
        return None


def parse_money(value: str | None) -> float | None:
    """Parse '4 615.41' or '4,615.41' → 4615.41."""
    if not value:
        return None
    cleaned = re.sub(r"[^\d.,\-+]", "", value).replace(",", ".")
    try:
        return float(cleaned)
    except ValueError:
        return None


def run_combo(target_vol: float, max_lev: float, run_id: str,
              timeout: int = 600) -> dict:
    """Lance un backtest avec les overrides. Renvoie un dict de métriques."""
    report_name = f"sens_{run_id}"
    cmd = [
        "python3", str(WRAPPER),
        "--report-name", report_name,
        "--ini-name", f"sens_{run_id}.ini",
        "--input", f"Inp_GlobalTargetVol={target_vol}",
        "--input", f"Inp_GlobalMaxLeverage={max_lev}",
    ]
    started = time.monotonic()
    result = subprocess.run(cmd, cwd=str(ROOT), capture_output=True,
                            text=True, timeout=timeout)
    duration = time.monotonic() - started

    if result.returncode != 0:
        return {
            "target_vol": target_vol, "max_lev": max_lev,
            "run_id": run_id, "duration_s": duration,
            "error": f"exit={result.returncode}",
            "stderr_tail": result.stderr[-300:] if result.stderr else "",
        }

    json_files = sorted((ROOT / "reports/mt5").glob("run_*.json"),
                        key=lambda p: p.stat().st_mtime)
    if not json_files:
        return {"target_vol": target_vol, "max_lev": max_lev,
                "run_id": run_id, "error": "no JSON output"}
    payload = json.loads(json_files[-1].read_text())
    metrics = payload.get("metrics", {})

    net_profit = parse_money(metrics.get("total_net_profit"))
    sharpe = parse_money(metrics.get("sharpe_ratio"))
    profit_factor = parse_money(metrics.get("profit_factor"))
    recovery_factor = parse_money(metrics.get("recovery_factor"))
    trades = parse_money(metrics.get("total_trades"))
    eq_dd_pct = parse_pct(metrics.get("equity_dd_max"))
    bal_dd_pct = parse_pct(metrics.get("balance_dd_max"))

    if net_profit is not None:
        final_balance = INITIAL_DEPOSIT + net_profit
        if final_balance > 0:
            cagr_pct = ((final_balance / INITIAL_DEPOSIT) ** (1.0 / WINDOW_YEARS)
                        - 1.0) * 100.0
        else:
            cagr_pct = None
    else:
        cagr_pct = None

    calmar = (cagr_pct / abs(eq_dd_pct)
              if (cagr_pct is not None and eq_dd_pct not in (None, 0))
              else None)

    return {
        "target_vol": target_vol,
        "max_lev": max_lev,
        "run_id": run_id,
        "duration_s": round(duration, 1),
        "cagr_pct": cagr_pct,
        "net_profit": net_profit,
        "equity_dd_pct": eq_dd_pct,
        "balance_dd_pct": bal_dd_pct,
        "sharpe": sharpe,
        "calmar": calmar,
        "profit_factor": profit_factor,
        "recovery_factor": recovery_factor,
        "trades": int(trades) if trades is not None else None,
        "json_file": str(json_files[-1].name),
    }


def parse_grid(spec: str) -> list[float]:
    return [float(x.strip()) for x in spec.split(",") if x.strip()]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--voltarget-grid", default="0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50")
    p.add_argument("--maxlev-grid",   default="2,4,6,8,10,12,14,16,18,20")
    p.add_argument("--out-prefix",    default=None)
    p.add_argument("--no-plot",       action="store_true")
    args = p.parse_args()

    vol_grid = parse_grid(args.voltarget_grid)
    lev_grid = parse_grid(args.maxlev_grid)
    n_combos = len(vol_grid) * len(lev_grid)
    print(f"=== Sensitivity sweep — {len(vol_grid)} × {len(lev_grid)} = "
          f"{n_combos} combos ===", flush=True)
    print(f"  TargetVol grid : {vol_grid}", flush=True)
    print(f"  MaxLev grid    : {lev_grid}", flush=True)
    estimated_min = n_combos * 22 / 60
    print(f"  Estimated time : ~{estimated_min:.0f} min séquentiel\n", flush=True)

    rows = []
    started_total = time.monotonic()
    for i, vt in enumerate(vol_grid):
        for j, lev in enumerate(lev_grid):
            idx = i * len(lev_grid) + j + 1
            run_id = f"vt{vt:.2f}_lev{lev:.0f}".replace(".", "p")
            elapsed = time.monotonic() - started_total
            print(f"[{idx:3d}/{n_combos}] vt={vt:.2f} lev={lev:.0f} "
                  f"(elapsed={elapsed/60:.1f} min)", flush=True)
            r = run_combo(vt, lev, run_id)
            if "error" in r:
                print(f"  ✗ ERROR: {r['error']}", flush=True)
            else:
                cagr_str = f"{r['cagr_pct']:+.2f}%" if r['cagr_pct'] is not None else "n/a"
                dd_str = f"{r['equity_dd_pct']:+.2f}%" if r['equity_dd_pct'] is not None else "n/a"
                print(f"  ✓ CAGR={cagr_str}  DD={dd_str}  "
                      f"Sharpe={r['sharpe']:+.2f}  Calmar={r['calmar']:+.2f}"
                      if r['sharpe'] is not None and r['calmar'] is not None
                      else f"  ✓ partial: {r}", flush=True)
            rows.append(r)

    df = pd.DataFrame(rows)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    prefix = args.out_prefix or f"voltarget_leverage_sensitivity_{ts}"
    csv_path = OUT_DIR / f"{prefix}.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n=== {len(df)} rows → {csv_path} ===")

    # Plot heatmap CAGR & Calmar
    if args.no_plot:
        return 0
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[warn] matplotlib not available — skipping plots", flush=True)
        return 0

    pivot_cagr = df.pivot(index="target_vol", columns="max_lev",
                          values="cagr_pct")
    pivot_dd = df.pivot(index="target_vol", columns="max_lev",
                        values="equity_dd_pct")
    pivot_sharpe = df.pivot(index="target_vol", columns="max_lev",
                            values="sharpe")
    pivot_calmar = df.pivot(index="target_vol", columns="max_lev",
                            values="calmar")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax, pv, title, cmap in [
        (axes[0, 0], pivot_cagr, "CAGR (%)", "RdYlGn"),
        (axes[0, 1], pivot_dd, "Equity Max DD (%)", "RdYlGn"),
        (axes[1, 0], pivot_sharpe, "Sharpe Ratio", "RdYlGn"),
        (axes[1, 1], pivot_calmar, "Calmar (CAGR/|DD|)", "RdYlGn"),
    ]:
        im = ax.imshow(pv.values, aspect="auto", cmap=cmap, origin="lower")
        ax.set_xticks(range(len(pv.columns)))
        ax.set_xticklabels([f"{c:g}" for c in pv.columns])
        ax.set_yticks(range(len(pv.index)))
        ax.set_yticklabels([f"{i:.2f}" for i in pv.index])
        ax.set_xlabel("Inp_GlobalMaxLeverage")
        ax.set_ylabel("Inp_GlobalTargetVol")
        ax.set_title(title)
        # Annoter chaque cellule
        for ii in range(pv.shape[0]):
            for jj in range(pv.shape[1]):
                v = pv.values[ii, jj]
                if pd.notna(v):
                    ax.text(jj, ii, f"{v:.1f}", ha="center", va="center",
                            fontsize=7, color="black")
        plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)

    plt.suptitle("Sensitivity (Inp_GlobalTargetVol × Inp_GlobalMaxLeverage) "
                 "— FxMultiSleeve combined portfolio, 2020-11 → 2026-04",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    png_path = OUT_DIR / f"{prefix}.png"
    plt.savefig(png_path, dpi=110, bbox_inches="tight")
    print(f"=== Heatmap → {png_path} ===")

    # Top 5 par CAGR
    print(f"\n=== TOP 5 par CAGR ===")
    df_sorted = df.dropna(subset=["cagr_pct"]).sort_values(
        "cagr_pct", ascending=False)
    for _, r in df_sorted.head(5).iterrows():
        print(f"  vt={r['target_vol']:.2f} lev={r['max_lev']:.0f}: "
              f"CAGR={r['cagr_pct']:+.2f}% DD={r['equity_dd_pct']:+.2f}% "
              f"Sharpe={r['sharpe']:+.2f} Calmar={r['calmar']:+.2f}")

    print(f"\n=== TOP 5 par Calmar (CAGR/|DD|) ===")
    df_sorted = df.dropna(subset=["calmar"]).sort_values(
        "calmar", ascending=False)
    for _, r in df_sorted.head(5).iterrows():
        print(f"  vt={r['target_vol']:.2f} lev={r['max_lev']:.0f}: "
              f"CAGR={r['cagr_pct']:+.2f}% DD={r['equity_dd_pct']:+.2f}% "
              f"Sharpe={r['sharpe']:+.2f} Calmar={r['calmar']:+.2f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
