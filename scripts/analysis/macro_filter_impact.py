"""macro_filter_impact — Phase B.4 (Plan CAGR docs/investigations).

Mesure l'impact du filtre macro sur le sleeve MR_Macro en comparant deux runs :

1. Baseline           : `Inp_MR_SpreadThresh = 0.5` (filtre actif normal)
2. Macro-filter-off   : `Inp_MR_SpreadThresh = -10.0` (toujours satisfait)

Sur la fenêtre fold5 (2025-11-01 → 2026-04-30) — la fenêtre où walkforward N=5
a observé une dégradation OOS. Hypothèse : le filtre macro bloque trop de
signaux MR sur ce régime → relâcher pourrait débloquer.

Usage :
    python scripts/analysis/macro_filter_impact.py

Output :
    reports/analysis/macro_filter_impact.csv (table comparaison)
    stdout : verdict (relâcher OUI/NON)
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RUN_CLI = REPO / "src/mt5/bridge/run_backtest_cli.py"
REPORTS = REPO / "reports/mt5"
ANALYSIS_OUT = REPO / "reports/analysis"

FOLD5_FROM = "2025.11.01"
FOLD5_TO = "2026.04.30"


def run_backtest(report_name: str, extra_inputs: list[str]) -> dict:
    """Run un backtest fold5 et retourne le JSON dump."""
    cmd = [
        sys.executable,
        str(RUN_CLI),
        "--from", FOLD5_FROM,
        "--to", FOLD5_TO,
        "--report-name", report_name,
        "--input", "Inp_ExportDeals=true",
    ]
    for inp in extra_inputs:
        cmd += ["--input", inp]
    print(f"[run] {' '.join(cmd[2:])}")
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if proc.returncode != 0:
        print("STDOUT:\n", proc.stdout)
        print("STDERR:\n", proc.stderr)
        raise RuntimeError(f"run_backtest_cli exit={proc.returncode}")
    # Find latest JSON
    jsons = sorted(REPORTS.glob("run_*.json"), key=lambda p: p.stat().st_mtime)
    if not jsons:
        raise RuntimeError("no JSON dump found")
    return json.loads(jsons[-1].read_text())


def main() -> int:
    ANALYSIS_OUT.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"Phase B.4 — Macro filter impact on fold5 [{FOLD5_FROM} → {FOLD5_TO}]")
    print("=" * 60)

    # Run 1 : baseline (filtre macro actif)
    baseline = run_backtest(
        "fold5_macro_on",
        extra_inputs=[],
    )
    # Run 2 : filtre macro complètement bypass (force macro_ok=true)
    no_filter = run_backtest(
        "fold5_macro_off",
        extra_inputs=["Inp_MR_DisableMacroFilter=true"],
    )

    def extract(d: dict) -> dict:
        import re

        m = d.get("metrics", {}) or {}

        def _num(s):
            if s is None:
                return None
            try:
                return float(str(s).replace(" ", "").replace(",", ""))
            except ValueError:
                return None

        def _dd_pct(s):
            if s is None:
                return None
            mt = re.search(r"\(([\d.]+)%\)", str(s))
            return float(mt.group(1)) if mt else None

        return {
            "sharpe": _num(m.get("sharpe_ratio")),
            "net_profit": _num(m.get("total_net_profit")),
            "profit_factor": _num(m.get("profit_factor")),
            "trades": _num(m.get("total_trades")),
            "max_dd_pct": _dd_pct(m.get("equity_dd_max")),
        }

    a = extract(baseline)
    b = extract(no_filter)

    rows = [
        ("config", "Sharpe", "Net", "PF", "Trades", "MaxDD %"),
        ("baseline (macro filter ON)",
         f"{a['sharpe']}", f"{a['net_profit']}", f"{a['profit_factor']}",
         f"{a['trades']}", f"{a['max_dd_pct']}"),
        ("macro filter RELAXED",
         f"{b['sharpe']}", f"{b['net_profit']}", f"{b['profit_factor']}",
         f"{b['trades']}", f"{b['max_dd_pct']}"),
    ]
    csv_lines = [",".join(r) for r in rows]
    out_csv = ANALYSIS_OUT / "macro_filter_impact.csv"
    out_csv.write_text("\n".join(csv_lines))
    print(f"\n[write] {out_csv}\n")

    print(f"{'config':30s} {'Sharpe':>8s} {'Net':>10s} {'PF':>6s} {'Trades':>7s} {'DD%':>7s}")
    for r in rows[1:]:
        print(f"{r[0]:30s} {r[1]:>8s} {r[2]:>10s} {r[3]:>6s} {r[4]:>7s} {r[5]:>7s}")

    # Verdict
    s_a, s_b = a.get("sharpe"), b.get("sharpe")
    if s_a is None or s_b is None:
        print("\n[verdict] missing sharpe values — re-check run logs")
        return 1
    delta = float(s_b) - float(s_a)
    print(f"\nΔSharpe (relaxed - baseline) = {delta:+.3f}")
    if delta > 0.10:
        print("→ VERDICT : RELÂCHER LE FILTRE MACRO sur fold5 améliore Sharpe ≥ 0.10.")
        print("  Recommandation : envisager de baisser Inp_MR_SpreadThresh.")
    elif delta < -0.10:
        print("→ VERDICT : LE FILTRE MACRO PROTÈGE sur fold5 (ΔSharpe < -0.10).")
        print("  Recommandation : garder ou durcir le filtre.")
    else:
        print("→ VERDICT : IMPACT NEUTRE (|ΔSharpe| < 0.10).")
        print("  Le filtre macro n'est pas le facteur dominant sur fold5.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
