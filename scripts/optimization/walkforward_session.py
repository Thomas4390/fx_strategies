"""walkforward_session — Phase E.1 (Plan CAGR docs/investigations).

Sweep 4 fenêtres horaires UTC pour MR Macro :
  6-14  (actuelle, London open + early NY)
  8-16  (London full + early NY)
  13-21 (NY full)
  0-23  (24h, no session restriction)

Compare baseline 80/10/10 sur full 5.4 ans. Critère go : ΔSharpe ≥ +0.05.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RUN_CLI = REPO / "src/mt5/bridge/run_backtest_cli.py"
REPORTS_MT5 = REPO / "reports/mt5"
OUT_DIR = REPO / "reports/optimization/sessions"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SESSIONS = [
    ("baseline_6_14", 6, 14),
    ("london_8_16", 8, 16),
    ("ny_13_21", 13, 21),
    ("full_0_23", 0, 23),
]


def run_backtest(label: str, start: int, end: int) -> dict:
    cmd = [
        sys.executable,
        str(RUN_CLI),
        "--report-name", f"e1_session_{label}",
        "--input", f"Inp_MR_SessionStart={start}",
        "--input", f"Inp_MR_SessionEnd={end}",
    ]
    print(f"  [run] session {start}-{end} UTC ({label})")
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if proc.returncode != 0:
        print("STDOUT tail:\n", proc.stdout[-800:])
        raise RuntimeError(f"backtest exit={proc.returncode}")
    jsons = sorted(REPORTS_MT5.glob("run_*.json"), key=lambda p: p.stat().st_mtime)
    return json.loads(jsons[-1].read_text())


def parse_metrics(d: dict) -> dict:
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


def main() -> int:
    print("=" * 60)
    print("Phase E.1 — MR Macro session sweep")
    print("=" * 60 + "\n")

    rows = []
    for label, s, e in SESSIONS:
        d = run_backtest(label, s, e)
        m = parse_metrics(d)
        rows.append({
            "session": f"{s}-{e}",
            "label": label,
            **m,
        })

    import pandas as pd

    df = pd.DataFrame(rows).sort_values("sharpe", ascending=False)
    df.to_csv(OUT_DIR / "session_sweep.csv", index=False)

    print(f"\n{'session':<12} {'label':<18} {'Sharpe':>7} {'Net':>10} "
          f"{'PF':>5} {'DD%':>7} {'Trades':>7}")
    print("-" * 70)
    for _, r in df.iterrows():
        print(f"{r['session']:<12} {r['label']:<18} {r['sharpe']:>7.2f} "
              f"{r['net_profit']:>10.0f} {r['profit_factor']:>5.2f} "
              f"{r['max_dd_pct']:>7.2f} {int(r['trades']):>7d}")

    base = df[df["label"] == "baseline_6_14"].iloc[0]
    print(f"\n=== ΔSharpe vs baseline 6-14 (Sharpe={base['sharpe']:.2f}) ===")
    for _, r in df.iterrows():
        if r["label"] == "baseline_6_14":
            continue
        ds = r["sharpe"] - base["sharpe"]
        verdict = "✓ retain" if ds >= 0.05 else "✗"
        print(f"  {r['session']:<12} ΔSharpe={ds:+.2f}  → {verdict}")

    print(f"\n[write] {OUT_DIR / 'session_sweep.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
