"""walkforward_session_n5 — Phase E.1 validation N=5 sur top session 8-16.

Compare baseline (6-14 UTC) vs candidate (8-16 UTC) sur 5 folds OOS.
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

CANDIDATES = [
    ("baseline_6_14", 6, 14),
    ("london_8_16", 8, 16),
]

FOLDS = [
    ("fold1_oos", "2021.11.01", "2022.10.31"),
    ("fold2_oos", "2022.11.01", "2023.10.31"),
    ("fold3_oos", "2023.11.01", "2024.10.31"),
    ("fold4_oos", "2024.11.01", "2025.10.31"),
    ("fold5_oos", "2025.11.01", "2026.04.30"),
]


def run_bt(label, s, e, fold, fd, td):
    cmd = [
        sys.executable, str(RUN_CLI),
        "--from", fd, "--to", td,
        "--report-name", f"e1n5_{label}_{fold}",
        "--input", f"Inp_MR_SessionStart={s}",
        "--input", f"Inp_MR_SessionEnd={e}",
    ]
    print(f"  [run] {label} {fold} {fd}→{td}")
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if proc.returncode != 0:
        print("STDOUT:\n", proc.stdout[-600:])
        raise RuntimeError(f"exit={proc.returncode}")
    jsons = sorted(REPORTS_MT5.glob("run_*.json"), key=lambda p: p.stat().st_mtime)
    return json.loads(jsons[-1].read_text())


def parse(d):
    m = d.get("metrics", {}) or {}

    def _num(s):
        try:
            return float(str(s).replace(" ", "").replace(",", "")) if s else None
        except ValueError:
            return None

    def _dd(s):
        mt = re.search(r"\(([\d.]+)%\)", str(s)) if s else None
        return float(mt.group(1)) if mt else None

    return {
        "sharpe": _num(m.get("sharpe_ratio")),
        "net": _num(m.get("total_net_profit")),
        "max_dd_pct": _dd(m.get("equity_dd_max")),
        "trades": _num(m.get("total_trades")),
    }


def main():
    print("=" * 60)
    print("Phase E.1 — Walk-forward N=5 sur sessions 6-14 vs 8-16")
    print("=" * 60 + "\n")

    rows = []
    for label, s, e in CANDIDATES:
        for fold, fd, td in FOLDS:
            d = run_bt(label, s, e, fold, fd, td)
            m = parse(d)
            rows.append({"label": label, "fold": fold, "from": fd, "to": td, **m})

    import pandas as pd

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "n5_session.csv", index=False)

    print(f"\n{'='*70}\n  WALK-FORWARD N=5 OOS PAR CANDIDATE\n{'='*70}")
    summary = []
    for label in ["baseline_6_14", "london_8_16"]:
        sub = df[df["label"] == label]
        s = {
            "label": label,
            "sharpe_med": sub["sharpe"].median(),
            "sharpe_avg": sub["sharpe"].mean(),
            "net_avg": sub["net"].mean(),
            "max_dd": sub["max_dd_pct"].max(),
        }
        summary.append(s)
        print(f"  {label:<18} Sharpe_med={s['sharpe_med']:.2f} "
              f"Sharpe_avg={s['sharpe_avg']:.2f} "
              f"Net_avg={s['net_avg']:.0f} "
              f"DD_max={s['max_dd']:.2f}%")

    base = summary[0]
    cand = summary[1]
    ds = cand["sharpe_med"] - base["sharpe_med"]
    dn = cand["net_avg"] - base["net_avg"]
    dd_delta = cand["max_dd"] - base["max_dd"]

    print(f"\n=== VERDICT ===")
    print(f"ΔSharpe_med = {ds:+.2f}")
    print(f"ΔNet_avg    = {dn:+.0f} USD/fold")
    print(f"ΔDD_max     = {dd_delta:+.2f} pp")
    if ds >= 0.05 and dd_delta <= 2.0:
        print(f"→ ✓ RETAIN session 8-16 UTC")
    else:
        print(f"→ ✗ keep baseline 6-14 UTC")
    return 0


if __name__ == "__main__":
    sys.exit(main())
