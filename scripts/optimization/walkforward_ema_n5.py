"""walkforward_ema_n5 — Phase E.2 N=5 OOS validation sur top 2 EMA combos."""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RUN_CLI = REPO / "src/mt5/bridge/run_backtest_cli.py"
REPORTS_MT5 = REPO / "reports/mt5"
OUT_DIR = REPO / "reports/optimization/ts_ema"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CANDIDATES = [
    ("baseline_20_50", 20, 50),
    ("ts_14_50", 14, 50),
    ("ts_30_50", 30, 50),
]

FOLDS = [
    ("fold1_oos", "2021.11.01", "2022.10.31"),
    ("fold2_oos", "2022.11.01", "2023.10.31"),
    ("fold3_oos", "2023.11.01", "2024.10.31"),
    ("fold4_oos", "2024.11.01", "2025.10.31"),
    ("fold5_oos", "2025.11.01", "2026.04.30"),
]


def run_bt(label, fast, slow, fold, fd, td):
    cmd = [
        sys.executable, str(RUN_CLI),
        "--from", fd, "--to", td,
        "--report-name", f"e2n5_{label}_{fold}",
        "--input", f"Inp_TS_FastEMA={fast}",
        "--input", f"Inp_TS_SlowEMA={slow}",
    ]
    print(f"  [run] {label} {fold}")
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if proc.returncode != 0:
        print(proc.stdout[-500:])
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
    print("=" * 70)
    print("Phase E.2 — Walk-forward N=5 OOS sur top 2 EMA combos")
    print("=" * 70 + "\n")

    rows = []
    for label, f, s in CANDIDATES:
        for fold, fd, td in FOLDS:
            d = run_bt(label, f, s, fold, fd, td)
            m = parse(d)
            rows.append({"label": label, "fast": f, "slow": s,
                         "fold": fold, **m})

    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "n5.csv", index=False)

    print(f"\n{'='*70}\n  N=5 OOS\n{'='*70}")
    summary = []
    for label, f, s in CANDIDATES:
        sub = df[df["label"] == label]
        rec = {
            "label": label,
            "ema": f"{f}/{s}",
            "sharpe_med": sub["sharpe"].median(),
            "sharpe_avg": sub["sharpe"].mean(),
            "net_avg": sub["net"].mean(),
            "max_dd": sub["max_dd_pct"].max(),
        }
        summary.append(rec)
        print(f"  {label:<18} EMA={f}/{s:<3} "
              f"Sharpe_med={rec['sharpe_med']:.2f} "
              f"Sharpe_avg={rec['sharpe_avg']:.2f} "
              f"Net_avg={rec['net_avg']:.0f} "
              f"DD_max={rec['max_dd']:.2f}%")

    base = summary[0]
    print(f"\n=== VERDICT vs baseline 20/50 ===")
    for s in summary[1:]:
        ds = s["sharpe_med"] - base["sharpe_med"]
        dn = s["net_avg"] - base["net_avg"]
        dd_delta = s["max_dd"] - base["max_dd"]
        verdict = "✓" if ds >= 0.05 and dd_delta <= 2.0 else "✗"
        print(f"  {s['label']:<18} ΔSharpe={ds:+.2f} ΔNet={dn:+.0f} "
              f"ΔDD={dd_delta:+.2f}pp → {verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
