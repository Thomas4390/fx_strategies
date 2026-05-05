"""walkforward_rsi_thresh — Phase E.3 (Plan CAGR docs/investigations).

Phase B révèle RSI Daily flat (PF 1.01, net +6.82 USD sur 5.4 ans) avec
USDJPY drag -295 USD. Test :
  1. Grid (oversold, overbought, exit_mid) sur full 5.4y
  2. Variante : retirer USDJPY de Inp_RSI_Pairs

Critère : Sharpe sleeve standalone (alloc 0/0/1/0) ≥ 0.5 OR ΔSharpe global
(combiné) ≥ +0.05.
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
OUT_DIR = REPO / "reports/optimization/rsi_thresh"
OUT_DIR.mkdir(parents=True, exist_ok=True)

VARIANTS = [
    # label, oversold, overbought, exit_mid, pairs
    ("baseline", 25, 75, 50, "EURUSD,GBPUSD,USDJPY,USDCAD"),
    ("strict_20_80", 20, 80, 50, "EURUSD,GBPUSD,USDJPY,USDCAD"),
    ("loose_30_70", 30, 70, 50, "EURUSD,GBPUSD,USDJPY,USDCAD"),
    ("exit_55", 25, 75, 55, "EURUSD,GBPUSD,USDJPY,USDCAD"),
    ("exit_60", 20, 80, 60, "EURUSD,GBPUSD,USDJPY,USDCAD"),
    ("no_jpy", 25, 75, 50, "EURUSD,GBPUSD,USDCAD"),
    ("no_jpy_strict", 20, 80, 50, "EURUSD,GBPUSD,USDCAD"),
    ("no_jpy_loose", 30, 70, 50, "EURUSD,GBPUSD,USDCAD"),
]


def run_bt(label, oversold, overbought, exit_mid, pairs):
    cmd = [
        sys.executable, str(RUN_CLI),
        "--report-name", f"e3_rsi_{label}",
        "--input", f"Inp_RSI_Oversold={oversold}",
        "--input", f"Inp_RSI_Overbought={overbought}",
        "--input", f"Inp_RSI_ExitMid={exit_mid}",
        "--input", f"Inp_RSI_Pairs={pairs}",
    ]
    print(f"  [run] {label} ({oversold}/{overbought}/{exit_mid}) {pairs}")
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
    print("Phase E.3 — RSI Daily seuils + pairs (combiné, full 5.4y)")
    print("=" * 70 + "\n")

    rows = []
    for v in VARIANTS:
        d = run_bt(*v)
        m = parse(d)
        rows.append({"label": v[0], "oversold": v[1], "overbought": v[2],
                     "exit_mid": v[3], "pairs": v[4], **m})

    import pandas as pd
    df = pd.DataFrame(rows).sort_values("sharpe", ascending=False)
    df.to_csv(OUT_DIR / "grid.csv", index=False)

    print(f"\n{'label':<18} {'os/ob/ex':<10} {'pairs':<28} "
          f"{'Sharpe':>7} {'Net':>9} {'DD%':>6} {'Trades':>7}")
    print("-" * 90)
    for _, r in df.iterrows():
        thr = f"{int(r['oversold'])}/{int(r['overbought'])}/{int(r['exit_mid'])}"
        print(f"{r['label']:<18} {thr:<10} {r['pairs']:<28} "
              f"{r['sharpe']:>7.2f} {r['net']:>9.0f} "
              f"{r['max_dd_pct']:>6.2f} {int(r['trades']):>7d}")

    base = df[df["label"] == "baseline"].iloc[0]
    print(f"\nBaseline RSI : Sharpe={base['sharpe']:.2f} Net={base['net']:.0f}")
    print(f"\nTop variants vs baseline :")
    for _, r in df.head(5).iterrows():
        ds = r["sharpe"] - base["sharpe"]
        verdict = "✓" if ds >= 0.05 else "✗"
        print(f"  {r['label']:<18} ΔSharpe={ds:+.2f} → {verdict}")

    print(f"\n[write] {OUT_DIR / 'grid.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
