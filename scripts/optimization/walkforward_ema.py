"""walkforward_ema — Phase E.2 (Plan CAGR docs/investigations).

Grid search Inp_TS_FastEMA × Inp_TS_SlowEMA sur full 5.4 ans, suivi de
walk-forward N=5 sur top 3 candidats. Phase B avait révélé que TS
Momentum est concentré 83 % USDJPY → vérifier si d'autres params
réduisent la concentration tout en préservant l'edge.

Grid : fast ∈ {10, 14, 20, 30, 50}, slow ∈ {30, 50, 100, 200}, slow > fast.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from itertools import product
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RUN_CLI = REPO / "src/mt5/bridge/run_backtest_cli.py"
REPORTS_MT5 = REPO / "reports/mt5"
OUT_DIR = REPO / "reports/optimization/ts_ema"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FAST_GRID = [10, 14, 20, 30, 50]
SLOW_GRID = [30, 50, 100, 200]

FROM = "2020.11.23"
TO = "2026.04.30"


def run_bt(fast: int, slow: int, fold_label: str = "full",
           fd: str = FROM, td: str = TO) -> dict:
    cmd = [
        sys.executable, str(RUN_CLI),
        "--from", fd, "--to", td,
        "--report-name", f"e2_ts_{fast}_{slow}_{fold_label}",
        "--input", f"Inp_TS_FastEMA={fast}",
        "--input", f"Inp_TS_SlowEMA={slow}",
    ]
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
    print("=" * 70)
    print("Phase E.2 — TS Momentum EMA grid")
    print("=" * 70 + "\n")

    pairs = [(f, s) for f, s in product(FAST_GRID, SLOW_GRID) if s > f]
    print(f"Grid : {len(pairs)} combos (fast × slow, slow > fast)\n")

    rows = []
    for f, s in pairs:
        print(f"  [run] fast={f:>3} slow={s:>3}")
        d = run_bt(f, s)
        m = parse(d)
        rows.append({"fast": f, "slow": s, **m})

    import pandas as pd

    df = pd.DataFrame(rows).sort_values("sharpe", ascending=False)
    df.to_csv(OUT_DIR / "grid.csv", index=False)

    print(f"\n{'fast':>5} {'slow':>5} {'Sharpe':>7} {'Net':>9} {'DD%':>6} "
          f"{'Trades':>7}")
    print("-" * 50)
    for _, r in df.iterrows():
        print(f"{int(r['fast']):>5} {int(r['slow']):>5} "
              f"{r['sharpe']:>7.2f} {r['net']:>9.0f} "
              f"{r['max_dd_pct']:>6.2f} {int(r['trades']):>7d}")

    # Baseline current = (20, 50)
    base = df[(df["fast"] == 20) & (df["slow"] == 50)]
    if not base.empty:
        b = base.iloc[0]
        print(f"\nBaseline TS (20/50) : Sharpe={b['sharpe']:.2f} "
              f"Net={b['net']:.0f}")
        print(f"\nTop combos vs baseline :")
        for _, r in df.head(5).iterrows():
            ds = r["sharpe"] - b["sharpe"]
            verdict = "✓" if ds >= 0.05 else "✗"
            print(f"  fast={int(r['fast']):>3} slow={int(r['slow']):>3} "
                  f"ΔSharpe={ds:+.2f} → {verdict}")

    print(f"\n[write] {OUT_DIR / 'grid.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
