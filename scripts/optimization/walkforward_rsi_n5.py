"""walkforward_rsi_n5 — Phase E.3 N=5 OOS validation."""
from __future__ import annotations

import json, re, subprocess, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RUN_CLI = REPO / "src/mt5/bridge/run_backtest_cli.py"
REPORTS_MT5 = REPO / "reports/mt5"
OUT_DIR = REPO / "reports/optimization/rsi_thresh"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CANDIDATES = [
    ("baseline", "EURUSD,GBPUSD,USDJPY,USDCAD"),
    ("no_jpy", "EURUSD,GBPUSD,USDCAD"),
]
FOLDS = [
    ("fold1_oos", "2021.11.01", "2022.10.31"),
    ("fold2_oos", "2022.11.01", "2023.10.31"),
    ("fold3_oos", "2023.11.01", "2024.10.31"),
    ("fold4_oos", "2024.11.01", "2025.10.31"),
    ("fold5_oos", "2025.11.01", "2026.04.30"),
]


def run_bt(label, pairs, fold, fd, td):
    cmd = [sys.executable, str(RUN_CLI),
           "--from", fd, "--to", td,
           "--report-name", f"e3n5_{label}_{fold}",
           "--input", f"Inp_RSI_Pairs={pairs}"]
    print(f"  [run] {label} {fold}")
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if proc.returncode != 0:
        raise RuntimeError(proc.stdout[-500:])
    jsons = sorted(REPORTS_MT5.glob("run_*.json"), key=lambda p: p.stat().st_mtime)
    return json.loads(jsons[-1].read_text())


def parse(d):
    m = d.get("metrics", {}) or {}
    def _num(s):
        try: return float(str(s).replace(" ", "").replace(",", "")) if s else None
        except ValueError: return None
    def _dd(s):
        mt = re.search(r"\(([\d.]+)%\)", str(s)) if s else None
        return float(mt.group(1)) if mt else None
    return {"sharpe": _num(m.get("sharpe_ratio")),
            "net": _num(m.get("total_net_profit")),
            "max_dd_pct": _dd(m.get("equity_dd_max")),
            "trades": _num(m.get("total_trades"))}


def main():
    print("Phase E.3 — N=5 OOS RSI no_jpy validation\n")
    rows = []
    for label, pairs in CANDIDATES:
        for fold, fd, td in FOLDS:
            d = run_bt(label, pairs, fold, fd, td)
            m = parse(d)
            rows.append({"label": label, "pairs": pairs, "fold": fold, **m})
    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "n5.csv", index=False)
    summary = []
    print(f"\n{'cand':<12} {'Sharpe_med':>10} {'Sharpe_avg':>10} "
          f"{'Net_avg':>9} {'DD_max':>8}")
    for label, _ in CANDIDATES:
        sub = df[df["label"] == label]
        rec = {"label": label,
               "sharpe_med": sub["sharpe"].median(),
               "sharpe_avg": sub["sharpe"].mean(),
               "net_avg": sub["net"].mean(),
               "max_dd": sub["max_dd_pct"].max()}
        summary.append(rec)
        print(f"  {label:<12} {rec['sharpe_med']:>10.2f} {rec['sharpe_avg']:>10.2f} "
              f"{rec['net_avg']:>9.0f} {rec['max_dd']:>8.2f}")
    base = summary[0]
    cand = summary[1]
    ds = cand["sharpe_med"] - base["sharpe_med"]
    dn = cand["net_avg"] - base["net_avg"]
    dd_delta = cand["max_dd"] - base["max_dd"]
    print(f"\n=== VERDICT ===")
    print(f"ΔSharpe_med = {ds:+.2f}  ΔNet_avg = {dn:+.0f}  ΔDD = {dd_delta:+.2f}pp")
    print("→ ✓ RETAIN no_jpy" if ds >= 0.05 and dd_delta <= 2.0 else "→ ✗ keep baseline")
    return 0


if __name__ == "__main__":
    sys.exit(main())
