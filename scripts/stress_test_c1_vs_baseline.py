#!/usr/bin/env python3
"""stress_test_c1_vs_baseline — backtest 3 fenêtres adversariales pour C1 vs prod.

Fenêtres (data EURUSD.c M1 commence Nov 2020, donc COVID 2020-Q1 exclus):
  W1 yen/BoJ          : 2022-08-01 → 2022-11-30
  W2 banking SVB/CS   : 2023-03-01 → 2023-04-15
  W3 yen carry unwind : 2024-08-01 → 2024-09-30

Configs:
  C1   : vt=0.75 lev=64 vfl=0.02 EnableDDCap=false (champion walkforward N=5)
  prod : vt=0.28 lev=12 vfl=0.02 EnableDDCap=false (current default)

Critères pass:
  - DD(C1) ≤ 25% (hard cap)
  - DD(C1) ≤ 1.5 * DD(prod) sur même fenêtre
  - Sharpe(C1) ≥ 0
"""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RUNNER = ROOT / "src/mt5/bridge/run_backtest_cli.py"
OUT_DIR = ROOT / "reports/stress"
OUT_DIR.mkdir(parents=True, exist_ok=True)
MT5_CACHE_DIR = Path("/home/thomas/.mt5/drive_c/Program Files/MetaTrader 5/Tester/cache")

WINDOWS = [
    ("W1_yen_2022",       "2022.08.01", "2022.11.30"),
    ("W2_banking_2023",   "2023.03.01", "2023.04.15"),
    ("W3_yen_2024",       "2024.08.01", "2024.09.30"),
]

CONFIGS = {
    "C1":   {"Inp_GlobalTargetVol": "0.75", "Inp_GlobalMaxLeverage": "64",
             "Inp_GlobalVolFloor": "0.02", "Inp_EnableDDCap": "false"},
    "prod": {"Inp_GlobalTargetVol": "0.28", "Inp_GlobalMaxLeverage": "12",
             "Inp_GlobalVolFloor": "0.02", "Inp_EnableDDCap": "false"},
}


def clear_cache() -> int:
    n = 0
    if MT5_CACHE_DIR.exists():
        for f in MT5_CACHE_DIR.glob("*.opt"):
            try:
                f.unlink(); n += 1
            except OSError:
                pass
        for f in MT5_CACHE_DIR.glob("*.tst"):
            try:
                f.unlink(); n += 1
            except OSError:
                pass
    return n


def run_one(label: str, win_name: str, frm: str, to: str,
            params: dict, timeout: int = 600) -> dict | None:
    n_cleared = clear_cache()
    report_name = f"fx_stress_{label}_{win_name}"
    cmd = ["python3", str(RUNNER),
           "--from", frm, "--to", to,
           "--report-name", report_name,
           "--timeout", str(timeout)]
    for k, v in params.items():
        cmd.extend(["--input", f"{k}={v}"])
    print(f"\n=== {label} × {win_name} ({frm}→{to}) cleared {n_cleared} cache ===",
          flush=True)
    result = subprocess.run(cmd, cwd=str(ROOT), check=False,
                            capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ✗ failed exit={result.returncode}")
        print(result.stderr[-500:] if result.stderr else "")
        return None
    # Find latest reports/mt5/run_*.json
    runs = sorted((ROOT / "reports/mt5").glob("run_*.json"),
                  key=lambda p: p.stat().st_mtime, reverse=True)
    if not runs:
        return None
    with open(runs[0]) as f:
        data = json.load(f)
    return data


def main() -> int:
    rows = []
    for cfg_name, params in CONFIGS.items():
        for win_name, frm, to in WINDOWS:
            data = run_one(cfg_name, win_name, frm, to, params)
            if data is None:
                rows.append({"config": cfg_name, "window": win_name,
                             "status": "FAIL"})
                continue
            metrics = data.get("metrics", {}) or {}
            rows.append({
                "config": cfg_name,
                "window": win_name,
                "from": frm,
                "to": to,
                "sharpe": metrics.get("sharpe"),
                "net_profit": metrics.get("net_profit"),
                "profit_factor": metrics.get("profit_factor"),
                "equity_dd_money": metrics.get("equity_dd_money"),
                "equity_dd_pct": metrics.get("equity_dd_pct"),
                "trades": metrics.get("trades"),
                "recovery_factor": metrics.get("recovery_factor"),
                "status": "OK",
            })

    df = pd.DataFrame(rows)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_csv = OUT_DIR / f"stress_summary_{ts}.csv"
    df.to_csv(out_csv, index=False)

    print(f"\n{'='*78}\n  STRESS TESTS RESULTATS\n{'='*78}")
    print(f"  {'Config':<6} {'Window':<20} {'Sharpe':>8} {'DD%':>7} "
          f"{'PF':>6} {'Net':>10} {'Trades':>7}")
    print("  " + "-" * 70)
    for _, r in df.iterrows():
        if r["status"] != "OK":
            print(f"  {r['config']:<6} {r['window']:<20} FAIL")
            continue
        print(f"  {r['config']:<6} {r['window']:<20} "
              f"{r['sharpe']:>+8.2f} {r['equity_dd_pct']:>+7.2f} "
              f"{r['profit_factor']:>+6.2f} {r['net_profit']:>+10.2f} "
              f"{int(r['trades']):>7}")

    # Comparaison C1 vs prod par fenêtre
    print(f"\n{'='*78}\n  COMPARAISON C1 / PROD (par fenêtre)\n{'='*78}")
    print(f"  {'Window':<20} {'DD_C1':>7} {'DD_prod':>8} "
          f"{'Ratio':>6} {'C1_PASS?':>10}")
    print("  " + "-" * 60)
    for win_name, _, _ in WINDOWS:
        c1 = df[(df["config"] == "C1") & (df["window"] == win_name)]
        pd_row = df[(df["config"] == "prod") & (df["window"] == win_name)]
        if c1.empty or pd_row.empty:
            continue
        dd_c1 = c1.iloc[0]["equity_dd_pct"]
        dd_pr = pd_row.iloc[0]["equity_dd_pct"]
        sh_c1 = c1.iloc[0]["sharpe"]
        ratio = dd_c1 / dd_pr if dd_pr > 0 else float('inf')
        # Critères: DD<25% hard, ratio<=1.5, Sharpe>=0
        pass_dd = dd_c1 <= 25.0
        pass_ratio = ratio <= 1.5
        pass_sharpe = sh_c1 >= 0.0
        verdict = "✓ ALL" if (pass_dd and pass_ratio and pass_sharpe) else \
                  f"FAIL[{'DD' if not pass_dd else ''}{'R' if not pass_ratio else ''}{'S' if not pass_sharpe else ''}]"
        print(f"  {win_name:<20} {dd_c1:>+7.2f} {dd_pr:>+8.2f} "
              f"{ratio:>6.2f} {verdict:>10}")

    print(f"\n  → {out_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
