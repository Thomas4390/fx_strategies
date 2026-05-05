"""walkforward_allocations — Phase C (Plan CAGR docs/investigations).

Sweep 6 allocations (MR / TS / RSI) × N=5 walk-forward folds. Étape 1 :
screening rapide sur full 5.4 ans pour identifier top variants. Étape 2 :
walk-forward N=5 sur top 2 + baseline 80/10/10 pour valider la robustesse.

Critères §3.1 plan source pour retenir une variante :
  - CAGR_avg ≥ baseline + 1pp
  - ΔSharpe_med ≥ +0.05 sur N=5 folds
  - MaxDD ≤ baseline + 2pp

Output :
  reports/optimization/allocations/screening.csv
  reports/optimization/allocations/walkforward.csv
  reports/optimization/allocations/findings.md

Usage :
    python scripts/optimization/walkforward_allocations.py
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
OUT_DIR = REPO / "reports/optimization/allocations"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# (label, alloc_mr, alloc_ts, alloc_rsi)
ALLOCATIONS = [
    ("conservative", 0.50, 0.25, 0.25),
    ("balanced", 0.60, 0.20, 0.20),
    ("baseline", 0.80, 0.10, 0.10),       # current
    ("mr_heavy", 0.90, 0.05, 0.05),
    ("equal", 0.34, 0.33, 0.33),
    ("ts_heavy", 0.40, 0.50, 0.10),
]

FULL_FROM = "2020.11.23"
FULL_TO = "2026.04.30"

# Reuse N=5 folds from walkforward_n5.py (sliding 1y OOS, IS expanding)
FOLDS = [
    ("fold1_is",  "2020.11.23", "2021.10.31"),
    ("fold1_oos", "2021.11.01", "2022.10.31"),
    ("fold2_is",  "2020.11.23", "2022.10.31"),
    ("fold2_oos", "2022.11.01", "2023.10.31"),
    ("fold3_is",  "2020.11.23", "2023.10.31"),
    ("fold3_oos", "2023.11.01", "2024.10.31"),
    ("fold4_is",  "2020.11.23", "2024.10.31"),
    ("fold4_oos", "2024.11.01", "2025.10.31"),
    ("fold5_is",  "2020.11.23", "2025.10.31"),
    ("fold5_oos", "2025.11.01", "2026.04.30"),
]


def run_backtest(
    label: str,
    alloc: tuple[float, float, float],
    from_date: str,
    to_date: str,
    report_name: str,
) -> dict:
    """Run un backtest avec une alloc spécifique. Retourne metrics dict."""
    mr, ts, rsi = alloc
    cmd = [
        sys.executable,
        str(RUN_CLI),
        "--from", from_date,
        "--to", to_date,
        "--report-name", report_name,
        "--input", f"Inp_AllocMRMacro={mr:.2f}",
        "--input", f"Inp_AllocTSMomentum={ts:.2f}",
        "--input", f"Inp_AllocRSIDaily={rsi:.2f}",
    ]
    print(f"  [run] alloc={mr}/{ts}/{rsi} {from_date}→{to_date}")
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if proc.returncode != 0:
        print("STDOUT:\n", proc.stdout[-1000:])
        print("STDERR:\n", proc.stderr[-1000:])
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


def cagr_from_net(net: float, deposit: float = 10_000.0, years: float = 5.432) -> float:
    """CAGR % depuis Net Profit USD (simplification, fenêtre fixe)."""
    final = deposit + (net or 0)
    if final <= 0 or deposit <= 0:
        return -100.0
    return 100.0 * ((final / deposit) ** (1.0 / years) - 1.0)


def years_window(from_date: str, to_date: str) -> float:
    from datetime import datetime as dt
    a = dt.strptime(from_date, "%Y.%m.%d")
    b = dt.strptime(to_date, "%Y.%m.%d")
    return (b - a).days / 365.25


def main() -> int:
    print("=" * 70)
    print("Phase C — allocation sweep (Plan CAGR)")
    print("=" * 70)

    # ===== STAGE 1 — screening full 5.4y =====
    print(f"\n--- Stage 1 : screening {FULL_FROM} → {FULL_TO} (6 variants) ---\n")
    rows = []
    for label, mr, ts, rsi in ALLOCATIONS:
        d = run_backtest(label, (mr, ts, rsi), FULL_FROM, FULL_TO, f"alloc_{label}")
        m = parse_metrics(d)
        cagr = cagr_from_net(m["net_profit"], years=years_window(FULL_FROM, FULL_TO))
        rows.append({
            "alloc_label": label,
            "mr": mr, "ts": ts, "rsi": rsi,
            "sharpe": m["sharpe"], "cagr_pct": cagr,
            "net_profit": m["net_profit"],
            "profit_factor": m["profit_factor"],
            "max_dd_pct": m["max_dd_pct"],
            "trades": m["trades"],
        })

    import pandas as pd

    screening_df = pd.DataFrame(rows)
    screening_df = screening_df.sort_values("sharpe", ascending=False).reset_index(drop=True)
    screening_csv = OUT_DIR / "screening.csv"
    screening_df.to_csv(screening_csv, index=False)

    print(f"\n{'label':<14} {'MR/TS/RSI':<14} {'Sharpe':>7} {'CAGR%':>7} "
          f"{'Net':>9} {'PF':>5} {'DD%':>6} {'Trades':>7}")
    print("-" * 75)
    for _, r in screening_df.iterrows():
        print(f"{r['alloc_label']:<14} "
              f"{r['mr']:.2f}/{r['ts']:.2f}/{r['rsi']:.2f}  "
              f"{r['sharpe']:>7.2f} {r['cagr_pct']:>+7.2f} "
              f"{r['net_profit']:>9.0f} {r['profit_factor']:>5.2f} "
              f"{r['max_dd_pct']:>6.2f} {int(r['trades']):>7d}")
    print(f"\n[write] {screening_csv}")

    # Pick top 2 by Sharpe + always include baseline for delta computation
    baseline_sharpe = screening_df.loc[screening_df["alloc_label"] == "baseline", "sharpe"].iloc[0]
    top_labels = screening_df.head(2)["alloc_label"].tolist()
    candidates = list(dict.fromkeys(top_labels + ["baseline"]))
    print(f"\nTop 2 by Sharpe : {top_labels}")
    print(f"Walk-forward N=5 candidates : {candidates}\n")

    # ===== STAGE 2 — walk-forward N=5 sur candidates =====
    print(f"\n--- Stage 2 : walk-forward N=5 (3 candidates × 10 folds = 30 runs) ---\n")
    wf_rows = []
    alloc_map = {a[0]: a[1:] for a in ALLOCATIONS}
    for cand in candidates:
        mr, ts, rsi = alloc_map[cand]
        for fname, fd, td in FOLDS:
            try:
                d = run_backtest(
                    cand,
                    (mr, ts, rsi),
                    fd, td,
                    f"alloc_{cand}_{fname}",
                )
                m = parse_metrics(d)
                yr = years_window(fd, td)
                cagr = cagr_from_net(m["net_profit"], years=yr)
                wf_rows.append({
                    "alloc_label": cand,
                    "fold": fname,
                    "from": fd, "to": td,
                    "years": yr,
                    "sharpe": m["sharpe"],
                    "cagr_pct": cagr,
                    "net_profit": m["net_profit"],
                    "max_dd_pct": m["max_dd_pct"],
                    "trades": m["trades"],
                })
            except Exception as e:
                print(f"  ✗ {cand} {fname}: {e}")

    wf_df = pd.DataFrame(wf_rows)
    wf_csv = OUT_DIR / "walkforward.csv"
    wf_df.to_csv(wf_csv, index=False)

    # Aggregate per candidate (OOS folds only)
    print(f"\n{'='*70}\n  WALK-FORWARD OOS STATS PAR CANDIDATE\n{'='*70}")
    oos_df = wf_df[wf_df["fold"].str.endswith("_oos")].copy()
    print(f"\n{'cand':<14} {'Sharpe_med':>10} {'Sharpe_avg':>10} "
          f"{'CAGR_avg%':>10} {'DD_max%':>8} {'#OOS':>5}")
    print("-" * 60)
    summary = []
    for cand in candidates:
        sub = oos_df[oos_df["alloc_label"] == cand]
        if sub.empty:
            continue
        s_med = sub["sharpe"].median()
        s_avg = sub["sharpe"].mean()
        c_avg = sub["cagr_pct"].mean()
        dd = sub["max_dd_pct"].max()
        summary.append({
            "alloc_label": cand,
            "sharpe_med_oos": s_med,
            "sharpe_avg_oos": s_avg,
            "cagr_avg_oos": c_avg,
            "max_dd_oos": dd,
            "n_oos": len(sub),
        })
        print(f"{cand:<14} {s_med:>10.2f} {s_avg:>10.2f} "
              f"{c_avg:>10.2f} {dd:>8.2f} {len(sub):>5}")

    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(OUT_DIR / "walkforward_summary.csv", index=False)

    # Verdict
    base = next((s for s in summary if s["alloc_label"] == "baseline"), None)
    print(f"\n{'='*70}\n  VERDICT (vs baseline 80/10/10)\n{'='*70}")
    if base:
        for s in summary:
            if s["alloc_label"] == "baseline":
                continue
            ds = s["sharpe_med_oos"] - base["sharpe_med_oos"]
            dc = s["cagr_avg_oos"] - base["cagr_avg_oos"]
            dd_delta = s["max_dd_oos"] - base["max_dd_oos"]
            verdict = "✗"
            reasons = []
            if ds >= 0.05 and dc >= 1.0 and dd_delta <= 2.0:
                verdict = "✓ RETAIN"
            else:
                if ds < 0.05:
                    reasons.append(f"ΔSharpe={ds:+.2f}<0.05")
                if dc < 1.0:
                    reasons.append(f"ΔCAGR={dc:+.2f}pp<1.0")
                if dd_delta > 2.0:
                    reasons.append(f"ΔDD={dd_delta:+.2f}pp>2.0")
            print(f"  {s['alloc_label']:<14} "
                  f"ΔSharpe={ds:+.2f} ΔCAGR={dc:+.2f}pp ΔDD={dd_delta:+.2f}pp "
                  f"→ {verdict} {' / '.join(reasons)}")
    print(f"\n[write] {wf_csv}\n[write] {OUT_DIR / 'walkforward_summary.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
