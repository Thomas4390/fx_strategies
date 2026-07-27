"""Compare MT5 C1 reference backtest vs QuantConnect Phase I refresh.

Loads:
  - MT5 reference: reports/qc_phase_i/mt5_c1_reference.json (run_20260505T171514Z)
  - MT5 deals:     reports/mt5/deals_c1_full.csv (UTF-16LE)
  - QC sleeves:    mr_macro_phase_i_v1.json, ts_momentum_phase_i_v1.json,
                   rsi_daily_phase_i_v1.json, tri_signaux_phase_i_v1.json

Emits:
  - Per-metric PASS/FAIL with strict tolerance gates:
    Sharpe rf=0 ±10% relative, Vol ±2% absolute, deals ±10% relative,
    CAGR ±10% relative, MaxDD ±10% relative.
  - Sleeve breakdown vs MT5 magic 831 / 832 / 833.
  - Verdict global ≥ 4/5 gates AND MR Macro standalone Sharpe rf=0 > 0.40.

Run: python reports/qc_phase_i/compare_mt5_qc_phase_i.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
QC_DIR = REPO_ROOT / "reports" / "qc_phase_i"
MT5_DEALS_CSV = REPO_ROOT / "reports" / "mt5" / "deals_c1_full.csv"

MT5_MAGIC_TO_SLEEVE = {"831": "MR_MACRO", "832": "TS_MOMENTUM", "833": "RSI_DAILY"}


def parse_pct(s: str) -> float:
    return float(s.strip().rstrip("%"))


def parse_money(s: str) -> float:
    return float(s.replace("$", "").replace(",", "").replace(" ", "").strip())


def load_qc(path: Path) -> dict:
    with path.open() as f:
        d = json.load(f)
    if "statistics" in d:
        s = d["statistics"]
        return {
            "name": d["name"],
            "total_orders": int(s["Total Orders"]),
            "cagr_pct": parse_pct(s["Compounding Annual Return"]),
            "drawdown_pct": parse_pct(s["Drawdown"]),
            "vol_pct": float(s["Annual Standard Deviation"]) * 100,
            "sharpe_qc": float(s["Sharpe Ratio"]),
            "win_rate_pct": parse_pct(s["Win Rate"]),
            "avg_win_pct": parse_pct(s["Average Win"]),
            "avg_loss_pct": parse_pct(s["Average Loss"]),
            "net_profit_pct": parse_pct(s["Net Profit"]),
            "end_equity": float(s["End Equity"]),
        }
    # Combined synthetic: stats computed from chart equity in upstream script
    lev = d["leveraged"]
    return {
        "name": d["name"],
        "total_orders": 0,  # synthetic, no real positions
        "cagr_pct": lev["cagr_pct"],
        "drawdown_pct": abs(lev["drawdown_pct"]),
        "vol_pct": lev["vol_pct"],
        "sharpe_qc": float("nan"),
        "win_rate_pct": float("nan"),
        "avg_win_pct": float("nan"),
        "avg_loss_pct": float("nan"),
        "net_profit_pct": lev["total_return_pct"],
        "end_equity": d["end_equity_lev"],
    }


def sharpe_rf0(cagr_pct: float, vol_pct: float) -> float:
    """Sharpe assuming rf=0, matching MT5's OnTester convention."""
    return cagr_pct / vol_pct if vol_pct > 0 else 0.0


def load_mt5_reference() -> dict:
    with (QC_DIR / "mt5_c1_reference.json").open() as f:
        d = json.load(f)
    m = d["metrics"]
    init_dep = parse_money(m["initial_deposit"])
    net = parse_money(m["total_net_profit"])
    sharpe = float(m["sharpe_ratio"])
    trades = int(m["total_trades"])
    raw_tail = d["log_summary"]["raw_tail"]
    cagr = None
    dd_eq_pct = None
    for line in raw_tail.splitlines():
        if "[OPTIM]" in line and "cagr=" in line:
            for tok in line.split():
                if tok.startswith("cagr="):
                    cagr = float(tok.split("=")[1]) * 100
                if tok.startswith("dd="):
                    dd_eq_pct = float(tok.split("=")[1])
    if dd_eq_pct is None:
        # Fallback parse "13.00%" out of "(13.00%)"
        dd_str = m["equity_dd_max"].split("(")[1].rstrip(")")
        dd_eq_pct = parse_pct(dd_str)
    if cagr is None:
        cagr = ((init_dep + net) / init_dep) ** (1 / 5.432) - 1
        cagr *= 100
    vol_implied = cagr / sharpe if sharpe > 0 else float("nan")
    return {
        "sharpe_rf0": sharpe,
        "cagr_pct": cagr,
        "drawdown_pct": dd_eq_pct,
        "vol_pct_implied": vol_implied,
        "deals_total": trades,
        "init_deposit": init_dep,
        "net_profit": net,
        "profit_factor": float(m["profit_factor"]),
        "recovery_factor": float(m["recovery_factor"]),
        "years": 5.432,
    }


def count_mt5_deals_per_sleeve(csv_path: Path) -> dict:
    """Read UTF-16LE CSV, count entry deals per sleeve (round trips)."""
    text = csv_path.read_bytes().decode("utf-16-le").lstrip("﻿")
    reader = csv.DictReader(text.splitlines())
    sleeve_count = {"MR_MACRO": 0, "TS_MOMENTUM": 0, "RSI_DAILY": 0, "OTHER": 0}
    entry_count = {"MR_MACRO": 0, "TS_MOMENTUM": 0, "RSI_DAILY": 0}
    for row in reader:
        sleeve = row.get("sleeve", "OTHER")
        sleeve_count[sleeve] = sleeve_count.get(sleeve, 0) + 1
        if row.get("entry") == "0" and sleeve in entry_count:
            entry_count[sleeve] += 1
    return {
        "deal_events_per_sleeve": sleeve_count,
        "entry_deals_per_sleeve": entry_count,
        "total_deal_events": sum(sleeve_count.values()),
        "total_entries": sum(entry_count.values()),
    }


def gate(name: str, qc_val: float, mt5_val: float, kind: str, tol: float) -> tuple[str, str]:
    """Return (status, reason)."""
    if kind == "rel":
        if mt5_val == 0:
            return ("SKIP", "mt5 reference zero")
        diff = abs(qc_val - mt5_val) / abs(mt5_val)
        ok = diff <= tol
        return ("PASS" if ok else "FAIL",
                f"{name}: MT5={mt5_val:.4f} QC={qc_val:.4f} diff={diff*100:.1f}% (tol {tol*100:.0f}%)")
    if kind == "abs":
        diff = abs(qc_val - mt5_val)
        ok = diff <= tol
        return ("PASS" if ok else "FAIL",
                f"{name}: MT5={mt5_val:.4f} QC={qc_val:.4f} abs_diff={diff:.4f} (tol {tol:.4f})")
    return ("SKIP", "unknown gate kind")


def main() -> int:
    print("=" * 72)
    print(" Validation reproductibilite MT5 C1 <-> QuantConnect Phase I")
    print(" Periode 2020-11-23 -> 2026-04-30 (5.432y), tolerance stricte")
    print("=" * 72)

    mt5 = load_mt5_reference()
    deals = count_mt5_deals_per_sleeve(MT5_DEALS_CSV)

    print("\n--- MT5 C1 reference (deals_c1_full.csv + run JSON) ---")
    print(f"  Sharpe rf=0     : {mt5['sharpe_rf0']:.4f}")
    print(f"  CAGR            : {mt5['cagr_pct']:.2f} %")
    print(f"  MaxDD equity    : {mt5['drawdown_pct']:.2f} %")
    print(f"  Vol implied     : {mt5['vol_pct_implied']:.2f} % (cagr / sharpe)")
    print(f"  Total trades    : {mt5['deals_total']} (round trips per OnTester)")
    print(f"  Profit factor   : {mt5['profit_factor']:.2f}")
    print(f"  Recovery factor : {mt5['recovery_factor']:.2f}")
    print(f"  Years           : {mt5['years']:.3f}")
    print(f"  Deal events/sleeve : {deals['deal_events_per_sleeve']}")
    print(f"  Entry deals/sleeve : {deals['entry_deals_per_sleeve']}")

    qc_files = {
        "mr_macro": QC_DIR / "mr_macro_phase_i_v1.json",
        "ts_momentum": QC_DIR / "ts_momentum_phase_i_v1.json",
        "rsi_daily": QC_DIR / "rsi_daily_phase_i_v1.json",
        "combined": QC_DIR / "tri_signaux_phase_i_v1.json",
    }
    qc = {}
    for k, p in qc_files.items():
        if p.exists():
            qc[k] = load_qc(p)
            qc[k]["sharpe_rf0"] = sharpe_rf0(qc[k]["cagr_pct"], qc[k]["vol_pct"])
        else:
            qc[k] = None

    print("\n--- QC Phase I sleeves standalone ---")
    for k in ("mr_macro", "ts_momentum", "rsi_daily"):
        if qc[k] is None:
            print(f"  {k:<12} : MISSING ({qc_files[k].name})")
            continue
        v = qc[k]
        print(f"  {k:<12}: orders={v['total_orders']:<5} CAGR={v['cagr_pct']:>5.2f}%  "
              f"Vol={v['vol_pct']:>4.2f}%  DD={v['drawdown_pct']:>5.2f}%  "
              f"Sharpe(rf=0)={v['sharpe_rf0']:>5.2f}  Win={v['win_rate_pct']:.0f}%")

    # Combined comparison vs MT5
    print("\n--- Combined Phase I vs MT5 C1 ---")
    if qc["combined"] is None:
        print("  Combined backtest result NOT FOUND -- run the QC backtest first.")
        return 2

    c = qc["combined"]
    # Combined synthetic in QC: orders==0 (no real positions). Sleeve counts
    # are summed from standalone backtests below.
    qc_total_orders_sum = sum(qc[k]["total_orders"] for k in ("mr_macro", "ts_momentum", "rsi_daily")
                               if qc[k] is not None)
    print(f"  QC combined synth: CAGR={c['cagr_pct']:.2f}% Vol={c['vol_pct']:.2f}% "
          f"DD={c['drawdown_pct']:.2f}% Sharpe(rf=0)={c['sharpe_rf0']:.2f}")
    print(f"  Sum standalone QC orders (MR+TS+RSI) = {qc_total_orders_sum}")

    print("\n--- Tolerance gates (combined leveraged) ---")
    gates = [
        gate("Sharpe rf=0",     c["sharpe_rf0"],     mt5["sharpe_rf0"],          "rel", 0.10),
        gate("CAGR (%)",        c["cagr_pct"],       mt5["cagr_pct"],            "rel", 0.10),
        gate("MaxDD (%)",       c["drawdown_pct"],   mt5["drawdown_pct"],        "rel", 0.10),
        gate("Vol annual (%)",  c["vol_pct"],        mt5["vol_pct_implied"],     "abs", 2.0),
        gate("Deal count",      qc_total_orders_sum, mt5["deals_total"],         "rel", 0.10),
    ]
    for status, reason in gates:
        marker = "[OK]  " if status == "PASS" else "[FAIL]" if status == "FAIL" else "[SKIP]"
        print(f"  {marker} {reason}")

    pass_count = sum(1 for s, _ in gates if s == "PASS")
    fail_count = sum(1 for s, _ in gates if s == "FAIL")
    mr_sharpe = qc["mr_macro"]["sharpe_rf0"] if qc["mr_macro"] else 0
    mr_ok = mr_sharpe > 0.40

    print("\n--- Verdict ---")
    print(f"  Gates passes : {pass_count}/5  (failed {fail_count})")
    print(f"  MR Macro standalone Sharpe rf=0 = {mr_sharpe:.2f}  (>0.40 ? {'YES' if mr_ok else 'NO'})")
    overall = "PASS" if pass_count >= 4 and mr_ok else "FAIL"
    print(f"  VERDICT GLOBAL : {overall}")
    return 0 if overall == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
