#!/usr/bin/env python3
"""compare_vbt_vs_mt5_c1 — Parité vbt pro vs MT5 C1 (champion Phase I).

Run le portfolio production vbt avec defaults C1 (vt=0.75, lev=64,
RSI no JPY, TS no CAD) et compare metrics avec le backtest MT5 C1
de référence sur même fenêtre.

Tolérances acceptables :
  - Sharpe : ±0.10
  - CAGR   : ±2pp
  - MaxDD  : ±2pp
  - PF     : ±0.10
  - Trades : ±10%

Si écart > tolérance → flag dans output markdown.

Usage :
    python scripts/compare_vbt_vs_mt5_c1.py
"""
from __future__ import annotations

import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

OUT_MD = ROOT / "reports/investigations/vbt_vs_mt5_c1_parity.md"
MT5_C1_JSON = ROOT / "reports/mt5/run_20260505T172809Z.json"  # smoke C1 propre


def load_mt5_c1_metrics() -> dict:
    """Read MT5 C1 reference backtest metrics from latest C1 smoke."""
    if not MT5_C1_JSON.exists():
        # Fallback : find most recent C1 backtest in reports/mt5/
        candidates = sorted(
            ROOT.glob("reports/mt5/run_*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        for p in candidates:
            try:
                d = json.loads(p.read_text())
                raw = d.get("log_summary", {}).get("raw_tail", "")
                m = re.search(r"\[OPTIM\] vt=0\.7500 maxlev=64\.0000", raw)
                if m:
                    return _parse_mt5_json(d)
            except Exception:
                continue
        raise FileNotFoundError("No MT5 C1 backtest found")
    return _parse_mt5_json(json.loads(MT5_C1_JSON.read_text()))


def _parse_mt5_json(d: dict) -> dict:
    m = d.get("metrics", {})
    raw = d.get("log_summary", {}).get("raw_tail", "")
    optim = re.search(
        r"\[OPTIM\] vt=([\d.]+) maxlev=([\d.]+) volfloor=([\d.]+) "
        r"cagr=([\d.]+) dd=([\d.]+) sharpe=([\d.]+) pf=([\d.]+) "
        r"rf=([\d.]+) trades=(\d+) net=([-\d.]+) years=([\d.]+)",
        raw,
    )
    if not optim:
        raise ValueError(f"No OPTIM line in MT5 JSON {d.get('run_id')}")
    cagr_pct = float(optim.group(4)) * 100
    dd_pct = float(optim.group(5))
    sharpe = float(optim.group(6))
    pf = float(optim.group(7))
    trades = int(optim.group(9))
    net = float(optim.group(10))
    return {
        "vt": float(optim.group(1)),
        "lev": float(optim.group(2)),
        "vfloor": float(optim.group(3)),
        "cagr_pct": cagr_pct,
        "dd_pct": dd_pct,
        "sharpe": sharpe,
        "profit_factor": pf,
        "trades": trades,
        "net_profit": net,
        "years": float(optim.group(11)),
    }


def run_vbt() -> dict:
    """Build vbt production portfolio (defaults C1) and extract metrics."""
    from strategies.combined_portfolio_v2 import (
        PRODUCTION_MAX_LEVERAGE,
        PRODUCTION_TARGET_VOL,
        PRODUCTION_WEIGHTS,
        build_production_portfolio,
    )

    print(f"[vbt] Defaults: vt={PRODUCTION_TARGET_VOL}, lev={PRODUCTION_MAX_LEVERAGE}")
    print(f"[vbt] Weights : {PRODUCTION_WEIGHTS}")
    print("[vbt] Building production portfolio (slow ~60-90s warm cache)...")

    # Phase M.1 : pipelines have built-in leverage=12. Disable Python vol-target
    # to avoid double-leverage stacking. MT5 GlobalLeverage applied uniformly to
    # all sleeves via pipeline leverage param, not via portfolio-level scaling.
    res = build_production_portfolio(target_vol=None)

    pf = res.get("portfolio")
    sharpe = res.get("sharpe", float("nan"))
    annual_ret = res.get("annual_return", float("nan"))
    max_dd = res.get("max_drawdown", float("nan"))
    # PF and trades from pf if available
    pf_val = float("nan")
    trades = 0
    if pf is not None:
        try:
            stats = pf.stats() if hasattr(pf, "stats") else {}
            pf_val = float(stats.get("Profit Factor", float("nan")))
            trades = int(stats.get("Total Trades", 0))
        except Exception as exc:
            print(f"[vbt] Warning extracting pf.stats: {exc}")

    return {
        "vt": PRODUCTION_TARGET_VOL,
        "lev": PRODUCTION_MAX_LEVERAGE,
        "cagr_pct": annual_ret * 100,
        "dd_pct": abs(max_dd) * 100,
        "sharpe": sharpe,
        "profit_factor": pf_val,
        "trades": trades,
        "net_profit": float("nan"),  # vbt does not report $ net profit on synthetic price
    }


def main() -> int:
    print("=" * 70)
    print("  COMPARAISON vbt pro vs MT5 C1 (champion Phase I)")
    print("=" * 70)

    mt5 = load_mt5_c1_metrics()
    print(f"\n[mt5] Reference: vt={mt5['vt']} lev={mt5['lev']} "
          f"({mt5['years']:.2f}y backtest)")
    print(f"[mt5] Sharpe={mt5['sharpe']:.2f} CAGR={mt5['cagr_pct']:.2f}% "
          f"DD={mt5['dd_pct']:.2f}% PF={mt5['profit_factor']:.2f} "
          f"Trades={mt5['trades']} Net=${mt5['net_profit']:.0f}")

    vbt = run_vbt()
    print(f"\n[vbt] Sharpe={vbt['sharpe']:.2f} CAGR={vbt['cagr_pct']:.2f}% "
          f"DD={vbt['dd_pct']:.2f}% PF={vbt['profit_factor']:.2f} "
          f"Trades={vbt['trades']}")

    # Comparaison
    tol = {"sharpe": 0.10, "cagr_pct": 2.0, "dd_pct": 2.0, "profit_factor": 0.10}
    rows = []
    for k in ("vt", "lev"):
        rows.append((k, mt5[k], vbt[k], "config", "—"))
    for k, t in tol.items():
        delta = vbt[k] - mt5[k]
        within = abs(delta) <= t if not (pd.isna(vbt[k]) or pd.isna(mt5[k])) else False
        flag = "✓" if within else "✗"
        rows.append((k, mt5[k], vbt[k], f"±{t}", f"{delta:+.3f} {flag}"))
    delta_tr = (vbt["trades"] - mt5["trades"]) / max(mt5["trades"], 1) * 100
    within_tr = abs(delta_tr) <= 10
    rows.append(("trades", mt5["trades"], vbt["trades"], "±10%",
                 f"{delta_tr:+.1f}% {'✓' if within_tr else '✗'}"))

    print(f"\n{'metric':<15} {'mt5_c1':>12} {'vbt':>12} {'tolerance':>12} {'verdict':>15}")
    print("-" * 72)
    for row in rows:
        m_str = f"{row[1]:>12.4f}" if isinstance(row[1], float) else f"{row[1]:>12}"
        v_str = f"{row[2]:>12.4f}" if isinstance(row[2], float) else f"{row[2]:>12}"
        print(f"{row[0]:<15} {m_str} {v_str} {row[3]:>12} {row[4]:>15}")

    # Verdict global
    failures = [r for r in rows if "✗" in str(r[4])]
    overall = "✅ PASS" if not failures else f"⚠️  {len(failures)} écart(s) hors tolérance"

    # Markdown report
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    md = [
        f"# Parité vbt pro vs MT5 C1 — {ts}",
        "",
        f"**MT5 C1 reference** : `{MT5_C1_JSON.name}` ({mt5['years']:.2f}y backtest, vt={mt5['vt']}, lev={mt5['lev']})",
        "",
        f"**Verdict global** : {overall}",
        "",
        "## Métriques side-by-side",
        "",
        "| Métrique | MT5 C1 | vbt pro | Tolérance | Δ |",
        "|---|---|---|---|---|",
    ]
    for row in rows:
        m_str = f"{row[1]:.4f}" if isinstance(row[1], float) else str(row[1])
        v_str = f"{row[2]:.4f}" if isinstance(row[2], float) else str(row[2])
        md.append(f"| {row[0]} | {m_str} | {v_str} | {row[3]} | {row[4]} |")

    if failures:
        md.append("")
        md.append("## Écarts à investiguer")
        md.append("")
        for r in failures:
            md.append(f"- **{r[0]}** : MT5={r[1]} vs vbt={r[2]}, delta {r[4]}")
        md.append("")
        md.append("Causes probables :")
        md.append("1. Sizing model (MT5 lots discrets vs vbt fraction continue)")
        md.append("2. Slippage application (MT5 ajuste SL distance, vbt uniforme)")
        md.append("3. Vol recompute timing (MT5 21:00 UTC daily, vbt rolling shift(1))")
        md.append("4. Sub-equity calculation (MT5 par sleeve, vbt aggregate)")
        md.append("5. Macro filter execution timing (signal-level vs entry-level)")

    OUT_MD.write_text("\n".join(md) + "\n")
    print(f"\n→ {OUT_MD}")
    print(f"\n{overall}")

    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
