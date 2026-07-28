#!/usr/bin/env python3
"""Parity gate for the gold sleeve: vbt against an MT5 tester run.

Companion to ``compare_vbt_vs_mt5_c1.py``, same shape and same tolerances, but
with a different **goal**, and the difference matters. MT5 runs on the broker
feed with real spread, server-time bar boundaries and lot rounding, so the
target is not equality — it is a divergence that stays bounded and gets
attributed. A nil divergence would mean the broker backtest had been idealised.

The MT5 side is read from a ``reports/mt5/run_*.json`` produced by
``run_backtest_cli.py``; the vbt side is recomputed on the same window.

Two things to know before reading the output:

* The MT5 run must isolate the sleeve — ``Inp_AllocGoldMomentum=1.0`` and the
  other allocations at 0. In production the gold allocation is **0.0**, so a
  default run trades no gold at all and this script would compare vbt against
  the three FX sleeves.
* Runs made with ``--model 1`` (OHLC) interpolate fills and flatter the Sharpe.
  Those figures are an upper bound, not a measurement.

Usage:
    python scripts/compare_vbt_vs_mt5_gold.py
    python scripts/compare_vbt_vs_mt5_gold.py --run reports/mt5/run_2026....json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SRC = _PROJECT_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

MT5_REPORTS = _PROJECT_ROOT / "reports/mt5"

# Repo-wide parity tolerances (compare_vbt_vs_mt5_c1.py).
TOLERANCES = {"sharpe": 0.10, "maxdd_pp": 2.0, "trades_pct": 10.0}


def _to_float(text: str | None) -> float | None:
    """MT5 writes '17 384.72' and '18 404.66 (42.28%)'."""
    if not text:
        return None
    cleaned = str(text).replace(" ", " ").replace(" ", "")
    match = re.search(r"-?\d+\.?\d*", cleaned)
    return float(match.group()) if match else None


def _dd_pct(text: str | None) -> float | None:
    """Extract the percentage from 'amount (pp%)'."""
    if not text:
        return None
    match = re.search(r"\(([\d.]+)%\)", str(text))
    return float(match.group(1)) if match else None


def load_mt5_run(path: Path) -> dict:
    with path.open() as fh:
        payload = json.load(fh)
    metrics = payload.get("metrics") or {}
    period = metrics.get("period", "")
    dates = re.findall(r"(\d{4})\.(\d{2})\.(\d{2})", period)
    window = None
    if len(dates) == 2:
        window = (
            pd.Timestamp("-".join(dates[0])),
            pd.Timestamp("-".join(dates[1])),
        )
    return {
        "path": path,
        "symbol": metrics.get("symbol"),
        "window": window,
        "sharpe": _to_float(metrics.get("sharpe_ratio")),
        "trades": _to_float(metrics.get("total_trades")),
        "maxdd": _dd_pct(metrics.get("equity_dd_max")),
        "net_profit": _to_float(metrics.get("total_net_profit")),
        "exit_code": payload.get("exit_code"),
        "init_ok": (payload.get("log_summary") or {}).get("init_ok"),
    }


def run_vbt(
    window: tuple[pd.Timestamp, pd.Timestamp] | None,
    symbol: str = "XAU-USD",
    loader: str | None = None,
) -> dict:
    """Recompute the sleeve on ``symbol``, over the MT5 run's window.

    ``loader`` matters more than it looks. The registry default for XAG-USD is
    the long Yahoo series, which is a **rolled futures** history: comparing it
    to the broker's spot CFD would fail the parity gate for a reason that has
    nothing to do with the engines. Pass ``mt5`` to compare like for like.
    """
    from strategies import tsmom
    from utils import apply_vbt_settings

    apply_vbt_settings()
    pf, _ = tsmom.pipeline(symbol, loader_override=loader)

    rets = pf.returns
    trades = pf.trades.records_readable
    entries = pd.to_datetime(trades["Entry Index"])
    if window is not None:
        lo, hi = window
        rets = rets[(rets.index >= lo) & (rets.index <= hi)]
        trades = trades[(entries >= lo) & (entries <= hi)]

    equity = (1 + rets).cumprod()
    years = (rets.index[-1] - rets.index[0]).days / 365.25
    return {
        "sharpe": float((rets.mean() * 252) / (rets.std() * np.sqrt(252))),
        "cagr": float((equity.iloc[-1] ** (1 / years) - 1) * 100),
        "vol": float(rets.std() * np.sqrt(252) * 100),
        "maxdd": float(-((equity / equity.cummax()) - 1).min() * 100),
        "trades": len(trades),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, help="reports/mt5/run_*.json (default: most recent)")
    ap.add_argument("--symbol", default="XAU-USD",
                    help="instrument du registre tsmom (défaut: XAU-USD)")
    ap.add_argument("--loader", default=None,
                    help="loader_override — 'mt5' pour comparer au CFD du courtier "
                         "plutôt qu'à la série longue (obligatoire pour XAG-USD)")
    args = ap.parse_args()

    path = args.run
    if path is None:
        runs = sorted(MT5_REPORTS.glob("run_*.json"))
        if not runs:
            raise SystemExit(f"Aucun run MT5 dans {MT5_REPORTS}")
        path = runs[-1]

    mt5 = load_mt5_run(path)
    print(f"[mt5] {path.name}  symbole={mt5['symbol']}  exit={mt5['exit_code']}  init_ok={mt5['init_ok']}")
    if mt5["window"]:
        print(f"[mt5] fenêtre {mt5['window'][0].date()} → {mt5['window'][1].date()}")
    if args.symbol == "XAG-USD" and args.loader != "mt5":
        print("[!] XAG-USD sans --loader mt5 : le registre servirait la série Yahoo, "
              "qui est un continu de futures à rolls. La comparaison échouerait pour "
              "une raison de source, pas de moteur.")

    vbt = run_vbt(mt5["window"], symbol=args.symbol, loader=args.loader)
    print(f"[vbt] instrument={args.symbol} loader={args.loader or 'défaut registre'}")
    print(f"[vbt] Sharpe={vbt['sharpe']:.2f}  CAGR={vbt['cagr']:.2f}%  "
          f"vol={vbt['vol']:.2f}%  maxDD={vbt['maxdd']:.2f}%  trades={vbt['trades']}")

    rows = []
    d_sharpe = vbt["sharpe"] - (mt5["sharpe"] or 0.0)
    rows.append(("sharpe", mt5["sharpe"], vbt["sharpe"], f"±{TOLERANCES['sharpe']}",
                 d_sharpe, abs(d_sharpe) <= TOLERANCES["sharpe"]))
    d_dd = vbt["maxdd"] - (mt5["maxdd"] or 0.0)
    rows.append(("maxdd %", mt5["maxdd"], vbt["maxdd"], f"±{TOLERANCES['maxdd_pp']}pp",
                 d_dd, abs(d_dd) <= TOLERANCES["maxdd_pp"]))
    mt5_tr = mt5["trades"] or 0
    d_tr = (vbt["trades"] - mt5_tr) / max(mt5_tr, 1) * 100
    rows.append(("trades", mt5_tr, vbt["trades"], f"±{TOLERANCES['trades_pct']}%",
                 d_tr, abs(d_tr) <= TOLERANCES["trades_pct"]))

    print(f"\n{'métrique':<12}{'MT5':>12}{'vbt':>12}{'tolérance':>14}{'Δ':>12}   verdict")
    print("-" * 74)
    for name, ref, cand, tol, delta, ok in rows:
        ref_s = f"{ref:.2f}" if isinstance(ref, float) else str(ref)
        cand_s = f"{cand:.2f}" if isinstance(cand, float) else str(cand)
        print(f"{name:<12}{ref_s:>12}{cand_s:>12}{tol:>14}{delta:>+12.2f}   {'✓' if ok else '✗'}")

    failures = [r[0] for r in rows if not r[5]]
    print()
    if failures:
        print(f"⚠️  hors tolérance : {', '.join(failures)}")
        print("   L'égalité n'est PAS l'objectif — mais chaque écart doit être attribué")
        print("   nommément (spread, stop de sécurité 4%, arrondi de lots, swap),")
        print("   jamais laissé « résiduel ».")
        print("   Voir reports/investigations/vbt_vs_mt5_gold_parity.md")
        return 1
    print("✅ les trois métriques tiennent dans les tolérances du repo.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
