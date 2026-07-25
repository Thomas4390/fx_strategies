#!/usr/bin/env python3
"""Diff the daily traces of the vbt / QuantConnect / MT5 engines, rung by rung.

The reconciliation method is a ladder, and its whole point is that a divergence
lands on a *named quantity* rather than on an aggregate. Comparing Sharpe ratios
is the weakest available test: two implementations can agree on Sharpe through
offsetting errors, and a Sharpe gap never says where the problem is.

    rung 1  close           bar boundaries, timezone, calendar
    rung 2  score           lookback indexing, warmup
    rung 3  target_weight   sigma window/ddof, floor, causal shift
    rung 4  position_units  fill timing, lot rounding, margin
    rung 5  equity          costs, spread, swap

**A divergence at rung N makes every rung beyond it uninterpretable**, so the
report marks them as such instead of inviting you to read noise.

Two pairs, two different targets — this is the part that is easy to get wrong:

* vbt vs QC share their data (the local parquet was exported from QC), so any
  divergence is engine semantics and is exactly solvable. Tolerance is tight.
* vbt vs MT5 do not: broker feed, server-time bar boundaries, real bid-ask
  spread. The goal there is a divergence that is **bounded and attributed**,
  never nil — a nil divergence would mean the broker backtest was idealised.

Trace format: docs/specs/gold_momentum_spec.md §9.

Usage
-----
    python scripts/reconcile_three_way.py --vbt trace_vbt.csv --qc trace_qc.csv
    python scripts/reconcile_three_way.py --vbt a.csv --qc b.csv --mt5 c.csv
    python scripts/reconcile_three_way.py --vbt a.csv --mt5 c.csv --strict

Exit code is 1 when a pair breaches its tolerance, so the script can gate CI.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

RUNGS: tuple[str, ...] = (
    "close",
    "score",
    "target_weight",
    "position_units",
    "equity",
)

REQUIRED_COLUMNS: tuple[str, ...] = ("date", *RUNGS)


@dataclass(frozen=True)
class Tolerance:
    """Per-pair tolerance. ``atol`` matters: score and weight legitimately hit 0."""

    rtol: float
    atol: float
    label: str


# vbt and QC eat the same bytes, so anything above float noise is a real defect.
TOL_SAME_DATA = Tolerance(rtol=1e-6, atol=1e-6, label="données identiques")

# MT5 runs on the broker feed. These bounds are not a convergence target, they
# are the threshold past which a gap must be attributed to a named cost rather
# than left as a residual.
TOL_BROKER_FEED = Tolerance(rtol=2e-2, atol=1e-4, label="flux broker, écart à attribuer")


@dataclass(frozen=True)
class RungReport:
    rung: str
    n_common: int
    n_breaching: int
    first_date: str | None
    max_abs: float
    max_rel: float
    passed: bool


def load_trace(path: Path, name: str) -> pd.DataFrame:
    """Load one engine trace, failing loudly on a malformed contract."""
    if not path.exists():
        raise SystemExit(f"[{name}] fichier introuvable : {path}")
    frame = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in frame.columns]
    if missing:
        raise SystemExit(
            f"[{name}] colonnes manquantes {missing} dans {path}. "
            f"Contrat attendu : {', '.join(REQUIRED_COLUMNS)} "
            "(docs/specs/gold_momentum_spec.md §9)"
        )
    frame["date"] = pd.to_datetime(frame["date"]).dt.strftime("%Y-%m-%d")
    duplicated = frame["date"].duplicated()
    if duplicated.any():
        raise SystemExit(
            f"[{name}] {int(duplicated.sum())} date(s) en double, "
            f"première : {frame.loc[duplicated, 'date'].iloc[0]}"
        )
    return frame.set_index("date").sort_index()


def compare_rung(
    left: pd.DataFrame,
    right: pd.DataFrame,
    rung: str,
    tol: Tolerance,
) -> RungReport:
    """Compare one column on the common dates."""
    common = left.index.intersection(right.index)
    a = left.loc[common, rung].to_numpy(dtype=float)
    b = right.loc[common, rung].to_numpy(dtype=float)

    delta = np.abs(a - b)
    scale = np.maximum(np.abs(a), np.abs(b))
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.where(scale > 0, delta / scale, 0.0)

    breaching = delta > (tol.atol + tol.rtol * np.abs(b))
    first_idx = int(np.argmax(breaching)) if breaching.any() else None

    return RungReport(
        rung=rung,
        n_common=len(common),
        n_breaching=int(breaching.sum()),
        first_date=str(common[first_idx]) if first_idx is not None else None,
        max_abs=float(delta.max()) if len(delta) else 0.0,
        max_rel=float(np.nanmax(rel)) if len(rel) else 0.0,
        passed=not breaching.any(),
    )


def report_calendar(left: pd.DataFrame, right: pd.DataFrame, a: str, b: str) -> int:
    """Report calendar mismatch — rung 0, and a cause of every rung above it."""
    only_left = left.index.difference(right.index)
    only_right = right.index.difference(left.index)
    common = left.index.intersection(right.index)

    print(f"  séances communes      : {len(common)}")
    print(f"  seulement dans {a:<6} : {len(only_left)}", end="")
    if len(only_left):
        print(f"   (ex. {', '.join(only_left[:3])})", end="")
    print()
    print(f"  seulement dans {b:<6} : {len(only_right)}", end="")
    if len(only_right):
        print(f"   (ex. {', '.join(only_right[:3])})", end="")
    print()

    if not len(common):
        print("\n  ⛔ aucune séance commune — calendriers ou fuseaux incompatibles.")
    return len(common)


def compare_pair(left: pd.DataFrame, right: pd.DataFrame, a: str, b: str, tol: Tolerance) -> bool:
    """Run the full ladder on a pair. Returns True when every rung holds."""
    print(f"\n{'═' * 78}")
    print(f"  {a.upper()} ↔ {b.upper()}   —   tolérance : {tol.label} "
          f"(rtol={tol.rtol:g}, atol={tol.atol:g})")
    print("═" * 78)

    if not report_calendar(left, right, a, b):
        return False

    print(f"\n  {'barreau':<16} {'écart max':>13} {'rel. max':>11} "
          f"{'jours hors':>11} {'1er écart':>12}   verdict")
    print("  " + "-" * 74)

    all_passed = True
    broken_at: str | None = None
    for rung in RUNGS:
        rep = compare_rung(left, right, rung, tol)
        if broken_at is not None:
            print(f"  {rung:<16} {'—':>13} {'—':>11} {'—':>11} {'—':>12}   "
                  f"ininterprétable (cassé au barreau « {broken_at} »)")
            continue
        verdict = "✓" if rep.passed else "✗"
        print(f"  {rung:<16} {rep.max_abs:>13.6g} {rep.max_rel:>11.3e} "
              f"{rep.n_breaching:>11} {str(rep.first_date or '—'):>12}   {verdict}")
        if not rep.passed:
            all_passed = False
            broken_at = rung

    if broken_at is not None:
        print(f"\n  ⛔ premier barreau cassé : « {broken_at} » → {_hint(broken_at)}")
        print("     Réparer celui-ci avant de lire quoi que ce soit au-dessus.")
    else:
        print("\n  ✅ les cinq barreaux tiennent dans la tolérance.")
    return all_passed


def _hint(rung: str) -> str:
    return {
        "close": "bornes de barre, fuseau, calendrier (spec §2)",
        "score": "indexation du lookback ou warmup (spec §3)",
        "target_weight": "fenêtre/ddof de sigma, plancher, décalage causal (spec §4)",
        "position_units": "timing de fill, arrondi de lots, marge (spec §6)",
        "equity": "coûts, spread, swap (spec §8)",
    }[rung]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vbt", type=Path, help="trace vbt")
    ap.add_argument("--qc", type=Path, help="trace QuantConnect")
    ap.add_argument("--mt5", type=Path, help="trace MT5")
    ap.add_argument(
        "--strict",
        action="store_true",
        help="appliquer aussi la tolérance stricte à la paire MT5 (diagnostic seulement : "
             "l'égalité avec MT5 n'est PAS l'objectif, voir docstring)",
    )
    args = ap.parse_args()

    traces = {
        name: load_trace(path, name)
        for name, path in (("vbt", args.vbt), ("qc", args.qc), ("mt5", args.mt5))
        if path is not None
    }
    if len(traces) < 2:
        raise SystemExit("Au moins deux traces sont nécessaires (--vbt/--qc/--mt5).")

    for name, frame in traces.items():
        span = f"{frame.index[0]} → {frame.index[-1]}" if len(frame) else "vide"
        print(f"[{name:<3}] {len(frame):>5} séances   {span}")

    mt5_tol = TOL_SAME_DATA if args.strict else TOL_BROKER_FEED
    pairs = [
        ("vbt", "qc", TOL_SAME_DATA),
        ("vbt", "mt5", mt5_tol),
        ("qc", "mt5", mt5_tol),
    ]

    results: list[tuple[str, bool]] = []
    for a, b, tol in pairs:
        if a in traces and b in traces:
            results.append((f"{a}↔{b}", compare_pair(traces[a], traces[b], a, b, tol)))

    print(f"\n{'═' * 78}")
    failures = [name for name, ok in results if not ok]
    if failures:
        print(f"  VERDICT : {len(failures)} paire(s) hors tolérance — {', '.join(failures)}")
        print("  Un écart hors tolérance doit être attribué nommément, jamais laissé « résiduel ».")
        return 1
    print(f"  VERDICT : {len(results)} paire(s) réconciliée(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
