"""Generate frozen stats snapshots from the legacy ``backtest_*`` API.

This script is run ONCE per strategy, BEFORE the legacy function is deleted.
Each phase of the refactor produces a handful of ``tests/snapshots/<strat>_<label>.json``
files that become the immutable baseline for ``test_pipeline_equivalence.py``.

Usage:
    python tests/_generate_snapshots.py --strat mr_turbo
    python tests/_generate_snapshots.py --strat mr_macro
    python tests/_generate_snapshots.py --strat all

Each entry in ``SNAPSHOT_CASES`` is:
    (strat, label, backtest_callable_name, params_dict)

After running, inspect the generated JSON files, commit them, then proceed
with the strategy rewrite.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_TESTS_DIR = Path(__file__).resolve().parent
_SRC_DIR = _TESTS_DIR.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

_SNAPSHOT_DIR = _TESTS_DIR / "snapshots"
_SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)

# Each case: (strategy_key, label_slug, module_import_path, callable_name, params)
SNAPSHOT_CASES: list[tuple[str, str, str, str, dict[str, Any]]] = [
    # ── Phase 1: mr_turbo ────────────────────────────────────────────
    (
        "mr_turbo",
        "default",
        "strategies.mr_turbo",
        "backtest_mr_turbo",
        dict(bb_window=80, bb_alpha=5.0, sl_stop=0.005, tp_stop=0.006),
    ),
    (
        "mr_turbo",
        "tight_bands",
        "strategies.mr_turbo",
        "backtest_mr_turbo",
        dict(bb_window=60, bb_alpha=4.0, sl_stop=0.004, tp_stop=0.006),
    ),
    (
        "mr_turbo",
        "loose_bands",
        "strategies.mr_turbo",
        "backtest_mr_turbo",
        dict(bb_window=120, bb_alpha=6.0, sl_stop=0.005, tp_stop=0.008),
    ),
    # ── Phase 2: mr_macro ───────────────────────────────────────────
    (
        "mr_macro",
        "default",
        "strategies.mr_macro",
        "backtest_mr_macro",
        dict(bb_window=80, bb_alpha=5.0, sl_stop=0.005, tp_stop=0.006, spread_threshold=0.5),
    ),
    (
        "mr_macro",
        "strict_spread",
        "strategies.mr_macro",
        "backtest_mr_macro",
        dict(bb_window=60, bb_alpha=5.0, sl_stop=0.005, tp_stop=0.006, spread_threshold=0.3),
    ),
    (
        "mr_macro",
        "loose_bands",
        "strategies.mr_macro",
        "backtest_mr_macro",
        dict(bb_window=120, bb_alpha=6.0, sl_stop=0.005, tp_stop=0.008, spread_threshold=0.5),
    ),
    # ── Phase 3: rsi_daily ───────────────────────────────────────────
    # ── Phase 4: daily_momentum ──────────────────────────────────────
    # ── Phase 5: ou_mean_reversion ───────────────────────────────────
    # ── Phase 6: composite_fx_alpha ──────────────────────────────────
]


def _stats_to_dict(stats: Any) -> dict[str, Any]:
    import numpy as np
    import pandas as pd

    out: dict[str, Any] = {}
    for key, val in stats.items():
        if isinstance(val, (int, float, np.integer, np.floating)):
            out[str(key)] = None if np.isnan(val) else float(val)
        else:
            out[str(key)] = str(val)
    return out


def generate_case(case: tuple[str, str, str, str, dict[str, Any]]) -> Path:
    import importlib

    from utils import apply_vbt_settings, load_fx_data

    strat, label, module, fn_name, params = case
    snapshot_path = _SNAPSHOT_DIR / f"{strat}_{label}.json"

    print(f"▶ {strat}/{label} via {module}.{fn_name}({params})")

    apply_vbt_settings()
    _, data = load_fx_data("data/EUR-USD_minute.parquet")

    mod = importlib.import_module(module)
    fn = getattr(mod, fn_name)
    pf = fn(data, **params)
    stats = pf.stats()

    payload = {
        "strat": strat,
        "label": label,
        "module": module,
        "callable": fn_name,
        "params": params,
        "stats": _stats_to_dict(stats),
    }
    with snapshot_path.open("w") as fh:
        json.dump(payload, fh, indent=2, default=str)
    print(f"  ✔ wrote {snapshot_path.name}")
    return snapshot_path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--strat", default="all", help="Strategy key, or 'all'")
    args = ap.parse_args()

    cases = (
        SNAPSHOT_CASES
        if args.strat == "all"
        else [c for c in SNAPSHOT_CASES if c[0] == args.strat]
    )

    if not cases:
        print(f"No snapshot cases for strat={args.strat!r}")
        return 1

    for case in cases:
        try:
            generate_case(case)
        except Exception as e:
            print(f"  ✗ FAILED: {type(e).__name__}: {e}")
            return 2

    print(f"\n✔ Generated {len(cases)} snapshot(s) in {_SNAPSHOT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
