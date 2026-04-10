"""Equivalence tests between legacy ``backtest_*`` API and new ``pipeline()`` API.

Each strategy phase populates this file with a ``test_<strat>_equivalent``
case. The baseline is a JSON snapshot of ``pf.stats()`` produced by the
legacy ``backtest_<strat>`` BEFORE it is deleted (see
``tests/_generate_snapshots.py``).

During the refactor the rule is:
1. Generate the snapshot from the legacy API (one-off).
2. Rewrite the strategy with ``pipeline()``.
3. Run this test file — ``pipeline(**params).stats()`` must match the
   frozen snapshot to ``rtol=1e-10``.
4. Only then delete the legacy ``backtest_*``.

This file is intentionally a skeleton at Phase 0. Phases 1-6 add their
own cases via ``@pytest.mark.parametrize``.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

_SNAPSHOT_DIR = Path(__file__).parent / "snapshots"

# Tolerance for float equivalence — tight enough to catch any numerical drift
# but loose enough to absorb platform-dependent FP rounding.
_RTOL = 1e-10
_ATOL = 1e-12


def _snapshot_path(strat: str, label: str) -> Path:
    return _SNAPSHOT_DIR / f"{strat}_{label}.json"


def _load_snapshot(path: Path) -> dict[str, Any]:
    if not path.exists():
        pytest.skip(f"Snapshot {path.name} not generated yet — run tests/_generate_snapshots.py")
    with path.open() as fh:
        return json.load(fh)


def _stats_to_dict(stats: pd.Series) -> dict[str, Any]:
    """Convert pf.stats() Series to a JSON-serializable dict.

    Non-numeric entries (strings, Timedeltas, ...) are stringified so the
    snapshot captures the full state but comparison focuses on numerics.
    """
    out: dict[str, Any] = {}
    for key, val in stats.items():
        if isinstance(val, (int, float, np.integer, np.floating)):
            if np.isnan(val):
                out[str(key)] = None
            else:
                out[str(key)] = float(val)
        else:
            out[str(key)] = str(val)
    return out


def assert_stats_equivalent(
    reference: dict[str, Any],
    candidate: pd.Series,
    *,
    rtol: float = _RTOL,
    atol: float = _ATOL,
) -> None:
    """Assert numeric stats in ``candidate`` match ``reference`` dict."""
    cand_dict = _stats_to_dict(candidate)
    mismatches: list[str] = []
    for key, ref_val in reference.items():
        cand_val = cand_dict.get(key)
        if ref_val is None and cand_val is None:
            continue
        if isinstance(ref_val, (int, float)) and isinstance(cand_val, (int, float)):
            if not np.isclose(ref_val, cand_val, rtol=rtol, atol=atol, equal_nan=True):
                mismatches.append(f"{key}: ref={ref_val!r} vs cand={cand_val!r}")
        else:
            if str(ref_val) != str(cand_val):
                mismatches.append(f"{key} (non-numeric): ref={ref_val!r} vs cand={cand_val!r}")
    if mismatches:
        raise AssertionError(
            f"Stats diverge from snapshot ({len(mismatches)} fields):\n  "
            + "\n  ".join(mismatches)
        )


# ═══════════════════════════════════════════════════════════════════════
# Phase 1+ cases are appended below as strategies are migrated.
# ═══════════════════════════════════════════════════════════════════════


@pytest.fixture(scope="module")
def fx_data():
    """Full EUR-USD dataset — required so snapshots produced by the legacy
    backtest on the full history can be replayed against ``pipeline()``."""
    import sys
    from pathlib import Path

    src_dir = Path(__file__).resolve().parent.parent / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    from utils import apply_vbt_settings, load_fx_data

    apply_vbt_settings()
    _, data = load_fx_data("data/EUR-USD_minute.parquet")
    return data


MR_TURBO_CASES = [
    ("default", dict(bb_window=80, bb_alpha=5.0, sl_stop=0.005, tp_stop=0.006)),
    ("tight_bands", dict(bb_window=60, bb_alpha=4.0, sl_stop=0.004, tp_stop=0.006)),
    ("loose_bands", dict(bb_window=120, bb_alpha=6.0, sl_stop=0.005, tp_stop=0.008)),
]


@pytest.mark.parametrize("label,params", MR_TURBO_CASES, ids=[c[0] for c in MR_TURBO_CASES])
def test_mr_turbo_pipeline_equivalent(label, params, fx_data):
    """pipeline(data, **params) must match the legacy backtest_mr_turbo snapshot."""
    import importlib

    strategies_mr_turbo = importlib.import_module("strategies.mr_turbo")
    pipeline = strategies_mr_turbo.pipeline

    snapshot = _load_snapshot(_snapshot_path("mr_turbo", label))
    pf, _ = pipeline(fx_data, **params)
    assert_stats_equivalent(snapshot["stats"], pf.stats())


def test_helper_roundtrip(tmp_path):
    """Sanity: dict → JSON → compare against the same pf.stats() passes."""
    stats = pd.Series(
        {
            "Total Return [%]": 12.34,
            "Sharpe Ratio": 1.56,
            "Max Drawdown [%]": -4.5,
            "Win Rate [%]": np.nan,
            "Start": pd.Timestamp("2024-01-01"),
            "Duration": pd.Timedelta(days=30),
        }
    )
    snapshot = _stats_to_dict(stats)
    snap_path = tmp_path / "dummy.json"
    with snap_path.open("w") as fh:
        json.dump(snapshot, fh)
    reference = _load_snapshot(snap_path)
    assert_stats_equivalent(reference, stats)
