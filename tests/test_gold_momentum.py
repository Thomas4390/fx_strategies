"""Regression lock for the Gold Momentum sleeve.

Three things are pinned here, and the middle one is the reason this file
exists at all:

1. ``pipeline().stats()`` against frozen snapshots at ``rtol=1e-10`` — the same
   contract as ``test_pipeline_equivalence.py``.
2. **The session boundary.** The sleeve cut its sessions at midnight until
   2026-07-25, which turned every Sunday evening into a session of its own: 392
   of them, ~356 minutes each against 1375 for a real one. Session count was
   inflated 20% and every lookback shortened by the same proportion. Nothing in
   any output said so — which is precisely why it needs a test rather than a
   comment.
3. The daily trace contract, since three engines diff against it.

Snapshots are regenerated with ``python tests/_generate_snapshots.py --strat
gold_momentum``. Regenerating them to make a red test pass is only correct when
the behaviour change was intended.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

_SNAPSHOT_DIR = Path(__file__).parent / "snapshots"
_RTOL = 1e-10
_ATOL = 1e-12

# Cut at 17:00 New York over 2019-01-01 → 2026-07-23, the gold minute export
# yields exactly this many sessions, none of them a Sunday.
_EXPECTED_SESSIONS = 1971

GOLD_CASES: list[tuple[str, dict[str, Any]]] = [
    ("default", {}),
    ("no_vol_target", {"target_vol": None}),
    ("short_enabled", {"allow_short": True}),
]


@pytest.fixture(scope="module")
def gold_data():
    """Minute-level XAUUSD, loaded once — the parquet is large."""
    from utils import apply_vbt_settings, load_gold_data

    apply_vbt_settings()
    _, data = load_gold_data()
    return data


def _load_snapshot(strat: str, label: str) -> dict[str, Any]:
    path = _SNAPSHOT_DIR / f"{strat}_{label}.json"
    if not path.exists():
        pytest.skip(
            f"Snapshot {path.name} not generated — run "
            "python tests/_generate_snapshots.py --strat gold_momentum"
        )
    with path.open() as fh:
        return json.load(fh)


def _assert_stats_match(reference: dict[str, Any], candidate: pd.Series) -> None:
    mismatches: list[str] = []
    for key, ref in reference.items():
        got = candidate.get(key)
        if isinstance(got, (np.integer, np.floating)):
            got = float(got)
        if ref is None and (got is None or (isinstance(got, float) and np.isnan(got))):
            continue
        if isinstance(ref, (int, float)) and isinstance(got, (int, float)):
            if not np.isclose(ref, got, rtol=_RTOL, atol=_ATOL, equal_nan=True):
                mismatches.append(f"{key}: snapshot={ref!r} vs got={got!r}")
        elif str(ref) != str(got):
            mismatches.append(f"{key} (non-numeric): snapshot={ref!r} vs got={got!r}")
    if mismatches:
        raise AssertionError(
            f"Gold sleeve diverges from snapshot ({len(mismatches)} field(s)):\n  "
            + "\n  ".join(mismatches)
        )


@pytest.mark.parametrize("label,params", GOLD_CASES, ids=[c[0] for c in GOLD_CASES])
def test_gold_pipeline_matches_snapshot(label, params, gold_data):
    """pipeline(**params).stats() must reproduce the frozen snapshot."""
    from strategies.gold_momentum import pipeline

    snapshot = _load_snapshot("gold_momentum", label)
    pf, _ = pipeline(gold_data, **params)
    _assert_stats_match(snapshot["stats"], pf.stats())


def test_sessions_close_at_five_pm_new_york(gold_data):
    """No Sunday sessions, and the session count matches the 17:00 boundary."""
    from strategies.gold_momentum import _daily_close

    close = _daily_close(gold_data)
    sundays = int((close.index.dayofweek == 6).sum())
    saturdays = int((close.index.dayofweek == 5).sum())

    assert sundays == 0, (
        f"{sundays} Sunday session(s): the boundary regressed to the calendar "
        "day. A Sunday-evening stub is ~356 minutes against 1375 for a real "
        "session, and counting it shortens every lookback."
    )
    assert saturdays == 0, f"{saturdays} Saturday session(s) — gold does not trade Saturday"
    assert len(close) == _EXPECTED_SESSIONS, (
        f"{len(close)} sessions, expected {_EXPECTED_SESSIONS}. A count near "
        "2363 means midnight boundaries are back."
    )


def test_session_dates_maps_around_the_boundary():
    """16:59 stays on the session, 17:01 starts the next one."""
    from strategies.gold_momentum import session_dates

    stamps = pd.DatetimeIndex([
        "2024-03-05 16:59",
        "2024-03-05 17:00",
        "2024-03-05 17:01",
        "2024-03-05 23:59",
    ])
    mapped = session_dates(stamps)
    assert str(mapped[0].date()) == "2024-03-05"
    assert str(mapped[1].date()) == "2024-03-05", "17:00 exactly still closes that session"
    assert str(mapped[2].date()) == "2024-03-06"
    assert str(mapped[3].date()) == "2024-03-06"


def test_daily_trace_contract(gold_data, tmp_path):
    """The six-column trace the three engines are diffed on."""
    from strategies.gold_momentum import (
        DEFAULT_LOOKBACKS,
        TRACE_COLUMNS,
        emit_daily_trace,
        pipeline,
    )

    pf, indicator = pipeline(gold_data)
    out = tmp_path / "trace.csv"
    trace = emit_daily_trace(pf, indicator, out)

    assert list(trace.columns) == list(TRACE_COLUMNS)
    assert out.exists()
    reread = pd.read_csv(out)
    assert list(reread.columns) == list(TRACE_COLUMNS)
    assert len(reread) == len(trace)

    # Warmup rows are dropped, not zero-filled: a 0.0 score means "the horizons
    # disagree", which is a different statement from "no score yet".
    assert trace["score"].notna().all()
    # The warmup is exactly the longest lookback, read from the sleeve rather
    # than hardcoded — retuning the lookbacks must not silently redefine what
    # this test checks.
    assert len(trace) == _EXPECTED_SESSIONS - max(DEFAULT_LOOKBACKS)

    # Flat whenever the signal does not call for a position.
    assert ((trace["target_weight"] == 0.0) == (trace["score"] <= 0.0)).all()
    assert (trace["equity"] > 0).all()
    assert trace["date"].is_monotonic_increasing
