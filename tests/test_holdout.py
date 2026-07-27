"""Tests for framework.holdout — code-level guard of the holdout policy.

Scope:
- :func:`assert_not_optimizing` — raises on an index touching the frozen
  slice, passes on a pre-freeze index or with ``allow_frozen_oos=True``.
- :func:`trim_insample` / :func:`frozen_oos_slice` — exact partition of a
  series / frame, with a caller-supplied ``holdout_start``.
- tz-aware indexes are accepted (compared on the naive wall clock).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from framework import holdout  # noqa: E402


def _index(start: str, periods: int, tz: str | None = None) -> pd.DatetimeIndex:
    return pd.bdate_range(start, periods=periods, freq="B", tz=tz)


def test_policy_constant_matches_policy_doc():
    assert holdout.HOLDOUT_START == pd.Timestamp("2026-01-01")


def test_assert_raises_when_index_crosses_the_freeze():
    idx = _index("2025-11-01", 60)  # runs into 2026
    assert idx.max() >= holdout.HOLDOUT_START
    with pytest.raises(RuntimeError, match="HOLDOUT_POLICY.md"):
        holdout.assert_not_optimizing(idx)


def test_assert_passes_on_a_2025_index():
    holdout.assert_not_optimizing(_index("2025-01-01", 200))


def test_assert_passes_with_allow_frozen_oos():
    idx = _index("2025-11-01", 60)
    holdout.assert_not_optimizing(idx, allow_frozen_oos=True)


def test_assert_honors_a_custom_holdout_start():
    idx = _index("2025-01-01", 200)  # crosses 2025-07-01
    holdout.assert_not_optimizing(idx)
    with pytest.raises(RuntimeError):
        holdout.assert_not_optimizing(idx, holdout_start=pd.Timestamp("2025-07-01"))


def test_assert_accepts_tz_aware_index():
    holdout.assert_not_optimizing(_index("2025-01-01", 200, tz="UTC"))
    with pytest.raises(RuntimeError):
        holdout.assert_not_optimizing(_index("2025-11-01", 60, tz="America/New_York"))


def test_trim_and_frozen_partition_a_series_exactly():
    idx = _index("2025-11-01", 80)
    series = pd.Series(range(len(idx)), index=idx, dtype=float)

    insample = holdout.trim_insample(series)
    frozen = holdout.frozen_oos_slice(series)

    assert len(insample) and len(frozen)
    assert len(insample) + len(frozen) == len(series)
    assert insample.index.max() < holdout.HOLDOUT_START
    assert frozen.index.min() >= holdout.HOLDOUT_START
    pd.testing.assert_series_equal(pd.concat([insample, frozen]), series)


def test_trim_and_frozen_partition_a_frame_with_custom_start():
    idx = _index("2025-01-01", 300)
    frame = pd.DataFrame({"close": range(len(idx))}, index=idx, dtype=float)
    start = pd.Timestamp("2025-07-01")

    insample = holdout.trim_insample(frame, holdout_start=start)
    frozen = holdout.frozen_oos_slice(frame, holdout_start=start)

    assert len(insample) + len(frozen) == len(frame)
    assert insample.index.max() < start <= frozen.index.min()


def test_frozen_oos_slice_logs_every_read(caplog):
    idx = _index("2025-12-15", 40)
    series = pd.Series(1.0, index=idx)
    with caplog.at_level("WARNING", logger="framework.holdout"):
        holdout.frozen_oos_slice(series)
        holdout.frozen_oos_slice(series)
    reads = [r for r in caplog.records if "FROZEN_OOS read" in r.message]
    assert len(reads) == 2
