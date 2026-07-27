"""Code-level guard for the repo holdout policy.

Single source of truth for the rule specified in
``docs/research/HOLDOUT_POLICY.md`` (Phase 21, active) : **all data on or
after 2026-01-01 is locked out of model selection**. The policy file
describes this module as the enforcement layer ; the numbers here must
never drift from it.

Two ways to use it :

- ``assert_not_optimizing(index)`` at the top of a sweep, on the index of
  the data actually used for ranking — fails loudly if the frozen slice
  leaked into selection ;
- ``trim_insample`` / ``frozen_oos_slice`` to partition a series or frame,
  the second one logging every read because a holdout is a budget, not a
  resource (see the consumption log in the policy).

Timezone note. Comparisons are made on a naive wall clock : the repo's
price indexes are naive New York timestamps, while the policy states the
freeze in UTC. A tz-aware index is therefore converted with
``tz_localize(None)`` (wall clock kept, offset dropped) rather than
converted to UTC, so a naive and a tz-aware view of the same bars are cut
at the same place.

``holdout_start`` is a parameter everywhere : some sweeps deliberately
freeze earlier than the repo policy (``sweep_gold_sizing.py`` locks
2025-07-01), which is stricter and therefore allowed.
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)

POLICY_DOC = "docs/research/HOLDOUT_POLICY.md"

HOLDOUT_START = pd.Timestamp("2026-01-01")


def _naive_index(index: pd.Index) -> pd.DatetimeIndex:
    """Datetime index stripped of its timezone (see module timezone note)."""
    idx = pd.DatetimeIndex(index)
    return idx.tz_localize(None) if idx.tz is not None else idx


def _naive_ts(ts: pd.Timestamp) -> pd.Timestamp:
    ts = pd.Timestamp(ts)
    return ts.tz_localize(None) if ts.tz is not None else ts


def assert_not_optimizing(
    index: pd.DatetimeIndex,
    *,
    holdout_start: pd.Timestamp = HOLDOUT_START,
    allow_frozen_oos: bool = False,
) -> None:
    """Raise ``RuntimeError`` if ``index`` touches the frozen slice.

    Parameters
    ----------
    index
        Index of the data used for ranking / selection.
    holdout_start
        Freeze date. Defaults to the repo policy ; pass an earlier date
        for a sweep that locks more than the policy requires.
    allow_frozen_oos
        Escape hatch for the single inference pass the policy allows once
        per phase. The caller is expected to mark the result
        ``FROZEN_OOS_RESULT`` in its report.
    """
    if allow_frozen_oos:
        return
    naive = _naive_index(index)
    start = _naive_ts(holdout_start)
    n_frozen = int((naive >= start).sum())
    if n_frozen:
        raise RuntimeError(
            f"{n_frozen} timestamps >= {start.date()} in the selection index "
            f"(up to {naive.max()}): the frozen slice cannot be used for model "
            f"selection ({POLICY_DOC}). Use trim_insample() to rank, or pass "
            "allow_frozen_oos=True for the single inference pass allowed per phase."
        )


def trim_insample(obj, *, holdout_start: pd.Timestamp = HOLDOUT_START):
    """Return the ``< holdout_start`` slice of a datetime-indexed object."""
    return obj[_naive_index(obj.index) < _naive_ts(holdout_start)]


def frozen_oos_slice(obj, *, holdout_start: pd.Timestamp = HOLDOUT_START):
    """Return the ``>= holdout_start`` slice — and record the read.

    Every call is logged : reads of the frozen slice are budgeted, and the
    consumption log in the policy is only honest if the code says out loud
    when it touches the blind period.
    """
    logger.warning("FROZEN_OOS read — one per phase (%s)", POLICY_DOC)
    return obj[_naive_index(obj.index) >= _naive_ts(holdout_start)]
