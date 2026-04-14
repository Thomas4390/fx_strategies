"""Tests for framework.cpcv — Combinatorially Purged CV.

Scope:
- :func:`build_cpcv_splitter` produces ``C(n_groups, n_test_groups)``
  splits and every train/test pair is disjoint after purge+embargo.
- The splitter is usable via ``vbt.Splitter`` helpers
  (``n_splits``, ``index_bounds``).
- :func:`cpcv_oos_distribution` returns one row per config with all
  expected percentile columns, sorted by median descending.
- :func:`cpcv_summary` matches the combinatorial math.
"""

from __future__ import annotations

import sys
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


from framework.cpcv import (  # noqa: E402
    _embargo_mask,
    _group_bounds,
    _purge_mask,
    build_cpcv_splitter,
    cpcv_oos_distribution,
    cpcv_summary,
)


# ═══════════════════════════════════════════════════════════════════════
# Splitter construction
# ═══════════════════════════════════════════════════════════════════════


@pytest.fixture(scope="module")
def daily_index() -> pd.DatetimeIndex:
    return pd.date_range("2020-01-01", periods=600, freq="D")


def test_cpcv_combination_count(daily_index):
    splitter = build_cpcv_splitter(
        daily_index, n_groups=6, n_test_groups=2, purge_td="1 day", embargo_pct=0.005
    )
    expected_splits = comb(6, 2)
    assert splitter.n_splits == expected_splits


def test_cpcv_train_test_disjoint_at_helpers(daily_index):
    """Reconstruct one CPCV split manually and verify disjointness."""
    n = len(daily_index)
    bounds = _group_bounds(n, n_groups=4)
    # Test = group 2 only.
    test_mask = np.zeros(n, dtype=bool)
    s, e = bounds[2]
    test_mask[s:e] = True
    purge_m = _purge_mask(daily_index, test_mask, pd.Timedelta("1 day"))
    emb_m = _embargo_mask(test_mask, embargo_bars=5)
    train_mask = ~test_mask & ~purge_m & ~emb_m

    assert not np.any(train_mask & test_mask)
    assert train_mask.sum() > 0
    assert test_mask.sum() > 0


def test_cpcv_splitter_has_expected_split_count(daily_index):
    """Smoke test : the splitter exposes ``n_splits`` matching C(N, k)."""
    splitter = build_cpcv_splitter(
        daily_index, n_groups=5, n_test_groups=2, purge_td="1 day", embargo_pct=0.005
    )
    assert splitter.n_splits == comb(5, 2)
    # Each split has 2 sets (train, test).
    assert splitter.n_sets == 2


def test_cpcv_invalid_test_groups(daily_index):
    with pytest.raises(ValueError):
        build_cpcv_splitter(daily_index, n_groups=3, n_test_groups=3)
    with pytest.raises(ValueError):
        build_cpcv_splitter(daily_index, n_groups=3, n_test_groups=0)


def test_cpcv_too_short_index():
    short_idx = pd.date_range("2020-01-01", periods=5, freq="D")
    with pytest.raises(ValueError):
        build_cpcv_splitter(short_idx, n_groups=6, n_test_groups=2)


# ═══════════════════════════════════════════════════════════════════════
# OOS distribution
# ═══════════════════════════════════════════════════════════════════════


def _fake_grid_perf(n_configs: int, n_splits: int, seed: int = 0) -> pd.Series:
    """Build a synthetic ``@vbt.cv_split``-style Series indexed by
    ``(param, split, set)``."""
    rng = np.random.default_rng(seed)
    rows: list[tuple[int, int, str, float]] = []
    for p in range(n_configs):
        mean_sr = rng.normal(0.0, 0.3)
        for s in range(n_splits):
            for set_name in ("train", "test"):
                rows.append(
                    (p, s, set_name, mean_sr + rng.normal(0.0, 0.1))
                )
    df = pd.DataFrame(rows, columns=["param", "split", "set", "value"])
    return df.set_index(["param", "split", "set"])["value"]


def test_cpcv_oos_distribution_columns():
    perf = _fake_grid_perf(n_configs=5, n_splits=10, seed=1)
    dist = cpcv_oos_distribution(perf, metric_label="Sharpe")
    expected = {"mean", "median", "std", "min", "max", "q05", "q95", "pct_positive", "n_splits"}
    assert expected <= set(dist.columns)
    assert dist.shape[0] == 5
    # Sorted by median descending.
    medians = dist["median"].values
    assert (np.diff(medians) <= 0).all()


def test_cpcv_summary_math():
    perf = _fake_grid_perf(n_configs=4, n_splits=15, seed=2)
    summary = cpcv_summary(perf, n_groups=6, n_test_groups=2)
    assert summary["n_splits_expected"] == comb(6, 2)
    assert summary["n_reconstructed_paths"] == comb(5, 1)
    assert summary["n_configs"] == 4
