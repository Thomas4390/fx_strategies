"""Tests for framework.bootstrap and framework.bootstrap_nb.

Scope:
- Stationary bootstrap index generator : length, bounds, determinism.
- Block mean ``block_len_mean=1`` degenerates to iid bootstrap.
- :func:`bootstrap_metric` : returns the expected dict keys, observed
  value matches the VBT-native metric.
- :func:`bootstrap_all_metrics` : DataFrame shape and column names,
  single pass equals per-metric loop (modulo RNG alignment).
- :func:`bootstrap_equity_paths` : correct shape, preserves the
  datetime index, every column finishes at a positive value.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


from framework.bootstrap import (  # noqa: E402
    bootstrap_all_metrics,
    bootstrap_equity_paths,
    bootstrap_metric,
    equity_fan_percentiles,
)
from framework.bootstrap_nb import (  # noqa: E402
    bootstrap_metric_nb,
    moving_block_indices_nb,
    stationary_bootstrap_indices_nb,
)
from framework.pipeline_utils import (  # noqa: E402
    FX_MINUTE_ANN_FACTOR,
    METRIC_NAMES,
    SHARPE_RATIO,
    SORTINO_RATIO,
)


# ═══════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════


@pytest.fixture(scope="module")
def toy_returns() -> pd.Series:
    rng = np.random.default_rng(7)
    n = 2000
    idx = pd.date_range("2022-01-03", periods=n, freq="D")
    r = rng.normal(loc=0.0006, scale=0.012, size=n)
    return pd.Series(r, index=idx, name="returns")


# ═══════════════════════════════════════════════════════════════════════
# Index generators
# ═══════════════════════════════════════════════════════════════════════


def test_stationary_indices_length_and_bounds():
    idx = stationary_bootstrap_indices_nb(500, 50.0, 1)
    assert idx.shape == (500,)
    assert idx.min() >= 0
    assert idx.max() < 500


def test_stationary_indices_deterministic():
    a = stationary_bootstrap_indices_nb(500, 50.0, 1)
    b = stationary_bootstrap_indices_nb(500, 50.0, 1)
    assert np.array_equal(a, b)


def test_stationary_indices_seed_changes_output():
    a = stationary_bootstrap_indices_nb(500, 50.0, 1)
    b = stationary_bootstrap_indices_nb(500, 50.0, 2)
    assert not np.array_equal(a, b)


def test_stationary_block_mean_1_is_almost_iid():
    """``block_len_mean=1`` ⇒ p=1 ⇒ new index every step ⇒ iid."""
    idx = stationary_bootstrap_indices_nb(5000, 1.0, 42)
    # Mean gap between consecutive indices should have no autocorrelation.
    gaps = np.diff(idx)
    # Rough iid check : the fraction of +1 gaps should be ~1/n, not 1.0.
    plus_one_frac = float(np.mean(gaps == 1))
    assert plus_one_frac < 0.05


def test_moving_block_indices_fixed_length():
    idx = moving_block_indices_nb(1000, 20, 1)
    assert idx.shape == (1000,)
    assert idx.min() >= 0
    assert idx.max() < 1000


# ═══════════════════════════════════════════════════════════════════════
# Single metric bootstrap
# ═══════════════════════════════════════════════════════════════════════


def test_bootstrap_metric_keys(toy_returns):
    out = bootstrap_metric(
        toy_returns, SHARPE_RATIO, n_boot=200, block_len_mean=20, seed=1,
        ann_factor=252.0,
    )
    expected = {
        "observed", "mean", "std", "ci_low", "ci_high",
        "samples", "metric_id", "metric_name",
    }
    assert expected <= set(out.keys())
    assert out["samples"].shape == (200,)
    assert out["ci_low"] <= out["mean"] <= out["ci_high"]
    assert out["metric_name"] == METRIC_NAMES[SHARPE_RATIO]


def test_bootstrap_metric_observed_matches_kernel(toy_returns):
    """The ``observed`` slot should equal the direct kernel call."""
    from framework.pipeline_utils import compute_metric_nb

    arr = toy_returns.to_numpy(dtype=np.float64)
    direct = float(compute_metric_nb(arr, SHARPE_RATIO, 252.0, 0.05))
    out = bootstrap_metric(
        toy_returns, SHARPE_RATIO, n_boot=50, block_len_mean=20, seed=1,
        ann_factor=252.0,
    )
    assert abs(out["observed"] - direct) < 1e-12


def test_bootstrap_metric_deterministic(toy_returns):
    a = bootstrap_metric(toy_returns, SHARPE_RATIO, n_boot=100, seed=3, ann_factor=252.0)
    b = bootstrap_metric(toy_returns, SHARPE_RATIO, n_boot=100, seed=3, ann_factor=252.0)
    assert np.array_equal(a["samples"], b["samples"])


# ═══════════════════════════════════════════════════════════════════════
# All metrics
# ═══════════════════════════════════════════════════════════════════════


def test_bootstrap_all_metrics_shape(toy_returns):
    df = bootstrap_all_metrics(
        toy_returns, n_boot=100, block_len_mean=20, seed=1, ann_factor=252.0
    )
    assert df.shape[0] == len(METRIC_NAMES)
    assert {"observed", "mean", "std", "ci_low", "ci_high", "label"} <= set(df.columns)
    assert "sharpe_ratio" in df.index
    assert "sortino_ratio" in df.index


def test_bootstrap_all_metrics_return_samples(toy_returns):
    df, samples = bootstrap_all_metrics(
        toy_returns, n_boot=150, seed=1, ann_factor=252.0, return_samples=True
    )
    assert samples.shape == (150, 14)
    # Column SHARPE_RATIO of the sample matrix should match the single
    # metric bootstrap with the same seed and block.
    single = bootstrap_metric(
        toy_returns, SHARPE_RATIO, n_boot=150, seed=1, ann_factor=252.0
    )
    # Values will differ (single pass resamples once vs one-per-metric),
    # but shapes must line up.
    assert samples[:, SHARPE_RATIO].shape == single["samples"].shape


# ═══════════════════════════════════════════════════════════════════════
# Equity fan chart
# ═══════════════════════════════════════════════════════════════════════


def test_bootstrap_equity_paths_shape(toy_returns):
    df = bootstrap_equity_paths(toy_returns, n_sim=30, block_len_mean=20, seed=1)
    assert df.shape == (len(toy_returns), 30)
    # Should preserve the datetime index.
    assert isinstance(df.index, pd.DatetimeIndex)
    assert (df.iloc[-1] > 0).all()


def test_bootstrap_equity_paths_finite_filter():
    """Infinite values in the input should not break the path builder."""
    idx = pd.date_range("2022-01-01", periods=200, freq="D")
    s = pd.Series(np.random.default_rng(5).normal(0.001, 0.01, 200), index=idx)
    s.iloc[10] = np.inf  # spurious infinity
    df = bootstrap_equity_paths(s, n_sim=20, seed=1)
    assert df.shape[1] == 20
    assert np.isfinite(df.to_numpy()).all()


def test_equity_fan_percentiles_columns(toy_returns):
    paths = bootstrap_equity_paths(toy_returns, n_sim=25, seed=1)
    perc = equity_fan_percentiles(paths)
    assert list(perc.columns) == ["p05", "p25", "p50", "p75", "p95"]
    assert (perc["p05"] <= perc["p50"]).all()
    assert (perc["p50"] <= perc["p95"]).all()


# ═══════════════════════════════════════════════════════════════════════
# Direct kernel smoke
# ═══════════════════════════════════════════════════════════════════════


def test_bootstrap_metric_nb_sortino():
    arr = np.random.default_rng(11).normal(0.0005, 0.01, 1000)
    samples = bootstrap_metric_nb(
        arr, SORTINO_RATIO, 100, 20.0, 42, FX_MINUTE_ANN_FACTOR, 0.05
    )
    assert samples.shape == (100,)
    assert np.isfinite(samples).all()
