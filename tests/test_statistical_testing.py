"""Tests for framework.statistical_testing — DSR, PSR and CSCV PBO.

Scope:
- :func:`probabilistic_sharpe_ratio` — monotonicity vs the benchmark SR.
- :func:`expected_max_sharpe` — known asymptotic limits.
- :func:`deflated_sharpe_ratio` — input validation, DSR monotonic in
  ``n_trials`` (more trials → lower DSR).
- :func:`probability_of_backtest_overfitting` — CSCV returns
  ``pbo ∈ [0, 1]``, real-alpha case has PBO strictly below pure noise.
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


import vectorbtpro as vbt  # noqa: E402

vbt.settings.returns.year_freq = pd.Timedelta(days=252)

from framework.statistical_testing import (  # noqa: E402
    deflated_sharpe_ratio,
    expected_max_sharpe,
    probabilistic_sharpe_ratio,
    probability_of_backtest_overfitting,
)


# ═══════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════


@pytest.fixture(scope="module")
def positive_edge_returns() -> pd.Series:
    """Synthetic returns with a clearly positive Sharpe."""
    rng = np.random.default_rng(20260413)
    n = 1500
    idx = pd.bdate_range("2018-01-01", periods=n, freq="B")
    return pd.Series(rng.normal(0.0006, 0.01, n), index=idx, name="alpha")


@pytest.fixture(scope="module")
def noise_returns() -> pd.Series:
    """Pure-noise returns — Sharpe ≈ 0."""
    rng = np.random.default_rng(20260414)
    n = 1500
    idx = pd.bdate_range("2018-01-01", periods=n, freq="B")
    return pd.Series(rng.normal(0.0, 0.01, n), index=idx, name="noise")


# ═══════════════════════════════════════════════════════════════════════
# Probabilistic Sharpe Ratio
# ═══════════════════════════════════════════════════════════════════════


def test_psr_is_bounded_01(positive_edge_returns):
    psr = probabilistic_sharpe_ratio(positive_edge_returns, sr_benchmark=0.0)
    assert 0.0 <= psr <= 1.0


def test_psr_monotonic_in_benchmark(positive_edge_returns):
    """Higher benchmark → lower PSR (the bar to clear is higher)."""
    psr_low = probabilistic_sharpe_ratio(positive_edge_returns, sr_benchmark=0.0)
    psr_mid = probabilistic_sharpe_ratio(positive_edge_returns, sr_benchmark=0.5)
    psr_high = probabilistic_sharpe_ratio(positive_edge_returns, sr_benchmark=2.0)
    assert psr_low >= psr_mid >= psr_high


def test_psr_noise_against_observed_sr(noise_returns):
    """PSR vs the empirical Sharpe should be exactly 0.5 (SR = SR*)."""
    sr_hat = float(noise_returns.vbt.returns(freq="1D").sharpe_ratio())
    psr = probabilistic_sharpe_ratio(noise_returns, sr_benchmark=sr_hat)
    # PSR(SR*, SR*) = Φ(0) = 0.5 up to floating point.
    assert abs(psr - 0.5) < 1e-9, f"expected 0.5, got {psr}"


# ═══════════════════════════════════════════════════════════════════════
# Expected max Sharpe
# ═══════════════════════════════════════════════════════════════════════


def test_expected_max_sharpe_zero_variance():
    """Zero variance across trials ⇒ expected max SR = 0."""
    assert expected_max_sharpe(n_trials=100, trial_sr_var=0.0) == 0.0


def test_expected_max_sharpe_one_trial():
    """N = 1 ⇒ expected max = 0 (no selection bias)."""
    assert expected_max_sharpe(n_trials=1, trial_sr_var=0.5) == 0.0


def test_expected_max_sharpe_monotonic_in_n_trials():
    """More trials ⇒ higher expected maximum."""
    v = 0.25
    exp2 = expected_max_sharpe(2, v)
    exp10 = expected_max_sharpe(10, v)
    exp100 = expected_max_sharpe(100, v)
    assert exp2 < exp10 < exp100


def test_expected_max_sharpe_monotonic_in_variance():
    """More variance across trials ⇒ higher expected max."""
    n = 50
    low = expected_max_sharpe(n, 0.1)
    high = expected_max_sharpe(n, 1.0)
    assert low < high


# ═══════════════════════════════════════════════════════════════════════
# Deflated Sharpe Ratio
# ═══════════════════════════════════════════════════════════════════════


def test_dsr_requires_exactly_one_variance_arg(positive_edge_returns):
    with pytest.raises(ValueError, match="exactly one"):
        deflated_sharpe_ratio(positive_edge_returns, n_trials=10)
    with pytest.raises(ValueError, match="exactly one"):
        deflated_sharpe_ratio(
            positive_edge_returns,
            n_trials=10,
            trial_sharpes=[0.1, 0.2],
            trial_sr_var=0.5,
        )


def test_dsr_rejects_too_few_sharpes(positive_edge_returns):
    with pytest.raises(ValueError, match="at least 2"):
        deflated_sharpe_ratio(positive_edge_returns, n_trials=10, trial_sharpes=[0.5])


def test_dsr_output_schema(positive_edge_returns):
    rng = np.random.default_rng(1)
    sharpes = rng.normal(0.3, 0.2, 20)
    result = deflated_sharpe_ratio(
        positive_edge_returns, n_trials=20, trial_sharpes=sharpes
    )
    for key in ("sharpe", "sharpe_std", "expected_max_sharpe", "dsr", "n_trials"):
        assert key in result
    assert result["n_trials"] == 20
    assert 0.0 <= result["dsr"] <= 1.0


def test_dsr_decreases_with_more_trials(positive_edge_returns):
    """Hold trial variance constant, vary n_trials — DSR must drop."""
    var = 0.09  # σ_SR = 0.3
    d_low = deflated_sharpe_ratio(positive_edge_returns, n_trials=5, trial_sr_var=var)
    d_high = deflated_sharpe_ratio(
        positive_edge_returns, n_trials=500, trial_sr_var=var
    )
    assert d_low["expected_max_sharpe"] < d_high["expected_max_sharpe"]
    # More trials ⇒ higher threshold ⇒ DSR is ≤.
    assert d_high["dsr"] <= d_low["dsr"] + 1e-9


def test_dsr_accepts_pandas_series_for_trials(positive_edge_returns):
    sharpes = pd.Series([0.1, 0.2, 0.3, 0.4, 0.5])
    result = deflated_sharpe_ratio(
        positive_edge_returns, n_trials=5, trial_sharpes=sharpes
    )
    assert np.isfinite(result["dsr"])


# ═══════════════════════════════════════════════════════════════════════
# Probability of Backtest Overfitting (CSCV)
# ═══════════════════════════════════════════════════════════════════════


def test_pbo_bounded_01():
    rng = np.random.default_rng(0)
    mat = pd.DataFrame(rng.normal(0, 0.01, (800, 16)))
    result = probability_of_backtest_overfitting(mat, n_bins=8)
    assert 0.0 <= result["pbo"] <= 1.0
    assert result["n_configs"] == 16
    assert result["n_bins"] == 8


def test_pbo_validates_n_bins():
    mat = pd.DataFrame(np.zeros((100, 4)))
    with pytest.raises(ValueError, match="even positive"):
        probability_of_backtest_overfitting(mat, n_bins=7)
    with pytest.raises(ValueError, match="even positive"):
        probability_of_backtest_overfitting(mat, n_bins=0)


def test_pbo_validates_objective():
    mat = pd.DataFrame(np.ones((100, 4)))
    with pytest.raises(ValueError, match="objective"):
        probability_of_backtest_overfitting(mat, n_bins=4, objective="foo")


def test_pbo_requires_two_configs():
    mat = pd.DataFrame(np.ones((100, 1)))
    with pytest.raises(ValueError, match="at least 2 configs"):
        probability_of_backtest_overfitting(mat, n_bins=4)


def test_pbo_persistent_edge_lower_than_noise():
    """A real alpha column should lower PBO vs pure noise."""
    rng = np.random.default_rng(42)
    n, k = 1600, 20
    noise = rng.normal(0, 0.01, (n, k))
    pbo_noise = probability_of_backtest_overfitting(
        pd.DataFrame(noise), n_bins=10, objective="sharpe"
    )

    alpha = noise.copy()
    alpha[:, 3] += 0.0015  # persistent edge on col 3
    pbo_alpha = probability_of_backtest_overfitting(
        pd.DataFrame(alpha), n_bins=10, objective="sharpe"
    )
    # Real alpha must reduce the overfit probability by a clear margin.
    assert pbo_alpha["pbo"] < pbo_noise["pbo"] - 0.1


def test_pbo_logits_length_matches_splits():
    rng = np.random.default_rng(1)
    mat = pd.DataFrame(rng.normal(0, 0.01, (400, 8)))
    result = probability_of_backtest_overfitting(mat, n_bins=8)
    # C(8, 4) = 70 splits.
    assert result["n_splits"] == 70
    assert len(result["logits"]) == 70
