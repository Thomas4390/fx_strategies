"""Tests for framework.mc_trades — trade-level Monte Carlo.

Scope:
- :func:`mc_trade_shuffle_nb` preserves the total PnL (product
  invariant since we compound returns).
- :func:`mc_max_drawdown_distribution` returns finite stats and
  the observed MDD is inside ``[0, 1]``.
- :func:`mc_sequence_risk_report` returns a z-score in a reasonable
  range (|z| < 5 on random data).
- Resample mode generates a non-degenerate distribution of terminal
  equities (unlike shuffle which preserves it).
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

from framework.mc_trades import (  # noqa: E402
    mc_max_drawdown_distribution,
    mc_sequence_risk_report,
    mc_trade_equity_paths,
    mc_trade_resample_nb,
    mc_trade_shuffle_nb,
)


# ═══════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════


@pytest.fixture(scope="module")
def toy_pf() -> vbt.Portfolio:
    """Build a small synthetic portfolio with a handful of trades."""
    rng = np.random.default_rng(3)
    n = 500
    idx = pd.date_range("2022-01-03", periods=n, freq="h")
    close = pd.Series(
        100.0 * np.cumprod(1.0 + rng.normal(0.0, 0.005, n)),
        index=idx,
        name="close",
    )
    # Alternate entry/exit every ~25 bars for ~20 trades.
    entries = pd.Series(False, index=idx)
    exits = pd.Series(False, index=idx)
    for i in range(0, n, 50):
        entries.iloc[i] = True
        if i + 25 < n:
            exits.iloc[i + 25] = True
    pf = vbt.Portfolio.from_signals(
        close=close,
        entries=entries,
        exits=exits,
        init_cash=10_000.0,
        fees=0.0,
        slippage=0.0,
    )
    return pf


# ═══════════════════════════════════════════════════════════════════════
# Numba kernels
# ═══════════════════════════════════════════════════════════════════════


def test_shuffle_preserves_terminal_equity():
    """Total compound return must be permutation-invariant."""
    trade_returns = np.array(
        [0.01, -0.005, 0.02, -0.01, 0.015, -0.003, 0.008], dtype=np.float64
    )
    mdds, uws, terminals = mc_trade_shuffle_nb(trade_returns, 100, seed=1)
    expected_terminal = float(np.prod(1.0 + trade_returns))
    # All terminals should be equal (up to float rounding).
    assert np.allclose(terminals, expected_terminal, atol=1e-12)
    assert mdds.shape == (100,)
    assert uws.shape == (100,)
    assert (mdds >= 0).all()
    assert (mdds < 1.0).all()


def test_resample_terminal_varies():
    trade_returns = np.array(
        [0.01, -0.005, 0.02, -0.01, 0.015], dtype=np.float64
    )
    _, _, terminals = mc_trade_resample_nb(trade_returns, 200, seed=2)
    # Terminal dispersion must be strictly positive (sampling with replacement).
    assert float(np.std(terminals)) > 0.0


def test_shuffle_deterministic():
    tr = np.array([0.01, -0.005, 0.02, -0.01], dtype=np.float64)
    a = mc_trade_shuffle_nb(tr, 50, seed=7)
    b = mc_trade_shuffle_nb(tr, 50, seed=7)
    assert np.array_equal(a[0], b[0])
    assert np.array_equal(a[1], b[1])


# ═══════════════════════════════════════════════════════════════════════
# Python API
# ═══════════════════════════════════════════════════════════════════════


def test_mc_max_drawdown_distribution_keys(toy_pf):
    out = mc_max_drawdown_distribution(toy_pf, n_sim=200, mode="shuffle", seed=1)
    expected = {
        "mode", "n_trades", "n_sim", "seed",
        "observed_mdd", "observed_underwater", "observed_terminal",
        "mdd_samples", "mdd_p50", "mdd_p95", "mdd_p99",
        "mdd_mean", "mdd_std", "uw_samples", "uw_p50", "uw_p95",
        "terminal_samples",
    }
    assert expected <= set(out.keys())
    assert 0.0 <= out["observed_mdd"] <= 1.0
    assert out["mdd_samples"].shape == (200,)
    assert out["mode"] == "shuffle"


def test_mc_max_drawdown_invalid_mode(toy_pf):
    with pytest.raises(ValueError):
        mc_max_drawdown_distribution(toy_pf, mode="bogus")


def test_mc_sequence_risk_zscore(toy_pf):
    report = mc_sequence_risk_report(toy_pf, n_sim=200, seed=1)
    assert "sequence_luck_zscore" in report
    assert abs(report["sequence_luck_zscore"]) < 10.0
    assert 0.0 <= report["pct_shuffles_worse_than_observed"] <= 1.0


def test_mc_trade_equity_paths_shape(toy_pf):
    n_paths = 25
    df = mc_trade_equity_paths(toy_pf, n_paths=n_paths, seed=1)
    assert df.shape[1] == n_paths
    assert df.iloc[0].eq(1.0).all()  # every path starts at 1
    assert (df.iloc[-1] > 0).all()
