"""Tests for combined_portfolio_v2 (Phase 3).

Covers:
- ``test_v2_matches_v1_when_leverage_disabled`` — regression guard that
  v2 with ``allocation="risk_parity"``, ``target_vol=None`` and
  ``dd_cap_enabled=False`` reproduces v1 bit-identically on a synthetic
  multi-strategy return set.
- ``test_vol_regime_no_lookahead`` — flipping returns at ``t=last``
  must not change vol_regime at ``t < last``.
- ``test_dd_cap_activates_after_drawdown`` — injecting a synthetic
  drawdown series triggers the lagged leverage scale correctly.
- ``test_regime_adaptive_weights_shape`` — weights DataFrame has no
  NaN, sums to 1 per row, and hits at least 2 different vol regimes
  on a stationary random walk.
- ``test_no_lookahead_on_real_data`` — end-to-end fuzz: mutating the
  last day of each component returns must not change the portfolio
  returns before that day.
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


# ═══════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════


@pytest.fixture(scope="module")
def synthetic_strat_rets() -> dict[str, pd.Series]:
    """Three synthetic daily return series with distinct profiles.

    - MR_Macro : low-vol, slight positive drift (Sharpe ~0.5).
    - XS_Momentum : medium-vol trending up (Sharpe ~0.4).
    - TS_Momentum_RSI : medium-vol mean-reverting around zero.

    Identical RNG seeding guarantees reproducibility of the equivalence
    test across runs.
    """
    rng = np.random.default_rng(20260413)
    n = 2500  # ~10 years of daily bars
    idx = pd.bdate_range("2016-01-04", periods=n, freq="B")
    mr = pd.Series(rng.normal(0.0004, 0.006, n), index=idx, name="MR_Macro")
    xs = pd.Series(rng.normal(0.0003, 0.009, n), index=idx, name="XS_Momentum")
    ts = pd.Series(
        rng.normal(0.0002, 0.008, n), index=idx, name="TS_Momentum_RSI"
    )
    return {"MR_Macro": mr, "XS_Momentum": xs, "TS_Momentum_RSI": ts}


# ═══════════════════════════════════════════════════════════════════════
# Equivalence: v2(risk_parity, no leverage, no DD cap) == v1(risk_parity)
# ═══════════════════════════════════════════════════════════════════════


def test_v2_matches_v1_when_leverage_disabled(synthetic_strat_rets):
    """The disabled-layers v2 path must reproduce v1 risk_parity exactly."""
    from strategies.combined_portfolio import build_combined_portfolio
    from strategies.combined_portfolio_v2 import build_combined_portfolio_v2

    res_v1 = build_combined_portfolio(synthetic_strat_rets, allocation="risk_parity")
    res_v2 = build_combined_portfolio_v2(
        synthetic_strat_rets,
        allocation="risk_parity",
        target_vol=None,
        dd_cap_enabled=False,
    )

    # Portfolio returns must be identical to the last bit.
    np.testing.assert_allclose(
        res_v1["portfolio_returns"].values,
        res_v2["portfolio_returns"].values,
        rtol=0,
        atol=0,
        err_msg="v2 (disabled layers) diverges from v1 portfolio returns",
    )
    # And the Sharpe ratio rounded to ~10 decimals must match (VBT may
    # differ on the last ULP due to internal caching).
    assert abs(res_v1["sharpe"] - res_v2["sharpe"]) < 1e-10


# ═══════════════════════════════════════════════════════════════════════
# Look-ahead guards
# ═══════════════════════════════════════════════════════════════════════


def test_vol_regime_no_lookahead(synthetic_strat_rets):
    """Flipping only the last return must not alter regime at t < last."""
    from strategies.combined_portfolio_v2 import compute_vol_regime

    common = pd.DataFrame(synthetic_strat_rets).dropna()
    ew = common.mean(axis=1)
    regime_orig = compute_vol_regime(ew)

    ew_perturbed = ew.copy()
    ew_perturbed.iloc[-1] = 0.50  # 50% one-day return = massive outlier
    regime_perturbed = compute_vol_regime(ew_perturbed)

    # All rows except the last must be unchanged. The last row is
    # allowed to change because the vol ratio at t uses vol up to t-1,
    # and perturbing t=last affects vol for t>last which doesn't exist.
    pd.testing.assert_series_equal(
        regime_orig.iloc[:-1],
        regime_perturbed.iloc[:-1],
        check_names=False,
    )


def test_no_lookahead_end_to_end(synthetic_strat_rets):
    """End-to-end fuzz: mutating the last day must not change prior bars.

    Rebuilds the full v2 portfolio with target_vol + DD cap active, then
    perturbs the last row of all three components by +50% and rebuilds.
    All portfolio returns strictly before the last bar must be identical.
    """
    from strategies.combined_portfolio_v2 import build_combined_portfolio_v2

    res_orig = build_combined_portfolio_v2(
        synthetic_strat_rets,
        allocation="regime_adaptive",
        target_vol=0.12,
        max_leverage=3.0,
        dd_cap_enabled=True,
    )

    perturbed = {k: s.copy() for k, s in synthetic_strat_rets.items()}
    for s in perturbed.values():
        s.iloc[-1] = 0.50

    res_pert = build_combined_portfolio_v2(
        perturbed,
        allocation="regime_adaptive",
        target_vol=0.12,
        max_leverage=3.0,
        dd_cap_enabled=True,
    )

    orig = res_orig["portfolio_returns"]
    pert = res_pert["portfolio_returns"]
    # Everything strictly before the last index must be bit-identical.
    np.testing.assert_allclose(
        orig.iloc[:-1].values,
        pert.iloc[:-1].values,
        rtol=0,
        atol=0,
        err_msg="Look-ahead detected: prior portfolio returns mutated",
    )


# ═══════════════════════════════════════════════════════════════════════
# DD cap activation
# ═══════════════════════════════════════════════════════════════════════


def test_dd_cap_activates_after_drawdown():
    """Injecting a synthetic DD must produce the documented leverage scale."""
    from strategies.combined_portfolio_v2 import compute_dd_cap_scale

    # Synthetic prelev series: +0.1% daily, then a single -25% shock,
    # then +0.1% daily. Equity peaks at the shock date, drops 25%,
    # then flatlines-ish.
    n = 120
    idx = pd.bdate_range("2020-01-01", periods=n, freq="B")
    rets = pd.Series(0.001, index=idx)
    shock_idx = 40
    rets.iloc[shock_idx] = -0.25
    scale = compute_dd_cap_scale(rets)

    # Before the shock the scale must be 1 (no DD).
    assert (scale.iloc[: shock_idx + 1] == 1.0).all()

    # After the shock (from shock+2 onward) the drawdown exceeds 20% so
    # the scale must drop below 1.0 AND above 0.15 (not fully flat).
    # shock+1: shift(1) means DD is still 0 at this bar.
    after_shock = scale.iloc[shock_idx + 2 :]
    assert (after_shock < 1.0).all(), "DD cap never activated"
    assert (after_shock >= 0.15).all(), "DD cap over-activated below floor"

    # Specifically at DD ~ -25% the scale must be ~0.475 (halfway
    # between 0.6 at DD=0.20 and 0.35 at DD=0.30 in the interp table).
    actual_scale = scale.iloc[shock_idx + 2]
    assert 0.4 < actual_scale < 0.55, (
        f"DD cap scale at -25% DD expected in [0.4, 0.55], got {actual_scale}"
    )


# ═══════════════════════════════════════════════════════════════════════
# Regime-adaptive weights shape
# ═══════════════════════════════════════════════════════════════════════


def test_regime_adaptive_weights_shape(synthetic_strat_rets):
    from strategies.combined_portfolio_v2 import (
        compute_regime_adaptive_weights,
    )

    common = pd.DataFrame(synthetic_strat_rets).dropna()
    weights, vol_regime, trend_score = compute_regime_adaptive_weights(common)

    # No NaN in weights, each row sums to ~1.
    assert not weights.isna().any().any(), "weights contain NaN"
    sums = weights.sum(axis=1)
    np.testing.assert_allclose(sums.values, 1.0, rtol=1e-10, atol=1e-12)

    # Must hit at least 2 vol regimes over a 10-year random walk (the
    # ratio wanders enough to produce some low/high cells).
    unique_regimes = set(vol_regime.unique())
    assert len(unique_regimes) >= 2, (
        f"Synthetic data only produced one vol regime: {unique_regimes}"
    )


# ═══════════════════════════════════════════════════════════════════════
# Sanity: v2 regime-adaptive with leverage beats v2 no-leverage on CAGR
# ═══════════════════════════════════════════════════════════════════════


def test_v2_leverage_boosts_cagr(synthetic_strat_rets):
    """target_vol=0.12 must produce a higher annualized return than no-leverage.

    On the synthetic set (slight positive drift for MR and XS) the
    regime-adaptive + vol-targeting path should materially boost the
    annualized return compared to the bare weighted sum.
    """
    from strategies.combined_portfolio_v2 import build_combined_portfolio_v2

    res_nolev = build_combined_portfolio_v2(
        synthetic_strat_rets,
        allocation="regime_adaptive",
        target_vol=None,
        dd_cap_enabled=False,
    )
    res_lev = build_combined_portfolio_v2(
        synthetic_strat_rets,
        allocation="regime_adaptive",
        target_vol=0.12,
        max_leverage=3.0,
        dd_cap_enabled=True,
    )

    # Levered vol must be closer to the 12% target than the unlevered.
    assert abs(res_lev["annual_vol"] - 0.12) < abs(res_nolev["annual_vol"] - 0.12)
    # And the annual return must be strictly higher (positive drift).
    assert res_lev["annual_return"] > res_nolev["annual_return"]
