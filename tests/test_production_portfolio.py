"""Tests for the production portfolio one-liner in combined_portfolio_v2.

``build_production_portfolio`` is a convenience wrapper around
``build_combined_portfolio_v2`` that pins the recipe to the canonical
production configuration (80/10/10 weights, tv=0.28, ml=12, DD cap off).
These tests verify the wrapper is bit-identical to a manual call with
the same arguments, so reports and reproduction scripts can treat it
as the canonical entry point without worrying about drift.
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


@pytest.fixture(scope="module")
def synthetic_production_rets() -> dict[str, pd.Series]:
    """Synthetic 3-sleeve return set matching the production trio keys.

    Uses the same generator as ``test_combined_portfolio_v2`` so the
    fixture is reproducible; only the strategy names differ to match
    the production sleeves (MR_Macro / TS_Momentum_3p / RSI_Daily_4p).
    """
    rng = np.random.default_rng(20260413)
    n = 2500
    idx = pd.bdate_range("2016-01-04", periods=n, freq="B")
    return {
        "MR_Macro": pd.Series(rng.normal(0.0004, 0.006, n), index=idx, name="MR_Macro"),
        "TS_Momentum_3p": pd.Series(
            rng.normal(0.0003, 0.009, n), index=idx, name="TS_Momentum_3p"
        ),
        "RSI_Daily_4p": pd.Series(
            rng.normal(0.0002, 0.005, n), index=idx, name="RSI_Daily_4p"
        ),
    }


def test_production_helper_matches_manual_call(synthetic_production_rets):
    """build_production_portfolio must equal a manual build_combined_portfolio_v2 call."""
    from strategies.combined_portfolio_v2 import (
        PRODUCTION_MAX_LEVERAGE,
        PRODUCTION_TARGET_VOL,
        PRODUCTION_WEIGHTS,
        build_combined_portfolio_v2,
        build_production_portfolio,
    )

    res_helper = build_production_portfolio(synthetic_production_rets)
    res_manual = build_combined_portfolio_v2(
        synthetic_production_rets,
        allocation="custom",
        custom_weights=PRODUCTION_WEIGHTS,
        target_vol=PRODUCTION_TARGET_VOL,
        max_leverage=PRODUCTION_MAX_LEVERAGE,
        dd_cap_enabled=False,
    )

    np.testing.assert_allclose(
        res_helper["portfolio_returns"].values,
        res_manual["portfolio_returns"].values,
        rtol=0,
        atol=0,
        err_msg="Production helper diverges from the manual call",
    )
    assert abs(res_helper["sharpe"] - res_manual["sharpe"]) < 1e-12
    assert abs(res_helper["annual_return"] - res_manual["annual_return"]) < 1e-12
    assert abs(res_helper["max_drawdown"] - res_manual["max_drawdown"]) < 1e-12


def test_production_helper_uses_canonical_weights(synthetic_production_rets):
    """The weights_ts returned by the helper must sum to 1 with the canonical ratios."""
    from strategies.combined_portfolio_v2 import (
        PRODUCTION_WEIGHTS,
        build_production_portfolio,
    )

    res = build_production_portfolio(synthetic_production_rets)
    weights_ts = res["weights_ts"]

    assert list(weights_ts.columns) == list(PRODUCTION_WEIGHTS.keys())
    # Every row sums to 1 (renormalization guard in compute_regime_adaptive_weights
    # is not triggered here because allocation='custom', but the static
    # branch of _compute_weights_ts normalizes too).
    np.testing.assert_allclose(weights_ts.sum(axis=1).values, 1.0, rtol=1e-12, atol=0)
    # And each column's time-series mean matches the production target share
    # exactly (static allocation → constant weights).
    for strat, expected in PRODUCTION_WEIGHTS.items():
        assert abs(weights_ts[strat].mean() - expected) < 1e-12


def test_production_helper_pins_custom_config_params(synthetic_production_rets):
    """Helper must record the canonical target_vol / max_leverage / dd_cap=False."""
    from strategies.combined_portfolio_v2 import (
        PRODUCTION_MAX_LEVERAGE,
        PRODUCTION_TARGET_VOL,
        build_production_portfolio,
    )

    res = build_production_portfolio(synthetic_production_rets)
    assert res["allocation"] == "custom"
    assert res["target_vol"] == PRODUCTION_TARGET_VOL
    assert res["max_leverage"] == PRODUCTION_MAX_LEVERAGE
    assert res["dd_cap_enabled"] is False


def test_production_helper_accepts_param_overrides(synthetic_production_rets):
    """Caller may override target_vol and max_leverage (e.g. for sensitivity sweeps)."""
    from strategies.combined_portfolio_v2 import build_production_portfolio

    res_default = build_production_portfolio(synthetic_production_rets)
    res_override = build_production_portfolio(
        synthetic_production_rets, target_vol=0.20, max_leverage=8.0
    )

    # Overrides must propagate into the result dict.
    assert res_override["target_vol"] == 0.20
    assert res_override["max_leverage"] == 8.0
    # Lower target_vol + lower leverage should yield lower annualized vol.
    assert res_override["annual_vol"] < res_default["annual_vol"]
