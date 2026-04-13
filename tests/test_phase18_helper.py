"""Tests for the Phase 18 one-liner helper in combined_portfolio_v2.

``build_phase18_portfolio`` is a convenience wrapper around
``build_combined_portfolio_v2`` that pins the recipe to the final
Phase 18 recommended configuration. These tests verify the wrapper
is bit-identical to a manual call with the same arguments, so the
report and reproduction scripts can treat it as a canonical entry
point without worrying about drift.
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
def synthetic_phase18_rets() -> dict[str, pd.Series]:
    """Synthetic 3-sleeve return set matching Phase 18 keys.

    Uses the same generator as ``test_combined_portfolio_v2`` so the
    fixture is reproducible; only the strategy names differ to match
    the Phase 18 sleeves.
    """
    rng = np.random.default_rng(20260413)
    n = 2500
    idx = pd.bdate_range("2016-01-04", periods=n, freq="B")
    return {
        "MR_Macro": pd.Series(
            rng.normal(0.0004, 0.006, n), index=idx, name="MR_Macro"
        ),
        "TS_Momentum_3p": pd.Series(
            rng.normal(0.0003, 0.009, n), index=idx, name="TS_Momentum_3p"
        ),
        "RSI_Daily_4p": pd.Series(
            rng.normal(0.0002, 0.005, n), index=idx, name="RSI_Daily_4p"
        ),
    }


def test_phase18_helper_matches_manual_call(synthetic_phase18_rets):
    """build_phase18_portfolio must equal a manual build_combined_portfolio_v2 call."""
    from strategies.combined_portfolio_v2 import (
        PHASE18_MAX_LEVERAGE,
        PHASE18_TARGET_VOL,
        PHASE18_WEIGHTS,
        build_combined_portfolio_v2,
        build_phase18_portfolio,
    )

    res_helper = build_phase18_portfolio(synthetic_phase18_rets)
    res_manual = build_combined_portfolio_v2(
        synthetic_phase18_rets,
        allocation="custom",
        custom_weights=PHASE18_WEIGHTS,
        target_vol=PHASE18_TARGET_VOL,
        max_leverage=PHASE18_MAX_LEVERAGE,
        dd_cap_enabled=False,
    )

    np.testing.assert_allclose(
        res_helper["portfolio_returns"].values,
        res_manual["portfolio_returns"].values,
        rtol=0,
        atol=0,
        err_msg="Phase 18 helper diverges from the manual call",
    )
    assert abs(res_helper["sharpe"] - res_manual["sharpe"]) < 1e-12
    assert abs(res_helper["annual_return"] - res_manual["annual_return"]) < 1e-12
    assert abs(res_helper["max_drawdown"] - res_manual["max_drawdown"]) < 1e-12


def test_phase18_helper_uses_canonical_weights(synthetic_phase18_rets):
    """The weights_ts returned by the helper must sum to 1 with the Phase 18 ratios."""
    from strategies.combined_portfolio_v2 import (
        PHASE18_WEIGHTS,
        build_phase18_portfolio,
    )

    res = build_phase18_portfolio(synthetic_phase18_rets)
    weights_ts = res["weights_ts"]

    assert list(weights_ts.columns) == list(PHASE18_WEIGHTS.keys())
    # Every row sums to 1 (renormalization guard in compute_regime_adaptive_weights
    # is not triggered here because allocation='custom', but the static
    # branch of _compute_weights_ts normalizes too).
    np.testing.assert_allclose(
        weights_ts.sum(axis=1).values, 1.0, rtol=1e-12, atol=0
    )
    # And each column's time-series mean matches the Phase 18 target share
    # exactly (static allocation → constant weights).
    for strat, expected in PHASE18_WEIGHTS.items():
        assert abs(weights_ts[strat].mean() - expected) < 1e-12


def test_phase18_helper_pins_custom_config_params(synthetic_phase18_rets):
    """Helper must record the canonical target_vol / max_leverage / dd_cap=False."""
    from strategies.combined_portfolio_v2 import (
        PHASE18_MAX_LEVERAGE,
        PHASE18_TARGET_VOL,
        build_phase18_portfolio,
    )

    res = build_phase18_portfolio(synthetic_phase18_rets)
    assert res["allocation"] == "custom"
    assert res["target_vol"] == PHASE18_TARGET_VOL
    assert res["max_leverage"] == PHASE18_MAX_LEVERAGE
    assert res["dd_cap_enabled"] is False


def test_phase18_helper_accepts_param_overrides(synthetic_phase18_rets):
    """Caller may override target_vol and max_leverage (e.g. for sensitivity sweeps)."""
    from strategies.combined_portfolio_v2 import build_phase18_portfolio

    res_default = build_phase18_portfolio(synthetic_phase18_rets)
    res_override = build_phase18_portfolio(
        synthetic_phase18_rets, target_vol=0.20, max_leverage=8.0
    )

    # Overrides must propagate into the result dict.
    assert res_override["target_vol"] == 0.20
    assert res_override["max_leverage"] == 8.0
    # Lower target_vol + lower leverage should yield lower annualized vol.
    assert res_override["annual_vol"] < res_default["annual_vol"]
