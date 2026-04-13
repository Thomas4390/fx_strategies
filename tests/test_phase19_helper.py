"""Tests for the Phase 19 *balanced* and *aggressive* helpers.

``build_phase19_balanced_portfolio`` and ``build_phase19_aggressive_portfolio``
are the canonical one-liners for the two Phase 19 recommended configurations
identified by the refined leverage sweep
(``docs/research/phase19_2026-04-13_refined_leverage.md``). These tests
mirror the ``test_phase18_helper`` contract to guarantee each wrapper is
bit-identical to a manual ``build_combined_portfolio_v2`` call and to
record the pinned parameters.
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
def synthetic_phase19_rets() -> dict[str, pd.Series]:
    """Reuse the Phase 18 trio fixture (same keys and RNG seed)."""
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


# ═══════════════════════════════════════════════════════════════════════
# BALANCED helper
# ═══════════════════════════════════════════════════════════════════════


def test_phase19_balanced_matches_manual_call(synthetic_phase19_rets):
    """Helper must equal a manual build_combined_portfolio_v2 call bit-identically."""
    from strategies.combined_portfolio_v2 import (
        PHASE19_BALANCED_DD_CAP_ENABLED,
        PHASE19_BALANCED_MAX_LEVERAGE,
        PHASE19_BALANCED_TARGET_VOL,
        PHASE19_BALANCED_WEIGHTS,
        build_combined_portfolio_v2,
        build_phase19_balanced_portfolio,
    )

    res_helper = build_phase19_balanced_portfolio(synthetic_phase19_rets)
    res_manual = build_combined_portfolio_v2(
        synthetic_phase19_rets,
        allocation="custom",
        custom_weights=PHASE19_BALANCED_WEIGHTS,
        target_vol=PHASE19_BALANCED_TARGET_VOL,
        max_leverage=PHASE19_BALANCED_MAX_LEVERAGE,
        dd_cap_enabled=PHASE19_BALANCED_DD_CAP_ENABLED,
    )

    np.testing.assert_allclose(
        res_helper["portfolio_returns"].values,
        res_manual["portfolio_returns"].values,
        rtol=0, atol=0,
        err_msg="Phase 19 balanced helper diverges from manual call",
    )
    assert abs(res_helper["sharpe"] - res_manual["sharpe"]) < 1e-12
    assert abs(res_helper["annual_return"] - res_manual["annual_return"]) < 1e-12
    assert abs(res_helper["max_drawdown"] - res_manual["max_drawdown"]) < 1e-12


def test_phase19_balanced_pins_canonical_params(synthetic_phase19_rets):
    """Helper must record the canonical tv=0.25 / ml=14 / dd=False triple."""
    from strategies.combined_portfolio_v2 import build_phase19_balanced_portfolio

    res = build_phase19_balanced_portfolio(synthetic_phase19_rets)
    assert res["allocation"] == "custom"
    assert res["target_vol"] == 0.25
    assert res["max_leverage"] == 14.0
    assert res["dd_cap_enabled"] is False


def test_phase19_balanced_weights_are_phase18(synthetic_phase19_rets):
    """Phase 19 balanced reuses exact Phase 18 sleeves/weights."""
    from strategies.combined_portfolio_v2 import (
        PHASE18_WEIGHTS,
        PHASE19_BALANCED_WEIGHTS,
    )

    assert PHASE19_BALANCED_WEIGHTS == PHASE18_WEIGHTS


# ═══════════════════════════════════════════════════════════════════════
# AGGRESSIVE helper
# ═══════════════════════════════════════════════════════════════════════


def test_phase19_aggressive_matches_manual_call(synthetic_phase19_rets):
    """Helper must equal a manual build_combined_portfolio_v2 call bit-identically."""
    from strategies.combined_portfolio_v2 import (
        PHASE19_AGGRESSIVE_DD_CAP_ENABLED,
        PHASE19_AGGRESSIVE_MAX_LEVERAGE,
        PHASE19_AGGRESSIVE_TARGET_VOL,
        PHASE19_AGGRESSIVE_WEIGHTS,
        build_combined_portfolio_v2,
        build_phase19_aggressive_portfolio,
    )

    res_helper = build_phase19_aggressive_portfolio(synthetic_phase19_rets)
    res_manual = build_combined_portfolio_v2(
        synthetic_phase19_rets,
        allocation="custom",
        custom_weights=PHASE19_AGGRESSIVE_WEIGHTS,
        target_vol=PHASE19_AGGRESSIVE_TARGET_VOL,
        max_leverage=PHASE19_AGGRESSIVE_MAX_LEVERAGE,
        dd_cap_enabled=PHASE19_AGGRESSIVE_DD_CAP_ENABLED,
    )

    np.testing.assert_allclose(
        res_helper["portfolio_returns"].values,
        res_manual["portfolio_returns"].values,
        rtol=0, atol=0,
        err_msg="Phase 19 aggressive helper diverges from manual call",
    )


def test_phase19_aggressive_pins_canonical_params(synthetic_phase19_rets):
    """Helper must record tv=0.35 / ml=18 / dd=False."""
    from strategies.combined_portfolio_v2 import build_phase19_aggressive_portfolio

    res = build_phase19_aggressive_portfolio(synthetic_phase19_rets)
    assert res["target_vol"] == 0.35
    assert res["max_leverage"] == 18.0
    assert res["dd_cap_enabled"] is False


# ═══════════════════════════════════════════════════════════════════════
# Cross-variant invariants
# ═══════════════════════════════════════════════════════════════════════


def test_phase19_aggressive_higher_vol_than_balanced(synthetic_phase19_rets):
    """Aggressive target_vol=0.35 should produce higher realized vol than balanced=0.25."""
    from strategies.combined_portfolio_v2 import (
        build_phase19_aggressive_portfolio,
        build_phase19_balanced_portfolio,
    )

    res_bal = build_phase19_balanced_portfolio(synthetic_phase19_rets)
    res_agg = build_phase19_aggressive_portfolio(synthetic_phase19_rets)

    assert res_agg["annual_vol"] > res_bal["annual_vol"], (
        f"Aggressive vol ({res_agg['annual_vol']:.4f}) should exceed "
        f"balanced vol ({res_bal['annual_vol']:.4f})"
    )
    # And — since both sit on the 0.966 Sharpe plateau — the Sharpe should
    # be within a tight band (we use 0.05 absolute to accommodate RNG
    # noise on the synthetic fixture rather than the real data's 0.001).
    assert abs(res_agg["sharpe"] - res_bal["sharpe"]) < 0.05


def test_phase19_override_params(synthetic_phase19_rets):
    """Caller may override target_vol / max_leverage for sensitivity sweeps."""
    from strategies.combined_portfolio_v2 import build_phase19_balanced_portfolio

    res_default = build_phase19_balanced_portfolio(synthetic_phase19_rets)
    res_override = build_phase19_balanced_portfolio(
        synthetic_phase19_rets, target_vol=0.18, max_leverage=6.0
    )

    assert res_override["target_vol"] == 0.18
    assert res_override["max_leverage"] == 6.0
    # Lower target_vol + lower max_leverage => lower annualized vol.
    assert res_override["annual_vol"] < res_default["annual_vol"]
