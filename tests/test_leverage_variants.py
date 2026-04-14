"""Tests for the conservative and aggressive portfolio helpers.

``build_conservative_portfolio`` and ``build_aggressive_portfolio`` are
the canonical one-liners for the two leverage variants of the
production trio. These tests mirror the ``test_production_portfolio``
contract to guarantee each wrapper is bit-identical to a manual
``build_combined_portfolio_v2`` call and to record the pinned parameters.
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
def synthetic_leverage_variants_rets() -> dict[str, pd.Series]:
    """Reuse the production trio fixture (same keys and RNG seed)."""
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


# ═══════════════════════════════════════════════════════════════════════
# CONSERVATIVE helper
# ═══════════════════════════════════════════════════════════════════════


def test_conservative_matches_manual_call(synthetic_leverage_variants_rets):
    """Helper must equal a manual build_combined_portfolio_v2 call bit-identically."""
    from strategies.combined_portfolio_v2 import (
        CONSERVATIVE_DD_CAP_ENABLED,
        CONSERVATIVE_MAX_LEVERAGE,
        CONSERVATIVE_TARGET_VOL,
        CONSERVATIVE_WEIGHTS,
        build_combined_portfolio_v2,
        build_conservative_portfolio,
    )

    res_helper = build_conservative_portfolio(synthetic_leverage_variants_rets)
    res_manual = build_combined_portfolio_v2(
        synthetic_leverage_variants_rets,
        allocation="custom",
        custom_weights=CONSERVATIVE_WEIGHTS,
        target_vol=CONSERVATIVE_TARGET_VOL,
        max_leverage=CONSERVATIVE_MAX_LEVERAGE,
        dd_cap_enabled=CONSERVATIVE_DD_CAP_ENABLED,
    )

    np.testing.assert_allclose(
        res_helper["portfolio_returns"].values,
        res_manual["portfolio_returns"].values,
        rtol=0,
        atol=0,
        err_msg="Conservative helper diverges from manual call",
    )
    assert abs(res_helper["sharpe"] - res_manual["sharpe"]) < 1e-12
    assert abs(res_helper["annual_return"] - res_manual["annual_return"]) < 1e-12
    assert abs(res_helper["max_drawdown"] - res_manual["max_drawdown"]) < 1e-12


def test_conservative_pins_canonical_params(synthetic_leverage_variants_rets):
    """Helper must record the canonical tv=0.25 / ml=14 / dd=False triple."""
    from strategies.combined_portfolio_v2 import build_conservative_portfolio

    res = build_conservative_portfolio(synthetic_leverage_variants_rets)
    assert res["allocation"] == "custom"
    assert res["target_vol"] == 0.25
    assert res["max_leverage"] == 14.0
    assert res["dd_cap_enabled"] is False


def test_conservative_weights_match_production(synthetic_leverage_variants_rets):
    """Conservative variant reuses the exact production sleeves/weights."""
    from strategies.combined_portfolio_v2 import (
        CONSERVATIVE_WEIGHTS,
        PRODUCTION_WEIGHTS,
    )

    assert CONSERVATIVE_WEIGHTS == PRODUCTION_WEIGHTS


# ═══════════════════════════════════════════════════════════════════════
# AGGRESSIVE helper
# ═══════════════════════════════════════════════════════════════════════


def test_aggressive_matches_manual_call(synthetic_leverage_variants_rets):
    """Helper must equal a manual build_combined_portfolio_v2 call bit-identically."""
    from strategies.combined_portfolio_v2 import (
        AGGRESSIVE_DD_CAP_ENABLED,
        AGGRESSIVE_MAX_LEVERAGE,
        AGGRESSIVE_TARGET_VOL,
        AGGRESSIVE_WEIGHTS,
        build_aggressive_portfolio,
        build_combined_portfolio_v2,
    )

    res_helper = build_aggressive_portfolio(synthetic_leverage_variants_rets)
    res_manual = build_combined_portfolio_v2(
        synthetic_leverage_variants_rets,
        allocation="custom",
        custom_weights=AGGRESSIVE_WEIGHTS,
        target_vol=AGGRESSIVE_TARGET_VOL,
        max_leverage=AGGRESSIVE_MAX_LEVERAGE,
        dd_cap_enabled=AGGRESSIVE_DD_CAP_ENABLED,
    )

    np.testing.assert_allclose(
        res_helper["portfolio_returns"].values,
        res_manual["portfolio_returns"].values,
        rtol=0,
        atol=0,
        err_msg="Aggressive helper diverges from manual call",
    )


def test_aggressive_pins_canonical_params(synthetic_leverage_variants_rets):
    """Helper must record tv=0.35 / ml=18 / dd=False."""
    from strategies.combined_portfolio_v2 import build_aggressive_portfolio

    res = build_aggressive_portfolio(synthetic_leverage_variants_rets)
    assert res["target_vol"] == 0.35
    assert res["max_leverage"] == 18.0
    assert res["dd_cap_enabled"] is False


# ═══════════════════════════════════════════════════════════════════════
# Cross-variant invariants
# ═══════════════════════════════════════════════════════════════════════


def test_aggressive_higher_vol_than_conservative(synthetic_leverage_variants_rets):
    """Aggressive target_vol=0.35 should produce higher realized vol than conservative=0.25."""
    from strategies.combined_portfolio_v2 import (
        build_aggressive_portfolio,
        build_conservative_portfolio,
    )

    res_bal = build_conservative_portfolio(synthetic_leverage_variants_rets)
    res_agg = build_aggressive_portfolio(synthetic_leverage_variants_rets)

    assert res_agg["annual_vol"] > res_bal["annual_vol"], (
        f"Aggressive vol ({res_agg['annual_vol']:.4f}) should exceed "
        f"conservative vol ({res_bal['annual_vol']:.4f})"
    )
    # Both variants sit on the same Sharpe plateau, so the Sharpe should
    # be within a tight band (we use 0.05 absolute to accommodate RNG
    # noise on the synthetic fixture rather than the real data's 0.001).
    assert abs(res_agg["sharpe"] - res_bal["sharpe"]) < 0.05


def test_conservative_override_params(synthetic_leverage_variants_rets):
    """Caller may override target_vol / max_leverage for sensitivity sweeps."""
    from strategies.combined_portfolio_v2 import build_conservative_portfolio

    res_default = build_conservative_portfolio(synthetic_leverage_variants_rets)
    res_override = build_conservative_portfolio(
        synthetic_leverage_variants_rets, target_vol=0.18, max_leverage=6.0
    )

    assert res_override["target_vol"] == 0.18
    assert res_override["max_leverage"] == 6.0
    # Lower target_vol + lower max_leverage => lower annualized vol.
    assert res_override["annual_vol"] < res_default["annual_vol"]
