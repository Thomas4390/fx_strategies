"""Regression lock on the sizing convention of each sleeve.

This file exists because of a defect that lived from 2026-05-05 to 2026-07-26
without a single test going red: ``combined_portfolio`` multiplied
``TS_Momentum_3p`` by ``MT5_LEV_AVG = 12`` even though that sleeve already
carries ``vol_target_leverage(target=0.10, cap=3.0)`` — and its MQL5port
(``FxSleeveTSMomentum.mqh:234``) sizes on that same per-pair leverage, never on
``CRiskManager::GlobalLeverage()``. The scaling stacked a second leverage layer
with no counterpart in the deployed EA.

Why nothing caught it — and what each test below locks:

1. **The aggregate metrics could not move.** The combined portfolio is capped by
   a global vol-targeting layer (``target_vol=0.75``): inflate a sleeve and the
   global leverage drops by the same factor, so realized vol, max drawdown and
   Sharpe stay pinned to the target. Watching them proves nothing about sleeve
   sizing. ``test_vol_targeted_sleeves_realize_their_target`` watches each
   sleeve *before* aggregation instead — the symptom was a sleeve targeting 10%
   vol and realizing 67%.

2. **The weights kept reading 80/10/10 the whole time.** What moved was the
   *risk* budget: MR Macro fell to 39.4% of portfolio risk while a 10%-weight
   sleeve took 55.1%. ``test_production_risk_budget_follows_weights`` asserts no
   sleeve's risk share runs away from its nominal weight, which is the
   proportionality a nominal weight is supposed to express.

Both run off the disk cache, so they are cheap on a warm cache and ~2 min after
a ``_SLEEVES_VERSION`` bump.
"""

from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest

# A sleeve that vol-targets internally must land near its target once realized.
# The factor is deliberately loose — it is there to catch a stacked leverage
# layer (which lands 6-12x off), not to pin the sleeve's behaviour.
_VOL_TARGET_TOLERANCE = 3.0

# Below this, a sleeve has effectively stopped trading and the vol check would
# pass vacuously.
_VOL_FLOOR = 0.005

# A sleeve may carry more risk than its nominal weight — sleeves do not target
# the same vol — but not by an arbitrary factor. Same spirit as the tolerance
# above: catch a stacked leverage layer (the defect ran at 5.5x), not the
# ordinary spread between a 55%-vol-target sleeve and a 7%-vol one.
_RISK_SHARE_TOLERANCE = 3.0


@pytest.fixture(scope="module")
def strategy_returns() -> dict[str, pd.Series]:
    """Component daily returns, served from the parquet cache when fresh."""
    from strategies.combined_portfolio import get_strategy_daily_returns

    try:
        return get_strategy_daily_returns()
    except FileNotFoundError as exc:  # pragma: no cover - depends on local data
        pytest.skip(f"FX/gold data unavailable: {exc}")


def _default_target_vol(func, param: str = "target_vol") -> float:
    """Read the sleeve's own vol target from its signature.

    Reading the default rather than hardcoding it keeps this test honest when
    a target is retuned: the assertion follows the code instead of freezing a
    number that silently stops describing it.
    """
    default = inspect.signature(func).parameters[param].default
    assert default is not None, f"{func.__name__} has no default {param}"
    return float(default)


def _annualized_vol(rets: pd.Series) -> float:
    return float(rets.dropna().std(ddof=1) * np.sqrt(252))


def _vol_targeted_sleeves() -> list[tuple[str, float]]:
    from strategies.daily_momentum import backtest_ts_momentum_portfolio
    from strategies.gold_momentum import pipeline as gold_pipeline

    ts_target = _default_target_vol(backtest_ts_momentum_portfolio)
    return [
        ("TS_Momentum_3p", ts_target),
        ("TS_Momentum_RSI", ts_target),
        ("Gold_Momentum", _default_target_vol(gold_pipeline)),
    ]


@pytest.mark.parametrize("sleeve,target", _vol_targeted_sleeves())
def test_vol_targeted_sleeves_realize_their_target(
    strategy_returns: dict[str, pd.Series], sleeve: str, target: float
) -> None:
    """A sleeve carrying its own vol target must realize a vol of that order.

    A sleeve wearing two leverage layers blows through this by construction:
    the stacked ``× MT5_LEV_AVG = 12`` put TS_Momentum_3p at 67% realized
    against a 10% target.
    """
    assert sleeve in strategy_returns, f"{sleeve} missing from the sleeve set"
    realized = _annualized_vol(strategy_returns[sleeve])

    assert realized > _VOL_FLOOR, (
        f"{sleeve}: realized vol {realized:.2%} is ~zero — the sleeve is not "
        "trading, so the target check below would pass vacuously"
    )
    assert realized <= target * _VOL_TARGET_TOLERANCE, (
        f"{sleeve}: realized vol {realized:.2%} exceeds {_VOL_TARGET_TOLERANCE}x "
        f"its own {target:.0%} target. A sleeve that vol-targets internally must "
        "not be re-scaled by the portfolio layer — check for a stacked leverage "
        "factor in combined_portfolio._compute_strategy_daily_returns()."
    )


def test_production_risk_budget_follows_weights(
    strategy_returns: dict[str, pd.Series],
) -> None:
    """No sleeve may carry a risk share disproportionate to its weight.

    Risk contribution is ``w_i * (Σw)_i / σ_p``, so it tracks ``w_i * σ_i``:
    re-scaling one sleeve moves the risk budget while every nominal weight — and
    every table built from them — keeps reading 80/10/10.

    The assertion is a *ratio* rather than « the heaviest sleeve dominates the
    risk »: since 2026-07-27 the momentum sleeve weighs 20 % and carries ~53 %
    of the risk **by design** — it vol-targets at 55 % against ~7 % for MR
    Macro, and the sweep that sized it published that contribution before the
    allocation was acted (``reports/research/momentum_weights_sweep_2026H2.csv``,
    48 % at w=0,20 under the cycle's execution conventions). The stacked-leverage
    defect still trips the ratio: 10 % of the weight for 55,1 % of the risk is
    5,5x.
    """
    from strategies.combined_portfolio_v2 import PRODUCTION_WEIGHTS

    rets = pd.DataFrame(
        {k: strategy_returns[k].dropna() for k in PRODUCTION_WEIGHTS}
    ).dropna()
    weights = np.array([PRODUCTION_WEIGHTS[c] for c in rets.columns])

    cov = rets.cov().values * 252.0
    port_vol = float(np.sqrt(weights @ cov @ weights))
    assert port_vol > 0.0, "degenerate covariance — cannot attribute risk"

    contributions = weights * ((cov @ weights) / port_vol) / port_vol
    shares = dict(zip(rets.columns, contributions))

    detail = ", ".join(
        f"{c}: w={PRODUCTION_WEIGHTS[c]:.0%} risk={s:.1%}" for c, s in shares.items()
    )
    for sleeve, share in shares.items():
        weight = PRODUCTION_WEIGHTS[sleeve]
        assert share <= weight * _RISK_SHARE_TOLERANCE, (
            f"{sleeve} weighs {weight:.0%} but carries {share:.1%} of portfolio "
            f"risk, more than {_RISK_SHARE_TOLERANCE:.0f}x its nominal share — the "
            f"allocation no longer describes the risk taken ({detail}). Check for a "
            f"stacked leverage factor in "
            f"combined_portfolio._compute_strategy_daily_returns()."
        )
