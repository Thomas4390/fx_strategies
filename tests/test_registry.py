"""Smoke test: every registered strategy backtests without error."""

import pytest
from strategies import REGISTRY
from framework.runner import StrategyRunner


@pytest.mark.parametrize("name", list(REGISTRY.keys()))
def test_strategy_backtests(name, raw, data):
    """Each strategy should construct, backtest, and return a Portfolio."""
    spec = REGISTRY[name]
    runner = StrategyRunner(spec, raw, data)
    pf, ind = runner.backtest()

    assert pf is not None
    assert pf.wrapper.shape_2d[0] > 0
    stats = pf.stats()
    assert stats is not None
