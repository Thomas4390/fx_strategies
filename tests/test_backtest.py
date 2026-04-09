"""Test single backtest with prepare_fn produces valid results."""
import numpy as np
from framework.runner import StrategyRunner


def test_backtest_ou_mr(raw, data):
    """OU MR strategy with native VWAP+ADX should produce trades."""
    from strategies.ou_mean_reversion import spec

    runner = StrategyRunner(spec, raw, data)
    pf, ind = runner.backtest()

    assert pf.trades.count() > 0
    assert not np.isnan(pf.sharpe_ratio)
    assert hasattr(ind, "upper_band")
    assert hasattr(ind, "lower_band")
    assert hasattr(ind, "leverage")
