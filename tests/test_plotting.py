"""Test all plot functions return valid figures."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from framework.runner import StrategyRunner


def test_plot_monthly_heatmap(raw, data):
    from strategies.ou_mean_reversion import spec
    from framework.plotting import plot_monthly_heatmap

    runner = StrategyRunner(spec, raw, data)
    pf, _ = runner.backtest()
    fig = plot_monthly_heatmap(pf)
    assert isinstance(fig, go.Figure)


def test_plot_portfolio_summary(raw, data):
    from strategies.ou_mean_reversion import spec
    from framework.plotting import plot_portfolio_summary

    runner = StrategyRunner(spec, raw, data)
    pf, _ = runner.backtest()
    fig = plot_portfolio_summary(pf)
    assert isinstance(fig, go.Figure)


def test_plot_trade_analysis(raw, data):
    from strategies.ou_mean_reversion import spec
    from framework.plotting import plot_trade_analysis

    runner = StrategyRunner(spec, raw, data)
    pf, _ = runner.backtest()
    fig = plot_trade_analysis(pf)
    assert isinstance(fig, go.Figure)


def test_plot_cv_stability():
    from framework.plotting import plot_cv_stability

    idx = pd.MultiIndex.from_tuples(
        [(0, "train", 60, 2.0), (1, "train", 60, 2.0)],
        names=["split", "set", "lookback", "band_width"],
    )
    grid = pd.Series([1.5, 1.2], index=idx)
    fig = plot_cv_stability(grid)
    assert isinstance(fig, go.Figure)


def test_plot_partial_dependence():
    from framework.plotting import plot_partial_dependence

    idx = pd.MultiIndex.from_tuples(
        [
            (0, "train", 40, 2.0),
            (0, "train", 60, 2.0),
            (0, "train", 40, 2.5),
            (0, "train", 60, 2.5),
        ],
        names=["split", "set", "lookback", "band_width"],
    )
    grid = pd.Series([1.0, 1.5, 1.2, 1.8], index=idx)
    fig = plot_partial_dependence(
        grid, {"lookback": [40, 60], "band_width": [2.0, 2.5]}
    )
    assert isinstance(fig, go.Figure)


def test_plot_rolling_sharpe(raw, data):
    from strategies.ou_mean_reversion import spec
    from framework.plotting import plot_rolling_sharpe

    runner = StrategyRunner(spec, raw, data)
    pf, _ = runner.backtest()
    fig = plot_rolling_sharpe(pf, window=5)  # small window for test data
    assert isinstance(fig, go.Figure)


def test_plot_train_vs_test():
    from framework.plotting import plot_train_vs_test

    idx = pd.MultiIndex.from_tuples(
        [
            (0, "train", 60, 2.0),
            (0, "test", 60, 2.0),
            (0, "train", 40, 2.5),
            (0, "test", 40, 2.5),
        ],
        names=["split", "set", "lookback", "band_width"],
    )
    grid = pd.Series([1.5, 1.0, 1.2, 0.8], index=idx)
    fig = plot_train_vs_test(grid)
    assert isinstance(fig, go.Figure)
