"""Test all plot functions return valid figures.

Uses ``strategies.mr_turbo.pipeline`` as the portfolio source so the tests
are decoupled from the removed StrategyRunner / StrategySpec API.
"""

import pandas as pd
import plotly.graph_objects as go


def _make_pf(raw_and_data):
    """Build a small portfolio suitable for plot tests."""
    _, data = raw_and_data
    # Use the last 5 days to keep tests fast and independent of full history.
    data_mini = data.iloc[-6300:]
    from strategies.mr_turbo import pipeline

    pf, _ = pipeline(data_mini, bb_window=60, bb_alpha=4.0)
    return pf


def test_plot_monthly_heatmap(raw_and_data):
    from framework.plotting import plot_monthly_heatmap

    pf = _make_pf(raw_and_data)
    fig = plot_monthly_heatmap(pf)
    assert isinstance(fig, go.Figure)


def test_plot_portfolio_summary(raw_and_data):
    from framework.plotting import plot_portfolio_summary

    pf = _make_pf(raw_and_data)
    fig = plot_portfolio_summary(pf)
    assert isinstance(fig, go.Figure)


def test_plot_trade_analysis(raw_and_data):
    from framework.plotting import plot_trade_analysis

    pf = _make_pf(raw_and_data)
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


def test_plot_rolling_sharpe(raw_and_data):
    from framework.plotting import plot_rolling_sharpe

    pf = _make_pf(raw_and_data)
    fig = plot_rolling_sharpe(pf, window=5)
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
