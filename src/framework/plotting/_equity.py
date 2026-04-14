"""Equity, return-distribution, drawdown, rolling-metric, and
CV-stability visualizations.

Extracted from ``_core`` to stay under the 800-line rule. Depends on
``_core`` for layout helpers (``_apply_title_layout``, ``make_fullscreen``)
and on ``_helpers`` for ``_pick_first_column``.
"""

from __future__ import annotations

import calendar

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import vectorbtpro as vbt
from plotly.subplots import make_subplots

from ._core import _apply_title_layout, make_fullscreen
from ._helpers import _pick_first_column


# ═══════════════════════════════════════════════════════════════════════
# BEST-COMBINATION ANALYSIS PLOTS
# ═══════════════════════════════════════════════════════════════════════


def plot_equity_top_n(
    pf_sweep: vbt.Portfolio,
    n: int = 5,
    title: str = "Top Parameter Combos — Equity",
) -> go.Figure:
    """Overlay equity curves of the top N combos by Sharpe."""
    sharpes = pf_sweep.sharpe_ratio
    top_idx = sharpes.nlargest(n).index
    top_pf = pf_sweep[top_idx]
    fig = top_pf.value.vbt.plot()
    fig.update_traces(
        hovertemplate=(
            "<b>Portfolio Value</b>: %{y:,.2f}"
            "<br>%{x|%Y-%m-%d}<extra>%{fullData.name}</extra>"
        ),
    )
    fig.update_layout(title=title, height=500)
    return fig


def plot_cv_stability(
    grid_perf: pd.Series,
    title: str = "CV Stability — Sharpe per Fold",
) -> go.Figure:
    """Bar chart of best-combo Sharpe per CV fold."""
    if "split" not in grid_perf.index.names:
        fig = go.Figure()
        fig.add_annotation(text="No CV splits available", showarrow=False)
        return fig

    train_mask = grid_perf.index.get_level_values("set").isin(
        ["train", "set_0", 0]
    )
    train = grid_perf[train_mask]
    sweep_levels = [
        n for n in train.index.names if n not in ("split", "set")
    ]

    if sweep_levels:
        best_combo = train.groupby(sweep_levels).mean().idxmax()
        if isinstance(best_combo, tuple):
            mask = pd.Series(True, index=train.index)
            for level, val in zip(sweep_levels, best_combo):
                mask &= train.index.get_level_values(level) == val
            fold_sharpes = train[mask]
        else:
            fold_sharpes = train.xs(best_combo, level=sweep_levels[0])
    else:
        fold_sharpes = train

    split_vals = fold_sharpes.index.get_level_values("split")
    fig = go.Figure(
        data=go.Bar(
            x=[f"Fold {s}" for s in split_vals], y=fold_sharpes.values,
            hovertemplate=(
                "<b>Sharpe Ratio</b>: %{y:.2f}"
                "<br>%{x}<extra></extra>"
            ),
        )
    )
    fig.add_hline(
        y=fold_sharpes.mean(),
        line_dash="dash",
        line_color="red",
        annotation_text=f"Mean: {fold_sharpes.mean():.2f}",
    )
    fig.update_layout(title=title, height=400, yaxis_title="Sharpe Ratio")
    return fig


# ═══════════════════════════════════════════════════════════════════════
# ROLLING RISK-ADJUSTED METRICS
# ═══════════════════════════════════════════════════════════════════════


def plot_rolling_sharpe(
    pf: vbt.Portfolio,
    window: int = 252,
    title: str = "Rolling 1-Year Risk-Adjusted Metrics",
    y_range: tuple[float, float] = (-4.0, 6.0),
) -> go.Figure:
    """Rolling Sharpe + Sortino + Calmar on daily-resampled portfolio.

    All three metrics share the same y-axis (annualized risk-adjusted
    reward). Values are hard-clipped into ``y_range`` to avoid extreme
    spikes when the rolling denominator (downside deviation, max
    drawdown) approaches zero — e.g. a low-volatility window with very
    few losing days inflates Sortino towards infinity.

    Sortino returns NaN when downside deviation is below a small
    epsilon (window had no meaningful losses) and NaN gaps are left
    visible so the reader can tell when the metric is undefined.
    """
    pf_daily = pf.resample("1D")
    rets = pf_daily.returns
    if isinstance(rets, pd.DataFrame):
        rets = rets.mean(axis=1)
    rets = rets.dropna()
    sqrt_n = np.sqrt(252)
    eps = 1e-6

    def _sharpe(x: np.ndarray) -> float:
        s = x.std()
        if s < eps:
            return np.nan
        return float(x.mean() / s * sqrt_n)

    def _sortino(x: np.ndarray) -> float:
        downside = x[x < 0]
        if downside.size < 5:
            return np.nan
        dd = downside.std()
        if dd < eps:
            return np.nan
        return float(x.mean() / dd * sqrt_n)

    def _calmar(x: np.ndarray) -> float:
        if len(x) == 0:
            return np.nan
        cum = (1 + pd.Series(x)).cumprod()
        peak = cum.cummax()
        dd = (cum / peak - 1).min()
        if dd > -eps:
            return np.nan
        ann_ret = cum.iloc[-1] ** (252 / len(x)) - 1
        return float(ann_ret / abs(dd))

    roll_sharpe = rets.rolling(window).apply(_sharpe, raw=True)
    roll_sortino = rets.rolling(window).apply(_sortino, raw=True)
    roll_calmar = rets.rolling(window).apply(_calmar, raw=True)

    # Clip outliers so a single blow-up does not flatten the rest of
    # the curve. NaN values are kept (plotly draws them as gaps).
    lo, hi = y_range
    roll_sharpe = roll_sharpe.clip(lo, hi)
    roll_sortino = roll_sortino.clip(lo, hi)
    roll_calmar = roll_calmar.clip(lo, hi)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=roll_sharpe.index, y=roll_sharpe.values,
            mode="lines", name="Rolling Sharpe",
            line=dict(color="#636EFA", width=2),
            connectgaps=False,
            hovertemplate=(
                "<b>Sharpe Ratio</b>: %{y:.2f}"
                "<br>%{x|%Y-%m-%d}<extra></extra>"
            ),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=roll_sortino.index, y=roll_sortino.values,
            mode="lines", name="Rolling Sortino",
            line=dict(color="#00CC96", width=2),
            connectgaps=False,
            hovertemplate=(
                "<b>Sortino Ratio</b>: %{y:.2f}"
                "<br>%{x|%Y-%m-%d}<extra></extra>"
            ),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=roll_calmar.index, y=roll_calmar.values,
            mode="lines", name="Rolling Calmar",
            line=dict(color="#AB63FA", width=2, dash="dot"),
            connectgaps=False,
            hovertemplate=(
                "<b>Calmar Ratio</b>: %{y:.2f}"
                "<br>%{x|%Y-%m-%d}<extra></extra>"
            ),
        )
    )
    fig.add_hline(y=0, line_color="gray", line_width=1)
    fig.add_hline(
        y=1, line_dash="dot", line_color="#00CC96", opacity=0.5,
        annotation_text="= 1",
    )
    fig.update_yaxes(
        title_text="Annualized risk-adjusted return",
        range=list(y_range),
    )
    fig.update_xaxes(title_text="Date")
    _apply_title_layout(
        fig, title,
        subtitle=(
            f"Rolling window: {window} trading days  ·  "
            f"clipped to [{lo}, {hi}]"
        ),
    )
    make_fullscreen(fig)
    return fig


# ═══════════════════════════════════════════════════════════════════════
# PARAMETER SENSITIVITY / TRAIN-TEST COMPARISON
# ═══════════════════════════════════════════════════════════════════════


def plot_partial_dependence(
    sweep_sharpes: pd.Series,
    param_grid: dict[str, list],
    title: str = "Parameter Sensitivity",
    metric_name: str = "Sharpe Ratio",
) -> go.Figure:
    """Marginal mean of *metric_name* for each swept parameter."""
    params = list(param_grid.keys())
    n_params = len(params)
    if n_params == 0:
        return go.Figure()

    fig = make_subplots(rows=1, cols=n_params, subplot_titles=params)
    idx_names = list(sweep_sharpes.index.names)

    for i, param in enumerate(params):
        matched = param
        for name in idx_names:
            if name and name.endswith(param):
                matched = name
                break
        if matched not in idx_names:
            continue
        marginal = sweep_sharpes.groupby(matched).mean()
        fig.add_trace(
            go.Bar(
                x=[str(v) for v in marginal.index],
                y=marginal.values,
                name=param,
                hovertemplate=(
                    f"<b>{metric_name}</b>: %{{y:.2f}}"
                    f"<br>{param}=%{{x}}<extra></extra>"
                ),
            ),
            row=1,
            col=i + 1,
        )

    fig.update_layout(title=title, height=350, showlegend=False)
    return fig


def plot_train_vs_test(
    grid_perf: pd.Series,
    title: str = "Train vs Test Sharpe (Overfitting Check)",
    metric_name: str = "Sharpe Ratio",
) -> go.Figure:
    """Scatter: train *metric_name* (x) vs test *metric_name* (y) per combo."""
    if "set" not in grid_perf.index.names:
        return go.Figure()

    train_mask = grid_perf.index.get_level_values("set").isin(
        ["train", "set_0", 0]
    )
    test_mask = grid_perf.index.get_level_values("set").isin(
        ["test", "set_1", 1]
    )
    train = grid_perf[train_mask]
    test = grid_perf[test_mask]

    sweep_levels = [
        n for n in train.index.names if n not in ("split", "set")
    ]
    if not sweep_levels:
        return go.Figure()

    train_avg = train.groupby(sweep_levels).mean()
    test_avg = test.groupby(sweep_levels).mean()
    common = train_avg.index.intersection(test_avg.index)

    if len(common) == 0:
        return go.Figure()

    fig = go.Figure(
        data=go.Scatter(
            x=train_avg.loc[common].values,
            y=test_avg.loc[common].values,
            mode="markers",
            text=[str(c) for c in common],
            name="Combos",
            hovertemplate=(
                f"<b>Train {metric_name}</b>: %{{x:.2f}}"
                f"<br><b>Test {metric_name}</b>: %{{y:.2f}}"
                "<br>params=%{text}<extra></extra>"
            ),
        )
    )
    max_val = max(train_avg.max(), test_avg.max(), 0) * 1.1
    min_val = min(train_avg.min(), test_avg.min(), 0) * 0.9
    fig.add_trace(
        go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode="lines",
            line=dict(dash="dash", color="gray"),
            name="y=x",
            hoverinfo="skip",
        )
    )
    fig.update_layout(
        title=title,
        height=500,
        xaxis_title=f"Train {metric_name}",
        yaxis_title=f"Test {metric_name}",
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════
# RETURNS DISTRIBUTION ANALYTICS
# ═══════════════════════════════════════════════════════════════════════


def plot_returns_distribution(
    pf: vbt.Portfolio,
    title: str = "Returns Analysis",
    height: int | None = None,
) -> go.Figure:
    """4-panel returns analysis.

    Layout:
    - (1,1) Daily returns per calendar day as a green/red bar chart.
    - (1,2) Monthly returns boxplot (distribution per calendar month).
    - (2,1) Cumulative returns curve.
    - (2,2) Returns ECDF.
    """
    pf_daily = pf.resample("1D")
    daily_rets = pf_daily.returns.dropna()
    if isinstance(daily_rets, pd.DataFrame):
        daily_rets = daily_rets.iloc[:, 0]

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Avg Strategy Return by Weekday",
            "Monthly Returns Boxplot",
            "Cumulative Returns",
            "Returns ECDF",
        ),
        vertical_spacing=0.14,
        horizontal_spacing=0.1,
    )

    # (1,1) Mean daily return aggregated by day of week
    dow_order = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    dow_index = daily_rets.index.dayofweek  # 0=Mon .. 6=Sun
    dow_group = pd.Series(daily_rets.values, index=dow_index).groupby(level=0)
    dow_mean = dow_group.mean().reindex(range(7))
    dow_std = dow_group.std().reindex(range(7))
    dow_count = dow_group.count().reindex(range(7)).fillna(0).astype(int)
    colors = np.where(dow_mean.fillna(0).values >= 0, "#00CC96", "#EF553B")
    hover = [
        f"{lbl}: mean={m:.3%}<br>std={s:.3%}<br>n={c}"
        if not pd.isna(m) else f"{lbl}: no data"
        for lbl, m, s, c in zip(dow_order, dow_mean.values, dow_std.values, dow_count.values)
    ]
    fig.add_trace(
        go.Bar(
            x=dow_order,
            y=dow_mean.fillna(0).values,
            error_y=dict(
                type="data",
                array=dow_std.fillna(0).values,
                visible=True,
                color="rgba(90,90,90,0.4)",
                thickness=1.2,
            ),
            marker_color=colors,
            name="Avg daily return",
            hovertemplate="%{hovertext}<extra></extra>",
            hovertext=hover,
        ),
        row=1, col=1,
    )
    fig.add_hline(y=0, line_color="gray", line_width=1, row=1, col=1)
    fig.update_yaxes(tickformat=".3%", row=1, col=1, title_text="Mean return")

    # (1,2) Monthly returns boxplot
    mo_rets = pf_daily.resample("ME").returns
    if isinstance(mo_rets, pd.DataFrame):
        mo_rets = mo_rets.mean(axis=1)
    mo_rets = mo_rets.dropna()
    months = mo_rets.index.month
    for m in range(1, 13):
        m_data = mo_rets[months == m].values
        if len(m_data) > 0:
            fig.add_trace(
                go.Box(
                    y=m_data, name=calendar.month_abbr[m],
                    marker_color="#636EFA", showlegend=False,
                ),
                row=1, col=2,
            )
    fig.update_yaxes(tickformat=".1%", row=1, col=2)

    # (2,1) Cumulative returns
    cum_rets = (1 + daily_rets).cumprod() - 1
    fig.add_trace(
        go.Scatter(
            x=cum_rets.index, y=cum_rets.values,
            mode="lines", name="Cumulative",
            line=dict(color="#00CC96", width=2),
            hovertemplate=(
                "<b>Cumulative Return</b>: %{y:.2%}"
                "<br>%{x|%Y-%m-%d}<extra></extra>"
            ),
        ),
        row=2, col=1,
    )
    fig.update_yaxes(tickformat=".1%", row=2, col=1)

    # (2,2) ECDF
    sorted_rets = np.sort(daily_rets.values)
    ecdf_y = np.arange(1, len(sorted_rets) + 1) / len(sorted_rets)
    fig.add_trace(
        go.Scatter(
            x=sorted_rets, y=ecdf_y,
            mode="lines", name="ECDF",
            line=dict(color="#EF553B", width=2),
            hovertemplate=(
                "<b>Daily Return</b>: %{x:.2%}"
                "<br><b>Cumulative Probability</b>: %{y:.2f}<extra></extra>"
            ),
        ),
        row=2, col=2,
    )
    fig.update_xaxes(tickformat=".2%", row=2, col=2)

    fig.update_layout(showlegend=False)
    _apply_title_layout(fig, title)
    if height is not None:
        fig.layout.height = height
    else:
        make_fullscreen(fig)
    for ann in fig.layout.annotations:
        if ann.yref == "paper":
            ann.y = min(ann.y, 0.94)
    return fig


# ═══════════════════════════════════════════════════════════════════════
# DRAWDOWN ANALYSIS
# ═══════════════════════════════════════════════════════════════════════


def plot_drawdown_analysis(
    pf: vbt.Portfolio,
    title: str = "Drawdown Analysis",
    top_n: int = 5,
    height: int | None = None,
) -> go.Figure:
    """Underwater plot with top N drawdowns highlighted.

    Multi-column portfolios fall back to the first column so VBT's
    ``drawdowns``/``underwater`` subplots render correctly.
    """
    pf_single, col_label = _pick_first_column(pf)
    pf_daily = pf_single.resample("1D")
    try:
        fig = pf_daily.plot(subplots=["drawdowns", "underwater"])
    except Exception:
        # Fallback: manual underwater plot
        rets = pf_daily.returns
        if isinstance(rets, pd.DataFrame):
            rets = rets.mean(axis=1)
        cum = (1 + rets).cumprod()
        dd = cum / cum.cummax() - 1
        fig = go.Figure(
            go.Scatter(
                x=dd.index, y=dd.values,
                fill="tozeroy", mode="lines",
                line=dict(color="#EF553B"),
                name="Drawdown",
                hovertemplate=(
                    "<b>Max Drawdown</b>: %{y:.2%}"
                    "<br>%{x|%Y-%m-%d}<extra></extra>"
                ),
            )
        )
    if col_label:
        title = f"{title} — {col_label}"
    _apply_title_layout(fig, title)
    if height is not None:
        fig.layout.height = height
    else:
        make_fullscreen(fig)
    return fig


# ═══════════════════════════════════════════════════════════════════════
# MULTI-STRATEGY COMPARISON
# ═══════════════════════════════════════════════════════════════════════


def plot_multi_strategy_equity(
    portfolios: dict[str, vbt.Portfolio],
    title: str = "Strategy Comparison — Equity Curves",
    height: int = 500,
    normalize: bool = True,
) -> go.Figure:
    """Overlay equity curves from multiple portfolios.

    If normalize=True, all curves start at 1.0 for fair comparison.
    """
    colors = [
        "#636EFA", "#EF553B", "#00CC96", "#AB63FA",
        "#FFA15A", "#19D3F3", "#FF6692", "#B6E880",
    ]
    fig = go.Figure()

    metric_label = "Normalized Equity" if normalize else "Portfolio Value"
    for i, (name, pf) in enumerate(portfolios.items()):
        pf_daily = pf.resample("1D")
        values = pf_daily.value
        if normalize:
            values = values / values.iloc[0]
        fig.add_trace(
            go.Scatter(
                x=values.index, y=values.values,
                mode="lines", name=name,
                line=dict(color=colors[i % len(colors)], width=2),
                hovertemplate=(
                    f"<b>{name}</b><br>"
                    f"<b>{metric_label}</b>: %{{y:,.2f}}"
                    "<br>%{x|%Y-%m-%d}<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title=title, height=height,
        yaxis_title="Normalized Value" if normalize else "Portfolio Value",
        hovermode="closest",
    )
    return fig
