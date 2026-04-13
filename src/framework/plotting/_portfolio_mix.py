"""Multi-strategy portfolio visualizations (weights, contributions, regime).

These plots are dedicated to aggregator portfolios where multiple sub-strategies
are combined via time-varying allocations — produced by
``strategies.combined_portfolio.build_combined_portfolio`` and
``strategies.combined_portfolio_v2.build_combined_portfolio_v2``.

Every function returns a ``plotly.graph_objects.Figure`` with the same layout
conventions as the other ``framework.plotting`` modules (fullscreen-friendly,
unified hover, explicit title). Designed to be passed to ``show_browser`` or
``save_fullscreen_html`` downstream.
"""
from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ._core import _apply_title_layout


_PALETTE: tuple[str, ...] = (
    "#636EFA", "#EF553B", "#00CC96", "#AB63FA",
    "#FFA15A", "#19D3F3", "#FF6692", "#B6E880",
)


def _color(i: int) -> str:
    return _PALETTE[i % len(_PALETTE)]


# ═══════════════════════════════════════════════════════════════════════
# 1. WEIGHTS OVER TIME — stacked area
# ═══════════════════════════════════════════════════════════════════════


def plot_weights_stacked_area(
    weights_ts: pd.DataFrame,
    title: str = "Strategy Weights Over Time",
    height: int = 500,
    normalize: bool = False,
) -> go.Figure:
    """Stacked-area chart of strategy weights through time.

    Parameters
    ----------
    weights_ts : DataFrame
        Time-varying allocation DataFrame (index=dates, columns=strategy names).
        Typically the ``weights_ts`` key produced by the combined-portfolio
        build functions.
    normalize : bool
        If True, each row is divided by its sum so the stack always reaches
        1.0 — useful when the raw weights carry a leverage factor that
        would otherwise make the stack drift above/below 1.
    """
    df = weights_ts.copy()
    if normalize:
        row_sum = df.sum(axis=1)
        df = df.div(row_sum.where(row_sum != 0, 1.0), axis=0)

    fig = go.Figure()
    for i, col in enumerate(df.columns):
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df[col].values,
                mode="lines",
                name=str(col),
                stackgroup="weights",
                line=dict(width=0.5, color=_color(i)),
                hovertemplate=(
                    f"<b>{col}</b><br>"
                    "weight: %{y:.2%}<br>%{x|%Y-%m-%d}<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title=title,
        height=height,
        yaxis=dict(title="Weight", tickformat=".0%"),
        hovermode="x unified",
        legend=dict(orientation="h", y=-0.15),
    )
    _apply_title_layout(fig, title)
    return fig


def plot_weights_rolling_mean(
    weights_ts: pd.DataFrame,
    window: int = 21,
    title: str | None = None,
    height: int = 450,
) -> go.Figure:
    """Rolling-mean smoothed line chart of weights (non-stacked overlay).

    Great for seeing where allocations trend over time without the visual
    noise of daily rebalancing jitter.
    """
    smoothed = weights_ts.rolling(window, min_periods=1).mean()
    title = title or f"Strategy Weights — {window}-day Rolling Mean"

    fig = go.Figure()
    for i, col in enumerate(smoothed.columns):
        fig.add_trace(
            go.Scatter(
                x=smoothed.index,
                y=smoothed[col].values,
                mode="lines",
                name=str(col),
                line=dict(width=2, color=_color(i)),
                hovertemplate=(
                    f"<b>{col}</b><br>"
                    "smoothed weight: %{y:.2%}<br>%{x|%Y-%m-%d}<extra></extra>"
                ),
            )
        )
    fig.update_layout(
        title=title,
        height=height,
        yaxis=dict(title="Weight", tickformat=".0%"),
        hovermode="x unified",
    )
    _apply_title_layout(fig, title)
    return fig


def plot_weights_distribution(
    weights_ts: pd.DataFrame,
    title: str = "Weights Distribution per Strategy",
    height: int = 400,
) -> go.Figure:
    """Box-plot summary of the distribution of each strategy's weight.

    Quick diagnostic to see which sleeves have the widest allocation range
    (e.g. risk-parity tends to compress MR Macro while leaving diversifiers
    volatile) and whether any sleeve hits zero or the cap frequently.
    """
    fig = go.Figure()
    for i, col in enumerate(weights_ts.columns):
        vals = weights_ts[col].dropna().values
        fig.add_trace(
            go.Box(
                y=vals,
                name=str(col),
                boxmean=True,
                marker_color=_color(i),
                hovertemplate=(
                    f"<b>{col}</b><br>weight: %{{y:.2%}}<extra></extra>"
                ),
            )
        )
    fig.update_layout(
        title=title,
        height=height,
        yaxis=dict(title="Weight", tickformat=".0%"),
        showlegend=False,
    )
    _apply_title_layout(fig, title)
    return fig


# ═══════════════════════════════════════════════════════════════════════
# 2. STRATEGY CONTRIBUTION TO PORTFOLIO P&L
# ═══════════════════════════════════════════════════════════════════════


def plot_strategy_contribution(
    component_returns: pd.DataFrame,
    weights_ts: pd.DataFrame,
    title: str = "Cumulative Contribution per Strategy",
    height: int = 500,
) -> go.Figure:
    """Stacked area of cumulative weighted contribution per strategy.

    For each bar ``contrib_i_t = weight_i_{t-1} × ret_i_t`` (causal), and the
    cumulative sum over time gives the stackable per-strategy contribution
    to the aggregated P&L. The sum across strategies at any point equals
    the combined portfolio cumulative return (net of leverage/DD cap).
    """
    aligned_w = weights_ts.reindex(
        index=component_returns.index, columns=component_returns.columns
    ).fillna(0.0)
    # Causal shift: weights at t-1 drive return at t.
    contrib = component_returns * aligned_w.shift(1).fillna(0.0)
    cum = contrib.cumsum()

    fig = go.Figure()
    for i, col in enumerate(cum.columns):
        fig.add_trace(
            go.Scatter(
                x=cum.index,
                y=cum[col].values,
                mode="lines",
                name=str(col),
                stackgroup="contrib",
                line=dict(width=0.5, color=_color(i)),
                hovertemplate=(
                    f"<b>{col}</b><br>"
                    "cum contribution: %{y:+.2%}<br>"
                    "%{x|%Y-%m-%d}<extra></extra>"
                ),
            )
        )
    fig.update_layout(
        title=title,
        height=height,
        yaxis=dict(title="Cumulative return contribution", tickformat=".1%"),
        hovermode="x unified",
        legend=dict(orientation="h", y=-0.15),
    )
    _apply_title_layout(fig, title)
    return fig


# ═══════════════════════════════════════════════════════════════════════
# 3. ROLLING CORRELATION
# ═══════════════════════════════════════════════════════════════════════


def plot_rolling_correlation_heatmap(
    component_returns: pd.DataFrame,
    title: str = "Pairwise Correlation (Full Period)",
    height: int = 500,
) -> go.Figure:
    """Static full-period pairwise correlation heatmap (quick diagnostic)."""
    corr = component_returns.corr()
    fig = go.Figure(
        data=go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.index,
            zmin=-1, zmax=1,
            colorscale="RdBu_r",
            text=corr.round(3).values,
            texttemplate="%{text}",
            hovertemplate="%{x} vs %{y}<br>ρ = %{z:.3f}<extra></extra>",
        )
    )
    fig.update_layout(title=title, height=height)
    _apply_title_layout(fig, title)
    return fig


def plot_rolling_correlation_pairs(
    component_returns: pd.DataFrame,
    window: int = 63,
    title: str | None = None,
    height: int = 500,
) -> go.Figure:
    """Rolling pairwise correlation of every strategy pair over ``window`` days.

    One line per pair. Typically 3 strategies ⇒ 3 pairs. Lets you see when
    orthogonality breaks down (e.g. during regime shifts).
    """
    title = title or f"Rolling Pairwise Correlation — {window}-day window"
    cols = list(component_returns.columns)
    pairs = [(cols[i], cols[j]) for i in range(len(cols)) for j in range(i + 1, len(cols))]

    fig = go.Figure()
    for i, (a, b) in enumerate(pairs):
        roll = component_returns[a].rolling(window, min_periods=window // 2).corr(component_returns[b])
        fig.add_trace(
            go.Scatter(
                x=roll.index,
                y=roll.values,
                mode="lines",
                name=f"{a} × {b}",
                line=dict(width=2, color=_color(i)),
                hovertemplate=(
                    f"<b>{a} × {b}</b><br>"
                    "ρ: %{y:+.3f}<br>%{x|%Y-%m-%d}<extra></extra>"
                ),
            )
        )
    fig.add_hline(y=0.0, line_dash="dash", line_color="gray", opacity=0.5)
    fig.update_layout(
        title=title,
        height=height,
        yaxis=dict(title="Rolling correlation", range=[-1, 1]),
        hovermode="x unified",
    )
    _apply_title_layout(fig, title)
    return fig


# ═══════════════════════════════════════════════════════════════════════
# 4. LEVERAGE & VOL TARGETING (v2 specific)
# ═══════════════════════════════════════════════════════════════════════


def plot_leverage_and_vol(
    leverage_ts: pd.Series,
    port_rets_base: pd.Series,
    target_vol: float,
    max_leverage: float,
    title: str = "Global Leverage & Realized Vol",
    height: int = 500,
) -> go.Figure:
    """Leverage multiplier over time with realized vol and target-vol reference.

    Top subplot: leverage series with max-leverage cap line.
    Bottom subplot: realized 21-day annualized vol with the target-vol line.
    """
    realized_vol = port_rets_base.rolling(21, min_periods=10).std() * np.sqrt(252)

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.5, 0.5],
        vertical_spacing=0.08,
        subplot_titles=("Leverage multiplier", "Realized (21d) vs target vol"),
    )

    fig.add_trace(
        go.Scatter(
            x=leverage_ts.index, y=leverage_ts.values,
            mode="lines", name="leverage",
            line=dict(color=_color(0), width=2),
            hovertemplate="lev: %{y:.2f}x<br>%{x|%Y-%m-%d}<extra></extra>",
        ),
        row=1, col=1,
    )
    fig.add_hline(
        y=max_leverage, line_dash="dash", line_color="red",
        annotation_text=f"max={max_leverage:.1f}x", annotation_position="top right",
        row=1, col=1,
    )
    fig.add_hline(y=1.0, line_dash="dot", line_color="gray", row=1, col=1)

    fig.add_trace(
        go.Scatter(
            x=realized_vol.index, y=realized_vol.values,
            mode="lines", name="realized vol (21d)",
            line=dict(color=_color(1), width=2),
            hovertemplate="vol: %{y:.2%}<br>%{x|%Y-%m-%d}<extra></extra>",
        ),
        row=2, col=1,
    )
    fig.add_hline(
        y=target_vol, line_dash="dash", line_color="green",
        annotation_text=f"target={target_vol:.1%}", annotation_position="top right",
        row=2, col=1,
    )

    fig.update_yaxes(title="Leverage", row=1, col=1)
    fig.update_yaxes(title="Annualized vol", tickformat=".0%", row=2, col=1)
    fig.update_layout(title=title, height=height, hovermode="x unified", showlegend=False)
    _apply_title_layout(fig, title)
    return fig


# ═══════════════════════════════════════════════════════════════════════
# 5. DD CAP ACTIVITY (v2 specific)
# ═══════════════════════════════════════════════════════════════════════


def plot_dd_cap_activity(
    port_rets_prelev: pd.Series,
    dd_scale_ts: pd.Series,
    title: str = "Drawdown Cap Activity",
    height: int = 500,
) -> go.Figure:
    """Pre-lev equity with drawdown shaded and the DD-cap scale overlay.

    Top subplot: pre-leverage equity (cumprod of 1 + port_rets_prelev) with
    running max and drawdown region shaded.
    Bottom subplot: ``dd_scale_ts`` (1.0 = full leverage, < 1 = de-levered).
    """
    equity = (1.0 + port_rets_prelev.fillna(0.0)).cumprod()
    running_max = equity.expanding().max()
    drawdown = equity / running_max - 1.0

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.6, 0.4],
        vertical_spacing=0.08,
        subplot_titles=("Pre-lev equity & drawdown", "DD cap scale (1.0 = full leverage)"),
    )

    fig.add_trace(
        go.Scatter(
            x=equity.index, y=equity.values,
            mode="lines", name="equity",
            line=dict(color=_color(0), width=2),
            hovertemplate="equity: %{y:.4f}<br>%{x|%Y-%m-%d}<extra></extra>",
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=running_max.index, y=running_max.values,
            mode="lines", name="running max",
            line=dict(color="gray", width=1, dash="dot"),
            hovertemplate="peak: %{y:.4f}<extra></extra>",
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=drawdown.index, y=drawdown.values,
            mode="lines", name="drawdown",
            line=dict(color=_color(1), width=1),
            fill="tozeroy",
            fillcolor="rgba(239, 85, 59, 0.2)",
            yaxis="y3",
            hovertemplate="DD: %{y:+.2%}<br>%{x|%Y-%m-%d}<extra></extra>",
        ),
        row=1, col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=dd_scale_ts.index, y=dd_scale_ts.values,
            mode="lines", name="dd scale",
            line=dict(color=_color(2), width=2),
            hovertemplate="scale: %{y:.2f}<br>%{x|%Y-%m-%d}<extra></extra>",
        ),
        row=2, col=1,
    )
    fig.add_hline(y=1.0, line_dash="dot", line_color="gray", row=2, col=1)

    fig.update_yaxes(title="Equity", row=1, col=1)
    fig.update_yaxes(title="DD cap scale", range=[0, 1.1], row=2, col=1)
    fig.update_layout(title=title, height=height, hovermode="x unified")
    _apply_title_layout(fig, title)
    return fig


# ═══════════════════════════════════════════════════════════════════════
# 6. REGIME OVERLAY (v2 specific)
# ═══════════════════════════════════════════════════════════════════════


def plot_regime_overlay(
    port_rets: pd.Series,
    vol_regime_ts: pd.Series,
    title: str = "Portfolio Equity × Vol Regime",
    height: int = 500,
) -> go.Figure:
    """Equity curve with vol regime shown as background color bands.

    Low regime → green tint, normal → neutral, high → red tint.
    """
    equity = (1.0 + port_rets.fillna(0.0)).cumprod()

    fig = go.Figure()

    # Background regime bands as filled rectangles via shapes.
    regime_map = {"low": "rgba(0, 204, 150, 0.12)",
                  "normal": "rgba(200, 200, 200, 0.08)",
                  "high": "rgba(239, 85, 59, 0.15)"}
    shapes = []
    prev = None
    span_start = None
    for idx, label in vol_regime_ts.items():
        if label != prev:
            if prev is not None and span_start is not None and prev in regime_map:
                shapes.append(dict(
                    type="rect", xref="x", yref="paper",
                    x0=span_start, x1=idx, y0=0, y1=1,
                    fillcolor=regime_map[prev], line_width=0, layer="below",
                ))
            span_start = idx
            prev = label
    if prev is not None and span_start is not None and prev in regime_map:
        shapes.append(dict(
            type="rect", xref="x", yref="paper",
            x0=span_start, x1=vol_regime_ts.index[-1], y0=0, y1=1,
            fillcolor=regime_map[prev], line_width=0, layer="below",
        ))

    fig.add_trace(
        go.Scatter(
            x=equity.index, y=equity.values,
            mode="lines", name="equity",
            line=dict(color=_color(0), width=2),
            hovertemplate=(
                "equity: %{y:.4f}<br>%{x|%Y-%m-%d}<extra></extra>"
            ),
        )
    )

    # Add legend proxies for regime colors.
    for name, color in regime_map.items():
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers",
            name=f"{name} vol",
            marker=dict(size=15, color=color.replace("0.12", "1.0").replace("0.08", "1.0").replace("0.15", "1.0")),
            showlegend=True,
        ))

    fig.update_layout(
        title=title,
        height=height,
        shapes=shapes,
        hovermode="x unified",
        yaxis=dict(title="Equity"),
    )
    _apply_title_layout(fig, title)
    return fig


# ═══════════════════════════════════════════════════════════════════════
# 7. TURNOVER
# ═══════════════════════════════════════════════════════════════════════


def plot_turnover(
    weights_ts: pd.DataFrame,
    title: str = "Rebalancing Turnover",
    height: int = 400,
) -> go.Figure:
    """Daily turnover = 0.5 × Σ|Δw_i_t|, and its 21-day rolling mean.

    The 0.5 factor converts absolute weight change into one-way turnover
    (standard convention). Useful to audit how aggressively the allocation
    scheme rebalances — e.g. regime-adaptive should jump at regime flips,
    while static allocations should be flat near zero.
    """
    dw = weights_ts.diff().abs().sum(axis=1) * 0.5
    roll = dw.rolling(21, min_periods=5).mean()

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=dw.index, y=dw.values,
            name="daily turnover",
            marker_color=_color(0),
            opacity=0.4,
            hovertemplate="turnover: %{y:.2%}<br>%{x|%Y-%m-%d}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=roll.index, y=roll.values,
            mode="lines", name="21d rolling mean",
            line=dict(color=_color(1), width=2),
            hovertemplate="roll: %{y:.2%}<br>%{x|%Y-%m-%d}<extra></extra>",
        )
    )
    fig.update_layout(
        title=title,
        height=height,
        yaxis=dict(title="Turnover (one-way)", tickformat=".1%"),
        hovermode="x unified",
    )
    _apply_title_layout(fig, title)
    return fig


# ═══════════════════════════════════════════════════════════════════════
# 8. ONE-CALL REPORT — all relevant plots for a combined-portfolio result
# ═══════════════════════════════════════════════════════════════════════


def generate_portfolio_mix_plots(
    result: Mapping[str, object],
    show: bool = True,
) -> dict[str, go.Figure]:
    """Generate every applicable mix plot from a combined-portfolio result dict.

    Accepts the dict returned by ``build_combined_portfolio`` (v1) or
    ``build_combined_portfolio_v2``. Skips v2-only plots when the relevant
    keys are missing.

    Returns a dict ``{plot_name: figure}`` so the caller can also save them
    to disk via ``save_fullscreen_html`` or embed them in a tearsheet.
    """
    from ._core import show_browser  # lazy to avoid circular

    figs: dict[str, go.Figure] = {}

    weights_ts = result.get("weights_ts")
    component_returns = result.get("component_returns")
    allocation = result.get("allocation", "unknown")

    if weights_ts is not None:
        figs["weights_area"] = plot_weights_stacked_area(
            weights_ts, title=f"Weights Over Time — {allocation}"
        )
        figs["weights_rolling"] = plot_weights_rolling_mean(weights_ts)
        figs["weights_box"] = plot_weights_distribution(weights_ts)
        figs["turnover"] = plot_turnover(weights_ts)

    if component_returns is not None and weights_ts is not None:
        figs["contribution"] = plot_strategy_contribution(
            component_returns, weights_ts
        )

    if component_returns is not None and len(component_returns.columns) > 1:
        figs["corr_heatmap"] = plot_rolling_correlation_heatmap(component_returns)
        figs["corr_rolling"] = plot_rolling_correlation_pairs(component_returns)

    # v2-specific panels.
    leverage_ts = result.get("leverage_ts")
    port_rets_base = result.get("port_rets_base")
    target_vol = result.get("target_vol")
    max_leverage = result.get("max_leverage")
    if (
        leverage_ts is not None
        and port_rets_base is not None
        and target_vol is not None
        and max_leverage is not None
    ):
        figs["leverage_vol"] = plot_leverage_and_vol(
            leverage_ts,
            port_rets_base,
            target_vol=float(target_vol),
            max_leverage=float(max_leverage),
        )

    dd_scale_ts = result.get("dd_scale_ts")
    port_rets_prelev = result.get("port_rets_prelev")
    if dd_scale_ts is not None and port_rets_prelev is not None:
        # Only show if DD cap actually ever activates (else the plot is flat).
        dd_scale_series = dd_scale_ts if isinstance(dd_scale_ts, pd.Series) else pd.Series(dd_scale_ts)
        if (dd_scale_series < 1.0).any():
            figs["dd_cap"] = plot_dd_cap_activity(port_rets_prelev, dd_scale_ts)

    vol_regime_ts = result.get("vol_regime_ts")
    port_rets = result.get("portfolio_returns")
    if vol_regime_ts is not None and port_rets is not None:
        vol_regime_series = vol_regime_ts if isinstance(vol_regime_ts, pd.Series) else pd.Series(vol_regime_ts)
        if vol_regime_series.nunique() > 1:
            figs["regime_overlay"] = plot_regime_overlay(port_rets, vol_regime_ts)

    if show:
        for name, fig in figs.items():
            show_browser(fig)

    return figs
