"""Robustness and overfitting-detection visualizations.

Sibling of ``_equity`` and ``_params``. Houses the plots produced by
:mod:`framework.robustness` : bootstrap metric distributions, equity
fan charts, MaxDD CDFs, PBO logit histograms, SPA p-values, and
rolling-stability overlays.

All figures are Plotly, wrapped by ``make_fullscreen`` so they render
in the existing fullscreen HTML shell used by ``analyze_portfolio``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ._core import _apply_title_layout, make_fullscreen


# ═══════════════════════════════════════════════════════════════════════
# Bootstrap distributions
# ═══════════════════════════════════════════════════════════════════════


def plot_bootstrap_distribution(
    samples: np.ndarray,
    observed: float,
    *,
    metric_label: str = "Metric",
    ci_low: float | None = None,
    ci_high: float | None = None,
    title: str | None = None,
    n_bins: int = 60,
) -> go.Figure:
    """Histogram of bootstrap samples with observed value and 95% CI.

    Parameters
    ----------
    samples
        ``(n_boot,)`` array of bootstrap replicates.
    observed
        The point value computed on the original series.
    metric_label
        Human-readable metric name, used in axis and title.
    ci_low, ci_high
        Lower/upper bounds of the confidence interval (usually the
        2.5% / 97.5% quantiles of ``samples``). Drawn as dashed
        vertical lines when supplied.
    """
    arr = np.asarray(samples, dtype=float)
    arr = arr[np.isfinite(arr)]

    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=arr,
            nbinsx=n_bins,
            marker=dict(color="#4C78A8", line=dict(color="#1F3A5F", width=0.5)),
            opacity=0.85,
            name=f"{metric_label} bootstrap",
            hovertemplate=f"{metric_label}=%{{x:.4f}}<br>count=%{{y}}<extra></extra>",
        )
    )
    y_ref = "paper"
    # Observed value line.
    fig.add_shape(
        type="line",
        x0=observed,
        x1=observed,
        y0=0,
        y1=1,
        yref=y_ref,
        line=dict(color="#E45756", width=3),
    )
    fig.add_annotation(
        x=observed,
        y=1.0,
        yref=y_ref,
        text=f"<b>observed = {observed:.3f}</b>",
        showarrow=False,
        font=dict(color="#E45756", size=12),
        yanchor="bottom",
    )
    if ci_low is not None:
        fig.add_shape(
            type="line",
            x0=ci_low,
            x1=ci_low,
            y0=0,
            y1=1,
            yref=y_ref,
            line=dict(color="#54A24B", width=2, dash="dash"),
        )
    if ci_high is not None:
        fig.add_shape(
            type="line",
            x0=ci_high,
            x1=ci_high,
            y0=0,
            y1=1,
            yref=y_ref,
            line=dict(color="#54A24B", width=2, dash="dash"),
        )
    fig.update_layout(
        xaxis_title=metric_label,
        yaxis_title="Frequency",
        bargap=0.02,
    )
    _apply_title_layout(fig, title or f"Bootstrap distribution — {metric_label}")
    return make_fullscreen(fig)


def plot_metric_ci_forest(
    bootstrap_df: pd.DataFrame,
    *,
    title: str = "Bootstrap 95% CI — Metrics Forest Plot",
    include_metrics: list[str] | None = None,
) -> go.Figure:
    """Forest plot of observed value ± bootstrap CI for each metric.

    Parameters
    ----------
    bootstrap_df
        Output of :func:`framework.bootstrap.bootstrap_all_metrics`.
    include_metrics
        Optional subset of metric names (rows of ``bootstrap_df``) to
        display. Defaults to all rows.
    """
    df = bootstrap_df
    if include_metrics is not None:
        df = df.loc[[m for m in include_metrics if m in df.index]]
    df = df.iloc[::-1]  # top metric at the top of the plot

    fig = go.Figure()
    for metric, row in df.iterrows():
        label = row.get("label", metric)
        fig.add_trace(
            go.Scatter(
                x=[row["ci_low"], row["ci_high"]],
                y=[label, label],
                mode="lines",
                line=dict(color="#4C78A8", width=4),
                showlegend=False,
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[row["observed"]],
                y=[label],
                mode="markers",
                marker=dict(color="#E45756", size=12, symbol="diamond"),
                name="observed",
                showlegend=False,
                hovertemplate=(
                    f"<b>{label}</b><br>observed=%{{x:.4f}}<br>"
                    f"CI=[{row['ci_low']:.4f}, {row['ci_high']:.4f}]<extra></extra>"
                ),
            )
        )
    fig.update_layout(
        xaxis_title="Metric value (95% bootstrap CI)",
        yaxis=dict(type="category"),
    )
    _apply_title_layout(fig, title)
    return make_fullscreen(fig)


# ═══════════════════════════════════════════════════════════════════════
# Equity fan chart
# ═══════════════════════════════════════════════════════════════════════


def plot_equity_fan_chart(
    equity_paths: pd.DataFrame,
    *,
    observed_equity: pd.Series | None = None,
    title: str = "Bootstrap Equity Fan Chart",
    percentiles: tuple[float, ...] = (5.0, 25.0, 50.0, 75.0, 95.0),
) -> go.Figure:
    """Fan chart (P5/P25/P50/P75/P95) over a matrix of bootstrap paths.

    Parameters
    ----------
    equity_paths
        Output of :func:`framework.bootstrap.bootstrap_equity_paths` —
        DataFrame of shape ``(T, n_sim)``.
    observed_equity
        Optional observed equity curve (same index as ``equity_paths``)
        overlaid as a solid line.
    """
    arr = equity_paths.to_numpy(dtype=np.float64)
    perc = np.percentile(arr, list(percentiles), axis=1)
    x = equity_paths.index

    fig = go.Figure()
    # Outer band P5–P95.
    fig.add_trace(
        go.Scatter(
            x=x,
            y=perc[0],
            line=dict(color="rgba(76,120,168,0)"),
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=perc[-1],
            fill="tonexty",
            fillcolor="rgba(76,120,168,0.15)",
            line=dict(color="rgba(76,120,168,0)"),
            name="P5–P95",
            hoverinfo="skip",
        )
    )
    # Inner band P25–P75.
    fig.add_trace(
        go.Scatter(
            x=x,
            y=perc[1],
            line=dict(color="rgba(76,120,168,0)"),
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=perc[3],
            fill="tonexty",
            fillcolor="rgba(76,120,168,0.35)",
            line=dict(color="rgba(76,120,168,0)"),
            name="P25–P75",
            hoverinfo="skip",
        )
    )
    # Median.
    fig.add_trace(
        go.Scatter(
            x=x,
            y=perc[2],
            mode="lines",
            line=dict(color="#4C78A8", width=2.5),
            name="Median (P50)",
        )
    )
    if observed_equity is not None:
        fig.add_trace(
            go.Scatter(
                x=observed_equity.index,
                y=observed_equity.values,
                mode="lines",
                line=dict(color="#E45756", width=2.5),
                name="Observed",
            )
        )
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Equity (normalized)",
    )
    _apply_title_layout(fig, title)
    return make_fullscreen(fig)


# ═══════════════════════════════════════════════════════════════════════
# Max drawdown CDF
# ═══════════════════════════════════════════════════════════════════════


def plot_mdd_distribution(
    mdd_samples: np.ndarray,
    observed_mdd: float,
    *,
    title: str = "MC Max Drawdown CDF",
    show_cdf: bool = True,
) -> go.Figure:
    """CDF + histogram of Monte Carlo max drawdowns.

    Parameters
    ----------
    mdd_samples
        Array of bootstrap / shuffle max drawdowns (fractional).
    observed_mdd
        Observed max drawdown — marked on both panels.
    show_cdf
        When True, a second subplot shows the empirical CDF.
    """
    arr = np.asarray(mdd_samples, dtype=float)
    arr = arr[np.isfinite(arr)]

    if show_cdf:
        fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=("Histogram", "Empirical CDF"),
            horizontal_spacing=0.08,
        )
    else:
        fig = go.Figure()

    fig.add_trace(
        go.Histogram(
            x=arr,
            nbinsx=60,
            marker=dict(color="#4C78A8", line=dict(color="#1F3A5F", width=0.5)),
            opacity=0.85,
            name="MC MDD",
            hovertemplate="MDD=%{x:.3f}<br>count=%{y}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    # Observed line on histogram.
    fig.add_shape(
        type="line",
        x0=observed_mdd,
        x1=observed_mdd,
        y0=0,
        y1=1,
        yref="y domain",
        xref="x",
        line=dict(color="#E45756", width=3),
    )

    if show_cdf:
        sorted_arr = np.sort(arr)
        cdf = np.arange(1, arr.size + 1) / arr.size
        fig.add_trace(
            go.Scatter(
                x=sorted_arr,
                y=cdf,
                mode="lines",
                line=dict(color="#4C78A8", width=2.5),
                name="CDF",
            ),
            row=1,
            col=2,
        )
        # Locate the observed MDD on the CDF.
        frac_worse = float(np.mean(arr >= observed_mdd))
        fig.add_shape(
            type="line",
            x0=observed_mdd,
            x1=observed_mdd,
            y0=0,
            y1=1,
            yref="y2 domain",
            xref="x2",
            line=dict(color="#E45756", width=3),
        )
        fig.add_annotation(
            x=observed_mdd,
            y=1 - frac_worse,
            xref="x2",
            yref="y2",
            text=f"<b>observed={observed_mdd:.3f}</b><br>{100*(1-frac_worse):.0f}% of shuffles better",
            showarrow=True,
            arrowhead=2,
            ax=40,
            ay=-40,
            font=dict(color="#E45756", size=11),
        )
        fig.update_xaxes(title_text="Max drawdown", row=1, col=2)
        fig.update_yaxes(title_text="CDF", row=1, col=2)

    fig.update_xaxes(title_text="Max drawdown", row=1, col=1)
    fig.update_yaxes(title_text="Frequency", row=1, col=1)
    _apply_title_layout(fig, title)
    return make_fullscreen(fig)


# ═══════════════════════════════════════════════════════════════════════
# PBO logits
# ═══════════════════════════════════════════════════════════════════════


def plot_pbo_logits(
    logits: np.ndarray,
    pbo: float,
    *,
    title: str | None = None,
    n_bins: int = 50,
) -> go.Figure:
    """Histogram of CSCV logits with PBO annotation.

    A healthy strategy should have its logits centred well above zero
    (IS winner stays in the top half OOS). ``pbo = P(logit < 0)``.
    """
    arr = np.asarray(logits, dtype=float)
    arr = arr[np.isfinite(arr)]

    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=arr,
            nbinsx=n_bins,
            marker=dict(color="#72B7B2", line=dict(color="#2C5C5C", width=0.5)),
            opacity=0.85,
            name="CSCV logits",
            hovertemplate="logit=%{x:.3f}<br>count=%{y}<extra></extra>",
        )
    )
    fig.add_shape(
        type="line",
        x0=0,
        x1=0,
        y0=0,
        y1=1,
        yref="paper",
        line=dict(color="#E45756", width=3, dash="dash"),
    )
    verdict = "healthy" if pbo < 0.5 else "overfit"
    fig.add_annotation(
        x=0.0,
        y=0.98,
        xref="x",
        yref="paper",
        text=f"<b>PBO = {pbo:.3f}</b><br>({verdict})",
        showarrow=False,
        font=dict(color="#E45756", size=14),
        bgcolor="rgba(255,255,255,0.85)",
        bordercolor="#E45756",
        borderwidth=1,
    )
    fig.update_layout(
        xaxis_title="CSCV logit  (>0 = IS winner is above OOS median)",
        yaxis_title="Frequency",
    )
    _apply_title_layout(fig, title or "Probability of Backtest Overfitting — CSCV logits")
    return make_fullscreen(fig)


# ═══════════════════════════════════════════════════════════════════════
# SPA / StepM p-values
# ═══════════════════════════════════════════════════════════════════════


def plot_spa_pvalues(
    spa_result: dict[str, Any],
    *,
    title: str = "SPA / Reality Check p-values",
) -> go.Figure:
    """Bar plot of Hansen SPA lower / consistent / upper p-values."""
    labels = ["lower", "consistent", "upper"]
    vals = [
        float(spa_result.get("pvalue_lower", np.nan)),
        float(spa_result.get("pvalue_consistent", np.nan)),
        float(spa_result.get("pvalue_upper", np.nan)),
    ]
    colors = [
        "#54A24B" if v < 0.05 else "#E45756" for v in vals
    ]
    fig = go.Figure(
        go.Bar(
            x=labels,
            y=vals,
            marker=dict(color=colors),
            text=[f"{v:.4f}" for v in vals],
            textposition="outside",
        )
    )
    fig.add_shape(
        type="line",
        x0=-0.5,
        x1=2.5,
        y0=0.05,
        y1=0.05,
        line=dict(color="#E45756", width=2, dash="dash"),
    )
    fig.add_annotation(
        x=2.5,
        y=0.05,
        text="α = 0.05",
        showarrow=False,
        font=dict(color="#E45756", size=11),
        xanchor="right",
    )
    fig.update_layout(
        xaxis_title="SPA bound",
        yaxis_title="p-value",
        yaxis=dict(range=[0, max(max(vals) * 1.2, 0.1)]),
    )
    subtitle = (
        f"best strategy: {spa_result.get('best_strategy_label', '?')} · "
        f"n={spa_result.get('n_obs', '?')} × {spa_result.get('n_strategies', '?')} strategies"
    )
    _apply_title_layout(fig, title, subtitle=subtitle)
    return make_fullscreen(fig)


# ═══════════════════════════════════════════════════════════════════════
# Rolling metric stability
# ═══════════════════════════════════════════════════════════════════════


def plot_rolling_metric_stability(
    returns: pd.Series,
    *,
    window: str = "365D",
    ann_factor: float = 252.0,
    title: str = "Rolling Metric Stability",
) -> go.Figure:
    """Overlay rolling Sharpe / Sortino / Calmar on a common scale.

    Uses plain pandas rolling to avoid dragging VBT annualization
    surprises on sparse windows — the goal here is a qualitative
    stability view, not an exact metric.
    """
    r = returns.dropna()
    if r.empty:
        raise ValueError("returns is empty")

    try:
        roll = r.rolling(window)
    except Exception:
        # Fallback for numeric windows on a RangeIndex.
        roll = r.rolling(int(window.strip("D")))

    roll_mean = roll.mean()
    roll_std = roll.std(ddof=1)
    sharpe = (roll_mean / roll_std) * np.sqrt(ann_factor)

    neg = r.where(r < 0, 0.0)
    down_std = neg.rolling(window).std(ddof=1)
    sortino = (roll_mean / down_std.replace(0, np.nan)) * np.sqrt(ann_factor)

    # Rolling Calmar via cummax of the rolling cumulative return.
    cum = (1.0 + r).cumprod()
    roll_max = cum.rolling(window).max()
    roll_min = cum.rolling(window).min()
    roll_dd = (roll_min - roll_max) / roll_max
    ann_ret = roll_mean * ann_factor
    calmar = ann_ret / roll_dd.abs().replace(0, np.nan)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=sharpe.index, y=sharpe.values, mode="lines", name="Rolling Sharpe")
    )
    fig.add_trace(
        go.Scatter(x=sortino.index, y=sortino.values, mode="lines", name="Rolling Sortino")
    )
    fig.add_trace(
        go.Scatter(x=calmar.index, y=calmar.values, mode="lines", name="Rolling Calmar")
    )
    fig.add_shape(
        type="line",
        x0=sharpe.index[0] if not sharpe.empty else None,
        x1=sharpe.index[-1] if not sharpe.empty else None,
        y0=0,
        y1=0,
        line=dict(color="#888", width=1, dash="dot"),
    )
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Annualized metric",
        legend=dict(orientation="h"),
    )
    _apply_title_layout(fig, title, subtitle=f"window={window}, ann_factor={ann_factor}")
    return make_fullscreen(fig)


# ═══════════════════════════════════════════════════════════════════════
# CPCV OOS distribution
# ═══════════════════════════════════════════════════════════════════════


def plot_cpcv_distribution(
    cpcv_dist: pd.DataFrame,
    *,
    top_n: int = 20,
    metric_label: str | None = None,
    title: str = "CPCV — OOS Distribution per Config",
) -> go.Figure:
    """Boxplot of the per-config OOS metric distribution.

    Input = output of :func:`framework.cpcv.cpcv_oos_distribution`.
    """
    if cpcv_dist.empty:
        fig = go.Figure()
        _apply_title_layout(fig, title, subtitle="(empty)")
        return make_fullscreen(fig)

    label = metric_label or cpcv_dist.attrs.get("metric_label", "Metric")
    top = cpcv_dist.head(top_n)

    fig = go.Figure()
    for key, row in top.iterrows():
        fig.add_trace(
            go.Box(
                x=[
                    row["min"],
                    row["q05"],
                    row["median"],
                    row["q95"],
                    row["max"],
                ],
                name=str(key),
                boxpoints=False,
                marker=dict(color="#4C78A8"),
                line=dict(color="#1F3A5F"),
                orientation="h",
                hovertemplate=(
                    f"<b>{key}</b><br>median={row['median']:.3f}<br>"
                    f"std={row['std']:.3f}<br>pct_pos={row['pct_positive']:.0%}<extra></extra>"
                ),
            )
        )
    fig.update_layout(
        xaxis_title=f"{label} (OOS across CPCV splits)",
        showlegend=False,
    )
    _apply_title_layout(fig, title, subtitle=f"top {top_n} configs by median OOS")
    return make_fullscreen(fig)


__all__ = [
    "plot_bootstrap_distribution",
    "plot_metric_ci_forest",
    "plot_equity_fan_chart",
    "plot_mdd_distribution",
    "plot_pbo_logits",
    "plot_spa_pvalues",
    "plot_rolling_metric_stability",
    "plot_cpcv_distribution",
]
