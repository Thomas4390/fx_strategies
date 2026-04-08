"""Shared plotting utilities for strategy results.

Leverages VBT Pro native plotting methods for trade signals, portfolio
summaries, and trade-level analysis.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import vectorbtpro as vbt
from plotly.subplots import make_subplots

# ═══════════════════════════════════════════════════════════════════════
# MONTHLY HEATMAP
# ═══════════════════════════════════════════════════════════════════════


# Maximum bars for minute-frequency charts (~1 week of FX data: 5d × 21h × 60min)
_MAX_MINUTE_BARS = 7_200


def _infer_sim_start(idx: pd.DatetimeIndex, max_bars: int = _MAX_MINUTE_BARS) -> Any:
    """Return a sim_start timestamp if the index exceeds max_bars, else None."""
    if len(idx) > max_bars:
        return idx[-max_bars]
    return None


def plot_monthly_heatmap(
    pf: vbt.Portfolio,
    title: str = "Monthly Returns (%)",
) -> go.Figure:
    """Create a year x month heatmap of portfolio returns."""
    rets = pf.returns
    monthly = rets.resample("ME").apply(lambda x: (1 + x).prod() - 1)
    df = pd.DataFrame(
        {
            "year": monthly.index.year,
            "month": monthly.index.month,
            "ret": monthly.values,
        }
    )
    pivot = df.pivot(index="year", columns="month", values="ret")
    month_labels = [
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
    ]
    pivot.columns = [month_labels[m - 1] for m in pivot.columns]
    fig = go.Figure(
        data=go.Heatmap(
            z=pivot.values,
            x=pivot.columns,
            y=pivot.index.astype(str),
            colorscale="RdYlGn",
            zmid=0,
            text=np.round(pivot.values * 100, 1),
            texttemplate="%{text}%",
        )
    )
    fig.update_layout(title=title, height=400)
    return fig


# ═══════════════════════════════════════════════════════════════════════
# TRADE SIGNALS (price + entry/exit markers + indicator overlays)
# ═══════════════════════════════════════════════════════════════════════


def plot_trade_signals(
    pf: vbt.Portfolio,
    title: str = "Trade Signals",
    overlays: dict[str, tuple[pd.Series, str | None, str | None]] | None = None,
    height: int = 700,
) -> go.Figure:
    """Price chart with entry/exit markers, position zones, and indicator overlays.

    Automatically windows to ~1 week for minute-frequency data.
    """
    idx = pf.wrapper.index
    sim_start = _infer_sim_start(idx)

    fig = pf.plot_trade_signals(
        plot_positions="zones",
        sim_start=sim_start,
    )

    # Add indicator overlay lines
    if overlays:
        for label, (series, color, dash) in overlays.items():
            if sim_start is not None:
                series = series.loc[sim_start:]
            line_kwargs = {}
            if color:
                line_kwargs["color"] = color
            if dash:
                line_kwargs["dash"] = dash
            fig.add_trace(
                go.Scatter(
                    x=series.index,
                    y=series.values,
                    mode="lines",
                    name=label,
                    line=line_kwargs,
                    opacity=0.7,
                )
            )

    fig.update_layout(title=title, height=height)
    return fig


# ═══════════════════════════════════════════════════════════════════════
# PORTFOLIO SUMMARY (enhanced composite subplots)
# ═══════════════════════════════════════════════════════════════════════


def plot_portfolio_summary(
    pf: vbt.Portfolio,
    title: str = "Portfolio Summary",
    height: int = 1000,
) -> go.Figure:
    """Composite VBT subplot figure with cumulative returns, drawdowns,
    underwater, and trade P&L. Automatically windows to ~1 week for minute data."""
    sim_start = _infer_sim_start(pf.wrapper.index)
    plot_kwargs: dict[str, Any] = {
        "subplots": ["cumulative_returns", "drawdowns", "underwater", "trade_pnl"],
    }
    if sim_start is not None:
        plot_kwargs["sim_start"] = sim_start
    fig = pf.plot(**plot_kwargs)
    fig.update_layout(title=title, height=height)
    return fig


# ═══════════════════════════════════════════════════════════════════════
# TRADE ANALYSIS (PnL, MAE, MFE, edge ratio)
# ═══════════════════════════════════════════════════════════════════════


def plot_trade_analysis(
    pf: vbt.Portfolio,
    title: str = "Trade Analysis",
    height: int = 800,
) -> go.Figure:
    """2x2 grid: trade PnL scatter, MAE, MFE, running edge ratio.

    Returns an empty figure with a message if no trades exist.
    """
    trades = pf.trades
    if trades.count() == 0:
        fig = go.Figure()
        fig.add_annotation(text="No trades to analyze", showarrow=False)
        fig.update_layout(title=title, height=400)
        return fig

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Trade PnL",
            "MAE (Max Adverse Excursion)",
            "MFE (Max Favorable Excursion)",
            "Running Edge Ratio",
        ),
    )

    trades.plot_pnl(
        pct_scale=True,
        fig=fig,
        add_trace_kwargs=dict(row=1, col=1),
    )
    trades.plot_mae(
        fig=fig,
        add_trace_kwargs=dict(row=1, col=2),
    )
    trades.plot_mfe(
        fig=fig,
        add_trace_kwargs=dict(row=2, col=1),
    )
    trades.plot_running_edge_ratio(
        fig=fig,
        add_trace_kwargs=dict(row=2, col=2),
    )

    fig.update_layout(title=title, height=height, showlegend=False)
    return fig


# ═══════════════════════════════════════════════════════════════════════
# TRADE REPORT (text-based stats)
# ═══════════════════════════════════════════════════════════════════════


def build_trade_report(pf: vbt.Portfolio) -> str:
    """Build comprehensive text report: portfolio + returns + trade stats."""
    sections = []

    sections.append(f"PORTFOLIO STATS\n{'-' * 40}")
    sections.append(pf.stats().to_string())

    sections.append(f"\nRETURNS STATS\n{'-' * 40}")
    sections.append(pf.returns_stats().to_string())

    if pf.trades.count() > 0:
        sections.append(f"\nTRADE STATS\n{'-' * 40}")
        sections.append(pf.trades.stats().to_string())

    return "\n".join(sections)


# ═══════════════════════════════════════════════════════════════════════
# OVERLAY RESOLUTION (PlotConfig -> concrete Series dict)
# ═══════════════════════════════════════════════════════════════════════


def resolve_overlays(
    spec: Any,  # StrategySpec (avoid circular import)
    raw: pd.DataFrame,
    ind_result: Any,
) -> dict[str, tuple[pd.Series, str | None, str | None]]:
    """Resolve ``PlotConfig.overlays`` into ``{label: (series, color, dash)}``.

    Uses the same ``"ind.<name>"`` / ``"data.<col>"`` convention as ``signal_args_map``.
    """
    result = {}
    for overlay in spec.plot_config.overlays:
        prefix, _, name = overlay.source.partition(".")
        if prefix == "ind":
            values = getattr(ind_result, name).values
        elif prefix == "data":
            values = raw[name].values
        else:
            continue
        series = pd.Series(values, index=raw.index, name=overlay.label)
        result[overlay.label] = (series, overlay.color, overlay.dash)
    return result
