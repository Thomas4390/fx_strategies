"""Trade-level visualizations: signals, analysis, duration, and orders.

Extracted from ``_core`` to stay under the 800-line rule. Shared
helpers (``_slice_pf_last``, ``_pick_first_column``) live in
``_helpers`` to avoid a circular import with ``_core``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import vectorbtpro as vbt
from plotly.subplots import make_subplots

from ._core import (
    _MAX_MINUTE_BARS,
    _apply_title_layout,
    _find_featured_trade_window,
    _infer_sim_start,
    make_fullscreen,
)
from ._helpers import _pick_first_column, _slice_pf_last
from ._reports import _total_trade_count


# ═══════════════════════════════════════════════════════════════════════
# TRADE SIGNALS (price + entry/exit markers + indicator overlays)
# ═══════════════════════════════════════════════════════════════════════


def plot_trade_signals(
    pf: vbt.Portfolio,
    title: str = "Trade Signals",
    overlays: dict[str, tuple[pd.Series, str | None, str | None]] | None = None,
    indicator: Any | None = None,
    height: int | None = None,
) -> go.Figure:
    """Price chart with entry/exit markers, position zones, and indicator overlays.

    Picks a representative trade with fully-populated indicator data
    (see :func:`_find_featured_trade_window`) so the band fills and
    indicator lines are drawn consistently over the whole window. Falls
    back to the last ~7200 bars if no trade qualifies.
    """
    idx = pf.wrapper.index

    # 1) Prefer a window centred on a featured trade (full indicator coverage)
    sim_start, sim_end = _find_featured_trade_window(pf, indicator, max_bars=_MAX_MINUTE_BARS)
    if sim_start is None:
        sim_start = _infer_sim_start(idx)
        sim_end = None

    fig = pf.plot_trade_signals(
        plot_positions="zones",
        sim_start=sim_start,
        sim_end=sim_end,
    )

    # 2) Overlay the indicator on the same figure, sliced to the window.
    if indicator is not None and callable(getattr(indicator, "plot", None)):
        try:
            ds_ind = indicator
            if sim_start is not None or sim_end is not None:
                import copy
                ds_ind = copy.copy(indicator)
                for attr_name in dir(ds_ind):
                    if attr_name.startswith("_"):
                        continue
                    try:
                        val = getattr(ds_ind, attr_name)
                    except Exception:
                        continue
                    if isinstance(val, (pd.Series, pd.DataFrame)):
                        try:
                            sliced = val.loc[sim_start:sim_end]
                            object.__setattr__(ds_ind, attr_name, sliced)
                        except Exception:
                            pass
            ds_ind.plot(fig=fig)
        except Exception as e:
            print(f"  [plot_trade_signals] indicator.plot() failed: {e}")

    # 3) Legacy overlay dict
    if overlays:
        for label, (series, color, dash) in overlays.items():
            if sim_start is not None:
                series = series.loc[sim_start:sim_end]
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
                    hovertemplate=(
                        f"<b>{label}</b>: %{{y:.2f}}"
                        "<br>%{x|%Y-%m-%d %H:%M}<extra></extra>"
                    ),
                )
            )

    subtitle = None
    if sim_start is not None and sim_end is not None:
        subtitle = (
            f"{pd.Timestamp(sim_start).strftime('%Y-%m-%d %H:%M')}"
            f"  →  {pd.Timestamp(sim_end).strftime('%Y-%m-%d %H:%M')}"
        )
    _apply_title_layout(fig, title, subtitle=subtitle)
    # Move the legend INSIDE the plot area (top-left) so it no longer
    # collides with the title anchored above the plot.
    fig.update_layout(
        legend=dict(
            orientation="v",
            x=0.01,
            y=0.985,
            xanchor="left",
            yanchor="top",
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="rgba(120,120,120,0.4)",
            borderwidth=1,
            font=dict(size=11),
        ),
    )
    if height is not None:
        fig.layout.height = height
    else:
        make_fullscreen(fig)
    return fig


# ═══════════════════════════════════════════════════════════════════════
# TRADE ANALYSIS (PnL, MAE, MFE, edge ratio)
# ═══════════════════════════════════════════════════════════════════════


def plot_trade_analysis(
    pf: vbt.Portfolio,
    title: str = "Trade Analysis",
    height: int | None = None,
) -> go.Figure:
    """2x2 grid: trade PnL scatter, MAE, MFE, running edge ratio.

    Returns an empty figure with a message if no trades exist.
    Multi-asset portfolios fall back to the first column since VBT's
    per-trade plots require a single column.
    """
    pf_single, col_label = _pick_first_column(pf)
    trades = pf_single.trades
    if trades.count() == 0:
        fig = go.Figure()
        fig.add_annotation(text="No trades to analyze", showarrow=False)
        fig.update_layout(title=title)
        make_fullscreen(fig)
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

    # Build a time-indexed Trade PnL scatter manually so the x-axis
    # spans the full backtest (first year → last year) instead of the
    # default trade-index (0..N). VBT's trades.plot_pnl() uses the
    # trade record ID on the x-axis which hides the chronology.
    try:
        records = trades.records_readable
        if not records.empty:
            entry_col = (
                "Entry Index" if "Entry Index" in records.columns
                else "Entry Timestamp"
            )
            entry_times = pd.to_datetime(records[entry_col].values)
            pnl_pct = records["Return"].astype(float).values
            # Clip y-axis to 2-98 percentile so outliers don't crush
            # the scale, but keep markers inside the clipped range.
            mask = ~np.isnan(pnl_pct)
            if mask.sum() >= 5:
                lo = float(np.percentile(pnl_pct[mask], 2))
                hi = float(np.percentile(pnl_pct[mask], 98))
                span = max(hi - lo, 1e-6)
                pad = span * 0.12
                y_range = [lo - pad, hi + pad]
            else:
                y_range = None
            wins = pnl_pct > 0
            losses = ~wins
            fig.add_trace(
                go.Scatter(
                    x=entry_times[wins], y=pnl_pct[wins],
                    mode="markers", name="Winners",
                    marker=dict(
                        color="#00CC96", size=6,
                        line=dict(color="rgba(0,0,0,0.25)", width=0.5),
                    ),
                    hovertemplate=(
                        "<b>Winning Trade Return</b>: %{y:.2%}"
                        "<br>Entry: %{x|%Y-%m-%d %H:%M}<extra></extra>"
                    ),
                ),
                row=1, col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=entry_times[losses], y=pnl_pct[losses],
                    mode="markers", name="Losers",
                    marker=dict(
                        color="#EF553B", size=6,
                        line=dict(color="rgba(0,0,0,0.25)", width=0.5),
                    ),
                    hovertemplate=(
                        "<b>Losing Trade Return</b>: %{y:.2%}"
                        "<br>Entry: %{x|%Y-%m-%d %H:%M}<extra></extra>"
                    ),
                ),
                row=1, col=1,
            )
            fig.add_hline(y=0, line_color="gray", line_width=1, row=1, col=1)
            # Force x-axis to span the full data range
            try:
                data_idx = pf_single.wrapper.index
                fig.update_xaxes(
                    range=[data_idx[0], data_idx[-1]],
                    row=1, col=1,
                )
            except Exception:
                pass
            fig.update_yaxes(tickformat=".2%", row=1, col=1)
            if y_range is not None:
                fig.update_yaxes(range=y_range, row=1, col=1)
    except Exception as e:
        print(f"  [plot_trade_analysis] Trade PnL subplot failed: {e}")

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

    if col_label:
        title = f"{title} — {col_label}"
    fig.update_layout(showlegend=False)
    _apply_title_layout(fig, title)
    if height is not None:
        fig.layout.height = height
    else:
        make_fullscreen(fig)
    # Push subplot annotations down slightly to leave room for the main title
    for ann in fig.layout.annotations:
        if ann.yref == "paper":
            ann.y = min(ann.y, 0.94)
    return fig


# ═══════════════════════════════════════════════════════════════════════
# TRADE DURATION ANALYSIS
# ═══════════════════════════════════════════════════════════════════════


def plot_trade_duration(
    pf: vbt.Portfolio,
    title: str = "Trade Duration Analysis",
    height: int | None = None,
) -> go.Figure:
    """Scatter PnL vs duration + duration histogram.

    Aggregates trades across columns for multi-asset portfolios.
    """
    trades = pf.trades
    if _total_trade_count(pf) == 0:
        fig = go.Figure()
        fig.add_annotation(text="No trades", showarrow=False)
        make_fullscreen(fig)
        return fig

    pnls = np.asarray(trades.pnl.values).ravel()
    durations = np.asarray(trades.duration.values).ravel()
    # Drop NaN padding (multi-col trade records are padded to max-length)
    mask = ~pd.isna(pnls)
    pnls = pnls[mask]
    durations = durations[mask]
    if len(pnls) == 0:
        fig = go.Figure()
        fig.add_annotation(text="No trades after NaN filter", showarrow=False)
        make_fullscreen(fig)
        return fig
    # ``trades.duration`` is an int64 array of bar counts. For FX minute
    # data that corresponds directly to minutes (1 bar = 1 min). Cast
    # through timedelta64[m] so the conversion is explicit and safe.
    dur_minutes = durations.astype("timedelta64[m]").astype(float)
    is_win = pnls > 0

    # Build a subtitle with key duration stats so short-duration trades
    # (e.g. SL hit on the bar after entry) are immediately visible.
    dur_stats = (
        f"n={len(dur_minutes)}  "
        f"min={int(dur_minutes.min())}m  "
        f"median={int(np.median(dur_minutes))}m  "
        f"mean={dur_minutes.mean():.0f}m  "
        f"max={int(dur_minutes.max())}m"
    )

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("PnL vs Duration", "Duration Distribution"),
        horizontal_spacing=0.12,
    )

    # Scatter PnL vs duration (wins green, losses red)
    fig.add_trace(
        go.Scatter(
            x=dur_minutes[is_win], y=pnls[is_win],
            mode="markers", name="Winners",
            marker=dict(color="#00CC96", size=6),
            hovertemplate=(
                "<b>Winning Trade PnL</b>: $%{y:,.2f}"
                "<br><b>Duration</b>: %{x:.0f} min<extra></extra>"
            ),
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=dur_minutes[~is_win], y=pnls[~is_win],
            mode="markers", name="Losers",
            marker=dict(color="#EF553B", size=6),
            hovertemplate=(
                "<b>Losing Trade PnL</b>: $%{y:,.2f}"
                "<br><b>Duration</b>: %{x:.0f} min<extra></extra>"
            ),
        ),
        row=1, col=1,
    )

    # Duration histogram
    fig.add_trace(
        go.Histogram(
            x=dur_minutes, nbinsx=30,
            name="Duration", marker_color="#636EFA",
            hovertemplate=(
                "<b>Trade Duration</b>: %{x} min"
                "<br><b>Trade Count</b>: %{y}<extra></extra>"
            ),
        ),
        row=1, col=2,
    )

    fig.update_xaxes(title_text="Duration (minutes)", row=1, col=1)
    fig.update_xaxes(title_text="Duration (minutes)", row=1, col=2)
    fig.update_yaxes(title_text="PnL ($)", row=1, col=1)
    fig.update_yaxes(title_text="Trade count", row=1, col=2)
    _apply_title_layout(fig, title, subtitle=dur_stats)
    if height is not None:
        fig.layout.height = height
    else:
        make_fullscreen(fig)
    for ann in fig.layout.annotations:
        if ann.yref == "paper":
            ann.y = min(ann.y, 0.94)
    return fig


# ═══════════════════════════════════════════════════════════════════════
# NATIVE VBT PLOTS: ORDERS / EXPOSURE / ALLOCATIONS
# ═══════════════════════════════════════════════════════════════════════


def plot_orders_on_price(
    pf: vbt.Portfolio,
    title: str = "Orders on Price",
    max_bars: int = _MAX_MINUTE_BARS,
) -> go.Figure:
    """Price chart with buy/sell order markers (VBT native).

    Uses :meth:`vbt.Portfolio.orders.plot` which overlays entry/exit
    markers on the close price. Minute-frequency data is automatically
    clipped to the last ``max_bars`` bars — required to keep the HTML
    file size manageable. Multi-asset portfolios fall back to the
    first column since VBT's native plot requires a single column.
    """
    pf_slice = _slice_pf_last(pf, max_bars)
    pf_single, col_label = _pick_first_column(pf_slice)
    fig = pf_single.orders.plot()
    if col_label:
        title = f"{title} — {col_label}"
    fig.update_layout(title=title)
    make_fullscreen(fig)
    return fig


def plot_trades_on_price(
    pf: vbt.Portfolio,
    title: str = "Trades on Price",
    max_bars: int = _MAX_MINUTE_BARS,
) -> go.Figure:
    """Price chart with trade P&L zones overlaid (VBT native).

    Uses :meth:`vbt.Portfolio.trades.plot` which highlights each trade
    as a green (win) or red (loss) zone on top of the close price.
    Multi-asset portfolios fall back to the first column.
    """
    pf_slice = _slice_pf_last(pf, max_bars)
    pf_single, col_label = _pick_first_column(pf_slice)
    fig = pf_single.trades.plot()
    if col_label:
        title = f"{title} — {col_label}"
    fig.update_layout(title=title)
    make_fullscreen(fig)
    return fig


def plot_exposure(
    pf: vbt.Portfolio,
    title: str = "Gross Exposure",
) -> go.Figure:
    """Gross-exposure timeline (VBT native).

    Shows the ratio of position notional to portfolio value over time,
    which is the best lens for verifying that leverage is being
    applied as expected.
    """
    pf_daily = pf.resample("1D")
    try:
        exposure = pf_daily.gross_exposure
    except Exception:
        exposure = pf_daily.allocations
    fig = exposure.vbt.plot()
    fig.update_traces(
        hovertemplate=(
            "<b>Gross Exposure</b>: %{y:.2f}"
            "<br>%{x|%Y-%m-%d}<extra></extra>"
        ),
    )
    fig.add_hline(y=1.0, line_dash="dot", line_color="gray",
                  annotation_text="100%")
    fig.update_layout(title=title, yaxis_title="Gross Exposure")
    make_fullscreen(fig)
    return fig


def plot_value_and_cash(
    pf: vbt.Portfolio,
    title: str = "Portfolio Value vs Cash",
) -> go.Figure:
    """Two-panel value and cash timelines (VBT native)."""
    pf_daily = pf.resample("1D")
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        subplot_titles=("Portfolio Value", "Cash"),
        row_heights=[0.6, 0.4],
    )
    value = pf_daily.value
    cash = pf_daily.cash
    fig.add_trace(
        go.Scatter(
            x=value.index, y=value.values,
            mode="lines", name="Portfolio Value",
            line=dict(color="#00CC96", width=2),
            hovertemplate=(
                "<b>Portfolio Value</b>: $%{y:,.2f}"
                "<br>%{x|%Y-%m-%d}<extra></extra>"
            ),
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=cash.index, y=cash.values,
            mode="lines", name="Cash",
            line=dict(color="#636EFA", width=2),
            hovertemplate=(
                "<b>Cash Balance</b>: $%{y:,.2f}"
                "<br>%{x|%Y-%m-%d}<extra></extra>"
            ),
        ),
        row=2, col=1,
    )
    fig.update_layout(title=title)
    make_fullscreen(fig)
    return fig


def plot_orders_heatmap(
    pf: vbt.Portfolio,
    title: str = "Trade Frequency by Hour/Weekday",
) -> go.Figure:
    """Weekday × hour heatmap of trade entry count (VBT native data).

    Aggregates trades across columns for multi-asset portfolios.
    """
    if _total_trade_count(pf) == 0:
        fig = go.Figure()
        fig.add_annotation(text="No trades to analyze", showarrow=False)
        make_fullscreen(fig)
        return fig

    trades = pf.trades
    entry_idx_raw = np.asarray(trades.entry_idx.values).ravel()
    # Drop NaN/invalid padding from multi-column trade records
    valid_mask = ~pd.isna(entry_idx_raw)
    entry_idx_raw = entry_idx_raw[valid_mask].astype(int)
    if len(entry_idx_raw) == 0:
        fig = go.Figure()
        fig.add_annotation(text="No trades after filter", showarrow=False)
        make_fullscreen(fig)
        return fig
    entry_idx = pf.wrapper.index[entry_idx_raw]
    df = pd.DataFrame({"ts": entry_idx})
    df["hour"] = df["ts"].dt.hour
    df["weekday"] = df["ts"].dt.day_name()
    weekday_order = [
        "Monday", "Tuesday", "Wednesday", "Thursday",
        "Friday", "Saturday", "Sunday",
    ]
    counts = (
        df.groupby(["weekday", "hour"])
        .size()
        .unstack(fill_value=0)
        .reindex(weekday_order, fill_value=0)
    )
    fig = go.Figure(
        data=go.Heatmap(
            z=counts.values,
            x=counts.columns.values,
            y=counts.index.values,
            colorscale="YlOrRd",
            hovertemplate="Weekday=%{y}<br>Hour=%{x}<br>Trades=%{z}<extra></extra>",
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Hour (UTC)",
        yaxis_title="Weekday",
    )
    make_fullscreen(fig)
    return fig
