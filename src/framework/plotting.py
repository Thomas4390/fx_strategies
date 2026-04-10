"""Shared plotting utilities for strategy results.

Leverages VBT Pro native plotting methods for trade signals, portfolio
summaries, parameter heatmaps, 3D volumes, and trade-level analysis.

All figures saved via :func:`save_fullscreen_html` or displayed via
:func:`show_browser` are wrapped in an HTML shell that forces the
Plotly div to fill the entire browser viewport (100vh × 100vw).
"""

from __future__ import annotations

import calendar
import os
import tempfile
import webbrowser
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import vectorbtpro as vbt
from plotly.subplots import make_subplots


# ═══════════════════════════════════════════════════════════════════════
# FULLSCREEN BROWSER RENDERING
# ═══════════════════════════════════════════════════════════════════════

# HTML shell that forces the Plotly graph div to fill the viewport.
# Plotly's default behaviour is a fixed-size container; with
# ``autosize=True`` on the layout and the CSS rules below, the chart
# resizes dynamically with the browser window.
_FULLSCREEN_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
  html, body {{
    margin: 0;
    padding: 0;
    height: 100%;
    width: 100%;
    background: #ffffff;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    overflow: hidden;
  }}
  #plot-root, .plot-container, .plotly-graph-div {{
    height: 100vh !important;
    width: 100vw !important;
  }}
  .js-plotly-plot .plotly .modebar {{ opacity: 0.85; }}
</style>
</head>
<body>
<div id="plot-root">
{plot_div}
</div>
</body>
</html>
"""


def make_fullscreen(fig: go.Figure) -> go.Figure:
    """Configure a Plotly figure to fill the entire browser viewport.

    Drops fixed width/height, enables autosize, and tightens margins so
    the chart stretches to the full window when embedded in the
    fullscreen HTML shell.
    """
    fig.update_layout(
        autosize=True,
        margin={"l": 60, "r": 40, "t": 110, "b": 50},
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.015,
            "xanchor": "center",
            "x": 0.5,
        },
    )
    # Clear any fixed pixel dimensions so CSS 100vh/100vw wins.
    fig.layout.height = None
    fig.layout.width = None
    return fig


def _apply_title_layout(
    fig: go.Figure,
    title: str,
    *,
    subtitle: str | None = None,
) -> go.Figure:
    """Apply a consistent, non-overlapping title style to a figure.

    Places the title well above the plot area with a larger font so it
    does not collide with subplot annotations or axis labels. Optional
    subtitle appears as a smaller secondary line.
    """
    text = f"<b>{title}</b>"
    if subtitle:
        text += f"<br><span style='font-size:13px;color:#888'>{subtitle}</span>"
    fig.update_layout(
        title=dict(
            text=text,
            x=0.5,
            xanchor="center",
            y=0.985,
            yanchor="top",
            font=dict(size=20, color="#222"),
            pad=dict(t=10, b=10),
        ),
    )
    return fig


def save_fullscreen_html(
    fig: go.Figure,
    path: str | os.PathLike,
    title: str | None = None,
) -> str:
    """Save *fig* to *path* as a standalone HTML that fills the browser.

    Uses Plotly's responsive config and a CSS wrapper so the chart
    resizes dynamically with the window. Returns the written path.
    """
    make_fullscreen(fig)
    if title is None:
        layout_title = getattr(fig.layout.title, "text", None)
        title = layout_title or "Plot"
    plot_div = fig.to_html(
        include_plotlyjs="cdn",
        full_html=False,
        config={
            "responsive": True,
            "displayModeBar": True,
            "scrollZoom": True,
        },
        div_id="main-plot",
    )
    html = _FULLSCREEN_HTML_TEMPLATE.format(title=title, plot_div=plot_div)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    return str(out)


def show_browser(
    fig: go.Figure,
    title: str | None = None,
    fullscreen: bool = True,
) -> None:
    """Open *fig* in the default web browser.

    ``fullscreen=True`` (default) writes a temporary HTML shell that
    forces the Plotly div to fill the viewport (100vh × 100vw), which
    is needed for true fullscreen since Plotly's default ``.show()``
    template renders at a fixed container size.

    ``fullscreen=False`` uses the purely native
    ``fig.show(renderer="browser")`` call — simpler, but the plot will
    not stretch to fill the entire window.
    """
    if fullscreen:
        tmp = tempfile.NamedTemporaryFile(
            prefix="fx_plot_",
            suffix=".html",
            delete=False,
            mode="w",
            encoding="utf-8",
        )
        tmp.close()
        save_fullscreen_html(fig, tmp.name, title=title)
        webbrowser.open(f"file://{tmp.name}")
    else:
        make_fullscreen(fig)
        fig.show(renderer="browser", config={"responsive": True})


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


def _find_featured_trade_window(
    pf: vbt.Portfolio,
    indicator: Any | None,
    max_bars: int = _MAX_MINUTE_BARS,
) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    """Pick a representative trade and return ``(sim_start, sim_end)``
    bounds that comfortably contain entry, exit and some context.

    Selection criteria:
      1. All indicator Series/DataFrame fields must be fully populated
         across the entry→exit window (no NaN).
      2. Trade duration should fit within ~40% of the target window so
         that there is visible pre/post context.
      3. Prefer trades with larger absolute PnL (more informative).

    Returns ``(None, None)`` if no trade qualifies (caller falls back to
    the default "last N bars" window).
    """
    try:
        records = pf.trades.records_readable
    except Exception:
        return None, None
    if records.empty:
        return None, None

    entry_col = "Entry Index" if "Entry Index" in records.columns else "Entry Timestamp"
    exit_col = "Exit Index" if "Exit Index" in records.columns else "Exit Timestamp"
    records = records.copy()
    records[entry_col] = pd.to_datetime(records[entry_col])
    records[exit_col] = pd.to_datetime(records[exit_col])
    records["_dur_min"] = (
        records[exit_col] - records[entry_col]
    ).dt.total_seconds() / 60.0

    # Gather indicator Series fields for NaN checks.
    ind_series: list[pd.Series] = []
    if indicator is not None:
        for attr_name in dir(indicator):
            if attr_name.startswith("_"):
                continue
            try:
                val = getattr(indicator, attr_name)
            except Exception:
                continue
            if isinstance(val, pd.Series):
                ind_series.append(val)

    def _has_nan(trade_start: pd.Timestamp, trade_end: pd.Timestamp) -> bool:
        for s in ind_series:
            try:
                sl = s.loc[trade_start:trade_end]
                if sl.isna().any():
                    return True
            except Exception:
                return True
        return False

    target_dur = max_bars * 0.4  # aim for 40% of window being the trade
    records["_score"] = (
        records["_dur_min"].sub(target_dur).abs() / max(target_dur, 1.0)
        - records["PnL"].abs().rank(pct=True)
    )
    records = records.sort_values("_score")

    for _, trade in records.iterrows():
        entry = trade[entry_col]
        exit_ = trade[exit_col]
        if _has_nan(entry, exit_):
            continue
        dur = trade["_dur_min"]
        pad = pd.Timedelta(minutes=max(max_bars * 0.3, dur * 1.0))
        win_start = entry - pad
        win_end = exit_ + pad
        # Make sure the window is clamped to the data index so nothing
        # is sliced out of bounds.
        try:
            data_idx = pf.wrapper.index
            if win_start < data_idx[0]:
                win_start = data_idx[0]
            if win_end > data_idx[-1]:
                win_end = data_idx[-1]
        except Exception:
            pass
        return win_start, win_end

    return None, None


def plot_monthly_heatmap(
    pf: vbt.Portfolio,
    title: str = "Monthly Returns (%)",
) -> go.Figure:
    """Create a year x month heatmap of portfolio returns using native VBT.

    For multi-column portfolios the returns are averaged across columns
    so the heatmap shows the aggregate strategy return per month.
    """
    pf_daily = pf.resample("1D")
    mo_rets = pf_daily.resample("ME").returns
    # For multi-column portfolios, aggregate to 1D by averaging columns
    if isinstance(mo_rets, pd.DataFrame):
        mo_rets = mo_rets.mean(axis=1)
    mo_matrix = pd.Series(
        mo_rets.values,
        index=pd.MultiIndex.from_arrays(
            [mo_rets.index.year, mo_rets.index.month],
            names=["year", "month"],
        ),
    ).unstack("month")
    mo_matrix.columns = [calendar.month_abbr[m] for m in mo_matrix.columns]
    fig = mo_matrix.vbt.heatmap(
        is_x_category=True,
        trace_kwargs=dict(
            zmid=0,
            colorscale="RdYlGn",
            text=np.round(mo_matrix.values * 100, 1),
            texttemplate="%{text}%",
        ),
    )
    _apply_title_layout(fig, title)
    make_fullscreen(fig)
    return fig


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
# PORTFOLIO SUMMARY (enhanced composite subplots)
# ═══════════════════════════════════════════════════════════════════════


def plot_portfolio_summary(
    pf: vbt.Portfolio,
    title: str = "Portfolio Summary",
    height: int | None = None,
) -> go.Figure:
    """Composite VBT subplot figure with cumulative returns, drawdowns,
    underwater, and trade P&L. Resamples to daily for fast rendering.

    For multi-column portfolios, falls back to the first column since
    VBT's ``cumulative_returns`` / ``trade_pnl`` subplots require a
    single column.
    """
    pf_single, col_label = _pick_first_column(pf)
    pf_daily = pf_single.resample("1D")
    try:
        fig = pf_daily.plot(
            subplots=["cumulative_returns", "drawdowns", "underwater", "trade_pnl"],
        )
    except Exception:
        # Degraded fallback: just cumulative returns + drawdowns
        fig = pf_daily.plot(subplots=["cumulative_returns", "drawdowns"])
    if col_label:
        title = f"{title} — {col_label}"
    _apply_title_layout(fig, title)
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
# TRADE REPORT (text-based stats)
# ═══════════════════════════════════════════════════════════════════════


def _stats_to_rows(stats: pd.Series) -> list[list[str]]:
    """Convert a ``pd.Series`` of stats to ``[[label, value], ...]`` rows
    with nicely formatted values (durations, percentages, floats).
    """
    rows: list[list[str]] = []
    for label, val in stats.items():
        if val is None:
            rows.append([str(label), "—"])
            continue
        if isinstance(val, pd.Timedelta):
            total_min = int(val.total_seconds() / 60)
            if total_min < 60:
                rows.append([str(label), f"{total_min} min"])
            elif total_min < 1440:
                rows.append([str(label), f"{total_min / 60:.1f} h"])
            else:
                rows.append([str(label), f"{total_min / 1440:.1f} d"])
            continue
        if isinstance(val, pd.Timestamp):
            rows.append([str(label), val.strftime("%Y-%m-%d %H:%M")])
            continue
        try:
            f = float(val)
        except (TypeError, ValueError):
            rows.append([str(label), str(val)])
            continue
        if np.isnan(f):
            rows.append([str(label), "—"])
            continue
        if abs(f) >= 10_000:
            rows.append([str(label), f"{f:,.2f}"])
        else:
            rows.append([str(label), f"{f:.2f}"])
    return rows


def _box_title(title: str, width: int = 78) -> str:
    """Return a Unicode-boxed title header line block."""
    title = title.strip()
    inner = f"  {title}  "
    pad = max(width - len(inner) - 2, 0)
    top = "╔" + "═" * (width - 2) + "╗"
    mid = "║" + inner + " " * pad + "║"
    bot = "╠" + "═" * (width - 2) + "╣"
    return f"{top}\n{mid}\n{bot}"


def build_trade_report(pf: vbt.Portfolio, name: str = "Strategy") -> str:
    """Build a text report with portfolio stats and trade stats,
    rendered as aligned tabulate boxes with section headers.

    ``pf.returns_stats()`` is skipped to avoid the heavy returns
    computation.
    """
    from tabulate import tabulate

    sections: list[str] = []
    sections.append(_box_title(f"{name} — Backtest Report"))

    # ---- Portfolio stats ----
    try:
        stats = pf.stats()
        if isinstance(stats, pd.DataFrame):
            stats = stats.iloc[:, 0]
        rows = _stats_to_rows(stats)
        sections.append("\n  ── Portfolio Stats ──")
        sections.append(
            tabulate(rows, headers=["Metric", "Value"], tablefmt="rounded_outline")
        )
    except Exception as e:
        sections.append(f"\n  [portfolio stats failed: {e}]")

    # ---- Trade stats ----
    trade_count = pf.trades.count()
    if isinstance(trade_count, pd.Series):
        has_trades = bool((trade_count > 0).any())
    else:
        has_trades = trade_count > 0

    if has_trades:
        try:
            ts = pf.trades.stats()
            if isinstance(ts, pd.DataFrame):
                ts = ts.iloc[:, 0]
            rows = _stats_to_rows(ts)
            sections.append("\n  ── Trade Stats ──")
            sections.append(
                tabulate(rows, headers=["Metric", "Value"], tablefmt="rounded_outline")
            )
        except Exception as e:
            sections.append(f"\n  [trade stats failed: {e}]")

        # Quick win/loss summary directly from trade records
        try:
            pnls = np.asarray(pf.trades.pnl.values).ravel()
            pnls = pnls[~np.isnan(pnls)]
            if pnls.size:
                wins = pnls[pnls > 0]
                losses = pnls[pnls < 0]
                extra = [
                    ["Mean PnL", f"{pnls.mean():,.2f}"],
                    ["Median PnL", f"{np.median(pnls):,.2f}"],
                    ["Std PnL", f"{pnls.std():,.2f}"],
                    ["Largest Win", f"{pnls.max():,.2f}"],
                    ["Largest Loss", f"{pnls.min():,.2f}"],
                    ["Avg Win", f"{wins.mean():,.2f}" if wins.size else "—"],
                    ["Avg Loss", f"{losses.mean():,.2f}" if losses.size else "—"],
                    ["Win / Loss ratio",
                     f"{abs(wins.mean() / losses.mean()):.2f}"
                     if wins.size and losses.size else "—"],
                ]
                sections.append("\n  ── Trade Distribution ──")
                sections.append(
                    tabulate(extra, headers=["Metric", "Value"], tablefmt="rounded_outline")
                )
        except Exception:
            pass

    return "\n".join(sections)


# ═══════════════════════════════════════════════════════════════════════
# OVERLAY RESOLUTION (PlotConfig -> concrete Series dict)
# ═══════════════════════════════════════════════════════════════════════


def resolve_overlays(
    spec: Any,  # StrategySpec (avoid circular import)
    raw: pd.DataFrame,
    ind_result: Any,
    prepared: dict[str, Any] | None = None,
) -> dict[str, tuple[pd.Series, str | None, str | None]]:
    """Resolve ``PlotConfig.overlays`` into ``{label: (series, color, dash)}``.

    Uses the same ``"ind.<name>"`` / ``"data.<col>"`` / ``"pre.<name>"``
    convention as ``signal_args_map``.
    """
    if prepared is None:
        prepared = {}
    result = {}
    for overlay in spec.plot_config.overlays:
        prefix, _, name = overlay.source.partition(".")
        if prefix == "ind":
            values = getattr(ind_result, name).values
        elif prefix == "data":
            values = raw[name].values
        elif prefix == "pre":
            values = prepared[name]
        else:
            continue
        series = pd.Series(values, index=raw.index, name=overlay.label)
        result[overlay.label] = (series, overlay.color, overlay.dash)
    return result


# ===================================================================
# QUANTSTATS HTML TEARSHEET
# ===================================================================


def generate_html_tearsheet(
    pf: vbt.Portfolio,
    output_path: str = "results/tearsheet.html",
    title: str = "Strategy Tearsheet",
) -> str | None:
    """Generate a full QuantStats HTML tearsheet.

    Returns the output path on success, None on failure.
    """
    try:
        pf_daily = pf.resample("1D")
        pf_daily.qs.html_report(
            output=output_path,
            title=title,
            periods_per_year=252,
        )
        print(f"  Tearsheet saved to: {output_path}")
        return output_path
    except Exception as e:
        print(f"  Tearsheet generation failed: {e}")
        return None


# ===================================================================
# RETURNS DISTRIBUTION ANALYTICS
# ===================================================================


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


# ===================================================================
# EXTENDED STATS (text output)
# ===================================================================


def _total_trade_count(pf: vbt.Portfolio) -> int:
    """Return total number of trades across all columns/groups."""
    cnt = pf.trades.count()
    if hasattr(cnt, "sum"):
        try:
            return int(cnt.sum())
        except Exception:
            pass
    try:
        return int(cnt)
    except Exception:
        return 0


def print_extended_stats(pf: vbt.Portfolio, name: str = "Strategy") -> None:
    """Print comprehensive stats: portfolio, returns, trades, drawdowns.

    Handles multi-column / grouped portfolios: uses ``.sum()`` on
    trade counts and shows the per-column table rather than crashing
    on ambiguous Series truthiness.
    """
    print(f"\n{'=' * 60}")
    print(f"  {name} — Extended Statistics")
    print(f"{'=' * 60}")

    print(f"\n--- Portfolio Stats ---")
    print(pf.stats().to_string())

    print(f"\n--- Returns Stats ---")
    print(pf.returns_stats().to_string())

    n_trades = _total_trade_count(pf)
    if n_trades > 0:
        print(f"\n--- Trade Stats ---")
        print(pf.trades.stats().to_string())

        # Per-column aggregated trade metrics (works for single or multi-col)
        try:
            pnls = np.asarray(pf.trades.pnl.values).ravel()
            pnls = pnls[~np.isnan(pnls)]
            if pnls.size:
                print(f"\n--- Trade Distribution ---")
                print(f"  Mean PnL: {pnls.mean():.2f}")
                print(f"  Median PnL: {np.median(pnls):.2f}")
                print(f"  Skew: {pd.Series(pnls).skew():.2f}")
                print(f"  Kurtosis: {pd.Series(pnls).kurtosis():.2f}")
        except Exception as e:
            print(f"  (trade distribution skipped: {e})")

    try:
        dd = pf.drawdowns
        dd_count = dd.count()
        if hasattr(dd_count, "sum"):
            dd_count = int(dd_count.sum())
        if dd_count > 0:
            print(f"\n--- Drawdown Stats ---")
            print(dd.stats().to_string())
    except Exception:
        pass


# ===================================================================
# TRADE DURATION ANALYSIS
# ===================================================================


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


# ===================================================================
# DRAWDOWN ANALYSIS
# ===================================================================


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


# ===================================================================
# MULTI-STRATEGY COMPARISON
# ===================================================================


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
        hovermode="x unified",
    )
    return fig


# ===================================================================
# FULL REPORT GENERATOR
# ===================================================================


def generate_full_report(
    pf: vbt.Portfolio,
    name: str = "Strategy",
    output_dir: str = "results",
) -> None:
    """Generate all analytics: text stats, plots, and HTML tearsheet.

    Saves HTML tearsheet to output_dir/ and shows plots in browser.
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    # 1. Extended text stats
    print_extended_stats(pf, name)

    # 2. HTML tearsheet
    tearsheet_path = os.path.join(output_dir, f"{name}_tearsheet.html")
    generate_html_tearsheet(pf, tearsheet_path, name)

    # 3. Plots
    plots = [
        (plot_portfolio_summary, f"{name} — Portfolio Summary"),
        (plot_monthly_heatmap, f"{name} — Monthly Returns"),
        (plot_returns_distribution, f"{name} — Returns Distribution"),
        (plot_trade_analysis, f"{name} — Trade Analysis"),
        (plot_trade_duration, f"{name} — Trade Duration"),
        (plot_drawdown_analysis, f"{name} — Drawdown Analysis"),
        (plot_rolling_sharpe, f"{name} — Rolling Sharpe"),
    ]

    for plot_fn, plot_title in plots:
        try:
            fig = plot_fn(pf, title=plot_title)
            show_browser(fig)
        except Exception as e:
            print(f"  Plot '{plot_title}' failed: {e}")


# ═══════════════════════════════════════════════════════════════════════
# PARAMETER GRID PLOTS (2D heatmaps + 3D volumes, VBT native)
# ═══════════════════════════════════════════════════════════════════════


def _resolve_level_name(index: pd.Index, param: str) -> str:
    """Resolve a level name inside a MultiIndex, handling VBT's prefixes.

    VBT prefixes indicator parameters with the short name
    (e.g. ``"mrmacro_bb_window"`` for ``bb_window``). This function
    accepts either the raw or prefixed form and returns the actual
    level name present on *index*.
    """
    names = list(index.names) if isinstance(index, pd.MultiIndex) else [index.name]
    if param in names:
        return param
    for n in names:
        if n and n.endswith("_" + param):
            return n
        if n and n.endswith(param) and n != param:
            return n
    raise ValueError(
        f"Cannot find level matching {param!r} in index names {names}"
    )


def _text_matrix(series: pd.Series, x_level: str, y_level: str) -> np.ndarray:
    """Return a 2D numeric matrix (y × x) for annotation of a heatmap."""
    try:
        mat = series.unstack(x_level)
        return np.round(mat.values.astype(float), 2)
    except Exception:
        return np.array([[]])


def plot_param_heatmap(
    perf: pd.Series,
    x_param: str,
    y_param: str,
    title: str = "Parameter Heatmap",
    metric_name: str = "metric",
    colorscale: str = "RdYlGn",
    zmid: float | None = 0.0,
    aggregate: str = "mean",
) -> go.Figure:
    """Static 2D heatmap of a metric across two parameters.

    Uses VBT Pro native :meth:`Series.vbt.heatmap`. If *perf* has extra
    MultiIndex levels beyond (*x_param*, *y_param*), they are reduced
    via ``aggregate`` (``"mean"``, ``"median"``, ``"min"``, ``"max"``).

    Parameters
    ----------
    perf
        Metric values indexed by parameter combinations (typically a
        ``pd.Series`` returned by ``getattr(pf, metric)`` on a
        multi-column portfolio, or by ``runner.cv(...)``).
    x_param, y_param
        Parameter names to place on the x and y axes. Matched against
        MultiIndex level names, tolerating VBT's short-name prefix.
    metric_name
        Human label used for the colorbar title.
    zmid
        Value to center the diverging colorscale on (0 for Sharpe, etc.).
        Set to ``None`` to disable centring.
    aggregate
        Aggregation function over non-axis levels.
    """
    x_level = _resolve_level_name(perf.index, x_param)
    y_level = _resolve_level_name(perf.index, y_param)

    other = [n for n in perf.index.names if n not in (x_level, y_level)]
    if other:
        agg_fn = {"mean": "mean", "median": "median", "min": "min", "max": "max"}[aggregate]
        perf_agg = getattr(perf.groupby([y_level, x_level]), agg_fn)()
    else:
        perf_agg = perf

    text = _text_matrix(perf_agg, x_level, y_level)
    trace_kwargs: dict[str, Any] = {
        "colorscale": colorscale,
        "colorbar": dict(title=metric_name),
        "text": text,
        "texttemplate": "%{text}",
        "hovertemplate": (
            f"{x_param}=%{{x}}<br>{y_param}=%{{y}}<br><b>{metric_name}</b>: %{{z:.2f}}"
            "<extra></extra>"
        ),
    }
    if zmid is not None:
        trace_kwargs["zmid"] = zmid

    fig = perf_agg.vbt.heatmap(
        x_level=x_level,
        y_level=y_level,
        trace_kwargs=trace_kwargs,
    )
    fig.update_layout(
        title=title,
        xaxis_title=x_param,
        yaxis_title=y_param,
    )
    make_fullscreen(fig)
    return fig


def plot_param_heatmap_slider(
    grid_perf: pd.Series,
    x_param: str,
    y_param: str,
    slider_level: str = "split",
    set_filter: str | int | None = "train",
    title: str = "Per-Split Parameter Heatmap",
    metric_name: str = "metric",
    colorscale: str = "RdYlGn",
    zmid: float | None = 0.0,
) -> go.Figure:
    """2D parameter heatmap with a slider across splits.

    Uses VBT Pro native :meth:`Series.vbt.heatmap` with the
    ``slider_level`` kwarg, producing an interactive chart where the
    user scrubs through splits to inspect parameter stability.

    If *grid_perf* has a ``"set"`` level (train/test), ``set_filter``
    selects which slice to display.
    """
    x_level = _resolve_level_name(grid_perf.index, x_param)
    y_level = _resolve_level_name(grid_perf.index, y_param)

    if "set" in grid_perf.index.names and set_filter is not None:
        set_vals = grid_perf.index.get_level_values("set")
        if set_filter == "train":
            mask = set_vals.isin(["train", "set_0", 0])
        elif set_filter == "test":
            mask = set_vals.isin(["test", "set_1", 1])
        else:
            mask = set_vals == set_filter
        grid_perf = grid_perf[mask]
        # Drop the "set" level to simplify the MultiIndex
        grid_perf = grid_perf.droplevel("set")

    trace_kwargs: dict[str, Any] = {
        "colorscale": colorscale,
        "colorbar": dict(title=metric_name),
        "hovertemplate": (
            f"{x_param}=%{{x}}<br>{y_param}=%{{y}}<br><b>{metric_name}</b>: %{{z:.2f}}"
            "<extra></extra>"
        ),
    }
    if zmid is not None:
        trace_kwargs["zmid"] = zmid

    fig = grid_perf.vbt.heatmap(
        x_level=x_level,
        y_level=y_level,
        slider_level=slider_level,
        trace_kwargs=trace_kwargs,
    )
    fig.update_layout(
        title=title,
        xaxis_title=x_param,
        yaxis_title=y_param,
    )
    make_fullscreen(fig)
    return fig


def plot_param_volume(
    grid_perf: pd.Series,
    x_param: str,
    y_param: str,
    z_param: str,
    set_filter: str | int | None = "train",
    title: str = "3D Parameter Volume",
    metric_name: str = "metric",
    colorscale: str = "RdYlGn",
    aggregate: str = "mean",
) -> go.Figure:
    """3D volume plot over three parameter/split dimensions.

    Uses VBT Pro native :meth:`Series.vbt.volume`, which renders a
    ``plotly.graph_objects.Volume`` with the three chosen levels on
    x/y/z axes and the metric value as the scalar field. Ideal for
    visualising how the best parameter combination evolves across
    splits: pass e.g. ``x_param="bb_window", y_param="bb_alpha",
    z_param="split"``.
    """
    x_level = _resolve_level_name(grid_perf.index, x_param)
    y_level = _resolve_level_name(grid_perf.index, y_param)
    z_level = _resolve_level_name(grid_perf.index, z_param)

    if "set" in grid_perf.index.names and set_filter is not None:
        set_vals = grid_perf.index.get_level_values("set")
        if set_filter == "train":
            mask = set_vals.isin(["train", "set_0", 0])
        elif set_filter == "test":
            mask = set_vals.isin(["test", "set_1", 1])
        else:
            mask = set_vals == set_filter
        grid_perf = grid_perf[mask].droplevel("set")

    other = [
        n for n in grid_perf.index.names if n not in (x_level, y_level, z_level)
    ]
    if other:
        agg_fn = {"mean": "mean", "median": "median", "min": "min", "max": "max"}[aggregate]
        grid_perf = getattr(
            grid_perf.groupby([x_level, y_level, z_level]), agg_fn
        )()

    trace_kwargs: dict[str, Any] = {
        "colorscale": colorscale,
        "colorbar": dict(title=metric_name),
        "opacity": 0.3,
        "surface_count": 17,
        "caps": dict(x_show=False, y_show=False, z_show=False),
    }
    fig = grid_perf.vbt.volume(
        x_level=x_level,
        y_level=y_level,
        z_level=z_level,
        trace_kwargs=trace_kwargs,
    )
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title=x_param,
            yaxis_title=y_param,
            zaxis_title=z_param,
        ),
    )
    make_fullscreen(fig)
    return fig


def plot_param_surface(
    perf: pd.Series,
    x_param: str,
    y_param: str,
    title: str = "Parameter Surface (3D)",
    metric_name: str = "metric",
    colorscale: str = "RdYlGn",
    aggregate: str = "mean",
) -> go.Figure:
    """3D surface of a metric across two parameters.

    Complement to :func:`plot_param_heatmap` — same data, different
    perspective. Useful to spot peaks vs plateaus that are harder to
    see in a flat heatmap.
    """
    x_level = _resolve_level_name(perf.index, x_param)
    y_level = _resolve_level_name(perf.index, y_param)

    other = [n for n in perf.index.names if n not in (x_level, y_level)]
    if other:
        agg_fn = {"mean": "mean", "median": "median", "min": "min", "max": "max"}[aggregate]
        perf = getattr(perf.groupby([y_level, x_level]), agg_fn)()

    mat = perf.unstack(x_level)
    fig = go.Figure(
        data=go.Surface(
            z=mat.values,
            x=mat.columns.values,
            y=mat.index.values,
            colorscale=colorscale,
            colorbar=dict(title=metric_name),
            hovertemplate=(
                f"{x_param}=%{{x}}<br>{y_param}=%{{y}}<br><b>{metric_name}</b>: %{{z:.2f}}"
                "<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title=x_param,
            yaxis_title=y_param,
            zaxis_title=metric_name,
        ),
    )
    make_fullscreen(fig)
    return fig


# ═══════════════════════════════════════════════════════════════════════
# NATIVE VBT PLOTS: ORDERS / EXPOSURE / ALLOCATIONS
# ═══════════════════════════════════════════════════════════════════════


def _slice_pf_last(pf: vbt.Portfolio, max_bars: int) -> vbt.Portfolio:
    """Return a portfolio sliced to the last *max_bars* rows.

    Essential on minute-frequency data: embedding a 3M-bar price series
    in an HTML plot produces hundred-megabyte files. We slice the
    portfolio itself rather than passing ``sim_start`` so every
    downstream accessor (``orders``, ``trades``) operates on the
    reduced window.
    """
    if len(pf.wrapper.index) <= max_bars:
        return pf
    start = pf.wrapper.index[-max_bars]
    return pf.loc[start:]


def _pick_first_column(pf: vbt.Portfolio) -> tuple[vbt.Portfolio, str | None]:
    """For multi-column portfolios, return a single-column slice.

    VBT's trade / order / MAE / MFE plots only accept a single column.
    When faced with a grouped / multi-asset portfolio, this helper
    falls back to the first column so the plot still works. Returns
    ``(pf_single, label)`` where *label* is the selected column name
    to append to the chart title (``None`` if the input was already
    single-column).
    """
    try:
        n_cols = pf.wrapper.shape_2d[1]
    except Exception:
        return pf, None
    if n_cols <= 1:
        return pf, None

    columns = pf.wrapper.columns
    first_col = columns[0]
    # Try to ungroup first if grouped (cash_sharing portfolios need this)
    try:
        pf_ungrouped = pf.regroup(False)
    except Exception:
        pf_ungrouped = pf
    try:
        return pf_ungrouped[first_col], str(first_col)
    except Exception:
        try:
            return pf_ungrouped.iloc[:, 0], str(first_col)
        except Exception:
            return pf, None


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


# ═══════════════════════════════════════════════════════════════════════
# STATIC PARAMETER GRID RUNNER (no splits, full dataset)
# ═══════════════════════════════════════════════════════════════════════


def compute_static_param_grid(
    runner: Any,  # StrategyRunner (late import to avoid cycle)
    param_grid: dict[str, list],
    metric: str = "sharpe_ratio",
) -> pd.Series:
    """Run a parameter grid on the full dataset (no CV) and return metric.

    Mirrors :meth:`StrategyRunner.cv` but without the splitter: one
    multi-column portfolio is built over the whole dataset and the
    requested *metric* is pulled out as a Series indexed by the
    parameter combinations. Used by :func:`plot_param_heatmap` and
    friends to build *static* heatmaps / surfaces covering the entire
    period rather than per-split snapshots.

    Parameters
    ----------
    runner
        A :class:`StrategyRunner` bound to the target data.
    param_grid
        Mapping of parameter names to lists of candidate values. Will
        be merged with the spec defaults.
    metric
        Name of a :class:`vbt.Portfolio` attribute to extract (e.g.
        ``"sharpe_ratio"``, ``"sortino_ratio"``, ``"annualized_return"``).
    """
    params = runner.spec.default_params()
    params.update(param_grid)

    prepared = runner._run_prepare(runner.raw, runner.data)
    ind = runner._run_indicator(
        runner.raw,
        runner.index_ns,
        params,
        parallel=True,
        prepared=prepared,
    )
    pf = runner._run_portfolio(
        runner.raw,
        runner.index_ns,
        ind,
        params,
        parallel=True,
        prepared=prepared,
    )
    values = getattr(pf, metric)
    if isinstance(values, pd.Series):
        return values
    # Scalar (single combo) → wrap in a Series
    return pd.Series([float(values)], index=pd.Index(["default"], name="combo"))


# ═══════════════════════════════════════════════════════════════════════
# STANDALONE REPORT HELPERS (for strategies that bypass StrategyRunner)
# ═══════════════════════════════════════════════════════════════════════


def run_standalone_grid(
    backtest_fn: Any,
    data: Any,
    param_grid: dict[str, list],
    metric: str = "sharpe_ratio",
    fixed_params: dict[str, Any] | None = None,
    verbose: bool = True,
    backtest_multi_fn: Any = None,
) -> pd.Series:
    """Run a parameter grid and return a metric Series.

    Two execution modes:

    1. **Broadcasted Numba path** (preferred, 5-10× faster): pass a
       ``backtest_multi_fn`` that accepts lists for its sweepable
       parameters and returns a single multi-column ``vbt.Portfolio``
       built with ``Portfolio.from_signals(chunked="threadpool")``.
       The entire sweep runs as one Numba call.

    2. **Sequential fallback** (when ``backtest_multi_fn is None``):
       iterates over the Cartesian product, calling *backtest_fn*
       once per combination. Slower but works for strategies that
       cannot be easily broadcast (e.g. ``rsi_daily``, daily
       momentum variants with stateful signals).
    """
    import itertools
    import time

    fixed = fixed_params or {}
    keys = list(param_grid.keys())

    # ─── Broadcasted Numba-parallel path ─────────────────────────────
    if backtest_multi_fn is not None:
        if verbose:
            n_expected = 1
            for v in param_grid.values():
                n_expected *= len(v)
            print(
                f"  Running broadcasted grid: {n_expected} combos in ONE "
                f"multi-column Portfolio.from_signals(chunked='threadpool')"
            )
        t0 = time.time()
        pf = backtest_multi_fn(data, **fixed, **param_grid)
        series = getattr(pf, metric)
        if not isinstance(series, pd.Series):
            series = pd.Series([float(series)],
                               index=pd.Index(["default"], name="combo"),
                               name=metric)
        elapsed = time.time() - t0
        if verbose:
            print(f"  Done in {elapsed:.1f}s ({len(series)} values)")
        # VBT prefixes param names (e.g. "bbands_window"). Rename to
        # match the user-provided grid keys so downstream heatmaps work.
        try:
            if isinstance(series.index, pd.MultiIndex):
                new_names = []
                for lvl in series.index.names:
                    matched = None
                    for k in keys:
                        if lvl and (lvl == k or lvl.endswith("_" + k) or lvl.endswith(k)):
                            matched = k
                            break
                    new_names.append(matched or lvl)
                series.index = series.index.set_names(new_names)
        except Exception:
            pass
        series.name = metric
        return series

    # ─── Sequential fallback ─────────────────────────────────────────
    combos = list(itertools.product(*(param_grid[k] for k in keys)))
    n = len(combos)
    if verbose:
        print(
            f"  Running sequential grid: {n} combos × 1 backtest each "
            f"(no backtest_multi_fn supplied)"
        )

    values: list[float] = []
    tuples: list[tuple] = []
    t0 = time.time()
    for i, combo in enumerate(combos, 1):
        params = dict(zip(keys, combo))
        pf = backtest_fn(data, **fixed, **params)
        val = getattr(pf, metric)
        if hasattr(val, "item"):
            try:
                val = val.item()
            except Exception:
                val = float(val)
        values.append(float(val))
        tuples.append(combo)
        if verbose and (i % max(1, n // 10) == 0 or i == n):
            elapsed = time.time() - t0
            eta = elapsed / i * (n - i)
            print(
                f"    [{i:>3}/{n}] {params} → {metric}={float(val):+.3f} "
                f"(elapsed {elapsed:.0f}s, eta {eta:.0f}s)"
            )
    idx = pd.MultiIndex.from_tuples(tuples, names=keys)
    return pd.Series(values, index=idx, name=metric)


def generate_single_run_plots(
    pf: vbt.Portfolio,
    name: str = "Strategy",
    output_dir: str | None = None,
    show: bool = False,
    skip: tuple[str, ...] = (),
) -> dict[str, go.Figure]:
    """Generate every single-run plot in the framework for a given portfolio.

    Produces a dict ``{plot_key: figure}``. If *output_dir* is given,
    each figure is additionally saved as a fullscreen HTML file under
    ``{output_dir}/{plot_key}.html``. If *show* is true, every figure
    is opened in the browser via :func:`show_browser`.

    Use *skip* to exclude individual plots by key.
    """
    plots_spec: list[tuple[str, Any, str]] = [
        ("portfolio_summary", plot_portfolio_summary, f"{name} — Portfolio Summary"),
        ("monthly_returns", plot_monthly_heatmap, f"{name} — Monthly Returns"),
        ("returns_distribution", plot_returns_distribution, f"{name} — Returns Distribution"),
        ("trade_analysis", plot_trade_analysis, f"{name} — Trade Analysis"),
        ("trade_duration", plot_trade_duration, f"{name} — Trade Duration"),
        ("drawdown", plot_drawdown_analysis, f"{name} — Drawdown Analysis"),
        ("rolling_sharpe", plot_rolling_sharpe, f"{name} — Rolling 1Y Sharpe"),
        ("orders_on_price", plot_orders_on_price, f"{name} — Orders on Price"),
        ("trades_on_price", plot_trades_on_price, f"{name} — Trades on Price"),
        ("exposure", plot_exposure, f"{name} — Gross Exposure"),
        ("value_and_cash", plot_value_and_cash, f"{name} — Value & Cash"),
        ("orders_frequency", plot_orders_heatmap, f"{name} — Trades by Hour/Weekday"),
    ]
    figures: dict[str, go.Figure] = {}
    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    for key, fn, title in plots_spec:
        if key in skip:
            continue
        try:
            fig = fn(pf, title=title)
        except Exception as e:
            print(f"  ⚠ {key} skipped: {type(e).__name__}: {e}")
            continue
        figures[key] = fig
        if output_dir:
            save_fullscreen_html(fig, f"{output_dir}/{key}.html")
        if show:
            show_browser(fig)
    return figures


def generate_param_grid_plots(
    perf: pd.Series,
    name: str = "Strategy",
    output_dir: str | None = None,
    metric_name: str = "Sharpe",
    show: bool = False,
) -> dict[str, go.Figure]:
    """Generate all parameter-grid plots from a standalone or CV sweep.

    Uses the first two levels of *perf*'s MultiIndex as the (x, y)
    axes. If a third level exists, a 3D volume is also produced with
    that level as the z-axis. A static heatmap and a 3D surface are
    always generated.
    """
    figures: dict[str, go.Figure] = {}
    if not isinstance(perf.index, pd.MultiIndex):
        print(f"  ⚠ generate_param_grid_plots: perf is not a MultiIndex, skipping")
        return figures

    level_names = [n for n in perf.index.names if n is not None]
    if len(level_names) < 2:
        print(
            f"  ⚠ generate_param_grid_plots: need ≥2 param levels, got {level_names}"
        )
        return figures

    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    x_param, y_param = level_names[0], level_names[1]

    # 2D static heatmap (aggregated over extra levels)
    try:
        fig_hm = plot_param_heatmap(
            perf,
            x_param=x_param,
            y_param=y_param,
            title=f"{name} — {metric_name} by ({x_param} × {y_param})",
            metric_name=metric_name,
        )
        figures["param_heatmap_static"] = fig_hm
    except Exception as e:
        print(f"  ⚠ param_heatmap_static skipped: {e}")

    # 3D parameter surface
    try:
        fig_surf = plot_param_surface(
            perf,
            x_param=x_param,
            y_param=y_param,
            title=f"{name} — {metric_name} surface ({x_param} × {y_param})",
            metric_name=metric_name,
        )
        figures["param_surface"] = fig_surf
    except Exception as e:
        print(f"  ⚠ param_surface skipped: {e}")

    # 3D volume if we have a third dimension
    if len(level_names) >= 3:
        z_param = level_names[2]
        try:
            fig_vol = plot_param_volume(
                perf,
                x_param=x_param,
                y_param=y_param,
                z_param=z_param,
                set_filter=None,
                title=f"{name} — {metric_name} volume ({x_param} × {y_param} × {z_param})",
                metric_name=metric_name,
            )
            figures["param_volume"] = fig_vol
        except Exception as e:
            print(f"  ⚠ param_volume skipped: {e}")

    # Per-"slider" heatmap if ≥3 dims (treat level 3 as slider axis)
    if len(level_names) >= 3:
        try:
            fig_slider = plot_param_heatmap_slider(
                perf,
                x_param=x_param,
                y_param=y_param,
                slider_level=level_names[2],
                set_filter=None,
                title=f"{name} — {metric_name} per {level_names[2]}",
                metric_name=metric_name,
            )
            figures["param_heatmap_slider"] = fig_slider
        except Exception as e:
            print(f"  ⚠ param_heatmap_slider skipped: {e}")

    if output_dir:
        for key, fig in figures.items():
            save_fullscreen_html(fig, f"{output_dir}/{key}.html")
    if show:
        for fig in figures.values():
            show_browser(fig)
    return figures


def generate_standalone_report(
    backtest_fn: Any,
    data: Any,
    name: str,
    param_grid: dict[str, list] | None = None,
    fixed_params: dict[str, Any] | None = None,
    output_dir: str | None = None,
    metric: str = "sharpe_ratio",
    metric_name: str = "Sharpe",
    show: bool = True,
    print_stats: bool = True,
    backtest_multi_fn: Any = None,
) -> dict[str, Any]:
    """One-call entry point for standalone strategy scripts.

    Runs a single backtest with *fixed_params*, prints stats and saves
    all single-run plots. If *param_grid* is provided, additionally
    runs a grid sweep via :func:`run_standalone_grid` and saves
    parameter heatmaps / surface / volume plots.

    When *backtest_multi_fn* is provided, the grid sweep runs as a
    single multi-column ``Portfolio.from_signals(chunked="threadpool")``
    call — the full pipeline is fully Numba-parallel instead of a
    Python for-loop. This typically yields a 5-10× speedup on FX
    minute data.

    Parameters
    ----------
    backtest_fn
        Scalar-param backtest function: ``backtest_fn(data, **params) -> vbt.Portfolio``.
    backtest_multi_fn
        Optional broadcasted variant accepting lists for the sweepable
        parameters, returning a multi-column ``vbt.Portfolio``. When
        present it is preferred over *backtest_fn* for grid execution.
    data
        The :class:`vbt.Data` object (or pandas input) passed as the
        first argument to the backtest.
    name
        Human label used in plot titles and filenames.
    param_grid
        Optional mapping of parameter names to value lists for the sweep.
    fixed_params
        Parameters held constant in both the single run and the grid
        sweep (e.g. ``{"leverage": 3.0}``).
    output_dir
        Directory to save fullscreen HTML plots. Created if missing.
    metric
        :class:`vbt.Portfolio` attribute to extract in the grid sweep.
    metric_name
        Human label for the metric colorbar.
    show
        If true, open every figure in the browser.

    Returns
    -------
    dict
        ``{"pf": single-run portfolio, "grid_perf": grid Series or None,
        "figures": all generated figures keyed by plot name}``.
    """
    fixed = fixed_params or {}

    print(f"\n{'=' * 60}\n{name} — Standalone Report\n{'=' * 60}")
    print(f"  Running single backtest with {fixed or 'defaults'} ...")
    pf = backtest_fn(data, **fixed)

    if print_stats:
        print_extended_stats(pf, name)

    # HTML tearsheet (daily-resampled QuantStats)
    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        generate_html_tearsheet(
            pf,
            output_path=f"{output_dir}/tearsheet.html",
            title=f"{name} — Tearsheet",
        )

    print(f"\n  Generating single-run plots ...")
    figures = generate_single_run_plots(
        pf, name=name, output_dir=output_dir, show=show
    )

    grid_perf: pd.Series | None = None
    if param_grid:
        print(f"\n  Running standalone parameter grid ...")
        grid_perf = run_standalone_grid(
            backtest_fn=backtest_fn,
            backtest_multi_fn=backtest_multi_fn,
            data=data,
            param_grid=param_grid,
            metric=metric,
            fixed_params=fixed,
        )
        # Persist grid values so users can inspect / plot later
        if output_dir:
            grid_perf.to_csv(f"{output_dir}/param_grid.csv", header=True)

        print(f"\n  Generating parameter-grid plots ...")
        grid_figures = generate_param_grid_plots(
            grid_perf,
            name=name,
            output_dir=output_dir,
            metric_name=metric_name,
            show=show,
        )
        figures.update(grid_figures)

    if output_dir:
        print(f"\n  ✔ All results saved under {output_dir}/")

    return {"pf": pf, "grid_perf": grid_perf, "figures": figures}


# ═══════════════════════════════════════════════════════════════════════
# TERMINAL PRETTY-PRINT HELPERS (grid-search + CV results)
# ═══════════════════════════════════════════════════════════════════════


def _format_metric_header(metric_name: str) -> str:
    """Convert a snake_case metric name into a Title Case header."""
    return metric_name.replace("_", " ").title()


def _format_cell(val: Any) -> str:
    """Always format numeric values with 2 decimal places."""
    if val is None:
        return "—"
    try:
        f = float(val)
    except (TypeError, ValueError):
        return str(val)
    if np.isnan(f):
        return "—"
    if abs(f) >= 10_000:
        return f"{f:,.2f}"
    return f"{f:.2f}"


def print_grid_results(
    grid: pd.Series,
    *,
    title: str = "Grid Search Results",
    metric_name: str = "metric",
    top_n: int = 20,
    ascending: bool = False,
) -> None:
    """Pretty-print a grid-search ``pd.Series`` to the terminal.

    Uses ``tabulate`` for a clean aligned layout. The index levels
    become the left-hand columns and the metric becomes the right-hand
    column. Shows a header with summary statistics (best/median/worst).
    """
    from tabulate import tabulate

    if not isinstance(grid, pd.Series) or len(grid) == 0:
        print(f"\n(empty {title})")
        return

    sorted_grid = grid.sort_values(ascending=ascending)
    head = sorted_grid.head(top_n)
    df = head.reset_index()
    metric_col = df.columns[-1]
    df.rename(columns={metric_col: _format_metric_header(metric_name)}, inplace=True)
    for col in df.columns[:-1]:
        df[col] = df[col].apply(
            lambda v: f"{v:g}" if isinstance(v, (int, float, np.floating)) else str(v)
        )
    df[df.columns[-1]] = df[df.columns[-1]].apply(_format_cell)

    bar = "═" * 78
    print(f"\n{bar}")
    print(f"  {title}")
    print(f"{bar}")
    print(
        f"  Combos: {len(grid)}   "
        f"Best: {_format_cell(sorted_grid.iloc[0])}   "
        f"Median: {_format_cell(sorted_grid.median())}   "
        f"Worst: {_format_cell(sorted_grid.iloc[-1])}"
    )
    print(f"  Top {min(top_n, len(grid))} combos by {metric_name}:")
    print()
    print(tabulate(df, headers="keys", tablefmt="rounded_outline", showindex=False))
    print()


def print_cv_results(
    grid_perf: pd.Series,
    best_perf: pd.Series | None = None,
    splitter: Any | None = None,
    *,
    title: str = "Cross-Validation Results",
    metric_name: str = "metric",
    top_n: int = 5,
) -> None:
    """Pretty-print CV grid + best-per-split results.

    Displays two sections:
      1. Best combo per (split, set) — typically best on train per fold.
      2. Aggregated ranking across splits (mean metric per combo).
    """
    from tabulate import tabulate

    if not isinstance(grid_perf, pd.Series) or len(grid_perf) == 0:
        print(f"\n(empty {title})")
        return

    names = list(grid_perf.index.names)
    has_split = "split" in names
    has_set = "set" in names

    date_labels: dict[int, str] = {}
    if splitter is not None and has_split:
        try:
            bounds = splitter.index_bounds
            for (split_i, set_i), row in bounds.iterrows():
                if set_i in ("test", 1):
                    start = pd.Timestamp(row["start"]).strftime("%Y-%m-%d")
                    end = pd.Timestamp(row["end"]).strftime("%Y-%m-%d")
                    date_labels[split_i] = f"{start} → {end}"
        except Exception:
            pass

    bar = "═" * 78
    print(f"\n{bar}")
    print(f"  {title}")
    print(f"{bar}")
    print(
        f"  Splits: {len(date_labels) or '?'}   "
        f"Combos: {len(grid_perf)}   "
        f"Metric: {metric_name}"
    )

    # -------- Section 1: best per split (from best_perf if provided) --------
    if best_perf is not None and len(best_perf) > 0 and has_split:
        print(f"\n  ── Best combo per split (test range) ──")
        rows = []
        bp = best_perf
        if has_set:
            try:
                bp = bp.xs("test", level="set")
            except (KeyError, ValueError):
                try:
                    bp = bp.xs(1, level="set")
                except Exception:
                    pass
        for idx, val in bp.items():
            if not isinstance(idx, tuple):
                idx = (idx,)
            split_val = idx[0]
            param_vals = idx[1:]
            param_names = [n for n in bp.index.names if n != "split"]
            date_str = date_labels.get(split_val, f"Split {split_val}")
            params_str = ", ".join(
                f"{n}={v}" for n, v in zip(param_names, param_vals)
            )
            rows.append([date_str, params_str, _format_cell(val)])
        print(
            tabulate(
                rows,
                headers=["Test Range", "Best Params", _format_metric_header(metric_name)],
                tablefmt="rounded_outline",
            )
        )

    # -------- Section 2: aggregated ranking across splits --------
    sweep_levels = [
        n for n in names if n not in ("split", "set")
    ]
    if sweep_levels:
        test_perf = grid_perf
        if has_set:
            try:
                test_perf = test_perf.xs("test", level="set")
            except (KeyError, ValueError):
                try:
                    test_perf = test_perf.xs(1, level="set")
                except Exception:
                    pass
        agg = test_perf.groupby(sweep_levels).agg(["mean", "std", "min", "max"])
        agg = agg.sort_values("mean", ascending=False).head(top_n)
        df = agg.reset_index()
        for col in sweep_levels:
            df[col] = df[col].apply(lambda v: f"{v:g}" if isinstance(v, (int, float, np.floating)) else str(v))
        for col in ["mean", "std", "min", "max"]:
            df[col] = df[col].apply(_format_cell)
        print(f"\n  ── Top {top_n} combos by mean test {metric_name} (across all splits) ──")
        print(tabulate(df, headers="keys", tablefmt="rounded_outline", showindex=False))
    print()
