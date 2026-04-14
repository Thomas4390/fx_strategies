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

# ``_pick_first_column`` lives in ``_helpers`` so ``_trades`` and
# ``_core`` can share it without a circular import.
from ._helpers import _pick_first_column
from ._reports import print_extended_stats


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

    The legend is anchored to the bottom of the viewport so it never
    collides with the title bar above the plot area, and the hover mode
    is set to ``closest`` so hovering a curve shows only that curve's
    label rather than every trace at the same x-coordinate.
    """
    fig.update_layout(
        autosize=True,
        margin={"l": 60, "r": 40, "t": 110, "b": 110},
        legend={
            "orientation": "h",
            "yanchor": "top",
            "y": -0.12,
            "xanchor": "center",
            "x": 0.5,
            "bgcolor": "rgba(255,255,255,0.85)",
            "bordercolor": "rgba(120,120,120,0.3)",
            "borderwidth": 1,
        },
        hovermode="closest",
        hoverlabel=dict(
            bgcolor="rgba(255,255,255,0.95)",
            bordercolor="rgba(80,80,80,0.6)",
            font=dict(size=12),
        ),
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








def generate_full_report(
    pf: vbt.Portfolio,
    name: str = "Strategy",
    output_dir: str = "results",
) -> None:
    """Generate all analytics: text stats, plots, and HTML tearsheet.

    Saves HTML tearsheet to output_dir/ and shows plots in browser.
    """
    # Function-local imports avoid a circular dependency: ``_trades``
    # and ``_equity`` both import from ``_core``, so pulling their
    # functions at module load time would break the import graph.
    from ._equity import (
        plot_drawdown_analysis,
        plot_returns_distribution,
        plot_rolling_sharpe,
    )
    from ._trades import plot_trade_analysis, plot_trade_duration

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



