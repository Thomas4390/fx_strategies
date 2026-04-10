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
        margin={"l": 50, "r": 30, "t": 70, "b": 40},
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "center",
            "x": 0.5,
        },
    )
    # Clear any fixed pixel dimensions so CSS 100vh/100vw wins.
    fig.layout.height = None
    fig.layout.width = None
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
    fig.update_layout(title=title)
    make_fullscreen(fig)
    return fig


# ═══════════════════════════════════════════════════════════════════════
# TRADE SIGNALS (price + entry/exit markers + indicator overlays)
# ═══════════════════════════════════════════════════════════════════════


def plot_trade_signals(
    pf: vbt.Portfolio,
    title: str = "Trade Signals",
    overlays: dict[str, tuple[pd.Series, str | None, str | None]] | None = None,
    height: int | None = None,
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

    fig.update_layout(title=title)
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
    fig.update_layout(title=title)
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

    if col_label:
        title = f"{title} — {col_label}"
    fig.update_layout(title=title, showlegend=False)
    if height is not None:
        fig.layout.height = height
    else:
        make_fullscreen(fig)
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
            x=[f"Fold {s}" for s in split_vals], y=fold_sharpes.values
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
    title: str = "Rolling 1-Year Sharpe Ratio",
) -> go.Figure:
    """Rolling Sharpe on daily-resampled portfolio."""
    pf_daily = pf.resample("1D")
    rets = pf_daily.returns
    rolling_sr = rets.rolling(window).apply(
        lambda x: x.mean() / x.std() * np.sqrt(252) if x.std() > 0 else 0
    )
    fig = rolling_sr.vbt.plot()
    fig.add_hline(y=0, line_color="gray")
    fig.add_hline(
        y=1, line_dash="dot", line_color="green", annotation_text="Sharpe=1"
    )
    fig.update_layout(title=title, height=400)
    return fig


def plot_partial_dependence(
    sweep_sharpes: pd.Series,
    param_grid: dict[str, list],
    title: str = "Parameter Sensitivity",
) -> go.Figure:
    """Marginal mean Sharpe for each swept parameter."""
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
            ),
            row=1,
            col=i + 1,
        )

    fig.update_layout(title=title, height=350, showlegend=False)
    return fig


def plot_train_vs_test(
    grid_perf: pd.Series,
    title: str = "Train vs Test Sharpe (Overfitting Check)",
) -> go.Figure:
    """Scatter: train Sharpe (x) vs test Sharpe (y) per param combo."""
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
        )
    )
    fig.update_layout(
        title=title,
        height=500,
        xaxis_title="Train Sharpe",
        yaxis_title="Test Sharpe",
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════
# TRADE REPORT (text-based stats)
# ═══════════════════════════════════════════════════════════════════════


def build_trade_report(pf: vbt.Portfolio) -> str:
    """Build a text report with portfolio stats and trade stats.

    ``pf.returns_stats()`` used to be included here, but it triggers a
    heavy computation (full returns analysis) that is rarely consulted
    and visibly slows down single-run reports. Dropped intentionally.
    """
    sections = []

    sections.append(f"PORTFOLIO STATS\n{'-' * 40}")
    sections.append(pf.stats().to_string())

    trade_count = pf.trades.count()
    # Multi-column portfolios return a Series of per-column counts.
    if isinstance(trade_count, pd.Series):
        has_trades = bool((trade_count > 0).any())
    else:
        has_trades = trade_count > 0
    if has_trades:
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
    title: str = "Returns Distribution",
    height: int = 800,
) -> go.Figure:
    """3-panel returns analysis: histogram, QQ plot, monthly boxplot."""
    pf_daily = pf.resample("1D")
    daily_rets = pf_daily.returns.dropna()

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Daily Returns Histogram",
            "Monthly Returns Boxplot",
            "Cumulative Returns",
            "Returns ECDF",
        ),
    )

    # Histogram
    fig.add_trace(
        go.Histogram(
            x=daily_rets.values, nbinsx=80,
            name="Daily Returns", marker_color="#636EFA",
        ),
        row=1, col=1,
    )

    # Monthly boxplot
    mo_rets = pf_daily.resample("ME").returns
    months = mo_rets.index.month
    month_names = [
        calendar.month_abbr[m] for m in range(1, 13)
    ]
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

    # Cumulative returns
    cum_rets = (1 + daily_rets).cumprod() - 1
    fig.add_trace(
        go.Scatter(
            x=cum_rets.index, y=cum_rets.values,
            mode="lines", name="Cumulative",
            line=dict(color="#00CC96"),
        ),
        row=2, col=1,
    )

    # ECDF
    sorted_rets = np.sort(daily_rets.values)
    ecdf_y = np.arange(1, len(sorted_rets) + 1) / len(sorted_rets)
    fig.add_trace(
        go.Scatter(
            x=sorted_rets, y=ecdf_y,
            mode="lines", name="ECDF",
            line=dict(color="#EF553B"),
        ),
        row=2, col=2,
    )

    fig.update_layout(title=title, height=height, showlegend=False)
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
                print(f"  Skew: {pd.Series(pnls).skew():.3f}")
                print(f"  Kurtosis: {pd.Series(pnls).kurtosis():.3f}")
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
    dur_hours = durations.astype("timedelta64[m]").astype(float) / 60
    is_win = pnls > 0

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("PnL vs Duration", "Duration Distribution"),
    )

    # Scatter PnL vs duration (wins green, losses red)
    fig.add_trace(
        go.Scatter(
            x=dur_hours[is_win], y=pnls[is_win],
            mode="markers", name="Winners",
            marker=dict(color="#00CC96", size=6),
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=dur_hours[~is_win], y=pnls[~is_win],
            mode="markers", name="Losers",
            marker=dict(color="#EF553B", size=6),
        ),
        row=1, col=1,
    )

    # Duration histogram
    fig.add_trace(
        go.Histogram(
            x=dur_hours, nbinsx=30,
            name="Duration", marker_color="#636EFA",
        ),
        row=1, col=2,
    )

    fig.update_xaxes(title_text="Duration (hours)", row=1, col=1)
    fig.update_xaxes(title_text="Duration (hours)", row=1, col=2)
    fig.update_yaxes(title_text="PnL ($)", row=1, col=1)
    fig.update_layout(title=title)
    if height is not None:
        fig.layout.height = height
    else:
        make_fullscreen(fig)
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
            )
        )
    if col_label:
        title = f"{title} — {col_label}"
    fig.update_layout(title=title)
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
            f"{x_param}=%{{x}}<br>{y_param}=%{{y}}<br>{metric_name}=%{{z:.3f}}"
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
            f"{x_param}=%{{x}}<br>{y_param}=%{{y}}<br>{metric_name}=%{{z:.3f}}"
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
                f"{x_param}=%{{x}}<br>{y_param}=%{{y}}<br>{metric_name}=%{{z:.3f}}"
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
            mode="lines", name="Value",
            line=dict(color="#00CC96", width=2),
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=cash.index, y=cash.values,
            mode="lines", name="Cash",
            line=dict(color="#636EFA", width=2),
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
