"""Shared plotting utilities for strategy results.

Leverages VBT Pro native plotting methods for trade signals, portfolio
summaries, and trade-level analysis.
"""

from __future__ import annotations

import calendar
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import vectorbtpro as vbt
from plotly.subplots import make_subplots


# ═══════════════════════════════════════════════════════════════════════
# BROWSER RENDERING
# ═══════════════════════════════════════════════════════════════════════


def show_browser(fig: go.Figure) -> None:
    """Show a plotly figure in the default web browser."""
    fig.show(renderer="browser")


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
    """Create a year x month heatmap of portfolio returns using native VBT."""
    pf_daily = pf.resample("1D")
    mo_rets = pf_daily.resample("ME").returns
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
    underwater, and trade P&L. Resamples to daily for fast rendering."""
    pf_daily = pf.resample("1D")
    fig = pf_daily.plot(
        subplots=["cumulative_returns", "drawdowns", "underwater", "trade_pnl"],
    )
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


def print_extended_stats(pf: vbt.Portfolio, name: str = "Strategy") -> None:
    """Print comprehensive stats: portfolio, returns, trades, drawdowns."""
    print(f"\n{'=' * 60}")
    print(f"  {name} — Extended Statistics")
    print(f"{'=' * 60}")

    print(f"\n--- Portfolio Stats ---")
    print(pf.stats().to_string())

    print(f"\n--- Returns Stats ---")
    print(pf.returns_stats().to_string())

    if pf.trades.count() > 0:
        print(f"\n--- Trade Stats ---")
        print(pf.trades.stats().to_string())

        # Additional trade metrics
        trades = pf.trades
        pnls = trades.pnl.values
        print(f"\n--- Trade Distribution ---")
        print(f"  Mean PnL: {pnls.mean():.2f}")
        print(f"  Median PnL: {np.median(pnls):.2f}")
        print(f"  Skew: {pd.Series(pnls).skew():.3f}")
        print(f"  Kurtosis: {pd.Series(pnls).kurtosis():.3f}")

        durations = trades.duration.values
        dur_hours = durations.astype("timedelta64[m]").astype(float) / 60
        print(f"  Avg duration: {dur_hours.mean():.1f}h")
        print(f"  Median duration: {np.median(dur_hours):.1f}h")

    try:
        dd = pf.drawdowns
        if dd.count() > 0:
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
    height: int = 600,
) -> go.Figure:
    """Scatter PnL vs duration + duration histogram."""
    trades = pf.trades
    if trades.count() == 0:
        fig = go.Figure()
        fig.add_annotation(text="No trades", showarrow=False)
        return fig

    pnls = trades.pnl.values
    durations = trades.duration.values
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
    fig.update_layout(title=title, height=height)
    return fig


# ===================================================================
# DRAWDOWN ANALYSIS
# ===================================================================


def plot_drawdown_analysis(
    pf: vbt.Portfolio,
    title: str = "Drawdown Analysis",
    top_n: int = 5,
    height: int = 600,
) -> go.Figure:
    """Underwater plot with top N drawdowns highlighted."""
    pf_daily = pf.resample("1D")
    try:
        fig = pf_daily.plot(subplots=["drawdowns", "underwater"])
        fig.update_layout(title=title, height=height)
        return fig
    except Exception:
        # Fallback: manual underwater plot
        rets = pf_daily.returns
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
        fig.update_layout(title=title, height=height // 2)
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
