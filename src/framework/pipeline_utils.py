"""Shared utilities for the pipeline / pipeline_nb / create_cv_pipeline pattern.

This module is the socle of the refactoring towards the ims_pipeline-style
architecture. Each strategy file in ``src/strategies/`` imports from here
and exposes three entry points:

- ``pipeline(data, **params) -> (pf, indicator)`` — investigation path,
  uses ``vbt.IF`` + ``vbt.Portfolio.from_signals``, returns a Portfolio
  and its Indicator (with ``.plot()`` overlay).
- ``pipeline_nb(...)`` — grid-search path, decorated with
  ``@vbt.parameterized`` + ``@njit(nogil=True)``, returns a scalar metric
  via ``compute_metric_nb``.
- ``create_cv_pipeline(splitter, metric_type, **defaults)`` — factory
  returning a function decorated with ``@vbt.cv_split`` + ``@njit``.

The constants and dispatch below mirror ``example/ims_pipeline.py`` but
are adapted for FX minute-frequency data (24h market, annualization via
``year_freq = 24h * 252``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import vectorbtpro as vbt
from numba import njit

# ═══════════════════════════════════════════════════════════════════════
# METRIC CONSTANTS (Numba-safe integer dispatch)
# ═══════════════════════════════════════════════════════════════════════

TOTAL_RETURN = 0
SHARPE_RATIO = 1
CALMAR_RATIO = 2
SORTINO_RATIO = 3
OMEGA_RATIO = 4
ANNUALIZED_RETURN = 5
MAX_DRAWDOWN = 6
PROFIT_FACTOR = 7
VALUE_AT_RISK = 8
TAIL_RATIO = 9
ANNUALIZED_VOLATILITY = 10
INFORMATION_RATIO = 11
DOWNSIDE_RISK = 12
COND_VALUE_AT_RISK = 13

METRIC_NAMES: dict[int, str] = {
    TOTAL_RETURN: "total_return",
    SHARPE_RATIO: "sharpe_ratio",
    CALMAR_RATIO: "calmar_ratio",
    SORTINO_RATIO: "sortino_ratio",
    OMEGA_RATIO: "omega_ratio",
    ANNUALIZED_RETURN: "annualized_return",
    MAX_DRAWDOWN: "max_drawdown",
    PROFIT_FACTOR: "profit_factor",
    VALUE_AT_RISK: "value_at_risk",
    TAIL_RATIO: "tail_ratio",
    ANNUALIZED_VOLATILITY: "annualized_volatility",
    INFORMATION_RATIO: "information_ratio",
    DOWNSIDE_RISK: "downside_risk",
    COND_VALUE_AT_RISK: "cond_value_at_risk",
}

METRIC_NAME_TO_ID: dict[str, int] = {v: k for k, v in METRIC_NAMES.items()}

# FX minute default: 24h × 252 = 362880 bars/year
FX_MINUTE_ANN_FACTOR: float = 24.0 * 60.0 * 252.0

# Stock intraday default: 6.5h × 252 = 98280 bars/year
STOCK_MINUTE_ANN_FACTOR: float = 6.5 * 60.0 * 252.0


# ═══════════════════════════════════════════════════════════════════════
# METRIC DISPATCH (pure Numba, nogil)
# ═══════════════════════════════════════════════════════════════════════


@njit(nogil=True, cache=True)
def compute_metric_nb(
    returns: np.ndarray,
    metric_type: int,
    ann_factor: float = FX_MINUTE_ANN_FACTOR,
    cutoff: float = 0.05,
) -> float:
    """Dispatch to a ``vbt.ret_nb.*_1d_nb`` kernel based on ``metric_type``.

    ``returns`` must be 1D. If your pipeline stores 2D returns from
    ``vbt.pf_nb.from_signal_func_nb``, pass ``returns[:, 0]`` or
    ``returns.flatten()``.

    Signs are flipped for metrics where "lower is better" (drawdown, VaR,
    volatility, downside risk, CVaR) so that every metric can be maximized
    by the CV selection logic without further post-processing.
    """
    if metric_type == TOTAL_RETURN:
        return vbt.ret_nb.total_return_1d_nb(returns=returns)

    elif metric_type == SHARPE_RATIO:
        return vbt.ret_nb.sharpe_ratio_1d_nb(returns=returns, ann_factor=ann_factor)

    elif metric_type == CALMAR_RATIO:
        return vbt.ret_nb.calmar_ratio_1d_nb(returns=returns, ann_factor=ann_factor)

    elif metric_type == SORTINO_RATIO:
        return vbt.ret_nb.sortino_ratio_1d_nb(returns=returns, ann_factor=ann_factor)

    elif metric_type == OMEGA_RATIO:
        return vbt.ret_nb.omega_ratio_1d_nb(returns=returns)

    elif metric_type == ANNUALIZED_RETURN:
        return vbt.ret_nb.annualized_return_1d_nb(
            returns=returns, ann_factor=ann_factor
        )

    elif metric_type == MAX_DRAWDOWN:
        return -vbt.ret_nb.max_drawdown_1d_nb(returns=returns)

    elif metric_type == PROFIT_FACTOR:
        return vbt.ret_nb.profit_factor_1d_nb(returns=returns)

    elif metric_type == VALUE_AT_RISK:
        return -vbt.ret_nb.value_at_risk_1d_nb(returns=returns, cutoff=cutoff)

    elif metric_type == TAIL_RATIO:
        return vbt.ret_nb.tail_ratio_1d_nb(returns=returns)

    elif metric_type == ANNUALIZED_VOLATILITY:
        return -vbt.ret_nb.annualized_volatility_1d_nb(
            returns=returns, ann_factor=ann_factor
        )

    elif metric_type == INFORMATION_RATIO:
        return vbt.ret_nb.information_ratio_1d_nb(returns=returns)

    elif metric_type == DOWNSIDE_RISK:
        return -vbt.ret_nb.downside_risk_1d_nb(
            returns=returns, ann_factor=ann_factor
        )

    elif metric_type == COND_VALUE_AT_RISK:
        return -vbt.ret_nb.cond_value_at_risk_1d_nb(
            returns=returns, cutoff=cutoff
        )

    else:
        return vbt.ret_nb.total_return_1d_nb(returns=returns)


# ═══════════════════════════════════════════════════════════════════════
# PLOTTING SETTINGS
# ═══════════════════════════════════════════════════════════════════════


def apply_vbt_plot_defaults() -> None:
    """Install fullscreen layout + FX 24h annualization as VBT global defaults.

    Wrapper around ``utils.apply_vbt_settings`` that the strategies' mains
    call at startup. Kept as a thin re-export so strategies only import from
    ``framework.pipeline_utils``.
    """
    from utils import apply_vbt_settings

    apply_vbt_settings()


# ═══════════════════════════════════════════════════════════════════════
# ANNUALIZATION
# ═══════════════════════════════════════════════════════════════════════


def resolve_ann_factor(index: pd.DatetimeIndex | None = None) -> float:
    """Return annualization factor.

    - If ``index`` given, compute empirically via ``utils.compute_ann_factor``
      (counts actual bars per day).
    - Otherwise return the FX-minute default (24h × 252).
    """
    if index is None:
        return FX_MINUTE_ANN_FACTOR
    from utils import compute_ann_factor

    return float(compute_ann_factor(index))


# ═══════════════════════════════════════════════════════════════════════
# PORTFOLIO ANALYZER (investigation reports)
# ═══════════════════════════════════════════════════════════════════════


def analyze_portfolio(
    pf: vbt.Portfolio,
    *,
    name: str = "Strategy",
    output_dir: str | Path | None = None,
    show_charts: bool = False,
    save_excel: bool = False,
    indicator: Any | None = None,
) -> dict[str, Any]:
    """Generate stats + equity + drawdowns + trades + tearsheet.

    Thin wrapper around ``framework.plotting`` helpers. Returns a dict with
    ``{"stats": pd.Series, "figures": dict[str, go.Figure], "html_path": Path | None}``.

    The single-run report is the successor of ``backtest_<strat>()`` + the
    ad-hoc ``_walk_forward_report`` prints scattered in the old main blocks.
    """
    from framework.plotting import (
        build_trade_report,
        generate_html_tearsheet,
        plot_monthly_heatmap,
        plot_portfolio_summary,
        plot_trade_analysis,
        plot_trade_signals,
    )

    figures: dict[str, go.Figure] = {}
    stats = pf.stats()

    try:
        figures["summary"] = plot_portfolio_summary(pf, title=f"{name} — Summary")
    except Exception as e:
        print(f"  [analyze_portfolio] plot_portfolio_summary failed: {e}")

    try:
        figures["monthly_heatmap"] = plot_monthly_heatmap(
            pf, title=f"{name} — Monthly Returns"
        )
    except Exception as e:
        print(f"  [analyze_portfolio] plot_monthly_heatmap failed: {e}")

    try:
        figures["trade_analysis"] = plot_trade_analysis(
            pf, title=f"{name} — Trade Analysis"
        )
    except Exception as e:
        print(f"  [analyze_portfolio] plot_trade_analysis failed: {e}")

    if indicator is not None:
        try:
            fig = indicator.plot() if callable(getattr(indicator, "plot", None)) else None
            if fig is not None:
                figures["indicator_overlay"] = fig
        except Exception as e:
            print(f"  [analyze_portfolio] indicator.plot() failed: {e}")

    try:
        figures["trade_signals"] = plot_trade_signals(pf, title=f"{name} — Signals")
    except Exception as e:
        print(f"  [analyze_portfolio] plot_trade_signals failed: {e}")

    report_text = build_trade_report(pf)
    print(report_text)

    html_path: Path | None = None
    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        for fig_name, fig in figures.items():
            fig_path = out / f"{name.replace(' ', '_')}_{fig_name}.html"
            fig.write_html(str(fig_path))
        try:
            html_path = out / f"{name.replace(' ', '_')}_tearsheet.html"
            generate_html_tearsheet(pf, output_path=str(html_path), title=name)
        except Exception as e:
            print(f"  [analyze_portfolio] generate_html_tearsheet failed: {e}")
            html_path = None
        report_path = out / f"{name.replace(' ', '_')}_stats.txt"
        report_path.write_text(report_text, encoding="utf-8")

    if show_charts:
        for fig in figures.values():
            try:
                fig.show()
            except Exception as e:
                print(f"  [analyze_portfolio] fig.show() failed: {e}")

    if save_excel and output_dir is not None:
        try:
            xlsx_path = Path(output_dir) / f"{name.replace(' ', '_')}_stats.xlsx"
            with pd.ExcelWriter(str(xlsx_path)) as writer:
                stats.to_frame("value").to_excel(writer, sheet_name="stats")
                if hasattr(pf, "trades"):
                    trades_df = pf.trades.records_readable
                    if isinstance(trades_df, pd.DataFrame) and not trades_df.empty:
                        trades_df.to_excel(writer, sheet_name="trades", index=False)
        except Exception as e:
            print(f"  [analyze_portfolio] save_excel failed: {e}")

    return {"stats": stats, "figures": figures, "html_path": html_path}


# ═══════════════════════════════════════════════════════════════════════
# CV PLOTTING HELPERS
# ═══════════════════════════════════════════════════════════════════════


def plot_cv_splitter(splitter: Any, title: str = "CV Splitter") -> go.Figure:
    """Visualize train/test ranges of a ``vbt.Splitter``."""
    from utils import configure_figure_for_fullscreen

    fig = splitter.plot()
    fig.update_layout(title=title)
    return configure_figure_for_fullscreen(fig)


def plot_cv_heatmap(
    grid_perf: pd.Series,
    *,
    x_level: str,
    y_level: str,
    slider_level: str | None = "split",
    title: str | None = None,
    **layout_kwargs: Any,
) -> go.Figure:
    """Render a CV grid as a heatmap, sliced by split.

    ``grid_perf`` must be a ``pd.Series`` with a ``MultiIndex`` containing at
    least ``x_level``, ``y_level`` and (optionally) ``slider_level``.
    """
    from utils import configure_figure_for_fullscreen

    kwargs: dict[str, Any] = {"x_level": x_level, "y_level": y_level}
    if slider_level is not None and slider_level in (grid_perf.index.names or []):
        kwargs["slider_level"] = slider_level
    fig = grid_perf.vbt.heatmap(**kwargs)
    if title:
        layout_kwargs.setdefault("title", title)
    if layout_kwargs:
        fig.update_layout(**layout_kwargs)
    return configure_figure_for_fullscreen(fig)


def plot_cv_volume(
    grid_perf: pd.Series,
    *,
    x_level: str,
    y_level: str,
    z_level: str,
    slider_level: str | None = "split",
    title: str | None = None,
    **layout_kwargs: Any,
) -> go.Figure:
    """Render a CV grid as a 3D volume plot, sliced by split."""
    from utils import configure_figure_for_fullscreen

    kwargs: dict[str, Any] = {
        "x_level": x_level,
        "y_level": y_level,
        "z_level": z_level,
    }
    if slider_level is not None and slider_level in (grid_perf.index.names or []):
        kwargs["slider_level"] = slider_level
    fig = grid_perf.vbt.volume(**kwargs)
    if title:
        layout_kwargs.setdefault("title", title)
    if layout_kwargs:
        fig.update_layout(**layout_kwargs)
    return configure_figure_for_fullscreen(fig)


# ═══════════════════════════════════════════════════════════════════════
# EQUIVALENCE TEST HELPERS (used by tests/test_pipeline_equivalence.py)
# ═══════════════════════════════════════════════════════════════════════


def assert_pf_equivalent(
    pf_reference: vbt.Portfolio,
    pf_candidate: vbt.Portfolio,
    *,
    rtol: float = 1e-10,
    atol: float = 1e-12,
) -> None:
    """Assert two portfolios produce the same results to float tolerance.

    Compares ``total_return``, ``sharpe_ratio``, ``max_drawdown`` and the
    per-bar ``returns`` array. Used by phase-equivalence tests during the
    refactor.
    """

    def _scalar(x: Any) -> float:
        if isinstance(x, pd.Series):
            return float(x.iloc[0])
        return float(x)

    for metric in ("total_return", "sharpe_ratio", "max_drawdown"):
        ref = _scalar(getattr(pf_reference, metric))
        cand = _scalar(getattr(pf_candidate, metric))
        if np.isnan(ref) and np.isnan(cand):
            continue
        if not np.isclose(ref, cand, rtol=rtol, atol=atol, equal_nan=True):
            raise AssertionError(
                f"{metric}: reference={ref!r} vs candidate={cand!r} "
                f"(rtol={rtol}, atol={atol})"
            )

    ref_ret = pf_reference.returns
    cand_ret = pf_candidate.returns
    if isinstance(ref_ret, pd.DataFrame):
        ref_ret = ref_ret.iloc[:, 0]
    if isinstance(cand_ret, pd.DataFrame):
        cand_ret = cand_ret.iloc[:, 0]
    np.testing.assert_allclose(
        ref_ret.values,
        cand_ret.values,
        rtol=rtol,
        atol=atol,
        equal_nan=True,
    )
