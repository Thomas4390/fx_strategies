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
# EXECUTION DEFAULTS — progress bars + parallelization
# ═══════════════════════════════════════════════════════════════════════

# Default execution config for @vbt.parameterized and @vbt.cv_split.
#
# engine="threadpool" is the right choice for our pipelines because the
# heavy lifting is done inside vbt.Portfolio.from_signals which releases
# the GIL in its Numba kernels (from_signals_nb). The outer Python layer
# holds the GIL briefly but the simulation itself runs nogil.
#
# For pure @njit(nogil=True) pipelines, threadpool + chunk_len="auto"
# scales near-linearly with the number of cores. For Python-level
# pipelines that delegate to VBT, the scaling is sublinear but still
# provides a 2-4x speedup on typical grids vs serial execution.
#
# If a strategy becomes GIL-bound (e.g. heavy Python-level post-processing
# per combo), switch to engine="pathos" with distribute="chunks" to use
# multiprocessing instead — but note the overhead of pickling vbt.Data.
#
# NOTE: ``show_progress`` is intentionally omitted. VBT enables progress bars
# by default (with a delay), and passing ``show_progress=True`` via
# ``parameterized_kwargs`` inside ``@vbt.cv_split`` triggers a VBT bug where
# ``cv_split`` injects its own ``show_progress=False`` at the top-level of
# parameterized_kwargs, causing a conflict with the nested value and leading
# to ``Parameterizer doesn't expect arguments ['show_progress']``. By omitting
# it from our defaults, we let VBT's automatic default (enabled with delay)
# kick in, and ``pbar_kwargs=dict(delay=0)`` forces immediate display.
DEFAULT_EXECUTE_KWARGS: dict[str, Any] = dict(
    chunk_len="auto",
    engine="threadpool",
    warmup=True,  # compile first chunk serially before parallelizing
)


def make_execute_kwargs(desc: str, **overrides: Any) -> dict[str, Any]:
    """Build an ``execute_kwargs`` dict with a labelled tqdm progress bar.

    Parameters
    ----------
    desc
        Label shown on the progress bar (e.g. ``"MR Turbo grid"``).
    **overrides
        Any key in :data:`DEFAULT_EXECUTE_KWARGS` to override.

    Returns
    -------
    dict
        Dictionary suitable for
        ``@vbt.parameterized(execute_kwargs=...)`` and
        ``@vbt.cv_split(execute_kwargs=...)``.

    Example
    -------
    >>> @vbt.parameterized(
    ...     merge_func="concat",
    ...     execute_kwargs=make_execute_kwargs("MR Turbo grid"),
    ... )
    ... def pipeline_nb(...): ...
    """
    out: dict[str, Any] = dict(DEFAULT_EXECUTE_KWARGS)
    # Allow callers to override pbar_kwargs while preserving the desc.
    pbar_overrides = overrides.pop("pbar_kwargs", None)
    for key, val in overrides.items():
        out[key] = val
    pbar_kwargs: dict[str, Any] = dict(delay=0)  # force immediate display
    if pbar_overrides:
        pbar_kwargs.update(pbar_overrides)
    pbar_kwargs["desc"] = desc
    out["pbar_kwargs"] = pbar_kwargs
    return out


# ═══════════════════════════════════════════════════════════════════════
# PLOTTING SETTINGS
# ═══════════════════════════════════════════════════════════════════════


def apply_vbt_plot_defaults() -> None:
    """Install fullscreen layout + FX 24h annualization + downsampling defaults.

    In addition to ``utils.apply_vbt_settings`` (which installs
    ``configure_figure_for_fullscreen`` and the 24h year-freq), this also
    enables the Plotly performance optimizations needed to plot long
    minute-frequency series without freezing the browser / IDE:

    - ``use_gl=True``: enable ``Scattergl`` (WebGL) rendering for traces
      with more than 10000 points. This gives a 10-50x speedup for line
      plots of long histories.
    - ``use_resampler=True``: activate the ``plotly-resampler`` plugin if
      installed. This plugin dynamically downsamples the data client-side
      so only the visible points are rendered. Silently skipped if the
      package is not installed.
    """
    from utils import apply_vbt_settings

    apply_vbt_settings()

    # Enable WebGL scatter rendering for large traces (>10k points).
    try:
        vbt.settings.plotting["use_gl"] = True
    except Exception:
        pass

    # Enable plotly-resampler if available — fallback gracefully otherwise.
    try:
        import plotly_resampler  # noqa: F401

        vbt.settings.plotting["use_resampler"] = True
    except ImportError:
        pass


# ═══════════════════════════════════════════════════════════════════════
# DOWNSAMPLING HELPERS
# ═══════════════════════════════════════════════════════════════════════


def _pick_resample_freq(n_source: int, total_seconds: float, max_points: int) -> str:
    """Choose a human-readable pandas offset that yields ~max_points samples."""
    if n_source <= max_points or total_seconds <= 0:
        return ""
    target_sec = total_seconds / max_points
    if target_sec < 60:
        return "1min"
    if target_sec < 300:
        return "5min"
    if target_sec < 900:
        return "15min"
    if target_sec < 1800:
        return "30min"
    if target_sec < 3600:
        return "1H"
    if target_sec < 14400:
        return "4H"
    if target_sec < 86400:
        return "1D"
    return "1W"


def downsample_for_plot(obj: Any, max_points: int = 20_000) -> Any:
    """Downsample a Series/DataFrame to <= ``max_points`` via temporal resample.

    Returns the object unchanged if it has no DatetimeIndex, if it is
    already short enough, or if downsampling fails. Used by
    :func:`analyze_portfolio` before building indicator overlays.
    """
    if obj is None:
        return obj
    if not hasattr(obj, "index") or not hasattr(obj, "resample"):
        return obj
    idx = obj.index
    if not isinstance(idx, pd.DatetimeIndex) or len(obj) <= max_points:
        return obj
    total_sec = (idx[-1] - idx[0]).total_seconds()
    freq = _pick_resample_freq(len(obj), total_sec, max_points)
    if not freq:
        return obj
    try:
        return obj.resample(freq).last().dropna()
    except Exception:
        return obj


def downsample_portfolio_for_plot(
    pf: vbt.Portfolio, max_points: int = 20_000
) -> vbt.Portfolio:
    """Resample a ``vbt.Portfolio`` down to <= ``max_points`` for plotting.

    Uses ``pf.resample(freq)`` which correctly handles intrabar stops and
    position carry-over. Returns the original ``pf`` if already short
    enough or if resampling fails.

    The original full-resolution ``pf`` should still be used for
    ``pf.stats()`` / ``pf.sharpe_ratio`` — only the plotting layer needs
    the downsampled version.
    """
    try:
        idx = pf.wrapper.index
    except Exception:
        return pf
    if not isinstance(idx, pd.DatetimeIndex) or len(idx) <= max_points:
        return pf
    total_sec = (idx[-1] - idx[0]).total_seconds()
    freq = _pick_resample_freq(len(idx), total_sec, max_points)
    if not freq:
        return pf
    try:
        return pf.resample(freq)
    except Exception as e:
        print(f"  [downsample_portfolio_for_plot] resample({freq!r}) failed: {e}")
        return pf


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


def _downsample_indicator_fields(indicator: Any, max_points: int) -> Any:
    """Return a copy of ``indicator`` with all Series/DataFrame fields
    downsampled to ``max_points``.

    Works on dataclass instances that hold pandas Series attributes
    (``MRTurboIndicator``, ``MRMacroIndicator``, ``OUMRIndicator``,
    ``CompositeAlphaIndicator``, ``TSMomentumIndicator``,
    ``XSMomentumIndicator``, ``RSIDailyIndicator``) without requiring any
    strategy-specific knowledge — inspected at runtime.
    """
    if indicator is None:
        return None
    # Prefer dataclass copy via __dict__ mutation, fall back to returning
    # the original if we cannot build a new instance.
    import copy

    try:
        ds = copy.copy(indicator)
    except Exception:
        return indicator
    for attr_name in dir(ds):
        if attr_name.startswith("_"):
            continue
        try:
            val = getattr(ds, attr_name)
        except Exception:
            continue
        if isinstance(val, (pd.Series, pd.DataFrame)):
            try:
                object.__setattr__(ds, attr_name, downsample_for_plot(val, max_points))
            except Exception:
                pass
    return ds


def analyze_portfolio(
    pf: vbt.Portfolio,
    *,
    name: str = "Strategy",
    output_dir: str | Path | None = None,
    show_charts: bool = False,
    save_excel: bool = False,
    indicator: Any | None = None,
    max_plot_points: int = 20_000,
) -> dict[str, Any]:
    """Generate stats + equity + drawdowns + trades + tearsheet.

    Thin wrapper around ``framework.plotting`` helpers. Returns a dict with
    ``{"stats": pd.Series, "figures": dict[str, go.Figure], "html_path": Path | None}``.

    Parameters
    ----------
    pf
        Full-resolution portfolio. ``pf.stats()`` is computed on this
        object so the reported metrics reflect the original frequency.
    indicator
        Optional indicator holder with a ``.plot()`` method. Any pandas
        Series/DataFrame attribute is downsampled via
        :func:`downsample_for_plot` before being passed to ``.plot()``.
    max_plot_points
        Target max number of bars per trace. Long histories (e.g. FX
        minute over 8 years = ~1.2M points) are resampled to a coarser
        frequency before plotting to prevent Plotly/IDE freezes.
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
    # Stats always computed on the full-resolution portfolio.
    stats = pf.stats()

    # Downsample the portfolio for plotting only — leaves pf.stats() untouched.
    pf_plot = downsample_portfolio_for_plot(pf, max_points=max_plot_points)

    try:
        figures["summary"] = plot_portfolio_summary(pf_plot, title=f"{name} — Summary")
    except Exception as e:
        print(f"  [analyze_portfolio] plot_portfolio_summary failed: {e}")

    try:
        figures["monthly_heatmap"] = plot_monthly_heatmap(
            pf_plot, title=f"{name} — Monthly Returns"
        )
    except Exception as e:
        print(f"  [analyze_portfolio] plot_monthly_heatmap failed: {e}")

    try:
        figures["trade_analysis"] = plot_trade_analysis(
            pf_plot, title=f"{name} — Trade Analysis"
        )
    except Exception as e:
        print(f"  [analyze_portfolio] plot_trade_analysis failed: {e}")

    if indicator is not None:
        # Rebuild the indicator with each Series attribute downsampled.
        try:
            ds_indicator = _downsample_indicator_fields(indicator, max_plot_points)
        except Exception as e:
            print(f"  [analyze_portfolio] indicator downsample failed: {e}")
            ds_indicator = indicator
        try:
            fig = ds_indicator.plot() if callable(getattr(ds_indicator, "plot", None)) else None
            if fig is not None:
                figures["indicator_overlay"] = fig
        except Exception as e:
            print(f"  [analyze_portfolio] indicator.plot() failed: {e}")

    # Trade signals need fine resolution to show individual trade markers,
    # but plotting 1M+ points freezes Plotly. Compromise: if the full pf is
    # too long, slice the last N bars at native resolution instead of
    # resampling.
    try:
        pf_signals = pf
        try:
            idx = pf.wrapper.index
            if isinstance(idx, pd.DatetimeIndex) and len(idx) > max_plot_points:
                pf_signals = pf.iloc[-max_plot_points:]
        except Exception:
            pass
        figures["trade_signals"] = plot_trade_signals(
            pf_signals, title=f"{name} — Signals (last {max_plot_points} bars)"
        )
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
