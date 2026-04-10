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

# Human-readable display labels — used for chart hovers, colorbars, and
# table headers so the reader sees "Sharpe Ratio: 1.65" instead of the
# raw snake_case identifier "sharpe_ratio".
METRIC_LABELS: dict[int, str] = {
    TOTAL_RETURN: "Total Return",
    SHARPE_RATIO: "Sharpe Ratio",
    CALMAR_RATIO: "Calmar Ratio",
    SORTINO_RATIO: "Sortino Ratio",
    OMEGA_RATIO: "Omega Ratio",
    ANNUALIZED_RETURN: "Annualized Return",
    MAX_DRAWDOWN: "Max Drawdown",
    PROFIT_FACTOR: "Profit Factor",
    VALUE_AT_RISK: "Value at Risk",
    TAIL_RATIO: "Tail Ratio",
    ANNUALIZED_VOLATILITY: "Annualized Volatility",
    INFORMATION_RATIO: "Information Ratio",
    DOWNSIDE_RISK: "Downside Risk",
    COND_VALUE_AT_RISK: "Conditional VaR",
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

    # Silence the "Using right bound of source/target index without
    # frequency" warnings from vbt.Splitter during @vbt.cv_split. FX
    # data has weekend gaps so pd.DatetimeIndex.freq is None, and VBT
    # falls back to the right bound of the index — which is the
    # correct behaviour, just noisy when repeated per split × param.
    import warnings as _warnings
    try:
        from vectorbtpro.utils.warnings_ import VBTWarning

        _warnings.filterwarnings(
            "ignore",
            message=".*right bound of.*index without frequency.*",
            category=VBTWarning,
        )
    except Exception:
        # Fallback: match by message only in case VBTWarning is not
        # importable.
        _warnings.filterwarnings(
            "ignore",
            message=".*right bound of.*index without frequency.*",
        )
    try:
        vbt.settings.resampling["silence_warnings"] = True
    except Exception:
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
        return "1h"
    if target_sec < 14400:
        return "4h"
    if target_sec < 86400:
        return "1D"
    return "1W"


def downsample_for_plot(obj: Any, max_points: int = 20_000) -> Any:
    """Downsample a Series/DataFrame to <= ``max_points`` via VBT-native resample.

    Uses ``obj.vbt.resample_apply(freq, "last")`` — the VBT Pro native
    equivalent of ``obj.resample(freq).last()`` which accepts modern
    lowercase pandas offsets (``"1h"``, ``"4h"``) without FutureWarning
    and integrates with VBT's internal Resampler.

    Returns the object unchanged if it has no DatetimeIndex, if it is
    already short enough, or if downsampling fails. Used by
    :func:`analyze_portfolio` before building indicator overlays.
    """
    if obj is None:
        return obj
    if not hasattr(obj, "index"):
        return obj
    idx = obj.index
    if not isinstance(idx, pd.DatetimeIndex) or len(obj) <= max_points:
        return obj
    total_sec = (idx[-1] - idx[0]).total_seconds()
    freq = _pick_resample_freq(len(obj), total_sec, max_points)
    if not freq:
        return obj
    try:
        return obj.vbt.resample_apply(freq, "last")
    except Exception:
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
        plot_drawdown_analysis,
        plot_monthly_heatmap,
        plot_portfolio_summary,
        plot_returns_distribution,
        plot_rolling_sharpe,
        plot_trade_analysis,
        plot_trade_duration,
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

    try:
        figures["returns_distribution"] = plot_returns_distribution(
            pf_plot, title=f"{name} — Returns Distribution"
        )
    except Exception as e:
        print(f"  [analyze_portfolio] plot_returns_distribution failed: {e}")

    try:
        figures["drawdown_analysis"] = plot_drawdown_analysis(
            pf_plot, title=f"{name} — Drawdown Analysis"
        )
    except Exception as e:
        print(f"  [analyze_portfolio] plot_drawdown_analysis failed: {e}")

    try:
        figures["rolling_sharpe"] = plot_rolling_sharpe(
            pf_plot, title=f"{name} — Rolling Sharpe"
        )
    except Exception as e:
        print(f"  [analyze_portfolio] plot_rolling_sharpe failed: {e}")

    try:
        figures["trade_duration"] = plot_trade_duration(
            pf_plot, title=f"{name} — Trade Duration"
        )
    except Exception as e:
        print(f"  [analyze_portfolio] plot_trade_duration failed: {e}")

    # Trade signals + indicator overlays combined into a single figure
    # so all entry/exit context is in one place. The indicator is
    # downsampled field-by-field first (series attributes only) to keep
    # the overlay snappy on long histories.
    ds_indicator: Any | None = None
    if indicator is not None:
        try:
            ds_indicator = _downsample_indicator_fields(indicator, max_plot_points)
        except Exception as e:
            print(f"  [analyze_portfolio] indicator downsample failed: {e}")
            ds_indicator = indicator

    try:
        pf_signals = pf
        try:
            idx = pf.wrapper.index
            if isinstance(idx, pd.DatetimeIndex) and len(idx) > max_plot_points:
                pf_signals = pf.iloc[-max_plot_points:]
        except Exception:
            pass
        figures["signals_and_indicator"] = plot_trade_signals(
            pf_signals,
            title=f"{name} — Signals + Indicators (last {max_plot_points} bars)",
            indicator=ds_indicator,
        )
    except Exception as e:
        print(f"  [analyze_portfolio] plot_trade_signals failed: {e}")

    report_text = build_trade_report(pf, name=name)
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
    from framework.plotting import _apply_title_layout, make_fullscreen

    fig = splitter.plot()
    _apply_title_layout(fig, title)
    return make_fullscreen(fig)


def _split_date_labels(splitter: Any, prefer_set: str = "test") -> dict[int, str]:
    """Return a ``{split_int: "YYYY-MM-DD → YYYY-MM-DD"}`` mapping.

    Looks up the preferred set ("test" or "train") in
    ``splitter.index_bounds`` and falls back gracefully if the splitter
    does not expose bounds.
    """
    try:
        bounds = splitter.index_bounds
    except Exception:
        return {}
    out: dict[int, str] = {}
    try:
        for (split_i, set_i), row in bounds.iterrows():
            if set_i == prefer_set or set_i == (1 if prefer_set == "test" else 0):
                start = pd.Timestamp(row["start"]).strftime("%Y-%m-%d")
                end = pd.Timestamp(row["end"]).strftime("%Y-%m-%d")
                out[split_i] = f"{start}/{end}"
    except Exception:
        return {}
    return out


def _relabel_split_level(
    grid_perf: pd.Series, splitter: Any | None
) -> pd.Series:
    """Return a copy of ``grid_perf`` with the ``split`` level renamed
    to human-readable date ranges, if the splitter is provided and has
    bounds.
    """
    if splitter is None or "split" not in (grid_perf.index.names or []):
        return grid_perf
    labels = _split_date_labels(splitter, prefer_set="test")
    if not labels:
        return grid_perf
    try:
        return grid_perf.rename(index=labels, level="split")
    except Exception:
        return grid_perf


def _select_set(grid_perf: pd.Series, set_name: str = "test") -> pd.Series:
    """Return the subset of ``grid_perf`` matching the given ``set`` level."""
    if "set" not in (grid_perf.index.names or []):
        return grid_perf
    try:
        return grid_perf.xs(set_name, level="set")
    except (KeyError, ValueError):
        try:
            return grid_perf.xs(1 if set_name == "test" else 0, level="set")
        except Exception:
            return grid_perf


def _reduce_to_plot_levels(
    data: pd.Series,
    keep_levels: list[str],
) -> pd.Series:
    """Collapse ``data`` down to exactly ``keep_levels`` by averaging
    over every other MultiIndex level.

    Necessary before calling ``data.vbt.heatmap(...)`` or building a
    Surface because any extra index level (e.g. a parameter the user
    did not map to an axis) produces an ambiguous render.
    """
    if not isinstance(data.index, pd.MultiIndex):
        return data
    extras = [n for n in (data.index.names or []) if n not in keep_levels]
    if not extras:
        return data
    # Group by the desired levels only, averaging across the rest.
    return data.groupby(keep_levels).mean()


def _pretty_metric(metric_name: str) -> str:
    """``sharpe_ratio`` → ``Sharpe Ratio`` for display labels."""
    return metric_name.replace("_", " ").title()


def plot_grid_heatmap(
    grid_perf: pd.Series,
    *,
    x_level: str,
    y_level: str,
    slider_level: str | None = None,
    splitter: Any | None = None,
    set_name: str = "test",
    title: str | None = None,
    metric_name: str = "metric",
    **layout_kwargs: Any,
) -> go.Figure:
    """Render a grid-search (or CV grid) as a 2D heatmap.

    Extra index levels that are neither ``x_level``/``y_level`` nor
    ``slider_level``/``"split"`` are averaged out so the heatmap is
    always built from a clean Series with exactly 2 (+ optional
    slider) dimensions.

    Parameters
    ----------
    grid_perf
        ``pd.Series`` with a ``MultiIndex`` containing ``x_level``,
        ``y_level`` and optionally ``slider_level``, ``split``, ``set``.
    x_level, y_level
        Index level names to map to x and y axes.
    slider_level
        Optional level used as a slider. Pass ``"split"`` to page
        through CV folds, or any swept parameter to slide through its
        values.  Pass ``None`` for a single aggregated heatmap.
    splitter
        Optional ``vbt.Splitter``. When provided, the ``split`` level
        is relabelled with date ranges for human-readable sliders.
    set_name
        Which ``set`` to keep when a ``set`` level exists (``"test"``
        or ``"train"``). Ignored if no ``set`` level present.
    """
    from framework.plotting import _apply_title_layout, make_fullscreen

    data = _select_set(grid_perf, set_name=set_name)
    data = _relabel_split_level(data, splitter)

    # Build the exact list of index levels needed for rendering.
    if slider_level is None:
        keep = [x_level, y_level]
    else:
        keep = [x_level, y_level, slider_level]
    data = _reduce_to_plot_levels(data, keep)

    # Report NaN cells. The VBT heatmap leaves NaN as blank cells,
    # which is visually correct ("this combo failed to compute") but
    # can look like a bug without a diagnostic.
    nan_count = int(data.isna().sum())
    if nan_count > 0:
        print(
            f"  [heatmap] {nan_count}/{len(data)} NaN cells — rendered "
            f"as blanks (combos where metric computation yielded NaN)"
        )

    pretty = _pretty_metric(metric_name)
    kwargs: dict[str, Any] = {
        "x_level": x_level,
        "y_level": y_level,
        "trace_kwargs": dict(
            colorbar=dict(title=pretty),
            hovertemplate=(
                f"{x_level}: %{{x}}<br>"
                f"{y_level}: %{{y}}<br>"
                f"{pretty}: %{{z:.2f}}<extra></extra>"
            ),
        ),
    }
    if slider_level is not None and slider_level in (data.index.names or []):
        kwargs["slider_level"] = slider_level
    fig = data.vbt.heatmap(**kwargs)
    if title:
        _apply_title_layout(fig, title)
    if layout_kwargs:
        fig.update_layout(**layout_kwargs)
    return make_fullscreen(fig)


def _surface_matrix_from_series(
    s: pd.Series,
    x_level: str,
    y_level: str,
    fill_method: str = "median",
) -> tuple[list, list, np.ndarray]:
    """Convert a ``pd.Series`` with at least ``(x_level, y_level)`` in
    its MultiIndex to a ``(x_values, y_values, z_matrix)`` tuple ready
    for ``go.Surface``.

    NaN cells are filled with ``median`` (default), ``mean`` or a
    constant float (``0``/``np.nan``/etc.) so 3D surfaces render as
    continuous sheets even when a few combos fail. A diagnostic
    message is printed if any cell was filled.
    """
    # Force the Series down to the two plot levels before unstacking.
    s = _reduce_to_plot_levels(s, [x_level, y_level])
    mat = s.unstack(y_level)
    # Sort axes numerically if possible so surfaces read left-to-right.
    try:
        mat = mat.sort_index(axis=0)
        mat = mat.sort_index(axis=1)
    except Exception:
        pass
    x = list(mat.index)
    y = list(mat.columns)
    z = mat.values.astype(float)
    nan_count = int(np.isnan(z).sum())
    if nan_count > 0:
        all_nan = nan_count == z.size
        if all_nan:
            fill_val = 0.0
        elif fill_method == "median":
            import warnings as _w
            with _w.catch_warnings():
                _w.simplefilter("ignore", RuntimeWarning)
                med = np.nanmedian(z)
            fill_val = float(med) if np.isfinite(med) else 0.0
        elif fill_method == "mean":
            import warnings as _w
            with _w.catch_warnings():
                _w.simplefilter("ignore", RuntimeWarning)
                mn = np.nanmean(z)
            fill_val = float(mn) if np.isfinite(mn) else 0.0
        else:
            try:
                fill_val = float(fill_method)
            except (TypeError, ValueError):
                fill_val = 0.0
        print(
            f"  [surface] {nan_count}/{z.size} NaN cells filled with "
            f"{fill_method}={fill_val:.4f}"
            + ("  (ALL NaN — degenerate frame)" if all_nan else "")
        )
        z = np.where(np.isnan(z), fill_val, z)
    # Plotly expects z[i_y, i_x], i.e. the transpose of a row=x / col=y matrix.
    return x, y, z.T


def plot_grid_surface(
    grid_perf: pd.Series,
    *,
    x_level: str,
    y_level: str,
    slider_level: str | None = None,
    splitter: Any | None = None,
    set_name: str = "test",
    title: str | None = None,
    colorscale: str = "Viridis",
    fill_method: str = "median",
    metric_name: str = "metric",
    **layout_kwargs: Any,
) -> go.Figure:
    """Render a grid-search result as a 3D ``go.Surface`` plot.

    Unlike :func:`plot_grid_volume` which needs three swept parameters
    (x, y, z), the surface uses **two** parameter axes and the metric
    itself as height. NaN cells (e.g. from CV folds with no trades)
    are filled with the median of the remaining values so the surface
    stays a continuous sheet instead of disappearing.

    If ``slider_level`` is provided, one surface is emitted per slider
    frame. Otherwise extra levels are averaged out.
    """
    from framework.plotting import _apply_title_layout, make_fullscreen

    data = _select_set(grid_perf, set_name=set_name)
    data = _relabel_split_level(data, splitter)

    # Drop every level that is neither x, y nor the slider (start, end,
    # other params) so the surface matrix is unambiguous.
    if slider_level is None:
        keep = [x_level, y_level]
    else:
        keep = [x_level, y_level, slider_level]
    data = _reduce_to_plot_levels(data, keep)

    pretty = _pretty_metric(metric_name)
    hovertemplate = (
        f"{x_level}: %{{x}}<br>"
        f"{y_level}: %{{y}}<br>"
        f"{pretty}: %{{z:.2f}}<extra></extra>"
    )

    if slider_level is not None and slider_level in (data.index.names or []):
        slider_vals = sorted(
            data.index.get_level_values(slider_level).unique(),
            key=lambda v: str(v),
        )
        frames = []
        first_x = first_y = first_z = None
        for sv in slider_vals:
            sub = data.xs(sv, level=slider_level)
            x, y, z = _surface_matrix_from_series(
                sub, x_level, y_level, fill_method=fill_method
            )
            if first_z is None:
                first_x, first_y, first_z = x, y, z
            frames.append(
                go.Frame(
                    data=[
                        go.Surface(
                            x=x, y=y, z=z,
                            colorscale=colorscale,
                            hovertemplate=hovertemplate,
                        )
                    ],
                    name=str(sv),
                )
            )
        fig = go.Figure(
            data=[
                go.Surface(
                    x=first_x, y=first_y, z=first_z,
                    colorscale=colorscale,
                    colorbar=dict(title=pretty),
                    hovertemplate=hovertemplate,
                )
            ],
            frames=frames,
        )
        fig.update_layout(
            sliders=[
                dict(
                    active=0,
                    pad=dict(t=50),
                    currentvalue=dict(prefix=f"{slider_level}: "),
                    steps=[
                        dict(
                            method="animate",
                            label=str(sv),
                            args=[
                                [str(sv)],
                                dict(
                                    mode="immediate",
                                    frame=dict(duration=0, redraw=True),
                                    transition=dict(duration=0),
                                ),
                            ],
                        )
                        for sv in slider_vals
                    ],
                )
            ],
        )
    else:
        x, y, z = _surface_matrix_from_series(
            data, x_level, y_level, fill_method=fill_method
        )
        fig = go.Figure(
            go.Surface(
                x=x, y=y, z=z,
                colorscale=colorscale,
                colorbar=dict(title=pretty),
                hovertemplate=hovertemplate,
            )
        )

    fig.update_layout(
        scene=dict(
            xaxis_title=x_level,
            yaxis_title=y_level,
            zaxis_title=pretty,
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.1)),
        ),
    )
    if title:
        _apply_title_layout(fig, title)
    if layout_kwargs:
        fig.update_layout(**layout_kwargs)
    return make_fullscreen(fig)


def plot_grid_volume(
    grid_perf: pd.Series,
    *,
    x_level: str,
    y_level: str,
    z_level: str,
    slider_level: str | None = None,
    splitter: Any | None = None,
    set_name: str = "test",
    title: str | None = None,
    fill_method: str = "median",
    metric_name: str = "metric",
    **layout_kwargs: Any,
) -> go.Figure:
    """Render a grid-search (or CV grid) as a 3D volume plot.

    NaN cells are filled with the median of the remaining values so
    the volume stays a continuous blob. See :func:`plot_grid_heatmap`
    for the rest of the parameter semantics.
    """
    from framework.plotting import _apply_title_layout, make_fullscreen

    data = _select_set(grid_perf, set_name=set_name)
    data = _relabel_split_level(data, splitter)

    if slider_level is None:
        keep = [x_level, y_level, z_level]
    else:
        keep = [x_level, y_level, z_level, slider_level]
    data = _reduce_to_plot_levels(data, keep)

    nan_count = int(data.isna().sum())
    if nan_count > 0:
        if fill_method == "median":
            fill_val = float(data.median(skipna=True))
        elif fill_method == "mean":
            fill_val = float(data.mean(skipna=True))
        else:
            try:
                fill_val = float(fill_method)
            except (TypeError, ValueError):
                fill_val = 0.0
        print(
            f"  [volume] {nan_count}/{len(data)} NaN cells filled with "
            f"{fill_method}={fill_val:.4f}"
        )
        data = data.fillna(fill_val)

    pretty = _pretty_metric(metric_name)
    kwargs: dict[str, Any] = {
        "x_level": x_level,
        "y_level": y_level,
        "z_level": z_level,
        "trace_kwargs": dict(
            colorbar=dict(title=pretty),
            hovertemplate=(
                f"{x_level}: %{{x}}<br>"
                f"{y_level}: %{{y}}<br>"
                f"{z_level}: %{{z}}<br>"
                f"{pretty}: %{{value:.2f}}<extra></extra>"
            ),
        ),
    }
    if slider_level is not None and slider_level in (data.index.names or []):
        kwargs["slider_level"] = slider_level
    fig = data.vbt.volume(**kwargs)
    if title:
        _apply_title_layout(fig, title)
    if layout_kwargs:
        fig.update_layout(**layout_kwargs)
    return make_fullscreen(fig)


# Back-compat wrappers keeping the old ``plot_cv_*`` names.
def plot_cv_heatmap(
    grid_perf: pd.Series,
    *,
    x_level: str,
    y_level: str,
    slider_level: str | None = "split",
    splitter: Any | None = None,
    set_name: str = "test",
    title: str | None = None,
    metric_name: str = "metric",
    **layout_kwargs: Any,
) -> go.Figure:
    """CV heatmap with split slider. See :func:`plot_grid_heatmap`."""
    return plot_grid_heatmap(
        grid_perf,
        x_level=x_level,
        y_level=y_level,
        slider_level=slider_level,
        splitter=splitter,
        set_name=set_name,
        title=title,
        metric_name=metric_name,
        **layout_kwargs,
    )


def plot_cv_volume(
    grid_perf: pd.Series,
    *,
    x_level: str,
    y_level: str,
    z_level: str,
    slider_level: str | None = "split",
    splitter: Any | None = None,
    set_name: str = "test",
    title: str | None = None,
    metric_name: str = "metric",
    **layout_kwargs: Any,
) -> go.Figure:
    """CV 3D volume with split slider. See :func:`plot_grid_volume`."""
    return plot_grid_volume(
        grid_perf,
        x_level=x_level,
        y_level=y_level,
        z_level=z_level,
        slider_level=slider_level,
        splitter=splitter,
        set_name=set_name,
        title=title,
        metric_name=metric_name,
        **layout_kwargs,
    )


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
