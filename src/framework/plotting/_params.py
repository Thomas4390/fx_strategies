"""Parameter-grid visualizations: 2D heatmaps, 3D surfaces and volumes.

Extracted from ``_core`` to stay under the 800-line rule. Depends on
``make_fullscreen`` from ``_core`` for consistent fullscreen layout.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from ._core import make_fullscreen


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
