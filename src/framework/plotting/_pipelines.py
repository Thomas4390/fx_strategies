"""Standalone runner pipelines: parameter grids and end-to-end reports.

Extracted from ``_core`` to keep each plotting module under the 800-line
rule. These functions compose the lower-level ``plot_*`` helpers from
``_core`` into one-call entry points for strategy scripts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import vectorbtpro as vbt

from ._core import (
    generate_html_tearsheet,
    plot_monthly_heatmap,
    plot_portfolio_summary,
    save_fullscreen_html,
    show_browser,
)
from ._equity import (
    plot_drawdown_analysis,
    plot_returns_distribution,
    plot_rolling_sharpe,
)
from ._params import (
    plot_param_heatmap,
    plot_param_heatmap_slider,
    plot_param_surface,
    plot_param_volume,
)
from ._reports import print_extended_stats
from ._trades import (
    plot_exposure,
    plot_orders_heatmap,
    plot_orders_on_price,
    plot_trade_analysis,
    plot_trade_duration,
    plot_trades_on_price,
    plot_value_and_cash,
)


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
