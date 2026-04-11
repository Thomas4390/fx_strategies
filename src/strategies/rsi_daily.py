"""RSI Mean Reversion (Daily) — ims_pipeline format.

Single-pair daily RSI mean reversion:
- Long when RSI crosses below ``oversold``
- Exit long when RSI crosses above ``exit_mid``
- Short when RSI crosses above ``overbought``
- Exit short when RSI crosses below ``exit_mid``

Three entry points:
- ``pipeline(data, **params) -> (pf, ind)`` — investigation path
- ``pipeline_nb(data, **params)`` — ``@vbt.parameterized`` scalar metric
- ``create_cv_pipeline(splitter, metric_type)`` — ``@vbt.cv_split`` factory
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import vectorbtpro as vbt

from framework.pipeline_utils import (
    SHARPE_RATIO,
    compute_metric_nb,
    make_execute_kwargs,
)

# RSI daily is resampled to daily frequency → annualization factor = 252
RSI_DAILY_ANN_FACTOR: float = 252.0


# ═══════════════════════════════════════════════════════════════════════
# 1. INVESTIGATION PATH — pipeline() returns (pf, indicator)
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class RSIDailyIndicator:
    """Lightweight indicator wrapper with a Plotly overlay."""

    close: pd.Series
    rsi: pd.Series
    oversold: float
    overbought: float
    exit_mid: float

    def plot(self, fig: go.Figure | None = None, **layout_kwargs) -> go.Figure:
        fig = fig or go.Figure()
        self.rsi.vbt.plot(
            fig=fig,
            trace_kwargs=dict(name="RSI", line=dict(width=2, color="purple")),
        )
        for level, color, name in (
            (self.oversold, "green", f"Oversold ({self.oversold:.0f})"),
            (self.overbought, "red", f"Overbought ({self.overbought:.0f})"),
            (self.exit_mid, "grey", f"Exit ({self.exit_mid:.0f})"),
        ):
            fig.add_hline(
                y=level,
                line=dict(color=color, dash="dash", width=1),
                annotation_text=name,
                annotation_position="right",
            )
        fig.update_layout(**layout_kwargs)
        return fig


def backtest_rsi_daily(
    data: vbt.Data,
    **kwargs: Any,
) -> vbt.Portfolio:
    """DEPRECATED shim — use ``pipeline(data, ...)`` instead.

    Kept temporarily for callers that still import ``backtest_rsi_daily``.
    """
    pf, _ = pipeline(data, **kwargs)
    return pf


def pipeline(
    data: vbt.Data,
    rsi_period: int = 14,
    oversold: float = 25.0,
    overbought: float = 75.0,
    exit_mid: float = 50.0,
    leverage: float = 1.0,
    init_cash: float = 1_000_000.0,
    slippage: float = 0.0001,
) -> tuple[vbt.Portfolio, RSIDailyIndicator]:
    """Investigation path — bit-equivalent to the legacy ``backtest_rsi_daily``.

    The input ``data`` can be either a ``vbt.Data`` or a raw Series/DataFrame
    indexed at any frequency. It is resampled to daily close before the RSI
    computation to match the original semantics.
    """
    if not (oversold < exit_mid < overbought):
        raise ValueError(
            f"RSI thresholds must satisfy oversold < exit_mid < overbought, "
            f"got oversold={oversold}, exit_mid={exit_mid}, overbought={overbought}"
        )

    if hasattr(data, "close"):
        close_any = data.close
    else:
        close_any = data
    close_daily = close_any.resample("1D").last().dropna()

    rsi = vbt.RSI.run(close_daily, window=rsi_period).rsi

    entries = rsi.vbt.crossed_below(oversold)
    exits = rsi.vbt.crossed_above(exit_mid)
    short_entries = rsi.vbt.crossed_above(overbought)
    short_exits = rsi.vbt.crossed_below(exit_mid)

    pf = vbt.Portfolio.from_signals(
        close=close_daily,
        entries=entries,
        exits=exits,
        short_entries=short_entries,
        short_exits=short_exits,
        init_cash=init_cash,
        leverage=leverage,
        slippage=slippage,
        freq="1D",
    )

    indicator = RSIDailyIndicator(
        close=close_daily,
        rsi=rsi.rename("RSI"),
        oversold=oversold,
        overbought=overbought,
        exit_mid=exit_mid,
    )
    return pf, indicator


# ═══════════════════════════════════════════════════════════════════════
# 2. GRID-SEARCH PATH — pipeline_nb (@vbt.parameterized)
# ═══════════════════════════════════════════════════════════════════════


@vbt.parameterized(
    merge_func="concat",
    execute_kwargs=make_execute_kwargs("RSI Daily grid"),
)
def pipeline_nb(
    data: vbt.Data,
    rsi_period: int,
    oversold: float,
    overbought: float,
    exit_mid: float = 50.0,
    leverage: float = 1.0,
    init_cash: float = 1_000_000.0,
    slippage: float = 0.0001,
    ann_factor: float = RSI_DAILY_ANN_FACTOR,
    cutoff: float = 0.05,
    metric_type: int = SHARPE_RATIO,
) -> float:
    """Grid-search path — scalar metric per param combo."""
    pf, _ = pipeline(
        data,
        rsi_period=rsi_period,
        oversold=oversold,
        overbought=overbought,
        exit_mid=exit_mid,
        leverage=leverage,
        init_cash=init_cash,
        slippage=slippage,
    )
    returns = pf.returns.values
    if returns.ndim > 1:
        returns = returns[:, 0]
    return float(compute_metric_nb(returns, metric_type, ann_factor, cutoff))


def run_grid(
    data: vbt.Data,
    *,
    rsi_period: list[int] | int,
    oversold: list[float] | float,
    overbought: list[float] | float,
    metric_type: int = SHARPE_RATIO,
    **kwargs: Any,
) -> pd.Series:
    """Wrap list inputs as ``vbt.Param`` and call ``pipeline_nb``."""

    def _param(v):
        if isinstance(v, (list, tuple, np.ndarray)):
            return vbt.Param(list(v))
        return v

    return pipeline_nb(
        data,
        rsi_period=_param(rsi_period),
        oversold=_param(oversold),
        overbought=_param(overbought),
        metric_type=metric_type,
        **kwargs,
    )


# ═══════════════════════════════════════════════════════════════════════
# 3. CV FACTORY — create_cv_pipeline (@vbt.cv_split)
# ═══════════════════════════════════════════════════════════════════════


def create_cv_pipeline(
    splitter: Any,
    metric_type: int = SHARPE_RATIO,
    **pipeline_defaults: Any,
):
    """Build a ``@vbt.cv_split`` pipeline for walk-forward cross-validation."""
    splitter_kwargs = pipeline_defaults.pop("splitter_kwargs", {})

    defaults = dict(
        exit_mid=50.0,
        leverage=1.0,
        init_cash=1_000_000.0,
        slippage=0.0001,
        ann_factor=RSI_DAILY_ANN_FACTOR,
        cutoff=0.05,
        metric_type=metric_type,
    )
    defaults.update(pipeline_defaults)

    @vbt.cv_split(
        splitter=splitter,
        splitter_kwargs=splitter_kwargs,
        takeable_args=["data"],
        parameterized_kwargs=dict(
            execute_kwargs=make_execute_kwargs(
                "RSI Daily combos", pbar_kwargs=dict(leave=False)
            ),
            merge_func="concat",
        ),
        execute_kwargs=make_execute_kwargs("RSI Daily CV splits"),
        merge_func="concat",
        return_grid="all",
        attach_bounds="index",
    )
    def cv_pipeline(
        data: vbt.Data,
        rsi_period: int,
        oversold: float,
        overbought: float,
        exit_mid: float = defaults["exit_mid"],
        leverage: float = defaults["leverage"],
        init_cash: float = defaults["init_cash"],
        slippage: float = defaults["slippage"],
        ann_factor: float = defaults["ann_factor"],
        cutoff: float = defaults["cutoff"],
        metric_type: int = defaults["metric_type"],
    ) -> float:
        pf, _ = pipeline(
            data,
            rsi_period=rsi_period,
            oversold=oversold,
            overbought=overbought,
            exit_mid=exit_mid,
            leverage=leverage,
            init_cash=init_cash,
            slippage=slippage,
        )
        returns = pf.returns.values
        if returns.ndim > 1:
            returns = returns[:, 0]
        return float(compute_metric_nb(returns, metric_type, ann_factor, cutoff))

    return cv_pipeline


# ═══════════════════════════════════════════════════════════════════════
# 4. CLI — single / grid / cv modes
# ═══════════════════════════════════════════════════════════════════════


if __name__ == "__main__":
    import sys
    from pathlib import Path as _Path

    _SRC = _Path(__file__).resolve().parent.parent
    if str(_SRC) not in sys.path:
        sys.path.insert(0, str(_SRC))

    from framework.pipeline_utils import (
        METRIC_LABELS,
        analyze_portfolio,
        apply_vbt_plot_defaults,
        plot_cv_heatmap,
        plot_cv_splitter,
        plot_cv_volume,
        plot_grid_heatmap,
        plot_grid_surface,
        plot_grid_volume,
    )
    from framework.plotting import print_cv_results, print_grid_results
    from utils import load_fx_data

    # ─────────────────────────────────────────────────────────────────
    # CONFIGURATION
    # ─────────────────────────────────────────────────────────────────
    PAIR = "EUR-USD"  # EUR-USD | GBP-USD | USD-JPY | USD-CAD
    DATA_PATH = f"data/{PAIR}_minute.parquet"
    OUTPUT_DIR = f"results/rsi_daily_{PAIR.lower()}"
    SHOW_CHARTS = True
    N_FOLDS = 10

    SINGLE_PARAMS: dict[str, Any] = dict(
        rsi_period=14,
        oversold=25.0,
        overbought=75.0,
        leverage=1.0,
    )
    GRID_PARAMS: dict[str, list] = dict(
        rsi_period=[7, 10, 14, 21],
        oversold=[20.0, 25.0, 30.0],
        overbought=[70.0, 75.0, 80.0],
    )
    CV_PARAMS: dict[str, list] = dict(
        rsi_period=[7, 14, 21],
        oversold=[20.0, 25.0, 30.0],
        overbought=[70.0, 75.0, 80.0],
    )
    _METRIC_NAME = METRIC_LABELS[SHARPE_RATIO]

    def _header(label: str) -> None:
        bar = "█" * 78
        print(f"\n{bar}")
        print(f"██  {label.ljust(72)}  ██")
        print(f"{bar}\n")

    apply_vbt_plot_defaults()
    print(f"Loading {PAIR} ...")
    _, data = load_fx_data(DATA_PATH)

    # 1) SINGLE RUN
    _header(f"RSI DAILY ({PAIR})  ·  SINGLE RUN")
    pf, ind = pipeline(data, **SINGLE_PARAMS)
    analyze_portfolio(
        pf,
        name=f"RSI Daily ({PAIR})",
        output_dir=OUTPUT_DIR,
        show_charts=SHOW_CHARTS,
        indicator=ind,
    )

    # 2) GRID SEARCH
    _header(f"RSI DAILY ({PAIR})  ·  GRID SEARCH")
    grid = run_grid(data, metric_type=SHARPE_RATIO, **GRID_PARAMS)
    print_grid_results(
        grid,
        title=f"RSI Daily ({PAIR}) — Grid Search",
        metric_name=_METRIC_NAME,
        top_n=20,
    )
    if SHOW_CHARTS:
        plot_grid_heatmap(
            grid,
            x_level="oversold",
            y_level="overbought",
            slider_level="rsi_period",
            title=f"RSI Daily — {_METRIC_NAME} heatmap (slider: rsi_period)",
            metric_name=_METRIC_NAME,
        ).show()
        plot_grid_heatmap(
            grid,
            x_level="oversold",
            y_level="overbought",
            slider_level=None,
            title=f"RSI Daily — {_METRIC_NAME} heatmap (aggregated)",
            metric_name=_METRIC_NAME,
        ).show()
        plot_grid_volume(
            grid,
            x_level="oversold",
            y_level="overbought",
            z_level="rsi_period",
            title=f"RSI Daily — {_METRIC_NAME} volume",
            metric_name=_METRIC_NAME,
        ).show()
        plot_grid_surface(
            grid,
            x_level="oversold",
            y_level="overbought",
            slider_level="rsi_period",
            title=f"RSI Daily — {_METRIC_NAME} surface (slider: rsi_period)",
            metric_name=_METRIC_NAME,
        ).show()
        plot_grid_surface(
            grid,
            x_level="oversold",
            y_level="overbought",
            slider_level=None,
            title=f"RSI Daily — {_METRIC_NAME} surface (aggregated)",
            metric_name=_METRIC_NAME,
        ).show()

    # 3) CROSS-VALIDATION
    _header(f"RSI DAILY ({PAIR})  ·  WALK-FORWARD CV")
    # The splitter operates on the resampled daily index to keep
    # the train/test ranges aligned with the RSI computation.
    daily_index = data.close.vbt.resample_apply("1D", "last").dropna().index
    splitter = vbt.Splitter.from_purged_walkforward(
        daily_index,
        n_folds=N_FOLDS,
        n_test_folds=1,
        purge_td="1 day",
        min_train_folds=3,
    )
    if SHOW_CHARTS:
        plot_cv_splitter(splitter, title="RSI Daily — CV Splits").show()
    cv_pipeline = create_cv_pipeline(splitter, metric_type=SHARPE_RATIO)
    grid_perf, best_perf = cv_pipeline(
        data,
        rsi_period=vbt.Param(CV_PARAMS["rsi_period"]),
        oversold=vbt.Param(CV_PARAMS["oversold"]),
        overbought=vbt.Param(CV_PARAMS["overbought"]),
    )
    print_cv_results(
        grid_perf,
        best_perf,
        splitter=splitter,
        title=f"RSI Daily ({PAIR}) — Walk-Forward CV",
        metric_name=_METRIC_NAME,
        top_n=10,
    )
    if SHOW_CHARTS:
        plot_cv_heatmap(
            grid_perf,
            x_level="oversold",
            y_level="overbought",
            slider_level="split",
            splitter=splitter,
            title=f"RSI Daily — CV {_METRIC_NAME} heatmap (per split)",
            metric_name=_METRIC_NAME,
        ).show()
        plot_cv_heatmap(
            grid_perf,
            x_level="oversold",
            y_level="overbought",
            slider_level=None,
            splitter=splitter,
            title=f"RSI Daily — CV {_METRIC_NAME} heatmap (mean across splits)",
            metric_name=_METRIC_NAME,
        ).show()
        plot_cv_volume(
            grid_perf,
            x_level="oversold",
            y_level="overbought",
            z_level="rsi_period",
            slider_level="split",
            splitter=splitter,
            title=f"RSI Daily — CV {_METRIC_NAME} volume (per split)",
            metric_name=_METRIC_NAME,
        ).show()
        plot_cv_volume(
            grid_perf,
            x_level="oversold",
            y_level="overbought",
            z_level="rsi_period",
            slider_level=None,
            splitter=splitter,
            title=f"RSI Daily — CV {_METRIC_NAME} volume (mean across splits)",
            metric_name=_METRIC_NAME,
        ).show()
        plot_grid_surface(
            grid_perf,
            x_level="oversold",
            y_level="overbought",
            slider_level="split",
            splitter=splitter,
            title=f"RSI Daily — CV {_METRIC_NAME} surface (per split)",
            metric_name=_METRIC_NAME,
        ).show()
        plot_grid_surface(
            grid_perf,
            x_level="oversold",
            y_level="overbought",
            slider_level=None,
            splitter=splitter,
            title=f"RSI Daily — CV {_METRIC_NAME} surface (mean across splits)",
            metric_name=_METRIC_NAME,
        ).show()

    print("\nAll modes done.")
