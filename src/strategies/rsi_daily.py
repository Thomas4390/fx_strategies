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
    execute_kwargs=dict(chunk_len="auto", engine="threadpool"),
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
            execute_kwargs=dict(chunk_len="auto", engine="threadpool"),
            merge_func="concat",
        ),
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
    import argparse
    import sys
    from pathlib import Path as _Path

    _SRC = _Path(__file__).resolve().parent.parent
    if str(_SRC) not in sys.path:
        sys.path.insert(0, str(_SRC))

    from framework.pipeline_utils import (
        analyze_portfolio,
        apply_vbt_plot_defaults,
        plot_cv_heatmap,
        plot_cv_splitter,
    )
    from utils import load_fx_data

    ap = argparse.ArgumentParser(description="RSI Daily pipeline (ims format)")
    ap.add_argument(
        "--pair",
        default="EUR-USD",
        choices=["EUR-USD", "GBP-USD", "USD-JPY", "USD-CAD"],
    )
    ap.add_argument("--mode", choices=["single", "grid", "cv"], default="single")
    ap.add_argument("--rsi-period", type=int, default=14)
    ap.add_argument("--oversold", type=float, default=25.0)
    ap.add_argument("--overbought", type=float, default=75.0)
    ap.add_argument("--leverage", type=float, default=1.0)
    ap.add_argument("--n-folds", type=int, default=10)
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--output-dir", default=None)
    args = ap.parse_args()

    apply_vbt_plot_defaults()
    data_path = f"data/{args.pair}_minute.parquet"
    print(f"Loading {args.pair} ...")
    _, data = load_fx_data(data_path)
    output_dir = args.output_dir or f"results/rsi_daily_{args.pair.lower()}"

    if args.mode == "single":
        pf, ind = pipeline(
            data,
            rsi_period=args.rsi_period,
            oversold=args.oversold,
            overbought=args.overbought,
            leverage=args.leverage,
        )
        print(pf.stats())
        analyze_portfolio(
            pf,
            name=f"RSI Daily ({args.pair})",
            output_dir=output_dir,
            show_charts=args.show,
            indicator=ind,
        )

    elif args.mode == "grid":
        grid = run_grid(
            data,
            rsi_period=[7, 10, 14, 21],
            oversold=[20.0, 25.0, 30.0],
            overbought=[70.0, 75.0, 80.0],
            metric_type=SHARPE_RATIO,
        )
        print("\nTop 20 combos by Sharpe:")
        print(grid.sort_values(ascending=False).head(20))
        if args.show:
            fig = grid.vbt.heatmap(
                x_level="oversold",
                y_level="overbought",
                slider_level="rsi_period",
            )
            fig.show()

    elif args.mode == "cv":
        # The splitter operates on the resampled daily index to keep
        # the train/test ranges aligned with the RSI computation.
        daily_index = data.close.resample("1D").last().dropna().index
        splitter = vbt.Splitter.from_purged_walkforward(
            daily_index,
            n_folds=args.n_folds,
            n_test_folds=1,
            purge_td="1 day",
            min_train_folds=3,
        )
        if args.show:
            plot_cv_splitter(splitter, title="RSI Daily — CV Splits").show()
        cv_pipeline = create_cv_pipeline(splitter, metric_type=SHARPE_RATIO)
        grid_perf, best_perf = cv_pipeline(
            data,
            rsi_period=vbt.Param([7, 14, 21]),
            oversold=vbt.Param([20.0, 25.0, 30.0]),
            overbought=vbt.Param([70.0, 75.0, 80.0]),
        )
        print("\n▶ Best per split:")
        print(best_perf)
        if args.show:
            plot_cv_heatmap(
                grid_perf,
                x_level="oversold",
                y_level="overbought",
                slider_level="split",
                title=f"RSI Daily ({args.pair}) — CV Sharpe Heatmap",
            ).show()

    print("\nDone.")
