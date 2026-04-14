"""OU Mean Reversion — VWAP MR + vol-targeted dynamic leverage (ims_pipeline).

Combines the mr_turbo signal base (session-filtered BB mean reversion with
fixed SL/TP/EOD) with a per-bar leverage array driven by a rolling 20-day
realized vol targeting ``sigma_target``:

    leverage(t) = min(sigma_target / rolling_vol(t), max_leverage)

Three entry points:
- ``pipeline(data, **params) -> (pf, OUMRIndicator)`` — investigation
- ``pipeline_nb(data, **params)`` — ``@vbt.parameterized`` scalar metric
- ``create_cv_pipeline(splitter, metric_type)`` — ``@vbt.cv_split`` factory
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import vectorbtpro as vbt

from framework.pipeline_utils import (
    FX_MINUTE_ANN_FACTOR,
    SHARPE_RATIO,
    compute_metric_nb,
    make_execute_kwargs,
)
from framework.project_config import PROJECT_CONFIG
from utils import compute_daily_rolling_volatility_nb, compute_leverage_nb


# ═══════════════════════════════════════════════════════════════════════
# 1. INVESTIGATION PATH — pipeline() returns (pf, indicator)
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class OUMRIndicator:
    """VWAP + BB bands + dynamic leverage overlay."""

    close: pd.Series
    vwap: pd.Series
    upper: pd.Series
    lower: pd.Series
    leverage: pd.Series

    def plot(self, fig: go.Figure | None = None, **layout_kwargs) -> go.Figure:
        fig = fig or go.Figure()
        self.close.vbt.plot(
            fig=fig,
            trace_kwargs=dict(name="Close", line=dict(width=2, color="blue")),
        )
        self.lower.vbt.plot(
            fig=fig,
            trace_kwargs=dict(name="Lower Band", line=dict(width=1.2, color="grey")),
        )
        self.upper.vbt.plot(
            fig=fig,
            trace_kwargs=dict(
                name="Upper Band",
                line=dict(width=1.2, color="grey"),
                fill="tonexty",
                fillcolor="rgba(255, 255, 0, 0.2)",
            ),
        )
        self.vwap.vbt.plot(
            fig=fig,
            trace_kwargs=dict(name="VWAP", line=dict(color="red", width=1, dash="dot")),
        )
        fig.update_layout(**layout_kwargs)
        return fig


def backtest_ou_mr(
    data: vbt.Data,
    **kwargs: Any,
) -> vbt.Portfolio:
    """DEPRECATED shim — use ``pipeline(data, ...)`` instead."""
    pf, _ = pipeline(data, **kwargs)
    return pf


def pipeline(
    data: vbt.Data,
    bb_window: int = 80,
    bb_alpha: float = 5.0,
    sl_stop: float = 0.005,
    tp_stop: float = 0.006,
    session_start: int = 6,
    session_end: int = 14,
    dt_stop: str = "21:00",
    td_stop: str = "6h",
    vol_window: int = 20,
    sigma_target: float = 0.10,
    max_leverage: float = 3.0,
    leverage_mult: float = 1.0,
    init_cash: float | None = None,
    slippage: float | None = PROJECT_CONFIG["slippage_intraday"],
    fees: float | None = None,
) -> tuple[vbt.Portfolio, OUMRIndicator]:
    """Investigation path — bit-equivalent to the legacy ``backtest_ou_mr``.

    VWAP MR + vol-targeted dynamic leverage. The ``leverage`` array is
    passed to ``Portfolio.from_signals`` which broadcasts it per-bar.
    """
    close = data.close

    vwap = vbt.VWAP.run(data.high, data.low, close, data.volume, anchor="D").vwap
    deviation = close - vwap
    bb = vbt.BBANDS.run(deviation, window=bb_window, alpha=bb_alpha)
    upper = vwap + bb.upper
    lower = vwap + bb.lower

    hours = close.index.hour
    session = (hours >= session_start) & (hours < session_end)

    entries = (close < lower) & session
    short_entries = (close > upper) & session

    index_ns = vbt.dt.to_ns(close.index)
    close_vals = np.ascontiguousarray(close.values, dtype=np.float64)
    rolling_vol = compute_daily_rolling_volatility_nb(index_ns, close_vals, vol_window)
    leverage_arr = compute_leverage_nb(rolling_vol, sigma_target, max_leverage)
    leverage_arr = leverage_arr * float(leverage_mult)
    nan_count = int(np.isnan(leverage_arr).sum())
    if nan_count > 0:
        warnings.warn(
            f"ou_mean_reversion: {nan_count} NaN leverage value(s) replaced "
            f"with 1.0 (out of {leverage_arr.size}). Likely vol_window warmup "
            f"or a data gap — check input close series.",
            RuntimeWarning,
            stacklevel=2,
        )
        leverage_arr = np.where(np.isnan(leverage_arr), 1.0, leverage_arr)

    pf = vbt.Portfolio.from_signals(
        data,
        entries=entries,
        exits=False,
        short_entries=short_entries,
        short_exits=False,
        sl_stop=sl_stop,
        tp_stop=tp_stop,
        dt_stop=dt_stop,
        td_stop=td_stop,
        slippage=slippage,
        init_cash=init_cash,
        leverage=leverage_arr,
        fees=fees,
        freq="1min",
    )
    indicator = OUMRIndicator(
        close=close,
        vwap=vwap.rename("VWAP"),
        upper=upper.rename("Upper Band"),
        lower=lower.rename("Lower Band"),
        leverage=pd.Series(leverage_arr, index=close.index, name="Leverage"),
    )
    return pf, indicator


# ═══════════════════════════════════════════════════════════════════════
# 2. GRID-SEARCH PATH — pipeline_nb (@vbt.parameterized)
# ═══════════════════════════════════════════════════════════════════════


@vbt.parameterized(
    merge_func="concat",
    execute_kwargs=make_execute_kwargs("OU MR grid"),
)
def pipeline_nb(
    data: vbt.Data,
    bb_window: int,
    bb_alpha: float,
    sl_stop: float = 0.005,
    tp_stop: float = 0.006,
    sigma_target: float = 0.10,
    max_leverage: float = 3.0,
    vol_window: int = 20,
    leverage_mult: float = 1.0,
    session_start: int = 6,
    session_end: int = 14,
    dt_stop: str = "21:00",
    td_stop: str = "6h",
    init_cash: float | None = None,
    slippage: float | None = PROJECT_CONFIG["slippage_intraday"],
    fees: float | None = None,
    ann_factor: float = FX_MINUTE_ANN_FACTOR,
    cutoff: float = 0.05,
    metric_type: int = SHARPE_RATIO,
) -> float:
    """Grid-search path — scalar metric per param combo."""
    pf, _ = pipeline(
        data,
        bb_window=bb_window,
        bb_alpha=bb_alpha,
        sl_stop=sl_stop,
        tp_stop=tp_stop,
        session_start=session_start,
        session_end=session_end,
        dt_stop=dt_stop,
        td_stop=td_stop,
        vol_window=vol_window,
        sigma_target=sigma_target,
        max_leverage=max_leverage,
        leverage_mult=leverage_mult,
        init_cash=init_cash,
        slippage=slippage,
        fees=fees,
    )
    returns = pf.returns.values
    if returns.ndim > 1:
        returns = returns[:, 0]
    return float(compute_metric_nb(returns, metric_type, ann_factor, cutoff))


def run_grid(
    data: vbt.Data,
    *,
    bb_window: list[int] | int,
    bb_alpha: list[float] | float,
    sigma_target: list[float] | float = 0.10,
    max_leverage: list[float] | float = 3.0,
    metric_type: int = SHARPE_RATIO,
    **kwargs: Any,
) -> pd.Series:
    def _param(v):
        if isinstance(v, (list, tuple, np.ndarray)):
            return vbt.Param(list(v))
        return v

    return pipeline_nb(
        data,
        bb_window=_param(bb_window),
        bb_alpha=_param(bb_alpha),
        sigma_target=_param(sigma_target),
        max_leverage=_param(max_leverage),
        metric_type=metric_type,
        **kwargs,
    )


# ═══════════════════════════════════════════════════════════════════════
# 3. CV FACTORY — create_cv_pipeline
# ═══════════════════════════════════════════════════════════════════════


def create_cv_pipeline(
    splitter: Any,
    metric_type: int = SHARPE_RATIO,
    **pipeline_defaults: Any,
):
    splitter_kwargs = pipeline_defaults.pop("splitter_kwargs", {})

    defaults = dict(
        sl_stop=0.005,
        tp_stop=0.006,
        sigma_target=0.10,
        max_leverage=3.0,
        vol_window=20,
        leverage_mult=1.0,
        session_start=6,
        session_end=14,
        dt_stop="21:00",
        td_stop="6h",
        init_cash=None,
        slippage=PROJECT_CONFIG["slippage_intraday"],
        fees=None,
        ann_factor=FX_MINUTE_ANN_FACTOR,
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
                "OU MR combos", pbar_kwargs=dict(leave=False)
            ),
            merge_func="concat",
        ),
        execute_kwargs=make_execute_kwargs("OU MR CV splits"),
        merge_func="concat",
        return_grid="all",
        attach_bounds="index",
    )
    def cv_pipeline(
        data: vbt.Data,
        bb_window: int,
        bb_alpha: float,
        sl_stop: float = defaults["sl_stop"],
        tp_stop: float = defaults["tp_stop"],
        sigma_target: float = defaults["sigma_target"],
        max_leverage: float = defaults["max_leverage"],
        vol_window: int = defaults["vol_window"],
        leverage_mult: float = defaults["leverage_mult"],
        session_start: int = defaults["session_start"],
        session_end: int = defaults["session_end"],
        dt_stop: str = defaults["dt_stop"],
        td_stop: str = defaults["td_stop"],
        init_cash: float | None = defaults["init_cash"],
        slippage: float | None = defaults["slippage"],
        fees: float | None = defaults["fees"],
        ann_factor: float = defaults["ann_factor"],
        cutoff: float = defaults["cutoff"],
        metric_type: int = defaults["metric_type"],
    ) -> float:
        pf, _ = pipeline(
            data,
            bb_window=bb_window,
            bb_alpha=bb_alpha,
            sl_stop=sl_stop,
            tp_stop=tp_stop,
            session_start=session_start,
            session_end=session_end,
            dt_stop=dt_stop,
            td_stop=td_stop,
            vol_window=vol_window,
            sigma_target=sigma_target,
            max_leverage=max_leverage,
            leverage_mult=leverage_mult,
            init_cash=init_cash,
            slippage=slippage,
            fees=fees,
        )
        returns = pf.returns.values
        if returns.ndim > 1:
            returns = returns[:, 0]
        return float(compute_metric_nb(returns, metric_type, ann_factor, cutoff))

    return cv_pipeline


# ═══════════════════════════════════════════════════════════════════════
# 4. CLI — single / grid / cv
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
    from framework.project_config import data_path, results_dir
    from utils import load_fx_data

    PAIR = PROJECT_CONFIG["default_pair"]
    DATA_PATH = str(data_path(PAIR))
    OUTPUT_DIR = str(results_dir("ou_mr"))
    SHOW_CHARTS = PROJECT_CONFIG["show_charts"]
    N_FOLDS = 15  # ou_mr-specific: finer-grained walk-forward

    SINGLE_PARAMS: dict[str, Any] = dict(
        bb_window=80,
        bb_alpha=5.0,
        sigma_target=0.10,
        max_leverage=3.0,
        leverage_mult=1.0,
    )
    GRID_PARAMS: dict[str, list] = dict(
        bb_window=[60, 80, 120],
        bb_alpha=[4.0, 5.0, 6.0],
        sigma_target=[0.05, 0.10, 0.20],
    )
    _METRIC_NAME = METRIC_LABELS[SHARPE_RATIO]

    def _header(label: str) -> None:
        bar = "█" * 78
        print(f"\n{bar}")
        print(f"██  {label.ljust(72)}  ██")
        print(f"{bar}\n")

    apply_vbt_plot_defaults()
    print("Loading data...")
    _, data = load_fx_data(DATA_PATH)

    # 1) SINGLE RUN
    _header("OU MEAN REVERSION  ·  SINGLE RUN")
    pf, ind = pipeline(data, **SINGLE_PARAMS)
    analyze_portfolio(
        pf,
        name="OU Mean Reversion",
        output_dir=OUTPUT_DIR,
        show_charts=SHOW_CHARTS,
        indicator=ind,
    )

    # 2) GRID SEARCH
    _header("OU MEAN REVERSION  ·  GRID SEARCH")
    grid = run_grid(data, metric_type=SHARPE_RATIO, **GRID_PARAMS)
    print_grid_results(
        grid,
        title="OU Mean Reversion — Grid Search",
        metric_name=_METRIC_NAME,
        top_n=20,
    )
    if SHOW_CHARTS:
        plot_grid_heatmap(
            grid,
            x_level="bb_window",
            y_level="bb_alpha",
            slider_level="sigma_target",
            title=f"OU MR — {_METRIC_NAME} heatmap (slider: sigma_target)",
            metric_name=_METRIC_NAME,
        ).show()
        plot_grid_heatmap(
            grid,
            x_level="bb_window",
            y_level="bb_alpha",
            slider_level=None,
            title=f"OU MR — {_METRIC_NAME} heatmap (aggregated)",
            metric_name=_METRIC_NAME,
        ).show()
        plot_grid_volume(
            grid,
            x_level="bb_window",
            y_level="bb_alpha",
            z_level="sigma_target",
            title=f"OU MR — {_METRIC_NAME} volume",
            metric_name=_METRIC_NAME,
        ).show()
        plot_grid_surface(
            grid,
            x_level="bb_window",
            y_level="bb_alpha",
            slider_level="sigma_target",
            title=f"OU MR — {_METRIC_NAME} surface (slider: sigma_target)",
            metric_name=_METRIC_NAME,
        ).show()
        plot_grid_surface(
            grid,
            x_level="bb_window",
            y_level="bb_alpha",
            slider_level=None,
            title=f"OU MR — {_METRIC_NAME} surface (aggregated)",
            metric_name=_METRIC_NAME,
        ).show()

    # 3) CROSS-VALIDATION
    _header("OU MEAN REVERSION  ·  WALK-FORWARD CV")
    # Daily-resampled index keeps the splitter plot light and fast.
    daily_index = data.close.vbt.resample_apply("1D", "last").dropna().index
    splitter = vbt.Splitter.from_purged_walkforward(
        daily_index,
        n_folds=N_FOLDS,
        n_test_folds=1,
        purge_td="1 day",
        min_train_folds=3,
    )
    if SHOW_CHARTS:
        plot_cv_splitter(splitter, title="OU MR — CV Splits").show()
    cv_pipeline = create_cv_pipeline(splitter, metric_type=SHARPE_RATIO)
    grid_perf, best_perf = cv_pipeline(
        data,
        bb_window=vbt.Param(GRID_PARAMS["bb_window"]),
        bb_alpha=vbt.Param(GRID_PARAMS["bb_alpha"]),
        sigma_target=vbt.Param(GRID_PARAMS["sigma_target"]),
    )
    print_cv_results(
        grid_perf,
        best_perf,
        splitter=splitter,
        title="OU Mean Reversion — Walk-Forward CV",
        metric_name=_METRIC_NAME,
        top_n=10,
    )
    if SHOW_CHARTS:
        plot_cv_heatmap(
            grid_perf,
            x_level="bb_window",
            y_level="bb_alpha",
            slider_level="split",
            splitter=splitter,
            title=f"OU MR — CV {_METRIC_NAME} heatmap (per split)",
            metric_name=_METRIC_NAME,
        ).show()
        plot_cv_heatmap(
            grid_perf,
            x_level="bb_window",
            y_level="bb_alpha",
            slider_level=None,
            splitter=splitter,
            title=f"OU MR — CV {_METRIC_NAME} heatmap (mean across splits)",
            metric_name=_METRIC_NAME,
        ).show()
        plot_cv_volume(
            grid_perf,
            x_level="bb_window",
            y_level="bb_alpha",
            z_level="sigma_target",
            slider_level="split",
            splitter=splitter,
            title=f"OU MR — CV {_METRIC_NAME} volume (per split)",
            metric_name=_METRIC_NAME,
        ).show()
        plot_cv_volume(
            grid_perf,
            x_level="bb_window",
            y_level="bb_alpha",
            z_level="sigma_target",
            slider_level=None,
            splitter=splitter,
            title=f"OU MR — CV {_METRIC_NAME} volume (mean across splits)",
            metric_name=_METRIC_NAME,
        ).show()
        plot_grid_surface(
            grid_perf,
            x_level="bb_window",
            y_level="bb_alpha",
            slider_level="split",
            splitter=splitter,
            title=f"OU MR — CV {_METRIC_NAME} surface (per split)",
            metric_name=_METRIC_NAME,
        ).show()
        plot_grid_surface(
            grid_perf,
            x_level="bb_window",
            y_level="bb_alpha",
            slider_level=None,
            splitter=splitter,
            title=f"OU MR — CV {_METRIC_NAME} surface (mean across splits)",
            metric_name=_METRIC_NAME,
        ).show()

    print("\nAll modes done.")
