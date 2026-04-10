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
            trace_kwargs=dict(
                name="VWAP", line=dict(color="red", width=1, dash="dot")
            ),
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
    init_cash: float = 1_000_000.0,
    slippage: float = 0.00015,
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
    init_cash: float = 1_000_000.0,
    slippage: float = 0.00015,
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
        init_cash=1_000_000.0,
        slippage=0.00015,
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
        init_cash: float = defaults["init_cash"],
        slippage: float = defaults["slippage"],
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

    ap = argparse.ArgumentParser(description="OU Mean Reversion pipeline (ims format)")
    ap.add_argument("--data", default="data/EUR-USD_minute.parquet")
    ap.add_argument("--mode", choices=["single", "grid", "cv"], default="single")
    ap.add_argument("--bb-window", type=int, default=80)
    ap.add_argument("--bb-alpha", type=float, default=5.0)
    ap.add_argument("--sigma-target", type=float, default=0.10)
    ap.add_argument("--max-leverage", type=float, default=3.0)
    ap.add_argument("--leverage-mult", type=float, default=1.0)
    ap.add_argument("--n-folds", type=int, default=15)
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--output-dir", default="results/ou_mr")
    args = ap.parse_args()

    apply_vbt_plot_defaults()
    print("Loading data...")
    _, data = load_fx_data(args.data)

    if args.mode == "single":
        pf, ind = pipeline(
            data,
            bb_window=args.bb_window,
            bb_alpha=args.bb_alpha,
            sigma_target=args.sigma_target,
            max_leverage=args.max_leverage,
            leverage_mult=args.leverage_mult,
        )
        print(pf.stats())
        analyze_portfolio(
            pf,
            name="OU Mean Reversion",
            output_dir=args.output_dir,
            show_charts=args.show,
            indicator=ind,
        )

    elif args.mode == "grid":
        grid = run_grid(
            data,
            bb_window=[60, 80, 120],
            bb_alpha=[4.0, 5.0, 6.0],
            sigma_target=[0.05, 0.10, 0.20],
            metric_type=SHARPE_RATIO,
        )
        print("\nTop 20 combos by Sharpe:")
        print(grid.sort_values(ascending=False).head(20))
        if args.show:
            grid.vbt.heatmap(
                x_level="bb_window",
                y_level="bb_alpha",
                slider_level="sigma_target",
            ).show()

    elif args.mode == "cv":
        splitter = vbt.Splitter.from_purged_walkforward(
            data.index,
            n_folds=args.n_folds,
            n_test_folds=1,
            purge_td="1 day",
            min_train_folds=3,
        )
        if args.show:
            plot_cv_splitter(splitter, title="OU MR — CV Splits").show()
        cv_pipeline = create_cv_pipeline(splitter, metric_type=SHARPE_RATIO)
        grid_perf, best_perf = cv_pipeline(
            data,
            bb_window=vbt.Param([60, 80, 120]),
            bb_alpha=vbt.Param([4.0, 5.0, 6.0]),
            sigma_target=vbt.Param([0.05, 0.10, 0.20]),
        )
        print("\n▶ Best per split:")
        print(best_perf)
        if args.show:
            plot_cv_heatmap(
                grid_perf,
                x_level="bb_window",
                y_level="bb_alpha",
                slider_level="split",
                title="OU MR — CV Sharpe Heatmap",
            ).show()

    print("\nDone.")
