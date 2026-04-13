"""Composite FX Alpha — Momentum + Volatility Timing (Daily, ims_pipeline format).

Multi-factor daily strategy: blended momentum (w_short/w_long), volatility
regime detection, ewma vol targeting, drawdown control with hysteresis, and
K-overlapping Jegadeesh-Titman sub-portfolios.

Three entry points:
- ``pipeline(data, **params) -> (pf, CompositeAlphaIndicator)``
- ``pipeline_nb(data, **params)`` — ``@vbt.parameterized`` scalar metric
- ``create_cv_pipeline(splitter, metric_type)`` — ``@vbt.cv_split`` factory

Note
----
The Phase 6 rewrite replaces the legacy ``composite_signal_nb`` delta-sizing
path (which was coupled to StrategyRunner + StrategySpec) with a simpler
``vbt.Portfolio.from_orders(size=target_weights, size_type='targetpercent')``.
The five Numba kernels that produce the target weights are unchanged. The
result is economically equivalent but not bit-identical to the legacy
StrategyRunner output — see the refactor plan for details.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import vectorbtpro as vbt
from numba import njit

from framework.pipeline_utils import (
    SHARPE_RATIO,
    compute_metric_nb,
    make_execute_kwargs,
)

# Daily strategy → annualization factor = 252
COMPOSITE_ANN_FACTOR: float = 252.0


# ═══════════════════════════════════════════════════════════════════════
# NUMBA KERNELS (unchanged from the legacy StrategySpec implementation)
# ═══════════════════════════════════════════════════════════════════════


@njit(nogil=True)
def momentum_signal_nb(close: np.ndarray, w_short: int, w_long: int) -> np.ndarray:
    """Blended momentum: 0.5 * log_return(21d) + 0.5 * log_return(63d)."""
    n = len(close)
    out = np.full(n, np.nan)
    for i in range(w_long, n):
        r_short = np.log(close[i] / close[i - w_short])
        r_long = np.log(close[i] / close[i - w_long])
        out[i] = 0.5 * r_short + 0.5 * r_long
    return out


@njit(nogil=True)
def regime_weight_nb(
    vr: np.ndarray,
    low_th: float,
    high_th: float,
    w_low: float,
    w_normal: float,
    w_high: float,
) -> np.ndarray:
    """Map volatility regime ratio to momentum weight."""
    n = len(vr)
    out = np.full(n, np.nan)
    for i in range(n):
        if np.isnan(vr[i]):
            continue
        elif vr[i] < low_th:
            out[i] = w_low
        elif vr[i] > high_th:
            out[i] = w_high
        else:
            out[i] = w_normal
    return out


@njit(nogil=True)
def vol_scaling_nb(ewma_vol: np.ndarray, target: float, cap: float) -> np.ndarray:
    """lambda_t = min(sigma_target / sigma_Pt, lambda_max)."""
    n = len(ewma_vol)
    out = np.full(n, np.nan)
    for i in range(n):
        v = ewma_vol[i]
        if not np.isnan(v) and v > 0:
            out[i] = min(target / v, cap)
    return out


@njit(nogil=True)
def drawdown_control_nb(
    dd: np.ndarray,
    soft: float,
    hard: float,
    recovery: float,
) -> np.ndarray:
    """State machine: NORMAL(1.0) -> REDUCED(0.5) -> FLAT(0.0) with hysteresis."""
    n = len(dd)
    mult = np.ones(n)
    state = 0
    for i in range(n):
        d = dd[i]
        if state == 0:
            if d > hard:
                state = 2
            elif d > soft:
                state = 1
        elif state == 1:
            if d > hard:
                state = 2
            elif d < recovery:
                state = 0
        elif state == 2 and d < recovery:
            state = 0
        if state == 1:
            mult[i] = 0.5
        elif state == 2:
            mult[i] = 0.0
    return mult


@njit(nogil=True)
def sub_portfolio_weights_nb(
    direction: np.ndarray,
    regime_wt: np.ndarray,
    vol_scale: np.ndarray,
    dd_mult: np.ndarray,
    n_days: int,
    k: int,
) -> np.ndarray:
    """K=5 overlapping sub-portfolio weights (Jegadeesh-Titman)."""
    sub_w = np.zeros((k, n_days))
    for j in range(k):
        current = 0.0
        for i in range(n_days):
            if i >= j * 5 and (i - j * 5) % (k * 5) == 0:
                d = direction[i]
                r = regime_wt[i]
                v = vol_scale[i]
                m = dd_mult[i]
                if not (np.isnan(d) or np.isnan(r) or np.isnan(v) or np.isnan(m)):
                    current = d * r * v * m
            sub_w[j, i] = current

    weights = np.zeros(n_days)
    for i in range(n_days):
        total = 0.0
        for j in range(k):
            total += sub_w[j, i]
        weights[i] = total / k
    return weights


@njit(nogil=True)
def compute_composite_nb(
    close: np.ndarray,
    returns: np.ndarray,
    w_short: int,
    w_long: int,
    vol_short: int,
    vol_long: int,
    ewma_span: int,
    target_vol: float,
    leverage_cap: float,
    vr_low: float,
    vr_high: float,
    mom_w_low: float,
    mom_w_normal: float,
    mom_w_high: float,
    dd_soft: float,
    dd_hard: float,
    dd_recovery: float,
    n_sub: int,
) -> tuple:
    """Master kernel: compute all signals -> daily target weights."""
    n = len(close)

    momentum = momentum_signal_nb(close, w_short, w_long)
    direction = np.full(n, 0.0)
    for i in range(n):
        if not np.isnan(momentum[i]):
            direction[i] = (
                1.0 if momentum[i] > 0 else (-1.0 if momentum[i] < 0 else 0.0)
            )

    sigma_short = vbt.generic.nb.rolling_std_1d_nb(
        returns, vol_short, minp=vol_short, ddof=1
    )
    sigma_long = vbt.generic.nb.rolling_std_1d_nb(
        returns, vol_long, minp=vol_long, ddof=1
    )
    vr = np.full(n, np.nan)
    for i in range(n):
        if (
            not np.isnan(sigma_short[i])
            and not np.isnan(sigma_long[i])
            and sigma_long[i] > 0
        ):
            vr[i] = sigma_short[i] / sigma_long[i]

    regime_wt = regime_weight_nb(
        vr, vr_low, vr_high, mom_w_low, mom_w_normal, mom_w_high
    )

    sq_returns = np.empty(n)
    for i in range(n):
        sq_returns[i] = returns[i] ** 2 if not np.isnan(returns[i]) else np.nan
    ewma_var = vbt.generic.nb.ewm_mean_1d_nb(sq_returns, ewma_span, minp=1, adjust=True)
    ewma_vol = np.full(n, np.nan)
    for i in range(n):
        if not np.isnan(ewma_var[i]) and ewma_var[i] > 0:
            ewma_vol[i] = np.sqrt(ewma_var[i]) * np.sqrt(252)
    vol_scale = vol_scaling_nb(ewma_vol, target_vol, leverage_cap)

    proxy_equity = np.ones(n)
    for i in range(1, n):
        d = direction[i - 1]
        rw = regime_wt[i - 1]
        v = vol_scale[i - 1]
        r = returns[i]
        if not (np.isnan(d) or np.isnan(rw) or np.isnan(v) or np.isnan(r)):
            proxy_equity[i] = proxy_equity[i - 1] * (1 + r * d * rw * v)
        else:
            proxy_equity[i] = proxy_equity[i - 1]

    lookback = 63
    dd = np.zeros(n)
    for i in range(n):
        peak = proxy_equity[max(0, i - lookback + 1)]
        for j in range(max(0, i - lookback + 1), i + 1):
            if proxy_equity[j] > peak:
                peak = proxy_equity[j]
        dd[i] = 1.0 - proxy_equity[i] / peak if peak > 0 else 0.0

    dd_mult = drawdown_control_nb(dd, dd_soft, dd_hard, dd_recovery)

    weights = sub_portfolio_weights_nb(
        direction, regime_wt, vol_scale, dd_mult, n, n_sub
    )

    return momentum, direction, vr, regime_wt, ewma_vol, vol_scale, dd, dd_mult, weights


# ═══════════════════════════════════════════════════════════════════════
# 1. INVESTIGATION PATH — pipeline() returns (pf, indicator)
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class CompositeAlphaIndicator:
    """Composite alpha signals for overlay plotting."""

    close: pd.Series
    momentum: pd.Series
    vr: pd.Series
    vol_scale: pd.Series
    target_weight: pd.Series
    drawdown: pd.Series

    def plot(self, fig: go.Figure | None = None, **layout_kwargs) -> go.Figure:
        from plotly.subplots import make_subplots

        fig = fig or make_subplots(
            rows=4, cols=1,
            shared_xaxes=True,
            subplot_titles=("Close", "Momentum", "Vol Scale", "Target Weight"),
            vertical_spacing=0.05,
        )
        fig.add_scatter(
            x=self.close.index, y=self.close.values, name="Close", row=1, col=1,
        )
        fig.add_scatter(
            x=self.momentum.index, y=self.momentum.values, name="Momentum", row=2, col=1,
        )
        fig.add_scatter(
            x=self.vol_scale.index, y=self.vol_scale.values, name="Vol Scale", row=3, col=1,
        )
        fig.add_scatter(
            x=self.target_weight.index, y=self.target_weight.values, name="Target Weight", row=4, col=1,
        )
        fig.update_layout(**layout_kwargs)
        return fig


def pipeline(
    data: vbt.Data | pd.Series,
    w_short: int = 21,
    w_long: int = 63,
    vol_short: int = 21,
    vol_long: int = 252,
    ewma_span: int = 30,
    target_vol: float = 0.10,
    leverage_cap: float = 3.0,
    vr_low: float = 0.8,
    vr_high: float = 1.2,
    mom_w_low: float = 0.20,
    mom_w_normal: float = 0.30,
    mom_w_high: float = 0.50,
    dd_soft: float = 0.12,
    dd_hard: float = 0.20,
    dd_recovery: float = 0.10,
    n_sub: int = 5,
    leverage: float = 2.0,
    fees: float = 0.00035,
    init_cash: float = 1_000_000.0,
) -> tuple[vbt.Portfolio, CompositeAlphaIndicator]:
    """Investigation path — Composite FX Alpha daily rebalance.

    Resamples minute to daily, runs the 5 Numba kernels through
    ``compute_composite_nb``, and rebalances to ``target_weight`` via
    :meth:`vbt.Portfolio.from_orders` with ``size_type='targetpercent'``.

    This differs from the legacy StrategyRunner path (which used
    ``composite_signal_nb`` delta sizing) — see module docstring.
    """
    if hasattr(data, "close"):
        close_any = data.close
    else:
        close_any = data
    close_daily = close_any.vbt.resample_apply("1D", "last").dropna()
    returns_daily = np.log(close_daily / close_daily.shift(1)).fillna(0.0)

    (
        momentum,
        direction,
        vr,
        regime_wt,
        ewma_vol,
        vol_scale,
        dd,
        dd_mult,
        target_weights,
    ) = compute_composite_nb(
        close_daily.values,
        returns_daily.values,
        w_short, w_long,
        vol_short, vol_long,
        ewma_span,
        target_vol,
        leverage_cap,
        vr_low, vr_high,
        mom_w_low, mom_w_normal, mom_w_high,
        dd_soft, dd_hard, dd_recovery,
        n_sub,
    )

    # Shift by 1 to avoid look-ahead and convert NaN -> 0
    target_w_series = pd.Series(
        target_weights, index=close_daily.index, name="target_weight"
    ).shift(1).fillna(0.0)

    pf = vbt.Portfolio.from_orders(
        close=close_daily,
        size=target_w_series,
        size_type="targetpercent",
        init_cash=init_cash,
        leverage=leverage,
        fees=fees,
        freq="1D",
    )

    indicator = CompositeAlphaIndicator(
        close=close_daily,
        momentum=pd.Series(momentum, index=close_daily.index, name="momentum"),
        vr=pd.Series(vr, index=close_daily.index, name="vol_regime"),
        vol_scale=pd.Series(vol_scale, index=close_daily.index, name="vol_scale"),
        target_weight=target_w_series,
        drawdown=pd.Series(dd, index=close_daily.index, name="drawdown"),
    )
    return pf, indicator


# ═══════════════════════════════════════════════════════════════════════
# 2. GRID-SEARCH PATH — pipeline_nb (@vbt.parameterized)
# ═══════════════════════════════════════════════════════════════════════


@vbt.parameterized(
    merge_func="concat",
    execute_kwargs=make_execute_kwargs("Composite FX Alpha grid"),
)
def pipeline_nb(
    data: vbt.Data | pd.Series,
    w_short: int,
    w_long: int,
    target_vol: float = 0.10,
    leverage_cap: float = 3.0,
    dd_soft: float = 0.12,
    dd_hard: float = 0.20,
    vol_short: int = 21,
    vol_long: int = 252,
    ewma_span: int = 30,
    vr_low: float = 0.8,
    vr_high: float = 1.2,
    mom_w_low: float = 0.20,
    mom_w_normal: float = 0.30,
    mom_w_high: float = 0.50,
    dd_recovery: float = 0.10,
    n_sub: int = 5,
    leverage: float = 2.0,
    fees: float = 0.00035,
    init_cash: float = 1_000_000.0,
    ann_factor: float = COMPOSITE_ANN_FACTOR,
    cutoff: float = 0.05,
    metric_type: int = SHARPE_RATIO,
) -> float:
    """Grid-search path — scalar metric per param combo."""
    pf, _ = pipeline(
        data,
        w_short=w_short,
        w_long=w_long,
        vol_short=vol_short,
        vol_long=vol_long,
        ewma_span=ewma_span,
        target_vol=target_vol,
        leverage_cap=leverage_cap,
        vr_low=vr_low,
        vr_high=vr_high,
        mom_w_low=mom_w_low,
        mom_w_normal=mom_w_normal,
        mom_w_high=mom_w_high,
        dd_soft=dd_soft,
        dd_hard=dd_hard,
        dd_recovery=dd_recovery,
        n_sub=n_sub,
        leverage=leverage,
        fees=fees,
        init_cash=init_cash,
    )
    returns = pf.returns.values
    if returns.ndim > 1:
        returns = returns[:, 0]
    return float(compute_metric_nb(returns, metric_type, ann_factor, cutoff))


def run_grid(
    data: vbt.Data | pd.Series,
    *,
    w_short: list[int] | int,
    w_long: list[int] | int,
    target_vol: list[float] | float = 0.10,
    metric_type: int = SHARPE_RATIO,
    **kwargs: Any,
) -> pd.Series:
    def _param(v):
        if isinstance(v, (list, tuple, np.ndarray)):
            return vbt.Param(list(v))
        return v

    return pipeline_nb(
        data,
        w_short=_param(w_short),
        w_long=_param(w_long),
        target_vol=_param(target_vol),
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
    splitter_kwargs = pipeline_defaults.pop("splitter_kwargs", {})
    defaults = dict(
        target_vol=0.10,
        leverage_cap=3.0,
        dd_soft=0.12,
        dd_hard=0.20,
        vol_short=21,
        vol_long=252,
        ewma_span=30,
        vr_low=0.8,
        vr_high=1.2,
        mom_w_low=0.20,
        mom_w_normal=0.30,
        mom_w_high=0.50,
        dd_recovery=0.10,
        n_sub=5,
        leverage=2.0,
        fees=0.00035,
        init_cash=1_000_000.0,
        ann_factor=COMPOSITE_ANN_FACTOR,
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
                "Composite FX Alpha combos", pbar_kwargs=dict(leave=False)
            ),
            merge_func="concat",
        ),
        execute_kwargs=make_execute_kwargs("Composite FX Alpha CV splits"),
        merge_func="concat",
        return_grid="all",
        attach_bounds="index",
    )
    def cv_pipeline(
        data: vbt.Data | pd.Series,
        w_short: int,
        w_long: int,
        target_vol: float = defaults["target_vol"],
        leverage_cap: float = defaults["leverage_cap"],
        dd_soft: float = defaults["dd_soft"],
        dd_hard: float = defaults["dd_hard"],
        vol_short: int = defaults["vol_short"],
        vol_long: int = defaults["vol_long"],
        ewma_span: int = defaults["ewma_span"],
        vr_low: float = defaults["vr_low"],
        vr_high: float = defaults["vr_high"],
        mom_w_low: float = defaults["mom_w_low"],
        mom_w_normal: float = defaults["mom_w_normal"],
        mom_w_high: float = defaults["mom_w_high"],
        dd_recovery: float = defaults["dd_recovery"],
        n_sub: int = defaults["n_sub"],
        leverage: float = defaults["leverage"],
        fees: float = defaults["fees"],
        init_cash: float = defaults["init_cash"],
        ann_factor: float = defaults["ann_factor"],
        cutoff: float = defaults["cutoff"],
        metric_type: int = defaults["metric_type"],
    ) -> float:
        pf, _ = pipeline(
            data,
            w_short=w_short,
            w_long=w_long,
            vol_short=vol_short,
            vol_long=vol_long,
            ewma_span=ewma_span,
            target_vol=target_vol,
            leverage_cap=leverage_cap,
            vr_low=vr_low,
            vr_high=vr_high,
            mom_w_low=mom_w_low,
            mom_w_normal=mom_w_normal,
            mom_w_high=mom_w_high,
            dd_soft=dd_soft,
            dd_hard=dd_hard,
            dd_recovery=dd_recovery,
            n_sub=n_sub,
            leverage=leverage,
            fees=fees,
            init_cash=init_cash,
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
    from utils import load_fx_data

    DATA_PATH = "data/EUR-USD_minute.parquet"
    OUTPUT_DIR = "results/composite_fx_alpha"
    SHOW_CHARTS = True
    N_FOLDS = 8

    SINGLE_PARAMS: dict[str, Any] = dict(target_vol=0.10, leverage=2.0)
    GRID_PARAMS: dict[str, list] = dict(
        w_short=[10, 21, 42],
        w_long=[42, 63, 126],
        target_vol=[0.05, 0.08, 0.10, 0.15],
    )
    CV_PARAMS: dict[str, list] = dict(
        w_short=[10, 21, 42],
        w_long=[42, 63, 126],
        target_vol=[0.08, 0.10, 0.15],
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
    _header("COMPOSITE FX ALPHA  ·  SINGLE RUN")
    pf, ind = pipeline(data, **SINGLE_PARAMS)
    analyze_portfolio(
        pf,
        name="Composite FX Alpha",
        output_dir=OUTPUT_DIR,
        show_charts=SHOW_CHARTS,
        indicator=ind,
    )

    # 2) GRID SEARCH
    _header("COMPOSITE FX ALPHA  ·  GRID SEARCH")
    grid = run_grid(data, metric_type=SHARPE_RATIO, **GRID_PARAMS)
    print_grid_results(
        grid,
        title="Composite FX Alpha — Grid Search",
        metric_name=_METRIC_NAME,
        top_n=20,
    )
    if SHOW_CHARTS:
        plot_grid_heatmap(
            grid,
            x_level="w_short",
            y_level="w_long",
            slider_level="target_vol",
            title=f"Composite Alpha — {_METRIC_NAME} heatmap (slider: target_vol)",
            metric_name=_METRIC_NAME,
        ).show()
        plot_grid_heatmap(
            grid,
            x_level="w_short",
            y_level="w_long",
            slider_level=None,
            title=f"Composite Alpha — {_METRIC_NAME} heatmap (aggregated)",
            metric_name=_METRIC_NAME,
        ).show()
        plot_grid_volume(
            grid,
            x_level="w_short",
            y_level="w_long",
            z_level="target_vol",
            title=f"Composite Alpha — {_METRIC_NAME} volume",
            metric_name=_METRIC_NAME,
        ).show()
        plot_grid_surface(
            grid,
            x_level="w_short",
            y_level="w_long",
            slider_level="target_vol",
            title=f"Composite Alpha — {_METRIC_NAME} surface (slider: target_vol)",
            metric_name=_METRIC_NAME,
        ).show()
        plot_grid_surface(
            grid,
            x_level="w_short",
            y_level="w_long",
            slider_level=None,
            title=f"Composite Alpha — {_METRIC_NAME} surface (aggregated)",
            metric_name=_METRIC_NAME,
        ).show()

    # 3) CROSS-VALIDATION
    _header("COMPOSITE FX ALPHA  ·  WALK-FORWARD CV")
    daily_index = data.close.vbt.resample_apply("1D", "last").dropna().index
    splitter = vbt.Splitter.from_purged_walkforward(
        daily_index,
        n_folds=N_FOLDS,
        n_test_folds=1,
        purge_td="1 day",
        min_train_folds=3,
    )
    if SHOW_CHARTS:
        plot_cv_splitter(splitter, title="Composite Alpha — CV Splits").show()
    cv_pipeline = create_cv_pipeline(splitter, metric_type=SHARPE_RATIO)
    grid_perf, best_perf = cv_pipeline(
        data,
        w_short=vbt.Param(CV_PARAMS["w_short"]),
        w_long=vbt.Param(CV_PARAMS["w_long"]),
        target_vol=vbt.Param(CV_PARAMS["target_vol"]),
    )
    print_cv_results(
        grid_perf,
        best_perf,
        splitter=splitter,
        title="Composite FX Alpha — Walk-Forward CV",
        metric_name=_METRIC_NAME,
        top_n=10,
    )
    if SHOW_CHARTS:
        plot_cv_heatmap(
            grid_perf,
            x_level="w_short",
            y_level="w_long",
            slider_level="split",
            splitter=splitter,
            title=f"Composite Alpha — CV {_METRIC_NAME} heatmap (per split)",
            metric_name=_METRIC_NAME,
        ).show()
        plot_cv_heatmap(
            grid_perf,
            x_level="w_short",
            y_level="w_long",
            slider_level=None,
            splitter=splitter,
            title=f"Composite Alpha — CV {_METRIC_NAME} heatmap (mean across splits)",
            metric_name=_METRIC_NAME,
        ).show()
        plot_cv_volume(
            grid_perf,
            x_level="w_short",
            y_level="w_long",
            z_level="target_vol",
            slider_level="split",
            splitter=splitter,
            title=f"Composite Alpha — CV {_METRIC_NAME} volume (per split)",
            metric_name=_METRIC_NAME,
        ).show()
        plot_cv_volume(
            grid_perf,
            x_level="w_short",
            y_level="w_long",
            z_level="target_vol",
            slider_level=None,
            splitter=splitter,
            title=f"Composite Alpha — CV {_METRIC_NAME} volume (mean across splits)",
            metric_name=_METRIC_NAME,
        ).show()
        plot_grid_surface(
            grid_perf,
            x_level="w_short",
            y_level="w_long",
            slider_level="split",
            splitter=splitter,
            title=f"Composite Alpha — CV {_METRIC_NAME} surface (per split)",
            metric_name=_METRIC_NAME,
        ).show()
        plot_grid_surface(
            grid_perf,
            x_level="w_short",
            y_level="w_long",
            slider_level=None,
            splitter=splitter,
            title=f"Composite Alpha — CV {_METRIC_NAME} surface (mean across splits)",
            metric_name=_METRIC_NAME,
        ).show()

    print("\nAll modes done.")
