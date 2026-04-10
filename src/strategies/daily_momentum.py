"""Daily FX Momentum Strategies — ims_pipeline format.

Two validated daily strategies, each with its own triplet of entry points:

1. **Cross-sectional (XS) momentum** across 4 pairs:
   - ``pipeline_xs(closes, **params) -> (pf, XSMomentumIndicator)``
   - ``pipeline_xs_nb(closes, **params)`` — ``@vbt.parameterized`` scalar
   - ``create_cv_pipeline_xs(splitter, ...)`` — ``@vbt.cv_split`` factory

2. **Time-series (TS) momentum** on a single pair with RSI confirmation:
   - ``pipeline_ts(close_daily, **params) -> (pf, TSMomentumIndicator)``
   - ``pipeline_ts_nb(close_daily, **params)`` — scalar grid search
   - ``create_cv_pipeline_ts(splitter, ...)`` — CV factory

Research findings (walk-forward 2019-2025):
  XS Momentum (21/63): Sharpe 0.72, 6/7 years positive
  TS Momentum GBP-USD EMA(20/50)+RSI7: Sharpe 0.70, 7/7 years positive
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
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
from utils import load_fx_data

warnings.filterwarnings(
    "ignore",
    message=r"Downcasting object dtype arrays on .fillna",
    category=FutureWarning,
)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
FX_PAIRS = ["EUR-USD", "GBP-USD", "USD-JPY", "USD-CAD"]

# Daily strategies → annualization factor = 252
DAILY_ANN_FACTOR: float = 252.0


# ═══════════════════════════════════════════════════════════════════════
# DATA
# ═══════════════════════════════════════════════════════════════════════


def load_daily_closes(pairs: list[str] | None = None) -> pd.DataFrame:
    """Load FX pairs as daily close prices (used by XS momentum)."""
    if pairs is None:
        pairs = FX_PAIRS
    closes = {}
    for pair in pairs:
        _, data = load_fx_data(f"data/{pair}_minute.parquet")
        closes[pair] = data.close.resample("1D").last().dropna()
    return pd.DataFrame(closes).dropna()


# ═══════════════════════════════════════════════════════════════════════
# HELPERS — returns-based proxies (used by combined_portfolio.py)
# ═══════════════════════════════════════════════════════════════════════


def compute_xs_momentum_weights(
    closes: pd.DataFrame,
    w_short: int = 21,
    w_long: int = 63,
) -> pd.DataFrame:
    """Dollar-neutral cross-sectional momentum weights (z-score normalized)."""
    ret_s = np.log(closes / closes.shift(w_short))
    ret_l = np.log(closes / closes.shift(w_long))
    momentum = 0.5 * ret_s + 0.5 * ret_l

    cs_mean = momentum.mean(axis=1)
    cs_std = momentum.std(axis=1).clip(lower=1e-10)
    z = momentum.sub(cs_mean, axis=0).div(cs_std, axis=0)

    weights = z.div(z.abs().sum(axis=1), axis=0).fillna(0)
    return weights.shift(1)  # no look-ahead


def backtest_xs_momentum(
    closes: pd.DataFrame,
    w_short: int = 21,
    w_long: int = 63,
    target_vol: float = 0.10,
) -> pd.Series:
    """XS momentum as raw daily returns Series (used by combined_portfolio)."""
    weights = compute_xs_momentum_weights(closes, w_short, w_long)
    daily_rets = closes.pct_change()
    port_ret = (weights * daily_rets).sum(axis=1).dropna()

    vol_21 = port_ret.rolling(21, min_periods=10).std() * np.sqrt(252)
    lev = (target_vol / vol_21.clip(lower=0.01)).clip(upper=5.0).shift(1)
    return port_ret * lev.fillna(1.0)


def backtest_ts_momentum_rsi(
    close_daily: pd.Series,
    fast_ema: int = 20,
    slow_ema: int = 50,
    rsi_period: int = 7,
    rsi_low: int = 40,
    rsi_high: int = 60,
    target_vol: float = 0.10,
) -> pd.Series:
    """TS momentum + RSI raw returns for a single pair (used by combined)."""
    ema_f = close_daily.ewm(span=fast_ema, min_periods=fast_ema).mean()
    ema_s = close_daily.ewm(span=slow_ema, min_periods=slow_ema).mean()
    rsi = vbt.RSI.run(close_daily, window=rsi_period).rsi

    signal = pd.Series(0.0, index=close_daily.index)
    trend_long = ema_f > ema_s
    trend_short = ema_f < ema_s
    rsi_ok_long = rsi < rsi_high
    rsi_ok_short = rsi > rsi_low

    signal[trend_long & rsi_ok_long] = 1.0
    signal[trend_short & rsi_ok_short] = -1.0
    signal = signal.shift(1)

    daily_ret = close_daily.pct_change()
    vol_21 = daily_ret.rolling(21, min_periods=10).std() * np.sqrt(252)
    lev = (target_vol / vol_21.clip(lower=0.01)).clip(upper=3.0).shift(1)
    return (signal * daily_ret * lev.fillna(1.0)).dropna()


def backtest_ts_momentum_portfolio(
    closes: pd.DataFrame,
    fast_ema: int = 20,
    slow_ema: int = 50,
    rsi_period: int = 7,
    rsi_low: int = 40,
    rsi_high: int = 60,
    target_vol: float = 0.10,
) -> pd.Series:
    """Equal-weight TS momentum portfolio across all pairs (used by combined)."""
    pair_rets = []
    for pair in closes.columns:
        rets = backtest_ts_momentum_rsi(
            closes[pair],
            fast_ema,
            slow_ema,
            rsi_period,
            rsi_low,
            rsi_high,
            target_vol,
        )
        pair_rets.append(rets)
    return pd.concat(pair_rets, axis=1).fillna(0).mean(axis=1)


def backtest_rsi_mr(
    close_daily: pd.Series,
    rsi_period: int = 14,
    oversold: int = 25,
    overbought: int = 75,
    target_vol: float = 0.10,
) -> pd.Series:
    """RSI mean reversion raw returns — kept for combined_portfolio imports."""
    rsi = vbt.RSI.run(close_daily, window=rsi_period).rsi

    signal = pd.Series(0.0, index=close_daily.index)
    signal[rsi < oversold] = 1.0
    signal[rsi > overbought] = -1.0
    signal = signal.shift(1)

    daily_ret = close_daily.pct_change()
    vol_21 = daily_ret.rolling(21, min_periods=10).std() * np.sqrt(252)
    lev = (target_vol / vol_21.clip(lower=0.01)).clip(upper=3.0).shift(1)
    return (signal * daily_ret * lev.fillna(1.0)).dropna()


# ═══════════════════════════════════════════════════════════════════════
# 1. XS MOMENTUM — pipeline_xs / pipeline_xs_nb / create_cv_pipeline_xs
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class XSMomentumIndicator:
    """XS momentum weights + closes for overlay plotting."""

    closes: pd.DataFrame
    weights: pd.DataFrame
    w_short: int
    w_long: int

    def plot(self, fig: go.Figure | None = None, **layout_kwargs) -> go.Figure:
        fig = fig or go.Figure()
        self.weights.vbt.plot(fig=fig)
        fig.update_layout(title="XS Momentum target weights", **layout_kwargs)
        return fig


def backtest_xs_momentum_pf(
    closes: pd.DataFrame,
    **kwargs: Any,
) -> vbt.Portfolio:
    """DEPRECATED shim — use ``pipeline_xs(closes, ...)`` instead."""
    pf, _ = pipeline_xs(closes, **kwargs)
    return pf


def pipeline_xs(
    closes: pd.DataFrame,
    w_short: int = 21,
    w_long: int = 63,
    target_vol: float = 0.10,
    leverage: float = 1.0,
    init_cash: float = 1_000_000.0,
    slippage: float = 0.0001,
    fees: float = 0.00005,
) -> tuple[vbt.Portfolio, XSMomentumIndicator]:
    """Investigation path — bit-equivalent to the legacy backtest_xs_momentum_pf.

    Returns a 4-column :class:`vbt.Portfolio` plus an ``XSMomentumIndicator``.
    """
    weights = compute_xs_momentum_weights(closes, w_short, w_long)

    daily_rets = closes.pct_change().fillna(0)
    proxy_port_ret = (weights * daily_rets).sum(axis=1)
    vol_21 = proxy_port_ret.rolling(21, min_periods=10).std() * np.sqrt(252)
    lev_mult = (
        (target_vol / vol_21.clip(lower=0.01))
        .clip(upper=5.0)
        .shift(1)
        .fillna(1.0)
    )

    scaled_weights = weights.mul(lev_mult, axis=0).fillna(0.0)

    pf = vbt.Portfolio.from_orders(
        close=closes,
        size=scaled_weights,
        size_type="targetpercent",
        init_cash=init_cash,
        leverage=leverage,
        slippage=slippage,
        fees=fees,
        freq="1D",
    )
    indicator = XSMomentumIndicator(
        closes=closes,
        weights=scaled_weights,
        w_short=w_short,
        w_long=w_long,
    )
    return pf, indicator


@vbt.parameterized(
    merge_func="concat",
    execute_kwargs=make_execute_kwargs("XS Momentum grid"),
)
def pipeline_xs_nb(
    closes: pd.DataFrame,
    w_short: int,
    w_long: int,
    target_vol: float = 0.10,
    leverage: float = 1.0,
    init_cash: float = 1_000_000.0,
    slippage: float = 0.0001,
    fees: float = 0.00005,
    ann_factor: float = DAILY_ANN_FACTOR,
    cutoff: float = 0.05,
    metric_type: int = SHARPE_RATIO,
) -> float:
    """XS momentum grid-search path — scalar metric per param combo."""
    pf, _ = pipeline_xs(
        closes,
        w_short=w_short,
        w_long=w_long,
        target_vol=target_vol,
        leverage=leverage,
        init_cash=init_cash,
        slippage=slippage,
        fees=fees,
    )
    returns = pf.returns.values
    if returns.ndim > 1:
        # Aggregate per-asset returns to a single equally-weighted series.
        returns = returns.mean(axis=1)
    return float(compute_metric_nb(returns, metric_type, ann_factor, cutoff))


def run_grid_xs(
    closes: pd.DataFrame,
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

    return pipeline_xs_nb(
        closes,
        w_short=_param(w_short),
        w_long=_param(w_long),
        target_vol=_param(target_vol),
        metric_type=metric_type,
        **kwargs,
    )


def create_cv_pipeline_xs(
    splitter: Any,
    metric_type: int = SHARPE_RATIO,
    **pipeline_defaults: Any,
):
    """Build ``@vbt.cv_split`` pipeline for XS momentum CV."""
    splitter_kwargs = pipeline_defaults.pop("splitter_kwargs", {})
    defaults = dict(
        target_vol=0.10,
        leverage=1.0,
        init_cash=1_000_000.0,
        slippage=0.0001,
        fees=0.00005,
        ann_factor=DAILY_ANN_FACTOR,
        cutoff=0.05,
        metric_type=metric_type,
    )
    defaults.update(pipeline_defaults)

    @vbt.cv_split(
        splitter=splitter,
        splitter_kwargs=splitter_kwargs,
        takeable_args=["closes"],
        parameterized_kwargs=dict(
            execute_kwargs=make_execute_kwargs(
                "XS Momentum combos", pbar_kwargs=dict(leave=False)
            ),
            merge_func="concat",
        ),
        execute_kwargs=make_execute_kwargs("XS Momentum CV splits"),
        merge_func="concat",
        return_grid="all",
        attach_bounds="index",
    )
    def cv_pipeline(
        closes: pd.DataFrame,
        w_short: int,
        w_long: int,
        target_vol: float = defaults["target_vol"],
        leverage: float = defaults["leverage"],
        init_cash: float = defaults["init_cash"],
        slippage: float = defaults["slippage"],
        fees: float = defaults["fees"],
        ann_factor: float = defaults["ann_factor"],
        cutoff: float = defaults["cutoff"],
        metric_type: int = defaults["metric_type"],
    ) -> float:
        pf, _ = pipeline_xs(
            closes,
            w_short=w_short,
            w_long=w_long,
            target_vol=target_vol,
            leverage=leverage,
            init_cash=init_cash,
            slippage=slippage,
            fees=fees,
        )
        returns = pf.returns.values
        if returns.ndim > 1:
            returns = returns.mean(axis=1)
        return float(compute_metric_nb(returns, metric_type, ann_factor, cutoff))

    return cv_pipeline


# ═══════════════════════════════════════════════════════════════════════
# 2. TS MOMENTUM — pipeline_ts / pipeline_ts_nb / create_cv_pipeline_ts
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class TSMomentumIndicator:
    """TS momentum EMA + RSI overlays."""

    close: pd.Series
    ema_fast: pd.Series
    ema_slow: pd.Series
    rsi: pd.Series
    rsi_low: float
    rsi_high: float

    def plot(self, fig: go.Figure | None = None, **layout_kwargs) -> go.Figure:
        fig = fig or go.Figure()
        self.close.vbt.plot(
            fig=fig,
            trace_kwargs=dict(name="Close", line=dict(width=2, color="blue")),
        )
        self.ema_fast.vbt.plot(
            fig=fig,
            trace_kwargs=dict(name="EMA Fast", line=dict(color="orange")),
        )
        self.ema_slow.vbt.plot(
            fig=fig,
            trace_kwargs=dict(name="EMA Slow", line=dict(color="red")),
        )
        fig.update_layout(**layout_kwargs)
        return fig


def backtest_ts_momentum_pf(
    close_daily: pd.Series,
    **kwargs: Any,
) -> vbt.Portfolio:
    """DEPRECATED shim — use ``pipeline_ts(close_daily, ...)`` instead."""
    pf, _ = pipeline_ts(close_daily, **kwargs)
    return pf


def pipeline_ts(
    close_daily: pd.Series,
    fast_ema: int = 20,
    slow_ema: int = 50,
    rsi_period: int = 7,
    rsi_low: int = 40,
    rsi_high: int = 60,
    target_vol: float = 0.10,
    leverage: float = 1.0,
    init_cash: float = 1_000_000.0,
    slippage: float = 0.0001,
) -> tuple[vbt.Portfolio, TSMomentumIndicator]:
    """Investigation path — bit-equivalent to legacy backtest_ts_momentum_pf."""
    ema_f = close_daily.ewm(span=fast_ema, min_periods=fast_ema).mean()
    ema_s = close_daily.ewm(span=slow_ema, min_periods=slow_ema).mean()
    rsi = vbt.RSI.run(close_daily, window=rsi_period).rsi

    trend_long = ema_f > ema_s
    trend_short = ema_f < ema_s

    long_ok = trend_long & (rsi < rsi_high)
    short_ok = trend_short & (rsi > rsi_low)

    entries = long_ok & ~long_ok.shift(1).fillna(False)
    exits = ~long_ok & long_ok.shift(1).fillna(False)
    short_entries = short_ok & ~short_ok.shift(1).fillna(False)
    short_exits = ~short_ok & short_ok.shift(1).fillna(False)

    daily_ret = close_daily.pct_change()
    vol_21 = daily_ret.rolling(21, min_periods=10).std() * np.sqrt(252)
    dyn_lev = (
        (target_vol / vol_21.clip(lower=0.01)).clip(upper=3.0).shift(1).fillna(1.0)
    )
    dyn_lev_arr = (dyn_lev * leverage).values

    pf = vbt.Portfolio.from_signals(
        close=close_daily,
        entries=entries,
        exits=exits,
        short_entries=short_entries,
        short_exits=short_exits,
        leverage=dyn_lev_arr,
        init_cash=init_cash,
        slippage=slippage,
        freq="1D",
    )
    indicator = TSMomentumIndicator(
        close=close_daily,
        ema_fast=ema_f.rename("EMA Fast"),
        ema_slow=ema_s.rename("EMA Slow"),
        rsi=rsi.rename("RSI"),
        rsi_low=rsi_low,
        rsi_high=rsi_high,
    )
    return pf, indicator


@vbt.parameterized(
    merge_func="concat",
    execute_kwargs=make_execute_kwargs("TS Momentum grid"),
)
def pipeline_ts_nb(
    close_daily: pd.Series,
    fast_ema: int,
    slow_ema: int,
    rsi_period: int,
    rsi_low: int = 40,
    rsi_high: int = 60,
    target_vol: float = 0.10,
    leverage: float = 1.0,
    init_cash: float = 1_000_000.0,
    slippage: float = 0.0001,
    ann_factor: float = DAILY_ANN_FACTOR,
    cutoff: float = 0.05,
    metric_type: int = SHARPE_RATIO,
) -> float:
    """TS momentum grid-search path — scalar metric per param combo."""
    pf, _ = pipeline_ts(
        close_daily,
        fast_ema=fast_ema,
        slow_ema=slow_ema,
        rsi_period=rsi_period,
        rsi_low=rsi_low,
        rsi_high=rsi_high,
        target_vol=target_vol,
        leverage=leverage,
        init_cash=init_cash,
        slippage=slippage,
    )
    returns = pf.returns.values
    if returns.ndim > 1:
        returns = returns[:, 0]
    return float(compute_metric_nb(returns, metric_type, ann_factor, cutoff))


def run_grid_ts(
    close_daily: pd.Series,
    *,
    fast_ema: list[int] | int,
    slow_ema: list[int] | int,
    rsi_period: list[int] | int,
    metric_type: int = SHARPE_RATIO,
    **kwargs: Any,
) -> pd.Series:
    def _param(v):
        if isinstance(v, (list, tuple, np.ndarray)):
            return vbt.Param(list(v), condition="fast_ema < slow_ema" if False else None)
        return v

    # Note: condition support requires all params to share the same level.
    # Kept simple for Phase 4 — run all combos then filter post-hoc if needed.
    def _plain(v):
        if isinstance(v, (list, tuple, np.ndarray)):
            return vbt.Param(list(v))
        return v

    return pipeline_ts_nb(
        close_daily,
        fast_ema=_plain(fast_ema),
        slow_ema=_plain(slow_ema),
        rsi_period=_plain(rsi_period),
        metric_type=metric_type,
        **kwargs,
    )


def create_cv_pipeline_ts(
    splitter: Any,
    metric_type: int = SHARPE_RATIO,
    **pipeline_defaults: Any,
):
    """Build ``@vbt.cv_split`` pipeline for TS momentum CV."""
    splitter_kwargs = pipeline_defaults.pop("splitter_kwargs", {})
    defaults = dict(
        rsi_low=40,
        rsi_high=60,
        target_vol=0.10,
        leverage=1.0,
        init_cash=1_000_000.0,
        slippage=0.0001,
        ann_factor=DAILY_ANN_FACTOR,
        cutoff=0.05,
        metric_type=metric_type,
    )
    defaults.update(pipeline_defaults)

    @vbt.cv_split(
        splitter=splitter,
        splitter_kwargs=splitter_kwargs,
        takeable_args=["close_daily"],
        parameterized_kwargs=dict(
            execute_kwargs=make_execute_kwargs(
                "TS Momentum combos", pbar_kwargs=dict(leave=False)
            ),
            merge_func="concat",
        ),
        execute_kwargs=make_execute_kwargs("TS Momentum CV splits"),
        merge_func="concat",
        return_grid="all",
        attach_bounds="index",
    )
    def cv_pipeline(
        close_daily: pd.Series,
        fast_ema: int,
        slow_ema: int,
        rsi_period: int,
        rsi_low: int = defaults["rsi_low"],
        rsi_high: int = defaults["rsi_high"],
        target_vol: float = defaults["target_vol"],
        leverage: float = defaults["leverage"],
        init_cash: float = defaults["init_cash"],
        slippage: float = defaults["slippage"],
        ann_factor: float = defaults["ann_factor"],
        cutoff: float = defaults["cutoff"],
        metric_type: int = defaults["metric_type"],
    ) -> float:
        pf, _ = pipeline_ts(
            close_daily,
            fast_ema=fast_ema,
            slow_ema=slow_ema,
            rsi_period=rsi_period,
            rsi_low=rsi_low,
            rsi_high=rsi_high,
            target_vol=target_vol,
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
# 3. CLI — run either strategy in single / grid / cv mode
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

    # ─────────────────────────────────────────────────────────────────
    # CONFIGURATION
    # ─────────────────────────────────────────────────────────────────
    TS_PAIR = "GBP-USD"
    XS_OUTPUT_DIR = "results/daily_xs"
    TS_OUTPUT_DIR = f"results/daily_ts_{TS_PAIR.lower()}"
    SHOW_CHARTS = True
    N_FOLDS = 8
    LEVERAGE = 1.0

    XS_GRID_PARAMS: dict[str, list] = dict(
        w_short=[10, 21, 42],
        w_long=[42, 63, 126],
        target_vol=[0.08, 0.10, 0.12],
    )
    XS_CV_PARAMS: dict[str, list] = dict(
        w_short=[10, 21, 42],
        w_long=[42, 63, 126],
    )
    TS_GRID_PARAMS: dict[str, list] = dict(
        fast_ema=[10, 20, 30],
        slow_ema=[40, 50, 100],
        rsi_period=[7, 14],
    )
    _METRIC_NAME = METRIC_LABELS[SHARPE_RATIO]

    def _header(label: str) -> None:
        bar = "█" * 78
        print(f"\n{bar}")
        print(f"██  {label.ljust(72)}  ██")
        print(f"{bar}\n")

    apply_vbt_plot_defaults()
    print("Loading daily closes for 4 pairs ...")
    closes = load_daily_closes()
    print(f"  {len(closes)} days, {list(closes.columns)}")

    # ═════════════════════════════════════════════════════════════════
    # STRATEGY A: CROSS-SECTIONAL MOMENTUM (XS)
    # ═════════════════════════════════════════════════════════════════

    # 1) XS SINGLE
    _header("XS MOMENTUM (4-pair)  ·  SINGLE RUN")
    pf_xs, ind_xs = pipeline_xs(closes, leverage=LEVERAGE)
    analyze_portfolio(
        pf_xs,
        name="XS Momentum (4-pair)",
        output_dir=XS_OUTPUT_DIR,
        show_charts=SHOW_CHARTS,
        indicator=ind_xs,
    )

    # 2) XS GRID
    _header("XS MOMENTUM  ·  GRID SEARCH")
    grid_xs = run_grid_xs(closes, metric_type=SHARPE_RATIO, **XS_GRID_PARAMS)
    print_grid_results(
        grid_xs,
        title="XS Momentum — Grid Search",
        metric_name=_METRIC_NAME,
        top_n=20,
    )
    if SHOW_CHARTS:
        plot_grid_heatmap(
            grid_xs,
            x_level="w_short",
            y_level="w_long",
            slider_level="target_vol",
            title=f"XS Momentum — {_METRIC_NAME} heatmap (slider: target_vol)",
            metric_name=_METRIC_NAME,
        ).show()
        plot_grid_heatmap(
            grid_xs,
            x_level="w_short",
            y_level="w_long",
            slider_level=None,
            title=f"XS Momentum — {_METRIC_NAME} heatmap (aggregated)",
            metric_name=_METRIC_NAME,
        ).show()
        plot_grid_volume(
            grid_xs,
            x_level="w_short",
            y_level="w_long",
            z_level="target_vol",
            title=f"XS Momentum — {_METRIC_NAME} volume",
            metric_name=_METRIC_NAME,
        ).show()
        plot_grid_surface(
            grid_xs,
            x_level="w_short",
            y_level="w_long",
            slider_level="target_vol",
            title=f"XS Momentum — {_METRIC_NAME} surface (slider: target_vol)",
            metric_name=_METRIC_NAME,
        ).show()
        plot_grid_surface(
            grid_xs,
            x_level="w_short",
            y_level="w_long",
            slider_level=None,
            title=f"XS Momentum — {_METRIC_NAME} surface (aggregated)",
            metric_name=_METRIC_NAME,
        ).show()

    # 3) XS CV
    _header("XS MOMENTUM  ·  WALK-FORWARD CV")
    splitter_xs = vbt.Splitter.from_purged_walkforward(
        closes.index,
        n_folds=N_FOLDS,
        n_test_folds=1,
        purge_td="1 day",
        min_train_folds=3,
    )
    if SHOW_CHARTS:
        plot_cv_splitter(splitter_xs, title="XS Momentum — CV Splits").show()
    cv_pipeline_xs = create_cv_pipeline_xs(splitter_xs, metric_type=SHARPE_RATIO)
    grid_perf_xs, best_perf_xs = cv_pipeline_xs(
        closes,
        w_short=vbt.Param(XS_CV_PARAMS["w_short"]),
        w_long=vbt.Param(XS_CV_PARAMS["w_long"]),
    )
    print_cv_results(
        grid_perf_xs,
        best_perf_xs,
        splitter=splitter_xs,
        title="XS Momentum — Walk-Forward CV",
        metric_name=_METRIC_NAME,
        top_n=10,
    )
    if SHOW_CHARTS:
        plot_cv_heatmap(
            grid_perf_xs,
            x_level="w_short",
            y_level="w_long",
            slider_level="split",
            splitter=splitter_xs,
            title=f"XS Momentum — CV {_METRIC_NAME} heatmap (per split)",
            metric_name=_METRIC_NAME,
        ).show()
        plot_cv_heatmap(
            grid_perf_xs,
            x_level="w_short",
            y_level="w_long",
            slider_level=None,
            splitter=splitter_xs,
            title=f"XS Momentum — CV {_METRIC_NAME} heatmap (mean across splits)",
            metric_name=_METRIC_NAME,
        ).show()

    # ═════════════════════════════════════════════════════════════════
    # STRATEGY B: TIME-SERIES MOMENTUM + RSI (TS)
    # ═════════════════════════════════════════════════════════════════

    close_daily = closes[TS_PAIR]

    # 1) TS SINGLE
    _header(f"TS MOMENTUM+RSI ({TS_PAIR})  ·  SINGLE RUN")
    pf_ts, ind_ts = pipeline_ts(close_daily, leverage=LEVERAGE)
    analyze_portfolio(
        pf_ts,
        name=f"TS Momentum+RSI ({TS_PAIR})",
        output_dir=TS_OUTPUT_DIR,
        show_charts=SHOW_CHARTS,
        indicator=ind_ts,
    )

    # 2) TS GRID
    _header(f"TS MOMENTUM+RSI ({TS_PAIR})  ·  GRID SEARCH")
    grid_ts = run_grid_ts(close_daily, metric_type=SHARPE_RATIO, **TS_GRID_PARAMS)
    print_grid_results(
        grid_ts,
        title=f"TS Momentum ({TS_PAIR}) — Grid Search",
        metric_name=_METRIC_NAME,
        top_n=20,
    )
    if SHOW_CHARTS:
        plot_grid_heatmap(
            grid_ts,
            x_level="fast_ema",
            y_level="slow_ema",
            slider_level="rsi_period",
            title=f"TS Momentum — {_METRIC_NAME} heatmap (slider: rsi_period)",
            metric_name=_METRIC_NAME,
        ).show()
        plot_grid_heatmap(
            grid_ts,
            x_level="fast_ema",
            y_level="slow_ema",
            slider_level=None,
            title=f"TS Momentum — {_METRIC_NAME} heatmap (aggregated)",
            metric_name=_METRIC_NAME,
        ).show()
        plot_grid_volume(
            grid_ts,
            x_level="fast_ema",
            y_level="slow_ema",
            z_level="rsi_period",
            title=f"TS Momentum — {_METRIC_NAME} volume",
            metric_name=_METRIC_NAME,
        ).show()
        plot_grid_surface(
            grid_ts,
            x_level="fast_ema",
            y_level="slow_ema",
            slider_level="rsi_period",
            title=f"TS Momentum — {_METRIC_NAME} surface (slider: rsi_period)",
            metric_name=_METRIC_NAME,
        ).show()
        plot_grid_surface(
            grid_ts,
            x_level="fast_ema",
            y_level="slow_ema",
            slider_level=None,
            title=f"TS Momentum — {_METRIC_NAME} surface (aggregated)",
            metric_name=_METRIC_NAME,
        ).show()

    # 3) TS CV
    _header(f"TS MOMENTUM+RSI ({TS_PAIR})  ·  WALK-FORWARD CV")
    splitter_ts = vbt.Splitter.from_purged_walkforward(
        close_daily.index,
        n_folds=N_FOLDS,
        n_test_folds=1,
        purge_td="1 day",
        min_train_folds=3,
    )
    if SHOW_CHARTS:
        plot_cv_splitter(splitter_ts, title="TS Momentum — CV Splits").show()
    cv_pipeline_ts = create_cv_pipeline_ts(splitter_ts, metric_type=SHARPE_RATIO)
    grid_perf_ts, best_perf_ts = cv_pipeline_ts(
        close_daily,
        fast_ema=vbt.Param(TS_GRID_PARAMS["fast_ema"]),
        slow_ema=vbt.Param(TS_GRID_PARAMS["slow_ema"]),
        rsi_period=vbt.Param(TS_GRID_PARAMS["rsi_period"]),
    )
    print_cv_results(
        grid_perf_ts,
        best_perf_ts,
        splitter=splitter_ts,
        title=f"TS Momentum ({TS_PAIR}) — Walk-Forward CV",
        metric_name=_METRIC_NAME,
        top_n=10,
    )
    if SHOW_CHARTS:
        plot_cv_heatmap(
            grid_perf_ts,
            x_level="fast_ema",
            y_level="slow_ema",
            slider_level="split",
            splitter=splitter_ts,
            title=f"TS Momentum — CV {_METRIC_NAME} heatmap (per split)",
            metric_name=_METRIC_NAME,
        ).show()
        plot_cv_heatmap(
            grid_perf_ts,
            x_level="fast_ema",
            y_level="slow_ema",
            slider_level=None,
            splitter=splitter_ts,
            title=f"TS Momentum — CV {_METRIC_NAME} heatmap (mean across splits)",
            metric_name=_METRIC_NAME,
        ).show()
        plot_cv_volume(
            grid_perf_ts,
            x_level="fast_ema",
            y_level="slow_ema",
            z_level="rsi_period",
            slider_level="split",
            splitter=splitter_ts,
            title=f"TS Momentum — CV {_METRIC_NAME} volume (per split)",
            metric_name=_METRIC_NAME,
        ).show()

    print("\nAll modes done.")
