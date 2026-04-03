#!/usr/bin/env python
"""
Intraday TWAP Mean Reversion — EUR/USD Minute Data

Robust backtesting pipeline:
- TWAP anchor (no volume data available) with rolling z-score
- Native vbt.generic.nb.rolling_zscore_1d_nb for signal computation
- ADX regime filter to disable MR in trending markets
- Volatility-targeted position sizing
- Purged k-fold + walk-forward cross-validation
- 20% hold-out set for final validation
- Realistic transaction costs (1.5 pip slippage)
"""

import os
from typing import Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import vectorbtpro as vbt
from numba import njit
from plotly.subplots import make_subplots

from utils import (
    apply_vbt_settings,
    compute_adx_nb,
    compute_ann_factor,
    compute_daily_rolling_volatility_nb,
    compute_leverage_nb,
    find_day_boundaries_nb,
)


# ═══════════════════════════════════════════════════════════════════════
# 0. SETTINGS
# ═══════════════════════════════════════════════════════════════════════

apply_vbt_settings()

SLIPPAGE = 0.00015       # 1.5 pips — realistic for EUR/USD minute bars
FIXED_FEES = 0.0         # No fixed commission for spot FX
INIT_CASH = 1_000_000.0
VOL_WINDOW = 20          # Days for rolling volatility
SIGMA_TARGET = 0.01      # Daily vol target for position sizing
MAX_LEVERAGE = 3.0       # Cap on vol-targeted leverage
ADX_PERIOD = 14          # ADX lookback (daily-equivalent, applied to minute)
ADX_THRESHOLD = 30.0     # ADX above this = trending, disable MR


# ═══════════════════════════════════════════════════════════════════════
# 1. NUMBA KERNELS
# ═══════════════════════════════════════════════════════════════════════

@njit
def compute_intraday_twap_nb(
    index_ns: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
) -> np.ndarray:
    """Session-resetting TWAP: cumulative mean of typical price per day."""
    n = len(close)
    twap = np.full(n, np.nan)
    if n == 0:
        return twap

    start_arr, end_arr, n_days = find_day_boundaries_nb(index_ns)

    for d in range(n_days):
        s = start_arr[d]
        e = end_arr[d]
        cum_tp = 0.0
        count = 0
        for i in range(s, e):
            tp = (high[i] + low[i] + close[i]) / 3.0
            if not np.isnan(tp):
                cum_tp += tp
                count += 1
                twap[i] = cum_tp / count

    return twap


@njit
def compute_intraday_mr_indicators_nb(
    index_ns: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    lookback: int,
    band_width: float,
    adx_period: int,
    adx_threshold: float,
    vol_window: int,
    sigma_target: float,
    max_leverage: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute TWAP, z-score, bands, ADX filter, and leverage."""
    n = len(close)

    # TWAP per session
    twap = compute_intraday_twap_nb(index_ns, high, low, close)

    # Deviation from TWAP
    deviation = np.empty(n)
    for i in range(n):
        if np.isnan(twap[i]) or np.isnan(close[i]):
            deviation[i] = np.nan
        else:
            deviation[i] = close[i] - twap[i]

    # Native rolling z-score (one-pass accumulator)
    zscore = vbt.generic.nb.rolling_zscore_1d_nb(deviation, lookback, minp=lookback, ddof=1)

    # Bands from rolling std
    rolling_std = vbt.generic.nb.rolling_std_1d_nb(deviation, lookback, minp=lookback, ddof=1)
    upper_band = np.full(n, np.nan)
    lower_band = np.full(n, np.nan)
    for i in range(n):
        s = rolling_std[i]
        if not np.isnan(s) and s > 1e-10 and not np.isnan(twap[i]):
            upper_band[i] = twap[i] + band_width * s
            lower_band[i] = twap[i] - band_width * s

    # ADX regime filter (1 = MR allowed, 0 = trending)
    adx = compute_adx_nb(high, low, close, adx_period)
    regime_ok = np.ones(n, dtype=np.float64)
    for i in range(n):
        if not np.isnan(adx[i]) and adx[i] > adx_threshold:
            regime_ok[i] = 0.0

    # Volatility-targeted leverage
    rolling_vol = compute_daily_rolling_volatility_nb(index_ns, close, vol_window)
    leverage = compute_leverage_nb(rolling_vol, sigma_target, max_leverage)

    return twap, zscore, upper_band, lower_band, regime_ok, leverage


# ═══════════════════════════════════════════════════════════════════════
# 2. METRIC CONSTANTS + DISPATCH
# ═══════════════════════════════════════════════════════════════════════

TOTAL_RETURN = 0
SHARPE_RATIO = 1
CALMAR_RATIO = 2
SORTINO_RATIO = 3
OMEGA_RATIO = 4
ANNUALIZED_RETURN = 5
MAX_DRAWDOWN = 6
PROFIT_FACTOR = 7


@njit(nogil=True)
def compute_metric_nb(returns, metric_type, ann_factor, cutoff=0.05):
    if metric_type == TOTAL_RETURN:
        return vbt.ret_nb.total_return_nb(returns=returns)
    elif metric_type == SHARPE_RATIO:
        return vbt.ret_nb.sharpe_ratio_nb(returns=returns, ann_factor=ann_factor)
    elif metric_type == CALMAR_RATIO:
        return vbt.ret_nb.calmar_ratio_nb(returns=returns, ann_factor=ann_factor)
    elif metric_type == SORTINO_RATIO:
        return vbt.ret_nb.sortino_ratio_nb(returns=returns, ann_factor=ann_factor)
    elif metric_type == OMEGA_RATIO:
        return vbt.ret_nb.omega_ratio_nb(returns=returns)
    elif metric_type == ANNUALIZED_RETURN:
        return vbt.ret_nb.annualized_return_nb(returns=returns, ann_factor=ann_factor)
    elif metric_type == MAX_DRAWDOWN:
        return -vbt.ret_nb.max_drawdown_nb(returns=returns)
    elif metric_type == PROFIT_FACTOR:
        return vbt.ret_nb.profit_factor_nb(returns=returns)
    else:
        return vbt.ret_nb.total_return_nb(returns=returns)


# ═══════════════════════════════════════════════════════════════════════
# 3. SIGNAL FUNCTION
# ═══════════════════════════════════════════════════════════════════════

@njit
def intraday_mr_signal_nb(
    c,
    close_arr: np.ndarray,
    upper_arr: np.ndarray,
    lower_arr: np.ndarray,
    twap_arr: np.ndarray,
    regime_ok_arr: np.ndarray,
    index_ns_arr: np.ndarray,
    eod_hour_arr: np.ndarray,
    eod_minute_arr: np.ndarray,
    eval_freq_arr: np.ndarray,
):
    ts_ns = index_ns_arr[c.i]
    cur_hour = vbt.dt_nb.hour_nb(ts_ns)
    cur_minute = vbt.dt_nb.minute_nb(ts_ns)

    eod_hour = vbt.pf_nb.select_nb(c, eod_hour_arr)
    eod_minute = vbt.pf_nb.select_nb(c, eod_minute_arr)

    # EOD forced exit
    is_eod = (cur_hour > eod_hour) or (cur_hour == eod_hour and cur_minute >= eod_minute)
    if is_eod:
        el = vbt.pf_nb.ctx_helpers.in_long_position_nb(c)
        es = vbt.pf_nb.ctx_helpers.in_short_position_nb(c)
        return False, el, False, es

    # Evaluate at parametric frequency
    eval_freq = vbt.pf_nb.select_nb(c, eval_freq_arr)
    if eval_freq > 0 and cur_minute % eval_freq != 0:
        return False, False, False, False

    px = vbt.pf_nb.select_nb(c, close_arr)
    ub = vbt.pf_nb.select_nb(c, upper_arr)
    lb = vbt.pf_nb.select_nb(c, lower_arr)
    tw = vbt.pf_nb.select_nb(c, twap_arr)
    regime = vbt.pf_nb.select_nb(c, regime_ok_arr)

    if np.isnan(px) or np.isnan(ub) or np.isnan(lb) or np.isnan(tw):
        return False, False, False, False

    in_long = vbt.pf_nb.ctx_helpers.in_long_position_nb(c)
    in_short = vbt.pf_nb.ctx_helpers.in_short_position_nb(c)

    # Regime filter: no new entries in trending market, but allow exits
    if not in_long and not in_short:
        if regime < 0.5:
            return False, False, False, False
        if px < lb:
            return True, False, False, False
        elif px > ub:
            return False, False, True, False
    elif in_long:
        if px >= tw:
            return False, True, False, False
    elif in_short:
        if px <= tw:
            return False, False, False, True

    return False, False, False, False


# ═══════════════════════════════════════════════════════════════════════
# 4. STANDARD BACKTEST
# ═══════════════════════════════════════════════════════════════════════

def run_standard_backtest(
    raw: pd.DataFrame,
    index_ns: np.ndarray,
    ann_factor: float,
    lookback: int = 60,
    band_width: float = 2.0,
    eod_hour: int = 21,
    eod_minute: int = 0,
    eval_freq: int = 5,
    sl_stop: float = 0.003,
    tp_stop: float = 0.0045,
    adx_period: int = ADX_PERIOD,
    adx_threshold: float = ADX_THRESHOLD,
    vol_window: int = VOL_WINDOW,
    sigma_target: float = SIGMA_TARGET,
    max_leverage: float = MAX_LEVERAGE,
):
    IMR = vbt.IF(
        class_name="IntradayMR",
        short_name="imr",
        input_names=["index_ns", "high_minute", "low_minute", "close_minute"],
        param_names=["lookback", "band_width", "adx_period", "adx_threshold",
                      "vol_window", "sigma_target", "max_leverage"],
        output_names=["twap", "zscore", "upper_band", "lower_band", "regime_ok", "leverage"],
    ).with_apply_func(
        compute_intraday_mr_indicators_nb,
        takes_1d=True,
        lookback=lookback,
        band_width=band_width,
        adx_period=adx_period,
        adx_threshold=adx_threshold,
        vol_window=vol_window,
        sigma_target=sigma_target,
        max_leverage=max_leverage,
    )

    imr = IMR.run(
        index_ns=index_ns,
        high_minute=raw["high"],
        low_minute=raw["low"],
        close_minute=raw["close"],
        lookback=lookback,
        band_width=band_width,
        adx_period=adx_period,
        adx_threshold=adx_threshold,
        vol_window=vol_window,
        sigma_target=sigma_target,
        max_leverage=max_leverage,
        jitted_loop=True,
        jitted_warmup=True,
        execute_kwargs=dict(engine="threadpool", n_chunks="auto"),
    )

    pf = vbt.Portfolio.from_signals(
        raw["close"],
        open=raw["open"],
        high=raw["high"],
        low=raw["low"],
        signal_func_nb=intraday_mr_signal_nb,
        signal_args=(
            vbt.Rep("close_arr"),
            vbt.Rep("upper_arr"),
            vbt.Rep("lower_arr"),
            vbt.Rep("twap_arr"),
            vbt.Rep("regime_ok_arr"),
            vbt.Rep("index_arr"),
            vbt.Rep("eod_hour"),
            vbt.Rep("eod_minute"),
            vbt.Rep("eval_freq"),
        ),
        broadcast_named_args=dict(
            close_arr=raw["close"],
            upper_arr=imr.upper_band.values,
            lower_arr=imr.lower_band.values,
            twap_arr=imr.twap.values,
            regime_ok_arr=imr.regime_ok.values,
            index_arr=index_ns,
            eod_hour=eod_hour,
            eod_minute=eod_minute,
            eval_freq=eval_freq,
        ),
        leverage=imr.leverage.values,
        slippage=SLIPPAGE,
        fixed_fees=FIXED_FEES,
        sl_stop=sl_stop,
        tp_stop=tp_stop,
        init_cash=INIT_CASH,
        freq="1min",
    )

    return pf, imr


# ═══════════════════════════════════════════════════════════════════════
# 5. FULL NUMBA PIPELINE (for parameter sweeps)
# ═══════════════════════════════════════════════════════════════════════

@vbt.parameterized(
    execute_kwargs=dict(chunk_len="auto", engine="threadpool"),
    merge_func="concat",
)
@njit(nogil=True)
def pipeline_nb(
    high_arr,
    low_arr,
    close_arr,
    open_arr,
    idx_ns,
    lookback,
    band_width,
    eod_hour,
    eod_minute,
    eval_freq,
    adx_period: int = ADX_PERIOD,
    adx_threshold: float = ADX_THRESHOLD,
    vol_window: int = VOL_WINDOW,
    sigma_target: float = SIGMA_TARGET,
    max_leverage: float = MAX_LEVERAGE,
    init_cash: float = INIT_CASH,
    fixed_fees: float = FIXED_FEES,
    sl_stop: float = 0.003,
    ann_factor: float = 299_124.0,
    metric_type: int = SHARPE_RATIO,
):
    target_shape = close_arr.shape

    twap, zscore, upper, lower, regime_ok, leverage = compute_intraday_mr_indicators_nb(
        idx_ns,
        high_arr[:, 0],
        low_arr[:, 0],
        close_arr[:, 0],
        lookback,
        band_width,
        adx_period,
        adx_threshold,
        vol_window,
        sigma_target,
        max_leverage,
    )

    eod_hour_a = np.full(target_shape, eod_hour, dtype=np.int32)
    eod_minute_a = np.full(target_shape, eod_minute, dtype=np.int32)
    eval_freq_a = np.full(target_shape, eval_freq, dtype=np.int32)
    group_lens = np.full(close_arr.shape[1], 1)
    leverage_arr = leverage.reshape(-1, 1)
    fixed_fees_arr = np.full(1, fixed_fees)
    fees_arr = np.full(1, 0.0)

    sim_out = vbt.pf_nb.from_signal_func_nb(
        target_shape=target_shape,
        group_lens=group_lens,
        init_cash=init_cash,
        cash_sharing=False,
        open=open_arr,
        high=high_arr,
        low=low_arr,
        close=close_arr,
        signal_func_nb=intraday_mr_signal_nb,
        signal_args=(
            close_arr,
            upper.reshape(-1, 1),
            lower.reshape(-1, 1),
            twap.reshape(-1, 1),
            regime_ok.reshape(-1, 1),
            idx_ns,
            eod_hour_a,
            eod_minute_a,
            eval_freq_a,
        ),
        fees=fees_arr,
        fixed_fees=fixed_fees_arr,
        leverage=leverage_arr,
        sl_stop=sl_stop,
        post_segment_func_nb=vbt.pf_nb.save_post_segment_func_nb,
        in_outputs=vbt.pf_nb.init_FSInOutputs_nb(target_shape, group_lens, cash_sharing=False),
    )

    returns = sim_out.in_outputs.returns
    return compute_metric_nb(returns, metric_type, ann_factor)


# ═══════════════════════════════════════════════════════════════════════
# 6. CV PIPELINES
# ═══════════════════════════════════════════════════════════════════════

def _build_cv_runner(splitter):
    """Build a CV-wrapped backtest function for a given splitter."""

    def _run_pipeline(
        high_arr, low_arr, close_arr, open_arr, idx_ns,
        lookback, band_width, sl_stop=0.003, tp_stop=0.0045,
        eod_hour=21, eod_minute=0, eval_freq=5,
        adx_period=ADX_PERIOD, adx_threshold=ADX_THRESHOLD,
        vol_window=VOL_WINDOW, sigma_target=SIGMA_TARGET,
        max_leverage=MAX_LEVERAGE,
        metric="sharpe_ratio",
    ):
        IMR = vbt.IF(
            class_name="IntradayMR",
            short_name="imr",
            input_names=["index_ns", "high_minute", "low_minute", "close_minute"],
            param_names=["lookback", "band_width", "adx_period", "adx_threshold",
                          "vol_window", "sigma_target", "max_leverage"],
            output_names=["twap", "zscore", "upper_band", "lower_band", "regime_ok", "leverage"],
        ).with_apply_func(
            compute_intraday_mr_indicators_nb,
            takes_1d=True,
            lookback=lookback,
            band_width=band_width,
            adx_period=adx_period,
            adx_threshold=adx_threshold,
            vol_window=vol_window,
            sigma_target=sigma_target,
            max_leverage=max_leverage,
        )

        close_s = pd.Series(close_arr[:, 0]) if close_arr.ndim > 1 else pd.Series(close_arr)
        high_s = pd.Series(high_arr[:, 0]) if high_arr.ndim > 1 else pd.Series(high_arr)
        low_s = pd.Series(low_arr[:, 0]) if low_arr.ndim > 1 else pd.Series(low_arr)
        open_s = pd.Series(open_arr[:, 0]) if open_arr.ndim > 1 else pd.Series(open_arr)

        imr = IMR.run(
            index_ns=idx_ns,
            high_minute=high_s,
            low_minute=low_s,
            close_minute=close_s,
            lookback=lookback,
            band_width=band_width,
            adx_period=adx_period,
            adx_threshold=adx_threshold,
            vol_window=vol_window,
            sigma_target=sigma_target,
            max_leverage=max_leverage,
        )

        pf = vbt.Portfolio.from_signals(
            close_s,
            open=open_s,
            high=high_s,
            low=low_s,
            signal_func_nb=intraday_mr_signal_nb,
            signal_args=(
                vbt.Rep("close_arr"), vbt.Rep("upper_arr"), vbt.Rep("lower_arr"),
                vbt.Rep("twap_arr"), vbt.Rep("regime_ok_arr"), vbt.Rep("index_arr"),
                vbt.Rep("eod_hour"), vbt.Rep("eod_minute"), vbt.Rep("eval_freq"),
            ),
            broadcast_named_args=dict(
                close_arr=close_s,
                upper_arr=imr.upper_band.values,
                lower_arr=imr.lower_band.values,
                twap_arr=imr.twap.values,
                regime_ok_arr=imr.regime_ok.values,
                index_arr=idx_ns,
                eod_hour=eod_hour,
                eod_minute=eod_minute,
                eval_freq=eval_freq,
            ),
            leverage=imr.leverage.values,
            slippage=SLIPPAGE,
            fixed_fees=FIXED_FEES,
            sl_stop=sl_stop,
            tp_stop=tp_stop,
            init_cash=INIT_CASH,
            freq="1min",
        )
        return pf.deep_getattr(metric)

    return vbt.cv_split(
        _run_pipeline,
        splitter=splitter,
        takeable_args=["high_arr", "low_arr", "close_arr", "open_arr", "idx_ns"],
        parameterized_kwargs=dict(engine="threadpool", chunk_len="auto"),
        merge_func="concat",
    )


# ═══════════════════════════════════════════════════════════════════════
# 7. PLOTTING
# ═══════════════════════════════════════════════════════════════════════

def plot_monthly_heatmap(pf: vbt.Portfolio, title: str = "Monthly Returns (%)") -> go.Figure:
    rets = pf.returns
    monthly = rets.resample("ME").apply(lambda x: (1 + x).prod() - 1)
    df = pd.DataFrame({
        "year": monthly.index.year,
        "month": monthly.index.month,
        "ret": monthly.values,
    })
    pivot = df.pivot(index="year", columns="month", values="ret")
    pivot.columns = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ]
    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=pivot.columns,
        y=pivot.index.astype(str),
        colorscale="RdYlGn",
        zmid=0,
        text=np.round(pivot.values * 100, 1),
        texttemplate="%{text}%",
    ))
    fig.update_layout(title=title, height=400)
    return fig


# ═══════════════════════════════════════════════════════════════════════
# 8. MAIN
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    results_dir = "results/intraday_mr"
    os.makedirs(results_dir, exist_ok=True)

    # ── Load data ──────────────────────────────────────────────────
    print("Loading EUR-USD minute data...")
    raw = pd.read_parquet("data/EUR-USD.parquet")
    raw["date"] = pd.to_datetime(raw["date"])
    raw = raw.set_index("date").sort_index()
    index_ns = vbt.dt.to_ns(raw.index)

    ann_factor = compute_ann_factor(raw.index)
    print(f"  {len(raw)} bars: {raw.index[0]} -> {raw.index[-1]}")
    print(f"  Annualization factor: {ann_factor:.0f}")

    # ── Hold-out split: 80% train / 20% test ──────────────────────
    holdout_date = raw.index[int(len(raw) * 0.8)]
    raw_train = raw.loc[:holdout_date]
    raw_test = raw.loc[holdout_date:]
    index_ns_train = vbt.dt.to_ns(raw_train.index)
    index_ns_test = vbt.dt.to_ns(raw_test.index)
    print(f"  Train: {len(raw_train)} bars ({raw_train.index[0]} -> {raw_train.index[-1]})")
    print(f"  Test:  {len(raw_test)} bars ({raw_test.index[0]} -> {raw_test.index[-1]})")

    # ── Standard backtest on FULL data ─────────────────────────────
    print("\nRunning standard backtest (full data, default params)...")
    pf, imr = run_standard_backtest(
        raw, index_ns, ann_factor,
        lookback=60, band_width=2.0, sl_stop=0.003, tp_stop=0.0045,
        eval_freq=5,
    )

    print("\n" + "=" * 60)
    print("INTRADAY TWAP MEAN REVERSION — STANDARD (realistic costs)")
    print("=" * 60)
    print(pf.stats().to_string())
    print("=" * 60)

    # ── Plots (standard) ──────────────────────────────────────────
    fig_pf = pf.plot(subplots=["cumulative_returns", "drawdowns", "underwater"])
    fig_pf.update_layout(title="Intraday MR — Portfolio Overview (realistic)", height=900)
    fig_pf.write_html(f"{results_dir}/portfolio_overview.html")

    fig_monthly = plot_monthly_heatmap(pf, "Intraday MR — Monthly Returns (%)")
    fig_monthly.write_html(f"{results_dir}/monthly_returns.html")

    print(f"\nPlots saved to {results_dir}/")

    # ── Cross-validation: Purged K-Fold (on train set) ─────────────
    print("\n" + "=" * 60)
    print("CV 1: Purged K-Fold (Lopez de Prado)")
    print("=" * 60)

    high_arr_train = vbt.to_2d_array(raw_train["high"])
    low_arr_train = vbt.to_2d_array(raw_train["low"])
    close_arr_train = vbt.to_2d_array(raw_train["close"])
    open_arr_train = vbt.to_2d_array(raw_train["open"])

    splitter_kfold = vbt.Splitter.from_purged_kfold(
        raw_train.index,
        n_folds=10,
        n_test_folds=2,
        purge_td="1 hour",
        embargo_td="30 min",
    )

    cv_kfold = _build_cv_runner(splitter_kfold)

    param_grid = dict(
        lookback=vbt.Param([20, 30, 60, 120, 240]),
        band_width=vbt.Param([1.5, 2.0, 2.5, 3.0, 3.5]),
        sl_stop=vbt.Param([0.002, 0.003, 0.005]),
    )

    n_combos = 5 * 5 * 3
    n_splits = len(splitter_kfold.splits)
    print(f"  Splitter: {n_splits} splits")
    print(f"  Grid: {n_combos} combos x {n_splits} splits = {n_combos * n_splits} backtests")

    grid_perf_kfold, best_perf_kfold = cv_kfold(
        high_arr=high_arr_train,
        low_arr=low_arr_train,
        close_arr=close_arr_train,
        open_arr=open_arr_train,
        idx_ns=index_ns_train,
        **param_grid,
        tp_stop=0.0045,
        eod_hour=21,
        eod_minute=0,
        eval_freq=5,
        metric="sharpe_ratio",
        _return_grid="all",
        _index=raw_train.index,
    )

    fig_cv_kfold = grid_perf_kfold.vbt.heatmap(
        x_level="lookback",
        y_level="band_width",
        slider_level="split",
    )
    fig_cv_kfold.write_html(f"{results_dir}/cv_kfold_heatmap.html")
    print(f"  CV k-fold heatmap saved.")

    # ── Cross-validation: Walk-Forward (on train set) ──────────────
    print("\n" + "=" * 60)
    print("CV 2: Purged Walk-Forward")
    print("=" * 60)

    splitter_wf = vbt.Splitter.from_purged_walkforward(
        raw_train.index,
        n_folds=10,
        n_test_folds=1,
        min_train_folds=3,
        purge_td="1 hour",
    )

    cv_wf = _build_cv_runner(splitter_wf)
    n_splits_wf = len(splitter_wf.splits)
    print(f"  Splitter: {n_splits_wf} splits (walk-forward)")
    print(f"  Grid: {n_combos} combos x {n_splits_wf} splits = {n_combos * n_splits_wf} backtests")

    grid_perf_wf, best_perf_wf = cv_wf(
        high_arr=high_arr_train,
        low_arr=low_arr_train,
        close_arr=close_arr_train,
        open_arr=open_arr_train,
        idx_ns=index_ns_train,
        **param_grid,
        tp_stop=0.0045,
        eod_hour=21,
        eod_minute=0,
        eval_freq=5,
        metric="sharpe_ratio",
        _return_grid="all",
        _index=raw_train.index,
    )

    fig_cv_wf = grid_perf_wf.vbt.heatmap(
        x_level="lookback",
        y_level="band_width",
        slider_level="split",
    )
    fig_cv_wf.write_html(f"{results_dir}/cv_walkforward_heatmap.html")
    print(f"  CV walk-forward heatmap saved.")

    # ── Best params consensus ──────────────────────────────────────
    print("\n" + "=" * 60)
    print("PARAMETER SELECTION")
    print("=" * 60)

    best_idx_kfold = best_perf_kfold.idxmax() if isinstance(best_perf_kfold.index, pd.MultiIndex) else None
    best_idx_wf = best_perf_wf.idxmax() if isinstance(best_perf_wf.index, pd.MultiIndex) else None

    print(f"  K-Fold best: {best_idx_kfold} (Sharpe: {best_perf_kfold.max():.4f})")
    print(f"  Walk-Forward best: {best_idx_wf} (Sharpe: {best_perf_wf.max():.4f})")

    # Use walk-forward params as primary (more conservative / realistic)
    if best_idx_wf is not None:
        level_names = best_perf_wf.index.names
        best_row = best_perf_wf[best_perf_wf == best_perf_wf.max()]
        opt_params = {}
        for name in ["lookback", "band_width", "sl_stop"]:
            if name in level_names:
                opt_params[name] = best_row.index.get_level_values(name)[0]
    elif best_idx_kfold is not None:
        level_names = best_perf_kfold.index.names
        best_row = best_perf_kfold[best_perf_kfold == best_perf_kfold.max()]
        opt_params = {}
        for name in ["lookback", "band_width", "sl_stop"]:
            if name in level_names:
                opt_params[name] = best_row.index.get_level_values(name)[0]
    else:
        opt_params = {"lookback": 60, "band_width": 2.0, "sl_stop": 0.003}

    print(f"  Selected params: {opt_params}")

    # ── Re-run optimized on train set ──────────────────────────────
    print("\nRe-running optimized on TRAIN set...")
    pf_opt_train, _ = run_standard_backtest(
        raw_train, index_ns_train, ann_factor,
        lookback=int(opt_params.get("lookback", 60)),
        band_width=float(opt_params.get("band_width", 2.0)),
        sl_stop=float(opt_params.get("sl_stop", 0.003)),
        tp_stop=float(opt_params.get("sl_stop", 0.003)) * 1.5,
        eval_freq=5,
    )

    print("\n" + "=" * 60)
    print("OPTIMIZED — TRAIN SET")
    print("=" * 60)
    print(pf_opt_train.stats().to_string())

    # ── Hold-out test: final validation ────────────────────────────
    print("\n" + "=" * 60)
    print("HOLD-OUT TEST (20% unseen data)")
    print("=" * 60)

    pf_holdout, _ = run_standard_backtest(
        raw_test, index_ns_test, ann_factor,
        lookback=int(opt_params.get("lookback", 60)),
        band_width=float(opt_params.get("band_width", 2.0)),
        sl_stop=float(opt_params.get("sl_stop", 0.003)),
        tp_stop=float(opt_params.get("sl_stop", 0.003)) * 1.5,
        eval_freq=5,
    )

    print(pf_holdout.stats().to_string())
    print("=" * 60)

    comparison = pd.DataFrame({
        "Standard (full)": pf.stats(),
        "Optimized (train)": pf_opt_train.stats(),
        "Hold-out (test)": pf_holdout.stats(),
    })
    print("\n" + "=" * 60)
    print("COMPARISON: Standard vs Optimized vs Hold-Out")
    print("=" * 60)
    print(comparison.to_string())
    print("=" * 60)

    # ── Save hold-out plots ────────────────────────────────────────
    fig_holdout = pf_holdout.plot(subplots=["cumulative_returns", "drawdowns", "underwater"])
    fig_holdout.update_layout(title="Intraday MR — Hold-Out Test", height=900)
    fig_holdout.write_html(f"{results_dir}/portfolio_holdout.html")

    fig_monthly_holdout = plot_monthly_heatmap(pf_holdout, "Intraday MR — Hold-Out Monthly Returns (%)")
    fig_monthly_holdout.write_html(f"{results_dir}/monthly_returns_holdout.html")

    fig_opt = pf_opt_train.plot(subplots=["cumulative_returns", "drawdowns", "underwater"])
    fig_opt.update_layout(title="Intraday MR — Optimized (Train)", height=900)
    fig_opt.write_html(f"{results_dir}/portfolio_optimized.html")

    fig_monthly_opt = plot_monthly_heatmap(pf_opt_train, "Intraday MR — Optimized Monthly Returns (%)")
    fig_monthly_opt.write_html(f"{results_dir}/monthly_returns_optimized.html")

    print(f"\nAll results saved to {results_dir}/")
