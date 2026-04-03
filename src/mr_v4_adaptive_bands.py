#!/usr/bin/env python
"""
Intraday TWAP Mean Reversion V4 — Adaptive EWM Bands — EUR/USD Minute Data

Variation of the base TWAP MR strategy replacing fixed rolling_std bands
with exponentially weighted (EWM) std for faster adaptation to volatility
regime changes.

Key differences from base (V1):
- EWM std instead of rolling_std → bands widen quickly on vol spikes
- EWM mean for deviation smoothing → faster signal response
- Symmetric SL/TP (tp_stop = sl_stop)
- No vol-targeting leverage (leverage = 1.0)
- ADX regime filter retained
- EOD forced exit retained
"""

import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import vectorbtpro as vbt
from numba import njit

from utils import (
    ANNUALIZED_RETURN,
    CALMAR_RATIO,
    MAX_DRAWDOWN,
    OMEGA_RATIO,
    PROFIT_FACTOR,
    SHARPE_RATIO,
    SORTINO_RATIO,
    TOTAL_RETURN,
    apply_vbt_settings,
    compute_ann_factor,
    compute_daily_adx_broadcast_nb,
    compute_intraday_twap_nb,
    compute_metric_nb,
    find_day_boundaries_nb,
    load_fx_data,
)

# ═══════════════════════════════════════════════════════════════════════
# 0. SETTINGS
# ═══════════════════════════════════════════════════════════════════════

apply_vbt_settings()

SLIPPAGE = 0.00015  # 1.5 pips — realistic for EUR/USD minute bars
FIXED_FEES = 0.0  # No fixed commission for spot FX
INIT_CASH = 1_000_000.0
ADX_PERIOD = 14  # ADX lookback (daily-equivalent, applied to minute)
ADX_THRESHOLD = 30.0  # ADX above this = trending, disable MR


# ═══════════════════════════════════════════════════════════════════════
# 1. NUMBA KERNELS
# ═══════════════════════════════════════════════════════════════════════


@njit(nogil=True)
def compute_mr_v4_indicators_nb(
    index_ns: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    open_: np.ndarray,
    ewm_span: int,
    band_width: float,
    adx_period: int,
    adx_threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute TWAP, z-score (EWM-based), adaptive bands, and ADX filter.

    Returns:
        (twap, zscore, upper_band, lower_band, regime_ok)
    """
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

    # EWM std for adaptive band width (reacts faster to vol spikes)
    ewm_std = vbt.generic.nb.ewm_std_1d_nb(
        deviation, span=ewm_span, minp=ewm_span, adjust=False
    )

    # EWM mean for smoothed deviation
    smoothed_deviation = vbt.generic.nb.ewm_mean_1d_nb(
        deviation, span=ewm_span, minp=ewm_span, adjust=False
    )

    # Z-score = smoothed_deviation / ewm_std
    zscore = np.full(n, np.nan)
    for i in range(n):
        sd = smoothed_deviation[i]
        es = ewm_std[i]
        if not np.isnan(sd) and not np.isnan(es) and es > 1e-10:
            zscore[i] = sd / es

    # Adaptive bands
    upper_band = np.full(n, np.nan)
    lower_band = np.full(n, np.nan)
    for i in range(n):
        s = ewm_std[i]
        if not np.isnan(s) and s > 1e-10 and not np.isnan(twap[i]):
            upper_band[i] = twap[i] + band_width * s
            lower_band[i] = twap[i] - band_width * s

    # ADX regime filter (1 = MR allowed, 0 = trending)
    adx = compute_daily_adx_broadcast_nb(index_ns, high, low, close, open_, adx_period)
    regime_ok = np.ones(n, dtype=np.float64)
    for i in range(n):
        if not np.isnan(adx[i]) and adx[i] > adx_threshold:
            regime_ok[i] = 0.0

    return twap, zscore, upper_band, lower_band, regime_ok


# ═══════════════════════════════════════════════════════════════════════
# 2. SIGNAL FUNCTION
# ═══════════════════════════════════════════════════════════════════════


@njit(nogil=True)
def mr_v4_signal_nb(
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
    """Signal logic: flat+regime_ok → enter on band cross, exit at TWAP.

    Long when price < lower_band, short when price > upper_band.
    Exit long at TWAP, exit short at TWAP.
    EOD forced exit.
    """
    ts_ns = index_ns_arr[c.i]
    cur_hour = vbt.dt_nb.hour_nb(ts_ns)
    cur_minute = vbt.dt_nb.minute_nb(ts_ns)

    eod_hour = vbt.pf_nb.select_nb(c, eod_hour_arr)
    eod_minute = vbt.pf_nb.select_nb(c, eod_minute_arr)

    # EOD forced exit
    is_eod = (cur_hour > eod_hour) or (
        cur_hour == eod_hour and cur_minute >= eod_minute
    )
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
# 3. STANDARD BACKTEST
# ═══════════════════════════════════════════════════════════════════════


def run_backtest(
    raw: pd.DataFrame,
    index_ns: np.ndarray,
    ann_factor: float,
    ewm_span: int = 60,
    band_width: float = 2.0,
    eod_hour: int = 21,
    eod_minute: int = 0,
    eval_freq: int = 5,
    sl_stop: float = 0.005,
    adx_period: int = ADX_PERIOD,
    adx_threshold: float = ADX_THRESHOLD,
) -> tuple[vbt.Portfolio, object]:
    """Run V4 adaptive-band backtest with symmetric SL/TP and leverage=1.0."""
    MR_V4 = vbt.IF(
        class_name="MR_V4",
        short_name="mr_v4",
        input_names=["index_ns", "high_minute", "low_minute", "close_minute", "open_minute"],
        param_names=[
            "ewm_span",
            "band_width",
            "adx_period",
            "adx_threshold",
        ],
        output_names=[
            "twap",
            "zscore",
            "upper_band",
            "lower_band",
            "regime_ok",
        ],
    ).with_apply_func(
        compute_mr_v4_indicators_nb,
        takes_1d=True,
        ewm_span=ewm_span,
        band_width=band_width,
        adx_period=adx_period,
        adx_threshold=adx_threshold,
    )

    mr_v4 = MR_V4.run(
        index_ns=index_ns,
        high_minute=raw["high"],
        low_minute=raw["low"],
        close_minute=raw["close"],
        open_minute=raw["open"],
        ewm_span=ewm_span,
        band_width=band_width,
        adx_period=adx_period,
        adx_threshold=adx_threshold,
        jitted_loop=True,
        jitted_warmup=True,
        execute_kwargs={"engine": "threadpool", "n_chunks": "auto"},
    )

    pf = vbt.Portfolio.from_signals(
        raw["close"],
        open=raw["open"],
        high=raw["high"],
        low=raw["low"],
        jitted={"parallel": True},
        chunked="threadpool",
        signal_func_nb=mr_v4_signal_nb,
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
        broadcast_named_args={
            "close_arr": raw["close"],
            "upper_arr": mr_v4.upper_band.values,
            "lower_arr": mr_v4.lower_band.values,
            "twap_arr": mr_v4.twap.values,
            "regime_ok_arr": mr_v4.regime_ok.values,
            "index_arr": index_ns,
            "eod_hour": eod_hour,
            "eod_minute": eod_minute,
            "eval_freq": eval_freq,
        },
        leverage=1.0,
        slippage=SLIPPAGE,
        fixed_fees=FIXED_FEES,
        sl_stop=sl_stop,
        init_cash=INIT_CASH,
        freq="1min",
    )

    return pf, mr_v4


# ═══════════════════════════════════════════════════════════════════════
# 4. CV PIPELINE
# ═══════════════════════════════════════════════════════════════════════


def _build_cv_runner(splitter):
    """Build a CV-wrapped backtest function for a given splitter."""

    def _run_pipeline(
        high_arr,
        low_arr,
        close_arr,
        open_arr,
        idx_ns,
        ewm_span,
        band_width,
        sl_stop=0.005,
        eod_hour=21,
        eod_minute=0,
        eval_freq=5,
        adx_period=ADX_PERIOD,
        adx_threshold=ADX_THRESHOLD,
        metric="sharpe_ratio",
    ):
        MR_V4 = vbt.IF(
            class_name="MR_V4",
            short_name="mr_v4",
            input_names=["index_ns", "high_minute", "low_minute", "close_minute", "open_minute"],
            param_names=[
                "ewm_span",
                "band_width",
                "adx_period",
                "adx_threshold",
            ],
            output_names=[
                "twap",
                "zscore",
                "upper_band",
                "lower_band",
                "regime_ok",
            ],
        ).with_apply_func(
            compute_mr_v4_indicators_nb,
            takes_1d=True,
            ewm_span=ewm_span,
            band_width=band_width,
            adx_period=adx_period,
            adx_threshold=adx_threshold,
        )

        close_s = (
            pd.Series(close_arr[:, 0]) if close_arr.ndim > 1 else pd.Series(close_arr)
        )
        high_s = pd.Series(high_arr[:, 0]) if high_arr.ndim > 1 else pd.Series(high_arr)
        low_s = pd.Series(low_arr[:, 0]) if low_arr.ndim > 1 else pd.Series(low_arr)
        open_s = pd.Series(open_arr[:, 0]) if open_arr.ndim > 1 else pd.Series(open_arr)

        mr_v4 = MR_V4.run(
            index_ns=idx_ns,
            high_minute=high_s,
            low_minute=low_s,
            close_minute=close_s,
            open_minute=open_s,
            ewm_span=ewm_span,
            band_width=band_width,
            adx_period=adx_period,
            adx_threshold=adx_threshold,
        )

        pf = vbt.Portfolio.from_signals(
            close_s,
            open=open_s,
            high=high_s,
            low=low_s,
            jitted={"parallel": True},
            chunked="threadpool",
            signal_func_nb=mr_v4_signal_nb,
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
            broadcast_named_args={
                "close_arr": close_s,
                "upper_arr": mr_v4.upper_band.values,
                "lower_arr": mr_v4.lower_band.values,
                "twap_arr": mr_v4.twap.values,
                "regime_ok_arr": mr_v4.regime_ok.values,
                "index_arr": idx_ns,
                "eod_hour": eod_hour,
                "eod_minute": eod_minute,
                "eval_freq": eval_freq,
            },
            leverage=1.0,
            slippage=SLIPPAGE,
            fixed_fees=FIXED_FEES,
            sl_stop=sl_stop,
            init_cash=INIT_CASH,
            freq="1min",
        )
        return pf.deep_getattr(metric)

    return vbt.cv_split(
        _run_pipeline,
        splitter=splitter,
        takeable_args=["high_arr", "low_arr", "close_arr", "open_arr", "idx_ns"],
        parameterized_kwargs={"engine": "threadpool", "chunk_len": "auto"},
        merge_func="concat",
    )


# ═══════════════════════════════════════════════════════════════════════
# 5. PLOTTING
# ═══════════════════════════════════════════════════════════════════════


def plot_monthly_heatmap(
    pf: vbt.Portfolio, title: str = "Monthly Returns (%)"
) -> go.Figure:
    rets = pf.returns
    monthly = rets.resample("ME").apply(lambda x: (1 + x).prod() - 1)
    df = pd.DataFrame(
        {
            "year": monthly.index.year,
            "month": monthly.index.month,
            "ret": monthly.values,
        }
    )
    pivot = df.pivot(index="year", columns="month", values="ret")
    pivot.columns = [
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
    ]
    fig = go.Figure(
        data=go.Heatmap(
            z=pivot.values,
            x=pivot.columns,
            y=pivot.index.astype(str),
            colorscale="RdYlGn",
            zmid=0,
            text=np.round(pivot.values * 100, 1),
            texttemplate="%{text}%",
        )
    )
    fig.update_layout(title=title, height=400)
    return fig


# ═══════════════════════════════════════════════════════════════════════
# 6. MAIN
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    results_dir = "results/mr_v4"
    os.makedirs(results_dir, exist_ok=True)

    # ── Load data (VBT Pro native) ───────────────────────────────
    print("Loading EUR-USD minute data via vbt.Data...")
    raw, data = load_fx_data()
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
    train_start, train_end = raw_train.index[0], raw_train.index[-1]
    print(f"  Train: {len(raw_train)} bars ({train_start} -> {train_end})")
    print(
        f"  Test:  {len(raw_test)} bars ({raw_test.index[0]} -> {raw_test.index[-1]})"
    )

    # ── Walk-forward CV (on train set) ────────────────────────────
    print("\n" + "=" * 60)
    print("CV: Purged Walk-Forward (V4 Adaptive Bands)")
    print("=" * 60)

    high_arr_train = vbt.to_2d_array(raw_train["high"])
    low_arr_train = vbt.to_2d_array(raw_train["low"])
    close_arr_train = vbt.to_2d_array(raw_train["close"])
    open_arr_train = vbt.to_2d_array(raw_train["open"])

    splitter = vbt.Splitter.from_purged_walkforward(
        raw_train.index,
        n_folds=10,
        n_test_folds=1,
        min_train_folds=3,
        purge_td="1 hour",
    )

    cv_runner = _build_cv_runner(splitter)

    param_grid = {
        "ewm_span": vbt.Param([20, 40, 60, 120, 240]),
        "band_width": vbt.Param([1.5, 2.0, 2.5, 3.0]),
        "sl_stop": vbt.Param([0.001, 0.002, 0.003, 0.005]),
    }

    n_combos = 5 * 4 * 4
    n_splits = len(splitter.splits)
    print(f"  Splitter: {n_splits} splits (walk-forward)")
    total = n_combos * n_splits
    print(f"  Grid: {n_combos} combos x {n_splits} splits = {total} backtests")

    grid_perf, best_perf = cv_runner(
        high_arr=high_arr_train,
        low_arr=low_arr_train,
        close_arr=close_arr_train,
        open_arr=open_arr_train,
        idx_ns=index_ns_train,
        **param_grid,
        eod_hour=21,
        eod_minute=0,
        eval_freq=5,
        metric="sharpe_ratio",
        _return_grid="all",
        _index=raw_train.index,
    )

    fig_cv = grid_perf.vbt.heatmap(
        x_level="ewm_span",
        y_level="band_width",
        slider_level="split",
    )
    fig_cv.write_html(f"{results_dir}/cv_walkforward_heatmap.html")
    print("  CV walk-forward heatmap saved.")

    # ── Best params selection ─────────────────────────────────────
    print("\n" + "=" * 60)
    print("PARAMETER SELECTION")
    print("=" * 60)

    best_idx = (
        best_perf.idxmax() if isinstance(best_perf.index, pd.MultiIndex) else None
    )
    print(f"  Walk-Forward best: {best_idx} (Sharpe: {best_perf.max():.4f})")

    if best_idx is not None:
        level_names = best_perf.index.names
        best_row = best_perf[best_perf == best_perf.max()]
        opt_params = {}
        for name in ["ewm_span", "band_width", "sl_stop"]:
            if name in level_names:
                opt_params[name] = best_row.index.get_level_values(name)[0]
    else:
        opt_params = {"ewm_span": 60, "band_width": 2.0, "sl_stop": 0.005}

    opt_sl = float(opt_params.get("sl_stop", 0.005))

    print(f"  Selected params: {opt_params}")

    # ── Re-run optimized on train set ─────────────────────────────
    print("\nRe-running optimized on TRAIN set...")
    pf_opt_train, _ = run_backtest(
        raw_train,
        index_ns_train,
        ann_factor,
        ewm_span=int(opt_params.get("ewm_span", 60)),
        band_width=float(opt_params.get("band_width", 2.0)),
        sl_stop=opt_sl,
        eval_freq=5,
    )

    print("\n" + "=" * 60)
    print("MR V4 ADAPTIVE BANDS — OPTIMIZED (TRAIN SET)")
    print("=" * 60)
    print(pf_opt_train.stats().to_string())

    # ── Hold-out test: final validation ───────────────────────────
    print("\n" + "=" * 60)
    print("HOLD-OUT TEST (20% unseen data)")
    print("=" * 60)

    pf_holdout, _ = run_backtest(
        raw_test,
        index_ns_test,
        ann_factor,
        ewm_span=int(opt_params.get("ewm_span", 60)),
        band_width=float(opt_params.get("band_width", 2.0)),
        sl_stop=opt_sl,
        eval_freq=5,
    )

    print(pf_holdout.stats().to_string())
    print("=" * 60)

    comparison = pd.DataFrame(
        {
            "Optimized (train)": pf_opt_train.stats(),
            "Hold-out (test)": pf_holdout.stats(),
        }
    )
    print("\n" + "=" * 60)
    print("COMPARISON: Optimized vs Hold-Out")
    print("=" * 60)
    print(comparison.to_string())
    print("=" * 60)

    # ── Save plots ────────────────────────────────────────────────
    fig_opt = pf_opt_train.plot(
        subplots=["cumulative_returns", "drawdowns", "underwater"]
    )
    fig_opt.update_layout(
        title="MR V4 Adaptive Bands — Optimized (Train)", height=900
    )
    fig_opt.write_html(f"{results_dir}/portfolio_optimized.html")

    fig_monthly_opt = plot_monthly_heatmap(
        pf_opt_train, "MR V4 — Optimized Monthly Returns (%)"
    )
    fig_monthly_opt.write_html(f"{results_dir}/monthly_returns_optimized.html")

    fig_holdout = pf_holdout.plot(
        subplots=["cumulative_returns", "drawdowns", "underwater"]
    )
    fig_holdout.update_layout(
        title="MR V4 Adaptive Bands — Hold-Out Test", height=900
    )
    fig_holdout.write_html(f"{results_dir}/portfolio_holdout.html")

    fig_monthly_holdout = plot_monthly_heatmap(
        pf_holdout, "MR V4 — Hold-Out Monthly Returns (%)"
    )
    fig_monthly_holdout.write_html(f"{results_dir}/monthly_returns_holdout.html")

    print(f"\nAll results saved to {results_dir}/")
