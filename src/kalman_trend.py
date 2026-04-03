#!/usr/bin/env python
"""
Intraday Kalman Filter Trend Following — EUR/USD Minute Data

Full-Numba kernels + VBT PRO pipeline:
- Kalman filter 2-state [price, velocity] for trend extraction
- EMA crossover on Kalman output for direction
- VWAP as confirmation filter
- EOD forced exit
"""

import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import vectorbtpro as vbt
from numba import njit

from utils import apply_vbt_settings, find_day_boundaries_nb, load_fx_data

# ═══════════════════════════════════════════════════════════════════════
# 0. SETTINGS
# ═══════════════════════════════════════════════════════════════════════

apply_vbt_settings()


# ═══════════════════════════════════════════════════════════════════════
# 1. NUMBA KERNELS
# ═══════════════════════════════════════════════════════════════════════


@njit
def kalman_filter_1d_nb(
    close: np.ndarray,
    process_var: float,
    measurement_var: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Scalar 2-state Kalman filter: [price, velocity]."""
    n = len(close)
    kf_price = np.full(n, np.nan)
    kf_velocity = np.full(n, np.nan)

    if n == 0:
        return kf_price, kf_velocity

    # Find first valid close
    start = 0
    while start < n and np.isnan(close[start]):
        start += 1
    if start >= n:
        return kf_price, kf_velocity

    price_est = close[start]
    vel_est = 0.0
    p11 = 1.0
    p12 = 0.0
    p22 = 1.0
    kf_price[start] = price_est
    kf_velocity[start] = vel_est

    for i in range(start + 1, n):
        if np.isnan(close[i]):
            kf_price[i] = price_est + vel_est
            kf_velocity[i] = vel_est
            continue

        # Predict
        price_pred = price_est + vel_est
        vel_pred = vel_est
        p11_pred = p11 + 2.0 * p12 + p22 + process_var
        p12_pred = p12 + p22
        p22_pred = p22 + process_var * 0.01

        # Update (scalar measurement)
        innovation = close[i] - price_pred
        S = p11_pred + measurement_var
        if abs(S) < 1e-15:
            kf_price[i] = price_pred
            kf_velocity[i] = vel_pred
            continue
        k1 = p11_pred / S
        k2 = p12_pred / S

        price_est = price_pred + k1 * innovation
        vel_est = vel_pred + k2 * innovation
        p11 = (1.0 - k1) * p11_pred
        p12 = p12_pred - k1 * p12_pred
        p22 = p22_pred - k2 * p12_pred

        kf_price[i] = price_est
        kf_velocity[i] = vel_est

    return kf_price, kf_velocity


@njit
def compute_intraday_kalman_indicators_nb(
    index_ns: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    open_: np.ndarray,
    volume: np.ndarray,
    process_var: float,
    measurement_var: float,
    ema_fast: int,
    ema_slow: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Kalman filter
    kf_price, kf_velocity = kalman_filter_1d_nb(close, process_var, measurement_var)

    # EMA fast/slow on Kalman price
    ema_fast_line = vbt.generic.nb.ewm_mean_1d_nb(
        kf_price, ema_fast, minp=1, adjust=True
    )
    ema_slow_line = vbt.generic.nb.ewm_mean_1d_nb(
        kf_price, ema_slow, minp=1, adjust=True
    )

    # VWAP
    start_arr, end_arr, n_days = find_day_boundaries_nb(index_ns)
    group_lens = end_arr[:n_days] - start_arr[:n_days]
    vwap = vbt.indicators.nb.vwap_1d_nb(high, low, close, volume, group_lens)

    return kf_price, kf_velocity, ema_fast_line, ema_slow_line, vwap


# ═══════════════════════════════════════════════════════════════════════
# 2. SIGNAL FUNCTION
# ═══════════════════════════════════════════════════════════════════════


@njit
def intraday_kalman_signal_nb(
    c,
    close_arr: np.ndarray,
    ema_fast_arr: np.ndarray,
    ema_slow_arr: np.ndarray,
    velocity_arr: np.ndarray,
    vwap_arr: np.ndarray,
    index_ns_arr: np.ndarray,
    eod_hour_arr: np.ndarray,
    eod_minute_arr: np.ndarray,
):
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

    # Evaluate every 15 minutes
    if cur_minute % 15 != 0:
        return False, False, False, False

    px = vbt.pf_nb.select_nb(c, close_arr)
    ef = vbt.pf_nb.select_nb(c, ema_fast_arr)
    es_val = vbt.pf_nb.select_nb(c, ema_slow_arr)
    vel = vbt.pf_nb.select_nb(c, velocity_arr)
    vw = vbt.pf_nb.select_nb(c, vwap_arr)

    if (
        np.isnan(px)
        or np.isnan(ef)
        or np.isnan(es_val)
        or np.isnan(vel)
        or np.isnan(vw)
    ):
        return False, False, False, False

    in_long = vbt.pf_nb.ctx_helpers.in_long_position_nb(c)
    in_short = vbt.pf_nb.ctx_helpers.in_short_position_nb(c)

    if not in_long and not in_short:
        if ef > es_val and vel > 0.0 and px > vw:
            return True, False, False, False
        elif ef < es_val and vel < 0.0 and px < vw:
            return False, False, True, False
    elif in_long:
        if ef < es_val or vel < 0.0:
            return False, True, False, False
    elif in_short and (ef > es_val or vel > 0.0):
        return False, False, False, True

    return False, False, False, False


# ═══════════════════════════════════════════════════════════════════════
# 3. STANDARD BACKTEST
# ═══════════════════════════════════════════════════════════════════════


def run_standard_backtest(
    raw,
    index_ns,
    process_var=0.001,
    measurement_var=1.0,
    ema_fast=100,
    ema_slow=500,
    eod_hour=21,
    eod_minute=0,
):
    IKT = vbt.IF(
        class_name="IntradayKalman",
        short_name="ikt",
        input_names=[
            "index_ns",
            "high_minute",
            "low_minute",
            "close_minute",
            "open_minute",
            "volume_minute",
        ],
        param_names=["process_var", "measurement_var", "ema_fast", "ema_slow"],
        output_names=[
            "kalman_price",
            "kalman_velocity",
            "ema_fast_line",
            "ema_slow_line",
            "vwap",
        ],
    ).with_apply_func(
        compute_intraday_kalman_indicators_nb,
        takes_1d=True,
        process_var=process_var,
        measurement_var=measurement_var,
        ema_fast=ema_fast,
        ema_slow=ema_slow,
    )

    ikt = IKT.run(
        index_ns=index_ns,
        high_minute=raw["high"],
        low_minute=raw["low"],
        close_minute=raw["close"],
        open_minute=raw["open"],
        volume_minute=raw["volume"],
        process_var=process_var,
        measurement_var=measurement_var,
        ema_fast=ema_fast,
        ema_slow=ema_slow,
        jitted_loop=True,
        jitted_warmup=True,
        execute_kwargs={"engine": "threadpool", "n_chunks": "auto"},
    )

    pf = vbt.Portfolio.from_signals(
        raw["close"],
        signal_func_nb=intraday_kalman_signal_nb,
        signal_args=(
            vbt.Rep("close_arr"),
            vbt.Rep("ema_fast_arr"),
            vbt.Rep("ema_slow_arr"),
            vbt.Rep("velocity_arr"),
            vbt.Rep("vwap_arr"),
            vbt.Rep("index_arr"),
            vbt.Rep("eod_hour"),
            vbt.Rep("eod_minute"),
        ),
        broadcast_named_args={
            "close_arr": raw["close"],
            "ema_fast_arr": ikt.ema_fast_line.values,
            "ema_slow_arr": ikt.ema_slow_line.values,
            "velocity_arr": ikt.kalman_velocity.values,
            "vwap_arr": ikt.vwap.values,
            "index_arr": index_ns,
            "eod_hour": eod_hour,
            "eod_minute": eod_minute,
        },
        slippage=0.00015,
        fixed_fees=0.0,
        init_cash=1_000_000,
        freq="1T",
    )
    return pf, ikt


# ═══════════════════════════════════════════════════════════════════════
# 4. CV PIPELINE
# ═══════════════════════════════════════════════════════════════════════


def create_cv_pipeline(splitter):
    def _run(
        high_arr,
        low_arr,
        close_arr,
        open_arr,
        volume_arr,
        idx_ns,
        process_var,
        measurement_var,
        ema_fast,
        ema_slow,
        eod_hour=21,
        eod_minute=0,
        metric="sharpe_ratio",
    ):
        close_s = (
            pd.Series(close_arr[:, 0]) if close_arr.ndim > 1 else pd.Series(close_arr)
        )
        high_s = pd.Series(high_arr[:, 0]) if high_arr.ndim > 1 else pd.Series(high_arr)
        low_s = pd.Series(low_arr[:, 0]) if low_arr.ndim > 1 else pd.Series(low_arr)
        open_s = pd.Series(open_arr[:, 0]) if open_arr.ndim > 1 else pd.Series(open_arr)
        vol_s = (
            pd.Series(volume_arr[:, 0])
            if volume_arr.ndim > 1
            else pd.Series(volume_arr)
        )

        IKT = vbt.IF(
            class_name="IntradayKalman",
            short_name="ikt",
            input_names=[
                "index_ns",
                "high_minute",
                "low_minute",
                "close_minute",
                "open_minute",
                "volume_minute",
            ],
            param_names=["process_var", "measurement_var", "ema_fast", "ema_slow"],
            output_names=[
                "kalman_price",
                "kalman_velocity",
                "ema_fast_line",
                "ema_slow_line",
                "vwap",
            ],
        ).with_apply_func(
            compute_intraday_kalman_indicators_nb,
            takes_1d=True,
            process_var=process_var,
            measurement_var=measurement_var,
            ema_fast=ema_fast,
            ema_slow=ema_slow,
        )

        ikt = IKT.run(
            index_ns=idx_ns,
            high_minute=high_s,
            low_minute=low_s,
            close_minute=close_s,
            open_minute=open_s,
            volume_minute=vol_s,
            process_var=process_var,
            measurement_var=measurement_var,
            ema_fast=ema_fast,
            ema_slow=ema_slow,
        )

        pf = vbt.Portfolio.from_signals(
            close_s,
            signal_func_nb=intraday_kalman_signal_nb,
            signal_args=(
                vbt.Rep("close_arr"),
                vbt.Rep("ema_fast_arr"),
                vbt.Rep("ema_slow_arr"),
                vbt.Rep("velocity_arr"),
                vbt.Rep("vwap_arr"),
                vbt.Rep("index_arr"),
                vbt.Rep("eod_hour"),
                vbt.Rep("eod_minute"),
            ),
            broadcast_named_args={
                "close_arr": close_s,
                "ema_fast_arr": ikt.ema_fast_line.values,
                "ema_slow_arr": ikt.ema_slow_line.values,
                "velocity_arr": ikt.kalman_velocity.values,
                "vwap_arr": ikt.vwap.values,
                "index_arr": idx_ns,
                "eod_hour": eod_hour,
                "eod_minute": eod_minute,
            },
            slippage=0.00015,
            fixed_fees=0.0,
            init_cash=1_000_000,
            freq="1T",
        )
        return pf.deep_getattr(metric)

    return vbt.cv_split(
        _run,
        splitter=splitter,
        takeable_args=[
            "high_arr",
            "low_arr",
            "close_arr",
            "open_arr",
            "volume_arr",
            "idx_ns",
        ],
        parameterized_kwargs={"engine": "threadpool", "chunk_len": "auto"},
        merge_func="concat",
    )


# ═══════════════════════════════════════════════════════════════════════
# 5. PLOTTING
# ═══════════════════════════════════════════════════════════════════════


def plot_monthly_heatmap(pf: vbt.Portfolio) -> go.Figure:
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
    fig.update_layout(title="Kalman Trend — Monthly Returns (%)", height=400)
    return fig


# ═══════════════════════════════════════════════════════════════════════
# 6. MAIN
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    results_dir = "results/intraday_kalman"
    os.makedirs(results_dir, exist_ok=True)

    print("Loading EUR-USD minute data via vbt.Data...")
    raw, data = load_fx_data()
    raw["volume"] = 1.0
    index_ns = vbt.dt.to_ns(raw.index)
    print(f"  {len(raw)} bars: {raw.index[0]} → {raw.index[-1]}")

    # ── Standard backtest ──────────────────────────────────────────
    print(
        "\nRunning standard backtest (process_var=0.001, measurement_var=1.0, ema=100/500)..."
    )
    pf, ikt = run_standard_backtest(raw, index_ns)

    print("\n" + "=" * 60)
    print("INTRADAY KALMAN TREND — STANDARD")
    print("=" * 60)
    print(pf.stats().to_string())
    print("=" * 60)

    fig_pf = pf.plot(subplots=["cumulative_returns", "drawdowns", "underwater"])
    fig_pf.update_layout(title="Kalman Trend — Portfolio Overview", height=900)
    fig_pf.write_html(f"{results_dir}/portfolio_overview.html")

    fig_monthly = plot_monthly_heatmap(pf)
    fig_monthly.write_html(f"{results_dir}/monthly_returns.html")
    print(f"\nPlots saved to {results_dir}/")

    # ── Cross-validation ───────────────────────────────────────────
    print("\nPreparing CV...")
    high_arr = vbt.to_2d_array(raw["high"])
    low_arr = vbt.to_2d_array(raw["low"])
    close_arr = vbt.to_2d_array(raw["close"])
    open_arr = vbt.to_2d_array(raw["open"])
    volume_arr = vbt.to_2d_array(raw["volume"])

    splitter = vbt.Splitter.from_n_rolling(
        raw.index,
        n=5,
        length=1_500_000,
        split=0.5,
        set_labels=["train", "test"],
    )
    cv = create_cv_pipeline(splitter)

    n_combos = 3 * 3 * 3 * 3
    print(f"Running CV: {n_combos} combos x 5 splits = {n_combos * 5} backtests...")

    grid_perf, best_perf = cv(
        high_arr=high_arr,
        low_arr=low_arr,
        close_arr=close_arr,
        open_arr=open_arr,
        volume_arr=volume_arr,
        idx_ns=index_ns,
        process_var=vbt.Param([0.0001, 0.001, 0.01]),
        measurement_var=vbt.Param([0.5, 1.0, 5.0]),
        ema_fast=vbt.Param([50, 100, 200]),
        ema_slow=vbt.Param([200, 500, 1000]),
        eod_hour=21,
        eod_minute=0,
        metric="sharpe_ratio",
        _return_grid="all",
        _index=raw.index,
    )

    print(f"  Grid shape: {grid_perf.shape}")
    print(f"  Best perf shape: {best_perf.shape}")

    fig_cv = grid_perf.vbt.heatmap(
        x_level="process_var",
        y_level="measurement_var",
        slider_level="split",
    )
    fig_cv.write_html(f"{results_dir}/cv_heatmap.html")

    if isinstance(best_perf.index, pd.MultiIndex):
        print(f"\nBest params: {best_perf.idxmax()}")
        print(f"Best Sharpe: {best_perf.max():.4f}")

    print(f"\nAll results saved to {results_dir}/")
