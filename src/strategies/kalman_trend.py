"""Kalman Trend Following: Kalman Filter + EMA Crossover + VWAP Confirmation."""

import numpy as np
import vectorbtpro as vbt
from numba import njit

from framework.spec import (
    IndicatorSpec,
    OverlayLine,
    ParamDef,
    PlotConfig,
    PortfolioConfig,
    StrategySpec,
)
from utils import find_day_boundaries_nb

# ═══════════════════════════════════════════════════════════════════════
# NUMBA KERNELS
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

        price_pred = price_est + vel_est
        vel_pred = vel_est
        p11_pred = p11 + 2.0 * p12 + p22 + process_var
        p12_pred = p12 + p22
        p22_pred = p22 + process_var * 0.01

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
    kf_price, kf_velocity = kalman_filter_1d_nb(close, process_var, measurement_var)

    ema_fast_line = vbt.generic.nb.ewm_mean_1d_nb(
        kf_price, ema_fast, minp=1, adjust=True
    )
    ema_slow_line = vbt.generic.nb.ewm_mean_1d_nb(
        kf_price, ema_slow, minp=1, adjust=True
    )

    start_arr, end_arr, n_days = find_day_boundaries_nb(index_ns)
    group_lens = end_arr[:n_days] - start_arr[:n_days]
    vwap = vbt.indicators.nb.vwap_1d_nb(high, low, close, volume, group_lens)

    return kf_price, kf_velocity, ema_fast_line, ema_slow_line, vwap


# ═══════════════════════════════════════════════════════════════════════
# SIGNAL FUNCTION
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

    is_eod = (cur_hour > eod_hour) or (
        cur_hour == eod_hour and cur_minute >= eod_minute
    )
    if is_eod:
        el = vbt.pf_nb.ctx_helpers.in_long_position_nb(c)
        es = vbt.pf_nb.ctx_helpers.in_short_position_nb(c)
        return False, el, False, es

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
# STRATEGY SPECIFICATION
# ═══════════════════════════════════════════════════════════════════════

spec = StrategySpec(
    name="Kalman Trend Following",
    indicator=IndicatorSpec(
        class_name="IntradayKalman",
        short_name="ikt",
        input_names=(
            "index_ns",
            "high_minute",
            "low_minute",
            "close_minute",
            "open_minute",
            "volume_minute",
        ),
        param_names=("process_var", "measurement_var", "ema_fast", "ema_slow"),
        output_names=(
            "kalman_price",
            "kalman_velocity",
            "ema_fast_line",
            "ema_slow_line",
            "vwap",
        ),
        kernel_func=compute_intraday_kalman_indicators_nb,
    ),
    signal_func=intraday_kalman_signal_nb,
    signal_args_map=(
        ("close_arr", "data.close"),
        ("ema_fast_arr", "ind.ema_fast_line"),
        ("ema_slow_arr", "ind.ema_slow_line"),
        ("velocity_arr", "ind.kalman_velocity"),
        ("vwap_arr", "ind.vwap"),
        ("index_arr", "extra.index_ns"),
        ("eod_hour", "param.eod_hour"),
        ("eod_minute", "param.eod_minute"),
    ),
    params={
        "process_var": ParamDef(0.001),
        "measurement_var": ParamDef(1.0),
        "ema_fast": ParamDef(100),
        "ema_slow": ParamDef(500),
        "eod_hour": ParamDef(21),
        "eod_minute": ParamDef(0),
    },
    portfolio_config=PortfolioConfig(freq="1min"),
    plot_config=PlotConfig(
        overlays=(
            OverlayLine("ind.kalman_price", "Kalman Price", color="#2196F3"),
            OverlayLine("ind.ema_fast_line", "EMA Fast", color="#4CAF50", dash="dash"),
            OverlayLine("ind.ema_slow_line", "EMA Slow", color="#FF5722", dash="dash"),
            OverlayLine("ind.vwap", "VWAP", color="#9C27B0", dash="dot"),
        ),
    ),
    takeable_args=(
        "high_arr",
        "low_arr",
        "close_arr",
        "open_arr",
        "volume_arr",
        "idx_ns",
    ),
)


if __name__ == "__main__":
    from framework.runner import run_strategy

    run_strategy(spec, mode="full")
