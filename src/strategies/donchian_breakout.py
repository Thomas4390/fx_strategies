"""Donchian Channel Breakout: Rolling Max/Min Channels + VWAP Confirmation."""

import numpy as np
import vectorbtpro as vbt
from numba import njit

from framework.spec import IndicatorSpec, ParamDef, PortfolioConfig, StrategySpec
from utils import find_day_boundaries_nb


# ═══════════════════════════════════════════════════════════════════════
# NUMBA KERNELS
# ═══════════════════════════════════════════════════════════════════════


@njit
def rolling_max_nb(data: np.ndarray, period: int) -> np.ndarray:
    """Rolling max over [i-period, i-1] (lagged by 1 bar)."""
    n = len(data)
    out = np.full(n, np.nan)
    for i in range(period + 1, n):
        mx = data[i - period]
        for j in range(i - period + 1, i):
            if data[j] > mx:
                mx = data[j]
        out[i] = mx
    return out


@njit
def rolling_min_nb(data: np.ndarray, period: int) -> np.ndarray:
    """Rolling min over [i-period, i-1] (lagged by 1 bar)."""
    n = len(data)
    out = np.full(n, np.nan)
    for i in range(period + 1, n):
        mn = data[i - period]
        for j in range(i - period + 1, i):
            if data[j] < mn:
                mn = data[j]
        out[i] = mn
    return out


@njit
def compute_intraday_donchian_nb(
    index_ns: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    open_: np.ndarray,
    volume: np.ndarray,
    entry_period: int,
    exit_period: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    upper = rolling_max_nb(high, entry_period)
    lower = rolling_min_nb(low, entry_period)
    exit_upper = rolling_max_nb(high, exit_period)
    exit_lower = rolling_min_nb(low, exit_period)

    start_arr, end_arr, n_days = find_day_boundaries_nb(index_ns)
    group_lens = end_arr[:n_days] - start_arr[:n_days]
    vwap = vbt.indicators.nb.vwap_1d_nb(high, low, close, volume, group_lens)

    return upper, lower, exit_upper, exit_lower, vwap


# ═══════════════════════════════════════════════════════════════════════
# SIGNAL FUNCTION
# ═══════════════════════════════════════════════════════════════════════


@njit
def intraday_donchian_signal_nb(
    c,
    close_arr: np.ndarray,
    upper_arr: np.ndarray,
    lower_arr: np.ndarray,
    exit_upper_arr: np.ndarray,
    exit_lower_arr: np.ndarray,
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
    ub = vbt.pf_nb.select_nb(c, upper_arr)
    lb = vbt.pf_nb.select_nb(c, lower_arr)
    exu = vbt.pf_nb.select_nb(c, exit_upper_arr)
    exl = vbt.pf_nb.select_nb(c, exit_lower_arr)
    vw = vbt.pf_nb.select_nb(c, vwap_arr)

    if np.isnan(px) or np.isnan(ub) or np.isnan(lb) or np.isnan(vw):
        return False, False, False, False

    in_long = vbt.pf_nb.ctx_helpers.in_long_position_nb(c)
    in_short = vbt.pf_nb.ctx_helpers.in_short_position_nb(c)

    if not in_long and not in_short:
        if px > ub and px > vw:
            return True, False, False, False
        elif px < lb and px < vw:
            return False, False, True, False
    elif in_long:
        if not np.isnan(exl) and px < exl:
            return False, True, False, False
    elif in_short and not np.isnan(exu) and px > exu:
        return False, False, False, True

    return False, False, False, False


# ═══════════════════════════════════════════════════════════════════════
# STRATEGY SPECIFICATION
# ═══════════════════════════════════════════════════════════════════════

spec = StrategySpec(
    name="Donchian Channel Breakout",
    indicator=IndicatorSpec(
        class_name="IntradayDonchian",
        short_name="idb",
        input_names=(
            "index_ns",
            "high_minute",
            "low_minute",
            "close_minute",
            "open_minute",
            "volume_minute",
        ),
        param_names=("entry_period", "exit_period"),
        output_names=(
            "upper_channel",
            "lower_channel",
            "exit_upper",
            "exit_lower",
            "vwap",
        ),
        kernel_func=compute_intraday_donchian_nb,
    ),
    signal_func=intraday_donchian_signal_nb,
    signal_args_map=(
        ("close_arr", "data.close"),
        ("upper_arr", "ind.upper_channel"),
        ("lower_arr", "ind.lower_channel"),
        ("exit_upper_arr", "ind.exit_upper"),
        ("exit_lower_arr", "ind.exit_lower"),
        ("vwap_arr", "ind.vwap"),
        ("index_arr", "extra.index_ns"),
        ("eod_hour", "param.eod_hour"),
        ("eod_minute", "param.eod_minute"),
    ),
    params={
        "entry_period": ParamDef(240, sweep=[120, 180, 240, 360]),
        "exit_period": ParamDef(60, sweep=[30, 60, 90, 120]),
        "eod_hour": ParamDef(21),
        "eod_minute": ParamDef(0),
    },
    portfolio_config=PortfolioConfig(freq="1min"),
    takeable_args=(
        "high_arr",
        "low_arr",
        "close_arr",
        "open_arr",
        "volume_arr",
        "idx_ns",
    ),
)
