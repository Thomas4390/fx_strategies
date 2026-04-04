"""MR V2: Z-Score Based Exit.

Entry at |z| > entry_z, exit at |z| < exit_z. No fixed bands, no leverage.
"""

import numpy as np
import vectorbtpro as vbt
from numba import njit

from framework.spec import IndicatorSpec, ParamDef, PortfolioConfig, StrategySpec
from utils import (
    compute_daily_adx_broadcast_nb,
    compute_intraday_twap_nb,
    compute_intraday_zscore_nb,
)


# ═══════════════════════════════════════════════════════════════════════
# INDICATOR KERNEL
# ═══════════════════════════════════════════════════════════════════════


@njit(nogil=True)
def compute_mr_v2_indicators_nb(
    index_ns: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    open_: np.ndarray,
    lookback: int,
    adx_period: int,
    adx_threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute TWAP, rolling z-score, and ADX regime filter."""
    n = len(close)

    twap = compute_intraday_twap_nb(index_ns, high, low, close)

    deviation = np.empty(n)
    for i in range(n):
        if np.isnan(twap[i]) or np.isnan(close[i]):
            deviation[i] = np.nan
        else:
            deviation[i] = close[i] - twap[i]

    zscore = compute_intraday_zscore_nb(index_ns, deviation, lookback)

    adx = compute_daily_adx_broadcast_nb(index_ns, high, low, close, open_, adx_period)
    regime_ok = np.ones(n, dtype=np.float64)
    for i in range(n):
        if not np.isnan(adx[i]) and adx[i] > adx_threshold:
            regime_ok[i] = 0.0

    return twap, zscore, regime_ok


# ═══════════════════════════════════════════════════════════════════════
# SIGNAL FUNCTION
# ═══════════════════════════════════════════════════════════════════════


@njit(nogil=True)
def mr_v2_signal_nb(
    c,
    zscore_arr: np.ndarray,
    regime_ok_arr: np.ndarray,
    index_ns_arr: np.ndarray,
    eod_hour_arr: np.ndarray,
    eod_minute_arr: np.ndarray,
    eval_freq_arr: np.ndarray,
    entry_z_arr: np.ndarray,
    exit_z_arr: np.ndarray,
):
    """Z-score entry AND exit: enter at |z|>entry_z, exit at |z|<exit_z."""
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

    eval_freq = vbt.pf_nb.select_nb(c, eval_freq_arr)
    if eval_freq > 0 and cur_minute % eval_freq != 0:
        return False, False, False, False

    z = vbt.pf_nb.select_nb(c, zscore_arr)
    regime = vbt.pf_nb.select_nb(c, regime_ok_arr)
    entry_z = vbt.pf_nb.select_nb(c, entry_z_arr)
    exit_z = vbt.pf_nb.select_nb(c, exit_z_arr)

    if np.isnan(z):
        return False, False, False, False

    in_long = vbt.pf_nb.ctx_helpers.in_long_position_nb(c)
    in_short = vbt.pf_nb.ctx_helpers.in_short_position_nb(c)

    if in_long:
        if z >= -exit_z:
            return False, True, False, False
        return False, False, False, False

    if in_short:
        if z <= exit_z:
            return False, False, False, True
        return False, False, False, False

    if regime < 0.5:
        return False, False, False, False

    if z < -entry_z:
        return True, False, False, False
    elif z > entry_z:
        return False, False, True, False

    return False, False, False, False


# ═══════════════════════════════════════════════════════════════════════
# STRATEGY SPECIFICATION
# ═══════════════════════════════════════════════════════════════════════

spec = StrategySpec(
    name="MR V2: Z-Score Exit",
    indicator=IndicatorSpec(
        class_name="IntradayMRv2",
        short_name="mrv2",
        input_names=(
            "index_ns",
            "high_minute",
            "low_minute",
            "close_minute",
            "open_minute",
        ),
        param_names=("lookback", "adx_period", "adx_threshold"),
        output_names=("twap", "zscore", "regime_ok"),
        kernel_func=compute_mr_v2_indicators_nb,
    ),
    signal_func=mr_v2_signal_nb,
    signal_args_map=(
        ("zscore_arr", "ind.zscore"),
        ("regime_ok_arr", "ind.regime_ok"),
        ("index_arr", "extra.index_ns"),
        ("eod_hour", "param.eod_hour"),
        ("eod_minute", "param.eod_minute"),
        ("eval_freq", "param.eval_freq"),
        ("entry_z", "param.entry_z"),
        ("exit_z", "param.exit_z"),
    ),
    params={
        "lookback": ParamDef(60, sweep=[20, 40, 60, 120, 240]),
        "adx_period": ParamDef(14),
        "adx_threshold": ParamDef(30.0),
        "eod_hour": ParamDef(21),
        "eod_minute": ParamDef(0),
        "eval_freq": ParamDef(5),
        "entry_z": ParamDef(2.0, sweep=[1.5, 2.0, 2.5, 3.0]),
        "exit_z": ParamDef(0.5, sweep=[0.3, 0.5, 0.7]),
        "sl_stop": ParamDef(0.005, sweep=[0.002, 0.003, 0.005]),
    },
    portfolio_config=PortfolioConfig(leverage=1.0, sl_stop=0.005),
)
