"""MR V4: Adaptive EWM Bands.

EWM std replaces rolling std for faster volatility adaptation.
"""

import numpy as np
import vectorbtpro as vbt
from numba import njit

from framework.spec import IndicatorSpec, ParamDef, PortfolioConfig, StrategySpec
from utils import (
    compute_daily_adx_broadcast_nb,
    compute_intraday_twap_nb,
)


# ═══════════════════════════════════════════════════════════════════════
# INDICATOR KERNEL
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
    """Compute TWAP, EWM z-score, adaptive bands, and ADX filter."""
    n = len(close)

    twap = compute_intraday_twap_nb(index_ns, high, low, close)

    deviation = np.empty(n)
    for i in range(n):
        if np.isnan(twap[i]) or np.isnan(close[i]):
            deviation[i] = np.nan
        else:
            deviation[i] = close[i] - twap[i]

    ewm_std = vbt.generic.nb.ewm_std_1d_nb(
        deviation, span=ewm_span, minp=ewm_span, adjust=False
    )
    smoothed_deviation = vbt.generic.nb.ewm_mean_1d_nb(
        deviation, span=ewm_span, minp=ewm_span, adjust=False
    )

    zscore = np.full(n, np.nan)
    for i in range(n):
        sd = smoothed_deviation[i]
        es = ewm_std[i]
        if not np.isnan(sd) and not np.isnan(es) and es > 1e-10:
            zscore[i] = sd / es

    upper_band = np.full(n, np.nan)
    lower_band = np.full(n, np.nan)
    for i in range(n):
        s = ewm_std[i]
        if not np.isnan(s) and s > 1e-10 and not np.isnan(twap[i]):
            upper_band[i] = twap[i] + band_width * s
            lower_band[i] = twap[i] - band_width * s

    adx = compute_daily_adx_broadcast_nb(index_ns, high, low, close, open_, adx_period)
    regime_ok = np.ones(n, dtype=np.float64)
    for i in range(n):
        if not np.isnan(adx[i]) and adx[i] > adx_threshold:
            regime_ok[i] = 0.0

    return twap, zscore, upper_band, lower_band, regime_ok


# ═══════════════════════════════════════════════════════════════════════
# SIGNAL FUNCTION
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
    """Band-based MR with adaptive EWM bands."""
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

    px = vbt.pf_nb.select_nb(c, close_arr)
    ub = vbt.pf_nb.select_nb(c, upper_arr)
    lb = vbt.pf_nb.select_nb(c, lower_arr)
    tw = vbt.pf_nb.select_nb(c, twap_arr)
    regime = vbt.pf_nb.select_nb(c, regime_ok_arr)

    if np.isnan(px) or np.isnan(ub) or np.isnan(lb) or np.isnan(tw):
        return False, False, False, False

    in_long = vbt.pf_nb.ctx_helpers.in_long_position_nb(c)
    in_short = vbt.pf_nb.ctx_helpers.in_short_position_nb(c)

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
# STRATEGY SPECIFICATION
# ═══════════════════════════════════════════════════════════════════════

spec = StrategySpec(
    name="MR V4: Adaptive EWM Bands",
    indicator=IndicatorSpec(
        class_name="MR_V4",
        short_name="mr_v4",
        input_names=(
            "index_ns",
            "high_minute",
            "low_minute",
            "close_minute",
            "open_minute",
        ),
        param_names=("ewm_span", "band_width", "adx_period", "adx_threshold"),
        output_names=("twap", "zscore", "upper_band", "lower_band", "regime_ok"),
        kernel_func=compute_mr_v4_indicators_nb,
    ),
    signal_func=mr_v4_signal_nb,
    signal_args_map=(
        ("close_arr", "data.close"),
        ("upper_arr", "ind.upper_band"),
        ("lower_arr", "ind.lower_band"),
        ("twap_arr", "ind.twap"),
        ("regime_ok_arr", "ind.regime_ok"),
        ("index_arr", "extra.index_ns"),
        ("eod_hour", "param.eod_hour"),
        ("eod_minute", "param.eod_minute"),
        ("eval_freq", "param.eval_freq"),
    ),
    params={
        "ewm_span": ParamDef(60, sweep=[20, 40, 60, 120]),
        "band_width": ParamDef(2.0, sweep=[1.5, 2.0, 2.5, 3.0]),
        "adx_period": ParamDef(14),
        "adx_threshold": ParamDef(30.0),
        "eod_hour": ParamDef(21),
        "eod_minute": ParamDef(0),
        "eval_freq": ParamDef(5),
        "sl_stop": ParamDef(0.005, sweep=[0.001, 0.002, 0.003, 0.005]),
    },
    portfolio_config=PortfolioConfig(leverage=1.0, sl_stop=0.005),
)
