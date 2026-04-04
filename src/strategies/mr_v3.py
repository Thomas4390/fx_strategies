"""MR V3: Session Filter (London/NY Overlap).

Band-based mean reversion restricted to high-liquidity hours.
"""

import numpy as np
import vectorbtpro as vbt
from numba import njit

from framework.spec import IndicatorSpec, ParamDef, PortfolioConfig, StrategySpec
from utils import (
    compute_daily_adx_broadcast_nb,
    compute_intraday_rolling_std_nb,
    compute_intraday_twap_nb,
    compute_intraday_zscore_nb,
)


# ═══════════════════════════════════════════════════════════════════════
# INDICATOR KERNEL
# ═══════════════════════════════════════════════════════════════════════


@njit(nogil=True)
def compute_mr_v3_indicators_nb(
    index_ns: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    open_: np.ndarray,
    lookback: int,
    band_width: float,
    adx_period: int,
    adx_threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute TWAP, z-score, bands, and ADX regime filter."""
    n = len(close)

    twap = compute_intraday_twap_nb(index_ns, high, low, close)

    deviation = np.empty(n)
    for i in range(n):
        if np.isnan(twap[i]) or np.isnan(close[i]):
            deviation[i] = np.nan
        else:
            deviation[i] = close[i] - twap[i]

    zscore = compute_intraday_zscore_nb(index_ns, deviation, lookback)

    rolling_std = compute_intraday_rolling_std_nb(index_ns, deviation, lookback)
    upper_band = np.full(n, np.nan)
    lower_band = np.full(n, np.nan)
    for i in range(n):
        s = rolling_std[i]
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
def mr_v3_signal_nb(
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
    session_start_arr: np.ndarray,
    session_end_arr: np.ndarray,
):
    """Band-based MR with session filter for London/NY overlap."""
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

    session_start = vbt.pf_nb.select_nb(c, session_start_arr)
    session_end = vbt.pf_nb.select_nb(c, session_end_arr)
    in_session = (cur_hour >= session_start) and (cur_hour < session_end)

    # Outside session: allow exits, no new entries
    if not in_session:
        if in_long and px >= tw:
            return False, True, False, False
        if in_short and px <= tw:
            return False, False, False, True
        return False, False, False, False

    # Inside session
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
    name="MR V3: Session Filter",
    indicator=IndicatorSpec(
        class_name="MR_V3",
        short_name="mr_v3",
        input_names=(
            "index_ns",
            "high_minute",
            "low_minute",
            "close_minute",
            "open_minute",
        ),
        param_names=("lookback", "band_width", "adx_period", "adx_threshold"),
        output_names=("twap", "zscore", "upper_band", "lower_band", "regime_ok"),
        kernel_func=compute_mr_v3_indicators_nb,
    ),
    signal_func=mr_v3_signal_nb,
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
        ("session_start", "param.session_start"),
        ("session_end", "param.session_end"),
    ),
    params={
        "lookback": ParamDef(60, sweep=[20, 40, 60, 120, 240]),
        "band_width": ParamDef(2.0, sweep=[1.5, 2.0, 2.5, 3.0]),
        "adx_period": ParamDef(14),
        "adx_threshold": ParamDef(30.0),
        "eod_hour": ParamDef(21),
        "eod_minute": ParamDef(0),
        "eval_freq": ParamDef(5),
        "session_start": ParamDef(8),
        "session_end": ParamDef(16),
        "sl_stop": ParamDef(0.005, sweep=[0.001, 0.002, 0.003, 0.005]),
    },
    portfolio_config=PortfolioConfig(leverage=1.0, sl_stop=0.005),
)


if __name__ == "__main__":
    from framework.runner import run_strategy

    run_strategy(spec, mode="full")
