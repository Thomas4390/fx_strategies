"""
Shared utilities for FX intraday strategies.

Numba-compiled kernels and common settings reused across all strategy modules.
"""


import numpy as np
import pandas as pd
import vectorbtpro as vbt
from numba import njit, prange

# ═══════════════════════════════════════════════════════════════════════
# PLOTTING
# ═══════════════════════════════════════════════════════════════════════


def configure_figure_for_fullscreen(fig):
    fig.update_layout(
        width=None,
        height=None,
        autosize=True,
        margin={"l": 30, "r": 30, "t": 60, "b": 30},
        title={"font": {"size": 20}, "x": 0.5, "xanchor": "center"},
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "center",
            "x": 0.5,
            "font": {"size": 12},
        },
    )
    return fig


def apply_vbt_settings() -> None:
    vbt.settings.set("plotting.pre_show_func", configure_figure_for_fullscreen)
    vbt.settings.returns.year_freq = pd.Timedelta(hours=24) * 252


# ═══════════════════════════════════════════════════════════════════════
# TIME UTILITIES
# ═══════════════════════════════════════════════════════════════════════


@njit(nogil=True)
def find_day_boundaries_nb(
    index_ns: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Return (start_idx, end_idx, n_days) for each trading day."""
    n = len(index_ns)
    start_idx = np.empty(n, dtype=np.int64)
    end_idx = np.empty(n, dtype=np.int64)

    if n == 0:
        return start_idx, end_idx, 0

    day_number = vbt.dt_nb.days_nb(ts=index_ns)
    current_day = day_number[0]
    day_counter = 0
    current_start = 0

    for i in range(1, n):
        if day_number[i] != current_day:
            start_idx[day_counter] = current_start
            end_idx[day_counter] = i
            day_counter += 1
            current_day = day_number[i]
            current_start = i

    start_idx[day_counter] = current_start
    end_idx[day_counter] = n
    day_counter += 1

    return start_idx, end_idx, day_counter


def compute_ann_factor(index: pd.DatetimeIndex) -> float:
    """Compute annualization factor from actual data: 252 * avg_bars_per_day."""
    bars_per_day = index.to_series().groupby(index.date).count()
    return 252.0 * bars_per_day.mean()


# ═══════════════════════════════════════════════════════════════════════
# VOLATILITY & LEVERAGE
# ═══════════════════════════════════════════════════════════════════════


@njit(nogil=True)
def compute_daily_rolling_volatility_nb(
    index_ns: np.ndarray,
    close_minute: np.ndarray,
    window_size: int,
) -> np.ndarray:
    """Close-to-close rolling volatility broadcast to minute bars."""
    n = len(close_minute)
    if n == 0 or window_size <= 0:
        return np.full(n, np.nan)

    start_arr, end_arr, n_days = find_day_boundaries_nb(index_ns)
    if n_days < 2:
        return np.full(n, np.nan)

    last_close = np.full(n_days, np.nan)
    for d in range(n_days):
        if end_arr[d] > 0:
            last_close[d] = close_minute[end_arr[d] - 1]

    returns = np.full(n_days - 1, np.nan)
    for i in range(1, n_days):
        prev = last_close[i - 1]
        if not np.isnan(prev) and np.abs(prev) > 1e-9:
            returns[i - 1] = last_close[i] / prev - 1.0

    if len(returns) < window_size:
        return np.full(n, np.nan)

    rolling_std = vbt.generic.nb.rolling_std_1d_nb(
        returns,
        window=window_size,
        minp=window_size,
        ddof=1,
    )

    vol_per_minute = np.full(n, np.nan)
    for d in range(1, n_days):
        if d - 1 < rolling_std.size:
            std_val = rolling_std[d - 1]
            if start_arr[d] < end_arr[d]:
                vol_per_minute[start_arr[d] : end_arr[d]] = std_val

    return vol_per_minute


@njit(nogil=True)
def compute_leverage_nb(
    rolling_vol_per_minute: np.ndarray,
    sigma_target: float,
    max_leverage: float,
) -> np.ndarray:
    """Volatility-targeted leverage capped at max_leverage."""
    n = len(rolling_vol_per_minute)
    leverage = np.full(n, 1.0)

    for i in range(n):
        vol = rolling_vol_per_minute[i]
        if not np.isnan(vol) and vol > 1e-9:
            val = sigma_target / vol
            leverage[i] = min(val, max_leverage)

    return leverage


# ═══════════════════════════════════════════════════════════════════════
# ADX REGIME FILTER
# ═══════════════════════════════════════════════════════════════════════


@njit(nogil=True)
def compute_adx_nb(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    period: int,
) -> np.ndarray:
    """Compute ADX (Average Directional Index) for regime filtering."""
    n = len(close)
    adx = np.full(n, np.nan)
    if n < period + 1:
        return adx

    plus_dm = np.empty(n)
    minus_dm = np.empty(n)
    tr = np.empty(n)

    plus_dm[0] = 0.0
    minus_dm[0] = 0.0
    tr[0] = high[0] - low[0]

    for i in range(1, n):
        up_move = high[i] - high[i - 1]
        down_move = low[i - 1] - low[i]

        plus_dm[i] = up_move if (up_move > down_move and up_move > 0) else 0.0
        minus_dm[i] = down_move if (down_move > up_move and down_move > 0) else 0.0

        hl = high[i] - low[i]
        hc = abs(high[i] - close[i - 1])
        lc = abs(low[i] - close[i - 1])
        tr[i] = max(hl, max(hc, lc))

    smoothed_plus_dm = vbt.generic.nb.ewm_mean_1d_nb(
        plus_dm, span=period, minp=period, adjust=False
    )
    smoothed_minus_dm = vbt.generic.nb.ewm_mean_1d_nb(
        minus_dm, span=period, minp=period, adjust=False
    )
    smoothed_tr = vbt.generic.nb.ewm_mean_1d_nb(
        tr, span=period, minp=period, adjust=False
    )

    dx = np.full(n, np.nan)
    for i in range(n):
        atr_val = smoothed_tr[i]
        if not np.isnan(atr_val) and atr_val > 1e-10:
            plus_di = smoothed_plus_dm[i] / atr_val
            minus_di = smoothed_minus_dm[i] / atr_val
            di_sum = plus_di + minus_di
            if di_sum > 1e-10:
                dx[i] = abs(plus_di - minus_di) / di_sum * 100.0

    adx[:] = vbt.generic.nb.ewm_mean_1d_nb(dx, span=period, minp=period, adjust=False)

    return adx
