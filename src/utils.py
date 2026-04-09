"""
Shared utilities for FX intraday strategies.

Numba-compiled kernels and common settings reused across all strategy modules.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import vectorbtpro as vbt
from numba import njit
from numba.core.errors import NumbaPerformanceWarning

# Suppress Numba prange warning from vectorbtpro internals (from_signal_func_nb).
# This is a known VBT Pro issue — their prange loop has multiple exit points.
warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)

# Suppress VBT chunking warnings when signal_func_nb args are not in take_spec.
warnings.filterwarnings(
    "ignore",
    message="Argument at index .* not found in SequenceTaker",
    module=r"vectorbtpro\.utils\.chunking",
)

# Project root: two levels up from this file (src/utils.py -> project/)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ═══════════════════════════════════════════════════════════════════════
# PLOTTING
# ═══════════════════════════════════════════════════════════════════════


def configure_figure_for_fullscreen(fig: go.Figure) -> go.Figure:
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
    import multiprocessing

    from numba import get_num_threads

    n_cores = multiprocessing.cpu_count()
    print(f"  Parallelization: {n_cores} cores available")
    print(f"  Numba threads: {get_num_threads()}")

    vbt.settings.set("plotting.pre_show_func", configure_figure_for_fullscreen)
    vbt.settings.returns.year_freq = pd.Timedelta(hours=24) * 252


# ═══════════════════════════════════════════════════════════════════════
# DATA LOADING (VBT Pro native)
# ═══════════════════════════════════════════════════════════════════════


def load_fx_data(
    path: str = "data/EUR-USD.parquet",
    shift_hours: int = 0,
) -> tuple[pd.DataFrame, vbt.Data]:
    """Load EUR-USD parquet via vbt.Data.from_parquet and prep for OHLCV.

    Uses VBT Pro native parquet loading, then sets the date index
    and capitalizes column names for OHLCV recognition.

    Parameters
    ----------
    path : str
        Path to parquet file.
    shift_hours : int
        Hours to shift index (7 for FX 5pm ET convention on daily, 0 for intraday).

    Returns
    -------
    raw : pd.DataFrame
        Raw OHLC DataFrame with lowercase columns (for Numba kernels).
    data : vbt.Data
        VBT Data wrapper with capitalized columns (for native VBT functions).
    """
    # Resolve relative paths against project root
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = _PROJECT_ROOT / resolved

    # Load via VBT Pro native parquet reader
    data_raw = vbt.Data.from_parquet(str(resolved))
    symbol = data_raw.symbols[0]
    df = data_raw.data[symbol]
    df = df.set_index("date").sort_index()
    if shift_hours:
        df.index = df.index + pd.Timedelta(hours=shift_hours)

    # Raw DataFrame with lowercase columns for Numba kernels
    raw = df.copy()
    raw.columns = [c.lower() for c in raw.columns]

    # Add dummy volume if missing (FX data has no volume)
    if "volume" not in raw.columns:
        raw["volume"] = 1.0

    # VBT Data wrapper with capitalized columns for native functions
    df_cap = raw.copy()
    df_cap.columns = [c.capitalize() for c in df_cap.columns]
    data = vbt.Data.from_data({symbol: df_cap}, tz_localize=False, tz_convert=False)
    return raw, data


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


@njit(nogil=True)
def compute_daily_adx_broadcast_nb(
    index_ns: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    open_: np.ndarray,
    adx_period: int,
) -> np.ndarray:
    """Compute ADX on daily-resampled OHLC, then broadcast to minute bars.

    The raw ADX is calculated on 14 *minutes* when applied directly to
    minute data — meaningless noise.  This helper resamples to daily first
    (open=first, high=max, low=min, close=last), computes ADX on daily bars,
    then broadcasts each daily value to the *next* trading day's minute bars
    (1-day lag to avoid look-ahead bias).
    """
    n = len(close)
    start_arr, end_arr, n_days = find_day_boundaries_nb(index_ns)

    if n_days < adx_period + 2:
        return np.full(n, np.nan)

    # ── Resample to daily OHLC ───────────────────────────────────────
    d_high = np.empty(n_days)
    d_low = np.empty(n_days)
    d_close = np.empty(n_days)

    for d in range(n_days):
        s = start_arr[d]
        e = end_arr[d]
        mx = high[s]
        mn = low[s]
        for i in range(s + 1, e):
            if high[i] > mx:
                mx = high[i]
            if low[i] < mn:
                mn = low[i]
        d_high[d] = mx
        d_low[d] = mn
        d_close[d] = close[e - 1]

    # ── ADX on daily bars ────────────────────────────────────────────
    daily_adx = compute_adx_nb(d_high, d_low, d_close, adx_period)

    # ── Broadcast to minute bars with 1-day lag ──────────────────────
    adx_minute = np.full(n, np.nan)
    for d in range(1, n_days):
        adx_val = daily_adx[d - 1]  # previous day's ADX
        s = start_arr[d]
        e = end_arr[d]
        for i in range(s, e):
            adx_minute[i] = adx_val

    return adx_minute


# ═══════════════════════════════════════════════════════════════════════
# INTRADAY TWAP (shared across MR strategies)
# ═══════════════════════════════════════════════════════════════════════


@njit(nogil=True)
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


# ═══════════════════════════════════════════════════════════════════════
# INTRADAY Z-SCORE & ROLLING STD (day-boundary aware)
# ═══════════════════════════════════════════════════════════════════════


@njit(nogil=True)
def compute_intraday_rolling_std_nb(
    index_ns: np.ndarray,
    data: np.ndarray,
    lookback: int,
) -> np.ndarray:
    """Rolling std that resets at each day boundary.

    Prevents cross-day contamination when TWAP resets at midnight.
    Uses minp=min(lookback, 20) to allow early-day values.
    """
    n = len(data)
    out = np.full(n, np.nan)
    if n == 0:
        return out

    start_arr, end_arr, n_days = find_day_boundaries_nb(index_ns)
    minp = min(lookback, 20)

    for d in range(n_days):
        s = start_arr[d]
        e = end_arr[d]
        day_len = e - s
        if day_len < minp:
            continue

        day_data = data[s:e]
        day_std = vbt.generic.nb.rolling_std_1d_nb(
            day_data, lookback, minp=minp, ddof=1
        )
        for i in range(day_len):
            out[s + i] = day_std[i]

    return out


@njit(nogil=True)
def compute_intraday_zscore_nb(
    index_ns: np.ndarray,
    data: np.ndarray,
    lookback: int,
) -> np.ndarray:
    """Rolling z-score that resets at each day boundary.

    Prevents spurious spikes when TWAP resets produce discontinuities
    in the deviation series across day boundaries.
    """
    n = len(data)
    out = np.full(n, np.nan)
    if n == 0:
        return out

    start_arr, end_arr, n_days = find_day_boundaries_nb(index_ns)
    minp = min(lookback, 20)

    for d in range(n_days):
        s = start_arr[d]
        e = end_arr[d]
        day_len = e - s
        if day_len < minp:
            continue

        day_data = data[s:e]
        day_zscore = vbt.generic.nb.rolling_zscore_1d_nb(
            day_data, lookback, minp=minp, ddof=1
        )
        for i in range(day_len):
            out[s + i] = day_zscore[i]

    return out


# ═══════════════════════════════════════════════════════════════════════
# SHARED MR INDICATOR HELPERS
# ═══════════════════════════════════════════════════════════════════════


@njit(nogil=True)
def compute_deviation_nb(
    close: np.ndarray,
    twap: np.ndarray,
) -> np.ndarray:
    """Compute close-to-TWAP deviation, NaN-safe."""
    n = len(close)
    deviation = np.empty(n)
    for i in range(n):
        if np.isnan(twap[i]) or np.isnan(close[i]):
            deviation[i] = np.nan
        else:
            deviation[i] = close[i] - twap[i]
    return deviation


@njit(nogil=True)
def compute_intraday_bands_nb(
    index_ns: np.ndarray,
    twap: np.ndarray,
    deviation: np.ndarray,
    lookback: int,
    band_width: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute upper/lower bands from intraday rolling std of deviation."""
    n = len(twap)
    rolling_std = compute_intraday_rolling_std_nb(index_ns, deviation, lookback)
    upper_band = np.full(n, np.nan)
    lower_band = np.full(n, np.nan)
    for i in range(n):
        s = rolling_std[i]
        if not np.isnan(s) and s > 1e-10 and not np.isnan(twap[i]):
            upper_band[i] = twap[i] + band_width * s
            lower_band[i] = twap[i] - band_width * s
    return upper_band, lower_band


@njit(nogil=True)
def compute_adx_regime_nb(
    index_ns: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    open_: np.ndarray,
    adx_period: int,
    adx_threshold: float,
) -> np.ndarray:
    """Compute ADX and return binary regime_ok array (1=MR allowed, 0=trending)."""
    n = len(close)
    adx = compute_daily_adx_broadcast_nb(index_ns, high, low, close, open_, adx_period)
    regime_ok = np.ones(n, dtype=np.float64)
    for i in range(n):
        if not np.isnan(adx[i]) and adx[i] > adx_threshold:
            regime_ok[i] = 0.0
    return regime_ok


@njit(nogil=True)
def compute_mr_base_indicators_nb(
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
    """Full MR base indicators: TWAP + zscore + bands + regime_ok."""
    twap = compute_intraday_twap_nb(index_ns, high, low, close)
    deviation = compute_deviation_nb(close, twap)
    zscore = compute_intraday_zscore_nb(index_ns, deviation, lookback)
    upper_band, lower_band = compute_intraday_bands_nb(
        index_ns, twap, deviation, lookback, band_width
    )
    regime_ok = compute_adx_regime_nb(
        index_ns, high, low, close, open_, adx_period, adx_threshold
    )
    return twap, zscore, upper_band, lower_band, regime_ok


# ═══════════════════════════════════════════════════════════════════════
# SHARED MR SIGNAL FUNCTION
# ═══════════════════════════════════════════════════════════════════════


@njit(nogil=True)
def mr_band_signal_nb(
    c: object,
    close_arr: np.ndarray,
    upper_arr: np.ndarray,
    lower_arr: np.ndarray,
    twap_arr: np.ndarray,
    regime_ok_arr: np.ndarray,
    index_ns_arr: np.ndarray,
    eod_hour_arr: np.ndarray,
    eod_minute_arr: np.ndarray,
    eval_freq_arr: np.ndarray,
) -> tuple[bool, bool, bool, bool]:
    """Shared MR signal: band entry, TWAP exit, EOD forced exit, regime filter."""
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
