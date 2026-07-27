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

    # vbt 2026.6.27 a renomme `plotting.pre_show_func` en `plotting.pre_render_func`.
    # La config etant gelee, ecrire une cle absente leve un KeyError au lieu d'etre ignore.
    hook_key = (
        "pre_render_func"
        if "pre_render_func" in vbt.settings["plotting"]
        else "pre_show_func"
    )
    vbt.settings.set(f"plotting.{hook_key}", configure_figure_for_fullscreen)
    vbt.settings.returns.year_freq = pd.Timedelta(hours=24) * 252


# ═══════════════════════════════════════════════════════════════════════
# DATA LOADING (VBT Pro native)
# ═══════════════════════════════════════════════════════════════════════


def load_fx_data(
    path: str = "data/EUR-USD_minute.parquet",
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


# Gold (XAUUSD) — QuantConnect export, OANDA CFD.
# Differs from the FX parquets: timestamp lives in a tz-aware UTC index named
# "time" (not a "date" column), and there is no volume column.
GOLD_SYMBOL = "XAU-USD"
GOLD_DATA_PATH = "data/XAU-USD_minute_qc.parquet"

# The gold signal is anchored on US session clock times (10:00 / 15:30 / 16:00).
# Those boundaries move in UTC across DST, so the index is converted to New York
# time and then made naive, keeping the repo-wide "naive index" convention while
# letting `index.hour` mean exactly what the strategy means.
GOLD_TZ = "America/New_York"


def validate_ohlc_frame(
    df: pd.DataFrame,
    *,
    name: str,
    min_rows: int = 1000,
) -> None:
    """Fail fast on a malformed OHLC frame.

    Checks the invariants no other loader in this repo verifies: a strictly
    increasing index, no duplicate timestamps, no NaN, and OHLC coherence
    (high is the bar maximum, low the bar minimum, all prices strictly
    positive). Raises rather than silently repairing — a silent fix on price
    data produces a backtest that cannot be trusted.

    Parameters
    ----------
    df : pd.DataFrame
        Frame with lowercase open/high/low/close columns and a DatetimeIndex.
    name : str
        Dataset name, used in error messages.
    min_rows : int
        Minimum acceptable row count.

    Raises
    ------
    ValueError
        On any violated invariant, with a message naming the first offenders.
    """
    if len(df) < min_rows:
        raise ValueError(f"{name}: only {len(df)} rows, expected >= {min_rows}")

    missing = [c for c in ("open", "high", "low", "close") if c not in df.columns]
    if missing:
        raise ValueError(f"{name}: missing OHLC columns {missing}")

    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError(f"{name}: index is {type(df.index).__name__}, expected DatetimeIndex")

    n_dup = int(df.index.duplicated().sum())
    if n_dup:
        dups = df.index[df.index.duplicated()][:5].tolist()
        raise ValueError(f"{name}: {n_dup} duplicate timestamps, first {dups}")

    if not df.index.is_monotonic_increasing:
        raise ValueError(f"{name}: index is not monotonically increasing")

    ohlc = df[["open", "high", "low", "close"]]
    n_nan = int(ohlc.isna().to_numpy().sum())
    if n_nan:
        raise ValueError(f"{name}: {n_nan} NaN values in OHLC")

    if bool((ohlc <= 0).to_numpy().any()):
        raise ValueError(f"{name}: non-positive prices found")

    body_high = ohlc[["open", "close"]].max(axis=1)
    body_low = ohlc[["open", "close"]].min(axis=1)
    bad = (ohlc["high"] < body_high) | (ohlc["low"] > body_low) | (ohlc["high"] < ohlc["low"])
    n_bad = int(bad.sum())
    if n_bad:
        raise ValueError(
            f"{name}: {n_bad} bars violate OHLC coherence, first {df.index[bad][:5].tolist()}"
        )


def load_gold_data(
    path: str = GOLD_DATA_PATH,
    tz: str = GOLD_TZ,
    validate: bool = True,
) -> tuple[pd.DataFrame, vbt.Data]:
    """Load the XAUUSD minute parquet, indexed on naive New York time.

    Mirrors `load_fx_data` (raw lowercase frame for Numba kernels, capitalized
    `vbt.Data` for native VBT functions) but handles the two ways the gold
    export differs from the FX parquets: a tz-aware UTC index named "time"
    rather than a "date" column, and no volume column.

    Parameters
    ----------
    path : str
        Path to the gold parquet, relative to the project root or absolute.
    tz : str
        Target timezone. The index is converted to it and then made naive.
    validate : bool
        Run `validate_ohlc_frame` on the loaded frame.

    Returns
    -------
    raw : pd.DataFrame
        OHLCV with lowercase columns, naive index in `tz` (for Numba kernels).
    data : vbt.Data
        VBT Data wrapper with capitalized columns (for native VBT functions).
    """
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = _PROJECT_ROOT / path

    data_raw = vbt.Data.from_parquet(str(resolved))
    df = data_raw.data[data_raw.symbols[0]].sort_index()

    if df.index.tz is None:
        raise ValueError(
            f"{path}: index is tz-naive; expected tz-aware UTC. Its timezone would "
            "have to be guessed, and a wrong guess silently shifts every session boundary."
        )
    df.index = df.index.tz_convert(tz).tz_localize(None)

    raw = df.copy()
    raw.columns = [c.lower() for c in raw.columns]

    if validate:
        validate_ohlc_frame(raw, name=GOLD_SYMBOL)

    # No volume on the OANDA gold CFD; VWAP-style indicators need the column present.
    if "volume" not in raw.columns:
        raw["volume"] = 1.0

    df_cap = raw.copy()
    df_cap.columns = [c.capitalize() for c in df_cap.columns]
    data = vbt.Data.from_data({GOLD_SYMBOL: df_cap}, tz_localize=False, tz_convert=False)
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
def compute_mr_bands_nb(
    index_ns: np.ndarray,
    close: np.ndarray,
    vwap: np.ndarray,
    lookback: int,
    band_width: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Session-resetting Bollinger bands around pre-computed VWAP.

    Returns (zscore, upper_band, lower_band).
    """
    deviation = compute_deviation_nb(close, vwap)
    zscore = compute_intraday_zscore_nb(index_ns, deviation, lookback)
    upper, lower = compute_intraday_bands_nb(
        index_ns, vwap, deviation, lookback, band_width
    )
    return zscore, upper, lower


# ═══════════════════════════════════════════════════════════════════════
# NATIVE VBT PREPARE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════


def prepare_mr(
    raw: pd.DataFrame,
    data: vbt.Data | None,
    adx_period: int = 14,
    adx_threshold: float = 30.0,
) -> dict[str, np.ndarray]:
    """Pre-compute VWAP (native) and ADX regime filter (native talib).

    Works with both full vbt.Data and raw-only (CV splits where data=None).
    """
    # Native VWAP — session-anchored, resets daily
    vwap_ind = vbt.VWAP.run(
        raw["high"], raw["low"], raw["close"], raw["volume"], anchor="D"
    )
    vwap = vwap_ind.vwap.values

    # Native ADX on daily timeframe with anti-look-ahead realignment.
    # Always build a temporary Data from raw to guarantee shape alignment
    # (data may span the full dataset while raw is a CV/holdout slice).
    df_cap = raw.copy()
    df_cap.columns = [c.capitalize() for c in df_cap.columns]
    temp_data = vbt.Data.from_data(
        {"tmp": df_cap}, tz_localize=False, tz_convert=False
    )
    adx_result = temp_data.run("talib:ADX", timeframe="1D", timeperiod=adx_period)
    adx_values = adx_result.real.values

    # Regime filter: 1.0 = MR allowed (low ADX), 0.0 = trending
    regime_ok = np.where(
        np.isnan(adx_values) | (adx_values < adx_threshold), 1.0, 0.0
    )

    return {"vwap": vwap, "regime_ok": regime_ok}


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


# ═══════════════════════════════════════════════════════════════════════
# BROKER (METATRADER 5) DATA LOADING
# ═══════════════════════════════════════════════════════════════════════

# One parquet per pair and per period, exported from the broker terminal.
# Shares the gold export's shape — tz-aware UTC index named "time" — and adds
# three broker columns QuantConnect does not carry: tick_volume (number of
# price changes, the only volume a retail FX feed has), spread and real_volume.
MT5_DATA_TEMPLATE = "data/{pair}_{period}_mt5.parquet"


def load_mt5_daily(
    pair: str,
    period: str = "daily",
    tz: str = GOLD_TZ,
    validate: bool = True,
) -> tuple[pd.DataFrame, vbt.Data]:
    """Load a broker MT5 parquet, indexed on naive New York time.

    Mirrors `load_gold_data` (raw lowercase frame for Numba kernels,
    capitalized `vbt.Data` for native VBT functions) so that a sleeve written
    against the gold export runs unchanged on a broker export: same index
    convention, same column casing, same fail-fast validation.

    Parameters
    ----------
    pair : str
        Pair as it appears in the file name, e.g. "EUR-USD".
    period : str
        Bar period, e.g. "daily" or "minute".
    tz : str
        Target timezone. The index is converted to it and then made naive.
    validate : bool
        Run `validate_ohlc_frame` on the loaded frame.

    Returns
    -------
    raw : pd.DataFrame
        OHLCV with lowercase columns, naive index in `tz` (for Numba kernels).
        Keeps the broker's `tick_volume` and `spread` columns.
    data : vbt.Data
        VBT Data wrapper with capitalized columns (for native VBT functions).
    """
    path = MT5_DATA_TEMPLATE.format(pair=pair, period=period)
    resolved = _PROJECT_ROOT / path

    data_raw = vbt.Data.from_parquet(str(resolved))
    df = data_raw.data[data_raw.symbols[0]].sort_index()

    if df.index.tz is None:
        raise ValueError(
            f"{path}: index is tz-naive; expected tz-aware UTC. Its timezone would "
            "have to be guessed, and a wrong guess silently shifts every session boundary."
        )
    df.index = df.index.tz_convert(tz).tz_localize(None)

    raw = df.copy()
    raw.columns = [c.lower() for c in raw.columns]

    if validate:
        validate_ohlc_frame(raw, name=f"{pair}_mt5")

    # Retail FX has no traded volume; tick_volume is the broker's proxy and is
    # mirrored into `volume` so VWAP-style indicators find the column they
    # expect, without dropping the original name.
    if "tick_volume" in raw.columns:
        raw["volume"] = raw["tick_volume"].astype(float)
    elif "volume" not in raw.columns:
        raw["volume"] = 1.0

    df_cap = raw.copy()
    df_cap.columns = [c.capitalize() for c in df_cap.columns]
    data = vbt.Data.from_data({pair: df_cap}, tz_localize=False, tz_convert=False)
    return raw, data


def list_mt5_pairs(period: str = "daily") -> list[str]:
    """Pairs with a broker parquet for `period`, sorted, e.g. ``["AUD-USD", ...]``.

    Reads the data directory rather than a hardcoded list: adding a pair to the
    export is then enough to make it visible to the callers.
    """
    suffix = f"_{period}_mt5.parquet"
    return sorted(
        p.name[: -len(suffix)] for p in (_PROJECT_ROOT / "data").glob(f"*{suffix}")
    )


# ═══════════════════════════════════════════════════════════════════════
# SCREENING DATA LOADING (LONG DAILY HISTORIES)
# ═══════════════════════════════════════════════════════════════════════

# One parquet per screened instrument, downloaded for the multi-instrument
# screen: the broker exports start in 2020-2022, which is too short to judge an
# edge, while these go back to 1990-2000. Same shape as the gold export — a
# tz-aware UTC index named "time" — with OHLC only, no volume of any kind.
SCREENING_DATA_TEMPLATE = "data/{name}_daily_yahoo.parquet"


def load_screening_daily(
    name: str,
    tz: str = GOLD_TZ,
    validate: bool = True,
) -> tuple[pd.DataFrame, vbt.Data]:
    """Load a long daily screening parquet, indexed on naive New York time.

    Mirrors `load_mt5_daily` (raw lowercase frame for Numba kernels,
    capitalized `vbt.Data` for native VBT functions) so a sleeve written
    against the broker exports runs unchanged on the long history.

    Parameters
    ----------
    name : str
        Instrument as it appears in the file name, e.g. "US500" or "XAG-USD".
    tz : str
        Target timezone. The index is converted to it and then made naive.
    validate : bool
        Run `validate_ohlc_frame` on the loaded frame.

    Returns
    -------
    raw : pd.DataFrame
        OHLCV with lowercase columns, naive index in `tz` (for Numba kernels).
    data : vbt.Data
        VBT Data wrapper with capitalized columns (for native VBT functions).
    """
    path = SCREENING_DATA_TEMPLATE.format(name=name)
    resolved = _PROJECT_ROOT / path

    data_raw = vbt.Data.from_parquet(str(resolved))
    df = data_raw.data[data_raw.symbols[0]].sort_index()

    if df.index.tz is None:
        raise ValueError(
            f"{path}: index is tz-naive; expected tz-aware UTC. Its timezone would "
            "have to be guessed, and a wrong guess silently shifts every session boundary."
        )
    df.index = df.index.tz_convert(tz).tz_localize(None)

    raw = df.copy()
    raw.columns = [c.lower() for c in raw.columns]

    if validate:
        validate_ohlc_frame(raw, name=f"{name}_yahoo", min_rows=1000)

    # No volume in these exports; VWAP-style indicators need the column present.
    if "volume" not in raw.columns:
        raw["volume"] = 1.0

    df_cap = raw.copy()
    df_cap.columns = [c.capitalize() for c in df_cap.columns]
    data = vbt.Data.from_data({name: df_cap}, tz_localize=False, tz_convert=False)
    return raw, data
