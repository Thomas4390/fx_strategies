"""Pandas assembly of the XAUUSD x10 context (``docs/specs/xau_x10_spec.md`` §6).

The kernels of ``x10_kernels`` do arithmetic; this module does calendars. It
owns the four places where a wall clock, a resampling boundary or a reindex can
silently move information backwards in time:

1. **session** (§2) — 18:00 to 17:00 New York, DST included;
2. **aggregation** M1 -> M5 / H1 (``label="left"``, ``closed="left"``), a bar
   existing only if at least one M1 fell into it;
3. **H1 -> M5 alignment** (§6.1) — ``shift(1)`` *then* ``reindex(...).ffill()``,
   the order being the whole point: without the shift, the 08:59 close is served
   to the 08:00 decision. That exact defect turned a Sharpe of 3.10 into -1.48
   on the EURUSD BB-MR study (``docs/research/eur-usd-bb-mr-research-plan.md``);
4. **session VWAP** (§6.2) — a cumulative mean *including* bar ``t``, which the
   rule above must NOT be applied to.

The pandas idiom is preferred over ``vbt.Resampler.realign_closing`` here: the
same three lines have to be portable to QuantConnect and MQL5, and an engine
helper has no counterpart there.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from framework.x10_kernels import (
    ATR_PERIOD,
    VWAP_MIN_BARS,
    atr_wilder_nb,
    session_mean_typical_nb,
    x10_levels_nb,
)

# ═══════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════

# Spec §2. The session closes at 17:00 and reopens at 18:00 New York; the hour
# in between carries no gold bar, so a single close hour describes the boundary.
SESSION_TZ = "America/New_York"
SESSION_CLOSE_HOUR = 17

EMA_SPAN_H1 = 50  # alpha = 2/51, spec §6.1

# Spec §6.3 — ICE weights renormalised to sum 1, and the ICE scaling constant.
_ICE_WEIGHTS = {"eurusd": 0.576, "usdjpy": 0.136, "gbpusd": 0.119, "usdcad": 0.091}
_ICE_TOTAL = sum(_ICE_WEIGHTS.values())
DXY4_WEIGHTS = {leg: w / _ICE_TOTAL for leg, w in _ICE_WEIGHTS.items()}
# Sign of each log leg: +1 when the pair is quoted USD-base (USDJPY, USDCAD).
DXY4_SIGNS = {"eurusd": -1.0, "usdjpy": 1.0, "gbpusd": -1.0, "usdcad": 1.0}
DXY4_CONSTANT = 50.14348112
DXY4_FFILL_LIMIT = 5  # bars, §6.3: beyond that the basket is undefined = neutral

_EPOCH = pd.Timestamp("1970-01-01")

__all__ = [
    "ATR_PERIOD",
    "DXY4_CONSTANT",
    "DXY4_FFILL_LIMIT",
    "DXY4_SIGNS",
    "DXY4_WEIGHTS",
    "EMA_SPAN_H1",
    "SESSION_CLOSE_HOUR",
    "SESSION_TZ",
    "VWAP_MIN_BARS",
    "align_h1_to_m5",
    "atr_wilder",
    "context_scores",
    "dxy4",
    "ema_h1",
    "resample_ohlc",
    "session_dates",
    "session_ids",
    "session_mean_typical",
    "to_session_clock",
    "x10_levels",
]


# ═══════════════════════════════════════════════════════════════════════
# SESSION CLOCK (§2)
# ═══════════════════════════════════════════════════════════════════════


def to_session_clock(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Naive New York view of ``index``.

    A tz-aware index (the raw gold export is UTC) is converted and made naive; a
    naive index is assumed to already be New York, which is what
    ``utils.load_gold_data`` produces and what the FX minute parquets store.
    """
    if index.tz is not None:
        return index.tz_convert(SESSION_TZ).tz_localize(None)
    return index


def session_dates(
    index: pd.DatetimeIndex,
    session_close_hour: int = SESSION_CLOSE_HOUR,
) -> pd.DatetimeIndex:
    """Session date of each stamp, under a 17:00 New York close — spec §2.

    Same convention, same nanosecond nudge, as
    ``strategies.gold_momentum.session_dates``: the interval is closed on the
    right, ``(17:00 of J-1, 17:00 of J]``, so 17:00 closes session J and the
    18:00 reopen starts J+1. A unit test asserts the two agree bar for bar.

    It is restated here rather than imported because ``framework`` must not
    depend on ``strategies`` — that import direction is the one the package
    layout forbids — and because this variant additionally accepts a tz-aware
    index, which the gold and FX loaders both produce at some stage.
    """
    local = to_session_clock(index)
    offset = pd.Timedelta(hours=24 - session_close_hour) - pd.Timedelta(nanoseconds=1)
    return (local + offset).normalize()


def session_ids(
    index: pd.DatetimeIndex,
    session_close_hour: int = SESSION_CLOSE_HOUR,
) -> np.ndarray:
    """Session labels as int64 days since epoch, for the Numba kernels."""
    dates = session_dates(index, session_close_hour)
    return (dates - _EPOCH).days.to_numpy().astype(np.int64)


# ═══════════════════════════════════════════════════════════════════════
# AGGREGATION M1 -> M5 / H1
# ═══════════════════════════════════════════════════════════════════════


def resample_ohlc(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    """Aggregate OHLC bars, left-labelled and left-closed.

    ``label="left", closed="left"`` means the bar stamped 08:00 covers
    ``[08:00, 08:05)`` and is complete at 08:05 — the convention the H1
    alignment of §6.1 relies on. Bins containing no source bar are dropped
    rather than forward-filled: an invented bar during a market halt would arm a
    level on a price nobody could trade.
    """
    agg = {"open": "first", "high": "max", "low": "min", "close": "last"}
    out = df[list(agg)].resample(freq, label="left", closed="left").agg(agg)
    return out.dropna(subset=["open"])


# ═══════════════════════════════════════════════════════════════════════
# H1 CONTEXT (§6.1)
# ═══════════════════════════════════════════════════════════════════════


def align_h1_to_m5(h1_series: pd.Series, m5_index: pd.DatetimeIndex) -> pd.Series:
    """Serve an H1 series to M5 bars, one hour late by construction — §6.1.

    ``shift(1)`` first, ``reindex(...).ffill()`` second. An H1 bar is only
    visible once closed, so the M5 bars of hour ``h`` see the bar of ``h-1`` and
    nothing newer. Applies to EMA50_H1, ATR_H1 and DXY — and to nothing else:
    the session VWAP is *not* an H1 series and must not go through here.
    """
    return h1_series.shift(1).reindex(m5_index, method="ffill")


def ema_h1(close_h1: pd.Series, span: int = EMA_SPAN_H1) -> pd.Series:
    """EMA of the H1 closes, ``alpha = 2/(span+1)``, undefined before ``span``."""
    return close_h1.ewm(span=span, adjust=False, min_periods=span).mean()


def atr_wilder(ohlc: pd.DataFrame, n: int = ATR_PERIOD) -> pd.Series:
    """Wilder ATR on any timeframe — the M5 unit of §4, the H1 one of §6.1."""
    values = atr_wilder_nb(
        ohlc["high"].to_numpy(dtype=np.float64),
        ohlc["low"].to_numpy(dtype=np.float64),
        ohlc["close"].to_numpy(dtype=np.float64),
        n,
    )
    return pd.Series(values, index=ohlc.index, name="atr")


# ═══════════════════════════════════════════════════════════════════════
# SESSION CONTEXT (§6.2) AND LEVELS (§3)
# ═══════════════════════════════════════════════════════════════════════


def session_mean_typical(
    m5: pd.DataFrame,
    min_bars: int = VWAP_MIN_BARS,
) -> pd.Series:
    """Session VWAP of the spec: unweighted cumulative mean of ``(H+L+C)/3``.

    Any ``volume`` column present in ``m5`` is ignored — gold carries none on QC
    (§1), and weighting it in one engine only would make the three incomparable.
    """
    values = session_mean_typical_nb(
        m5["high"].to_numpy(dtype=np.float64),
        m5["low"].to_numpy(dtype=np.float64),
        m5["close"].to_numpy(dtype=np.float64),
        session_ids(m5.index),
        min_bars,
    )
    return pd.Series(values, index=m5.index, name="vwap")


def x10_levels(close: pd.Series) -> tuple[pd.Series, pd.Series]:
    """``(L_inf, L_sup)`` bracketing each close — spec §3."""
    l_inf, l_sup = x10_levels_nb(close.to_numpy(dtype=np.float64))
    return (
        pd.Series(l_inf, index=close.index, name="l_inf"),
        pd.Series(l_sup, index=close.index, name="l_sup"),
    )


# ═══════════════════════════════════════════════════════════════════════
# DXY4 (§6.3)
# ═══════════════════════════════════════════════════════════════════════


def dxy4(
    eurusd: pd.Series,
    usdjpy: pd.Series,
    gbpusd: pd.Series,
    usdcad: pd.Series,
    ffill_limit: int = DXY4_FFILL_LIMIT,
) -> pd.Series:
    """Four-leg synthetic dollar basket — spec §6.3.

    ``50.14348112 * EURUSD^-w_e * USDJPY^w_j * GBPUSD^-w_g * USDCAD^w_c`` with
    the ICE weights renormalised to 1, computed in logs so the four exponents
    are a single dot product.

    The legs are aligned on the union of their indices and each is forward
    filled by at most ``ffill_limit`` bars. Beyond that the basket is NaN =
    **undefined = neutral** (``ctx_dxy = 0``), never a stale value: a frozen leg
    would keep scoring the dollar on a quote that no longer exists, and
    multi-symbol history is not guaranteed in the MT5 tester.
    """
    legs = {"eurusd": eurusd, "usdjpy": usdjpy, "gbpusd": gbpusd, "usdcad": usdcad}
    index = legs["eurusd"].index
    for series in legs.values():
        index = index.union(series.index)

    log_value = pd.Series(np.log(DXY4_CONSTANT), index=index)
    for leg, series in legs.items():
        filled = series.reindex(index).ffill(limit=ffill_limit)
        log_value = log_value + DXY4_SIGNS[leg] * DXY4_WEIGHTS[leg] * np.log(filled)
    return np.exp(log_value).rename("dxy4")


# ═══════════════════════════════════════════════════════════════════════
# CONTEXT SCORES (§6.4)
# ═══════════════════════════════════════════════════════════════════════


def _sign_score(values) -> np.ndarray:
    """``signe()`` of §6.4, with ``signe(0) = 0`` and NaN mapped to 0.

    An undefined quantity scores 0 — neutral — rather than propagating NaN: the
    VWAP before its 12-bar warmup and the DXY with a stale leg are both
    explicitly neutral in the spec, and a NaN here would silently poison the
    trace instead.
    """
    arr = np.asarray(values, dtype=np.float64)
    out = np.zeros(arr.shape, dtype=np.int8)
    out[arr > 0.0] = 1
    out[arr < 0.0] = -1
    return out


def context_scores(
    q,
    close,
    ema50_h1,
    vwap,
    dxy,
    dxy_ema50_h1,
):
    """The ``(ctx_ema, ctx_vwap, ctx_dxy)`` triplet of §6.4.

    ``q`` is the direction of the **trade**, not the approach direction ``d``:
    for a breakout they coincide, for a reversal they are opposite, and a
    "favourable" context must mean favourable to the position actually taken.

    ``ctx_dxy`` carries the extra minus of the spec — a dollar above its own
    EMA50 is adverse to a long gold trade. Returned as -1/0/+1 integers, with
    the index of ``close`` kept when it is a Series.
    """
    q_arr = np.asarray(q, dtype=np.float64)
    close_arr = np.asarray(close, dtype=np.float64)

    ctx_ema = _sign_score(q_arr * (close_arr - np.asarray(ema50_h1, dtype=np.float64)))
    ctx_vwap = _sign_score(q_arr * (close_arr - np.asarray(vwap, dtype=np.float64)))
    ctx_dxy = _sign_score(
        -q_arr
        * (np.asarray(dxy, dtype=np.float64) - np.asarray(dxy_ema50_h1, dtype=np.float64))
    )

    if isinstance(close, pd.Series):
        return (
            pd.Series(ctx_ema, index=close.index, name="ctx_ema"),
            pd.Series(ctx_vwap, index=close.index, name="ctx_vwap"),
            pd.Series(ctx_dxy, index=close.index, name="ctx_dxy"),
        )
    return ctx_ema, ctx_vwap, ctx_dxy
