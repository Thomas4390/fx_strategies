"""Minute aggregation and session clock for the QuantConnect x10 port.

Pure Python: no ``AlgorithmImports``, no pandas, no numpy. Everything here has a
line-by-line counterpart in the reference engine and is unit-tested against it
by ``tests/test_qc_x10_parity.py``.

Three conventions, each of which the reference implementation fixes and which a
QuantConnect port gets wrong by default:

1. **Left label, left close.** ``resample_ohlc`` (``framework/x10_context.py``)
   stamps a bar with the *start* of its bin: the bar labelled 08:00 covers
   ``[08:00, 08:05)``. LEAN stamps a minute bar with its ``end_time``, so every
   minute is shifted back by one before it is binned.
2. **Empty bins are dropped, never forward filled.** ``resample_ohlc`` ends on a
   ``dropna(subset=["open"])``; a bin with no minute simply does not exist, and
   the H1 alignment of §6.1 counts *existing* bars, not wall-clock hours.
3. **The reference bins on the naive New York clock** (``prepare_inputs`` calls
   ``to_session_clock`` before resampling). New York is a whole number of hours
   away from UTC in both DST states, and both 5 and 60 divide 60, so the M5 and
   H1 bins land on exactly the same instants whichever of the two clocks is used
   — binning on UTC minutes here is therefore not an approximation. The New York
   clock is still needed, but only for the two quantities that read a wall
   clock: the minute of day (§10) and the session id (§2).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import NamedTuple

try:  # QuantConnect ships tzdata; a bare container might not.
    from zoneinfo import ZoneInfo

    _NY = ZoneInfo("America/New_York")
except Exception:  # pragma: no cover - fallback used only on QC if zoneinfo fails
    import pytz

    _NY = pytz.timezone("America/New_York")

# Spec §2 — the session runs 18:00 (J-1) -> 17:00 (J) New York.
SESSION_CLOSE_HOUR = 17

_EPOCH_DATE = date(1970, 1, 1)
_EPOCH_DT = datetime(1970, 1, 1, tzinfo=timezone.utc)

M5_MINUTES = 5
H1_MINUTES = 60

__all__ = [
    "H1_MINUTES",
    "M5_MINUTES",
    "SESSION_CLOSE_HOUR",
    "Bar",
    "BarBuilder",
    "MinuteQuote",
    "minute_index",
    "minute_to_utc",
    "ny_fields",
]


# ═══════════════════════════════════════════════════════════════════════
# CLOCK
# ═══════════════════════════════════════════════════════════════════════


def minute_index(stamp: datetime) -> int:
    """Whole minutes between the UTC epoch and ``stamp``.

    ``stamp`` is the *end* of a LEAN minute bar; the caller subtracts one to get
    the left label the reference indexes by.
    """
    if stamp.tzinfo is not None:
        stamp = stamp.astimezone(timezone.utc).replace(tzinfo=None)
    return int((stamp - _EPOCH_DT.replace(tzinfo=None)).total_seconds()) // 60


def minute_to_utc(minute: int) -> datetime:
    """Inverse of ``minute_index``, as a naive UTC datetime."""
    return _EPOCH_DT.replace(tzinfo=None) + timedelta(minutes=minute)


def ny_fields(minute: int) -> tuple[int, int]:
    """``(minute_of_day, session_id)`` of a bar label, on the New York clock.

    ``minute_of_day`` is what §10 compares to the 16:30/18:15 window and to the
    16:55 flat. ``session_id`` reproduces ``x10_context.session_ids``: the
    interval is closed on the right, ``(17:00 of J-1, 17:00 of J]``, so 17:00
    still belongs to session J and the 18:00 reopen starts J+1. The nanosecond
    nudge of the pandas version becomes an explicit midnight test here — minute
    stamps cannot land anywhere else inside it.
    """
    local = _EPOCH_DT.replace(tzinfo=timezone.utc)
    local = (local + timedelta(minutes=minute)).astimezone(_NY).replace(tzinfo=None)
    shifted = local + timedelta(hours=24 - SESSION_CLOSE_HOUR)
    day = shifted.date()
    if shifted.hour == 0 and shifted.minute == 0:
        day = day - timedelta(days=1)
    return local.hour * 60 + local.minute, (day - _EPOCH_DATE).days


# ═══════════════════════════════════════════════════════════════════════
# BARS
# ═══════════════════════════════════════════════════════════════════════


class MinuteQuote(NamedTuple):
    """One LEAN ``QuoteBar``, flattened: mid for signals, bid/ask for fills.

    The mid is what the reference works on (the gold parquet is mid OHLC); the
    bid and the ask replace the ``mid ± s/2`` the reference synthesises from a
    constant spread, and are the trigger sides of §9 — bid for a long exit, ask
    for a short one.
    """

    mid_open: float
    mid_high: float
    mid_low: float
    mid_close: float
    bid_open: float
    bid_high: float
    bid_low: float
    bid_close: float
    ask_open: float
    ask_high: float
    ask_low: float
    ask_close: float


@dataclass
class Bar:
    """An aggregated OHLC bar, ``start`` being its left label in minutes."""

    start: int
    open: float
    high: float
    low: float
    close: float


class BarBuilder:
    """Accumulate minutes into left-labelled bins of ``period`` minutes.

    A native ``QuoteBarConsolidator`` would emit on its own schedule and stamp
    the result with the bin *end*; this accumulator is driven explicitly so the
    caller decides the order in which an M5 bar and an H1 bar that close on the
    same instant are handed to the automaton — the M5 decision first, the H1
    context second, which is what ``align_h1_to_m5`` means in streaming form.

    Three methods, to be called in this order for every minute:

    * ``roll(minute)`` closes a bin that a gap left open;
    * ``add(minute, ...)`` accumulates;
    * ``seal()`` closes the bin when the minute just added was its last.
    """

    __slots__ = (
        "period",
        "dropped",
        "_start",
        "_last",
        "_closed",
        "_live",
        "_o",
        "_h",
        "_l",
        "_c",
    )

    def __init__(self, period: int) -> None:
        self.period = period
        # Minutes refused because their bin had already been emitted. Must stay
        # at zero: a non-zero count means the session flush closed a bin the
        # feed had not finished, and the bars stop matching the reference.
        self.dropped = 0
        self._start = -1
        self._last = -1
        self._closed = -1
        self._live = False
        self._o = self._h = self._l = self._c = float("nan")

    def bin_of(self, minute: int) -> int:
        return (minute // self.period) * self.period

    def roll(self, minute: int) -> Bar | None:
        """Close the running bin when ``minute`` belongs to a later one."""
        if self._live and self.bin_of(minute) != self._start:
            return self._emit()
        return None

    def flush(self) -> Bar | None:
        """Close the running bin unconditionally, whatever its bin."""
        return self._emit() if self._live else None

    def add(self, minute: int, o: float, h: float, low: float, c: float) -> None:
        start = self.bin_of(minute)
        if not self._live:
            if start <= self._closed:
                self.dropped += 1
                return  # duplicate or out-of-order minute: the bin is already out
            self._start = start
            self._o, self._h, self._l, self._c = o, h, low, c
            self._live = True
        else:
            if h > self._h:
                self._h = h
            if low < self._l:
                self._l = low
            self._c = c
        self._last = minute

    def seal(self) -> Bar | None:
        """Close the running bin when its last minute has been fed."""
        if self._live and self._last + 1 >= self._start + self.period:
            return self._emit()
        return None

    def _emit(self) -> Bar:
        bar = Bar(self._start, self._o, self._h, self._l, self._c)
        self._live = False
        self._closed = self._start
        return bar
