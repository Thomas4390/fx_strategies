"""Streaming indicators of the x10 spec, recomputed by hand.

Pure Python: no ``AlgorithmImports``, no pandas, no numpy. Every class here is
the streaming twin of one function of ``framework/x10_kernels.py`` or
``framework/x10_context.py`` and is compared to it bar for bar by
``tests/test_qc_x10_parity.py``.

No native QuantConnect indicator is used, and §4 says why: ``AverageTrueRange``
seeds on the simple mean of the first 14 true ranges, the spec seeds on the
first true range alone, and a seeding gap does not decay — it shifts every
distance of the strategy, which are all expressed in ATR.

The one subtlety worth reading twice is ``Ema`` on a series that carries holes.
The reference computes the DXY EMA with ``ewm(..., adjust=False)`` and pandas'
default ``ignore_na=False``, which treats a missing observation as an elapsed
step: the next valid observation gets weight ``1 - (1-alpha)^(k+1)`` for ``k``
skipped bars, not ``alpha``. Skipping the gap instead — the obvious reading —
gives a different series as soon as one leg of the basket goes stale.
"""

from __future__ import annotations

from collections import deque
from math import exp, isfinite, log, sqrt

NAN = float("nan")

# §4, §5, §6 — frozen horizons.
ATR_PERIOD = 14
H_VELOCITY = 3
H_MOMENTUM = 12
VWAP_MIN_BARS = 12
EMA_SPAN_H1 = 50

# §3 — the broker grid: XAUUSD quotes to the mil, levels 10 000 points apart.
X10_POINTS_PER_UNIT = 1000
X10_POINTS_PER_LEVEL = 10_000
X10_LEVEL_SIZE = 10.0

# §6.3 — ICE weights renormalised to sum 1, and the ICE scaling constant. The
# arithmetic is written exactly as in ``x10_context`` so the floats are bit
# identical, not merely close.
_ICE_WEIGHTS = {"eurusd": 0.576, "usdjpy": 0.136, "gbpusd": 0.119, "usdcad": 0.091}
_ICE_TOTAL = sum(_ICE_WEIGHTS.values())
DXY4_WEIGHTS = {leg: w / _ICE_TOTAL for leg, w in _ICE_WEIGHTS.items()}
DXY4_SIGNS = {"eurusd": -1.0, "usdjpy": 1.0, "gbpusd": -1.0, "usdcad": 1.0}
DXY4_CONSTANT = 50.14348112
DXY4_FFILL_LIMIT = 5
DXY4_LEGS: tuple[str, ...] = ("eurusd", "usdjpy", "gbpusd", "usdcad")

__all__ = [
    "ATR_PERIOD",
    "DXY4_CONSTANT",
    "DXY4_FFILL_LIMIT",
    "DXY4_LEGS",
    "DXY4_SIGNS",
    "DXY4_WEIGHTS",
    "EMA_SPAN_H1",
    "H_MOMENTUM",
    "H_VELOCITY",
    "VWAP_MIN_BARS",
    "Dxy4Basket",
    "Ema",
    "Kinematics",
    "SessionTypicalMean",
    "WilderAtr",
    "x10_levels",
]


# ═══════════════════════════════════════════════════════════════════════
# §4 — ATR de Wilder
# ═══════════════════════════════════════════════════════════════════════


class WilderAtr:
    """Wilder ATR, ``alpha = 1/n``, seeded on the first true range — §4.

    ``TR[0]`` does not exist (it needs ``C[-1]``), so the first bar only records
    its close. The value stays undefined until ``n`` true ranges are in, and is
    NaN rather than 0 before that: a zero ATR makes every distance of the spec
    infinite in ATR units.
    """

    __slots__ = ("period", "value", "_prev_close", "_run", "_seen")

    def __init__(self, period: int = ATR_PERIOD) -> None:
        self.period = period
        self.value = NAN
        self._prev_close = NAN
        self._run = 0.0
        self._seen = 0

    def update(self, high: float, low: float, close: float) -> float:
        prev = self._prev_close
        self._prev_close = close
        if not isfinite(prev):
            self.value = NAN
            return NAN

        tr = high - low
        up = abs(high - prev)
        dn = abs(low - prev)
        if up > tr:
            tr = up
        if dn > tr:
            tr = dn

        self._seen += 1
        self._run = tr if self._seen == 1 else self._run + (tr - self._run) / self.period
        self.value = self._run if self._seen >= self.period else NAN
        return self.value


# ═══════════════════════════════════════════════════════════════════════
# §6.1 — EMA
# ═══════════════════════════════════════════════════════════════════════


class Ema:
    """``ewm(span=..., adjust=False, min_periods=...)``, holes included — §6.1.

    A NaN observation leaves the value untouched (pandas carries the previous
    mean forward) but still counts as an elapsed step, so the next valid
    observation is weighted ``1 - (1-alpha)^(gap+1)``. That is pandas'
    ``ignore_na=False``, which is the default the reference runs under.
    """

    __slots__ = ("alpha", "min_periods", "value", "_value", "_n", "_gap")

    def __init__(self, span: int, min_periods: int | None = None) -> None:
        self.alpha = 2.0 / (span + 1.0)
        self.min_periods = span if min_periods is None else min_periods
        self.value = NAN
        self._value = NAN
        self._n = 0
        self._gap = 0

    def update(self, x: float) -> float:
        if not isfinite(x):
            self._gap += 1
            return self.value

        self._n += 1
        if self._n == 1:
            self._value = x
        else:
            weight = 1.0 - (1.0 - self.alpha) ** (self._gap + 1)
            self._value = self._value + weight * (x - self._value)
        self._gap = 0
        self.value = self._value if self._n >= self.min_periods else NAN
        return self.value


# ═══════════════════════════════════════════════════════════════════════
# §5 — cinematique
# ═══════════════════════════════════════════════════════════════════════


class Kinematics:
    """Velocity, momentum and acceleration in ATR units — §5.

    ``v = (C[t]-C[t-3])/(3A)``, ``m = (C[t]-C[t-12])/(A*sqrt(12))`` and
    ``a = v[t]-v[t-3]``. All three are undefined while the ATR is, and ``a``
    warms up three bars after ``v`` because it needs ``v[t-3]``. NaN propagates
    and is never substituted by 0: §5 freezes every transition of the automaton
    on a bar where any of the three is undefined.
    """

    __slots__ = ("h_v", "h_m", "v", "m", "a", "_closes", "_vs", "_sqrt_h_m")

    def __init__(self, h_v: int = H_VELOCITY, h_m: int = H_MOMENTUM) -> None:
        self.h_v = h_v
        self.h_m = h_m
        self.v = self.m = self.a = NAN
        self._closes: deque[float] = deque(maxlen=h_m + 1)
        self._vs: deque[float] = deque(maxlen=h_v + 1)
        self._sqrt_h_m = sqrt(float(h_m))

    def update(self, close: float, atr: float) -> tuple[float, float, float]:
        self._closes.append(close)
        v = m = a = NAN
        if isfinite(atr) and atr > 0.0:
            if len(self._closes) > self.h_v:
                v = (close - self._closes[-1 - self.h_v]) / (self.h_v * atr)
            if len(self._closes) > self.h_m:
                m = (close - self._closes[-1 - self.h_m]) / (atr * self._sqrt_h_m)

        self._vs.append(v)
        if len(self._vs) > self.h_v:
            prev = self._vs[-1 - self.h_v]
            if isfinite(v) and isfinite(prev):
                a = v - prev

        self.v, self.m, self.a = v, m, a
        return v, m, a


# ═══════════════════════════════════════════════════════════════════════
# §6.2 — VWAP de seance
# ═══════════════════════════════════════════════════════════════════════


class SessionTypicalMean:
    """Unweighted cumulative session mean of ``(H+L+C)/3`` — §6.2.

    The mean includes the decision bar itself: it is fully known at the close of
    that bar, which is the decision instant, so no shift applies here. Reset
    whenever the session id changes, undefined before ``min_bars`` bars.
    """

    __slots__ = ("min_bars", "value", "_cum", "_count", "_session")

    def __init__(self, min_bars: int = VWAP_MIN_BARS) -> None:
        self.min_bars = min_bars
        self.value = NAN
        self._cum = 0.0
        self._count = 0
        self._session = None

    def update(self, high: float, low: float, close: float, session_id: int) -> float:
        if self._session is not None and session_id != self._session:
            self._cum = 0.0
            self._count = 0
        self._session = session_id

        self._cum += (high + low + close) / 3.0
        self._count += 1
        self.value = self._cum / self._count if self._count >= self.min_bars else NAN
        return self.value


# ═══════════════════════════════════════════════════════════════════════
# §3 — grille x10
# ═══════════════════════════════════════════════════════════════════════


def x10_levels(close: float) -> tuple[float, float]:
    """``(L_inf, L_sup)`` bracketing ``close`` — §3.

    Integer arithmetic on broker points (``digits=3``), so a close stored as
    4409.999999996 cannot fall on the wrong side of 4410. Python's ``round`` and
    NumPy's ``rint`` both round halves to even, which is what makes this the
    same function as ``x10_levels_nb``.
    """
    points = int(round(close * X10_POINTS_PER_UNIT))
    floored = (points // X10_POINTS_PER_LEVEL) * X10_POINTS_PER_LEVEL
    l_inf = floored / X10_POINTS_PER_UNIT
    return l_inf, l_inf + X10_LEVEL_SIZE


# ═══════════════════════════════════════════════════════════════════════
# §6.3 — panier DXY4
# ═══════════════════════════════════════════════════════════════════════


class Dxy4Basket:
    """Four-leg synthetic dollar, computed in logs — §6.3.

    One call per H1 bin of the union index the reference builds (a bin exists as
    soon as *one* leg produced an H1 bar). A leg missing from the call is
    forward filled for at most ``DXY4_FFILL_LIMIT`` consecutive bins, which is
    ``reindex(...).ffill(limit=5)``; past that the basket is **undefined =
    neutral**, never a stale value.
    """

    __slots__ = ("ffill_limit", "value", "_last", "_stale")

    def __init__(self, ffill_limit: int = DXY4_FFILL_LIMIT) -> None:
        self.ffill_limit = ffill_limit
        self.value = NAN
        self._last: dict[str, float] = {}
        self._stale: dict[str, int] = {leg: ffill_limit + 1 for leg in DXY4_LEGS}

    def update(self, closes: dict[str, float]) -> float:
        for leg in DXY4_LEGS:
            price = closes.get(leg)
            if price is not None and isfinite(price) and price > 0.0:
                self._last[leg] = price
                self._stale[leg] = 0
            else:
                self._stale[leg] += 1

        log_value = log(DXY4_CONSTANT)
        for leg in DXY4_LEGS:
            if self._stale[leg] > self.ffill_limit:
                self.value = NAN
                return NAN
            log_value += DXY4_SIGNS[leg] * DXY4_WEIGHTS[leg] * log(self._last[leg])

        self.value = exp(log_value)
        return self.value
