"""Causal kernels for the XAUUSD x10 strategy (``docs/specs/xau_x10_spec.md``).

Four quantities, each read at the close of an M5 bar and never one tick later:
the x10 grid (§3), the Wilder ATR that normalises every distance in the spec
(§4), the kinematics triplet v/m/a (§5) and the session mean of the typical
price the spec calls VWAP (§6.2).

They are kept in plain Numba loops, free of any pandas or engine API, because
the same arithmetic has to be reproduced line by line in QuantConnect Python and
in MQL5: a vectorised pandas trick here would have no counterpart there, and the
three-way reconciliation of §13 would fail on rung 2 with nothing to point at.

Every function is a left-to-right scan: the value written at index ``i`` reads
only ``[0, i]``. That is the property ``tests/test_x10_kernels.py`` enforces by
truncating the input and recomputing.
"""

from __future__ import annotations

import numpy as np
from numba import njit

# Broker grid: XAUUSD quotes to the mil (`digits=3`, `point=0.001`), and the x10
# levels are 10 000 points apart. Integer arithmetic on points is what keeps a
# price stored as 4409.999999996 on the correct side of 4410 (spec §3).
X10_POINTS_PER_UNIT = 1000
X10_POINTS_PER_LEVEL = 10_000
X10_LEVEL_SIZE = 10.0

ATR_PERIOD = 14
H_VELOCITY = 3
H_MOMENTUM = 12
VWAP_MIN_BARS = 12


@njit(nogil=True, cache=True)
def x10_levels_nb(close: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Bracketing x10 levels of each close — spec §3.

    ``L_inf = floor(round(C*1000) / 10000) * 10`` and ``L_sup = L_inf + 10``,
    computed on integer points. A price sitting exactly on a level opens the box
    above it (``L_inf == C``), so the opposable property ``L_inf <= C < L_sup``
    holds without a special case.
    """
    n = len(close)
    l_inf = np.empty(n, dtype=np.float64)
    l_sup = np.empty(n, dtype=np.float64)
    for i in range(n):
        points = np.int64(np.rint(close[i] * X10_POINTS_PER_UNIT))
        floored = (points // X10_POINTS_PER_LEVEL) * X10_POINTS_PER_LEVEL
        l_inf[i] = floored / X10_POINTS_PER_UNIT
        l_sup[i] = l_inf[i] + X10_LEVEL_SIZE
    return l_inf, l_sup


@njit(nogil=True, cache=True)
def atr_wilder_nb(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    n: int = ATR_PERIOD,
) -> np.ndarray:
    """Wilder ATR (``alpha = 1/n``, recursive) — spec §4.

    ``TR[0]`` is undefined (it needs ``C[-1]``), so the recursion is seeded on
    ``TR[1]`` and the output stays NaN until ``n`` true ranges have been seen:
    the first defined value lands on bar ``n``. NaN, never 0 — a zero ATR would
    make every distance of the spec infinite in ATR units.

    Equivalent to ``tr.ewm(alpha=1/n, adjust=False, min_periods=n).mean()`` with
    ``tr[0] = NaN``, which is the reference the unit test recomputes.

    Seeding, which §4 does not pin down: the recursion starts on ``TR[1]``, not
    on the simple mean of the first 14 true ranges. MQL5's built-in ``iATR``
    seeds on that simple mean, so the MT5 port must reproduce *this* seed by
    hand or rung 2 of the §13 ladder will show a decaying, warmup-only gap.
    """
    size = len(close)
    atr = np.full(size, np.nan)
    if size == 0 or n < 1:
        return atr

    alpha = 1.0 / n
    run = 0.0
    seen = 0
    for i in range(1, size):
        prev_close = close[i - 1]
        tr = high[i] - low[i]
        up = abs(high[i] - prev_close)
        dn = abs(low[i] - prev_close)
        if up > tr:
            tr = up
        if dn > tr:
            tr = dn

        seen += 1
        run = tr if seen == 1 else run + alpha * (tr - run)
        if seen >= n:
            atr[i] = run
    return atr


@njit(nogil=True, cache=True)
def kinematics_nb(
    close: np.ndarray,
    atr: np.ndarray,
    h_v: int = H_VELOCITY,
    h_m: int = H_MOMENTUM,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Velocity, momentum and acceleration in ATR units — spec §5.

    ``v[t] = (C[t]-C[t-3]) / (3*A[t])``, ``m[t] = (C[t]-C[t-12]) / (A[t]*sqrt(12))``
    and ``a[t] = v[t] - v[t-3]``. The three are undefined while ``A[t]`` is, or
    while the lookback is not covered; ``a`` additionally needs ``v[t-3]``, so it
    warms up three bars after ``v``.

    ``atr`` is passed in rather than recomputed: the decision bar's ATR is the
    single unit of the whole spec, and recomputing it here would be a second
    definition waiting to drift from the first.
    """
    size = len(close)
    v = np.full(size, np.nan)
    m = np.full(size, np.nan)
    a = np.full(size, np.nan)
    if size == 0:
        return v, m, a

    sqrt_h_m = np.sqrt(np.float64(h_m))
    for i in range(size):
        atr_i = atr[i]
        if not np.isfinite(atr_i) or atr_i <= 0.0:
            continue
        if i >= h_v:
            v[i] = (close[i] - close[i - h_v]) / (h_v * atr_i)
        if i >= h_m:
            m[i] = (close[i] - close[i - h_m]) / (atr_i * sqrt_h_m)

    for i in range(h_v, size):
        if np.isfinite(v[i]) and np.isfinite(v[i - h_v]):
            a[i] = v[i] - v[i - h_v]
    return v, m, a


@njit(nogil=True, cache=True)
def session_mean_typical_nb(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    session_id: np.ndarray,
    min_bars: int = VWAP_MIN_BARS,
) -> np.ndarray:
    """Unweighted cumulative session mean of ``(H+L+C)/3`` — spec §6.2.

    The mean includes bar ``t`` itself: it is known at the close of ``t``, which
    is the decision instant (§2), so shifting it would falsify the quantity
    rather than protect anything. Undefined before ``min_bars`` bars of session,
    and reset whenever ``session_id`` changes — the caller owns the 18:00 New
    York boundary, which is a wall-clock question pandas answers better than a
    Numba loop (DST).
    """
    size = len(close)
    out = np.full(size, np.nan)
    if size == 0:
        return out

    cum = 0.0
    count = 0
    current = session_id[0]
    for i in range(size):
        if session_id[i] != current:
            current = session_id[i]
            cum = 0.0
            count = 0
        cum += (high[i] + low[i] + close[i]) / 3.0
        count += 1
        if count >= min_bars:
            out[i] = cum / count
    return out
