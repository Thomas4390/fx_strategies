"""Tests for the x10 automaton and its intrabar execution (spec §7-§11).

Every series here is written by hand, bar by bar, and every expected number is
recomputed in the test from the formula of the spec rather than copied from a
run. Two reasons, both structural:

* this engine is the **reference** implementation — an MQL5 expert and a
  QuantConnect algorithm will be written against it, then reconciled event by
  event (§13) — so an expectation captured from the code itself would simply
  propagate whatever the code does into the two ports;
* the indicators are injected (constant or scripted ``atr``/``v``/``m``/``a``),
  which isolates the state machine from the kernels of ``x10_kernels``: a test
  that fails here fails on a transition, never on a warmup.

``_events_str`` renders one line per event and is what the four nominal
scenarios assert against; it is also the artefact quoted in the delivery report.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from framework.x10_engine import (
    CANCEL_REASON_NAMES,
    CONTRACT_SIZE,
    EVENT_COLUMNS,
    EVENT_NAMES,
    EXIT_REASON_NAMES,
    SCENARIO_NAMES,
    TRADE_COLUMNS,
    run_engine,
)
from framework.x10_kernels import atr_wilder_nb, kinematics_nb, x10_levels_nb

# ═══════════════════════════════════════════════════════════════════════
# SCENARIO BUILDER
# ═══════════════════════════════════════════════════════════════════════


def _m1_of(bar: tuple[float, float, float, float]) -> list[tuple[float, ...]]:
    """Five M1 bars aggregating exactly to one M5 bar, high before low.

    The path is ``open -> high -> low -> close``. It is deterministic and
    pessimistic-neutral: a test that needs the low first, or a single M1 bar
    carrying both extremes, passes an explicit override.
    """
    o, h, low, c = bar
    return [(o, o, o, o), (o, h, o, h), (h, h, low, low), (low, low, low, low), (low, c, low, c)]


def _as_array(value, n: int) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    return np.full(n, float(value)) if arr.ndim == 0 else arr.astype(np.float64)


def build_case(
    bars: list[tuple[float, float, float, float]],
    *,
    atr=1.0,
    v=0.0,
    m=0.0,
    a=0.0,
    ema=1990.0,
    vwap=1995.0,
    atr_h1=10.0,
    dxy=100.0,
    dxy_ema=100.0,
    spread=0.0,
    minute0=600,
    session=0,
    m1: dict[int, list[tuple[float, float, float, float]]] | None = None,
    gap_before: dict[int, int] | None = None,
) -> dict:
    """Kernel inputs from hand-written M5 bars.

    ``bars`` are ``(open, high, low, close)`` M5 bars; ``m1`` overrides the M1
    resolution of individual bars. Indicators are scalars (broadcast) or
    per-bar sequences. The levels come from the real ``x10_levels_nb`` — the
    grid of §3 is never worth faking — and the wall clock advances five minutes
    per bar, modulo a day, so the 16:30/16:55/18:15 rules of §10 can be walked
    into by choosing ``minute0``.

    ``gap_before={i: 10}`` makes bar ``i`` start ten minutes after bar ``i-1``
    instead of five — a data hole inside a session, which §2 forbids filling
    across.
    """
    n = len(bars)
    m1 = m1 or {}
    rows: list[tuple[float, ...]] = []
    starts = np.empty(n, dtype=np.int64)
    ends = np.empty(n, dtype=np.int64)
    for i, bar in enumerate(bars):
        sub = m1.get(i) or _m1_of(bar)
        starts[i] = len(rows)
        rows.extend(sub)
        ends[i] = len(rows)
    m1_arr = np.asarray(rows, dtype=np.float64)

    m5 = np.asarray(bars, dtype=np.float64)
    close = m5[:, 3].copy()
    l_inf, l_sup = x10_levels_nb(close)

    gap_before = gap_before or {}
    steps = np.array([0] + [gap_before.get(i, 5) for i in range(1, n)], dtype=np.int64)
    offsets = np.cumsum(steps)
    minutes = (minute0 + offsets) % 1440

    return dict(
        m5_open=m5[:, 0].copy(),
        m5_high=m5[:, 1].copy(),
        m5_low=m5[:, 2].copy(),
        m5_close=close,
        atr=_as_array(atr, n),
        v=_as_array(v, n),
        m=_as_array(m, n),
        a=_as_array(a, n),
        l_inf=l_inf,
        l_sup=l_sup,
        vwap=_as_array(vwap, n),
        ema50_h1=_as_array(ema, n),
        atr_h1=_as_array(atr_h1, n),
        dxy=_as_array(dxy, n),
        dxy_ema50_h1=_as_array(dxy_ema, n),
        minute_of_day=minutes.astype(np.float64),
        session_id=_as_array(session, n).astype(np.int64),
        bar_minute=(minute0 + offsets).astype(np.int64),
        spread=_as_array(spread, n),
        m1_open=m1_arr[:, 0].copy(),
        m1_high=m1_arr[:, 1].copy(),
        m1_low=m1_arr[:, 2].copy(),
        m1_close=m1_arr[:, 3].copy(),
        m1_start=starts,
        m1_end=ends,
    )


def run_case(case: dict, **params) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the engine and return the two registries as named frames."""
    events, trades = run_engine(**case, **params)
    return (
        pd.DataFrame(events, columns=list(EVENT_COLUMNS)),
        pd.DataFrame(trades, columns=list(TRADE_COLUMNS)),
    )


def _events_str(events: pd.DataFrame) -> list[str]:
    """One line per event: ``bar EVENT[(reason)] SCENARIO``."""
    out = []
    for row in events.itertuples():
        label = EVENT_NAMES[int(row.event)]
        if label == "CANCEL":
            label += f"({CANCEL_REASON_NAMES[int(row.cancel_reason)]})"
        elif label == "EXIT":
            label += f"({EXIT_REASON_NAMES[int(row.exit_reason)]})"
        scn = SCENARIO_NAMES[int(row.scenario)]
        out.append(f"{int(row.i_decision)} {label} {scn}".rstrip())
    return out


# Bars 0-4 of a Breakout Long on L = 2000: arm, break, then the three bars of
# the hold window (§7.2), the third of which carries a retest wick.
BASE_LONG: list[tuple[float, float, float, float]] = [
    (1998.4, 1999.6, 1998.3, 1999.5),  # 0  ARM  (gap 0.5 <= z*A)
    (1999.5, 2000.6, 1999.4, 2000.5),  # 1  BREAK (0.5 >= k_b*A, close upper half)
    (2000.5, 2001.0, 2000.2, 2000.8),  # 2  hold 1
    (2000.8, 2001.2, 1999.9, 2000.6),  # 3  hold 2, wick back on the level = retest
    (2000.6, 2001.5, 2000.4, 2001.2),  # 4  hold 3 -> entry decision
]
LONG_KIN = dict(v=0.3, m=0.5, a=0.25)  # arms d=+1 on every bar it is read


def long_case(tail, **kwargs):
    """``BASE_LONG`` plus the caller's bars, with the breakout kinematics."""
    params = dict(LONG_KIN)
    params.update(kwargs)
    return build_case(BASE_LONG + list(tail), **params)


# ═══════════════════════════════════════════════════════════════════════
# 1. THE FOUR NOMINAL SCENARIOS (§7.2, §7.3)
# ═══════════════════════════════════════════════════════════════════════


def test_breakout_long_nominal():
    case = long_case(
        [
            (2001.3, 2002.0, 2001.0, 2001.8),  # 5 fill at the open
            (2005.0, 2010.5, 2004.0, 2010.2),  # 6 target touched
            (2010.2, 2010.4, 2010.0, 2010.1),  # 7
        ]
    )
    events, trades = run_case(case)

    assert _events_str(events) == [
        "0 ARM",
        "1 BREAK BREAK_LONG",
        "4 ENTRY BREAK_LONG",
        "6 EXIT(TARGET) BREAK_LONG",
    ]
    t = trades.iloc[0]
    assert len(trades) == 1
    assert t.q == 1.0
    assert t.level == 2000.0
    assert t.stop == 2000.0 - 1.0 * 1.0  # L - d*k_s*A
    assert t.target == 2010.0  # L + 10*d
    # R = (|target - C| - s/2) / (|C - stop| + s/2), C = close of the decision bar
    assert t.r_est == pytest.approx((2010.0 - 2001.2) / (2001.2 - 1999.0))
    assert t.fill_px == 2001.3  # open of bar 5, mid + s/2 with s = 0
    assert t.exit_px == 2010.0
    assert t.retest == 1.0  # the wick of bar 3, traced and not required
    assert t.lots == pytest.approx(0.21)  # floor(50 / (2.3*100) / 0.01) * 0.01


def test_breakout_short_nominal():
    bars = [
        (2000.6, 2000.7, 2000.4, 2000.5),  # 0 ARM d=-1
        (2000.5, 2000.6, 1999.4, 1999.5),  # 1 BREAK
        (1999.5, 1999.8, 1999.0, 1999.2),  # 2 hold 1
        (1999.2, 2000.1, 1999.0, 1999.4),  # 3 hold 2 + retest
        (1999.4, 1999.6, 1998.8, 1998.9),  # 4 hold 3 -> entry
        (1998.8, 1999.2, 1998.0, 1998.4),  # 5 fill
        (1998.4, 1998.6, 1989.5, 1991.5),  # 6 target
        (1991.5, 1991.7, 1991.3, 1991.4),  # 7
    ]
    events, trades = run_case(
        build_case(bars, v=-0.3, m=-0.5, a=-0.25, ema=2010.0, vwap=2005.0)
    )

    assert _events_str(events) == [
        "0 ARM",
        "1 BREAK BREAK_SHORT",
        "4 ENTRY BREAK_SHORT",
        "6 EXIT(TARGET) BREAK_SHORT",
    ]
    t = trades.iloc[0]
    assert t.q == -1.0
    assert t.stop == 2001.0
    assert t.target == 1990.0
    assert t.r_est == pytest.approx((1998.9 - 1990.0) / (2001.0 - 1998.9))
    assert t.fill_px == 1998.8  # sell at mid - s/2
    assert t.exit_px == 1990.0
    assert t.retest == 1.0
    assert t.lots == pytest.approx(0.22)  # floor(50 / (2.2*100) / 0.01) * 0.01


def test_reversal_short_nominal():
    """Sweep above L = 2000 without a break, then reintegration (§7.3)."""
    bars = [
        (1998.4, 1999.6, 1998.3, 1999.5),  # 0 ARM d=+1
        (1999.5, 2000.5, 1999.4, 2000.1),  # 1 excursion starts, no break
        (2000.1, 2000.6, 1999.2, 1999.4),  # 2 reintegration -> SWEEP
        (1999.3, 1999.5, 1998.6, 1998.8),  # 3 fill
        (1998.8, 1999.0, 1989.0, 1991.5),  # 4 target
        (1991.5, 1991.7, 1991.3, 1991.4),  # 5
    ]
    case = build_case(
        bars,
        v=[0.3, 0.3, 0.05, 0.0, 0.0, 0.0],
        a=[0.25, 0.25, -0.3, 0.0, 0.0, 0.0],
        m=0.5,
    )
    events, trades = run_case(case)

    assert _events_str(events) == [
        "0 ARM",
        "2 SWEEP REV_SHORT",
        "2 ENTRY REV_SHORT",
        "4 EXIT(TARGET) REV_SHORT",
    ]
    t = trades.iloc[0]
    assert t.q == -1.0
    assert t.level == 2000.0
    assert t.stop == 2000.6 + 0.5 * 1.0  # extreme of the excursion + 0.5 ATR
    assert t.target == 1990.0  # L - 10*d
    assert t.r_est == pytest.approx((1999.4 - 1990.0) / (2001.1 - 1999.4))
    assert t.fill_px == 1999.3
    assert t.exit_px == 1990.0
    assert t.lots == pytest.approx(0.27)  # floor(50 / (1.8*100) / 0.01) * 0.01


def test_reversal_long_nominal():
    """Mirror image below L = 2000."""
    bars = [
        (2000.6, 2000.7, 2000.4, 2000.5),  # 0 ARM d=-1
        (2000.5, 2000.6, 1999.5, 1999.9),  # 1 excursion starts
        (1999.9, 2000.6, 1999.4, 2000.4),  # 2 reintegration -> SWEEP
        (2000.5, 2001.0, 2000.2, 2000.8),  # 3 fill
        (2000.8, 2010.5, 2000.6, 2008.5),  # 4 target
        (2008.5, 2008.7, 2008.3, 2008.4),  # 5
    ]
    case = build_case(
        bars,
        v=[-0.3, -0.3, -0.05, 0.0, 0.0, 0.0],
        a=[-0.25, -0.25, 0.3, 0.0, 0.0, 0.0],
        m=-0.5,
        ema=2010.0,
        vwap=2005.0,
    )
    events, trades = run_case(case)

    assert _events_str(events) == [
        "0 ARM",
        "2 SWEEP REV_LONG",
        "2 ENTRY REV_LONG",
        "4 EXIT(TARGET) REV_LONG",
    ]
    t = trades.iloc[0]
    assert t.q == 1.0
    assert t.stop == 1999.4 - 0.5 * 1.0
    assert t.target == 2010.0
    assert t.r_est == pytest.approx((2010.0 - 2000.4) / (2000.4 - 1998.9))
    assert t.fill_px == 2000.5
    assert t.lots == pytest.approx(0.31)  # floor(50 / (1.6*100) / 0.01) * 0.01


# ═══════════════════════════════════════════════════════════════════════
# 2. FAILED BREAKOUT -> REVERSAL (§7.2, points 1-5)
# ═══════════════════════════════════════════════════════════════════════

_FAILED_BREAK_BARS = [
    (1998.4, 1999.6, 1998.3, 1999.5),  # 0 ARM
    (1999.5, 2000.6, 1999.4, 2000.5),  # 1 BREAK
    (2000.5, 2001.0, 1999.0, 1999.2),  # 2 close back through L
    (1999.1, 1999.3, 1998.4, 1998.6),  # 3
    (1998.6, 1998.8, 1989.0, 1991.5),  # 4
    (1991.5, 1991.7, 1991.3, 1991.4),  # 5
]


def test_failed_breakout_flips_into_a_reversal_on_the_reintegration_bar():
    case = build_case(
        _FAILED_BREAK_BARS,
        v=[0.3, 0.3, 0.05, 0.0, 0.0, 0.0],
        a=[0.25, 0.25, -0.3, 0.0, 0.0, 0.0],
        m=0.5,
    )
    events, trades = run_case(case)

    assert _events_str(events) == [
        "0 ARM",
        "1 BREAK BREAK_LONG",
        "2 CANCEL(hold) BREAK_LONG",
        "2 SWEEP REV_SHORT",
        "2 ENTRY REV_SHORT",
        "4 EXIT(TARGET) REV_SHORT",
    ]
    t = trades.iloc[0]
    # E is the extreme reached since the break bar b, not since the arming.
    assert t.stop == 2001.0 + 0.5
    assert t.target == 1990.0
    assert t.fill_px == 1999.1
    assert t.lots == pytest.approx(0.20)  # floor(50 / (2.4*100) / 0.01) * 0.01


def test_failed_breakout_without_a_flip_goes_back_to_idle():
    case = build_case(
        _FAILED_BREAK_BARS,
        v=[0.3, 0.3, 0.05, 0.0, 0.0, 0.0],
        a=[0.25, 0.25, 0.0, 0.0, 0.0, 0.0],  # no flip: d*a > -a_min
        m=0.5,
    )
    events, trades = run_case(case)
    assert _events_str(events) == [
        "0 ARM",
        "1 BREAK BREAK_LONG",
        "2 CANCEL(hold) BREAK_LONG",
    ]
    assert len(trades) == 0


# ═══════════════════════════════════════════════════════════════════════
# 3. ARMING (§7.1)
# ═══════════════════════════════════════════════════════════════════════

_ZONE_BARS = [(1998.4, 1999.6, 1998.3, 1999.5)] * 3


@pytest.mark.parametrize(
    ("kin", "why"),
    [
        (dict(v=0.3, m=-0.5, a=0.25), "d*m <= 0"),
        (dict(v=0.3, m=0.0, a=0.25), "d*m == 0 is not > 0"),
        (dict(v=0.1, m=0.5, a=0.25), "d*v < v_min"),
        (dict(v=0.3, m=0.5, a=0.1), "d*a < a_min"),
    ],
)
def test_arming_refused_when_a_kinematic_condition_fails(kin, why):
    events, _ = run_case(build_case(_ZONE_BARS, **kin))
    assert len(events) == 0, why


def test_arming_needs_a_defined_kinematics():
    """§5: a NaN in v, m or a forbids arming — no substitution by zero."""
    events, _ = run_case(build_case(_ZONE_BARS, v=0.3, m=np.nan, a=0.25))
    assert len(events) == 0


def test_a_second_armable_level_is_ignored_while_one_is_armed():
    """§7.1: no replacement, no queue — and N_arm closes the state at bar 12."""
    bars = [(1999.2, 1999.7, 1999.1, 1999.5)] * 16
    events, _ = run_case(build_case(bars, **LONG_KIN))
    assert _events_str(events) == ["0 ARM", "12 CANCEL(narm)"]


def test_arming_cancelled_when_the_price_leaves_the_zone_backwards():
    bars = [
        (1998.4, 1999.6, 1998.3, 1999.5),  # 0 ARM, gap 0.5
        (1999.5, 1999.6, 1998.4, 1998.5),  # 1 gap 1.5 > z*A
    ] + [(1999.2, 1999.7, 1999.1, 1999.5)] * 10
    events, _ = run_case(build_case(bars, **LONG_KIN))
    # The level then sits on the 6-bar cooldown of §9: bars 2-7 cannot re-arm.
    assert _events_str(events) == ["0 ARM", "1 CANCEL(zone)", "8 ARM"]


# ═══════════════════════════════════════════════════════════════════════
# 4. HOLD WINDOW AND RETEST (§7.2)
# ═══════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize("bar_back", [2, 3, 4])
def test_a_close_back_through_the_level_cancels_the_breakout(bar_back):
    """A close on the wrong side at hold bar 1, 2 or 3 kills the breakout."""
    bars = list(BASE_LONG) + [(2001.3, 2002.0, 2001.0, 2001.8)] * 3
    bars[bar_back] = (2000.5, 2001.0, 1999.0, 1999.2)
    # Acceleration only lives long enough to arm: no flip, so a plain cancel.
    a = np.where(np.arange(len(bars)) < 2, 0.25, 0.0)
    case = build_case(bars, v=0.3, m=0.5, a=a)
    events, trades = run_case(case)
    assert _events_str(events) == [
        "0 ARM",
        "1 BREAK BREAK_LONG",
        f"{bar_back} CANCEL(hold) BREAK_LONG",
    ]
    assert len(trades) == 0


def test_the_retest_is_traced_but_never_required():
    """Same breakout with no wick back on the level: it still enters."""
    bars = list(BASE_LONG) + [
        (2001.3, 2002.0, 2001.0, 2001.8),
        (2005.0, 2010.5, 2004.0, 2010.2),
        (2010.2, 2010.4, 2010.0, 2010.1),
    ]
    bars[3] = (2000.8, 2001.2, 2000.4, 2000.6)  # low no longer touches 2000
    _, trades = run_case(build_case(bars, **LONG_KIN))
    assert len(trades) == 1
    assert trades.iloc[0].retest == 0.0


# ═══════════════════════════════════════════════════════════════════════
# 5. DIRECT SWEEP — the N_sweep window (§7.3)
# ═══════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize(("bar_back", "taken"), [(1, True), (2, True), (3, True), (4, False)])
def test_reintegration_is_accepted_within_n_sweep_bars_and_refused_after(bar_back, taken):
    """The window counts from the first bar whose extreme passes L, inclusive.

    Bars ``i_exc`` .. ``i_exc + N_sweep - 1`` can reintegrate; the bar after
    that cancels the arming with ``nsweep`` before its close is even read.
    """
    above = (1999.5, 2000.5, 1999.4, 2000.1)  # extreme past L, close still above
    back = (2000.1, 2000.6, 1999.2, 1999.4)  # the reintegration bar
    bars = [(1998.4, 1999.6, 1998.3, 1999.5)] + [above] * 4
    bars[bar_back] = back
    bars += [
        (1999.3, 1999.5, 1998.6, 1998.8),
        (1998.8, 1999.0, 1989.0, 1991.5),
        (1991.5, 1991.7, 1991.3, 1991.4),
    ]
    n = len(bars)
    v = np.full(n, 0.3)
    a = np.full(n, 0.25)
    v[bar_back] = 0.05  # deceleration
    a[bar_back] = -0.3  # flip
    events, trades = run_case(build_case(bars, v=v, a=a, m=0.5))

    labels = _events_str(events)
    if taken:
        assert f"{bar_back} SWEEP REV_SHORT" in labels
        assert len(trades) == 1
    else:
        assert labels == ["0 ARM", "4 CANCEL(nsweep)"]
        assert len(trades) == 0


# ═══════════════════════════════════════════════════════════════════════
# 6. THE R RULE (§8)
# ═══════════════════════════════════════════════════════════════════════


def _r_case(ratio: float, spread: float):
    """A reversal short whose R is, by construction, exactly ``ratio``.

    The target sits at ``L - 10`` and the close of the reintegration bar is
    fixed, so the stop — hence the extreme of the excursion — is solved for
    from ``R = (T - s/2) / (S + s/2)``.
    """
    close = 1999.4
    half = 0.5 * spread
    t_dist = close - 1990.0
    s_dist = (t_dist - half) / ratio - half
    extreme = close + s_dist - 0.5  # stop = extreme + 0.5*A
    bars = [
        (1998.4, 1999.6, 1998.3, 1999.5),
        (1999.5, extreme, 1999.4, 2000.1),
        (2000.1, 2000.2, 1999.2, close),
        (1999.3, 1999.5, 1998.6, 1998.8),
        (1998.8, 1999.0, 1998.6, 1998.7),
    ]
    case = build_case(
        bars,
        v=[0.3, 0.3, 0.05, 0.0, 0.0],
        a=[0.25, 0.25, -0.3, 0.0, 0.0],
        m=0.5,
        spread=spread,
    )
    return case, extreme + 0.5


@pytest.mark.parametrize("spread", [0.0, 0.29])
def test_r_below_one_cancels_the_candidate(spread):
    case, stop = _r_case(0.999, spread)
    events, trades = run_case(case)
    assert _events_str(events) == ["0 ARM", "2 SWEEP REV_SHORT", "2 CANCEL(r) REV_SHORT"]
    assert len(trades) == 0
    assert events.iloc[-1].r_est == pytest.approx(0.999)
    assert events.iloc[-1].stop == pytest.approx(stop)


@pytest.mark.parametrize("spread", [0.0, 0.29])
def test_r_just_above_one_enters(spread):
    case, _ = _r_case(1.001, spread)
    events, trades = run_case(case)
    assert "2 ENTRY REV_SHORT" in _events_str(events)
    assert trades.iloc[0].r_est == pytest.approx(1.001)


def test_the_spread_enters_r_once_on_each_side():
    """§8: ``s/2`` is subtracted from the numerator and added to the denominator."""
    case, stop = _r_case(1.5, 0.29)
    _, trades = run_case(case)
    close = 1999.4
    naive = (2010.0 - close) / (stop - close)  # the same ratio without any spread
    assert trades.iloc[0].r_est == pytest.approx((close - 1990.0 - 0.145) / (stop - close + 0.145))
    assert trades.iloc[0].r_est != pytest.approx(naive)


# ═══════════════════════════════════════════════════════════════════════
# 7. EXECUTION (§9, §10)
# ═══════════════════════════════════════════════════════════════════════

_QUIET = (2001.5, 2001.6, 2001.4, 2001.5)
_FILL_BAR = (2001.3, 2002.0, 2001.0, 2001.8)
_TARGET_BAR = (2005.0, 2010.5, 2004.0, 2010.2)
_AFTER_TARGET = (2010.2, 2010.4, 2010.0, 2010.1)
# Fill on bar 5, target on bar 6: the shortest complete trade.
_TARGET_TAIL = [_FILL_BAR, _TARGET_BAR, _QUIET]
# Same, with one idle bar in between: the position has to survive two closes.
_SLOW_TARGET_TAIL = [_FILL_BAR, (2001.8, 2002.5, 2001.5, 2002.0), _TARGET_BAR, _AFTER_TARGET]


def test_the_fill_is_the_next_open_plus_half_a_spread():
    case = long_case(
        [_FILL_BAR, (2005.0, 2010.5, 2004.0, 2010.2), _QUIET], spread=0.29
    )
    _, trades = run_case(case)
    t = trades.iloc[0]
    assert t.fill_px == pytest.approx(2001.3 + 0.145)  # buy at mid + s/2
    assert t.spread == 0.29
    # Sizing and R both read the spread-adjusted entry, not the mid.
    assert t.lots == pytest.approx(np.floor(50.0 / (2001.445 - 1999.0) / CONTRACT_SIZE / 0.01) * 0.01)


def test_a_double_contact_inside_one_m1_is_a_stop():
    case = long_case(
        [_FILL_BAR, (2005.0, 2010.5, 1998.0, 2000.0), _QUIET],
        m1={6: [(2005.0, 2010.5, 1998.0, 2000.0)]},
    )
    _, trades = run_case(case)
    t = trades.iloc[0]
    assert EXIT_REASON_NAMES[int(t.exit_reason)] == "STOP"
    assert t.exit_px == 1999.0


def test_a_gap_beyond_the_stop_fills_at_the_open():
    case = long_case(
        [_FILL_BAR, (1995.0, 1996.0, 1994.0, 1995.5), _QUIET],
        m1={6: [(1995.0, 1996.0, 1994.0, 1995.5)]},
    )
    _, trades = run_case(case)
    t = trades.iloc[0]
    assert EXIT_REASON_NAMES[int(t.exit_reason)] == "STOP"
    assert t.exit_px == 1995.0  # the open, not the theoretical 1999.0


def test_a_gap_beyond_the_target_exits_at_the_open_even_if_the_stop_follows():
    """§9: the open is the first price of the minute, and it is known.

    The pessimistic rule of §9 — "double contact inside one M1, the stop wins" —
    arbitrates the order the bar does *not* tell us. It does not apply here: the
    open is chronologically first, the position is already closed at the target
    when the low is printed, and the low belongs to a position that no longer
    exists. Frozen by this test; §9 is being amended to say so.
    """
    case = long_case(
        [_FILL_BAR, (2012.0, 2012.5, 1990.0, 1991.0), _QUIET],
        m1={6: [(2012.0, 2012.5, 1990.0, 1991.0)]},
    )
    _, trades = run_case(case)
    t = trades.iloc[0]
    assert EXIT_REASON_NAMES[int(t.exit_reason)] == "TARGET"
    assert t.exit_px == 2012.0  # the open, on the bid side (s = 0 here)


def test_the_target_is_only_hit_when_the_trigger_side_reaches_it():
    """Long exits are sold at the bid: a mid high grazing the target is not one."""
    case = long_case(
        [
            _FILL_BAR,
            (2005.0, 2010.10, 2004.9, 2010.05),  # 6 bid high 2009.955 < 2010
            (2010.0, 2010.20, 2009.9, 2010.15),  # 7 bid high 2010.055 >= 2010
            _QUIET,
        ],
        spread=0.29,
        m1={6: [(2005.0, 2010.10, 2004.9, 2010.05)], 7: [(2010.0, 2010.20, 2009.9, 2010.15)]},
    )
    _, trades = run_case(case)
    t = trades.iloc[0]
    assert t.i_exit_m5 == 7.0
    assert EXIT_REASON_NAMES[int(t.exit_reason)] == "TARGET"
    assert t.exit_px == 2010.0


def test_the_position_is_closed_after_48_m5_bars():
    case = long_case([_FILL_BAR] + [_QUIET] * 48)
    _, trades = run_case(case)
    t = trades.iloc[0]
    assert EXIT_REASON_NAMES[int(t.exit_reason)] == "TIME"
    assert t.bars_held == 48.0
    assert t.i_exit_m5 == 52.0  # filled on bar 5, held through bar 52
    assert t.exit_px == _QUIET[3]


def test_the_position_is_flat_at_1655_new_york():
    case = long_case([_FILL_BAR] + [_QUIET] * 8, minute0=960)
    _, trades = run_case(case)
    t = trades.iloc[0]
    assert case["minute_of_day"][8] == 16 * 60 + 40  # still open at 16:40
    assert EXIT_REASON_NAMES[int(t.exit_reason)] == "SESSION"
    assert t.i_exit_m5 == 11.0  # 16:00 + 55 min
    assert case["minute_of_day"][11] == 16 * 60 + 55


@pytest.mark.parametrize("fill_minute", [18 * 60 + 15, 19 * 60, 21 * 60, 23 * 60 + 30])
def test_an_evening_position_is_not_closed_on_its_own_fill_bar(fill_minute):
    """§10: 16:55 flattens the session, it does not flatten the whole evening.

    ``minute_of_day`` counts from New York midnight, so an unbounded
    ``>= 16:55`` test is also true at 18:15, 21:00 and 23:30 — every evening
    entry would be shut on the bar that opened it, one bar held and the spread
    paid for nothing. The forced close lives in ``[16:55, 18:00)``.
    """
    case = long_case(_SLOW_TARGET_TAIL, minute0=(fill_minute - 25) % 1440)
    _, trades = run_case(case)
    t = trades.iloc[0]
    assert case["minute_of_day"][5] == fill_minute
    assert EXIT_REASON_NAMES[int(t.exit_reason)] == "TARGET"
    assert t.bars_held == 3.0
    assert t.i_exit_m5 == 7.0


def test_a_position_crossing_new_york_midnight_is_kept():
    """Midnight is the middle of a session (§2), not a boundary of any kind."""
    case = long_case(_SLOW_TARGET_TAIL, minute0=23 * 60 + 25)
    _, trades = run_case(case)
    t = trades.iloc[0]
    assert case["minute_of_day"][6] == 23 * 60 + 55
    assert case["minute_of_day"][7] == 0  # the wrap happens inside the trade
    assert EXIT_REASON_NAMES[int(t.exit_reason)] == "TARGET"
    assert t.bars_held == 3.0


@pytest.mark.parametrize(
    ("fill_minute", "entered"),
    [(16 * 60 + 25, True), (16 * 60 + 30, False), (18 * 60 + 10, False), (18 * 60 + 15, True)],
)
def test_no_entry_fills_between_1630_and_1815(fill_minute, entered):
    """§10: the window bites on the fill bar, which is the decision bar plus one."""
    case = long_case(_TARGET_TAIL, minute0=(fill_minute - 25) % 1440)
    events, trades = run_case(case)
    labels = _events_str(events)
    assert ("4 ENTRY BREAK_LONG" in labels) is entered
    assert ("4 CANCEL(window) BREAK_LONG" in labels) is not entered
    if entered:
        # An entry allowed at 18:15 must also be allowed to live past its bar.
        assert EXIT_REASON_NAMES[int(trades.iloc[0].exit_reason)] == "TARGET"
        assert trades.iloc[0].bars_held == 2.0


def test_an_entry_is_refused_when_the_next_bar_is_not_the_next_five_minutes():
    """§2: ``fill[t] := open of bar t+1`` means the bar five minutes later.

    After a hole in the feed the next *existing* bar can be hours away; filling
    there would execute a decision on a price the decision never saw.
    """
    case = long_case(_TARGET_TAIL, gap_before={5: 10})
    events, trades = run_case(case)
    assert "4 CANCEL(window) BREAK_LONG" in _events_str(events)
    assert len(trades) == 0


def test_a_contiguous_grid_still_fills_on_the_next_bar():
    case = long_case(_TARGET_TAIL)
    events, trades = run_case(case)
    assert "4 ENTRY BREAK_LONG" in _events_str(events)
    assert len(trades) == 1


def test_only_one_position_at_a_time():
    """A level armable while a position is open is simply not seen (§7.1, §9)."""
    case = long_case(
        [
            (2001.3, 2009.9, 2001.0, 2009.6),  # 5 fill; close 0.4 under 2010
            (2009.6, 2010.5, 2009.4, 2010.2),  # 6 target
            (2010.2, 2010.4, 2010.0, 2010.1),
        ]
    )
    events, trades = run_case(case)
    assert _events_str(events) == [
        "0 ARM",
        "1 BREAK BREAK_LONG",
        "4 ENTRY BREAK_LONG",
        "6 EXIT(TARGET) BREAK_LONG",
    ]
    assert len(trades) == 1


# ═══════════════════════════════════════════════════════════════════════
# 8. SIZING (§11)
# ═══════════════════════════════════════════════════════════════════════


def test_lots_follow_the_risk_fraction_and_round_down():
    _, trades = run_case(long_case(_TARGET_TAIL))
    t = trades.iloc[0]
    raw = 0.005 * 10_000.0 / (abs(2001.3 - 1999.0) * CONTRACT_SIZE)
    assert raw == pytest.approx(0.2173913)
    assert t.lots == pytest.approx(0.21)
    assert t.risk_amount == pytest.approx(0.21 * 2.3 * CONTRACT_SIZE)


def test_an_adverse_dollar_halves_the_risk_without_blocking_the_trade():
    case = long_case(_TARGET_TAIL, dxy=101.0, dxy_ema=100.0)
    _, trades = run_case(case)
    t = trades.iloc[0]
    assert t.ctx_dxy == -1.0
    assert t.lots == pytest.approx(0.10)  # floor(25 / 230 / 0.01) * 0.01


def test_a_size_below_volume_min_cancels_the_entry():
    events, trades = run_case(long_case(_TARGET_TAIL), init_cash=100.0)
    assert _events_str(events) == [
        "0 ARM",
        "1 BREAK BREAK_LONG",
        "4 CANCEL(size) BREAK_LONG",
    ]
    assert len(trades) == 0


def test_the_leverage_cap_truncates_an_oversized_position():
    _, trades = run_case(long_case(_TARGET_TAIL), risk_frac=0.5)
    # 1:100 on 10 000 of equity = 1 000 000 of notional / (100 oz * 2001.3)
    assert trades.iloc[0].lots == pytest.approx(4.99)


def test_equity_compounds_from_one_trade_to_the_next():
    bars = list(BASE_LONG) + [
        (2001.3, 2002.0, 2001.0, 2001.8),  # 5  fill long at 2001.3
        (2001.3, 2001.4, 1998.0, 1998.2),  # 6  stopped at 1999.0
        (1998.2, 1998.3, 1990.5, 1990.8),  # 7  ARM d=-1 on 1990
        (1990.8, 1990.9, 1989.3, 1989.5),  # 8  BREAK
        (1989.5, 1989.8, 1989.0, 1989.2),  # 9  hold 1
        (1989.2, 1989.4, 1988.8, 1989.0),  # 10 hold 2
        (1989.0, 1989.2, 1988.4, 1988.6),  # 11 hold 3 -> entry
        (1988.5, 1988.7, 1987.9, 1988.1),  # 12 fill short at 1988.5
        (1988.1, 1988.3, 1979.5, 1981.5),  # 13 target 1980
        (1981.5, 1981.7, 1981.3, 1981.4),  # 14
    ]
    n = len(bars)
    v = np.where(np.arange(n) < 7, 0.3, -0.3)
    a = np.where(np.arange(n) < 7, 0.25, -0.25)
    m = np.where(np.arange(n) < 7, 0.5, -0.5)
    _, trades = run_case(build_case(bars, v=v, a=a, m=m, ema=1995.0, vwap=1995.0))

    assert len(trades) == 2
    loss = (1999.0 - 2001.3) * 0.21 * CONTRACT_SIZE
    assert trades.iloc[0].equity_before == 10_000.0
    assert trades.iloc[0].equity_after == pytest.approx(10_000.0 + loss)
    # The second trade is sized on the equity left by the first.
    assert trades.iloc[1].equity_before == pytest.approx(10_000.0 + loss)
    assert trades.iloc[1].lots == pytest.approx(
        np.floor(0.005 * (10_000.0 + loss) / (2.5 * CONTRACT_SIZE) / 0.01) * 0.01
    )
    gain = (1988.5 - 1980.0) * trades.iloc[1].lots * CONTRACT_SIZE
    assert trades.iloc[1].equity_after == pytest.approx(10_000.0 + loss + gain)


# ═══════════════════════════════════════════════════════════════════════
# 9. CONTEXT (§6.4)
# ═══════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize(
    ("ema", "vwap"),
    [(2010.0, 1995.0), (2001.2, 1995.0), (1990.0, 2010.0), (1990.0, 2001.2)],
)
def test_a_breakout_needs_both_the_ema_and_the_vwap_in_its_favour(ema, vwap):
    """§6.4 usage 1 — and ``signe(0) = 0`` is not favourable."""
    events, trades = run_case(long_case(_TARGET_TAIL, ema=ema, vwap=vwap))
    assert _events_str(events) == [
        "0 ARM",
        "1 BREAK BREAK_LONG",
        "4 CANCEL(ctx) BREAK_LONG",
    ]
    assert len(trades) == 0
    assert np.isnan(events.iloc[-1].r_est)  # refused before R is even computed


def test_a_reversal_is_allowed_against_both_the_ema_and_the_vwap():
    """§6.4 usage 2 — counter-trend by construction, so no context gate."""
    bars = [
        (1998.4, 1999.6, 1998.3, 1999.5),
        (1999.5, 2000.5, 1999.4, 2000.1),
        (2000.1, 2000.6, 1999.2, 1999.4),
        (1999.3, 1999.5, 1998.6, 1998.8),
        (1998.8, 1999.0, 1989.0, 1991.5),
        (1991.5, 1991.7, 1991.3, 1991.4),
    ]
    case = build_case(
        bars,
        v=[0.3, 0.3, 0.05, 0.0, 0.0, 0.0],
        a=[0.25, 0.25, -0.3, 0.0, 0.0, 0.0],
        m=0.5,
        ema=1990.0,
        vwap=1995.0,
    )
    _, trades = run_case(case)
    t = trades.iloc[0]
    assert (t.ctx_ema, t.ctx_vwap) == (-1.0, -1.0)  # both against the short
    assert t.lots > 0.0


@pytest.mark.parametrize(("atr_h1", "extended"), [(10.0, 0.0), (2.0, 1.0)])
def test_the_vwap_extension_is_measured_and_never_blocks(atr_h1, extended):
    bars = [
        (1998.4, 1999.6, 1998.3, 1999.5),
        (1999.5, 2000.5, 1999.4, 2000.1),
        (2000.1, 2000.6, 1999.2, 1999.4),
        (1999.3, 1999.5, 1998.6, 1998.8),
        (1998.8, 1999.0, 1989.0, 1991.5),
        (1991.5, 1991.7, 1991.3, 1991.4),
    ]
    case = build_case(
        bars,
        v=[0.3, 0.3, 0.05, 0.0, 0.0, 0.0],
        a=[0.25, 0.25, -0.3, 0.0, 0.0, 0.0],
        m=0.5,
        atr_h1=atr_h1,
    )
    _, trades = run_case(case)
    assert len(trades) == 1  # taken in both cases
    assert trades.iloc[0].extension_vwap == extended


# ═══════════════════════════════════════════════════════════════════════
# 10-11. DETERMINISM AND CAUSALITY
# ═══════════════════════════════════════════════════════════════════════


def random_case(n: int = 2500, seed: int = 7, spread: float = 0.29) -> dict:
    """A long pseudo-random M5 series with real indicators on top of it."""
    rng = np.random.default_rng(seed)
    close = 2000.0 + np.cumsum(rng.normal(0.0, 0.45, n))
    high = close + np.abs(rng.normal(0.0, 0.35, n)) + 0.05
    low = close - np.abs(rng.normal(0.0, 0.35, n)) - 0.05
    open_ = np.empty(n)
    open_[0] = close[0]
    open_[1:] = close[:-1]
    open_ = np.clip(open_, low, high)

    atr = atr_wilder_nb(high, low, close)
    v, m, a = kinematics_nb(close, atr)
    ema = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().shift(1).to_numpy()
    vwap = pd.Series(close).expanding().mean().to_numpy()
    bars = list(zip(open_, high, low, close, strict=True))

    case = build_case(bars, spread=spread)
    case["atr"] = atr
    case["v"] = v
    case["m"] = m
    case["a"] = a
    case["ema50_h1"] = ema
    case["vwap"] = vwap
    case["atr_h1"] = atr * 3.0
    case["session_id"] = ((600 + 5 * np.arange(n) + 360) // 1440).astype(np.int64)
    return case


def test_two_runs_produce_bit_identical_registries():
    case = random_case()
    first = run_engine(**case)
    second = run_engine(**case)
    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_array_equal(first[1], second[1])
    assert len(first[0]) > 50  # the fixture is not vacuous


def _truncate(case: dict, k: int) -> dict:
    cut = dict(case)
    j = int(case["m1_end"][k - 1])
    for key, value in case.items():
        if key.startswith("m1_") and key not in ("m1_start", "m1_end"):
            cut[key] = value[:j]
        elif key in ("m1_start", "m1_end"):
            cut[key] = value[:k]
        else:
            cut[key] = value[:k]
    return cut


def test_truncating_the_input_leaves_the_earlier_events_untouched():
    """End-to-end causality: nothing decided before bar ``k-1`` sees bar ``k``.

    The last two bars are excluded from the comparison and that exclusion is
    the honest statement of the property, not a loophole: bar ``k-1`` is the
    final bar of the truncated input, where the engine takes no decision (it
    has no bar to fill on) and force-closes any open position (§10).
    """
    case = random_case()
    full, _ = run_engine(**case)
    for k in (900, 1500, 2100):
        cut, _ = run_engine(**_truncate(case, k))
        keep = full[full[:, 0] < k - 1]
        got = cut[cut[:, 0] < k - 1]
        assert len(keep) > 10
        np.testing.assert_array_equal(keep, got)


# ═══════════════════════════════════════════════════════════════════════
# 12. ABLATION SWITCHES (campaign only — annexe A.3)
# ═══════════════════════════════════════════════════════════════════════
#
# The three flags remove one *decision* each, never a measurement: the scores
# keep being computed and traced. Each test below pins both halves — what the
# spec does, and what the ablation does instead — so that a flag silently
# widening its reach fails here rather than in a campaign JSON.

# Bar 5 fills the breakout decided at bar 4, bar 6 reaches the target.
ABLATION_TAIL = [
    (2001.3, 2002.0, 2001.0, 2001.8),
    (2005.0, 2010.5, 2004.0, 2010.2),
    (2010.2, 2010.4, 2010.0, 2010.1),
]


def test_use_ema_false_lets_a_breakout_through_an_adverse_ema():
    """§6.4 usage 1, half of it: ``ctx_ema > 0`` stops the entry, or does not."""
    case = long_case(ABLATION_TAIL, ema=2010.0, vwap=1995.0)

    events, trades = run_case(case)
    assert _events_str(events)[-1] == "4 CANCEL(ctx) BREAK_LONG"
    assert len(trades) == 0

    events, trades = run_case(case, use_ema=False)
    assert _events_str(events) == [
        "0 ARM",
        "1 BREAK BREAK_LONG",
        "4 ENTRY BREAK_LONG",
        "6 EXIT(TARGET) BREAK_LONG",
    ]
    # The score is still measured and still adverse — only the gate is gone.
    assert trades.iloc[0].ctx_ema == -1.0
    assert trades.iloc[0].ctx_vwap == 1.0


def test_use_vwap_false_lets_a_breakout_through_an_adverse_vwap():
    """§6.4 usage 1, the other half. ``use_ema`` is untouched and still bites."""
    case = long_case(ABLATION_TAIL, ema=1990.0, vwap=2010.0)

    events, trades = run_case(case)
    assert _events_str(events)[-1] == "4 CANCEL(ctx) BREAK_LONG"
    assert len(trades) == 0

    events, trades = run_case(case, use_vwap=False)
    assert _events_str(events) == [
        "0 ARM",
        "1 BREAK BREAK_LONG",
        "4 ENTRY BREAK_LONG",
        "6 EXIT(TARGET) BREAK_LONG",
    ]
    assert trades.iloc[0].ctx_ema == 1.0
    assert trades.iloc[0].ctx_vwap == -1.0


def test_use_dxy_false_stops_halving_the_risk_on_an_adverse_dollar():
    """§6.4 usage 3 / §11: the flag moves the lots, never the entry itself."""
    # q = +1 and DXY above its EMA50 H1 -> ctx_dxy = sign(-(dxy - ema)) = -1.
    case = long_case(ABLATION_TAIL, dxy=101.0, dxy_ema=100.0)

    _, trades = run_case(case)
    assert len(trades) == 1
    assert trades.iloc[0].ctx_dxy == -1.0
    # 0.25 % of 10 000 over a 2.3 $ stop on a 100 oz contract.
    assert trades.iloc[0].lots == pytest.approx(0.10)

    _, ablated = run_case(case, use_dxy=False)
    assert len(ablated) == 1
    assert ablated.iloc[0].ctx_dxy == -1.0  # still measured, still adverse
    assert ablated.iloc[0].lots == pytest.approx(0.21)  # the undivided 0.5 %


@pytest.mark.parametrize(
    "flags",
    [
        {},
        dict(use_ema=True),
        dict(use_vwap=True),
        dict(use_dxy=True),
        dict(use_ema=True, use_vwap=True, use_dxy=True),
    ],
)
def test_the_flags_at_their_defaults_change_nothing(flags):
    """Bit-for-bit: the spec run is the ``True, True, True`` run."""
    case = random_case()
    reference = run_engine(**case)
    got = run_engine(**case, **flags)
    np.testing.assert_array_equal(reference[0], got[0])
    np.testing.assert_array_equal(reference[1], got[1])
    assert len(reference[0]) > 50  # the fixture is not vacuous
