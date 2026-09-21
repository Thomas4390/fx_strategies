"""Streaming twin of ``framework/x10_engine.py`` — the x10 automaton (§7-§11).

Pure Python: no ``AlgorithmImports``, no pandas, no numpy. ``main.py`` is the
only file of this folder that knows QuantConnect exists; everything decided here
is decided from prices, and ``tests/test_qc_x10_parity.py`` replays it against
the reference kernel on synthetic minutes.

The reference walks two preallocated arrays in one Numba loop; a QuantConnect
algorithm is handed one minute at a time and has to place its orders while the
minute is still tradable. The loop is therefore split into the three steps the
kernel docstring numbers, kept in the same order and with the same contents:

1. ``open_bar`` — the entry decided at the previous M5 close fills at the open
   of this bar, which is also where the lots of §11 are computed and where a
   position too small to size, or a fill bar that is not the immediate
   successor of the decision bar, cancels;
2. ``on_minute`` — an open position is resolved minute by minute, the open
   price first on a gap and the stop first on a double contact inside the same
   minute (§9);
3. ``close_bar`` — the time exits of §10, then the automaton reads the close and
   may arm, break, sweep, cancel or decide an entry for the next bar.

Two rules the kernel resolves with array indexing and which streaming has to
state explicitly:

* the decision bar's ``v[t-1]`` is kept in ``_prev_v`` and refreshed on **every**
  M5 bar, including the frozen ones (§5), because the kernel reads a plain array;
* the fill bar must start exactly five minutes after the decision bar (§2). The
  kernel compares two entries of its ``bar_minute`` column; here the test is
  deferred to the fill itself, which produces the same trace — nothing can be
  emitted between the end of step 3 of bar ``i`` and the start of step 1 of bar
  ``i+1`` — with the cooldown still counted from the decision bar.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import floor, isfinite
from typing import NamedTuple

from x10_bars import (
    H1_MINUTES,
    M5_MINUTES,
    Bar,
    BarBuilder,
    MinuteQuote,
    minute_to_utc,
    ny_fields,
)
from x10_indicators import (
    ATR_PERIOD,
    DXY4_LEGS,
    EMA_SPAN_H1,
    Dxy4Basket,
    Ema,
    Kinematics,
    SessionTypicalMean,
    WilderAtr,
    x10_levels,
)

NAN = float("nan")

# ═══════════════════════════════════════════════════════════════════════
# CODES — events, scenarios, exit and cancel reasons (§7.4, §10)
# ═══════════════════════════════════════════════════════════════════════

EVENT_ARM = 0
EVENT_BREAK = 1
EVENT_SWEEP = 2
EVENT_ENTRY = 3
EVENT_EXIT = 4
EVENT_CANCEL = 5
EVENT_NAMES: tuple[str, ...] = ("ARM", "BREAK", "SWEEP", "ENTRY", "EXIT", "CANCEL")

SC_NONE = 0
SC_BREAK_LONG = 1
SC_BREAK_SHORT = 2
SC_REV_LONG = 3
SC_REV_SHORT = 4
SCENARIO_NAMES: tuple[str, ...] = ("", "BREAK_LONG", "BREAK_SHORT", "REV_LONG", "REV_SHORT")

XR_NONE = 0
XR_STOP = 1
XR_TARGET = 2
XR_TIME = 3
XR_SESSION = 4
EXIT_REASON_NAMES: tuple[str, ...] = ("", "STOP", "TARGET", "TIME", "SESSION")

CR_NONE = 0
CR_ZONE = 1  # §7.1 — price left the zone backwards
CR_NARM = 2  # §7.1 — N_arm bars without an event
CR_HOLD = 3  # §7.2 — a close came back through the level during the hold
CR_NSWEEP = 4  # §7.3 — no reintegration within N_sweep bars
CR_SWEEP = 5  # §7.3 — reintegration without depth / deceleration / flip
CR_CTX = 6  # §6.4 — breakout refused by EMA50 H1 or VWAP
CR_R = 7  # §8 — R < 1
CR_WINDOW = 8  # §10 / §2 — forbidden window, or fill bar not at decision + 5 min
CR_SIZE = 9  # §11 — lots below volume_min
CANCEL_REASON_NAMES: tuple[str, ...] = (
    "",
    "zone",
    "narm",
    "hold",
    "nsweep",
    "sweep",
    "ctx",
    "r",
    "window",
    "size",
)

# ═══════════════════════════════════════════════════════════════════════
# FROZEN PARAMETERS — annexe A.1. Not arguments, on purpose.
# ═══════════════════════════════════════════════════════════════════════

V_MIN = 0.2  # §7.1
K_B = 0.25  # §7.2 breakout strength
S_MIN = 0.3  # §7.3 sweep depth
N_ARM = 12  # §7.1 arming lifetime, in M5 bars
N_HOLD = 3  # §7.2 hold window
N_SWEEP = 3  # §7.3 reintegration window
REV_STOP_ATR = 0.5  # §7.3 stop = extreme + 0.5 ATR
R_MIN = 1.0  # §8
MAX_HOLD_BARS = 48  # §10, four hours of M5
COOLDOWN_BARS = 6  # §9, per level
CONTRACT_SIZE = 100.0  # §11, ounces per lot
LOT_STEP = 0.01  # §11
MIN_LOTS = 0.01  # §11, volume_min
LEVERAGE_CAP = 100.0  # §11, 1:100
RISK_FRAC = 0.005  # §11
DXY_ADVERSE_FACTOR = 0.5  # §11, the risk is halved when ctx_dxy < 0
LEVEL_SIZE = 10.0  # §3

# §10, minutes since New York midnight. No entry in [16:30, 18:15) and an
# unconditional flat in [16:55, 18:00) — the daily break, and only it: a
# position opened in the evening lives its normal life and crosses NY midnight.
ENTRY_BLOCK_START = 16 * 60 + 30
ENTRY_BLOCK_END = 18 * 60 + 15
SESSION_FORCE_START = 16 * 60 + 55
SESSION_FORCE_END = 18 * 60

_ST_IDLE = 0
_ST_ARMED = 1
_ST_BROKEN = 2

# §13 — the trace contract, plus the cancel reason the reconciliation needs.
TRACE_COLUMNS: tuple[str, ...] = (
    "ts_decision",
    "event",
    "scenario",
    "level",
    "d",
    "atr",
    "v",
    "m",
    "a",
    "ctx_ema",
    "ctx_vwap",
    "ctx_dxy",
    "vwap",
    "ema50_h1",
    "dxy",
    "stop",
    "target",
    "r_est",
    "fill_px",
    "exit_px",
    "exit_reason",
    "cancel_reason",
)
_TRACE_DECIMALS: dict[str, int] = {
    "level": 3,
    "atr": 6,
    "v": 6,
    "m": 6,
    "a": 6,
    "vwap": 6,
    "ema50_h1": 6,
    "dxy": 6,
    "stop": 6,
    "target": 6,
    "r_est": 6,
    "fill_px": 3,
    "exit_px": 3,
}

__all__ = [
    "CANCEL_REASON_NAMES",
    "CONTRACT_SIZE",
    "EVENT_NAMES",
    "EXIT_REASON_NAMES",
    "SCENARIO_NAMES",
    "TRACE_COLUMNS",
    "BarContext",
    "X10Event",
    "X10Feed",
    "X10Machine",
    "trace_header",
    "trace_row",
]


# ═══════════════════════════════════════════════════════════════════════
# RECORDS
# ═══════════════════════════════════════════════════════════════════════


class BarContext(NamedTuple):
    """Everything the automaton reads at the close of one M5 bar (§3-§6)."""

    atr: float
    v: float
    m: float
    a: float
    l_inf: float
    l_sup: float
    vwap: float
    ema50_h1: float
    atr_h1: float
    dxy: float
    dxy_ema50_h1: float


@dataclass
class X10Event:
    """One row of the §13 trace, plus the fields the order router needs.

    ``bar_minute`` is the left label of the decision bar in minutes since the
    UTC epoch; ``lots`` and ``q`` are not part of the contract and are not
    written to the CSV — they exist so ``main.py`` can size a market order
    without recomputing anything.
    """

    bar_index: int
    bar_minute: int
    event: int
    scenario: int = SC_NONE
    level: float = NAN
    d: int = 0
    atr: float = NAN
    v: float = NAN
    m: float = NAN
    a: float = NAN
    ctx_ema: int = 0
    ctx_vwap: int = 0
    ctx_dxy: int = 0
    vwap: float = NAN
    ema50_h1: float = NAN
    dxy: float = NAN
    stop: float = NAN
    target: float = NAN
    r_est: float = NAN
    fill_px: float = NAN
    exit_px: float = NAN
    exit_reason: int = XR_NONE
    cancel_reason: int = CR_NONE
    lots: float = NAN
    q: int = 0


@dataclass
class _Pending:
    """An entry decided at the close of ``bar_index``, filling at the next bar."""

    bar_index: int
    bar_minute: int
    q: int
    d: int
    scenario: int
    level: float
    stop: float
    target: float
    r_est: float
    ctx_ema: int
    ctx_vwap: int
    ctx_dxy: int
    atr: float
    v: float
    m: float
    a: float
    vwap: float
    ema50_h1: float
    dxy: float
    extension_vwap: float = 0.0
    retest: float = 0.0


@dataclass
class _Position:
    """An open position — one at a time, all scenarios confounded (§9)."""

    q: int
    lots: float
    stop: float
    target: float
    fill_px: float
    level: float
    scenario: int
    r_est: float
    session_id: int
    extension_vwap: float = 0.0
    retest: float = 0.0
    bars_held: int = 0


# ═══════════════════════════════════════════════════════════════════════
# SMALL KERNELS
# ═══════════════════════════════════════════════════════════════════════


def _sign(x: float) -> int:
    """``signe()`` of §6.4: ``signe(0) = 0``, and an undefined value is neutral."""
    if not isfinite(x):
        return 0
    if x > 0.0:
        return 1
    if x < 0.0:
        return -1
    return 0


def _floor_lots(raw: float) -> float:
    """Lot step rounding of §11 — downwards only, and without an epsilon.

    ``NormalizeLots`` (``FxTradeHelpers.mqh:17-27``) has no epsilon either, and
    adding one here would buy 0.22 lots where the EA buys 0.21 on the same
    inputs.
    """
    if raw <= 0.0:
        return 0.0
    return floor(raw / LOT_STEP) * LOT_STEP


# ═══════════════════════════════════════════════════════════════════════
# THE AUTOMATON
# ═══════════════════════════════════════════════════════════════════════


class X10Machine:
    """The x10 state machine of §7, fed one M5 bar and its minutes at a time."""

    def __init__(
        self,
        z: float = 1.0,
        a_min: float = 0.2,
        k_s: float = 1.0,
        init_cash: float = 10_000.0,
        risk_frac: float = RISK_FRAC,
    ) -> None:
        self.z = float(z)
        self.a_min = float(a_min)
        self.k_s = float(k_s)
        self.risk_frac = float(risk_frac)
        self.equity = float(init_cash)
        self.events: list[X10Event] = []

        self._state = _ST_IDLE
        self._level = NAN
        self._d = 0
        self._i_arm = -1
        self._i_break = -1
        self._has_exc = False
        self._i_exc = -1
        self._extreme = NAN
        self._retest = 0.0

        self._pend: _Pending | None = None
        self._pos: _Position | None = None
        self._cooldown: dict[float, int] = {}
        self._prev_v = NAN
        self._exited_this_bar = False
        self._exit_to_patch: tuple[X10Event, int] | None = None

    # ── step 1 ────────────────────────────────────────────────────────
    def open_bar(
        self,
        bar_index: int,
        bar_minute: int,
        session_id: int,
        quote: MinuteQuote,
    ) -> None:
        """Fill the pending entry at this bar's open, then age the position (§9)."""
        self._exited_this_bar = False

        pending = self._pend
        if pending is not None:
            self._pend = None
            self._state = _ST_IDLE
            self._fill(bar_index, bar_minute, session_id, pending, quote)

        pos = self._pos
        if pos is not None:
            pos.bars_held += 1
            if session_id != pos.session_id:
                # The daily break was crossed with a position on: flatten at the
                # first price of the new session. A 16:55 close normally fires
                # first; this is the net under it.
                price = quote.bid_open if pos.q > 0 else quote.ask_open
                self._exit(bar_index, bar_minute, price, XR_SESSION)

    def _fill(
        self,
        bar_index: int,
        bar_minute: int,
        session_id: int,
        p: _Pending,
        quote: MinuteQuote,
    ) -> None:
        event = X10Event(
            bar_index=p.bar_index,
            bar_minute=p.bar_minute,
            event=EVENT_ENTRY,
            scenario=p.scenario,
            level=p.level,
            d=p.d,
            atr=p.atr,
            v=p.v,
            m=p.m,
            a=p.a,
            ctx_ema=p.ctx_ema,
            ctx_vwap=p.ctx_vwap,
            ctx_dxy=p.ctx_dxy,
            vwap=p.vwap,
            ema50_h1=p.ema50_h1,
            dxy=p.dxy,
            stop=p.stop,
            target=p.target,
            r_est=p.r_est,
            q=p.q,
        )
        self.events.append(event)

        # §2 — the fill bar must be the immediate successor of the decision bar.
        # A halt, a weekend or a session break between the two is a fill on a
        # price the decision never looked at, so the candidate is dropped.
        if bar_minute != p.bar_minute + M5_MINUTES:
            event.event = EVENT_CANCEL
            event.cancel_reason = CR_WINDOW
            self._cooldown[p.level] = p.bar_index + COOLDOWN_BARS
            return

        fill_px = quote.ask_open if p.q > 0 else quote.bid_open
        # §6.4 usage 3 / §11: an adverse dollar halves the risk, it never blocks
        # the trade — and it does so for the four scenarios.
        frac = self.risk_frac * DXY_ADVERSE_FACTOR if p.ctx_dxy < 0 else self.risk_frac
        stop_dist = abs(fill_px - p.stop)
        lots = 0.0
        if stop_dist > 0.0 and self.equity > 0.0:
            lots = _floor_lots(frac * self.equity / (stop_dist * CONTRACT_SIZE))
            cap = _floor_lots(self.equity * LEVERAGE_CAP / (CONTRACT_SIZE * fill_px))
            if lots > cap:
                lots = cap

        if lots < MIN_LOTS:
            event.event = EVENT_CANCEL
            event.cancel_reason = CR_SIZE
            self._cooldown[p.level] = bar_index + COOLDOWN_BARS
            return

        event.fill_px = fill_px
        event.lots = lots
        self._pos = _Position(
            q=p.q,
            lots=lots,
            stop=p.stop,
            target=p.target,
            fill_px=fill_px,
            level=p.level,
            scenario=p.scenario,
            r_est=p.r_est,
            session_id=session_id,
            extension_vwap=p.extension_vwap,
            retest=p.retest,
            bars_held=0,
        )

    # ── step 2 ────────────────────────────────────────────────────────
    def on_minute(self, bar_index: int, bar_minute: int, quote: MinuteQuote) -> None:
        """Resolve stop and target inside one M1 bar (§9).

        The trigger side only: a long is stopped and taken out on the **bid**, a
        short on the **ask**. The open is tested first, so a gap beyond either
        level fills at the open rather than at the level; inside the bar the
        stop wins a double contact, without exception.
        """
        pos = self._pos
        if pos is None or self._exited_this_bar:
            return

        price = NAN
        reason = XR_NONE
        if pos.q > 0:
            if quote.bid_open <= pos.stop:
                price, reason = quote.bid_open, XR_STOP
            elif quote.bid_open >= pos.target:
                price, reason = quote.bid_open, XR_TARGET
            elif quote.bid_low <= pos.stop:
                price, reason = pos.stop, XR_STOP
            elif quote.bid_high >= pos.target:
                price, reason = pos.target, XR_TARGET
        else:
            if quote.ask_open >= pos.stop:
                price, reason = quote.ask_open, XR_STOP
            elif quote.ask_open <= pos.target:
                price, reason = quote.ask_open, XR_TARGET
            elif quote.ask_high >= pos.stop:
                price, reason = pos.stop, XR_STOP
            elif quote.ask_low <= pos.target:
                price, reason = pos.target, XR_TARGET

        if reason != XR_NONE:
            self._exit(bar_index, bar_minute, price, reason)

    def _exit(self, bar_index: int, bar_minute: int, price: float, reason: int) -> None:
        pos = self._pos
        assert pos is not None
        self.equity += (price - pos.fill_px) * pos.q * pos.lots * CONTRACT_SIZE

        event = X10Event(
            bar_index=bar_index,
            bar_minute=bar_minute,
            event=EVENT_EXIT,
            scenario=pos.scenario,
            level=pos.level,
            d=self._d,
            stop=pos.stop,
            target=pos.target,
            r_est=pos.r_est,
            fill_px=pos.fill_px,
            exit_px=price,
            exit_reason=reason,
            lots=pos.lots,
            q=pos.q,
        )
        self.events.append(event)
        # The indicators of the exit bar are only known at its close; the row is
        # completed there, which is also where the reference reads them.
        self._exit_to_patch = (event, pos.q)

        self._cooldown[pos.level] = bar_index + COOLDOWN_BARS
        self._pos = None
        self._state = _ST_IDLE
        self._exited_this_bar = True

    # ── step 3 ────────────────────────────────────────────────────────
    def close_bar(
        self,
        bar_index: int,
        bar: Bar,
        minute_of_day: int,
        next_minute_of_day: int,
        half_spread: float,
        ctx: BarContext,
        quote: MinuteQuote,
    ) -> None:
        """Time exits of §10, then the automaton reads the close of ``bar``."""
        pos = self._pos
        if pos is not None and not self._exited_this_bar:
            forced = SESSION_FORCE_START <= minute_of_day < SESSION_FORCE_END
            if forced or pos.bars_held >= MAX_HOLD_BARS:
                price = quote.bid_close if pos.q > 0 else quote.ask_close
                self._exit(
                    bar_index, bar.start, price, XR_SESSION if forced else XR_TIME
                )

        if self._exit_to_patch is not None:
            event, q = self._exit_to_patch
            self._exit_to_patch = None
            event.atr = ctx.atr
            event.v = ctx.v
            event.m = ctx.m
            event.a = ctx.a
            event.ctx_ema = _sign(q * (bar.close - ctx.ema50_h1))
            event.ctx_vwap = _sign(q * (bar.close - ctx.vwap))
            event.ctx_dxy = _sign(-q * (ctx.dxy - ctx.dxy_ema50_h1))
            event.vwap = ctx.vwap
            event.ema50_h1 = ctx.ema50_h1
            event.dxy = ctx.dxy

        # ``v[t-1]`` of §7.3 is a plain array lookup in the reference: it must
        # advance on every M5 bar, frozen or not, in position or not.
        prev_v = self._prev_v
        self._prev_v = ctx.v

        if self._pos is not None or self._pend is not None:
            return
        if (
            not isfinite(ctx.atr)
            or ctx.atr <= 0.0
            or not isfinite(ctx.v)
            or not isfinite(ctx.m)
            or not isfinite(ctx.a)
        ):
            return

        if self._state == _ST_IDLE:
            self._try_arm(bar_index, bar, ctx)
            return

        self._run_armed(
            bar_index, bar, ctx, prev_v, next_minute_of_day, half_spread
        )

    # ── §7.1 ──────────────────────────────────────────────────────────
    def _try_arm(self, bar_index: int, bar: Bar, ctx: BarContext) -> None:
        close = bar.close
        atr = ctx.atr
        if len(self._cooldown) > 64:
            # At most one cooldown opens per bar and each lasts six, so the live
            # set is tiny; the dict is pruned rather than left to grow over the
            # seven years of the sample.
            self._cooldown = {
                lv: until for lv, until in self._cooldown.items() if until >= bar_index
            }
        cand_d = 0
        cand_level = NAN
        for dd in (1, -1):
            level = ctx.l_sup if dd == 1 else ctx.l_inf
            gap = dd * (level - close)
            if gap <= 0.0 or gap > self.z * atr:
                continue
            if dd * ctx.v < V_MIN or dd * ctx.a < self.a_min or dd * ctx.m <= 0.0:
                continue
            if self._cooldown.get(level, -1) >= bar_index:
                continue
            cand_d = dd
            cand_level = level
            break

        if cand_d == 0:
            return

        self._d = cand_d
        self._level = cand_level
        self._i_arm = bar_index
        self._state = _ST_ARMED
        self._retest = 0.0
        self._has_exc = False
        self._i_exc = -1
        self._extreme = NAN

        self._emit(bar_index, bar, ctx, EVENT_ARM, SC_NONE, cand_d)

        # §7.3: the reintegration window counts from the first bar whose extreme
        # goes past the level, which may be this one.
        bar_ext = bar.high if cand_d == 1 else bar.low
        if cand_d * (bar_ext - cand_level) > 0.0:
            self._has_exc = True
            self._i_exc = bar_index
            self._extreme = bar_ext

    # ── §7.2 / §7.3 ───────────────────────────────────────────────────
    def _run_armed(
        self,
        bar_index: int,
        bar: Bar,
        ctx: BarContext,
        prev_v: float,
        next_minute_of_day: int,
        half_spread: float,
    ) -> None:
        d = self._d
        level = self._level
        close = bar.close
        atr = ctx.atr
        scn_break = SC_BREAK_LONG if d == 1 else SC_BREAK_SHORT
        scn_rev = SC_REV_SHORT if d == 1 else SC_REV_LONG
        bar_ext = bar.high if d == 1 else bar.low

        emitted = False
        cancel_reason = CR_NONE
        cancel_scn = SC_NONE
        entry_q = 0
        entry_scn = SC_NONE
        entry_stop = NAN
        entry_target = NAN
        entry_ext = 0.0
        sweep_taken = False

        if self._state == _ST_ARMED:
            broke = d * (close - level) >= K_B * atr and d * (
                close - 0.5 * (bar.high + bar.low)
            ) > 0.0
            if broke:
                self._state = _ST_BROKEN
                self._i_break = bar_index
                self._extreme = bar_ext
                self._has_exc = False
                self._i_exc = -1
                self._retest = 0.0
                self._emit(bar_index, bar, ctx, EVENT_BREAK, scn_break, d)
                return

            if not self._has_exc:
                if d * (bar_ext - level) > 0.0:
                    self._has_exc = True
                    self._i_exc = bar_index
                    self._extreme = bar_ext
            elif d * (bar_ext - self._extreme) > 0.0:
                self._extreme = bar_ext

            if self._has_exc and bar_index - self._i_exc >= N_SWEEP:
                emitted = True
                cancel_reason = CR_NSWEEP
            elif self._has_exc and d * (close - level) < 0.0:
                deep = d * (self._extreme - level) >= S_MIN * atr
                decel = isfinite(prev_v) and d * ctx.v < d * prev_v
                flip = d * ctx.a <= -self.a_min
                if deep and decel and flip:
                    sweep_taken = True
                    entry_q = -d
                    entry_scn = scn_rev
                    entry_target = level - LEVEL_SIZE * d
                    entry_stop = self._extreme + REV_STOP_ATR * atr * d
                else:
                    emitted = True
                    cancel_reason = CR_SWEEP
            elif d * (level - close) > self.z * atr:
                emitted = True
                cancel_reason = CR_ZONE
            elif bar_index - self._i_arm >= N_ARM:
                emitted = True
                cancel_reason = CR_NARM

        else:  # _ST_BROKEN — the hold window of §7.2
            if d * (bar_ext - self._extreme) > 0.0:
                self._extreme = bar_ext
            if (d == 1 and bar.low <= level) or (d == -1 and bar.high >= level):
                self._retest = 1.0

            if d * (close - level) < 0.0:
                # Failed breakout: cancel it, then §7.2 points 2-5 — this very
                # bar is the candidate reintegration of a reversal, with no
                # re-arming and no N_sweep countdown.
                emitted = True
                cancel_reason = CR_HOLD
                cancel_scn = scn_break
                deep = d * (self._extreme - level) >= S_MIN * atr
                decel = isfinite(prev_v) and d * ctx.v < d * prev_v
                flip = d * ctx.a <= -self.a_min
                if deep and decel and flip:
                    sweep_taken = True
                    entry_q = -d
                    entry_scn = scn_rev
                    entry_target = level - LEVEL_SIZE * d
                    entry_stop = self._extreme + REV_STOP_ATR * atr * d
            elif bar_index - self._i_break >= N_HOLD:
                entry_q = d
                entry_scn = scn_break
                entry_target = level + LEVEL_SIZE * d
                entry_stop = level - d * self.k_s * atr

        if emitted:
            self._emit(
                bar_index,
                bar,
                ctx,
                EVENT_CANCEL,
                cancel_scn,
                d,
                cancel_reason=cancel_reason,
            )
            if not sweep_taken:
                self._state = _ST_IDLE
                self._cooldown[level] = bar_index + COOLDOWN_BARS
                return

        if entry_q == 0:
            return

        # ── entry candidate: context (§6.4), R (§8), window (§10) ─────
        q = entry_q
        ctx_ema = _sign(q * (close - ctx.ema50_h1))
        ctx_vwap = _sign(q * (close - ctx.vwap))
        ctx_dxy = _sign(-q * (ctx.dxy - ctx.dxy_ema50_h1))

        if sweep_taken:
            self._emit(
                bar_index,
                bar,
                ctx,
                EVENT_SWEEP,
                entry_scn,
                d,
                stop=entry_stop,
                target=entry_target,
                scores=(ctx_ema, ctx_vwap, ctx_dxy),
            )
            # §7.3, traced and measured, never blocking.
            if (
                isfinite(ctx.vwap)
                and isfinite(ctx.atr_h1)
                and abs(close - ctx.vwap) >= ctx.atr_h1
            ):
                entry_ext = 1.0

        refuse = CR_NONE
        r_est = NAN
        if entry_scn in (SC_BREAK_LONG, SC_BREAK_SHORT) and (
            ctx_ema <= 0 or ctx_vwap <= 0
        ):
            refuse = CR_CTX
        if refuse == CR_NONE:
            # §8: one half-spread, once, on each side of the ratio.
            denom = abs(close - entry_stop) + half_spread
            num = abs(entry_target - close) - half_spread
            r_est = num / denom if denom > 0.0 else NAN
            if not isfinite(r_est) or r_est < R_MIN:
                refuse = CR_R
        if refuse == CR_NONE and (
            ENTRY_BLOCK_START <= next_minute_of_day < ENTRY_BLOCK_END
        ):
            refuse = CR_WINDOW

        if refuse != CR_NONE:
            self._emit(
                bar_index,
                bar,
                ctx,
                EVENT_CANCEL,
                entry_scn,
                d,
                stop=entry_stop,
                target=entry_target,
                r_est=r_est,
                cancel_reason=refuse,
                scores=(ctx_ema, ctx_vwap, ctx_dxy),
            )
            self._state = _ST_IDLE
            self._cooldown[level] = bar_index + COOLDOWN_BARS
            return

        self._state = _ST_IDLE
        self._pend = _Pending(
            bar_index=bar_index,
            bar_minute=bar.start,
            q=entry_q,
            d=d,
            scenario=entry_scn,
            level=level,
            stop=entry_stop,
            target=entry_target,
            r_est=r_est,
            ctx_ema=ctx_ema,
            ctx_vwap=ctx_vwap,
            ctx_dxy=ctx_dxy,
            atr=ctx.atr,
            v=ctx.v,
            m=ctx.m,
            a=ctx.a,
            vwap=ctx.vwap,
            ema50_h1=ctx.ema50_h1,
            dxy=ctx.dxy,
            # §7.2 / §7.3 trade attributes: measured and carried, never blocking,
            # and outside the 21 columns of §13.
            extension_vwap=entry_ext,
            retest=self._retest,
        )

    # ── event emission ────────────────────────────────────────────────
    def _emit(
        self,
        bar_index: int,
        bar: Bar,
        ctx: BarContext,
        event: int,
        scenario: int,
        d: int,
        *,
        stop: float = NAN,
        target: float = NAN,
        r_est: float = NAN,
        cancel_reason: int = CR_NONE,
        scores: tuple[int, int, int] | None = None,
    ) -> None:
        """Append one trace row; the scores default to ``q = d`` (§6.4)."""
        if scores is None:
            scores = (
                _sign(d * (bar.close - ctx.ema50_h1)),
                _sign(d * (bar.close - ctx.vwap)),
                _sign(-d * (ctx.dxy - ctx.dxy_ema50_h1)),
            )
        self.events.append(
            X10Event(
                bar_index=bar_index,
                bar_minute=bar.start,
                event=event,
                scenario=scenario,
                level=self._level,
                d=d,
                atr=ctx.atr,
                v=ctx.v,
                m=ctx.m,
                a=ctx.a,
                ctx_ema=scores[0],
                ctx_vwap=scores[1],
                ctx_dxy=scores[2],
                vwap=ctx.vwap,
                ema50_h1=ctx.ema50_h1,
                dxy=ctx.dxy,
                stop=stop,
                target=target,
                r_est=r_est,
                cancel_reason=cancel_reason,
            )
        )


# ═══════════════════════════════════════════════════════════════════════
# THE FEED — minutes in, events out
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class _FeedState:
    """Published H1 context — the value of the last *closed* H1 bar (§6.1)."""

    ema50_h1: float = NAN
    atr_h1: float = NAN
    dxy: float = NAN
    dxy_ema50_h1: float = NAN


class X10Feed:
    """Drive the bars, the indicators and the automaton from raw minutes.

    One object owns the whole causal chain so that the QuantConnect algorithm
    and the parity test exercise **the same** code: ``main.py`` only translates
    a ``Slice`` into ``on_minute`` calls and turns the returned events into
    orders.

    Order inside one minute, and it is the whole contract:

    1. bins a gap left open are closed — M5 first (its decision), H1 second (its
       context), so an M5 bar and the H1 bar that close on the same instant obey
       ``align_h1_to_m5``: the 16:55 bar still sees the 15:00 hour;
    2. the hour of the DXY union index rolls over, after the M5 decisions of the
       hour that just ended and before those of the hour starting;
    3. a new M5 bin opens: the pending entry fills at its open and the position
       ages one bar;
    4. the minute resolves the open position;
    5. the minute is accumulated and, if it was the last of its bin, the bin is
       closed — M5 decision first, H1 context second, again.
    """

    def __init__(
        self,
        *,
        z: float = 1.0,
        a_min: float = 0.2,
        k_s: float = 1.0,
        init_cash: float = 10_000.0,
        risk_frac: float = RISK_FRAC,
    ) -> None:
        self.machine = X10Machine(
            z=z, a_min=a_min, k_s=k_s, init_cash=init_cash, risk_frac=risk_frac
        )
        self.bar_index = -1

        self._m5 = BarBuilder(M5_MINUTES)
        self._h1 = BarBuilder(H1_MINUTES)
        self._legs = {leg: BarBuilder(H1_MINUTES) for leg in DXY4_LEGS}

        self._atr5 = WilderAtr(ATR_PERIOD)
        self._kin = Kinematics()
        self._vwap = SessionTypicalMean()
        self._ema50 = Ema(EMA_SPAN_H1)
        self._atr_h1 = WilderAtr(ATR_PERIOD)
        self._basket = Dxy4Basket()
        self._dxy_ema = Ema(EMA_SPAN_H1)

        self._ctx = _FeedState()
        self._open_bin = -1
        self._dxy_hour: int | None = None
        self._last_quote: MinuteQuote | None = None
        self.last_context: BarContext | None = None

    # ── public API ────────────────────────────────────────────────────
    def on_minute(
        self,
        minute: int,
        gold: MinuteQuote | None,
        legs: dict[str, float] | None = None,
        live: bool = True,
    ) -> list[X10Event]:
        """Feed one minute; return the events it produced, in order."""
        first = len(self.machine.events)

        # 1. bins a gap left open
        bar = self._m5.roll(minute)
        if bar is not None:
            self._close_m5(bar, live)
        bar = self._h1.roll(minute)
        if bar is not None:
            self._close_h1(bar)

        # 2. the union hour of the DXY basket
        hour = minute // H1_MINUTES
        if self._dxy_hour is None:
            self._dxy_hour = hour
        elif hour != self._dxy_hour:
            self._close_dxy_hour()
            self._dxy_hour = hour

        if gold is not None:
            # 3. a new M5 bin: fills, ageing, session exit
            start = self._m5.bin_of(minute)
            if start != self._open_bin:
                self._open_bin = start
                self.bar_index += 1
                if live:
                    _, session_id = ny_fields(start)
                    self.machine.open_bar(self.bar_index, start, session_id, gold)

            # 4. resolve the open position on this minute
            if live:
                self.machine.on_minute(self.bar_index, self._open_bin, gold)

            # 5. accumulate, then close the bins this minute completes
            self._m5.add(minute, gold.mid_open, gold.mid_high, gold.mid_low, gold.mid_close)
            self._h1.add(minute, gold.mid_open, gold.mid_high, gold.mid_low, gold.mid_close)
            self._last_quote = gold

            bar = self._m5.seal()
            if bar is not None:
                self._close_m5(bar, live)
            bar = self._h1.seal()
            if bar is not None:
                self._close_h1(bar)

        if legs:
            for leg, price in legs.items():
                builder = self._legs.get(leg)
                if builder is not None:
                    builder.add(minute, price, price, price, price)

        return self.machine.events[first:]

    # ── internals ─────────────────────────────────────────────────────
    def _close_m5(self, bar: Bar, live: bool) -> None:
        minute_of_day, session_id = ny_fields(bar.start)
        atr = self._atr5.update(bar.high, bar.low, bar.close)
        v, m, a = self._kin.update(bar.close, atr)
        vwap = self._vwap.update(bar.high, bar.low, bar.close, session_id)
        l_inf, l_sup = x10_levels(bar.close)
        ctx = BarContext(
            atr=atr,
            v=v,
            m=m,
            a=a,
            l_inf=l_inf,
            l_sup=l_sup,
            vwap=vwap,
            ema50_h1=self._ctx.ema50_h1,
            atr_h1=self._ctx.atr_h1,
            dxy=self._ctx.dxy,
            dxy_ema50_h1=self._ctx.dxy_ema50_h1,
        )
        self.last_context = ctx

        quote = self._last_quote
        if not live or quote is None:
            return
        next_minute_of_day, _ = ny_fields(bar.start + M5_MINUTES)
        half_spread = 0.5 * (quote.ask_close - quote.bid_close)
        self.machine.close_bar(
            self.bar_index,
            bar,
            minute_of_day,
            next_minute_of_day,
            half_spread,
            ctx,
            quote,
        )

    def _close_h1(self, bar: Bar) -> None:
        self._ctx.ema50_h1 = self._ema50.update(bar.close)
        self._ctx.atr_h1 = self._atr_h1.update(bar.high, bar.low, bar.close)

    def _close_dxy_hour(self) -> None:
        closes: dict[str, float] = {}
        for leg, builder in self._legs.items():
            bar = builder.flush()
            if bar is not None:
                closes[leg] = bar.close
        if not closes:
            return  # no leg produced a bar: the union index has no entry here
        self._ctx.dxy = self._basket.update(closes)
        self._ctx.dxy_ema50_h1 = self._dxy_ema.update(self._ctx.dxy)


# ═══════════════════════════════════════════════════════════════════════
# §13 — TRACE FORMATTING
# ═══════════════════════════════════════════════════════════════════════


def _fmt(value: float, decimals: int | None) -> str:
    if value != value:  # NaN
        return ""
    if decimals is None:
        return str(value)
    return f"{round(value, decimals):.{decimals}f}"


def trace_header() -> str:
    return ",".join(TRACE_COLUMNS)


def trace_row(event: X10Event) -> str:
    """One CSV line of §13, in the frozen column order, with its precisions."""
    stamp = minute_to_utc(event.bar_minute).strftime("%Y-%m-%d %H:%M:%S")
    values = [
        stamp,
        EVENT_NAMES[event.event],
        SCENARIO_NAMES[event.scenario],
        _fmt(event.level, _TRACE_DECIMALS["level"]),
        str(int(event.d)),
        _fmt(event.atr, 6),
        _fmt(event.v, 6),
        _fmt(event.m, 6),
        _fmt(event.a, 6),
        str(int(event.ctx_ema)),
        str(int(event.ctx_vwap)),
        str(int(event.ctx_dxy)),
        _fmt(event.vwap, 6),
        _fmt(event.ema50_h1, 6),
        _fmt(event.dxy, 6),
        _fmt(event.stop, 6),
        _fmt(event.target, 6),
        _fmt(event.r_est, 6),
        _fmt(event.fill_px, 3),
        _fmt(event.exit_px, 3),
        EXIT_REASON_NAMES[event.exit_reason],
        CANCEL_REASON_NAMES[event.cancel_reason],
    ]
    return ",".join(values)
