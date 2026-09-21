"""State machine and intrabar execution of the XAUUSD x10 strategy (spec §7-§11).

One Numba kernel walks the M5 bars — where every decision is taken, at the
close, on information no newer than that close — and, whenever a position is
open, the M1 bars inside each M5 bar, where the stop and the target are
resolved. Nothing else runs: the indicators arrive precomputed from
``x10_kernels`` / ``x10_context``, and the portfolio is assembled downstream
from the order registry this kernel produces.

The kernel writes two registries, both plain ``float64`` matrices with named
column indices:

* **events** (``EVENT_COLUMNS``) — one row per ``ARM``/``BREAK``/``SWEEP``/
  ``ENTRY``/``EXIT``/``CANCEL``, the first 21 columns being exactly the trace
  contract of §13 in its order, plus ``cancel_reason`` and ``trade_id`` which
  §13 does not carry but a three-way reconciliation cannot debug without;
* **trades** (``TRADE_COLUMNS``) — one row per position, carrying the decision
  bar, both M1 fill indices, the sizing inputs and the realised exit.

Matrices rather than record arrays because the same layout has to be mirrored
in MQL5 and in QuantConnect Python, where a NumPy record dtype has no
counterpart; integer quantities (bar indices, codes, ``-1/0/+1`` scores) are
exactly representable in ``float64`` far beyond the size of any history here.

Everything the annexe A.1 freezes is a module constant, not an argument: only
``z``, ``a_min`` and ``k_s`` (the 27-point grid of A.2), the initial equity and
the risk fraction are passed in. A frozen parameter that a caller can override
is a frozen parameter waiting to be optimised.
"""

from __future__ import annotations

import numpy as np
from numba import njit

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

# Cancel reasons are NOT part of the §13 contract — the trace keeps
# ``exit_reason`` for the four exits of §10 — but an unexplained CANCEL is the
# hardest kind of divergence to attribute on rung 3 of the reconciliation
# ladder, so the registry names them.
CR_NONE = 0
CR_ZONE = 1  # §7.1 — price left the zone backwards
CR_NARM = 2  # §7.1 — N_arm bars without an event
CR_HOLD = 3  # §7.2 — a close came back through the level during the hold
CR_NSWEEP = 4  # §7.3 — no reintegration within N_sweep bars
CR_SWEEP = 5  # §7.3 — reintegration without depth / deceleration / flip
CR_CTX = 6  # §6.4 — breakout refused by EMA50 H1 or VWAP
CR_R = 7  # §8 — R < 1
CR_WINDOW = 8  # §10 — the fill bar falls in the 16:30-18:15 window
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
# REGISTRY LAYOUTS
# ═══════════════════════════════════════════════════════════════════════

# Columns 0..20 are the §13 trace, in order (``i_decision`` becomes
# ``ts_decision`` once the caller maps it back onto the M5 index).
E_I_DECISION = 0
E_EVENT = 1
E_SCENARIO = 2
E_LEVEL = 3
E_D = 4
E_ATR = 5
E_V = 6
E_M = 7
E_A = 8
E_CTX_EMA = 9
E_CTX_VWAP = 10
E_CTX_DXY = 11
E_VWAP = 12
E_EMA50_H1 = 13
E_DXY = 14
E_STOP = 15
E_TARGET = 16
E_R_EST = 17
E_FILL_PX = 18
E_EXIT_PX = 19
E_EXIT_REASON = 20
E_CANCEL_REASON = 21
E_TRADE_ID = 22
N_EVENT_FIELDS = 23

EVENT_COLUMNS: tuple[str, ...] = (
    "i_decision",
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
    "trade_id",
)

T_I_DECISION = 0
T_I_FILL_M1 = 1
T_I_EXIT_M1 = 2
T_I_EXIT_M5 = 3
T_Q = 4
T_SCENARIO = 5
T_LEVEL = 6
T_STOP = 7
T_TARGET = 8
T_R_EST = 9
T_LOTS = 10
T_FILL_PX = 11
T_EXIT_PX = 12
T_EXIT_REASON = 13
T_CTX_EMA = 14
T_CTX_VWAP = 15
T_CTX_DXY = 16
T_EXTENSION_VWAP = 17
T_RETEST = 18
T_RISK_AMOUNT = 19
T_EQUITY_BEFORE = 20
T_EQUITY_AFTER = 21
T_ATR = 22
T_SPREAD = 23
T_BARS_HELD = 24
N_TRADE_FIELDS = 25

TRADE_COLUMNS: tuple[str, ...] = (
    "i_decision",
    "i_fill_m1",
    "i_exit_m1",
    "i_exit_m5",
    "q",
    "scenario",
    "level",
    "stop",
    "target",
    "r_est",
    "lots",
    "fill_px",
    "exit_px",
    "exit_reason",
    "ctx_ema",
    "ctx_vwap",
    "ctx_dxy",
    "extension_vwap",
    "retest",
    "risk_amount",
    "equity_before",
    "equity_after",
    "atr",
    "spread",
    "bars_held",
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
LEVERAGE_CAP = 100.0  # 1:100
RISK_FRAC = 0.005  # §11
DXY_ADVERSE_FACTOR = 0.5  # §11, the risk is halved when ctx_dxy < 0
RISK_FRAC_DXY_ADVERSE = RISK_FRAC * DXY_ADVERSE_FACTOR  # 0.0025, §11
LEVEL_SIZE = 10.0  # §3

# §10, minutes since New York midnight. No entry in [16:30, 18:15) and an
# unconditional flat at 16:55.
ENTRY_BLOCK_START = 16 * 60 + 30
ENTRY_BLOCK_END = 18 * 60 + 15
SESSION_FORCE_MINUTE = 16 * 60 + 55
# ... and the flat applies until the session reopens at 18:00, not until
# midnight. ``minute_of_day`` is counted from New York midnight, so an
# unbounded ``>= 16:55`` test is equally true at 21:00 and would close every
# evening position on the bar that filled it.
SESSION_FORCE_END = 18 * 60

# §2 — ``fill[t] := open of bar t+1``: the bar five minutes later, not merely
# the next one that exists. Across a hole in the feed the two differ.
M5_MINUTES = 5

# Cooldowns last 6 bars and at most one is opened per bar, so a handful of
# slots always suffices; the ring is sized well above that and scanned in full.
_COOLDOWN_SLOTS = 16

# Internal states of the automaton (§7).
_ST_IDLE = 0
_ST_ARMED = 1
_ST_BROKEN = 2

__all__ = [
    "CANCEL_REASON_NAMES",
    "CONTRACT_SIZE",
    "COOLDOWN_BARS",
    "DXY_ADVERSE_FACTOR",
    "EVENT_COLUMNS",
    "EVENT_NAMES",
    "EXIT_REASON_NAMES",
    "K_B",
    "LEVERAGE_CAP",
    "LOT_STEP",
    "MAX_HOLD_BARS",
    "MIN_LOTS",
    "N_ARM",
    "N_HOLD",
    "N_SWEEP",
    "RISK_FRAC",
    "R_MIN",
    "SCENARIO_NAMES",
    "S_MIN",
    "TRADE_COLUMNS",
    "V_MIN",
    "run_engine",
    "x10_engine_nb",
]


# ═══════════════════════════════════════════════════════════════════════
# SMALL KERNELS
# ═══════════════════════════════════════════════════════════════════════


@njit(nogil=True, cache=True)
def _sign_nb(x: float) -> int:
    """``signe()`` of §6.4: ``signe(0) = 0``, and an undefined value is neutral."""
    if not np.isfinite(x):
        return 0
    if x > 0.0:
        return 1
    if x < 0.0:
        return -1
    return 0


@njit(nogil=True, cache=True)
def _floor_lots_nb(raw: float) -> float:
    """Lot step rounding of §11 — ``MathFloor(raw/step)*step``, downwards only.

    No epsilon: ``NormalizeLots`` (``FxTradeHelpers.mqh:17-27``) has none either,
    and an epsilon here would make the Python engine buy 0.22 lots where the EA
    buys 0.21 on the same inputs.
    """
    if raw <= 0.0:
        return 0.0
    return np.floor(raw / LOT_STEP) * LOT_STEP


# ═══════════════════════════════════════════════════════════════════════
# THE ENGINE
# ═══════════════════════════════════════════════════════════════════════


@njit(nogil=True, cache=True)
def x10_engine_nb(
    m5_open: np.ndarray,
    m5_high: np.ndarray,
    m5_low: np.ndarray,
    m5_close: np.ndarray,
    atr: np.ndarray,
    v: np.ndarray,
    m: np.ndarray,
    a: np.ndarray,
    l_inf: np.ndarray,
    l_sup: np.ndarray,
    vwap: np.ndarray,
    ema50_h1: np.ndarray,
    atr_h1: np.ndarray,
    dxy: np.ndarray,
    dxy_ema50_h1: np.ndarray,
    minute_of_day: np.ndarray,
    session_id: np.ndarray,
    bar_minute: np.ndarray,
    spread: np.ndarray,
    m1_open: np.ndarray,
    m1_high: np.ndarray,
    m1_low: np.ndarray,
    m1_close: np.ndarray,
    m1_start: np.ndarray,
    m1_end: np.ndarray,
    z: float,
    a_min: float,
    k_s: float,
    init_cash: float,
    risk_frac: float,
    events: np.ndarray,
    trades: np.ndarray,
) -> tuple[int, int]:
    """Run the x10 automaton over M5 decisions and M1 resolution (§7-§11).

    ``events`` and ``trades`` are preallocated output matrices of shape
    ``(capacity, N_EVENT_FIELDS)`` and ``(capacity, N_TRADE_FIELDS)``; the
    function returns ``(n_events, n_trades)``, or ``(-1, -1)`` when either
    capacity would be exceeded — the caller reallocates and retries, which is
    what ``run_engine`` does.

    Ordering inside one M5 bar ``i``, and it is the whole contract:

    1. an entry decided at ``i-1`` fills at the open of the first M1 of ``i``
       (§2, §9), which is also where the lots of §11 are computed and where a
       position too small to size cancels;
    2. an open position is resolved M1 by M1 — stop first on a double contact,
       open price on a gap (§9) — then by the time exits of §10;
    3. the automaton reads the close of ``i`` and may arm, break, sweep, cancel
       or decide an entry for ``i+1``.

    No decision is taken while a position is open or an entry is pending (one
    position at a time, §9), on a bar where ``atr``, ``v``, ``m`` or ``a`` is
    undefined (§5), or on the last bar of the input — which has no ``i+1`` to
    fill in, and is also what makes the whole scan causal.

    ``bar_minute`` is the start of each M5 bar in whole minutes on a monotonic
    clock (epoch minutes in production). It exists for one test: an entry fills
    only if the next bar starts exactly five minutes later. Every other counter
    of the spec — ``N_arm``, ``N_hold``, ``N_sweep``, the 48-bar life and the
    cooldown — keeps counting *existing* bars.
    """
    n5 = len(m5_close)
    cap_ev = events.shape[0]
    cap_tr = trades.shape[0]

    n_ev = 0
    n_tr = 0

    # ── automaton state ────────────────────────────────────────────────
    state = _ST_IDLE
    level = np.nan
    d = 0
    i_arm = -1
    i_break = -1
    has_exc = False
    i_exc = -1
    extreme = np.nan
    retest = 0.0

    # ── entry decided at the previous bar, filling at this one ────────
    pend = False
    p_i = -1
    p_q = 0
    p_scn = SC_NONE
    p_level = np.nan
    p_stop = np.nan
    p_target = np.nan
    p_r = np.nan
    p_ctx_ema = 0
    p_ctx_vwap = 0
    p_ctx_dxy = 0
    p_ext = 0.0
    p_retest = 0.0
    p_atr = np.nan

    # ── open position ─────────────────────────────────────────────────
    in_pos = False
    pos_q = 0
    pos_lots = 0.0
    pos_stop = np.nan
    pos_target = np.nan
    pos_fill = np.nan
    pos_row = -1
    pos_session = -1
    bars_held = 0

    equity = init_cash

    cd_level = np.full(_COOLDOWN_SLOTS, np.nan)
    cd_until = np.full(_COOLDOWN_SLOTS, -1, dtype=np.int64)
    cd_next = 0

    for i in range(n5):
        # Worst case within one bar: ENTRY (or CANCEL size), EXIT, CANCEL,
        # SWEEP, ENTRY-decision CANCEL. Eight rows of head room, checked once.
        if n_ev + 8 > cap_ev or n_tr + 2 > cap_tr:
            return -1, -1

        half = 0.5 * spread[i]
        j0 = m1_start[i]
        j1 = m1_end[i]

        # ── 1. pending entry: fill at the open of this bar (§2, §9, §11) ──
        if pend:
            mid_open = m1_open[j0]
            fill_px = mid_open + half if p_q > 0 else mid_open - half
            # §6.4 usage 3 / §11: an adverse dollar halves the risk, it never
            # blocks the trade — and it does so for the four scenarios.
            frac = risk_frac * DXY_ADVERSE_FACTOR if p_ctx_dxy < 0 else risk_frac
            stop_dist = abs(fill_px - p_stop)
            lots = 0.0
            if stop_dist > 0.0 and equity > 0.0:
                lots = _floor_lots_nb(frac * equity / (stop_dist * CONTRACT_SIZE))
                cap_lots = _floor_lots_nb(equity * LEVERAGE_CAP / (CONTRACT_SIZE * fill_px))
                if lots > cap_lots:
                    lots = cap_lots

            k = n_ev
            n_ev += 1
            events[k, E_I_DECISION] = p_i
            events[k, E_SCENARIO] = p_scn
            events[k, E_LEVEL] = p_level
            events[k, E_D] = d
            events[k, E_ATR] = atr[p_i]
            events[k, E_V] = v[p_i]
            events[k, E_M] = m[p_i]
            events[k, E_A] = a[p_i]
            events[k, E_CTX_EMA] = p_ctx_ema
            events[k, E_CTX_VWAP] = p_ctx_vwap
            events[k, E_CTX_DXY] = p_ctx_dxy
            events[k, E_VWAP] = vwap[p_i]
            events[k, E_EMA50_H1] = ema50_h1[p_i]
            events[k, E_DXY] = dxy[p_i]
            events[k, E_STOP] = p_stop
            events[k, E_TARGET] = p_target
            events[k, E_R_EST] = p_r
            events[k, E_EXIT_PX] = np.nan
            events[k, E_EXIT_REASON] = XR_NONE

            if lots < MIN_LOTS:
                events[k, E_EVENT] = EVENT_CANCEL
                events[k, E_FILL_PX] = np.nan
                events[k, E_CANCEL_REASON] = CR_SIZE
                events[k, E_TRADE_ID] = np.nan
                cd_level[cd_next] = p_level
                cd_until[cd_next] = i + COOLDOWN_BARS
                cd_next = (cd_next + 1) % _COOLDOWN_SLOTS
            else:
                row = n_tr
                n_tr += 1
                events[k, E_EVENT] = EVENT_ENTRY
                events[k, E_FILL_PX] = fill_px
                events[k, E_CANCEL_REASON] = CR_NONE
                events[k, E_TRADE_ID] = row

                trades[row, T_I_DECISION] = p_i
                trades[row, T_I_FILL_M1] = j0
                trades[row, T_I_EXIT_M1] = np.nan
                trades[row, T_I_EXIT_M5] = np.nan
                trades[row, T_Q] = p_q
                trades[row, T_SCENARIO] = p_scn
                trades[row, T_LEVEL] = p_level
                trades[row, T_STOP] = p_stop
                trades[row, T_TARGET] = p_target
                trades[row, T_R_EST] = p_r
                trades[row, T_LOTS] = lots
                trades[row, T_FILL_PX] = fill_px
                trades[row, T_EXIT_PX] = np.nan
                trades[row, T_EXIT_REASON] = XR_NONE
                trades[row, T_CTX_EMA] = p_ctx_ema
                trades[row, T_CTX_VWAP] = p_ctx_vwap
                trades[row, T_CTX_DXY] = p_ctx_dxy
                trades[row, T_EXTENSION_VWAP] = p_ext
                trades[row, T_RETEST] = p_retest
                trades[row, T_RISK_AMOUNT] = lots * stop_dist * CONTRACT_SIZE
                trades[row, T_EQUITY_BEFORE] = equity
                trades[row, T_EQUITY_AFTER] = np.nan
                trades[row, T_ATR] = p_atr
                trades[row, T_SPREAD] = spread[i]
                trades[row, T_BARS_HELD] = np.nan

                in_pos = True
                pos_q = p_q
                pos_lots = lots
                pos_stop = p_stop
                pos_target = p_target
                pos_fill = fill_px
                pos_row = row
                pos_session = session_id[i]
                bars_held = 0

            state = _ST_IDLE
            pend = False

        # ── 2. resolve the open position on the M1 bars of this bar (§9) ──
        if in_pos:
            bars_held += 1
            exit_px = np.nan
            exit_reason = XR_NONE
            j_exit = -1

            if session_id[i] != pos_session:
                # The daily break was crossed with a position on: flatten at
                # the first price of the new session (§10). A 16:55 close
                # normally fires first; this is the net under it.
                j_exit = j0
                exit_px = m1_open[j0] - half if pos_q > 0 else m1_open[j0] + half
                exit_reason = XR_SESSION
            else:
                for j in range(j0, j1):
                    if pos_q > 0:
                        o_t = m1_open[j] - half
                        h_t = m1_high[j] - half
                        l_t = m1_low[j] - half
                        if o_t <= pos_stop:
                            j_exit = j
                            exit_px = o_t
                            exit_reason = XR_STOP
                        elif o_t >= pos_target:
                            j_exit = j
                            exit_px = o_t
                            exit_reason = XR_TARGET
                        elif l_t <= pos_stop:
                            j_exit = j
                            exit_px = pos_stop
                            exit_reason = XR_STOP
                        elif h_t >= pos_target:
                            j_exit = j
                            exit_px = pos_target
                            exit_reason = XR_TARGET
                    else:
                        o_t = m1_open[j] + half
                        h_t = m1_high[j] + half
                        l_t = m1_low[j] + half
                        if o_t >= pos_stop:
                            j_exit = j
                            exit_px = o_t
                            exit_reason = XR_STOP
                        elif o_t <= pos_target:
                            j_exit = j
                            exit_px = o_t
                            exit_reason = XR_TARGET
                        elif h_t >= pos_stop:
                            j_exit = j
                            exit_px = pos_stop
                            exit_reason = XR_STOP
                        elif l_t <= pos_target:
                            j_exit = j
                            exit_px = pos_target
                            exit_reason = XR_TARGET
                    if j_exit >= 0:
                        break

            # §10: flat from 16:55 up to the 18:00 reopen — a window, not a
            # half-line. Outside it the position lives on, evening included.
            session_close = (
                minute_of_day[i] >= SESSION_FORCE_MINUTE
                and minute_of_day[i] < SESSION_FORCE_END
            )
            if j_exit < 0 and (session_close or bars_held >= MAX_HOLD_BARS or i == n5 - 1):
                j_exit = j1 - 1
                exit_px = (
                    m1_close[j1 - 1] - half if pos_q > 0 else m1_close[j1 - 1] + half
                )
                if session_close or i == n5 - 1:
                    exit_reason = XR_SESSION
                else:
                    exit_reason = XR_TIME

            if j_exit >= 0:
                pnl = (exit_px - pos_fill) * pos_q * pos_lots * CONTRACT_SIZE
                equity += pnl

                trades[pos_row, T_I_EXIT_M1] = j_exit
                trades[pos_row, T_I_EXIT_M5] = i
                trades[pos_row, T_EXIT_PX] = exit_px
                trades[pos_row, T_EXIT_REASON] = exit_reason
                trades[pos_row, T_EQUITY_AFTER] = equity
                trades[pos_row, T_BARS_HELD] = bars_held

                q = pos_q
                k = n_ev
                n_ev += 1
                events[k, E_I_DECISION] = i
                events[k, E_EVENT] = EVENT_EXIT
                events[k, E_SCENARIO] = trades[pos_row, T_SCENARIO]
                events[k, E_LEVEL] = trades[pos_row, T_LEVEL]
                events[k, E_D] = d
                events[k, E_ATR] = atr[i]
                events[k, E_V] = v[i]
                events[k, E_M] = m[i]
                events[k, E_A] = a[i]
                events[k, E_CTX_EMA] = _sign_nb(q * (m5_close[i] - ema50_h1[i]))
                events[k, E_CTX_VWAP] = _sign_nb(q * (m5_close[i] - vwap[i]))
                events[k, E_CTX_DXY] = _sign_nb(-q * (dxy[i] - dxy_ema50_h1[i]))
                events[k, E_VWAP] = vwap[i]
                events[k, E_EMA50_H1] = ema50_h1[i]
                events[k, E_DXY] = dxy[i]
                events[k, E_STOP] = pos_stop
                events[k, E_TARGET] = pos_target
                events[k, E_R_EST] = trades[pos_row, T_R_EST]
                events[k, E_FILL_PX] = pos_fill
                events[k, E_EXIT_PX] = exit_px
                events[k, E_EXIT_REASON] = exit_reason
                events[k, E_CANCEL_REASON] = CR_NONE
                events[k, E_TRADE_ID] = pos_row

                cd_level[cd_next] = trades[pos_row, T_LEVEL]
                cd_until[cd_next] = i + COOLDOWN_BARS
                cd_next = (cd_next + 1) % _COOLDOWN_SLOTS

                in_pos = False
                state = _ST_IDLE

        # ── 3. decisions, at the close of this bar (§2) ───────────────
        if in_pos or pend or i == n5 - 1:
            continue
        if (
            not np.isfinite(atr[i])
            or atr[i] <= 0.0
            or not np.isfinite(v[i])
            or not np.isfinite(m[i])
            or not np.isfinite(a[i])
        ):
            continue

        close = m5_close[i]
        atr_i = atr[i]

        # Everything below shares the same snapshot write; ``q`` is the sign of
        # the trade under consideration (§6.4), which is ``d`` until a reversal
        # is identified.
        if state == _ST_IDLE:
            cand_d = 0
            cand_level = np.nan
            for side in range(2):
                dd = 1 if side == 0 else -1
                lv = l_sup[i] if dd == 1 else l_inf[i]
                gap = dd * (lv - close)
                if gap <= 0.0 or gap > z * atr_i:
                    continue
                if dd * v[i] < V_MIN or dd * a[i] < a_min or dd * m[i] <= 0.0:
                    continue
                blocked = False
                for c in range(_COOLDOWN_SLOTS):
                    if cd_until[c] >= i and cd_level[c] == lv:
                        blocked = True
                        break
                if blocked:
                    continue
                cand_d = dd
                cand_level = lv
                break

            if cand_d != 0:
                d = cand_d
                level = cand_level
                i_arm = i
                state = _ST_ARMED
                retest = 0.0
                has_exc = False
                i_exc = -1
                extreme = np.nan

                q = d
                k = n_ev
                n_ev += 1
                events[k, E_I_DECISION] = i
                events[k, E_EVENT] = EVENT_ARM
                events[k, E_SCENARIO] = SC_NONE
                events[k, E_LEVEL] = level
                events[k, E_D] = d
                events[k, E_ATR] = atr_i
                events[k, E_V] = v[i]
                events[k, E_M] = m[i]
                events[k, E_A] = a[i]
                events[k, E_CTX_EMA] = _sign_nb(q * (close - ema50_h1[i]))
                events[k, E_CTX_VWAP] = _sign_nb(q * (close - vwap[i]))
                events[k, E_CTX_DXY] = _sign_nb(-q * (dxy[i] - dxy_ema50_h1[i]))
                events[k, E_VWAP] = vwap[i]
                events[k, E_EMA50_H1] = ema50_h1[i]
                events[k, E_DXY] = dxy[i]
                events[k, E_STOP] = np.nan
                events[k, E_TARGET] = np.nan
                events[k, E_R_EST] = np.nan
                events[k, E_FILL_PX] = np.nan
                events[k, E_EXIT_PX] = np.nan
                events[k, E_EXIT_REASON] = XR_NONE
                events[k, E_CANCEL_REASON] = CR_NONE
                events[k, E_TRADE_ID] = np.nan

                # §7.3: the reintegration window counts from the first bar
                # whose extreme goes past the level, which may be this one.
                ext_i = m5_high[i] if d == 1 else m5_low[i]
                if d * (ext_i - level) > 0.0:
                    has_exc = True
                    i_exc = i
                    extreme = ext_i
            continue

        # ── the automaton is busy on ``level`` / ``d`` ─────────────────
        scn_break = SC_BREAK_LONG if d == 1 else SC_BREAK_SHORT
        scn_rev = SC_REV_SHORT if d == 1 else SC_REV_LONG
        bar_ext = m5_high[i] if d == 1 else m5_low[i]

        emitted = False
        cancel_reason = CR_NONE
        cancel_scn = SC_NONE
        entry_q = 0
        entry_scn = SC_NONE
        entry_stop = np.nan
        entry_target = np.nan
        entry_ext = 0.0
        sweep_taken = False

        if state == _ST_ARMED:
            broke = d * (close - level) >= K_B * atr_i and d * (
                close - 0.5 * (m5_high[i] + m5_low[i])
            ) > 0.0
            if broke:
                state = _ST_BROKEN
                i_break = i
                extreme = bar_ext
                has_exc = False
                i_exc = -1
                retest = 0.0

                q = d
                k = n_ev
                n_ev += 1
                events[k, E_I_DECISION] = i
                events[k, E_EVENT] = EVENT_BREAK
                events[k, E_SCENARIO] = scn_break
                events[k, E_LEVEL] = level
                events[k, E_D] = d
                events[k, E_ATR] = atr_i
                events[k, E_V] = v[i]
                events[k, E_M] = m[i]
                events[k, E_A] = a[i]
                events[k, E_CTX_EMA] = _sign_nb(q * (close - ema50_h1[i]))
                events[k, E_CTX_VWAP] = _sign_nb(q * (close - vwap[i]))
                events[k, E_CTX_DXY] = _sign_nb(-q * (dxy[i] - dxy_ema50_h1[i]))
                events[k, E_VWAP] = vwap[i]
                events[k, E_EMA50_H1] = ema50_h1[i]
                events[k, E_DXY] = dxy[i]
                events[k, E_STOP] = np.nan
                events[k, E_TARGET] = np.nan
                events[k, E_R_EST] = np.nan
                events[k, E_FILL_PX] = np.nan
                events[k, E_EXIT_PX] = np.nan
                events[k, E_EXIT_REASON] = XR_NONE
                events[k, E_CANCEL_REASON] = CR_NONE
                events[k, E_TRADE_ID] = np.nan
                continue

            if not has_exc:
                if d * (bar_ext - level) > 0.0:
                    has_exc = True
                    i_exc = i
                    extreme = bar_ext
            elif d * (bar_ext - extreme) > 0.0:
                extreme = bar_ext

            if has_exc and i - i_exc >= N_SWEEP:
                emitted = True
                cancel_reason = CR_NSWEEP
            elif has_exc and d * (close - level) < 0.0:
                deep = d * (extreme - level) >= S_MIN * atr_i
                decel = i > 0 and np.isfinite(v[i - 1]) and d * v[i] < d * v[i - 1]
                flip = d * a[i] <= -a_min
                if deep and decel and flip:
                    sweep_taken = True
                    entry_q = -d
                    entry_scn = scn_rev
                    entry_target = level - LEVEL_SIZE * d
                    entry_stop = extreme + REV_STOP_ATR * atr_i * d
                else:
                    emitted = True
                    cancel_reason = CR_SWEEP
            elif d * (level - close) > z * atr_i:
                emitted = True
                cancel_reason = CR_ZONE
            elif i - i_arm >= N_ARM:
                emitted = True
                cancel_reason = CR_NARM

        else:  # _ST_BROKEN — the hold window of §7.2
            if d * (bar_ext - extreme) > 0.0:
                extreme = bar_ext
            if (d == 1 and m5_low[i] <= level) or (d == -1 and m5_high[i] >= level):
                retest = 1.0

            if d * (close - level) < 0.0:
                # Failed breakout: cancel it, then §7.2 point 2-5 — this very
                # bar is the candidate reintegration of a reversal, with no
                # re-arming and no N_sweep countdown.
                emitted = True
                cancel_reason = CR_HOLD
                cancel_scn = scn_break
                deep = d * (extreme - level) >= S_MIN * atr_i
                decel = i > 0 and np.isfinite(v[i - 1]) and d * v[i] < d * v[i - 1]
                flip = d * a[i] <= -a_min
                if deep and decel and flip:
                    sweep_taken = True
                    entry_q = -d
                    entry_scn = scn_rev
                    entry_target = level - LEVEL_SIZE * d
                    entry_stop = extreme + REV_STOP_ATR * atr_i * d
            elif i - i_break >= N_HOLD:
                entry_q = d
                entry_scn = scn_break
                entry_target = level + LEVEL_SIZE * d
                entry_stop = level - d * k_s * atr_i

        if emitted:
            q = d
            k = n_ev
            n_ev += 1
            events[k, E_I_DECISION] = i
            events[k, E_EVENT] = EVENT_CANCEL
            events[k, E_SCENARIO] = cancel_scn
            events[k, E_LEVEL] = level
            events[k, E_D] = d
            events[k, E_ATR] = atr_i
            events[k, E_V] = v[i]
            events[k, E_M] = m[i]
            events[k, E_A] = a[i]
            events[k, E_CTX_EMA] = _sign_nb(q * (close - ema50_h1[i]))
            events[k, E_CTX_VWAP] = _sign_nb(q * (close - vwap[i]))
            events[k, E_CTX_DXY] = _sign_nb(-q * (dxy[i] - dxy_ema50_h1[i]))
            events[k, E_VWAP] = vwap[i]
            events[k, E_EMA50_H1] = ema50_h1[i]
            events[k, E_DXY] = dxy[i]
            events[k, E_STOP] = np.nan
            events[k, E_TARGET] = np.nan
            events[k, E_R_EST] = np.nan
            events[k, E_FILL_PX] = np.nan
            events[k, E_EXIT_PX] = np.nan
            events[k, E_EXIT_REASON] = XR_NONE
            events[k, E_CANCEL_REASON] = cancel_reason
            events[k, E_TRADE_ID] = np.nan
            if not sweep_taken:
                state = _ST_IDLE
                cd_level[cd_next] = level
                cd_until[cd_next] = i + COOLDOWN_BARS
                cd_next = (cd_next + 1) % _COOLDOWN_SLOTS
                continue

        if entry_q == 0:
            continue

        # ── entry candidate: context (§6.4), R (§8), window (§10) ─────
        q = entry_q
        ctx_ema = _sign_nb(q * (close - ema50_h1[i]))
        ctx_vwap = _sign_nb(q * (close - vwap[i]))
        ctx_dxy = _sign_nb(-q * (dxy[i] - dxy_ema50_h1[i]))

        if sweep_taken:
            k = n_ev
            n_ev += 1
            events[k, E_I_DECISION] = i
            events[k, E_EVENT] = EVENT_SWEEP
            events[k, E_SCENARIO] = entry_scn
            events[k, E_LEVEL] = level
            events[k, E_D] = d
            events[k, E_ATR] = atr_i
            events[k, E_V] = v[i]
            events[k, E_M] = m[i]
            events[k, E_A] = a[i]
            events[k, E_CTX_EMA] = ctx_ema
            events[k, E_CTX_VWAP] = ctx_vwap
            events[k, E_CTX_DXY] = ctx_dxy
            events[k, E_VWAP] = vwap[i]
            events[k, E_EMA50_H1] = ema50_h1[i]
            events[k, E_DXY] = dxy[i]
            events[k, E_STOP] = entry_stop
            events[k, E_TARGET] = entry_target
            events[k, E_R_EST] = np.nan
            events[k, E_FILL_PX] = np.nan
            events[k, E_EXIT_PX] = np.nan
            events[k, E_EXIT_REASON] = XR_NONE
            events[k, E_CANCEL_REASON] = CR_NONE
            events[k, E_TRADE_ID] = np.nan
            # §7.3, traced and measured, never blocking.
            if (
                np.isfinite(vwap[i])
                and np.isfinite(atr_h1[i])
                and abs(close - vwap[i]) >= atr_h1[i]
            ):
                entry_ext = 1.0

        refuse = CR_NONE
        r_est = np.nan
        if entry_scn == SC_BREAK_LONG or entry_scn == SC_BREAK_SHORT:
            if ctx_ema <= 0 or ctx_vwap <= 0:
                refuse = CR_CTX
        if refuse == CR_NONE:
            # §8: one half-spread, once, on each side of the ratio.
            denom = abs(close - entry_stop) + half
            num = abs(entry_target - close) - half
            r_est = num / denom if denom > 0.0 else np.nan
            if not np.isfinite(r_est) or r_est < R_MIN:
                refuse = CR_R
        if refuse == CR_NONE and bar_minute[i + 1] - bar_minute[i] != M5_MINUTES:
            # §2: the fill bar is the one five minutes later. After a hole in
            # the feed the next existing bar is not it, and filling there would
            # execute a decision at a price it never saw.
            refuse = CR_WINDOW
        if refuse == CR_NONE and (
            minute_of_day[i + 1] >= ENTRY_BLOCK_START
            and minute_of_day[i + 1] < ENTRY_BLOCK_END
        ):
            refuse = CR_WINDOW

        if refuse != CR_NONE:
            k = n_ev
            n_ev += 1
            events[k, E_I_DECISION] = i
            events[k, E_EVENT] = EVENT_CANCEL
            events[k, E_SCENARIO] = entry_scn
            events[k, E_LEVEL] = level
            events[k, E_D] = d
            events[k, E_ATR] = atr_i
            events[k, E_V] = v[i]
            events[k, E_M] = m[i]
            events[k, E_A] = a[i]
            events[k, E_CTX_EMA] = ctx_ema
            events[k, E_CTX_VWAP] = ctx_vwap
            events[k, E_CTX_DXY] = ctx_dxy
            events[k, E_VWAP] = vwap[i]
            events[k, E_EMA50_H1] = ema50_h1[i]
            events[k, E_DXY] = dxy[i]
            events[k, E_STOP] = entry_stop
            events[k, E_TARGET] = entry_target
            events[k, E_R_EST] = r_est
            events[k, E_FILL_PX] = np.nan
            events[k, E_EXIT_PX] = np.nan
            events[k, E_EXIT_REASON] = XR_NONE
            events[k, E_CANCEL_REASON] = refuse
            events[k, E_TRADE_ID] = np.nan

            state = _ST_IDLE
            cd_level[cd_next] = level
            cd_until[cd_next] = i + COOLDOWN_BARS
            cd_next = (cd_next + 1) % _COOLDOWN_SLOTS
            continue

        pend = True
        state = _ST_IDLE
        p_i = i
        p_q = entry_q
        p_scn = entry_scn
        p_level = level
        p_stop = entry_stop
        p_target = entry_target
        p_r = r_est
        p_ctx_ema = ctx_ema
        p_ctx_vwap = ctx_vwap
        p_ctx_dxy = ctx_dxy
        p_ext = entry_ext
        p_retest = retest
        p_atr = atr_i

    return n_ev, n_tr


# ═══════════════════════════════════════════════════════════════════════
# PYTHON WRAPPER — buffer allocation
# ═══════════════════════════════════════════════════════════════════════


def run_engine(
    m5_open: np.ndarray,
    m5_high: np.ndarray,
    m5_low: np.ndarray,
    m5_close: np.ndarray,
    atr: np.ndarray,
    v: np.ndarray,
    m: np.ndarray,
    a: np.ndarray,
    l_inf: np.ndarray,
    l_sup: np.ndarray,
    vwap: np.ndarray,
    ema50_h1: np.ndarray,
    atr_h1: np.ndarray,
    dxy: np.ndarray,
    dxy_ema50_h1: np.ndarray,
    minute_of_day: np.ndarray,
    session_id: np.ndarray,
    bar_minute: np.ndarray,
    spread: np.ndarray,
    m1_open: np.ndarray,
    m1_high: np.ndarray,
    m1_low: np.ndarray,
    m1_close: np.ndarray,
    m1_start: np.ndarray,
    m1_end: np.ndarray,
    *,
    z: float = 1.0,
    a_min: float = 0.2,
    k_s: float = 1.0,
    init_cash: float = 10_000.0,
    risk_frac: float = RISK_FRAC,
) -> tuple[np.ndarray, np.ndarray]:
    """``x10_engine_nb`` with its output buffers sized and trimmed.

    Returns the two registries as ``(events, trades)`` matrices, already cut to
    the number of rows written. The capacity starts small and quadruples on
    overflow: events are rare (a handful per hundred bars) and a buffer sized
    for the worst case would dwarf the price history it scans.
    """
    n5 = len(m5_close)
    cap = max(256, n5 // 8)
    while True:
        events = np.full((cap + 16, N_EVENT_FIELDS), np.nan)
        trades = np.full((cap + 16, N_TRADE_FIELDS), np.nan)
        n_ev, n_tr = x10_engine_nb(
            m5_open,
            m5_high,
            m5_low,
            m5_close,
            atr,
            v,
            m,
            a,
            l_inf,
            l_sup,
            vwap,
            ema50_h1,
            atr_h1,
            dxy,
            dxy_ema50_h1,
            minute_of_day,
            session_id,
            bar_minute,
            spread,
            m1_open,
            m1_high,
            m1_low,
            m1_close,
            m1_start,
            m1_end,
            float(z),
            float(a_min),
            float(k_s),
            float(init_cash),
            float(risk_frac),
            events,
            trades,
        )
        if n_ev >= 0:
            return events[:n_ev].copy(), trades[:n_tr].copy()
        cap *= 4
