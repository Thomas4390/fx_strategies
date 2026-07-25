"""Path-dependent position sizing overlays for ``Portfolio.from_signals``.

Four regimes behind one Numba ``signal_func_nb``, so a strategy can be run with
identical signals and only the sizing rule changed:

- ``MODE_FLAT``       constant size — the control
- ``MODE_MARTINGALE`` size multiplied after each *losing* trade, reset on a win
- ``MODE_GRID``       add-ons at successive adverse excursions within a trade
- ``MODE_COMBO``      martingale base size *and* an intra-trade grid
- ``MODE_ANTI_MART``  size multiplied after each *winning* trade, reset on a loss

Why this is written as a state machine rather than a size array: martingale
depends on the outcome of the *previous* trade, and grid depends on the price
path *inside* the current trade. Neither is knowable before the simulation, so
both must be computed inside the simulation loop. The structure follows the
"entry laddering" pattern from the VBT docs (from-signals / Dynamic): named
tuples to group arrays, a structured dtype for per-column state, and
``accumulate=True`` so add-ons enlarge the open position.

Every path-dependent regime carries three hard guards, without which the
backtest is fiction rather than a simulation:

1. ``max_total``   caps cumulative deployed size, standing in for margin.
2. ``basket_stop`` closes the whole position at a loss threshold.
3. ``kill_dd``     forces the sleeve back to flat sizing after a drawdown.

A martingale without these does not blow up in a backtest only because the
backtest lets equity go arbitrarily negative. A broker does not.
"""

from __future__ import annotations

from collections import namedtuple

import numpy as np
import vectorbtpro as vbt
from numba import njit

# ═══════════════════════════════════════════════════════════════════════
# MODES
# ═══════════════════════════════════════════════════════════════════════

MODE_FLAT = 0
MODE_MARTINGALE = 1
MODE_GRID = 2
MODE_COMBO = 3
MODE_ANTI_MART = 4

MODE_LABELS = {
    MODE_FLAT: "flat",
    MODE_MARTINGALE: "martingale",
    MODE_GRID: "grid",
    MODE_COMBO: "combo",
    MODE_ANTI_MART: "anti_martingale",
}

# Fraction of portfolio value ordered per unit of size multiplier.
_SIZE_TYPE = vbt.pf_enums.SizeType.ValuePercent

# ═══════════════════════════════════════════════════════════════════════
# CONTAINERS
# ═══════════════════════════════════════════════════════════════════════

#: Signal arrays forwarded from the strategy, unchanged by the overlay.
Signals = namedtuple("Signals", ["entries", "exits", "short_entries", "short_exits"])

#: Order arrays the overlay writes into (full-shaped, see `size` template).
Order = namedtuple("Order", ["size", "size_type"])

#: One parameter set per column.
sizing_params_dt = np.dtype(
    [
        ("mode", np.int64),
        ("base_size", np.float64),  # fraction of portfolio value at 1x
        ("mult", np.float64),  # martingale / anti-martingale multiplier
        ("n_max", np.int64),  # max multiplier steps (caps mult ** n_max)
        ("grid_k", np.float64),  # grid spacing, in ATR units
        ("n_levels", np.int64),  # max add-ons per trade
        ("grid_mult", np.float64),  # size multiplier per successive grid level
        ("basket_stop", np.float64),  # close all at this open loss (frac. of value)
        ("max_total", np.float64),  # cap on cumulative size multiple
        ("kill_dd", np.float64),  # drawdown past which sizing reverts to flat
    ],
    align=True,
)

#: Mutable per-column state carried across bars.
sizing_state_dt = np.dtype(
    [
        ("loss_streak", np.int64),
        ("win_streak", np.int64),
        ("anchor", np.float64),  # entry price of the open position
        ("level", np.int64),  # grid add-ons already filled
        ("total_mult", np.float64),  # cumulative size multiple deployed
        ("peak_value", np.float64),  # running equity peak, for the kill switch
        ("killed", np.int64),  # 1 once the kill switch has fired
        ("prev_in_pos", np.int64),  # were we in a position on the previous bar
        ("max_total_seen", np.float64),  # diagnostic: largest multiple ever deployed
        ("n_addons", np.int64),  # diagnostic: total grid add-ons executed
        ("n_killed", np.int64),  # diagnostic: kill-switch activations
    ],
    align=True,
)


def make_params(
    mode: int,
    *,
    base_size: float = 1.0,
    mult: float = 2.0,
    n_max: int = 3,
    grid_k: float = 1.0,
    n_levels: int = 3,
    grid_mult: float = 1.0,
    basket_stop: float = 0.25,
    max_total: float = 4.0,
    kill_dd: float = 0.5,
) -> np.ndarray:
    """Build a one-element ``sizing_params_dt`` array.

    Defaults are deliberately conservative: three martingale steps at 2x caps
    exposure at 8x base, and the basket stop fires well before the kill switch.
    """
    if mode not in MODE_LABELS:
        raise ValueError(f"unknown mode {mode}, expected one of {sorted(MODE_LABELS)}")
    if mult < 1.0:
        raise ValueError(f"mult must be >= 1.0, got {mult}")
    if n_max < 0 or n_levels < 0:
        raise ValueError("n_max and n_levels must be >= 0")
    if grid_k <= 0.0:
        raise ValueError(f"grid_k must be > 0, got {grid_k}")
    if max_total < base_size:
        raise ValueError(f"max_total ({max_total}) must be >= base_size ({base_size})")

    return np.array(
        [
            (
                mode,
                base_size,
                mult,
                n_max,
                grid_k,
                n_levels,
                grid_mult,
                basket_stop,
                max_total,
                kill_dd,
            )
        ],
        dtype=sizing_params_dt,
    )


def make_state(n_cols: int) -> np.ndarray:
    """Fresh per-column state array."""
    state = np.zeros(n_cols, dtype=sizing_state_dt)
    state["anchor"] = np.nan
    return state


def build_overlay_kwargs(
    params: np.ndarray,
    atr: np.ndarray,
    *,
    memory: dict | None = None,
) -> dict:
    """``from_signals`` kwargs wiring the overlay to a strategy's signals.

    Keeps the template plumbing in one place: the ``size`` and ``size_type``
    arrays must be materialised at full shape (the overlay writes into them per
    bar), the state array must be allocated once broadcasting is resolved, and
    ``accumulate`` must be on for grid add-ons to enlarge a position.

    Pass a ``memory`` dict to get the state array back after the simulation —
    it carries the diagnostics (``max_total_seen``, ``n_addons``, ``n_killed``)
    that decide whether a regime is survivable.

    Merge the result into the ``pipeline(...)`` call:

        kw = build_overlay_kwargs(make_params(MODE_MARTINGALE), atr)
        pf, _ = pipeline(data, **kw)
    """
    mem = memory if memory is not None else {}
    return dict(
        signal_func_nb=sizing_signal_func_nb,
        signal_args=(
            vbt.RepEval(
                "Signals(entries, exits, short_entries, short_exits)",
                context=dict(Signals=Signals),
            ),
            vbt.RepEval("Order(size, size_type)", context=dict(Order=Order)),
            params,
            vbt.RepEval(
                "mem.setdefault('state', make_state(wrapper.shape_2d[1]))",
                context=dict(mem=mem, make_state=make_state),
                context_merge_kwargs=dict(nested=False),
            ),
            atr,
        ),
        size=vbt.RepEval("np.full(wrapper.shape_2d, np.nan)"),
        size_type=vbt.RepEval("np.full(wrapper.shape_2d, -1)"),
        accumulate=True,
    )


# ═══════════════════════════════════════════════════════════════════════
# KERNEL
# ═══════════════════════════════════════════════════════════════════════


@njit(nogil=True)
def _step_multiplier_nb(mode: int, mult: float, n_max: int, loss_streak: int, win_streak: int) -> float:
    """Size multiple for a fresh entry, from the streak of prior outcomes."""
    if mode == MODE_MARTINGALE or mode == MODE_COMBO:
        steps = loss_streak
    elif mode == MODE_ANTI_MART:
        steps = win_streak
    else:
        return 1.0
    if steps > n_max:
        steps = n_max
    return mult**steps


@njit(nogil=True)
def sizing_signal_func_nb(c, sig, order, params, state, atr):
    """``signal_func_nb`` applying a path-dependent sizing regime.

    Returns the usual ``(long_entry, long_exit, short_entry, short_exit)`` and
    writes the size of any order it emits into ``order.size`` / ``order.size_type``.
    """
    col = c.col
    p = params[0] if params.shape[0] == 1 else params[col]
    st = state[col]

    is_entry = vbt.pf_nb.select_nb(c, sig.entries)
    is_exit = vbt.pf_nb.select_nb(c, sig.exits)
    is_short_entry = vbt.pf_nb.select_nb(c, sig.short_entries)
    is_short_exit = vbt.pf_nb.select_nb(c, sig.short_exits)

    position = c.last_position[col]
    in_pos = position != 0.0
    value = c.last_value[c.group]
    price = c.close[c.i, col]

    # ── equity peak and kill switch ────────────────────────────────────
    if value > st["peak_value"] or st["peak_value"] == 0.0:
        st["peak_value"] = value
    if not st["killed"] and st["peak_value"] > 0.0:
        drawdown = 1.0 - value / st["peak_value"]
        if drawdown >= p["kill_dd"]:
            st["killed"] = 1
            st["n_killed"] += 1

    # Once killed, the overlay degrades to flat sizing. It never re-arms:
    # a system that re-arms its own risk engine after a 50% loss is how the
    # second 50% happens.
    mode = p["mode"]
    if st["killed"]:
        mode = MODE_FLAT

    # ── a position closed since the previous bar → update streaks ──────
    if st["prev_in_pos"] and not in_pos:
        pos_info = c.last_pos_info[col]
        if pos_info["status"] == vbt.pf_enums.TradeStatus.Closed:
            if pos_info["pnl"] < 0.0:
                st["loss_streak"] += 1
                st["win_streak"] = 0
            else:
                st["win_streak"] += 1
                st["loss_streak"] = 0
        st["anchor"] = np.nan
        st["level"] = 0
        st["total_mult"] = 0.0
    st["prev_in_pos"] = 1 if in_pos else 0

    # ── open position: basket stop, then grid add-ons ──────────────────
    if in_pos and not np.isnan(st["anchor"]):
        direction = 1.0 if position > 0.0 else -1.0
        adverse = -direction * (price - st["anchor"]) / st["anchor"]

        # Basket stop: unrealized loss on the whole stack, as a fraction of
        # portfolio value. Checked before any add-on, so a grid can never
        # double down through its own stop.
        if value > 0.0:
            open_loss = adverse * st["total_mult"] * p["base_size"]
            if open_loss >= p["basket_stop"]:
                order.size[c.i, col] = np.inf
                order.size_type[c.i, col] = vbt.pf_enums.SizeType.Amount
                if direction > 0.0:
                    return False, True, False, False
                return False, False, False, True

        if (mode == MODE_GRID or mode == MODE_COMBO) and st["level"] < p["n_levels"]:
            spacing = atr[c.i, col] / st["anchor"] * p["grid_k"]
            if spacing > 0.0 and adverse >= (st["level"] + 1) * spacing:
                add = p["base_size"] * p["grid_mult"] ** (st["level"] + 1)
                if st["total_mult"] + add <= p["max_total"]:
                    st["level"] += 1
                    st["total_mult"] += add
                    st["n_addons"] += 1
                    if st["total_mult"] > st["max_total_seen"]:
                        st["max_total_seen"] = st["total_mult"]
                    order.size[c.i, col] = add
                    order.size_type[c.i, col] = _SIZE_TYPE
                    if direction > 0.0:
                        return True, False, False, False
                    return False, False, True, False

    # ── exits pass through untouched ───────────────────────────────────
    if is_exit or is_short_exit:
        order.size[c.i, col] = np.inf
        order.size_type[c.i, col] = vbt.pf_enums.SizeType.Amount
        return False, is_exit, False, is_short_exit

    # ── fresh entry ────────────────────────────────────────────────────
    if (is_entry or is_short_entry) and not in_pos:
        mult = _step_multiplier_nb(mode, p["mult"], p["n_max"], st["loss_streak"], st["win_streak"])
        size = p["base_size"] * mult
        if size > p["max_total"]:
            size = p["max_total"]
        st["anchor"] = price
        st["level"] = 0
        st["total_mult"] = size
        if size > st["max_total_seen"]:
            st["max_total_seen"] = size
        order.size[c.i, col] = size
        order.size_type[c.i, col] = _SIZE_TYPE
        return is_entry, False, is_short_entry, False

    return False, False, False, False
