"""Dual / acceleration momentum — the TSMOM ensemble, filtered by acceleration.

The base score is the one the gold sleeve trades: the mean of the sign of four
fixed lookbacks (``gold_momentum.momentum_ensemble``). What this family adds is
a **second derivative**: a trend whose 3-month return no longer outruns its
12-month return is decelerating, and the literature on momentum crashes
(Daniel & Moskowitz, "Momentum crashes", JFE 122(2), 2016) documents that the
bad tail of a momentum book is concentrated in exactly those regimes.

``accel_score`` is that quantity, ``pct_change(fast) - pct_change(slow)``. Two
ways to use it, and they are the two modes of this sleeve:

- ``"gate"`` — the filter is a door: no long unless the score AND the
  acceleration are positive. Trades are removed outright.
- ``"brake"`` — the filter is a dimmer: the long is kept, but sized at half
  leverage when the trend decelerates. Nothing is removed, exposure is.

**What "brake" means exactly, because the engine constrains it.** Entries and
exits are edge transitions, so ``from_signals`` places an order only when the
state flips and a per-bar ``leverage`` array is read at order time, nowhere
else. The brake therefore halves a *trade opened while the trend decelerates*,
for that trade's whole life; it does not re-size an open position when the
acceleration flips mid-trade. That is the limitation the vol-target layer
already has here (a gold trade carries the leverage of its entry bar), which is
why the two go through the same array. On EUR-USD 23 of the 32 entries are
braked, so the mode is not inert — it is narrower than "half exposure whenever
accel < 0" reads.

Neither mode touches the signal itself: the lookbacks stay unselected, which is
the property that protects the base sleeve from overfitting (see the reversion
record in ``gold_momentum.DEFAULT_LOOKBACKS``). This is a filter on a fixed
score, so it will be correlated with the score it filters — the screening
script measures that correlation rather than assuming it away.

One entry point, matching the other sleeves:
- ``pipeline(data, **params) -> (pf, ind)`` — investigation path
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import vectorbtpro as vbt

from framework.leverage import vol_target_leverage
from strategies.gold_momentum import (
    DEFAULT_LOOKBACKS,
    GOLD_DAILY_ANN_FACTOR,
    SESSION_CLOSE_HOUR,
    VOL_WINDOW,
    _daily_close,
    _daily_open,
    momentum_ensemble,
)

# ~3 months against ~12, the horizon pair the momentum-crash literature uses.
FAST_N: int = 63
SLOW_N: int = 252

# What "brake" multiplies the vol-target leverage by while the trend
# decelerates. Half, not a tuned number: the point of the mode is to compare a
# dimmer with a door, not to calibrate the dimmer.
BRAKE_FACTOR: float = 0.5

MODES: tuple[str, ...] = ("gate", "brake")

# The screening grid, closed. Two modes times two horizon pairs — the second
# pair (21, 126) halves both horizons so the family is not judged on a single
# speed. Nothing else is to be added here without a new trial budget.
CONFIGS: tuple[tuple[str, int, int], ...] = (
    ("gate", FAST_N, SLOW_N),
    ("brake", FAST_N, SLOW_N),
    ("gate", 21, 126),
    ("brake", 21, 126),
)


def accel_score(
    close_daily: pd.Series,
    fast_n: int = FAST_N,
    slow_n: int = SLOW_N,
) -> pd.Series:
    """Acceleration: the fast horizon's return minus the slow one's.

    Positive means the recent leg of the trend is steeper than the whole of it
    — the trend is still gaining. Negative means it is running out, whatever
    the sign of the trend itself.

    Unlike ``momentum_ensemble`` this is a raw return difference, not a vote:
    it is used only through its sign, so its scale never enters a sizing
    decision and does not need normalizing across instruments.
    """
    if not 0 < fast_n < slow_n:
        raise ValueError(f"expected 0 < fast_n < slow_n, got {fast_n} and {slow_n}")
    return close_daily.vbt.pct_change(fast_n) - close_daily.vbt.pct_change(slow_n)


@dataclass
class DualMomentumIndicator:
    """Signal diagnostics: both scores, the state they produce, the leverage."""

    close: pd.Series
    score: pd.Series
    accel: pd.Series
    long_ok: pd.Series
    leverage: pd.Series


def pipeline(
    data: vbt.Data,
    lookbacks: tuple[int, ...] = DEFAULT_LOOKBACKS,
    mode: str = "gate",
    fast_n: int = FAST_N,
    slow_n: int = SLOW_N,
    allow_short: bool = False,
    target_vol: float | None = 0.55,
    max_leverage: float = 6.6,
    base_size: float = 1.0,
    leverage: float | None = None,
    init_cash: float | None = None,
    slippage: float | None = None,
    fees: float | None = None,
    session_close_hour: int = SESSION_CLOSE_HOUR,
    ann_factor: float = GOLD_DAILY_ANN_FACTOR,
    fill: str = "next_open",
    **pf_kwargs: Any,
) -> tuple[vbt.Portfolio, DualMomentumIndicator]:
    """Investigation path — the gold sleeve's plumbing, one filter added.

    Everything that is not the filter is deliberately identical to
    ``gold_momentum.pipeline``: same session cut, same ensemble score, same
    edge-transition entries and exits, same vol-target layer. A difference
    between the two families must therefore come from the acceleration and
    from nothing else.

    ``fill`` defaults to ``"next_open"`` here rather than to ``"close"``: this
    sleeve is born after the protocol decision of
    ``docs/research/momentum_expansion_2026H2.md`` §2, so its default is the
    convention MT5 executes, and there is no snapshot pinning the idealised one.

    ``mode`` selects how the acceleration is used — see the module docstring.
    The brake multiplier is applied on the *signal* bar, exactly like the score
    itself, so the two are read on the same information set; under
    ``fill="next_open"`` both are then applied to the following open.
    """
    if not lookbacks:
        raise ValueError("lookbacks must be a non-empty tuple")
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    if base_size <= 0:
        raise ValueError(f"base_size must be > 0, got {base_size}")
    if fill not in ("close", "next_open"):
        raise ValueError(f"fill must be 'close' or 'next_open', got {fill!r}")

    close_daily = _daily_close(data, session_close_hour)
    score = momentum_ensemble(close_daily, lookbacks)
    accel = accel_score(close_daily, fast_n, slow_n)

    rising = accel > 0.0
    falling = accel < 0.0  # NaN during the warmup is neither, by construction

    if mode == "gate":
        long_ok = (score > 0.0) & rising
        short_ok = ((score < 0.0) & falling) if allow_short else pd.Series(False, index=score.index)
    else:
        long_ok = score > 0.0
        short_ok = (score < 0.0) if allow_short else pd.Series(False, index=score.index)

    # Edge transitions: one entry when the state turns on, one exit when it
    # turns off — the structure of gold_momentum.pipeline, unchanged.
    prev_long = long_ok.vbt.fshift(1, fill_value=False)
    prev_short = short_ok.vbt.fshift(1, fill_value=False)
    entries = long_ok & ~prev_long
    exits = ~long_ok & prev_long
    short_entries = short_ok & ~prev_short
    short_exits = ~short_ok & prev_short

    if target_vol is not None:
        daily_ret = close_daily.vbt.pct_change()
        realized = daily_ret.vbt.rolling_std(
            VOL_WINDOW, minp=VOL_WINDOW, ddof=1
        ) * np.sqrt(ann_factor)
        lev_ts = vol_target_leverage(realized, target_vol, max_leverage=max_leverage)
    else:
        lev_ts = pd.Series(1.0, index=close_daily.index)

    if mode == "brake":
        # A dimmer, not a door: the position stays, its size does not. Read at
        # order time only — see the module docstring on what that scopes.
        lev_ts = lev_ts * np.where(falling.to_numpy(), BRAKE_FACTOR, 1.0)

    lev_arr = (lev_ts * float(leverage if leverage is not None else 1.0)).to_numpy()

    pf_args: dict[str, Any] = dict(
        close=close_daily,
        entries=entries,
        exits=exits,
        short_entries=short_entries,
        short_exits=short_exits,
        size=base_size,
        size_type="percent",
        leverage=lev_arr,
        init_cash=init_cash,
        slippage=slippage,
        fees=fees,
        freq="1D",
    )
    if fill == "next_open":
        pf_args["open"] = _daily_open(data, session_close_hour).reindex(close_daily.index)
        pf_args["price"] = "nextopen"
    pf_args.update(pf_kwargs)

    pf = vbt.Portfolio.from_signals(**pf_args)

    indicator = DualMomentumIndicator(
        close=close_daily,
        score=score.rename("score"),
        accel=accel.rename("accel"),
        long_ok=long_ok.rename("long_ok"),
        leverage=lev_ts.rename("leverage"),
    )
    return pf, indicator
