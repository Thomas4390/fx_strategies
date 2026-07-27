"""Donchian breakout (N-session channel) — ims_pipeline format.

Price-based trend following in the Turtle lineage (Donchian's original moving
channel, then Dennis & Eckhardt's 20/55-day rule): go long when the close
prints a new ``entry_n``-session high, stay long until it prints a new
``exit_n``-session low. Nothing is estimated, nothing is fitted — the rule is a
pure ordering statistic of the last N closes, which is the point: it carries
the same trend premium as TSMOM through a completely different functional form.

Why it earns a slot next to ``gold_momentum``. TSMOM votes on the *sign of a
return* over fixed horizons; Donchian reacts to the *extremes* of the price
path. Whether that is a second family or the same one in disguise is settled
by the correlation of the two return streams — a measurement that belongs to
the screening script, not here.

Two asymmetries, both deliberate and both classic. **The exit channel is
shorter than the entry one** (``exit_n`` defaults to ``entry_n // 2``): a
symmetric rule gives back the whole channel width on every reversal. And
**long-only by default**, same argument as the gold sleeve — the instruments
screened here carry a structural positive drift, so a sustained short fights
the drift instead of harvesting a premium.

``fill`` defaults to ``"next_open"`` rather than to the sleeve's historical
``"close"``: a breakout is by construction decided on a bar whose close is
extreme, so banking that same close as a fill price is the one idealisation
this family is most exposed to.

Two entry points, matching the other sleeves:
- ``pipeline(data, **params) -> (pf, ind)`` — investigation path
- ``create_cv_pipeline(splitter, metric_type)`` — ``@vbt.cv_split`` factory
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import vectorbtpro as vbt

from framework.leverage import vol_target_leverage
from framework.pipeline_utils import (
    SHARPE_RATIO,
    compute_metric_nb,
    make_execute_kwargs,
)
from strategies.gold_momentum import (
    SESSION_CLOSE_HOUR,
    VOL_WINDOW,
    _daily_close,
    _daily_open,
)

# Signals live on daily bars, like every other daily sleeve — same
# annualization, directly comparable in the combiner.
DONCHIAN_DAILY_ANN_FACTOR: float = 252.0

# Entry channel, in sessions. 100 sits between the two Turtle systems (20 and
# 55 days) and the "one-year high" rule the momentum literature uses; it is a
# starting point for the screen, not a selected value. The screening grid is
# closed at {55, 100, 252} precisely so that this number never becomes one.
DEFAULT_ENTRY_N: int = 100


def donchian_channel(
    close_daily: pd.Series, entry_n: int, exit_n: int
) -> tuple[pd.Series, pd.Series]:
    """Entry (upper) and exit (lower) channels, both already lagged one session.

    The lag is what makes the rule causal: ``upper[t]`` is the highest close
    over ``[t - entry_n, t - 1]``, so ``close[t] > upper[t]`` reads "today
    closed above everything the channel had seen", never "today closed above
    its own maximum" (which is trivially false and would never fire).
    """
    upper = close_daily.vbt.rolling_max(entry_n, minp=entry_n).vbt.fshift(1)
    lower = close_daily.vbt.rolling_min(exit_n, minp=exit_n).vbt.fshift(1)
    return upper, lower


def _latched_state(entry_signal: pd.Series, exit_signal: pd.Series) -> pd.Series:
    """Boolean position state: on at ``entry_signal``, off at ``exit_signal``.

    A breakout is an *event*, not a state — the close only pierces the channel
    on one bar — so the state has to be carried forward until the opposite
    event fires. Everything before the first event is flat rather than NaN.
    Entry wins on a bar where both fire; on the long side the two cannot both
    be true, so that only settles a boundary case.
    """
    latch = pd.Series(np.nan, index=entry_signal.index)
    latch[exit_signal] = 0.0
    latch[entry_signal] = 1.0
    return latch.ffill().fillna(0.0) > 0.0


# ═══════════════════════════════════════════════════════════════════════
# 1. INVESTIGATION PATH — pipeline() returns (pf, indicator)
# ═══════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class DonchianIndicator:
    """Signal diagnostics: the price and the two channels it is compared to."""

    close: pd.Series
    upper: pd.Series
    lower: pd.Series

    def plot(self, fig: go.Figure | None = None, **layout_kwargs) -> go.Figure:
        fig = fig or go.Figure()
        for series, name, color in (
            (self.close, "close", "black"),
            (self.upper, "entry channel", "teal"),
            (self.lower, "exit channel", "crimson"),
        ):
            series.vbt.plot(
                fig=fig, trace_kwargs=dict(name=name, line=dict(width=1, color=color))
            )
        fig.update_layout(**layout_kwargs)
        return fig


def pipeline(
    data: vbt.Data,
    entry_n: int = DEFAULT_ENTRY_N,
    exit_n: int | None = None,
    allow_short: bool = False,
    target_vol: float | None = 0.55,
    max_leverage: float = 6.6,
    session_close_hour: int = SESSION_CLOSE_HOUR,
    ann_factor: float = DONCHIAN_DAILY_ANN_FACTOR,
    fill: str = "next_open",
    slippage: float | None = None,
    fees: float | None = None,
    init_cash: float | None = None,
    **pf_kwargs: Any,
) -> tuple[vbt.Portfolio, DonchianIndicator]:
    """Investigation path — channel breakout, vol-targeted, one instrument.

    ``exit_n`` defaults to ``entry_n // 2``: the Turtle asymmetry, see the
    module docstring.

    ``target_vol`` scales the position by inverse realized volatility through
    ``framework.leverage.vol_target_leverage`` — the same sizing layer as the
    gold sleeve, on purpose: two sleeves sized differently cannot be compared
    on a Sharpe. Set to None to size flat. ``session_close_hour`` and
    ``ann_factor`` are the conventions ``gold_momentum`` exposes so another
    instrument runs through the same sleeve without a fork (``strategies.tsmom``).

    ``fill="next_open"`` (the default here) decides on ``close[t]`` and fills
    at ``open[t+1]``, which is what MT5 does; VBT resolves it through
    ``price="nextopen"`` (an implicit ``from_ago=1``), so the ``shift(1)``
    already applied by ``vol_target_leverage`` is not doubled. ``"close"``
    keeps the idealised same-bar fill.
    """
    if entry_n <= 0:
        raise ValueError(f"entry_n must be > 0, got {entry_n}")
    exit_n = entry_n // 2 if exit_n is None else exit_n
    if exit_n <= 0:
        raise ValueError(f"exit_n must be > 0, got {exit_n}")
    if fill not in ("close", "next_open"):
        raise ValueError(f"fill must be 'close' or 'next_open', got {fill!r}")

    close_daily = _daily_close(data, session_close_hour)
    upper, lower = donchian_channel(close_daily, entry_n, exit_n)

    long_ok = _latched_state(close_daily > upper, close_daily < lower)
    if allow_short:
        # Mirror image: the entry channel's low opens the short, the exit
        # channel's high closes it.
        short_upper, short_lower = donchian_channel(close_daily, exit_n, entry_n)
        short_ok = _latched_state(close_daily < short_lower, close_daily > short_upper)
    else:
        short_ok = pd.Series(False, index=close_daily.index)

    # Edge transitions: one entry when the state turns on, one exit when it
    # turns off — the same block as the gold sleeve, so a trade means the same
    # thing in both.
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

    # Per-bar leverage array rather than a percent-of-cash size: with
    # `size_type="percent"` VBT caps the order at the available cash, silently
    # truncating any target above 1x (gold_momentum.pipeline, same plumbing).
    pf_args: dict[str, Any] = dict(
        close=close_daily, entries=entries, exits=exits,
        short_entries=short_entries, short_exits=short_exits,
        size=1.0, size_type="percent", leverage=lev_ts.to_numpy(),
        init_cash=init_cash, slippage=slippage, fees=fees, freq="1D",
    )
    if fill == "next_open":
        pf_args["open"] = _daily_open(data, session_close_hour).reindex(close_daily.index)
        pf_args["price"] = "nextopen"
    pf_args.update(pf_kwargs)

    pf = vbt.Portfolio.from_signals(**pf_args)

    indicator = DonchianIndicator(
        close=close_daily, upper=upper.rename("upper"), lower=lower.rename("lower")
    )
    return pf, indicator


# ═══════════════════════════════════════════════════════════════════════
# 2. CV FACTORY — create_cv_pipeline (@vbt.cv_split)
# ═══════════════════════════════════════════════════════════════════════


def create_cv_pipeline(
    splitter: Any,
    metric_type: int = SHARPE_RATIO,
    **pipeline_defaults: Any,
):
    """Build a ``@vbt.cv_split`` pipeline for walk-forward cross-validation.

    Mirrors ``strategies.tsmom.create_cv_pipeline``, with the channel
    parameters in place of the lookbacks and the same production sizing.
    """
    splitter_kwargs = pipeline_defaults.pop("splitter_kwargs", {})

    defaults = dict(
        entry_n=DEFAULT_ENTRY_N, exit_n=None, allow_short=False, target_vol=0.55,
        max_leverage=6.6, init_cash=None, slippage=None, fees=None,
        ann_factor=DONCHIAN_DAILY_ANN_FACTOR, cutoff=0.05, metric_type=metric_type,
    )
    defaults.update(pipeline_defaults)

    @vbt.cv_split(
        splitter=splitter,
        splitter_kwargs=splitter_kwargs,
        takeable_args=["data"],
        parameterized_kwargs=dict(
            execute_kwargs=make_execute_kwargs(
                "Donchian combos", pbar_kwargs=dict(leave=False)
            ),
            merge_func="concat",
        ),
        execute_kwargs=make_execute_kwargs("Donchian CV splits"),
        merge_func="concat",
        return_grid="all",
        attach_bounds="index",
    )
    def cv_pipeline(
        data: vbt.Data,
        entry_n: int = defaults["entry_n"],
        exit_n: int | None = defaults["exit_n"],
        allow_short: bool = defaults["allow_short"],
        target_vol: float | None = defaults["target_vol"],
        max_leverage: float = defaults["max_leverage"],
        init_cash: float | None = defaults["init_cash"],
        slippage: float | None = defaults["slippage"],
        fees: float | None = defaults["fees"],
        ann_factor: float = defaults["ann_factor"],
        cutoff: float = defaults["cutoff"],
        metric_type: int = defaults["metric_type"],
    ) -> float:
        pf, _ = pipeline(
            data, entry_n=entry_n, exit_n=exit_n, allow_short=allow_short,
            target_vol=target_vol, max_leverage=max_leverage, init_cash=init_cash,
            slippage=slippage, fees=fees, ann_factor=ann_factor,
        )
        returns = pf.returns.values
        if returns.ndim > 1:
            returns = returns[:, 0]
        return float(compute_metric_nb(returns, metric_type, ann_factor, cutoff))

    return cv_pipeline
