"""Cross-sectional momentum — one portfolio, the whole screened universe.

Every other momentum family here is a *sleeve per instrument*, the sleeves being
added up afterwards. This one is not: cross-sectional momentum only exists as a
**single portfolio** that ranks the instruments against each other every month,
buys the best ``n_long`` and (optionally) sells the worst ``n_short``. One
return series, not a sum of series — and the family only earns its place if
that series is not the equal-weight TSMOM basket in disguise.

Three conventions carry the design, all of them forced by the data:

- **Ragged histories.** The universe mixes 36-year index series with 3-year
  broker exports, so the book ranks only what *quotes on the decision date* and
  stays flat below ``min_available`` quoting instruments. That threshold also
  defines the trading grid: the portfolio does not exist on days where the
  cross-section is too thin to be a cross-section (before ~2000, here).
- **No forward fill.** A missing close is a missing instrument, never a stale
  price. A positional ``shift(126)`` on the joined frame would therefore count
  *other* instruments' sessions, so :func:`xs_scores` scores each instrument on
  its own sessions and reindexes afterwards.
- **Skip the last month** (``skip=21``, Jegadeesh-Titman): the (t-21, t-252)
  return, not the (t, t-252) one, because the most recent month reverses.

Execution goes through ``from_orders`` with ``size_type="targetpercent"`` —
``from_signals`` has no target-percent sizing, and a target weight is exactly
what a rebalanced book is. Cash sharing plus ``leverage_mode="eager"`` are not
decoration: under the default lazy mode the first long consumes every dollar of
free cash and the shorts of the same rebalance come back ``Rejected(NoCash)``,
silently turning a 3/3 book into a 3/0 one.

Entry points: ``xs_scores`` (ranking signal), ``xs_weights`` (vol-targeted
target weights, one row per rebalance), ``pipeline`` (the simulated book).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import vectorbtpro as vbt

from framework.leverage import vol_target_leverage

# Daily strategies → annualization factor = 252, the repo convention.
DAILY_ANN_FACTOR: float = 252.0

# Realized-volatility window for the portfolio-level vol target, in sessions.
VOL_WINDOW: int = 21

# Below this many quoting instruments there is no cross-section to rank: the
# book stays flat and the day is not even part of the trading grid.
MIN_AVAILABLE: int = 6

# Headroom on the leverage *permission* handed to ``from_orders``. Gross
# exposure is set by the target weights, never by this number; the margin exists
# because a book that drifted above its target between two rebalances is briefly
# worth more than ``max_leverage``, and a permission set exactly at the cap makes
# VBT reject the very orders that would bring it back down.
LEVERAGE_HEADROOM: float = 2.0

# Closed grid, 6 configurations. Long-only (3, 0) against dollar-neutral
# (3, 3), on the three usual momentum horizons. Nothing else is tested — the
# trial budget of this family was declared at n=6.
GRID: tuple[dict[str, int], ...] = tuple(
    dict(lookback=lookback, n_long=n_long, n_short=n_short)
    for lookback in (63, 126, 252)
    for n_long, n_short in ((3, 0), (3, 3))
)


def xs_scores(
    closes: pd.DataFrame,
    lookback: int = 126,
    skip: int = 21,
) -> pd.DataFrame:
    """Momentum score per instrument: ``close[t-skip] / close[t-lookback] - 1``.

    Computed column by column on each instrument's **own** sessions: on a frame
    that mixes calendars, row ``t-126`` of the joint index is not 126 sessions
    ago for an instrument that does not trade every one of those days, and it is
    often a NaN — dropping it from the ranking for a reason that has nothing to
    do with its momentum. Reindexed on the joint frame at the end, so a date
    where the instrument does not quote stays NaN and is not ranked.
    """
    if lookback <= 0:
        raise ValueError(f"lookback must be > 0, got {lookback}")
    if not 0 <= skip < lookback:
        raise ValueError(f"skip must satisfy 0 <= skip < lookback, got {skip}/{lookback}")

    scores = {}
    for column in closes.columns:
        own = closes[column].dropna()
        scores[column] = (own.shift(skip) / own.shift(lookback) - 1.0).reindex(closes.index)
    return pd.DataFrame(scores, index=closes.index, columns=closes.columns)


def trading_grid(closes: pd.DataFrame, min_available: int = MIN_AVAILABLE) -> pd.DatetimeIndex:
    """Dates on which at least ``min_available`` instruments quote — the book's index.

    A day where three of them quote is a Sunday FX session or the far end of a
    broker export, not a day where a cross-section can be ranked.
    """
    return pd.DatetimeIndex(closes.index[closes.notna().sum(axis=1) >= min_available])


def rebalance_dates(index: pd.DatetimeIndex, rebalance: str = "ME") -> pd.DatetimeIndex:
    """Last grid date of each ``rebalance`` period — the decision dates."""
    stamps = pd.Series(index, index=index).resample(rebalance).last().dropna()
    return pd.DatetimeIndex(stamps.to_numpy())


def xs_weights(
    closes: pd.DataFrame,
    lookback: int = 126,
    skip: int = 21,
    n_long: int = 3,
    n_short: int = 0,
    rebalance: str = "ME",
    target_vol: float | None = 0.25,
    max_leverage: float = 3.0,
    min_available: int = MIN_AVAILABLE,
    ann_factor: float = DAILY_ANN_FACTOR,
) -> pd.DataFrame:
    """Target weights of the book, one row per rebalance date.

    Gross exposure is normalized to 1 *before* the vol target: ``1 / n_long`` per
    leg long-only, ``±0.5 / n`` per leg dollar-neutral. Both books then carry the
    same gross, so ``target_vol`` means the same thing on either side of the grid
    and ``max_leverage`` caps gross exposure rather than one leg of it.

    ``target_vol`` is applied at the **portfolio** level — the realized vol of
    the unlevered basket, not of an instrument. A diversified basket does not
    need the 0.55 of the mono-instrument sleeve; 0.25 is the default here, and
    ``None`` sizes flat.
    """
    if n_long < 1:
        raise ValueError(f"n_long must be >= 1, got {n_long}")
    if n_short < 0:
        raise ValueError(f"n_short must be >= 0, got {n_short}")

    grid = trading_grid(closes, min_available)
    on_grid = closes.loc[grid]
    scores = xs_scores(closes, lookback, skip).loc[grid].where(on_grid.notna())

    at_reb = scores.loc[rebalance_dates(grid, rebalance)]
    n_ranked = at_reb.notna().sum(axis=1)
    # method="first" so a tie never puts the same instrument on both legs.
    best = at_reb.rank(axis=1, ascending=False, method="first")
    worst = at_reb.rank(axis=1, ascending=True, method="first")

    long_gross = 1.0 if n_short == 0 else 0.5
    weights = (best <= n_long).astype(float) * (long_gross / n_long)
    if n_short:
        weights -= (worst <= n_short).astype(float) * (0.5 / n_short)
    # Flat whenever the cross-section is too thin to rank, or too thin to fill
    # both legs of the requested book.
    tradable = n_ranked >= max(min_available, n_long + n_short)
    weights = weights.where(tradable, 0.0)

    if target_vol is None:
        return weights

    held = weights.reindex(grid).ffill().fillna(0.0)
    rets = on_grid.vbt.pct_change().fillna(0.0)
    # Proxy book return: weights decided on close[t] earn from t+1 on. Ignores
    # the intra-month drift of the held weights, which is a second-order effect
    # on a volatility estimate and keeps the estimate independent of the fill.
    proxy = (held.vbt.fshift(1).fillna(0.0) * rets).sum(axis=1)
    realized = proxy.vbt.rolling_std(VOL_WINDOW, minp=VOL_WINDOW, ddof=1) * np.sqrt(ann_factor)
    lev = vol_target_leverage(realized, target_vol, max_leverage=max_leverage)
    return weights.mul(lev.reindex(weights.index), axis=0)


def _price_table(on_grid: pd.DataFrame, opens: pd.DataFrame | None, fill: str) -> pd.DataFrame:
    """Execution price per cell, built explicitly rather than left to a keyword.

    ``price="nextopen"`` is a ``from_signals`` convenience; ``from_orders`` takes
    a matrix, so the next-open convention is the order placed one grid row after
    the decision, quoted at that row's open. Where an instrument has no open
    (source without OHLC, missing session) the close of the same row is used,
    then the last known close — a degradation written down rather than hidden,
    as ``gold_momentum._daily_open`` does for the mono-instrument sleeve.
    """
    price = on_grid
    if fill == "next_open" and opens is not None:
        aligned = opens.reindex(index=on_grid.index, columns=on_grid.columns)
        price = aligned.combine_first(on_grid)
    return price.ffill().bfill()


def _slippage_table(
    slippage: pd.Series | float | None,
    index: pd.DatetimeIndex,
    columns: pd.Index,
) -> pd.DataFrame | None:
    """Per-instrument slippage broadcast column by column over the grid."""
    if slippage is None:
        return None
    if isinstance(slippage, pd.Series):
        values = slippage.reindex(columns).to_numpy(dtype=float)
        if np.isnan(values).any():
            missing = [c for c, v in zip(columns, values) if np.isnan(v)]
            raise ValueError(f"no slippage for {missing}")
    else:
        values = np.full(len(columns), float(slippage))
    return pd.DataFrame(np.tile(values, (len(index), 1)), index=index, columns=columns)


def pipeline(
    closes: pd.DataFrame,
    opens: pd.DataFrame | None = None,
    lookback: int = 126,
    skip: int = 21,
    n_long: int = 3,
    n_short: int = 0,
    rebalance: str = "ME",
    target_vol: float | None = 0.25,
    max_leverage: float = 3.0,
    min_available: int = MIN_AVAILABLE,
    fill: str = "next_open",
    slippage: pd.Series | float | None = None,
    init_cash: float = 100_000.0,
    ann_factor: float = DAILY_ANN_FACTOR,
    **pf_kwargs,
) -> vbt.Portfolio:
    """Simulate the cross-sectional book as one cash-shared portfolio.

    ``fill="next_open"`` decides on ``close[t]`` and fills on the next grid
    row's open, which is what a broker does; ``fill="close"`` fills on the
    decision close, which is the idealised convention.

    The returned portfolio is **grouped**: ``pf.returns`` is the one series this
    family produces, and the per-column view exists only for order-level
    inspection.
    """
    if fill not in ("close", "next_open"):
        raise ValueError(f"fill must be 'close' or 'next_open', got {fill!r}")

    weights = xs_weights(
        closes, lookback=lookback, skip=skip, n_long=n_long, n_short=n_short,
        rebalance=rebalance, target_vol=target_vol, max_leverage=max_leverage,
        min_available=min_available, ann_factor=ann_factor,
    )
    grid = trading_grid(closes, min_available)
    on_grid = closes.loc[grid]

    rows = grid.get_indexer(weights.index)
    if fill == "next_open":
        rows = rows + 1
    keep = rows < len(grid)  # a decision on the last grid row never gets filled
    size = pd.DataFrame(np.nan, index=grid, columns=closes.columns)
    size.iloc[rows[keep]] = weights.to_numpy()[keep]

    return vbt.Portfolio.from_orders(
        close=on_grid,
        size=size,
        size_type="targetpercent",
        price=_price_table(on_grid, opens, fill),
        slippage=_slippage_table(slippage, grid, closes.columns),
        group_by=True,
        cash_sharing=True,
        call_seq="auto",
        # A permission, not a target: gross exposure is capped by construction
        # at ``max_leverage`` (unit gross × the vol-target multiplier), and
        # eager mode is what lets the whole book be filled in one pass.
        leverage=max_leverage * LEVERAGE_HEADROOM,
        leverage_mode="eager",
        init_cash=init_cash,
        freq="1D",
        **pf_kwargs,
    )
