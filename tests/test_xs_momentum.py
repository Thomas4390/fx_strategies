"""Contract of the cross-sectional momentum book, on synthetic panels only.

Six properties, each of which would produce a plausible-looking screen row
while being wrong:

1. **The score is the score.** ``close[t-skip] / close[t-lookback] - 1``, to the
   last digit, and *blind to the last ``skip`` sessions* — the skip is the whole
   Jegadeesh-Titman convention and a one-row slip would silently trade the
   reversal it is meant to avoid.
2. **Ragged calendars do not rank instruments.** An instrument that does not
   trade on the same days as the others must still be scored, on its own
   sessions; a positional shift on the joined frame would drop it for a
   calendar reason and call it momentum.
3. **The availability gate.** No weight before ``min_available`` instruments
   quote — otherwise the 1990s, where five index series exist, would be traded
   as a "cross-section" of five.
4. **Gross normalization.** Long-only and dollar-neutral books carry the same
   gross exposure, so ``target_vol`` and ``max_leverage`` mean the same thing
   on both.
5. **The next-open fill.** The order lands one grid row after the decision, at
   that row's open — checked against the open itself, not against a snapshot.
6. **The book is filled.** Under cash sharing, a rejected order turns a 3/3
   book into a 3/0 one without a single error message.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

INDEX = pd.date_range("2019-01-01", periods=500, freq="B")

# Column carrying the fill date in ``orders.records_readable``: "Index" for the
# order-level records of ``from_orders``, "Fill Index" for the signal-level ones.
_FILL_INDEX = "Index"


def _geometric(rates: dict[str, float], index: pd.DatetimeIndex = INDEX) -> pd.DataFrame:
    """Panel of constant-growth series — every score is known in closed form."""
    steps = np.arange(len(index))
    return pd.DataFrame(
        {name: 100.0 * (1.0 + rate) ** steps for name, rate in rates.items()},
        index=index,
    )


@pytest.fixture()
def panel() -> pd.DataFrame:
    """Eight instruments, all quoting, ordered by momentum: I0 fastest."""
    return _geometric({f"I{i}": 0.0010 - 0.0001 * i for i in range(8)})


def test_score_matches_the_closed_form_and_skips_the_last_month(panel):
    from strategies.xs_momentum import xs_scores

    lookback, skip = 126, 21
    scores = xs_scores(panel, lookback=lookback, skip=skip)
    when = INDEX[400]

    for i in range(8):
        rate = 0.0010 - 0.0001 * i
        expected = (1.0 + rate) ** (lookback - skip) - 1.0
        assert scores.loc[when, f"I{i}"] == pytest.approx(expected, abs=1e-12)

    # The skipped month is genuinely skipped: rewriting the last 21 closes
    # leaves the score on the last date untouched.
    perturbed = panel.copy()
    perturbed.iloc[-skip:] *= 1.5
    assert xs_scores(perturbed, lookback=lookback, skip=skip).iloc[-1].equals(
        scores.iloc[-1]
    )


def test_a_ragged_calendar_does_not_erase_a_score(panel):
    """An instrument trading every other day keeps a score on its own sessions."""
    from strategies.xs_momentum import xs_scores

    ragged = panel.copy()
    ragged.iloc[1::2, ragged.columns.get_loc("I3")] = np.nan

    scores = xs_scores(ragged, lookback=126, skip=21)
    last = ragged["I3"].dropna().index[-1]

    assert not np.isnan(scores.loc[last, "I3"]), "own-session score lost to the joint index"
    # Its own sessions are twice as far apart, so its 126-session lookback spans
    # twice the calendar — the score is bigger, not missing.
    assert scores.loc[last, "I3"] > scores.loc[last, "I2"]
    # And a date where it does not quote stays unranked.
    silent = ragged.index[ragged["I3"].isna()][-1]
    assert np.isnan(scores.loc[silent, "I3"])


def test_nothing_is_traded_before_the_universe_exists():
    from strategies.xs_momentum import MIN_AVAILABLE, xs_weights

    starts = [0, 0, 0, 0, 0, 300, 320, 340]
    steps = np.arange(len(INDEX))
    staggered = pd.DataFrame(
        {
            f"I{i}": np.where(steps >= start, 100.0 * (1.0 + 0.0002 * (i + 1)) ** steps, np.nan)
            for i, start in enumerate(starts)
        },
        index=INDEX,
    )
    quoting = staggered.notna().sum(axis=1)
    first_six = INDEX[quoting >= MIN_AVAILABLE][0]

    weights = xs_weights(staggered, lookback=63, skip=5, n_long=3, n_short=0)
    early = weights.reindex(INDEX).fillna(0.0).loc[INDEX < first_six]

    assert not early.to_numpy().any(), "traded a cross-section of five"
    assert weights.ne(0.0).any(axis=1).any(), "never traded at all — test is vacuous"


def test_both_books_carry_the_same_gross(panel):
    from strategies.xs_momentum import xs_weights

    common = dict(lookback=126, skip=21, target_vol=None)
    long_only = xs_weights(panel, n_long=3, n_short=0, **common)
    neutral = xs_weights(panel, n_long=3, n_short=3, **common)

    traded = long_only.abs().sum(axis=1) > 0
    assert traded.any()
    assert np.allclose(long_only[traded].abs().sum(axis=1), 1.0)
    assert np.allclose(neutral[traded].abs().sum(axis=1), 1.0)
    # Dollar neutral means dollar neutral.
    assert neutral[traded].sum(axis=1).abs().max() == pytest.approx(0.0, abs=1e-12)
    # Long-only holds the three fastest, the neutral book shorts the slowest.
    assert list(long_only[traded].iloc[-1].nlargest(3).index) == ["I0", "I1", "I2"]
    assert list(neutral[traded].iloc[-1].nsmallest(3).index) == ["I5", "I6", "I7"]


def test_next_open_fills_one_row_after_the_decision(panel):
    from strategies.xs_momentum import pipeline, rebalance_dates, trading_grid

    opens = panel * 0.99  # an open that cannot be confused with any close
    grid = trading_grid(panel)
    decisions = rebalance_dates(grid, "ME")

    # slippage pinned to zero: ``vbt.yml`` puts a global 1 bp on every fill and
    # the point here is the price itself, not the friction on top of it.
    pf = pipeline(panel, opens=opens, lookback=126, skip=21, n_long=3,
                  target_vol=None, slippage=0.0)
    orders = pf.orders.records_readable
    assert len(orders) > 0

    for _, order in orders.iterrows():
        fill_at = order[_FILL_INDEX]
        row = grid.get_loc(fill_at)
        assert grid[row - 1] in decisions, f"fill on {fill_at} is not a next-open fill"
        assert order["Price"] == pytest.approx(opens.loc[fill_at, order["Column"]])

    # Same book on the decision close under fill="close".
    at_close = pipeline(panel, lookback=126, skip=21, n_long=3, target_vol=None,
                        slippage=0.0, fill="close")
    fills = at_close.orders.records_readable[_FILL_INDEX]
    assert set(fills).issubset(set(decisions))


def test_every_leg_of_a_long_short_book_is_filled(panel):
    from strategies.xs_momentum import pipeline

    pf = pipeline(panel, lookback=126, skip=21, n_long=3, n_short=3, log=True)
    logs = pf.logs.records_readable

    assert (logs["[RES] Status"] == "Rejected").sum() == 0, "cash sharing ate an order"
    assert pf.returns.ndim == 1, "the book must be one grouped return series"

    first = pf.orders.records_readable.groupby(_FILL_INDEX).size().iloc[0]
    assert first == 6, f"first rebalance filled {first} legs, expected 6"


def test_the_grid_is_closed():
    from strategies.xs_momentum import GRID

    assert len(GRID) == 6
    assert {(c["lookback"], c["n_long"], c["n_short"]) for c in GRID} == {
        (lookback, n_long, n_short)
        for lookback in (63, 126, 252)
        for n_long, n_short in ((3, 0), (3, 3))
    }
