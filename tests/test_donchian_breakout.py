"""Contract of the Donchian breakout sleeve.

Five properties, each of which would fail silently in a backtest report:

1. **The channel is lagged.** ``upper[t]`` must be the highest close *before*
   t. Drop the lag and the rule compares a close to a window containing
   itself — it stops firing, or fires on the wrong bar, and the equity curve
   still looks perfectly plausible.
2. **Causality.** Truncating the sample must not change a single order placed
   before the cut.
3. **The next-open fill.** This sleeve defaults to ``fill="next_open"``
   because a breakout is decided on an extreme close; the claim "filled at the
   next session's open" is checked against the session opens themselves.
4. **Transitions.** A breakout is an event, the position is a state: one Buy
   per crossing of the entry channel, one Sell per crossing of the exit
   channel, strictly alternating. A rule that re-entered on every new high
   would trade the same trend a dozen times.
5. The exit channel defaults to half the entry channel (the Turtle asymmetry).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# Columns compared across runs: what was traded, when, at what price and size.
_ORDER_COLUMNS: list[str] = ["Fill Index", "Size", "Price", "Side"]


@pytest.fixture(scope="module")
def eur_usd():
    """Broker daily EUR-USD, loaded once."""
    from strategies.tsmom import load_instrument

    _, data = load_instrument("EUR-USD")
    return data


def synthetic_daily(n: int = 400, seed: int = 7) -> pd.DataFrame:
    """A daily OHLC-ish frame — enough for the sleeve, cheap enough for a test.

    ``_daily_close`` / ``_daily_open`` read ``.close`` / ``.open`` and leave a
    daily index alone, so a plain DataFrame is a valid ``data`` here. The drift
    is positive so that the long-only rule actually takes trades.
    """
    idx = pd.bdate_range("2020-01-01", periods=n)
    rng = np.random.default_rng(seed)
    close = pd.Series(
        100.0 * np.exp(np.cumsum(rng.normal(0.0006, 0.011, n))), index=idx
    )
    open_ = close.vbt.fshift(1).fillna(close.iloc[0])
    return pd.DataFrame({"open": open_, "high": close, "low": close, "close": close})


def _orders_before(pf, cutoff: pd.Timestamp) -> pd.DataFrame:
    records = pf.orders.records_readable
    return records[records["Fill Index"] < cutoff][_ORDER_COLUMNS].reset_index(drop=True)


def test_channel_is_lagged_by_one_session():
    """upper[t] is the max close over [t-entry_n, t-1], and never includes t."""
    from strategies.donchian_breakout import donchian_channel

    close = pd.Series(
        np.arange(20.0), index=pd.bdate_range("2021-01-01", periods=20)
    )
    upper, lower = donchian_channel(close, entry_n=5, exit_n=3)

    # Warmup: 5 closes to fill the window, plus the one-session lag.
    assert upper.iloc[:5].isna().all()
    assert upper.iloc[5] == 4.0, "the channel must exclude the current close"
    assert lower.iloc[5] == 2.0  # min of closes 2..4, the exit window of 3
    # On a strictly increasing series the close is always above the lagged max,
    # which is exactly what makes it a breakout.
    assert (close.iloc[5:] > upper.iloc[5:]).all()


def test_orders_are_causal(eur_usd):
    """Orders before t are identical whether or not the data after t exists."""
    from strategies.donchian_breakout import pipeline

    cut = int(len(eur_usd.close) * 0.8)
    cutoff = eur_usd.wrapper.index[cut]

    pf_full, _ = pipeline(eur_usd, entry_n=100, exit_n=50)
    pf_truncated, _ = pipeline(eur_usd.iloc[:cut], entry_n=100, exit_n=50)

    full = _orders_before(pf_full, cutoff)
    truncated = _orders_before(pf_truncated, cutoff)

    assert len(full) > 0, "no order before the cut — the test would be vacuous"
    pd.testing.assert_frame_equal(
        full,
        truncated,
        check_exact=False,
        obj=f"orders before {cutoff} (future data leaked into the past)",
    )


def test_next_open_fills_at_the_following_session_open(eur_usd):
    """Every order price is the open of the session after the signal bar."""
    from strategies.donchian_breakout import pipeline
    from strategies.gold_momentum import _daily_open

    pf, _ = pipeline(eur_usd, entry_n=55, exit_n=27, fill="next_open", slippage=0.0)

    opens = _daily_open(eur_usd).reindex(pf.wrapper.index)
    records = pf.orders.records_readable
    assert len(records) > 0

    index = pf.wrapper.index
    for _, order in records.iterrows():
        signal_pos = index.get_loc(order["Signal Index"])
        fill_stamp = index[signal_pos + 1]
        assert order["Fill Index"] == fill_stamp, (
            f"signal on {order['Signal Index']} filled on {order['Fill Index']}, "
            f"expected the next session {fill_stamp}"
        )
        assert order["Price"] == pytest.approx(float(opens.loc[fill_stamp]), rel=1e-12), (
            f"order filled at {order['Price']} on {fill_stamp}, "
            f"expected that session's open {float(opens.loc[fill_stamp])}"
        )


def test_entries_and_exits_are_channel_transitions():
    """One Buy per entry-channel crossing, one Sell per exit-channel crossing."""
    from strategies.donchian_breakout import pipeline

    frame = synthetic_daily()
    pf, ind = pipeline(
        frame, entry_n=20, exit_n=10, target_vol=None, fill="close", slippage=0.0
    )
    orders = pf.orders.records_readable
    assert len(orders) >= 4, f"only {len(orders)} orders — the test would be vacuous"

    sides = list(orders["Side"])
    assert sides[0] == "Buy"
    assert all(a != b for a, b in zip(sides, sides[1:])), (
        f"orders do not alternate Buy/Sell ({sides}): the position state is not "
        "latched and the sleeve re-enters on every new high"
    )

    for _, order in orders.iterrows():
        stamp = order["Signal Index"]
        if order["Side"] == "Buy":
            assert ind.close[stamp] > ind.upper[stamp], (
                f"Buy signalled on {stamp} without a new {20}-session high"
            )
        else:
            assert ind.close[stamp] < ind.lower[stamp], (
                f"Sell signalled on {stamp} without a new {10}-session low"
            )


def test_exit_channel_defaults_to_half_the_entry_channel():
    """exit_n=None is the Turtle asymmetry, not a symmetric channel."""
    from strategies.donchian_breakout import pipeline

    frame = synthetic_daily()
    _, implicit = pipeline(frame, entry_n=40, target_vol=None)
    _, explicit = pipeline(frame, entry_n=40, exit_n=20, target_vol=None)
    _, symmetric = pipeline(frame, entry_n=40, exit_n=40, target_vol=None)

    pd.testing.assert_series_equal(implicit.lower, explicit.lower)
    assert not implicit.lower.equals(symmetric.lower)
