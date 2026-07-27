"""Contract of the multi-instrument TSMOM wrapper.

Four properties, each of which would fail silently in a backtest report:

1. **Causality.** Truncating the sample must not change a single order placed
   before the cut. Anything that peeks — a full-sample volatility, a centred
   rolling window — shows up here and nowhere else.
2. **The next-open fill.** ``fill="next_open"`` claims to reproduce what MT5
   does: decide on ``close[t]``, fill at ``open[t+1]``. That claim is checked
   against the session opens themselves, not against a snapshot.
3. **No drift against the gold sleeve.** ``tsmom.pipeline("XAU-USD")`` must be
   the gold sleeve, bit for bit — the registry is data plumbing, not a variant.
4. An unknown symbol fails loudly.
"""

from __future__ import annotations

import pandas as pd
import pytest

# Columns compared across runs: what was traded, when, at what price and size.
_ORDER_COLUMNS: list[str] = ["Fill Index", "Size", "Price", "Side"]

_EQUIVALENCE_METRICS: list[str] = ["total_return", "total_trades", "sharpe_ratio"]


@pytest.fixture(scope="module")
def eur_usd():
    """Broker daily EUR-USD, loaded once."""
    from strategies.tsmom import load_instrument

    _, data = load_instrument("EUR-USD")
    return data


def _orders_before(pf, cutoff: pd.Timestamp) -> pd.DataFrame:
    records = pf.orders.records_readable
    return records[records["Fill Index"] < cutoff][_ORDER_COLUMNS].reset_index(drop=True)


def test_orders_are_causal(eur_usd):
    """Orders before t are identical whether or not the data after t exists."""
    from strategies.gold_momentum import pipeline

    cut = int(len(eur_usd.close) * 0.8)
    cutoff = eur_usd.wrapper.index[cut]

    pf_full, _ = pipeline(eur_usd)
    pf_truncated, _ = pipeline(eur_usd.iloc[:cut])

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
    from strategies.gold_momentum import _daily_open
    from strategies.tsmom import pipeline

    pf, _ = pipeline("EUR-USD", fill="next_open", slippage=0.0)

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


def test_gold_runs_identically_through_the_registry():
    """tsmom.pipeline('XAU-USD') is the gold sleeve, not a variant of it."""
    from strategies import gold_momentum, tsmom
    from utils import load_gold_data

    _, gold_data = load_gold_data()

    pf_direct, _ = gold_momentum.pipeline(gold_data)
    pf_registry, _ = tsmom.pipeline("XAU-USD")

    direct = pf_direct.stats(_EQUIVALENCE_METRICS)
    registry = pf_registry.stats(_EQUIVALENCE_METRICS)

    pd.testing.assert_series_equal(
        direct,
        registry,
        obj="XAU-USD stats through tsmom vs gold_momentum",
    )


def test_yahoo_loader_matches_the_broker_conventions():
    """A screening instrument loads like a broker one: naive index, daily bars.

    The sleeve reads `index.hour` to cut its sessions, so a tz-aware or a
    calendar-shifted index would move every boundary without any error.
    """
    from strategies.tsmom import load_instrument

    raw, data = load_instrument("US500")

    assert raw.index.tz is None
    assert raw.index.is_monotonic_increasing
    assert {"open", "high", "low", "close", "volume"} <= set(raw.columns)
    assert list(data.close.index) == list(raw.index)
    assert raw.index.to_series().diff().median() >= pd.Timedelta(days=1)


def test_loader_override_switches_the_source():
    """`loader_override` reads the other history of the same instrument."""
    from strategies.tsmom import INSTRUMENTS, load_instrument

    assert INSTRUMENTS["JPN225"].loader == "yahoo"
    long_raw, _ = load_instrument("JPN225")
    broker_raw, _ = load_instrument("JPN225", loader_override="mt5")

    assert broker_raw.index.min() > long_raw.index.min()


def test_unknown_symbol_raises():
    """An unknown symbol names the ones that exist."""
    from strategies.tsmom import load_instrument

    with pytest.raises(KeyError, match="ZZZ"):
        load_instrument("ZZZ")
