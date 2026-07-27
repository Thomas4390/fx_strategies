"""Contract of the dual / acceleration momentum sleeve.

What is pinned here is what a backtest report cannot show:

1. **The acceleration is what it claims to be** — the fast horizon's return
   minus the slow one's, checked against the arithmetic rather than against a
   snapshot.
2. **The gate is a door.** No long session may coexist with a negative
   acceleration; if it can, the mode is a label rather than a filter.
3. **The brake is a dimmer.** It must not remove a single trade — same entries,
   same exits as the unfiltered sleeve — and must halve the leverage exactly
   where the acceleration is negative.
4. **Causality.** Truncating the sample must not move an order placed before
   the cut. The acceleration reads 252 sessions back, which is exactly the kind
   of window that leaks if it is written the wrong way round.
5. An unknown mode fails loudly, and so does an inverted horizon pair.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

_ORDER_COLUMNS: list[str] = ["Fill Index", "Size", "Price", "Side"]


@pytest.fixture(scope="module")
def eur_usd():
    """Broker daily EUR-USD, loaded once — the same fixture as test_tsmom."""
    from strategies.tsmom import load_instrument

    _, data = load_instrument("EUR-USD")
    return data


def test_accel_score_is_the_horizon_difference(eur_usd):
    """accel = pct_change(fast) - pct_change(slow), NaN through the warmup."""
    from strategies.dual_momentum import accel_score
    from strategies.gold_momentum import _daily_close

    close = _daily_close(eur_usd)
    accel = accel_score(close, fast_n=63, slow_n=252)

    expected = close.vbt.pct_change(63) - close.vbt.pct_change(252)
    pd.testing.assert_series_equal(accel, expected, check_names=False)

    assert accel.iloc[:252].isna().all(), "the slow horizon must not resolve early"
    assert accel.iloc[252:].notna().any()


def test_gate_never_holds_a_decelerating_long(eur_usd):
    """The gate mode is a door: score > 0 AND accel > 0, never one alone."""
    from strategies.dual_momentum import pipeline

    _, ind = pipeline(eur_usd, mode="gate")

    assert ind.long_ok.any(), "no long at all — the test would be vacuous"
    assert not (ind.long_ok & (ind.accel < 0)).any()
    assert (ind.long_ok == ((ind.score > 0) & (ind.accel > 0))).all()

    # And it removes trades rather than resizing them: that is what separates it
    # from the brake.
    pf_gate, _ = pipeline(eur_usd, mode="gate")
    pf_brake, _ = pipeline(eur_usd, mode="brake")
    assert int(pf_gate.trades.count()) < int(pf_brake.trades.count())


def test_brake_halves_the_leverage_without_touching_the_trades(eur_usd):
    """The brake mode is a dimmer: same state, leverage x0.5 where accel < 0."""
    from strategies.dual_momentum import BRAKE_FACTOR, pipeline

    pf_brake, brake = pipeline(eur_usd, mode="brake")
    pf_gate, gate = pipeline(eur_usd, mode="gate")

    # The traded state is the unfiltered one — no trade is removed.
    assert (brake.long_ok == (brake.score > 0)).all()

    falling = brake.accel < 0
    assert falling.any(), "no decelerating session — the test would be vacuous"
    expected = gate.leverage * np.where(falling.to_numpy(), BRAKE_FACTOR, 1.0)
    pd.testing.assert_series_equal(brake.leverage, expected, check_names=False)

    # Sizing only: the entries and exits are those of the unfiltered sleeve, so
    # the trade count matches it and not the gate's.
    from strategies.gold_momentum import pipeline as base_pipeline

    pf_base, _ = base_pipeline(eur_usd, fill="next_open")
    assert int(pf_brake.trades.count()) == int(pf_base.trades.count())
    assert int(pf_gate.trades.count()) != int(pf_base.trades.count())


@pytest.mark.parametrize("mode", ["gate", "brake"])
def test_orders_are_causal(eur_usd, mode):
    """Orders before t are identical whether or not the data after t exists."""
    from strategies.dual_momentum import pipeline

    cut = int(len(eur_usd.close) * 0.8)
    cutoff = eur_usd.wrapper.index[cut]

    def orders_before(pf) -> pd.DataFrame:
        rec = pf.orders.records_readable
        return rec[rec["Fill Index"] < cutoff][_ORDER_COLUMNS].reset_index(drop=True)

    pf_full, _ = pipeline(eur_usd, mode=mode)
    pf_truncated, _ = pipeline(eur_usd.iloc[:cut], mode=mode)

    full = orders_before(pf_full)
    assert len(full) > 0, "no order before the cut — the test would be vacuous"
    pd.testing.assert_frame_equal(
        full,
        orders_before(pf_truncated),
        check_exact=False,
        obj=f"orders before {cutoff} in mode {mode} (future data leaked into the past)",
    )


def test_invalid_parameters_raise(eur_usd):
    """An unknown mode and an inverted horizon pair both fail loudly."""
    from strategies.dual_momentum import accel_score, pipeline
    from strategies.gold_momentum import _daily_close

    with pytest.raises(ValueError, match="mode"):
        pipeline(eur_usd, mode="filter")
    with pytest.raises(ValueError, match="fast_n"):
        accel_score(_daily_close(eur_usd), fast_n=252, slow_n=63)
