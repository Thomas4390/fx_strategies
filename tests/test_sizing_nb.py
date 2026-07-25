"""Unit tests for the path-dependent sizing overlays.

Every case uses a hand-built price path and hand-placed signals, so the size of
each order is computable with pen and paper. That is the only way to validate a
state machine whose output depends on its own history.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import vectorbtpro as vbt

from framework.sizing_nb import (
    MODE_ANTI_MART,
    MODE_COMBO,
    MODE_FLAT,
    MODE_GRID,
    MODE_MARTINGALE,
    build_overlay_kwargs,
    make_params,
)

INIT_CASH = 100_000.0


def _run(prices, entries, exits, params, atr=None):
    """Simulate a single column and return (portfolio, state)."""
    idx = pd.date_range("2020-01-01", periods=len(prices), freq="D")
    close = pd.Series(np.asarray(prices, dtype=np.float64), index=idx)
    if atr is None:
        atr = np.full((len(prices), 1), 1.0)
    memory: dict = {}
    kwargs = build_overlay_kwargs(params, np.asarray(atr, dtype=np.float64), memory=memory)
    pf = vbt.Portfolio.from_signals(
        close=close,
        entries=pd.Series(entries, index=idx),
        exits=pd.Series(exits, index=idx),
        short_entries=False,
        short_exits=False,
        init_cash=INIT_CASH,
        fees=0.0,
        slippage=0.0,
        freq="1D",
        **kwargs,
    )
    return pf, memory["state"]


def _entry_sizes(pf):
    """Value ordered on each buy, in portfolio-value fractions at the time."""
    orders = pf.orders.records_readable
    buys = orders[orders["Side"] == "Buy"]
    return buys["Size"].to_numpy() * buys["Price"].to_numpy()


# ── fixtures: a price path producing 3 losses then a win ───────────────
# Trades open on the entry bar's close and shut on the exit bar's close.
# A trailing idle bar is appended so the final close is observable: the state
# machine can only register a trade's outcome on the bar *after* it goes flat.
LOSS_LOSS_LOSS_WIN = [
    100.0, 100.0, 99.0, 98.0,  # trade 1: 100 -> 98   LOSS
    98.0, 98.0, 97.0, 96.0,    # trade 2:  98 -> 96   LOSS
    96.0, 96.0, 95.0, 94.0,    # trade 3:  96 -> 94   LOSS
    94.0, 94.0, 96.0, 98.0,    # trade 4:  94 -> 98   WIN
    98.0,                       # idle
]
ENTRIES_4 = [False, True, False, False] * 4 + [False]
EXITS_4 = [False, False, False, True] * 4 + [False]


def test_flat_keeps_size_constant():
    params = make_params(MODE_FLAT, base_size=0.10)
    pf, state = _run(LOSS_LOSS_LOSS_WIN, ENTRIES_4, EXITS_4, params)
    sizes = _entry_sizes(pf)
    assert len(sizes) == 4
    ratios = sizes / sizes[0]
    np.testing.assert_allclose(ratios, [1.0, 1.0, 1.0, 1.0], rtol=0.05)
    assert state["n_addons"][0] == 0
    assert state["loss_streak"][0] == 0  # reset by the final win


def test_martingale_doubles_after_each_loss():
    """3 losses then a win -> multiples 1, 2, 4, 8 (mult=2, n_max=3)."""
    params = make_params(MODE_MARTINGALE, base_size=0.05, mult=2.0, n_max=3, max_total=10.0)
    pf, state = _run(LOSS_LOSS_LOSS_WIN, ENTRIES_4, EXITS_4, params)
    sizes = _entry_sizes(pf)
    assert len(sizes) == 4
    ratios = sizes / sizes[0]
    np.testing.assert_allclose(ratios, [1.0, 2.0, 4.0, 8.0], rtol=0.10)
    assert state["loss_streak"][0] == 0
    assert state["win_streak"][0] == 1


def test_martingale_caps_at_n_max():
    """With n_max=2 the multiple saturates at 4x: 1, 2, 4, 4."""
    params = make_params(MODE_MARTINGALE, base_size=0.05, mult=2.0, n_max=2, max_total=10.0)
    prices = LOSS_LOSS_LOSS_WIN[:12] + [94.0, 94.0, 93.0, 92.0, 92.0]  # 4th trade also loses
    pf, _ = _run(prices, ENTRIES_4, EXITS_4, params)
    ratios = _entry_sizes(pf) / _entry_sizes(pf)[0]
    np.testing.assert_allclose(ratios, [1.0, 2.0, 4.0, 4.0], rtol=0.10)


def test_max_total_caps_the_martingale():
    """max_total clamps the order even when the streak asks for more."""
    params = make_params(MODE_MARTINGALE, base_size=0.05, mult=2.0, n_max=3, max_total=0.15)
    pf, _ = _run(LOSS_LOSS_LOSS_WIN, ENTRIES_4, EXITS_4, params)
    ratios = _entry_sizes(pf) / _entry_sizes(pf)[0]
    # 0.05, 0.10, then clamped to 0.15 (not 0.20), and 0.15 again (not 0.40)
    np.testing.assert_allclose(ratios, [1.0, 2.0, 3.0, 3.0], rtol=0.10)


def test_anti_martingale_doubles_after_each_win():
    """3 wins then a loss -> multiples 1, 2, 4, 8."""
    prices = [
        100.0, 100.0, 101.0, 102.0,
        102.0, 102.0, 103.0, 104.0,
        104.0, 104.0, 105.0, 106.0,
        106.0, 106.0, 105.0, 104.0,
        104.0,
    ]
    params = make_params(MODE_ANTI_MART, base_size=0.05, mult=2.0, n_max=3, max_total=10.0)
    pf, state = _run(prices, ENTRIES_4, EXITS_4, params)
    ratios = _entry_sizes(pf) / _entry_sizes(pf)[0]
    np.testing.assert_allclose(ratios, [1.0, 2.0, 4.0, 8.0], rtol=0.10)
    assert state["win_streak"][0] == 0
    assert state["loss_streak"][0] == 1


def test_grid_adds_on_adverse_excursion():
    """ATR=1 on a 100 price, grid_k=2 -> a level every 2% adverse."""
    # Enter at 100, drift down through -2%, -4%, -6%, then exit.
    prices = [100.0, 100.0, 98.0, 96.0, 94.0, 94.0]
    entries = [False, True, False, False, False, False]
    exits = [False, False, False, False, False, True]
    params = make_params(
        MODE_GRID, base_size=0.05, grid_k=2.0, n_levels=3, grid_mult=1.0, max_total=1.0,
        basket_stop=0.99,
    )
    pf, state = _run(prices, entries, exits, params)
    sizes = _entry_sizes(pf)
    assert state["n_addons"][0] == 3, f"expected 3 add-ons, got {state['n_addons'][0]}"
    assert len(sizes) == 4  # initial + 3 add-ons
    np.testing.assert_allclose(sizes / sizes[0], [1.0, 1.0, 1.0, 1.0], rtol=0.15)


def test_grid_respects_n_levels():
    prices = [100.0, 100.0, 98.0, 96.0, 94.0, 92.0, 90.0, 90.0]
    entries = [False, True] + [False] * 6
    exits = [False] * 7 + [True]
    params = make_params(
        MODE_GRID, base_size=0.05, grid_k=2.0, n_levels=2, grid_mult=1.0, max_total=1.0,
        basket_stop=0.99,
    )
    _, state = _run(prices, entries, exits, params)
    assert state["n_addons"][0] == 2


def test_grid_martingale_levels_scale_by_grid_mult():
    """grid_mult=2 -> add-ons of 1x, 2x, 4x the base size."""
    prices = [100.0, 100.0, 98.0, 96.0, 94.0, 94.0]
    entries = [False, True, False, False, False, False]
    exits = [False, False, False, False, False, True]
    params = make_params(
        MODE_GRID, base_size=0.02, grid_k=2.0, n_levels=3, grid_mult=2.0, max_total=1.0,
        basket_stop=0.99,
    )
    pf, state = _run(prices, entries, exits, params)
    assert state["n_addons"][0] == 3
    sizes = _entry_sizes(pf)
    np.testing.assert_allclose(sizes / sizes[0], [1.0, 2.0, 4.0, 8.0], rtol=0.15)


def test_basket_stop_closes_the_stack():
    """A deep adverse move must close the position, not keep averaging down."""
    prices = [100.0, 100.0, 98.0, 96.0, 80.0, 80.0, 80.0]
    entries = [False, True] + [False] * 5
    exits = [False] * 7
    params = make_params(
        MODE_GRID, base_size=0.50, grid_k=2.0, n_levels=5, grid_mult=1.0,
        basket_stop=0.05, max_total=5.0,
    )
    pf, _ = _run(prices, entries, exits, params)
    # Position is flat before the end despite no exit signal ever firing.
    assert pf.assets.iloc[-1] == pytest.approx(0.0, abs=1e-9)


def test_kill_switch_reverts_to_flat_and_never_rearms():
    """Past kill_dd the overlay stops escalating, permanently."""
    # Six consecutive losing trades, each -10%, on an oversized martingale.
    prices, entries, exits = [], [], []
    px = 100.0
    for _ in range(6):
        prices += [px, px, px * 0.95, px * 0.90]
        entries += [False, True, False, False]
        exits += [False, False, False, True]
        px *= 0.90
    params = make_params(
        MODE_MARTINGALE, base_size=0.30, mult=2.0, n_max=5, max_total=10.0,
        basket_stop=0.99, kill_dd=0.20,
    )
    _, state = _run(prices, entries, exits, params)
    assert state["killed"][0] == 1
    assert state["n_killed"][0] == 1, "the kill switch must fire exactly once, never re-arm"


def test_combo_applies_both_martingale_and_grid():
    """COMBO escalates the *initial* size on losses and ladders *within* a trade.

    The two mechanisms are deliberately independent: the martingale multiplier
    scales the opening order, the grid multiplier scales the add-ons. Composing
    them multiplicatively would make peak exposure ``mult**n_max * grid_mult**
    n_levels``, which reaches four digits before any guard notices.
    """
    # ATR is 1.0 throughout, so with grid_k=2 a level needs ~2% of adverse move.
    prices = [
        100.0,          # 0 idle
        100.0,          # 1 trade 1 opens at 100
        99.5,           # 2 -0.5%: too shallow for a grid level
        99.5,           # 3 trade 1 closes -> LOSS
        99.5,           # 4 idle: the loss is registered here
        99.5,           # 5 trade 2 opens at 2x base
        97.0,           # 6 -2.5% -> one grid add-on at 1x base
        97.0,           # 7
        97.0,           # 8 trade 2 closes
        97.0,           # 9 idle
    ]
    entries = [False, True, False, False, False, True, False, False, False, False]
    exits = [False, False, False, True, False, False, False, False, True, False]
    params = make_params(
        MODE_COMBO, base_size=0.02, mult=2.0, n_max=3, grid_k=2.0, n_levels=2,
        grid_mult=1.0, max_total=1.0, basket_stop=0.99,
    )
    pf, state = _run(prices, entries, exits, params)
    sizes = _entry_sizes(pf)
    assert len(sizes) == 3, f"expected open/open/add-on, got {len(sizes)} buys"
    # 1x base, then 2x base after the loss, then a 1x base grid add-on.
    np.testing.assert_allclose(sizes / sizes[0], [1.0, 2.0, 1.0], rtol=0.10)
    assert state["n_addons"][0] == 1, "COMBO must also ladder inside the trade"


def test_diagnostics_track_peak_exposure():
    params = make_params(MODE_MARTINGALE, base_size=0.05, mult=2.0, n_max=3, max_total=10.0)
    _, state = _run(LOSS_LOSS_LOSS_WIN, ENTRIES_4, EXITS_4, params)
    assert state["max_total_seen"][0] == pytest.approx(0.05 * 8.0)


def test_make_params_rejects_incoherent_inputs():
    with pytest.raises(ValueError, match="mult"):
        make_params(MODE_MARTINGALE, mult=0.5)
    with pytest.raises(ValueError, match="grid_k"):
        make_params(MODE_GRID, grid_k=0.0)
    with pytest.raises(ValueError, match="max_total"):
        make_params(MODE_FLAT, base_size=1.0, max_total=0.5)
    with pytest.raises(ValueError, match="unknown mode"):
        make_params(99)
