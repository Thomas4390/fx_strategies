"""Tests for the x10 strategy module (``docs/specs/xau_x10_spec.md``).

The state machine itself is covered bar by bar in ``test_x10_engine.py``; what
is checked here is the assembly around it — the M1/M5 indexing of
``prepare_inputs``, the equivalence between the engine's own trade registry and
the ``vbt.Portfolio`` built from it, the §13 trace contract, and the holdout
guard on the two sweep paths.

The equivalence test is the load-bearing one: the registry is what the MQL5 and
QuantConnect ports will be reconciled against, while the portfolio is what any
future campaign will measure. If the two ever disagree, every number published
from one of them describes the other's strategy.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import vectorbtpro as vbt

from framework.x10_engine import CONTRACT_SIZE
from strategies.xau_x10 import (
    TRACE_COLUMNS,
    count_summary,
    create_cv_pipeline,
    emit_event_trace,
    pipeline,
    pipeline_nb,
    prepare_inputs,
    run_grid,
)

# ═══════════════════════════════════════════════════════════════════════
# FIXTURES
# ═══════════════════════════════════════════════════════════════════════


def synthetic_m1(
    start: str = "2024-03-01", days: int = 30, seed: int = 3, sigma: float = 0.12
) -> pd.DataFrame:
    """A gold-shaped M1 random walk: no volume, three decimals, naive New York.

    The volatility is picked so the walk crosses several x10 levels over the
    window — a series that never approaches a level would make every test here
    pass vacuously.
    """
    n = days * 24 * 60
    rng = np.random.default_rng(seed)
    close = 2000.0 + np.cumsum(rng.normal(0.0, sigma, n))
    high = close + np.abs(rng.normal(0.0, sigma, n))
    low = close - np.abs(rng.normal(0.0, sigma, n))
    open_ = np.empty(n)
    open_[0] = close[0]
    open_[1:] = close[:-1]
    open_ = np.clip(open_, low, high)
    index = pd.date_range(start, periods=n, freq="1min", name="date")
    return pd.DataFrame(
        {
            "open": np.round(open_, 3),
            "high": np.round(high, 3),
            "low": np.round(low, 3),
            "close": np.round(close, 3),
        },
        index=index,
    )


@pytest.fixture(scope="module")
def m1() -> pd.DataFrame:
    return synthetic_m1()


@pytest.fixture(scope="module")
def run(m1):
    return pipeline(m1)


# ═══════════════════════════════════════════════════════════════════════
# prepare_inputs — the M1/M5 indexing (§1, §2)
# ═══════════════════════════════════════════════════════════════════════


def test_prepare_inputs_partitions_the_m1_bars_across_the_m5_grid(m1):
    inputs = prepare_inputs(m1, dxy=None, dxy_path="/nonexistent")
    assert inputs.m1_start[0] == 0
    assert inputs.m1_end[-1] == len(inputs.m1)
    # Contiguous, ordered, and every M1 bar owned by exactly one M5 bar.
    np.testing.assert_array_equal(inputs.m1_start[1:], inputs.m1_end[:-1])
    assert int((inputs.m1_end - inputs.m1_start).sum()) == len(inputs.m1)
    assert bool(np.all(inputs.m1_end > inputs.m1_start))

    # And the M5 bar really is the aggregate of the M1 bars it points at.
    for i in (0, 17, 5_000, len(inputs.m5) - 1):
        j0, j1 = int(inputs.m1_start[i]), int(inputs.m1_end[i])
        chunk = inputs.m1.iloc[j0:j1]
        assert inputs.m5["open"].iloc[i] == chunk["open"].iloc[0]
        assert inputs.m5["close"].iloc[i] == chunk["close"].iloc[-1]
        assert inputs.m5["high"].iloc[i] == chunk["high"].max()
        assert inputs.m5["low"].iloc[i] == chunk["low"].min()


def test_prepare_inputs_accepts_a_vbt_data_wrapper(m1):
    capped = m1.copy()
    capped.columns = [c.capitalize() for c in capped.columns]
    data = vbt.Data.from_data({"XAUUSD": capped}, tz_localize=False, tz_convert=False)
    from_frame = prepare_inputs(m1, dxy=None, dxy_path="/nonexistent")
    from_data = prepare_inputs(data, dxy=None, dxy_path="/nonexistent")
    pd.testing.assert_frame_equal(from_frame.m5, from_data.m5)


def test_an_absent_dollar_basket_is_neutral_not_fatal(m1):
    """§6.3: an undefined DXY scores 0; it never stops a run."""
    inputs = prepare_inputs(m1, dxy=None, dxy_path="/nonexistent")
    assert np.isnan(inputs.dxy).all()
    pf, ind = pipeline(m1, inputs=inputs)
    assert (ind.trades["ctx_dxy"] == 0.0).all()


def test_an_h1_dollar_basket_is_served_one_hour_late(m1):
    """§6.1 applies to the DXY exactly as to the EMA50: shift(1) then ffill."""
    hourly = pd.Series(
        100.0 + np.arange(24 * 30, dtype=float),
        index=pd.date_range("2024-03-01", periods=24 * 30, freq="1h"),
    )
    inputs = prepare_inputs(m1, dxy=hourly)
    m5 = inputs.m5.index
    at_0100 = int(np.flatnonzero(m5 == pd.Timestamp("2024-03-01 01:00"))[0])
    at_0155 = int(np.flatnonzero(m5 == pd.Timestamp("2024-03-01 01:55"))[0])
    assert np.isnan(inputs.dxy[0])  # no closed hour yet
    assert inputs.dxy[at_0100] == hourly.iloc[0]
    assert inputs.dxy[at_0155] == hourly.iloc[0]


# ═══════════════════════════════════════════════════════════════════════
# Registry <-> portfolio equivalence (§9, §11)
# ═══════════════════════════════════════════════════════════════════════


def test_the_fixture_actually_trades(run):
    _, ind = run
    assert len(ind.trades) >= 5
    assert set(ind.events["event_name"]) >= {"ARM", "ENTRY", "EXIT"}


def test_the_portfolio_holds_exactly_the_trades_of_the_registry(run):
    pf, ind = run
    assert pf.orders.count() == 2 * len(ind.trades)
    assert pf.trades.count() == len(ind.trades)


def test_every_trade_pnl_matches_the_registry_to_the_cent(run):
    pf, ind = run
    registry = (
        (ind.trades["exit_px"] - ind.trades["fill_px"])
        * ind.trades["q"]
        * ind.trades["lots"]
        * CONTRACT_SIZE
    ).to_numpy()
    engine_pnl = np.sort(registry)
    portfolio_pnl = np.sort(pf.trades.records_readable["PnL"].to_numpy())
    assert float(np.max(np.abs(engine_pnl - portfolio_pnl))) < 1e-8


def test_every_trade_is_closed_and_directional(run):
    _, ind = run
    assert ind.trades["exit_px"].notna().all()
    assert ind.trades["exit_reason"].gt(0).all()
    assert set(np.unique(ind.trades["q"])) <= {-1.0, 1.0}
    assert (ind.trades["lots"] >= 0.01).all()


def test_positions_never_overlap(run):
    """§9: one position at a time, all four scenarios pooled."""
    _, ind = run
    fills = ind.trades["i_fill_m1"].to_numpy()
    exits = ind.trades["i_exit_m1"].to_numpy()
    assert bool(np.all(fills[1:] > exits[:-1]))


def test_two_runs_of_the_pipeline_agree_bit_for_bit(m1):
    inputs = prepare_inputs(m1, dxy=None, dxy_path="/nonexistent")
    _, first = pipeline(m1, inputs=inputs)
    _, second = pipeline(m1, inputs=inputs)
    pd.testing.assert_frame_equal(first.events, second.events)
    pd.testing.assert_frame_equal(first.trades, second.trades)


# ═══════════════════════════════════════════════════════════════════════
# The §13 trace contract
# ═══════════════════════════════════════════════════════════════════════


def test_the_trace_is_the_twenty_one_columns_of_the_spec(run, tmp_path):
    _, ind = run
    path = tmp_path / "x10_trace.csv"
    trace = emit_event_trace(ind, path)

    assert tuple(trace.columns) == TRACE_COLUMNS
    assert len(trace) == len(ind.events)
    header = path.read_text().splitlines()[0]
    assert header == ",".join(TRACE_COLUMNS)

    reread = pd.read_csv(path)
    assert tuple(reread.columns) == TRACE_COLUMNS
    assert set(reread["event"]) <= {"ARM", "BREAK", "SWEEP", "ENTRY", "EXIT", "CANCEL"}
    assert set(reread["scenario"].dropna()) <= {
        "BREAK_LONG",
        "BREAK_SHORT",
        "REV_LONG",
        "REV_SHORT",
    }
    assert set(reread["exit_reason"].dropna()) <= {"STOP", "TARGET", "TIME", "SESSION"}
    assert set(reread["d"]) <= {-1, 1}
    assert set(reread["ctx_ema"]) <= {-1, 0, 1}


def test_the_trace_stamps_are_utc(run):
    _, ind = run
    trace = emit_event_trace(ind)
    first_local = ind.events["ts_decision"].iloc[0]
    first_utc = pd.Timestamp(trace["ts_decision"].iloc[0])
    offset = first_utc - first_local
    assert offset in (pd.Timedelta(hours=4), pd.Timedelta(hours=5))  # EDT / EST


def test_prices_are_only_carried_by_the_events_that_have_them(run):
    _, ind = run
    trace = emit_event_trace(ind)
    no_fill = trace[~trace["event"].isin(["ENTRY", "EXIT"])]
    assert no_fill["fill_px"].isna().all()
    assert no_fill["exit_px"].isna().all()
    assert (trace.loc[trace["event"] == "EXIT", "exit_reason"] != "").all()
    assert (trace.loc[trace["event"] == "ARM", "exit_reason"] == "").all()


def test_the_extended_trace_adds_the_cancel_reason(run):
    _, ind = run
    trace = emit_event_trace(ind, extended=True)
    assert tuple(trace.columns) == (*TRACE_COLUMNS, "cancel_reason")
    cancels = trace[trace["event"] == "CANCEL"]
    assert (cancels["cancel_reason"] != "").all()


def test_the_counts_add_up(run):
    _, ind = run
    summary = count_summary(ind)
    assert summary["events_by_type"].sum() == len(ind.events)
    assert summary["trades_by_scenario"].sum() == len(ind.trades)
    assert summary["exits_by_reason"].sum() == len(ind.trades)
    # §8 asks for a rejection rate per year: both halves must be countable.
    assert summary["r_candidates_by_year"].sum() >= len(ind.trades)


# ═══════════════════════════════════════════════════════════════════════
# Holdout guard (docs/research/HOLDOUT_POLICY.md)
# ═══════════════════════════════════════════════════════════════════════


def test_run_grid_refuses_the_frozen_slice():
    frozen = synthetic_m1(start="2025-12-20", days=20)
    assert frozen.index.max() >= pd.Timestamp("2026-01-01")
    with pytest.raises(RuntimeError, match="frozen slice"):
        run_grid(frozen, z=[0.5, 1.0], a_min=0.2, k_s=1.0)


@pytest.mark.parametrize("tz", [None, "UTC"])
def test_pipeline_nb_refuses_the_frozen_slice(tz):
    """The scalar-metric path is a ranking path: it needs the guard too.

    ``run_grid`` is the usual door but not the only one — a sweep script can
    call ``pipeline_nb`` directly with ``vbt.Param`` values, and a guard that
    only lives in the wrapper is a guard one import away from being bypassed.
    """
    frozen = synthetic_m1(start="2025-12-20", days=20)
    if tz is not None:
        frozen.index = frozen.index.tz_localize("America/New_York").tz_convert(tz)
    with pytest.raises(RuntimeError, match="frozen slice"):
        pipeline_nb(frozen, z=1.0, a_min=0.2, k_s=1.0)


def test_the_cv_pipeline_refuses_the_frozen_slice():
    frozen = synthetic_m1(start="2025-12-20", days=20)
    cv = create_cv_pipeline("from_purged_walkforward")
    with pytest.raises(RuntimeError, match="frozen slice"):
        cv(frozen, z=1.0, a_min=0.2, k_s=1.0)


def test_a_splitter_built_on_frozen_bars_is_refused_at_construction():
    daily = pd.date_range("2025-10-01", "2026-02-01", freq="1D")
    splitter = vbt.Splitter.from_purged_walkforward(daily, n_folds=5, n_test_folds=1)
    with pytest.raises(RuntimeError, match="frozen slice"):
        create_cv_pipeline(splitter)
