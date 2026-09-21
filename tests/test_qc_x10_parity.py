"""Parity of the QuantConnect x10 port against the reference engine (§13).

The QuantConnect algorithm cannot be run here, but everything it decides lives
in three modules that import nothing from ``AlgorithmImports``:
``src/qc/xau_x10/x10_bars.py``, ``x10_indicators.py`` and ``x10_state.py``.
This file replays them minute by minute on synthetic quotes and walks the
reconciliation ladder of §13 on the **same** inputs the reference kernel gets:

* rung 1 — the M5 and H1 bars the streaming accumulator builds;
* rung 2 — ``atr``, ``v``, ``m``, ``a``, ``vwap``, ``ema50_h1``, ``atr_h1`` and
  the two DXY series, at 1e-9;
* rung 3/4 — the event sequence: type, scenario, level, ``d`` and decision bar;
* rung 4/5 — ``stop``, ``target``, ``r_est`` at 1e-9, and the lots of §11.

Synthetic data only, and on purpose: the reconciliation is a statement about
two implementations of the same arithmetic, not about the gold history, and
reading a real run here would spend a trial of annexe A.3 without declaring it.

The sample is a ten-business-day random walk carrying the two discontinuities
that make the port hard — the 17:00-18:00 New York break and the weekend — so
that the "fill bar must start five minutes after the decision bar" rule of §2
and the 16:55 flat of §10 are both exercised.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from framework.x10_context import SESSION_TZ, dxy4, resample_ohlc, to_session_clock
from framework.x10_engine import (
    CANCEL_REASON_NAMES,
    E_CANCEL_REASON,
    E_D,
    E_EVENT,
    E_EXIT_PX,
    E_EXIT_REASON,
    E_FILL_PX,
    E_I_DECISION,
    E_LEVEL,
    E_R_EST,
    E_SCENARIO,
    E_STOP,
    E_TARGET,
    EVENT_ENTRY,
    EVENT_NAMES,
    EXIT_REASON_NAMES,
    T_LOTS,
    run_engine,
)
from strategies.xau_x10 import prepare_inputs

_QC_DIR = Path(__file__).resolve().parent.parent / "src" / "qc" / "xau_x10"
if str(_QC_DIR) not in sys.path:
    sys.path.insert(0, str(_QC_DIR))

from x10_bars import MinuteQuote  # noqa: E402
from x10_state import EVENT_EXIT as QC_EVENT_EXIT  # noqa: E402
from x10_state import X10Feed  # noqa: E402

# Two points of the grid of annexe A.2: the one the port defaults to, and a
# permissive one that arms far more often — the second is what makes a §10
# ``window`` refusal appear at all in ten days of data.
GRIDS = ((1.0, 0.2, 1.0), (1.5, 0.1, 0.75))

# §12: with ``bid = mid - s/2`` and ``ask = mid + s/2`` the streaming execution
# and the kernel's ``mid ± s/2`` are the same arithmetic, bit for bit.
SPREAD = 0.29
HALF = 0.145
INIT_CASH = 10_000.0

_EPOCH_UTC = pd.Timestamp("1970-01-01", tz="UTC")
_TOL = 1e-9


# ═══════════════════════════════════════════════════════════════════════
# SYNTHETIC SAMPLE
# ═══════════════════════════════════════════════════════════════════════


def _gold_minutes() -> pd.DatetimeIndex:
    """Ten business days of gold minutes, with the three holes that matter.

    The 17:00-18:00 daily break and the weekend are the calendar of §2. The
    three daily outages are not: they are deliberate holes in the feed, and
    they are the only way to separate "the next M5 bar" from "the bar five
    minutes later" — the distinction §2 turns into a ``window`` refusal, which
    the port decides at the fill and the kernel at the decision. The 21:00 one
    also puts bars either side of the New York evening, where the flat of §10
    must **not** fire.
    """
    idx = pd.date_range("2024-06-02 22:00", "2024-06-14 21:00", freq="1min", tz="UTC")
    ny = idx.tz_convert(SESSION_TZ)
    minute = ny.hour * 60 + ny.minute
    dow = ny.dayofweek  # Monday = 0, Sunday = 6
    keep = ~((minute >= 17 * 60) & (minute < 18 * 60))
    keep &= ~((dow == 4) & (minute >= 17 * 60))
    keep &= ~(dow == 5)
    keep &= ~((dow == 6) & (minute < 18 * 60))
    for hour, first, last in ((8, 0, 37), (13, 12, 49), (21, 5, 42)):
        keep &= ~((ny.hour == hour) & (ny.minute >= first) & (ny.minute < last))
    return idx[keep]


def _walk(index: pd.DatetimeIndex, start: float, sigma: float, seed: int) -> pd.DataFrame:
    """Minute OHLC random walk quoted to three decimals, open glued to the close."""
    rng = np.random.default_rng(seed)
    n = len(index)
    close = np.round(start + np.cumsum(rng.normal(0.0, sigma, n)), 3)
    open_ = np.empty(n)
    open_[0] = close[0]
    open_[1:] = close[:-1]
    span = np.abs(rng.normal(0.0, sigma, n))
    high = np.round(np.maximum(open_, close) + span, 3)
    low = np.round(np.minimum(open_, close) - span, 3)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close}, index=index
    )


def _flat(index: pd.DatetimeIndex, start: float, sigma: float, seed: int) -> pd.DataFrame:
    """A currency leg: only its close is read, so the bar is a single price."""
    rng = np.random.default_rng(seed)
    close = start + np.cumsum(rng.normal(0.0, sigma, len(index)))
    return pd.DataFrame(
        {"open": close, "high": close, "low": close, "close": close}, index=index
    )


@pytest.fixture(scope="module", params=GRIDS, ids=lambda g: f"z{g[0]}_a{g[1]}_k{g[2]}")
def sample(request) -> dict:
    """Gold minutes, four currency legs, and everything both engines read."""
    z, a_min, k_s = request.param
    index = _gold_minutes()
    gold = _walk(index, 2000.0, 0.6, seed=20260921)
    legs_utc = {
        "eurusd": _flat(index, 1.0800, 3e-5, seed=1),
        "usdjpy": _flat(index, 150.00, 3e-3, seed=2),
        "gbpusd": _flat(index, 1.2700, 3e-5, seed=3),
        "usdcad": _flat(index, 1.3600, 3e-5, seed=4),
    }

    # The reference clock is naive New York (``prepare_inputs`` converts the
    # gold index itself; the DXY series is handed over already converted).
    legs_ny = {}
    for leg, frame in legs_utc.items():
        local = frame.copy()
        local.index = to_session_clock(pd.DatetimeIndex(local.index))
        legs_ny[leg] = local
    dxy_h1 = dxy4(**{leg: resample_ohlc(f, "1h")["close"] for leg, f in legs_ny.items()})

    inputs = prepare_inputs(gold, dxy=dxy_h1)
    events, trades = run_engine(
        **inputs.kernel_kwargs(SPREAD),
        z=z,
        a_min=a_min,
        k_s=k_s,
        init_cash=INIT_CASH,
    )
    return {
        "grid": (z, a_min, k_s),
        "index": index,
        "gold": gold,
        "legs": legs_utc,
        "inputs": inputs,
        "events": events,
        "trades": trades,
    }


# ═══════════════════════════════════════════════════════════════════════
# THE PORT, REPLAYED
# ═══════════════════════════════════════════════════════════════════════


class _RecordingFeed(X10Feed):
    """``X10Feed`` with the bars and the per-bar context kept for comparison."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.m5_bars: list = []
        self.h1_bars: list = []
        self.contexts: list = []

    def _close_m5(self, bar, live: bool) -> None:  # noqa: D102 - see base class
        super()._close_m5(bar, live)
        self.m5_bars.append(bar)
        self.contexts.append(self.last_context)

    def _close_h1(self, bar) -> None:  # noqa: D102 - see base class
        super()._close_h1(bar)
        self.h1_bars.append(bar)


@pytest.fixture(scope="module")
def replay(sample: dict) -> _RecordingFeed:
    """Run the pure QuantConnect modules over the same minutes, one by one."""
    z, a_min, k_s = sample["grid"]
    feed = _RecordingFeed(z=z, a_min=a_min, k_s=k_s, init_cash=INIT_CASH)
    index = sample["index"]
    gold = sample["gold"]
    legs = sample["legs"]

    minutes = ((index - _EPOCH_UTC) // pd.Timedelta(minutes=1)).to_numpy(dtype=np.int64)
    o, h, low, c = (gold[col].to_numpy(dtype=np.float64) for col in gold.columns)
    leg_closes = {leg: frame["close"].to_numpy(dtype=np.float64) for leg, frame in legs.items()}

    for i, minute in enumerate(minutes):
        quote = MinuteQuote(
            o[i], h[i], low[i], c[i],
            o[i] - HALF, h[i] - HALF, low[i] - HALF, c[i] - HALF,
            o[i] + HALF, h[i] + HALF, low[i] + HALF, c[i] + HALF,
        )
        feed.on_minute(
            int(minute), quote, {leg: arr[i] for leg, arr in leg_closes.items()}
        )
    return feed


def _utc_minutes(index: pd.DatetimeIndex) -> np.ndarray:
    """Naive New York bar labels back to epoch minutes on the UTC clock."""
    utc = pd.DatetimeIndex(index).tz_localize(SESSION_TZ).tz_convert("UTC")
    return ((utc - _EPOCH_UTC) // pd.Timedelta(minutes=1)).to_numpy(dtype=np.int64)


# ═══════════════════════════════════════════════════════════════════════
# RUNG 0 — the sample has to be worth comparing
# ═══════════════════════════════════════════════════════════════════════


def test_sample_carries_the_discontinuities_and_every_event_type(sample: dict):
    """A vacuous sample would make every assertion below pass for free."""
    minutes = _utc_minutes(sample["inputs"].m5.index)
    gaps = np.diff(minutes)
    assert (gaps == 5).sum() > 2_000  # the ordinary case dominates
    assert (gaps == 65).sum() >= 8  # the 17:00-18:00 daily breaks
    assert gaps.max() > 60 * 24  # one weekend
    outages = (gaps != 5) & (gaps != 65) & (gaps < 60 * 24)
    assert outages.sum() >= 20  # and the deliberate feed holes

    codes = set(sample["events"][:, E_EVENT].astype(int).tolist())
    assert codes == set(range(len(EVENT_NAMES))), "every event of §7.4 must occur"

    # The two refusals the port does not decide where the kernel does.
    reasons = sample["events"][:, E_CANCEL_REASON].astype(int)
    assert (reasons == CANCEL_REASON_NAMES.index("window")).sum() >= 1
    assert (reasons == CANCEL_REASON_NAMES.index("ctx")).sum() >= 1


# ═══════════════════════════════════════════════════════════════════════
# RUNG 1 — barres M5 / H1
# ═══════════════════════════════════════════════════════════════════════


def test_m5_bars_match_the_reference_resampling(sample: dict, replay: _RecordingFeed):
    ref = sample["inputs"].m5
    assert len(replay.m5_bars) == len(ref)
    np.testing.assert_array_equal(
        np.array([bar.start for bar in replay.m5_bars]), _utc_minutes(ref.index)
    )
    for col in ("open", "high", "low", "close"):
        np.testing.assert_array_equal(
            np.array([getattr(bar, col) for bar in replay.m5_bars]),
            ref[col].to_numpy(dtype=np.float64),
        )


def test_h1_bars_match_the_reference_resampling(sample: dict, replay: _RecordingFeed):
    m1 = sample["gold"].copy()
    m1.index = to_session_clock(pd.DatetimeIndex(m1.index))
    ref = resample_ohlc(m1, "1h")
    assert len(replay.h1_bars) == len(ref)
    np.testing.assert_array_equal(
        np.array([bar.start for bar in replay.h1_bars]), _utc_minutes(ref.index)
    )
    for col in ("open", "high", "low", "close"):
        np.testing.assert_array_equal(
            np.array([getattr(bar, col) for bar in replay.h1_bars]),
            ref[col].to_numpy(dtype=np.float64),
        )


# ═══════════════════════════════════════════════════════════════════════
# RUNG 2 — indicateurs
# ═══════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize(
    "field",
    ["atr", "v", "m", "a", "vwap", "ema50_h1", "atr_h1", "dxy", "dxy_ema50_h1"],
)
def test_indicators_match_bar_for_bar(sample: dict, replay: _RecordingFeed, field: str):
    """§13 rung 2, at 1e-9: lissage, warmup and H1 causality, all three at once."""
    got = np.array([getattr(ctx, field) for ctx in replay.contexts], dtype=np.float64)
    expected = np.asarray(getattr(sample["inputs"], field), dtype=np.float64)
    np.testing.assert_allclose(got, expected, rtol=0.0, atol=_TOL, equal_nan=True)


def test_x10_levels_match_bar_for_bar(sample: dict, replay: _RecordingFeed):
    inputs = sample["inputs"]
    for field in ("l_inf", "l_sup"):
        got = np.array([getattr(ctx, field) for ctx in replay.contexts])
        np.testing.assert_array_equal(got, np.asarray(getattr(inputs, field)))


# ═══════════════════════════════════════════════════════════════════════
# RUNGS 3-5 — evenements, niveaux de sortie, lots
# ═══════════════════════════════════════════════════════════════════════


def _reference_rows(sample: dict) -> list[tuple]:
    """Reference events, minus the ones the last bar owns.

    The kernel flattens any open position on ``i == n5 - 1`` and takes no
    decision there — it has no ``i+1`` to fill into. A streaming algorithm has
    no such notion of a last bar, so the tail is out of the comparison on both
    sides rather than faked on one of them.
    """
    last = len(sample["inputs"].m5) - 1
    rows = []
    for row in sample["events"]:
        if int(row[E_I_DECISION]) >= last:
            continue
        rows.append(
            (
                int(row[E_I_DECISION]),
                int(row[E_EVENT]),
                int(row[E_SCENARIO]),
                float(row[E_LEVEL]),
                int(row[E_D]),
                int(row[E_EXIT_REASON]),
                int(row[E_CANCEL_REASON]),
            )
        )
    return rows


def _port_rows(sample: dict, replay: _RecordingFeed) -> list[tuple]:
    last = len(sample["inputs"].m5) - 1
    return [
        (
            event.bar_index,
            event.event,
            event.scenario,
            event.level,
            event.d,
            event.exit_reason,
            event.cancel_reason,
        )
        for event in replay.machine.events
        if event.bar_index < last
    ]


def test_event_sequence_is_identical(sample: dict, replay: _RecordingFeed):
    """§13 rungs 3 and 4: same events, same order, same level and direction."""
    expected = _reference_rows(sample)
    got = _port_rows(sample, replay)
    assert got == expected


def test_event_counts_by_type_are_reported(sample: dict, replay: _RecordingFeed, capsys):
    """Print the per-type census the acceptance criterion asks for."""
    rows = _port_rows(sample, replay)
    counts = {
        name: sum(1 for row in rows if row[1] == code)
        for code, name in enumerate(EVENT_NAMES)
    }
    cancels = {
        name: sum(1 for row in rows if row[6] == code)
        for code, name in enumerate(CANCEL_REASON_NAMES)
        if code and any(row[6] == code for row in rows)
    }
    exits = {
        name: sum(1 for row in rows if row[5] == code)
        for code, name in enumerate(EXIT_REASON_NAMES)
        if code and any(row[5] == code for row in rows)
    }
    with capsys.disabled():
        print(
            f"\n[qc x10 parity] grid={sample['grid']}  total={len(rows)}"
            f"\n    events  {counts}"
            f"\n    cancels {cancels}"
            f"\n    exits   {exits}"
        )
    assert sum(counts.values()) == len(rows)


def test_stop_target_and_r_est_match(sample: dict, replay: _RecordingFeed):
    """§13 rung 4: the three numbers an entry decision is made of, at 1e-9."""
    last = len(sample["inputs"].m5) - 1
    expected = np.array(
        [
            [row[E_STOP], row[E_TARGET], row[E_R_EST]]
            for row in sample["events"]
            if int(row[E_I_DECISION]) < last
        ]
    )
    got = np.array(
        [
            [event.stop, event.target, event.r_est]
            for event in replay.machine.events
            if event.bar_index < last
        ]
    )
    np.testing.assert_allclose(got, expected, rtol=0.0, atol=_TOL, equal_nan=True)


def test_fill_and_exit_prices_match(sample: dict, replay: _RecordingFeed):
    """§13 rung 5 on the *theoretical* prices — LEAN's own fills are not these."""
    last = len(sample["inputs"].m5) - 1
    expected = np.array(
        [
            [row[E_FILL_PX], row[E_EXIT_PX]]
            for row in sample["events"]
            if int(row[E_I_DECISION]) < last
        ]
    )
    got = np.array(
        [
            [event.fill_px, event.exit_px]
            for event in replay.machine.events
            if event.bar_index < last
        ]
    )
    np.testing.assert_allclose(got, expected, rtol=0.0, atol=_TOL, equal_nan=True)


def test_lots_match_trade_for_trade(sample: dict, replay: _RecordingFeed):
    """§11: same risk fraction, same 1:100 cap, same downward 0.01 rounding."""
    last = len(sample["inputs"].m5) - 1
    # One trade row per ENTRY event, in order: the kernel appends both together.
    entries = [row for row in sample["events"] if int(row[E_EVENT]) == EVENT_ENTRY]
    expected = np.array(
        [
            sample["trades"][k, T_LOTS]
            for k, row in enumerate(entries)
            if int(row[E_I_DECISION]) < last
        ]
    )
    got = np.array(
        [
            event.lots
            for event in replay.machine.events
            if event.event == EVENT_ENTRY and event.bar_index < last
        ]
    )
    assert len(got) == len(expected)
    np.testing.assert_allclose(got, expected, rtol=0.0, atol=_TOL)


def test_the_port_closes_positions_and_does_not_leak_them(replay: _RecordingFeed):
    """One position at a time (§9): exits never outnumber entries."""
    entries = [e for e in replay.machine.events if e.event == EVENT_ENTRY]
    exits = [e for e in replay.machine.events if e.event == QC_EVENT_EXIT]
    assert entries, "the sample must open at least one position"
    assert len(entries) - len(exits) in (0, 1)
