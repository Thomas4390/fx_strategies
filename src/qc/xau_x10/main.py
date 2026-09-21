"""XAUUSD x10 levels — QuantConnect algorithm (``docs/specs/xau_x10_spec.md``).

Third engine of the strategy, after the vectorbt reference
(``src/strategies/xau_x10.py``) and the MQL5 expert. It shares the **data** of
the reference — the gold minute parquet was exported from this very account —
so §13 asks the two to agree on at least 98 % of the entries; any gap is a
defect of one of the two ports, not a market fact.

This file is glue and nothing else. Every decision is taken in three modules
that import nothing from QuantConnect and are replayed against the reference
kernel by ``tests/test_qc_x10_parity.py``:

* ``x10_bars``       — left-labelled M5/H1 accumulation and the New York clock;
* ``x10_indicators`` — ATR, EMA, kinematics, session VWAP, DXY4 basket;
* ``x10_state``      — the automaton of §7, the sizing of §11, the §13 trace.

What this file owns, and the traps it is written around:

* **XAUUSD on OANDA is a CFD, not a forex pair.** It delivers ``QuoteBar``s, so
  the algorithm reads ``slice.quote_bars`` and never ``slice.bars``: the latter
  is silently empty and the first backtest places zero orders with no error.
* **No attribute named ``symbol``** — it shadows a ``QCAlgorithm`` method, and
  the compiler only warns.
* **Execution is handled by hand** in ``on_data``. No ``StopMarketOrder``, no
  ``LimitOrder``: bracket orders submitted before the entry has filled are the
  documented way to get a backtest that exits on the timer instead of the stop
  (``docs/quantconnect_validation_report.md`` §5.2). The stop and the target of
  §9 are tested minute by minute on the trigger side — bid for a long, ask for
  a short — with the stop winning a double contact.
* **No slippage model and a zero fee model.** The execution cost is the real
  bid-ask spread of the quote bars, once; a ``ConstantSlippageModel`` would
  charge it a second time on every exit (§5.3 of the same report).
* **The trace is emitted twice** (§13): into the ObjectStore *and* into the log
  with a ``TRACE,`` prefix, because the ObjectStore is not exportable through
  the API on this account. Only ``ENTRY`` and ``EXIT`` go to the log — the log
  quota does not survive one line per ``ARM``.

The orders carry ``scenario|level|ts_decision|stop|target|r_est`` as their tag:
orders *are* retrievable through the API, which makes them the primary
reconciliation channel and the trace the secondary one.
"""

# region imports
from AlgorithmImports import *  # noqa: F403
# endregion

from datetime import datetime, timedelta

from x10_bars import MinuteQuote, minute_index, minute_to_utc
from x10_indicators import DXY4_LEGS
from x10_state import (
    CONTRACT_SIZE,
    EVENT_ENTRY,
    EVENT_EXIT,
    SCENARIO_NAMES,
    X10Feed,
    trace_header,
    trace_row,
)

# §1 / §14 — the four legs of the synthetic dollar of §6.3, OANDA majors.
LEG_TICKERS: dict[str, str] = {
    "eurusd": "EURUSD",
    "usdjpy": "USDJPY",
    "gbpusd": "GBPUSD",
    "usdcad": "USDCAD",
}

# §1 — selection window; anything at or after this date is the frozen slice.
FROZEN_FROM = "2026-01-01"
DEFAULT_START = "2019-01-01"
DEFAULT_END = "2025-12-31"

# §6.1 — the EMA50 H1 needs 50 closed H1 bars and the ATR_H1 fourteen more.
# Twelve calendar days cover 60 trading hours with two weekends of slack.
WARMUP = timedelta(days=12)

INIT_CASH = 10_000.0


class ApogeeInvestS2XauX10(QCAlgorithm):  # noqa: F405
    """Strategy 2 — XAUUSD psychological x10 levels, M5 decisions, M1 fills."""

    # ── setup ─────────────────────────────────────────────────────────
    def initialize(self) -> None:
        start, end = self._dates()
        self.set_time_zone(TimeZones.UTC)  # noqa: F405
        self.set_start_date(start.year, start.month, start.day)
        self.set_end_date(end.year, end.month, end.day)
        self.set_cash(INIT_CASH)
        self.set_brokerage_model(
            BrokerageName.OANDA_BROKERAGE,  # noqa: F405
            AccountType.MARGIN,  # noqa: F405
        )

        gold = self.add_cfd("XAUUSD", Resolution.MINUTE, Market.OANDA)  # noqa: F405
        gold.set_fee_model(ConstantFeeModel(0))  # noqa: F405
        self._xau = gold.symbol

        self._legs: dict = {}
        for leg, ticker in LEG_TICKERS.items():
            security = self.add_forex(ticker, Resolution.MINUTE, Market.OANDA)  # noqa: F405
            security.set_fee_model(ConstantFeeModel(0))  # noqa: F405
            self._legs[security.symbol] = leg

        # Annexe A.2 — the three axes of the grid, and nothing else.
        self._feed = X10Feed(
            z=self._float("z", 1.0),
            a_min=self._float("a_min", 0.2),
            k_s=self._float("k_s", 1.0),
            init_cash=INIT_CASH,
        )

        self._trace_key = f"xau_x10/trace_{start:%Y-%m-%d}_{end:%Y-%m-%d}.csv"
        self._open_qty = 0.0
        self._last_minute = -1
        self.set_warm_up(WARMUP)

    def _dates(self) -> tuple[datetime, datetime]:
        """Parse the window and refuse the frozen slice unless it is unlocked.

        ``docs/research/HOLDOUT_POLICY.md``: 2026 onwards is read once, after
        the in-sample campaign has closed. An accidental backtest over it is not
        a mistake that can be undone by ignoring the number.
        """
        start = self._string("start", DEFAULT_START)
        end = self._string("end", DEFAULT_END)
        if end >= FROZEN_FROM and self._string("allow_frozen_oos", "0") != "1":
            raise ValueError(
                f"end={end} reaches the frozen out-of-sample slice ({FROZEN_FROM}+); "
                "set allow_frozen_oos=1 only inside a declared holdout read"
            )
        return (
            datetime.strptime(start, "%Y-%m-%d"),
            datetime.strptime(end, "%Y-%m-%d"),
        )

    def _string(self, name: str, default: str) -> str:
        value = self.get_parameter(name)
        return default if value is None or value == "" else str(value)

    def _float(self, name: str, default: float) -> float:
        return float(self._string(name, str(default)))

    # ── the minute loop ───────────────────────────────────────────────
    def on_data(self, slice: Slice) -> None:  # noqa: F405, A002
        """Hand one minute to the feed and route the events it returns.

        ``self.time`` is the **end** of the minute just closed; the reference
        indexes a bar by its start, hence the ``- 1``. Nothing else in this
        method decides anything: the feed owns the order of the five steps
        inside a minute, and it is the same object the parity test drives.
        """
        minute = minute_index(self.time) - 1
        if minute <= self._last_minute:
            return
        self._last_minute = minute

        legs: dict = {}
        for symbol, leg in self._legs.items():
            if symbol in slice.quote_bars:
                legs[leg] = float(slice.quote_bars[symbol].close)

        gold = None
        if self._xau in slice.quote_bars:
            gold = self._quote(slice.quote_bars[self._xau])
        if gold is None and not legs:
            return

        # §11 sizes on the equity; the broker's own valuation is the one that
        # matters here, not the running total the automaton keeps for the
        # reference engine.
        self._feed.machine.equity = float(self.portfolio.total_portfolio_value)

        events = self._feed.on_minute(minute, gold, legs, live=not self.is_warming_up)
        for event in events:
            self._route(event)

    @staticmethod
    def _quote(bar) -> MinuteQuote | None:
        """Flatten a ``QuoteBar``; a one-sided quote is not tradable here."""
        bid, ask = bar.bid, bar.ask
        if bid is None or ask is None:
            return None
        return MinuteQuote(
            float(bar.open), float(bar.high), float(bar.low), float(bar.close),
            float(bid.open), float(bid.high), float(bid.low), float(bid.close),
            float(ask.open), float(ask.high), float(ask.low), float(ask.close),
        )

    # ── orders ────────────────────────────────────────────────────────
    def _route(self, event) -> None:
        """Turn an ``ENTRY``/``EXIT`` row into a market order, and nothing else.

        The quantity is in ounces: §11 sizes in lots of 100. The price LEAN
        fills at is **not** the theoretical price carried by the event — the
        order is sent while the deciding minute has just closed, so LEAN fills
        it at the next minute's bid/ask. Both numbers are kept: the theoretical
        one in the trace, the realised one in the order log.
        """
        if event.event == EVENT_ENTRY:
            quantity = event.q * event.lots * CONTRACT_SIZE
            if quantity == 0.0:
                return
            self._open_qty = quantity
            self.market_order(self._xau, quantity, tag=self._tag(event))
        elif event.event == EVENT_EXIT and self._open_qty != 0.0:
            self.market_order(self._xau, -self._open_qty, tag=self._tag(event))
            self._open_qty = 0.0

    @staticmethod
    def _tag(event) -> str:
        """``scenario|level|ts_decision|stop|target|r_est`` — the API channel."""
        stamp = minute_to_utc(event.bar_minute).strftime("%Y-%m-%d %H:%M:%S")
        return "|".join(
            (
                SCENARIO_NAMES[event.scenario],
                f"{event.level:.3f}",
                stamp,
                f"{event.stop:.6f}",
                f"{event.target:.6f}",
                f"{event.r_est:.6f}",
            )
        )

    # ── §13 trace ─────────────────────────────────────────────────────
    def on_end_of_algorithm(self) -> None:
        """Emit the trace twice: ObjectStore for the file, log for the export.

        The ObjectStore cannot be downloaded through the API on this account
        (``docs/specs/gold_momentum_spec.md`` §8), so the log carries a second
        copy — but only of ``ENTRY`` and ``EXIT``, which is what rungs 4 and 5
        of §13 compare. One line per ``ARM`` would blow the log quota on seven
        years of minutes and buy nothing.
        """
        events = self._feed.machine.events
        lines = [trace_header()]
        for event in events:
            row = trace_row(event)
            lines.append(row)
            if event.event in (EVENT_ENTRY, EVENT_EXIT):
                self.log("TRACE," + row)
        self.object_store.save(self._trace_key, "\n".join(lines) + "\n")
        self.log(
            f"x10 trace: {len(events)} events, {len(DXY4_LEGS)} dollar legs, "
            f"object store key {self._trace_key}"
        )
