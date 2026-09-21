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

from datetime import date, datetime, timedelta

from x10_bars import MinuteQuote, minute_index, minute_to_utc, ny_fields
from x10_indicators import DXY4_LEGS
from x10_state import (
    CONTRACT_SIZE,
    EVENT_ENTRY,
    EVENT_EXIT,
    EXIT_REASON_NAMES,
    SCENARIO_NAMES,
    BrokerSync,
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

# Last minute the XAUUSD CFD quotes before the daily break, New York clock,
# measured on the very export the reference engine reads
# (``data/XAU-USD_minute_qc.parquet``: the last stamp of each of the 261 2024
# sessions is 16:58, 16:59 and 17:00 never occur, and the feed resumes at
# 18:04). The M5 bin 16:55 therefore never receives a 16:59 minute and cannot
# close on its own — see ``X10Feed.on_minute(last_of_session=...)``.
SESSION_LAST_MINUTE = 16 * 60 + 58

# First minute the CFD quotes again after the break, New York clock. It bounds
# the sealing window above: ``minute_of_day >= SESSION_LAST_MINUTE`` alone is
# true from 16:58 to 23:59, so every evening minute flushed its own M5 bin and
# the whole evening session was cut into one-minute bars (backtest v3).
SESSION_REOPEN_MINUTE = 18 * 60

# LEAN keeps an order tag as free text; the API and the UI truncate long ones,
# so the enriched entry tag is kept comfortably under this.
TAG_MAX = 200

# ``diag=1`` plots the decision bar and its ATR for this many days. Chart
# series are the only per-bar quantity retrievable through the API, and QC
# thins a series past a few thousand points, so the window is kept short.
DIAG_DAYS = 3
DIAG_MAX_POINTS = 900


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

        # ``fill_forward=False`` is load-bearing, not a preference. The export
        # the reference reads has a bar for every quoted minute and none at all
        # between 16:58 and 18:04 New York; LEAN's market-hours database has the
        # CFD open from 18:00, so with fill-forward on it would invent four
        # phantom minutes at every session open, land them in the 18:00 M5 bin
        # and move that bar's open, high, low and therefore the ATR. That is a
        # first-order suspect for the M5 disagreement of §2.1 / D8.
        gold = self.add_cfd(  # noqa: F405
            "XAUUSD", Resolution.MINUTE, Market.OANDA, fill_forward=False  # noqa: F405
        )
        gold.set_fee_model(ConstantFeeModel(0))  # noqa: F405
        self._xau = gold.symbol

        self._legs: dict = {}
        for leg, ticker in LEG_TICKERS.items():
            security = self.add_forex(  # noqa: F405
                ticker, Resolution.MINUTE, Market.OANDA, fill_forward=False  # noqa: F405
            )
            security.set_fee_model(ConstantFeeModel(0))  # noqa: F405
            self._legs[security.symbol] = leg

        # Annexe A.2 — the three axes of the grid, and nothing else. The spread
        # override is not a grid axis: it switches the run between "real
        # execution" and "signals comparable to the reference" (§8, D7).
        override = self._string("spread_override", "")
        self._feed = X10Feed(
            z=self._float("z", 1.0),
            a_min=self._float("a_min", 0.2),
            k_s=self._float("k_s", 1.0),
            init_cash=INIT_CASH,
            spread_override=float(override) if override else None,
        )

        self._trace_key = f"xau_x10/trace_{start:%Y-%m-%d}_{end:%Y-%m-%d}.csv"
        # Intent against reality: ``_broker`` holds what the strategy wants,
        # the portfolio holds what the account has, and ``_sync`` is the single
        # place an order is ever sent.
        self._broker = BrokerSync()
        self._last_minute = -1
        self._clock_checked = False
        self._audit_day = -1
        # §10 / D5 — session label of the last day of the window. The position
        # is squared on the last quoted minute of that session, market open,
        # because an order issued from ``on_end_of_algorithm`` is too late:
        # LEAN turns it into ``MarketOnOpen`` and OANDA refuses the type.
        self._final_session = (end.date() - date(1970, 1, 1)).days
        self._blocked_session = -1
        self._diag = self._string("diag", "0") == "1"
        self._diag_until = minute_index(start) + DIAG_DAYS * 24 * 60
        self._diag_points = 0
        self._m5_seen = 0
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

        The bar label is ``utc_time`` minus one minute, and **not**
        ``quote_bar.time``. Both are the start of the bar just closed, but they
        are not on the same clock: ``utc_time`` is UTC by construction, whereas
        a LEAN bar carries a *naive* stamp on its **exchange** clock — New York
        for an OANDA CFD. Feeding the second to ``minute_index``, which reads a
        naive stamp as UTC, produced an index five hours behind the truth;
        ``ny_fields`` then subtracted the offset a second time and the whole
        session clock of §2/§10 ran five hours late. The 16:55 flat fired at
        21:55 New York, on a market that had closed at 17:00 (backtest v3, and
        the ``xr=SESSION`` tag of its order 16 proves it).
        """
        legs: dict = {}
        for symbol, leg in self._legs.items():
            if symbol in slice.quote_bars:
                legs[leg] = float(slice.quote_bars[symbol].close)

        gold = None
        exchange_stamp = None
        if self._xau in slice.quote_bars:
            bar = slice.quote_bars[self._xau]
            exchange_stamp = bar.time
            gold = self._quote(bar)
        if gold is None and not legs:
            return

        minute = minute_index(self.utc_time) - 1
        if minute <= self._last_minute:
            return
        self._last_minute = minute

        if not self._clock_checked and exchange_stamp is not None:
            # Kept as a probe, never as the clock: it measures the very offset
            # described above, so it must be reported and not consumed.
            self._clock_checked = True
            offset = minute - minute_index(exchange_stamp)
            message = (
                f"x10 clock probe: utc_time-1min={minute_to_utc(minute)} "
                f"quote_bar.time={exchange_stamp} offset={offset}min"
            )
            self.log(message) if offset == 0 else self.error(message)

        minute_of_day, session_id = ny_fields(minute)

        # §11 sizes on the equity; the broker's own valuation is the one that
        # matters here, not the running total the automaton keeps for the
        # reference engine. It is only trustworthy because ``_sync`` keeps the
        # account on the strategy's inventory — D6 was a consequence of D1.
        self._feed.machine.equity = float(self.portfolio.total_portfolio_value)
        held = float(self.portfolio[self._xau].quantity)
        blocked = not self._broker.matches(held)
        self._feed.machine.broker_blocked = blocked
        self._watchdog(blocked, session_id, held)
        self._audit(minute, held)

        events = self._feed.on_minute(
            minute,
            gold,
            legs,
            live=not self.is_warming_up,
            last_of_session=(
                gold is not None
                and SESSION_LAST_MINUTE <= minute_of_day < SESSION_REOPEN_MINUTE
            ),
        )
        for event in events:
            self._route(event)

        if self._feed.m5_closed != self._m5_seen:
            self._m5_seen = self._feed.m5_closed
            self._plot_diagnostics(minute)

        # Last quoted minute of the last session: square up while the market is
        # still open. This overrides any entry the automaton just asked for.
        if session_id >= self._final_session and minute_of_day >= SESSION_LAST_MINUTE:
            self._broker.want(0.0, "SESSION|end_of_window")

        self._sync(minute)

    def _watchdog(self, blocked: bool, session_id: int, held: float) -> None:
        """Refuse to stay frozen silently for more than one session.

        ``broker_blocked`` is meant to last a few minutes — the time for an
        order to fill. In v3 it lasted eleven and a half months because the
        intent and the fill differed by one ulp. A freeze that survives a
        session boundary is a defect, and it says so with the state that
        explains it.
        """
        if not blocked:
            self._blocked_session = -1
            return
        if self._blocked_session < 0:
            self._blocked_session = session_id
            return
        if session_id > self._blocked_session:
            self.error(
                f"x10 AUTOMATON FROZEN across sessions {self._blocked_session}->{session_id} "
                f"held={held!r} target={self._broker.target_qty!r} "
                f"inflight={self._broker.inflight} deferred={self._broker.deferred} "
                f"rejected={self._broker.rejected} abandoned={self._broker.abandoned}"
            )
            self._blocked_session = session_id

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
        """Record the inventory the strategy wants; ``_sync`` is what orders it.

        Separating intent from submission is the whole fix for D1-D4. The old
        code assumed a sent order was a filled order and wrote its position
        state before the broker had answered; eight refused exits later, the
        2024 account was carrying 40 ounces nobody had asked for.
        """
        if event.event == EVENT_ENTRY:
            quantity = event.q * event.lots * CONTRACT_SIZE
            if quantity == 0.0:
                return
            self._broker.want(quantity, self._entry_tag(event))
        elif event.event == EVENT_EXIT:
            self._broker.want(0.0, self._exit_tag(event))

    def _sync(self, minute: int) -> None:
        """Drive the account onto the wanted inventory, never onto a shut market.

        One market order at a time, and none while the XAUUSD exchange is
        closed: ``market_order`` silently becomes ``MarketOnOpen`` there, and
        ``OandaBrokerageModel`` rejects that type — eight times out of eight in
        2024, always at 17:04 New York (D2). A deferred order is not lost, it
        is re-attempted on the next quoted minute. The decision itself lives in
        ``BrokerSync`` so that the parity test can exercise it without LEAN.
        """
        quantity = self._broker.order(
            float(self.portfolio[self._xau].quantity),
            bool(self.securities[self._xau].exchange.exchange_open),
            minute,
        )
        if quantity is not None:
            self.market_order(self._xau, quantity, tag=self._broker.tag or "x10|sync")

    def on_order_event(self, order_event) -> None:
        """Learn what actually happened, and resynchronise on the portfolio.

        The automaton is allowed to be wrong about the account for exactly one
        event. A refused **entry** leaves it holding a position that does not
        exist, so it is forced flat with a ``broker`` cancel in the trace; a
        refused **exit** leaves real inventory, so the position stays open for
        the automaton (``broker_blocked``) until the account is flat again.
        """
        status = order_event.status
        if status == OrderStatus.FILLED:  # noqa: F405
            self._broker.on_settled()
            return
        if status not in (OrderStatus.INVALID, OrderStatus.CANCELED):  # noqa: F405
            return

        held = float(self.portfolio[self._xau].quantity)
        phantom = self._broker.on_refused(held)
        self.error(
            f"x10 ORDER REFUSED id={order_event.order_id} status={status} "
            f"held={held} phantom_entry={phantom} msg={order_event.message}"
        )
        if phantom:
            # D3 / §4.2: the entry never existed. Without this the next exit
            # order opens an inverse position out of nothing.
            self._feed.force_flat()

    def _audit(self, minute: int, held: float) -> None:
        """Once a day, shout if the account is not where the automaton thinks."""
        day = minute // (24 * 60)
        if day == self._audit_day:
            return
        self._audit_day = day
        believed = self._feed.machine.position_quantity
        target = self._broker.target_qty
        if not self._broker.matches(held) or (
            abs(target) > 0.0 and abs(held - believed) > 1e-6
        ):
            self.error(
                f"x10 INVARIANT BROKEN held={held} target={target} "
                f"automaton={believed} deferred={self._broker.deferred} "
                f"rejected={self._broker.rejected} abandoned={self._broker.abandoned}"
            )

    def _plot_diagnostics(self, minute: int) -> None:
        """``diag=1``: publish the decision bar and its ATR as chart series.

        Chart series **are** retrievable through the API (``read_backtest_chart``)
        where the ObjectStore and the log are not, so this is the only way to
        compare the M5 bars themselves against the reference without a trace.
        No order is ever placed for diagnostics.
        """
        if not self._diag or minute > self._diag_until:
            return
        if self._diag_points >= DIAG_MAX_POINTS:
            return
        bar = self._feed.last_bar
        context = self._feed.last_context
        if bar is None or context is None:
            return
        if context.atr != context.atr:  # NaN until 14 TRs exist; LEAN cannot plot it
            return
        self._diag_points += 1
        self.plot("X10Diag", "m5_close", bar.close)
        self.plot("X10Diag", "atr", context.atr)

    # ── order tags — the only channel the API gives back ──────────────
    @staticmethod
    def _legacy_tag(event) -> str:
        """``scenario|level|ts_decision|stop|target|r_est``, frozen prefix.

        ``scripts/reconcile_x10_events.py`` reads these six fields positionally
        and must keep working: anything new goes strictly after them.
        """
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

    def _entry_tag(self, event) -> str:
        """The frozen six fields, then the quantities D8 needs to be measurable.

        Rung 2 of §13 cannot be read from the outside: the trace lives in the
        ObjectStore and the log is not exportable. The ATR, the decision close
        and the spread that went into ``R`` therefore travel on the order, which
        is exportable, in ``k=v`` fields after the frozen prefix. They are
        ordered by diagnostic value so that a truncation loses the least.
        """
        bar = self._feed.last_bar
        extra = [
            f"bm={event.bar_minute}",
            "c=" if bar is None else f"c={bar.close:.4f}",
            f"atr={event.atr:.6f}",
            f"sp={self._feed.last_half_spread * 2.0:.6f}",
            f"v={event.v:.6f}",
            f"a={event.a:.6f}",
            f"vw={event.vwap:.6f}",
            f"ema={event.ema50_h1:.6f}",
        ]
        tag = self._legacy_tag(event)
        for field in extra:
            if len(tag) + 1 + len(field) > TAG_MAX:
                break
            tag = f"{tag}|{field}"
        return tag

    def _exit_tag(self, event) -> str:
        """The frozen six fields plus the exit reason, which §8.2 had to infer."""
        tag = self._legacy_tag(event)
        field = f"xr={EXIT_REASON_NAMES[event.exit_reason]}|xp={event.exit_px:.3f}"
        return tag if len(tag) + 1 + len(field) > TAG_MAX else f"{tag}|{field}"

    # ── §13 trace ─────────────────────────────────────────────────────
    def on_end_of_algorithm(self) -> None:
        """Emit the trace twice: ObjectStore for the file, log for the export.

        The ObjectStore cannot be downloaded through the API on this account
        (``docs/specs/gold_momentum_spec.md`` §8), so the log carries a second
        copy — but only of ``ENTRY`` and ``EXIT``, which is what rungs 4 and 5
        of §13 compare. One line per ``ARM`` would blow the log quota on seven
        years of minutes and buy nothing.
        """
        # D5 / §10: no position survives the end of the file. The real close
        # happened on the last quoted minute of the last session, inside
        # ``on_data``, because an order issued from here is stamped after the
        # exchange has shut and LEAN turns it into a ``MarketOnOpen`` that OANDA
        # refuses (v3, order 20). This ``liquidate`` is the belt to that braces:
        # if it ever has anything to do, the close above failed and that is the
        # defect to report, not the leftover.
        held = float(self.portfolio[self._xau].quantity)
        if held != 0.0:
            self.error(
                f"x10 NOT FLAT at end of algorithm: {held} oz still held — "
                "the end-of-window close did not fill; liquidating late"
            )
            self.liquidate(self._xau, tag="SESSION|end_of_algorithm")
        else:
            self.log("x10 flat at end of algorithm")
        self.log(
            f"x10 order health: {self._broker.rejected} refused, "
            f"{self._broker.deferred} deferred on a closed exchange, "
            f"{self._broker.abandoned} abandoned without an event, "
            f"{self._feed.dropped_minutes} minutes dropped"
        )

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
