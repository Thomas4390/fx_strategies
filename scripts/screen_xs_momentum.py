#!/usr/bin/env python
"""Pre-filter the cross-sectional momentum family on the screened universe.

Same pre-filter status as ``screen_tsmom_universe.py`` — the vbt numbers kill
the hopeless, the MT5 tester ranks what survives — but the object is different:
one **portfolio**, not one sleeve per instrument. Six configurations, six return
series, six rows.

Universe: the 14 instruments the TSMOM screen marked ``PASS``
(``reports/research/tsmom_screen_2026H2.csv``), each read from the source that
file used. Their histories are ragged — 36 years for the yahoo indices, 2 to 8
for the broker exports — so the book ranks only what quotes on the decision
date and does not exist at all before six instruments quote together (August
2000 on this universe). Closes are joined on the session date, outer join,
**never forward filled**: a missing close is a missing instrument.

Conventions, inherited from the TSMOM screen so the two families can be
compared at all:

- ``fill="next_open"``: decide on close[t], fill on the next session's open.
- per-symbol half spread of ``costs.yml`` as slippage, charged on every fill,
  which is where the turnover of a monthly rebalance is paid.
- 0.5 bp/night swap drag on **gross** exposure (sum of |weights|, leverage
  included) — a long/short book pays it on both legs.
- selection window ends 2025-12-31 (``framework.holdout``); full history in,
  window out.
- Sharpe annualized on 252 sessions.

The trial budget of this family (n=6, closed grid) is declared in the phase
note, not re-logged here.

    python scripts/screen_xs_momentum.py --selfcheck
    python scripts/screen_xs_momentum.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from framework import holdout  # noqa: E402
from framework.costs import cost_for  # noqa: E402
from strategies import tsmom, xs_momentum  # noqa: E402
from strategies.gold_momentum import (  # noqa: E402
    SESSION_CLOSE_HOUR,
    _daily_close,
    _daily_open,
    session_dates,
)

TSMOM_SCREEN_PATH = _ROOT / "reports" / "research" / "tsmom_screen_2026H2.csv"
OUTPUT_PATH = _ROOT / "reports" / "research" / "xs_screen_2026H2.csv"

SELECTION_END = pd.Timestamp("2025-12-31")  # same cut as HOLDOUT_START, spelled out
SWAP_BPS_PER_NIGHT = 0.00005
ANN_DAYS = 252

# Family gates, stated before the run: a book that neither clears a floor of
# raw edge nor brings anything the equal-weight TSMOM basket does not already
# have is not a family, it is a re-weighting.
MIN_SHARPE = 0.15
MAX_CORR_TO_TSMOM = 0.8

CSV_COLUMNS: tuple[str, ...] = (
    "config", "lookback", "n_long", "n_short", "n_bars", "first_date", "last_date",
    "sharpe_net", "cagr", "maxdd", "trades_total", "turnover_annuel",
    "n_instruments_moyen", "gross_median", "corr_to_tsmom",
)


def by_session_date(series: pd.Series) -> pd.Series:
    """Series re-indexed on its session date under the 17:00 New York cut.

    Not ``normalize()``: a broker daily bar is stamped 20:00 New York and would
    be pushed back one session by a calendar-day truncation, which is exactly
    the misalignment ``screen_tsmom_universe._by_date`` was written to avoid.
    """
    out = series.copy()
    out.index = session_dates(pd.DatetimeIndex(out.index), SESSION_CLOSE_HOUR)
    return out[~out.index.duplicated()]


def read_universe() -> list[tuple[str, str]]:
    """``(symbol, loader)`` for every instrument the TSMOM screen passed."""
    screen = pd.read_csv(TSMOM_SCREEN_PATH)
    passed = screen.loc[screen["verdict"] == "PASS", ["symbol", "source"]]
    if passed.empty:
        raise RuntimeError(f"{TSMOM_SCREEN_PATH.name} carries no PASS instrument")
    return list(passed.itertuples(index=False, name=None))


def load_panel(universe: list[tuple[str, str]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Closes and opens of the universe, joined on the session date.

    Outer join, no forward fill. An instrument that fails to load is dropped
    with a log rather than silently: a universe that loses a column without
    saying so reads as if that instrument had been ranked and lost.
    """
    closes: dict[str, pd.Series] = {}
    opens: dict[str, pd.Series] = {}
    for symbol, loader in universe:
        try:
            _, data = tsmom.load_instrument(symbol, loader_override=loader)
        except (ValueError, KeyError, FileNotFoundError) as exc:
            print(f"  !! {symbol} [{loader}]: excluded, {exc}")
            continue
        closes[symbol] = by_session_date(_daily_close(data, SESSION_CLOSE_HOUR))
        opens[symbol] = by_session_date(_daily_open(data, SESSION_CLOSE_HOUR))
        print(f"  {symbol:<9} [{loader:<9}] {len(closes[symbol]):>5} sessions, "
              f"{closes[symbol].index.min().date()} -> {closes[symbol].index.max().date()}")
    if not closes:
        raise RuntimeError("no instrument loaded — nothing to rank")
    close_df = pd.DataFrame(closes).sort_index()
    return close_df, pd.DataFrame(opens).reindex(close_df.index)


def slippage_series(columns: pd.Index) -> pd.Series:
    """Per-symbol half spread, one value per column of the panel."""
    return pd.Series({symbol: cost_for(symbol) for symbol in columns}, dtype=float)


def run_config(
    closes: pd.DataFrame,
    opens: pd.DataFrame,
    slippage: pd.Series,
    config: dict[str, int],
) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    """Simulate one configuration; return (net returns, trades, weights), windowed.

    The swap drag is charged on the full series, with the gross exposure of the
    session it belongs to, and only then is the selection window cut out.
    """
    params = dict(lookback=config["lookback"], n_long=config["n_long"],
                  n_short=config["n_short"])
    weights = xs_momentum.xs_weights(closes, **params)
    pf = xs_momentum.pipeline(closes, opens=opens, slippage=slippage, **params)

    ret = pf.returns
    gross = weights.reindex(ret.index).ffill().fillna(0.0).abs().sum(axis=1)
    ret_net = ret - SWAP_BPS_PER_NIGHT * gross

    trades = pf.trades.records_readable
    column = "Exit Index" if "Exit Index" in trades.columns else "Entry Index"
    trades = holdout.trim_insample(trades.set_index(column))

    ret_net = holdout.trim_insample(ret_net)
    holdout.assert_not_optimizing(ret_net.index)
    return ret_net, trades, holdout.trim_insample(weights)


def tsmom_basket(universe: list[tuple[str, str]]) -> pd.Series:
    """Equal-weight basket of the mono-instrument TSMOM sleeves, net returns.

    The additivity yardstick: the XS book only earns its place if its returns
    are not this series under another name. Same costs, same swap, same window
    as the TSMOM screen that produced the universe.
    """
    sleeves = []
    for symbol, loader in universe:
        try:
            pf, _ = tsmom.pipeline(
                symbol, loader_override=loader, fill="next_open",
                slippage=cost_for(symbol),
            )
        except (ValueError, KeyError, FileNotFoundError) as exc:
            print(f"  !! {symbol}: excluded from the TSMOM basket, {exc}")
            continue
        ret = pf.returns
        exposure = (pf.asset_value / pf.value).reindex(ret.index).fillna(0.0).abs()
        sleeves.append(by_session_date(ret - SWAP_BPS_PER_NIGHT * exposure).rename(symbol))
    # skipna: an instrument whose history has not started yet must not drag the
    # basket toward zero.
    basket = pd.concat(sleeves, axis=1, sort=True).mean(axis=1, skipna=True)
    return holdout.trim_insample(basket.dropna())


def metrics(
    config: dict[str, int],
    ret_net: pd.Series,
    trades: pd.DataFrame,
    weights: pd.DataFrame,
    basket: pd.Series,
) -> dict[str, object]:
    """One CSV row, everything computed on the post-cost, post-swap series."""
    acc = ret_net.vbt.returns(freq="1D", year_freq=pd.Timedelta(days=ANN_DAYS))
    years = len(ret_net) / ANN_DAYS
    held = weights.reindex(ret_net.index).ffill().fillna(0.0)
    # Turnover of the *targets*: weights are constant between two rebalances, so
    # the diff is non-zero only on decision rows. Intra-month drift is not a
    # traded turnover and is deliberately not counted.
    turnover = float(held.diff().abs().sum(axis=1).sum() / years) if years else float("nan")
    n_held = weights.ne(0.0).sum(axis=1)

    joined = pd.concat([ret_net, basket], axis=1, join="inner").dropna()
    corr = float(joined.iloc[:, 0].corr(joined.iloc[:, 1])) if len(joined) > 2 else float("nan")

    return dict(
        config=f"lb{config['lookback']}_L{config['n_long']}S{config['n_short']}",
        lookback=config["lookback"], n_long=config["n_long"], n_short=config["n_short"],
        n_bars=len(ret_net),
        first_date=str(ret_net.index.min().date()),
        last_date=str(ret_net.index.max().date()),
        sharpe_net=float(acc.sharpe_ratio()),
        cagr=float(acc.annualized()),
        maxdd=float(acc.max_drawdown()),
        trades_total=int(len(trades)),
        turnover_annuel=turnover,
        n_instruments_moyen=float(n_held[n_held > 0].mean()),
        gross_median=float(held.abs().sum(axis=1).median()),
        corr_to_tsmom=corr,
    )


def format_table(df: pd.DataFrame) -> str:
    """Screen table, ranked by net Sharpe."""
    shown = df.copy()
    for column, fmt in (
        ("sharpe_net", "{:.3f}"), ("cagr", "{:.1%}"), ("maxdd", "{:.1%}"),
        ("turnover_annuel", "{:.2f}"), ("n_instruments_moyen", "{:.1f}"),
        ("gross_median", "{:.2f}"), ("corr_to_tsmom", "{:.2f}"),
    ):
        shown[column] = shown[column].map(lambda v, f=fmt: "" if pd.isna(v) else f.format(v))
    return shown.to_string(index=False)


def selfcheck() -> int:
    """Two properties, checked on synthetic data. Returns a shell exit code."""
    index = pd.date_range("2020-01-01", periods=400, freq="B")

    # (a) The ranking at one date, against a hand computation on 3 instruments.
    #     Constant daily growth rates, so score = (1+g)**(lookback-skip) - 1.
    rates = dict(A=0.0010, B=0.0002, C=0.0006)
    closes = pd.DataFrame(
        {name: 100.0 * (1.0 + g) ** np.arange(len(index)) for name, g in rates.items()},
        index=index,
    )
    lookback, skip = 126, 21
    scores = xs_momentum.xs_scores(closes, lookback=lookback, skip=skip)
    when = index[300]
    expected = {n: (1.0 + g) ** (lookback - skip) - 1.0 for n, g in rates.items()}
    got = scores.loc[when]
    max_err = max(abs(got[n] - expected[n]) for n in rates)
    order_ok = list(got.sort_values(ascending=False).index) == ["A", "C", "B"]

    weights = xs_momentum.xs_weights(
        closes, lookback=lookback, skip=skip, n_long=1, n_short=1,
        min_available=3, target_vol=None,
    )
    last = weights.iloc[-1]
    book_ok = bool(np.isclose(last["A"], 0.5) and np.isclose(last["B"], -0.5)
                   and np.isclose(last["C"], 0.0))

    print("\nSELFCHECK (a) — rank against a hand computation, 3 synthetic instruments")
    for name in rates:
        print(f"  {name}: score {got[name]:.6f}   expected {expected[name]:.6f}")
    print(f"  max abs error       : {max_err:.2e}")
    print(f"  ranking A > C > B   : {order_ok}")
    print(f"  book (+A / -B / 0 C): {book_ok}   weights {last.to_dict()}")

    # (b) Nothing is traded before six instruments quote together.
    starts = [0, 10, 20, 30, 40, 260, 300, 320]
    staggered = pd.DataFrame(
        {f"I{i}": np.where(np.arange(len(index)) >= s,
                           100.0 * (1.0 + 0.0003 * (i + 1)) ** np.arange(len(index)), np.nan)
         for i, s in enumerate(starts)},
        index=index,
    )
    w_stag = xs_momentum.xs_weights(staggered, lookback=63, skip=5, n_long=3, n_short=0)
    n_available = staggered.notna().sum(axis=1)
    first_six = index[n_available >= xs_momentum.MIN_AVAILABLE][0]
    non_zero = w_stag.index[w_stag.ne(0.0).any(axis=1)]
    first_trade = non_zero[0] if len(non_zero) else None
    # On the *full* calendar, not only on the rebalance rows: the weights are
    # undefined before the grid starts, and undefined must read as flat.
    before = w_stag.reindex(index).fillna(0.0).loc[index < first_six]
    gate_ok = (first_trade is not None and first_trade >= first_six
               and not bool(before.ne(0.0).to_numpy().any()))

    print("\nSELFCHECK (b) — no weight before six instruments quote")
    print(f"  6th instrument quotes on   : {first_six.date()}")
    print(f"  first non-zero weight on   : "
          f"{first_trade.date() if first_trade is not None else '-'}")
    print(f"  non-zero weights before it : {int(before.ne(0.0).to_numpy().sum())} "
          f"(over {len(before)} earlier sessions)")
    print(f"  gate holds                 : {gate_ok}")

    ok = max_err < 1e-9 and order_ok and book_ok and gate_ok
    print(f"\n  {'OK' if ok else 'FAIL'}\n")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selfcheck", action="store_true",
                    help="check the ranking and the availability gate, then exit")
    args = ap.parse_args()

    if args.selfcheck:
        return selfcheck()

    universe = read_universe()
    print(f"\n{'=' * 92}")
    print("  XS momentum screen — vbt pre-filter, one portfolio per config")
    print(f"  {len(universe)} instruments (TSMOM screen PASS), selection window <= "
          f"{SELECTION_END.date()} (holdout {holdout.HOLDOUT_START.date()})")
    print(f"  next-open fill, per-symbol half spread, swap {SWAP_BPS_PER_NIGHT * 1e4:.1f} bp/night "
          f"on gross exposure, Sharpe annualized on {ANN_DAYS} sessions")
    print(f"  gates: kill the family if best Sharpe < {MIN_SHARPE} or corr to the "
          f"TSMOM basket > {MAX_CORR_TO_TSMOM} without beating it")
    print(f"{'=' * 92}\n")

    closes, opens = load_panel(universe)
    grid = xs_momentum.trading_grid(closes)
    print(f"\n  panel {closes.shape[0]} session dates x {closes.shape[1]} instruments; "
          f"tradable grid {len(grid)} days from {grid.min().date()} "
          f"(>= {xs_momentum.MIN_AVAILABLE} quoting)")
    print(f"  instruments quoting per grid day: mean "
          f"{closes.loc[grid].notna().sum(axis=1).mean():.1f}\n")

    print("  equal-weight TSMOM basket (additivity yardstick) ...", flush=True)
    basket = tsmom_basket(universe)
    basket_acc = basket.vbt.returns(freq="1D", year_freq=pd.Timedelta(days=ANN_DAYS))
    basket_sharpe = float(basket_acc.sharpe_ratio())
    print(f"    {len(basket)} sessions, Sharpe {basket_sharpe:.3f}\n")

    slippage = slippage_series(closes.columns)
    rows = []
    for config in xs_momentum.GRID:
        print(f"  lookback {config['lookback']:>3}  long {config['n_long']}  "
              f"short {config['n_short']}", flush=True)
        ret_net, trades, weights = run_config(closes, opens, slippage, config)
        rows.append(metrics(config, ret_net, trades, weights, basket))

    df = pd.DataFrame(rows, columns=list(CSV_COLUMNS))
    df = df.sort_values("sharpe_net", ascending=False, na_position="last")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"\n{format_table(df)}\n")

    best = df.iloc[0]
    print(f"TSMOM equal-weight basket: Sharpe {basket_sharpe:.3f} "
          f"({len(basket)} sessions in window)")
    print(f"Best XS config: {best['config']} — Sharpe {best['sharpe_net']:.3f}, "
          f"corr to the basket {best['corr_to_tsmom']:.2f}")

    if best["sharpe_net"] < MIN_SHARPE:
        verdict = (f"KILL_FAMILY — best net Sharpe {best['sharpe_net']:.3f} < {MIN_SHARPE}")
    elif best["corr_to_tsmom"] > MAX_CORR_TO_TSMOM and best["sharpe_net"] <= basket_sharpe:
        verdict = (f"KILL_FAMILY — corr {best['corr_to_tsmom']:.2f} > {MAX_CORR_TO_TSMOM} "
                   f"and Sharpe {best['sharpe_net']:.3f} <= basket {basket_sharpe:.3f}")
    else:
        verdict = (f"PASS_FAMILY — {best['config']}, Sharpe {best['sharpe_net']:.3f}, "
                   f"corr {best['corr_to_tsmom']:.2f}")
    print(f"\nVERDICT: {verdict}")
    print(f"\nWritten: {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
