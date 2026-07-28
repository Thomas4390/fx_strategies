#!/usr/bin/env python
"""Pre-filter the Donchian breakout family across the TSMOM survivors.

Six configurations — ``entry_n`` in {55, 100, 252} crossed with an exit channel
at half and at a quarter of it — run long-only on the **14 instruments the
TSMOM screen passed** (``reports/research/tsmom_screen_2026H2.csv``), each with
the same source that screen used. This is a **pre-filter of raw edge**, not a
ranking: what survives goes to the MT5 tester, and the tester is what ranks
(``docs/research/momentum_expansion_2026H2.md`` §2).

Conventions, identical to ``screen_tsmom_universe.py`` on purpose — two
families that are not costed and annualized the same way cannot be compared:

- **Full history in, window out.** The pipeline simulates the whole series and
  the *returns* are sliced afterwards, so the 252-session channel never starts
  on a stranded warmup.
- **Selection window** ends 2025-12-31 (``framework.holdout.trim_insample``).
- **Costs**: ``fill="next_open"`` plus the per-symbol half spread of
  ``costs.yml`` as slippage, plus a 0.5 bp/night swap drag on gross exposure.
- **Annualization**: 252 sessions a year.

**One selection metric, fixed before the run: the simple mean of the per
instrument net Sharpe.** Not the best instrument, not a weighted blend — a
config that only works on one tradable is a config that was chosen by looking
at the answer. The per-instrument detail is printed, but it decides nothing.

    python scripts/screen_donchian.py --selfcheck
    python scripts/screen_donchian.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
if str(_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(_ROOT / "scripts"))

from framework import holdout, trials  # noqa: E402
from framework.costs import cost_for  # noqa: E402
from strategies import donchian_breakout, tsmom  # noqa: E402
from strategies.gold_momentum import (  # noqa: E402
    SESSION_CLOSE_HOUR,
    _daily_open,
    session_dates,
)
from update_data_manifest import assert_manifest_fresh  # noqa: E402

TSMOM_SCREEN_PATH = _ROOT / "reports" / "research" / "tsmom_screen_2026H2.csv"
OUTPUT_PATH = _ROOT / "reports" / "research" / "donchian_screen_2026H2.csv"

GOLD = "XAU-USD"
SELECTION_END = pd.Timestamp("2025-12-31")  # same cut as HOLDOUT_START, spelled out

# Closed grid, written down before the run: three entry channels (the two
# Turtle systems bracket 55, 252 is the one-year high) times the two classic
# exit asymmetries. Six configs, no more — the family is judged on these.
ENTRY_LENGTHS: tuple[int, ...] = (55, 100, 252)
EXIT_DIVISORS: tuple[int, ...] = (2, 4)

# 0.5 bp per night on gross exposure, the EA's Inp_SwapBpsPerNight.
SWAP_BPS_PER_NIGHT = 0.00005

ANN_DAYS = 252  # sessions a year — repo convention for annualization

# Gates, fixed before the run (momentum_expansion_2026H2.md §5).
KILL_SHARPE_AGG = 0.15
KEEP_SHARPE_INSTRUMENT = 0.30
MAX_CORR_TO_TSMOM = 0.5  # above this the family is not additive to TSMOM

CSV_COLUMNS: tuple[str, ...] = (
    "entry_n", "exit_n", "symbol", "source", "n_bars", "first_date", "last_date",
    "sharpe_net", "cagr_net", "maxdd", "trades_total", "trades_per_year",
    "exposure_median", "years_in_window", "status",
)


def build_grid() -> list[tuple[int, int]]:
    """The six ``(entry_n, exit_n)`` pairs, in a fixed order."""
    return [(n, n // d) for n in ENTRY_LENGTHS for d in EXIT_DIVISORS]


def read_universe() -> list[tuple[str, str]]:
    """``(symbol, source)`` for the TSMOM survivors, read off the screen CSV.

    The source is taken from the same file rather than re-derived: running
    Donchian on a different history than TSMOM would make the two families
    incomparable, which is precisely what the additivity check needs them to be.
    """
    screen = pd.read_csv(TSMOM_SCREEN_PATH)
    passed = screen[screen["verdict"] == "PASS"]
    if passed.empty:
        raise RuntimeError(f"no PASS verdict in {TSMOM_SCREEN_PATH.name}")
    return list(zip(passed["symbol"], passed["source"]))


def _by_date(series: pd.Series) -> pd.Series:
    """Same series, re-indexed on the session date under the 17:00 New York cut.

    Not ``normalize()``: a broker daily bar stamped 00:00 UTC lands at 20:00 the
    *previous* New York day (``screen_tsmom_universe._by_date``, same reason).
    """
    out = series.copy()
    out.index = session_dates(pd.DatetimeIndex(out.index), SESSION_CLOSE_HOUR)
    return out[~out.index.duplicated()]


def net_returns(pf) -> tuple[pd.Series, pd.Series]:
    """(net returns, gross exposure), both cut to the selection window.

    The swap drag is subtracted before the cut, on the full series, so it is
    charged with the exposure of the session it belongs to.
    """
    ret = pf.returns
    exposure = (pf.asset_value / pf.value).reindex(ret.index).fillna(0.0).abs()
    ret_net = holdout.trim_insample(ret - SWAP_BPS_PER_NIGHT * exposure)
    holdout.assert_not_optimizing(ret_net.index)
    return ret_net, holdout.trim_insample(exposure)


def count_trades(pf) -> int:
    """Trades closed inside the selection window, counted like the TSMOM screen.

    On the exit date, not the entry one: a trade still open at the cut has no
    realized result yet, and the two screens have to count the same thing.
    """
    trades = pf.trades.records_readable
    col = "Exit Index" if "Exit Index" in trades.columns else "Entry Index"
    return int(len(holdout.trim_insample(trades.set_index(col))))


def metrics(
    entry_n: int,
    exit_n: int,
    symbol: str,
    source: str,
    ret_net: pd.Series,
    exposure: pd.Series,
    n_trades: int,
) -> dict[str, object]:
    """One CSV row. Everything is computed on the post-swap, post-slippage series."""
    acc = ret_net.vbt.returns(freq="1D", year_freq=pd.Timedelta(days=ANN_DAYS))
    years = len(ret_net) / ANN_DAYS
    return dict(
        entry_n=entry_n,
        exit_n=exit_n,
        symbol=symbol,
        source=source,
        n_bars=len(ret_net),
        first_date=str(ret_net.index.min().date()),
        last_date=str(ret_net.index.max().date()),
        sharpe_net=float(acc.sharpe_ratio()),
        cagr_net=float(acc.annualized()),
        maxdd=float(acc.max_drawdown()),
        trades_total=n_trades,
        trades_per_year=n_trades / years if years else float("nan"),
        exposure_median=float(exposure.median()),
        years_in_window=years,
        status="OK",
    )


def aggregate_row(entry_n: int, exit_n: int, rows: list[dict[str, object]]) -> dict:
    """The AGG line: the simple mean of the per-instrument Sharpe, and context.

    ``source`` carries the instrument count rather than a loader name — an
    aggregate has no source, and the count is what the mean has to be read
    against.
    """
    frame = pd.DataFrame(rows)
    return dict(
        entry_n=entry_n,
        exit_n=exit_n,
        symbol="AGG",
        source=f"{len(frame)} instruments",
        n_bars=int(frame["n_bars"].sum()),
        first_date=min(frame["first_date"]),
        last_date=max(frame["last_date"]),
        sharpe_net=float(frame["sharpe_net"].mean()),
        cagr_net=float(frame["cagr_net"].mean()),
        maxdd=float(frame["maxdd"].mean()),
        trades_total=int(frame["trades_total"].sum()),
        trades_per_year=float(frame["trades_per_year"].sum()),
        exposure_median=float(frame["exposure_median"].median()),
        years_in_window=float(frame["years_in_window"].mean()),
        status="AGG",
    )


def invalid_row(entry_n: int, exit_n: int, symbol: str, source: str, reason: str) -> dict:
    """Row for an instrument that does not load: no metric, no aggregation.

    Written rather than dropped — a screen that silently loses an instrument
    reads as if it had been tested and rejected.
    """
    print(f"  !! {symbol}: {reason}", flush=True)
    row: dict[str, object] = {c: float("nan") for c in CSV_COLUMNS}
    row.update(
        entry_n=entry_n, exit_n=exit_n, symbol=symbol, source=source,
        status="DATA_INVALID",
    )
    return row


def basket(streams: dict[str, pd.Series]) -> pd.Series:
    """Equal-weight daily return of a basket, on the session date.

    Instruments enter the mean only on the sessions they actually have, so the
    1990 histories are not diluted by the ones starting in 2022.
    """
    frame = pd.DataFrame({name: _by_date(s) for name, s in streams.items()})
    return frame.mean(axis=1).dropna()


def tsmom_stream(symbol: str, source: str) -> pd.Series:
    """Net TSMOM returns for one instrument — the reference the family must add to."""
    pf, _ = tsmom.pipeline(
        symbol, loader_override=source, fill="next_open", slippage=cost_for(symbol)
    )
    return net_returns(pf)[0]


def format_table(df: pd.DataFrame) -> str:
    """Screen table, values formatted for reading rather than for re-parsing."""
    shown = df.copy()
    for col, fmt in (
        ("sharpe_net", "{:.3f}"), ("cagr_net", "{:.1%}"), ("maxdd", "{:.1%}"),
        ("trades_per_year", "{:.1f}"), ("exposure_median", "{:.2f}"),
        ("years_in_window", "{:.1f}"),
    ):
        if col in shown:
            shown[col] = shown[col].map(lambda v, f=fmt: "" if pd.isna(v) else f.format(v))
    for col in ("n_bars", "trades_total"):
        if col in shown:
            shown[col] = shown[col].map(lambda v: "" if pd.isna(v) else str(int(v)))
    return shown.to_string(index=False)


def selfcheck() -> int:
    """Causality and fill convention on XAU-USD, config (100, 50). Exit code."""
    entry_n, exit_n = 100, 50
    _, data = tsmom.load_instrument(GOLD)
    order_columns = ["Fill Index", "Size", "Price", "Side"]

    print(f"\nSELFCHECK — {GOLD}, Donchian ({entry_n}, {exit_n}), long-only")

    # 1. Causality: the orders placed before a cut cannot depend on data after it.
    cut = int(len(data.close) * 0.8)
    cutoff = data.wrapper.index[cut]
    pf_full, _ = donchian_breakout.pipeline(data, entry_n=entry_n, exit_n=exit_n)
    pf_trunc, _ = donchian_breakout.pipeline(
        data.iloc[:cut], entry_n=entry_n, exit_n=exit_n
    )

    def _before(pf) -> pd.DataFrame:
        rec = pf.orders.records_readable
        return rec[rec["Fill Index"] < cutoff][order_columns].reset_index(drop=True)

    full, truncated = _before(pf_full), _before(pf_trunc)
    causal = len(full) > 0 and full.equals(truncated)
    print(f"  truncation at 80 %  : {cutoff.date()}")
    print(f"  orders before cut   : {len(full)} full vs {len(truncated)} truncated")
    print(f"  causality           : {'OK' if causal else 'FAIL'}")

    # 2. Fill: every order price is the open of the session after the signal.
    pf, _ = donchian_breakout.pipeline(
        data, entry_n=entry_n, exit_n=exit_n, fill="next_open", slippage=0.0
    )
    opens = _daily_open(data).reindex(pf.wrapper.index)
    index = pf.wrapper.index
    records = pf.orders.records_readable
    mismatches = 0
    for _, order in records.iterrows():
        fill_stamp = index[index.get_loc(order["Signal Index"]) + 1]
        if order["Fill Index"] != fill_stamp:
            mismatches += 1
        elif abs(order["Price"] - float(opens.loc[fill_stamp])) > 1e-9:
            mismatches += 1
    filled = len(records) > 0 and mismatches == 0
    print(f"  orders checked      : {len(records)}")
    print(f"  next-open fill      : {'OK' if filled else f'FAIL ({mismatches} off)'}")

    ok = causal and filled
    print(f"\n  {'OK' if ok else 'FAIL'}\n")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--selfcheck", action="store_true", help="check causality and fill, then exit"
    )
    args = ap.parse_args()

    assert_manifest_fresh()
    if args.selfcheck:
        return selfcheck()

    universe = read_universe()
    grid = build_grid()

    trials.log_trials(
        "donchian", len(grid),
        "N {55,100,252} x sortie {N/2, N/4}, selection par Sharpe agrege",
        config_key="donchian:6cfg:tsmom_pass_universe",
    )

    print(f"\n{'=' * 92}")
    print("  Donchian breakout screen — vbt pre-filter, closed grid of 6 configs")
    print(f"  {len(universe)} instruments (TSMOM PASS list), selection window "
          f"<= {SELECTION_END.date()} (holdout {holdout.HOLDOUT_START.date()})")
    print(f"  next-open fill, per-symbol half spread, swap {SWAP_BPS_PER_NIGHT * 1e4:.1f} "
          f"bp/night, Sharpe annualized on {ANN_DAYS} sessions")
    print("  selection metric = simple mean of the per-instrument net Sharpe")
    print(f"{'=' * 92}\n")

    rows: list[dict[str, object]] = []
    # (entry_n, exit_n) -> {symbol: net returns}, kept for the best config's basket.
    streams: dict[tuple[int, int], dict[str, pd.Series]] = {g: {} for g in grid}

    for symbol, source in universe:
        print(f"  {symbol:<9} [{source}]", flush=True)
        try:
            _, data = tsmom.load_instrument(symbol, loader_override=source)
        except (ValueError, FileNotFoundError, KeyError) as exc:
            rows += [invalid_row(n, x, symbol, source, str(exc)) for n, x in grid]
            continue
        for entry_n, exit_n in grid:
            pf, _ = donchian_breakout.pipeline(
                data,
                entry_n=entry_n,
                exit_n=exit_n,
                fill="next_open",
                slippage=cost_for(symbol),
            )
            ret_net, exposure = net_returns(pf)
            rows.append(
                metrics(entry_n, exit_n, symbol, source, ret_net, exposure, count_trades(pf))
            )
            streams[(entry_n, exit_n)][symbol] = ret_net

    detail = pd.DataFrame(rows, columns=list(CSV_COLUMNS))
    agg = [
        aggregate_row(n, x, [r for r in rows
                             if r["status"] == "OK" and (r["entry_n"], r["exit_n"]) == (n, x)])
        for n, x in grid
    ]
    agg_df = pd.DataFrame(agg, columns=list(CSV_COLUMNS)).sort_values(
        "sharpe_net", ascending=False
    )

    out = pd.concat([detail, pd.DataFrame(agg, columns=list(CSV_COLUMNS))])
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_PATH, index=False)

    print(f"\n{'-' * 92}\n  AGGREGATE — the 6 configs, ranked by the mean per-instrument "
          f"Sharpe\n{'-' * 92}")
    print(format_table(agg_df.drop(columns=["symbol", "n_bars"])))

    best = agg_df.iloc[0]
    best_key = (int(best["entry_n"]), int(best["exit_n"]))
    best_detail = detail[
        (detail["entry_n"] == best_key[0]) & (detail["exit_n"] == best_key[1])
    ].sort_values("sharpe_net", ascending=False)
    print(f"\n{'-' * 92}\n  BEST CONFIG — entry_n={best_key[0]}, exit_n={best_key[1]}, "
          f"per instrument\n{'-' * 92}")
    print(format_table(best_detail.drop(columns=["entry_n", "exit_n"])))

    # Additivity: two families that produce the same daily return stream are one
    # family. Baskets, not pairs — the sleeve would be held as a basket.
    print("\n  TSMOM reference basket (same instruments, same sources)...", flush=True)
    tsmom_streams: dict[str, pd.Series] = {}
    for symbol, source in universe:
        try:
            tsmom_streams[symbol] = tsmom_stream(symbol, source)
        except (ValueError, FileNotFoundError, KeyError) as exc:
            print(f"  !! {symbol}: {exc}", flush=True)
    joined = pd.concat(
        [basket(streams[best_key]).rename("donchian"),
         basket(tsmom_streams).rename("tsmom")],
        axis=1, join="inner",
    ).dropna()
    corr = float(joined["donchian"].corr(joined["tsmom"]))
    print(f"  corr(donchian basket, tsmom basket) = {corr:.3f} on {len(joined)} "
          f"common sessions (additivity needs < {MAX_CORR_TO_TSMOM})")

    print(f"\n{'=' * 92}")
    if float(best["sharpe_net"]) < KILL_SHARPE_AGG:
        print(f"  VERDICT: KILL_FAMILY — best aggregated Sharpe "
              f"{float(best['sharpe_net']):.3f} < {KILL_SHARPE_AGG}")
    else:
        keep = best_detail[best_detail["sharpe_net"] >= KEEP_SHARPE_INSTRUMENT]
        print(f"  VERDICT: PASS — config entry_n={best_key[0]}, exit_n={best_key[1]}, "
              f"aggregated Sharpe {float(best['sharpe_net']):.3f} >= {KILL_SHARPE_AGG}")
        print(f"  instruments with individual Sharpe >= {KEEP_SHARPE_INSTRUMENT} "
              f"({len(keep)}/{len(best_detail)}): "
              + ", ".join(f"{r.symbol} {r.sharpe_net:.2f}" for r in keep.itertuples()))
        print(f"  additivity to TSMOM: corr {corr:.3f} — "
              f"{'OK' if corr < MAX_CORR_TO_TSMOM else 'REDUNDANT'}")
    print("  final ranking belongs to the MT5 tester — this only kills the hopeless")
    print(f"{'=' * 92}")
    print(f"\nWritten: {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
