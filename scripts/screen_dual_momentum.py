#!/usr/bin/env python
"""Pre-filter the dual / acceleration momentum family on the TSMOM survivors.

Four configurations — ``{gate, brake} x {(63, 252), (21, 126)}``, the closed
grid of ``strategies.dual_momentum.CONFIGS`` — run on the **14 instruments the
TSMOM screen passed** (``reports/research/tsmom_screen_2026H2.csv``, verdict
PASS), each on the same source that screen used. As with the TSMOM pass, this
is a pre-filter of raw edge and not a ranking: the ranking belongs to the MT5
tester.

The conventions are those of ``scripts/screen_tsmom_universe.py``, on purpose —
a family is only comparable to its base if nothing else moved:

- full history in, selection window out (``framework.holdout.trim_insample``,
  cut at 2025-12-31);
- ``fill="next_open"``, per-symbol half spread from ``costs.yml`` as slippage,
  0.5 bp/night swap drag on gross exposure;
- Sharpe annualized on 252 sessions.

**Selection metric: the simple mean of the per-instrument net Sharpes.** One
number per configuration, unweighted — an average that a single long history
cannot dominate, and the only quantity these four configurations are ranked on.

Because this family filters the very score TSMOM trades, its basket will be
correlated with the TSMOM basket by construction. The correlation is measured
here rather than assumed away, and the family is killed if it is high without a
better aggregate Sharpe than the base it filters.

    python scripts/screen_dual_momentum.py --selfcheck
    python scripts/screen_dual_momentum.py
    python scripts/screen_dual_momentum.py --only XAU-USD
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

from framework import holdout  # noqa: E402
from framework.costs import cost_for  # noqa: E402
from strategies import dual_momentum, gold_momentum, tsmom  # noqa: E402
from strategies.dual_momentum import CONFIGS  # noqa: E402
from strategies.gold_momentum import SESSION_CLOSE_HOUR, session_dates  # noqa: E402
from update_data_manifest import assert_manifest_fresh  # noqa: E402

TSMOM_SCREEN_PATH = _ROOT / "reports" / "research" / "tsmom_screen_2026H2.csv"
OUTPUT_PATH = _ROOT / "reports" / "research" / "dual_screen_2026H2.csv"

GOLD = "XAU-USD"
SELECTION_END = pd.Timestamp("2025-12-31")  # same cut as HOLDOUT_START, spelled out

SWAP_BPS_PER_NIGHT = 0.00005  # the EA's Inp_SwapBpsPerNight, on gross exposure
ANN_DAYS = 252

# The baseline this family has to beat: the same sleeve without the filter,
# long-only, on the same 14 instruments and the same conventions.
BASELINE = "tsmom_ref"

# Kill gates, from the brief of the cycle. The second one is a pair: a filter
# that merely tracks the score it filters adds nothing unless it also beats it.
MIN_AGG_SHARPE = 0.15
MAX_BASELINE_CORR = 0.80

CSV_COLUMNS: tuple[str, ...] = (
    "config", "mode", "fast_n", "slow_n", "instrument", "source", "n_bars",
    "first_date", "last_date", "sharpe_net", "cagr_net", "maxdd",
    "trades_total", "trades_per_year", "years_in_window",
)


def config_name(mode: str, fast_n: int, slow_n: int) -> str:
    """``gate_63_252`` — the key every table and CSV row is joined on."""
    return f"{mode}_{fast_n}_{slow_n}"


def read_pass_universe() -> list[tuple[str, str]]:
    """``(symbol, source)`` for the TSMOM PASS instruments, read off the screen.

    The source is taken from the screen's own ``source`` column rather than from
    the registry: which of two histories was judged trustworthy is a decision
    that file already recorded, and re-deciding it here would silently screen a
    different universe than the one the base family was screened on.
    """
    df = pd.read_csv(TSMOM_SCREEN_PATH)
    passed = df[df["verdict"] == "PASS"]
    if passed.empty:
        raise RuntimeError(f"no PASS instrument in {TSMOM_SCREEN_PATH.name}")
    return [(row.symbol, row.source) for row in passed.itertuples()]


def net_returns(pf) -> tuple[pd.Series, pd.DataFrame]:
    """Net returns and trades of a run, both cut to the selection window.

    The swap drag is subtracted on the full series, before the cut, so every
    session is charged with the exposure it actually carried.
    """
    ret = pf.returns
    exposure = (pf.asset_value / pf.value).reindex(ret.index).fillna(0.0)
    ret_net = holdout.trim_insample(ret - SWAP_BPS_PER_NIGHT * exposure.abs())
    holdout.assert_not_optimizing(ret_net.index)

    trades = pf.trades.records_readable
    col = "Exit Index" if "Exit Index" in trades.columns else "Entry Index"
    return ret_net, holdout.trim_insample(trades.set_index(col))


def metrics(
    config: str, mode: str, fast_n: int, slow_n: int,
    symbol: str, source: str, ret_net: pd.Series, trades: pd.DataFrame,
) -> dict[str, object]:
    """One CSV row, all of it computed post-slippage and post-swap."""
    acc = ret_net.vbt.returns(freq="1D", year_freq=pd.Timedelta(days=ANN_DAYS))
    years = len(ret_net) / ANN_DAYS
    n_trades = int(len(trades))
    return dict(
        config=config, mode=mode, fast_n=fast_n, slow_n=slow_n,
        instrument=symbol, source=source, n_bars=len(ret_net),
        first_date=str(ret_net.index.min().date()),
        last_date=str(ret_net.index.max().date()),
        sharpe_net=float(acc.sharpe_ratio()),
        cagr_net=float(acc.annualized()),
        maxdd=float(acc.max_drawdown()),
        trades_total=n_trades,
        trades_per_year=n_trades / years if years else float("nan"),
        years_in_window=years,
    )


def agg_row(config: str, rows: list[dict[str, object]]) -> dict[str, object]:
    """The aggregate line: the simple mean of the per-instrument Sharpes.

    Unweighted deliberately. Weighting by history would hand the verdict to
    US100's 36 years, and the question here is whether the family carries an
    edge across instruments, not whether it carries one on the longest series.
    """
    sharpes = [float(r["sharpe_net"]) for r in rows]
    row = {c: float("nan") for c in CSV_COLUMNS}
    row.update(
        config=config, mode=rows[0]["mode"], fast_n=rows[0]["fast_n"],
        slow_n=rows[0]["slow_n"], instrument="AGG", source=f"{len(rows)} instruments",
        sharpe_net=float(pd.Series(sharpes).mean()),
        trades_total=int(sum(int(r["trades_total"]) for r in rows)),
    )
    return row


def _by_date(series: pd.Series) -> pd.Series:
    """Same series, re-indexed on the session date under the 17:00 New York cut.

    Copied from the TSMOM screen for the same reason: two instruments carry the
    wall clock of their own source, so an exact-timestamp join between them
    returns an empty intersection.
    """
    out = series.copy()
    out.index = session_dates(pd.DatetimeIndex(out.index), SESSION_CLOSE_HOUR)
    return out[~out.index.duplicated()]


def basket(returns_by_symbol: dict[str, pd.Series]) -> pd.Series:
    """Equally weighted basket of the per-instrument net returns.

    Equal weight among whatever is trading that session: the histories start
    30 years apart, so requiring all 14 would shrink the basket to 2022-2025.
    """
    # sort=True: the 14 indexes start 30 years apart, and their union is only a
    # calendar if it is sorted (pandas 4 stops sorting it by default).
    frame = pd.concat(
        {sym: _by_date(ret) for sym, ret in returns_by_symbol.items()}, axis=1, sort=True
    )
    return frame.mean(axis=1).dropna()


def basket_corr(left: pd.Series, right: pd.Series) -> float:
    """Correlation of two baskets on their common sessions."""
    joined = pd.concat([left, right], axis=1, join="inner").dropna()
    if len(joined) < 3:
        return float("nan")
    return float(joined.iloc[:, 0].corr(joined.iloc[:, 1]))


def run_symbol(symbol: str, source: str) -> dict[str, tuple[pd.Series, pd.DataFrame]]:
    """All five simulations of one instrument, on one load of its data.

    Four dual configurations plus the unfiltered TSMOM baseline. Loading once
    matters: the FX minute exports are the slow part of this screen, not the
    simulations.
    """
    _, data = tsmom.load_instrument(symbol, loader_override=source)
    inst = tsmom.INSTRUMENTS[symbol]
    common = dict(
        session_close_hour=inst.session_close_hour,
        ann_factor=inst.ann_factor,
        fill="next_open",
        slippage=cost_for(symbol),
    )

    out: dict[str, tuple[pd.Series, pd.DataFrame]] = {}
    for mode, fast_n, slow_n in CONFIGS:
        pf, _ = dual_momentum.pipeline(
            data, mode=mode, fast_n=fast_n, slow_n=slow_n, **common
        )
        out[config_name(mode, fast_n, slow_n)] = net_returns(pf)
    pf_base, _ = gold_momentum.pipeline(data, **common)
    out[BASELINE] = net_returns(pf_base)
    return out


def selfcheck() -> int:
    """Causality and next-open fill on XAU-USD, gate (63, 252). Exit code."""
    mode, fast_n, slow_n = CONFIGS[0]
    _, data = tsmom.load_instrument(GOLD)
    params = dict(mode=mode, fast_n=fast_n, slow_n=slow_n, fill="next_open", slippage=0.0)

    print(f"\nSELFCHECK — {GOLD}, {config_name(mode, fast_n, slow_n)}, next_open, no slippage")

    cut = int(len(data.close) * 0.8)
    cutoff = data.wrapper.index[cut]
    columns = ["Fill Index", "Size", "Price", "Side"]

    def orders_before(pf) -> pd.DataFrame:
        rec = pf.orders.records_readable
        return rec[rec["Fill Index"] < cutoff][columns].reset_index(drop=True)

    pf_full, ind = dual_momentum.pipeline(data, **params)
    pf_cut, _ = dual_momentum.pipeline(data.iloc[:cut], **params)
    full, truncated = orders_before(pf_full), orders_before(pf_cut)

    causal = len(full) > 0 and full.equals(truncated)
    print(f"  sessions / orders   : {len(ind.close)} / {len(pf_full.orders.records_readable)}")
    print(f"  orders before {cutoff.date()} : {len(full)} full vs {len(truncated)} truncated")
    print(f"  causality           : {'OK' if causal else 'FAIL'} "
          f"(truncating the sample must not move a past order)")

    opens = gold_momentum._daily_open(data).reindex(pf_full.wrapper.index)
    index = pf_full.wrapper.index
    bad = []
    for _, order in pf_full.orders.records_readable.iterrows():
        stamp = index[index.get_loc(order["Signal Index"]) + 1]
        expected = float(opens.loc[stamp])
        if order["Fill Index"] != stamp or abs(order["Price"] - expected) > 1e-9 * abs(expected):
            bad.append(str(order["Signal Index"]))
    filled = not bad
    print(f"  next-open fill      : {'OK' if filled else 'FAIL'} "
          f"({len(bad)} order(s) not filled at the following session open)")

    # The gate must never hold a long into a negative acceleration — the whole
    # point of the mode, and it is checked on the state, not on a metric.
    gated = int((ind.long_ok & (ind.accel < 0)).sum())
    print(f"  gate discipline     : {'OK' if gated == 0 else 'FAIL'} "
          f"({gated} long session(s) with accel < 0)")

    ok = causal and filled and gated == 0
    print(f"\n  {'OK' if ok else 'FAIL'}\n")
    return 0 if ok else 1


def format_table(df: pd.DataFrame) -> str:
    """Table with the money columns rounded to what they can honestly carry."""
    shown = df.copy()
    for col, fmt in (
        ("sharpe_net", "{:.3f}"), ("cagr_net", "{:.1%}"), ("maxdd", "{:.1%}"),
        ("trades_per_year", "{:.1f}"), ("years_in_window", "{:.1f}"),
    ):
        if col in shown.columns:
            shown[col] = shown[col].map(lambda v, f=fmt: "" if pd.isna(v) else f.format(v))
    for col in ("n_bars", "trades_total", "fast_n", "slow_n"):
        if col in shown.columns:
            shown[col] = shown[col].map(lambda v: "" if pd.isna(v) else str(int(v)))
    return shown.to_string(index=False)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selfcheck", action="store_true", help="causality + fill check, then exit")
    ap.add_argument("--only", metavar="SYMBOL", help="run a single instrument")
    args = ap.parse_args()

    assert_manifest_fresh()

    if args.selfcheck:
        return selfcheck()

    universe = read_pass_universe()
    if args.only:
        universe = [(sym, src) for sym, src in universe if sym == args.only]
        if not universe:
            raise SystemExit(f"--only {args.only}: not a PASS instrument of the TSMOM screen")

    names = [config_name(*cfg) for cfg in CONFIGS]
    print(f"\n{'=' * 92}")
    print("  Dual / acceleration momentum screen — vbt pre-filter, closed grid of 4 configs")
    print(f"  {len(universe)} TSMOM PASS instruments, selection window <= {SELECTION_END.date()} "
          f"(holdout {holdout.HOLDOUT_START.date()})")
    print(f"  next-open fill, per-symbol half spread, swap {SWAP_BPS_PER_NIGHT * 1e4:.1f} bp/night, "
          f"Sharpe annualized on {ANN_DAYS} sessions")
    print("  selection metric = simple mean of the per-instrument net Sharpes")
    print(f"{'=' * 92}\n")

    rows: list[dict[str, object]] = []
    per_config: dict[str, list[dict[str, object]]] = {name: [] for name in [*names, BASELINE]}
    returns: dict[str, dict[str, pd.Series]] = {name: {} for name in [*names, BASELINE]}

    for symbol, source in universe:
        print(f"  {symbol:<9} [{source}]", flush=True)
        try:
            results = run_symbol(symbol, source)
        except (ValueError, KeyError, FileNotFoundError) as exc:
            # Logged and skipped: an instrument that cannot load has no verdict,
            # and dropping it silently would read as if it had been tested.
            print(f"  !! {symbol}: {exc}")
            continue
        for name, (ret_net, trades) in results.items():
            mode, fast_n, slow_n = (
                (BASELINE, 0, 0) if name == BASELINE
                else CONFIGS[names.index(name)]
            )
            row = metrics(name, mode, fast_n, slow_n, symbol, source, ret_net, trades)
            rows.append(row)
            per_config[name].append(row)
            returns[name][symbol] = ret_net

    for name in [*names, BASELINE]:
        if per_config[name]:
            rows.append(agg_row(name, per_config[name]))

    df = pd.DataFrame(rows, columns=list(CSV_COLUMNS))
    df = df.astype({"n_bars": "Int64", "trades_total": "Int64"})
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    aggregates = df[df["instrument"] == "AGG"].sort_values("sharpe_net", ascending=False)
    print(f"\n{'-' * 92}\n  AGGREGATE — simple mean of the {len(universe)} net Sharpes\n{'-' * 92}")
    print(format_table(aggregates[["config", "instrument", "sharpe_net", "trades_total"]]))

    best = aggregates[aggregates["config"] != BASELINE].iloc[0]
    best_name = str(best["config"])
    base_sharpe = float(
        aggregates.loc[aggregates["config"] == BASELINE, "sharpe_net"].iloc[0]
    )
    detail = df[(df["config"] == best_name) & (df["instrument"] != "AGG")]
    print(f"\n{'-' * 92}\n  BEST CONFIG — {best_name}, per instrument\n{'-' * 92}")
    print(format_table(detail.drop(columns=["config", "mode", "fast_n", "slow_n"])
                       .sort_values("sharpe_net", ascending=False)))

    corr = basket_corr(basket(returns[best_name]), basket(returns[BASELINE]))
    best_sharpe = float(best["sharpe_net"])
    print(f"\n  aggregate Sharpe, {best_name:<12}: {best_sharpe:.3f}")
    print(f"  aggregate Sharpe, {BASELINE:<12}: {base_sharpe:.3f}  (unfiltered long-only TSMOM)")
    print(f"  basket correlation vs TSMOM     : {corr:.3f}  "
          f"(equally weighted baskets, common sessions)")

    reasons = []
    if best_sharpe < MIN_AGG_SHARPE:
        reasons.append(f"best aggregate Sharpe {best_sharpe:.3f} < {MIN_AGG_SHARPE}")
    if corr > MAX_BASELINE_CORR and best_sharpe <= base_sharpe:
        reasons.append(
            f"basket correlation {corr:.3f} > {MAX_BASELINE_CORR} while the aggregate "
            f"Sharpe {best_sharpe:.3f} does not beat the TSMOM baseline {base_sharpe:.3f}"
        )
    if reasons:
        print(f"\n  VERDICT: KILL_FAMILY — {'; '.join(reasons)}")
    else:
        print(f"\n  VERDICT: SURVIVES pre-filter — best {best_name}, aggregate Sharpe "
              f"{best_sharpe:.3f} vs {base_sharpe:.3f} baseline, basket corr {corr:.3f}. "
              "Ranking belongs to the MT5 tester.")

    print(f"\nWritten: {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
