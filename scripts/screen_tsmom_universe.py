#!/usr/bin/env python
"""Pre-filter the TSMOM sleeve across the whole tradable universe.

One sleeve, one configuration — the production gold defaults — run on the 21
instruments of the registry with the longest history each of them has. This is
a **pre-filter of raw edge**, not a ranking: what survives here goes to the MT5
tester, and the tester is what ranks. Nothing is tuned per instrument, so the
only thing the numbers can say is "this tradable does / does not carry a
positive momentum edge net of costs".

Conventions, all of them deliberate:

- **Full history in, window out.** The pipeline always simulates the whole
  series and the *returns* are sliced afterwards. Slicing the prices first
  would strand the 250-session lookback with no history — the bug documented
  in ``sweep_gold_sizing.load_daily``.
- **Selection window** ends 2025-12-31: ``framework.holdout.trim_insample``,
  repo policy (``HOLDOUT_START`` = 2026-01-01). The 2026 slice is frozen.
- **Costs**: ``fill="next_open"`` (decide on close[t], fill on open[t+1], what
  MT5 does) plus the per-symbol half spread of ``costs.yml`` as slippage, plus
  a 0.5 bp/night swap drag charged on gross exposure — the EA's
  ``Inp_SwapBpsPerNight=0.5``.
- **Annualization**: 252 sessions a year, the repo convention. ``--selfcheck``
  reproduces the parity oracle, which is stated in VBT's own default (365
  days), and prints both so the two cannot be confused.
- **Sources**: the long daily parquets for the non-FX instruments, except where
  ``reports/research/screening_source_check.json`` says the long series does
  not match the broker's (``BROKER_ONLY``), and the long minute parquets for
  the four majors that have one. The minute exports are indexed on naive broker
  time rather than New York, so the session boundary is cut at a different wall
  clock — immaterial to a daily signal, which only needs one cut per day.

    python scripts/screen_tsmom_universe.py --selfcheck
    python scripts/screen_tsmom_universe.py
    python scripts/screen_tsmom_universe.py --only EUR-USD
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
if str(_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(_ROOT / "scripts"))

from framework import holdout, trials  # noqa: E402
from framework.costs import cost_for  # noqa: E402
from strategies import tsmom  # noqa: E402
from strategies.gold_momentum import SESSION_CLOSE_HOUR, session_dates  # noqa: E402
from update_data_manifest import assert_manifest_fresh  # noqa: E402

# The ten majors/crosses, in registry order; the first four also have a long
# minute export (2018 ->) and are read from it rather than from the broker
# daily one (2020-11 ->).
_FX_PAIRS: tuple[str, ...] = (
    "EUR-USD", "GBP-USD", "USD-JPY", "USD-CAD", "USD-CHF",
    "AUD-USD", "NZD-USD", "EUR-GBP", "EUR-JPY", "GBP-JPY",
)
_FX_LONG_MINUTE: tuple[str, ...] = ("EUR-USD", "GBP-USD", "USD-JPY", "USD-CAD")

_NON_FX: tuple[str, ...] = (
    "XAG-USD", "XTI-USD", "XBR-USD", "XNG-USD", "US500",
    "US100", "US30", "GER40", "JPN225", "UK100",
)

# Crosses yen ajoutés le 2026-07-28, lus sur la série longue Yahoo (~22 ans).
# Pas de verdict de source pour eux : `check_screening_vs_broker` compare une
# série longue au CFD du courtier, et ces paires n'ont pas encore d'export
# broker. Ce sont du FX comptant, pas des futures à rolls ni des indices cash,
# donc la divergence structurelle que ce verdict cherche n'a pas lieu d'être —
# mais leur spread reste un PROXY (celui de GBP-JPY, le plus large des crosses
# yen mesurés) tant qu'aucun historique broker ne le mesure.
_FX_LONG_YAHOO: tuple[str, ...] = ("AUD-JPY", "NZD-JPY", "CAD-JPY")

SOURCE_CHECK_PATH = _ROOT / "reports" / "research" / "screening_source_check.json"
OUTPUT_PATH = _ROOT / "reports" / "research" / "tsmom_screen_2026H2.csv"

GOLD = "XAU-USD"
SELECTION_END = pd.Timestamp("2025-12-31")  # same cut as HOLDOUT_START, spelled out

# 0.5 bp per night on gross exposure, the EA's Inp_SwapBpsPerNight. Charged on
# every session because a daily sleeve holds overnight by construction.
SWAP_BPS_PER_NIGHT = 0.00005

ANN_DAYS = 252  # sessions a year — repo convention for annualization
MIN_TRADES_PER_YEAR = 3.0

# Parity oracle: XAU-USD, next-open fill, 2 bp slippage, returns over
# 2021-01-01 -> 2026-04-30, Sharpe stated in VBT's default annualization
# (365 days). Not the selection window — a fixture, nothing else.
ORACLE = dict(
    symbol=GOLD, slippage=0.0002,
    start="2021-01-01", end="2026-04-30",
    sharpe_raw=1.154, sharpe_net=1.082, tol=0.01,
)

CSV_COLUMNS: tuple[str, ...] = (
    "symbol", "source", "n_bars", "first_date", "last_date", "sharpe_net",
    "cagr_net", "maxdd", "trades_total", "trades_per_year",
    "avg_price_ret_per_trade", "exposure_median", "years_in_window",
    "corr_to_gold", "verdict",
)


def read_source_verdicts() -> dict[str, str]:
    """Source per non-FX instrument, read off the verdict file, never guessed.

    ``BROKER_ONLY`` means the long series disagrees with what the broker will
    actually fill, so the short broker export is the honest one; anything else
    keeps the long history.
    """
    payload = json.loads(SOURCE_CHECK_PATH.read_text())
    verdicts = {r["instrument"]: r["verdict"] for r in payload["results"]}
    missing = [name for name in _NON_FX if name not in verdicts]
    if missing:
        raise RuntimeError(
            f"{SOURCE_CHECK_PATH.name} carries no verdict for {missing} — "
            "run scripts/investigations/check_screening_vs_broker.py first."
        )
    return {name: ("mt5" if verdicts[name] == "BROKER_ONLY" else "yahoo") for name in _NON_FX}


def build_universe() -> list[tuple[str, str]]:
    """``(symbol, loader)`` per instrument, longest usable history each."""
    universe = [(GOLD, "qc")]
    universe += [
        (pair, "fx_minute" if pair in _FX_LONG_MINUTE else "mt5") for pair in _FX_PAIRS
    ]
    sources = read_source_verdicts()
    universe += [(name, sources[name]) for name in _NON_FX]
    universe += [(name, "yahoo") for name in _FX_LONG_YAHOO]
    return universe


def run_instrument(symbol: str, loader: str) -> tuple[pd.Series, pd.Series, pd.DataFrame]:
    """Simulate ``symbol`` over its full history; return (net returns, exposure, trades).

    The three outputs are already cut to the selection window. The swap drag is
    subtracted before the cut, on the full series, so it is charged with the
    exposure of the session it belongs to.
    """
    pf, _ = tsmom.pipeline(
        symbol,
        loader_override=loader,
        fill="next_open",
        slippage=cost_for(symbol),
    )
    ret = pf.returns
    exposure = (pf.asset_value / pf.value).reindex(ret.index).fillna(0.0)
    ret_net = ret + SWAP_BPS_PER_NIGHT * exposure.abs() * tsmom.carry_sign(symbol)

    trades = pf.trades.records_readable
    col = "Exit Index" if "Exit Index" in trades.columns else "Entry Index"
    trades = holdout.trim_insample(trades.set_index(col))

    ret_net = holdout.trim_insample(ret_net)
    holdout.assert_not_optimizing(ret_net.index)
    return ret_net, holdout.trim_insample(exposure.abs()), trades


def avg_price_return(trades: pd.DataFrame) -> float:
    """Mean signed price move per trade, entry to exit — the raw edge per bet.

    Deliberately not ``Return``: that one is a return on the invested value and
    therefore carries the leverage, which is a sizing choice, not an edge.
    """
    if not len(trades):
        return float("nan")
    sign = np.where(trades["Direction"].eq("Short"), -1.0, 1.0)
    move = trades["Avg Exit Price"].to_numpy() / trades["Avg Entry Price"].to_numpy() - 1.0
    return float((sign * move).mean())


def _by_date(series: pd.Series) -> pd.Series:
    """Same series, re-indexed on the session date under the 17:00 New York cut.

    Not ``normalize()``: a broker daily bar stamped 00:00 UTC lands at 20:00 the
    *previous* New York day, so truncating to the calendar day would shift the
    whole series back by one session against a resampled minute export. The
    session boundary the sleeve itself uses puts both back on the same day.
    """
    out = series.copy()
    out.index = session_dates(pd.DatetimeIndex(out.index), SESSION_CLOSE_HOUR)
    return out[~out.index.duplicated()]


def metrics(
    symbol: str,
    source: str,
    ret_net: pd.Series,
    exposure: pd.Series,
    trades: pd.DataFrame,
    gold_ret: pd.Series | None,
) -> dict[str, object]:
    """One CSV row. Everything is computed on the post-swap, post-slippage series."""
    acc = ret_net.vbt.returns(freq="1D", year_freq=pd.Timedelta(days=ANN_DAYS))
    years = len(ret_net) / ANN_DAYS
    sharpe = float(acc.sharpe_ratio())
    n_trades = int(len(trades))
    per_year = n_trades / years if years else float("nan")

    if gold_ret is None:
        corr = 1.0 if symbol == GOLD else float("nan")
    else:
        # Aligned on the calendar date, not on the timestamp: a session carries
        # the wall clock of its source (midnight for a resampled minute export,
        # 09:30 New York for a long daily one), so an exact-timestamp join
        # returns an empty intersection between two perfectly comparable series.
        joined = pd.concat(
            [_by_date(ret_net), _by_date(gold_ret)], axis=1, join="inner"
        ).dropna()
        corr = float(joined.iloc[:, 0].corr(joined.iloc[:, 1])) if len(joined) > 2 else float("nan")

    if sharpe < 0:
        verdict = "KILL_NEG_EDGE"
    elif per_year < MIN_TRADES_PER_YEAR:
        verdict = "KILL_LOW_TRADES"
    else:
        verdict = "PASS"

    return dict(
        symbol=symbol,
        source=source,
        n_bars=len(ret_net),
        first_date=str(ret_net.index.min().date()),
        last_date=str(ret_net.index.max().date()),
        sharpe_net=sharpe,
        cagr_net=float(acc.annualized()),
        maxdd=float(acc.max_drawdown()),
        trades_total=n_trades,
        trades_per_year=per_year,
        avg_price_ret_per_trade=avg_price_return(trades),
        exposure_median=float(exposure.median()),
        years_in_window=years,
        corr_to_gold=corr,
        verdict=verdict,
    )


def invalid_row(symbol: str, source: str, reason: str) -> dict[str, object]:
    """Row for an instrument whose data does not load: no metric, no verdict.

    Written rather than dropped — a screen that silently loses an instrument
    reads as if it had been tested and rejected.
    """
    print(f"  !! {symbol}: {reason}")
    row = {c: float("nan") for c in CSV_COLUMNS}
    row.update(symbol=symbol, source=source, verdict="DATA_INVALID")
    return row


def selfcheck() -> int:
    """Reproduce the parity oracle. Returns a shell exit code."""
    pf, _ = tsmom.pipeline(ORACLE["symbol"], fill="next_open", slippage=ORACLE["slippage"])
    ret = pf.returns
    exposure = (pf.asset_value / pf.value).reindex(ret.index).fillna(0.0).abs()
    window = (ret.index >= ORACLE["start"]) & (ret.index <= ORACLE["end"])
    raw = ret[window]
    net = (ret - SWAP_BPS_PER_NIGHT * exposure)[window]

    def sharpe(series: pd.Series, days: int) -> float:
        acc = series.vbt.returns(freq="1D", year_freq=pd.Timedelta(days=days))
        return float(acc.sharpe_ratio())

    got_raw, got_net = sharpe(raw, 365), sharpe(net, 365)
    print(f"\nORACLE — {ORACLE['symbol']}, next_open, slippage "
          f"{ORACLE['slippage'] * 1e4:.0f} bp, {ORACLE['start']} -> {ORACLE['end']}")
    print(f"  sessions            : {len(raw)}")
    print(f"  sharpe raw  (365 d) : {got_raw:.4f}   expected {ORACLE['sharpe_raw']:.3f}")
    print(f"  sharpe net  (365 d) : {got_net:.4f}   expected {ORACLE['sharpe_net']:.3f}")
    print(f"  exposure median     : {float(exposure[window].median()):.4f}")
    print(f"  swap drag mean      : {float(SWAP_BPS_PER_NIGHT * exposure[window].mean()) * ANN_DAYS:.4%}/yr")
    print(f"  same, {ANN_DAYS} d convention (what the CSV reports): "
          f"raw {sharpe(raw, ANN_DAYS):.4f}, net {sharpe(net, ANN_DAYS):.4f}")

    tol = ORACLE["tol"]
    ok = abs(got_raw - ORACLE["sharpe_raw"]) <= tol and abs(got_net - ORACLE["sharpe_net"]) <= tol
    print(f"\n  {'OK' if ok else 'FAIL'} — tolerance +/-{tol}\n")
    return 0 if ok else 1


def format_table(df: pd.DataFrame) -> str:
    """Screen table, ranked by net Sharpe."""
    shown = df.copy()
    for col, fmt in (
        ("sharpe_net", "{:.2f}"), ("cagr_net", "{:.1%}"), ("maxdd", "{:.1%}"),
        ("trades_per_year", "{:.1f}"), ("avg_price_ret_per_trade", "{:.2%}"),
        ("exposure_median", "{:.2f}"), ("years_in_window", "{:.1f}"),
        ("corr_to_gold", "{:.2f}"),
    ):
        shown[col] = shown[col].map(lambda v, f=fmt: "" if pd.isna(v) else f.format(v))
    # `.map` on a nullable Int64 hands the callable a float, hence the int().
    for col in ("n_bars", "trades_total"):
        shown[col] = shown[col].map(lambda v: "" if pd.isna(v) else str(int(v)))
    for col in ("first_date", "last_date"):
        shown[col] = shown[col].map(lambda v: "" if pd.isna(v) else str(v))
    return shown.to_string(index=False)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selfcheck", action="store_true", help="reproduce the parity oracle and exit")
    ap.add_argument("--only", metavar="SYMBOL", help="run a single instrument")
    args = ap.parse_args()

    assert_manifest_fresh()

    if args.selfcheck:
        return selfcheck()

    universe = build_universe()
    if args.only:
        universe = [(sym, loader) for sym, loader in universe if sym == args.only]
        if not universe:
            raise SystemExit(f"--only {args.only}: not in the universe")

    print(f"\n{'=' * 92}")
    print("  TSMOM universe screen — vbt pre-filter, one config (production gold defaults)")
    print(f"  {len(universe)} instruments, selection window <= {SELECTION_END.date()} "
          f"(holdout {holdout.HOLDOUT_START.date()})")
    print(f"  next-open fill, per-symbol half spread, swap {SWAP_BPS_PER_NIGHT * 1e4:.1f} bp/night, "
          f"Sharpe annualized on {ANN_DAYS} sessions")
    print("  final ranking belongs to the MT5 tester — this only kills the hopeless")
    print(f"{'=' * 92}\n")

    trials.log_trials(
        "tsmom_universe", len(universe),
        "passe 1 pré-filtre vbt, config unique défauts or",
        config_key="tsmom_universe:21instr:gold_prod_defaults:selection_end_2025-12-31",
    )

    gold_ret: pd.Series | None = None
    rows: list[dict[str, object]] = []
    for symbol, loader in universe:
        print(f"  {symbol:<9} [{loader}]", flush=True)
        try:
            ret_net, exposure, trades = run_instrument(symbol, loader)
        except ValueError as exc:
            rows.append(invalid_row(symbol, loader, str(exc)))
            continue
        rows.append(metrics(symbol, loader, ret_net, exposure, trades, gold_ret))
        if symbol == GOLD:
            gold_ret = ret_net.rename(GOLD)

    df = pd.DataFrame(rows, columns=list(CSV_COLUMNS))
    # Nullable ints: a failed instrument has no count, and "9067.0 sessions" is
    # not a thing.
    df = df.astype({"n_bars": "Int64", "trades_total": "Int64"})
    df = df.sort_values("sharpe_net", ascending=False, na_position="last")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"\n{format_table(df)}\n")
    print(df["verdict"].value_counts().to_string())
    print(f"\nWritten: {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
