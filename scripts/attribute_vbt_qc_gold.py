#!/usr/bin/env python3
"""Attribute the vbt <-> QuantConnect gap on the gold sleeve, one convention at a time.

The two engines eat the same bytes: `data/XAU-USD_minute_qc.parquet` was exported
from QuantConnect. Every remaining difference is therefore engine semantics, and
is exactly solvable — which is what makes attribution, rather than tolerance,
the right goal here.

The QC trace itself cannot be pulled through the API (object-store export is
gated to Institutional accounts, and the MCP server exposes no backtest-log
endpoint), so this script does not diff two traces. It does something stronger:
it starts from the vbt sleeve and applies the QC conventions **one at a time**,
measuring what each one is worth. If the published QC metrics are recovered at
the end, the attribution is complete and every step is named.

The five conventions, each isolated:

    1. bar boundary     midnight NY          -> 17:00 NY
    2. sigma returns    arithmetic           -> log
    3. sigma floor      0.01                 -> 0.05
    4. causal shift     leverage lagged 1 bar-> not lagged
    5. fill timing      signal-bar close     -> T+1 open

Usage:
    python scripts/attribute_vbt_qc_gold.py
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import vectorbtpro as vbt  # noqa: E402
from utils import apply_vbt_settings, load_gold_data  # noqa: E402

LOOKBACKS = (40, 60, 120, 250)
VOL_WINDOW = 21
TARGET_VOL = 0.25
MAX_LEVERAGE = 3.0

# Passed explicitly rather than inherited from vbt.yml. That file is applied as
# a side effect of importing `framework`, which this script does not do; relying
# on it silently cost 0.36 pp of CAGR in an earlier run — the replica was
# trading with zero slippage while the production pipeline pays 1 bp.
INIT_CASH = 1_000_000
SLIPPAGE = 0.0001
FEES = 0.0

# Published QC run, project 34489845 (LEAN 2.5.0.0.17941), same window.
QC_REFERENCE = dict(cagr=20.17, vol=23.30, sharpe=0.575, maxdd=51.90)


@dataclass(frozen=True)
class Convention:
    """One engine-semantics switch. Defaults are the current vbt behaviour."""

    boundary: str = "midnight"   # "midnight" | "1700"
    sigma_returns: str = "arith"  # "arith" | "log"
    vol_floor: float = 0.01
    causal_shift: bool = True
    fill: str = "close"          # "close" | "nextopen"


def daily_bars(close_min: pd.Series, open_min: pd.Series, boundary: str):
    """Aggregate minute bars into daily ones under the requested convention."""
    if boundary == "midnight":
        close = close_min.vbt.resample_apply("1D", "last").dropna()
        opens = open_min.vbt.resample_apply("1D", "first").dropna()
        return close, opens

    # `origin` is silently ignored for a non-tick-like freq such as "1D",
    # so the 17:00 boundary has to be expressed as a 24h window.
    origin = pd.Timestamp("2019-01-01 17:00")
    close = close_min.resample("24h", origin=origin).last().dropna()
    opens = open_min.resample("24h", origin=origin).first().dropna()
    # A bar closing at 17:00 on day J covers J-1 17:00 -> J 17:00; QC dates it J.
    for series in (close, opens):
        series.index = (series.index + pd.Timedelta(hours=24)).normalize()
    return (close.groupby(close.index).last(),
            opens.groupby(opens.index).first())


def momentum_score(close: pd.Series) -> pd.Series:
    votes = [np.sign(close.vbt.pct_change(n)) for n in LOOKBACKS]
    return sum(votes) / float(len(LOOKBACKS))


def leverage_series(close: pd.Series, conv: Convention) -> pd.Series:
    if conv.sigma_returns == "log":
        rets = np.log(close / close.shift(1))
    else:
        rets = close.vbt.pct_change()
    sigma = rets.vbt.rolling_std(VOL_WINDOW, minp=VOL_WINDOW, ddof=1) * np.sqrt(252)
    lev = (TARGET_VOL / sigma.clip(lower=conv.vol_floor)).clip(upper=MAX_LEVERAGE)
    if conv.causal_shift:
        lev = lev.shift(1)
    return lev.fillna(1.0)


def run(close_min: pd.Series, open_min: pd.Series, conv: Convention):
    close, opens = daily_bars(close_min, open_min, conv.boundary)
    score = momentum_score(close)
    long_ok = score > 0.0
    prev = long_ok.vbt.fshift(1, fill_value=False)
    entries, exits = long_ok & ~prev, ~long_ok & prev

    lev = leverage_series(close, conv)

    price = None
    if conv.fill == "nextopen":
        # Decide on bar t, execute at the open of t+1: shift the signals rather
        # than the price, so the whole bar's accounting moves with the fill.
        entries = entries.vbt.fshift(1, fill_value=False)
        exits = exits.vbt.fshift(1, fill_value=False)
        price = opens.reindex(close.index).ffill()

    kwargs = dict(
        close=close, entries=entries, exits=exits,
        size=1.0, size_type="percent", leverage=lev.to_numpy(), freq="1D",
        init_cash=INIT_CASH, slippage=SLIPPAGE, fees=FEES,
    )
    if price is not None:
        kwargs["price"] = price
    return vbt.Portfolio.from_signals(**kwargs)


def measure(pf) -> dict:
    stats = pf.stats()
    years = (pf.wrapper.index[-1] - pf.wrapper.index[0]).days / 365.25
    total = float(stats["Total Return [%]"]) / 100.0
    return dict(
        cagr=((1 + total) ** (1 / years) - 1) * 100,
        vol=float(pf.returns.std() * np.sqrt(252)) * 100,
        sharpe=float(stats["Sharpe Ratio"]),
        maxdd=float(stats["Max Drawdown [%]"]),
        trades=int(float(stats["Total Trades"])),
    )


def main() -> int:
    apply_vbt_settings()
    _, data = load_gold_data()
    close_min, open_min = data.close, data.open

    # Cumulative ladder: each step keeps the previous ones, so the last row is
    # "vbt with every QC convention applied" and should land on the QC figures.
    steps: list[tuple[str, Convention]] = [
        ("vbt tel quel", Convention()),
        ("+ bornes 17:00", Convention(boundary="1700")),
        ("+ sigma log", Convention(boundary="1700", sigma_returns="log")),
        ("+ plancher 0.05",
         Convention(boundary="1700", sigma_returns="log", vol_floor=0.05)),
        ("+ sans décalage causal",
         Convention(boundary="1700", sigma_returns="log", vol_floor=0.05,
                    causal_shift=False)),
        ("+ fill au T+1 open",
         Convention(boundary="1700", sigma_returns="log", vol_floor=0.05,
                    causal_shift=False, fill="nextopen")),
    ]

    print(f"{'étape':<26} {'CAGR':>8} {'vol':>8} {'Sharpe':>8} {'maxDD':>8} {'trades':>7}"
          f"   {'ΔCAGR':>8} {'ΔmaxDD':>8}")
    print("-" * 96)

    previous = None
    rows = []
    for label, conv in steps:
        m = measure(run(close_min, open_min, conv))
        d_cagr = "" if previous is None else f"{m['cagr'] - previous['cagr']:+8.2f}"
        d_dd = "" if previous is None else f"{m['maxdd'] - previous['maxdd']:+8.2f}"
        print(f"{label:<26} {m['cagr']:>7.2f}% {m['vol']:>7.2f}% {m['sharpe']:>8.3f} "
              f"{m['maxdd']:>7.2f}% {m['trades']:>7}   {d_cagr:>8} {d_dd:>8}")
        rows.append((label, m))
        previous = m

    print("-" * 96)
    q = QC_REFERENCE
    print(f"{'QC (référence publiée)':<26} {q['cagr']:>7.2f}% {q['vol']:>7.2f}% "
          f"{q['sharpe']:>8.3f} {q['maxdd']:>7.2f}% {'128 ord':>7}")

    final = rows[-1][1]
    print(f"\n{'résidu après attribution':<26} "
          f"{final['cagr'] - q['cagr']:>+7.2f}pp {final['vol'] - q['vol']:>+7.2f}pp "
          f"{final['sharpe'] - q['sharpe']:>+8.3f} {final['maxdd'] - q['maxdd']:>+7.2f}pp")
    return 0


if __name__ == "__main__":
    sys.exit(main())
