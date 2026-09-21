#!/usr/bin/env python3
"""Build the four-leg synthetic dollar basket of the x10 spec (§6.3).

    DXY4 := 50.14348112 * EURUSD^-w_e * USDJPY^w_j * GBPUSD^-w_g * USDCAD^w_c

with the ICE weights renormalised to 1. Two outputs, because two consumers:

* ``data/DXY4_h1.parquet``     — the context series of §6.1, aligned to M5 by
  ``x10_context.align_h1_to_m5`` (shift then ffill);
* ``data/DXY4_minute.parquet`` — the M1 resolution grid, kept for the order
  resolution of §9 and for the MT5 side, which reads minute bars.

The basket is rebuilt at each frequency rather than downsampled from the minute
series: "a leg absent for more than 5 bars" is a statement about the bars of the
frequency being built, and 5 minutes of staleness is not 5 hours of it.

Why a synthetic basket at all: no DXY future is available in the same feed as
the gold minute bars, and a daily broad index (DTWEXBGS) cannot filter an M5
decision. ``--check`` measures how far the synthetic drifts from that daily
index; the answer is reported, never tuned — DTWEXBGS is a 26-currency trade
weighted index, not the 6-leg ICE dollar, so agreement is expected to be high
but not exact.

Usage
-----
    python scripts/build_dxy_synthetic.py --check    # measure, write nothing
    python scripts/build_dxy_synthetic.py            # write both parquets

``data/`` is gitignored: these files are build artefacts, rebuilt from the FX
minute parquets, never committed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SRC = _PROJECT_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from framework.x10_context import dxy4, resample_ohlc  # noqa: E402
from utils import load_fx_data, validate_ohlc_frame  # noqa: E402

# Leg -> source parquet. The four files share the naive New York minute index of
# the FX exports (weekly reopen at 17:00 New York, asserted in
# tests/test_dxy_synthetic.py), which is also the clock `load_gold_data` puts
# the gold bars on. No timezone conversion is needed, and none must be added —
# but a one-minute re-dating is, see `SOURCE_STAMP` below.
LEG_SOURCES: dict[str, str] = {
    "eurusd": "data/EUR-USD_minute.parquet",
    "usdjpy": "data/USD-JPY_minute.parquet",
    "gbpusd": "data/GBP-USD_minute.parquet",
    "usdcad": "data/USD-CAD_minute.parquet",
}

OUT_H1 = "data/DXY4_h1.parquet"
OUT_MINUTE = "data/DXY4_minute.parquet"
DTWEXBGS = "data/DTWEXBGS_daily.parquet"

# The FX minute parquets stamp a bar at its **close**, exactly like the gold
# parquet: measured against the MT5 export of EURUSD, which is open-stamped,
# their minute returns correlate at 0.991 at a +1 minute lag and at 0.07
# everywhere else, over 61 175 overlapping minutes
# (docs/research/xau_x10_reconciliation.md §11.6 bis). The basket must therefore
# be re-dated to the bar open before any resampling, or its H1 bins cover
# [h-1 min, h+59 min) while the M5 grid they feed covers [h, h+5 min) — the very
# defect that held the Python/QC entry matching at 33 %.
#
# This is deliberately local to the x10 chain: `utils.load_fx_data` is shared
# with strategy 1, whose published figures were produced under the old
# convention and must not move under it.
SOURCE_STAMP = "close"
BAR_OPEN_SHIFT = pd.Timedelta(minutes=1)


def to_bar_open(frame: pd.DataFrame, source_stamp: str = SOURCE_STAMP) -> pd.DataFrame:
    """Re-stamp a minute frame by the instant its bar opens.

    Pure, and tested on its own: a row stamped ``10:05`` under the ``close``
    convention describes ``[10:04, 10:05)`` and becomes ``10:04``. Under
    ``open`` the frame is returned untouched.
    """
    if source_stamp == "open":
        return frame
    if source_stamp != "close":
        raise ValueError(f"source_stamp={source_stamp!r} inconnu ; 'close' ou 'open'")
    shifted = frame.copy()
    shifted.index = shifted.index - BAR_OPEN_SHIFT
    return shifted

# FRED fixes DTWEXBGS at noon New York; the H1 bar stamped 11:00 closes there.
NOON_FIXING_HOUR = 11


def load_legs(data_dir: Path) -> dict[str, pd.DataFrame]:
    """Load and validate the four minute OHLC frames."""
    legs: dict[str, pd.DataFrame] = {}
    for leg, rel in LEG_SOURCES.items():
        path = data_dir / Path(rel).name
        if not path.exists():
            raise FileNotFoundError(f"{leg}: missing source {path}")
        raw, _ = load_fx_data(str(path))
        validate_ohlc_frame(raw, name=leg.upper())
        # Validation d'abord, redatation ensuite : `validate_ohlc_frame` parle
        # des barres, pas de leur étiquette, et la preuve doit porter sur le
        # fichier tel qu'il est écrit.
        raw = to_bar_open(raw)
        legs[leg] = raw
        span = f"{raw.index[0]} -> {raw.index[-1]}"
        print(f"  {leg.upper():7s} {len(raw):>9,} M1 bars   {span}")
    return legs


def build_basket(legs: dict[str, pd.DataFrame], freq: str | None) -> pd.Series:
    """DXY4 at M1 (``freq=None``) or at the given resampling frequency."""
    if freq is None:
        closes = {leg: df["close"] for leg, df in legs.items()}
    else:
        closes = {leg: resample_ohlc(df, freq)["close"] for leg, df in legs.items()}
    return dxy4(**closes)


def _coverage(name: str, series: pd.Series) -> None:
    n = len(series)
    n_nan = int(series.isna().sum())
    print(
        f"  {name:<12s} {n:>9,} bars   {series.index[0]} -> {series.index[-1]}   "
        f"NaN {n_nan:>7,} ({n_nan / n:.4%})"
    )
    # Level smell test: the ICE constant should land the basket in the 90-120 band.
    print(
        f"  {'':<12s} level min {series.min():.2f}  median {series.median():.2f}  "
        f"max {series.max():.2f}"
    )


def daily_correlation(dxy_h1: pd.Series, dtwex_path: Path) -> None:
    """Correlation of daily log returns against the FRED broad dollar index.

    Two sampling points are reported because the comparison is timing sensitive
    and the honest one is not the obvious one. DTWEXBGS is a **noon New York
    fixing**, so the H1 bar stamped 11:00 — which closes at 12:00 — is the
    apples-to-apples sample; a calendar-day close compares an 11-hour offset
    window and loses correlation for a purely clock-related reason. Both are
    printed so the gap is visible rather than chosen.

    Levels are inner-joined *before* differencing so that both series step over
    the same weekends and holidays; differencing first would compare a Friday-to-
    Monday move with a Friday-to-Tuesday one, another calendar artefact.

    The remaining gap to 1 is structural and must not be tuned away: DTWEXBGS is
    a 26-currency trade-weighted index — CNY and MXN included — while DXY4 has
    four legs. It is a sanity check on sign and scale, not a benchmark.
    """
    if not dtwex_path.exists():
        print(f"  DTWEXBGS absent ({dtwex_path}) — correlation not measured")
        return

    ref = pd.read_parquet(dtwex_path).set_index("date")["dtwexbgs"].sort_index()
    noon = dxy_h1[dxy_h1.index.hour == NOON_FIXING_HOUR]
    samples = {
        "noon NY fixing  ": pd.Series(noon.to_numpy(), index=noon.index.normalize()),
        "calendar-day end": dxy_h1.resample("1D").last().dropna(),
    }
    for label, daily in samples.items():
        joined = pd.concat({"dxy4": daily, "dtwexbgs": ref}, axis=1, join="inner").dropna()
        returns = np.log(joined).diff().dropna()
        corr = float(returns["dxy4"].corr(returns["dtwexbgs"]))
        print(
            f"  DTWEXBGS daily log-return corr [{label}]: {corr:.4f} "
            f"on {len(returns):,} days "
            f"({returns.index[0].date()} -> {returns.index[-1].date()})"
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=_PROJECT_ROOT / "data",
        help="Directory holding the FX minute parquets (default: <repo>/data)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Dry-run: report coverage, NaN share and DTWEXBGS correlation",
    )
    args = parser.parse_args()

    print("Legs:")
    legs = load_legs(args.data_dir)

    minute = build_basket(legs, None)
    h1 = build_basket(legs, "1h")

    print("Basket:")
    _coverage("DXY4 M1", minute)
    _coverage("DXY4 H1", h1)
    daily_correlation(h1, args.data_dir / Path(DTWEXBGS).name)

    if args.check:
        print("(check) nothing written")
        return 0

    for rel, series in ((OUT_MINUTE, minute), (OUT_H1, h1)):
        out = args.data_dir / Path(rel).name
        frame = series.rename("close").to_frame()
        frame.index.name = "date"
        frame.reset_index().to_parquet(out, engine="pyarrow", compression="snappy")
        print(f"  wrote {out} ({len(frame):,} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
