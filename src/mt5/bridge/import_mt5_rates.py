#!/usr/bin/env python3
"""import_mt5_rates — convert CSVs exported by FxExportRates.mq5 to Parquet.

The MQL5 script `Scripts/FxExportRates.mq5` writes one CSV per
(symbol, timeframe) into the MT5 portable data folder under
``MQL5/Files/exports/<SYMBOL>_<TF>.csv``. This script reads them and
produces tidy Parquet files in ``data/`` named like
``EUR-USD_minute_mt5.parquet``, matching the pre-existing convention
(``EUR-USD_minute.parquet``) but with a ``_mt5`` suffix that flags the
provenance.

Usage:
    # Default: read from the portable MT5 prefix on this machine
    python import_mt5_rates.py

    # Custom export dir (e.g. user-profile data folder, non-portable):
    python import_mt5_rates.py --src "/home/.../MQL5/Files/exports"

    # Dry-run (don't write parquets, just list what would be produced)
    python import_mt5_rates.py --check
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "data"

DEFAULT_SRC = Path(
    "/home/thomas/.mt5/drive_c/Program Files/MetaTrader 5/MQL5/Files/exports"
)

# Map MT5 timeframe label → file naming suffix used in data/
TF_LABEL_TO_PERIOD = {
    "M1": "minute",
    "M5": "5min",
    "M15": "15min",
    "M30": "30min",
    "H1": "hourly",
    "H4": "4hour",
    "D1": "daily",
    "W1": "weekly",
    "MN1": "monthly",
}


def parse_pair(symbol: str) -> str:
    """Convert broker symbol like ``EURUSD.c`` → ``EUR-USD`` (project convention)."""
    base = symbol.split(".")[0]  # strip suffix
    if len(base) != 6:
        raise ValueError(f"unexpected symbol format: {symbol!r}")
    return f"{base[:3]}-{base[3:]}"


def parse_csv_filename(name: str) -> tuple[str, str]:
    """``EURUSD.c_M1.csv`` → ("EURUSD.c", "M1"). Raises if format unexpected."""
    stem = Path(name).stem
    parts = stem.rsplit("_", 1)
    if len(parts) != 2:
        raise ValueError(f"unexpected CSV filename: {name!r}")
    return parts[0], parts[1]


def import_one(csv_path: Path, dst_dir: Path, check_only: bool) -> Path | None:
    symbol, tf_label = parse_csv_filename(csv_path.name)
    pair = parse_pair(symbol)
    period = TF_LABEL_TO_PERIOD.get(tf_label)
    if period is None:
        print(f"SKIP {csv_path.name}: unknown timeframe {tf_label!r}")
        return None

    df = pd.read_csv(
        csv_path,
        dtype={
            "open": "float64", "high": "float64",
            "low": "float64", "close": "float64",
            "tick_volume": "int64", "spread": "int64", "real_volume": "int64",
        },
        parse_dates=["time"],
    )
    if df.empty:
        print(f"SKIP {csv_path.name}: empty CSV")
        return None
    df = df.set_index("time").sort_index()

    out_name = f"{pair}_{period}_mt5.parquet"
    out_path = dst_dir / out_name
    bars = len(df)
    span = f"{df.index.min().isoformat()} → {df.index.max().isoformat()}"
    arrow = "(check)" if check_only else "→"
    print(f"OK   {csv_path.name}: {bars:>8} bars, {span}  {arrow} {out_path.name}")

    if check_only:
        return None
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, engine="pyarrow", compression="snappy")
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", type=Path, default=DEFAULT_SRC,
                        help="MT5 export dir (default: portable MQL5\\Files\\exports)")
    parser.add_argument("--dst", type=Path, default=DATA_DIR,
                        help="Output dir for parquets (default: <repo>/data)")
    parser.add_argument("--check", action="store_true",
                        help="Dry-run: list what would be written")
    args = parser.parse_args()

    if not args.src.exists():
        print(f"ERROR: source dir not found: {args.src}", file=sys.stderr)
        print(f"  Run the MQL5 script `Scripts/FxExportRates` from MT5 first, "
              f"or pass --src to point to your export dir.", file=sys.stderr)
        return 1

    csvs = sorted(args.src.glob("*.csv"))
    if not csvs:
        print(f"No CSVs in {args.src}")
        return 0

    print(f"Source : {args.src}")
    print(f"Target : {args.dst}")
    print(f"Found  : {len(csvs)} CSV file(s)\n")

    n_written = 0
    for csv_path in csvs:
        try:
            written = import_one(csv_path, args.dst, args.check)
            if written is not None:
                n_written += 1
        except Exception as exc:  # noqa: BLE001
            print(f"FAIL {csv_path.name}: {exc}", file=sys.stderr)

    suffix = "would be written" if args.check else "written"
    print(f"\n=== {n_written} parquet(s) {suffix} ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
