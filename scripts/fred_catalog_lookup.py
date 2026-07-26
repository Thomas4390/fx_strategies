#!/usr/bin/env python3
"""Quick lookup helper for the FRED catalog.

Usage:
    python scripts/fred_catalog_lookup.py inflation
    python scripts/fred_catalog_lookup.py CPIAUCSL --info
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from _fred_paths import CATALOG_DIR  # noqa: E402

SERIES_PARQUET = CATALOG_DIR / "series.parquet"


def load() -> pd.DataFrame:
    if not SERIES_PARQUET.exists():
        sys.exit(f"catalog missing: {SERIES_PARQUET} — run fred_full_database.py --phase=catalog")
    return pd.read_parquet(SERIES_PARQUET)


def find_series(query: str, top: int = 30) -> pd.DataFrame:
    df = load()
    q = query.lower()
    mask = (df["id"].str.lower().str.contains(q, na=False) |
            df["title"].str.lower().str.contains(q, na=False))
    return df[mask].nlargest(top, "popularity")[
        ["id", "title", "frequency_short", "popularity", "observation_end", "category_root"]]


def series_info(series_id: str) -> pd.Series | None:
    df = load()
    rows = df[df["id"] == series_id]
    if rows.empty:
        return None
    return rows.iloc[0]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("query", help="Series ID or search text")
    p.add_argument("--info", action="store_true",
                   help="Treat query as exact series_id and dump full metadata")
    p.add_argument("--top", type=int, default=30, help="Max results (default 30)")
    args = p.parse_args(argv)

    if args.info:
        s = series_info(args.query)
        if s is None:
            print(f"{args.query} not found"); return 1
        for k, v in s.items():
            print(f"{k:<28} {v}")
        return 0

    df = find_series(args.query, top=args.top)
    if df.empty:
        print(f"no matches for '{args.query}'"); return 1
    pd.set_option("display.max_colwidth", 80)
    print(df.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
