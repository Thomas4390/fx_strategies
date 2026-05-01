#!/usr/bin/env python3
"""fx_macro_history — fetch historical FRED data and emit multi-row CSV for MT5 backtest.

Fetches T10Y2Y (daily Treasury 10Y-2Y spread) and UNRATE (monthly US unemployment
rate) from the FRED API for the requested window, computes the 2-stage macro
filter for each daily spread observation, and emits a multi-row CSV consumed by
FxMultiSleeve in Strategy Tester (mode MACRO_SOURCE_HISTORY / AUTO).

Counterpart of fx_macro_bridge.py — same CSV schema but multiple rows for
historical lookup. The live bridge writes 1 row to macro_cache.csv; this script
writes N rows to macro_history.csv.

Usage:
    python fx_macro_history.py                            # 2019-01-01 -> today
    python fx_macro_history.py --start 2019-01-01 --end 2026-04-30
    python fx_macro_history.py --threshold 0.5
    python fx_macro_history.py --output /tmp/macro_history.csv

Reads FRED_API_KEY from <repo-root>/.env (gitignored). Side-effect outputs:
    1. <repo>/data/SPREAD_10Y2Y_daily.parquet      (raw FRED data, ~daily)
    2. <repo>/data/UNEMPLOYMENT_monthly.parquet    (raw FRED data, monthly)
And the primary output:
    3. <MT5 Common>/Files/macro_history.csv        (multi-row, ASCII, for MT5)
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
DATA_DIR = REPO_ROOT / "data"
ENV_FILE = REPO_ROOT / ".env"

FRED_BASE = "https://api.stlouisfed.org/fred/series/observations"
DEFAULT_START = "2019-01-01"
DEFAULT_THRESHOLD = 0.5
OUTPUT_CSV_NAME = "macro_history.csv"


def read_env_var(name: str) -> str:
    if not ENV_FILE.exists():
        raise RuntimeError(f"{ENV_FILE} not found — create it with FRED_API_KEY=…")
    for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith(name + "="):
            return line[len(name) + 1:].strip().strip('"').strip("'")
    raise RuntimeError(f"{name} not found in {ENV_FILE}")


def fetch_fred_series(series_id: str, api_key: str,
                      start: str, end: str) -> pd.Series:
    qs = urllib.parse.urlencode({
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "observation_start": start,
        "observation_end": end,
    })
    url = f"{FRED_BASE}?{qs}"
    print(f"GET {series_id} from FRED ({start} -> {end})")
    with urllib.request.urlopen(url, timeout=30) as resp:
        if resp.status != 200:
            raise RuntimeError(f"FRED HTTP {resp.status} for {series_id}")
        body = resp.read().decode("utf-8")
    data = json.loads(body)
    obs = data.get("observations", [])
    if not obs:
        raise RuntimeError(f"FRED returned 0 observations for {series_id}")
    rows = []
    for o in obs:
        v = o["value"]
        if v in (".", ""):
            continue  # FRED missing-value sentinel
        rows.append((pd.Timestamp(o["date"]), float(v)))
    if not rows:
        raise RuntimeError(f"FRED returned only missing values for {series_id}")
    s = (pd.DataFrame(rows, columns=["date", series_id])
         .set_index("date").sort_index()[series_id])
    print(f"  {len(s)} observations from {s.index.min().date()} to {s.index.max().date()}")
    return s


def detect_mt5_common_files() -> Path:
    candidates = [
        Path.home() / "AppData/Roaming/MetaQuotes/Terminal/Common/Files",
        Path.home() / ".wine/drive_c/users" / Path.home().name
        / "AppData/Roaming/MetaQuotes/Terminal/Common/Files",
    ]
    for c in candidates:
        if c.exists():
            return c
    fallback = REPO_ROOT / "output"
    fallback.mkdir(exist_ok=True)
    return fallback


def compute_macro_history(spread: pd.Series, unrate: pd.Series,
                          threshold: float) -> pd.DataFrame:
    """For each spread date, compute unemp_rising (3m diff on UNRATE) and macro_ok.

    Matches fx_macro_bridge.py logic exactly:
      - unemp_rising = (UNRATE[latest ≤ d] - UNRATE[3 monthly obs earlier]) > 0
      - macro_ok = (spread < threshold) AND NOT unemp_rising
    """
    if len(unrate) < 4:
        raise RuntimeError(f"UNRATE has only {len(unrate)} observations, need ≥ 4")
    unrate = unrate.sort_index()
    spread = spread.sort_index()

    rows = []
    for d, sp in spread.items():
        idx = unrate.index.searchsorted(d, side="right") - 1
        if idx < 3:
            continue  # need 4 monthly observations
        u_now = unrate.iloc[idx]
        u_3m = unrate.iloc[idx - 3]
        unemp_rising = bool((u_now - u_3m) > 0.0)
        macro_ok = bool((sp < threshold) and not unemp_rising)
        rows.append({
            "timestamp_utc": d.strftime("%Y-%m-%dT00:00:00Z"),
            "spread_10y2y": f"{sp:.6f}",
            "unemp_rising": int(unemp_rising),
            "spread_threshold": f"{threshold:.4f}",
            "macro_ok": int(macro_ok),
        })
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--start", default=DEFAULT_START,
                        help=f"Start date YYYY-MM-DD (default {DEFAULT_START})")
    parser.add_argument("--end", default=None,
                        help="End date YYYY-MM-DD (default today UTC)")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                        help=f"Spread threshold (default {DEFAULT_THRESHOLD})")
    parser.add_argument("--output", type=Path, default=None,
                        help="CSV output path (default <MT5 Common>/macro_history.csv)")
    args = parser.parse_args()

    end = args.end or dt.datetime.now(dt.timezone.utc).date().isoformat()
    unrate_start = (pd.Timestamp(args.start) - pd.DateOffset(months=6)).date().isoformat()

    api_key = read_env_var("FRED_API_KEY")

    spread = fetch_fred_series("T10Y2Y", api_key, args.start, end)
    unrate = fetch_fred_series("UNRATE", api_key, unrate_start, end)

    DATA_DIR.mkdir(exist_ok=True)
    spread.to_frame().to_parquet(DATA_DIR / "SPREAD_10Y2Y_daily.parquet")
    unrate.to_frame().to_parquet(DATA_DIR / "UNEMPLOYMENT_monthly.parquet")
    print(f"Saved parquets to {DATA_DIR}/")

    df = compute_macro_history(spread, unrate, args.threshold)
    if df.empty:
        raise RuntimeError("No rows produced (check input dates)")

    out_dir = args.output.parent if args.output else detect_mt5_common_files()
    out_path = args.output if args.output else (out_dir / OUTPUT_CSV_NAME)
    out_dir.mkdir(parents=True, exist_ok=True)

    header = "timestamp_utc,spread_10y2y,unemp_rising,spread_threshold,macro_ok"
    body = df.to_csv(index=False, header=False, lineterminator="\n")
    out_path.write_text(f"{header}\n{body}", encoding="ascii")
    print(f"OK wrote {out_path}: {len(df)} rows "
          f"from {df['timestamp_utc'].iloc[0]} to {df['timestamp_utc'].iloc[-1]}")
    print(f"  macro_ok=1 fraction: {df['macro_ok'].astype(int).mean() * 100:.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
