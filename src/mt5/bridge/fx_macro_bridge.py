#!/usr/bin/env python3
"""fx_macro_bridge — relay des données macro pour l'EA MT5.

Lit les parquets `data/SPREAD_10Y2Y_daily.parquet` et
`data/UNEMPLOYMENT_monthly.parquet`, calcule le filtre macro à 2 étages utilisé
par le sleeve MR Macro (cf. `src/strategies/mr_macro.py:load_macro_filters`),
et écrit le résultat dans le dossier `MQL5/Files/` du terminal MT5 sous forme
d'un CSV mono-ligne.

Schéma du CSV produit :

    timestamp_utc,spread_10y2y,unemp_rising,spread_threshold,macro_ok
    2026-04-24T18:00:00Z,0.3520,0,0.50,1

Convention :
- `unemp_rising` = 1 si `unemployment[-1] - unemployment[-4] > 0` (variation 3m)
- `macro_ok` = 1 si `(spread < threshold) AND (NOT unemp_rising)`

Usage :
    python fx_macro_bridge.py                  # défaut, écrit dans Common/Files
    python fx_macro_bridge.py --output PATH    # override destination
    python fx_macro_bridge.py --threshold 0.5  # seuil spread

Schedule cron horaire recommandé :
    0 * * * * /usr/bin/python3 /path/to/fx_macro_bridge.py
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "data"
DEFAULT_OUTPUT_NAMES = ("macro_cache.csv",)


def detect_mt5_files_dir() -> Path:
    """Détecte le dossier d'écriture par défaut (Common/Files MT5).

    Cherche le chemin standard Wine/Linux d'abord, puis Windows.
    Si rien trouvé, écrit dans le repo (`output/macro_cache.csv`) pour debug.
    """
    candidates = [
        Path.home() / ".wine/drive_c/users" / Path.home().name
        / "AppData/Roaming/MetaQuotes/Terminal/Common/Files",
        Path.home() / "AppData/Roaming/MetaQuotes/Terminal/Common/Files",
    ]
    for c in candidates:
        if c.exists():
            return c
    fallback = REPO_ROOT / "output"
    fallback.mkdir(exist_ok=True)
    return fallback


def load_spread_last() -> float:
    path = DATA_DIR / "SPREAD_10Y2Y_daily.parquet"
    df = pd.read_parquet(path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
    s = df.iloc[:, 0].sort_index()
    return float(s.iloc[-1])


def load_unemployment_3m_diff() -> float:
    path = DATA_DIR / "UNEMPLOYMENT_monthly.parquet"
    df = pd.read_parquet(path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
    s = df.iloc[:, 0].sort_index()
    if len(s) < 4:
        raise RuntimeError(f"Unemployment series too short ({len(s)} obs)")
    return float(s.iloc[-1] - s.iloc[-4])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None,
                        help="Destination CSV (défaut : Common/Files MT5)")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Seuil spread 10Y-2Y (défaut 0.5)")
    args = parser.parse_args()

    out_dir = args.output.parent if args.output else detect_mt5_files_dir()
    out_path = args.output if args.output else (out_dir / DEFAULT_OUTPUT_NAMES[0])
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        spread = load_spread_last()
        diff_3m = load_unemployment_3m_diff()
    except Exception as exc:
        print(f"ERROR loading macro data: {exc}", file=sys.stderr)
        return 1

    unemp_rising = bool(diff_3m > 0.0)
    macro_ok = (spread < args.threshold) and (not unemp_rising)

    now_utc = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    header = "timestamp_utc,spread_10y2y,unemp_rising,spread_threshold,macro_ok"
    line = (
        f"{now_utc},{spread:.6f},{int(unemp_rising)},"
        f"{args.threshold:.4f},{int(macro_ok)}"
    )
    content = f"{header}\n{line}\n"
    out_path.write_text(content, encoding="ascii")
    print(f"OK wrote {out_path}: {line}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
