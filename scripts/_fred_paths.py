"""Shared paths and small helpers for the FRED pipeline."""
from __future__ import annotations

import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
ENV_FILE = REPO_ROOT / ".env"

EXTERNAL_ROOT = Path("/run/media/thomas/New Volume/Datasets/Macro")
FRED_ROOT = EXTERNAL_ROOT / "fred"
CATALOG_DIR = FRED_ROOT / "catalog"
OBSERVATIONS_DIR = FRED_ROOT / "observations"
STATE_DIR = FRED_ROOT / "_state"
PROGRESS_DB = STATE_DIR / "progress.sqlite"
LATEST_RUN = STATE_DIR / "LATEST_RUN.txt"


_SLUG_RE = re.compile(r"[^A-Za-z0-9]+")


def slugify(text: str) -> str:
    s = _SLUG_RE.sub("_", text or "").strip("_")
    return s or "Other"


def freq_bucket(frequency_short: str | None) -> str:
    if not frequency_short:
        return "other"
    f = frequency_short.upper()
    return {
        "D": "daily",
        "W": "weekly",
        "BW": "weekly",   # bi-weekly bucketed with weekly
        "M": "monthly",
        "SM": "monthly",  # semi-monthly bucketed with monthly
        "Q": "quarterly",
        "SA": "annual",   # semi-annual → annual bucket
        "A": "annual",
    }.get(f, "other")


def observation_path(series_id: str, frequency_short: str | None,
                     category_root: str | None) -> Path:
    """<external>/fred/observations/<freq>/<category_root>/<aa>/<SERIES_ID>.parquet"""
    freq = freq_bucket(frequency_short)
    cat = slugify(category_root or "Other")
    prefix = (series_id[:2] or "_").upper()
    return OBSERVATIONS_DIR / freq / cat / prefix / f"{series_id}.parquet"


def meta_path_for(parquet_path: Path) -> Path:
    return parquet_path.with_suffix(".meta.json")


def external_mounted() -> bool:
    return EXTERNAL_ROOT.exists() and EXTERNAL_ROOT.is_dir()


def read_env_var(name: str) -> str:
    """Mirror of src/mt5/bridge/fx_macro_history.py:read_env_var."""
    if not ENV_FILE.exists():
        raise RuntimeError(f"{ENV_FILE} not found — create it with FRED_API_KEY=…")
    for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith(name + "="):
            return line[len(name) + 1:].strip().strip('"').strip("'")
    raise RuntimeError(f"{name} not found in {ENV_FILE}")
