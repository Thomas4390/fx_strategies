"""Disk-backed cache for the expensive ``get_strategy_daily_returns``.

The ``combined_portfolio.get_strategy_daily_returns`` helper runs four
backtests and reloads the full EUR-USD minute history on every call —
~60-90 seconds per sweep script. Phase 22 replaces each call with a
cached lookup :

1. Read ``data/MANIFEST.json`` → ``combined_fingerprint``.
2. Hash the fingerprint with a module-level ``_SLEEVES_VERSION`` to
   get the cache key.
3. If ``data/cache/strategy_daily_returns.parquet`` exists AND its
   side-car ``.json`` key file matches the cache key, load the parquet
   and return the columns as a dict of Series.
4. Otherwise rebuild via the fallback callable passed by the caller
   and write both files.

Design notes
------------
- Cache format is **parquet** (one file, wide columns). JSON side-car
  is separate so the key file can be inspected with ``cat``.
- No serialized Python objects — everything is parquet + JSON.
- Invalidation triggers: any change in the source data (manifest SHA
  changes) OR a bump in ``_SLEEVES_VERSION`` (used when the default
  backtest parameters of one of the sleeves changes).
- Thread safety : single-writer assumption (sweeps run sequentially).
  Concurrent reads are fine because pandas opens the parquet read-only.
- Reproducibility : the cache content is bit-identical to the fresh
  call because it uses default parameters end-to-end — see the
  ``test_cache_reproducibility`` test.

Bumping ``_SLEEVES_VERSION``
---------------------------
Whenever the default parameters of any sleeve change (e.g.
``backtest_mr_macro`` gets a new macro filter, or
``backtest_rsi_daily_portfolio`` switches to a different pair set),
bump ``_SLEEVES_VERSION`` here to invalidate the cache everywhere.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Callable

import pandas as pd


_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_MANIFEST_PATH = _PROJECT_ROOT / "data" / "MANIFEST.json"
_CACHE_DIR = _PROJECT_ROOT / "data" / "cache"
_CACHE_PARQUET = _CACHE_DIR / "strategy_daily_returns.parquet"
_CACHE_KEY_JSON = _CACHE_DIR / "strategy_daily_returns.key.json"

# Bump whenever the default parameters of a sleeve change. Current
# state : Phase 18 canonical trio + legacy 4-pair TS + XS momentum.
_SLEEVES_VERSION = "v1-phase18-canonical"


# ═══════════════════════════════════════════════════════════════════════
# Manifest fingerprint
# ═══════════════════════════════════════════════════════════════════════


class ManifestMissingError(RuntimeError):
    """Raised when ``data/MANIFEST.json`` is absent."""


def _read_manifest_fingerprint() -> str:
    if not _MANIFEST_PATH.exists():
        raise ManifestMissingError(
            f"Missing {_MANIFEST_PATH}. "
            "Run `python scripts/update_data_manifest.py` first."
        )
    with _MANIFEST_PATH.open("r") as fh:
        manifest = json.load(fh)
    fp = manifest.get("combined_fingerprint")
    if not fp:
        raise RuntimeError(
            f"Manifest at {_MANIFEST_PATH} has no combined_fingerprint. "
            "Regenerate with `python scripts/update_data_manifest.py`."
        )
    return str(fp)


def _cache_key(manifest_fingerprint: str, sleeves_version: str) -> str:
    """Combine manifest + sleeves version into a single short hash."""
    h = hashlib.sha256()
    h.update(f"{sleeves_version}|{manifest_fingerprint}".encode("utf-8"))
    return h.hexdigest()


# ═══════════════════════════════════════════════════════════════════════
# Cache read / write
# ═══════════════════════════════════════════════════════════════════════


def _load_key_file() -> dict[str, str] | None:
    if not _CACHE_KEY_JSON.exists():
        return None
    try:
        with _CACHE_KEY_JSON.open("r") as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError):
        return None


def _write_key_file(key: str, fingerprint: str, n_sleeves: int) -> None:
    _CACHE_KEY_JSON.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "cache_key": key,
        "sleeves_version": _SLEEVES_VERSION,
        "manifest_fingerprint": fingerprint,
        "n_sleeves": n_sleeves,
        "generated_at": datetime.now(UTC).isoformat(),
    }
    with _CACHE_KEY_JSON.open("w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=False)
        fh.write("\n")


def _load_cached(expected_key: str) -> dict[str, pd.Series] | None:
    """Return the cached sleeves dict if the key file matches."""
    key_payload = _load_key_file()
    if key_payload is None or key_payload.get("cache_key") != expected_key:
        return None
    if not _CACHE_PARQUET.exists():
        return None
    try:
        df = pd.read_parquet(_CACHE_PARQUET)
    except Exception:
        return None
    return {col: df[col].copy() for col in df.columns}


def _save_cached(sleeves: dict[str, pd.Series], key: str, fingerprint: str) -> None:
    """Materialize the sleeves dict to parquet + key file."""
    _CACHE_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(sleeves)
    # Reset index to a single datetime column for parquet portability,
    # then restore on read via pd.read_parquet (default index).
    df.to_parquet(_CACHE_PARQUET)
    _write_key_file(key, fingerprint, n_sleeves=len(sleeves))


# ═══════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════


def cached_strategy_daily_returns(
    rebuild: Callable[[], dict[str, pd.Series]],
    verbose: bool = True,
) -> dict[str, pd.Series]:
    """Return the sleeves dict, using the disk cache when valid.

    Parameters
    ----------
    rebuild : callable
        Zero-argument function that rebuilds the sleeves dict from
        scratch (typically ``_compute_strategy_daily_returns``). Only
        called on a cache miss.
    verbose : bool
        Print cache hit/miss status.

    Returns
    -------
    dict[str, pd.Series]
        Same shape as ``combined_portfolio.get_strategy_daily_returns``.
    """
    try:
        fingerprint = _read_manifest_fingerprint()
    except ManifestMissingError:
        if verbose:
            print("  [cache] MANIFEST.json missing → rebuilding (no cache)")
        return rebuild()

    key = _cache_key(fingerprint, _SLEEVES_VERSION)
    cached = _load_cached(key)
    if cached is not None:
        if verbose:
            print(
                f"  [cache] hit  key={key[:10]}…  "
                f"({len(cached)} sleeves, "
                f"manifest={fingerprint[:10]}…, "
                f"sleeves_version={_SLEEVES_VERSION})"
            )
        return cached

    if verbose:
        print(f"  [cache] miss key={key[:10]}… — rebuilding via {rebuild.__name__}")
    fresh = rebuild()
    try:
        _save_cached(fresh, key, fingerprint)
        if verbose:
            print(f"  [cache] wrote {_CACHE_PARQUET.name}")
    except Exception as exc:  # pragma: no cover — defensive
        if verbose:
            print(f"  [cache] WARNING: save failed: {exc}")
    return fresh


def clear_cache() -> None:
    """Delete both the parquet cache file and the key sidecar."""
    for p in (_CACHE_PARQUET, _CACHE_KEY_JSON):
        if p.exists():
            p.unlink()


def cache_info() -> dict[str, object]:
    """Return a small status dict for logging / debugging."""
    key_payload = _load_key_file()
    return {
        "parquet_exists": _CACHE_PARQUET.exists(),
        "key_file_exists": _CACHE_KEY_JSON.exists(),
        "cache_dir": str(_CACHE_DIR),
        "key_payload": key_payload,
        "sleeves_version": _SLEEVES_VERSION,
    }


__all__ = [
    "cached_strategy_daily_returns",
    "clear_cache",
    "cache_info",
    "ManifestMissingError",
]
