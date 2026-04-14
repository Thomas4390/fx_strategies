"""Generate/refresh ``data/MANIFEST.json`` with SHA256 + metadata.

The manifest is a small committed JSON file that captures the identity
and shape of every ``data/*.parquet`` used by the research pipeline.
Its purpose is twofold :

- **Reproducibility gate** : sweep scripts (Phase 22+) load the manifest
  at startup and refuse to run if a source file's SHA256 has changed
  since the manifest was committed. Silent data updates can no longer
  change backtest Sharpe without triggering an explicit manifest refresh.
- **Cache invalidation key** : the cached ``get_strategy_daily_returns``
  output (Phase 22) is keyed off the manifest fingerprint so it is
  automatically invalidated on any data change.

The manifest schema is intentionally flat and JSON-first. Schema
(dict, one top-level ``files`` key) :

```
{
  "generated_at": "2026-04-13T14:30:00+00:00",
  "project_root": "/abs/path",
  "files": {
    "EUR-USD_minute.parquet": {
      "sha256": "<hex>",
      "size_bytes": 49080320,
      "size_mb": 46.81,
      "row_count": 5256001,
      "column_count": 5,
      "columns": ["date", "open", "high", "low", "close"],
      "index_start": "2018-01-01 22:00:00",
      "index_end":   "2026-04-02 21:59:00"
    },
    ...
  },
  "combined_fingerprint": "<sha256 of concatenated per-file hashes>"
}
```

Usage
-----
    python scripts/update_data_manifest.py           # refresh in place
    python scripts/update_data_manifest.py --check   # non-zero exit if drift
    python scripts/update_data_manifest.py --dry-run # print, no write
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd


_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DATA_DIR = _PROJECT_ROOT / "data"
_MANIFEST_PATH = _DATA_DIR / "MANIFEST.json"
_CHUNK_SIZE = 1024 * 1024  # 1 MB


def _sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(_CHUNK_SIZE), b""):
            h.update(chunk)
    return h.hexdigest()


def _parquet_metadata(path: Path) -> dict[str, Any]:
    """Return shape + index range without loading the file into VBT."""
    df = pd.read_parquet(path)
    cols = [str(c) for c in df.columns]
    # Most project parquets keep the datetime column as a regular column
    # named "date" rather than an index.
    if "date" in df.columns and pd.api.types.is_datetime64_any_dtype(df["date"]):
        idx_start = str(df["date"].min())
        idx_end = str(df["date"].max())
    elif isinstance(df.index, pd.DatetimeIndex):
        idx_start = str(df.index.min())
        idx_end = str(df.index.max())
    else:
        idx_start = idx_end = ""
    return {
        "row_count": int(len(df)),
        "column_count": int(len(cols)),
        "columns": cols,
        "index_start": idx_start,
        "index_end": idx_end,
    }


def _file_entry(path: Path) -> dict[str, Any]:
    size_bytes = path.stat().st_size
    entry: dict[str, Any] = {
        "sha256": _sha256_of_file(path),
        "size_bytes": int(size_bytes),
        "size_mb": round(size_bytes / (1024 * 1024), 3),
    }
    try:
        entry.update(_parquet_metadata(path))
    except Exception as exc:  # pragma: no cover — defensive
        entry["metadata_error"] = f"{type(exc).__name__}: {exc}"
    return entry


def _combined_fingerprint(files: dict[str, dict[str, Any]]) -> str:
    h = hashlib.sha256()
    for name in sorted(files):
        sha = files[name]["sha256"]
        h.update(f"{name}:{sha}\n".encode("utf-8"))
    return h.hexdigest()


def build_manifest() -> dict[str, Any]:
    """Walk ``data/*.parquet`` and return a fresh manifest dict."""
    files: dict[str, dict[str, Any]] = {}
    for path in sorted(_DATA_DIR.glob("*.parquet")):
        files[path.name] = _file_entry(path)
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "project_root": str(_PROJECT_ROOT),
        "files": files,
        "combined_fingerprint": _combined_fingerprint(files),
    }


def load_manifest() -> dict[str, Any] | None:
    """Read ``data/MANIFEST.json`` if present."""
    if not _MANIFEST_PATH.exists():
        return None
    with _MANIFEST_PATH.open("r") as fh:
        return json.load(fh)


def save_manifest(manifest: dict[str, Any]) -> None:
    _MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _MANIFEST_PATH.open("w") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=False)
        fh.write("\n")


def diff_manifests(old: dict[str, Any], new: dict[str, Any]) -> list[str]:
    """Return a flat list of human-readable differences."""
    diffs: list[str] = []
    old_files = old.get("files", {})
    new_files = new.get("files", {})
    added = sorted(set(new_files) - set(old_files))
    removed = sorted(set(old_files) - set(new_files))
    for name in added:
        diffs.append(f"+ {name}  (new file)")
    for name in removed:
        diffs.append(f"- {name}  (removed)")
    for name in sorted(set(old_files) & set(new_files)):
        old_sha = old_files[name].get("sha256")
        new_sha = new_files[name].get("sha256")
        if old_sha != new_sha:
            diffs.append(
                f"~ {name}  sha {old_sha[:10]}… → {new_sha[:10]}…  "
                f"(rows {old_files[name].get('row_count')} → "
                f"{new_files[name].get('row_count')})"
            )
    if old.get("combined_fingerprint") != new.get("combined_fingerprint"):
        diffs.append(
            f"combined_fingerprint changed: "
            f"{old.get('combined_fingerprint', '')[:10]}… → "
            f"{new.get('combined_fingerprint', '')[:10]}…"
        )
    return diffs


def assert_manifest_fresh() -> dict[str, Any]:
    """Raise ``RuntimeError`` if the on-disk manifest drifts from the data.

    Called at the top of sweep scripts (Phase 22+). Returns the
    up-to-date manifest on success — the caller can then reuse its
    ``combined_fingerprint`` as a cache key.
    """
    if not _DATA_DIR.exists():
        raise RuntimeError(f"Data directory not found: {_DATA_DIR}")

    on_disk = load_manifest()
    current = build_manifest()

    if on_disk is None:
        raise RuntimeError(
            f"No manifest at {_MANIFEST_PATH}. "
            "Run `python scripts/update_data_manifest.py` to create one."
        )

    if on_disk.get("combined_fingerprint") != current.get("combined_fingerprint"):
        diffs = diff_manifests(on_disk, current)
        raise RuntimeError(
            "Data drift detected — MANIFEST.json is stale relative to "
            "`data/`. Review the diff and refresh with "
            "`python scripts/update_data_manifest.py`.\n\n" + "\n".join(diffs)
        )
    return current


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit non-zero if the manifest is out of date.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the manifest instead of writing it.",
    )
    args = parser.parse_args()

    new_manifest = build_manifest()

    if args.check:
        old = load_manifest()
        if old is None:
            print(
                "ERROR: data/MANIFEST.json is missing — run without --check to create."
            )
            return 2
        diffs = diff_manifests(old, new_manifest)
        if diffs:
            print("Data drift detected:")
            for d in diffs:
                print(f"  {d}")
            return 1
        print(f"OK — manifest up to date ({len(new_manifest['files'])} files).")
        return 0

    if args.dry_run:
        print(json.dumps(new_manifest, indent=2))
        return 0

    old = load_manifest()
    save_manifest(new_manifest)
    n_files = len(new_manifest["files"])
    print(
        f"Manifest written → {_MANIFEST_PATH}  "
        f"({n_files} files, fingerprint "
        f"{new_manifest['combined_fingerprint'][:12]}…)"
    )
    if old is not None:
        diffs = diff_manifests(old, new_manifest)
        if diffs:
            print("\nDiff vs previous manifest:")
            for d in diffs:
                print(f"  {d}")
        else:
            print("\nNo changes vs previous manifest.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
