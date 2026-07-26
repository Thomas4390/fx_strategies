"""Phase B — fetch observations and persist parquets.

Multiprocessing model:
    * Worker processes (Pool, default 3) call ``fetch_one()`` for each WorkItem.
      Each fetch_one acquires the shared TokenBucket, downloads observations,
      and returns a serializable result.
    * The main process consumes results via ``imap_unordered``, writes the
      parquet atomically and updates the SQLite progress DB. Single-writer
      avoids contention on the progress DB and the external drive.
"""
from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from _fred_client import FredError, FredHTTPError, fetch_observations
from _fred_paths import (DATA_DIR, OBSERVATIONS_DIR, external_mounted,
                          observation_path, meta_path_for, read_env_var)
from _fred_planner import WorkItem
from _fred_state import end_run, open_state, upsert_progress, utc_now_iso, begin_run
from _macro_registry import by_series_id


_GLOBAL_BUCKET = None
_GLOBAL_API_KEY = None


def _worker_init(bucket, api_key: str) -> None:
    global _GLOBAL_BUCKET, _GLOBAL_API_KEY
    _GLOBAL_BUCKET = bucket
    _GLOBAL_API_KEY = api_key


@dataclass
class FetchResult:
    series_id: str
    status: str                       # 'ok' | 'error' | 'skipped_non_numeric'
    rows: int = 0
    parquet_bytes: bytes | None = None
    metadata: dict | None = None
    date_min: str | None = None
    date_max: str | None = None
    error: str | None = None
    parquet_path: str | None = None
    parquet_sha256: str | None = None
    bytes: int = 0
    duration_sec: float = 0.0
    item: WorkItem | None = None


def _build_observations_df(observations: list[dict]) -> pd.DataFrame:
    rows = []
    for o in observations:
        v = o.get("value", "")
        if v in (".", "", None):
            continue
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        rows.append((o.get("date"), f))
    if not rows:
        return pd.DataFrame(columns=["date", "value"])
    df = pd.DataFrame(rows, columns=["date", "value"])
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df["value"] = df["value"].astype("float64")
    df = df.drop_duplicates(subset=["date"], keep="last").sort_values("date").reset_index(drop=True)
    return df


def fetch_one(item: WorkItem) -> FetchResult:
    """Worker entry — runs in a child process. Returns serializable result."""
    t0 = time.time()
    api_key = _GLOBAL_API_KEY
    bucket = _GLOBAL_BUCKET
    meta = item.metadata_snapshot or {"id": item.series_id}
    try:
        obs = fetch_observations(item.series_id, api_key, bucket=bucket)
    except FredError as e:
        return FetchResult(series_id=item.series_id, status="error",
                           error=f"observations: {e}", duration_sec=time.time() - t0,
                           metadata=meta, item=item)
    df = _build_observations_df(obs)
    if df.empty:
        return FetchResult(series_id=item.series_id, status="skipped_non_numeric",
                           error="no numeric observations after coercion",
                           duration_sec=time.time() - t0, metadata=meta, item=item)

    import io
    import pyarrow as pa  # noqa: F401
    buf = io.BytesIO()
    df.to_parquet(buf, index=False)
    parquet_bytes = buf.getvalue()
    sha = hashlib.sha256(parquet_bytes).hexdigest()
    return FetchResult(
        series_id=item.series_id,
        status="ok",
        rows=len(df),
        parquet_bytes=parquet_bytes,
        metadata=meta,
        date_min=df["date"].min().date().isoformat(),
        date_max=df["date"].max().date().isoformat(),
        parquet_sha256=sha,
        bytes=len(parquet_bytes),
        duration_sec=time.time() - t0,
        item=item,
    )


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(payload)
    os.replace(tmp, path)


def _legacy_curated_write(result: FetchResult) -> Path | None:
    """Mirror curated series to data/<STEM>_<freq>.parquet (legacy schema).

    The legacy schema is ``[date, <snake_metric>]`` (vs the bulk's
    ``[date, value]``). We rebuild that view from the canonical bytes.
    """
    spec = by_series_id(result.series_id)
    if spec is None:
        return None
    df = pd.read_parquet(observation_path(
        result.series_id, result.item.frequency_short, result.item.category_root))
    legacy = pd.DataFrame({"date": df["date"], spec.column: df["value"].astype("float64")})
    out = DATA_DIR / f"{spec.stem}_{spec.freq}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".parquet.tmp")
    legacy.to_parquet(tmp, index=False)
    os.replace(tmp, out)
    return out


def _write_one(result: FetchResult, db_conn) -> None:
    item = result.item
    if result.status != "ok":
        upsert_progress(db_conn, item.series_id,
                        status=result.status,
                        category_root=item.category_root,
                        frequency=item.frequency_short,
                        popularity=item.popularity,
                        last_error=result.error,
                        last_error_at=utc_now_iso(),
                        worker_pid=os.getpid())
        return

    parquet = observation_path(item.series_id, item.frequency_short, item.category_root)
    _atomic_write(parquet, result.parquet_bytes)
    if result.metadata is not None:
        meta_p = meta_path_for(parquet)
        meta_p.write_text(json.dumps(result.metadata, indent=2), encoding="utf-8")

    legacy = None
    if item.is_curated:
        try:
            legacy = _legacy_curated_write(result)
        except Exception as e:
            print(f"  [warn] legacy write failed for {item.series_id}: {e}")

    upsert_progress(db_conn, item.series_id,
                    status="done",
                    category_root=item.category_root,
                    frequency=item.frequency_short,
                    popularity=item.popularity,
                    rows=result.rows,
                    date_min=result.date_min,
                    date_max=result.date_max,
                    fetched_at=utc_now_iso(),
                    last_updated_fred=item.last_updated_fred,
                    worker_pid=os.getpid(),
                    parquet_path=str(parquet),
                    parquet_sha256=result.parquet_sha256,
                    bytes=result.bytes,
                    last_error=None, last_error_at=None)


def run_observations(items: list[WorkItem], *,
                     api_key: str,
                     n_workers: int = 3,
                     no_multiprocessing: bool = False,
                     state_db_path: Path,
                     run_log_path: Path,
                     mode_label: str) -> dict[str, Any]:
    if not external_mounted():
        raise RuntimeError(f"external drive not mounted: {OBSERVATIONS_DIR.parent}")

    n_workers = max(1, min(3, n_workers))
    n_items = len(items)
    summary = {"ok": 0, "error": 0, "skipped": 0, "rows": 0, "bytes": 0,
               "duration_sec": 0.0}
    t_start = time.time()

    print(f"[obs] starting run: {n_items} items, workers={n_workers}, mp={not no_multiprocessing}")

    log_f = run_log_path.open("a", encoding="utf-8")
    try:
        with open_state(state_db_path) as conn:
            run_id = begin_run(conn, mode_label)
            if no_multiprocessing or n_workers == 1:
                from _fred_token_bucket import LocalTokenBucket
                bucket = LocalTokenBucket(rpm=110)
                _worker_init(bucket, api_key)
                for idx, item in enumerate(items, start=1):
                    res = fetch_one(item)
                    _write_one(res, conn)
                    _post_progress(summary, res, idx, n_items, t_start, log_f)
            else:
                from _fred_token_bucket import TokenBucket
                with mp.Manager() as manager:
                    bucket = TokenBucket(manager, rpm=110)
                    with mp.Pool(processes=n_workers,
                                  initializer=_worker_init,
                                  initargs=(bucket, api_key)) as pool:
                        for idx, res in enumerate(pool.imap_unordered(fetch_one, items, chunksize=1), start=1):
                            _write_one(res, conn)
                            _post_progress(summary, res, idx, n_items, t_start, log_f)
            end_run(conn, run_id,
                    series_ok=summary["ok"], series_err=summary["error"],
                    rows_total=summary["rows"], bytes_total=summary["bytes"])
    finally:
        log_f.close()

    summary["duration_sec"] = time.time() - t_start
    print(f"[obs] done: {summary['ok']} ok, {summary['error']} err, "
          f"{summary['skipped']} skipped, {summary['rows']:,} rows, "
          f"{summary['bytes']/1e6:.1f} MB, {summary['duration_sec']:.0f}s")
    return summary


def _post_progress(summary: dict, res: FetchResult, idx: int, total: int,
                   t_start: float, log_f) -> None:
    if res.status == "ok":
        summary["ok"] += 1
        summary["rows"] += res.rows
        summary["bytes"] += res.bytes
    elif res.status == "skipped_non_numeric":
        summary["skipped"] += 1
    else:
        summary["error"] += 1
    log_f.write(json.dumps({
        "ts": utc_now_iso(),
        "series_id": res.series_id,
        "status": res.status,
        "rows": res.rows,
        "bytes": res.bytes,
        "duration_sec": round(res.duration_sec, 3),
        "error": res.error,
    }) + "\n")
    log_f.flush()
    if idx % 50 == 0 or idx == total or res.status != "ok":
        elapsed = time.time() - t_start
        rate = idx / elapsed if elapsed > 0 else 0.0
        eta = (total - idx) / rate if rate > 0 else 0.0
        flag = "✓" if res.status == "ok" else ("⊘" if res.status == "skipped_non_numeric" else "✗")
        print(f"  [{idx:>5}/{total}] {flag} {res.series_id:<24} "
              f"rows={res.rows:>7,} {res.duration_sec:.2f}s  "
              f"rate={rate:.1f}/s ETA {eta/60:.1f}min")
