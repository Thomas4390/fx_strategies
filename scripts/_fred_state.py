"""SQLite progress checkpoint store for the FRED bulk pipeline.

WAL mode + busy_timeout for safe concurrent reads while one process writes.
The writer process (or main thread in --no-multiprocessing mode) is the sole
writer. Workers post updates via a queue.
"""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS progress (
  series_id          TEXT PRIMARY KEY,
  status             TEXT NOT NULL,
  category_root      TEXT,
  frequency          TEXT,
  popularity         INTEGER,
  rows               INTEGER,
  date_min           TEXT,
  date_max           TEXT,
  fetched_at         TEXT,
  last_updated_fred  TEXT,
  attempts           INTEGER NOT NULL DEFAULT 0,
  last_error         TEXT,
  last_error_at      TEXT,
  worker_pid         INTEGER,
  parquet_path       TEXT,
  parquet_sha256     TEXT,
  bytes              INTEGER
);
CREATE INDEX IF NOT EXISTS idx_progress_status   ON progress(status);
CREATE INDEX IF NOT EXISTS idx_progress_priority ON progress(popularity DESC);
CREATE INDEX IF NOT EXISTS idx_progress_freq     ON progress(frequency);

CREATE TABLE IF NOT EXISTS runs (
  run_id      TEXT PRIMARY KEY,
  started_at  TEXT NOT NULL,
  ended_at    TEXT,
  mode        TEXT,
  series_ok   INTEGER,
  series_err  INTEGER,
  rows_total  INTEGER,
  bytes_total INTEGER
);
"""


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@contextmanager
def open_state(db_path: Path) -> Iterator[sqlite3.Connection]:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), timeout=30.0, isolation_level=None)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA busy_timeout=30000")
    conn.row_factory = sqlite3.Row
    try:
        conn.executescript(SCHEMA_SQL)
        yield conn
    finally:
        conn.close()


def upsert_progress(conn: sqlite3.Connection, series_id: str, **fields: Any) -> None:
    fields = {k: v for k, v in fields.items() if v is not None}
    fields["series_id"] = series_id
    cols = list(fields.keys())
    placeholders = ",".join(f":{c}" for c in cols)
    update = ",".join(f"{c}=excluded.{c}" for c in cols if c != "series_id")
    sql = (
        f"INSERT INTO progress ({','.join(cols)}) VALUES ({placeholders}) "
        f"ON CONFLICT(series_id) DO UPDATE SET {update}"
    )
    conn.execute(sql, fields)


def increment_attempts(conn: sqlite3.Connection, series_id: str) -> None:
    conn.execute(
        "INSERT INTO progress(series_id, status, attempts) VALUES(?, 'pending', 1) "
        "ON CONFLICT(series_id) DO UPDATE SET attempts=attempts+1",
        (series_id,),
    )


def get_progress(conn: sqlite3.Connection, series_id: str) -> dict[str, Any] | None:
    row = conn.execute("SELECT * FROM progress WHERE series_id=?", (series_id,)).fetchone()
    return dict(row) if row else None


def status_counts(conn: sqlite3.Connection) -> dict[str, int]:
    rows = conn.execute(
        "SELECT status, COUNT(*) c, COALESCE(SUM(rows),0) r, COALESCE(SUM(bytes),0) b "
        "FROM progress GROUP BY status ORDER BY status"
    ).fetchall()
    return {r["status"]: {"count": r["c"], "rows": r["r"], "bytes": r["b"]} for r in rows}


def all_done_series_ids(conn: sqlite3.Connection) -> set[str]:
    rows = conn.execute("SELECT series_id FROM progress WHERE status='done'").fetchall()
    return {r["series_id"] for r in rows}


def begin_run(conn: sqlite3.Connection, mode: str) -> str:
    run_id = utc_now_iso()
    conn.execute(
        "INSERT INTO runs(run_id, started_at, mode) VALUES(?,?,?)",
        (run_id, run_id, mode),
    )
    return run_id


def end_run(conn: sqlite3.Connection, run_id: str, *,
            series_ok: int, series_err: int, rows_total: int, bytes_total: int) -> None:
    conn.execute(
        "UPDATE runs SET ended_at=?, series_ok=?, series_err=?, rows_total=?, bytes_total=? "
        "WHERE run_id=?",
        (utc_now_iso(), series_ok, series_err, rows_total, bytes_total, run_id),
    )


def latest_run(conn: sqlite3.Connection) -> dict[str, Any] | None:
    row = conn.execute("SELECT * FROM runs ORDER BY started_at DESC LIMIT 1").fetchone()
    return dict(row) if row else None
