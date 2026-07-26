#!/usr/bin/env python3
"""FRED bulk database — orchestrator CLI.

Phase A: walk full FRED catalog (categories, series, releases, sources, tags).
Phase B: fetch observations for prioritized series, sharded onto external drive.
Phase C: validate parquets, regenerate data/MANIFEST.json.

See docs / .claude/plans/est-ce-que-tu-pourrais-ethereal-raccoon.md for design.

Usage examples:
    python scripts/fred_full_database.py --phase=catalog
    python scripts/fred_full_database.py --phase=observations --top-popular 5000
    python scripts/fred_full_database.py --status
    python scripts/fred_full_database.py --scan-gaps
    python scripts/fred_full_database.py --series DGS10,UNRATE
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from _fred_catalog import build_catalog, build_catalog_via_releases, save_catalog
from _fred_client import FredError
from _fred_observations import run_observations
from _fred_paths import (CATALOG_DIR, DATA_DIR, EXTERNAL_ROOT, FRED_ROOT,
                          OBSERVATIONS_DIR, PROGRESS_DB, STATE_DIR,
                          external_mounted, read_env_var)
from _fred_planner import build_queue, classify, scan_gaps
from _fred_state import (begin_run, end_run, latest_run, open_state,
                          status_counts, upsert_progress, utc_now_iso)
from _fred_token_bucket import LocalTokenBucket
from _macro_registry import REGISTRY as CURATED


SERIES_PARQUET = CATALOG_DIR / "series.parquet"


# ─────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────
def _ensure_external() -> None:
    if not external_mounted():
        sys.exit(f"error: external drive not mounted at {EXTERNAL_ROOT}")
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    OBSERVATIONS_DIR.mkdir(parents=True, exist_ok=True)


def _load_catalog() -> pd.DataFrame:
    if not SERIES_PARQUET.exists():
        sys.exit(f"error: catalog missing — run --phase=catalog first ({SERIES_PARQUET})")
    return pd.read_parquet(SERIES_PARQUET)


def _load_progress_index() -> dict[str, dict]:
    if not PROGRESS_DB.exists():
        return {}
    with open_state(PROGRESS_DB) as conn:
        rows = conn.execute("SELECT * FROM progress").fetchall()
    return {r["series_id"]: dict(r) for r in rows}


def _check_disk_space(min_gb: float) -> None:
    free_gb = shutil.disk_usage(EXTERNAL_ROOT).free / 1e9
    if free_gb < min_gb:
        sys.exit(f"error: only {free_gb:.1f} GB free on external drive, "
                 f"need at least {min_gb:.0f} GB. Aborting.")


# ─────────────────────────────────────────────────────────────────────────
# Phase A
# ─────────────────────────────────────────────────────────────────────────
def cmd_catalog(args: argparse.Namespace) -> int:
    _ensure_external()
    api_key = read_env_var("FRED_API_KEY")
    bucket = LocalTokenBucket(rpm=110)
    t0 = time.time()
    print(f"[catalog] writing to {CATALOG_DIR}")
    if getattr(args, "via_categories", False):
        dfs = build_catalog(api_key, bucket,
                            on_progress=lambda done, tot, n_series: print(
                                f"  [{done:>5}/{tot}] categories scanned, "
                                f"{n_series:,} unique series so far"))
    else:
        dfs = build_catalog_via_releases(api_key, bucket,
                            on_progress=lambda done, tot, n_series: print(
                                f"  [{done:>3}/{tot}] releases processed, "
                                f"{n_series:,} unique series so far"))
    paths = save_catalog(dfs)
    elapsed = time.time() - t0
    print(f"\n[catalog] saved in {elapsed:.0f}s ({elapsed/60:.1f}min)")
    for name, p in paths.items():
        size_mb = p.stat().st_size / 1e6
        print(f"  {name:>10}  {len(dfs[name]):>9,} rows  {size_mb:>7.2f} MB  {p}")
    print(f"\nNext: python {Path(__file__).name} --phase=observations --top-popular 5000")
    return 0


# ─────────────────────────────────────────────────────────────────────────
# Phase B
# ─────────────────────────────────────────────────────────────────────────
def cmd_observations(args: argparse.Namespace) -> int:
    _ensure_external()
    catalog = _load_catalog()
    progress = _load_progress_index()
    api_key = read_env_var("FRED_API_KEY")

    if args.full:
        _check_disk_space(min_gb=60.0)
        mode = "full"
    elif args.top_popular:
        _check_disk_space(min_gb=15.0)
        mode = f"top-{args.top_popular}"
    elif args.series:
        mode = "series-list"
    else:
        _check_disk_space(min_gb=15.0)
        args.top_popular = 5000
        mode = "top-5000"

    series_filter = None
    if args.series:
        series_filter = [s.strip() for s in args.series.split(",") if s.strip()]

    queue = build_queue(
        catalog, progress,
        top_popular=None if args.full else args.top_popular,
        full=args.full,
        series_filter=series_filter,
        freq_filter=args.freq,
        category_filter=args.category,
        include_curated=not args.no_curated,
        retry_errors=args.retry_errors,
        retry_skipped=args.retry_skipped,
    )
    if args.dry_run:
        print(f"[dry-run] {len(queue)} series would be fetched in mode={mode}")
        for w in queue[:20]:
            tag = "★" if w.is_curated else " "
            print(f"  {tag} {w.series_id:<24} pop={w.popularity:>5}  freq={w.frequency_short}  cat={w.category_root}")
        if len(queue) > 20:
            print(f"  ... and {len(queue) - 20} more")
        return 0

    if not queue:
        print("[obs] nothing to do — all series up to date")
        return 0

    run_log = STATE_DIR / f"run_{utc_now_iso().replace(':', '').replace('-', '')}.log"
    summary = run_observations(
        queue,
        api_key=api_key,
        n_workers=args.workers,
        no_multiprocessing=args.no_multiprocessing,
        state_db_path=PROGRESS_DB,
        run_log_path=run_log,
        mode_label=f"observations:{mode}",
    )

    (STATE_DIR / "LATEST_RUN.txt").write_text(
        f"finished_at={utc_now_iso()}\nmode={mode}\nsummary={summary}\n",
        encoding="utf-8",
    )

    if summary["ok"] > 0:
        manifest_script = SCRIPT_DIR / "update_data_manifest.py"
        if manifest_script.exists():
            try:
                subprocess.run([sys.executable, str(manifest_script)], check=True,
                               cwd=str(SCRIPT_DIR.parent))
                print("[manifest] data/MANIFEST.json regenerated")
            except subprocess.CalledProcessError as e:
                print(f"[manifest] update failed: {e}")
    return 0 if summary["error"] == 0 else 2


# ─────────────────────────────────────────────────────────────────────────
# Curated-only mode
# ─────────────────────────────────────────────────────────────────────────
def cmd_curated(args: argparse.Namespace) -> int:
    args.series = ",".join(s.series_id for s in CURATED)
    args.full = False
    args.top_popular = None
    args.no_curated = False
    return cmd_observations(args)


# ─────────────────────────────────────────────────────────────────────────
# Status / scan-gaps / report
# ─────────────────────────────────────────────────────────────────────────
def cmd_status(_args: argparse.Namespace) -> int:
    if not PROGRESS_DB.exists():
        print("no progress.sqlite yet — nothing to report")
        return 0
    with open_state(PROGRESS_DB) as conn:
        counts = status_counts(conn)
        run = latest_run(conn)
    if run:
        print(f"Latest run : {run['run_id']}")
        print(f"Mode       : {run['mode']}")
        print(f"Started    : {run['started_at']}    Ended: {run['ended_at'] or '(running)'}")
    print("─" * 65)
    print(f"{'Status':<22} {'Count':>9}  {'Rows':>14}  {'Bytes':>10}")
    print("─" * 65)
    for status, info in sorted(counts.items()):
        rows = info["rows"]
        bytes_ = info["bytes"]
        size = f"{bytes_/1e6:.1f} MB" if bytes_ < 1e9 else f"{bytes_/1e9:.2f} GB"
        print(f"{status:<22} {info['count']:>9,}  {rows:>14,}  {size:>10}")
    print("─" * 65)
    return 0


def cmd_scan_gaps(_args: argparse.Namespace) -> int:
    catalog = _load_catalog()
    progress = _load_progress_index()
    print(f"[scan-gaps] scanning {len(catalog):,} series...")
    df = scan_gaps(catalog, progress)
    counts = df["classification"].value_counts()
    print("\nClassification:")
    for cls, n in counts.items():
        print(f"  {cls:<18} {n:,}")
    out = STATE_DIR / f"gaps_{utc_now_iso().replace(':', '').replace('-', '')}.parquet"
    df.to_parquet(out, index=False)
    print(f"\nFull breakdown saved to: {out}")
    return 0


def cmd_report(_args: argparse.Namespace) -> int:
    """Produce a markdown summary of catalog + progress to stdout."""
    catalog = _load_catalog() if SERIES_PARQUET.exists() else None
    progress = _load_progress_index()
    print("# FRED database — report")
    print(f"\nGenerated: {utc_now_iso()}")
    if catalog is not None:
        print("\n## Catalog\n")
        print(f"- series: {len(catalog):,}")
        print(f"- top-level categories: {catalog['category_root'].nunique()}")
        print(f"- frequencies: {dict(catalog['frequency_short'].value_counts())}")
        print("\n### Top categories\n")
        counts = catalog["category_root"].value_counts().head(10)
        for name, n in counts.items():
            print(f"- {name}: {n:,}")
    if progress:
        print("\n## Progress\n")
        with open_state(PROGRESS_DB) as conn:
            counts = status_counts(conn)
        for status, info in sorted(counts.items()):
            size = info["bytes"]
            sz = f"{size/1e9:.2f} GB" if size > 1e9 else f"{size/1e6:.1f} MB"
            print(f"- **{status}**: {info['count']:,} series  ({info['rows']:,} rows, {sz})")
    return 0


def cmd_catalog_fixup(_args: argparse.Namespace) -> int:
    """Re-run failed categories from the latest run log and merge into catalog."""
    _ensure_external()
    if not SERIES_PARQUET.exists():
        sys.exit("error: catalog missing; run --phase=catalog first")

    logs = sorted(STATE_DIR.glob("run_*.log")) + [Path("/tmp/fred_catalog_run.log")]
    failed_ids: set[int] = set()
    for lp in logs:
        if not lp.exists():
            continue
        for line in lp.read_text(errors="ignore").splitlines():
            m = None
            if "[warn] children of" in line:
                try:
                    m = int(line.split("children of")[1].split("failed")[0].strip())
                except Exception:
                    pass
            elif "[warn] series for cat" in line:
                try:
                    m = int(line.split("cat")[1].split("failed")[0].strip())
                except Exception:
                    pass
            if m is not None:
                failed_ids.add(m)
    if not failed_ids:
        print("no failed categories detected in logs")
        return 0
    print(f"[fixup] retrying {len(failed_ids)} failed categories: {sorted(failed_ids)}")

    api_key = read_env_var("FRED_API_KEY")
    bucket = LocalTokenBucket(rpm=110)

    catalog_df = pd.read_parquet(SERIES_PARQUET)
    cats_df = pd.read_parquet(CATALOG_DIR / "categories.parquet")
    cats_index = {int(r["id"]): {"id": int(r["id"]), "name": r["name"],
                                  "parent_id": r["parent_id"]}
                   for _, r in cats_df.iterrows()}
    existing_ids = set(catalog_df["id"])

    from _fred_catalog import (fetch_series_for_category, retry_failed_categories,
                                category_path)
    n_before = len(catalog_df)

    new_cats = retry_failed_categories(list(failed_ids), api_key, bucket, cats_index)
    print(f"[fixup] discovered {new_cats} new sub-categories during retry")

    new_rows = []
    leaf_set = set(failed_ids)
    if new_cats:
        leaf_set.update(int(cid) for cid in cats_index if cid not in existing_ids and cid != 0)

    for cid in leaf_set:
        items = fetch_series_for_category(api_key, bucket, cid)
        if items is None:
            print(f"  [fixup-warn] cat {cid} still failing")
            continue
        for s in items:
            sid = s["id"]
            if sid in existing_ids:
                continue
            existing_ids.add(sid)
            paths = []
            roots = []
            p = category_path(cats_index, cid)
            if p:
                if len(p) > 1:
                    roots.append(p[1])
                paths.append(" / ".join(p[1:]))
            new_rows.append({
                "id": sid, "title": s.get("title", ""),
                "frequency": s.get("frequency", ""),
                "frequency_short": s.get("frequency_short", ""),
                "units": s.get("units", ""), "units_short": s.get("units_short", ""),
                "seasonal_adjustment": s.get("seasonal_adjustment", ""),
                "seasonal_adjustment_short": s.get("seasonal_adjustment_short", ""),
                "observation_start": s.get("observation_start", ""),
                "observation_end": s.get("observation_end", ""),
                "last_updated": s.get("last_updated", ""),
                "popularity": int(s.get("popularity", 0) or 0),
                "group_popularity": int(s.get("group_popularity", 0) or 0),
                "notes": s.get("notes", "") or "",
                "realtime_start": s.get("realtime_start", ""),
                "realtime_end": s.get("realtime_end", ""),
                "category_ids": f"[{cid}]",
                "category_root": roots[0] if roots else "Other",
                "category_paths": " | ".join(paths) if paths else "",
            })

    if not new_rows:
        print("[fixup] no new series recovered")
        return 0
    new_df = pd.DataFrame(new_rows)
    merged = pd.concat([catalog_df, new_df], ignore_index=True).drop_duplicates(subset=["id"])
    merged = merged.sort_values("popularity", ascending=False).reset_index(drop=True)
    tmp = SERIES_PARQUET.with_suffix(".parquet.tmp")
    merged.to_parquet(tmp, index=False)
    tmp.replace(SERIES_PARQUET)
    print(f"[fixup] series.parquet: {n_before:,} → {len(merged):,} (+{len(merged)-n_before})")
    return 0


def cmd_explain(args: argparse.Namespace) -> int:
    sid = args.explain
    catalog = _load_catalog()
    progress = _load_progress_index()
    row = catalog[catalog["id"] == sid]
    if row.empty:
        print(f"{sid} not found in catalog")
        return 1
    r = row.iloc[0]
    print(f"=== {sid} ===")
    print(f"Title         : {r['title']}")
    print(f"Frequency     : {r['frequency']} ({r['frequency_short']})")
    print(f"Units         : {r['units']}")
    print(f"Seasonal adj. : {r['seasonal_adjustment']}")
    print(f"Observations  : {r['observation_start']} → {r['observation_end']}")
    print(f"Last updated  : {r['last_updated']}")
    print(f"Popularity    : {r['popularity']}")
    print(f"Category root : {r.get('category_root', '?')}")
    print(f"Category paths: {r.get('category_paths', '?')}")
    p = progress.get(sid)
    print()
    if p:
        print(f"Local status  : {p['status']}")
        print(f"Rows on disk  : {p.get('rows')}")
        print(f"Date range    : {p.get('date_min')} → {p.get('date_max')}")
        print(f"Fetched at    : {p.get('fetched_at')}")
        print(f"Parquet path  : {p.get('parquet_path')}")
        print(f"Sha256        : {p.get('parquet_sha256')}")
        if p.get("last_error"):
            print(f"Last error    : {p['last_error']} (at {p['last_error_at']})")
    else:
        print("Local status  : not yet downloaded")
    cls = classify(sid, r, p)
    print(f"\nClassification: {cls}")
    return 0


# ─────────────────────────────────────────────────────────────────────────
# Validate
# ─────────────────────────────────────────────────────────────────────────
def cmd_validate(_args: argparse.Namespace) -> int:
    _ensure_external()
    progress = _load_progress_index()
    n_total = len(progress)
    if n_total == 0:
        print("no progress to validate")
        return 0
    n_ok = 0
    failures = []
    for sid, p in progress.items():
        if p["status"] != "done":
            continue
        path = Path(p["parquet_path"]) if p.get("parquet_path") else None
        if path is None or not path.exists():
            failures.append((sid, "parquet missing on disk"))
            continue
        try:
            df = pd.read_parquet(path)
            assert list(df.columns) == ["date", "value"], f"bad columns {df.columns.tolist()}"
            assert df["date"].is_monotonic_increasing, "date not sorted"
            assert not df["date"].duplicated().any(), "duplicate dates"
            assert not df["value"].isna().any(), "NaN values present"
            assert df["value"].dtype.name == "float64"
            n_ok += 1
        except Exception as e:
            failures.append((sid, str(e)))
    print(f"validated {n_ok}/{sum(1 for p in progress.values() if p['status']=='done')} done series")
    if failures:
        print(f"\n{len(failures)} failures:")
        for sid, err in failures[:20]:
            print(f"  {sid}: {err}")
        return 2
    return 0


# ─────────────────────────────────────────────────────────────────────────
# Run all phases
# ─────────────────────────────────────────────────────────────────────────
def cmd_all(args: argparse.Namespace) -> int:
    rc = cmd_catalog(args)
    if rc != 0:
        return rc
    return cmd_observations(args)


# ─────────────────────────────────────────────────────────────────────────
# argparse
# ─────────────────────────────────────────────────────────────────────────
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    p.add_argument("--phase", choices=["catalog", "observations", "curated", "all"],
                   default=None,
                   help="Phase to run. Default: all.")
    p.add_argument("--status", action="store_true",
                   help="Print progress summary and exit")
    p.add_argument("--scan-gaps", dest="scan_gaps", action="store_true",
                   help="Detect missing/stale series and exit")
    p.add_argument("--explain", metavar="SERIES_ID",
                   help="Show everything we know about a series")
    p.add_argument("--validate", action="store_true",
                   help="Re-read all done parquets and check invariants")
    p.add_argument("--report", action="store_true",
                   help="Print markdown summary of catalog and progress")
    p.add_argument("--catalog-fixup", dest="catalog_fixup", action="store_true",
                   help="Retry failed categories from logs and merge into series.parquet")
    p.add_argument("--via-categories", dest="via_categories", action="store_true",
                   help="Use slower category-tree walk (default uses /releases)")

    p.add_argument("--top-popular", type=int, default=None,
                   help="Limit observations to top N by popularity (default: 5000)")
    p.add_argument("--full", action="store_true",
                   help="Fetch ALL series in the catalog (multi-day)")
    p.add_argument("--series", help="Comma-separated list of series_ids")
    p.add_argument("--freq", help="Filter by frequency_short (D/W/M/Q/A)")
    p.add_argument("--category", help="Filter by category root substring")
    p.add_argument("--no-curated", action="store_true",
                   help="Skip the curated FX series (otherwise they go first)")
    p.add_argument("--retry-errors", action="store_true",
                   help="Re-attempt series in status=error")
    p.add_argument("--retry-skipped", action="store_true",
                   help="Re-attempt series in status=skipped_non_numeric")

    p.add_argument("--workers", type=int, default=3,
                   help="Worker processes (clamp 1-3)")
    p.add_argument("--no-multiprocessing", action="store_true",
                   help="Single-process mode for debug")
    p.add_argument("--dry-run", action="store_true",
                   help="Show what would be fetched without doing it")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.status:
        return cmd_status(args)
    if args.scan_gaps:
        return cmd_scan_gaps(args)
    if args.explain:
        return cmd_explain(args)
    if args.validate:
        return cmd_validate(args)
    if args.report:
        return cmd_report(args)
    if args.catalog_fixup:
        return cmd_catalog_fixup(args)

    phase = args.phase or "all"
    try:
        if phase == "catalog":
            return cmd_catalog(args)
        if phase == "observations":
            return cmd_observations(args)
        if phase == "curated":
            return cmd_curated(args)
        if phase == "all":
            return cmd_all(args)
    except FredError as e:
        sys.exit(f"FRED error: {e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
