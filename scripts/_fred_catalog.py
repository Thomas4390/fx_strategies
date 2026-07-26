"""Phase A — walk FRED categories and build the full series catalog.

Strategy:
    1. BFS from category_id=0 via /fred/category/children to enumerate every
       category in the tree.
    2. For each category, fetch /fred/category/series (paginated 1000/page) to
       enumerate the series it contains. A series can belong to multiple
       categories — we accumulate ``category_paths`` per series.
    3. Also fetch /fred/releases, /fred/sources, /fred/tags as flat lists.
    4. Persist 5 parquets in <external>/fred/catalog/.

The catalog walk is dominated by /fred/category/series calls. With ~5500
categories at avg 30-150 series each, plus pagination, expect 5k–10k
requests → 45-90 minutes at 110 req/min. The token bucket rate-limits.
"""
from __future__ import annotations

import json
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

import pandas as pd

from _fred_client import paginate, request_json
from _fred_paths import CATALOG_DIR


CHILDREN_PATH = "/category/children"
SERIES_PATH = "/category/series"
CATEGORY_PATH = "/category"


def fetch_all_categories(api_key: str, bucket, root_id: int = 0,
                         _failed: list[int] | None = None) -> dict[int, dict[str, Any]]:
    """BFS the FRED category tree starting from ``root_id``.

    Returns ``{category_id: {id, name, parent_id}}``. The root (id=0) is
    inserted synthetically. If ``_failed`` is provided, failed cat IDs are
    appended to it for a follow-up retry.
    """
    cats: dict[int, dict[str, Any]] = {0: {"id": 0, "name": "ROOT", "parent_id": None}}
    queue: deque[int] = deque([root_id])
    visited: set[int] = {root_id}
    n_done = 0

    while queue:
        cat_id = queue.popleft()
        try:
            data = request_json(CHILDREN_PATH, {"category_id": cat_id}, api_key, bucket=bucket)
        except Exception as e:
            print(f"  [warn] children of {cat_id} failed: {e}")
            if _failed is not None:
                _failed.append(cat_id)
            continue
        for c in data.get("categories", []):
            cid = int(c["id"])
            if cid in visited:
                continue
            visited.add(cid)
            cats[cid] = {"id": cid, "name": c.get("name", ""), "parent_id": cat_id}
            queue.append(cid)
        n_done += 1
        if n_done % 200 == 0:
            print(f"  [bfs] processed {n_done} cats, {len(cats):,} known so far, queue={len(queue)}")
    return cats


def retry_failed_categories(failed_ids: list[int], api_key: str, bucket,
                            cats: dict[int, dict[str, Any]]) -> int:
    """Re-attempt children fetch for previously-failed categories. In-place
    update of ``cats``. Returns count of categories newly added."""
    n_new = 0
    for cat_id in failed_ids:
        try:
            data = request_json(CHILDREN_PATH, {"category_id": cat_id}, api_key, bucket=bucket)
        except Exception as e:
            print(f"  [retry-warn] children of {cat_id} still failing: {e}")
            continue
        for c in data.get("categories", []):
            cid = int(c["id"])
            if cid in cats:
                continue
            cats[cid] = {"id": cid, "name": c.get("name", ""), "parent_id": cat_id}
            n_new += 1
    return n_new


def category_path(cats: dict[int, dict[str, Any]], cat_id: int) -> list[str]:
    """Return list of category names from root to ``cat_id``."""
    path: list[str] = []
    cur: int | None = cat_id
    seen: set[int] = set()
    while cur is not None and cur in cats and cur not in seen:
        seen.add(cur)
        path.append(cats[cur]["name"])
        cur = cats[cur].get("parent_id")
    path.reverse()
    return path


def fetch_series_for_category(api_key: str, bucket, cat_id: int) -> list[dict[str, Any]] | None:
    """Returns list of series dicts, or None if the call failed (caller can retry)."""
    items: list[dict[str, Any]] = []
    try:
        for s in paginate(SERIES_PATH, {"category_id": cat_id}, api_key,
                          list_key="seriess", bucket=bucket, page_size=1000):
            items.append(s)
        return items
    except Exception as e:
        print(f"  [warn] series for cat {cat_id} failed: {e}")
        return None


def fetch_flat_list(path: str, list_key: str, api_key: str, bucket) -> list[dict[str, Any]]:
    return list(paginate(path, {}, api_key, list_key=list_key, bucket=bucket, page_size=1000))


def build_catalog_via_releases(api_key: str, bucket, *,
                               on_progress=None) -> dict[str, pd.DataFrame]:
    """Faster catalog walk via /releases instead of BFS through categories.

    FRED has ~324 releases vs ~10K+ categories. Most series belong to a release
    and the cumulative API call count is ~1k vs ~15k. ``category_root`` is
    derived from the release group rather than the category tree.
    """
    t0 = time.time()
    print("[catalog] fetching releases, sources, tags...")
    releases = fetch_flat_list("/releases", "releases", api_key, bucket)
    sources = fetch_flat_list("/sources", "sources", api_key, bucket)
    tags = fetch_flat_list("/tags", "tags", api_key, bucket)
    print(f"[catalog] {len(releases)} releases, {len(sources)} sources, {len(tags):,} tags ({time.time()-t0:.0f}s)")

    series_by_id: dict[str, dict[str, Any]] = {}
    series_release: dict[str, int] = {}
    t1 = time.time()
    for i, r in enumerate(releases, start=1):
        rid = r["id"]
        rname = r.get("name", f"release_{rid}")
        try:
            for s in paginate("/release/series", {"release_id": rid}, api_key,
                              list_key="seriess", bucket=bucket, page_size=1000):
                sid = s["id"]
                if sid not in series_by_id:
                    series_by_id[sid] = s
                    series_release[sid] = rid
        except Exception as e:
            print(f"  [warn] release {rid} ({rname}) failed: {e}")
        if on_progress and i % 20 == 0:
            on_progress(i, len(releases), len(series_by_id))
    print(f"[catalog] {len(series_by_id):,} unique series via releases ({time.time()-t1:.0f}s)")

    # Build a synthetic categories table mapping release_id -> "category" so the
    # rest of the pipeline (sharding, planner) can use category_root identically.
    cats_df = pd.DataFrame.from_records([
        {"id": r["id"], "name": r.get("name", ""), "parent_id": None}
        for r in releases
    ])

    release_index = {r["id"]: r for r in releases}

    rows = []
    for sid, meta in series_by_id.items():
        rid = series_release.get(sid)
        rinfo = release_index.get(rid, {})
        rname = rinfo.get("name", "Other")
        rows.append({
            "id": sid,
            "title": meta.get("title", ""),
            "frequency": meta.get("frequency", ""),
            "frequency_short": meta.get("frequency_short", ""),
            "units": meta.get("units", ""),
            "units_short": meta.get("units_short", ""),
            "seasonal_adjustment": meta.get("seasonal_adjustment", ""),
            "seasonal_adjustment_short": meta.get("seasonal_adjustment_short", ""),
            "observation_start": meta.get("observation_start", ""),
            "observation_end": meta.get("observation_end", ""),
            "last_updated": meta.get("last_updated", ""),
            "popularity": int(meta.get("popularity", 0) or 0),
            "group_popularity": int(meta.get("group_popularity", 0) or 0),
            "notes": meta.get("notes", "") or "",
            "realtime_start": meta.get("realtime_start", ""),
            "realtime_end": meta.get("realtime_end", ""),
            "category_ids": "[]",
            "category_root": rname,
            "category_paths": rname,
            "release_id": rid,
        })
    series_df = pd.DataFrame(rows).sort_values("popularity", ascending=False).reset_index(drop=True)

    return {
        "categories": cats_df,
        "series": series_df,
        "releases": pd.DataFrame(releases) if releases else pd.DataFrame(),
        "sources": pd.DataFrame(sources) if sources else pd.DataFrame(),
        "tags": pd.DataFrame(tags) if tags else pd.DataFrame(),
    }


def build_catalog(api_key: str, bucket, *,
                  on_progress=None) -> dict[str, pd.DataFrame]:
    """Phase A driver. Returns a dict of dataframes ready to persist."""
    t0 = time.time()
    print("[catalog] walking categories...")
    failed_cats: list[int] = []
    cats = fetch_all_categories(api_key, bucket, _failed=failed_cats)
    print(f"[catalog] discovered {len(cats):,} categories ({time.time() - t0:.0f}s)")
    if failed_cats:
        print(f"[catalog] {len(failed_cats)} categories failed during BFS; retrying...")
        n_new = retry_failed_categories(failed_cats, api_key, bucket, cats)
        print(f"[catalog] retry recovered {n_new} new categories")

    series_by_id: dict[str, dict[str, Any]] = {}
    series_categories: dict[str, list[int]] = defaultdict(list)
    failed_series_cats: list[int] = []
    n_processed = 0
    t1 = time.time()
    leaf_cats = [cid for cid in cats if cid != 0]
    for cid in leaf_cats:
        items = fetch_series_for_category(api_key, bucket, cid)
        if items is None:
            failed_series_cats.append(cid)
            continue
        for s in items:
            sid = s["id"]
            if sid not in series_by_id:
                series_by_id[sid] = s
            series_categories[sid].append(cid)
        n_processed += 1
        if on_progress and n_processed % 50 == 0:
            on_progress(n_processed, len(leaf_cats), len(series_by_id))
    if failed_series_cats:
        print(f"[catalog] retrying {len(failed_series_cats)} failed series fetches...")
        for cid in failed_series_cats:
            items = fetch_series_for_category(api_key, bucket, cid)
            if items is None:
                continue
            for s in items:
                sid = s["id"]
                if sid not in series_by_id:
                    series_by_id[sid] = s
                series_categories[sid].append(cid)
    print(f"[catalog] discovered {len(series_by_id):,} unique series "
          f"across {n_processed:,} categories ({time.time() - t1:.0f}s)")

    print("[catalog] fetching releases, sources, tags...")
    releases = fetch_flat_list("/releases", "releases", api_key, bucket)
    sources = fetch_flat_list("/sources", "sources", api_key, bucket)
    tags = fetch_flat_list("/tags", "tags", api_key, bucket)
    print(f"[catalog] {len(releases)} releases, {len(sources)} sources, {len(tags):,} tags")

    cats_df = pd.DataFrame.from_records([
        {"id": c["id"], "name": c["name"], "parent_id": c["parent_id"]}
        for c in cats.values()
    ]).sort_values("id").reset_index(drop=True)

    rows = []
    for sid, meta in series_by_id.items():
        cids = series_categories.get(sid, [])
        roots = []
        paths = []
        for cid in cids:
            p = category_path(cats, cid)
            if p:
                if len(p) > 1:
                    roots.append(p[1])  # skip ROOT, take top-level
                paths.append(" / ".join(p[1:]))
        rows.append({
            "id": sid,
            "title": meta.get("title", ""),
            "frequency": meta.get("frequency", ""),
            "frequency_short": meta.get("frequency_short", ""),
            "units": meta.get("units", ""),
            "units_short": meta.get("units_short", ""),
            "seasonal_adjustment": meta.get("seasonal_adjustment", ""),
            "seasonal_adjustment_short": meta.get("seasonal_adjustment_short", ""),
            "observation_start": meta.get("observation_start", ""),
            "observation_end": meta.get("observation_end", ""),
            "last_updated": meta.get("last_updated", ""),
            "popularity": int(meta.get("popularity", 0) or 0),
            "group_popularity": int(meta.get("group_popularity", 0) or 0),
            "notes": meta.get("notes", "") or "",
            "realtime_start": meta.get("realtime_start", ""),
            "realtime_end": meta.get("realtime_end", ""),
            "category_ids": json.dumps(cids),
            "category_root": roots[0] if roots else "Other",
            "category_paths": " | ".join(paths) if paths else "",
        })
    series_df = pd.DataFrame(rows).sort_values("popularity", ascending=False).reset_index(drop=True)

    releases_df = pd.DataFrame(releases) if releases else pd.DataFrame()
    sources_df = pd.DataFrame(sources) if sources else pd.DataFrame()
    tags_df = pd.DataFrame(tags) if tags else pd.DataFrame()

    return {
        "categories": cats_df,
        "series": series_df,
        "releases": releases_df,
        "sources": sources_df,
        "tags": tags_df,
    }


def save_catalog(dfs: dict[str, pd.DataFrame], out_dir: Path = CATALOG_DIR) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, df in dfs.items():
        out = out_dir / f"{name}.parquet"
        tmp = out.with_suffix(".parquet.tmp")
        df.to_parquet(tmp, index=False)
        tmp.replace(out)
        paths[name] = out
    _write_index_md(out_dir, dfs)
    return paths


def _write_index_md(out_dir: Path, dfs: dict[str, pd.DataFrame]) -> None:
    cats = dfs["categories"]
    series = dfs["series"]
    lines = ["# FRED catalog index", ""]
    lines.append(f"- categories: {len(cats):,}")
    lines.append(f"- series: {len(series):,}")
    lines.append(f"- releases: {len(dfs['releases']):,}")
    lines.append(f"- sources: {len(dfs['sources']):,}")
    lines.append(f"- tags: {len(dfs['tags']):,}")
    lines.append("")
    if "category_root" in series.columns:
        lines.append("## Series by category root")
        counts = series["category_root"].value_counts().head(20)
        for name, n in counts.items():
            lines.append(f"- {name}: {n:,}")
        lines.append("")
    if "frequency_short" in series.columns:
        lines.append("## Series by frequency")
        for name, n in series["frequency_short"].value_counts().items():
            lines.append(f"- {name or '(blank)'}: {n:,}")
    (out_dir / "INDEX.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
