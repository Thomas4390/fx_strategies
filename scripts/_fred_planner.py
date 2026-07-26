"""Phase B planner — gap detection and prioritized work-queue building."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd

from _fred_paths import OBSERVATIONS_DIR, observation_path
from _macro_registry import REGISTRY as CURATED


@dataclass(frozen=True)
class WorkItem:
    series_id: str
    category_root: str
    frequency_short: str
    popularity: int
    last_updated_fred: str
    observation_end: str
    is_curated: bool
    metadata_snapshot: dict | None = None


def _parse_dt(s: str | None):
    if not s:
        return None
    try:
        return datetime.strptime(s[:10], "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except Exception:
        return None


def classify(series_id: str, catalog_row: pd.Series, progress_row: dict | None,
             observations_root: Path = OBSERVATIONS_DIR) -> str:
    """Return one of: missing, stale_obs, stale_fetch, done, error_pending, skipped."""
    parquet = observation_path(series_id, catalog_row.get("frequency_short"),
                               catalog_row.get("category_root"))
    p_status = (progress_row or {}).get("status")
    if p_status == "skipped_non_numeric":
        return "skipped"
    if not parquet.exists():
        return "missing"
    if p_status == "error":
        return "error_pending"
    last_updated_fred = _parse_dt(catalog_row.get("last_updated"))
    fetched_at = _parse_dt((progress_row or {}).get("fetched_at"))
    if last_updated_fred and fetched_at and last_updated_fred > fetched_at:
        return "stale_fetch"
    return "done"


def scan_gaps(catalog_df: pd.DataFrame, progress_index: dict[str, dict]) -> pd.DataFrame:
    """Compute per-series classification. ``progress_index`` is {series_id: row}."""
    rows = []
    for _, r in catalog_df.iterrows():
        sid = r["id"]
        cls = classify(sid, r, progress_index.get(sid))
        rows.append({
            "series_id": sid,
            "classification": cls,
            "popularity": int(r.get("popularity", 0) or 0),
            "frequency_short": r.get("frequency_short", ""),
            "category_root": r.get("category_root", "Other"),
            "last_updated": r.get("last_updated", ""),
            "observation_end": r.get("observation_end", ""),
        })
    return pd.DataFrame(rows)


def build_queue(catalog_df: pd.DataFrame, progress_index: dict[str, dict],
                *,
                top_popular: int | None = None,
                full: bool = False,
                series_filter: Iterable[str] | None = None,
                freq_filter: str | None = None,
                category_filter: str | None = None,
                include_curated: bool = True,
                retry_errors: bool = False,
                retry_skipped: bool = False) -> list[WorkItem]:
    """Build the ordered work queue for Phase B.

    Order: curated first (if not filtered out), then by popularity desc,
    then by last_updated desc.
    """
    cat = catalog_df.copy()
    cat["popularity"] = cat["popularity"].fillna(0).astype(int)

    if series_filter is not None:
        wanted = set(series_filter)
        cat = cat[cat["id"].isin(wanted)]
    elif full:
        pass
    elif top_popular is not None:
        cat = cat.nlargest(top_popular, "popularity")

    if freq_filter:
        cat = cat[cat["frequency_short"].str.upper() == freq_filter.upper()]
    if category_filter:
        cat = cat[cat["category_root"].str.contains(category_filter, case=False, na=False)]

    cat = cat.sort_values(
        ["popularity", "last_updated"], ascending=[False, False]
    ).reset_index(drop=True)

    curated_ids = {s.series_id for s in CURATED}

    queue: list[WorkItem] = []
    seen: set[str] = set()

    def _row_to_item(r: pd.Series, is_curated: bool) -> WorkItem:
        snapshot = {k: ("" if pd.isna(v) else v)
                    for k, v in r.to_dict().items()} if not r.empty else None
        return WorkItem(
            series_id=str(r.get("id", "")) if not r.empty else "",
            category_root=str(r.get("category_root", "Other")) if not r.empty else "Other",
            frequency_short=str(r.get("frequency_short", "")) if not r.empty else "",
            popularity=int(r.get("popularity", 0) or 0) if not r.empty else 0,
            last_updated_fred=str(r.get("last_updated", "")) if not r.empty else "",
            observation_end=str(r.get("observation_end", "")) if not r.empty else "",
            is_curated=is_curated,
            metadata_snapshot=snapshot,
        )

    if include_curated:
        for s in CURATED:
            if s.series_id in seen:
                continue
            row = catalog_df[catalog_df["id"] == s.series_id]
            if row.empty:
                queue.append(WorkItem(
                    series_id=s.series_id, category_root="Other",
                    frequency_short="", popularity=0,
                    last_updated_fred="", observation_end="",
                    is_curated=True, metadata_snapshot=None,
                ))
            else:
                queue.append(_row_to_item(row.iloc[0], True))
            seen.add(s.series_id)

    for _, r in cat.iterrows():
        sid = r["id"]
        if sid in seen:
            continue
        seen.add(sid)
        queue.append(_row_to_item(r, sid in curated_ids))

    filtered: list[WorkItem] = []
    for item in queue:
        prog = progress_index.get(item.series_id)
        status = (prog or {}).get("status")
        if status == "done":
            cls = classify(item.series_id, _row_for(catalog_df, item.series_id), prog)
            if cls == "done":
                continue
        if status == "skipped_non_numeric" and not retry_skipped:
            continue
        if status == "error" and not retry_errors:
            classification = classify(item.series_id, _row_for(catalog_df, item.series_id), prog)
            if classification != "missing" and classification != "stale_fetch":
                continue
        filtered.append(item)
    return filtered


def _row_for(catalog_df: pd.DataFrame, series_id: str) -> pd.Series:
    rows = catalog_df[catalog_df["id"] == series_id]
    if rows.empty:
        return pd.Series(dtype=object)
    return rows.iloc[0]
