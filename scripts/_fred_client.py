"""HTTP client for the FRED API — stdlib only (urllib + json).

Endpoints used:
    /fred/category/children, /fred/category/series, /fred/category
    /fred/series, /fred/series/observations
    /fred/releases, /fred/sources, /fred/tags

The client handles:
    * shared token-bucket rate limiting
    * exponential retry on 429/5xx and network errors
    * pagination via offset/limit
    * content-type guard (HTML error pages → retry/error)
    * FRED's "." missing-value sentinel (caller decides what to do)
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Iterator, Protocol


FRED_BASE = "https://api.stlouisfed.org/fred"
DEFAULT_TIMEOUT = 30.0
RETRY_DELAYS = (0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 45.0, 60.0, 60.0, 60.0)
RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504}
DNS_RESET_INTERVAL = 60.0  # force getaddrinfo refresh between connections


class Bucket(Protocol):
    def acquire(self) -> None: ...


class FredError(RuntimeError):
    pass


class FredHTTPError(FredError):
    def __init__(self, status: int, body: str, url: str) -> None:
        super().__init__(f"FRED HTTP {status}: {body[:200]}")
        self.status = status
        self.body = body
        self.url = url


def _build_url(path: str, params: dict[str, Any]) -> str:
    qs = urllib.parse.urlencode({k: v for k, v in params.items() if v is not None})
    return f"{FRED_BASE}{path}?{qs}"


def request_json(
    path: str,
    params: dict[str, Any],
    api_key: str,
    bucket: Bucket | None = None,
    timeout: float = DEFAULT_TIMEOUT,
    max_attempts: int = 12,
) -> dict[str, Any]:
    p = dict(params)
    p.setdefault("file_type", "json")
    p["api_key"] = api_key
    url = _build_url(path, p)

    attempt = 0
    while True:
        if bucket is not None:
            bucket.acquire()
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "fx_strategies/1.0"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                ctype = resp.headers.get("Content-Type", "")
                body = resp.read().decode("utf-8", errors="replace")
                if "json" not in ctype.lower():
                    raise FredHTTPError(resp.status, f"non-json content-type: {ctype} body={body[:120]}", url)
                return json.loads(body)
        except urllib.error.HTTPError as e:
            body = ""
            try:
                body = e.read().decode("utf-8", errors="replace")
            except Exception:
                pass
            if e.code in RETRYABLE_STATUS and attempt < max_attempts - 1:
                delay = RETRY_DELAYS[min(attempt, len(RETRY_DELAYS) - 1)]
                time.sleep(delay)
                attempt += 1
                continue
            raise FredHTTPError(e.code, body, url) from e
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
            if attempt < max_attempts - 1:
                delay = RETRY_DELAYS[min(attempt, len(RETRY_DELAYS) - 1)]
                time.sleep(delay)
                attempt += 1
                continue
            raise FredError(f"network error after {attempt + 1} attempts: {e}") from e


def paginate(
    path: str,
    params: dict[str, Any],
    api_key: str,
    list_key: str,
    bucket: Bucket | None = None,
    page_size: int = 1000,
    max_pages: int | None = None,
) -> Iterator[dict[str, Any]]:
    """Yield items from a paginated list endpoint (e.g. /fred/category/series).

    FRED list endpoints accept ``limit`` (max 1000) and ``offset``. Total count
    is reported as ``count`` in the response body.
    """
    offset = 0
    pages = 0
    while True:
        page_params = dict(params)
        page_params["limit"] = page_size
        page_params["offset"] = offset
        data = request_json(path, page_params, api_key, bucket=bucket)
        items = data.get(list_key, [])
        for item in items:
            yield item
        count = int(data.get("count", 0))
        offset += len(items)
        pages += 1
        if not items or offset >= count:
            return
        if max_pages is not None and pages >= max_pages:
            return


def fetch_observations(
    series_id: str,
    api_key: str,
    bucket: Bucket | None = None,
    observation_start: str | None = None,
    observation_end: str | None = None,
    page_size: int = 100_000,
) -> list[dict[str, Any]]:
    """Fetch all observations for a series, following pagination."""
    params: dict[str, Any] = {"series_id": series_id, "sort_order": "asc"}
    if observation_start:
        params["observation_start"] = observation_start
    if observation_end:
        params["observation_end"] = observation_end
    return list(paginate("/series/observations", params, api_key,
                         list_key="observations", bucket=bucket, page_size=page_size))


def fetch_series_metadata(series_id: str, api_key: str,
                          bucket: Bucket | None = None) -> dict[str, Any]:
    data = request_json("/series", {"series_id": series_id}, api_key, bucket=bucket)
    items = data.get("seriess", [])
    if not items:
        raise FredError(f"no metadata for series {series_id}")
    return items[0]
