"""Process-shared token bucket for FRED rate limiting.

FRED docs state 120 req/min/IP. We target 110/min for safety margin.
"""
from __future__ import annotations

import time
from multiprocessing.managers import SyncManager


class TokenBucket:
    """Evenly-spaced rate limiter shared across processes via Manager.

    Workers call ``acquire()`` before each FRED HTTP request. The call blocks
    until the bucket has a token, guaranteeing the global rate stays under
    ``rpm`` requests per minute regardless of worker count.
    """

    def __init__(self, manager: SyncManager, rpm: float = 110.0) -> None:
        self._lock = manager.Lock()
        self._next_allowed = manager.Value("d", 0.0)
        self._interval = 60.0 / float(rpm)
        self._counter = manager.Value("i", 0)

    def acquire(self) -> None:
        with self._lock:
            now = time.monotonic()
            wait = self._next_allowed.value - now
            if wait > 0:
                time.sleep(wait)
                now += wait
            self._next_allowed.value = now + self._interval
            self._counter.value += 1

    @property
    def total_acquired(self) -> int:
        return int(self._counter.value)


class LocalTokenBucket:
    """Single-process variant for --no-multiprocessing debug runs."""

    def __init__(self, rpm: float = 110.0) -> None:
        self._next_allowed = 0.0
        self._interval = 60.0 / float(rpm)
        self._counter = 0

    def acquire(self) -> None:
        now = time.monotonic()
        wait = self._next_allowed - now
        if wait > 0:
            time.sleep(wait)
            now += wait
        self._next_allowed = now + self._interval
        self._counter += 1

    @property
    def total_acquired(self) -> int:
        return self._counter
