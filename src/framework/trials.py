"""Append-only registry of how many configurations have been tested.

Deflated Sharpe is only as honest as the ``n_trials`` fed to it, and that
count is the one thing a research repo forgets first. Log the trials
**before** computing ``dsr_for_sweep_top`` (or any other deflation) : the
registry total is the honest ``n_trials`` bound, not the size of the sweep
you happen to be running right now.

    from framework import trials

    trials.log_trials("gold_sizing", 24, note="sizing regimes, phase 22")
    n = trials.total_trials()          # every family, the bound to deflate with

The file is a plain JSON list, append-only : entries are never rewritten,
because a trial that was run cannot be un-run. Single-writer assumption
(sweeps run sequentially) — writes are atomic via tmp + replace, no lock.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import UTC, datetime
from pathlib import Path

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
REGISTRY_PATH = _PROJECT_ROOT / "reports" / "research" / "trials_registry.json"

_SEED_FAMILY = "fx_legacy"
_SEED_N = 290
_SEED_NOTE = (
    "sweeps 2026-04 phases 18-21 sur les 4 paires FX "
    "(cf. docs/research/phase21_2026-04-13_dsr_retrofit.md)"
)


def _read() -> list[dict]:
    if not REGISTRY_PATH.exists():
        return []
    return json.loads(REGISTRY_PATH.read_text())


def _write(entries: list[dict]) -> None:
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = REGISTRY_PATH.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(entries, indent=2) + "\n")
    os.replace(tmp_path, REGISTRY_PATH)


def _entry(family: str, n_configs: int, note: str) -> dict:
    return {
        "ts": datetime.now(UTC).isoformat(),
        "family": family,
        "n": int(n_configs),
        "note": note,
    }


def log_trials(family: str, n_configs: int, note: str = "") -> None:
    """Append one sweep to the registry."""
    entries = _read()
    entries.append(_entry(family, n_configs, note))
    _write(entries)
    logger.info("trials: +%d for %s (total %d)", n_configs, family, total_trials())


def total_trials(family: str | None = None) -> int:
    """Sum of logged configurations, all families by default."""
    return sum(
        int(e["n"]) for e in _read() if family is None or e["family"] == family
    )


def seed_registry() -> None:
    """Create the registry with the pre-policy trial count. Idempotent."""
    if REGISTRY_PATH.exists():
        return
    _write([_entry(_SEED_FAMILY, _SEED_N, _SEED_NOTE)])
    logger.info("trials: seeded registry at %s", REGISTRY_PATH)
