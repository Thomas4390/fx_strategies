"""Append-only registry of how many configurations have been tested.

Deflated Sharpe is only as honest as the ``n_trials`` fed to it, and that
count is the one thing a research repo forgets first. Log the trials
**before** computing ``dsr_for_sweep_top`` (or any other deflation) : the
registry total is the honest ``n_trials`` bound, not the size of the sweep
you happen to be running right now.

    from framework import trials

    trials.log_trials("gold_sizing", 24, note="sizing regimes, phase 22",
                      config_key="gold_sizing:24regimes:v1")
    n = trials.total_trials()          # raw bound, re-runs included
    d = trials.distinct_trials()       # distinct configuration spaces

The file is a plain JSON list, append-only : entries are never rewritten,
because a trial that was run cannot be un-run. Single-writer assumption
(sweeps run sequentially) — writes are atomic via tmp + replace, no lock.

Re-running an identical sweep does not test anything new, yet the raw total
counts it twice — ``tsmom_universe`` was logged 7 times at n=21 for a single
21-configuration space. ``config_key`` names the *space of configurations
enumerated*, not the run : two entries sharing a key are re-runs, and
``distinct_trials`` folds them into one. Entries written before the field
existed are qualified after the fact by an **annotation** entry rather than
by a rewrite — the append-only guarantee holds, and ``total_trials`` keeps
returning what it always returned, so no published figure moves in silence.
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


_ANNOTATION = "annotation"


def _entry(
    family: str, n_configs: int, note: str, config_key: str | None = None
) -> dict:
    entry = {
        "ts": datetime.now(UTC).isoformat(),
        "family": family,
        "n": int(n_configs),
        "note": note,
    }
    if config_key is not None:
        entry["config_key"] = config_key
    return entry


def _is_sweep(entry: dict) -> bool:
    return entry.get("kind") != _ANNOTATION


def log_trials(
    family: str, n_configs: int, note: str = "", config_key: str | None = None
) -> None:
    """Append one sweep to the registry.

    ``config_key`` identifies the configuration space, so that re-running the
    same sweep does not inflate ``distinct_trials``. Omitting it keeps the
    legacy behaviour : the entry counts on its own.
    """
    entries = _read()
    entries.append(_entry(family, n_configs, note, config_key))
    _write(entries)
    logger.info("trials: +%d for %s (total %d)", n_configs, family, total_trials())


def annotate_config_key(target_ts: str, config_key: str, note: str = "") -> None:
    """Qualify an already-logged entry with a ``config_key``, append-only.

    The target entry is never rewritten : the annotation is a new record that
    ``distinct_trials`` folds in at read time. Used to mark the re-runs that
    predate the field.
    """
    entries = _read()
    if not any(e["ts"] == target_ts for e in entries if _is_sweep(e)):
        raise ValueError(f"no logged sweep at ts={target_ts!r}")
    entries.append(
        {
            "ts": datetime.now(UTC).isoformat(),
            "kind": _ANNOTATION,
            "target_ts": target_ts,
            "config_key": config_key,
            "note": note,
        }
    )
    _write(entries)


def _resolved_keys(entries: list[dict]) -> dict[str, str]:
    """Map ``ts`` -> ``config_key``, annotations overriding nothing in place."""
    keys = {
        e["ts"]: e["config_key"] for e in entries if _is_sweep(e) and "config_key" in e
    }
    for entry in entries:
        if entry.get("kind") == _ANNOTATION:
            keys[entry["target_ts"]] = entry["config_key"]
    return keys


def total_trials(family: str | None = None) -> int:
    """Sum of logged configurations, re-runs included — the conservative bound."""
    return sum(
        int(e["n"])
        for e in _read()
        if _is_sweep(e) and (family is None or e["family"] == family)
    )


def distinct_trials(family: str | None = None) -> int:
    """Sum over distinct configuration spaces : re-runs counted once.

    An entry with no ``config_key`` (and no annotation) counts on its own —
    absence of a key means "unqualified", never "same as some other sweep".
    """
    entries = _read()
    keys = _resolved_keys(entries)
    seen: set[str] = set()
    total = 0
    for entry in entries:
        if not _is_sweep(entry) or (family is not None and entry["family"] != family):
            continue
        key = keys.get(entry["ts"])
        if key is not None:
            if key in seen:
                continue
            seen.add(key)
        total += int(entry["n"])
    return total


def seed_registry() -> None:
    """Create the registry with the pre-policy trial count. Idempotent."""
    if REGISTRY_PATH.exists():
        return
    _write([_entry(_SEED_FAMILY, _SEED_N, _SEED_NOTE)])
    logger.info("trials: seeded registry at %s", REGISTRY_PATH)
