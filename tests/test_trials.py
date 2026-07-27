"""Tests for framework.trials — append-only trial registry.

The registry under test is redirected to a tmp_path so the real
``reports/research/trials_registry.json`` is never touched ; the last test
reads the committed one read-only, to check the legacy seed is there.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from framework import trials  # noqa: E402

_REAL_REGISTRY = trials.REGISTRY_PATH


@pytest.fixture()
def tmp_registry(tmp_path, monkeypatch) -> Path:
    path = tmp_path / "research" / "trials_registry.json"
    monkeypatch.setattr(trials, "REGISTRY_PATH", path)
    return path


def test_seed_creates_the_legacy_entry(tmp_registry):
    trials.seed_registry()

    entries = json.loads(tmp_registry.read_text())
    assert len(entries) == 1
    assert entries[0]["family"] == "fx_legacy"
    assert entries[0]["n"] == 290
    assert "phase21" in entries[0]["note"]
    assert entries[0]["ts"].startswith("20")


def test_seed_is_idempotent(tmp_registry):
    trials.seed_registry()
    before = tmp_registry.read_text()
    trials.seed_registry()
    assert tmp_registry.read_text() == before


def test_log_trials_appends(tmp_registry):
    trials.seed_registry()
    trials.log_trials("gold_sizing", 24, note="sizing regimes")
    trials.log_trials("gold_sizing", 6)

    entries = json.loads(tmp_registry.read_text())
    assert [e["family"] for e in entries] == ["fx_legacy", "gold_sizing", "gold_sizing"]
    assert entries[1]["note"] == "sizing regimes"
    assert entries[2]["note"] == ""


def test_total_trials_sums_and_filters_by_family(tmp_registry):
    trials.seed_registry()
    trials.log_trials("gold_sizing", 24)
    trials.log_trials("gold_sizing", 6)

    assert trials.total_trials() == 320
    assert trials.total_trials("gold_sizing") == 30
    assert trials.total_trials("fx_legacy") == 290
    assert trials.total_trials("unknown_family") == 0


def test_total_trials_on_a_missing_registry_is_zero(tmp_registry):
    assert not tmp_registry.exists()
    assert trials.total_trials() == 0


def test_committed_registry_holds_the_fx_legacy_seed():
    entries = json.loads(_REAL_REGISTRY.read_text())
    legacy = [e for e in entries if e["family"] == "fx_legacy"]
    assert len(legacy) == 1
    assert legacy[0]["n"] == 290
