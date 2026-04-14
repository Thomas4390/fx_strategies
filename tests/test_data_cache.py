"""Tests for framework.data_cache — sleeve cache + manifest invalidation.

The cache is tested with synthetic sleeve dicts (fast) and a manifest
file redirected to a tmp_path so the real project manifest is never
touched. The final reproducibility test (cold → warm returns the same
numbers) is covered by the end-to-end run at module import time in
``scripts/compute_dsr.py``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


from framework import data_cache  # noqa: E402


# ═══════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════


@pytest.fixture
def tmp_cache(tmp_path, monkeypatch) -> dict[str, Path]:
    """Redirect the data_cache module-level paths to a tmp directory.

    Returns the redirected paths as a dict so tests can inspect them.
    """
    manifest_path = tmp_path / "MANIFEST.json"
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = cache_dir / "strategy_daily_returns.parquet"
    key_path = cache_dir / "strategy_daily_returns.key.json"

    monkeypatch.setattr(data_cache, "_MANIFEST_PATH", manifest_path)
    monkeypatch.setattr(data_cache, "_CACHE_DIR", cache_dir)
    monkeypatch.setattr(data_cache, "_CACHE_PARQUET", parquet_path)
    monkeypatch.setattr(data_cache, "_CACHE_KEY_JSON", key_path)

    return {
        "manifest": manifest_path,
        "cache_dir": cache_dir,
        "parquet": parquet_path,
        "key": key_path,
    }


def _write_manifest(path: Path, fingerprint: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": "2026-04-13T00:00:00+00:00",
        "files": {},
        "combined_fingerprint": fingerprint,
    }
    path.write_text(json.dumps(payload))


def _synthetic_sleeves(seed: int = 0) -> dict[str, pd.Series]:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=500, freq="B")
    return {
        "MR_Macro": pd.Series(rng.normal(0.0005, 0.01, 500), index=idx),
        "TS_Momentum_3p": pd.Series(rng.normal(0.0003, 0.012, 500), index=idx),
        "RSI_Daily_4p": pd.Series(rng.normal(0.0002, 0.009, 500), index=idx),
    }


# ═══════════════════════════════════════════════════════════════════════
# Manifest fingerprint plumbing
# ═══════════════════════════════════════════════════════════════════════


def test_missing_manifest_raises_specific_error(tmp_cache):
    """No manifest file → ManifestMissingError from the private reader."""
    with pytest.raises(data_cache.ManifestMissingError):
        data_cache._read_manifest_fingerprint()


def test_manifest_without_fingerprint_raises(tmp_cache):
    """Manifest file with empty fingerprint → plain RuntimeError."""
    tmp_cache["manifest"].write_text(json.dumps({"files": {}}))
    with pytest.raises(RuntimeError, match="combined_fingerprint"):
        data_cache._read_manifest_fingerprint()


def test_cache_key_changes_with_fingerprint():
    k1 = data_cache._cache_key("fingerprint-A", "v1")
    k2 = data_cache._cache_key("fingerprint-B", "v1")
    k3 = data_cache._cache_key("fingerprint-A", "v2")
    assert k1 != k2
    assert k1 != k3
    assert len(k1) == 64  # SHA256 hex


# ═══════════════════════════════════════════════════════════════════════
# Cache miss → rebuild → write
# ═══════════════════════════════════════════════════════════════════════


def test_missing_manifest_falls_back_to_rebuild(tmp_cache):
    """When MANIFEST.json is missing, cached_* must still call rebuild."""
    call_count = {"n": 0}

    def _rebuild() -> dict[str, pd.Series]:
        call_count["n"] += 1
        return _synthetic_sleeves()

    result = data_cache.cached_strategy_daily_returns(_rebuild, verbose=False)
    assert call_count["n"] == 1
    assert set(result) == {"MR_Macro", "TS_Momentum_3p", "RSI_Daily_4p"}
    # No parquet/key file written on the fallback path.
    assert not tmp_cache["parquet"].exists()
    assert not tmp_cache["key"].exists()


def test_cold_call_writes_parquet_and_key(tmp_cache):
    _write_manifest(tmp_cache["manifest"], "fingerprint-xyz")
    expected = _synthetic_sleeves()

    def _rebuild() -> dict[str, pd.Series]:
        return expected

    result = data_cache.cached_strategy_daily_returns(_rebuild, verbose=False)
    assert tmp_cache["parquet"].exists()
    assert tmp_cache["key"].exists()
    # Content matches what the rebuild returned.
    for k in expected:
        np.testing.assert_array_equal(
            result[k].values, expected[k].values, err_msg=f"mismatch {k}"
        )


def test_warm_call_hits_cache_and_skips_rebuild(tmp_cache):
    _write_manifest(tmp_cache["manifest"], "fingerprint-xyz")
    expected = _synthetic_sleeves(seed=1)

    call_count = {"n": 0}

    def _rebuild() -> dict[str, pd.Series]:
        call_count["n"] += 1
        return expected

    # Cold: rebuild once, writes cache.
    data_cache.cached_strategy_daily_returns(_rebuild, verbose=False)
    assert call_count["n"] == 1

    # Warm: same fingerprint → no rebuild.
    result = data_cache.cached_strategy_daily_returns(_rebuild, verbose=False)
    assert call_count["n"] == 1
    for k in expected:
        np.testing.assert_array_equal(result[k].values, expected[k].values)


def test_fingerprint_change_invalidates_cache(tmp_cache):
    _write_manifest(tmp_cache["manifest"], "fingerprint-old")
    call_count = {"n": 0}

    def _rebuild() -> dict[str, pd.Series]:
        call_count["n"] += 1
        return _synthetic_sleeves(seed=call_count["n"])

    data_cache.cached_strategy_daily_returns(_rebuild, verbose=False)
    assert call_count["n"] == 1

    # Rewrite manifest with a new fingerprint → cache must miss.
    _write_manifest(tmp_cache["manifest"], "fingerprint-new")
    data_cache.cached_strategy_daily_returns(_rebuild, verbose=False)
    assert call_count["n"] == 2


def test_sleeves_version_bump_invalidates_cache(tmp_cache, monkeypatch):
    _write_manifest(tmp_cache["manifest"], "fingerprint-xyz")
    call_count = {"n": 0}

    def _rebuild() -> dict[str, pd.Series]:
        call_count["n"] += 1
        return _synthetic_sleeves(seed=call_count["n"])

    data_cache.cached_strategy_daily_returns(_rebuild, verbose=False)
    assert call_count["n"] == 1

    # Bump the sleeves version at the module level → cache miss.
    monkeypatch.setattr(data_cache, "_SLEEVES_VERSION", "v2-experimental")
    data_cache.cached_strategy_daily_returns(_rebuild, verbose=False)
    assert call_count["n"] == 2


# ═══════════════════════════════════════════════════════════════════════
# Cache API helpers
# ═══════════════════════════════════════════════════════════════════════


def test_clear_cache_removes_files(tmp_cache):
    _write_manifest(tmp_cache["manifest"], "fingerprint-xyz")
    data_cache.cached_strategy_daily_returns(
        lambda: _synthetic_sleeves(), verbose=False
    )
    assert tmp_cache["parquet"].exists()
    assert tmp_cache["key"].exists()

    data_cache.clear_cache()
    assert not tmp_cache["parquet"].exists()
    assert not tmp_cache["key"].exists()


def test_cache_info_reports_state(tmp_cache):
    info = data_cache.cache_info()
    assert info["parquet_exists"] is False
    assert info["key_file_exists"] is False
    assert info["key_payload"] is None

    _write_manifest(tmp_cache["manifest"], "fingerprint-xyz")
    data_cache.cached_strategy_daily_returns(
        lambda: _synthetic_sleeves(), verbose=False
    )

    info2 = data_cache.cache_info()
    assert info2["parquet_exists"] is True
    assert info2["key_file_exists"] is True
    assert info2["key_payload"]["manifest_fingerprint"] == "fingerprint-xyz"
    assert info2["key_payload"]["n_sleeves"] == 3


def test_corrupt_key_file_forces_rebuild(tmp_cache):
    _write_manifest(tmp_cache["manifest"], "fingerprint-xyz")
    data_cache.cached_strategy_daily_returns(
        lambda: _synthetic_sleeves(), verbose=False
    )

    # Corrupt the key file.
    tmp_cache["key"].write_text("not-valid-json {")

    call_count = {"n": 0}

    def _rebuild() -> dict[str, pd.Series]:
        call_count["n"] += 1
        return _synthetic_sleeves()

    data_cache.cached_strategy_daily_returns(_rebuild, verbose=False)
    assert call_count["n"] == 1
