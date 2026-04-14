"""Refactor invariance tests for the VBT native adoption audit.

Uses ``tests.helpers.portfolio_equivalence`` to assert that planned
refactors (see ``reports/audits/vbt_native_audit.md``) do not alter
portfolio outputs at machine precision.

Two layers:

1. **Helper self-tests** — fast, no data. Verify the fingerprint /
   assertion utility works on synthetic portfolios before we rely on
   it for real refactor validation.

2. **Pipeline determinism** — run each strategy pipeline twice in the
   same process and assert the fingerprints are bit-identical. This is
   the precondition for any refactor: if a pipeline is non-deterministic
   (rare but possible with parallel execution, hash ordering, or
   uninitialized state), no equivalence test can ever succeed.

3. **Coverage-gap snapshot** — ``composite_fx_alpha.pipeline`` is NOT
   covered by ``test_pipeline_equivalence.py``. Add a stats snapshot
   here following the same JSON-on-disk pattern, but scoped under
   ``tests/snapshots_refactor/`` so it stays out of the way of the
   legacy-migration snapshots.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import vectorbtpro as vbt

_TESTS_DIR = Path(__file__).resolve().parent
_SRC_DIR = _TESTS_DIR.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from tests.helpers.portfolio_equivalence import (  # noqa: E402
    PortfolioFingerprint,
    _hash_returns,
    assert_fingerprints_equivalent,
    assert_portfolio_equivalent,
    assert_returns_equivalent,
    diff_fingerprints,
    dump_fingerprint,
    fingerprint,
    load_fingerprint,
)

_REFACTOR_SNAPSHOT_DIR = _TESTS_DIR / "snapshots_refactor"


# ═══════════════════════════════════════════════════════════════════════
# 1. Helper self-tests — synthetic portfolios, no real data
# ═══════════════════════════════════════════════════════════════════════


@pytest.fixture(scope="module")
def synthetic_pf():
    """Deterministic toy portfolio — 250 bars, fixed RNG seed."""
    rng = np.random.default_rng(42)
    n = 250
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    close = pd.Series(
        100 * np.cumprod(1 + rng.normal(0.0002, 0.01, n)),
        index=idx,
        name="close",
    )
    entries = pd.Series(rng.random(n) < 0.05, index=idx)
    exits = pd.Series(rng.random(n) < 0.05, index=idx)
    return vbt.Portfolio.from_signals(
        close=close,
        entries=entries,
        exits=exits,
        init_cash=10_000.0,
        freq="1D",
    )


def test_fingerprint_extracts_all_fields(synthetic_pf):
    fp = fingerprint(synthetic_pf)
    assert isinstance(fp, PortfolioFingerprint)
    assert fp.n_rows == 250
    assert fp.num_orders >= 0
    assert fp.num_trades >= 0
    assert len(fp.returns_hash) == 16
    # Sharpe may be NaN for a toy — that's fine, we just check it's coerced
    assert isinstance(fp.sharpe_ratio, float)


def test_fingerprint_is_deterministic(synthetic_pf):
    """Calling fingerprint() twice on the same pf must give identical results."""
    fp1 = fingerprint(synthetic_pf)
    fp2 = fingerprint(synthetic_pf)
    assert_fingerprints_equivalent(fp1, fp2, label="self-equivalence")
    assert fp1.returns_hash == fp2.returns_hash


def test_assert_portfolio_equivalent_passes_on_self(synthetic_pf):
    """Trivial: a portfolio must be equivalent to itself."""
    assert_portfolio_equivalent(synthetic_pf, synthetic_pf, label="self")


def test_assert_portfolio_equivalent_detects_scalar_drift(synthetic_pf):
    """A synthetic drift in the returns must produce a loud failure."""
    fp_ref = fingerprint(synthetic_pf)
    # Forge a fingerprint with a small Sharpe drift beyond atol
    drifted = PortfolioFingerprint(
        **{**fp_ref.to_dict(), "sharpe_ratio": fp_ref.sharpe_ratio + 1e-6},
    )
    diffs = diff_fingerprints(fp_ref, drifted)
    assert len(diffs) == 1
    assert "sharpe_ratio" in diffs[0]


def test_assert_portfolio_equivalent_detects_categorical_drift(synthetic_pf):
    """A one-trade change must fail hard, regardless of tolerance."""
    fp_ref = fingerprint(synthetic_pf)
    drifted = PortfolioFingerprint(
        **{**fp_ref.to_dict(), "num_trades": fp_ref.num_trades + 1},
    )
    with pytest.raises(AssertionError, match="num_trades"):
        assert_fingerprints_equivalent(
            fp_ref, drifted, rtol=1e-3, atol=1e-3, label="synthetic"
        )


def test_returns_hash_stable_under_identical_input():
    idx = pd.date_range("2024-01-01", periods=10, freq="D")
    s = pd.Series([0.01, -0.02, 0.0, np.nan, 0.005, 0.0, -0.01, 0.02, 0.003, -0.004], index=idx)
    h1 = _hash_returns(s)
    h2 = _hash_returns(s.copy())
    assert h1 == h2


def test_returns_hash_sensitive_to_tiny_change():
    idx = pd.date_range("2024-01-01", periods=10, freq="D")
    s1 = pd.Series(np.ones(10) * 0.01, index=idx)
    # Sub-rounding perturbation: 1e-14 is below 0.5e-12 so round(x, 12)
    # absorbs it → hash is stable. Confirms we don't fire on FP noise.
    s2 = s1.copy()
    s2.iloc[5] += 1e-14
    assert _hash_returns(s1) == _hash_returns(s2)
    # Supra-rounding perturbation: 1e-9 is clearly above rounding → hash must differ
    s3 = s1.copy()
    s3.iloc[5] += 1e-9
    assert _hash_returns(s1) != _hash_returns(s3)


def test_assert_returns_equivalent_catches_first_divergence():
    idx = pd.date_range("2024-01-01", periods=5, freq="D")
    a = pd.Series([0.01, 0.02, 0.03, 0.04, 0.05], index=idx)
    b = a.copy()
    b.iloc[2] = 0.031  # clearly beyond tolerance
    with pytest.raises(AssertionError, match="idx=2"):
        assert_returns_equivalent(a, b, label="synthetic_drift")


def test_dump_and_load_fingerprint_roundtrip(synthetic_pf, tmp_path):
    fp = fingerprint(synthetic_pf)
    path = tmp_path / "fp.json"
    dump_fingerprint(fp, path)
    loaded = load_fingerprint(path)
    assert_fingerprints_equivalent(fp, loaded, label="roundtrip")


# ═══════════════════════════════════════════════════════════════════════
# 2. Pipeline determinism — shared fixture with the equivalence tests
# ═══════════════════════════════════════════════════════════════════════


@pytest.fixture(scope="module")
def fx_data():
    """Full EUR-USD dataset used by the minute-level strategies.

    Mirrors the fixture in ``test_pipeline_equivalence.py``. Duplicated
    (not imported) to keep these two test files independent.
    """
    from utils import apply_vbt_settings, load_fx_data

    apply_vbt_settings()
    _, data = load_fx_data("data/EUR-USD_minute.parquet")
    return data


def test_mr_turbo_pipeline_is_deterministic(fx_data):
    """Precondition for refactor: pipeline(data) must give identical
    fingerprints across two calls in the same process."""
    from strategies.mr_turbo import pipeline

    pf1, _ = pipeline(fx_data, bb_window=80, bb_alpha=5.0)
    pf2, _ = pipeline(fx_data, bb_window=80, bb_alpha=5.0)
    assert_portfolio_equivalent(pf1, pf2, label="mr_turbo_determinism")


def test_ou_mean_reversion_pipeline_is_deterministic(fx_data):
    from strategies.ou_mean_reversion import pipeline

    pf1, _ = pipeline(fx_data, bb_window=80, bb_alpha=5.0)
    pf2, _ = pipeline(fx_data, bb_window=80, bb_alpha=5.0)
    assert_portfolio_equivalent(pf1, pf2, label="ou_mr_determinism")


def test_composite_fx_alpha_pipeline_is_deterministic(fx_data):
    """Coverage-gap strategy: NOT present in test_pipeline_equivalence.py.

    The composite_fx_alpha pipeline has 5 @njit kernels and is the
    strategy with the highest raw numpy footprint in the codebase.
    This test provides the minimum-viable determinism baseline so any
    planned refactor (see audit R-KEEP-LEGACY for the kernels) can be
    validated against a fixed reference.
    """
    from strategies.composite_fx_alpha import pipeline

    pf1, _ = pipeline(fx_data)
    pf2, _ = pipeline(fx_data)
    assert_portfolio_equivalent(pf1, pf2, label="composite_fx_alpha_determinism")


# ═══════════════════════════════════════════════════════════════════════
# 3. Coverage-gap snapshot — composite_fx_alpha fingerprint on disk
# ═══════════════════════════════════════════════════════════════════════


def _snapshot_path(name: str) -> Path:
    return _REFACTOR_SNAPSHOT_DIR / f"{name}.json"


def _fingerprint_or_create(pf, path: Path, name: str) -> PortfolioFingerprint:
    """Disk-baseline helper: if snapshot exists, compare; else create
    and skip with a message telling the user to re-run."""
    fp = fingerprint(pf)
    if not path.exists():
        dump_fingerprint(fp, path)
        pytest.skip(
            f"Baseline {name} created at {path.relative_to(_TESTS_DIR.parent)}. "
            "Re-run this test to validate against it."
        )
    expected = load_fingerprint(path)
    assert_fingerprints_equivalent(expected, fp, label=name)
    return fp


def test_composite_fx_alpha_fingerprint_baseline(fx_data):
    """Disk-baseline test for composite_fx_alpha.pipeline.

    First run: creates the snapshot under tests/snapshots_refactor/
    and skips. Subsequent runs: asserts the fingerprint is unchanged.
    Use this to validate any refactor of composite_fx_alpha against a
    locked reference. Regenerate by deleting the snapshot file and
    re-running.
    """
    from strategies.composite_fx_alpha import pipeline

    pf, _ = pipeline(fx_data)
    _fingerprint_or_create(
        pf,
        _snapshot_path("composite_fx_alpha_default"),
        "composite_fx_alpha_default",
    )


# ═══════════════════════════════════════════════════════════════════════
# 4. Refactor unit tests — new helpers bit-equivalent to inline patterns
# ═══════════════════════════════════════════════════════════════════════


def test_vol_target_leverage_bit_equivalent_to_inline_pattern():
    """``framework.leverage.vol_target_leverage`` must produce exactly the
    same output as the inline ``(target/vol.clip).clip.shift.fillna``
    pattern it replaces across 4 sites in daily_momentum.py.

    Tested on a synthetic vol series with deliberately tricky values:
    zero (→ vol_floor clamp), NaN at warmup, tight cap, and a spike
    above max_leverage.
    """
    from framework.leverage import vol_target_leverage

    idx = pd.date_range("2024-01-01", periods=30, freq="D")
    vol = pd.Series(
        [np.nan] * 5  # rolling_std warmup
        + [0.05, 0.10, 0.15, 0.20, 0.25]
        + [0.0, 0.005]  # will be clamped by vol_floor=0.01
        + [0.08] * 13
        + [2.0] * 5,  # tiny target/vol → hits max_leverage clip
        index=idx,
    )

    for target_vol, max_leverage in [
        (0.10, 5.0),  # XS momentum config
        (0.10, 3.0),  # TS momentum config
        (0.20, 2.0),  # stress config
    ]:
        inline = (
            (target_vol / vol.clip(lower=0.01))
            .clip(upper=max_leverage)
            .shift(1)
            .fillna(1.0)
        )
        helper = vol_target_leverage(
            vol, target_vol=target_vol, max_leverage=max_leverage
        )
        label = f"target={target_vol} max_lev={max_leverage}"
        assert_returns_equivalent(inline, helper, label=label)


def test_vol_target_leverage_custom_default():
    """Non-default ``default`` value must propagate correctly."""
    from framework.leverage import vol_target_leverage

    idx = pd.date_range("2024-01-01", periods=5, freq="D")
    vol = pd.Series([np.nan, 0.10, 0.10, 0.10, 0.10], index=idx)
    result = vol_target_leverage(
        vol, target_vol=0.10, max_leverage=3.0, default=0.5
    )
    assert result.iloc[0] == 0.5  # shift-out-of-bounds → default
    assert result.iloc[1] == 0.5  # shift(1) of NaN → NaN → default
    assert result.iloc[2] == 1.0  # normal case: 0.10 / 0.10 = 1.0


def test_vbt_fshift_vs_shift_fillna_semantics():
    """Document the semantic difference so future readers know when it
    is safe to replace ``.shift(1).fillna(v)`` with ``.vbt.fshift(1,
    fill_value=v)``.

    Rule: ``.vbt.fshift(1, fill_value=v)`` only fills the position
    vacated by the shift (index 0). ``.shift(1).fillna(v)`` fills
    index 0 AND any pre-existing NaN in the input. The two are strictly
    equivalent ONLY when the input contains no NaN.
    """
    idx = pd.date_range("2024-01-01", periods=6, freq="D")

    # Case 1: no NaN in input → strict equivalence.
    s = pd.Series([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], index=idx)
    inline = s.shift(1).fillna(0.0)
    fshift = s.vbt.fshift(1, fill_value=0.0)
    assert_returns_equivalent(inline, fshift, label="no_nan_input")

    # Case 2: NaN in the middle → diverges. Inline replaces the middle
    # NaN with 0.0; fshift preserves it. The assertion below documents
    # that the refactor is UNSAFE for such inputs.
    s_nan = pd.Series([0.1, np.nan, 0.3, 0.4, 0.5, 0.6], index=idx)
    inline_nan = s_nan.shift(1).fillna(0.0)
    fshift_nan = s_nan.vbt.fshift(1, fill_value=0.0)
    # Position 2: inline = 0.0 (fillna), fshift = NaN (preserved)
    assert inline_nan.iloc[2] == 0.0
    assert np.isnan(fshift_nan.iloc[2])
