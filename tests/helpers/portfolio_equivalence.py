"""Bit-equivalence helper for vbt.Portfolio refactors.

Central utility used by ``tests/test_vbt_refactor_invariants.py`` and
other refactor-driven tests that need to assert ``pf_before == pf_after``
at machine precision.

The existing ``tests/test_pipeline_equivalence.py`` compares
``pf.stats()`` dicts with a float tolerance — good for the legacy →
pipeline migration but insufficient for fine-grained refactors where a
single bar shift can propagate silently through compounding without
moving Sharpe beyond 1e-10.

This helper adds three strictness levels:

1. **Categorical hard checks** — ``num_trades`` and ``num_orders`` must
   be strictly equal. Any drift here means a trade was created or
   destroyed, which is categorically a different strategy regardless
   of whether Sharpe moved.
2. **Returns hash** — a SHA-256 of the rounded returns series. Gives
   a single scalar answer to "did the return path change anywhere?"
3. **Elementwise tolerance** on scalar metrics and the returns series,
   with a default of ``atol=1e-12, rtol=1e-10`` (same as the existing
   snapshot tests).

See also: ``docs/vbt_native_cheatsheet.md`` §6 on ``cash_sharing``
which changes the semantics of ``pf.returns`` for grouped portfolios —
the helper handles grouped portfolios transparently because it reads
through the ``pf.returns`` / ``pf.sharpe_ratio`` properties which
respect grouping.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
import vectorbtpro as vbt


# ─────────────────────────────────────────────────────────────────────────
# Fingerprint
# ─────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class PortfolioFingerprint:
    """Compact signature of a vbt.Portfolio — enough to detect any
    refactor-induced drift without serializing the full object."""

    sharpe_ratio: float
    annualized_return: float
    annualized_volatility: float
    max_drawdown: float
    total_return: float
    final_value: float
    num_trades: int
    num_orders: int
    returns_hash: str
    n_rows: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "PortfolioFingerprint":
        return cls(**d)


def _safe_float(x: Any) -> float:
    """Coerce any scalar-like to float, mapping NaN → NaN."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return float("nan")
    return v


def _hash_returns(returns: pd.Series | pd.DataFrame, digits: int = 12) -> str:
    """SHA-256 of ``returns.round(digits).to_numpy().tobytes()``.

    Rounding to 12 digits absorbs the last ~3 bits of float64 noise
    without hiding genuine differences — the same tolerance used by
    the snapshot tests. A matching hash is a cryptographic guarantee
    that every rounded return is identical.
    """
    arr = returns.to_numpy()
    if arr.dtype != np.float64:
        arr = arr.astype(np.float64)
    rounded = np.round(np.nan_to_num(arr, nan=0.0), digits)
    return hashlib.sha256(rounded.tobytes()).hexdigest()[:16]


def fingerprint(pf: vbt.Portfolio) -> PortfolioFingerprint:
    """Compute a refactor-invariance fingerprint for a vbt.Portfolio.

    Reads through the native accessors (``sharpe_ratio``, ``returns``,
    ``trades.count()``, ``orders.count()``) so the result is correct
    for grouped / cash-sharing portfolios without special-casing.
    """
    returns = pf.returns
    return PortfolioFingerprint(
        sharpe_ratio=_safe_float(pf.sharpe_ratio),
        annualized_return=_safe_float(pf.annualized_return),
        annualized_volatility=_safe_float(pf.annualized_volatility),
        max_drawdown=_safe_float(pf.max_drawdown),
        total_return=_safe_float(pf.total_return),
        final_value=_safe_float(pf.final_value),
        num_trades=int(np.atleast_1d(pf.trades.count()).sum()),
        num_orders=int(np.atleast_1d(pf.orders.count()).sum()),
        returns_hash=_hash_returns(returns),
        n_rows=int(len(returns)),
    )


# ─────────────────────────────────────────────────────────────────────────
# Assertions
# ─────────────────────────────────────────────────────────────────────────


_DEFAULT_RTOL = 1e-10
_DEFAULT_ATOL = 1e-12

_SCALAR_FIELDS = (
    "sharpe_ratio",
    "annualized_return",
    "annualized_volatility",
    "max_drawdown",
    "total_return",
    "final_value",
)

_CATEGORICAL_FIELDS = ("num_trades", "num_orders", "n_rows")


def _isclose_or_nan(a: float, b: float, rtol: float, atol: float) -> bool:
    if np.isnan(a) and np.isnan(b):
        return True
    return bool(np.isclose(a, b, rtol=rtol, atol=atol))


def diff_fingerprints(
    before: PortfolioFingerprint,
    after: PortfolioFingerprint,
    *,
    rtol: float = _DEFAULT_RTOL,
    atol: float = _DEFAULT_ATOL,
) -> list[str]:
    """Return a list of human-readable divergence messages.

    Empty list means the two fingerprints are equivalent under the
    given tolerances. Used by ``assert_fingerprints_equivalent`` to
    build its error message, but also callable standalone for debug
    reporting.
    """
    diffs: list[str] = []

    # Categorical: hard equality
    for field in _CATEGORICAL_FIELDS:
        bv = getattr(before, field)
        av = getattr(after, field)
        if bv != av:
            diffs.append(f"{field}: {bv} → {av} (categorical drift — refactor is NOT bit-equivalent)")

    # Scalars: elementwise tolerance
    for field in _SCALAR_FIELDS:
        bv = getattr(before, field)
        av = getattr(after, field)
        if not _isclose_or_nan(bv, av, rtol, atol):
            abs_diff = abs(bv - av) if not (np.isnan(bv) or np.isnan(av)) else float("nan")
            diffs.append(
                f"{field}: {bv:.15g} → {av:.15g} (|diff|={abs_diff:.3e}, "
                f"tol=rtol={rtol:.1e} atol={atol:.1e})"
            )

    # Returns hash: cryptographic equality
    if before.returns_hash != after.returns_hash:
        diffs.append(
            f"returns_hash: {before.returns_hash} → {after.returns_hash} "
            "(at least one return differs beyond 1e-12 rounding)"
        )

    return diffs


def assert_fingerprints_equivalent(
    before: PortfolioFingerprint,
    after: PortfolioFingerprint,
    *,
    rtol: float = _DEFAULT_RTOL,
    atol: float = _DEFAULT_ATOL,
    label: str = "",
) -> None:
    """Raise AssertionError if two fingerprints disagree.

    ``label`` is prepended to the error message so tests comparing
    multiple pipelines produce localizable failures.
    """
    diffs = diff_fingerprints(before, after, rtol=rtol, atol=atol)
    if diffs:
        header = f"Portfolio fingerprint mismatch ({len(diffs)} field(s))"
        if label:
            header = f"[{label}] {header}"
        raise AssertionError(header + ":\n  - " + "\n  - ".join(diffs))


def assert_portfolio_equivalent(
    pf_before: vbt.Portfolio,
    pf_after: vbt.Portfolio,
    *,
    rtol: float = _DEFAULT_RTOL,
    atol: float = _DEFAULT_ATOL,
    label: str = "",
) -> None:
    """Top-level entry: compute fingerprints for both portfolios and
    assert equivalence.

    Use this for refactors that rebuild a ``vbt.Portfolio`` end-to-end
    (e.g. switching ``.shift(1).fillna(v)`` to ``.vbt.fshift``). For
    cases where the code under test returns a raw ``pd.Series`` of
    returns, use :func:`assert_returns_equivalent` instead.
    """
    fp_before = fingerprint(pf_before)
    fp_after = fingerprint(pf_after)
    assert_fingerprints_equivalent(
        fp_before, fp_after, rtol=rtol, atol=atol, label=label
    )


def assert_returns_equivalent(
    before: pd.Series | pd.DataFrame,
    after: pd.Series | pd.DataFrame,
    *,
    rtol: float = _DEFAULT_RTOL,
    atol: float = _DEFAULT_ATOL,
    label: str = "",
) -> None:
    """Elementwise equivalence for raw return Series / DataFrames.

    Used by the refactor target in ``daily_momentum.py`` where helpers
    like ``backtest_ts_momentum_rsi`` return a ``pd.Series`` rather than
    a Portfolio.
    """
    if before.shape != after.shape:
        raise AssertionError(
            f"[{label}] shape mismatch: {before.shape} → {after.shape}"
        )
    if not (before.index.equals(after.index)):
        raise AssertionError(f"[{label}] index mismatch")

    bv = np.asarray(before.to_numpy(), dtype=np.float64)
    av = np.asarray(after.to_numpy(), dtype=np.float64)

    mask_nan = np.isnan(bv) & np.isnan(av)
    close = np.isclose(bv, av, rtol=rtol, atol=atol) | mask_nan
    if not close.all():
        bad = (~close).sum()
        first_bad = int(np.argmax(~close))
        raise AssertionError(
            f"[{label}] returns diverged at {bad} / {close.size} points "
            f"(first at idx={first_bad}: before={bv.flat[first_bad]:.15g}, "
            f"after={av.flat[first_bad]:.15g}, |diff|="
            f"{abs(bv.flat[first_bad] - av.flat[first_bad]):.3e})"
        )


# ─────────────────────────────────────────────────────────────────────────
# JSON persistence — for disk-baseline workflow
# ─────────────────────────────────────────────────────────────────────────


def dump_fingerprint(fp: PortfolioFingerprint, path) -> None:
    """Write a fingerprint to JSON for the disk-baseline workflow."""
    import json
    from pathlib import Path

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        json.dump(fp.to_dict(), fh, indent=2, sort_keys=True)


def load_fingerprint(path) -> PortfolioFingerprint:
    """Load a fingerprint previously written by :func:`dump_fingerprint`."""
    import json
    from pathlib import Path

    with Path(path).open() as fh:
        return PortfolioFingerprint.from_dict(json.load(fh))
