"""Unit tests for the XAUUSD x10 causal kernels (``docs/specs/xau_x10_spec.md``).

These kernels feed three engines (vbt, QuantConnect, MQL5). A kernel that reads
one bar too far to the right invalidates all three at once and shows up nowhere
in the output, so every test here is either an opposable vector from the spec or
a causality property: truncate the series at ``t``, recompute, and demand the
same value at ``t``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from framework.x10_kernels import (
    atr_wilder_nb,
    kinematics_nb,
    session_mean_typical_nb,
    x10_levels_nb,
)

# ═══════════════════════════════════════════════════════════════════════
# helpers
# ═══════════════════════════════════════════════════════════════════════


def _ohlc_from_close(close: np.ndarray, spread: float = 0.5) -> tuple[np.ndarray, ...]:
    """Coherent high/low around a close path — enough for an ATR test."""
    return close + spread, close - spread, close


def _atr_reference(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, n: int = 14
) -> pd.Series:
    """Independent Wilder ATR, written in pandas straight from spec §4.

    ``TR[0]`` is undefined (it needs ``C[-1]``) and is forced to NaN rather than
    degraded to ``H-L``: with ``min_periods=n`` this is what makes the first
    defined ATR land on bar 14, as §4 requires.
    """
    h, lo, c = pd.Series(high), pd.Series(low), pd.Series(close)
    prev = c.shift(1)
    tr = pd.concat([h - lo, (h - prev).abs(), (lo - prev).abs()], axis=1).max(axis=1)
    tr.iloc[0] = np.nan
    return tr.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()


# ═══════════════════════════════════════════════════════════════════════
# §3 — x10 levels
# ═══════════════════════════════════════════════════════════════════════


def test_levels_spec_vectors():
    """The four opposable vectors: a price *on* a level opens the box above."""
    close = np.array([4400.000, 4409.999, 4410.000, 4399.999])
    l_inf, l_sup = x10_levels_nb(close)
    assert l_inf.tolist() == [4400.0, 4400.0, 4410.0, 4390.0]
    assert l_sup.tolist() == [4410.0, 4410.0, 4420.0, 4400.0]


def test_levels_bracket_property():
    """``L_inf <= C < L_sup`` and a 10 $ box, on 1e5 prices (spec §3)."""
    rng = np.random.default_rng(20260921)
    close = np.round(rng.uniform(1_000.0, 5_000.0, 100_000), 3)
    l_inf, l_sup = x10_levels_nb(close)
    assert bool(np.all(l_inf <= close))
    assert bool(np.all(close < l_sup))
    assert np.allclose(l_sup - l_inf, 10.0)


def test_levels_are_exact_multiples_of_ten():
    rng = np.random.default_rng(7)
    close = np.round(rng.uniform(1_000.0, 5_000.0, 10_000), 3)
    l_inf, _ = x10_levels_nb(close)
    assert bool(np.all(np.abs(l_inf / 10.0 - np.round(l_inf / 10.0)) < 1e-9))


def test_levels_float_noise_does_not_cross_the_grid():
    """``round(C*1000)`` is what keeps 4409.999999996 out of the wrong box."""
    l_inf, _ = x10_levels_nb(np.array([4409.999999996, 4410.000000004]))
    assert l_inf.tolist() == [4410.0, 4410.0]


# ═══════════════════════════════════════════════════════════════════════
# §4 — ATR de Wilder
# ═══════════════════════════════════════════════════════════════════════


def test_atr_matches_independent_wilder_reference():
    rng = np.random.default_rng(1)
    close = 2000.0 + np.cumsum(rng.normal(0.0, 1.0, 500))
    high, low, close = _ohlc_from_close(close, spread=1.5)
    got = atr_wilder_nb(high, low, close, 14)
    ref = _atr_reference(high, low, close, 14).to_numpy()
    both = np.isfinite(got) & np.isfinite(ref)
    assert both.sum() > 400
    assert np.array_equal(np.isfinite(got), np.isfinite(ref))
    np.testing.assert_allclose(got[both], ref[both], rtol=1e-10)


def test_atr_undefined_before_fourteen_bars():
    """§4: 'A[t] est indefini tant que t < 14' — never 0, never a partial mean."""
    rng = np.random.default_rng(2)
    close = 2000.0 + np.cumsum(rng.normal(0.0, 1.0, 40))
    high, low, close = _ohlc_from_close(close)
    atr = atr_wilder_nb(high, low, close, 14)
    assert bool(np.all(np.isnan(atr[:14])))
    assert bool(np.all(np.isfinite(atr[14:])))


def test_atr_is_causal():
    rng = np.random.default_rng(3)
    close = 2000.0 + np.cumsum(rng.normal(0.0, 1.0, 200))
    high, low, close = _ohlc_from_close(close)
    full = atr_wilder_nb(high, low, close, 14)
    for cut in (30, 87, 150):
        part = atr_wilder_nb(high[:cut], low[:cut], close[:cut], 14)
        np.testing.assert_allclose(part, full[:cut], rtol=1e-12, equal_nan=True)


# ═══════════════════════════════════════════════════════════════════════
# §5 — cinematique
# ═══════════════════════════════════════════════════════════════════════


def test_kinematics_linear_path_has_constant_speed_and_zero_acceleration():
    n = 60
    close = 2000.0 + 0.5 * np.arange(n, dtype=np.float64)
    atr = np.full(n, 2.0)
    v, m, a = kinematics_nb(close, atr, 3, 12)
    ok = np.isfinite(v)
    np.testing.assert_allclose(v[ok], 0.5 / 2.0, rtol=1e-12)
    ok_m = np.isfinite(m)
    np.testing.assert_allclose(m[ok_m], (0.5 * 12) / (2.0 * np.sqrt(12.0)), rtol=1e-12)
    np.testing.assert_allclose(a[np.isfinite(a)], 0.0, atol=1e-12)


def test_kinematics_quadratic_path_has_single_signed_acceleration():
    n = 80
    t = np.arange(n, dtype=np.float64)
    atr = np.full(n, 1.0)
    _, _, a_up = kinematics_nb(2000.0 + 0.01 * t**2, atr, 3, 12)
    _, _, a_dn = kinematics_nb(2000.0 - 0.01 * t**2, atr, 3, 12)
    assert bool(np.all(a_up[np.isfinite(a_up)] > 0))
    assert bool(np.all(a_dn[np.isfinite(a_dn)] < 0))


def test_acceleration_is_exactly_v_minus_v_lag3():
    rng = np.random.default_rng(4)
    n = 200
    close = 2000.0 + np.cumsum(rng.normal(0.0, 1.0, n))
    atr = 1.0 + np.abs(rng.normal(0.0, 0.1, n))
    v, _, a = kinematics_nb(close, atr, 3, 12)
    expected = np.full(n, np.nan)
    expected[3:] = v[3:] - v[:-3]
    np.testing.assert_allclose(a, expected, rtol=1e-12, equal_nan=True)


def test_kinematics_is_causal():
    rng = np.random.default_rng(5)
    n = 300
    close = 2000.0 + np.cumsum(rng.normal(0.0, 1.0, n))
    atr = 1.0 + np.abs(rng.normal(0.0, 0.1, n))
    v, m, a = kinematics_nb(close, atr, 3, 12)
    for cut in (40, 111, 250):
        v_c, m_c, a_c = kinematics_nb(close[:cut], atr[:cut], 3, 12)
        np.testing.assert_allclose(v_c, v[:cut], rtol=1e-12, equal_nan=True)
        np.testing.assert_allclose(m_c, m[:cut], rtol=1e-12, equal_nan=True)
        np.testing.assert_allclose(a_c, a[:cut], rtol=1e-12, equal_nan=True)


def test_kinematics_undefined_while_atr_is():
    n = 40
    close = 2000.0 + np.arange(n, dtype=np.float64)
    atr = np.full(n, np.nan)
    atr[20:] = 1.0
    v, m, a = kinematics_nb(close, atr, 3, 12)
    assert bool(np.all(np.isnan(v[:20])))
    assert bool(np.all(np.isnan(m[:20])))
    # a[t] needs v[t-3]: the first three bars after the ATR warmup stay undefined.
    assert bool(np.all(np.isnan(a[:23])))
    assert bool(np.all(np.isfinite(a[23:])))


# ═══════════════════════════════════════════════════════════════════════
# §6.2 — moyenne cumulee de seance du prix typique
# ═══════════════════════════════════════════════════════════════════════


def test_session_mean_typical_matches_expanding_mean():
    rng = np.random.default_rng(6)
    n = 300
    close = 2000.0 + np.cumsum(rng.normal(0.0, 0.5, n))
    high, low, close = _ohlc_from_close(close)
    session = np.repeat(np.arange(3, dtype=np.int64), 100)
    got = session_mean_typical_nb(high, low, close, session, 12)

    typ = pd.Series((high + low + close) / 3.0)
    ref = typ.groupby(pd.Series(session)).expanding().mean().to_numpy().copy()
    counts = pd.Series(session).groupby(pd.Series(session)).cumcount().to_numpy()
    ref[counts < 11] = np.nan
    np.testing.assert_allclose(got, ref, rtol=1e-12, equal_nan=True)


def test_session_mean_typical_warmup_and_reset():
    n = 60
    high, low, close = _ohlc_from_close(np.full(n, 2000.0), spread=0.0)
    session = np.repeat(np.array([0, 1], dtype=np.int64), 30)
    out = session_mean_typical_nb(high, low, close, session, 12)
    assert bool(np.all(np.isnan(out[:11])))
    assert np.isfinite(out[11])
    assert bool(np.all(np.isnan(out[30:41])))
    assert np.isfinite(out[41])


def test_session_mean_typical_is_causal():
    rng = np.random.default_rng(8)
    n = 200
    close = 2000.0 + np.cumsum(rng.normal(0.0, 0.5, n))
    high, low, close = _ohlc_from_close(close)
    session = np.repeat(np.arange(2, dtype=np.int64), 100)
    full = session_mean_typical_nb(high, low, close, session, 12)
    for cut in (20, 95, 160):
        part = session_mean_typical_nb(
            high[:cut], low[:cut], close[:cut], session[:cut], 12
        )
        np.testing.assert_allclose(part, full[:cut], rtol=1e-12, equal_nan=True)
