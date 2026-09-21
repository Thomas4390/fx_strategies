"""Tests for the pandas assembly of the x10 context (``docs/specs/xau_x10_spec.md`` §6).

Two families of quantities live here and **do not obey the same causality rule**
(spec §6): the H1 context (EMA50, ATR, DXY) is only visible after the H1 bar has
closed, while the session VWAP is a cumulative mean up to and including the
decision bar. Confusing the two is the documented first source of error, so the
pivot test of this file perturbs the M1 bars of one hour and demands that no M5
bar opening before the end of that hour sees anything move — with a naive
``resample().last() + ffill`` twin proving the test actually bites.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from framework.x10_context import (
    EMA_SPAN_H1,
    SESSION_TZ,
    VWAP_MIN_BARS,
    align_h1_to_m5,
    context_scores,
    ema_h1,
    resample_ohlc,
    session_dates,
    session_ids,
    session_mean_typical,
)
from strategies.gold_momentum import session_dates as gold_session_dates

# ═══════════════════════════════════════════════════════════════════════
# helpers
# ═══════════════════════════════════════════════════════════════════════


def _m1_frame(start: str, periods: int, seed: int = 0, tz: str | None = None) -> pd.DataFrame:
    """Synthetic M1 OHLC random walk, index named like the gold export."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=periods, freq="1min", tz=tz, name="date")
    close = 2000.0 + np.cumsum(rng.normal(0.0, 0.05, periods))
    return pd.DataFrame(
        {
            "open": close - rng.normal(0.0, 0.01, periods),
            "high": close + np.abs(rng.normal(0.0, 0.05, periods)) + 0.05,
            "low": close - np.abs(rng.normal(0.0, 0.05, periods)) - 0.05,
            "close": close,
        },
        index=idx,
    )


def _gold_clock_index(start: str, end: str, tz: str = "UTC") -> pd.DatetimeIndex:
    """M5 stamps over a date span, minus the 17:00-18:00 New York break."""
    idx = pd.date_range(start, end, freq="5min", tz=tz, name="date")
    ny = idx.tz_convert(SESSION_TZ) if idx.tz is not None else idx
    keep = ~((ny.hour == 17) & (ny.minute > 0))
    return idx[keep]


# ═══════════════════════════════════════════════════════════════════════
# §2 — seance 18:00 -> 17:00 New York
# ═══════════════════════════════════════════════════════════════════════


def test_session_dates_reproduce_the_gold_momentum_convention():
    """One boundary in the repo: this must agree bar for bar with strategy 1."""
    idx = pd.date_range("2024-05-01", periods=5_000, freq="5min", name="date")
    pd.testing.assert_index_equal(session_dates(idx), gold_session_dates(idx))


@pytest.mark.parametrize(
    ("switch", "utc_before", "utc_after"),
    [("2024-03-10", 23, 22), ("2024-11-03", 22, 23)],
)
def test_session_resets_at_18h_new_york_across_dst(switch, utc_before, utc_after):
    day = pd.Timestamp(switch)
    idx = _gold_clock_index(
        (day - pd.Timedelta(days=3)).strftime("%Y-%m-%d"),
        (day + pd.Timedelta(days=3)).strftime("%Y-%m-%d"),
    )
    sid = session_ids(idx)
    first = idx[np.r_[True, sid[1:] != sid[:-1]]][1:]  # drop the truncated first session
    ny = first.tz_convert(SESSION_TZ)

    assert bool(np.all(ny.hour == 18))
    assert bool(np.all(ny.minute == 0))
    # The wall-clock boundary is fixed; its UTC stamp is what moves across DST.
    assert set(first[first < day.tz_localize("UTC")].hour) == {utc_before}
    assert set(first[first > (day + pd.Timedelta(days=1)).tz_localize("UTC")].hour) == {
        utc_after
    }


def test_session_ids_are_monotonic_and_contiguous():
    idx = _gold_clock_index("2024-06-03", "2024-06-10")
    sid = session_ids(idx)
    assert bool(np.all(np.diff(sid) >= 0))
    assert len(np.unique(sid)) == 8


# ═══════════════════════════════════════════════════════════════════════
# §1/§6 — agregation M1 -> M5 / H1
# ═══════════════════════════════════════════════════════════════════════


def test_resample_ohlc_is_left_labelled_and_left_closed():
    m1 = _m1_frame("2024-06-03 00:00", 120, seed=1)
    m5 = resample_ohlc(m1, "5min")
    assert m5.index[0] == pd.Timestamp("2024-06-03 00:00")
    assert m5["open"].iloc[0] == m1["open"].iloc[0]
    assert m5["close"].iloc[0] == m1["close"].iloc[4]  # 00:04 closes the 00:00 bar
    assert m5["high"].iloc[0] == m1["high"].iloc[:5].max()
    assert m5["low"].iloc[0] == m1["low"].iloc[:5].min()
    assert len(m5) == 24


def test_resample_ohlc_drops_bins_without_any_m1():
    m1 = _m1_frame("2024-06-03 00:00", 60, seed=2)
    m1 = m1.drop(m1.index[20:40])  # a 20-minute hole
    m5 = resample_ohlc(m1, "5min")
    assert len(m5) == 8
    assert pd.Timestamp("2024-06-03 00:20") not in m5.index
    assert not m5.isna().to_numpy().any()


def test_resample_ohlc_h1_matches_manual_aggregation():
    m1 = _m1_frame("2024-06-03 00:00", 180, seed=3)
    h1 = resample_ohlc(m1, "1h")
    assert len(h1) == 3
    assert h1["close"].iloc[0] == m1["close"].iloc[59]
    assert h1["open"].iloc[1] == m1["open"].iloc[60]


# ═══════════════════════════════════════════════════════════════════════
# §6.1 — contexte H1, anti look-ahead (test pivot)
# ═══════════════════════════════════════════════════════════════════════


def _naive_align(h1_series: pd.Series, m5_index: pd.DatetimeIndex) -> pd.Series:
    """The faulty idiom of ``eur-usd-bb-mr-research-plan.md:148-197``, on purpose.

    ``resample().last()`` + ``ffill`` without ``shift(1)``: the 08:59 close is
    served to the 08:00 bar. Kept here as the twin that must FAIL the causality
    property the real implementation passes.
    """
    return h1_series.reindex(m5_index, method="ffill")


def test_align_h1_to_m5_serves_the_previous_closed_hour():
    m1 = _m1_frame("2024-06-03 00:00", 60 * 6, seed=4)
    m5 = resample_ohlc(m1, "5min")
    h1 = resample_ohlc(m1, "1h")
    aligned = align_h1_to_m5(h1["close"], m5.index)

    assert np.isnan(aligned.loc["2024-06-03 00:00"])  # no closed hour yet
    assert aligned.loc["2024-06-03 01:00"] == h1["close"].loc["2024-06-03 00:00"]
    assert aligned.loc["2024-06-03 01:55"] == h1["close"].loc["2024-06-03 00:00"]
    assert aligned.loc["2024-06-03 02:00"] == h1["close"].loc["2024-06-03 01:00"]
    assert aligned.index.equals(m5.index)


def _perturbation_deltas(align) -> tuple[np.ndarray, np.ndarray]:
    """Return (deltas before the end of hour h, deltas after) for one aligner."""
    m1 = _m1_frame("2024-06-03 00:00", 60 * 24 * 3, seed=5)
    hour = pd.Timestamp("2024-06-04 09:00")
    end = hour + pd.Timedelta(hours=1)

    shocked = m1.copy()
    mask = (shocked.index >= hour) & (shocked.index < end)
    shocked.loc[mask, ["open", "high", "low", "close"]] *= 1.5

    m5 = resample_ohlc(m1, "5min")
    base = align(ema_h1(resample_ohlc(m1, "1h")["close"], span=3), m5.index)
    shock = align(ema_h1(resample_ohlc(shocked, "1h")["close"], span=3), m5.index)

    diff = (base - shock).abs().fillna(0.0)
    return diff[m5.index < end].to_numpy(), diff[m5.index >= end].to_numpy()


def test_h1_context_cannot_see_the_hour_it_lives_in():
    """PIVOT: shocking the M1 of hour h moves no M5 bar opening before h+1h."""
    before, after = _perturbation_deltas(align_h1_to_m5)
    assert float(np.max(before)) == 0.0
    assert float(np.max(after)) > 0.0  # the shock is real, the test is not vacuous


def test_the_naive_alignment_leaks_and_the_pivot_test_catches_it():
    """Twin of the test above on the faulty idiom: it must fail."""
    before, after = _perturbation_deltas(_naive_align)
    assert float(np.max(after)) > 0.0
    with pytest.raises(AssertionError):
        assert float(np.max(before)) == 0.0, (
            f"look-ahead leak detected: {float(np.max(before)):.6f}"
        )


def test_ema_h1_is_the_spec_ema50():
    """§6.1: alpha = 2/51, recursive, undefined before 50 closes."""
    rng = np.random.default_rng(9)
    close = pd.Series(
        2000.0 + np.cumsum(rng.normal(0.0, 1.0, 200)),
        index=pd.date_range("2024-01-01", periods=200, freq="1h"),
    )
    got = ema_h1(close)
    ref = close.ewm(span=EMA_SPAN_H1, adjust=False, min_periods=EMA_SPAN_H1).mean()
    pd.testing.assert_series_equal(got, ref, check_names=False)
    assert got.iloc[:49].isna().all()
    assert got.iloc[49:].notna().all()


# ═══════════════════════════════════════════════════════════════════════
# §6.2 — VWAP de seance
# ═══════════════════════════════════════════════════════════════════════


def test_session_vwap_equals_expanding_mean_of_typical_price():
    idx = _gold_clock_index("2024-06-03", "2024-06-07")
    rng = np.random.default_rng(10)
    close = 2000.0 + np.cumsum(rng.normal(0.0, 0.3, len(idx)))
    m5 = pd.DataFrame(
        {"open": close, "high": close + 0.4, "low": close - 0.4, "close": close},
        index=idx,
    )
    got = session_mean_typical(m5)

    sid = pd.Series(session_ids(idx), index=idx)
    typ = (m5["high"] + m5["low"] + m5["close"]) / 3.0
    ref = typ.groupby(sid).expanding().mean().droplevel(0)
    ref[typ.groupby(sid).cumcount() < VWAP_MIN_BARS - 1] = np.nan
    pd.testing.assert_series_equal(got, ref, check_names=False)


def test_session_vwap_warms_up_over_twelve_bars_in_every_session():
    idx = _gold_clock_index("2024-06-03", "2024-06-07")
    m5 = pd.DataFrame(
        {"open": 2000.0, "high": 2001.0, "low": 1999.0, "close": 2000.0}, index=idx
    )
    got = session_mean_typical(m5)
    sid = pd.Series(session_ids(idx), index=idx)
    order = sid.groupby(sid).cumcount()
    assert got[order < 11].isna().all()
    assert got[order >= 11].notna().all()
    np.testing.assert_allclose(got[order >= 11].to_numpy(), 2000.0)


def test_session_vwap_resets_across_both_2024_dst_switches():
    for switch in ("2024-03-10", "2024-11-03"):
        day = pd.Timestamp(switch)
        idx = _gold_clock_index(
            (day - pd.Timedelta(days=2)).strftime("%Y-%m-%d"),
            (day + pd.Timedelta(days=2)).strftime("%Y-%m-%d"),
        )
        rng = np.random.default_rng(11)
        close = 2000.0 + np.cumsum(rng.normal(0.0, 0.3, len(idx)))
        m5 = pd.DataFrame(
            {"open": close, "high": close + 0.4, "low": close - 0.4, "close": close},
            index=idx,
        )
        got = session_mean_typical(m5)
        # A reset shows up as a fresh 11-bar NaN warmup at each 18:00 New York.
        ny = idx.tz_convert(SESSION_TZ)
        starts = np.flatnonzero((ny.hour == 18) & (ny.minute == 0))
        for s in starts:
            assert got.iloc[s : s + 11].isna().all()
            assert np.isfinite(got.iloc[s + 11])


def test_session_vwap_ignores_a_volume_column():
    """§6.2: unweighted by construction — gold has no volume on QC (§1)."""
    idx = _gold_clock_index("2024-06-03", "2024-06-05")
    rng = np.random.default_rng(12)
    close = 2000.0 + np.cumsum(rng.normal(0.0, 0.3, len(idx)))
    m5 = pd.DataFrame(
        {"open": close, "high": close + 0.4, "low": close - 0.4, "close": close},
        index=idx,
    )
    flat = m5.assign(volume=1.0)
    wild = m5.assign(volume=rng.uniform(1.0, 1e4, len(idx)))
    pd.testing.assert_series_equal(session_mean_typical(flat), session_mean_typical(wild))


# ═══════════════════════════════════════════════════════════════════════
# §6.4 — scores de contexte, dans le sens du trade
# ═══════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize("q", [1, -1])
def test_context_scores_truth_table(q):
    close = np.array([100.0, 100.0, 100.0, 100.0, 100.0, 100.0])
    ema = np.array([90.0, 110.0, 100.0, 100.0, 100.0, 100.0])
    vwap = np.array([90.0, 110.0, 100.0, np.nan, 90.0, 110.0])
    dxy = np.array([10.0, 10.0, 10.0, 10.0, np.nan, 10.0])
    dxy_ema = np.array([9.0, 11.0, 10.0, 10.0, 10.0, np.nan])

    ctx_ema, ctx_vwap, ctx_dxy = context_scores(q, close, ema, vwap, dxy, dxy_ema)

    assert ctx_ema.tolist() == [q, -q, 0, 0, 0, 0]
    assert ctx_vwap.tolist() == [q, -q, 0, 0, q, -q]
    # ctx_dxy = signe(-q * (DXY - EMA50_H1(DXY))): a stronger dollar is adverse
    # to a long gold trade, hence the leading minus.
    assert ctx_dxy.tolist() == [-q, q, 0, 0, 0, 0]


def test_context_scores_are_pure_signs_and_keep_the_index():
    idx = pd.date_range("2024-06-03", periods=4, freq="5min")
    close = pd.Series([100.0, 101.0, 99.0, 100.0], index=idx)
    ema = pd.Series(100.0, index=idx)
    ctx_ema, _, _ = context_scores(1, close, ema, ema, ema, ema)
    assert isinstance(ctx_ema, pd.Series)
    pd.testing.assert_index_equal(ctx_ema.index, idx)
    assert set(np.unique(ctx_ema.to_numpy())) <= {-1, 0, 1}


def test_context_scores_accept_a_vector_of_trade_directions():
    close = np.array([100.0, 100.0])
    ema = np.array([90.0, 90.0])
    q = np.array([1, -1])
    ctx_ema, _, _ = context_scores(q, close, ema, ema, ema, ema)
    assert ctx_ema.tolist() == [1, -1]
