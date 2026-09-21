"""Tests for the synthetic DXY4 basket (``docs/specs/xau_x10_spec.md`` §6.3).

The basket is a context filter, so a wrong sign or a stale leg does not crash
anything — it quietly halves the risk on the wrong trades (§11). Hence the
monotonicity table, the closed-form recomputation in logs, and the staleness
boundary at exactly five bars.

The timezone tests are descriptive rather than prescriptive: they pin down what
the parquets on disk actually are, because the whole gold/FX alignment rests on
the FX minute index being New York wall clock and the gold export being UTC.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from framework.x10_context import DXY4_CONSTANT, DXY4_WEIGHTS, dxy4
from utils import GOLD_DATA_PATH, load_fx_data

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
EURUSD_PATH = _PROJECT_ROOT / "data" / "EUR-USD_minute.parquet"
GOLD_PATH = _PROJECT_ROOT / GOLD_DATA_PATH

# ═══════════════════════════════════════════════════════════════════════
# helpers
# ═══════════════════════════════════════════════════════════════════════


def _legs(n: int = 50, freq: str = "1h") -> dict[str, pd.Series]:
    idx = pd.date_range("2024-06-03", periods=n, freq=freq, name="date")
    rng = np.random.default_rng(42)
    return {
        "eurusd": pd.Series(1.08 + rng.normal(0, 0.002, n), index=idx),
        "usdjpy": pd.Series(157.0 + rng.normal(0, 0.2, n), index=idx),
        "gbpusd": pd.Series(1.27 + rng.normal(0, 0.002, n), index=idx),
        "usdcad": pd.Series(1.37 + rng.normal(0, 0.002, n), index=idx),
    }


# ═══════════════════════════════════════════════════════════════════════
# §6.3 — formule
# ═══════════════════════════════════════════════════════════════════════


def test_renormalised_weights_sum_to_one():
    assert set(DXY4_WEIGHTS) == {"eurusd", "usdjpy", "gbpusd", "usdcad"}
    assert sum(DXY4_WEIGHTS.values()) == pytest.approx(1.0, abs=1e-12)
    # ICE weights, renormalised: the four legs keep their relative importance.
    raw = {"eurusd": 0.576, "usdjpy": 0.136, "gbpusd": 0.119, "usdcad": 0.091}
    total = sum(raw.values())
    for leg, w in raw.items():
        assert DXY4_WEIGHTS[leg] == pytest.approx(w / total, abs=1e-12)


def test_dxy4_matches_the_closed_form_recomputed_in_logs():
    legs = _legs()
    got = dxy4(**legs)
    w = DXY4_WEIGHTS
    log_ref = (
        np.log(DXY4_CONSTANT)
        - w["eurusd"] * np.log(legs["eurusd"])
        + w["usdjpy"] * np.log(legs["usdjpy"])
        - w["gbpusd"] * np.log(legs["gbpusd"])
        + w["usdcad"] * np.log(legs["usdcad"])
    )
    np.testing.assert_allclose(got.to_numpy(), np.exp(log_ref).to_numpy(), rtol=1e-12)


@pytest.mark.parametrize(
    ("leg", "sign"),
    [("eurusd", -1), ("usdjpy", +1), ("gbpusd", -1), ("usdcad", +1)],
)
def test_dxy4_is_monotonic_in_every_leg(leg, sign):
    """A leg quoted USD-base pushes the basket up, USD-quote pushes it down."""
    base = _legs()
    bumped = dict(base)
    bumped[leg] = base[leg] * 1.01
    delta = (dxy4(**bumped) - dxy4(**base)).to_numpy()
    assert bool(np.all(np.sign(delta) == sign))


def test_dxy4_is_dimensionless_level_around_one_hundred():
    """Sanity: the ICE constant puts the basket in its usual 90-120 range."""
    value = float(dxy4(**_legs()).iloc[0])
    assert 80.0 < value < 130.0


# ═══════════════════════════════════════════════════════════════════════
# §6.3 — jambe absente : ffill <= 5 barres, sinon indefini
# ═══════════════════════════════════════════════════════════════════════


def test_missing_leg_is_forward_filled_up_to_five_bars():
    legs = _legs(n=30)
    full = dxy4(**legs)
    hole = legs["usdjpy"].index[10:15]
    legs["usdjpy"] = legs["usdjpy"].drop(hole)
    got = dxy4(**legs)

    assert got.index.equals(full.index)
    assert got.loc[hole].notna().all()
    # Stale leg: the basket moves with the other three, not with the gap.
    assert not np.allclose(got.loc[hole].to_numpy(), full.loc[hole].to_numpy())


def test_missing_leg_becomes_undefined_on_the_sixth_bar():
    legs = _legs(n=30)
    hole = legs["usdjpy"].index[10:16]  # six bars
    legs["usdjpy"] = legs["usdjpy"].drop(hole)
    got = dxy4(**legs)

    assert got.loc[hole[:5]].notna().all()
    assert np.isnan(got.loc[hole[5]])
    assert got.loc[hole[5] + pd.Timedelta(hours=1) :].notna().all()


def test_nan_inside_a_leg_is_treated_as_an_absent_bar():
    legs = _legs(n=30)
    legs["gbpusd"].iloc[10:16] = np.nan
    got = dxy4(**legs)
    assert got.iloc[10:15].notna().all()
    assert np.isnan(got.iloc[15])


# ═══════════════════════════════════════════════════════════════════════
# fuseaux horaires des parquets — hypothese opposable
# ═══════════════════════════════════════════════════════════════════════


@pytest.mark.skipif(not EURUSD_PATH.exists(), reason="data/EUR-USD_minute.parquet absent")
def test_fx_minute_index_is_naive_new_york_wall_clock():
    """The FX minute parquets carry a naive index — this proves it is New York.

    The FX week reopens Sunday 17:00 New York. Taking the first bar after every
    gap longer than 24 h and reading its hour is a direct measurement of the
    index's clock: 17 means New York, 22/23 would mean UTC. Everything in this
    strategy that aligns gold (New York, via ``load_gold_data``) with the DXY
    basket depends on this.
    """
    raw, _ = load_fx_data("data/EUR-USD_minute.parquet")
    assert raw.index.tz is None

    gaps = raw.index.to_series().diff()
    reopen = raw.index[(gaps > pd.Timedelta(hours=24)).to_numpy()]
    assert len(reopen) > 300
    share_17h = float(np.mean(reopen.hour == 17))
    assert share_17h >= 0.99, f"weekly reopen hour histogram says this is not New York: {share_17h:.3f}"
    assert float(np.mean(reopen.dayofweek == 6)) >= 0.95  # Sunday


@pytest.mark.skipif(not GOLD_PATH.exists(), reason="data/XAU-USD_minute_qc.parquet absent")
def test_gold_minute_export_is_tz_aware_utc():
    """``load_gold_data`` converts this index to naive New York — §2 depends on it."""
    head = pd.read_parquet(GOLD_PATH, columns=["close"]).head(10)
    assert head.index.tz is not None
    assert str(head.index.tz) == "UTC"
