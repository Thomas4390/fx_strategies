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

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from framework.x10_context import DXY4_CONSTANT, DXY4_WEIGHTS, dxy4, resample_ohlc
from utils import GOLD_DATA_PATH, load_fx_data

_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from build_dxy_synthetic import to_bar_open  # noqa: E402

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


# ═══════════════════════════════════════════════════════════════════════
# Convention de datation des jambes FX (§1)
# ═══════════════════════════════════════════════════════════════════════
#
# Les parquets FX datent une barre de sa CLÔTURE, comme celui de l'or. Preuve :
# contre l'export MT5 d'EURUSD, daté à l'ouverture, la corrélation des
# rendements minute vaut 0,991 au décalage +1 min et 0,07 partout ailleurs, sur
# 61 175 minutes de recouvrement (`docs/research/xau_x10_reconciliation.md`
# §11.6 bis). Sans redatation, les bins H1 du panier couvrent
# `[h−1 min, h+59 min)` au lieu de `[h, h+60 min)`.


def test_a_leg_is_restamped_by_the_instant_its_bar_opens():
    index = pd.date_range("2024-06-03 10:00", periods=5, freq="1min", name="date")
    frame = pd.DataFrame({"close": np.arange(5.0)}, index=index)
    shifted = to_bar_open(frame)
    assert list(shifted.index) == list(index - pd.Timedelta(minutes=1))
    # Les valeurs ne bougent pas : seule l'étiquette change.
    np.testing.assert_array_equal(shifted["close"].to_numpy(), frame["close"].to_numpy())


def test_an_already_open_stamped_leg_is_left_alone():
    index = pd.date_range("2024-06-03 10:00", periods=5, freq="1min", name="date")
    frame = pd.DataFrame({"close": np.arange(5.0)}, index=index)
    assert list(to_bar_open(frame, "open").index) == list(index)


def test_an_unknown_leg_stamp_convention_is_refused():
    frame = pd.DataFrame(
        {"close": [1.0]}, index=pd.DatetimeIndex(["2024-06-03 10:00"], name="date")
    )
    with pytest.raises(ValueError, match="source_stamp"):
        to_bar_open(frame, "middle")


def test_restamping_moves_an_h1_bin_by_one_minute_of_content():
    """Non-régression : retirer la redatation change la clôture du bin H1."""
    index = pd.date_range("2024-06-03 10:00", periods=121, freq="1min", name="date")
    frame = pd.DataFrame({"close": np.arange(121.0)}, index=index)
    raw = resample_ohlc(frame.assign(open=frame["close"], high=frame["close"],
                                     low=frame["close"]), "1h")["close"]
    redated = to_bar_open(frame)
    fixed = resample_ohlc(redated.assign(open=redated["close"], high=redated["close"],
                                         low=redated["close"]), "1h")["close"]
    # Le bin 11:00 se fermait sur la minute 11:59 ; il se ferme maintenant sur
    # 12:00, c'est-à-dire la valeur suivante.
    assert float(raw.loc["2024-06-03 11:00"]) == pytest.approx(119.0)
    assert float(fixed.loc["2024-06-03 11:00"]) == pytest.approx(120.0)


@pytest.mark.skipif(not EURUSD_PATH.exists(), reason="data/EUR-USD_minute.parquet absent")
def test_the_weekly_reopen_stays_in_the_seventeen_oclock_hour_after_restamping():
    """La reprise hebdomadaire passe de 17:04 à 17:03 : même heure, même séance.

    Le test de fuseau au-dessus lit le fichier tel qu'il est écrit ; celui-ci
    vérifie que la redatation ne fait pas basculer la reprise dans l'heure
    précédente, ce qui déplacerait la frontière de séance de §2.
    """
    raw, _ = load_fx_data("data/EUR-USD_minute.parquet")
    redated = to_bar_open(raw)
    gaps = redated.index.to_series().diff()
    reopen = redated.index[(gaps > pd.Timedelta(hours=24)).to_numpy()]
    assert len(reopen) > 300
    assert float(np.mean(reopen.hour == 17)) >= 0.99
    assert float(np.mean(reopen.dayofweek == 6)) >= 0.95
