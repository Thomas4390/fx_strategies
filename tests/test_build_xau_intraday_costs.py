"""Tests des fonctions pures de `scripts/build_xau_intraday_costs.py`.

La table de coût n'est pas un détail d'implémentation : §12 de la spec en fait
le premier terme de l'équation, et §1.1 de la table de décision interdit de s'en
servir pour re-sélectionner quoi que ce soit. Deux propriétés portent tout :

1. **la construction de la table** agrège par heure de New York et par année,
   sur l'horloge locale et pas sur l'UTC — un décalage de fuseau déplacerait
   toute la courbe horaire du spread d'une poignée d'heures sans rien casser
   de visible ;
2. **la relecture de la table** rend un spread par barre, avec un repli déclaré
   et jamais un zéro : un spread nul est un backtest qui se ment.

Les années antérieures au dump sont extrapolées au prorata du prix et portent
`source: extrapolated` ; un test vérifie que la mention survit, parce que c'est
elle qui empêche de publier une mesure qui n'existe pas.

Tout est synthétique : aucun accès au dump du tester.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "scripts"))

from build_xau_intraday_costs import (  # noqa: E402
    POINT,
    SESSION_TZ,
    apply_price_scaling,
    build_cost_table,
    spread_from_table,
    spread_per_m5_bar,
)


# ═══════════════════════════════════════════════════════════════════════
# Fabriques
# ═══════════════════════════════════════════════════════════════════════


def synthetic_dump(
    start: str = "2023-01-02 00:00:00",
    hours: int = 48,
    *,
    expensive_hour_ny: int = 18,
    cheap_points: int = 200,
    expensive_points: int = 500,
    price: float = 2000.0,
) -> pd.DataFrame:
    """Un dump M1 où une seule heure de New York est chère, et on sait laquelle."""
    index = pd.date_range(start, periods=hours * 60, freq="1min", tz="UTC")
    local_hour = index.tz_convert(SESSION_TZ).hour
    points = np.where(local_hour == expensive_hour_ny, expensive_points, cheap_points)
    return pd.DataFrame(
        {"close": np.full(len(index), price), "spread_points": points}, index=index
    )


# ═══════════════════════════════════════════════════════════════════════
# Construction de la table
# ═══════════════════════════════════════════════════════════════════════


def test_the_table_is_cut_on_the_new_york_clock_not_on_utc():
    dump = synthetic_dump()
    table = build_cost_table(dump["spread_points"], dump["close"], base_year=2023)

    hours = table["by_year"]["2023"]["hours"]
    assert hours["18"]["median"] == pytest.approx(500 * POINT)
    # Toutes les autres heures locales sont au tarif bon marché.
    others = [h for key, h in hours.items() if key != "18"]
    assert all(h["median"] == pytest.approx(200 * POINT) for h in others)


def test_the_table_reports_the_point_value_and_the_measured_window():
    dump = synthetic_dump()
    table = build_cost_table(dump["spread_points"], dump["close"], base_year=2023)
    assert table["point"] == POINT
    assert table["timezone"] == SESSION_TZ
    assert table["measured_window_utc"][0].startswith("2023-01-02")
    assert table["unit"].startswith("USD")


def test_the_median_and_the_p75_are_not_the_same_number_when_the_spread_moves():
    index = pd.date_range("2023-03-01 12:00:00", periods=100, freq="1min", tz="UTC")
    points = pd.Series(np.arange(100) + 100, index=index)
    close = pd.Series(np.full(100, 2000.0), index=index)
    table = build_cost_table(points, close, base_year=2023)

    cell = next(iter(table["by_year"]["2023"]["hours"].values()))
    assert cell["p75"] > cell["median"]
    assert cell["n_bars"] > 0


def test_a_partial_year_says_so():
    """Le dump démarre en novembre 2022 : 2022 n'est pas une mesure de douze mois."""
    dump = synthetic_dump()
    table = build_cost_table(dump["spread_points"], dump["close"], base_year=2023)
    assert table["by_year"]["2023"]["partial_year"] is True


def test_years_before_the_dump_are_marked_extrapolated():
    dump = synthetic_dump()
    table = build_cost_table(
        dump["spread_points"], dump["close"], base_year=2023,
        extrapolated_years=(2019, 2020),
    )
    for year in ("2019", "2020"):
        block = table["by_year"][year]
        assert block["source"] == "extrapolated"
        assert block["extrapolated_from"] == 2023
        # Aucune barre ne les a produites : le compte doit le dire.
        assert all(cell["n_bars"] == 0 for cell in block["hours"].values())
    assert table["by_year"]["2023"]["source"] == "mt5_dump"


def test_a_measured_year_is_never_overwritten_by_an_extrapolation():
    dump = synthetic_dump()
    table = build_cost_table(
        dump["spread_points"], dump["close"], base_year=2023,
        extrapolated_years=(2023,),
    )
    assert table["by_year"]["2023"]["source"] == "mt5_dump"


def test_the_reference_year_must_exist_in_the_dump():
    dump = synthetic_dump()
    with pytest.raises(SystemExit, match="année de référence"):
        build_cost_table(dump["spread_points"], dump["close"], base_year=1999)


# ═══════════════════════════════════════════════════════════════════════
# Extrapolation au prorata du prix
# ═══════════════════════════════════════════════════════════════════════


def test_the_extrapolation_scales_the_profile_by_the_median_price():
    dump = synthetic_dump(price=2000.0)
    table = build_cost_table(
        dump["spread_points"], dump["close"], base_year=2023, extrapolated_years=(2019,)
    )
    apply_price_scaling(table, {2019: 1000.0})

    block = table["by_year"]["2019"]
    assert block["scale"] == pytest.approx(0.5)
    assert block["price_median_usd"] == pytest.approx(1000.0)
    # Deux fois moins cher pour un or deux fois moins cher, heure par heure.
    assert block["hours"]["18"]["median"] == pytest.approx(500 * POINT / 2)


def test_the_extrapolation_leaves_the_scale_empty_when_the_price_is_unknown():
    dump = synthetic_dump()
    table = build_cost_table(
        dump["spread_points"], dump["close"], base_year=2023, extrapolated_years=(2019,)
    )
    before = table["by_year"]["2019"]["hours"]["18"]["median"]
    apply_price_scaling(table, {})
    assert table["by_year"]["2019"]["scale"] is None
    assert table["by_year"]["2019"]["hours"]["18"]["median"] == pytest.approx(before)


def test_the_measured_years_are_never_rescaled():
    dump = synthetic_dump()
    table = build_cost_table(
        dump["spread_points"], dump["close"], base_year=2023, extrapolated_years=(2019,)
    )
    before = table["by_year"]["2023"]["hours"]["18"]["median"]
    apply_price_scaling(table, {2019: 1000.0, 2023: 4000.0})
    assert table["by_year"]["2023"]["hours"]["18"]["median"] == pytest.approx(before)


# ═══════════════════════════════════════════════════════════════════════
# Relecture : un spread par barre
# ═══════════════════════════════════════════════════════════════════════


def test_the_table_is_read_back_bar_by_bar_on_the_new_york_clock():
    dump = synthetic_dump()
    table = build_cost_table(dump["spread_points"], dump["close"], base_year=2023)

    # L'index du moteur est l'horloge New York NAÏVE de §2.
    index = pd.DatetimeIndex(
        ["2023-01-02 18:05:00", "2023-01-02 09:05:00"]
    )
    spread = spread_from_table(table, index, "median")
    assert spread[0] == pytest.approx(500 * POINT)
    assert spread[1] == pytest.approx(200 * POINT)


def test_the_p75_statistic_is_read_when_it_is_asked_for():
    index = pd.date_range("2023-03-01 12:00:00", periods=100, freq="1min", tz="UTC")
    points = pd.Series(np.arange(100) + 100, index=index)
    close = pd.Series(np.full(100, 2000.0), index=index)
    table = build_cost_table(points, close, base_year=2023)

    local = index.tz_convert(SESSION_TZ).tz_localize(None)[:1]
    median = spread_from_table(table, local, "median")[0]
    p75 = spread_from_table(table, local, "p75")[0]
    assert p75 > median


def test_a_missing_cell_falls_back_on_the_declared_value_never_on_zero():
    dump = synthetic_dump()
    table = build_cost_table(dump["spread_points"], dump["close"], base_year=2023)
    # 1998 n'est dans aucune branche de la table.
    spread = spread_from_table(table, pd.DatetimeIndex(["1998-06-01 10:00:00"]), "median")
    assert spread[0] == pytest.approx(0.29)
    assert (spread > 0).all()


# ═══════════════════════════════════════════════════════════════════════
# Spread par barre M5
# ═══════════════════════════════════════════════════════════════════════


def test_the_m5_spread_is_the_median_of_the_minutes_it_contains():
    index = pd.date_range("2023-01-02 00:00:00", periods=10, freq="1min", tz="UTC")
    dump = pd.DataFrame(
        {"spread_points": [100, 200, 300, 400, 500, 10, 10, 10, 10, 9000]}, index=index
    )
    m5 = pd.DatetimeIndex(["2023-01-02 00:00:00", "2023-01-02 00:05:00"], tz="UTC")
    spread = spread_per_m5_bar(dump, m5)
    assert spread[0] == pytest.approx(300 * POINT)
    # La médiane encaisse le pic à 9 $ sans le laisser porter la barre.
    assert spread[1] == pytest.approx(10 * POINT)


def test_the_m5_spread_refuses_minutes_that_fall_outside_the_grid():
    index = pd.date_range("2023-01-02 00:00:00", periods=10, freq="1min", tz="UTC")
    dump = pd.DataFrame({"spread_points": np.full(10, 250)}, index=index)
    m5 = pd.DatetimeIndex(["2023-01-02 00:00:00"], tz="UTC")
    with pytest.raises(SystemExit, match="hors de la grille M5"):
        spread_per_m5_bar(dump, m5)
