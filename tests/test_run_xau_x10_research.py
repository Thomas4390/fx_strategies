"""Les fonctions pures de la campagne in-sample XAUUSD x10.

Le script `scripts/run_xau_x10_research.py` produit des chiffres qui finissent
dans un rapport client et dans un verdict pré-gelé. Trois blocs y décident
quelque chose sans que personne puisse le vérifier à l'œil sur un run de sept
ans : la règle « centre du plateau » du §3 de la table de décision, l'agrégation
des rendements par séance, et la lecture de la table du §1. Ils sont donc
testés ici sur des données **synthétiques**, où la réponse est connue d'avance.

Aucun test de ce fichier ne charge de prix, ne lance le moteur, ni n'écrit dans
`results/` : ce sont des tests de règle, pas des tests de campagne. La campagne
elle-même n'est pas rejouable en test — elle consomme des essais.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import run_xau_x10_research as campaign  # noqa: E402


# ═══════════════════════════════════════════════════════════════════════
# 1. LE VOISINAGE 3x3x3 (§3 de la table)
# ═══════════════════════════════════════════════════════════════════════


def test_the_grid_is_the_twentyseven_of_annexe_a2():
    idx = campaign.grid_indices()
    assert len(idx) == 27
    assert len(set(idx)) == 27
    assert campaign.config_of((0, 0, 0)) == {"z": 0.5, "a_min": 0.1, "k_s": 0.75}
    assert campaign.config_of((2, 2, 2)) == {"z": 1.5, "a_min": 0.3, "k_s": 1.5}
    assert campaign.config_label((1, 1, 1)) == "z1_a0.2_k1"


@pytest.mark.parametrize(
    "idx,expected",
    [
        ((1, 1, 1), 6),  # centre géométrique
        ((0, 0, 0), 3),  # coin
        ((2, 2, 2), 3),  # coin
        ((1, 0, 0), 4),  # arête
        ((1, 1, 0), 5),  # face
    ],
)
def test_neighbour_counts_match_the_definition_of_the_table(idx, expected):
    """« 3 voisins à un coin, 4 ou 5 sur une arête ou une face, 6 au centre »."""
    assert len(campaign.neighbours(idx)) == expected


def test_a_neighbour_differs_on_exactly_one_axis_by_exactly_one():
    for idx in campaign.grid_indices():
        nbrs = campaign.neighbours(idx)
        assert idx not in nbrs, "une configuration n'est jamais son propre voisin"
        for other in nbrs:
            deltas = [abs(a - b) for a, b in zip(idx, other, strict=True)]
            assert sorted(deltas) == [0, 0, 1]
            assert idx in campaign.neighbours(other), "le voisinage est symétrique"


# ═══════════════════════════════════════════════════════════════════════
# 2. LE CENTRE DU PLATEAU ET SON DÉPARTAGE (§3)
# ═══════════════════════════════════════════════════════════════════════


def _flat_grid(value: float) -> dict[tuple[int, int, int], float]:
    return dict.fromkeys(campaign.grid_indices(), value)


def test_a_grid_that_is_negative_everywhere_selects_nothing():
    """Le cas pré-écrit par la table : pas de repli sur le pic, pas de retenue."""
    sharpes = _flat_grid(-1.5)
    sharpes[(0, 0, 0)] = -0.2  # un pic, isolé et toujours négatif
    report = campaign.select_plateau_center(sharpes)

    assert report["peak"]["config"] == campaign.config_label((0, 0, 0))
    assert report["n_passing_plateau"] == 0
    assert report["selected"] is None
    # Les deux moitiés du critère échouent, et la seconde n'est pas un accident :
    # quand le pic est négatif, `0,5 x pic` est PLUS HAUT que le pic, donc la
    # barre de la médiane monte au lieu de descendre. La règle du §3 ne se
    # dégrade pas en passoire sur un échantillon perdant.
    assert report["plateau_median_threshold"] == pytest.approx(-0.1)
    row = next(r for r in report["candidates"] if r["index"] == [1, 1, 1])
    assert row["median_neighbour_sharpe"] == pytest.approx(-1.5)
    assert row["passes_median"] is False
    assert row["passes_positive_share"] is False


def test_a_uniformly_positive_grid_selects_the_geometric_centre():
    """Tout est à égalité : le départage n°1 (Manhattan) tranche seul."""
    report = campaign.select_plateau_center(_flat_grid(1.0))
    assert report["n_passing_plateau"] == 27
    assert report["selected"]["index"] == [1, 1, 1]


def test_the_choice_maximises_the_neighbour_median_not_the_own_sharpe():
    """« on choisit un point dont l'entourage est bon, pas un point qui est bon »."""
    sharpes = _flat_grid(0.2)
    # Un pic isolé au coin — assez haut pour être le pic, assez bas pour que la
    # barre de la médiane (0,5 x pic = 1,5) reste franchissable par l'entourage.
    sharpes[(0, 0, 0)] = 3.0
    for nbr in campaign.neighbours((1, 1, 1)):
        sharpes[nbr] = 2.0
    sharpes[(1, 1, 1)] = 0.05  # le point retenu est lui-même médiocre

    report = campaign.select_plateau_center(sharpes)
    assert report["peak"]["config"] == campaign.config_label((0, 0, 0))
    assert report["plateau_median_threshold"] == pytest.approx(1.5)
    assert report["selected"]["index"] == [1, 1, 1]
    assert report["selected"]["median_neighbour_sharpe"] == pytest.approx(2.0)
    assert report["selected"]["sharpe_net"] == pytest.approx(0.05)


def test_the_own_sharpe_of_a_configuration_never_disqualifies_it():
    """Corollaire du §3 : un point médiocre entouré de bons points est retenu.

    Mettre le centre à -1 ne l'écarte de rien — le classement ne regarde que
    ses voisins, qui sont tous à 1.
    """
    sharpes = _flat_grid(1.0)
    sharpes[(1, 1, 1)] = -1.0

    report = campaign.select_plateau_center(sharpes)
    assert report["selected"]["index"] == [1, 1, 1]
    assert report["selected"]["sharpe_net"] == pytest.approx(-1.0)
    assert report["selected"]["median_neighbour_sharpe"] == pytest.approx(1.0)


def test_ties_are_broken_by_manhattan_then_by_the_widest_stop():
    """Départage 1 puis 2 : la distance de Manhattan, puis le plus grand `k_s`.

    Les quatre faces des axes `z` et `a_min` sont négatives : le centre (1,1,1)
    n'a plus 60 % de voisins positifs et sort du plateau. Les six candidats à
    distance 1 restants sont tous à médiane 1,0 — parfaitement à égalité — et
    seul le `k_s` le plus large les départage : (1,1,2), soit `k_s = 1,5`.
    """
    sharpes = _flat_grid(1.0)
    for face in ((0, 1, 1), (2, 1, 1), (1, 0, 1), (1, 2, 1)):
        sharpes[face] = -1.0

    report = campaign.select_plateau_center(sharpes)
    centre = next(r for r in report["candidates"] if r["index"] == [1, 1, 1])
    assert centre["share_neighbours_positive"] == pytest.approx(2 / 6)
    assert centre["passes_plateau"] is False

    selected = report["selected"]
    assert selected["index"] == [1, 1, 2]
    assert selected["k_s"] == max(campaign.GRID_K_S)
    assert sum(abs(a - b) for a, b in zip(selected["index"], (1, 1, 1), strict=True)) == 1


def test_every_candidate_row_carries_its_own_arithmetic():
    """Le JSON doit permettre de rejouer la règle à la main, sans le code."""
    report = campaign.select_plateau_center(_flat_grid(1.0))
    row = next(r for r in report["candidates"] if r["index"] == [0, 0, 0])
    assert len(row["neighbours"]) == 3 == row["n_neighbours"]
    assert row["neighbour_sharpes"] == [1.0, 1.0, 1.0]
    assert row["median_neighbour_sharpe"] == pytest.approx(1.0)
    assert report["plateau_median_threshold"] == pytest.approx(0.5)


# ═══════════════════════════════════════════════════════════════════════
# 3. RENDEMENTS PAR SÉANCE ET MÉTRIQUES DE TRADE
# ═══════════════════════════════════════════════════════════════════════


def _trades(rows: list[tuple[str, float, float]]) -> pd.DataFrame:
    """`(date de séance, PnL, risque engagé)` -> le registre minimal attendu."""
    dates = pd.to_datetime([r[0] for r in rows])
    pnl = np.array([r[1] for r in rows], dtype=float)
    equity = 10_000.0 + np.cumsum(pnl)
    return pd.DataFrame(
        {
            "session_date": dates,
            "equity_before": equity - pnl,
            "equity_after": equity,
            "risk_amount": [r[2] for r in rows],
        }
    )


def test_session_returns_keep_the_flat_sessions():
    """Une stratégie plate la plupart du temps ne doit pas voir son Sharpe gonfler."""
    sessions = pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"])
    trades = _trades([("2020-01-01", 100.0, 50.0), ("2020-01-03", -200.0, 50.0)])
    returns = campaign.session_returns(trades, sessions, init_cash=10_000.0)

    assert list(returns.index) == list(sessions)
    assert returns.iloc[0] == pytest.approx(100.0 / 10_000.0)
    assert returns.iloc[1] == pytest.approx(0.0)  # séance sans trade, conservée
    assert returns.iloc[2] == pytest.approx(-200.0 / 10_100.0)


def test_session_returns_of_an_empty_registry_are_flat():
    sessions = pd.to_datetime(["2020-01-01", "2020-01-02"])
    empty = _trades([]).astype({"session_date": "datetime64[ns]"})
    returns = campaign.session_returns(empty, sessions)
    assert (returns == 0.0).all()
    assert np.isnan(campaign.sharpe_net(returns))


def test_expectancy_is_the_mean_of_pnl_over_risk_engaged():
    trades = _trades(
        [("2020-01-01", 100.0, 50.0), ("2020-01-02", -50.0, 50.0)]
    )
    metrics = campaign.trade_metrics(trades)
    assert metrics["n"] == 2
    assert metrics["expectancy_r"] == pytest.approx((2.0 + -1.0) / 2)
    assert metrics["profit_factor"] == pytest.approx(100.0 / 50.0)
    assert metrics["pnl"] == pytest.approx(50.0)
    assert metrics["win_rate"] == pytest.approx(0.5)


def test_sharpe_and_drawdown_are_read_off_the_session_curve():
    returns = pd.Series(
        [0.01, -0.02, 0.01], index=pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"])
    )
    arr = returns.to_numpy()
    expected = arr.mean() / arr.std(ddof=1) * np.sqrt(252.0)
    assert campaign.sharpe_net(returns) == pytest.approx(expected)
    # Pic à 1,01 puis creux à 1,01 x 0,98 : -2 % exactement.
    assert campaign.max_drawdown(returns) == pytest.approx(-0.02)


def test_concentration_reports_the_share_carried_by_the_best_days():
    rows = [("2020-01-0%d" % (i + 1), 10.0, 5.0) for i in range(4)]
    rows.append(("2020-01-05", 60.0, 5.0))  # une seule séance porte 60 % du net
    conc = campaign.concentration(_trades(rows))

    assert conc["total_net"] == pytest.approx(100.0)
    assert conc["share_top5_days"] == pytest.approx(1.0)
    assert conc["share_top10_trades"] == pytest.approx(1.0)
    assert conc["share_best_month"] == pytest.approx(1.0)
    assert conc["pnl_top5_days"] == pytest.approx(100.0)
    assert conc["n_profitable_days"] == 5
    assert conc["n_losing_days"] == 0


def test_concentration_of_a_losing_sample_is_not_dressed_up():
    """Un net total négatif rend la part ininterprétable : `None`, pas un chiffre."""
    conc = campaign.concentration(
        _trades([("2020-01-01", 10.0, 5.0), ("2020-01-02", -50.0, 5.0)])
    )
    assert conc["total_net"] == pytest.approx(-40.0)
    assert np.isnan(conc["share_top5_days"])
    # Les montants, eux, restent publiables : c'est ce que la note commente.
    assert conc["pnl_top5_days"] == pytest.approx(-40.0)
    assert conc["gross_profit"] == pytest.approx(10.0)
    assert conc["n_profitable_days"] == 1
    assert conc["n_losing_days"] == 1


# ═══════════════════════════════════════════════════════════════════════
# 4. LA TABLE DE DÉCISION (§1), SUR UN DICT FACTICE
# ═══════════════════════════════════════════════════════════════════════


def _passing_measures() -> dict:
    """Un jeu de mesures qui passe les sept critères mesurables du §1."""
    return {
        "n_trades": 500,
        "trades_per_scenario": {
            "BREAK_LONG": 200,
            "BREAK_SHORT": 150,
            "REV_LONG": 100,
            "REV_SHORT": 50,
        },
        "expectancy_r": 0.20,
        "profit_factor": 1.40,
        "dsr": 0.99,
        "pbo": 0.10,
        "n_trials_logged": 34,
        "plateau": {
            "share_neighbours_positive": 1.0,
            "median_neighbour_sharpe": 1.2,
            "peak_sharpe": 1.5,
            "n_passing_plateau": 9,
            "selected_exists": True,
        },
        "walk_forward": {"share_positive_years": 0.85, "max_year_share_of_net": 0.30},
        "expectancy_r_spread_x1_5": 0.12,
    }


def test_the_table_has_the_nine_criteria_of_the_frozen_document():
    table = campaign.evaluate_decision_table(_passing_measures())
    assert table["n_criteria"] == 9
    assert set(table["criteria"]) == {
        "trades_in_sample",
        "expectancy_and_profit_factor",
        "dsr",
        "pbo",
        "plateau",
        "walk_forward",
        "spread_x1_5",
        "mt5_model_1",
        "oos_2026",
    }


def test_an_unmeasured_criterion_fails_and_never_says_not_applicable():
    """« Un critère non mesuré vaut échec, jamais "non applicable" » (§1)."""
    table = campaign.evaluate_decision_table(_passing_measures())
    for key in ("mt5_model_1", "oos_2026"):
        assert table["criteria"][key]["measured"] == campaign.NOT_MEASURED
        assert table["criteria"][key]["pass"] is False
    # Les sept mesurables passent : le seul échec est l'absence de mesure, donc
    # le verdict provisoire in-sample ne dit PAS « ne pas déployer ».
    assert table["failed"] == ["mt5_model_1", "oos_2026"]
    assert table["failed_with_an_interpretable_measure"] == []
    assert table["provisional_in_sample_verdict"] == "DÉPLOIEMENT À BLANC (démo)"


def test_one_interpretable_failure_is_enough_to_say_do_not_deploy():
    measures = _passing_measures()
    measures["expectancy_r"] = -0.16
    table = campaign.evaluate_decision_table(measures)

    criterion = table["criteria"]["expectancy_and_profit_factor"]
    assert criterion["pass"] is False
    assert criterion["measured"]["expectancy_r"] == pytest.approx(-0.16)
    assert "expectancy_and_profit_factor" in table["failed_with_an_interpretable_measure"]
    assert table["provisional_in_sample_verdict"] == "NE PAS DÉPLOYER"


def test_an_empty_plateau_fails_the_plateau_criterion():
    measures = _passing_measures()
    measures["plateau"] = {
        "share_neighbours_positive": 0.0,
        "median_neighbour_sharpe": -1.6,
        "peak_sharpe": -1.17,
        "n_passing_plateau": 0,
        "selected_exists": False,
    }
    table = campaign.evaluate_decision_table(measures)
    assert table["criteria"]["plateau"]["pass"] is False
    assert table["provisional_in_sample_verdict"] == "NE PAS DÉPLOYER"


def test_a_missing_measure_is_a_failure_not_a_crash():
    """Un `None` ou un NaN ne doit jamais passer pour un succès silencieux."""
    table = campaign.evaluate_decision_table(
        {"n_trades": None, "dsr": float("nan"), "pbo": None}
    )
    assert table["n_passing"] == 0
    assert table["provisional_in_sample_verdict"] == "NE PAS DÉPLOYER"


@pytest.mark.parametrize(
    "field,value",
    [
        ("n_trades", 299),
        ("profit_factor", 1.14),
        ("dsr", 0.94),
        ("pbo", 0.36),
        ("expectancy_r_spread_x1_5", 0.0),
    ],
)
def test_each_threshold_bites_exactly_at_the_frozen_value(field, value):
    measures = _passing_measures()
    measures[field] = value
    table = campaign.evaluate_decision_table(measures)
    assert table["failed_with_an_interpretable_measure"], (
        f"{field}={value} doit faire échouer un critère du §1"
    )


def test_the_walk_forward_criterion_needs_both_halves():
    measures = _passing_measures()
    measures["walk_forward"] = {
        "share_positive_years": 0.85,
        "max_year_share_of_net": 0.62,  # une année porte plus de la moitié du net
    }
    table = campaign.evaluate_decision_table(measures)
    assert table["criteria"]["walk_forward"]["pass"] is False


# ═══════════════════════════════════════════════════════════════════════
# 5. SÉRIALISATION
# ═══════════════════════════════════════════════════════════════════════


def test_non_finite_values_are_written_as_null():
    """`NaN` est du JSON invalide pour tout le monde sauf Python."""
    out = campaign.jsonable(
        {"a": float("nan"), "b": np.float64(np.inf), "c": np.int64(3), "d": True}
    )
    assert out == {"a": None, "b": None, "c": 3, "d": True}


def test_the_fallback_is_the_geometric_centre_and_is_never_called_a_selection():
    assert campaign.analysed_index({"selected": None}) == (1, 1, 1)
    assert campaign.analysed_index({"selected": {"index": [2, 0, 1]}}) == (2, 0, 1)
