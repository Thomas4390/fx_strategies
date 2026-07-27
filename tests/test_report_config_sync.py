"""Les scripts qui produisent le rapport client doivent décrire le portefeuille réel.

Le rapport a publié pendant trois mois des tables et des légendes bâties sur
80/10/10 à ``target_vol=0.28`` / ``max_leverage=12`` alors que la production
était passée à 72/9/9/10 à 0.37 / 31, sleeve or comprise. Rien ne l'a signalé :
chaque script portait sa propre copie de la configuration, et les six tests de
``test_stress_sanity.py`` ne vérifient que des formes — celui qui compte les 18
lignes du balayage est resté vert quand la grille a entièrement changé.

Ces tests comparent les copies à la source au lieu de faire confiance à la
discipline. Ils sont le pendant, côté rapport, de ``test_mt5_preset_sync.py``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
_SCRIPTS = _ROOT / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from strategies.combined_portfolio_v2 import (  # noqa: E402
    PRODUCTION_MAX_LEVERAGE,
    PRODUCTION_TARGET_VOL,
    PRODUCTION_WEIGHTS,
)

STRESS_JSON = _ROOT / "results/production_report/stress_test_report.json"
MT5_JSON = _ROOT / "results/production_report/mt5_reference.json"


def test_stress_test_runs_the_production_config():
    import stress_test_combined as stc

    assert stc.RECOMMENDED_CONFIG["custom_weights"] == PRODUCTION_WEIGHTS
    assert stc.RECOMMENDED_CONFIG["target_vol"] == PRODUCTION_TARGET_VOL
    assert stc.RECOMMENDED_CONFIG["max_leverage"] == PRODUCTION_MAX_LEVERAGE


def test_stress_test_writes_where_the_report_reads_it():
    """Les deux chemins ont divergé : relancer le test ne changeait rien au rapport."""
    import build_latex_report_assets as assets

    source = (_SCRIPTS / "stress_test_combined.py").read_text()

    assert str(assets.STRESS_JSON.relative_to(_ROOT)) in source


def test_weight_sensitivity_uses_the_production_risk_layer():
    import generate_weight_sensitivity_figures as wsf

    assert wsf.TARGET_VOL == PRODUCTION_TARGET_VOL
    assert wsf.MAX_LEVERAGE == PRODUCTION_MAX_LEVERAGE
    assert wsf.FIXED_WEIGHT == PRODUCTION_WEIGHTS[wsf.FIXED_KEY]
    # Le simplexe balaie le trio FX ; l'or est servi à part.
    assert set(wsf.SLEEVE_KEYS) | {wsf.FIXED_KEY} == set(PRODUCTION_WEIGHTS)


def test_weight_sensitivity_points_sum_to_one():
    import generate_weight_sensitivity_figures as wsf

    weights = wsf.make_weights(0.8, 0.1, 0.1)

    assert sum(weights.values()) == pytest.approx(1.0)
    assert weights == pytest.approx(PRODUCTION_WEIGHTS)


def test_every_production_sleeve_has_a_label_and_a_colour():
    """Une sleeve sans entrée ici disparaît des figures sans erreur."""
    import build_latex_report_assets as assets

    for key in PRODUCTION_WEIGHTS:
        assert key in assets.SLEEVE_DISPLAY
        assert assets.SLEEVE_COLOR_KEY[key] in assets.PALETTE
        assert f"{PRODUCTION_WEIGHTS[key] * 100:.0f}" in assets.sleeve_label_tex(key)


def test_published_stress_json_matches_the_current_config():
    """Un JSON d'une autre configuration produit un rapport qui se contredit.

    Ce test **échoue** quand l'artefact manque, il ne saute pas. Il a sauté
    pendant trois mois : ``results/`` était intégralement gitignoré, donc le
    fichier était absent par défaut sur tout clone frais et en CI. Le seul test
    qui comparait les artefacts publiés à la configuration courante était
    silencieusement inactif là où il aurait le plus servi.

    Les deux JSON de référence sont désormais versionnés (voir ``.gitignore``) :
    l'absence du fichier est un vrai défaut, pas un environnement incomplet.
    """
    assert STRESS_JSON.exists(), (
        f"{STRESS_JSON.relative_to(_ROOT)} absent. Ce fichier est versionné et "
        f"alimente les tables du rapport client : son absence rend le document "
        f"publié irreproductible. Le régénérer avec "
        f"`python scripts/stress_test_combined.py`."
    )

    payload = json.loads(STRESS_JSON.read_text())

    assert payload["custom_weights"] == PRODUCTION_WEIGHTS
    assert payload["config"]["target_vol"] == PRODUCTION_TARGET_VOL
    assert payload["config"]["max_leverage"] == PRODUCTION_MAX_LEVERAGE
    # La bande de CAGR visée doit être publiée, sinon la ligne « Cible
    # atteinte » du rapport ne dit pas contre quel mandat elle est mesurée.
    assert len(payload["target_cagr_band"]) == 2


def test_published_mt5_reference_covers_every_allocated_sleeve():
    """Le JSON MT5 doit décrire le portefeuille réellement alloué.

    ``build_mt5_assets()`` se contentait d'un avertissement quand ce fichier
    manquait : les trois tables MT5 gardaient alors le contenu du run précédent
    et la chaîne rendait un rapport d'apparence complète. Le fichier est
    désormais versionné et le générateur lève ; ce test ferme la boucle côté
    configuration.

    L'assertion porte sur le *nombre* de sleeves plutôt que sur leurs noms : les
    libellés MT5 (« Gold Momentum ») et les clés Python (``Gold_Momentum``) ne
    coïncident pas, et un test qui recopierait la correspondance serait une
    troisième copie à maintenir.
    """
    assert MT5_JSON.exists(), (
        f"{MT5_JSON.relative_to(_ROOT)} absent. Ce fichier est versionné et "
        f"porte les chiffres publiés au client. Le régénérer avec "
        f"`python scripts/parse_mt5_report.py`."
    )

    payload = json.loads(MT5_JSON.read_text())
    traded = {
        row["sleeve"] for row in payload["by_sleeve"]
        if row["sleeve"] != "Hors sleeve"
    }
    allocated = {k for k, w in PRODUCTION_WEIGHTS.items() if w > 0}
    assert len(traded) == len(allocated), (
        f"{len(traded)} sleeve(s) dans mt5_reference.json ({sorted(traded)}) "
        f"contre {len(allocated)} allouée(s) en production ({sorted(allocated)}). "
        f"Le backtest MT5 de référence ne décrit pas la configuration courante : "
        f"le relancer, puis scripts/parse_mt5_report.py."
    )


# ═══════════════════════════════════════════════════════════════════════
# Marqueur « production » des figures de sensibilité aux poids
# ═══════════════════════════════════════════════════════════════════════

_MARKED_FIGURES = ("plot_1d_sweep", "plot_simplex_ternary", "plot_pareto_frontier")


def test_weight_sensitivity_production_marker_is_derived_not_written():
    """Les trois figures marquaient le point de production « 80 / 10 / 10 » en dur.

    Ce libellé était le bon *par coïncidence* : 0.72/0.09/0.09 renormalisés sur
    les 90 % du trio donnent bien 80/10/10. Rien ne le rattachait à
    ``PRODUCTION_WEIGHTS`` — un changement d'allocation aurait laissé les six
    littéraux en place et les figures auraient étiqueté le mauvais point, en
    silence, pendant que les tables suivaient.
    """
    import ast

    import generate_weight_sensitivity_figures as wsf

    fx_total = sum(PRODUCTION_WEIGHTS[k] for k in wsf.SLEEVE_KEYS)
    expected = tuple(PRODUCTION_WEIGHTS[k] / fx_total for k in wsf.SLEEVE_KEYS)

    assert wsf.PRODUCTION_INTERNAL == pytest.approx(expected)
    assert sum(wsf.PRODUCTION_INTERNAL) == pytest.approx(1.0)
    assert wsf.PRODUCTION_ALLOC_LABEL == " / ".join(f"{w * 100:.0f}" for w in expected)

    source = Path(wsf.__file__).read_text()
    bodies = {
        node.name: ast.get_source_segment(source, node)
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.FunctionDef) and node.name in _MARKED_FIGURES
    }
    assert set(bodies) == set(_MARKED_FIGURES)

    for name, body in bodies.items():
        assert "PRODUCTION_INTERNAL" in body or "PRODUCTION_ALLOC_LABEL" in body, (
            f"{name} ne dérive pas le point de production de la configuration."
        )
        for literal in ("0.80", "80 / 10 / 10", "80/10/10"):
            assert literal not in body, (
                f"{name} localise ou étiquette la production avec le littéral "
                f"{literal!r} : le marqueur ne suivra pas PRODUCTION_WEIGHTS."
            )


def test_weight_sensitivity_marker_follows_a_change_of_allocation():
    """Preuve par mutation : le marqueur bouge quand l'allocation bouge."""
    import importlib

    import generate_weight_sensitivity_figures as wsf
    import strategies.combined_portfolio_v2 as cpv2

    original = cpv2.PRODUCTION_WEIGHTS
    try:
        cpv2.PRODUCTION_WEIGHTS = {
            "MR_Macro": 0.60,
            "TS_Momentum_3p": 0.15,
            "RSI_Daily_3p": 0.15,
            "Gold_Momentum": 0.10,
        }
        importlib.reload(wsf)
        assert wsf.PRODUCTION_INTERNAL == pytest.approx((2 / 3, 1 / 6, 1 / 6))
        assert wsf.PRODUCTION_ALLOC_LABEL == "67 / 17 / 17"
    finally:
        cpv2.PRODUCTION_WEIGHTS = original
        importlib.reload(wsf)

    assert wsf.PRODUCTION_WEIGHTS == original


def test_weight_sensitivity_named_points_all_live_on_the_simplex():
    """Tout ``WeightPoint`` porte des parts *internes* au trio, sommant à 1.

    Le point risk-parity était construit sur ``weights_ts.mean()``, soit les
    poids du portefeuille complet — or compris — sans renormalisation : ses
    coordonnées ne sommaient pas à 1 alors que toutes les autres le font. Rien
    ne le lisait encore, mais le Pareto localise déjà la production par ces
    coordonnées.
    """
    import pandas as pd

    import generate_weight_sensitivity_figures as wsf

    keys = (*wsf.SLEEVE_KEYS, wsf.FIXED_KEY)
    # Poids risk-parity du portefeuille complet : le trio n'y pèse que 65 %.
    weights_ts = pd.DataFrame([[0.30, 0.20, 0.15, 0.35]], columns=list(keys))

    def fake_build(returns, **kwargs):
        return {
            "wf_avg_sharpe": 1.0,
            "annual_return": 0.10,
            "annual_vol": 0.20,
            "max_drawdown": -0.05,
            "weights_ts": weights_ts,
        }

    points = wsf.run_named_allocations({k: pd.Series(dtype=float) for k in keys},
                                       fake_build)

    assert "risk-parity" in {p.label for p in points}
    for p in points:
        assert p.w_mr + p.w_ts + p.w_rsi == pytest.approx(1.0), (
            f"Le point « {p.label} » ne vit pas dans le repère du simplexe : "
            f"({p.w_mr:.4f}, {p.w_ts:.4f}, {p.w_rsi:.4f})."
        )
