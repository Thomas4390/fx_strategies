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
    """Un JSON d'une autre configuration produit un rapport qui se contredit."""
    if not STRESS_JSON.exists():  # pragma: no cover - artefact non versionné
        pytest.skip(f"{STRESS_JSON} absent : relancer stress_test_combined.py")

    payload = json.loads(STRESS_JSON.read_text())

    assert payload["custom_weights"] == PRODUCTION_WEIGHTS
    assert payload["config"]["target_vol"] == PRODUCTION_TARGET_VOL
    assert payload["config"]["max_leverage"] == PRODUCTION_MAX_LEVERAGE
    # La bande de CAGR visée doit être publiée, sinon la ligne « Cible
    # atteinte » du rapport ne dit pas contre quel mandat elle est mesurée.
    assert len(payload["target_cagr_band"]) == 2
