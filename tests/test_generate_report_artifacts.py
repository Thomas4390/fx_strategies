"""``generate_report_artifacts.py`` décrit la configuration qu'il a réellement jouée.

Ce script a longtemps produit un bloc ``config`` mensonger : la valeur retenue
passait par un ``... if False else {...}`` qui rendait la branche dérivée des
constantes inatteignable et figeait 80/10/10 à ``target_vol=0.28`` /
``max_leverage=12``, alors que les métriques du même dict étaient calculées sur
``PRODUCTION_WEIGHTS`` / ``PRODUCTION_TARGET_VOL`` / ``PRODUCTION_MAX_LEVERAGE``.
Le même dict était ensuite écrit dans ``results/production_report/stress_test_report.json``
— le chemin canonique de ``stress_test_combined.py``, avec un schéma différent.

Les tests ci-dessous ferment les trois brèches : la config décrite est celle qui
a servi au calcul, le script n'écrit plus l'artefact canonique, et ``summary.txt``
ne contient plus aucun littéral de configuration.
"""

from __future__ import annotations

import ast
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

GENERATOR = _SCRIPTS / "generate_report_artifacts.py"

# Artefact canonique : un seul producteur, scripts/stress_test_combined.py.
CANONICAL_STRESS_JSON = "stress_test_report.json"


# ═══════════════════════════════════════════════════════════════════════
# Helpers — littéraux de code, hors commentaires et docstrings
# ═══════════════════════════════════════════════════════════════════════


def _code_string_constants(source: str) -> list[str]:
    """Toutes les chaînes littérales du module sauf les docstrings.

    ``ast`` supprime déjà les commentaires ; on écarte en plus les docstrings
    de module, de classe et de fonction. Ce qui reste est du texte que le
    script peut réellement utiliser à l'exécution : un chemin de sortie, un
    nom de fichier, une ligne de summary.txt.
    """
    tree = ast.parse(source)

    docstrings: set[int] = set()
    holders = (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
    for node in ast.walk(tree):
        if not isinstance(node, holders):
            continue
        body = node.body
        if (
            body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            docstrings.add(id(body[0].value))

    return [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and id(node) not in docstrings
    ]


@pytest.fixture(scope="module")
def generator_code_strings() -> list[str]:
    return _code_string_constants(GENERATOR.read_text())


# ═══════════════════════════════════════════════════════════════════════
# Un seul producteur pour le JSON canonique
# ═══════════════════════════════════════════════════════════════════════


def test_the_generator_does_not_write_the_canonical_stress_json(generator_code_strings):
    """Deux producteurs d'un même artefact : c'est le dernier lancé qui gagne.

    ``stress_test_combined.py`` écrit ce fichier avec ``custom_weights``,
    ``target_cagr_band``, ``n_bootstrap``, ``block_size`` et ``oos_split_date``.
    ``build_latex_report_assets.py`` lit ces clés et
    ``tests/test_report_config_sync.py`` les vérifie. Un second écrivain au
    schéma plus pauvre transforme l'``AssertionError`` de synchronisation en
    ``KeyError`` et rend ``stress.get("target_cagr_band")`` silencieusement nul.

    L'assertion porte sur les littéraux de code : le docstring du module a le
    droit d'expliquer pourquoi ce fichier n'est pas produit ici.
    """
    offenders = [s for s in generator_code_strings if CANONICAL_STRESS_JSON in s]

    assert offenders == [], (
        f"{GENERATOR.name} manipule à nouveau {CANONICAL_STRESS_JSON} "
        f"({offenders}). Ce fichier n'a qu'un producteur : "
        f"scripts/stress_test_combined.py."
    )


def test_the_final_message_points_at_the_real_output_root(generator_code_strings):
    """Le script annonçait « artifacts in results/phase18/ » et écrivait ailleurs.

    Le garde-fou vise le *chemin* : « Phase 18 » reste un libellé légitime dans
    les titres de figures et de tearsheets.
    """
    import generate_report_artifacts as gra

    stale = [s for s in generator_code_strings if "results/phase18" in s.lower()]

    assert stale == [], f"Chemin de sortie périmé en dur : {stale}"
    assert gra.OUTPUT_ROOT == _ROOT / "results" / "production_report"


# ═══════════════════════════════════════════════════════════════════════
# La config décrite est la config jouée
# ═══════════════════════════════════════════════════════════════════════


class _FakeBootstrapStats:
    def to_dict(self) -> dict[str, float]:
        return {"cagr_mean": 0.0}


@pytest.fixture
def stubbed_stress_suite(monkeypatch):
    """Neutralise les quatre suites lourdes et capture la config qu'elles voient.

    ``run_stress_test`` lance ~1000 backtests ; seule la configuration passée
    aux helpers nous intéresse ici.
    """
    import stress_test_combined as stc

    seen: dict[str, object] = {}

    def fake_bootstrap(returns, n_runs=1000, block_size=20, seed=0):
        # Capture pendant l'exécution : la config est restaurée dans le finally.
        seen["config"] = stc.RECOMMENDED_CONFIG
        seen["n_runs"] = n_runs
        seen["block_size"] = block_size
        return _FakeBootstrapStats()

    monkeypatch.setattr(stc, "run_block_bootstrap", fake_bootstrap)
    monkeypatch.setattr(stc, "run_scenario_replay", lambda returns: [])
    monkeypatch.setattr(stc, "run_is_oos_split", lambda returns: {})
    monkeypatch.setattr(stc, "run_parameter_sensitivity", lambda returns: [])
    return seen


def test_run_stress_test_reports_the_config_it_actually_ran(stubbed_stress_suite):
    """Le bloc ``config`` doit être l'objet qui a piloté le calcul, pas une copie.

    C'est l'assertion qui tue la branche morte : une seconde copie figée
    passerait les vérifications sur ``PRODUCTION_*`` seulement par accident, et
    divergerait silencieusement au premier retune.
    """
    import generate_report_artifacts as gra

    report = gra.run_stress_test({}, n_bootstrap=7, block_size=3)

    assert report["config"] == stubbed_stress_suite["config"]
    assert report["config"]["custom_weights"] == PRODUCTION_WEIGHTS
    assert report["config"]["target_vol"] == PRODUCTION_TARGET_VOL
    assert report["config"]["max_leverage"] == PRODUCTION_MAX_LEVERAGE
    assert report["config"]["dd_cap_enabled"] is False
    # Les paramètres du bootstrap sont publiés, summary.txt les lit de là.
    assert (report["n_bootstrap"], report["block_size"]) == (7, 3)
    assert (stubbed_stress_suite["n_runs"], stubbed_stress_suite["block_size"]) == (7, 3)


def test_run_stress_test_restores_the_module_level_config(stubbed_stress_suite):
    """La surcharge de ``RECOMMENDED_CONFIG`` ne doit pas fuir hors de l'appel."""
    import generate_report_artifacts as gra
    import stress_test_combined as stc

    before = stc.RECOMMENDED_CONFIG
    gra.run_stress_test({})

    assert stc.RECOMMENDED_CONFIG is before


# ═══════════════════════════════════════════════════════════════════════
# summary.txt
# ═══════════════════════════════════════════════════════════════════════


def _synthetic_report() -> dict:
    """Rapport volontairement éloigné de la production.

    Si ``build_summary_text`` recopie une configuration en dur, aucune de ces
    valeurs ne ressortira.
    """
    metrics = {"cagr": 0.1, "vol": 0.2, "sharpe": 1.0, "max_dd": -0.3, "n": 100}
    boot = {
        "cagr_mean": 0.1, "cagr_p05": 0.0, "cagr_p50": 0.1, "cagr_p95": 0.2,
        "max_dd_mean": -0.3, "max_dd_p05": -0.5, "max_dd_p50": -0.3,
        "max_dd_p95": -0.1, "sharpe_mean": 1.0, "pos_fraction": 0.9,
        "target_hit_fraction": 0.5,
    }
    return {
        "config": {
            "allocation": "custom",
            "custom_weights": {"Alpha_Sleeve": 0.65, "Beta_Sleeve": 0.35},
            "target_vol": 0.11,
            "max_leverage": 3.0,
            "dd_cap_enabled": False,
        },
        "n_bootstrap": 250,
        "block_size": 10,
        "bootstrap": boot,
        "is_oos_summary": {
            "split_date": "2030-01-01",
            "in_sample": metrics,
            "out_of_sample": metrics,
            "wf_sharpes": [0.1] * 8,
            "wf_pos_years": 5,
        },
    }


def test_summary_derives_its_header_from_the_report():
    """L'en-tête a annoncé MR80 / 0.28 / 12 pendant que les chiffres disaient autre chose."""
    import generate_report_artifacts as gra

    text = gra.build_summary_text(_synthetic_report())

    assert "Config: Alpha_Sleeve 65% / Beta_Sleeve 35%" in text
    assert "target_vol=0.11, max_leverage=3, DDcap=OFF" in text
    assert "Bootstrap 250 runs, block=10d:" in text
    assert "2030-01-01" in text
    # Le dénominateur des années walk-forward était figé à /7.
    assert "WF positive years: 5/8" in text

    for stale in ("MR80", "target_vol=0.28", "max_leverage=12", "block=20d"):
        assert stale not in text, f"littéral périmé réapparu dans summary.txt : {stale}"


def test_summary_publishes_the_production_configuration(stubbed_stress_suite):
    """Bout en bout : ce que summary.txt affiche vient bien des constantes."""
    import generate_report_artifacts as gra

    report = gra.run_stress_test({})
    report["is_oos_summary"] = _synthetic_report()["is_oos_summary"]
    report["bootstrap"] = _synthetic_report()["bootstrap"]

    text = gra.build_summary_text(report)

    for name, weight in PRODUCTION_WEIGHTS.items():
        assert f"{name} {weight * 100:.0f}%" in text
    assert f"target_vol={PRODUCTION_TARGET_VOL}" in text
    assert f"max_leverage={PRODUCTION_MAX_LEVERAGE:.0f}" in text
