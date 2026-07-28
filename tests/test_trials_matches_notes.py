"""Le registre d'essais doit correspondre aux budgets déclarés dans les notes.

Le registre est la source de vérité du ``n_trials`` qui déflate les Sharpe
publiés. Rien ne le confrontait à ce que les notes de phase annoncent, si bien
que ``tsmom_universe`` a pu compter 147 essais — 7 re-runs d'un espace de 21
configurations — sans qu'aucun test ne rougisse. Ce fichier ferme ce trou.

Deux mesures cohabitent et il faut les garder distinctes :

- ``total_trials`` compte les runs, re-runs inclus. C'est une borne
  conservatrice, et c'est ce qui a été publié jusqu'au 2026-07-28 ;
- ``distinct_trials`` replie les re-runs par ``config_key`` et compte les
  **espaces de configurations** réellement explorés. C'est le chiffre qui a un
  sens statistique, et celui que ce test verrouille.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from framework import trials  # noqa: E402

# Budget déclaré par famille, tel qu'écrit dans les notes de phase.
# Source : docs/research/momentum_expansion_2026H2.md §4.1/§4.4/§4.5,
# momentum_integration_2026H2.md §3, phase21_2026-04-13_dsr_retrofit.md.
DECLARED: dict[str, int] = {
    "fx_legacy": 290,          # seed des phases 18-21
    "tsmom_universe": 21,      # 21 instruments, une seule configuration
    "tsmom_stability": 36,     # 3 grilles x 3 target_vol x 4 instruments
    "allow_short": 3,
    "donchian": 6,             # N {55,100,252} x sortie {N/2, N/4}
    "dual_momentum": 4,
    "xs_momentum": 6,          # lookback {63,126,252} x {long-only, long-short}
    "mt5_phase4_checks": 5,
    "integration_weights": 11,  # baseline + 2 compositions x 5 poids
    "gold_stop_spec": 0,       # correction de spécification dérivée
}


@pytest.mark.parametrize("family,expected", sorted(DECLARED.items()))
def test_distinct_trials_matches_the_declared_budget(family, expected):
    got = trials.distinct_trials(family)
    assert got == expected, (
        f"famille {family!r} : {got} configurations distinctes au registre "
        f"contre {expected} déclarées dans la note de phase. Soit un sweep a "
        f"été relancé sans config_key (le re-run compte double), soit la "
        f"grille a changé sans que la note suive."
    )


def test_no_family_is_logged_without_being_declared():
    """Un sweep nouveau doit entrer dans DECLARED, donc dans une note."""
    logged = {
        e["family"]
        for e in trials._read()
        if e.get("kind") != "annotation"
    }
    undeclared = sorted(logged - set(DECLARED))
    assert not undeclared, (
        f"familles loguées mais non déclarées : {undeclared}. Documenter leur "
        f"budget dans une note de phase et l'ajouter à DECLARED."
    )


def test_the_published_totals_hold():
    """Les trois chiffres que les livrables citent."""
    assert trials.distinct_trials() == sum(DECLARED.values()) == 382
    assert trials.distinct_trials() - trials.distinct_trials("fx_legacy") == 92
    assert trials.total_trials() >= trials.distinct_trials(), (
        "le total brut ne peut pas être inférieur au distinct"
    )
