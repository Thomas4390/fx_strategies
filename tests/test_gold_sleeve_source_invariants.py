"""Invariants de source de la sleeve momentum MQL5, que rien d'autre ne teste.

Le MQL5 ne tourne pas sous pytest : ces vérifications lisent le fichier. C'est
grossier, et c'est assumé — elles ne vérifient pas un comportement, elles
empêchent une régression précise dont on connaît le coût.

Deux constantes du fichier sont **inertes aux réglages livrés**, et l'audit du
2026-07-28 a conclu qu'il ne fallait pas les « corriger » :

- le plancher de volatilité ``0.05`` ne peut mordre que si
  ``TargetVol / 0.05 < MaxLeverage``, or ``0.55 / 0.05 = 11 > 6.6`` ;
- le repli de sigma ``0.16`` est inatteignable, parce que ``ComputeSigma21``
  n'est appelé qu'après un ``ComputeScore`` qui exige 252 barres — et qui a
  252 barres en a 22.

Ces deux propriétés sont vraies *par construction du code appelant*, pas par
nature. Un réordonnancement d'appels ou un retune réveillerait la constante
sans qu'aucune métrique ne bouge. D'où ces tests.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_SLEEVE = (
    Path(__file__).resolve().parent.parent
    / "src/mt5/Include/FxSleeveGoldMomentum.mqh"
)


@pytest.fixture(scope="module")
def source() -> str:
    return _SLEEVE.read_text(encoding="utf-8-sig")


def _process_symbol_body(source: str) -> str:
    """Corps de ``ProcessSymbol``, du début jusqu'à la méthode suivante."""
    start = source.index("void ProcessSymbol(")
    rest = source[start:]
    end = rest.index("\n    //--- Append one row")
    return rest[:end]


def test_sigma_is_only_reached_after_the_score_gate(source):
    """``ComputeScore`` doit garder ``ComputeSigma21``, sinon le repli 0.16 vit.

    Le repli de sigma est calibré sur l'or et serait appliqué tel quel à
    USD/JPY (vol ~7,6 %) et à l'argent (~25 %). Il n'est jamais atteint
    aujourd'hui, uniquement parce que le garde de 252 barres passe en premier.
    """
    body = _process_symbol_body(source)
    gate = body.index("if(!ComputeScore(")
    sigma = body.index("ComputeSigma21(")
    assert gate < sigma, (
        "ComputeSigma21 est appelé avant le garde ComputeScore : le repli de "
        "sigma 0.16, calibré sur l'or, devient atteignable sur les autres "
        "instruments. Voir docs/investigations/gold_stop_decoupling_2026-07-28.md."
    )
    assert "return;" in body[gate:sigma], (
        "le garde ComputeScore ne sort plus de la fonction — il ne garde rien"
    )


def test_the_inert_volatility_floor_is_announced_at_init(source):
    """Un retune qui réveille le plancher 0.05 doit le dire, pas le subir."""
    assert "Inp_Gold_TargetVol / 0.05 < Inp_Gold_MaxLeverage" in source, (
        "l'avertissement d'Init sur le plancher de volatilité a disparu : un "
        "futur retune pourrait le rendre actif sans que rien ne le signale"
    )


def test_sizing_and_protection_distances_stay_separate(source):
    """Le stop et le dénominateur du dimensionnement ne doivent pas refusionner.

    Ils étaient la même variable jusqu'au 2026-07-28 : élargir le stop divisait
    le notionnel d'autant, ce qui mêlait deux effets dans un seul chiffre.
    """
    assert "sl_dist_sizing" in source, "la distance de dimensionnement a disparu"
    assert re.search(r"LotsForRisk\([^)]*sl_dist_sizing\)", source), (
        "LotsForRisk ne dimensionne plus sur sl_dist_sizing : le stop et le "
        "dimensionnement sont de nouveau couplés"
    )
    assert "FX_GOLD_SL_SIGMAS" in source, "le plancher en sigmas a disparu"


def test_the_traced_instrument_is_selectable(source):
    """Sans cela, la parité n'est vérifiable que sur le premier instrument."""
    assert "TracedSymbol()" in source
    assert "Inp_Gold_TraceSymbol" in source
