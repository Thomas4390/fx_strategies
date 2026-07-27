"""La sleeve momentum combinée doit diviser par les *n configurés*, pas par n présents.

Deux règles du sweep de poids (`scripts/sweep_momentum_weights.py`) tiennent tout
le reste du tableau :

* un instrument sans séance ce jour-là rend 0 et **ne redistribue pas son poids**
  aux autres. Une moyenne qui saute les absents (le `mean(axis=1)` d'une frame
  laissée avec ses NaN) fabrique du rendement les jours creux — l'or seul un jour
  férié japonais compterait pour toute la sleeve, alors que le portage MQL5 ne
  lui alloue jamais que `sub_equity/n` ;
* les poids du portefeuille somment à 1,0 exactement et la réduction porte sur
  MR Macro seul : TS et RSI ne bougent pas d'un config à l'autre, sans quoi le
  sweep changerait deux choses à la fois.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import sweep_momentum_weights as smw  # noqa: E402


def _series(dates: list[str], values: list[float]) -> pd.Series:
    return pd.Series(values, index=pd.DatetimeIndex(dates))


def test_momentum_sleeve_divise_par_les_instruments_configures():
    """Le 2026-01-02 seul l'or trade : la sleeve rend 0,02/2, pas 0,02."""
    rets = {
        "XAU-USD": _series(["2026-01-01", "2026-01-02"], [0.01, 0.02]),
        "USD-JPY": _series(["2026-01-01"], [0.03]),
    }
    sleeve = smw.momentum_sleeve(rets)

    assert sleeve.loc["2026-01-01"] == pytest.approx(0.02)   # (0.01 + 0.03) / 2
    assert sleeve.loc["2026-01-02"] == pytest.approx(0.01)   # (0.02 + 0.00) / 2
    assert sleeve.name == smw.MOMENTUM_KEY


def test_momentum_sleeve_couvre_l_union_des_dates():
    """Une date portée par un seul instrument reste dans la sleeve."""
    rets = {
        "XAU-USD": _series(["2026-01-01"], [0.01]),
        "XAG-USD": _series(["2026-01-05"], [0.04]),
    }
    sleeve = smw.momentum_sleeve(rets)

    assert list(sleeve.index) == [pd.Timestamp("2026-01-01"), pd.Timestamp("2026-01-05")]
    assert sleeve.tolist() == pytest.approx([0.005, 0.02])


@pytest.mark.parametrize("w", smw.MOMENTUM_WEIGHTS)
def test_portfolio_weights_somme_exactement_a_un(w: float):
    weights = smw.portfolio_weights(w)

    assert sum(weights.values()) == pytest.approx(1.0, abs=1e-12)
    assert weights[smw.MOMENTUM_KEY] == w
    assert weights["TS_Momentum_3p"] == 0.09
    assert weights["RSI_Daily_3p"] == 0.09


def test_portfolio_weights_ne_reduit_que_mr_macro():
    """w = 0,10 doit redonner la production actuelle ; +1 pp de momentum = -1 pp de MR."""
    assert smw.portfolio_weights(0.10)["MR_Macro"] == pytest.approx(0.72)
    assert smw.portfolio_weights(0.20)["MR_Macro"] == pytest.approx(0.62)
