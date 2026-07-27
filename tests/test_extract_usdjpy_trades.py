"""L'extraction des trades USD/JPY doit échouer plutôt que publier un faux.

Mêmes pièges que l'extraction or (`tests/test_extract_gold_trades.py`) — le
journal multi-runs, l'appariement par ``position_id`` et non par l'ordre des
lignes — plus un piège propre au candidat : le swap est POSITIF (carry du long
dollar-yen) et pèse plus que le prix ; une extraction qui perdrait ou
tronquerait la colonne swap fausserait la thèse centrale du rapport.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

_REPO = Path(__file__).resolve().parent.parent
_SCRIPTS = _REPO / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import extract_usdjpy_trades as eut  # noqa: E402
import parse_mt5_report as pmr  # noqa: E402


# Deux positions USDJPY dont les deals s'entrelacent chronologiquement, plus un
# deal d'une autre sleeve et le dépôt : l'appariement positionnel les mélange.
_CSV = """\
deal_id,position_id,time_utc,symbol,magic,sleeve,type,entry,volume,price,profit,commission,swap
1,0,2021.01.01 00:00:00,,0,OTHER,2,0,0.0000,0.00000,100000.00,0.0000,0.0000
2,410,2021.09.14 21:00:17,USDJPY.c,835,GOLD_MOMENTUM,0,0,1.9500,109.68800,0.00,0.0000,0.0000
3,425,2021.10.01 21:00:17,USDJPY.c,835,GOLD_MOMENTUM,0,0,0.5000,111.05000,0.00,0.0000,0.0000
4,410,2021.11.24 21:00:17,USDJPY.c,835,GOLD_MOMENTUM,1,1,1.9500,115.37000,9585.00,0.0000,850.2500
5,900,2021.12.01 21:00:17,EURUSD.c,832,TS_MOMENTUM,0,0,0.0400,1.21078,0.00,0.0000,0.0000
6,900,2021.12.05 21:00:17,EURUSD.c,832,TS_MOMENTUM,1,1,0.0400,1.25000,1000.00,0.0000,-10.0000
7,425,2022.06.13 06:08:40,USDJPY.c,835,GOLD_MOMENTUM,1,1,0.5000,134.00000,8500.00,0.0000,1200.0000
"""


@pytest.fixture
def deals(tmp_path: Path) -> pd.DataFrame:
    path = tmp_path / "deals.csv"
    path.write_text(_CSV, encoding="utf-16")
    return pmr.load_deals(path)


def test_pairs_by_position_id_not_row_order(deals: pd.DataFrame) -> None:
    trades = eut.pair_usdjpy_trades(deals)
    assert list(trades["position_id"]) == [410, 425]
    t410 = trades.set_index("position_id").loc[410]
    assert t410["entry_price"] == pytest.approx(109.688)
    assert t410["exit_price"] == pytest.approx(115.37)


def test_only_the_usdjpy_sleeve_rows_are_kept(deals: pd.DataFrame) -> None:
    trades = eut.pair_usdjpy_trades(deals)
    assert len(trades) == 2  # ni le dépôt, ni la position TS_MOMENTUM


def test_positive_swap_survives_extraction(deals: pd.DataFrame) -> None:
    """Le carry est la thèse du rapport : le swap doit rester signé et sommé."""
    trades = eut.pair_usdjpy_trades(deals)
    assert trades["swap"].sum() == pytest.approx(850.25 + 1200.0)
    assert (trades["swap"] > 0).all()


def test_exit_out_of_session_is_flagged(deals: pd.DataFrame) -> None:
    """Une sortie hors 21:00 UTC n'a pas été décidée par le signal."""
    trades = eut.pair_usdjpy_trades(deals)
    by_id = trades.set_index("position_id")
    assert not bool(by_id.loc[410, "safety_stop"])
    assert bool(by_id.loc[425, "safety_stop"])


def test_published_artifacts_match_the_reference_run() -> None:
    """Le CSV publié reste accroché au run de recherche qui l'a produit."""
    csv_path = _REPO / "reports/mt5/usdjpy_trades_research.csv"
    if not csv_path.exists():
        pytest.skip("artefact non généré sur ce poste")
    trades = pd.read_csv(csv_path)
    assert len(trades) == 35
    assert trades["net"].sum() == pytest.approx(176822.48, abs=0.5)
    # Thèse du rapport : le carry pèse plus de la moitié du net.
    assert trades["swap"].sum() > 0.5 * trades["net"].sum()
