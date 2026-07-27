"""L'extraction des trades or doit échouer plutôt que publier un faux.

Le rapport d'analyse des trades or descend au trade individuel : une erreur
d'appariement n'y est pas visible, elle produit un tableau plausible et faux.
Trois pièges sont fixés ici :

* le journal du tester mélange **tous les runs de la journée** — 36 runs le jour
  du backtest de production. Sélectionner « le dernier » ou filtrer sur l'heure
  du log donne le mauvais score et le mauvais levier pour chaque trade ;
* les deals doivent être appariés par ``position_id``, jamais par l'ordre des
  lignes : MT5 les écrit triés par date, et une position longue de 214 jours
  s'intercale entre les deux deals d'une position courte ;
* une sortie hors de la borne de séance (21:00 UTC) n'a pas été décidée par le
  signal mais par le stop de sécurité. Confondre les deux fausse l'attribution.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import extract_gold_trades as egt  # noqa: E402
import parse_mt5_report as pmr  # noqa: E402


# Une position or courte (163) dont les deux deals encadrent, dans l'ordre
# chronologique, ceux d'une position or longue (177) : l'appariement positionnel
# les mélangerait.
_CSV = """\
deal_id,position_id,time_utc,symbol,magic,sleeve,type,entry,volume,price,profit,commission,swap
1,0,2021.01.01 00:00:00,,0,OTHER,2,0,0.0000,0.00000,10000.00,0.0000,0.0000
2,163,2021.11.09 21:00:17,XAUUSD.c,835,GOLD_MOMENTUM,0,0,0.1600,1832.46000,0.00,0.0000,0.0000
3,177,2021.11.10 21:00:17,XAUUSD.c,835,GOLD_MOMENTUM,0,0,0.1900,1785.69500,0.00,0.0000,0.0000
4,163,2021.11.24 21:00:17,XAUUSD.c,835,GOLD_MOMENTUM,1,1,0.1600,1788.08000,-710.08,0.0000,-141.4700
5,900,2021.12.01 21:00:17,EURUSD.c,832,TS_MOMENTUM,0,0,0.0400,1.21078,0.00,0.0000,0.0000
6,900,2021.12.05 21:00:17,EURUSD.c,832,TS_MOMENTUM,1,1,0.0400,1.25000,1000.00,0.0000,-10.0000
7,177,2022.06.13 06:08:40,XAUUSD.c,835,GOLD_MOMENTUM,1,1,0.1900,1900.00000,2000.00,0.0000,-500.0000
"""


@pytest.fixture
def deals(tmp_path: Path) -> pd.DataFrame:
    path = tmp_path / "deals.csv"
    path.write_text(_CSV, encoding="utf-16")
    return pmr.load_deals(path)


def test_pairs_by_position_id_not_row_order(deals: pd.DataFrame) -> None:
    """Chaque trade recolle son ouverture et sa sortie, malgré l'entrelacement."""
    trades = egt.pair_gold_trades(deals)

    assert list(trades["position_id"]) == [163, 177]

    first = trades.iloc[0]
    assert first["entry_price"] == pytest.approx(1832.46)
    assert first["exit_price"] == pytest.approx(1788.08)
    assert first["swap"] == pytest.approx(-141.47)
    assert first["net"] == pytest.approx(-710.08 - 141.47)
    assert first["duration_days"] == pytest.approx(15.0)

    second = trades.iloc[1]
    assert second["entry_price"] == pytest.approx(1785.695)
    assert second["exit_price"] == pytest.approx(1900.0)


def test_only_the_gold_sleeve_is_kept(deals: pd.DataFrame) -> None:
    """Le trade EURUSD et le dépôt initial n'entrent pas dans le compte."""
    trades = egt.pair_gold_trades(deals)
    assert len(trades) == 2
    assert not trades["position_id"].isin([0, 900]).any()


def test_out_of_session_exit_is_flagged_as_safety_stop(deals: pd.DataFrame) -> None:
    """Une sortie à 06:08 n'a pas été décidée par le signal."""
    trades = egt.pair_gold_trades(deals).set_index("position_id")
    assert not trades.loc[163, "safety_stop"]
    assert trades.loc[177, "safety_stop"]


def test_incomplete_round_trip_raises(tmp_path: Path) -> None:
    """Un CSV tronqué doit arrêter le script, pas produire un trade en moins."""
    truncated = "\n".join(
        line for line in _CSV.splitlines() if not line.startswith("4,163,")
    )
    path = tmp_path / "truncated.csv"
    path.write_text(truncated + "\n", encoding="utf-16")

    with pytest.raises(SystemExit, match="aller-retour"):
        egt.pair_gold_trades(pmr.load_deals(path))


def test_no_gold_deals_raises(tmp_path: Path) -> None:
    """Un run joué sans allocation or ne doit pas produire un CSV vide."""
    fx_only = "\n".join(
        line for line in _CSV.splitlines() if "GOLD_MOMENTUM" not in line
    )
    path = tmp_path / "fx_only.csv"
    path.write_text(fx_only + "\n", encoding="utf-16")

    with pytest.raises(SystemExit, match="Aucun deal"):
        egt.pair_gold_trades(pmr.load_deals(path))


# ---------------------------------------------------------------------------
# Journal du tester
# ---------------------------------------------------------------------------

_LOG_LINE = (
    "QK\t0\t21:42:{sec:02d}.127\tCore 01\t{dt}   [Gold_Momentum][INFO] "
    "Entry LONG XAUUSD.c lots={lots} price={price} score={score} lev={lev}\n"
)

# Deux runs consécutifs sur la même fenêtre : le second a un sizing différent.
# C'est exactement la situation du sweep, où seul le bon run doit être retenu.
_RUN_A = [("2021.11.09 21:00:17", "0.16", "1832.46", "0.50", "5.04"),
          ("2021.11.10 21:00:17", "0.19", "1785.69", "0.50", "6.46")]
_RUN_B = [("2021.11.09 21:00:17", "0.30", "1832.46", "1.00", "2.34"),
          ("2021.11.10 21:00:17", "0.27", "1785.69", "0.33", "5.40")]


def _write_log(path: Path, *runs: list[tuple[str, str, str, str, str]]) -> Path:
    text = ""
    for run in runs:
        for sec, (dt, lots, price, score, lev) in enumerate(run):
            text += _LOG_LINE.format(
                sec=sec, dt=dt, lots=lots, price=price, score=score, lev=lev
            )
    path.write_text(text, encoding="utf-16")
    return path


def test_log_is_split_into_runs_on_date_rewind(tmp_path: Path) -> None:
    """Le recul de date marque le début d'un nouveau run."""
    runs = egt.parse_log_entries(_write_log(tmp_path / "t.log", _RUN_A, _RUN_B))

    assert len(runs) == 2
    assert list(runs[0]["leverage"]) == [5.04, 6.46]
    assert list(runs[1]["leverage"]) == [2.34, 5.40]


def test_matching_run_is_selected_by_entry_dates(tmp_path: Path) -> None:
    """Le run retenu est celui dont les dates coïncident avec le CSV."""
    runs = egt.parse_log_entries(_write_log(tmp_path / "t.log", _RUN_B, _RUN_A))
    wanted = pd.Series(pd.to_datetime(["2021-11-09 21:00:17", "2021-11-10 21:00:17"]))

    chosen = egt.select_matching_run(runs, wanted)

    # Les deux runs portent les mêmes dates : la sélection doit rester
    # déterministe et rendre le premier qui correspond, pas « le dernier ».
    assert list(chosen["score"]) == [1.00, 0.33]


def test_no_matching_run_raises(tmp_path: Path) -> None:
    """Un journal qui ne contient pas le run doit arrêter le script."""
    runs = egt.parse_log_entries(_write_log(tmp_path / "t.log", _RUN_A))
    wanted = pd.Series(pd.to_datetime(["2024-01-01 21:00:00"]))

    with pytest.raises(SystemExit, match="Aucun run"):
        egt.select_matching_run(runs, wanted)
