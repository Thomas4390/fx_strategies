"""Les agrégats MT5 publiés au client doivent tenir sur le CSV des deals.

Le rapport client ne publie plus les métriques vbt mais celles du moteur qui
exécute. Trois pièges ont été trouvés en construisant cette extraction, tous
silencieux — ils ne cassent rien, ils décalent les chiffres :

* le dépôt initial est un deal de type ``balance`` : compté comme un gain, il
  doublait la base de calcul du CAGR (40 % devenait 26 %) ;
* les positions encore ouvertes au dernier tick sont liquidées par le tester
  avec ``magic = 0`` : 47,7 % du résultat tombait « hors sleeve » ;
* la courbe démarrait au premier trade et non à l'ouverture du backtest,
  raccourcissant la fenêtre du CAGR de trois semaines.

Ces tests fixent les trois comportements.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import parse_mt5_report as pmr  # noqa: E402


# deal_id, position_id, time_utc, symbol, magic, sleeve, type, entry, volume,
# price, profit, commission, swap
_CSV = """\
deal_id,position_id,time_utc,symbol,magic,sleeve,type,entry,volume,price,profit,commission,swap
1,0,2021.01.01 00:00:00,,0,OTHER,2,0,0.0000,0.00000,10000.00,0.0000,0.0000
2,2,2021.02.01 21:00:17,EURUSD.c,832,TS_MOMENTUM,0,0,0.0400,1.21078,0.00,0.0000,0.0000
3,2,2021.03.01 21:00:17,EURUSD.c,832,TS_MOMENTUM,1,1,0.0400,1.25000,1000.00,0.0000,-10.0000
4,4,2021.04.01 21:00:17,XAUUSD.c,835,GOLD_MOMENTUM,0,0,0.2300,1800.00000,0.00,0.0000,0.0000
5,4,2021.12.31 23:59:59,XAUUSD.c,0,OTHER,1,1,0.2300,2000.00000,4000.00,0.0000,0.0000
"""


@pytest.fixture()
def deals_csv(tmp_path) -> Path:
    """Le CSV tel que l'EA l'écrit : UTF-16, sur le disque."""
    path = tmp_path / "deals_test.csv"
    path.write_bytes(_CSV.encode("utf-16"))
    return path


@pytest.fixture()
def deals(deals_csv) -> pd.DataFrame:
    return pmr.load_deals(deals_csv)


def test_forced_close_is_attributed_to_its_sleeve(deals):
    """La liquidation de fin de test porte magic=0 ; sa position dit la sleeve."""
    closed = deals[deals["entry"] == pmr.DEAL_ENTRY_OUT]
    gold = closed[closed["symbol"] == "XAUUSD.c"]

    assert gold["sleeve"].tolist() == ["GOLD_MOMENTUM"]
    assert bool(gold["forced_close"].iloc[0]) is True


def test_strategy_exits_are_not_flagged_as_forced(deals):
    ts_exit = deals[(deals["symbol"] == "EURUSD.c") & (deals["entry"] == 1)]

    assert bool(ts_exit["forced_close"].iloc[0]) is False


def test_deposit_is_not_a_trade(deals):
    curve = pmr.balance_curve(deals, initial_deposit=10_000.0)

    # 10 000 de dépôt + 990 (1000 - 10 de swap) + 4 000, jamais 20 000.
    assert curve.iloc[-1] == pytest.approx(14_990.0)


def test_curve_starts_at_backtest_open_not_first_trade(deals):
    start = pd.Timestamp("2021-01-01")

    curve = pmr.balance_curve(deals, 10_000.0, start=start)

    assert curve.index[0] == start
    assert curve.iloc[0] == pytest.approx(10_000.0)


def test_sleeve_split_excludes_the_deposit_and_sums_to_net(deals):
    rows = pmr._group_metrics(deals, "sleeve")

    by_name = {row["sleeve"]: row for row in rows}
    assert set(by_name) == {"TS Momentum", "Gold Momentum"}
    assert by_name["Gold Momentum"]["net_profit"] == pytest.approx(4_000.0)
    assert by_name["TS Momentum"]["net_profit"] == pytest.approx(990.0)
    assert sum(r["share_of_net_pct"] for r in rows) == pytest.approx(1.0)
    # Le gain latent liquidé d'office doit rester identifiable.
    assert by_name["Gold Momentum"]["forced_close_net"] == pytest.approx(4_000.0)


def test_period_start_is_read_from_the_html_header():
    assert pmr._period_start("M1 (2021.01.01 - 2026.04.30)") == pd.Timestamp(
        "2021-01-01"
    )
    assert pmr._period_start(None) is None


def test_cagr_uses_the_full_window(deals):
    curve = pmr.balance_curve(deals, 10_000.0, start=pd.Timestamp("2021-01-01"))

    # 10 000 → 14 990 sur 364 jours, pas sur la fenêtre amputée qui commencerait
    # au premier trade.
    assert pmr._cagr(curve) == pytest.approx(0.5011, abs=1e-3)


# ---------------------------------------------------------------------------
# Unités du bloc `headline` et provenance du JSON
#
# Deux défauts relevés par l'audit du 2026-07-26 (§12), tous deux de forme :
# le JSON rangeait côte à côte une fraction signée et deux pourcentages positifs
# sous des noms qui se ressemblent, sans stocker la seule grandeur MT5
# comparable à la reconstruction ; et il ne disait pas de quel run il venait.
# ---------------------------------------------------------------------------


# MT5 inverse l'ordre entre les deux champs de repli : « montant (pct%) » pour
# `Maximal`, « pct% (montant) » pour `Relative`. Le fixture reproduit les deux.
_HTML = """\
<html><body><table>
   <tr align="right">
      <td nowrap colspan="3" >Expert:</td>
      <td nowrap colspan="10" align="left"><b>FxMultiSleeve</b></td>
   </tr>
   <tr align="right">
      <td nowrap colspan="3" >Symbol:</td>
      <td nowrap colspan="10" align="left"><b>EURUSD.c</b></td>
   </tr>
   <tr align="right">
      <td nowrap colspan="3" >Period:</td>
      <td nowrap colspan="10" align="left"><b>M1 (2021.01.01 - 2021.12.31)</b></td>
   </tr>
   <tr align="right">
      <td nowrap colspan="3" >Inputs:</td>
      <td nowrap colspan="10" align="left"><b>Inp_AllocGoldMomentum=0.1</b></td>
   </tr>
   <tr align="right">
      <td nowrap colspan="3"></td>
      <td nowrap colspan="10" align="left"><b>Inp_RiskScale=4.5</b></td>
   </tr>
   <tr align="right">
      <td nowrap colspan="3" >Initial Deposit:</td>
      <td nowrap colspan="10" align="left"><b>10 000.00</b></td>
   </tr>
   <tr align="right">
      <td nowrap colspan="3">Total Net Profit:</td>
      <td nowrap><b>4 990.00</b></td>
      <td nowrap colspan="3">Balance Drawdown Maximal:</td>
      <td nowrap><b>3 571.32 (6.66%)</b></td>
      <td nowrap colspan="3">Equity Drawdown Maximal:</td>
      <td nowrap colspan="2"><b>36 727.83 (44.33%)</b></td>
   </tr>
   <tr align="right">
      <td nowrap colspan="3">Gross Loss:</td>
      <td nowrap><b>-35 257.20</b></td>
      <td nowrap colspan="3">Balance Drawdown Relative:</td>
      <td nowrap><b>23.37% (3 263.01)</b></td>
      <td nowrap colspan="3">Equity Drawdown Relative:</td>
      <td nowrap colspan="2"><b>44.33% (36 727.83)</b></td>
   </tr>
   <tr align="right">
      <td nowrap colspan="3">Profit Factor:</td>
      <td nowrap><b>2.14</b></td>
      <td nowrap colspan="3">Recovery Factor:</td>
      <td nowrap><b>1.10</b></td>
      <td nowrap colspan="3">Sharpe Ratio:</td>
      <td nowrap colspan="2"><b>0.89</b></td>
   </tr>
   <tr align="right">
      <td nowrap colspan="3">Total Trades:</td>
      <td nowrap><b>2</b></td>
   </tr>
</table></body></html>
"""

# Un aller-retour perdant puis gagnant dans la même journée (1er juin), suivi
# d'une journée perdante isolée : le `resample("D")` ne voit que la seconde.
_CSV_INTRADAY = """\
deal_id,position_id,time_utc,symbol,magic,sleeve,type,entry,volume,price,profit,commission,swap
1,0,2021.01.01 00:00:00,,0,OTHER,2,0,0.0000,0.00000,10000.00,0.0000,0.0000
2,2,2021.06.01 08:00:00,EURUSD.c,832,TS_MOMENTUM,0,0,0.0400,1.21000,0.00,0.0000,0.0000
3,2,2021.06.01 12:00:00,EURUSD.c,832,TS_MOMENTUM,1,1,0.0400,1.20000,-2000.00,0.0000,0.0000
4,4,2021.06.01 13:00:00,EURUSD.c,832,TS_MOMENTUM,0,0,0.0400,1.20000,0.00,0.0000,0.0000
5,4,2021.06.01 20:00:00,EURUSD.c,832,TS_MOMENTUM,1,1,0.0400,1.22000,2000.00,0.0000,0.0000
6,6,2021.09.01 08:00:00,EURUSD.c,832,TS_MOMENTUM,0,0,0.0400,1.22000,0.00,0.0000,0.0000
7,6,2021.09.02 08:00:00,EURUSD.c,832,TS_MOMENTUM,1,1,0.0400,1.21000,-1000.00,0.0000,0.0000
8,8,2021.11.01 08:00:00,EURUSD.c,832,TS_MOMENTUM,0,0,0.0400,1.21000,0.00,0.0000,0.0000
9,8,2021.11.02 08:00:00,EURUSD.c,832,TS_MOMENTUM,1,1,0.0400,1.24000,3000.00,0.0000,0.0000
"""

_INI = """\
[Tester]
Expert=fx_strategies\\FxMultiSleeve.ex5
Symbol=EURUSD.c
Period=M1
Model=1
Spread=0
FromDate=2021.01.01
ToDate=2021.12.31
Deposit=10000
Leverage=1:100

[TesterInputs]
Inp_ExportDeals=true
"""


@pytest.fixture()
def html_report(tmp_path) -> Path:
    """L'en-tête du rapport HTML, dans l'UTF-16 que MT5 écrit."""
    path = tmp_path / "report_test.htm"
    path.write_bytes(_HTML.encode("utf-16"))
    return path


@pytest.fixture()
def intraday_deals(tmp_path) -> pd.DataFrame:
    path = tmp_path / "deals_intraday.csv"
    path.write_bytes(_CSV_INTRADAY.encode("utf-16"))
    return pmr.load_deals(path)


def test_balance_drawdown_relative_survives_its_inverted_format(html_report):
    """Le champ le plus utile est celui dont le format piège les lecteurs.

    « 23.37% (3 263.01) » met le pourcentage devant : ``_to_pct`` y rendrait
    ``None`` et ``_to_float`` prendrait 23.37 pour un montant.
    """
    header = pmr.load_html_header(html_report)

    assert header["balance_dd_relative_pct"] == pytest.approx(23.37)
    assert header["balance_dd_relative_amount"] == pytest.approx(3_263.01)
    # Et l'autre champ, au format inverse, n'a pas bougé.
    assert header["balance_dd_pct"] == pytest.approx(6.66)
    assert header["balance_dd_amount"] == pytest.approx(3_571.32)


def test_the_drawdown_block_is_uniformly_positive_percent(deals, html_report):
    """Une seule convention, et les deux grandeurs MT5 côte à côte.

    6.66 % est le repli maximal en *monnaie*, 23.37 % le repli maximal en
    *pourcentage* : les confondre était le défaut d'origine.
    """
    header = pmr.load_html_header(html_report)

    block = pmr.build_reference(deals, header)["headline"]["drawdowns"]

    assert "pourcentage positif" in block["unit"]
    numbers = [v for k, v in block.items() if k != "unit"]
    assert all(isinstance(v, float) and v >= 0.0 for v in numbers)
    assert block["balance_max_money_mt5_pct"] == pytest.approx(6.66)
    assert block["balance_relative_mt5_pct"] == pytest.approx(23.37)
    assert block["equity_relative_mt5_pct"] == pytest.approx(44.33)


def test_per_deal_drawdown_catches_the_trough_the_daily_curve_misses(
    intraday_deals, html_report
):
    """Le creux du 1er juin se referme avant la clôture : la journée dit 0.

    C'est l'écart de 0,62 point constaté sur le run publié (22,75 % contre
    23,37 %), réduit ici à un cas minimal.
    """
    header = pmr.load_html_header(html_report)

    block = pmr.build_reference(intraday_deals, header)["headline"]["drawdowns"]

    # Journalier : seul le 2 septembre creuse la courbe, 10 000 → 9 000.
    assert block["balance_relative_daily_pct"] == pytest.approx(10.0)
    # Par deal : l'aller-retour intra-journalier descend jusqu'à 8 000.
    assert block["balance_relative_per_deal_pct"] == pytest.approx(20.0)


def test_the_legacy_headline_keys_keep_their_own_convention(
    intraday_deals, html_report
):
    """`build_latex_report_assets.py` lit ces clés : elles ne bougent pas.

    Elles restent en fraction signée pour la reconstruction et en pourcentage
    positif pour MT5 — c'est précisément le mélange que `drawdowns` corrige,
    sans le casser.
    """
    header = pmr.load_html_header(html_report)

    head = pmr.build_reference(intraday_deals, header)["headline"]

    for key in ("total_net_profit", "cagr", "sharpe_ratio_mt5",
                "equity_dd_pct_mt5", "profit_factor", "total_trades"):
        assert key in head, f"clé consommée par le générateur LaTeX : {key}"
    assert head["balance_dd_pct_daily"] == pytest.approx(-0.10)
    assert head["equity_dd_pct_mt5"] == pytest.approx(44.33)
    # La même grandeur, dans la convention du bloc cohérent.
    assert head["drawdowns"]["balance_relative_daily_pct"] == pytest.approx(10.0)


def test_provenance_fingerprints_the_artefacts_actually_read(
    deals, deals_csv, html_report
):
    """Le nom du CSV vient de l'heure simulée : seuls mtime et sha256 le datent."""
    header = pmr.load_html_header(html_report)

    prov = pmr.build_reference(
        deals, header, deals_path=deals_csv, html_path=html_report
    )["provenance"]

    assert prov["deals_csv"]["path"] == str(deals_csv)
    assert prov["deals_csv"]["sha256"] == hashlib.sha256(
        deals_csv.read_bytes()
    ).hexdigest()
    assert prov["deals_csv"]["mtime_utc"].endswith("Z")
    assert prov["html_report"]["path"] == str(html_report)
    assert prov["generated_at_utc"].endswith("Z")
    assert prov["expert"] == "FxMultiSleeve"
    assert prov["deals_window"]["last_deal_utc"] == "2021-12-31 23:59:59"


def test_provenance_records_the_simulation_model_and_the_ea_inputs(
    deals, deals_csv, html_report, tmp_path
):
    """`Model` n'est ni dans le HTML ni dans le JSON de run — seul le .ini l'a.

    Le run publié tourne en ``Model=1`` (barres M1) alors que le défaut du CLI
    est ``--model 4`` : sans cette trace, rien ne le dit.
    """
    ini_path = tmp_path / "run_test.ini"
    ini_path.write_bytes(_INI.encode("utf-16"))
    run_json = tmp_path / "run_20210101T000000Z.json"
    run_json.write_text(json.dumps({
        "run_id": "20210101T000000Z",
        "ini_path": str(ini_path),
        "metrics": {"report_path": str(html_report)},
    }))
    header = pmr.load_html_header(html_report)

    prov = pmr.build_reference(
        deals, header, deals_path=deals_csv, html_path=html_report,
        run_json_path=run_json,
    )["provenance"]

    assert prov["run_json"]["run_id"] == "20210101T000000Z"
    assert prov["run_json"]["ini_path"] == str(ini_path)
    assert prov["tester"]["model"] == "1"
    assert prov["tester"]["model_label"] == "1 minute OHLC"
    assert prov["ea_inputs"]["Inp_RiskScale"] == "4.5"
    assert prov["ea_inputs"]["Inp_AllocGoldMomentum"] == "0.1"


def test_provenance_refuses_a_run_json_describing_another_report(
    deals, deals_csv, html_report, tmp_path
):
    """`_latest()` rend le run le plus récent, pas celui du rapport lu.

    Recopier son `Model` et ses inputs dans la provenance donnerait une
    configuration crédible et fausse — pire que pas de provenance du tout.
    """
    run_json = tmp_path / "run_20210101T000000Z.json"
    run_json.write_text(json.dumps({
        "run_id": "20210101T000000Z",
        "ini_path": str(tmp_path / "absent.ini"),
        "metrics": {"report_path": str(tmp_path / "un_autre_rapport.htm")},
    }))
    header = pmr.load_html_header(html_report)

    prov = pmr.build_reference(
        deals, header, deals_path=deals_csv, html_path=html_report,
        run_json_path=run_json,
    )["provenance"]

    assert "run_json" not in prov
    assert "tester" not in prov


def test_provenance_flags_a_deals_csv_from_another_run(
    deals, html_report, tmp_path
):
    """Le CSV est choisi par `mtime`, le HTML par le JSON de run.

    Rien ne garantit qu'ils décrivent le même backtest, et l'incohérence est
    muette : elle ne produit pas d'erreur, seulement des chiffres faux.
    """
    other_window = tmp_path / "report_2026.htm"
    other_window.write_bytes(
        _HTML.replace("2021.12.31", "2026.04.30").encode("utf-16")
    )

    matched = pmr.build_provenance(deals, pmr.load_html_header(html_report))
    mismatched = pmr.build_provenance(deals, pmr.load_html_header(other_window))

    assert matched["deals_match_html_period"] is True
    assert mismatched["deals_match_html_period"] is False
