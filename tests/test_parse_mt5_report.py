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
def deals(tmp_path) -> pd.DataFrame:
    path = tmp_path / "deals_test.csv"
    path.write_bytes(_CSV.encode("utf-16"))
    return pmr.load_deals(path)


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
