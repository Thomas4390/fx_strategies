"""Tests des fonctions pures de `scripts/reconcile_x10_events.py`.

Trois propriétés portent tout le rapport de réconciliation, et ce sont les trois
que ce fichier vérifie sur des données synthétiques, sans aucun accès réseau ni
au backtest QC :

1. **la reconstruction des trades depuis le journal d'ordres** suit l'intention
   de l'algorithme et non le signe des quantités — c'est ce qui rend lisible le
   cas, réellement observé, d'une sortie refusée par le broker ;
2. **l'appariement** est à ± 1 barre M5, un à un, au plus proche ;
3. **la décomposition en R télescope exactement** : la somme des cinq postes
   vaut l'écart total à la précision machine. Une décomposition qui ne boucle
   pas laisse un résidu où l'on peut cacher n'importe quoi, ce qui est
   précisément ce qu'une réconciliation doit rendre impossible.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "scripts"))

from reconcile_x10_events import (  # noqa: E402
    BAR_MINUTES,
    CONTRACT_SIZE,
    MT5_SUMMARY_KEYS,
    POSTS,
    REFERENCE_HALF_SPREAD,
    VOLUME_MIN,
    SUMMARY_KEYS,
    QcTag,
    _lots_for_risk,
    account_pnl,
    aggregate,
    busy_share,
    clock_offset_check,
    decompose,
    expectancy_table,
    implied_half_spread,
    infer_qc_exit_reason,
    load_mt5_deals,
    load_mt5_trace,
    match_entries,
    match_on_keys,
    mt5_positions,
    parse_tag,
    phantom_inventory,
    read_order_failures,
    reconstruct_qc_trades,
)

FILLED = 3
INVALID = 7
TYPE_MARKET = 0
TYPE_MARKET_ON_OPEN = 4


# ═══════════════════════════════════════════════════════════════════════
# Fabriques
# ═══════════════════════════════════════════════════════════════════════


def order(
    oid: int,
    time: str,
    qty: float,
    price: float,
    tag: str,
    *,
    status: int = FILLED,
    otype: int = TYPE_MARKET,
    message: str | None = None,
) -> dict:
    events = [{"status": "submitted", "message": None}]
    if message is not None:
        events = [{"status": "invalid", "message": message}]
    return {
        "id": oid,
        "time": f"{time}Z" if not time.endswith("Z") else time,
        "quantity": qty,
        "price": price,
        "tag": tag,
        "status": status,
        "type": otype,
        "events": events,
    }


def tag_of(
    scenario: str = "BREAK_LONG",
    level: float = 2000.0,
    ts: str = "2024-01-02 10:00:00",
    stop: float = 1998.0,
    target: float = 2010.0,
    r_est: float = 4.0,
) -> str:
    return f"{scenario}|{level:.3f}|{ts}|{stop:.6f}|{target:.6f}|{r_est:.6f}"


def py_trade(
    ts: str = "2024-01-02 10:00:00",
    scenario: str = "BREAK_LONG",
    level: float = 2000.0,
    stop: float = 1998.0,
    target: float = 2010.0,
    r_est: float = 4.0,
    lots: float = 0.10,
    fill_px: float = 2000.5,
    exit_px: float = 2010.0,
    exit_reason: str = "TARGET",
    equity_before: float = 10_000.0,
    hold_minutes: int = 60,
) -> pd.Series:
    side = 1 if scenario.endswith("LONG") else -1
    decision = pd.Timestamp(ts)
    return pd.Series(
        {
            "ts_decision_utc": decision,
            "ts_fill_utc": decision + pd.Timedelta(minutes=BAR_MINUTES),
            "ts_exit_utc": decision + pd.Timedelta(minutes=hold_minutes),
            "scenario_name": scenario,
            "level": level,
            "stop": stop,
            "target": target,
            "r_est": r_est,
            "lots": lots,
            "q": float(side),
            "fill_px": fill_px,
            "exit_px": exit_px,
            "exit_reason_name": exit_reason,
            "risk_amount": abs(fill_px - stop) * lots * CONTRACT_SIZE,
            "equity_before": equity_before,
            "bars_held": hold_minutes / BAR_MINUTES,
        }
    )


def py_frame(rows: list[pd.Series]) -> pd.DataFrame:
    return pd.DataFrame(rows).sort_values("ts_decision_utc").reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════════
# Le tag
# ═══════════════════════════════════════════════════════════════════════


def test_parse_tag_reads_the_six_fields():
    parsed = parse_tag("REV_SHORT|2060.000|2024-01-05 15:15:00|2065.383612|2050.000000|1.315550")
    assert parsed == QcTag(
        scenario="REV_SHORT",
        level=2060.0,
        ts_decision=pd.Timestamp("2024-01-05 15:15:00"),
        stop=2065.383612,
        target=2050.0,
        r_est=1.315550,
    )


@pytest.mark.parametrize(
    "tag",
    [
        "Margin Call",
        "",
        "BREAK_LONG|2000.000|2024-01-02 10:00:00",
        "NOT_A_SCENARIO|2000.000|2024-01-02 10:00:00|1.0|2.0|3.0",
        "BREAK_LONG|abc|2024-01-02 10:00:00|1.0|2.0|3.0",
    ],
)
def test_parse_tag_rejects_anything_that_is_not_a_strategy_order(tag):
    assert parse_tag(tag) is None


def test_entry_and_exit_of_one_trade_share_the_tag_key_but_not_the_stamp():
    entry = parse_tag(tag_of(ts="2024-01-02 10:00:00"))
    exit_ = parse_tag(tag_of(ts="2024-01-02 11:30:00"))
    assert entry.key == exit_.key
    assert entry.ts_decision != exit_.ts_decision


# ═══════════════════════════════════════════════════════════════════════
# Reconstruction des trades
# ═══════════════════════════════════════════════════════════════════════


def test_reconstruct_pairs_orders_two_by_two():
    t1, t2 = tag_of(level=2000.0), tag_of(level=2010.0, stop=2008.0, target=2020.0)
    ledger = reconstruct_qc_trades(
        [
            order(1, "2024-01-02T10:05:00", 10, 2000.5, t1),
            order(2, "2024-01-02T11:00:00", -10, 2010.0, t1),
            order(3, "2024-01-03T10:05:00", 8, 2010.5, t2),
            order(4, "2024-01-03T11:00:00", -8, 2005.0, t2),
        ]
    )
    assert len(ledger.trades) == 2
    assert all(t.closed for t in ledger.trades)
    assert ledger.trades[0].pnl == pytest.approx(10 * (2010.0 - 2000.5))
    assert ledger.trades[1].pnl == pytest.approx(8 * (2005.0 - 2010.5))
    assert ledger.tag_key_mismatches == []


def test_reconstruct_is_not_fooled_by_the_sign_of_the_quantity():
    """Deux ventes de suite : une entrée courte, puis une sortie courte refusée.

    Un appariement fondé sur le signe verrait deux entrées courtes. L'algorithme,
    lui, n'a émis qu'une position : c'est son intention qui fait foi.
    """
    t1 = tag_of("BREAK_SHORT", stop=2002.0, target=1990.0)
    t2 = tag_of("BREAK_SHORT", level=1990.0, ts="2024-01-02 12:00:00", stop=1992.0, target=1980.0)
    ledger = reconstruct_qc_trades(
        [
            order(1, "2024-01-02T10:05:00", -10, 2000.0, t1),
            order(
                2,
                "2024-01-02T11:00:00",
                10,
                0.0,
                t1,
                status=INVALID,
                otype=TYPE_MARKET_ON_OPEN,
                message="OandaBrokerageModel does not support MarketOnOpen order type",
            ),
            order(3, "2024-01-02T12:05:00", -5, 1990.0, t2),
            order(4, "2024-01-02T13:00:00", 5, 1980.0, t2),
        ]
    )
    assert len(ledger.trades) == 2
    assert ledger.trades[0].closed is False
    assert ledger.trades[0].exit_order_type == TYPE_MARKET_ON_OPEN
    assert "MarketOnOpen" in ledger.trades[0].exit_reject_reason
    assert len(ledger.rejected_exits) == 1
    # La seconde position est bien reconnue comme une ENTRÉE, pas comme la
    # sortie de la première : c'est tout l'enjeu.
    assert ledger.trades[1].closed is True
    assert ledger.trades[1].pnl == pytest.approx(-5 * (1980.0 - 1990.0))


def test_a_rejected_entry_still_consumes_its_exit_order():
    """`main.py:205-206` pose `_open_qty` avant d'envoyer l'ordre et ne lit jamais son sort.

    L'entrée est refusée, la position logique existe quand même, et la SORTIE
    part — puis se remplit, sur une position que personne n'a ouverte. Un
    appariement qui « saute » l'entrée refusée décalerait tout ce qui suit d'un
    ordre : c'est le défaut que ce test verrouille.
    """
    t1 = tag_of()
    t1_exit = tag_of(ts="2024-01-02 11:00:00")
    t2 = tag_of(level=2010.0, ts="2024-01-02 12:00:00", stop=2008.0, target=2020.0)
    ledger = reconstruct_qc_trades(
        [
            order(
                1,
                "2024-01-02T10:05:00",
                10,
                0.0,
                t1,
                status=INVALID,
                message="Insufficient buying power",
            ),
            order(2, "2024-01-02T11:05:00", -10, 2005.0, t1_exit),
            order(3, "2024-01-02T12:05:00", 4, 2010.5, t2),
            order(4, "2024-01-02T13:00:00", -4, 2020.0, t2),
        ]
    )
    assert len(ledger.rejected_entries) == 1
    assert len(ledger.trades) == 2
    assert ledger.trades[0].entry_filled is False
    assert ledger.trades[0].exit_filled is True  # la sortie, elle, s'exécute
    assert ledger.trades[0].closed is False
    assert ledger.trades[0].pnl == 0.0
    assert ledger.tag_key_mismatches == []
    # La position suivante reste correctement appariée : pas de décalage.
    assert ledger.trades[1].closed is True
    assert ledger.trades[1].pnl == pytest.approx(4 * (2020.0 - 2010.5))


def test_margin_call_orders_are_set_aside_and_never_paired():
    t1 = tag_of()
    ledger = reconstruct_qc_trades(
        [
            order(1, "2024-01-02T10:05:00", 10, 2000.5, t1),
            order(2, "2024-01-02T10:30:00", -3, 1999.0, "Margin Call"),
            order(3, "2024-01-02T11:00:00", -10, 2010.0, t1),
        ]
    )
    assert len(ledger.margin_calls) == 1
    assert len(ledger.trades) == 1
    assert ledger.trades[0].closed is True


def test_tag_key_mismatch_is_reported_not_silently_accepted():
    entry = tag_of(level=2000.0, stop=1998.0)
    wrong = tag_of(level=2000.0, ts="2024-01-02 11:00:00", stop=1997.0)
    ledger = reconstruct_qc_trades(
        [
            order(1, "2024-01-02T10:05:00", 10, 2000.5, entry),
            order(2, "2024-01-02T11:05:00", -10, 2010.0, wrong),
        ]
    )
    assert len(ledger.tag_key_mismatches) == 1


def test_risk_engaged_is_stop_distance_times_quantity():
    ledger = reconstruct_qc_trades(
        [
            order(1, "2024-01-02T10:05:00", 25, 2000.5, tag_of(stop=1998.0)),
            order(2, "2024-01-02T11:00:00", -25, 2010.0, tag_of(ts="2024-01-02 11:00:00")),
        ]
    )
    assert ledger.trades[0].risk_engaged == pytest.approx(abs(2000.5 - 1998.0) * 25)


# ═══════════════════════════════════════════════════════════════════════
# Inventaire fantôme
# ═══════════════════════════════════════════════════════════════════════


def test_phantom_inventory_sees_the_position_a_rejected_exit_leaves_open():
    t1 = tag_of("BREAK_SHORT", stop=2002.0, target=1990.0)
    t2 = tag_of("BREAK_SHORT", level=1990.0, ts="2024-01-02 12:00:00", stop=1992.0, target=1980.0)
    orders = [
        order(1, "2024-01-02T10:05:00", -10, 2000.0, t1),
        order(
            2,
            "2024-01-02T11:00:00",
            10,
            0.0,
            t1,
            status=INVALID,
            otype=TYPE_MARKET_ON_OPEN,
            message="OandaBrokerageModel does not support MarketOnOpen order type",
        ),
        order(3, "2024-01-02T12:05:00", -5, 1990.0, t2),
        order(4, "2024-01-02T13:00:00", 5, 1980.0, t2),
    ]
    report = phantom_inventory(orders, reconstruct_qc_trades(orders))
    assert report["n_rejected_exits"] == 1
    assert report["n_rejected_exits_market_on_open"] == 1
    assert report["final_net_position_oz"] == pytest.approx(-10.0)
    assert report["n_fills_leaving_account_flat"] == 0
    assert report["timeline"][0]["position_left_open_oz"] == pytest.approx(-10.0)


def test_account_pnl_is_pure_cash_flow_and_marks_the_open_position():
    t1 = tag_of()
    orders = [
        order(1, "2024-01-02T10:05:00", 10, 2000.0, t1),
        order(2, "2024-01-02T11:00:00", -4, 2010.0, t1),
    ]
    out = account_pnl(orders, mark_price=2005.0)
    # Trésorerie : -10 x 2 000 + 4 x 2 010 = -11 960 ; 6 oz restantes à 2 005.
    assert out["final_position"] == pytest.approx(6.0)
    assert out["realised_and_unrealised_pnl"] == pytest.approx(-11_960.0 + 6 * 2005.0)


def test_account_pnl_ignores_unfilled_orders():
    t1 = tag_of()
    orders = [
        order(1, "2024-01-02T10:05:00", 10, 2000.0, t1),
        order(2, "2024-01-02T11:00:00", -10, 0.0, t1, status=INVALID, message="nope"),
    ]
    out = account_pnl(orders, mark_price=2000.0)
    assert out["final_position"] == pytest.approx(10.0)
    assert out["realised_and_unrealised_pnl"] == pytest.approx(0.0)


# ═══════════════════════════════════════════════════════════════════════
# Appariement
# ═══════════════════════════════════════════════════════════════════════


def _qc(ts: str, scenario: str = "BREAK_LONG", level: float = 2000.0):
    tag = tag_of(scenario=scenario, level=level, ts=ts)
    entry = pd.Timestamp(ts) + pd.Timedelta(minutes=BAR_MINUTES)
    return [
        order(1, entry.strftime("%Y-%m-%dT%H:%M:%S"), 10, 2000.5, tag),
        order(2, (entry + pd.Timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%S"), -10, 2010.0, tag),
    ]


@pytest.mark.parametrize("offset", [-BAR_MINUTES, 0, BAR_MINUTES])
def test_match_accepts_exactly_one_bar_of_drift(offset):
    py = py_frame([py_trade(ts="2024-01-02 10:00:00")])
    shifted = (pd.Timestamp("2024-01-02 10:00:00") + pd.Timedelta(minutes=offset)).strftime(
        "%Y-%m-%d %H:%M:%S"
    )
    qc = reconstruct_qc_trades(_qc(shifted)).trades
    pairs, py_orphans, qc_orphans = match_entries(py, qc, tol_bars=1)
    assert pairs == [(0, 0)]
    assert not py_orphans and not qc_orphans


@pytest.mark.parametrize("offset", [-2 * BAR_MINUTES, 2 * BAR_MINUTES])
def test_match_refuses_two_bars_of_drift(offset):
    py = py_frame([py_trade(ts="2024-01-02 10:00:00")])
    shifted = (pd.Timestamp("2024-01-02 10:00:00") + pd.Timedelta(minutes=offset)).strftime(
        "%Y-%m-%d %H:%M:%S"
    )
    qc = reconstruct_qc_trades(_qc(shifted)).trades
    pairs, py_orphans, qc_orphans = match_entries(py, qc, tol_bars=1)
    assert pairs == []
    assert py_orphans == [0] and qc_orphans == [0]


def test_match_requires_the_same_scenario_and_the_same_level():
    py = py_frame([py_trade(ts="2024-01-02 10:00:00", scenario="BREAK_LONG", level=2000.0)])
    other_scenario = reconstruct_qc_trades(
        _qc("2024-01-02 10:00:00", scenario="REV_LONG")
    ).trades
    other_level = reconstruct_qc_trades(_qc("2024-01-02 10:00:00", level=2010.0)).trades
    assert match_entries(py, other_scenario, 1)[0] == []
    assert match_entries(py, other_level, 1)[0] == []


def test_match_is_one_to_one_and_takes_the_nearest():
    """Deux candidats QC dans la fenêtre : le plus proche gagne, l'autre reste orphelin."""
    py = py_frame([py_trade(ts="2024-01-02 10:00:00")])
    tag_a = tag_of(ts="2024-01-02 10:05:00")
    tag_b = tag_of(ts="2024-01-02 10:00:00")
    orders = [
        order(1, "2024-01-02T10:10:00", 10, 2000.5, tag_a),
        order(2, "2024-01-02T11:00:00", -10, 2010.0, tag_a),
        order(3, "2024-01-02T11:05:00", 10, 2000.5, tag_b),
        order(4, "2024-01-02T12:00:00", -10, 2010.0, tag_b),
    ]
    qc = reconstruct_qc_trades(orders).trades
    pairs, py_orphans, qc_orphans = match_entries(py, qc, tol_bars=1)
    assert pairs == [(0, 1)]  # le tag exact, pas celui décalé de cinq minutes
    assert py_orphans == [] and qc_orphans == [0]


def test_match_reports_orphans_on_both_sides():
    py = py_frame(
        [
            py_trade(ts="2024-01-02 10:00:00"),
            py_trade(ts="2024-01-03 10:00:00", level=2010.0, stop=2008.0, target=2020.0),
        ]
    )
    qc = reconstruct_qc_trades(
        [*_qc("2024-01-02 10:00:00"), *_qc("2024-01-05 10:00:00")]
    ).trades
    # Les ids se répètent dans le concaténé ; on refabrique proprement.
    orders = []
    for k, ts in enumerate(("2024-01-02 10:00:00", "2024-01-05 10:00:00")):
        tag = tag_of(ts=ts)
        base = pd.Timestamp(ts) + pd.Timedelta(minutes=BAR_MINUTES)
        orders.append(order(2 * k + 1, base.strftime("%Y-%m-%dT%H:%M:%S"), 10, 2000.5, tag))
        orders.append(
            order(
                2 * k + 2,
                (base + pd.Timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%S"),
                -10,
                2010.0,
                tag,
            )
        )
    qc = reconstruct_qc_trades(orders).trades
    pairs, py_orphans, qc_orphans = match_entries(py, qc, tol_bars=1)
    assert len(pairs) == 1
    assert py_orphans == [1]
    assert qc_orphans == [1]


# ═══════════════════════════════════════════════════════════════════════
# Décomposition en R
# ═══════════════════════════════════════════════════════════════════════


def _one_trade(
    *,
    scenario="BREAK_LONG",
    lots=0.10,
    qc_qty=10.0,
    py_fill=2000.5,
    qc_fill=2000.9,
    py_exit=2010.0,
    qc_exit=2009.4,
    py_reason="TARGET",
    stop=1998.0,
    target=2010.0,
):
    row = py_trade(
        scenario=scenario,
        lots=lots,
        fill_px=py_fill,
        exit_px=py_exit,
        exit_reason=py_reason,
        stop=stop,
        target=target,
    )
    tag = tag_of(scenario=scenario, stop=stop, target=target)
    orders = [
        order(1, "2024-01-02T10:05:00", qc_qty, qc_fill, tag),
        order(2, "2024-01-02T11:00:00", -qc_qty, qc_exit, tag),
    ]
    return row, reconstruct_qc_trades(orders).trades[0]


def test_decomposition_sums_exactly_to_the_total_gap():
    row, trade = _one_trade()
    out = decompose(row, trade)
    posts = sum(out[p] for p in POSTS)
    assert out["gap_r"] == pytest.approx(posts, abs=1e-12)
    assert out["closure_usd"] == pytest.approx(0.0, abs=1e-9)
    assert out["pnl_py"] - out["pnl_qc"] == pytest.approx(
        out["gap_r"] * row["risk_amount"], abs=1e-9
    )


@pytest.mark.parametrize("scenario", ["BREAK_LONG", "BREAK_SHORT", "REV_LONG", "REV_SHORT"])
@pytest.mark.parametrize("py_reason", ["STOP", "TARGET", "TIME", "SESSION"])
def test_decomposition_closes_for_every_side_and_every_exit_reason(scenario, py_reason):
    side = 1 if scenario.endswith("LONG") else -1
    stop = 2000.0 - side * 2.0
    target = 2000.0 + side * 10.0
    row, trade = _one_trade(
        scenario=scenario,
        stop=stop,
        target=target,
        py_fill=2000.0 + side * 0.15,
        qc_fill=2000.0 + side * 0.55,
        py_exit=2000.0 + side * 6.0,
        qc_exit=2000.0 + side * 5.4,
        py_reason=py_reason,
        qc_qty=side * 10.0,
    )
    out = decompose(row, trade)
    assert out["closure_usd"] == pytest.approx(0.0, abs=1e-9)
    assert out["gap_r"] == pytest.approx(sum(out[p] for p in POSTS), abs=1e-12)


def test_identical_engines_give_a_zero_gap_and_zero_posts():
    row, trade = _one_trade(
        lots=0.10, qc_qty=10.0, py_fill=2000.5, qc_fill=2000.5, py_exit=2010.0, qc_exit=2010.0
    )
    out = decompose(row, trade)
    assert out["gap_r"] == pytest.approx(0.0, abs=1e-12)
    for post in POSTS:
        assert out[post] == pytest.approx(0.0, abs=1e-12)
    assert out["r_py"] == pytest.approx(out["r_qc"], abs=1e-12)


def test_only_the_size_post_moves_when_only_the_size_differs():
    row, trade = _one_trade(
        lots=0.10, qc_qty=6.0, py_fill=2000.5, qc_fill=2000.5, py_exit=2010.0, qc_exit=2010.0
    )
    out = decompose(row, trade)
    assert out["r_size"] != pytest.approx(0.0)
    for post in ("r_entry_slippage", "r_py_exit_gap", "r_exit_reason", "r_exit_slippage"):
        assert out[post] == pytest.approx(0.0, abs=1e-12)


def test_only_the_entry_post_moves_when_only_the_entry_price_differs():
    row, trade = _one_trade(
        lots=0.10, qc_qty=10.0, py_fill=2000.5, qc_fill=2000.9, py_exit=2010.0, qc_exit=2010.0
    )
    out = decompose(row, trade)
    assert out["r_entry_slippage"] == pytest.approx(10.0 * (2000.9 - 2000.5) / row["risk_amount"])
    for post in ("r_size", "r_py_exit_gap", "r_exit_reason", "r_exit_slippage"):
        assert out[post] == pytest.approx(0.0, abs=1e-12)


def test_exit_slippage_carries_a_fill_beyond_the_stop():
    """L'hypothèse à tester : LEAN sort au marché *après* la détection du stop."""
    row, trade = _one_trade(
        py_reason="STOP",
        py_exit=1998.0,  # la référence sort AU stop
        qc_exit=1997.2,  # LEAN sort 0,80 $ au-delà
        qc_fill=2000.5,
        py_fill=2000.5,
        qc_qty=10.0,
        lots=0.10,
    )
    out = decompose(row, trade)
    assert out["exit_reason_qc"] == "STOP"
    assert out["exit_slippage_px"] == pytest.approx(-0.8)
    assert out["r_exit_slippage"] == pytest.approx(10.0 * 0.8 / row["risk_amount"])
    assert out["r_py_exit_gap"] == pytest.approx(0.0, abs=1e-12)


def test_reason_post_carries_a_disagreement_on_the_exit_reason():
    """Python sort à la cible, QC au stop : l'écart est un écart de RAISON."""
    row, trade = _one_trade(
        py_reason="TARGET", py_exit=2010.0, qc_exit=1998.0, py_fill=2000.5, qc_fill=2000.5
    )
    out = decompose(row, trade)
    assert out["exit_reason_py"] == "TARGET"
    assert out["exit_reason_qc"] == "STOP"
    assert out["reason_agrees"] is False
    assert out["r_exit_reason"] == pytest.approx(10.0 * (2010.0 - 1998.0) / row["risk_amount"])
    assert out["r_exit_slippage"] == pytest.approx(0.0, abs=1e-12)


def test_a_time_exit_puts_no_slippage_where_none_is_measurable():
    """Sans niveau théorique dans le tag, le poste de glissement est nul par construction."""
    row, trade = _one_trade(py_reason="TIME", py_exit=2003.0, qc_exit=2002.4)
    out = decompose(row, trade)
    assert out["exit_reason_qc"] == "TIME_OR_SESSION"
    assert out["r_exit_slippage"] == pytest.approx(0.0, abs=1e-12)
    assert out["closure_usd"] == pytest.approx(0.0, abs=1e-9)


def test_aggregate_posts_sum_to_the_total_gap():
    rows = [
        decompose(*_one_trade(qc_exit=2009.4)),
        decompose(*_one_trade(scenario="BREAK_SHORT", stop=2002.0, target=1990.0, qc_qty=-10.0)),
        decompose(*_one_trade(py_reason="STOP", py_exit=1998.0, qc_exit=1997.1)),
    ]
    out = aggregate(rows)["overall"]
    assert out["n"] == 3
    assert out["posts_sum_total"] == pytest.approx(out["gap_r_sum"], abs=1e-12)
    assert out["unattributed_r_sum"] == pytest.approx(0.0, abs=1e-12)
    assert out["gap_r_sum"] == pytest.approx(
        out["r_py_sum"] - out["r_qc_on_py_risk_sum"], abs=1e-12
    )


# ═══════════════════════════════════════════════════════════════════════
# Raison de sortie inférée, spread implicite, dimensionnement
# ═══════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize(
    ("qc_exit", "expected"),
    [(1998.0, "STOP"), (1997.5, "STOP"), (2010.0, "TARGET"), (2009.6, "TARGET"), (2004.0, "TIME_OR_SESSION")],
)
def test_infer_qc_exit_reason(qc_exit, expected):
    _, trade = _one_trade(qc_exit=qc_exit)
    assert infer_qc_exit_reason(trade) == expected


def test_infer_returns_not_closed_when_the_exit_was_rejected():
    tag = tag_of()
    orders = [
        order(1, "2024-01-02T10:05:00", 10, 2000.5, tag),
        order(2, "2024-01-02T11:00:00", -10, 0.0, tag, status=INVALID, message="nope"),
    ]
    trade = reconstruct_qc_trades(orders).trades[0]
    assert infer_qc_exit_reason(trade) == "NOT_CLOSED"


@pytest.mark.parametrize("side", [1, -1])
@pytest.mark.parametrize("half_qc", [0.145, 0.30, 0.75])
def test_implied_half_spread_round_trips(side, half_qc):
    """Fabriquer deux `r_est` depuis un même close et retrouver le spread de QC.

    C'est la mesure qui chiffre l'écart 5 du portage ; si l'inversion de §8 était
    fausse, le nombre publié dans la note le serait aussi.
    """
    close = 2000.0
    scenario = "BREAK_LONG" if side > 0 else "BREAK_SHORT"
    stop_py = close - side * 2.0
    stop_qc = close - side * 2.3  # les ATR diffèrent : l'inversion doit l'encaisser
    target = close + side * 10.0

    def r_of(stop: float, half: float) -> float:
        return (abs(target - close) - half) / (abs(close - stop) + half)

    row = py_trade(
        scenario=scenario,
        stop=stop_py,
        target=target,
        r_est=r_of(stop_py, REFERENCE_HALF_SPREAD),
    )
    tag = QcTag(
        scenario=scenario,
        level=2000.0,
        ts_decision=pd.Timestamp("2024-01-02 10:00:00"),
        stop=stop_qc,
        target=target,
        r_est=r_of(stop_qc, half_qc),
    )
    assert implied_half_spread(row, tag) == pytest.approx(half_qc, abs=1e-9)


@pytest.mark.parametrize(
    ("equity", "entry", "stop", "expected"),
    [
        (10_000.0, 2000.0, 1998.0, 0.25),  # 50 / (2 * 100) = 0,25 pile
        (10_000.0, 2000.0, 1997.93, 0.24),  # 0,2410 → arrondi INFÉRIEUR
        (800.0, 2000.0, 1996.0, 0.01),  # 4 / 400 = 0,01
        (800.0, 2000.0, 1990.0, 0.0),  # sous le pas : refus de taille
        (10_000.0, 2000.0, 2000.0, 0.0),  # distance nulle
        (0.0, 2000.0, 1998.0, 0.0),  # équité nulle
    ],
)
def test_lots_for_risk_floors_to_the_lot_step(equity, entry, stop, expected):
    lots = _lots_for_risk(equity, entry, stop)
    assert lots == pytest.approx(expected, abs=1e-12)
    # §11 : l'arrondi est INFÉRIEUR et sans epsilon (`x10_engine.py:293-302`),
    # donc le résultat est toujours un multiple du pas, jamais au-dessus du brut.
    assert lots <= 0.005 * equity / (abs(entry - stop) * CONTRACT_SIZE) + 1e-12 if (
        equity > 0 and entry != stop
    ) else lots == 0.0
    assert round(lots / VOLUME_MIN) == pytest.approx(lots / VOLUME_MIN, abs=1e-6)


# ═══════════════════════════════════════════════════════════════════════
# Tags enrichis v4 — les champs k=v derrière le préfixe gelé
# ═══════════════════════════════════════════════════════════════════════


V4_ENTRY = (
    "REV_SHORT|2070.000|2024-01-01 23:20:00|2071.119540|2060.000000|3.899469"
    "|bm=28402520|c=2069.9700|atr=0.789080|sp=0.360000"
    "|v=-0.287254|a=-0.484612|vw=2069.511562|ema=2068.558303"
)
V4_EXIT = (
    "REV_SHORT|2070.000|2024-01-02 00:10:00|2071.119540|2060.000000|3.899469"
    "|xr=STOP|xp=2071.120"
)


def test_the_frozen_six_fields_still_parse_the_same_way():
    """Le contrat v1 doit survivre à l'enrichissement : mêmes six champs, même clé."""
    v1 = parse_tag("REV_SHORT|2070.000|2024-01-01 23:20:00|2071.119540|2060.000000|3.899469")
    v4 = parse_tag(V4_ENTRY)
    assert v4.scenario == v1.scenario and v4.level == v1.level
    assert v4.stop == v1.stop and v4.target == v1.target and v4.r_est == v1.r_est
    assert v4.key == v1.key
    assert v1.extra == {}


def test_kv_fields_are_read_as_numbers():
    tag = parse_tag(V4_ENTRY)
    assert tag.number("bm") == 28402520
    assert tag.number("c") == pytest.approx(2069.97)
    assert tag.number("atr") == pytest.approx(0.789080)
    assert tag.number("sp") == pytest.approx(0.36)
    assert tag.number("absent") != tag.number("absent")  # NaN


def test_exit_reason_is_read_not_inferred_when_the_tag_carries_it():
    orders = [
        order(1, "2024-01-02T04:26:00", -38, 2069.79, V4_ENTRY),
        order(2, "2024-01-02T05:13:00", 38, 2071.31, V4_EXIT),
    ]
    trade = reconstruct_qc_trades(orders).trades[0]
    assert trade.exit_tag.exit_reason == "STOP"
    assert trade.exit_tag.exit_px == pytest.approx(2071.120)
    assert infer_qc_exit_reason(trade) == "STOP"


def test_a_read_time_exit_gets_a_measurable_slippage():
    """`xp=` chiffre le glissement des sorties TIME/SESSION, que v1 ne voyait pas."""
    entry = tag_of() + "|bm=1000|c=2000.5000|atr=1.000000|sp=0.290000"
    exit_tag = tag_of(ts="2024-01-02 11:00:00") + "|xr=TIME|xp=2003.000"
    orders = [
        order(1, "2024-01-02T10:05:00", 10, 2000.5, entry),
        order(2, "2024-01-02T11:00:00", -10, 2002.4, exit_tag),
    ]
    trade = reconstruct_qc_trades(orders).trades[0]
    row = decompose(py_trade(exit_reason="TIME", exit_px=2003.0), trade)
    assert row["exit_reason_qc"] == "TIME"
    assert row["exit_reason_qc_is_read"] is True
    # 2003,000 visé, 2002,400 rempli : 0,60 $ de glissement, plus un minorant.
    assert row["exit_slippage_px"] == pytest.approx(-0.6)
    assert row["r_exit_slippage"] != pytest.approx(0.0)
    assert row["closure_usd"] == pytest.approx(0.0, abs=1e-9)


def test_summary_keys_are_frozen():
    """Le rapport client lit ces noms : on peut en ajouter, jamais en renommer."""
    assert SUMMARY_KEYS == (
        "qc_backtest_id",
        "qc_trades",
        "py_trades",
        "match_rate_py_to_qc",
        "match_rate_qc_to_py",
        "py_expectancy_r",
        "qc_expectancy_r",
        "exit_slippage_r_per_trade",
        "qc_stop_realised_r",
        "qc_half_spread_usd",
        "unattributed_share",
        "qc_net_return_pct",
        "healthy",
    )


# ═══════════════════════════════════════════════════════════════════════
# Branche MT5 — trace de l'EA, deals, appariement, R réalisé
# ═══════════════════════════════════════════════════════════════════════
#
# Mêmes principes qu'au-dessus, sur des artefacts MT5 synthétiques : une trace
# §13 à 22 colonnes et un CSV de deals UTF-16. Aucun fichier du tester n'est lu.

TRACE_HEADER = (
    "ts_decision,event,scenario,level,d,atr,v,m,a,ctx_ema,ctx_vwap,ctx_dxy,"
    "vwap,ema50_h1,dxy,stop,target,r_est,fill_px,exit_px,exit_reason,cancel_reason"
)


def trace_row(
    ts: str,
    event: str,
    *,
    scenario: str = "",
    level: float = 2000.0,
    d: int = 1,
    atr: float = 1.0,
    vwap: float = 2000.0,
    ema50_h1: float = 1999.0,
    dxy: float = 100.0,
    stop: str = "",
    target: str = "",
    r_est: str = "",
    fill_px: str = "",
    exit_px: str = "",
    exit_reason: str = "",
    cancel_reason: str = "",
) -> str:
    return (
        f"{ts},{event},{scenario},{level:.3f},{d},{atr:.6f},0.1,0.2,0.3,1,1,1,"
        f"{vwap:.6f},{ema50_h1:.6f},{dxy:.6f},{stop},{target},{r_est},"
        f"{fill_px},{exit_px},{exit_reason},{cancel_reason}"
    )


def write_trace(tmp_path: Path, rows: list[str]) -> Path:
    path = tmp_path / "trace.csv"
    path.write_text("\n".join([TRACE_HEADER, *rows]) + "\n")
    return path


def write_deals(tmp_path: Path, rows: list[str]) -> Path:
    """Le CSV des deals tel que l'EA l'écrit : UTF-16 LE avec BOM."""
    header = (
        "deal_id,position_id,time_utc,symbol,magic,sleeve,type,entry,volume,"
        "price,profit,commission,swap"
    )
    path = tmp_path / "deals.csv"
    path.write_bytes(("\n".join([header, *rows]) + "\n").encode("utf-16"))
    return path


def one_position_artifacts(
    tmp_path: Path,
    *,
    scenario: str = "BREAK_LONG",
    magic: int = 841,
    fill_px: float = 2000.0,
    stop: float = 1995.0,
    volume: float = 0.10,
    profit: float = 50.0,
) -> tuple[Path, Path]:
    trace = write_trace(
        tmp_path,
        [
            trace_row("2024-01-02 10:00:00", "ARM"),
            trace_row(
                "2024-01-02 10:05:00",
                "ENTRY",
                scenario=scenario,
                stop=f"{stop:.6f}",
                target="2010.000000",
                r_est="2.000000",
                fill_px=f"{fill_px:.3f}",
            ),
            trace_row(
                "2024-01-02 11:00:00",
                "EXIT",
                scenario=scenario,
                exit_px="2010.000",
                exit_reason="TARGET",
            ),
        ],
    )
    deals = write_deals(
        tmp_path,
        [
            "1,0,2024.01.02 00:00:00,,0,OTHER,2,0,0.0000,0.00000,10000.00,0.0000,0.0000",
            f"2,2,2024.01.02 10:10:00,XAUUSD.c,{magic},{scenario},0,0,{volume:.4f},"
            f"{fill_px:.5f},0.00,0.0000,0.0000",
            f"3,2,2024.01.02 11:05:00,XAUUSD.c,{magic},{scenario},1,1,{volume:.4f},"
            f"2010.00000,{profit:.2f},0.0000,0.0000",
        ],
    )
    return trace, deals


def test_the_mt5_trace_is_read_with_its_twenty_second_column(tmp_path):
    path = write_trace(
        tmp_path,
        [
            trace_row("2024-01-02 10:00:00", "ARM"),
            trace_row("2024-01-02 10:05:00", "CANCEL", cancel_reason="zone"),
        ],
    )
    frame = load_mt5_trace(path)
    assert list(frame["event"]) == ["ARM", "CANCEL"]
    assert list(frame["cancel_reason"]) == ["", "zone"]
    # L'horodatage est UTC, jamais naïf : c'est ce qui le rend comparable.
    assert frame["ts_decision"].dt.tz is not None


def test_the_deals_csv_is_read_through_its_utf16_encoding(tmp_path):
    _, deals_path = one_position_artifacts(tmp_path)
    deals = load_mt5_deals(deals_path)
    assert len(deals) == 3
    # `net` réunit résultat, commission et swap : c'est lui qui boucle sur le
    # profit net du rapport HTML, pas la colonne `profit` seule.
    assert deals["net"].sum() == pytest.approx(10_050.0)


def test_realised_r_divides_the_net_by_the_risk_engaged_at_the_fill(tmp_path):
    # |2000 − 1995| × 0,10 lot × 100 onces = 50 $ de risque ; 50 $ gagnés = 1 R.
    trace_path, deals_path = one_position_artifacts(tmp_path)
    positions = mt5_positions(load_mt5_trace(trace_path), load_mt5_deals(deals_path))
    assert len(positions) == 1
    assert positions.loc[0, "risk_amount"] == pytest.approx(50.0)
    assert positions.loc[0, "r_realised"] == pytest.approx(1.0)
    assert positions.loc[0, "exit_reason_name"] == "TARGET"


def test_realised_r_is_not_r_est_when_the_lot_step_rounds_the_size(tmp_path):
    """0,07 lot au lieu de 0,10 : le R réalisé suit le risque VRAIMENT porté."""
    trace_path, deals_path = one_position_artifacts(tmp_path, volume=0.07, profit=50.0)
    positions = mt5_positions(load_mt5_trace(trace_path), load_mt5_deals(deals_path))
    assert positions.loc[0, "risk_amount"] == pytest.approx(35.0)
    assert positions.loc[0, "r_realised"] == pytest.approx(50.0 / 35.0)
    assert positions.loc[0, "r_est"] == pytest.approx(2.0)


def test_positions_refuse_to_be_built_when_trace_and_deals_disagree(tmp_path):
    trace_path, deals_path = one_position_artifacts(tmp_path)
    trace = load_mt5_trace(trace_path)
    extra = trace.iloc[[1]].copy()
    extra["ts_decision"] = pd.Timestamp("2024-01-03 10:05:00", tz="UTC")
    with pytest.raises(SystemExit, match="même nombre de positions"):
        mt5_positions(
            pd.concat([trace, extra], ignore_index=True), load_mt5_deals(deals_path)
        )


@pytest.mark.parametrize("offset", [0, -BAR_MINUTES, BAR_MINUTES])
def test_mt5_matching_accepts_exactly_one_bar_of_drift(offset):
    left = pd.DataFrame(
        {
            "ts_decision": [pd.Timestamp("2024-01-02 10:00:00", tz="UTC")],
            "scenario_name": ["BREAK_LONG"],
            "level": [2000.0],
        }
    )
    right = left.copy()
    right["ts_decision"] = right["ts_decision"] + pd.Timedelta(minutes=offset)
    pairs, _, _ = match_on_keys(left, right, ("scenario_name", "level"))
    assert pairs == [(0, 0)]


def test_mt5_matching_refuses_two_bars_of_drift():
    left = pd.DataFrame(
        {
            "ts_decision": [pd.Timestamp("2024-01-02 10:00:00", tz="UTC")],
            "scenario_name": ["BREAK_LONG"],
            "level": [2000.0],
        }
    )
    right = left.copy()
    right["ts_decision"] = right["ts_decision"] + pd.Timedelta(minutes=2 * BAR_MINUTES)
    pairs, left_orphans, right_orphans = match_on_keys(
        left, right, ("scenario_name", "level")
    )
    assert pairs == []
    assert left_orphans == [0] and right_orphans == [0]


def test_mt5_matching_is_one_to_one_and_takes_the_nearest():
    base = pd.Timestamp("2024-01-02 10:00:00", tz="UTC")
    left = pd.DataFrame(
        {
            "ts_decision": [base, base + pd.Timedelta(minutes=BAR_MINUTES)],
            "scenario_name": ["REV_LONG", "REV_LONG"],
            "level": [2000.0, 2000.0],
        }
    )
    right = left.copy()
    pairs, left_orphans, right_orphans = match_on_keys(
        left, right, ("scenario_name", "level")
    )
    assert pairs == [(0, 0), (1, 1)]
    assert not left_orphans and not right_orphans


def test_mt5_matching_requires_the_same_scenario():
    left = pd.DataFrame(
        {
            "ts_decision": [pd.Timestamp("2024-01-02 10:00:00", tz="UTC")],
            "scenario_name": ["BREAK_LONG"],
            "level": [2000.0],
        }
    )
    right = left.assign(scenario_name=["REV_LONG"])
    pairs, _, _ = match_on_keys(left, right, ("scenario_name", "level"))
    assert pairs == []


def test_expectancy_table_counts_trades_expectancy_and_profit_factor():
    frame = pd.DataFrame(
        {
            "scenario_name": ["BREAK_LONG"] * 3 + ["REV_SHORT"],
            "r_realised": [2.0, -1.0, -1.0, 1.0],
        }
    )
    table = expectancy_table(frame, "scenario_name")
    assert table["BREAK_LONG"]["trades"] == 3
    assert table["BREAK_LONG"]["expectancy_r"] == pytest.approx(0.0)
    assert table["BREAK_LONG"]["profit_factor"] == pytest.approx(1.0)
    # Aucune perte : le profit factor n'est pas `inf`, il n'est pas défini.
    assert table["REV_SHORT"]["profit_factor"] is None


def test_busy_share_counts_orphans_emitted_while_the_other_engine_held(tmp_path):
    other = pd.DataFrame(
        {
            "ts_fill": pd.to_datetime(["2024-01-02 10:00:00"], utc=True),
            "ts_exit": pd.to_datetime(["2024-01-02 12:00:00"], utc=True),
        }
    )
    share = busy_share(["2024-01-02 11:00:00+00:00", "2024-01-02 13:00:00+00:00"], other)
    assert share["n_orphans"] == 2
    assert share["n_counterpart_busy"] == 1
    assert share["share"] == pytest.approx(0.5)


def test_order_failures_separate_refused_entries_from_exit_retries(tmp_path):
    """`order_failures` compte des ORDRES, les `[EXIT][WARN]` des réessais."""
    lines = [
        "CS\t0\t16:52\tXauX10\t2023.04.16 22:55:00   [ENTRY][WARN] PositionOpen "
        "REV_LONG lots=0.13 failed retcode=10018",
        "CS\t0\t16:52\tXauX10\t2023.06.05 22:01:00   [EXIT][WARN] PositionClose "
        "268 failed retcode=10018",
        "CS\t0\t16:52\tXauX10\t2023.06.05 22:05:00   [EXIT][WARN] PositionClose "
        "268 failed retcode=10018",
    ]
    path = tmp_path / "agent.log"
    path.write_bytes(("\n".join(lines) + "\n").encode("utf-16"))
    failures = read_order_failures(path)
    assert failures["n_entry_failures"] == 1
    assert failures["n_exit_retry_lines"] == 2
    assert failures["n_positions_with_exit_retry"] == 1
    assert failures["retcodes"] == {"10018": 3}
    assert failures["entry_failures"][0]["time_utc"] == "2023-04-16 22:55:00"


def test_order_failures_say_so_when_the_log_is_missing(tmp_path):
    failures = read_order_failures(tmp_path / "absent.log")
    assert failures["measured"] is False


def test_the_clock_check_finds_no_hourly_offset_on_an_aligned_feed():
    index = pd.date_range("2024-01-02", periods=3000, freq="1min", tz="UTC")
    rng = np.random.default_rng(0)
    close = 2000.0 + np.cumsum(rng.normal(0, 0.05, len(index)))
    dump = pd.DataFrame({"close": close, "spread_points": 250}, index=index)
    reference = pd.DataFrame({"close": close}, index=index)
    path = Path(tempfile.mkdtemp()) / "ref.parquet"
    reference.to_parquet(path)

    check = clock_offset_check(dump, path)
    assert check["server_clock_offset_min"] == 0
    assert check["server_clock_is_utc"] is True
    assert check["best_lag_min"] == 0


def test_the_clock_check_sees_an_hour_of_offset_when_there_is_one():
    index = pd.date_range("2024-01-02", periods=3000, freq="1min", tz="UTC")
    rng = np.random.default_rng(1)
    close = 2000.0 + np.cumsum(rng.normal(0, 0.05, len(index)))
    dump = pd.DataFrame({"close": close, "spread_points": 250}, index=index)
    # Le même signal, publié une heure plus tôt sur l'horloge de référence.
    reference = pd.DataFrame({"close": close}, index=index - pd.Timedelta(hours=1))
    path = Path(tempfile.mkdtemp()) / "ref_shifted.parquet"
    reference.to_parquet(path)

    check = clock_offset_check(dump, path)
    assert check["server_clock_offset_min"] == -60
    assert check["server_clock_is_utc"] is False


def test_mt5_summary_keys_are_frozen():
    """La note de recherche lit ces noms : on peut en ajouter, jamais renommer."""
    assert MT5_SUMMARY_KEYS == (
        "match_rate_py_to_mt5_same_data",
        "match_rate_mt5_to_py_same_data",
        "match_rate_py_to_mt5_diff_data",
        "py_expectancy_r_same_data",
        "mt5_expectancy_r",
        "unattributed_share",
        "server_clock_offset_min",
    )
