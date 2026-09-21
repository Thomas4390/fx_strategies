#!/usr/bin/env python3
"""Réconciliation événementielle de la stratégie « XAUUSD niveaux x10 ».

Barreaux 3 à 5 de l'échelle de `docs/specs/xau_x10_spec.md` §13, entre la
**référence** Python (vectorbtpro + noyau Numba, `src/strategies/xau_x10.py`) et
le portage **QuantConnect** (`src/qc/xau_x10/`), sur le journal d'ordres du
backtest — seul canal exportable par l'API, §13 et `src/qc/xau_x10/README.md`.

La méthode est celle de `scripts/reconcile_three_way.py` : on descend l'échelle
jusqu'au premier barreau qui casse, et un écart doit atterrir sur une
**quantité nommée**. Comparer deux rendements annuels est le test le plus faible
disponible ; il ne dit jamais où est le problème.

    barreau 3/4  appariement des ENTRÉES        seuils, fenêtre, position unique
    barreau 4    stop / cible / r_est théoriques règle R, ATR, niveaux
    barreau 5    EXIT et PnL, en unités de R    fills LEAN, lots, coûts

Ce que le script reconstruit, et pourquoi il faut le reconstruire : LEAN ne
publie pas de « trades », seulement des ordres. Le portage envoie exactement
deux ordres market par position — l'ENTRÉE puis la SORTIE — et les tague tous
les deux `scenario|level|ts_decision|stop|target|r_est` (`src/qc/xau_x10/main.py`
`_tag`). L'appariement suit donc l'**intention de l'algorithme**, pas le signe
des quantités : un ordre qui arrive alors qu'aucune position logique n'est
ouverte est une entrée, le suivant est sa sortie. C'est la seule lecture
correcte lorsqu'une sortie a été **rejetée** par le broker, cas que ce script
existe en partie pour mesurer.

La décomposition du barreau 5 est un télescopage exact : partant du PnL Python
on substitue une à une la taille, le prix d'entrée, le niveau de sortie visé et
le prix de sortie réalisé, chaque substitution étant un poste nommé. La somme
des postes vaut l'écart total **à la précision machine**, ce que le JSON prouve
champ par champ (`decomposition_closure_max_abs`).

Usage
-----
    uv run python scripts/reconcile_x10_events.py \
        --py-trades reports/qc_x10/py_trades_2024_P2.csv \
        --qc-orders reports/qc_x10/x10_calage_2024_centre_orders.json \
        --qc-stats  reports/qc_x10/x10_calage_2024_centre.json \
        --out results/xau_x10/reconciliation_qc_2024.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "scripts"))

from reconcile_three_way import (  # noqa: E402
    TOL_BROKER_FEED,
    TOL_SAME_DATA,
    Tolerance,
)

# ── constantes du contrat ─────────────────────────────────────────────

#: §11 — taille de contrat XAUUSD, `data/broker/symbols_catalog_2026-07-28.csv`.
CONTRACT_SIZE = 100.0

#: §11 — fraction de l'équité risquée par trade, gelée.
RISK_FRAC = 0.005

#: §11 — pas de lot du broker ; un résultat sous ce seuil annule le trade.
VOLUME_MIN = 0.01

#: §2 — la grille de décision est en M5.
BAR_MINUTES = 5

#: §10 — durée maximale d'une position, en barres M5.
MAX_HOLD_BARS = 48

#: §13 — appariement des entrées attendu entre Python et QC (mêmes données).
MATCH_TARGET_SAME_DATA = 0.98

#: §13 — au-delà, la divergence non attribuée bloque la lecture hors échantillon.
UNATTRIBUTED_BLOCKER = 0.05

#: `src/qc/xau_x10/README.md` écart 1 — `set_warm_up(timedelta(days=12))`.
QC_WARMUP_DAYS = 12

#: LEAN `OrderStatus.Filled`; tout le reste est un ordre qui n'a pas existé.
LEAN_STATUS_FILLED = 3

#: LEAN `OrderType.MarketOnOpen` — celui que le modèle OANDA refuse.
LEAN_TYPE_MARKET_ON_OPEN = 4

SCENARIOS: tuple[str, ...] = ("BREAK_LONG", "BREAK_SHORT", "REV_LONG", "REV_SHORT")

#: Sens du trade par scénario (§7) : `q = +1` long, `q = -1` court.
SCENARIO_SIDE: dict[str, int] = {
    "BREAK_LONG": +1,
    "BREAK_SHORT": -1,
    "REV_LONG": +1,
    "REV_SHORT": -1,
}

#: §13 — liste fermée des raisons de sortie.
EXIT_REASONS: tuple[str, ...] = ("STOP", "TARGET", "TIME", "SESSION")


# ═══════════════════════════════════════════════════════════════════════
# 1. LE TAG QC — seul canal de réconciliation exportable par l'API
# ═══════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class QcTag:
    """`scenario|level|ts_decision|stop|target|r_est`, tel que `_tag` l'écrit."""

    scenario: str
    level: float
    ts_decision: pd.Timestamp
    stop: float
    target: float
    r_est: float

    @property
    def key(self) -> tuple[str, float, float, float, float]:
        """Ce qui identifie une *position* : la sortie porte le même quintuplet.

        Seul `ts_decision` diffère entre l'entrée et la sortie d'un même trade
        (§13, règle d'estampillage 1 : la sortie porte la barre M5 où elle est
        résolue). Le reste du tag est recopié tel quel par le portage, ce qui en
        fait un contrôle d'intégrité gratuit de l'appariement.
        """
        return (self.scenario, self.level, self.stop, self.target, self.r_est)


def parse_tag(tag: str) -> QcTag | None:
    """Décoder un tag d'ordre ; `None` pour tout ce qui n'est pas de la stratégie.

    LEAN pose ses propres tags — « Margin Call » en tête — sur les ordres qu'il
    émet lui-même. Les confondre avec des ordres de la stratégie fausserait
    l'appariement, donc on les rejette explicitement plutôt que de les laisser
    échouer au parsing.
    """
    parts = tag.split("|")
    if len(parts) != 6 or parts[0] not in SCENARIOS:
        return None
    try:
        return QcTag(
            scenario=parts[0],
            level=float(parts[1]),
            ts_decision=pd.Timestamp(parts[2]),
            stop=float(parts[3]),
            target=float(parts[4]),
            r_est=float(parts[5]),
        )
    except ValueError:
        return None


# ═══════════════════════════════════════════════════════════════════════
# 2. BARREAU 3/4 — reconstruire les trades QC depuis le journal d'ordres
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class QcTrade:
    """Une position QC : l'ordre d'entrée, l'ordre de sortie, et leurs sorts."""

    entry_id: int
    entry_time: pd.Timestamp
    entry_filled: bool
    entry_qty: float
    entry_px: float
    tag: QcTag
    exit_id: int | None = None
    exit_time: pd.Timestamp | None = None
    exit_filled: bool = False
    exit_qty: float = 0.0
    exit_px: float = float("nan")
    exit_tag: QcTag | None = None
    exit_reject_reason: str | None = None
    exit_order_type: int | None = None

    @property
    def closed(self) -> bool:
        """Vrai seulement si l'entrée **et** la sortie ont été remplies."""
        return self.entry_filled and self.exit_filled

    @property
    def side(self) -> int:
        return SCENARIO_SIDE[self.tag.scenario]

    @property
    def pnl(self) -> float:
        """PnL du couple d'ordres, en USD, aux prix réellement remplis."""
        if not self.closed:
            return 0.0
        return self.entry_qty * (self.exit_px - self.entry_px)

    @property
    def risk_engaged(self) -> float:
        """§11 — `|e_fill - stop| * quantité`, le risque réellement engagé."""
        if not self.entry_filled:
            return float("nan")
        return abs(self.entry_px - self.tag.stop) * abs(self.entry_qty)


@dataclass
class QcLedger:
    """Ce que la reconstruction a trouvé, y compris ce qui n'aurait pas dû y être."""

    trades: list[QcTrade] = field(default_factory=list)
    margin_calls: list[dict[str, Any]] = field(default_factory=list)
    rejected_entries: list[dict[str, Any]] = field(default_factory=list)
    rejected_exits: list[dict[str, Any]] = field(default_factory=list)
    tag_key_mismatches: list[dict[str, Any]] = field(default_factory=list)
    unparsed_tags: list[dict[str, Any]] = field(default_factory=list)


def _order_reject_reason(order: dict[str, Any]) -> str:
    events = order.get("events") or []
    for event in events:
        message = event.get("message")
        if message:
            return str(message)
    return f"status={order.get('status')}"


def reconstruct_qc_trades(orders: list[dict[str, Any]]) -> QcLedger:
    """Apparier les ordres QC deux à deux, dans l'ordre où l'algorithme les a émis.

    L'automate du portage n'a qu'une position à la fois (§9) et `_route`
    n'émet qu'un ordre par événement : la parité entrée/sortie est donc une
    propriété du code, pas une hypothèse. On la suit littéralement — le premier
    ordre stratégique est une entrée, le suivant sa sortie — et on **n'utilise
    pas le signe de la quantité**, qui ment dès qu'une sortie a été refusée par
    le broker et que la position réelle survit à la position logique.
    """
    ledger = QcLedger()
    open_trade: QcTrade | None = None

    for order in sorted(orders, key=lambda o: int(o["id"])):
        tag_text = str(order.get("tag") or "")
        tag = parse_tag(tag_text)
        filled = int(order.get("status", -1)) == LEAN_STATUS_FILLED
        record = {
            "order_id": int(order["id"]),
            "time": str(order["time"]),
            "quantity": float(order["quantity"]),
            "price": float(order["price"]),
            "type": int(order["type"]),
            "tag": tag_text,
        }

        if tag is None:
            if tag_text.startswith("Margin Call"):
                ledger.margin_calls.append(record)
            else:
                ledger.unparsed_tags.append(record)
            continue

        if open_trade is None:
            trade = QcTrade(
                entry_id=int(order["id"]),
                entry_time=pd.Timestamp(order["time"]).tz_localize(None),
                entry_filled=filled,
                entry_qty=float(order["quantity"]),
                entry_px=float(order["price"]),
                tag=tag,
            )
            if not filled:
                # Une entrée refusée ouvre quand même une position **logique** :
                # `main.py:205-206` pose `self._open_qty` **avant** d'envoyer
                # l'ordre et ne regarde jamais son sort. L'EXIT qui suivra sera
                # donc bel et bien émis, et remplira une position que personne
                # n'a ouverte. Sauter ce trade ici décalerait tout l'appariement
                # d'un ordre à partir de là — ce que `tag_key_mismatches`
                # signale, et ce que ce commentaire existe pour éviter.
                ledger.rejected_entries.append(
                    {**record, "reason": _order_reject_reason(order)}
                )
            open_trade = trade
            ledger.trades.append(trade)
            continue

        # Second ordre d'une position : c'est la sortie, remplie ou non.
        open_trade.exit_id = int(order["id"])
        open_trade.exit_time = pd.Timestamp(order["time"]).tz_localize(None)
        open_trade.exit_filled = filled
        open_trade.exit_qty = float(order["quantity"])
        open_trade.exit_px = float(order["price"]) if filled else float("nan")
        open_trade.exit_tag = tag
        open_trade.exit_order_type = int(order["type"])
        if tag.key != open_trade.tag.key:
            ledger.tag_key_mismatches.append(
                {
                    "entry_order_id": open_trade.entry_id,
                    "exit_order_id": open_trade.exit_id,
                    "entry_tag": open_trade.tag.key,
                    "exit_tag": tag.key,
                }
            )
        if not filled:
            open_trade.exit_reject_reason = _order_reject_reason(order)
            ledger.rejected_exits.append(
                {**record, "reason": open_trade.exit_reject_reason}
            )
        open_trade = None

    return ledger


# ═══════════════════════════════════════════════════════════════════════
# 3. COMPTABILITÉ DE COMPTE — ce que le journal d'ordres dit vraiment
# ═══════════════════════════════════════════════════════════════════════


def account_pnl(orders: list[dict[str, Any]], mark_price: float | None = None) -> dict[str, Any]:
    """PnL du compte par flux de trésorerie, sur **tous** les ordres remplis.

    Aucune notion de trade ici, et c'est le point : la somme
    `-Σ q_i p_i + position_finale * prix_final` est le résultat du compte quelles
    que soient les erreurs d'appariement. Comparée à la somme des PnL des trades
    reconstruits, elle isole exactement ce qui a été perdu **hors** des
    allers-retours voulus par la stratégie.
    """
    cash = 0.0
    position = 0.0
    last_price = float("nan")
    equity_curve: list[dict[str, Any]] = []
    for order in sorted(orders, key=lambda o: int(o["id"])):
        if int(order.get("status", -1)) != LEAN_STATUS_FILLED:
            continue
        qty = float(order["quantity"])
        px = float(order["price"])
        cash -= qty * px
        position += qty
        last_price = px
        equity_curve.append(
            {
                "order_id": int(order["id"]),
                "time": str(order["time"]),
                "position": position,
                "price": px,
                "pnl": cash + position * px,
            }
        )
    mark = mark_price if mark_price is not None else last_price
    return {
        "final_position": position,
        "mark_price": mark,
        "realised_and_unrealised_pnl": cash + position * mark,
        "equity_curve": equity_curve,
    }


# ═══════════════════════════════════════════════════════════════════════
# 4. BARREAU 4 — appariement des entrées Python ↔ QC
# ═══════════════════════════════════════════════════════════════════════


def load_py_trades(path: Path) -> pd.DataFrame:
    """Le registre de trades de la référence, horodaté en UTC.

    Le moteur de référence tourne sur l'horloge **New York naïve** de §2 ;
    `ts_decision_utc` est la même décision lue sur l'horloge des tags QC. On
    exige la colonne plutôt que de la recalculer : une conversion de fuseau faite
    deux fois à deux endroits est une source de décalage d'une heure qui ne se
    voit qu'en hiver.
    """
    if not path.exists():
        raise SystemExit(f"[py] fichier introuvable : {path}")
    frame = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    required = (
        "ts_decision_utc",
        "ts_fill_utc",
        "ts_exit_utc",
        "scenario_name",
        "level",
        "stop",
        "target",
        "r_est",
        "lots",
        "q",
        "fill_px",
        "exit_px",
        "exit_reason_name",
        "risk_amount",
        "equity_before",
    )
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise SystemExit(f"[py] colonnes manquantes {missing} dans {path}")
    for column in ("ts_decision_utc", "ts_fill_utc", "ts_exit_utc"):
        frame[column] = pd.to_datetime(frame[column])
    return frame.sort_values("ts_decision_utc").reset_index(drop=True)


def match_entries(
    py: pd.DataFrame,
    qc_trades: list[QcTrade],
    tol_bars: int = 1,
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Apparier sur `(scénario, niveau, ts_decision ± tol_bars barres M5)`.

    Un à un et au plus proche : à l'intérieur d'un groupe `(scénario, niveau)`
    les candidats sont triés par écart temporel croissant, et un trade déjà
    apparié ne l'est jamais deux fois. La tolérance existe parce que les deux
    moteurs peuvent résoudre la même décision sur deux barres adjacentes quand
    une barre M5 est vide d'un côté (§2) ; au-delà d'une barre il ne s'agit plus
    de la même décision et l'appariement serait une invention.
    """
    window = pd.Timedelta(minutes=tol_bars * BAR_MINUTES)

    groups: dict[tuple[str, float], list[int]] = {}
    for j, trade in enumerate(qc_trades):
        groups.setdefault((trade.tag.scenario, round(trade.tag.level, 3)), []).append(j)

    candidates: list[tuple[pd.Timedelta, int, int]] = []
    for i, row in py.iterrows():
        key = (str(row["scenario_name"]), round(float(row["level"]), 3))
        for j in groups.get(key, ()):
            delta = abs(qc_trades[j].tag.ts_decision - row["ts_decision_utc"])
            if delta <= window:
                candidates.append((delta, int(i), j))

    candidates.sort(key=lambda c: (c[0], c[1], c[2]))
    used_py: set[int] = set()
    used_qc: set[int] = set()
    pairs: list[tuple[int, int]] = []
    for _, i, j in candidates:
        if i in used_py or j in used_qc:
            continue
        used_py.add(i)
        used_qc.add(j)
        pairs.append((i, j))

    pairs.sort(key=lambda p: p[0])
    py_orphans = [int(i) for i in py.index if i not in used_py]
    qc_orphans = [j for j in range(len(qc_trades)) if j not in used_qc]
    return pairs, py_orphans, qc_orphans


def _lots_for_risk(equity: float, entry: float, stop: float) -> float:
    """§11 — `arrondi_inferieur_0,01( f * equite / ( |e_fill - stop| * 100 ) )`."""
    distance = abs(entry - stop)
    if distance <= 0.0 or equity <= 0.0:
        return 0.0
    raw = RISK_FRAC * equity / (distance * CONTRACT_SIZE)
    return math.floor(raw / VOLUME_MIN) * VOLUME_MIN


#: Fenêtre, en barres M5, au-delà de la tolérance d'appariement, dans laquelle
#: on cherche encore une décision « voisine » avant de conclure au désaccord.
NEIGHBOUR_BARS = 3

#: Causes candidates, dans l'ordre de priorité où elles sont testées. Une cause
#: structurelle connue explique le cas à elle seule et rend les suivantes non
#: informatives, d'où l'ordre ; seule `rung2_3_automaton_divergence` est un
#: constat de désaccord et non une explication d'exécution.
ORPHAN_CAUSES: tuple[str, ...] = (
    "sample_edge_warmup",
    "counterpart_position_open",
    "size_below_min",
    "adjacent_level",
    "near_miss_time",
    "other_candidate_same_window",
    "rung2_3_automaton_divergence",
)


def _classify(
    ts: pd.Timestamp,
    scenario: str,
    level: float,
    entry_px: float,
    stop: float,
    sample_start: pd.Timestamp,
    other_decisions: pd.DataFrame,
    other_holdings: list[tuple[pd.Timestamp, pd.Timestamp | None]],
    tol_bars: int,
    equity_for_size: float | None,
) -> str:
    """Cause candidate d'une entrée sans contrepartie, dans l'autre moteur.

    `other_decisions` porte les colonnes `ts`, `scenario`, `level` du moteur
    d'en face ; `other_holdings` ses intervalles de détention, qui interdisent
    une seconde position (§9). Le classement est volontairement conservateur :
    tout ce qui n'a pas de voisin nommable retombe sur le désaccord d'automate,
    qui n'est pas une excuse mais un renvoi au barreau 2/3.
    """
    if ts < sample_start + pd.Timedelta(days=QC_WARMUP_DAYS):
        return "sample_edge_warmup"

    for start, end in other_holdings:
        if end is not None and start <= ts <= end:
            return "counterpart_position_open"

    if equity_for_size is not None and _lots_for_risk(equity_for_size, entry_px, stop) < VOLUME_MIN:
        return "size_below_min"

    window = pd.Timedelta(minutes=(tol_bars + NEIGHBOUR_BARS) * BAR_MINUTES)
    near = other_decisions[(other_decisions["ts"] - ts).abs() <= window]
    if len(near):
        same_scenario = near[near["scenario"] == scenario]
        if len(same_scenario):
            if (same_scenario["level"] - level).abs().min() > 1e-9:
                return "adjacent_level"
            return "near_miss_time"
        return "other_candidate_same_window"

    return "rung2_3_automaton_divergence"


# ═══════════════════════════════════════════════════════════════════════
# 5. BARREAU 5 — décomposition de l'écart de PnL, en unités de R
# ═══════════════════════════════════════════════════════════════════════


def infer_qc_exit_reason(trade: QcTrade, tol: float = 0.35) -> str:
    """Deviner la raison de sortie QC ; le portage ne la met pas dans le tag.

    `_tag` recopie `stop` et `target` mais jamais la raison, et la trace §13 qui
    la porte vit dans l'ObjectStore, inaccessible par l'API sur ce compte. On la
    **déduit** du prix rempli : LEAN sort au marché juste après la détection, donc
    le fill reste au voisinage du niveau franchi, à un glissement d'une minute
    près. `tol` est exprimé en fraction de `|cible - stop|` — la borne est large
    parce que c'est exactement le glissement qu'on cherche à mesurer, et une
    sortie temporelle tombe presque toujours loin des deux bornes.

    Toute valeur renvoyée ici est une **inférence**, pas une lecture ; le JSON
    publie son taux d'accord avec la raison Python pour qu'on puisse la juger.
    """
    if not trade.closed:
        return "NOT_CLOSED"
    stop, target = trade.tag.stop, trade.tag.target
    span = abs(target - stop)
    if span <= 0.0:
        return "UNKNOWN"
    d_stop = abs(trade.exit_px - stop) / span
    d_target = abs(trade.exit_px - target) / span
    if min(d_stop, d_target) > tol:
        return "TIME_OR_SESSION"
    return "STOP" if d_stop <= d_target else "TARGET"


def theoretical_exit(reason: str, tag: QcTag, realised: float) -> float:
    """Le niveau que la spec fait viser à une sortie de raison `reason`.

    `TIME` et `SESSION` sortent à la clôture de la dernière M1 (§9) : aucun
    niveau théorique ne vit dans le tag, donc on rend le prix réalisé. Le poste
    de glissement correspondant vaut alors zéro **par construction**, ce qui est
    honnête : sur ces trades, le glissement de sortie n'est pas mesurable depuis
    le journal d'ordres, il n'est pas nul.
    """
    if reason == "STOP":
        return tag.stop
    if reason == "TARGET":
        return tag.target
    return realised


def decompose(py_row: pd.Series, trade: QcTrade) -> dict[str, Any]:
    """Télescoper l'écart de PnL d'un trade apparié en postes nommés.

    Partant du PnL de la référence, on substitue une à une :

    1. la **taille** (lots Python → quantité QC) ;
    2. le **prix d'entrée** (théorique §9 → fill LEAN) ;
    3. le **gap de sortie Python** (prix réalisé → niveau visé) ;
    4. la **raison de sortie** (niveau visé Python → niveau visé QC) ;
    5. le **prix de sortie** (niveau visé QC → fill LEAN).

    Chaque terme est la différence de deux expressions qui partagent tous leurs
    autres facteurs, donc la somme vaut `pnl_py - pnl_qc` exactement. Le
    dénominateur commun est le **risque Python** : normaliser chaque côté par son
    propre risque casserait le télescopage et interdirait toute somme.
    """
    side = SCENARIO_SIDE[str(py_row["scenario_name"])]
    units_py = side * float(py_row["lots"]) * CONTRACT_SIZE
    units_qc = trade.entry_qty
    entry_py = float(py_row["fill_px"])
    entry_qc = trade.entry_px
    exit_py = float(py_row["exit_px"])
    exit_qc = trade.exit_px

    reason_py = str(py_row["exit_reason_name"])
    reason_qc = infer_qc_exit_reason(trade)
    theo_py = theoretical_exit(reason_py, trade.tag, exit_py)
    theo_qc = theoretical_exit(reason_qc, trade.tag, exit_qc)

    pnl_py = units_py * (exit_py - entry_py)
    pnl_qc = units_qc * (exit_qc - entry_qc)

    size_usd = (units_py - units_qc) * (exit_py - entry_py)
    entry_slip_usd = units_qc * (entry_qc - entry_py)
    py_exit_gap_usd = units_qc * (exit_py - theo_py)
    reason_usd = units_qc * (theo_py - theo_qc)
    exit_slip_usd = units_qc * (theo_qc - exit_qc)

    risk_py = float(py_row["risk_amount"])
    risk_qc = trade.risk_engaged
    scale = risk_py if risk_py > 0 else float("nan")

    total = pnl_py - pnl_qc
    closure = total - (size_usd + entry_slip_usd + py_exit_gap_usd + reason_usd + exit_slip_usd)

    return {
        "ts_decision_py": str(py_row["ts_decision_utc"]),
        "ts_decision_qc": str(trade.tag.ts_decision),
        "scenario": str(py_row["scenario_name"]),
        "level": float(py_row["level"]),
        "entry_order_id": trade.entry_id,
        "exit_order_id": trade.exit_id,
        "exit_reason_py": reason_py,
        "exit_reason_qc": reason_qc,
        "reason_agrees": reason_py == reason_qc,
        "units_py": units_py,
        "units_qc": units_qc,
        "entry_px_py": entry_py,
        "entry_px_qc": entry_qc,
        "exit_px_py": exit_py,
        "exit_px_qc": exit_qc,
        "entry_slippage_px": side * (entry_qc - entry_py),
        "exit_slippage_px": side * (exit_qc - theo_qc),
        "risk_py": risk_py,
        "risk_qc": risk_qc,
        "pnl_py": pnl_py,
        "pnl_qc": pnl_qc,
        "r_py": pnl_py / scale,
        "r_qc": pnl_qc / risk_qc if risk_qc and risk_qc > 0 else float("nan"),
        "r_qc_on_py_risk": pnl_qc / scale,
        "gap_r": total / scale,
        "r_size": size_usd / scale,
        "r_entry_slippage": entry_slip_usd / scale,
        "r_py_exit_gap": py_exit_gap_usd / scale,
        "r_exit_reason": reason_usd / scale,
        "r_exit_slippage": exit_slip_usd / scale,
        "closure_usd": closure,
    }


# ═══════════════════════════════════════════════════════════════════════
# 6. AGRÉGATION
# ═══════════════════════════════════════════════════════════════════════

POSTS: tuple[str, ...] = (
    "r_size",
    "r_entry_slippage",
    "r_py_exit_gap",
    "r_exit_reason",
    "r_exit_slippage",
)


def _stats(values: np.ndarray) -> dict[str, float]:
    clean = values[np.isfinite(values)]
    if not len(clean):
        return {"n": 0, "sum": 0.0, "mean": float("nan")}
    return {
        "n": int(len(clean)),
        "sum": float(clean.sum()),
        "mean": float(clean.mean()),
        "median": float(np.median(clean)),
        "p05": float(np.quantile(clean, 0.05)),
        "p95": float(np.quantile(clean, 0.95)),
        "min": float(clean.min()),
        "max": float(clean.max()),
    }


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Les agrégats du barreau 5 : global, par raison de sortie, par scénario."""
    frame = pd.DataFrame(rows)
    if frame.empty:
        return {"n": 0}

    def block(sub: pd.DataFrame) -> dict[str, Any]:
        out: dict[str, Any] = {
            "n": int(len(sub)),
            "r_py_sum": float(sub["r_py"].sum()),
            "r_py_mean": float(sub["r_py"].mean()),
            "r_qc_on_py_risk_sum": float(sub["r_qc_on_py_risk"].sum()),
            "r_qc_on_py_risk_mean": float(sub["r_qc_on_py_risk"].mean()),
            "gap_r_sum": float(sub["gap_r"].sum()),
            "gap_r_mean": float(sub["gap_r"].mean()),
            "posts": {p: _stats(sub[p].to_numpy(dtype=float)) for p in POSTS},
        }
        out["posts_sum_total"] = sum(out["posts"][p]["sum"] for p in POSTS)
        out["unattributed_r_sum"] = out["gap_r_sum"] - out["posts_sum_total"]
        return out

    return {
        "overall": block(frame),
        "by_exit_reason_py": {
            str(k): block(g) for k, g in frame.groupby("exit_reason_py", sort=True)
        },
        "by_exit_reason_qc": {
            str(k): block(g) for k, g in frame.groupby("exit_reason_qc", sort=True)
        },
        "by_scenario": {str(k): block(g) for k, g in frame.groupby("scenario", sort=True)},
        "exit_reason_agreement_rate": float(frame["reason_agrees"].mean()),
        "decomposition_closure_max_abs": float(frame["closure_usd"].abs().max()),
        "entry_slippage_px": _stats(frame["entry_slippage_px"].to_numpy(dtype=float)),
        "exit_slippage_px": _stats(frame["exit_slippage_px"].to_numpy(dtype=float)),
    }


#: §12 — demi-fourchette de référence, `spread_current = 0,29 $` / 2.
REFERENCE_HALF_SPREAD = 0.145


def implied_half_spread(py_row: pd.Series, tag: QcTag) -> float:
    """La demi-fourchette que QC a réellement facturée, retrouvée par la règle R.

    §8 pose `R = (|cible - close| - h) / (|close - stop| + h)`. La référence y
    met `h = 0,145` constant ; le portage y met la vraie demi-fourchette de la
    dernière minute de la barre de décision (`x10_state.py:1065`). En inversant
    la relation on obtient, pour chaque côté,

        close + q*h = (cible + R*stop) / (R + 1)  =:  K(R, stop)

    donc, **à `close` commun aux deux moteurs**, `h_qc = q*(K_qc - K_py) + 0,145`.
    C'est la seule façon de chiffrer l'écart 5 du `README.md` du portage depuis le
    seul journal d'ordres : ni le spread ni le close ne sont dans le tag.

    L'hypothèse `close` commun est exactement ce que §13 suppose (mêmes barres) ;
    si elle est fausse, le nombre rendu absorbe aussi l'écart de close, ce que la
    note doit dire au lieu de le cacher.
    """
    side = SCENARIO_SIDE[str(py_row["scenario_name"])]
    target = float(py_row["target"])
    r_py, stop_py = float(py_row["r_est"]), float(py_row["stop"])
    r_qc, stop_qc = tag.r_est, tag.stop
    if r_py <= -1.0 or r_qc <= -1.0:
        return float("nan")
    k_py = (target + r_py * stop_py) / (r_py + 1.0)
    k_qc = (target + r_qc * stop_qc) / (r_qc + 1.0)
    return side * (k_qc - k_py) + REFERENCE_HALF_SPREAD


def rung4_theoretical(
    py: pd.DataFrame,
    qc_trades: list[QcTrade],
    pairs: list[tuple[int, int]],
    tol: Tolerance,
) -> dict[str, Any]:
    """Barreau 4 — `stop`, `cible`, `r_est` **théoriques**, issus du tag.

    Ces trois nombres ne dépendent d'aucun fill : ils sortent de l'ATR, du niveau
    et de la règle R (§7, §8). Un écart ici n'est pas un écart d'exécution, c'est
    un écart d'indicateur ou de seuil, et il rend le barreau 5 ininterprétable.
    """
    out: dict[str, Any] = {"tolerance": asdict(tol)}
    for name in ("stop", "target", "r_est"):
        a = np.array([float(py.loc[i, name]) for i, _ in pairs], dtype=float)
        b = np.array([getattr(qc_trades[j].tag, name) for _, j in pairs], dtype=float)
        delta = np.abs(a - b)
        rel = delta / np.maximum(np.abs(a), 1e-12)
        breaching = int((delta > tol.atol + tol.rtol * np.abs(b)).sum())
        out[name] = {
            "n": int(len(a)),
            "n_breaching": breaching,
            "max_abs": float(delta.max()) if len(delta) else 0.0,
            "max_rel": float(rel.max()) if len(rel) else 0.0,
            "q50_abs": float(np.quantile(delta, 0.50)) if len(delta) else 0.0,
            "q95_abs": float(np.quantile(delta, 0.95)) if len(delta) else 0.0,
            "q99_abs": float(np.quantile(delta, 0.99)) if len(delta) else 0.0,
            "passed": breaching == 0,
        }

    # L'ATR implicite du stop d'un BREAKOUT : `stop = level - d * k_s * atr`
    # (`x10_engine.py:860`, `x10_state.py:752`). La formule, la période, le
    # seeding et l'index de barre sont identiques des deux côtés ; un écart ici
    # ne peut venir que des **barres M5**, donc du barreau 2.
    ratios = []
    for i, j in pairs:
        if not str(py.loc[i, "scenario_name"]).startswith("BREAK"):
            continue
        level = float(py.loc[i, "level"])
        atr_py = abs(level - float(py.loc[i, "stop"]))
        atr_qc = abs(level - qc_trades[j].tag.stop)
        if atr_py > 0:
            ratios.append(atr_qc / atr_py)
    out["breakout_implied_atr_ratio_qc_over_py"] = _stats(np.array(ratios, dtype=float))

    halves = np.array(
        [implied_half_spread(py.loc[i], qc_trades[j].tag) for i, j in pairs], dtype=float
    )
    out["implied_half_spread_qc"] = _stats(halves)
    out["reference_half_spread"] = REFERENCE_HALF_SPREAD
    out["implied_spread_qc_over_reference_median"] = (
        float(np.nanmedian(halves)) / REFERENCE_HALF_SPREAD if len(halves) else float("nan")
    )
    return out


def phantom_inventory(orders: list[dict[str, Any]], ledger: QcLedger) -> dict[str, Any]:
    """La position que le portage croit fermée et que le broker porte encore.

    `main.py:207-209` remet `self._open_qty = 0.0` **sans regarder le sort de
    l'ordre de sortie**. Quand LEAN refuse cet ordre — et il le refuse, parce
    qu'un `market_order` envoyé pendant la coupure quotidienne du CFD est
    converti en `MarketOnOpen`, type que le modèle OANDA n'accepte pas — la
    position logique disparaît et la position réelle survit. Tout ce qui suit
    s'empile dessus.

    On mesure l'inventaire résiduel comme la position nette du compte à chaque
    ordre, moins la position que les allers-retours *voulus* justifient. Un
    résidu non nul entre deux trades est, par construction, de l'inventaire que
    personne n'a demandé.
    """
    rejected = {int(r["order_id"]) for r in ledger.rejected_exits}
    position = 0.0
    positions: list[float] = []
    timeline: list[dict[str, Any]] = []

    for order in sorted(orders, key=lambda o: int(o["id"])):
        oid = int(order["id"])
        stamp = str(order["time"])
        if int(order.get("status", -1)) != LEAN_STATUS_FILLED:
            if oid in rejected:
                timeline.append(
                    {
                        "order_id": oid,
                        "time": stamp,
                        "event": "exit_rejected",
                        "quantity_not_executed": float(order["quantity"]),
                        "order_type": int(order["type"]),
                        "position_left_open_oz": position,
                        "reason": _order_reject_reason(order),
                    }
                )
            continue
        position += float(order["quantity"])
        positions.append(position)
        if str(order.get("tag", "")).startswith("Margin Call"):
            timeline.append(
                {
                    "order_id": oid,
                    "time": stamp,
                    "event": "margin_call",
                    "quantity": float(order["quantity"]),
                    "price": float(order["price"]),
                    "position_after_oz": position,
                }
            )

    flat = [p for p in positions if abs(p) < 1e-9]
    return {
        "n_rejected_exits": len(ledger.rejected_exits),
        "n_rejected_exits_market_on_open": sum(
            1 for r in ledger.rejected_exits if int(r["type"]) == LEAN_TYPE_MARKET_ON_OPEN
        ),
        "first_residual_at": timeline[0]["time"] if timeline else None,
        "final_net_position_oz": position,
        "n_fills": len(positions),
        "n_fills_leaving_account_flat": len(flat),
        "max_abs_net_position_oz": max((abs(p) for p in positions), default=0.0),
        "timeline": timeline,
    }


def sizing_audit(qc_trades: list[QcTrade], equity_by_order: dict[int, float]) -> dict[str, Any]:
    """§11 — le risque engagé côté QC vaut-il bien 0,5 % de l'équité ?

    `risque = |fill - stop| * quantité`, rapporté à l'équité que le portage avait
    sous les yeux au moment de dimensionner. Un ratio nettement au-dessus de
    `f` est un bug de taille ou d'unité de quantité, pas un arrondi.
    """
    rows = []
    for trade in qc_trades:
        if not trade.entry_filled:
            continue
        equity = equity_by_order.get(trade.entry_id)
        if equity is None or equity <= 0:
            continue
        rows.append(
            {
                "entry_order_id": trade.entry_id,
                "time": str(trade.entry_time),
                "scenario": trade.tag.scenario,
                "quantity_oz": trade.entry_qty,
                "lots": abs(trade.entry_qty) / CONTRACT_SIZE,
                "entry_px": trade.entry_px,
                "stop": trade.tag.stop,
                "stop_distance": abs(trade.entry_px - trade.tag.stop),
                "risk_usd": trade.risk_engaged,
                "equity_before_usd": equity,
                "risk_pct_of_equity": trade.risk_engaged / equity,
            }
        )
    if not rows:
        return {"n": 0}
    frame = pd.DataFrame(rows)
    ratio = frame["risk_pct_of_equity"].to_numpy(dtype=float)
    worst = frame.reindex(frame["risk_pct_of_equity"].abs().sort_values(ascending=False).index)
    return {
        "n": int(len(frame)),
        "target_risk_frac": RISK_FRAC,
        "risk_pct_of_equity": _stats(ratio),
        "n_above_1_5x_target": int((ratio > 1.5 * RISK_FRAC).sum()),
        "n_below_0_5x_target": int((ratio < 0.5 * RISK_FRAC).sum()),
        "verdict": (
            "conforme"
            if abs(float(np.median(ratio)) - RISK_FRAC) <= 0.1 * RISK_FRAC
            else "NON CONFORME"
        ),
        "five_largest_examples": worst.head(5).to_dict(orient="records"),
    }


# ═══════════════════════════════════════════════════════════════════════
# 7. CLI
# ═══════════════════════════════════════════════════════════════════════


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--py-trades",
        type=Path,
        required=True,
        nargs="+",
        help="registre(s) de trades de la référence ; plusieurs = plusieurs variantes "
        "d'échantillon comparées au même backtest QC",
    )
    parser.add_argument("--qc-orders", type=Path, required=True)
    parser.add_argument("--qc-stats", type=Path, default=None)
    parser.add_argument(
        "--mt5-trace",
        type=Path,
        default=None,
        help="trace §13 de l'EA MQL5 — branche prévue, pas encore implémentée",
    )
    parser.add_argument("--tol-bars", type=int, default=1)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args(argv)


def _float_stat(stats: dict[str, Any], key: str) -> float | None:
    raw = stats.get(key)
    if raw is None:
        return None
    return float(str(raw).replace("%", "").replace("$", "").replace(",", ""))



@dataclass(frozen=True)
class QcContext:
    """Tout ce que le backtest QC apporte, indépendamment de la variante Python."""

    orders: list[dict[str, Any]]
    ledger: QcLedger
    stats: dict[str, Any]
    start_equity: float
    end_equity: float | None
    equity_before_order: dict[int, float]
    pnl: dict[str, Any]
    sample_start: pd.Timestamp


def load_qc(qc_orders: Path, qc_stats: Path | None) -> QcContext:
    """Charger le journal d'ordres, en reconstruire les trades et l'équité."""
    orders = json.loads(qc_orders.read_text())
    ledger = reconstruct_qc_trades(orders)
    if not ledger.trades:
        raise SystemExit(f"[qc] aucun ordre de stratégie reconnu dans {qc_orders}")

    stats: dict[str, Any] = {}
    if qc_stats and qc_stats.exists():
        stats = json.loads(qc_stats.read_text())
    statistics = stats.get("statistics", {})
    start_equity = _float_stat(statistics, "Start Equity") or 10_000.0
    end_equity = _float_stat(statistics, "End Equity")
    holdings = _float_stat(stats.get("runtimeStatistics", {}), "Holdings")

    raw = account_pnl(orders)
    # §11 dimensionne sur l'équité **avant** l'ordre courant : c'est celle que le
    # portage lit dans `portfolio.total_portfolio_value` à la minute précédente.
    equity_before_order: dict[int, float] = {}
    previous = start_equity
    for point in raw["equity_curve"]:
        equity_before_order[point["order_id"]] = previous
        previous = start_equity + point["pnl"]

    # La position finale est encore ouverte : la marquer au prix que le backtest
    # publie (`Holdings`) plutôt qu'au dernier prix traité, qui est arbitraire.
    mark = None
    if holdings and raw["final_position"]:
        mark = holdings / abs(raw["final_position"])
    pnl = account_pnl(orders, mark_price=mark)

    return QcContext(
        orders=orders,
        ledger=ledger,
        stats=stats,
        start_equity=start_equity,
        end_equity=end_equity,
        equity_before_order=equity_before_order,
        pnl=pnl,
        sample_start=min(t.entry_time for t in ledger.trades),
    )


def qc_only_sections(qc: QcContext, matched_pnl: float) -> dict[str, Any]:
    """Ce qui se lit dans le seul journal d'ordres, sans aucune référence Python.

    Cette partie est la plus importante du rapport et elle ne dépend d'aucun
    appariement : même si la réconciliation échouait entièrement, le compte dit
    tout seul où est passé l'argent.
    """
    ledger, pnl = qc.ledger, qc.pnl
    total = pnl["realised_and_unrealised_pnl"]
    closed = [t for t in ledger.trades if t.closed]
    reasons = pd.Series([infer_qc_exit_reason(t) for t in closed])

    reported = (qc.end_equity - qc.start_equity) if qc.end_equity is not None else None
    return {
        "order_book": {
            "n_orders": len(qc.orders),
            "n_strategy_positions": len(ledger.trades),
            "n_closed": len(closed),
            "n_margin_call_orders": len(ledger.margin_calls),
            "n_rejected_entries": len(ledger.rejected_entries),
            "n_rejected_exits": len(ledger.rejected_exits),
            "rejected_exits": ledger.rejected_exits,
            "rejected_entries": ledger.rejected_entries,
            "margin_calls": ledger.margin_calls,
            "tag_key_mismatches": ledger.tag_key_mismatches,
            "unparsed_tags": ledger.unparsed_tags,
            "inferred_exit_reason_mix": reasons.value_counts().to_dict(),
        },
        "phantom_inventory": phantom_inventory(qc.orders, ledger),
        "sizing_audit": sizing_audit(ledger.trades, qc.equity_before_order),
        "composition": {
            "start_equity": qc.start_equity,
            "end_equity_reported": qc.end_equity,
            "net_profit_reported": reported,
            "final_position_oz": pnl["final_position"],
            "mark_price": pnl["mark_price"],
            "account_pnl_from_orders": total,
            "accounting_closure_vs_reported": (
                total - reported if reported is not None else None
            ),
            "pnl_of_intended_round_trips": matched_pnl,
            "pnl_of_residual_inventory": total - matched_pnl,
            "residual_share_of_loss": (total - matched_pnl) / total if total else None,
            "expectancy_usd_per_intended_round_trip": (
                matched_pnl / len(closed) if closed else None
            ),
            "equity_curve": [
                {
                    "time": point["time"],
                    "order_id": point["order_id"],
                    "position_oz": point["position"],
                    "equity": qc.start_equity + point["pnl"],
                }
                for point in pnl["equity_curve"]
            ],
        },
    }


def build_variant(py_path: Path, qc: QcContext, tol_bars: int) -> dict[str, Any]:
    """Réconcilier **une** variante d'échantillon Python contre le backtest QC."""
    py = load_py_trades(py_path)
    trades = qc.ledger.trades
    pairs, py_orphans, qc_orphans = match_entries(py, trades, tol_bars=tol_bars)

    qc_decisions = pd.DataFrame(
        [{"ts": t.tag.ts_decision, "scenario": t.tag.scenario, "level": t.tag.level} for t in trades]
    )
    qc_holdings = [(t.entry_time, t.exit_time) for t in trades]
    py_decisions = pd.DataFrame(
        {
            "ts": py["ts_decision_utc"],
            "scenario": py["scenario_name"],
            "level": py["level"],
        }
    )
    py_holdings = list(zip(py["ts_fill_utc"], py["ts_exit_utc"], strict=True))

    py_orphan_rows = []
    for i in py_orphans:
        row = py.loc[i]
        py_orphan_rows.append(
            {
                "ts_decision_utc": str(row["ts_decision_utc"]),
                "scenario": str(row["scenario_name"]),
                "level": float(row["level"]),
                "exit_reason": str(row["exit_reason_name"]),
                "r_realised": float(row["risk_amount"])
                and float(
                    SCENARIO_SIDE[str(row["scenario_name"])]
                    * float(row["lots"])
                    * CONTRACT_SIZE
                    * (float(row["exit_px"]) - float(row["fill_px"]))
                )
                / float(row["risk_amount"]),
                "cause": _classify(
                    row["ts_decision_utc"],
                    str(row["scenario_name"]),
                    float(row["level"]),
                    float(row["fill_px"]),
                    float(row["stop"]),
                    qc.sample_start,
                    qc_decisions,
                    qc_holdings,
                    tol_bars,
                    equity_for_size=None,
                ),
            }
        )

    qc_orphan_rows = []
    for j in qc_orphans:
        trade = trades[j]
        before = py[py["ts_decision_utc"] <= trade.tag.ts_decision]
        equity = float(before.iloc[-1]["equity_before"]) if len(before) else None
        qc_orphan_rows.append(
            {
                "ts_decision_utc": str(trade.tag.ts_decision),
                "scenario": trade.tag.scenario,
                "level": trade.tag.level,
                "entry_order_id": trade.entry_id,
                "cause": _classify(
                    trade.tag.ts_decision,
                    trade.tag.scenario,
                    trade.tag.level,
                    trade.entry_px,
                    trade.tag.stop,
                    qc.sample_start,
                    py_decisions,
                    py_holdings,
                    tol_bars,
                    equity_for_size=equity,
                ),
            }
        )

    rows = [decompose(py.loc[i], trades[j]) for i, j in pairs if trades[j].closed]
    unclosed = [
        {
            "entry_order_id": trades[j].entry_id,
            "ts_decision_qc": str(trades[j].tag.ts_decision),
            "scenario": trades[j].tag.scenario,
            "level": trades[j].tag.level,
            "entry_qty_oz": trades[j].entry_qty,
            "entry_px": trades[j].entry_px,
            "exit_order_type": trades[j].exit_order_type,
            "exit_reject_reason": trades[j].exit_reject_reason,
        }
        for _, j in pairs
        if not trades[j].closed
    ]

    n_py, n_qc = len(py), len(trades)
    rate_py = len(pairs) / n_py if n_py else 0.0
    rate_qc = len(pairs) / n_qc if n_qc else 0.0
    unexplained = [
        o
        for o in (*py_orphan_rows, *qc_orphan_rows)
        if o["cause"] == "rung2_3_automaton_divergence"
    ]

    py_risk_pct = (py["risk_amount"] / py["equity_before"]).to_numpy(dtype=float)

    return {
        "py_trades": str(py_path),
        "rung3_4_entry_matching": {
            "n_py_entries": n_py,
            "n_qc_entries": n_qc,
            "n_matched": len(pairs),
            "match_rate_py": rate_py,
            "match_rate_qc": rate_qc,
            "target_same_data": MATCH_TARGET_SAME_DATA,
            "passed": min(rate_py, rate_qc) >= MATCH_TARGET_SAME_DATA,
            "cause_counts_py_orphans": pd.Series(
                [o["cause"] for o in py_orphan_rows]
            ).value_counts().to_dict(),
            "cause_counts_qc_orphans": pd.Series(
                [o["cause"] for o in qc_orphan_rows]
            ).value_counts().to_dict(),
            "n_unexplained_entries": len(unexplained),
            "unexplained_share_of_entries": len(unexplained) / max(n_py + n_qc, 1),
            "unattributed_blocker": UNATTRIBUTED_BLOCKER,
            "py_orphans": py_orphan_rows,
            "qc_orphans": qc_orphan_rows,
        },
        "rung4_theoretical": rung4_theoretical(py, trades, pairs, TOL_SAME_DATA),
        "rung5_attribution": aggregate(rows),
        "rung5_trades": rows,
        "matched_pairs_unclosed_in_qc": unclosed,
        "py_reference": {
            "n_trades": n_py,
            "exit_reason_mix": py["exit_reason_name"].value_counts().to_dict(),
            "risk_pct_of_equity": _stats(py_risk_pct),
            "mean_bars_held": float(py["bars_held"].mean()) if "bars_held" in py else None,
        },
    }


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    qc = load_qc(args.qc_orders, args.qc_stats)
    variants = {p.stem: build_variant(p, qc, args.tol_bars) for p in args.py_trades}

    matched_pnl = sum(t.pnl for t in qc.ledger.trades if t.closed)
    return {
        "meta": {
            "qc_orders": str(args.qc_orders),
            "qc_backtest": qc.stats.get("backtestId"),
            "qc_backtest_name": qc.stats.get("name"),
            "qc_parameters": qc.stats.get("parameterSet"),
            "qc_statistics": qc.stats.get("statistics"),
            "tol_bars": args.tol_bars,
            "py_variants": [str(p) for p in args.py_trades],
            "mt5": "en attente — aucune trace MT5 fournie",
        },
        "qc": qc_only_sections(qc, matched_pnl),
        "variants": variants,
        "tolerances": {
            "same_data": asdict(TOL_SAME_DATA),
            "broker_feed": asdict(TOL_BROKER_FEED),
            "unattributed_blocker": UNATTRIBUTED_BLOCKER,
        },
    }


# ═══════════════════════════════════════════════════════════════════════
# 8. RÉSUMÉ CONSOLE
# ═══════════════════════════════════════════════════════════════════════


def _print_variant(name: str, variant: dict[str, Any]) -> None:
    match = variant["rung3_4_entry_matching"]
    rung4 = variant["rung4_theoretical"]
    agg = variant["rung5_attribution"]

    print("-" * 74)
    print(f"VARIANTE {name}")
    print(
        f"barreau 3/4  entrées Python={match['n_py_entries']}  QC={match['n_qc_entries']}  "
        f"appariées={match['n_matched']}"
    )
    print(
        f"             taux Python→QC={match['match_rate_py']:.1%}  "
        f"QC→Python={match['match_rate_qc']:.1%}  "
        f"(cible ≥ {match['target_same_data']:.0%})  "
        f"{'OK' if match['passed'] else 'ÉCHEC'}"
    )
    print(f"             causes orphelins Python : {match['cause_counts_py_orphans']}")
    print(f"             causes orphelins QC     : {match['cause_counts_qc_orphans']}")
    print(
        f"             entrées sans cause nommée : {match['n_unexplained_entries']} "
        f"({match['unexplained_share_of_entries']:.1%} ; blocage au-delà de "
        f"{match['unattributed_blocker']:.0%})"
    )

    for field_name in ("stop", "target", "r_est"):
        cell = rung4[field_name]
        print(
            f"barreau 4    {field_name:<7} max|Δ|={cell['max_abs']:.4f}  "
            f"q95={cell['q95_abs']:.4f}  hors tolérance={cell['n_breaching']}/{cell['n']}  "
            f"{'OK' if cell['passed'] else 'ÉCHEC'}"
        )
    atr = rung4["breakout_implied_atr_ratio_qc_over_py"]
    half = rung4["implied_half_spread_qc"]
    print(
        f"             ATR M5 implicite QC/Python (BREAK, n={atr['n']}) : "
        f"médiane={atr.get('median', float('nan')):.4f}  "
        f"[{atr.get('min', float('nan')):.3f}, {atr.get('max', float('nan')):.3f}]"
    )
    print(
        f"             demi-spread QC implicite : médiane={half.get('median', float('nan')):.3f} $ "
        f"contre {rung4['reference_half_spread']:.3f} $ en référence "
        f"(x{rung4['implied_spread_qc_over_reference_median']:.2f})"
    )

    overall = agg.get("overall")
    if not overall:
        print("barreau 5    aucun trade apparié et clos des deux côtés")
        return
    print(f"barreau 5    {overall['n']} trades appariés ET clos des deux côtés")
    print(
        f"             R/trade  Python={overall['r_py_mean']:+.4f}   "
        f"QC={overall['r_qc_on_py_risk_mean']:+.4f}   "
        f"écart={overall['gap_r_mean']:+.4f} R/trade"
    )
    for post in POSTS:
        cell = overall["posts"][post]
        print(
            f"               {post:<20} somme={cell['sum']:+9.3f} R   "
            f"moyenne={cell['mean']:+.4f} R"
        )
    print(f"               {'non attribué':<20} somme={overall['unattributed_r_sum']:+9.3f} R")
    print(
        f"             fermeture de la décomposition : max|résidu| = "
        f"{agg['decomposition_closure_max_abs']:.3e} USD"
    )
    print(
        f"             glissement px (sens du trade) ENTRÉE moyen="
        f"{agg['entry_slippage_px']['mean']:+.4f} $   "
        f"SORTIE moyen={agg['exit_slippage_px']['mean']:+.4f} $"
    )
    print(f"             accord de raison de sortie : {agg['exit_reason_agreement_rate']:.1%}")


def print_summary(report: dict[str, Any]) -> None:
    meta = report["meta"]
    qc = report["qc"]
    book, phantom, sizing, comp = (
        qc["order_book"],
        qc["phantom_inventory"],
        qc["sizing_audit"],
        qc["composition"],
    )

    print("═" * 74)
    print(f"Réconciliation événementielle x10 — backtest QC « {meta['qc_backtest_name']} »")
    print("═" * 74)
    print(
        f"journal QC   {book['n_orders']} ordres → {book['n_strategy_positions']} positions "
        f"(clos={book['n_closed']})"
    )
    print(
        f"             sorties REJETÉES={book['n_rejected_exits']} "
        f"(dont MarketOnOpen={phantom['n_rejected_exits_market_on_open']})  "
        f"entrées rejetées={book['n_rejected_entries']}  "
        f"appels de marge={book['n_margin_call_orders']}"
    )
    print(
        f"             le compte n'est à plat que sur "
        f"{phantom['n_fills_leaving_account_flat']}/{phantom['n_fills']} remplissages ; "
        f"position nette max={phantom['max_abs_net_position_oz']:.0f} oz, "
        f"finale={phantom['final_net_position_oz']:+.0f} oz"
    )
    print(f"             raisons de sortie QC (inférées) : {book['inferred_exit_reason_mix']}")
    print("-" * 74)
    print(
        f"taille QC    risque/équité médian={sizing['risk_pct_of_equity']['median']:.4%}  "
        f"(cible {sizing['target_risk_frac']:.2%})  "
        f"p95={sizing['risk_pct_of_equity']['p95']:.4%}  "
        f"max={sizing['risk_pct_of_equity']['max']:.4%}  → {sizing['verdict']}"
    )
    print("-" * 74)
    print(
        f"composition  {comp['start_equity']:.0f} → {comp['end_equity_reported']:.2f} USD "
        f"({comp['net_profit_reported'] / comp['start_equity']:+.1%})"
    )
    print(
        f"             PnL du compte reconstruit={comp['account_pnl_from_orders']:+.2f} USD "
        f"(bouclage vs backtest : {comp['accounting_closure_vs_reported']:+.2f} USD)"
    )
    print(
        f"             dont allers-retours VOULUS = "
        f"{comp['pnl_of_intended_round_trips']:+.2f} USD "
        f"({comp['expectancy_usd_per_intended_round_trip']:+.2f} USD/trade)"
    )
    print(
        f"             dont inventaire RÉSIDUEL   = "
        f"{comp['pnl_of_residual_inventory']:+.2f} USD "
        f"→ {comp['residual_share_of_loss']:.1%} de la perte"
    )

    for name, variant in report["variants"].items():
        _print_variant(name, variant)
    print("═" * 74)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.mt5_trace is not None:
        raise SystemExit(
            "--mt5-trace : branche prévue par §13 mais pas encore implémentée ; "
            "l'EA MQL5 n'a pas encore tourné dans le tester."
        )
    report = build_report(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    print_summary(report)
    print(f"→ {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
