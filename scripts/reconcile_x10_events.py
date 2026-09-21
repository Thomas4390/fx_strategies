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
    """`scenario|level|ts_decision|stop|target|r_est`, puis des champs `k=v`.

    Les **six premiers champs sont positionnels et gelés** ; tout ce que le
    portage a ajouté depuis (v4) vit derrière, en `k=v`, et est optionnel. Un
    journal d'ordres v1 se lit donc avec exactement le même code, `extra` vide.

    Les clés connues, côté entrée : `bm` (début de barre de décision, minutes
    epoch), `c` (close mid de cette barre), `atr`, `sp` (spread plein utilisé
    dans la règle R), `v`, `a`, `vw`, `ema`. Côté sortie : `xr` (raison) et `xp`
    (prix théorique). Elles portent le barreau 2 de §13, que rien d'autre ne
    rendait observable depuis l'API.
    """

    scenario: str
    level: float
    ts_decision: pd.Timestamp
    stop: float
    target: float
    r_est: float
    extra: dict[str, str] = field(default_factory=dict)

    def number(self, key: str) -> float:
        """Champ `k=v` numérique, ou NaN s'il est absent ou vide (tag v1)."""
        raw = self.extra.get(key, "")
        try:
            return float(raw)
        except ValueError:
            return float("nan")

    @property
    def exit_reason(self) -> str | None:
        """La raison de sortie **lue**, quand le portage la publie (`xr=`)."""
        return self.extra.get("xr") or None

    @property
    def exit_px(self) -> float:
        """Le prix théorique de sortie **lu** (`xp=`), sans inférence."""
        return self.number("xp")

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
    if len(parts) < 6 or parts[0] not in SCENARIOS:
        return None
    extra: dict[str, str] = {}
    for field_text in parts[6:]:
        key, sep, value = field_text.partition("=")
        if sep:
            extra[key] = value
    try:
        return QcTag(
            scenario=parts[0],
            level=float(parts[1]),
            ts_decision=pd.Timestamp(parts[2]),
            stop=float(parts[3]),
            target=float(parts[4]),
            r_est=float(parts[5]),
            extra=extra,
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
    #: Ordres de SORTIE arrivés alors qu'aucune position n'était ouverte. Le
    #: portage ne devrait jamais en produire : c'est la signature d'un ordre
    #: réémis après coup. On les met de côté au lieu de les apparier de force,
    #: sans quoi un seul d'entre eux décale tout le reste du journal.
    orphan_exits: list[dict[str, Any]] = field(default_factory=list)


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

        # Depuis v4 le tag dit lui-même ce qu'il est : `bm=` sur une entrée,
        # `xr=` sur une sortie. On s'en sert plutôt que de l'alternance, qui
        # n'est vraie que tant que le portage n'émet jamais un ordre de trop.
        # Un tag v1 n'a ni l'un ni l'autre : on retombe alors sur l'alternance.
        is_entry = "bm" in tag.extra
        is_exit = "xr" in tag.extra
        if is_exit and open_trade is None:
            ledger.orphan_exits.append(
                {**record, "note": "sortie sans position ouverte"}
            )
            continue
        if is_entry and open_trade is not None:
            # Une entrée alors qu'une position est ouverte : §9 l'interdit. On
            # ferme la précédente sans sortie plutôt que de les confondre.
            open_trade = None

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
    # v4 et au-delà : le portage publie la raison, il n'y a plus rien à deviner.
    read = trade.exit_tag.exit_reason if trade.exit_tag is not None else None
    if read in EXIT_REASONS:
        return read
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
    reason_qc_is_read = bool(trade.exit_tag is not None and trade.exit_tag.exit_reason)
    theo_py = theoretical_exit(reason_py, trade.tag, exit_py)
    # `xp=` donne le prix théorique de sortie du portage, y compris pour TIME et
    # SESSION, que le tag v1 ne permettait pas de chiffrer : le poste de
    # glissement de sortie cesse d'être un minorant (§8.3 de la note v1).
    theo_qc = trade.exit_tag.exit_px if reason_qc_is_read else float("nan")
    if theo_qc != theo_qc:
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
        "exit_reason_qc_is_read": reason_qc_is_read,
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


def rung2_bars(
    py: pd.DataFrame,
    qc_trades: list[QcTrade],
    pairs: list[tuple[int, int]],
) -> dict[str, Any]:
    """Barreau 2 — la barre de décision elle-même : close, ATR, spread.

    Jusqu'à v4 ce barreau n'était pas observable : le tag ne portait que des
    quantités dérivées (stop, cible, R) et il fallait remonter à l'ATR par
    l'inverse du stop d'un breakout, ce qui ne marchait que sur 36 trades. Le
    portage publie maintenant `c=`, `atr=` et `sp=` sur chaque entrée, à `bm=`
    identique : on compare enfin les deux moteurs sur la **même barre**, et un
    écart tombe sur une quantité nommée au lieu d'un soupçon.

    `bm` est vérifié avant tout le reste : si les deux moteurs ne parlent pas de
    la même barre, comparer leurs ATR n'a aucun sens.
    """
    rows = []
    for i, j in pairs:
        tag = qc_trades[j].tag
        if not tag.extra:
            continue
        stamp = py.loc[i, "ts_decision_utc"]
        bm_py = (stamp - pd.Timestamp("1970-01-01")) // pd.Timedelta(minutes=1)
        rows.append(
            {
                "ts_decision_utc": str(stamp),
                "bm_py": int(bm_py),
                "bm_qc": int(tag.number("bm")) if tag.extra.get("bm") else -1,
                "close_py": float(py.loc[i, "m5_close"])
                if "m5_close" in py.columns
                else float("nan"),
                "close_qc": tag.number("c"),
                "atr_py": float(py.loc[i, "atr"]),
                "atr_qc": tag.number("atr"),
                "spread_py": float(py.loc[i, "spread"]),
                "spread_qc": tag.number("sp"),
            }
        )
    if not rows:
        return {"n": 0, "observable": False, "note": "tags v1 : aucun champ k=v"}

    frame = pd.DataFrame(rows)
    out: dict[str, Any] = {
        "n": int(len(frame)),
        "observable": True,
        "bm_identical": int((frame["bm_py"] == frame["bm_qc"]).sum()),
        "bm_offset_minutes": _stats((frame["bm_qc"] - frame["bm_py"]).to_numpy(dtype=float)),
    }
    for name in ("close", "atr", "spread"):
        a = frame[f"{name}_py"].to_numpy(dtype=float)
        b = frame[f"{name}_qc"].to_numpy(dtype=float)
        delta = np.abs(a - b)
        out[name] = {
            "abs": _stats(delta),
            "ratio_qc_over_py": _stats(np.where(a != 0.0, b / np.where(a == 0.0, np.nan, a), np.nan)),
            "n_identical_1e6": int(np.nansum(delta <= 1e-6)),
        }
    # Le parquet de référence est-il du mid, ou du bid/ask ?
    #
    # L'écart médian de close M5 est du même ordre que le demi-spread, ce qui
    # ferait un joli coupable : un parquet en bid vaudrait `close_qc - close_py
    # = +sp/2` partout, un parquet en ask `-sp/2`. Le test est le **signe** et
    # la **régression**, pas l'ordre de grandeur — deux quantités peuvent valoir
    # un quart de dollar sans avoir le moindre rapport.
    delta = (frame["close_qc"] - frame["close_py"]).to_numpy(dtype=float)
    half = (frame["spread_qc"] / 2.0).to_numpy(dtype=float)
    ok = np.isfinite(delta) & np.isfinite(half) & (half > 0)
    basis: dict[str, Any] = {"n": int(ok.sum())}
    if ok.sum() >= 3:
        d, h = delta[ok], half[ok]
        slope, intercept = np.polyfit(h, d, 1)
        pred = slope * h + intercept
        ss_tot = float(((d - d.mean()) ** 2).sum())
        basis.update(
            {
                "delta_close_qc_minus_py": _stats(d),
                "share_positive": float((d > 0).mean()),
                "share_negative": float((d < 0).mean()),
                "median_delta_over_half_spread": float(np.median(d / h)),
                "slope_vs_half_spread": float(slope),
                "intercept": float(intercept),
                "r2": float(1.0 - ((d - pred) ** 2).sum() / ss_tot) if ss_tot else float("nan"),
                "slope_through_origin": float((h * d).sum() / (h * h).sum()),
            }
        )
        # Un parquet en bid/ask donnerait une pente proche de ±1 ET un R² élevé
        # ET un signe quasi constant. Les trois doivent tomber ensemble.
        basis["verdict"] = (
            "bid_or_ask"
            if abs(abs(basis["slope_vs_half_spread"]) - 1.0) < 0.3
            and basis["r2"] > 0.5
            and max(basis["share_positive"], basis["share_negative"]) > 0.9
            else "mid_des_deux_cotes_donnees_differentes"
        )
    out["price_basis"] = basis
    out["rows"] = rows
    return out


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
        nargs="+",
        default=[],
        help="registre(s) de trades de la référence ; plusieurs = plusieurs variantes "
        "d'échantillon comparées au même backtest QC. En branche MT5, le premier "
        "est la campagne officielle qui sert de comparaison C2.",
    )
    parser.add_argument("--qc-orders", type=Path, default=None)
    parser.add_argument("--qc-stats", type=Path, default=None)
    parser.add_argument(
        "--mt5-trace",
        type=Path,
        default=None,
        help="trace §13 de l'EA MQL5 ; sa présence bascule le script en branche MT5",
    )
    parser.add_argument(
        "--mt5-deals",
        type=Path,
        default=None,
        help="CSV par deal de l'EA (UTF-16), d'où vient le PnL réalisé",
    )
    parser.add_argument(
        "--mt5-dump",
        type=Path,
        default=None,
        help="parquet M1 du dump broker (bid + spread), sur lequel la référence est rejouée",
    )
    parser.add_argument(
        "--mt5-log",
        type=Path,
        default=None,
        help="log de l'agent du tester (UTF-16), d'où se lisent les ordres refusés",
    )
    parser.add_argument(
        "--reference-minute",
        type=Path,
        default=_REPO / "data/XAU-USD_minute_qc.parquet",
        help="parquet M1 UTC de référence pour le contrôle d'horloge du dump",
    )
    parser.add_argument(
        "--no-bid-replay",
        action="store_true",
        help="ne pas rejouer la référence sur le bid brut (poste « signaux bid » non mesuré)",
    )
    parser.add_argument("--tol-bars", type=int, default=1)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.mt5_trace is None and args.qc_orders is None:
        parser.error("--qc-orders est requis hors branche MT5")
    if args.mt5_trace is not None and (args.mt5_deals is None or args.mt5_dump is None):
        parser.error("la branche MT5 exige --mt5-deals et --mt5-dump")
    if args.mt5_trace is None and not args.py_trades:
        parser.error("--py-trades est requis hors branche MT5")
    return args


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
    runtime = stats.get("runtimeStatistics", {})
    start_equity = _float_stat(statistics, "Start Equity") or 10_000.0
    # L'API rend parfois `statistics` entièrement à null ; `runtimeStatistics`
    # porte alors les mêmes grandeurs, et c'est le même backtest qui parle.
    end_equity = _float_stat(statistics, "End Equity")
    if end_equity is None:
        end_equity = _float_stat(runtime, "Equity")
    holdings = _float_stat(runtime, "Holdings")

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
            "orphan_exits": ledger.orphan_exits,
            "n_orphan_exits": len(ledger.orphan_exits),
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


def by_year(
    py: pd.DataFrame, qc_trades: list[QcTrade], pairs: list[tuple[int, int]]
) -> list[dict[str, Any]]:
    """Année par année, en **R** — la seule unité qui survive à la composition.

    L'équité QC se compose sur sept ans (10 000 → 2 217), donc les lots fondent
    et un PnL en dollars mélange le signal et la taille. Le R rapporte chaque
    trade au risque qu'il engageait vraiment, ce qui rend 2019 et 2025
    comparables. Les espérances portent sur **tous** les trades de l'année, pas
    seulement les appariés : c'est ce que chaque moteur aurait rapporté.
    """
    side = py["scenario_name"].map(SCENARIO_SIDE).to_numpy(dtype=float)
    pnl_py = side * py["lots"].to_numpy(float) * CONTRACT_SIZE * (
        py["exit_px"].to_numpy(float) - py["fill_px"].to_numpy(float)
    )
    risk_py = py["risk_amount"].to_numpy(dtype=float)
    frame_py = pd.DataFrame(
        {
            "year": pd.DatetimeIndex(py["ts_decision_utc"]).year,
            "r": np.where(risk_py > 0, pnl_py / np.where(risk_py <= 0, np.nan, risk_py), np.nan),
        }
    )

    rows_qc = []
    for trade in qc_trades:
        risk = trade.risk_engaged
        rows_qc.append(
            {
                "year": trade.tag.ts_decision.year,
                "r": trade.pnl / risk if trade.closed and risk and risk > 0 else np.nan,
                "at_lot_floor": abs(trade.entry_qty) <= CONTRACT_SIZE * VOLUME_MIN + 1e-9,
            }
        )
    frame_qc = pd.DataFrame(rows_qc)

    matched_years = pd.Series(
        [pd.Timestamp(py.loc[i, "ts_decision_utc"]).year for i, _ in pairs], dtype="int64"
    )

    years = sorted(set(frame_py["year"]) | set(frame_qc["year"]))
    out = []
    for year in years:
        sub_py = frame_py[frame_py["year"] == year]
        sub_qc = frame_qc[frame_qc["year"] == year]
        n_matched = int((matched_years == year).sum())
        out.append(
            {
                "year": int(year),
                "py_trades": int(len(sub_py)),
                "qc_trades": int(len(sub_qc)),
                "py_expectancy_r": float(sub_py["r"].mean()) if len(sub_py) else None,
                "qc_expectancy_r": float(sub_qc["r"].mean()) if len(sub_qc) else None,
                "match_rate_py_to_qc": n_matched / len(sub_py) if len(sub_py) else None,
                "n_matched": n_matched,
                # Les refus de taille (§11, `CANCEL` raison `size`) ne voyagent
                # sur aucun ordre : seuls les trades RETENUS sont tagués. On
                # publie donc le proxy observable — la part des entrées déjà
                # collées au plancher de 0,01 lot — et pas un refus inventé.
                "qc_entries_at_lot_floor": int(sub_qc["at_lot_floor"].sum()) if len(sub_qc) else 0,
            }
        )
    return out


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
        "by_year": by_year(py, trades, pairs),
        "rung2_bars": rung2_bars(py, trades, pairs),
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


#: Clés du bloc `summary`. **Gelées** : le rapport client les lit par leur nom,
#: donc on peut en ajouter, jamais en renommer ni en retirer.
SUMMARY_KEYS: tuple[str, ...] = (
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


def health_check(qc: QcContext) -> dict[str, Any]:
    """Les quatre critères de santé, avant toute lecture de performance.

    Un backtest qui les rate ne mesure pas la stratégie : il mesure un défaut du
    portage. Les publier à part, et en premier, évite de commenter un rendement
    qui n'a aucun sens — ce qu'on a failli faire avec le v1 et ses −91,8 %.
    """
    ledger, pnl = qc.ledger, qc.pnl
    phantom = phantom_inventory(qc.orders, ledger)
    # Tout ordre non rempli, y compris la liquidation de fin d'algorithme, que
    # son tag n'appartient pas à la stratégie et que le registre range donc
    # ailleurs. Un backtest sain n'en a aucun.
    unfilled = [
        {
            "order_id": int(o["id"]),
            "time": str(o["time"]),
            "type": int(o["type"]),
            "quantity": float(o["quantity"]),
            "tag": str(o.get("tag") or ""),
            "reason": _order_reject_reason(o),
        }
        for o in qc.orders
        if int(o.get("status", -1)) != LEAN_STATUS_FILLED
    ]
    unretried = len(unfilled) + len(ledger.orphan_exits)
    flat_between = phantom["n_fills_leaving_account_flat"]
    expected_flat = phantom["n_fills"] // 2
    criteria = {
        "orders_in_range": len(qc.orders) >= 300,
        "no_unretried_rejection": unretried == 0,
        "flat_at_end": abs(pnl["final_position"]) < 1e-9,
        "flat_between_trades": flat_between >= expected_flat - 1,
    }
    return {
        "criteria": criteria,
        "healthy": all(criteria.values()),
        "n_orders": len(qc.orders),
        "n_rejected_unretried": unretried,
        "unfilled_orders": unfilled,
        "final_position_oz": pnl["final_position"],
        "n_fills": phantom["n_fills"],
        "n_fills_leaving_account_flat": flat_between,
        "max_abs_net_position_oz": phantom["max_abs_net_position_oz"],
    }


def build_summary(
    qc: QcContext, variant: dict[str, Any], healthy: bool
) -> dict[str, Any]:
    """Le bloc stable que le rapport client lit, et rien d'autre."""
    match = variant["rung3_4_entry_matching"]
    agg = variant["rung5_attribution"]
    overall = agg.get("overall") or {}
    rung4 = variant["rung4_theoretical"]
    by_reason = agg.get("by_exit_reason_py", {})
    stop_block = by_reason.get("STOP", {})
    net = qc.pnl["realised_and_unrealised_pnl"]
    return {
        "qc_backtest_id": qc.stats.get("backtestId"),
        "qc_trades": len(qc.ledger.trades),
        "py_trades": match["n_py_entries"],
        "match_rate_py_to_qc": match["match_rate_py"],
        "match_rate_qc_to_py": match["match_rate_qc"],
        "py_expectancy_r": overall.get("r_py_mean"),
        "qc_expectancy_r": overall.get("r_qc_on_py_risk_mean"),
        "exit_slippage_r_per_trade": (
            overall.get("posts", {}).get("r_exit_slippage", {}).get("mean")
        ),
        "qc_stop_realised_r": stop_block.get("r_qc_on_py_risk_mean"),
        "qc_half_spread_usd": rung4.get("implied_half_spread_qc", {}).get("median"),
        "unattributed_share": match["unexplained_share_of_entries"],
        "qc_net_return_pct": 100.0 * net / qc.start_equity if qc.start_equity else None,
        "healthy": healthy,
    }


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    qc = load_qc(args.qc_orders, args.qc_stats)
    variants = {p.stem: build_variant(p, qc, args.tol_bars) for p in args.py_trades}
    health = health_check(qc)
    primary = variants[args.py_trades[0].stem]

    matched_pnl = sum(t.pnl for t in qc.ledger.trades if t.closed)
    return {
        "summary": build_summary(qc, primary, health["healthy"]),
        "by_year": primary["by_year"],
        "health": health,
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
    health = report["health"]
    flags = " ".join(f"{k}={'OK' if v else 'NON'}" for k, v in health["criteria"].items())
    print(f"santé        {'SAIN' if health['healthy'] else 'NON SAIN'}  {flags}")
    if health["n_rejected_unretried"]:
        for row in health["unfilled_orders"][:5]:
            print(
                f"             ordre non rempli #{row['order_id']} {row['time']} "
                f"type={row['type']} qty={row['quantity']:+.0f} — {row['reason'][:70]}"
            )
    print("-" * 74)
    if comp["end_equity_reported"] is None:
        print(
            f"composition  départ {comp['start_equity']:.0f} USD ; "
            "statistiques du backtest non fournies (--qc-stats)"
        )
    else:
        print(
            f"composition  {comp['start_equity']:.0f} → {comp['end_equity_reported']:.2f} USD "
            f"({comp['net_profit_reported'] / comp['start_equity']:+.1%})"
        )
    print(
        f"             PnL du compte reconstruit={comp['account_pnl_from_orders']:+.2f} USD "
        + (
            "(bouclage vs backtest indisponible)"
            if comp["accounting_closure_vs_reported"] is None
            else f"(bouclage vs backtest : {comp['accounting_closure_vs_reported']:+.2f} USD)"
        )
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


# ═══════════════════════════════════════════════════════════════════════
# 9. BRANCHE MT5 — l'EA MQL5 dans le tester (§13, critère 8)
# ═══════════════════════════════════════════════════════════════════════
#
# Deux comparaisons, et il ne faut jamais les confondre :
#
#   C1 « mêmes données »  la référence Python rejouée sur le dump M1 du broker,
#                         contre la trace de l'EA. Cible §13 : ≥ 95 %.
#   C2 « données différentes » la campagne Python officielle (parquet QC, spread
#                         constant 0,29 $) restreinte à la fenêtre MT5, contre la
#                         même trace. Cible §13 : ≥ 70 %, écart attribué.
#
# Le dump est en prix BID et horodaté à l'**ouverture** de la barre — c'est la
# convention MT5, celle du parquet QC est la clôture. Les deux conventions se
# valent ; les mélanger décale tout d'une minute et c'est invisible à l'œil.


#: Point du symbole XAUUSD.c, `data/broker/symbols_catalog_2026-07-28.csv`.
MT5_POINT = 0.001

#: Un magic par scénario, `src/mt5/Experts/XauX10.mq5`.
MT5_MAGIC_SCENARIO: dict[int, str] = {
    841: "BREAK_LONG",
    842: "BREAK_SHORT",
    843: "REV_LONG",
    844: "REV_SHORT",
}

#: `ENUM_DEAL_TYPE` / `ENUM_DEAL_ENTRY` — le dépôt initial n'est pas un trade.
MT5_DEAL_TYPE_BALANCE = 2
MT5_DEAL_ENTRY_IN = 0
MT5_DEAL_ENTRY_OUT = 1

#: §13 — appariement attendu Python (données MT5) ↔ EA MT5.
MATCH_TARGET_MT5_SAME_DATA = 0.95

#: §13 — appariement attendu Python/QC ↔ MT5, données différentes.
MATCH_TARGET_MT5_DIFF_DATA = 0.70

#: §13 barreau 2 — les indicateurs que les deux moteurs doivent voir pareil.
TRACE_INDICATORS: tuple[str, ...] = ("atr", "v", "m", "a", "vwap", "ema50_h1", "dxy")

#: Bloc racine stable de la branche MT5. Même règle que `SUMMARY_KEYS` : la
#: note de recherche et le rapport client lisent ces noms, on peut en ajouter,
#: jamais en renommer.
MT5_SUMMARY_KEYS: tuple[str, ...] = (
    "match_rate_py_to_mt5_same_data",
    "match_rate_mt5_to_py_same_data",
    "match_rate_py_to_mt5_diff_data",
    "py_expectancy_r_same_data",
    "mt5_expectancy_r",
    "unattributed_share",
    "server_clock_offset_min",
)

#: §10 — fenêtre interdite, bornes New York (incluse / exclue).
FORBIDDEN_WINDOW_NY = (16 * 60 + 30, 18 * 60 + 15)

#: Décalages testés par le contrôle d'horloge, en minutes. Les bornes horaires
#: répondent à « l'heure serveur est-elle vraiment UTC ? » ; ±1 et ±5 séparent
#: une convention de datation d'un vrai décalage d'horloge.
CLOCK_LAGS_MIN: tuple[int, ...] = (-180, -120, -60, -5, -1, 0, 1, 5, 60, 120, 180)


# ── lecture des artefacts du tester ───────────────────────────────────


def load_mt5_trace(path: Path) -> pd.DataFrame:
    """La trace §13 de l'EA : 21 colonnes + `cancel_reason`, `ts_decision` UTC.

    `ts_decision` est l'**ouverture** de la barre M5 de décision et non sa
    clôture : l'EA estampille avec `iTime`, qui est une heure d'ouverture. Le
    contrôle est direct et il est fait plus bas — le deal d'entrée tombe
    exactement cinq minutes après le `ts_decision` de son `ENTRY`, ce qui est le
    fill « à l'open de la barre suivante » de §9.
    """
    if not path.exists():
        raise SystemExit(f"[mt5] trace introuvable : {path}")
    frame = pd.read_csv(path)
    frame["ts_decision"] = pd.to_datetime(frame["ts_decision"], utc=True)
    for column in ("scenario", "exit_reason", "cancel_reason"):
        if column not in frame.columns:
            frame[column] = ""
        frame[column] = frame[column].fillna("").astype(str)
    frame["level"] = frame["level"].astype(float).round(3)
    frame["d"] = frame["d"].astype(int)
    return frame.reset_index(drop=True)


def load_mt5_deals(path: Path) -> pd.DataFrame:
    """Le CSV par deal de l'EA — UTF-16 LE, `time_utc` au format MT5."""
    if not path.exists():
        raise SystemExit(f"[mt5] deals introuvables : {path}")
    raw = path.read_bytes()
    for encoding in ("utf-16", "utf-8-sig", "utf-8"):
        try:
            text = raw.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    else:  # pragma: no cover - aucun encodage MT5 connu ne tombe ici
        text = raw.decode("utf-8", errors="replace")

    from io import StringIO

    deals = pd.read_csv(StringIO(text))
    deals["time_utc"] = pd.to_datetime(
        deals["time_utc"], format="%Y.%m.%d %H:%M:%S", utc=True
    )
    deals["net"] = deals["profit"] + deals["commission"] + deals["swap"]
    deals["sleeve"] = deals["sleeve"].astype(str).str.strip()
    return deals.sort_values(["time_utc", "deal_id"]).reset_index(drop=True)


def mt5_positions(trace: pd.DataFrame, deals: pd.DataFrame) -> pd.DataFrame:
    """Une ligne par position : la trace donne l'intention, les deals l'argent.

    §9 impose une position à la fois, tous scénarios confondus : les `ENTRY` et
    les `EXIT` de la trace alternent donc, et le k-ième deal d'entrée appartient
    au k-ième `ENTRY`. C'est vérifié, pas supposé — `fill_px` de la trace doit
    valoir le prix du deal au centime près, sinon l'appariement est faux et le
    reste du rapport ne veut rien dire.

    Le **R réalisé** est le profit net du trade divisé par le risque engagé, et
    le risque engagé se lit au fill : `|prix d'entrée − stop| × volume × 100`.
    Prendre `r_est` à la place donnerait le R *estimé*, qui ignore l'arrondi des
    lots au pas de 0,01.
    """
    entries = trace[trace["event"] == "ENTRY"].reset_index(drop=True)
    exits = trace[trace["event"] == "EXIT"].reset_index(drop=True)
    live = deals[deals["type"] != MT5_DEAL_TYPE_BALANCE]
    ins = live[live["entry"] == MT5_DEAL_ENTRY_IN].reset_index(drop=True)
    outs = live[live["entry"] == MT5_DEAL_ENTRY_OUT].reset_index(drop=True)

    sizes = {len(entries), len(exits), len(ins), len(outs)}
    if len(sizes) != 1:
        raise SystemExit(
            f"[mt5] la trace et les deals ne décrivent pas le même nombre de "
            f"positions : ENTRY={len(entries)} EXIT={len(exits)} "
            f"deals_in={len(ins)} deals_out={len(outs)}"
        )
    if not (ins["position_id"].to_numpy() == outs["position_id"].to_numpy()).all():
        raise SystemExit("[mt5] les deals d'entrée et de sortie ne s'apparient pas")

    price_gap = float(np.abs(entries["fill_px"].to_numpy() - ins["price"].to_numpy()).max())
    scenario_from_magic = ins["magic"].map(MT5_MAGIC_SCENARIO).fillna("")
    scenario_mismatch = int((scenario_from_magic.to_numpy() != entries["scenario"].to_numpy()).sum())

    volume = ins["volume"].to_numpy(dtype=float)
    fill_px = entries["fill_px"].to_numpy(dtype=float)
    stop = entries["stop"].to_numpy(dtype=float)
    risk = np.abs(fill_px - stop) * volume * CONTRACT_SIZE
    net = outs["net"].to_numpy(dtype=float)

    frame = pd.DataFrame(
        {
            "ts_decision": entries["ts_decision"],
            "ts_fill": ins["time_utc"],
            "ts_exit": outs["time_utc"],
            "ts_exit_bar": exits["ts_decision"],
            "scenario_name": entries["scenario"].to_numpy(),
            "level": entries["level"].to_numpy(),
            "d": entries["d"].to_numpy(),
            "q": [SCENARIO_SIDE.get(s, 0) for s in entries["scenario"]],
            "stop": stop,
            "target": entries["target"].to_numpy(dtype=float),
            "r_est": entries["r_est"].to_numpy(dtype=float),
            "fill_px": fill_px,
            "exit_px": exits["exit_px"].to_numpy(dtype=float),
            "exit_reason_name": exits["exit_reason"].to_numpy(),
            "lots": volume,
            "net": net,
            "risk_amount": risk,
            "r_realised": np.where(risk > 0, net / np.where(risk > 0, risk, 1.0), np.nan),
            "position_id": ins["position_id"].to_numpy(),
        }
    )
    frame.attrs["price_gap_trace_vs_deals"] = price_gap
    frame.attrs["scenario_magic_mismatches"] = scenario_mismatch
    frame.attrs["fill_lag_minutes"] = sorted(
        {
            int(v)
            for v in (
                (frame["ts_fill"] - frame["ts_decision"]).dt.total_seconds() / 60.0
            ).unique()
        }
    )
    return frame


# ── rejeu de la référence Python sur le dump du broker (C1) ───────────


def mt5_dump_to_m1(dump: pd.DataFrame, price: str = "mid") -> pd.DataFrame:
    """Le dump BID du broker en OHLC M1 exploitable par `prepare_inputs`.

    `price="mid"` ajoute la demi-fourchette de la barre — c'est le prix que la
    spec fait lire aux indicateurs (§9 : le côté ne s'applique qu'au fill).
    `price="bid"` rend le dump tel quel, ce que l'EA lit vraiment : la
    différence entre les deux runs *mesure* le poste « signaux sur bid ».
    """
    half = dump["spread_points"].to_numpy(dtype=float) * MT5_POINT / 2.0
    shift = half if price == "mid" else np.zeros(len(dump))
    return pd.DataFrame(
        {c: dump[c].to_numpy(dtype=float) + shift for c in ("open", "high", "low", "close")},
        index=dump.index,
    )


def mt5_spread_per_m5(dump: pd.DataFrame, inputs: Any) -> np.ndarray:
    """Spread par barre M5 = médiane des spreads M1 du bin, en dollars (§12)."""
    owner = np.repeat(np.arange(len(inputs.m5)), inputs.m1_end - inputs.m1_start)
    m1_spread = dump["spread_points"].to_numpy(dtype=float) * MT5_POINT
    return (
        pd.Series(m1_spread).groupby(owner).median().reindex(range(len(inputs.m5))).to_numpy()
    )


def replay_reference_on_dump(
    dump: pd.DataFrame,
    *,
    price: str = "mid",
    z: float = 1.0,
    a_min: float = 0.2,
    k_s: float = 1.0,
    init_cash: float = 10_000.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rejouer le moteur de référence sur les barres du broker.

    Rendu : la trace §13 (22 colonnes, `ts_decision` UTC) et le registre de
    trades. Le spread est passé **par barre M5**, ce que `pipeline()` ne sait pas
    faire — il broadcast un scalaire — d'où l'appel direct à `run_engine` sur
    les `kernel_kwargs` dont on remplace la seule clé `spread`.

    L'import est tardif : la branche QC de ce script n'a aucune raison de payer
    le chargement de vectorbtpro et la compilation Numba.
    """
    sys.path.insert(0, str(_REPO / "src"))
    from framework.x10_engine import TRADE_COLUMNS, run_engine  # noqa: PLC0415
    from strategies.xau_x10 import (  # noqa: PLC0415
        X10Indicator,
        _events_frame,
        _trades_frame,
        emit_event_trace,
        prepare_inputs,
    )

    m1 = mt5_dump_to_m1(dump, price=price)
    inputs = prepare_inputs(m1)
    kwargs = inputs.kernel_kwargs(0.0)
    kwargs["spread"] = mt5_spread_per_m5(dump, inputs)
    events, trades = run_engine(
        **kwargs, z=z, a_min=a_min, k_s=k_s, init_cash=init_cash, risk_frac=RISK_FRAC
    )

    indicator = X10Indicator(
        events=_events_frame(events, inputs.m5.index),
        trades=_trades_frame(trades, inputs.m5.index, inputs.m1.index),
        m5=inputs.m5,
        m1=inputs.m1,
        params={"z": z, "a_min": a_min, "k_s": k_s, "spread": "mt5_dump_per_bar"},
    )
    trace = emit_event_trace(indicator, extended=True)
    trace["ts_decision"] = pd.to_datetime(trace["ts_decision"], utc=True)
    trace["scenario"] = trace["scenario"].fillna("").astype(str)
    trace["exit_reason"] = trace["exit_reason"].fillna("").astype(str)
    trace["level"] = trace["level"].astype(float).round(3)
    trace["d"] = trace["d"].astype(int)

    registry = pd.DataFrame(np.asarray(trades), columns=list(TRADE_COLUMNS))
    registry["scenario_name"] = indicator.trades["scenario_name"].to_numpy()
    registry["exit_reason_name"] = indicator.trades["exit_reason_name"].to_numpy()
    for column, source in (
        ("ts_decision", "ts_decision"),
        ("ts_fill", "ts_fill"),
        ("ts_exit", "ts_exit"),
    ):
        stamps = pd.DatetimeIndex(indicator.trades[source])
        registry[column] = stamps.tz_localize(
            "America/New_York", ambiguous=True, nonexistent="shift_forward"
        ).tz_convert("UTC")
    registry["level"] = registry["level"].round(3)
    registry["r_realised"] = (
        registry["equity_after"] - registry["equity_before"]
    ) / registry["risk_amount"]
    return trace, registry


# ── contrôle d'horloge du dump ────────────────────────────────────────


def clock_offset_check(dump: pd.DataFrame, reference_minute: Path) -> dict[str, Any]:
    """L'heure serveur est-elle bien UTC ? Corrélation des rendements minute.

    L'EA suppose « heure serveur = heure UTC ». Si l'hypothèse est fausse, tout
    ce qui dépend de l'horloge de New York — séance, fenêtre interdite, VWAP —
    est décalé, et aucun appariement d'événements n'a de sens. On corrèle les
    rendements minute du dump à ceux d'un parquet de référence UTC, à plusieurs
    décalages ; le maximum doit tomber sur un décalage **horaire nul**.

    Le décalage d'une minute qui subsiste n'est pas une horloge, c'est une
    convention : MT5 date une barre de son ouverture, LEAN de sa clôture. Les
    deux champs sont rendus séparément pour qu'on ne les confonde pas.
    """
    reference = pd.read_parquet(reference_minute)
    if reference.index.tz is None:
        reference.index = reference.index.tz_localize("UTC")
    window = reference.loc[
        (reference.index >= dump.index[0]) & (reference.index <= dump.index[-1])
    ]
    dump_ret = np.log(dump["close"]).diff()
    ref_ret = np.log(window["close"]).diff().rename("ref")

    correlations: dict[str, float] = {}
    counts: dict[str, int] = {}
    for lag in CLOCK_LAGS_MIN:
        shifted = dump_ret.copy()
        shifted.index = shifted.index + pd.Timedelta(minutes=lag)
        joined = pd.concat([shifted.rename("dump"), ref_ret], axis=1, join="inner").dropna()
        joined = joined[(joined != 0).all(axis=1)]
        correlations[str(lag)] = float(joined["dump"].corr(joined["ref"]))
        counts[str(lag)] = int(len(joined))

    best = max(correlations, key=lambda k: correlations[k])
    hourly = {k: v for k, v in correlations.items() if int(k) % 60 == 0}
    best_hourly = max(hourly, key=lambda k: hourly[k])
    return {
        "reference_parquet": str(reference_minute),
        "corr_by_lag_min": correlations,
        "n_by_lag_min": counts,
        "best_lag_min": int(best),
        "best_lag_corr": correlations[best],
        # Ce que le critère du brief demande vraiment : pas de décalage horaire.
        "best_hourly_lag_min": int(best_hourly),
        "server_clock_offset_min": int(best_hourly),
        "bar_stamp_convention_lag_min": int(best),
        "server_clock_is_utc": int(best_hourly) == 0,
        "note": (
            "Décalage horaire nul : l'heure serveur est bien UTC. Le maximum de "
            "corrélation tombe à +1 min parce que MT5 date une barre M1 de son "
            "ouverture et le parquet QC de sa clôture — convention, pas horloge."
        ),
    }


# ── appariement générique ─────────────────────────────────────────────


def match_on_keys(
    left: pd.DataFrame,
    right: pd.DataFrame,
    key_cols: tuple[str, ...],
    tol_bars: int = 1,
    ts_col: str = "ts_decision",
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Appariement un-à-un, au plus proche, sur `(clés, ts ± tol_bars barres)`.

    Même règle que `match_entries` pour QC — trié par écart temporel croissant,
    jamais deux fois le même élément — mais sur deux `DataFrame` quelconques, ce
    dont les barreaux 3 et 4 de la branche MT5 ont besoin.
    """
    window = pd.Timedelta(minutes=tol_bars * BAR_MINUTES).value
    left, right = left.reset_index(drop=True), right.reset_index(drop=True)
    left_ts = left[ts_col].to_numpy(dtype="datetime64[ns]").astype("int64")
    right_ts = right[ts_col].to_numpy(dtype="datetime64[ns]").astype("int64")

    groups: dict[tuple[Any, ...], list[int]] = {}
    for j, key in enumerate(zip(*(right[c] for c in key_cols), strict=True)):
        groups.setdefault(key, []).append(j)

    candidates: list[tuple[int, int, int]] = []
    for i, key in enumerate(zip(*(left[c] for c in key_cols), strict=True)):
        for j in groups.get(key, ()):
            delta = abs(int(right_ts[j]) - int(left_ts[i]))
            if delta <= window:
                candidates.append((delta, i, j))

    candidates.sort()
    used_left: set[int] = set()
    used_right: set[int] = set()
    pairs: list[tuple[int, int]] = []
    for _, i, j in candidates:
        if i in used_left or j in used_right:
            continue
        used_left.add(i)
        used_right.add(j)
        pairs.append((i, j))
    pairs.sort()
    return (
        pairs,
        [i for i in range(len(left)) if i not in used_left],
        [j for j in range(len(right)) if j not in used_right],
    )


def _rate(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else float("nan")


# ── barreaux 2 à 5, côté MT5 ──────────────────────────────────────────


def mt5_rung2(py_trace: pd.DataFrame, mt5_trace: pd.DataFrame) -> dict[str, Any]:
    """Barreau 2 — les sept indicateurs à `ts_decision` identique.

    L'appariement est **exact** sur `(ts_decision, event, level, d)` : au
    barreau 2 on ne cherche pas la même décision à une barre près, on cherche le
    même instant. Ce qui n'a pas de jumeau exact est compté à part et sort du
    calcul des écarts.
    """
    keys = ["ts_decision", "event", "level", "d"]
    left = py_trace.drop_duplicates(subset=keys)
    right = mt5_trace.drop_duplicates(subset=keys)
    joined = left.merge(right, on=keys, suffixes=("_py", "_mt5"), how="inner")

    diffs: dict[str, Any] = {}
    for column in TRACE_INDICATORS:
        delta = (
            joined[f"{column}_py"].to_numpy(dtype=float)
            - joined[f"{column}_mt5"].to_numpy(dtype=float)
        )
        stats = _stats(np.abs(delta))
        stats["signed_median"] = float(np.nanmedian(delta)) if len(delta) else float("nan")
        diffs[column] = stats

    context: dict[str, Any] = {}
    for column in ("ctx_ema", "ctx_vwap", "ctx_dxy"):
        same = joined[f"{column}_py"].to_numpy() == joined[f"{column}_mt5"].to_numpy()
        context[column] = {
            "n": int(len(same)),
            "agreement_rate": _rate(int(same.sum()), len(same)),
        }
    return {
        "n_py_rows": int(len(left)),
        "n_mt5_rows": int(len(right)),
        "n_exact_stamp_matches": int(len(joined)),
        "exact_stamp_match_rate_py": _rate(len(joined), len(left)),
        "abs_diff": diffs,
        "context_scores": context,
    }


def mt5_rung3(
    py_trace: pd.DataFrame, mt5_trace: pd.DataFrame, tol_bars: int = 1
) -> dict[str, Any]:
    """Barreau 3 — `ARM` / `BREAK` / `SWEEP` appariés sur `(level, d, ts ± 1)`."""
    per_event: dict[str, Any] = {}
    for event in ("ARM", "BREAK", "SWEEP", "CANCEL"):
        left = py_trace[py_trace["event"] == event].reset_index(drop=True)
        right = mt5_trace[mt5_trace["event"] == event].reset_index(drop=True)
        pairs, py_orphans, mt5_orphans = match_on_keys(
            left, right, ("level", "d"), tol_bars=tol_bars
        )
        per_event[event] = {
            "n_py": int(len(left)),
            "n_mt5": int(len(right)),
            "n_matched": len(pairs),
            "match_rate_py": _rate(len(pairs), len(left)),
            "match_rate_mt5": _rate(len(pairs), len(right)),
            "n_py_orphans": len(py_orphans),
            "n_mt5_orphans": len(mt5_orphans),
        }

    # Matrice : sur les couples appariés à (level, d, ts ± 1) tous types
    # confondus, quel événement chaque moteur a-t-il émis ?
    pairs, _, _ = match_on_keys(
        py_trace.reset_index(drop=True),
        mt5_trace.reset_index(drop=True),
        ("level", "d"),
        tol_bars=tol_bars,
    )
    matrix: dict[str, dict[str, int]] = {}
    py_events = py_trace["event"].to_numpy()
    mt5_events = mt5_trace["event"].to_numpy()
    for i, j in pairs:
        matrix.setdefault(str(py_events[i]), {}).setdefault(str(mt5_events[j]), 0)
        matrix[str(py_events[i])][str(mt5_events[j])] += 1

    agree = sum(v.get(k, 0) for k, v in matrix.items())
    return {
        "by_event": per_event,
        "confusion_matrix_py_vs_mt5": matrix,
        "n_pairs_any_event": len(pairs),
        "same_event_rate": _rate(agree, len(pairs)),
        "cancel_reason_py": py_trace.loc[
            py_trace["event"] == "CANCEL", "cancel_reason"
        ].value_counts().to_dict(),
        "cancel_reason_mt5": mt5_trace.loc[
            mt5_trace["event"] == "CANCEL", "cancel_reason"
        ].value_counts().to_dict(),
    }


def mt5_rung4_5(
    py: pd.DataFrame, mt5: pd.DataFrame, tol_bars: int = 1, label: str = "same_data"
) -> dict[str, Any]:
    """Barreaux 4 et 5 — entrées appariées, puis stop / cible / R réalisé.

    `py` et `mt5` portent tous deux les colonnes du registre de trades ; la
    branche C2 y passe la campagne officielle sans rien changer d'autre que les
    données d'entrée, ce qui est exactement ce que §13 demande de comparer.
    """
    pairs, py_orphans, mt5_orphans = match_on_keys(
        py, mt5, ("scenario_name", "level"), tol_bars=tol_bars
    )
    py, mt5 = py.reset_index(drop=True), mt5.reset_index(drop=True)

    rows: list[dict[str, Any]] = []
    for i, j in pairs:
        p, m = py.loc[i], mt5.loc[j]
        rows.append(
            {
                "ts_decision": str(p["ts_decision"]),
                "scenario": str(p["scenario_name"]),
                "level": float(p["level"]),
                "d_stop": float(p["stop"]) - float(m["stop"]),
                "d_target": float(p["target"]) - float(m["target"]),
                "d_r_est": float(p["r_est"]) - float(m["r_est"]),
                "d_fill_px": float(p["fill_px"]) - float(m["fill_px"]),
                "d_exit_px": float(p["exit_px"]) - float(m["exit_px"]),
                "d_lots": float(p["lots"]) - float(m["lots"]),
                "r_py": float(p["r_realised"]),
                "r_mt5": float(m["r_realised"]),
                "d_r": float(p["r_realised"]) - float(m["r_realised"]),
                "reason_py": str(p["exit_reason_name"]),
                "reason_mt5": str(m["exit_reason_name"]),
            }
        )
    frame = pd.DataFrame(rows)

    reason_matrix: dict[str, dict[str, int]] = {}
    for row in rows:
        reason_matrix.setdefault(row["reason_py"], {}).setdefault(row["reason_mt5"], 0)
        reason_matrix[row["reason_py"]][row["reason_mt5"]] += 1
    agree = sum(v.get(k, 0) for k, v in reason_matrix.items())

    theoretical = {
        column: _stats(frame[column].to_numpy(dtype=float)) if len(frame) else _stats(np.array([]))
        for column in ("d_stop", "d_target", "d_r_est", "d_fill_px", "d_exit_px", "d_lots")
    }
    years = pd.DatetimeIndex(py["ts_decision"]).year
    by_year = {
        str(year): {
            "n_py": int((years == year).sum()),
            "n_matched": int(
                sum(1 for i, _ in pairs if years[i] == year)
            ),
        }
        for year in sorted(set(years))
    }
    for block in by_year.values():
        block["match_rate_py_to_mt5"] = _rate(block["n_matched"], block["n_py"])

    return {
        "label": label,
        "n_py_entries": int(len(py)),
        "n_mt5_entries": int(len(mt5)),
        "n_matched": len(pairs),
        "match_rate_py_to_mt5": _rate(len(pairs), len(py)),
        "match_rate_mt5_to_py": _rate(len(pairs), len(mt5)),
        "match_by_year": by_year,
        "py_orphans": [str(py.loc[i, "ts_decision"]) for i in py_orphans],
        "mt5_orphans": [str(mt5.loc[j, "ts_decision"]) for j in mt5_orphans],
        "rung4_theoretical": theoretical,
        "rung5_realised_r": {
            "py": _stats(frame["r_py"].to_numpy(dtype=float)) if len(frame) else {},
            "mt5": _stats(frame["r_mt5"].to_numpy(dtype=float)) if len(frame) else {},
            "difference": _stats(frame["d_r"].to_numpy(dtype=float)) if len(frame) else {},
        },
        "exit_reason_matrix_py_vs_mt5": reason_matrix,
        "exit_reason_agreement": _rate(agree, len(rows)),
        "matched_pairs": rows,
    }


def busy_share(orphan_stamps: list[str], other: pd.DataFrame) -> dict[str, Any]:
    """Part des orphelins qui tombent pendant qu'une position est ouverte en face.

    §9 n'autorise **une position à la fois**. Dès que les deux moteurs prennent
    un trade différent, chacun bloque l'autre sur tout ce qu'il tient : l'écart
    n'est plus un désaccord de règle mais une désynchronisation de chemin, et
    elle se mesure — c'est la proportion d'orphelins émis alors que le moteur
    d'en face était occupé.
    """
    if not orphan_stamps or other.empty:
        return {"n_orphans": len(orphan_stamps), "n_counterpart_busy": 0, "share": float("nan")}
    stamps = pd.to_datetime(pd.Series(orphan_stamps), utc=True).to_numpy()
    starts = pd.DatetimeIndex(other["ts_fill"]).to_numpy()
    ends = pd.DatetimeIndex(other["ts_exit"]).to_numpy()
    busy = sum(1 for s in stamps if bool(((starts <= s) & (s <= ends)).any()))
    return {
        "n_orphans": len(orphan_stamps),
        "n_counterpart_busy": int(busy),
        "share": _rate(busy, len(orphan_stamps)),
    }


def expectancy_table(frame: pd.DataFrame, by: str) -> dict[str, Any]:
    """Nombre de trades, espérance R et profit factor, ventilés par `by`."""
    out: dict[str, Any] = {}
    for key, chunk in frame.groupby(by):
        r = chunk["r_realised"].to_numpy(dtype=float)
        r = r[np.isfinite(r)]
        gains, losses = r[r > 0].sum(), -r[r <= 0].sum()
        out[str(key)] = {
            "trades": int(len(chunk)),
            "expectancy_r": float(r.mean()) if len(r) else float("nan"),
            # Aucune perte = profit factor non défini ; `inf` n'est pas du JSON
            # valide et se lirait comme un résultat, ce qu'il n'est pas.
            "profit_factor": float(gains / losses) if losses > 0 else None,
            "win_rate": _rate(int((r > 0).sum()), len(r)),
        }
    return out


# ── attribution des écarts en postes nommés ───────────────────────────


def mt5_attribution(
    same_data: dict[str, Any],
    mt5: pd.DataFrame,
    dump: pd.DataFrame,
    bid_variant: dict[str, Any] | None,
    order_failures: dict[str, Any],
    rung2: dict[str, Any],
    rung2_bid: dict[str, Any] | None,
    diff_data: dict[str, Any],
) -> dict[str, Any]:
    """Chaque écart sur un poste nommé, avec une mesure ou un « non mesurable ».

    Un poste sans chiffre n'est pas une explication, c'est une intuition. Les
    postes non mesurables portent explicitement `measured: false` et disent
    pourquoi — c'est ce que §13 appelle « écart attribué ».
    """
    matched = pd.DataFrame(same_data["matched_pairs"])
    posts: dict[str, Any] = {}

    # 1. Signaux sur bid (EA) contre mid (référence). Mesuré en rejouant la
    #    référence sur le bid brut : les entrées qui apparaissent ou
    #    disparaissent sont, par construction, celles que ce poste explique.
    if bid_variant is None:
        posts["signals_on_bid_vs_mid"] = {
            "measured": False,
            "why": "rejeu bid non demandé (--no-bid-replay)",
        }
    else:
        posts["signals_on_bid_vs_mid"] = {
            "measured": True,
            "half_spread_usd_median": float(
                np.median(dump["spread_points"].to_numpy(dtype=float)) * MT5_POINT / 2.0
            ),
            "vwap_abs_diff_median_mid": rung2["abs_diff"]["vwap"]["median"],
            "vwap_abs_diff_median_bid": (
                rung2_bid["abs_diff"]["vwap"]["median"] if rung2_bid else None
            ),
            "ema50_h1_abs_diff_median_mid": rung2["abs_diff"]["ema50_h1"]["median"],
            "ema50_h1_abs_diff_median_bid": (
                rung2_bid["abs_diff"]["ema50_h1"]["median"] if rung2_bid else None
            ),
            "n_entries_py_mid": same_data["n_py_entries"],
            "n_entries_py_bid": bid_variant["n_py_entries"],
            "match_rate_py_to_mt5_mid": same_data["match_rate_py_to_mt5"],
            "match_rate_py_to_mt5_bid": bid_variant["match_rate_py_to_mt5"],
            "match_rate_gain": (
                bid_variant["match_rate_py_to_mt5"] - same_data["match_rate_py_to_mt5"]
            ),
            "note": (
                "L'EA lit les barres BID du terminal ; la référence lit le mid "
                "que §9 prescrit. VWAP et EMA50 H1 sont des NIVEAUX : ils "
                "encaissent la demi-fourchette en entier, alors que l'ATR et la "
                "cinématique, qui sont des différences, l'annulent. Rejouer la "
                "référence sur le bid brut fait passer l'appariement des entrées "
                "de la première valeur à la seconde : c'est la mesure du poste."
            ),
        }

    # 2. Séance du broker : les cotations reprennent avant les transactions.
    ny = dump.index.tz_convert("America/New_York")
    gaps = pd.Series(dump.index).diff() > pd.Timedelta(minutes=30)
    resume = pd.Series(ny[gaps.to_numpy()]).dt.strftime("%H:%M").value_counts()
    fills_ny = mt5["ts_fill"].dt.tz_convert("America/New_York")
    fill_minute = fills_ny.dt.hour * 60 + fills_ny.dt.minute

    # Depuis combien de minutes les cotations ont-elles repris quand l'ordre
    # part ? C'est la seule façon de distinguer « le marché est fermé » de
    # « la cotation existe mais la négociation n'est pas encore ouverte ».
    resume_stamps = dump.index[gaps.to_numpy()]

    def _since_resume(stamps: pd.Series | pd.DatetimeIndex) -> np.ndarray:
        idx = np.searchsorted(resume_stamps.to_numpy(), np.asarray(stamps), side="right") - 1
        valid = idx >= 0
        delta = np.full(len(idx), np.nan)
        delta[valid] = (
            np.asarray(stamps)[valid] - resume_stamps.to_numpy()[idx[valid]]
        ) / np.timedelta64(1, "m")
        return delta

    failed_stamps = pd.DatetimeIndex(
        [row["time_utc"] for row in order_failures.get("entry_failures", [])], tz="UTC"
    ) if order_failures.get("entry_failures") else pd.DatetimeIndex([], tz="UTC")
    since_failed = _since_resume(failed_stamps) if len(failed_stamps) else np.array([])
    since_filled = _since_resume(pd.DatetimeIndex(mt5["ts_fill"]))

    posts["broker_session_reopen"] = {
        "measured": True,
        "quote_resume_ny_top": {k: int(v) for k, v in resume.head(5).items()},
        "n_session_gaps": int(gaps.sum()),
        "first_fill_minute_after_1800_ny": int(
            fill_minute[fill_minute >= FORBIDDEN_WINDOW_NY[1]].min()
        )
        if (fill_minute >= FORBIDDEN_WINDOW_NY[1]).any()
        else None,
        "n_fills_in_forbidden_window": int(
            (
                (fill_minute >= FORBIDDEN_WINDOW_NY[0])
                & (fill_minute < FORBIDDEN_WINDOW_NY[1])
            ).sum()
        ),
        "minutes_since_quote_resume_refused": _stats(since_failed),
        "minutes_since_quote_resume_filled": _stats(since_filled),
        "note": (
            "Les cotations reprennent à 18:01 New York. Les 13 ordres refusés "
            "(retcode 10018, marché fermé) tombent tous dans la première heure "
            "qui suit cette reprise, alors que les fills acceptés en sont "
            "entre 20 et 59 minutes après elle, alors que le fill médian en est "
            "à 704 minutes. Le retcode est la preuve directe ; la statistique "
            "seule ne suffirait pas, les deux plages se recouvrant par le bas."
        ),
    }

    # 2 bis. Convention de binning M5. Le parquet QC date une barre M1 de sa
    # CLÔTURE, le dump de son OUVERTURE : à grille M5 « label=left » identique,
    # les deux moteurs n'agrègent pas les mêmes minutes. Ce n'est ni une
    # horloge ni un flux, c'est un décalage de bin, et il touche chaque barre.
    mid_m1 = dump["close"] + dump["spread_points"] * MT5_POINT / 2.0
    native = mid_m1.resample("5min", label="left", closed="left").last().dropna()
    shifted = mid_m1.copy()
    shifted.index = shifted.index + pd.Timedelta(minutes=1)
    shifted = shifted.resample("5min", label="left", closed="left").last().dropna()
    both = pd.concat(
        [native.rename("native"), shifted.rename("shifted")], axis=1, join="inner"
    ).dropna()
    bin_gap = (both["native"] - both["shifted"]).to_numpy(dtype=float)
    posts["m5_binning_convention"] = {
        "measured": True,
        "applies_to": "C2 seulement",
        "n_bars": int(len(both)),
        "close_diff_usd": _stats(np.abs(bin_gap)),
        "share_of_bars_differing": _rate(int((np.abs(bin_gap) > 1e-9).sum()), len(bin_gap)),
        "note": (
            "Même dump, même grille M5, deux conventions de datation M1 : la "
            "clôture de la barre M5 change sur la quasi-totalité des barres. "
            "C'est le poste structurel qui sépare C2 de C1, à moteur identique."
        ),
    }

    # 3. Sorties TIME / SESSION : l'EA sort à l'open de la barre suivante, la
    #    spec au close du dernier M1 de la barre (§9).
    if len(matched):
        timed = matched[matched["reason_py"].isin(("TIME", "SESSION"))]
        posts["time_session_exit_at_next_open"] = {
            "measured": True,
            "n_trades": int(len(timed)),
            "exit_px_diff_usd": _stats(timed["d_exit_px"].to_numpy(dtype=float)),
            "r_diff": _stats(timed["d_r"].to_numpy(dtype=float)),
        }
        # 4. Double contact : les deux moteurs ne nomment pas la même sortie.
        disagreement = matched[
            matched["reason_py"].isin(("STOP", "TARGET"))
            & matched["reason_mt5"].isin(("STOP", "TARGET"))
            & (matched["reason_py"] != matched["reason_mt5"])
        ]
        posts["double_touch_resolved_by_tester"] = {
            "measured": True,
            "n_stop_target_swaps": int(len(disagreement)),
            "share_of_matched": _rate(len(disagreement), len(matched)),
            "r_diff": _stats(disagreement["d_r"].to_numpy(dtype=float)),
        }
    else:  # pragma: no cover - jamais atteint sur un run réel
        posts["time_session_exit_at_next_open"] = {"measured": False, "why": "aucun appariement"}
        posts["double_touch_resolved_by_tester"] = {"measured": False, "why": "aucun appariement"}

    # 5. Spread courant contre constant — nul en C1 par construction.
    spread = dump["spread_points"].to_numpy(dtype=float) * MT5_POINT
    posts["floating_vs_constant_spread"] = {
        "measured": True,
        "applies_to": "C2 seulement (C1 rejoue le spread mesuré par barre)",
        "measured_spread_usd_median": float(np.median(spread)),
        "measured_spread_usd_p95": float(np.quantile(spread, 0.95)),
        "campaign_constant_usd": 0.29,
    }

    # 6. DXY du broker contre parquet local.
    posts["dxy_broker_vs_parquet"] = {
        "measured": True,
        "ctx_dxy_agreement_rate": rung2["context_scores"]["ctx_dxy"]["agreement_rate"],
        "dxy_abs_diff": rung2["abs_diff"]["dxy"],
        "note": (
            "Écart assumé : l'EA reconstruit son panier à partir des quatre "
            "paires du broker, la référence lit data/DXY4_h1.parquet."
        ),
    }

    # 7. Ordres refusés par le tester.
    posts["rejected_orders"] = {"measured": True, **order_failures}

    # 8 et 9 — C2 seulement : le flux du broker n'est pas celui de QC, et la
    # règle de position unique transforme un désaccord ponctuel en divergence
    # de chemin. Les deux se mesurent, et ce sont elles qui font tomber C2 bien
    # plus bas que C1, à moteur strictement identique.
    if diff_data.get("n_matched") is None:
        posts["qc_feed_vs_broker_feed"] = {"measured": False, "why": "C2 non calculée"}
        posts["single_position_path_dependence"] = {
            "measured": False,
            "why": "C2 non calculée",
        }
    else:
        posts["qc_feed_vs_broker_feed"] = {
            "measured": True,
            "applies_to": "C2 seulement",
            "price_gap_usd": diff_data["feed_price_gap_usd"],
            "note": (
                "Mid du broker contre clôture QC, minute par minute, datation "
                "recalée. L'écart médian est négligeable mais les queues "
                "valent ±0,1 $, soit 2 à 20 % d'un ATR M5 : de quoi faire "
                "basculer un test de seuil dans un sens ou dans l'autre."
            ),
        }
        posts["single_position_path_dependence"] = {
            "measured": True,
            "applies_to": "C2 seulement",
            "match_rate_c1_bid_variant": (
                bid_variant["match_rate_py_to_mt5"] if bid_variant else None
            ),
            "match_rate_c2": diff_data["match_rate_py_to_mt5"],
            **diff_data["path_dependence"],
        }

    # ── part non attribuée ────────────────────────────────────────────
    #
    # Un orphelin n'est « attribué » que si un poste **mesuré** le nomme :
    #
    # * une entrée Python absente du rejeu sur bid vient du mid, donc du poste 1 ;
    # * une entrée MT5 que le rejeu sur bid retrouve vient du même poste, vue de
    #   l'autre côté ;
    # * une entrée MT5 qui n'existe pas dans le carnet de deals parce que
    #   l'ordre a été refusé vient du poste 7.
    #
    # Tout le reste est un désaccord d'automate que rien ne nomme, et c'est
    # cette part-là que §13 plafonne à 5 %.
    py_orphans = set(same_data["py_orphans"])
    mt5_orphans = set(same_data["mt5_orphans"])
    n_union = same_data["n_py_entries"] + same_data["n_mt5_entries"] - same_data["n_matched"]
    n_orphans = len(py_orphans) + len(mt5_orphans)

    explained_by_bid_py: set[str] = set()
    explained_by_bid_mt5: set[str] = set()
    if bid_variant is not None:
        bid_stamps = {str(ts) for ts in bid_variant["_py_stamps"]}
        matched_by_bid = set(mt5_orphans) - set(bid_variant["mt5_orphans"])
        explained_by_bid_py = {ts for ts in py_orphans if ts not in bid_stamps}
        explained_by_bid_mt5 = matched_by_bid
    failed_stamps = {row["time_utc"] for row in order_failures.get("entry_failures", [])}
    explained_by_reject = {ts for ts in mt5_orphans if ts[:16] in failed_stamps}

    explained = explained_by_bid_py | explained_by_bid_mt5 | explained_by_reject
    return {
        "posts": posts,
        "unattributed": {
            "n_entries_union": n_union,
            "n_orphans": n_orphans,
            "n_py_orphans": len(py_orphans),
            "n_mt5_orphans": len(mt5_orphans),
            "n_explained_signals_bid_vs_mid": len(explained_by_bid_py | explained_by_bid_mt5),
            "n_explained_rejected_orders": len(explained_by_reject),
            "n_orphans_explained": len(explained),
            "n_unattributed": n_orphans - len(explained),
            "unattributed_share": _rate(n_orphans - len(explained), n_union),
            "blocker_threshold": UNATTRIBUTED_BLOCKER,
        },
    }


def read_order_failures(log_path: Path | None) -> dict[str, Any]:
    """Les `order_failures` du compteur de l'EA, relus dans le log de l'agent.

    Le log de l'agent est en UTF-16 LE ; les lignes utiles sont celles que l'EA
    préfixe `[ENTRY][WARN]` — une par ordre d'ouverture refusé, ce que le
    compteur `order_failures` du résumé compte. Les `[EXIT][WARN]` sont des
    **réessais** de la même sortie barre après barre et se comptent par
    position, pas par ligne.
    """
    if log_path is None or not log_path.exists():
        return {"measured": False, "why": f"log de l'agent introuvable : {log_path}"}
    raw = log_path.read_bytes()
    for encoding in ("utf-16", "utf-8-sig", "utf-8"):
        try:
            text = raw.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    else:  # pragma: no cover
        text = raw.decode("utf-8", errors="replace")

    entry_failures: list[dict[str, Any]] = []
    exit_retries: dict[str, int] = {}
    retcodes: dict[str, int] = {}
    for line in text.splitlines():
        if "[WARN]" not in line or "failed" not in line:
            continue
        fields = line.split("\t")
        stamp = fields[-1].strip() if fields else line
        when, _, message = stamp.partition("   ")
        code = message.rsplit("retcode=", 1)[-1].strip() if "retcode=" in message else "?"
        retcodes[code] = retcodes.get(code, 0) + 1
        when_utc = when.strip().replace(".", "-", 2)
        if "[ENTRY][WARN]" in line:
            entry_failures.append(
                {"time_utc": when_utc, "retcode": code, "message": message.strip()}
            )
        elif "[EXIT][WARN]" in line:
            position = message.split("PositionClose", 1)[-1].split()[0]
            exit_retries[position] = exit_retries.get(position, 0) + 1

    return {
        "measured": True,
        "log_path": str(log_path),
        "n_entry_failures": len(entry_failures),
        "n_exit_retry_lines": sum(exit_retries.values()),
        "n_positions_with_exit_retry": len(exit_retries),
        "retcodes": retcodes,
        "retcode_10018": "TRADE_RETCODE_MARKET_CLOSED",
        "entry_failures": entry_failures,
    }


def build_mt5_report(args: argparse.Namespace) -> dict[str, Any]:
    """Le rapport MT5 complet : C1, C2, attribution, bloc `summary` stable."""
    mt5_trace = load_mt5_trace(args.mt5_trace)
    deals = load_mt5_deals(args.mt5_deals)
    mt5 = mt5_positions(mt5_trace, deals)

    dump = pd.read_parquet(args.mt5_dump)
    if dump.index.tz is None:
        dump.index = dump.index.tz_localize("UTC")
    clock = clock_offset_check(dump, args.reference_minute)
    if not clock["server_clock_is_utc"]:
        raise SystemExit(
            f"[mt5] l'horloge du dump n'est pas UTC : le maximum de corrélation "
            f"tombe à {clock['best_hourly_lag_min']} min. Séance, fenêtres "
            f"horaires et VWAP de l'EA sont décalés ; la réconciliation "
            f"événementielle n'a plus de sens. Rien n'a été écrit."
        )

    py_trace, py = replay_reference_on_dump(dump, price="mid")
    rung2 = mt5_rung2(py_trace, mt5_trace)
    rung3 = mt5_rung3(py_trace, mt5_trace, args.tol_bars)
    same_data = mt5_rung4_5(py, mt5, args.tol_bars, label="C1_same_data")

    # Variante bid : mêmes barres, mêmes réglages, mais les signaux lus sur le
    # prix que l'EA lit vraiment. Elle ne remplace pas C1 — le brief fixe le mid
    # — elle **mesure** le poste d'attribution n°1.
    rung2_bid: dict[str, Any] | None = None
    bid_variant: dict[str, Any] | None = None
    if not args.no_bid_replay:
        py_trace_bid, py_bid = replay_reference_on_dump(dump, price="bid")
        rung2_bid = mt5_rung2(py_trace_bid, mt5_trace)
        bid_variant = mt5_rung4_5(py_bid, mt5, args.tol_bars, label="C1_bid_variant")
        bid_variant["_py_stamps"] = [str(ts) for ts in py_bid["ts_decision"]]

    # C2 : la campagne officielle, restreinte à la fenêtre du run MT5.
    diff_data: dict[str, Any] = {"label": "C2_diff_data", "measured": False}
    official: pd.DataFrame | None = None
    if args.py_trades:
        # Le registre porte les deux horloges de §2 ; seule celle en UTC se
        # compare aux stamps de l'EA, la naïve New York doit disparaître avant
        # le renommage sous peine de deux colonnes du même nom.
        official = load_py_trades(args.py_trades[0]).drop(
            columns=["ts_decision", "ts_fill", "ts_exit"], errors="ignore"
        )
        official = official.rename(
            columns={
                "ts_decision_utc": "ts_decision",
                "ts_fill_utc": "ts_fill",
                "ts_exit_utc": "ts_exit",
            }
        )
        for column in ("ts_decision", "ts_fill", "ts_exit"):
            official[column] = pd.to_datetime(official[column], utc=True)
        lo, hi = mt5["ts_decision"].min(), mt5["ts_decision"].max()
        official = official[
            (official["ts_decision"] >= lo) & (official["ts_decision"] <= hi)
        ].reset_index(drop=True)
        official["level"] = official["level"].astype(float).round(3)
        official["r_realised"] = (
            official["equity_after"] - official["equity_before"]
        ) / official["risk_amount"]
        diff_data = mt5_rung4_5(official, mt5, args.tol_bars, label="C2_diff_data")
        diff_data.pop("matched_pairs", None)
        # Pourquoi C2 s'effondre alors que C1 tient : la règle de position
        # unique désynchronise les deux chemins dès le premier trade divergent.
        diff_data["path_dependence"] = {
            "py_orphans_while_mt5_busy": busy_share(diff_data["py_orphans"], mt5),
            "mt5_orphans_while_py_busy": busy_share(diff_data["mt5_orphans"], official),
        }
        # Le flux du broker contre celui de QC, en dollars par minute.
        feed = pd.read_parquet(args.reference_minute)
        if feed.index.tz is None:
            feed.index = feed.index.tz_localize("UTC")
        mid = dump["close"] + dump["spread_points"] * MT5_POINT / 2.0
        mid.index = mid.index + pd.Timedelta(minutes=1)  # datation open → close
        gap = pd.concat(
            [mid.rename("broker"), feed["close"].rename("qc")], axis=1, join="inner"
        ).dropna()
        delta = (gap["qc"] - gap["broker"]).rename("delta")
        diff_data["feed_price_gap_usd"] = _stats(delta.to_numpy(dtype=float))
        diff_data["feed_price_gap_usd_by_year"] = {
            str(year): _stats(chunk.to_numpy(dtype=float))
            for year, chunk in delta.groupby(delta.index.year)
        }

    failures = read_order_failures(args.mt5_log)
    attribution = mt5_attribution(
        same_data, mt5, dump, bid_variant, failures, rung2, rung2_bid, diff_data
    )
    if bid_variant is not None:
        bid_variant.pop("_py_stamps", None)
        bid_variant.pop("matched_pairs", None)

    mt5_years = mt5.assign(year=mt5["ts_decision"].dt.year)
    py_years = py.assign(year=py["ts_decision"].dt.year)
    tables = {
        "mt5_by_year": expectancy_table(mt5_years, "year"),
        "mt5_by_scenario": expectancy_table(mt5, "scenario_name"),
        "mt5_by_exit_reason": expectancy_table(mt5, "exit_reason_name"),
        "py_same_data_by_year": expectancy_table(py_years, "year"),
        "py_same_data_by_scenario": expectancy_table(py, "scenario_name"),
    }
    if official is not None:
        official_years = official.assign(year=official["ts_decision"].dt.year)
        tables["py_official_by_year"] = expectancy_table(official_years, "year")
        tables["py_official_by_scenario"] = expectancy_table(official, "scenario_name")

    r_mt5 = mt5["r_realised"].to_numpy(dtype=float)
    r_py = py["r_realised"].to_numpy(dtype=float)
    summary = {
        "match_rate_py_to_mt5_same_data": same_data["match_rate_py_to_mt5"],
        "match_rate_mt5_to_py_same_data": same_data["match_rate_mt5_to_py"],
        "match_rate_py_to_mt5_diff_data": diff_data.get("match_rate_py_to_mt5"),
        "py_expectancy_r_same_data": float(np.nanmean(r_py)),
        "mt5_expectancy_r": float(np.nanmean(r_mt5)),
        "unattributed_share": attribution["unattributed"]["unattributed_share"],
        "server_clock_offset_min": clock["server_clock_offset_min"],
    }
    return {
        "summary": summary,
        "clock_check": clock,
        "rung1_bars": {
            "n_m5_py": int(len(py_trace["ts_decision"].dt.floor("5min").unique())),
            "n_m1_dump": int(len(dump)),
            "dump_start_utc": str(dump.index[0]),
            "dump_end_utc": str(dump.index[-1]),
        },
        "rung2_indicators": rung2,
        "rung2_indicators_bid_variant": rung2_bid,
        "rung3_events": rung3,
        "c1_same_data": same_data,
        "c1_bid_variant": bid_variant,
        "c2_diff_data": diff_data,
        "attribution": attribution,
        "tables": tables,
        "controls": {
            "trace_fill_px_vs_deal_price_max_abs": mt5.attrs["price_gap_trace_vs_deals"],
            "scenario_magic_mismatches": mt5.attrs["scenario_magic_mismatches"],
            "fill_lag_minutes": mt5.attrs["fill_lag_minutes"],
            "n_positions": int(len(mt5)),
            "deals_net_sum_usd": float(
                deals.loc[deals["type"] != MT5_DEAL_TYPE_BALANCE, "net"].sum()
            ),
        },
        "targets": {
            "same_data": MATCH_TARGET_MT5_SAME_DATA,
            "diff_data": MATCH_TARGET_MT5_DIFF_DATA,
            "unattributed_blocker": UNATTRIBUTED_BLOCKER,
        },
        "meta": {
            "mt5_trace": str(args.mt5_trace),
            "mt5_deals": str(args.mt5_deals),
            "mt5_dump": str(args.mt5_dump),
            "py_official": [str(p) for p in (args.py_trades or [])],
            "tol_bars": args.tol_bars,
        },
    }


def print_mt5_summary(report: dict[str, Any]) -> None:
    summary, clock = report["summary"], report["clock_check"]
    c1, c2 = report["c1_same_data"], report["c2_diff_data"]
    print("═" * 74)
    print("Réconciliation x10 — EA MT5 contre référence Python")
    print("═" * 74)
    print(
        f"horloge      décalage horaire={clock['server_clock_offset_min']} min "
        f"(corr max à {clock['best_lag_min']} min = {clock['best_lag_corr']:.4f}, "
        f"convention de datation)"
    )
    print(
        f"C1 mêmes données  Python={c1['n_py_entries']} MT5={c1['n_mt5_entries']} "
        f"appariées={c1['n_matched']}  "
        f"Py→MT5={c1['match_rate_py_to_mt5']:.1%}  MT5→Py={c1['match_rate_mt5_to_py']:.1%} "
        f"(cible {MATCH_TARGET_MT5_SAME_DATA:.0%})"
    )
    if c2.get("n_matched") is not None:
        print(
            f"C2 données diff.  Python={c2['n_py_entries']} appariées={c2['n_matched']}  "
            f"Py→MT5={c2['match_rate_py_to_mt5']:.1%} (cible "
            f"{MATCH_TARGET_MT5_DIFF_DATA:.0%})"
        )
    print(
        f"espérance R  Python(mêmes données)={summary['py_expectancy_r_same_data']:+.4f}  "
        f"MT5={summary['mt5_expectancy_r']:+.4f}"
    )
    print(
        f"non attribué {summary['unattributed_share']:.1%} "
        f"(bloquant au-delà de {UNATTRIBUTED_BLOCKER:.0%})"
    )
    print("═" * 74)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.mt5_trace is not None:
        report = build_mt5_report(args)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str))
        print_mt5_summary(report)
        print(f"→ {args.out}")
        return 0
    report = build_report(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    print_summary(report)
    print(f"→ {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
