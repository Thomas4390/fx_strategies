#!/usr/bin/env python3
"""Figer les trades de la sleeve or d'un run MT5 dans le dépôt.

Le rapport d'analyse des trades or (``main_gold_trades.tex``) descend au trade
individuel. Ses deux sources vivent **hors du dépôt**, dans l'arborescence MT5,
et sont écrasées au run suivant :

* le CSV par deal écrit par l'EA quand ``Inp_ExportDeals=true`` — il porte le
  ``magic``, donc la sleeve, et surtout le **swap**, poste de coût majeur sur
  les tenues longues (jusqu'à -2 874 sur un seul trade) ;
* le journal du Strategy Tester, seul endroit où figurent le **score** de
  décision et le **levier** appliqués à chaque entrée. C'est ce que
  ``reports/investigations/vbt_vs_mt5_gold_parity.md`` désignait comme « la voie
  la plus courte vers l'attribution poste par poste », la trace
  ``gold_trace.csv`` n'ayant jamais été produite (``WriteTraceRow`` n'est jamais
  atteint).

Ce script apparie les deux et écrit deux artefacts versionnés, pour que le
rapport reste reproductible quand MT5 aura tourné à nouveau.

Le journal mélange **tous les runs de la journée** (sweep compris) : les entrées
d'un même run sont contiguës et leurs dates croissent, donc on découpe le flux à
chaque recul de date, puis on retient le run dont les dates d'entrée coïncident
exactement avec celles du CSV. Aucune heuristique sur l'heure du log.

Usage:
    python scripts/extract_gold_trades.py
    python scripts/extract_gold_trades.py --deals <chemin.csv> --log <chemin.log>
    python scripts/extract_gold_trades.py --expect 35
"""

from __future__ import annotations

import argparse
import codecs
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

# Le parser du rapport MT5 porte déjà les corrections qui rendent les agrégats
# justes (dépôt initial exclu, liquidations de fin de test rendues à leur
# sleeve). On le réutilise plutôt que de relire le CSV à côté.
from parse_mt5_report import (  # noqa: E402
    FILE_COMMON,
    _read_utf16_safe,
    load_deals,
)

MT5_REPORTS = _PROJECT_ROOT / "reports/mt5"
MT5_LOG_DIR = Path.home() / (
    ".mt5/drive_c/Program Files/MetaTrader 5/Tester/logs"
)
OUT_CSV = MT5_REPORTS / "gold_trades_production.csv"
OUT_JSON = MT5_REPORTS / "gold_trades_production.json"

SLEEVE = "GOLD_MOMENTUM"

# La sleeve décide à la clôture de séance — 17:00 New York, soit 21:00 UTC en
# heure d'été. Toute sortie qui tombe ailleurs n'a pas été décidée par le
# signal : c'est le stop de sécurité (Inp_Gold_SafetySL) qui a coupé.
SESSION_EXIT_HOUR_UTC = 21

_ENTRY_RE = re.compile(
    r"(?P<dt>\d{4}\.\d{2}\.\d{2} \d{2}:\d{2}:\d{2})\s+"
    r"\[Gold_Momentum\]\[INFO\] Entry (?P<direction>LONG|SHORT) (?P<symbol>\S+) "
    r"lots=(?P<lots>[\d.]+) price=(?P<price>[\d.]+) "
    r"score=(?P<score>-?[\d.]+) lev=(?P<lev>[\d.]+)"
)
_EXIT_RE = re.compile(
    r"(?P<dt>\d{4}\.\d{2}\.\d{2} \d{2}:\d{2}:\d{2})\s+"
    r"\[Gold_Momentum\]\[INFO\] Exit (?P<direction>LONG|SHORT) (?P<symbol>\S+) "
    r"\(score=(?P<score>-?[\d.]+)\)"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# Côté deals : apparier les aller-retours
# ---------------------------------------------------------------------------


def pair_gold_trades(deals: pd.DataFrame) -> pd.DataFrame:
    """Recomposer les positions or à partir des deals, par ``position_id``.

    MT5 écrit deux deals par position : l'ouverture (``entry == 0``) porte le
    prix d'entrée, la sortie (``entry == 1``) porte le résultat, le swap et la
    commission. Le rapprochement se fait sur ``position_id``, jamais sur
    l'ordre des lignes.
    """
    gold = deals[(deals["sleeve"] == SLEEVE) & (~deals["is_balance_op"])]
    if gold.empty:
        raise SystemExit(
            f"Aucun deal '{SLEEVE}' dans ce CSV. Le run a-t-il été joué avec "
            "Inp_AllocGoldMomentum > 0 et un EA exportant la sleeve or ?"
        )

    opens = gold[gold["entry"] == 0].set_index("position_id")
    closes = gold[gold["entry"] == 1].set_index("position_id")

    unmatched = opens.index.symmetric_difference(closes.index)
    if len(unmatched) > 0:
        raise SystemExit(
            f"{len(unmatched)} position(s) or sans aller-retour complet : "
            f"{sorted(unmatched)[:10]}. Le CSV est tronqué ou le run est "
            "encore en cours."
        )

    trades = pd.DataFrame(
        {
            "position_id": opens.index,
            "entry_time": opens["time_utc"].to_numpy(),
            "exit_time": closes.loc[opens.index, "time_utc"].to_numpy(),
            "direction": [
                "LONG" if t == 0 else "SHORT" for t in opens["type"].to_numpy()
            ],
            "lots": opens["volume"].to_numpy(),
            "entry_price": opens["price"].to_numpy(),
            "exit_price": closes.loc[opens.index, "price"].to_numpy(),
            "profit": closes.loc[opens.index, "profit"].to_numpy(),
            "commission": closes.loc[opens.index, "commission"].to_numpy(),
            "swap": closes.loc[opens.index, "swap"].to_numpy(),
            "net": closes.loc[opens.index, "net"].to_numpy(),
            "forced_close": closes.loc[opens.index, "forced_close"].to_numpy(),
        }
    ).sort_values("entry_time", ignore_index=True)

    trades["duration_days"] = (
        trades["exit_time"] - trades["entry_time"]
    ).dt.total_seconds() / 86400.0
    # Une sortie hors de la borne de séance n'a pas été décidée par le signal.
    trades["safety_stop"] = trades["exit_time"].dt.hour != SESSION_EXIT_HOUR_UTC
    return trades


# ---------------------------------------------------------------------------
# Côté journal : retrouver le run, en extraire score et levier
# ---------------------------------------------------------------------------


def parse_log_entries(log_path: Path) -> list[pd.DataFrame]:
    """Découper les entrées ``Gold_Momentum`` du journal en runs successifs.

    Le journal est en UTF-16 et pèse des dizaines de Mo : il est lu en flux, pas
    chargé en mémoire. Les dates de trade croissent à l'intérieur d'un run, donc
    tout recul marque le début d'un nouveau run.
    """
    runs: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    previous_dt: datetime | None = None

    with codecs.open(log_path, "r", encoding="utf-16", errors="replace") as handle:
        for line in handle:
            if "[Gold_Momentum]" not in line:
                continue
            match = _ENTRY_RE.search(line)
            if match is None:
                continue
            dt = datetime.strptime(match.group("dt"), "%Y.%m.%d %H:%M:%S")
            if previous_dt is not None and dt < previous_dt:
                runs.append(current)
                current = []
            current.append(
                {
                    "entry_time": dt,
                    "direction": match.group("direction"),
                    "log_lots": float(match.group("lots")),
                    "log_price": float(match.group("price")),
                    "score": float(match.group("score")),
                    "leverage": float(match.group("lev")),
                }
            )
            previous_dt = dt

    if current:
        runs.append(current)
    return [pd.DataFrame(run) for run in runs if run]


def select_matching_run(
    runs: list[pd.DataFrame], entry_times: pd.Series
) -> pd.DataFrame:
    """Retenir le run dont les dates d'entrée coïncident avec celles du CSV."""
    wanted = list(pd.to_datetime(entry_times))
    for run in runs:
        if list(run["entry_time"]) == wanted:
            return run

    sizes = sorted({len(run) for run in runs})
    raise SystemExit(
        f"Aucun run du journal ne correspond aux {len(wanted)} entrées du CSV "
        f"(runs trouvés : {len(runs)}, tailles {sizes}).\n"
        "Le journal du jour ne contient probablement pas le run qui a produit "
        "ce CSV — vérifier --log."
    )


# ---------------------------------------------------------------------------
# Sélection des sources
# ---------------------------------------------------------------------------


def pick_deals_csv(explicit: Path | None) -> Path:
    """Choisir le CSV de deals à lire.

    Par défaut : parmi les exports du dossier commun MT5, celui qui contient de
    l'or et couvre la fenêtre la plus tardive. La sélection par ``mtime`` la
    plus récente est un piège documenté — un CSV plus frais peut porter une
    fenêtre plus courte.
    """
    if explicit is not None:
        if not explicit.exists():
            raise SystemExit(f"CSV introuvable : {explicit}")
        return explicit

    candidates: list[tuple[pd.Timestamp, Path]] = []
    for path in sorted(FILE_COMMON.glob("deals_*.csv")):
        try:
            deals = load_deals(path)
        except Exception:  # noqa: BLE001 — un export corrompu ne doit pas bloquer
            continue
        gold = deals[deals["sleeve"] == SLEEVE]
        if gold.empty:
            continue
        candidates.append((gold["time_utc"].max(), path))

    if not candidates:
        raise SystemExit(
            f"Aucun export de deals contenant la sleeve '{SLEEVE}' dans "
            f"{FILE_COMMON}.\nRejouer un backtest avec Inp_ExportDeals=true et "
            "Inp_AllocGoldMomentum > 0."
        )
    return max(candidates)[1]


def pick_log(explicit: Path | None) -> Path:
    if explicit is not None:
        if not explicit.exists():
            raise SystemExit(f"Journal introuvable : {explicit}")
        return explicit
    logs = sorted(MT5_LOG_DIR.glob("*.log"))
    if not logs:
        raise SystemExit(f"Aucun journal de tester dans {MT5_LOG_DIR}")
    return logs[-1]


def link_production_run(deals_csv: Path) -> dict[str, Any] | None:
    """Rattacher l'extraction au run publié, via ``mt5_reference.json``.

    C'est ce fichier qui fait foi pour le rapport client : il nomme le run, le
    rapport HTML et le CSV de deals du backtest publié. On vérifie que le CSV
    lu ici est bien celui-là — sinon les trades or extraits ne seraient pas
    ceux du portefeuille publié, et le rapport comparerait deux runs.
    """
    reference = _PROJECT_ROOT / "results/production_report/mt5_reference.json"
    if not reference.exists():
        return None
    try:
        provenance = json.loads(reference.read_text()).get("provenance") or {}
    except (json.JSONDecodeError, OSError):
        return None

    published_csv = (provenance.get("deals_csv") or {}).get("sha256")
    same_run = published_csv == _sha256(deals_csv)
    if not same_run:
        print(
            "[!] Ce CSV de deals n'est pas celui du rapport MT5 publié "
            "(results/production_report/mt5_reference.json). Les trades or "
            "extraits ne décrivent donc pas le backtest publié."
        )
    return {
        "matches_published_report": same_run,
        "run_json": provenance.get("run_json"),
        "html_report": provenance.get("html_report"),
        "tester": provenance.get("tester"),
        "gold_inputs": {
            key: value
            for key, value in (provenance.get("ea_inputs") or {}).items()
            if "Gold" in key or key in {"Inp_RiskScale", "Inp_GlobalTargetVol"}
        },
    }


# ---------------------------------------------------------------------------


def build_provenance(
    deals_csv: Path, log_path: Path, trades: pd.DataFrame, n_runs: int
) -> dict[str, Any]:
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "generator": "scripts/extract_gold_trades.py",
        "sleeve": SLEEVE,
        "deals_csv": {
            "path": str(deals_csv),
            "sha256": _sha256(deals_csv),
            "size_bytes": deals_csv.stat().st_size,
        },
        "tester_log": {
            "path": str(log_path),
            "size_bytes": log_path.stat().st_size,
            "gold_runs_found": n_runs,
        },
        "window": {
            "first_entry_utc": trades["entry_time"].min().isoformat(),
            "last_exit_utc": trades["exit_time"].max().isoformat(),
        },
        "counts": {
            "trades": int(len(trades)),
            "safety_stops": int(trades["safety_stop"].sum()),
            "forced_closes": int(trades["forced_close"].sum()),
        },
        "totals": {
            "profit": round(float(trades["profit"].sum()), 2),
            "commission": round(float(trades["commission"].sum()), 2),
            "swap": round(float(trades["swap"].sum()), 2),
            "net": round(float(trades["net"].sum()), 2),
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--deals", type=Path, help="CSV de deals MT5 (défaut : auto)")
    ap.add_argument("--log", type=Path, help="journal du tester (défaut : le plus récent)")
    ap.add_argument(
        "--expect",
        type=int,
        default=None,
        help="nombre de trades attendu ; échoue si le compte diffère",
    )
    args = ap.parse_args()

    deals_csv = pick_deals_csv(args.deals)
    print(f"[deals] {deals_csv}")
    trades = pair_gold_trades(load_deals(deals_csv))
    print(
        f"[deals] {len(trades)} trades or  "
        f"{trades['entry_time'].min().date()} → {trades['exit_time'].max().date()}"
    )

    if args.expect is not None and len(trades) != args.expect:
        raise SystemExit(
            f"{len(trades)} trades appariés, {args.expect} attendus. "
            "Le CSV ne correspond pas au run de référence."
        )

    log_path = pick_log(args.log)
    print(f"[log]   {log_path}")
    runs = parse_log_entries(log_path)
    run = select_matching_run(runs, trades["entry_time"])
    print(f"[log]   run retrouvé parmi {len(runs)} — {len(run)} entrées appariées")

    # Le journal donne score et levier ; le CSV donne l'exécution. Les lots des
    # deux sources doivent coïncider — sinon les lignes ne parlent pas du même
    # trade et tout ce qui suit serait faux.
    merged = trades.join(run[["score", "leverage", "log_lots", "log_price"]])
    drift = (merged["lots"] - merged["log_lots"]).abs()
    if drift.max() > 1e-9:
        raise SystemExit(
            f"Les lots du journal et du CSV divergent (max {drift.max():.4f}) : "
            "l'appariement est faux, ne pas publier."
        )
    merged = merged.drop(columns=["log_lots", "log_price"])

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(OUT_CSV, index=False)

    provenance = build_provenance(deals_csv, log_path, trades, len(runs))
    published = link_production_run(deals_csv)
    if published is not None:
        provenance["published_report"] = published
    OUT_JSON.write_text(json.dumps(provenance, indent=2) + "\n")

    print(f"\n[out]   {OUT_CSV.relative_to(_PROJECT_ROOT)}")
    print(f"[out]   {OUT_JSON.relative_to(_PROJECT_ROOT)}")
    totals = provenance["totals"]
    print(
        f"\nnet {totals['net']:+,.2f}  (profit {totals['profit']:+,.2f}  "
        f"swap {totals['swap']:+,.2f}  commission {totals['commission']:+,.2f})"
    )
    print(
        f"stops de sécurité : {provenance['counts']['safety_stops']}  ·  "
        f"liquidations de fin de test : {provenance['counts']['forced_closes']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
