#!/usr/bin/env python3
"""Figer les trades du candidat USD/JPY d'un run MT5 de recherche.

Le rapport ``main_usdjpy_trades.tex`` descend au trade individuel sur la sleeve
TSMOM (moteur or) portée sur ``USDJPY.c`` — le meilleur candidat du cycle
d'expansion momentum 2026-H2 (``docs/research/momentum_expansion_2026H2.md``).
Ses deux sources vivent **hors du dépôt** :

* le CSV par deal écrit par l'EA quand ``Inp_ExportDeals=true`` — il porte le
  ``magic``, donc la sleeve, et surtout le **swap**, qui sur ce candidat n'est
  pas un coût mais une recette : un long USD/JPY encaisse le différentiel de
  taux, jusqu'à ``+25 862 $`` sur une seule position ;
* le journal du Strategy Tester, seul endroit où figurent le **score** de
  décision et le **levier** appliqués à chaque entrée.

À la différence de l'extraction or, le CSV de deals n'est **pas** cherché dans
l'arborescence MT5 : le sweep du cycle a rejoué vingt-et-un instruments dans la
même journée, chacun écrasant l'export du précédent. La copie de travail a donc
été sécurisée dans le dépôt (``reports/mt5/oos_usdjpy_deals.csv``) et c'est ce
chemin explicite qui fait foi. Chercher « le dernier export » rendrait ici le
mauvais instrument.

Le journal mélange **tous les runs de la journée** — cinquante-et-un, dont cinq
sur USD/JPY. Les entrées d'un même run sont contiguës et leurs dates croissent,
donc on découpe le flux à chaque recul de date après avoir filtré sur le
symbole, puis on retient le run dont les dates d'entrée coïncident exactement
avec celles du CSV. Aucune heuristique sur l'heure du log.

Usage:
    python scripts/extract_usdjpy_trades.py
    python scripts/extract_usdjpy_trades.py --deals <chemin.csv> --log <chemin.log>
    python scripts/extract_usdjpy_trades.py --expect 35
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
from parse_mt5_report import load_deals  # noqa: E402

MT5_REPORTS = _PROJECT_ROOT / "reports/mt5"
MT5_LOG_DIR = Path.home() / (
    ".mt5/drive_c/Program Files/MetaTrader 5/Tester/logs"
)
# Copie sécurisée hors de MT5 : le sweep du cycle écrase l'export à chaque
# instrument. C'est ce fichier-là, et pas « le plus récent », qui décrit le run.
DEALS_CSV = MT5_REPORTS / "oos_usdjpy_deals.csv"
OUT_CSV = MT5_REPORTS / "usdjpy_trades_research.csv"
OUT_JSON = MT5_REPORTS / "usdjpy_trades_research.json"

# L'EA porte le moteur or ; l'instrument est passé par ``Inp_Gold_Symbol``. Le
# label de sleeve reste donc GOLD_MOMENTUM dans les deals comme dans le journal,
# et c'est le symbole qui distingue ce run.
SLEEVE = "GOLD_MOMENTUM"
SYMBOL = "USDJPY.c"

# La sleeve décide à la clôture de séance — 21:00 UTC. Toute sortie qui tombe
# ailleurs n'a pas été décidée par le signal : c'est le stop de sécurité
# (Inp_Gold_SafetySL) qui a coupé, ou la liquidation de fin de test.
SESSION_EXIT_HOUR_UTC = 21

_ENTRY_RE = re.compile(
    r"(?P<dt>\d{4}\.\d{2}\.\d{2} \d{2}:\d{2}:\d{2})\s+"
    r"\[Gold_Momentum\]\[INFO\] Entry (?P<direction>LONG|SHORT) (?P<symbol>\S+) "
    r"lots=(?P<lots>[\d.]+) price=(?P<price>[\d.]+) "
    r"score=(?P<score>-?[\d.]+) lev=(?P<lev>[\d.]+)"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# Côté deals : apparier les aller-retours
# ---------------------------------------------------------------------------


def pair_usdjpy_trades(deals: pd.DataFrame) -> pd.DataFrame:
    """Recomposer les positions USD/JPY à partir des deals, par ``position_id``.

    MT5 écrit deux deals par position : l'ouverture (``entry == 0``) porte le
    prix d'entrée, la sortie (``entry == 1``) porte le résultat, le swap et la
    commission. Le rapprochement se fait sur ``position_id``, jamais sur
    l'ordre des lignes.

    Le filtre porte sur la sleeve **et** sur le symbole : le même magic a servi
    à vingt-et-un instruments pendant le sweep, et un CSV mélangé passerait
    autrement sans bruit.
    """
    sleeve = deals[
        (deals["sleeve"] == SLEEVE)
        & (deals["symbol"] == SYMBOL)
        & (~deals["is_balance_op"])
    ]
    if sleeve.empty:
        raise SystemExit(
            f"Aucun deal '{SLEEVE}' sur {SYMBOL} dans ce CSV. Le run a-t-il été "
            "joué avec Inp_Gold_Symbol=USDJPY et Inp_AllocGoldMomentum > 0 ?"
        )

    opens = sleeve[sleeve["entry"] == 0].set_index("position_id")
    closes = sleeve[sleeve["entry"] == 1].set_index("position_id")

    unmatched = opens.index.symmetric_difference(closes.index)
    if len(unmatched) > 0:
        raise SystemExit(
            f"{len(unmatched)} position(s) USD/JPY sans aller-retour complet : "
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
    # Une sortie hors de la borne de séance n'a pas été décidée par le signal —
    # sauf la liquidation de fin de test, qui tombe elle aussi hors borne mais
    # relève du tester, pas du stop. Les confondre gonflerait le compte des
    # stops de sécurité de un sur ce run, où il n'y en a aucun.
    trades["safety_stop"] = (
        trades["exit_time"].dt.hour != SESSION_EXIT_HOUR_UTC
    ) & (~trades["forced_close"])
    return trades


# ---------------------------------------------------------------------------
# Côté journal : retrouver le run, en extraire score et levier
# ---------------------------------------------------------------------------


def parse_log_entries(log_path: Path) -> list[pd.DataFrame]:
    """Découper les entrées USD/JPY du journal en runs successifs.

    Le journal est en UTF-16 et pèse des dizaines de Mo : il est lu en flux, pas
    chargé en mémoire. On ne retient que les entrées du symbole visé — le sweep
    en a écrit vingt-et-un dans le même fichier — puis les dates de trade
    croissant à l'intérieur d'un run, tout recul marque le début d'un nouveau
    run.
    """
    runs: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    previous_dt: datetime | None = None

    with codecs.open(log_path, "r", encoding="utf-16", errors="replace") as handle:
        for line in handle:
            if "[Gold_Momentum]" not in line:
                continue
            match = _ENTRY_RE.search(line)
            if match is None or match.group("symbol") != SYMBOL:
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
    """Retenir le run dont les dates d'entrée coïncident avec celles du CSV.

    Le CSV publié vient du **dernier** des runs USD/JPY de la journée, rejoué à
    l'identique après le premier. On retient donc le dernier candidat — mais
    seulement après avoir vérifié que tous les candidats portent les mêmes
    scores et les mêmes leviers. Deux runs de même fenêtre et de sizing
    différent (la situation du sweep) doivent arrêter le script, pas se
    départager par leur rang.
    """
    wanted = list(pd.to_datetime(entry_times))
    matches = [run for run in runs if list(run["entry_time"]) == wanted]

    if not matches:
        sizes = sorted({len(run) for run in runs})
        raise SystemExit(
            f"Aucun run du journal ne correspond aux {len(wanted)} entrées du CSV "
            f"(runs {SYMBOL} trouvés : {len(runs)}, tailles {sizes}).\n"
            "Le journal du jour ne contient probablement pas le run qui a produit "
            "ce CSV — vérifier --log."
        )

    chosen = matches[-1]
    columns = ["score", "leverage", "log_lots"]
    for other in matches[:-1]:
        if not other[columns].equals(chosen[columns]):
            raise SystemExit(
                f"{len(matches)} runs du journal portent ces {len(wanted)} dates "
                "mais des décisions différentes (score, levier ou lots). "
                "Impossible de savoir lequel a produit le CSV — ne pas publier."
            )
    return chosen


# ---------------------------------------------------------------------------
# Sélection des sources
# ---------------------------------------------------------------------------


def pick_deals_csv(explicit: Path | None) -> Path:
    """Choisir le CSV de deals à lire — par défaut la copie versionnée."""
    path = explicit if explicit is not None else DEALS_CSV
    if not path.exists():
        raise SystemExit(
            f"CSV introuvable : {path}\n"
            "Le run de recherche USD/JPY doit avoir été sécurisé dans le dépôt "
            "avant extraction (l'export MT5 est écrasé à chaque instrument)."
        )
    return path


def pick_run_json() -> dict[str, Any] | None:
    """Retrouver la fiche du run de recherche USD/JPY la plus récente.

    ``scripts/run_mt5_backtest.py`` dépose un ``run_*.json`` par exécution. On
    retient le plus récent dont le rapport HTML est celui d'USD/JPY : c'est lui
    qui nomme le journal du tester et porte les agrégats du simulateur, contre
    lesquels le total extrait est vérifié.
    """
    candidates: list[tuple[str, dict[str, Any]]] = []
    for path in sorted(MT5_REPORTS.glob("run_*.json")):
        try:
            payload = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        report = (payload.get("metrics") or {}).get("report_path") or ""
        if "oos_USDJPY" not in report:
            continue
        candidates.append((payload.get("run_id") or path.stem, payload))

    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])[1]


def pick_log(explicit: Path | None, run_json: dict[str, Any] | None) -> Path:
    """Journal du tester : celui que la fiche de run désigne, sinon le dernier."""
    if explicit is not None:
        if not explicit.exists():
            raise SystemExit(f"Journal introuvable : {explicit}")
        return explicit

    if run_json is not None:
        recorded = (run_json.get("log_summary") or {}).get("log_path")
        if recorded and Path(recorded).exists():
            return Path(recorded)

    logs = sorted(MT5_LOG_DIR.glob("*.log"))
    if not logs:
        raise SystemExit(f"Aucun journal de tester dans {MT5_LOG_DIR}")
    return logs[-1]


def link_research_run(
    run_json: dict[str, Any] | None, trades: pd.DataFrame
) -> dict[str, Any] | None:
    """Rattacher l'extraction au run MT5 et vérifier que le net y retombe.

    C'est le seul contrôle croisé disponible ici : contrairement à l'or, ce run
    n'est pas publié dans un rapport client, donc aucune empreinte de fichier ne
    fait foi. Le total net du simulateur, lui, est indiscutable — s'il ne
    coïncide pas au cent près avec la somme des trades appariés, le CSV et le
    rapport ne décrivent pas le même run.
    """
    if run_json is None:
        print(
            "[!] Aucune fiche de run USD/JPY dans reports/mt5/ : le total net "
            "extrait n'est vérifié contre rien."
        )
        return None

    metrics = run_json.get("metrics") or {}
    reported = metrics.get("total_net_profit")
    reported_value = (
        float(str(reported).replace(" ", "").replace("\xa0", "").replace(" ", ""))
        if reported
        else None
    )
    extracted = round(float(trades["net"].sum()), 2)
    if reported_value is not None and abs(reported_value - extracted) > 0.01:
        raise SystemExit(
            f"Le net des trades appariés ({extracted:+,.2f}) ne retombe pas sur "
            f"celui du rapport MT5 ({reported_value:+,.2f}). Le CSV de deals et "
            "la fiche de run ne décrivent pas le même backtest."
        )

    return {
        "run_id": run_json.get("run_id"),
        "ini_path": run_json.get("ini_path"),
        "html_report": metrics.get("report_path"),
        "period": metrics.get("period"),
        "initial_deposit": metrics.get("initial_deposit"),
        "total_net_profit": reported,
        "sharpe_ratio": metrics.get("sharpe_ratio"),
        "total_trades": metrics.get("total_trades"),
        "balance_dd_max": metrics.get("balance_dd_max"),
        "equity_dd_max": metrics.get("equity_dd_max"),
        "net_matches_report": reported_value is not None,
    }


# ---------------------------------------------------------------------------


def build_provenance(
    deals_csv: Path, log_path: Path, trades: pd.DataFrame, n_runs: int
) -> dict[str, Any]:
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "generator": "scripts/extract_usdjpy_trades.py",
        "sleeve": SLEEVE,
        "symbol": SYMBOL,
        "deals_csv": {
            "path": str(deals_csv),
            "sha256": _sha256(deals_csv),
            "size_bytes": deals_csv.stat().st_size,
        },
        "tester_log": {
            "path": str(log_path),
            "size_bytes": log_path.stat().st_size,
            "usdjpy_runs_found": n_runs,
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
    ap.add_argument(
        "--deals", type=Path, help=f"CSV de deals MT5 (défaut : {DEALS_CSV.name})"
    )
    ap.add_argument("--log", type=Path, help="journal du tester (défaut : celui du run)")
    ap.add_argument(
        "--expect",
        type=int,
        default=None,
        help="nombre de trades attendu ; échoue si le compte diffère",
    )
    args = ap.parse_args()

    deals_csv = pick_deals_csv(args.deals)
    print(f"[deals] {deals_csv}")
    trades = pair_usdjpy_trades(load_deals(deals_csv))
    print(
        f"[deals] {len(trades)} trades USD/JPY  "
        f"{trades['entry_time'].min().date()} → {trades['exit_time'].max().date()}"
    )

    if args.expect is not None and len(trades) != args.expect:
        raise SystemExit(
            f"{len(trades)} trades appariés, {args.expect} attendus. "
            "Le CSV ne correspond pas au run de référence."
        )

    run_json = pick_run_json()
    log_path = pick_log(args.log, run_json)
    print(f"[log]   {log_path}")
    runs = parse_log_entries(log_path)
    run = select_matching_run(runs, trades["entry_time"])
    print(
        f"[log]   run retrouvé parmi {len(runs)} runs {SYMBOL} — "
        f"{len(run)} entrées appariées"
    )

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
    research = link_research_run(run_json, trades)
    if research is not None:
        provenance["research_run"] = research
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
    if research is not None:
        print(
            f"rapport MT5 : net {research['total_net_profit']}  ·  "
            f"Sharpe {research['sharpe_ratio']}  ·  "
            f"{research['total_trades']} trades  ·  run {research['run_id']}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
