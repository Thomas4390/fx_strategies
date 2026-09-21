#!/usr/bin/env python3
"""Campagne in-sample officielle de la stratégie « XAUUSD niveaux x10 ».

Producteur **unique** de `results/xau_x10/*`. Tout chiffre publié dans
`docs/research/xau_x10_is_results.md` et dans le rapport client sort d'un JSON
écrit ici ; rien n'est recalculé à la main ailleurs.

Les seuils, la règle de sélection et le modèle de coût sont **gelés avant
mesure** (`docs/research/xau_x10_decision_table.md`, commit antérieur à ce
script). Ce fichier ne décide de rien : il mesure, et laisse la table arbitrer.
Un verdict négatif est un livrable prévu (spec §0).

Ce que la campagne s'interdit, et que le code rend impossible :

- toucher une barre ≥ 2026-01-01 — `assert_not_optimizing` en tête de chaque
  étape, sur l'index réellement consommé ;
- dépasser le budget d'essais de l'annexe A.3 — **deux** appels à `log_trials`,
  27 + 7 = 34, tous deux **avant** la mesure qu'ils déclarent ;
- re-sélectionner au vu d'un coût plus favorable — les sensibilités du §7 sont
  descriptives et ne rappellent jamais `select_plateau_center`.

Étapes, idempotentes et réexécutables une par une :

    grid        les 27 configurations de l'annexe A.2, au spread gelé de 0,29 $
    select      la règle « centre du plateau » du §3 de la table, à la lettre
    wf          walk-forward annuel 2019-2025 + CPCV sur la grille
    robustness  PSR / DSR / haircut / MinBTL / PBO / SPA / StepM / MC / ruine
    decomp      les douze décompositions du plan de campagne
    ablations   les 7 combinaisons de drapeaux EMA50 / VWAP / DXY
    costs       spread x1 / x1,5 / x2, slippage 0 / 0,05 / 0,10 $
    summary     `is_summary.json` — un champ par critère du §1 de la table
    all         tout ce qui précède, dans cet ordre

Usage :
    uv run python scripts/run_xau_x10_research.py --step all
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import vectorbtpro as vbt

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))

from framework import trials  # noqa: E402
from framework.cpcv import (  # noqa: E402
    build_cpcv_splitter,
    cpcv_oos_distribution,
    cpcv_summary,
)
from framework.holdout import (  # noqa: E402
    HOLDOUT_START,
    assert_not_optimizing,
    trim_insample,
)
from framework.robustness import robustness_report  # noqa: E402
from framework.ruin import ruin_report  # noqa: E402
from framework.x10_context import session_dates as session_dates_of  # noqa: E402
from framework.x10_engine import MAX_HOLD_BARS  # noqa: E402
from strategies.xau_x10 import (  # noqa: E402
    GRID_A_MIN,
    GRID_K_S,
    GRID_Z,
    X10_INIT_CASH,
    X10_SPREAD_DEFAULT,
    count_summary,
    create_cv_pipeline,
    pipeline,
    prepare_inputs,
)
from utils import load_gold_data  # noqa: E402

# ═══════════════════════════════════════════════════════════════════════
# CONSTANTES DE CAMPAGNE — gelées, et toutes citées dans is_summary.json
# ═══════════════════════════════════════════════════════════════════════

RESULTS_DIR = _REPO / "results" / "xau_x10"

IS_START = "2019-01-01"
IS_END = "2025-12-31"

# §1.1 de la table de décision : la sélection se fait au spread CONSTANT de
# 0,29 $ (catalogue broker 2026-07-28). Les sensibilités x1,5 et x2 sont
# obligatoires et n'ont pas le droit de re-sélectionner quoi que ce soit.
SELECTION_SPREAD = X10_SPREAD_DEFAULT
COST_SPREAD_MULTIPLIERS = (1.0, 1.5, 2.0)
COST_SLIPPAGE_PER_SIDE = (0.0, 0.05, 0.10)

# Rendements journaliers = équité par séance (§2). 252 séances par an.
ANN_DAYS = 252.0

SEED = 42
N_BOOT = 2000
N_MC = 2000
# Bloc du bootstrap stationnaire : le défaut du module (50) est calibré pour du
# M1. Sur ~1 800 séances, un mois de bourse est le bon ordre de grandeur.
BLOCK_LEN_DAILY = 20.0

CPCV_N_GROUPS = 6
CPCV_N_TEST_GROUPS = 2
CPCV_PURGE = "1 day"
CPCV_EMBARGO_PCT = 0.01

# Annexe A.3 : ce que la campagne consomme en entier. Le DSR est déflaté par ce
# chiffre et non par `distinct_trials` à l'instant du calcul, pour deux raisons :
# les 7 ablations sont loguées APRÈS l'étape de robustesse dans l'ordre du
# protocole, et un script relancé doit rendre le même JSON. C'est aussi la borne
# la plus conservatrice — déflater par 27 sous-estimerait la sélection.
N_TRIALS_GRID = 27
N_TRIALS_ABLATIONS = 7
CAMPAIGN_TRIALS = N_TRIALS_GRID + N_TRIALS_ABLATIONS

# §3 de la table : le centre géométrique de la grille. Utilisé UNIQUEMENT comme
# défaut descriptif pré-écrit quand aucune configuration ne passe le plateau —
# ce n'est alors pas une sélection, et le JSON le dit.
GRID_CENTRE = (1, 1, 1)

# Seuils du §1 de la table de décision, recopiés pour être opposables.
TH_MIN_TRADES = 300
TH_MIN_TRADES_PER_SCENARIO = 40
TH_MIN_EXPECTANCY_R = 0.10
TH_MIN_PROFIT_FACTOR = 1.15
TH_MIN_DSR = 0.95
TH_MAX_PBO = 0.35
TH_PLATEAU_POSITIVE_SHARE = 0.60
TH_PLATEAU_MEDIAN_FRACTION = 0.50
TH_WF_MIN_POSITIVE_YEARS = 0.60
TH_WF_MAX_YEAR_SHARE = 0.50

NOT_MEASURED = "non mesuré"

# §5 du plan : les fenêtres de séance New York, bornées sur l'heure de DÉCISION.
NY_SESSIONS: tuple[tuple[str, int, int], ...] = (
    ("Asie 18:00-03:00", 18 * 60, 3 * 60),
    ("Londres 03:00-08:00", 3 * 60, 8 * 60),
    ("New York 08:00-17:00", 8 * 60, 17 * 60),
)

# Tranches de « 10 $ / ATR M5 » du §5 — la grandeur non stationnaire de la note
# de faisabilité §1.
DOLLARS_PER_ATR_EDGES: tuple[float, ...] = (3.0, 6.0, 10.0)
DOLLARS_PER_ATR_LABELS: tuple[str, ...] = ("<3", "3-6", "6-10", ">10")


# ═══════════════════════════════════════════════════════════════════════
# 1. LA GRILLE ET SON VOISINAGE (annexe A.2, §3 de la table)
# ═══════════════════════════════════════════════════════════════════════


def grid_indices() -> list[tuple[int, int, int]]:
    """Les 27 triplets d'indices, ordre lexicographique croissant."""
    return [tuple(t) for t in product(range(3), repeat=3)]


def config_of(idx: tuple[int, int, int]) -> dict[str, float]:
    """Les valeurs `(z, a_min, k_s)` du triplet d'indices."""
    i_z, i_a, i_k = idx
    return {"z": GRID_Z[i_z], "a_min": GRID_A_MIN[i_a], "k_s": GRID_K_S[i_k]}


def config_label(idx: tuple[int, int, int]) -> str:
    """Étiquette stable d'une configuration, utilisée comme colonne partout."""
    cfg = config_of(idx)
    return f"z{cfg['z']:g}_a{cfg['a_min']:g}_k{cfg['k_s']:g}"


def neighbours(idx: tuple[int, int, int]) -> list[tuple[int, int, int]]:
    """Voisins au sens du §3 : un seul axe diffère, et de exactement 1.

    Trois voisins à un coin, quatre ou cinq sur une arête ou une face, six au
    centre géométrique. Une configuration n'est jamais son propre voisin.
    """
    out: list[tuple[int, int, int]] = []
    for axis in range(3):
        for step in (-1, 1):
            cand = list(idx)
            cand[axis] += step
            if 0 <= cand[axis] <= 2:
                out.append(tuple(cand))
    return sorted(out)


def select_plateau_center(sharpes: dict[tuple[int, int, int], float]) -> dict[str, Any]:
    """Applique le §3 de la table de décision, à la lettre.

    Critère plateau : au moins 60 % des voisins à Sharpe net > 0 **et** médiane
    des Sharpes des voisins ≥ 0,5 x pic. Parmi les seules configurations qui le
    passent, la retenue maximise la **médiane du Sharpe de ses voisins** — son
    Sharpe propre n'entre pas dans le classement. Départage : distance de
    Manhattan au centre géométrique, puis `k_s` le plus grand, puis ordre
    lexicographique.

    Aucune configuration ne passe ⇒ **aucune configuration retenue**, et le
    verdict est NE PAS DÉPLOYER. On ne se rabat pas sur le pic.

    Retourne tout l'intermédiaire, pas seulement la réponse : le rapport doit
    pouvoir rejouer la règle à la main.
    """
    values = {k: float(v) for k, v in sharpes.items()}
    peak_idx = max(
        values, key=lambda k: (values[k] if np.isfinite(values[k]) else -np.inf, k)
    )
    peak = values[peak_idx]
    threshold = TH_PLATEAU_MEDIAN_FRACTION * peak

    rows: list[dict[str, Any]] = []
    for idx in sorted(values):
        nbrs = [n for n in neighbours(idx) if n in values]
        nbr_sharpes = [values[n] for n in nbrs]
        share_positive = float(np.mean([s > 0.0 for s in nbr_sharpes])) if nbrs else 0.0
        median_nbr = float(np.median(nbr_sharpes)) if nbrs else float("nan")
        rows.append(
            {
                "index": list(idx),
                "config": config_label(idx),
                **config_of(idx),
                "sharpe_net": values[idx],
                "n_neighbours": len(nbrs),
                "neighbours": [config_label(n) for n in nbrs],
                "neighbour_sharpes": nbr_sharpes,
                "share_neighbours_positive": share_positive,
                "median_neighbour_sharpe": median_nbr,
                "passes_positive_share": share_positive >= TH_PLATEAU_POSITIVE_SHARE,
                "passes_median": bool(median_nbr >= threshold),
                "passes_plateau": bool(
                    share_positive >= TH_PLATEAU_POSITIVE_SHARE
                    and median_nbr >= threshold
                ),
            }
        )

    passing = [r for r in rows if r["passes_plateau"]]
    selected = None
    if passing:
        best = max(
            passing,
            key=lambda r: (
                r["median_neighbour_sharpe"],
                -sum(abs(a - b) for a, b in zip(r["index"], GRID_CENTRE, strict=True)),
                r["index"][2],
                tuple(-v for v in r["index"]),
            ),
        )
        selected = best

    return {
        "rule": "docs/research/xau_x10_decision_table.md §3",
        "peak": {
            "config": config_label(peak_idx),
            "index": list(peak_idx),
            **config_of(peak_idx),
            "sharpe_net": peak,
        },
        "plateau_median_threshold": threshold,
        "n_passing_plateau": len(passing),
        "selected": selected,
        "candidates": rows,
    }


# ═══════════════════════════════════════════════════════════════════════
# 2. MÉTRIQUES D'UN RUN — rien qui ne soit dans la spec ou la table
# ═══════════════════════════════════════════════════════════════════════


def trade_pnl(trades: pd.DataFrame) -> pd.Series:
    """PnL réalisé, tel que le moteur l'a comptabilisé sur l'équité (§11)."""
    if not len(trades):
        return pd.Series(dtype=float)
    return trades["equity_after"] - trades["equity_before"]


def realised_r(trades: pd.DataFrame) -> pd.Series:
    """`R` réalisé = PnL / risque engagé du trade, un multiple sans dimension."""
    if not len(trades):
        return pd.Series(dtype=float)
    risk = trades["risk_amount"].replace(0.0, np.nan)
    return trade_pnl(trades) / risk


def trade_metrics(trades: pd.DataFrame) -> dict[str, Any]:
    """`n`, espérance en R, profit factor, PnL — la cellule de toute décompo."""
    pnl = trade_pnl(trades)
    r = realised_r(trades)
    wins = float(pnl[pnl > 0].sum())
    losses = float(-pnl[pnl < 0].sum())
    return {
        "n": int(len(trades)),
        "expectancy_r": float(r.mean()) if len(r.dropna()) else float("nan"),
        "profit_factor": (wins / losses) if losses > 0 else float("inf" if wins > 0 else "nan"),
        "pnl": float(pnl.sum()),
        "win_rate": float((pnl > 0).mean()) if len(pnl) else float("nan"),
    }


def session_returns(
    trades: pd.DataFrame,
    session_dates: pd.DatetimeIndex,
    init_cash: float = X10_INIT_CASH,
) -> pd.Series:
    """Rendements par séance (§2) — l'équité de fin de séance, séances plates incluses.

    Le moteur compose déjà l'équité trade par trade (§11) : sommer les PnL par
    séance puis cumuler reproduit exactement sa courbe. Les séances sans trade
    portent un rendement nul et **restent dans l'index** — les retirer ferait
    passer une stratégie plate 80 % du temps pour une stratégie continue, et
    gonflerait son Sharpe d'autant.
    """
    dates = pd.DatetimeIndex(session_dates)
    per_session = pd.Series(0.0, index=dates)
    if len(trades):
        grouped = trade_pnl(trades).groupby(trades["session_date"].to_numpy()).sum()
        per_session = per_session.add(
            grouped.reindex(dates, fill_value=0.0), fill_value=0.0
        )
    curve = init_cash + per_session.cumsum()
    previous = curve.shift(1).fillna(init_cash)
    return (curve / previous - 1.0).rename("session_return")


def sharpe_net(returns: pd.Series, ann_days: float = ANN_DAYS) -> float:
    """Sharpe net annualisé sur les rendements de séance, taux sans risque nul."""
    arr = pd.Series(returns).dropna().to_numpy(dtype=np.float64)
    if arr.size < 2:
        return float("nan")
    std = float(arr.std(ddof=1))
    if std <= 0.0:
        return float("nan")
    return float(arr.mean() / std * np.sqrt(ann_days))


def max_drawdown(returns: pd.Series) -> float:
    """Drawdown maximal de la courbe composée des rendements de séance."""
    arr = pd.Series(returns).dropna()
    if arr.empty:
        return float("nan")
    equity = (1.0 + arr).cumprod()
    return float((equity / equity.cummax() - 1.0).min())


def concentration(trades: pd.DataFrame) -> dict[str, Any]:
    """Part du net portée par les 5 meilleurs jours, le meilleur mois, les 10 meilleurs trades.

    Le dénominateur est le net **total**. Une part supérieure à 1 n'est pas un
    bug : elle dit que le reste de l'échantillon perd de l'argent, et c'est
    précisément l'information qu'on cherche. Un net total négatif rend le
    ratio ininterprétable, et il est alors rendu `NaN` plutôt que maquillé.
    """
    pnl = trade_pnl(trades)
    total = float(pnl.sum())
    if not len(pnl):
        return {"total_net": 0.0, "note": "aucun trade"}

    by_day = pnl.groupby(trades["session_date"].to_numpy()).sum().sort_values()
    by_month = (
        pnl.groupby(pd.PeriodIndex(trades["session_date"], freq="M")).sum().sort_values()
    )

    def _share(x: float) -> float:
        return float(x / total) if total > 0 else float("nan")

    top5_days = float(by_day.tail(5).sum())
    best_month = float(by_month.tail(1).sum())
    top10_trades = float(pnl.nlargest(10).sum())
    gross_profit = float(pnl[pnl > 0].sum())

    return {
        "total_net": total,
        # Les parts sont nulles quand le net total est négatif ; les montants,
        # eux, restent lisibles et c'est sur eux que le rapport commente.
        "share_top5_days": _share(top5_days),
        "share_best_month": _share(best_month),
        "share_top10_trades": _share(top10_trades),
        "pnl_top5_days": top5_days,
        "pnl_best_month": best_month,
        "pnl_top10_trades": top10_trades,
        "gross_profit": gross_profit,
        "share_of_gross_profit_top5_days": (
            top5_days / gross_profit if gross_profit > 0 else float("nan")
        ),
        "share_of_gross_profit_top10_trades": (
            top10_trades / gross_profit if gross_profit > 0 else float("nan")
        ),
        "top5_days": {str(k.date()): float(v) for k, v in by_day.tail(5).items()},
        "best_month": str(by_month.index[-1]) if len(by_month) else None,
        "n_profitable_days": int((by_day > 0).sum()),
        "n_losing_days": int((by_day < 0).sum()),
    }


# ═══════════════════════════════════════════════════════════════════════
# 3. LA TABLE DE DÉCISION (§1) — évaluée sur un dict, jamais sur un run
# ═══════════════════════════════════════════════════════════════════════


def evaluate_decision_table(measures: dict[str, Any]) -> dict[str, Any]:
    """Un champ par critère du §1 : valeur mesurée, seuil, pass/fail.

    « Un critère non mesuré vaut **échec**, jamais "non applicable" » : les deux
    critères MT5 et OOS sont donc à `NOT_MEASURED` et à `pass = False`. Le
    verdict rendu est **provisoire** — il porte sur l'in-sample seul, et la
    table n'autorise qu'un seul verdict définitif, prononcé une seule fois.
    """

    def _ok(value: Any, test) -> bool:
        try:
            return bool(value is not None and np.isfinite(float(value)) and test(float(value)))
        except (TypeError, ValueError):
            return False

    n_trades = measures.get("n_trades")
    per_scenario = measures.get("trades_per_scenario") or {}
    expectancy = measures.get("expectancy_r")
    pf = measures.get("profit_factor")
    dsr = measures.get("dsr")
    pbo = measures.get("pbo")
    plateau = measures.get("plateau") or {}
    wf = measures.get("walk_forward") or {}
    exp_x15 = measures.get("expectancy_r_spread_x1_5")

    criteria: dict[str, Any] = {}

    criteria["trades_in_sample"] = {
        "label": "Trades in-sample 2019 → 2025-12-31",
        "measured": n_trades,
        "threshold": f">= {TH_MIN_TRADES}",
        "pass": _ok(n_trades, lambda v: v >= TH_MIN_TRADES),
        "scenarios_with_enough_trades": sorted(
            k for k, v in per_scenario.items() if v >= TH_MIN_TRADES_PER_SCENARIO
        ),
        "note": (
            f"un scénario n'est commenté qu'à >= {TH_MIN_TRADES_PER_SCENARIO} trades"
        ),
    }

    criteria["expectancy_and_profit_factor"] = {
        "label": "Espérance nette / profit factor",
        "measured": {"expectancy_r": expectancy, "profit_factor": pf},
        "threshold": f">= +{TH_MIN_EXPECTANCY_R} R / >= {TH_MIN_PROFIT_FACTOR}",
        "pass": _ok(expectancy, lambda v: v >= TH_MIN_EXPECTANCY_R)
        and _ok(pf, lambda v: v >= TH_MIN_PROFIT_FACTOR),
    }

    criteria["dsr"] = {
        "label": 'DSR (déflaté par distinct_trials("xau_x10"))',
        "measured": dsr,
        "threshold": f">= {TH_MIN_DSR}",
        "pass": _ok(dsr, lambda v: v >= TH_MIN_DSR),
        "n_trials": measures.get("n_trials_logged"),
    }

    criteria["pbo"] = {
        "label": "PBO (CSCV, matrice 27 x jours)",
        "measured": pbo,
        "threshold": f"<= {TH_MAX_PBO}",
        "pass": _ok(pbo, lambda v: v <= TH_MAX_PBO),
    }

    criteria["plateau"] = {
        "label": "Plateau",
        "measured": {
            "share_neighbours_positive": plateau.get("share_neighbours_positive"),
            "median_neighbour_sharpe": plateau.get("median_neighbour_sharpe"),
            "peak_sharpe": plateau.get("peak_sharpe"),
            "n_passing_plateau": plateau.get("n_passing_plateau"),
        },
        "threshold": (
            f">= {TH_PLATEAU_POSITIVE_SHARE:.0%} voisins à Sharpe > 0 ; "
            f"médiane des voisins >= {TH_PLATEAU_MEDIAN_FRACTION} x pic"
        ),
        "pass": bool(plateau.get("selected_exists", False)),
    }

    positive_years = wf.get("share_positive_years")
    max_year_share = wf.get("max_year_share_of_net")
    criteria["walk_forward"] = {
        "label": "Walk-forward annuel",
        "measured": {
            "share_positive_years": positive_years,
            "max_year_share_of_net": max_year_share,
        },
        "threshold": (
            f">= {TH_WF_MIN_POSITIVE_YEARS:.0%} d'années positives ; "
            f"aucune année > {TH_WF_MAX_YEAR_SHARE:.0%} du net total"
        ),
        "pass": _ok(positive_years, lambda v: v >= TH_WF_MIN_POSITIVE_YEARS)
        and _ok(max_year_share, lambda v: v <= TH_WF_MAX_YEAR_SHARE),
    }

    criteria["spread_x1_5"] = {
        "label": "Spread x1,5",
        "measured": exp_x15,
        "threshold": "espérance > 0",
        "pass": _ok(exp_x15, lambda v: v > 0.0),
    }

    criteria["mt5_model_1"] = {
        "label": "MT5 modèle 1, 2022-11-04 → 2025-12-31",
        "measured": NOT_MEASURED,
        "threshold": "espérance >= 0, même signe que Python",
        "pass": False,
        "note": "critère non mesuré = échec (table §1), jamais « non applicable »",
    }

    criteria["oos_2026"] = {
        "label": "OOS 2026, lecture unique",
        "measured": NOT_MEASURED,
        "threshold": "n >= 30, sinon NON CONCLUANT",
        "pass": False,
        "note": "holdout LOCKED : aucune lecture n'a eu lieu dans cette phase",
    }

    failed = sorted(k for k, v in criteria.items() if not v["pass"])
    interpretable = sorted(
        k for k in failed if criteria[k]["measured"] != NOT_MEASURED
    )
    verdict = "NE PAS DÉPLOYER" if interpretable else "DÉPLOIEMENT À BLANC (démo)"

    return {
        "criteria": criteria,
        "n_criteria": len(criteria),
        "n_passing": sum(1 for v in criteria.values() if v["pass"]),
        "failed": failed,
        "failed_with_an_interpretable_measure": interpretable,
        "provisional_in_sample_verdict": verdict,
        "verdict_scope": (
            "PROVISOIRE, in-sample seul. La table §2 n'autorise qu'un seul "
            "verdict définitif, prononcé une seule fois, après MT5 et OOS."
        ),
    }


# ═══════════════════════════════════════════════════════════════════════
# 4. SÉRIALISATION — un JSON qu'on peut diffuser et rejouer
# ═══════════════════════════════════════════════════════════════════════


def jsonable(obj: Any) -> Any:
    """Convertit récursivement en types JSON, NaN/inf compris.

    `json` sait écrire `NaN` mais aucun autre langage ne sait le relire ; les
    non-finis deviennent donc `null`, et l'absence de valeur est visible.
    """
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, pd.DataFrame):
        return jsonable(obj.reset_index().to_dict(orient="records"))
    if isinstance(obj, pd.Series):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (pd.Timestamp, pd.Period)):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return jsonable(obj.tolist())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        value = float(obj)
        return value if np.isfinite(value) else None
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if obj is None or isinstance(obj, (str, int)):
        return obj
    return str(obj)


def write_json(name: str, payload: Any) -> Path:
    path = RESULTS_DIR / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(payload), indent=2, ensure_ascii=False) + "\n")
    print(f"  → {path.relative_to(_REPO)}")
    return path


def read_json(name: str) -> Any:
    path = RESULTS_DIR / name
    if not path.exists():
        raise SystemExit(
            f"{path.relative_to(_REPO)} manque : lance l'étape qui le produit "
            f"(ou --step all)."
        )
    return json.loads(path.read_text())


def git_head() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=_REPO,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except Exception:  # pragma: no cover - dépôt absent
        return "unknown"


# ═══════════════════════════════════════════════════════════════════════
# 5. L'ÉCHANTILLON — chargé une fois, contrôlé à chaque étape
# ═══════════════════════════════════════════════════════════════════════


class Sample:
    """L'échantillon in-sample et son contexte, préparés une seule fois.

    `prepare_inputs` coûte l'essentiel d'un run isolé ; les 27 configurations
    de la grille, les 7 ablations et les 9 sensibilités de coût partagent la
    même instance. Le garde-fou holdout est rejoué à chaque étape sur l'index
    réellement consommé, pas une fois au chargement : une étape ajoutée plus
    tard ne doit pas pouvoir hériter d'un contrôle fait ailleurs.
    """

    def __init__(self) -> None:
        raw, _ = load_gold_data()
        raw = trim_insample(raw)
        raw = raw.loc[IS_START:IS_END]

        # Une séance court de 18:00 (J-1) à 17:00 (J) : les barres du soir du
        # 2025-12-31 appartiennent à une séance **étiquetée 2026-01-01**, dont
        # la seconde moitié est dans la tranche gelée. Les garder, c'est publier
        # un rendement de séance tronqué et faire entrer une étiquette ≥ 2026
        # dans un index de sélection. Elles sont retirées : plus conservateur
        # que de les inclure, et `assert_not_optimizing` devient exact au lieu
        # d'être contourné.
        labels = session_dates_of(pd.DatetimeIndex(raw.index))
        keep = labels < HOLDOUT_START
        n_dropped = int((~keep).sum())
        if n_dropped:
            print(
                f"[holdout] {n_dropped} barres M1 retirées : elles appartiennent "
                f"à la séance incomplète étiquetée {labels[~keep].min().date()}"
            )
        raw = raw[keep]
        self.n_bars_dropped_trailing_session = n_dropped
        self.raw = raw
        self.inputs = prepare_inputs(raw)
        self.m5 = self.inputs.m5
        epoch = pd.Timestamp("1970-01-01")
        self._session_of_bar = pd.DatetimeIndex(
            epoch + pd.to_timedelta(self.inputs.session_id, unit="D")
        )
        self.session_dates = pd.DatetimeIndex(sorted(set(self._session_of_bar)))
        self.guard("chargement")

    def guard(self, step: str) -> None:
        """`assert_not_optimizing` sur les trois index que l'étape consomme."""
        for index in (self.raw.index, self.m5.index, self.session_dates):
            assert_not_optimizing(pd.DatetimeIndex(index))
        print(f"[holdout] {step} : ok, rien >= {HOLDOUT_START.date()}")

    def max_dates(self) -> dict[str, str]:
        """La preuve A4 : la date maximale de chaque index utilisé."""
        return {
            "m1_max": str(self.raw.index.max()),
            "m5_max": str(self.m5.index.max()),
            "session_max": str(self.session_dates.max()),
            "holdout_start": str(HOLDOUT_START.date()),
        }

    def run(self, idx: tuple[int, int, int], **kwargs: Any):
        """Un run complet, avec `session_date` déjà posée sur chaque trade."""
        cfg = config_of(idx)
        pf, ind = pipeline(
            self.raw,
            z=cfg["z"],
            a_min=cfg["a_min"],
            k_s=cfg["k_s"],
            spread=kwargs.pop("spread", SELECTION_SPREAD),
            inputs=self.inputs,
            **kwargs,
        )
        trades = ind.trades
        if len(trades):
            trades = trades.copy()
            trades["session_date"] = self._session_of_bar[
                trades["i_exit_m5"].to_numpy(dtype=np.int64)
            ]
        else:
            trades = trades.assign(session_date=pd.Series(dtype="datetime64[ns]"))
        ind.trades = trades
        return pf, ind

    def summarise(self, idx: tuple[int, int, int], **kwargs: Any) -> dict[str, Any]:
        """Les métriques publiables d'une configuration + sa série de séance."""
        _, ind = self.run(idx, **kwargs)
        trades = ind.trades
        returns = session_returns(trades, self.session_dates)
        metrics = trade_metrics(trades)
        scenarios = (
            trades["scenario_name"].value_counts().to_dict() if len(trades) else {}
        )
        return {
            "index": list(idx),
            "config": config_label(idx),
            **config_of(idx),
            **metrics,
            "sharpe_net": sharpe_net(returns),
            "max_drawdown": max_drawdown(returns),
            "trades_per_scenario": {str(k): int(v) for k, v in scenarios.items()},
            "_returns": returns,
            "_indicator": ind,
        }


def daily_portfolio(returns: pd.Series) -> vbt.Portfolio:
    """Un portefeuille journalier dont les rendements SONT ceux de la séance.

    `robustness_report` attend un `vbt.Portfolio` et n'en lit que `pf.returns`.
    Le portefeuille M1 réel en compte 2,5 millions, tous nuls hors position :
    le bootstrap et le DSR y mesureraient la fréquence d'échantillonnage, pas
    la stratégie. Une barre d'amorce est ajoutée devant pour que le premier
    rendement de séance soit conservé au lieu d'être absorbé par l'achat.
    """
    idx = pd.DatetimeIndex(returns.index)
    equity = X10_INIT_CASH * (1.0 + returns.to_numpy(dtype=np.float64)).cumprod()
    full_index = pd.DatetimeIndex([idx[0] - pd.Timedelta(days=1)]).append(idx)
    curve = pd.Series(np.concatenate([[X10_INIT_CASH], equity]), index=full_index)
    return vbt.Portfolio.from_holding(curve, freq="1D")


# ═══════════════════════════════════════════════════════════════════════
# ÉTAPE 1 — LA GRILLE DES 27 (annexe A.2, spread gelé 0,29 $)
# ═══════════════════════════════════════════════════════════════════════


def step_grid(sample: Sample) -> None:
    sample.guard("grid")

    # Loguer AVANT de mesurer. `config_key` nomme l'espace de configurations :
    # un re-run du script ne gonfle pas `distinct_trials`.
    trials.log_trials(
        "xau_x10",
        27,
        note=(
            "grille annexe A.2 (z x a_min x k_s), XAUUSD x10, in-sample "
            f"{IS_START} → {IS_END}, spread constant {SELECTION_SPREAD} $"
        ),
        config_key="xau_x10:grid27:v1",
    )
    print(f"[trials] xau_x10 distinct = {trials.distinct_trials('xau_x10')}")

    rows: list[dict[str, Any]] = []
    returns_matrix: dict[str, pd.Series] = {}
    for idx in grid_indices():
        summary = sample.summarise(idx)
        returns_matrix[summary["config"]] = summary.pop("_returns")
        summary.pop("_indicator")
        rows.append(summary)
        print(
            f"  {summary['config']:>22}  sharpe={summary['sharpe_net']:+.3f}  "
            f"n={summary['n']:>4}  E[R]={summary['expectancy_r']:+.4f}  "
            f"PF={summary['profit_factor']:.3f}"
        )

    frame = pd.DataFrame(rows)
    frame["trades_per_scenario"] = frame["trades_per_scenario"].apply(json.dumps)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_csv(RESULTS_DIR / "grid27.csv", index=False)
    print(f"  → {(RESULTS_DIR / 'grid27.csv').relative_to(_REPO)}")

    matrix = pd.DataFrame(returns_matrix)
    matrix.index.name = "session_date"
    matrix.to_parquet(RESULTS_DIR / "grid27_daily_returns.parquet")
    print(f"  → {(RESULTS_DIR / 'grid27_daily_returns.parquet').relative_to(_REPO)}")
    print(f"  matrice : {matrix.shape[0]} séances x {matrix.shape[1]} configurations")


def load_grid(sample: Sample | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """La grille et sa matrice de rendements, telles qu'archivées."""
    csv = RESULTS_DIR / "grid27.csv"
    parquet = RESULTS_DIR / "grid27_daily_returns.parquet"
    if not csv.exists() or not parquet.exists():
        raise SystemExit("grid27.csv / grid27_daily_returns.parquet manquent : --step grid")
    frame = pd.read_csv(csv)
    frame["index"] = frame["index"].apply(lambda s: tuple(json.loads(s)))
    return frame, pd.read_parquet(parquet)


# ═══════════════════════════════════════════════════════════════════════
# ÉTAPE 2 — SÉLECTION (§3 de la table)
# ═══════════════════════════════════════════════════════════════════════


def step_select(sample: Sample) -> None:
    sample.guard("select")
    frame, _ = load_grid()
    sharpes = {
        tuple(row["index"]): float(row["sharpe_net"]) for _, row in frame.iterrows()
    }
    report = select_plateau_center(sharpes)

    if report["selected"] is None:
        report["retained_config"] = None
        report["fallback_for_descriptive_analyses"] = {
            "index": list(GRID_CENTRE),
            "config": config_label(GRID_CENTRE),
            **config_of(GRID_CENTRE),
            "why": (
                "AUCUNE configuration ne passe le critère plateau du §3. Il n'y a "
                "donc PAS de configuration retenue, et le verdict est NE PAS "
                "DÉPLOYER. Le centre géométrique de la grille sert de défaut "
                "descriptif, pré-écrit au brief de campagne AVANT mesure : ce "
                "n'est pas une sélection, et il ne doit jamais être présenté "
                "comme la configuration de la stratégie."
            ),
        }
        print("  aucune configuration ne passe le plateau → NE PAS DÉPLOYER")
        print(f"  défaut descriptif : {config_label(GRID_CENTRE)}")
    else:
        report["retained_config"] = report["selected"]
        report["fallback_for_descriptive_analyses"] = None
        print(f"  retenue : {report['selected']['config']}")

    peak = report["peak"]
    analysed = analysed_index(report)
    own = next(r for r in report["candidates"] if tuple(r["index"]) == analysed)
    report["plateau_versus_peak"] = {
        "peak_config": peak["config"],
        "peak_sharpe": peak["sharpe_net"],
        "analysed_config": own["config"],
        "analysed_sharpe": own["sharpe_net"],
        "sharpe_gap": own["sharpe_net"] - peak["sharpe_net"],
    }
    write_json("selection.json", report)


def analysed_index(selection: dict[str, Any]) -> tuple[int, int, int]:
    """La configuration sur laquelle portent les analyses descriptives.

    La retenue si le plateau a désigné quelqu'un ; sinon le centre de la
    grille, et le JSON de sélection dit en toutes lettres que ce n'est pas une
    sélection.
    """
    if selection.get("selected"):
        return tuple(selection["selected"]["index"])
    return GRID_CENTRE


# ═══════════════════════════════════════════════════════════════════════
# ÉTAPE 3 — WALK-FORWARD ANNUEL + CPCV
# ═══════════════════════════════════════════════════════════════════════


def step_walkforward(sample: Sample) -> None:
    sample.guard("wf")
    selection = read_json("selection.json")
    idx = analysed_index(selection)
    _, ind = sample.run(idx)
    trades = ind.trades

    total_net = float(trade_pnl(trades).sum())
    years: list[dict[str, Any]] = []
    for year, chunk in trades.groupby(trades["ts_decision"].dt.year):
        metrics = trade_metrics(chunk)
        years.append(
            {
                "year": int(year),
                **metrics,
                "share_of_total_net": (
                    float(metrics["pnl"] / total_net) if total_net else float("nan")
                ),
            }
        )
    years.sort(key=lambda r: r["year"])

    positive = [y for y in years if y["pnl"] > 0]
    shares = [abs(y["pnl"]) / abs(total_net) for y in years] if total_net else []
    payload = {
        "config": config_label(idx),
        "is_a_selection": bool(selection.get("selected")),
        "spread": SELECTION_SPREAD,
        "total_net": total_net,
        "years": years,
        "n_years": len(years),
        "n_positive_years": len(positive),
        "share_positive_years": (len(positive) / len(years)) if years else float("nan"),
        "max_year_share_of_net": max(shares) if shares else float("nan"),
        "note": (
            "`share_of_total_net` est signée et peut sortir de [0, 1] quand le "
            "net total est négatif ; `max_year_share_of_net` prend les valeurs "
            "absolues, qui est ce que le critère du §1 veut dire."
        ),
    }
    write_json("walkforward.json", payload)

    # ── CPCV sur la grille : distribution OOS des 27 configurations ──
    splitter = build_cpcv_splitter(
        sample.session_dates,
        n_groups=CPCV_N_GROUPS,
        n_test_groups=CPCV_N_TEST_GROUPS,
        purge_td=CPCV_PURGE,
        embargo_pct=CPCV_EMBARGO_PCT,
    )
    print(f"  CPCV : {splitter.n_splits} splits x 27 configurations")
    cv_pipeline = create_cv_pipeline(splitter, spread=SELECTION_SPREAD)
    grid_perf, _ = cv_pipeline(
        sample.raw,
        z=vbt.Param(list(GRID_Z)),
        a_min=vbt.Param(list(GRID_A_MIN)),
        k_s=vbt.Param(list(GRID_K_S)),
    )
    # `attach_bounds="index"` ajoute deux niveaux `start`/`end` à l'index.
    # `cpcv_oos_distribution` prend pour paramètre tout niveau qui n'est pas
    # `split` : les garder ferait 15 x 27 = 405 « configurations » d'un seul
    # point chacune, dont la médiane OOS serait la valeur elle-même.
    bound_levels = [n for n in (grid_perf.index.names or []) if n in ("start", "end")]
    if bound_levels:
        grid_perf = grid_perf.droplevel(bound_levels)
    dist = cpcv_oos_distribution(grid_perf)
    summary = cpcv_summary(
        grid_perf, n_groups=CPCV_N_GROUPS, n_test_groups=CPCV_N_TEST_GROUPS
    )
    if dist.shape[0] != 27:
        raise SystemExit(
            f"CPCV : {dist.shape[0]} configurations agrégées au lieu de 27 — "
            f"niveaux d'index {grid_perf.index.names}"
        )
    oos_values = dist[["mean", "median", "pct_positive"]]
    write_json(
        "cpcv.json",
        {
            "n_groups": CPCV_N_GROUPS,
            "n_test_groups": CPCV_N_TEST_GROUPS,
            "purge": CPCV_PURGE,
            "embargo_pct": CPCV_EMBARGO_PCT,
            "metric": (
                "Sharpe annualisé des rendements M1 du portefeuille "
                "(convention de `create_cv_pipeline`), pas les rendements de "
                "séance de grid27.csv : les deux ne sont comparables qu'en signe"
            ),
            "summary": summary,
            "distribution": dist.reset_index().to_dict(orient="records"),
            "share_of_configs_with_positive_median_oos": float(
                (dist["median"] > 0).mean()
            ),
            "overall_oos_mean": float(oos_values["mean"].mean()),
        },
    )


# ═══════════════════════════════════════════════════════════════════════
# ÉTAPE 4 — ROBUSTESSE
# ═══════════════════════════════════════════════════════════════════════


def step_robustness(sample: Sample) -> None:
    sample.guard("robustness")
    selection = read_json("selection.json")
    idx = analysed_index(selection)
    frame, matrix = load_grid()

    pf_m1, ind = sample.run(idx)
    returns = session_returns(ind.trades, sample.session_dates)
    pf_daily = daily_portfolio(returns)

    n_trials = max(trials.distinct_trials("xau_x10"), CAMPAIGN_TRIALS)
    grid_sharpes = pd.Series(
        frame["sharpe_net"].to_numpy(dtype=np.float64), index=frame["config"]
    )
    benchmark = pd.Series(0.0, index=matrix.index, name="benchmark")

    print(f"  n_trials (registre) = {n_trials} ; matrice {matrix.shape}")
    report = robustness_report(
        pf_daily,
        grid_sharpes=grid_sharpes,
        grid_returns_matrix=matrix,
        n_trials=n_trials,
        benchmark_returns=benchmark,
        n_boot=N_BOOT,
        n_mc=N_MC,
        block_len_mean=BLOCK_LEN_DAILY,
        seed=SEED,
        ann_factor=ANN_DAYS,
        include_equity_paths=False,
        include_mc_trades=False,
    )

    # Le Monte Carlo de trades se fait sur le portefeuille M1 réel : c'est le
    # seul qui porte les 1 700 trades dont on veut permuter l'ordre.
    from framework.mc_trades import (  # noqa: PLC0415 — import local, section dédiée
        mc_max_drawdown_distribution,
        mc_sequence_risk_report,
    )

    mc = mc_max_drawdown_distribution(pf_m1, n_sim=N_MC, mode="shuffle", seed=SEED)
    mc.pop("mdd_samples", None)
    mc.pop("terminal_samples", None)
    sequence = mc_sequence_risk_report(pf_m1, n_sim=N_MC, seed=SEED)

    ruin = ruin_report(
        returns.dropna(),
        label=f"xau_x10 {config_label(idx)}",
        n_boot=N_BOOT,
        block_mean=int(BLOCK_LEN_DAILY),
        ann_factor=ANN_DAYS,
        seed=SEED,
    )

    # Le CSCV rend un logit par combinaison — C(16, 8) = 12 870 valeurs, 500 ko
    # de JSON qui ne se lisent pas. La distribution est résumée, le tableau brut
    # part dans son propre CSV pour qui veut refaire l'histogramme.
    pbo = dict(report.get("pbo") or {})
    logits = np.asarray(pbo.pop("logits", []), dtype=np.float64)
    ranks = np.asarray(pbo.pop("ranks_oos", []), dtype=np.float64)
    if logits.size:
        pd.DataFrame({"logit": logits, "rank_oos": ranks}).to_csv(
            RESULTS_DIR / "pbo_logits.csv", index=False
        )
        finite = logits[np.isfinite(logits)]
        pbo["logit_distribution"] = {
            "n": int(logits.size),
            "n_finite": int(finite.size),
            "median": float(np.median(finite)) if finite.size else float("nan"),
            "q05": float(np.quantile(finite, 0.05)) if finite.size else float("nan"),
            "q95": float(np.quantile(finite, 0.95)) if finite.size else float("nan"),
            "detail": "results/xau_x10/pbo_logits.csv",
        }

    # ⚠ `arch.bootstrap.SPA` attend des **pertes** (plus bas = meilleur) et
    # `reality_check_via_arch` lui passe des **rendements** (plus haut =
    # meilleur). Sur une grille dont tous les rendements moyens sont négatifs,
    # l'inversion fait passer chaque configuration pour meilleure que le
    # benchmark plat et rend p = 0 là où la vérité est p ≈ 1. Défaut
    # préexistant de `src/framework/statistical_testing.py`, hors du périmètre
    # de cette campagne : le chiffre est publié **avec** son avertissement,
    # jamais seul, et il ne porte aucun critère du §1.
    spa = dict(report.get("spa") or {})
    if spa:
        spa["caveat"] = (
            "NON INTERPRÉTABLE EN L'ÉTAT. `reality_check_via_arch` passe des "
            "rendements à `arch.bootstrap.SPA`, qui documente attendre des "
            "pertes : l'orientation est inversée. Sur une grille intégralement "
            "perdante, la p-value rendue (0,0) signifie l'inverse de ce qu'elle "
            "paraît dire. À corriger dans le module, pas ici. Aucun critère de "
            "la table de décision ne dépend de ce test."
        )

    stepm = report.get("stepm")
    if stepm is None:
        stepm = {
            "status": "échec",
            "reason": (
                "`stepm_romano_wolf` lève sur une grille dont aucune "
                "configuration ne bat le benchmark plat : le step-down se "
                "retrouve avec un tableau vide. Limite préexistante du module, "
                "non corrigée ici ; le SPA, lui, a abouti."
            ),
        }

    bootstrap = report.get("bootstrap_df")
    payload = {
        "config": config_label(idx),
        "is_a_selection": bool(selection.get("selected")),
        "n_trials_used_for_deflation": n_trials,
        "seed": SEED,
        "n_boot": N_BOOT,
        "n_mc": N_MC,
        "block_len_mean": BLOCK_LEN_DAILY,
        "ann_factor": ANN_DAYS,
        "returns_basis": "rendements par séance (équité de fin de séance, §2)",
        "psr": report.get("psr"),
        "dsr": report.get("dsr"),
        "haircut": report.get("haircut"),
        "min_backtest_length": report.get("min_backtest_length"),
        "pbo": pbo,
        "spa": spa,
        "stepm": stepm,
        "bootstrap_ci_95": (
            bootstrap.reset_index().to_dict(orient="records")
            if isinstance(bootstrap, pd.DataFrame)
            else None
        ),
        "mc_trades_max_drawdown": mc,
        "sequence_risk": sequence,
        "ruin": ruin.as_row(),
        "config_section": report.get("config"),
    }
    write_json("robustness.json", payload)


# ═══════════════════════════════════════════════════════════════════════
# ÉTAPE 5 — DÉCOMPOSITIONS (aucune sélection ⇒ aucun essai)
# ═══════════════════════════════════════════════════════════════════════


def _cells(trades: pd.DataFrame, keys: pd.Series) -> dict[str, Any]:
    """`n`, E[R], PF, PnL par valeur de `keys`, dans l'ordre des clés."""
    out: dict[str, Any] = {}
    for key, chunk in trades.groupby(keys.to_numpy(), dropna=False):
        out[str(key)] = trade_metrics(chunk)
    return dict(sorted(out.items()))


def _ny_session_label(minutes: pd.Series) -> pd.Series:
    labels = pd.Series("hors séance", index=minutes.index, dtype=object)
    for name, start, end in NY_SESSIONS:
        if start < end:
            mask = (minutes >= start) & (minutes < end)
        else:  # fenêtre qui traverse minuit (Asie)
            mask = (minutes >= start) | (minutes < end)
        labels = labels.mask(mask, name)
    return labels


def step_decompositions(sample: Sample) -> None:
    sample.guard("decomp")
    selection = read_json("selection.json")
    idx = analysed_index(selection)
    _, ind = sample.run(idx)
    trades = ind.trades
    counts = count_summary(ind)

    minutes = trades["ts_decision"].dt.hour * 60 + trades["ts_decision"].dt.minute
    atr = trades["atr"]
    dollars_per_atr = 10.0 / atr
    reversals = trades[trades["scenario_name"].str.startswith("REV")]

    r_candidates = counts["r_candidates_by_year"]
    r_rejected = counts["r_rejected_by_year"]
    rejection = {
        str(int(year)): {
            "candidates": int(r_candidates[year]),
            "rejected": int(r_rejected.get(year, 0)),
            "rejection_rate": float(r_rejected.get(year, 0) / r_candidates[year])
            if r_candidates[year]
            else float("nan"),
        }
        for year in r_candidates.index
    }

    n_time_exits = int((trades["exit_reason_name"] == "TIME").sum())
    payload = {
        "config": config_label(idx),
        "is_a_selection": bool(selection.get("selected")),
        "spread": SELECTION_SPREAD,
        "overall": trade_metrics(trades),
        "by_scenario": _cells(trades, trades["scenario_name"]),
        "by_ny_session": _cells(trades, _ny_session_label(minutes)),
        "by_atr_tercile": _cells(
            trades, pd.qcut(atr, 3, labels=["ATR bas", "ATR moyen", "ATR haut"])
        ),
        "by_dollars_per_atr": _cells(
            trades,
            pd.cut(
                dollars_per_atr,
                [-np.inf, *DOLLARS_PER_ATR_EDGES, np.inf],
                labels=list(DOLLARS_PER_ATR_LABELS),
            ),
        ),
        "by_year": _cells(trades, trades["ts_decision"].dt.year),
        "by_exit_reason": _cells(trades, trades["exit_reason_name"]),
        "reversals_extended_vs_not": _cells(
            reversals,
            reversals["extension_vwap"].map({1.0: "étendu", 0.0: "non étendu"}),
        ),
        "by_dxy_context": _cells(
            trades,
            trades["ctx_dxy"].map({1.0: "favorable", 0.0: "neutre", -1.0: "adverse"}),
        ),
        "by_retest": _cells(
            trades, trades["retest"].map({1.0: "retest", 0.0: "sans retest"})
        ),
        "concentration": concentration(trades),
        "r_rule_rejection_by_year": rejection,
        "r_rule_note": (
            "candidats à R < 1 sur candidats AYANT ATTEINT le test R (§8) : un "
            "refus au contexte n'a pas de r_est et n'entre pas au dénominateur"
        ),
        "time_exits": {
            "n": n_time_exits,
            "share_of_trades": float(n_time_exits / len(trades)) if len(trades) else float("nan"),
            "max_hold_bars": int(MAX_HOLD_BARS),
        },
        "bars_held": {
            "mean": float(trades["bars_held"].mean()) if len(trades) else float("nan"),
            "median": float(trades["bars_held"].median()) if len(trades) else float("nan"),
        },
        "events_by_type": counts["events_by_type"],
        "cancels_by_reason": counts["cancels_by_reason"],
        "context_refusals": counts["ctx_refused"],
    }
    write_json("decompositions.json", payload)


# ═══════════════════════════════════════════════════════════════════════
# ÉTAPE 6 — ABLATIONS (7 essais, logués avant mesure)
# ═══════════════════════════════════════════════════════════════════════


def step_ablations(sample: Sample) -> None:
    sample.guard("ablations")
    selection = read_json("selection.json")
    idx = analysed_index(selection)

    trials.log_trials(
        "xau_x10",
        7,
        note=(
            "ablations EMA50 H1 / VWAP / DXY (7 combinaisons hors la spec) à la "
            f"configuration {config_label(idx)}, in-sample {IS_START} → {IS_END}"
        ),
        config_key="xau_x10:ablations:v1",
    )
    print(f"[trials] xau_x10 distinct = {trials.distinct_trials('xau_x10')}")

    rows: list[dict[str, Any]] = []
    for use_ema, use_vwap, use_dxy in product((True, False), repeat=3):
        flags = {"use_ema": use_ema, "use_vwap": use_vwap, "use_dxy": use_dxy}
        is_spec = all(flags.values())
        summary = sample.summarise(idx, **flags)
        summary.pop("_returns")
        summary.pop("_indicator")
        rows.append({**flags, "is_spec_baseline": is_spec, **summary})
        tag = "spec" if is_spec else "ablation"
        print(
            f"  ema={int(use_ema)} vwap={int(use_vwap)} dxy={int(use_dxy)} "
            f"[{tag:>8}]  sharpe={summary['sharpe_net']:+.3f}  n={summary['n']:>4}  "
            f"E[R]={summary['expectancy_r']:+.4f}"
        )

    write_json(
        "ablations.json",
        {
            "config": config_label(idx),
            "is_a_selection": bool(selection.get("selected")),
            "spread": SELECTION_SPREAD,
            "n_trials_logged": 7,
            "note": (
                "la ligne (True, True, True) est la spec elle-même : elle n'est "
                "pas un essai d'ablation et compte déjà dans la grille des 27"
            ),
            "runs": rows,
        },
    )


# ═══════════════════════════════════════════════════════════════════════
# ÉTAPE 7 — SENSIBILITÉ AUX COÛTS (descriptif, aucune re-sélection)
# ═══════════════════════════════════════════════════════════════════════


def step_costs(sample: Sample) -> None:
    sample.guard("costs")
    selection = read_json("selection.json")
    idx = analysed_index(selection)

    rows: list[dict[str, Any]] = []
    for mult in COST_SPREAD_MULTIPLIERS:
        for slip in COST_SLIPPAGE_PER_SIDE:
            # §12 : le slippage s'applique aux entrées ET aux sorties ; ajouté
            # au spread, il vaut donc 2 x slippage par aller-retour.
            spread = SELECTION_SPREAD * mult + 2.0 * slip
            summary = sample.summarise(idx, spread=spread)
            summary.pop("_returns")
            summary.pop("_indicator")
            rows.append(
                {
                    "spread_multiplier": mult,
                    "slippage_per_side": slip,
                    "effective_spread": spread,
                    **summary,
                }
            )
            print(
                f"  x{mult} + {slip:.2f}$ → spread={spread:.3f}  "
                f"E[R]={summary['expectancy_r']:+.4f}  PF={summary['profit_factor']:.3f}  "
                f"sharpe={summary['sharpe_net']:+.3f}"
            )

    # La grille entière au spread x2 : le plateau survit-il ? Descriptif —
    # aucune sélection n'est rejouée dessus, la table l'interdit (§1.1).
    print("  grille complète au spread x2 (descriptif, aucune re-sélection)")
    doubled = SELECTION_SPREAD * 2.0
    grid_x2 = {}
    for grid_idx in grid_indices():
        summary = sample.summarise(grid_idx, spread=doubled)
        summary.pop("_returns")
        summary.pop("_indicator")
        grid_x2[summary["config"]] = summary

    sharpes_x2 = {
        tuple(v["index"]): float(v["sharpe_net"]) for v in grid_x2.values()
    }
    plateau_x2 = select_plateau_center(sharpes_x2)

    write_json(
        "cost_sensitivity.json",
        {
            "config": config_label(idx),
            "is_a_selection": bool(selection.get("selected")),
            "base_spread": SELECTION_SPREAD,
            "model": (
                "spread constant (table §1.1) ; slippage modélisé en l'ajoutant "
                "au spread à raison de 2 x slippage par aller-retour (§12)"
            ),
            "sensitivities": rows,
            "grid_at_spread_x2": {
                "spread": doubled,
                "purpose": (
                    "descriptif : le plateau survit-il au doublement du coût ? "
                    "Aucune re-sélection — la configuration analysée ne change pas."
                ),
                "n_passing_plateau": plateau_x2["n_passing_plateau"],
                "peak": plateau_x2["peak"],
                "would_select": (
                    plateau_x2["selected"]["config"] if plateau_x2["selected"] else None
                ),
                "configs": grid_x2,
            },
        },
    )


# ═══════════════════════════════════════════════════════════════════════
# ÉTAPE 8 — is_summary.json
# ═══════════════════════════════════════════════════════════════════════


def step_summary(sample: Sample) -> None:
    sample.guard("summary")
    selection = read_json("selection.json")
    walkforward = read_json("walkforward.json")
    robustness = read_json("robustness.json")
    decomp = read_json("decompositions.json")
    costs = read_json("cost_sensitivity.json")
    frame, matrix = load_grid()

    idx = analysed_index(selection)
    label = config_label(idx)
    own = next(r for r in selection["candidates"] if tuple(r["index"]) == idx)
    overall = decomp["overall"]

    x15 = next(
        r
        for r in costs["sensitivities"]
        if r["spread_multiplier"] == 1.5 and r["slippage_per_side"] == 0.0
    )

    dsr_section = robustness.get("dsr") or {}
    pbo_section = robustness.get("pbo") or {}
    n_trials = trials.distinct_trials("xau_x10")

    measures = {
        "n_trades": overall["n"],
        "trades_per_scenario": {
            k: v["n"] for k, v in decomp["by_scenario"].items()
        },
        "expectancy_r": overall["expectancy_r"],
        "profit_factor": overall["profit_factor"],
        "dsr": dsr_section.get("dsr"),
        "pbo": pbo_section.get("pbo"),
        "n_trials_logged": robustness.get("n_trials_used_for_deflation", n_trials),
        "plateau": {
            "share_neighbours_positive": own["share_neighbours_positive"],
            "median_neighbour_sharpe": own["median_neighbour_sharpe"],
            "peak_sharpe": selection["peak"]["sharpe_net"],
            "n_passing_plateau": selection["n_passing_plateau"],
            "selected_exists": selection.get("selected") is not None,
        },
        "walk_forward": {
            "share_positive_years": walkforward["share_positive_years"],
            "max_year_share_of_net": walkforward["max_year_share_of_net"],
        },
        "expectancy_r_spread_x1_5": x15["expectancy_r"],
    }

    table = evaluate_decision_table(measures)

    payload = {
        "strategy": "XAUUSD niveaux x10 (Apogée Invest, stratégie 2)",
        "phase": "campagne in-sample officielle",
        "date": str(pd.Timestamp.today().date()),
        "git_head": git_head(),
        "spec": "docs/specs/xau_x10_spec.md",
        "decision_table_doc": "docs/research/xau_x10_decision_table.md",
        "holdout": {
            "state": "LOCKED",
            "touched_by_this_phase": False,
            **sample.max_dates(),
        },
        "sample": {
            "start": IS_START,
            "end": IS_END,
            "n_m1_bars": int(len(sample.raw)),
            "n_m5_bars": int(len(sample.m5)),
            "n_sessions": int(len(sample.session_dates)),
        },
        "cost_model": {
            "spread_usd": SELECTION_SPREAD,
            "source": "catalog_snapshot 2026-07-28 (table §1.1)",
            "constant": True,
            "slippage_sensitivities_usd_per_side": list(COST_SLIPPAGE_PER_SIDE),
            "spread_multipliers": list(COST_SPREAD_MULTIPLIERS),
        },
        "frozen_parameters": {
            "atr_period": 14,
            "ema_h1_period": 50,
            "v_horizon_bars": 3,
            "m_horizon_bars": 12,
            "v_min": 0.2,
            "k_b": 0.25,
            "n_arm": 12,
            "n_hold": 3,
            "n_sweep": 3,
            "s_min": 0.3,
            "rev_stop_atr": 0.5,
            "r_min": 1.0,
            "max_hold_bars": int(MAX_HOLD_BARS),
            "cooldown_bars": 6,
            "risk_frac": 0.005,
            "risk_frac_dxy_adverse": 0.0025,
            "leverage_cap": 100,
            "init_cash": X10_INIT_CASH,
        },
        "grid": {
            "z": list(GRID_Z),
            "a_min": list(GRID_A_MIN),
            "k_s": list(GRID_K_S),
            "n_configs": int(len(frame)),
            "returns_matrix_shape": list(matrix.shape),
        },
        "n_trials_logged": n_trials,
        "n_trials_used_for_deflation": robustness.get("n_trials_used_for_deflation"),
        "trial_budget_ceiling": 40,
        "selection": {
            "rule": "centre du plateau (table §3)",
            "retained_config": (
                selection["selected"]["config"] if selection.get("selected") else None
            ),
            "n_passing_plateau": selection["n_passing_plateau"],
            "peak_config": selection["peak"]["config"],
            "peak_sharpe_net": selection["peak"]["sharpe_net"],
            "analysed_config": label,
            "analysed_is_a_selection": selection.get("selected") is not None,
            "analysed_sharpe_net": own["sharpe_net"],
            "plateau_versus_peak": selection["plateau_versus_peak"],
        },
        "headline_metrics": {
            "sharpe_net_annualised_252": own["sharpe_net"],
            "n_trades": overall["n"],
            "expectancy_r": overall["expectancy_r"],
            "profit_factor": overall["profit_factor"],
            "net_pnl_usd": overall["pnl"],
            "win_rate": overall["win_rate"],
            "max_drawdown": float(frame.loc[frame["config"] == label, "max_drawdown"].iloc[0]),
        },
        "decision_table": table,
        "produced_by": "scripts/run_xau_x10_research.py",
    }
    write_json("is_summary.json", payload)
    print(f"\n  verdict in-sample PROVISOIRE : {table['provisional_in_sample_verdict']}")
    print(f"  critères en échec : {', '.join(table['failed'])}")


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════

STEPS = {
    "grid": step_grid,
    "select": step_select,
    "wf": step_walkforward,
    "robustness": step_robustness,
    "decomp": step_decompositions,
    "ablations": step_ablations,
    "costs": step_costs,
    "summary": step_summary,
}
ALL_STEPS = tuple(STEPS)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Campagne in-sample XAUUSD x10 (Apogée Invest)"
    )
    parser.add_argument(
        "--step",
        choices=(*ALL_STEPS, "all"),
        default="all",
        help="étape à exécuter ; `all` les enchaîne toutes dans l'ordre",
    )
    args = parser.parse_args(argv)

    steps = ALL_STEPS if args.step == "all" else (args.step,)

    print("=" * 72)
    print(f"XAUUSD x10 — campagne in-sample {IS_START} → {IS_END}")
    print(f"spread de sélection : {SELECTION_SPREAD} $ (constant, table §1.1)")
    print("=" * 72)

    sample = Sample()
    print(
        f"M1 : {len(sample.raw)} barres  {sample.raw.index.min()} → "
        f"{sample.raw.index.max()}"
    )
    print(
        f"M5 : {len(sample.m5)} barres  |  séances : {len(sample.session_dates)}  "
        f"jusqu'au {sample.session_dates.max().date()}"
    )
    for key, value in sample.max_dates().items():
        print(f"  [A4] {key} = {value}")

    for name in steps:
        print(f"\n--- {name} " + "-" * (68 - len(name)))
        STEPS[name](sample)

    print("\nFichiers produits :")
    for path in sorted(RESULTS_DIR.rglob("*")):
        if path.is_file():
            print(f"  {path.relative_to(_REPO)}  {path.stat().st_size:>10} o")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
