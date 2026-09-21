"""Produit les tables et figures du rapport client de la stratégie 2 (XAUUSD x10).

Run:
    uv run python scripts/build_x10_report_assets.py
    uv run python scripts/build_x10_report_assets.py --check

Sorties:
    reports/client/strategie2_xauusd_x10/tables/*.tex    (macros + tables booktabs)
    reports/client/strategie2_xauusd_x10/figures/*.png   (matplotlib, 300 DPI)
    results/xau_x10/feasibility_table.json               (descriptif 10 $/ATR)

Règle du dépôt : **un artefact, un producteur**. Tout nombre publié par le
rapport de la stratégie 2 sort d'ici, et tout nombre qui sort d'ici sort d'un
JSON de ``results/xau_x10/`` — jamais d'un littéral recopié à la main. Les
chiffres de la stratégie 1 ont dérivé pendant trois mois faute de cette règle.

``--check`` ne réécrit rien et rend un code de sortie non nul si un fichier
publié diffère de ce que ce script régénérerait. Le contrôle porte sur les
``tables/*.tex`` — les seuls artefacts qui portent des nombres — et non sur
les PNG, dont la sérialisation matplotlib n'est pas reproductible à l'octet.

Ce que ce script n'a **pas** le droit de faire, et ne fait pas :

- lire une barre postérieure au 2025-12-31 (le holdout est LOCKED) ;
- loguer un essai : la seule configuration réexécutée ici est le centre de la
  grille ``z1_a0.2_k1``, déjà logué sous ``xau_x10:grid27:v1`` ;
- publier un chiffre MT5 ou QuantConnect tant que leur JSON n'existe pas — les
  macros correspondantes valent ``n.d.``.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates  # noqa: E402
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SRC = _PROJECT_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

RESULTS_DIR = _PROJECT_ROOT / "results" / "xau_x10"
OUTPUT_ROOT = _PROJECT_ROOT / "reports" / "client" / "strategie2_xauusd_x10"
FIG_DIR = OUTPUT_ROOT / "figures"
TBL_DIR = OUTPUT_ROOT / "tables"

GOLD_PARQUET = _PROJECT_ROOT / "data" / "XAU-USD_minute_qc.parquet"
FEASIBILITY_JSON = RESULTS_DIR / "feasibility_table.json"

# Le document racine définit ces quatre couleurs ; les figures les reprennent
# pour que rien ne jure entre une capture et le corps du texte.
PALETTE = {
    "primary": "#0B2545",
    "accent": "#CC6B2F",
    "breakout": "#1F5582",
    "reversal": "#CC6B2F",
    "grey": "#4A4A4A",
    "light": "#E6E8EB",
    "green": "#2E8B57",
    "red": "#8E1616",
    "text": "#1A1A1A",
}

# La configuration réexécutée pour les figures de trajectoire et les exemples
# de trades : le centre géométrique de la grille, seul repli descriptif prévu
# au brief de campagne. Ce n'est PAS une sélection (selection.json le dit).
ANALYSED_KEY = "z1_a0.2_k1"
ANALYSED_PARAMS = {"z": 1.0, "a_min": 0.2, "k_s": 1.0}
SELECTION_SPREAD = 0.29
IS_START = "2019-01-01"
IS_END = "2025-12-31"

# Année de référence des exemples de trades. Fixée ici, citée en légende : le
# choix d'un exemple ne doit jamais dépendre de ce qu'on a envie de montrer.
EXAMPLE_YEAR = 2023

SCENARIO_LABELS = {
    "BREAK_LONG": "Breakout Long",
    "BREAK_SHORT": "Breakout Short",
    "REV_LONG": "Reversal Long",
    "REV_SHORT": "Reversal Short",
}
SCENARIO_SLUG = {
    "BREAK_LONG": "break_long",
    "BREAK_SHORT": "break_short",
    "REV_LONG": "rev_long",
    "REV_SHORT": "rev_short",
}

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "axes.edgecolor": PALETTE["grey"],
        "axes.labelcolor": PALETTE["text"],
        "xtick.color": PALETTE["grey"],
        "ytick.color": PALETTE["grey"],
        "axes.grid": True,
        "grid.color": PALETTE["light"],
        "grid.linewidth": 0.6,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "legend.frameon": False,
        "legend.fontsize": 9,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
    }
)


# ═══════════════════════════════════════════════════════════════════════
# 1. FORMATAGE — français, virgule décimale, espace fine de milliers
# ═══════════════════════════════════════════════════════════════════════

UNDETERMINED = "n.d."


def _group(digits: str) -> str:
    """``1723`` → ``1\\,723`` — l'espace fine typographique du français."""
    out = ""
    for i, char in enumerate(reversed(digits)):
        if i and i % 3 == 0:
            out = "\\," + out
        out = char + out
    return out


def _number(value: float, digits: int) -> str:
    """Le corps d'un nombre français, hors mode mathématique."""
    text = f"{abs(value):.{digits}f}"
    whole, _, frac = text.partition(".")
    body = _group(whole)
    if frac:
        body += "{,}" + frac
    return ("-" if value < 0 else "") + body


def fmt_int(value: Any) -> str:
    return f"${_number(float(value), 0)}$"


def fmt_dec(value: Any, digits: int = 3) -> str:
    return f"${_number(float(value), digits)}$"


def fmt_r(value: Any) -> str:
    """Une espérance en R : quatre décimales, comme la note d'analyse."""
    return fmt_dec(value, 4)


def fmt_pct(value: Any, digits: int = 1) -> str:
    """Une fraction (0,312) rendue en pourcentage français (31,2 %)."""
    return f"${_number(float(value) * 100.0, digits)}$\\,\\%"


def fmt_usd(value: Any, digits: int = 2) -> str:
    return f"${_number(float(value), digits)}$~\\$"


def fmt_text(value: Any) -> str:
    return str(value)


def fmt_config(value: Any) -> str:
    """``z1_a0.2_k1`` → ``$z = 1$ ; $a_{\\min} = 0{,}2$ ; $k_s = 1$``."""
    z, a, k = str(value).split("_")
    return (
        f"$z = {_number(float(z[1:]), 1).rstrip('0').rstrip('{,}')}$ ; "
        f"$a_{{\\min}} = {_number(float(a[1:]), 1)}$ ; "
        f"$k_s = {_number(float(k[1:]), 2).rstrip('0').rstrip('{,}')}$"
    )


def latex_escape(text: str) -> str:
    return text.replace("&", "\\&").replace("_", "\\_").replace("%", "\\%")


# Les libellés et seuils viennent des JSON en notation de travail (« >= », « x2 »,
# « 0.35 »). XeLaTeX ne les rendrait pas correctement, et une flèche Unicode
# dépend de la police : tout passe par cette table de réécriture.
_TEX_SUBSTITUTIONS: tuple[tuple[str, str], ...] = (
    ("→", "$\\rightarrow$"),
    (">=", "$\\geq$"),
    ("<=", "$\\leq$"),
    (" x ", " $\\times$ "),
    ("x1,5", "$\\times 1{,}5$"),
    ("+0.1", "$+0{,}10$"),
    ("1.15", "$1{,}15$"),
    ("0.95", "$0{,}95$"),
    ("0.35", "$0{,}35$"),
    ("0.5", "$0{,}5$"),
    # `latex_escape` est passé avant : le signe de pourcentage y est déjà échappé.
    ("60\\%", "$60$\\,\\%"),
    ("50\\%", "$50$\\,\\%"),
    (">", "$>$"),
)


def tex_sentence(text: str) -> str:
    """Un libellé de JSON rendu publiable, sans réécriture manuelle."""
    out = latex_escape(str(text))
    for needle, replacement in _TEX_SUBSTITUTIONS:
        out = out.replace(needle, replacement)
    return out


# ═══════════════════════════════════════════════════════════════════════
# 2. LECTURE DES RÉSULTATS
# ═══════════════════════════════════════════════════════════════════════

# Les JSON obligatoires : leur absence est une erreur, pas un « n.d. ».
REQUIRED_SOURCES = (
    "is_summary",
    "selection",
    "walkforward",
    "cpcv",
    "robustness",
    "decompositions",
    "ablations",
    "cost_sensitivity",
)
# Les JSON qui n'existent pas encore. Toute macro qui en dépend vaut « n.d. »
# tant que la mesure n'a pas été produite — jamais une valeur « en attendant ».
OPTIONAL_SOURCES = ("reconciliation_qc_2024", "mt5_reference")


def load_bundle() -> dict[str, Any]:
    """Tous les résultats lisibles, plus la grille et la table de faisabilité."""
    bundle: dict[str, Any] = {}
    for name in REQUIRED_SOURCES:
        path = RESULTS_DIR / f"{name}.json"
        if not path.is_file():
            raise SystemExit(f"résultat obligatoire absent : {path}")
        bundle[name] = json.loads(path.read_text(encoding="utf-8"))
    for name in OPTIONAL_SOURCES:
        path = RESULTS_DIR / f"{name}.json"
        bundle[name] = (
            json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None
        )
    bundle["grid27"] = pd.read_csv(RESULTS_DIR / "grid27.csv")
    bundle["feasibility"] = (
        json.loads(FEASIBILITY_JSON.read_text(encoding="utf-8"))
        if FEASIBILITY_JSON.is_file()
        else None
    )
    return bundle


def read_path(obj: Any, path: str) -> Any:
    """Résout ``a.b.0.c`` dans une structure JSON déjà chargée."""
    current = obj
    for part in path.split("."):
        if isinstance(current, list):
            current = current[int(part)]
        else:
            current = current[part]
    return current


# ═══════════════════════════════════════════════════════════════════════
# 3. LA TABLE MACRO → SOURCE
# ═══════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class MacroSource:
    """D'où vient une macro publiée, et comment elle se met en forme.

    ``source`` nomme le JSON de ``results/xau_x10/``. ``path`` y désigne la
    valeur par un chemin pointé ; quand le chiffre publié est une combinaison
    (une part, une pente), ``derive`` prend le relais et reçoit le *bundle*
    complet. ``source = "git"`` désigne une valeur lue dans l'historique du
    dépôt, pas dans un résultat.

    ``tests/test_x10_report_coherence.py`` importe cette table et refait le
    chemin lui-même : une macro publiée qui ne redescend pas sur sa source
    fait rougir la suite.
    """

    source: str
    fmt: Callable[[Any], str]
    path: str | None = None
    derive: Callable[[dict[str, Any]], Any] | None = None

    def value(self, bundle: dict[str, Any]) -> Any:
        if self.derive is not None:
            return self.derive(bundle)
        return read_path(bundle[self.source], self.path or "")


def _pct1(value: Any) -> str:
    return fmt_pct(value, 1)


def _pct2(value: Any) -> str:
    return fmt_pct(value, 2)


def _dec1(value: Any) -> str:
    return fmt_dec(value, 1)


def _dec2(value: Any) -> str:
    return fmt_dec(value, 2)


def _dec3(value: Any) -> str:
    return fmt_dec(value, 3)


def _dec4(value: Any) -> str:
    return fmt_dec(value, 4)


def _usd0(value: Any) -> str:
    return fmt_usd(value, 0)


def _usd2(value: Any) -> str:
    return fmt_usd(value, 2)


def _exit_share(reason: str) -> Callable[[dict[str, Any]], float]:
    def _inner(bundle: dict[str, Any]) -> float:
        decomp = bundle["decompositions"]
        return decomp["by_exit_reason"][reason]["n"] / decomp["overall"]["n"]

    return _inner


def _breakeven_target_rate(bundle: dict[str, Any]) -> float:
    """Le taux de cible qui annulerait l'espérance, aux deux R mesurés."""
    by_exit = bundle["decompositions"]["by_exit_reason"]
    loss = abs(by_exit["STOP"]["expectancy_r"])
    gain = by_exit["TARGET"]["expectancy_r"]
    return loss / (gain + loss)


def _worst_grid_sharpe(bundle: dict[str, Any]) -> float:
    return float(bundle["grid27"]["sharpe_net"].min())


def _asia_loss_share(bundle: dict[str, Any]) -> float:
    decomp = bundle["decompositions"]
    return decomp["by_ny_session"]["Asie 18:00-03:00"]["pnl"] / decomp["overall"]["pnl"]


def _asia_trade_share(bundle: dict[str, Any]) -> float:
    decomp = bundle["decompositions"]
    return decomp["by_ny_session"]["Asie 18:00-03:00"]["n"] / decomp["overall"]["n"]


def _rev_short_loss_share(bundle: dict[str, Any]) -> float:
    decomp = bundle["decompositions"]
    return decomp["by_scenario"]["REV_SHORT"]["pnl"] / decomp["overall"]["pnl"]


def _dxy_savings(bundle: dict[str, Any]) -> float:
    """Ce que le garde-fou DXY épargne : spec moins ablation ``use_dxy=False``."""
    runs = {
        (r["use_ema"], r["use_vwap"], r["use_dxy"]): r for r in bundle["ablations"]["runs"]
    }
    return runs[(True, True, True)]["pnl"] - runs[(True, True, False)]["pnl"]


def _ablation_all_off_sharpe(bundle: dict[str, Any]) -> float:
    runs = {
        (r["use_ema"], r["use_vwap"], r["use_dxy"]): r for r in bundle["ablations"]["runs"]
    }
    return runs[(False, False, False)]["sharpe_net"]


def _cost_slope_per_dime(bundle: dict[str, Any]) -> float:
    """Combien d'espérance en R coûtent 0,10 $ de plus par aller-retour.

    Pente entre les deux points de slippage du ×1, soit 0,20 $ d'écart de
    spread effectif, ramenée à 0,10 $.
    """
    rows = bundle["cost_sensitivity"]["sensitivities"]
    base = next(r for r in rows if r["spread_multiplier"] == 1.0 and r["slippage_per_side"] == 0.0)
    high = next(r for r in rows if r["spread_multiplier"] == 1.0 and r["slippage_per_side"] == 0.1)
    delta = high["effective_spread"] - base["effective_spread"]
    return abs(high["expectancy_r"] - base["expectancy_r"]) / delta * 0.10


def _zero_cost_extrapolation(bundle: dict[str, Any]) -> float:
    """Extrapolation linéaire de l'espérance à coût nul — indicative, non mesurée."""
    rows = bundle["cost_sensitivity"]["sensitivities"]
    base = next(r for r in rows if r["spread_multiplier"] == 1.0 and r["slippage_per_side"] == 0.0)
    return base["expectancy_r"] + _cost_slope_per_dime(bundle) * base["effective_spread"] / 0.10


def _trial_reserve(bundle: dict[str, Any]) -> int:
    summary = bundle["is_summary"]
    return int(summary["trial_budget_ceiling"]) - int(summary["n_trials_logged"])


def _n_failing(bundle: dict[str, Any]) -> int:
    table = bundle["is_summary"]["decision_table"]
    return int(table["n_criteria"]) - int(table["n_passing"])


def _cpcv_positive_share(bundle: dict[str, Any]) -> float:
    return float(bundle["cpcv"]["share_of_configs_with_positive_median_oos"])


def _boot(metric: str, field: str) -> Callable[[dict[str, Any]], float]:
    def _inner(bundle: dict[str, Any]) -> float:
        row = next(b for b in bundle["robustness"]["bootstrap_ci_95"] if b["metric"] == metric)
        return row[field]

    return _inner


def _feasibility_year(year: int, field: str) -> Callable[[dict[str, Any]], float]:
    def _inner(bundle: dict[str, Any]) -> float:
        rows = bundle["feasibility"]["years"]
        return next(r for r in rows if r["year"] == year)[field]

    return _inner


def _git_freeze_commit(_bundle: dict[str, Any]) -> str:
    """Le premier commit du document de table de décision — le hash du gel."""
    try:
        out = subprocess.run(
            [
                "git",
                "log",
                "--diff-filter=A",
                "--format=%h",
                "--",
                "docs/research/xau_x10_decision_table.md",
            ],
            cwd=_PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.split()
    except Exception:  # pragma: no cover — dépôt absent
        return UNDETERMINED
    return out[-1] if out else UNDETERMINED


MACRO_SOURCES: dict[str, MacroSource] = {
    # ── Tête de rapport ────────────────────────────────────────────────
    "XTenTradesIS": MacroSource("is_summary", fmt_int, "headline_metrics.n_trades"),
    "XTenExpectancyR": MacroSource("is_summary", fmt_r, "headline_metrics.expectancy_r"),
    "XTenProfitFactor": MacroSource("is_summary", _dec3, "headline_metrics.profit_factor"),
    "XTenSharpeIS": MacroSource("is_summary", _dec3, "headline_metrics.sharpe_net_annualised_252"),
    "XTenDSR": MacroSource("robustness", _dec3, "dsr.dsr"),
    "XTenPBO": MacroSource("robustness", _dec4, "pbo.pbo"),
    "XTenMaxDDPct": MacroSource("is_summary", _pct1, "headline_metrics.max_drawdown"),
    "XTenTrialsCount": MacroSource("is_summary", fmt_int, "n_trials_logged"),
    "XTenWinRate": MacroSource("is_summary", _pct1, "headline_metrics.win_rate"),
    "XTenNetPnl": MacroSource("is_summary", _usd2, "headline_metrics.net_pnl_usd"),
    # ── Exécution : rien tant que les moteurs n'ont pas rendu ──────────
    # Ces cinq chemins sont des emplacements réservés, PAS un contrat arrêté :
    # les fichiers de réconciliation et de référence MT5 appartiennent à une
    # autre chaîne de production, dont le schéma n'est pas figé. Tant qu'un
    # chemin ne résout pas, la macro vaut « n.d. » et le document porte son
    # bandeau « sections d'exécution en cours d'intégration ». Publier une
    # valeur devinée serait pire que ne rien publier.
    "XTenTradesOOS": MacroSource("oos_2026", fmt_int, "n_trades"),
    "XTenMTFiveExpectancyR": MacroSource("mt5_reference", fmt_r, "headline.expectancy_r"),
    "XTenQCExpectancyR": MacroSource("reconciliation_qc_2024", fmt_r, "headline.expectancy_r"),
    "XTenQCMatchRate": MacroSource("reconciliation_qc_2024", _pct1, "entries.match_rate"),
    "XTenMTFiveMatchRate": MacroSource("mt5_reference", _pct1, "entries.match_rate"),
    # ── Échantillon et budget d'essais ─────────────────────────────────
    "XTenSampleSessions": MacroSource("is_summary", fmt_int, "sample.n_sessions"),
    "XTenSampleBarsMFive": MacroSource("is_summary", fmt_int, "sample.n_m5_bars"),
    "XTenSampleBarsMOne": MacroSource("is_summary", fmt_int, "sample.n_m1_bars"),
    "XTenTrialsCeiling": MacroSource("is_summary", fmt_int, "trial_budget_ceiling"),
    "XTenTrialsReserve": MacroSource("is_summary", fmt_int, derive=_trial_reserve),
    "XTenSpreadUsd": MacroSource("is_summary", _usd2, "cost_model.spread_usd"),
    "XTenGitHead": MacroSource("is_summary", lambda v: str(v)[:7], "git_head"),
    "XTenFreezeCommit": MacroSource("git", fmt_text, derive=_git_freeze_commit),
    # ── Table de décision ──────────────────────────────────────────────
    "XTenCriteriaCount": MacroSource("is_summary", fmt_int, "decision_table.n_criteria"),
    "XTenCriteriaPassing": MacroSource("is_summary", fmt_int, "decision_table.n_passing"),
    "XTenCriteriaFailing": MacroSource("is_summary", fmt_int, derive=_n_failing),
    # ── Grille, plateau, sélection ─────────────────────────────────────
    "XTenPeakConfig": MacroSource("selection", fmt_config, "peak.config"),
    "XTenPeakSharpe": MacroSource("selection", _dec3, "peak.sharpe_net"),
    "XTenAnalysedConfig": MacroSource(
        "selection", fmt_config, "fallback_for_descriptive_analyses.config"
    ),
    "XTenWorstGridSharpe": MacroSource("grid27", _dec3, derive=_worst_grid_sharpe),
    "XTenPlateauPassing": MacroSource("selection", fmt_int, "n_passing_plateau"),
    "XTenPlateauThreshold": MacroSource("selection", _dec3, "plateau_median_threshold"),
    "XTenPlateauMedianNeighbour": MacroSource(
        "is_summary", _dec3, "decision_table.criteria.plateau.measured.median_neighbour_sharpe"
    ),
    "XTenPlateauShareNeighbours": MacroSource(
        "is_summary", _pct1, "decision_table.criteria.plateau.measured.share_neighbours_positive"
    ),
    "XTenSharpeGapPeak": MacroSource("selection", _dec3, "plateau_versus_peak.sharpe_gap"),
    # ── Walk-forward ───────────────────────────────────────────────────
    "XTenYearsCount": MacroSource("walkforward", fmt_int, "n_years"),
    "XTenPositiveYears": MacroSource("walkforward", fmt_int, "n_positive_years"),
    "XTenMaxYearShare": MacroSource("walkforward", _pct1, "max_year_share_of_net"),
    "XTenFirstYearExpectancy": MacroSource("walkforward", fmt_r, "years.0.expectancy_r"),
    "XTenLastYearExpectancy": MacroSource("walkforward", fmt_r, "years.6.expectancy_r"),
    "XTenWorstYearExpectancy": MacroSource("walkforward", fmt_r, "years.1.expectancy_r"),
    # ── Décompositions ─────────────────────────────────────────────────
    "XTenBestScenarioR": MacroSource("decompositions", fmt_r, "by_scenario.BREAK_LONG.expectancy_r"),
    "XTenWorstScenarioR": MacroSource("decompositions", fmt_r, "by_scenario.REV_SHORT.expectancy_r"),
    "XTenRevShortLossShare": MacroSource("decompositions", _pct1, derive=_rev_short_loss_share),
    "XTenAsiaExpectancyR": MacroSource(
        "decompositions", fmt_r, "by_ny_session.Asie 18:00-03:00.expectancy_r"
    ),
    "XTenNewYorkExpectancyR": MacroSource(
        "decompositions", fmt_r, "by_ny_session.New York 08:00-17:00.expectancy_r"
    ),
    "XTenAsiaLossShare": MacroSource("decompositions", _pct1, derive=_asia_loss_share),
    "XTenAsiaTradeShare": MacroSource("decompositions", _pct1, derive=_asia_trade_share),
    "XTenBestDollarBandR": MacroSource(
        "decompositions", fmt_r, "by_dollars_per_atr.6-10.expectancy_r"
    ),
    "XTenWorstDollarBandR": MacroSource(
        "decompositions", fmt_r, "by_dollars_per_atr.>10.expectancy_r"
    ),
    "XTenStopR": MacroSource("decompositions", fmt_r, "by_exit_reason.STOP.expectancy_r"),
    "XTenTargetR": MacroSource("decompositions", fmt_r, "by_exit_reason.TARGET.expectancy_r"),
    "XTenTimeR": MacroSource("decompositions", fmt_r, "by_exit_reason.TIME.expectancy_r"),
    "XTenSessionR": MacroSource("decompositions", fmt_r, "by_exit_reason.SESSION.expectancy_r"),
    "XTenStopShare": MacroSource("decompositions", _pct1, derive=_exit_share("STOP")),
    "XTenTargetShare": MacroSource("decompositions", _pct1, derive=_exit_share("TARGET")),
    "XTenTimeShare": MacroSource("decompositions", _pct1, "time_exits.share_of_trades"),
    "XTenSessionShare": MacroSource("decompositions", _pct1, derive=_exit_share("SESSION")),
    "XTenBreakevenTargetRate": MacroSource(
        "decompositions", _pct1, derive=_breakeven_target_rate
    ),
    "XTenBarsHeldMean": MacroSource("decompositions", _dec2, "bars_held.mean"),
    "XTenBarsHeldMedian": MacroSource("decompositions", fmt_int, "bars_held.median"),
    "XTenExtendedR": MacroSource(
        "decompositions", fmt_r, "reversals_extended_vs_not.étendu.expectancy_r"
    ),
    "XTenNotExtendedR": MacroSource(
        "decompositions", fmt_r, "reversals_extended_vs_not.non étendu.expectancy_r"
    ),
    "XTenRetestR": MacroSource("decompositions", fmt_r, "by_retest.retest.expectancy_r"),
    "XTenNoRetestR": MacroSource("decompositions", fmt_r, "by_retest.sans retest.expectancy_r"),
    "XTenDxyFavourableR": MacroSource(
        "decompositions", fmt_r, "by_dxy_context.favorable.expectancy_r"
    ),
    "XTenDxyAdverseR": MacroSource("decompositions", fmt_r, "by_dxy_context.adverse.expectancy_r"),
    "XTenArmCount": MacroSource("decompositions", fmt_int, "events_by_type.ARM"),
    "XTenCancelCount": MacroSource("decompositions", fmt_int, "events_by_type.CANCEL"),
    "XTenContextRefusals": MacroSource("decompositions", fmt_int, "context_refusals"),
    "XTenGrossProfit": MacroSource("decompositions", _usd2, "concentration.gross_profit"),
    "XTenTopFiveDaysShare": MacroSource(
        "decompositions", _pct1, "concentration.share_of_gross_profit_top5_days"
    ),
    "XTenTopTenTradesShare": MacroSource(
        "decompositions", _pct1, "concentration.share_of_gross_profit_top10_trades"
    ),
    "XTenProfitableSessions": MacroSource(
        "decompositions", fmt_int, "concentration.n_profitable_days"
    ),
    "XTenLosingSessions": MacroSource("decompositions", fmt_int, "concentration.n_losing_days"),
    "XTenBestMonthPnl": MacroSource("decompositions", _usd2, "concentration.pnl_best_month"),
    "XTenRRejectFirstYear": MacroSource(
        "decompositions", _pct1, "r_rule_rejection_by_year.2019.rejection_rate"
    ),
    "XTenRRejectLastYear": MacroSource(
        "decompositions", _pct1, "r_rule_rejection_by_year.2025.rejection_rate"
    ),
    # ── Ablations ──────────────────────────────────────────────────────
    "XTenAblationAllOffSharpe": MacroSource(
        "ablations", _dec3, derive=_ablation_all_off_sharpe
    ),
    "XTenDxySavings": MacroSource("ablations", _usd2, derive=_dxy_savings),
    # ── Coûts ──────────────────────────────────────────────────────────
    "XTenExpectancySpreadOneFive": MacroSource(
        "cost_sensitivity", fmt_r, "sensitivities.3.expectancy_r"
    ),
    "XTenExpectancySpreadTwo": MacroSource(
        "cost_sensitivity", fmt_r, "sensitivities.6.expectancy_r"
    ),
    "XTenSharpeSpreadTwo": MacroSource("cost_sensitivity", _dec3, "sensitivities.6.sharpe_net"),
    "XTenCostSlopePerDime": MacroSource("cost_sensitivity", _dec3, derive=_cost_slope_per_dime),
    "XTenZeroCostExtrapolation": MacroSource(
        "cost_sensitivity", _dec3, derive=_zero_cost_extrapolation
    ),
    "XTenPeakSharpeSpreadTwo": MacroSource(
        "cost_sensitivity", _dec3, "grid_at_spread_x2.peak.sharpe_net"
    ),
    "XTenPlateauPassingSpreadTwo": MacroSource(
        "cost_sensitivity", fmt_int, "grid_at_spread_x2.n_passing_plateau"
    ),
    # ── Robustesse ─────────────────────────────────────────────────────
    "XTenPSR": MacroSource("robustness", _dec3, "psr.psr"),
    "XTenExpectedMaxSharpe": MacroSource("robustness", _dec3, "dsr.expected_max_sharpe"),
    "XTenHaircutSharpe": MacroSource("robustness", _dec3, "haircut.haircut_sharpe"),
    "XTenHaircutPValue": MacroSource("robustness", _dec3, "haircut.adjusted_pvalue"),
    "XTenPBOSplits": MacroSource("robustness", fmt_int, "pbo.n_splits"),
    "XTenSharpeCILow": MacroSource("robustness", _dec3, derive=_boot("sharpe_ratio", "ci_low")),
    "XTenSharpeCIHigh": MacroSource("robustness", _dec3, derive=_boot("sharpe_ratio", "ci_high")),
    "XTenTotalReturn": MacroSource("robustness", _pct1, derive=_boot("total_return", "observed")),
    "XTenTotalReturnCILow": MacroSource(
        "robustness", _pct1, derive=_boot("total_return", "ci_low")
    ),
    "XTenTotalReturnCIHigh": MacroSource(
        "robustness", _pct1, derive=_boot("total_return", "ci_high")
    ),
    "XTenMCObservedMdd": MacroSource("robustness", _pct1, "mc_trades_max_drawdown.observed_mdd"),
    "XTenMCMddMedian": MacroSource("robustness", _pct1, "mc_trades_max_drawdown.mdd_p50"),
    "XTenMCMddPNinetyFive": MacroSource("robustness", _pct1, "mc_trades_max_drawdown.mdd_p95"),
    "XTenSequenceZ": MacroSource("robustness", _dec2, "sequence_risk.sequence_luck_zscore"),
    "XTenUnderwaterMedian": MacroSource("robustness", fmt_int, "mc_trades_max_drawdown.uw_p50"),
    "XTenMCTrades": MacroSource("robustness", fmt_int, "mc_trades_max_drawdown.n_trades"),
    "XTenMCSims": MacroSource("robustness", fmt_int, "mc_trades_max_drawdown.n_sim"),
    "XTenBootReplicates": MacroSource("robustness", fmt_int, "n_boot"),
    "XTenAnnReturn": MacroSource("robustness", _pct1, "ruin.ann_return"),
    "XTenAnnVol": MacroSource("robustness", _pct1, "ruin.ann_vol"),
    "XTenRuinProb": MacroSource("robustness", _pct1, "ruin.P(ruin)"),
    "XTenLossFiftyProb": MacroSource("robustness", _pct1, "ruin.P(loss>50%)"),
    "XTenTerminalMedian": MacroSource("robustness", _dec3, "ruin.terminal_p50"),
    "XTenBootDDPNinetyFive": MacroSource("robustness", _pct1, "ruin.boot_dd_p95"),
    "XTenRecoveryDays": MacroSource("robustness", fmt_int, "ruin.recovery_days"),
    # ── CPCV ───────────────────────────────────────────────────────────
    "XTenCPCVSplits": MacroSource("cpcv", fmt_int, "summary.n_splits_expected"),
    "XTenCPCVBestMedian": MacroSource("cpcv", _dec3, "summary.best_median"),
    "XTenCPCVMean": MacroSource("cpcv", _dec3, "overall_oos_mean"),
    "XTenCPCVPositiveShare": MacroSource("cpcv", _pct1, derive=_cpcv_positive_share),
    # ── Faisabilité : le pas de grille mesuré en ATR ───────────────────
    "XTenDollarsPerAtrFirst": MacroSource(
        "feasibility", _dec1, derive=_feasibility_year(2019, "dollars_per_atr")
    ),
    "XTenDollarsPerAtrLast": MacroSource(
        "feasibility", _dec1, derive=_feasibility_year(2025, "dollars_per_atr")
    ),
    "XTenCrossingsFirst": MacroSource(
        "feasibility", fmt_int, derive=_feasibility_year(2019, "crossings_per_day_median")
    ),
    "XTenCrossingsLast": MacroSource(
        "feasibility", fmt_int, derive=_feasibility_year(2025, "crossings_per_day_median")
    ),
    "XTenAtrFirst": MacroSource(
        "feasibility", _dec3, derive=_feasibility_year(2019, "atr_m5_median")
    ),
    "XTenAtrLast": MacroSource(
        "feasibility", _dec3, derive=_feasibility_year(2025, "atr_m5_median")
    ),
}

# Les macros dont la valeur ne redescend pas sur un JSON de résultats : le test
# les contrôle autrement (forme du hash), pas par égalité à une source.
NON_RESULT_MACROS = frozenset({"XTenFreezeCommit"})

# Les macros que seuls les moteurs d'exécution peuvent remplir. Tant que l'une
# d'elles vaut « n.d. », le document garde son bandeau : le verdict in-sample
# est acquis, mais les sections d'exécution ne sont pas intégrées.
EXECUTION_MACROS: tuple[str, ...] = (
    "XTenTradesOOS",
    "XTenMTFiveExpectancyR",
    "XTenQCExpectancyR",
    "XTenQCMatchRate",
    "XTenMTFiveMatchRate",
)


def build_macros(bundle: dict[str, Any]) -> dict[str, str]:
    """Chaque macro, ou ``n.d.`` quand sa source n'existe pas encore."""
    # Un nom de macro TeX ne contient que des lettres : `\XTenSampleM5Bars` se
    # lirait `\XTenSampleM` suivi du texte « 5Bars », et le document cesserait
    # de compiler dix sections plus loin, sur un message sans rapport.
    illegal = sorted(name for name in MACRO_SOURCES if not name.isalpha())
    if illegal:
        raise SystemExit(f"noms de macro non alphabétiques : {illegal}")
    out: dict[str, str] = {}
    for name, spec in MACRO_SOURCES.items():
        if name in NON_RESULT_MACROS:
            out[name] = spec.fmt(spec.value(bundle))
            continue
        if spec.source not in bundle or bundle.get(spec.source) is None:
            out[name] = UNDETERMINED
            continue
        try:
            out[name] = spec.fmt(spec.value(bundle))
        except (KeyError, IndexError, TypeError):
            out[name] = UNDETERMINED
    return out


# ═══════════════════════════════════════════════════════════════════════
# 4. TABLE DE FAISABILITÉ — 10 $/ATR par année, recalculée
# ═══════════════════════════════════════════════════════════════════════


def build_feasibility_table() -> dict[str, Any]:
    """Le pas de grille rapporté à l'ATR M5, année par année, 2019 → 2025.

    Purement descriptif : aucune performance, aucun signal, aucune barre
    postérieure au 2025-12-31. La méthode est celle de la note de faisabilité —
    agrégation M5, ATR de Wilder 14, un « jour » = une date UTC portant au moins
    100 barres M5, un « croisement » = changement de ``floor(close/10)`` entre
    deux clôtures M5 consécutives — mais les nombres sont recalculés ici, pour
    que le rapport n'en recopie aucun.
    """
    from framework.x10_context import atr_wilder, resample_ohlc

    raw = pd.read_parquet(GOLD_PARQUET)
    raw = raw.rename(columns=str.lower)[["open", "high", "low", "close"]]
    raw = raw.loc[: pd.Timestamp(IS_END, tz=raw.index.tz) + pd.Timedelta(days=1)]
    m5 = resample_ohlc(raw, "5min")
    frame = pd.DataFrame({"close": m5["close"], "atr": atr_wilder(m5)})
    frame["date"] = pd.DatetimeIndex(frame.index).date
    frame["year"] = pd.DatetimeIndex(frame.index).year

    # Un jour incomplet (veille de Noël, coupure) fausserait la médiane des
    # croisements : le seuil de 100 barres M5 les écarte.
    counts = frame.groupby("date").size()
    frame = frame[frame["date"].isin(set(counts[counts >= 100].index))]

    crossings = np.floor(frame["close"] / 10.0).diff().fillna(0.0) != 0.0
    frame = frame.assign(crossing=crossings.astype(int))

    years = []
    for year, group in frame.groupby("year"):
        if not 2019 <= int(year) <= 2025:
            continue
        atr_median = float(group["atr"].median())
        per_day = group.groupby("date")["crossing"].sum()
        years.append(
            {
                "year": int(year),
                "n_days": int(group["date"].nunique()),
                "price_median": float(group["close"].median()),
                "atr_m5_median": atr_median,
                "dollars_per_atr": 10.0 / atr_median,
                "spread_per_atr": SELECTION_SPREAD / atr_median,
                "crossings_per_day_median": float(per_day.median()),
                "crossings_per_day_mean": float(per_day.mean()),
            }
        )
    return {
        "source": str(GOLD_PARQUET.relative_to(_PROJECT_ROOT)),
        "window": {"start": IS_START, "end": IS_END},
        "method": (
            "agrégation M5 (label=left, closed=left) ; ATR de Wilder 14 sur M5 ; "
            "jour = date UTC portant >= 100 barres M5 ; croisement = changement "
            "de floor(close/10) entre deux clôtures M5 consécutives ; médianes annuelles"
        ),
        "scope": "descriptif — aucune performance, aucune barre >= 2026-01-01",
        "spread_usd": SELECTION_SPREAD,
        "produced_by": "scripts/build_x10_report_assets.py",
        "years": years,
    }


# ═══════════════════════════════════════════════════════════════════════
# 5. TABLES LATEX
# ═══════════════════════════════════════════════════════════════════════

GENERATED_HEADER = (
    "% GÉNÉRÉ par scripts/build_x10_report_assets.py — ne pas éditer à la main.\n"
    "% Toute valeur vient de results/xau_x10/*.json.\n"
)


def tex_table(
    *,
    header: str,
    rows: list[str],
    caption: str,
    label: str,
    col_spec: str,
    size: str = "\\small",
    placement: str = "htbp",
) -> str:
    body = "\n".join(rows)
    return (
        f"\\begin{{table}}[{placement}]\n"
        f"\\centering\n"
        f"{size}\n"
        f"\\caption{{{caption}}}\n"
        f"\\label{{{label}}}\n"
        f"\\begin{{tabular}}{{{col_spec}}}\n"
        f"\\toprule\n{header}\n\\midrule\n{body}\n\\bottomrule\n"
        f"\\end{{tabular}}\n"
        f"\\end{{table}}\n"
    )


def _verdict_cell(passed: bool) -> str:
    return (
        "\\textcolor{rsiGreen}{\\textbf{GO}}"
        if passed
        else "\\textcolor{combinedBurgundy}{\\textbf{échec}}"
    )


def _decision_measure(key: str, crit: dict[str, Any]) -> str:
    """La colonne « mesuré » de la table de décision, critère par critère."""
    measured = crit["measured"]
    if key == "trades_in_sample":
        return f"{fmt_int(measured)} trades"
    if key == "expectancy_and_profit_factor":
        return f"{fmt_r(measured['expectancy_r'])} R / PF {fmt_dec(measured['profit_factor'], 3)}"
    if key == "dsr":
        return fmt_dec(measured, 3)
    if key == "pbo":
        return fmt_dec(measured, 4)
    if key == "plateau":
        return (
            f"{fmt_pct(measured['share_neighbours_positive'], 0)} de voisins $> 0$ ; "
            f"médiane {fmt_dec(measured['median_neighbour_sharpe'], 3)}"
        )
    if key == "walk_forward":
        return (
            f"{fmt_pct(measured['share_positive_years'], 0)} d'années positives ; "
            f"max {fmt_pct(measured['max_year_share_of_net'], 1)}"
        )
    if key == "spread_x1_5":
        return f"{fmt_r(measured)} R"
    return "\\emph{non mesuré}"


def table_decision(bundle: dict[str, Any]) -> str:
    table = bundle["is_summary"]["decision_table"]
    rows = []
    for i, (key, crit) in enumerate(table["criteria"].items(), start=1):
        rows.append(
            f"{i} & {tex_sentence(crit['label'])} & {tex_sentence(crit['threshold'])} & "
            f"{_decision_measure(key, crit)} & {_verdict_cell(bool(crit['pass']))} \\\\"
        )
    return tex_table(
        header="\\textbf{\\#} & \\textbf{Critère} & \\textbf{Seuil GO} & "
        "\\textbf{Mesuré} & \\textbf{Verdict} \\\\",
        rows=rows,
        caption="Table de décision gelée avant mesure, confrontée aux résultats "
        "de la campagne in-sample. Un critère non mesuré vaut échec, jamais "
        "« non applicable ».",
        label="tab:x10_decision",
        col_spec="c p{4.3cm} p{3.3cm} p{3.6cm} c",
        size="\\footnotesize",
    )


def table_grid27(bundle: dict[str, Any]) -> str:
    grid = bundle["grid27"].sort_values("sharpe_net", ascending=False)
    rows = []
    for _, row in grid.iterrows():
        mark = "~$\\star$" if row["config"] == ANALYSED_KEY else ""
        rows.append(
            f"{fmt_dec(row['z'], 1)} & {fmt_dec(row['a_min'], 1)} & "
            f"{fmt_dec(row['k_s'], 2)} & {fmt_dec(row['sharpe_net'], 3)} & "
            f"{fmt_int(row['n'])} & {fmt_r(row['expectancy_r'])} & "
            f"{fmt_dec(row['profit_factor'], 3)} & {fmt_pct(row['max_drawdown'], 1)}{mark} \\\\"
        )
    return tex_table(
        header="$z$ & $a_{\\min}$ & $k_s$ & \\textbf{Sharpe net} & \\textbf{Trades} & "
        "\\textbf{E[R]} & \\textbf{PF} & \\textbf{Repli max} \\\\",
        rows=rows,
        caption="Les vingt-sept configurations de la grille, classées par Sharpe "
        "net décroissant, au spread constant de sélection. L'étoile marque le "
        "centre géométrique, support des décompositions~: ce n'est pas une "
        "configuration retenue.",
        label="tab:x10_grid27",
        col_spec="r r r r r r r r",
        size="\\footnotesize",
    )


def _cell_rows(cells: dict[str, dict[str, Any]], order: list[str] | None = None) -> list[str]:
    keys = order if order is not None else list(cells)
    rows = []
    for key in keys:
        cell = cells[key]
        pf = cell["profit_factor"]
        pf_text = "$\\infty$" if pf is None or not np.isfinite(pf) else fmt_dec(pf, 3)
        rows.append(
            f"{latex_escape(key)} & {fmt_int(cell['n'])} & {fmt_r(cell['expectancy_r'])} & "
            f"{pf_text} & {fmt_usd(cell['pnl'], 1)} & {fmt_pct(cell['win_rate'], 1)} \\\\"
        )
    return rows


_DECOMP_HEADER = (
    "\\textbf{Coupe} & \\textbf{n} & \\textbf{E[R]} & \\textbf{PF} & "
    "\\textbf{PnL} & \\textbf{Réussite} \\\\"
)
_DECOMP_SPEC = "l r r r r r"


def table_scenarios(bundle: dict[str, Any]) -> str:
    cells = bundle["decompositions"]["by_scenario"]
    order = ["BREAK_LONG", "BREAK_SHORT", "REV_LONG", "REV_SHORT"]
    rows = _cell_rows({SCENARIO_LABELS[k]: cells[k] for k in order})
    return tex_table(
        header=_DECOMP_HEADER,
        rows=rows,
        caption="Décomposition par scénario. Les quatre dépassent le seuil de "
        "commentabilité de la table de décision, et les quatre perdent.",
        label="tab:x10_scenarios",
        col_spec=_DECOMP_SPEC,
    )


def table_sessions(bundle: dict[str, Any]) -> str:
    cells = bundle["decompositions"]["by_ny_session"]
    order = ["Asie 18:00-03:00", "Londres 03:00-08:00", "New York 08:00-17:00"]
    return tex_table(
        header=_DECOMP_HEADER,
        rows=_cell_rows(cells, order),
        caption="Décomposition par séance, à l'heure de New York de la barre de "
        "décision. C'est la coupe la plus contrastée de la campagne.",
        label="tab:x10_sessions",
        col_spec=_DECOMP_SPEC,
    )


def table_atr_regimes(bundle: dict[str, Any]) -> str:
    decomp = bundle["decompositions"]
    rows = _cell_rows(decomp["by_atr_tercile"], ["ATR bas", "ATR moyen", "ATR haut"])
    rows.append("\\midrule")
    rows += _cell_rows(decomp["by_dollars_per_atr"], ["<3", "3-6", "6-10", ">10"])
    rows = [r.replace("<3", "$<3$").replace("3-6", "$3$--$6$") for r in rows]
    rows = [r.replace("6-10", "$6$--$10$").replace(">10", "$>10$") for r in rows]
    return tex_table(
        header=_DECOMP_HEADER,
        rows=rows,
        caption="Décomposition par régime de volatilité~: terciles d'ATR M5 en "
        "haut, tranches de pas de grille rapporté à l'ATR en bas. Aucune des "
        "quatre tranches n'est positive.",
        label="tab:x10_regimes",
        col_spec=_DECOMP_SPEC,
    )


def table_exit_reasons(bundle: dict[str, Any]) -> str:
    decomp = bundle["decompositions"]
    total = decomp["overall"]["n"]
    rows = []
    for reason, label in (
        ("STOP", "STOP"),
        ("TARGET", "TARGET (prochain x10)"),
        ("TIME", "TIME (48 barres)"),
        ("SESSION", "SESSION (16:55 New York)"),
    ):
        cell = decomp["by_exit_reason"][reason]
        pf = cell["profit_factor"]
        pf_text = "$\\infty$" if pf is None else fmt_dec(pf, 3)
        rows.append(
            f"{label} & {fmt_int(cell['n'])} & {fmt_pct(cell['n'] / total, 1)} & "
            f"{fmt_r(cell['expectancy_r'])} & {pf_text} & {fmt_usd(cell['pnl'], 1)} \\\\"
        )
    return tex_table(
        header="\\textbf{Sortie} & \\textbf{n} & \\textbf{Part} & \\textbf{E[R]} & "
        "\\textbf{PF} & \\textbf{PnL} \\\\",
        rows=rows,
        caption="Répartition des sorties. Le stop rend exactement $-1$~R, ce qui "
        "vérifie le dimensionnement de bout en bout~; la cible rend un peu plus "
        "de deux R, et elle n'arrive pas assez souvent.",
        label="tab:x10_exits",
        col_spec="l r r r r r",
    )


def table_context_cuts(bundle: dict[str, Any]) -> str:
    decomp = bundle["decompositions"]
    cells = {
        "Reversal étendu au VWAP": decomp["reversals_extended_vs_not"]["étendu"],
        "Reversal non étendu": decomp["reversals_extended_vs_not"]["non étendu"],
        "DXY favorable au trade": decomp["by_dxy_context"]["favorable"],
        "DXY adverse au trade": decomp["by_dxy_context"]["adverse"],
        "Avec retest du niveau": decomp["by_retest"]["retest"],
        "Sans retest": decomp["by_retest"]["sans retest"],
    }
    return tex_table(
        header=_DECOMP_HEADER,
        rows=_cell_rows(cells),
        caption="Les trois coupes descriptives du mandat~: extension VWAP, "
        "contexte dollar, retest. Aucune ne fait passer l'espérance du bon côté "
        "de zéro.",
        label="tab:x10_context_cuts",
        col_spec=_DECOMP_SPEC,
    )


def table_walkforward(bundle: dict[str, Any]) -> str:
    rows = []
    for year in bundle["walkforward"]["years"]:
        rows.append(
            f"{year['year']} & {fmt_int(year['n'])} & {fmt_r(year['expectancy_r'])} & "
            f"{fmt_dec(year['profit_factor'], 3)} & {fmt_usd(year['pnl'], 1)} & "
            f"{fmt_pct(year['win_rate'], 1)} & {fmt_pct(abs(year['share_of_total_net']), 1)} \\\\"
        )
    return tex_table(
        header="\\textbf{Année} & \\textbf{Trades} & \\textbf{E[R]} & \\textbf{PF} & "
        "\\textbf{PnL} & \\textbf{Réussite} & \\textbf{Part du net} \\\\",
        rows=rows,
        caption="Walk-forward annuel de la configuration analysée. Aucune année "
        "n'est positive~; la perte s'atténue avec le temps sans jamais changer "
        "de signe.",
        label="tab:x10_walkforward",
        col_spec="l r r r r r r",
    )


def table_ablations(bundle: dict[str, Any]) -> str:
    rows = []
    check, cross = "\\checkmark", "---"
    for run in bundle["ablations"]["runs"]:
        tag = " \\emph{(spec)}" if run["is_spec_baseline"] else ""
        rows.append(
            f"{check if run['use_ema'] else cross} & {check if run['use_vwap'] else cross} & "
            f"{check if run['use_dxy'] else cross}{tag} & {fmt_int(run['n'])} & "
            f"{fmt_dec(run['sharpe_net'], 3)} & {fmt_r(run['expectancy_r'])} & "
            f"{fmt_dec(run['profit_factor'], 3)} & {fmt_usd(run['pnl'], 1)} \\\\"
        )
    return tex_table(
        header="\\textbf{EMA50 H1} & \\textbf{VWAP} & \\textbf{DXY} & \\textbf{Trades} & "
        "\\textbf{Sharpe} & \\textbf{E[R]} & \\textbf{PF} & \\textbf{PnL} \\\\",
        rows=rows,
        caption="Ablations des trois filtres de contexte, à la configuration "
        "analysée. Le classement est monotone~: plus on retire de contexte, plus "
        "on perd.",
        label="tab:x10_ablations",
        col_spec="c c l r r r r r",
    )


def table_costs(bundle: dict[str, Any]) -> str:
    rows = []
    for row in bundle["cost_sensitivity"]["sensitivities"]:
        required = row["slippage_per_side"] == 0.0 and row["spread_multiplier"] in (1.5, 2.0)
        mark = "~$\\dagger$" if required else ""
        rows.append(
            f"$\\times{_number(row['spread_multiplier'], 1)}${mark} & "
            f"{fmt_usd(row['slippage_per_side'], 2)} & "
            f"{fmt_usd(row['effective_spread'], 3)} & {fmt_int(row['n'])} & "
            f"{fmt_r(row['expectancy_r'])} & {fmt_dec(row['profit_factor'], 3)} & "
            f"{fmt_dec(row['sharpe_net'], 3)} \\\\"
        )
    return tex_table(
        header="\\textbf{Spread} & \\textbf{Slippage / côté} & \\textbf{Spread effectif} & "
        "\\textbf{Trades} & \\textbf{E[R]} & \\textbf{PF} & \\textbf{Sharpe} \\\\",
        rows=rows,
        caption="Sensibilité aux coûts. Les deux lignes marquées d'une dague "
        "sont les sensibilités que la table de décision rend obligatoires.",
        label="tab:x10_costs",
        col_spec="l r r r r r r",
    )


def table_robustness(bundle: dict[str, Any]) -> str:
    rob = bundle["robustness"]
    boot = {b["metric"]: b for b in rob["bootstrap_ci_95"]}
    sharpe = boot["sharpe_ratio"]
    minbtl = rob["min_backtest_length"]
    rows = [
        f"Sharpe net (séance, 252) & {fmt_dec(sharpe['observed'], 3)} & "
        f"[{fmt_dec(sharpe['ci_low'], 3)} ; {fmt_dec(sharpe['ci_high'], 3)}] & --- \\\\",
        f"PSR (contre $0$) & {fmt_dec(rob['psr']['psr'], 3)} & --- & --- \\\\",
        f"DSR (déflaté, $N = {int(rob['dsr']['n_trials'])}$) & {fmt_dec(rob['dsr']['dsr'], 3)} & "
        f"--- & $\\geq 0{{,}}95$ \\\\",
        f"Sharpe attendu du maximum de $N$ essais & "
        f"{fmt_dec(rob['dsr']['expected_max_sharpe'], 3)} & --- & --- \\\\",
        f"Haircut Sharpe (BHY) & {fmt_dec(rob['haircut']['haircut_sharpe'], 3)} & --- & "
        f"$p$ ajustée {fmt_dec(rob['haircut']['adjusted_pvalue'], 3)} \\\\",
        "MinBTL & \\emph{non calculable} & --- & --- \\\\"
        if minbtl is None
        else f"MinBTL & {fmt_dec(minbtl, 2)} & --- & --- \\\\",
        f"PBO (CSCV, {fmt_int(rob['pbo']['n_bins'])} bins) & {fmt_dec(rob['pbo']['pbo'], 4)} & "
        f"--- & $\\leq 0{{,}}35$ \\\\",
        "\\midrule",
        f"Rendement total & {fmt_pct(boot['total_return']['observed'], 1)} & "
        f"[{fmt_pct(boot['total_return']['ci_low'], 1)} ; "
        f"{fmt_pct(boot['total_return']['ci_high'], 1)}] & --- \\\\",
        f"Repli maximal & {fmt_pct(boot['max_drawdown']['observed'], 1)} & "
        f"[{fmt_pct(boot['max_drawdown']['ci_low'], 1)} ; "
        f"{fmt_pct(boot['max_drawdown']['ci_high'], 1)}] & --- \\\\",
        f"Profit factor & {fmt_dec(boot['profit_factor']['observed'], 3)} & "
        f"[{fmt_dec(boot['profit_factor']['ci_low'], 3)} ; "
        f"{fmt_dec(boot['profit_factor']['ci_high'], 3)}] & --- \\\\",
        f"Ratio de Sortino & {fmt_dec(boot['sortino_ratio']['observed'], 3)} & "
        f"[{fmt_dec(boot['sortino_ratio']['ci_low'], 3)} ; "
        f"{fmt_dec(boot['sortino_ratio']['ci_high'], 3)}] & --- \\\\",
        "\\midrule",
        f"Repli maximal observé (Monte-Carlo) & "
        f"{fmt_pct(rob['mc_trades_max_drawdown']['observed_mdd'], 1)} & "
        f"médiane {fmt_pct(rob['mc_trades_max_drawdown']['mdd_p50'], 1)} ; "
        f"p95 {fmt_pct(rob['mc_trades_max_drawdown']['mdd_p95'], 1)} & --- \\\\",
        f"$z$ de chance de séquence & "
        f"{fmt_dec(rob['sequence_risk']['sequence_luck_zscore'], 2)} & --- & --- \\\\",
        f"P(ruine) & {fmt_pct(rob['ruin']['P(ruin)'], 1)} & --- & --- \\\\",
        f"P(perte $> 50$\\,\\%) & {fmt_pct(rob['ruin']['P(loss>50%)'], 1)} & --- & --- \\\\",
        f"Capital terminal médian (bootstrap) & "
        f"{fmt_dec(rob['ruin']['terminal_p50'], 3)} & --- & --- \\\\",
    ]
    return tex_table(
        header="\\textbf{Mesure} & \\textbf{Valeur} & \\textbf{IC 95\\,\\% / repère} & "
        "\\textbf{Seuil} \\\\",
        rows=rows,
        caption="Batterie de robustesse de la configuration analysée. Les "
        "intervalles sont des bootstraps stationnaires par blocs~; le DSR et le "
        "haircut sont déflatés par le registre d'essais.",
        label="tab:x10_robustness",
        col_spec="l r l r",
        size="\\footnotesize",
    )


def table_feasibility(bundle: dict[str, Any]) -> str:
    rows = []
    for year in bundle["feasibility"]["years"]:
        rows.append(
            f"{year['year']} & {fmt_int(year['n_days'])} & "
            f"{fmt_dec(year['price_median'], 2)} & {fmt_dec(year['atr_m5_median'], 3)} & "
            f"$\\mathbf{{{_number(year['dollars_per_atr'], 1)}}}$ & "
            f"{fmt_dec(year['spread_per_atr'], 3)} & "
            f"{fmt_int(year['crossings_per_day_median'])} & "
            f"{fmt_dec(year['crossings_per_day_mean'], 1)} \\\\"
        )
    return tex_table(
        header="\\textbf{Année} & \\textbf{Jours} & \\textbf{Prix méd.} & "
        "\\textbf{ATR M5 méd.} & \\textbf{10\\,\\$ / ATR} & \\textbf{Spread / ATR} & "
        "\\multicolumn{2}{c}{\\textbf{Crois. / jour}} \\\\\n"
        "& & & & & & \\textit{méd.} & \\textit{moy.} \\\\",
        rows=rows,
        caption="Le pas de la grille mesuré en ATR M5, année par année. La "
        "« même » stratégie vise une cible de dix-sept écarts-types de barre en "
        "début d'échantillon et de moins de quatre en fin.",
        label="tab:x10_feasibility",
        col_spec="l r r r r r r r",
        size="\\footnotesize",
    )


def table_r_rule(bundle: dict[str, Any]) -> str:
    rows = []
    for year, cell in sorted(bundle["decompositions"]["r_rule_rejection_by_year"].items()):
        rows.append(
            f"{year} & {fmt_int(cell['candidates'])} & {fmt_int(cell['rejected'])} & "
            f"{fmt_pct(cell['rejection_rate'], 1)} \\\\"
        )
    return tex_table(
        header="\\textbf{Année} & \\textbf{Candidats au test R} & \\textbf{Rejetés} & "
        "\\textbf{Taux de rejet} \\\\",
        rows=rows,
        caption="Taux de rejet du garde-fou $R \\geq 1$, par année. Le "
        "dénominateur ne compte que les candidats ayant atteint le test~: un "
        "refus au contexte n'a pas de $R$ estimé.",
        label="tab:x10_r_rule",
        col_spec="l r r r",
    )


def table_concentration(bundle: dict[str, Any]) -> str:
    conc = bundle["decompositions"]["concentration"]
    rows = [
        f"Profit brut cumulé & {fmt_usd(conc['gross_profit'], 2)} & --- \\\\",
        f"Cinq meilleures séances & {fmt_usd(conc['pnl_top5_days'], 2)} & "
        f"{fmt_pct(conc['share_of_gross_profit_top5_days'], 1)} \\\\",
        f"Dix meilleurs trades & {fmt_usd(conc['pnl_top10_trades'], 2)} & "
        f"{fmt_pct(conc['share_of_gross_profit_top10_trades'], 1)} \\\\",
        f"Meilleur mois ({conc['best_month']}) & {fmt_usd(conc['pnl_best_month'], 2)} & --- \\\\",
        "\\midrule",
        f"Séances gagnantes & {fmt_int(conc['n_profitable_days'])} & --- \\\\",
        f"Séances perdantes & {fmt_int(conc['n_losing_days'])} & --- \\\\",
    ]
    return tex_table(
        header="\\textbf{Concentration} & \\textbf{Montant} & \\textbf{Part du profit brut} \\\\",
        rows=rows,
        caption="Concentration du résultat. Les parts sont rapportées au profit "
        "\\emph{brut}~: une part d'un net négatif n'aurait pas de sens.",
        label="tab:x10_concentration",
        col_spec="l r r",
    )


def table_cpcv(bundle: dict[str, Any]) -> str:
    cpcv = bundle["cpcv"]
    best = sorted(cpcv["distribution"], key=lambda d: -d["median"])[:5]
    rows = []
    for entry in best:
        z, a, k = entry["key"]
        rows.append(
            f"$z={_number(z, 1)}$ ; $a_{{\\min}}={_number(a, 1)}$ ; $k_s={_number(k, 2)}$ & "
            f"{fmt_dec(entry['median'], 3)} & {fmt_dec(entry['q05'], 3)} & "
            f"{fmt_dec(entry['q95'], 3)} & {fmt_pct(entry['pct_positive'], 0)} \\\\"
        )
    return tex_table(
        header="\\textbf{Configuration} & \\textbf{Médiane OOS} & \\textbf{q05} & "
        "\\textbf{q95} & \\textbf{Splits positifs} \\\\",
        rows=rows,
        caption="Validation croisée combinatoire purgée~: les cinq meilleures "
        "configurations sur la médiane hors échantillon. Aucune des vingt-sept "
        "n'est positive sur un seul des splits. La métrique est le Sharpe des "
        "rendements minute du portefeuille et ne se compare au reste du rapport "
        "qu'en signe.",
        label="tab:x10_cpcv",
        col_spec="l r r r r",
    )


def table_trade_examples(examples: list[dict[str, Any]]) -> str:
    rows = []
    for ex in examples:
        rows.append(
            f"{SCENARIO_LABELS[ex['scenario']]} & {ex['ts_decision'][:16]} & "
            f"{fmt_dec(ex['level'], 0)} & {fmt_dec(ex['fill_px'], 3)} & "
            f"{fmt_dec(ex['stop'], 3)} & {fmt_dec(ex['target'], 0)} & "
            f"{fmt_dec(ex['exit_px'], 3)} & {ex['exit_reason']} & "
            f"{fmt_dec(ex['realised_r'], 2)} \\\\"
        )
    return tex_table(
        header="\\textbf{Scénario} & \\textbf{Décision (UTC)} & \\textbf{Niveau} & "
        "\\textbf{Entrée} & \\textbf{Stop} & \\textbf{Cible} & \\textbf{Sortie} & "
        "\\textbf{Raison} & \\textbf{R réalisé} \\\\",
        rows=rows,
        caption="Les quatre trades détaillés en annexe. Règle de choix, écrite "
        "avant de regarder~: le trade de $R$ réalisé médian de chaque scénario "
        "parmi ceux clôturés en 2023.",
        label="tab:x10_trade_examples",
        col_spec="l l r r r r r l r",
        size="\\scriptsize",
    )


def build_tables(bundle: dict[str, Any], examples: list[dict[str, Any]] | None) -> dict[str, str]:
    """Toutes les tables publiées, nom de fichier → contenu."""
    macros = build_macros(bundle)
    macro_lines = [
        GENERATED_HEADER,
        "% Les macros MT5, QuantConnect et hors échantillon valent « n.d. » tant\n"
        "% que leur JSON n'existe pas : le rapport ne publie pas une mesure qu'il\n"
        "% n'a pas faite.\n",
    ]
    for name, value in macros.items():
        macro_lines.append(f"\\newcommand{{\\{name}}}{{{value}}}")

    verdict = verdict_of(bundle)
    verdict_lines = [
        GENERATED_HEADER,
        "% \\XTenVerdict ∈ {GO, NOGO, INCONCLUSIF}, dérivé de is_summary.json.\n"
        "% \\XTenVerdictProvisoire vaut 1 tant que les sections d'exécution\n"
        "% (QuantConnect, MetaTrader 5) attendent leurs mesures ; le verdict\n"
        "% in-sample, lui, est acquis.\n",
        f"\\newcommand{{\\XTenVerdict}}{{{verdict}}}",
        f"\\newcommand{{\\XTenVerdictProvisoire}}{{{execution_pending(bundle)}}}",
    ]

    tables = {
        "x10_headline": "\n".join(macro_lines) + "\n",
        "x10_verdict": "\n".join(verdict_lines) + "\n",
        "x10_decision": GENERATED_HEADER + table_decision(bundle),
        "x10_grid27": GENERATED_HEADER + table_grid27(bundle),
        "x10_walkforward": GENERATED_HEADER + table_walkforward(bundle),
        "x10_scenarios": GENERATED_HEADER + table_scenarios(bundle),
        "x10_sessions": GENERATED_HEADER + table_sessions(bundle),
        "x10_regimes": GENERATED_HEADER + table_atr_regimes(bundle),
        "x10_exits": GENERATED_HEADER + table_exit_reasons(bundle),
        "x10_context_cuts": GENERATED_HEADER + table_context_cuts(bundle),
        "x10_ablations": GENERATED_HEADER + table_ablations(bundle),
        "x10_costs": GENERATED_HEADER + table_costs(bundle),
        "x10_robustness": GENERATED_HEADER + table_robustness(bundle),
        "x10_cpcv": GENERATED_HEADER + table_cpcv(bundle),
        "x10_concentration": GENERATED_HEADER + table_concentration(bundle),
        "x10_r_rule": GENERATED_HEADER + table_r_rule(bundle),
        "x10_feasibility": GENERATED_HEADER + table_feasibility(bundle),
    }
    if examples is not None:
        tables["x10_trade_examples"] = GENERATED_HEADER + table_trade_examples(examples)
    return tables


def verdict_of(bundle: dict[str, Any]) -> str:
    """``NE PAS DÉPLOYER`` du JSON → ``NOGO``, la valeur que le document teste."""
    raw = bundle["is_summary"]["decision_table"]["provisional_in_sample_verdict"]
    return {
        "NE PAS DÉPLOYER": "NOGO",
        "DÉPLOIEMENT À BLANC": "GO",
        "NON CONCLUANT": "INCONCLUSIF",
    }[raw]


def execution_pending(bundle: dict[str, Any]) -> int:
    """1 tant qu'une macro d'exécution n'a pas de valeur publiable.

    Le critère porte sur les macros, pas sur la présence des fichiers : un
    fichier présent dont le schéma ne résout pas laisse le rapport dans le même
    état d'incomplétude qu'un fichier absent, et le bandeau doit le dire.
    """
    macros = build_macros(bundle)
    return int(any(macros[name] == UNDETERMINED for name in EXECUTION_MACROS))


# ═══════════════════════════════════════════════════════════════════════
# 6. RÉEXÉCUTION DU MOTEUR — courbes et exemples de trades
# ═══════════════════════════════════════════════════════════════════════


def run_analysed_config() -> Any:
    """Le centre de la grille, sur l'échantillon in-sample et lui seul.

    Aucun essai n'est logué : cette configuration appartient déjà à la grille
    des 27 (``xau_x10:grid27:v1``). Aucune barre postérieure au 2025-12-31
    n'entre dans le calcul.
    """
    from strategies.xau_x10 import pipeline, prepare_inputs
    from utils import load_gold_data

    raw, _ = load_gold_data()
    raw = raw.loc[IS_START:IS_END]
    inputs = prepare_inputs(raw)
    _, indicator = pipeline(
        raw,
        z=ANALYSED_PARAMS["z"],
        a_min=ANALYSED_PARAMS["a_min"],
        k_s=ANALYSED_PARAMS["k_s"],
        spread=SELECTION_SPREAD,
        inputs=inputs,
    )
    return indicator


def pick_trade_examples(indicator: Any) -> list[dict[str, Any]]:
    """Un trade par scénario, choisi par une règle écrite, pas à la main.

    Règle : parmi les trades du scénario clôturés en 2023, celui dont le $R$
    réalisé est médian (rang ``n // 2`` en ordre croissant, départage par
    horodatage de décision). Aucune sélection esthétique.
    """
    trades = indicator.trades
    pnl = trades["equity_after"] - trades["equity_before"]
    frame = trades.assign(realised_r=pnl / trades["risk_amount"].replace(0.0, np.nan))
    frame = frame[pd.DatetimeIndex(frame["ts_exit"]).year == EXAMPLE_YEAR]

    picked = []
    for scenario in ("BREAK_LONG", "BREAK_SHORT", "REV_LONG", "REV_SHORT"):
        subset = frame[frame["scenario_name"] == scenario]
        if subset.empty:
            continue
        subset = subset.sort_values(["realised_r", "ts_decision"]).reset_index(drop=True)
        row = subset.iloc[len(subset) // 2]
        picked.append(
            {
                "scenario": scenario,
                "ts_decision": str(row["ts_decision"]),
                "ts_fill": row["ts_fill"],
                "ts_exit": row["ts_exit"],
                "i_decision": int(row["i_decision"]),
                "i_exit_m5": int(row["i_exit_m5"]),
                "level": float(row["level"]),
                "stop": float(row["stop"]),
                "target": float(row["target"]),
                "fill_px": float(row["fill_px"]),
                "exit_px": float(row["exit_px"]),
                "exit_reason": str(row["exit_reason_name"]),
                "realised_r": float(row["realised_r"]),
                "atr": float(row["atr"]),
                "q": float(row["q"]),
            }
        )
    return picked


# ═══════════════════════════════════════════════════════════════════════
# 7. FIGURES
# ═══════════════════════════════════════════════════════════════════════


def save_fig(fig, name: str) -> None:
    path = FIG_DIR / f"{name}.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {path.relative_to(_PROJECT_ROOT)}  ({path.stat().st_size / 1024:.0f} KB)")


def fig_grid_schema() -> None:
    """Le schéma de la grille x10, de la zone d'approche et des deux issues."""
    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    levels = [1990.0, 2000.0, 2010.0]
    for level in levels:
        ax.axhline(level, color=PALETTE["primary"], lw=1.6)
        ax.text(0.2, level + 0.35, f"niveau x10 — {level:.0f} $", color=PALETTE["primary"],
                fontsize=9, fontweight="bold")

    atr = 1.2
    zone_low, zone_high = 2000.0 - 1.0 * atr, 2000.0
    ax.add_patch(
        mpatches.Rectangle(
            (0.2, zone_low), 5.4, zone_high - zone_low,
            facecolor=PALETTE["accent"], alpha=0.18, edgecolor="none",
        )
    )
    ax.annotate(
        "zone d'approche  $z \\cdot ATR$", xy=(2.6, zone_low + 0.35),
        color=PALETTE["accent"], fontsize=9, fontweight="bold",
    )

    x = np.linspace(0.3, 5.6, 60)
    approach = 1996.0 + (2000.4 - 1996.0) * (x - 0.3) / 5.3 ** 1.0
    ax.plot(x, approach, color=PALETTE["grey"], lw=1.8, label="approche du niveau")

    xb = np.linspace(5.6, 9.4, 40)
    ax.plot(xb, 2000.4 + (2009.6 - 2000.4) * (xb - 5.6) / 3.8, color=PALETTE["breakout"],
            lw=2.2, label="Breakout — cible = prochain x10")
    ax.plot(xb, 2000.4 - (2000.4 - 1991.0) * (xb - 5.6) / 3.8, color=PALETTE["reversal"],
            lw=2.2, ls="--", label="Reversal — cible = x10 précédent")

    ax.annotate("", xy=(5.75, 2000.0 - 1.0 * atr), xytext=(5.75, 2000.0),
                arrowprops=dict(arrowstyle="<->", color=PALETTE["accent"], lw=1.2))
    ax.set_xlim(0, 9.6)
    ax.set_ylim(1988, 2012)
    ax.set_xticks([])
    ax.set_ylabel("prix ($)")
    ax.set_title("La grille des niveaux x10, la zone d'approche et les deux issues")
    ax.legend(loc="upper left", fontsize=8.5)
    save_fig(fig, "x10_fig01_grille")


def fig_dollars_per_atr(bundle: dict[str, Any]) -> None:
    years = bundle["feasibility"]["years"]
    labels = [str(y["year"]) for y in years]
    values = [y["dollars_per_atr"] for y in years]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10.0, 4.0))
    ax.bar(labels, values, color=PALETTE["primary"])
    for x, v in zip(labels, values):
        ax.text(x, v + 0.3, f"{v:.1f}".replace(".", ","), ha="center", fontsize=9,
                color=PALETTE["primary"], fontweight="bold")
    ax.set_ylabel("10 $ exprimés en ATR M5")
    ax.set_title("Le pas de grille rapporté à la volatilité")

    crossings = [y["crossings_per_day_median"] for y in years]
    ax2.bar(labels, crossings, color=PALETTE["accent"])
    for x, v in zip(labels, crossings):
        ax2.text(x, v + 0.8, f"{v:.0f}", ha="center", fontsize=9,
                 color=PALETTE["accent"], fontweight="bold")
    ax2.set_ylabel("croisements de niveau par jour (médiane)")
    ax2.set_title("Fréquence des croisements")
    fig.tight_layout()
    save_fig(fig, "x10_fig02_dollars_par_atr")


def fig_grid_heatmaps(bundle: dict[str, Any]) -> None:
    grid = bundle["grid27"]
    z_values = sorted(grid["z"].unique())
    a_values = sorted(grid["a_min"].unique())
    k_values = sorted(grid["k_s"].unique())
    vmin, vmax = grid["sharpe_net"].min(), grid["sharpe_net"].max()

    fig, axes = plt.subplots(1, len(k_values), figsize=(11.0, 3.8), sharey=True)
    for ax, k in zip(axes, k_values):
        block = np.array(
            [
                [
                    float(grid[(grid["z"] == z) & (grid["a_min"] == a) & (grid["k_s"] == k)][
                        "sharpe_net"
                    ].iloc[0])
                    for a in a_values
                ]
                for z in z_values
            ]
        )
        im = ax.imshow(block, cmap="RdYlGn", vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(a_values)))
        ax.set_xticklabels([f"{a:.1f}".replace(".", ",") for a in a_values])
        ax.set_yticks(range(len(z_values)))
        ax.set_yticklabels([f"{z:g}".replace(".", ",") for z in z_values])
        ax.set_xlabel("$a_{min}$")
        ax.set_title(f"$k_s$ = {k:g}".replace(".", ","))
        ax.grid(False)
        for i in range(len(z_values)):
            for j in range(len(a_values)):
                ax.text(j, i, f"{block[i, j]:.2f}".replace(".", ","), ha="center",
                        va="center", fontsize=8.5, color=PALETTE["text"])
    axes[0].set_ylabel("$z$")
    fig.colorbar(im, ax=axes, shrink=0.85, label="Sharpe net annualisé")
    fig.suptitle("Sharpe net des vingt-sept configurations — aucune case n'est positive",
                 fontsize=12, fontweight="bold")
    save_fig(fig, "x10_fig03_heatmaps")


def _analysed_returns() -> pd.Series:
    frame = pd.read_parquet(RESULTS_DIR / "grid27_daily_returns.parquet")
    return frame[ANALYSED_KEY]


def fig_equity(returns: pd.Series) -> None:
    equity = (1.0 + returns).cumprod()
    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    ax.plot(equity.index, equity.to_numpy(), color=PALETTE["primary"], lw=1.6)
    ax.axhline(1.0, color=PALETTE["grey"], lw=0.9, ls="--")
    ax.set_ylabel("capital, base 1")
    ax.set_title("Courbe d'équité de la configuration analysée, par séance")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    save_fig(fig, "x10_fig04_equite")


def fig_underwater(returns: pd.Series) -> None:
    equity = (1.0 + returns).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    fig, ax = plt.subplots(figsize=(9.5, 3.4))
    ax.fill_between(drawdown.index, drawdown.to_numpy() * 100.0, 0.0,
                    color=PALETTE["red"], alpha=0.35)
    ax.plot(drawdown.index, drawdown.to_numpy() * 100.0, color=PALETTE["red"], lw=1.0)
    ax.set_ylabel("repli depuis le plus haut (%)")
    ax.set_title("Courbe d'immersion — la stratégie n'est pratiquement jamais à son plus haut")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    save_fig(fig, "x10_fig05_immersion")


def fig_expectancy_per_year(bundle: dict[str, Any]) -> None:
    years = bundle["walkforward"]["years"]
    labels = [str(y["year"]) for y in years]
    values = [y["expectancy_r"] for y in years]
    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    ax.bar(labels, values, color=PALETTE["red"])
    ax.axhline(0.0, color=PALETTE["grey"], lw=1.0)
    for x, v in zip(labels, values):
        ax.text(x, v - 0.012, f"{v:.3f}".replace(".", ","), ha="center", va="top",
                fontsize=8.5, color=PALETTE["text"])
    ax.set_ylabel("espérance nette par trade (R)")
    ax.set_title("Espérance annuelle — zéro année positive sur sept")
    save_fig(fig, "x10_fig06_esperance_annuelle")


def fig_scenario_session(bundle: dict[str, Any]) -> None:
    decomp = bundle["decompositions"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 3.9))
    order = ["BREAK_LONG", "BREAK_SHORT", "REV_LONG", "REV_SHORT"]
    colors = [PALETTE["breakout"], PALETTE["breakout"], PALETTE["reversal"], PALETTE["reversal"]]
    ax1.barh(
        [SCENARIO_LABELS[k] for k in order],
        [decomp["by_scenario"][k]["expectancy_r"] for k in order],
        color=colors,
    )
    ax1.axvline(0.0, color=PALETTE["grey"], lw=1.0)
    ax1.set_xlabel("espérance nette (R)")
    ax1.set_title("Par scénario")
    ax1.invert_yaxis()

    sessions = ["Asie 18:00-03:00", "Londres 03:00-08:00", "New York 08:00-17:00"]
    ax2.barh(sessions, [decomp["by_ny_session"][s]["expectancy_r"] for s in sessions],
             color=PALETTE["primary"])
    ax2.axvline(0.0, color=PALETTE["grey"], lw=1.0)
    ax2.set_xlabel("espérance nette (R)")
    ax2.set_title("Par séance (heure de New York)")
    ax2.invert_yaxis()
    fig.tight_layout()
    save_fig(fig, "x10_fig07_scenario_session")


def fig_r_rule(bundle: dict[str, Any]) -> None:
    cells = bundle["decompositions"]["r_rule_rejection_by_year"]
    labels = sorted(cells)
    values = [cells[y]["rejection_rate"] * 100.0 for y in labels]
    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    ax.bar(labels, values, color=PALETTE["accent"])
    for x, v in zip(labels, values):
        ax.text(x, v + 1.0, f"{v:.1f}".replace(".", ",") + " %", ha="center",
                fontsize=8.5, color=PALETTE["text"])
    ax.set_ylabel("candidats rejetés (%)")
    ax.set_title("Le garde-fou $R \\geq 1$ mord, et de plus en plus")
    save_fig(fig, "x10_fig08_regle_r")


def fig_exit_reasons(bundle: dict[str, Any]) -> None:
    decomp = bundle["decompositions"]
    order = ["STOP", "TARGET", "TIME", "SESSION"]
    counts = [decomp["by_exit_reason"][r]["n"] for r in order]
    r_values = [decomp["by_exit_reason"][r]["expectancy_r"] for r in order]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 3.9))
    ax1.bar(order, counts, color=[PALETTE["red"], PALETTE["green"], PALETTE["primary"],
                                  PALETTE["accent"]])
    for x, v in zip(order, counts):
        ax1.text(x, v + 15, str(v), ha="center", fontsize=9, color=PALETTE["text"])
    ax1.set_ylabel("nombre de trades")
    ax1.set_title("Combien de trades sortent par quelle porte")
    ax2.bar(order, r_values, color=[PALETTE["red"], PALETTE["green"], PALETTE["primary"],
                                    PALETTE["accent"]])
    ax2.axhline(0.0, color=PALETTE["grey"], lw=1.0)
    ax2.set_ylabel("espérance de la sortie (R)")
    ax2.set_title("Ce que chaque porte rapporte")
    fig.tight_layout()
    save_fig(fig, "x10_fig09_raisons_sortie")


def fig_cost_sensitivity(bundle: dict[str, Any]) -> None:
    rows = bundle["cost_sensitivity"]["sensitivities"]
    fig, ax = plt.subplots(figsize=(8.8, 4.0))
    for slippage, colour in ((0.0, PALETTE["primary"]), (0.05, PALETTE["accent"]),
                             (0.1, PALETTE["red"])):
        subset = [r for r in rows if r["slippage_per_side"] == slippage]
        ax.plot(
            [r["effective_spread"] for r in subset],
            [r["expectancy_r"] for r in subset],
            marker="o", color=colour, lw=1.8,
            label=f"slippage {slippage:.2f} $ / côté".replace(".", ","),
        )
    ax.axhline(0.0, color=PALETTE["grey"], lw=1.2, ls="--")
    ax.set_xlabel("spread effectif par aller-retour ($)")
    ax.set_ylabel("espérance nette par trade (R)")
    ax.set_title("La droite de coût ne recoupe jamais l'axe des abscisses")
    ax.legend()
    save_fig(fig, "x10_fig10_sensibilite_couts")


def fig_bootstrap(bundle: dict[str, Any]) -> None:
    wanted = [
        ("sharpe_ratio", "Sharpe"),
        ("sortino_ratio", "Sortino"),
        ("calmar_ratio", "Calmar"),
        ("annualized_return", "Rendement annualisé"),
        ("total_return", "Rendement total"),
        ("information_ratio", "Ratio d'information"),
    ]
    boot = {b["metric"]: b for b in bundle["robustness"]["bootstrap_ci_95"]}
    fig, ax = plt.subplots(figsize=(8.8, 4.0))
    for i, (metric, label) in enumerate(wanted):
        row = boot[metric]
        ax.plot([row["ci_low"], row["ci_high"]], [i, i], color=PALETTE["primary"], lw=3.0)
        ax.plot([row["observed"]], [i], marker="D", color=PALETTE["accent"], ms=7)
        del label
    ax.axvline(0.0, color=PALETTE["red"], lw=1.4, ls="--")
    ax.set_yticks(range(len(wanted)))
    ax.set_yticklabels([label for _, label in wanted])
    ax.invert_yaxis()
    ax.set_xlabel("valeur (échelles hétérogènes, le signe seul est comparable)")
    ax.set_title("Intervalles de confiance à 95 % — aucun ne contient zéro")
    save_fig(fig, "x10_fig11_bootstrap")


def fig_trade_example(indicator: Any, example: dict[str, Any]) -> None:
    m5 = indicator.m5
    start = max(0, example["i_decision"] - 24)
    stop_i = min(len(m5) - 1, example["i_exit_m5"] + 12)
    window = m5.iloc[start : stop_i + 1]
    x = np.arange(len(window))

    fig, ax = plt.subplots(figsize=(9.5, 4.4))
    up = window["close"].to_numpy() >= window["open"].to_numpy()
    colours = np.where(up, PALETTE["green"], PALETTE["red"])
    ax.vlines(x, window["low"].to_numpy(), window["high"].to_numpy(), color=colours, lw=0.9)
    ax.vlines(x, window["open"].to_numpy(), window["close"].to_numpy(), color=colours, lw=3.0)

    atr = example["atr"]
    ax.axhline(example["level"], color=PALETTE["primary"], lw=1.6)
    ax.text(0.4, example["level"] + 0.05 * atr, f"niveau x10 — {example['level']:.0f} $",
            fontsize=8.5, color=PALETTE["primary"], fontweight="bold")
    direction = 1.0 if example["target"] > example["level"] else -1.0
    ax.add_patch(
        mpatches.Rectangle(
            (0, min(example["level"], example["level"] - direction * ANALYSED_PARAMS["z"] * atr)),
            len(window) - 1, ANALYSED_PARAMS["z"] * atr,
            facecolor=PALETTE["accent"], alpha=0.14, edgecolor="none",
        )
    )
    ax.axhline(example["stop"], color=PALETTE["red"], lw=1.2, ls="--")
    ax.text(0.4, example["stop"] + 0.05 * atr, "stop", fontsize=8.5, color=PALETTE["red"])
    ax.axhline(example["target"], color=PALETTE["green"], lw=1.2, ls="--")
    ax.text(0.4, example["target"] + 0.05 * atr, "cible", fontsize=8.5, color=PALETTE["green"])

    i_entry = example["i_decision"] - start + 1
    i_exit = example["i_exit_m5"] - start
    ax.plot([i_entry], [example["fill_px"]], marker="^", ms=11, color=PALETTE["primary"],
            zorder=5)
    ax.annotate("entrée", (i_entry, example["fill_px"]), textcoords="offset points",
                xytext=(6, 10), fontsize=8.5, color=PALETTE["primary"], fontweight="bold")
    ax.plot([i_exit], [example["exit_px"]], marker="v", ms=11, color=PALETTE["accent"],
            zorder=5)
    ax.annotate(f"sortie {example['exit_reason']}", (i_exit, example["exit_px"]),
                textcoords="offset points", xytext=(6, -14), fontsize=8.5,
                color=PALETTE["accent"], fontweight="bold")

    ticks = np.arange(0, len(window), max(1, len(window) // 8))
    ax.set_xticks(ticks)
    ax.set_xticklabels([window.index[t].strftime("%H:%M") for t in ticks])
    ax.set_ylabel("prix ($)")
    ax.set_xlabel(f"barres M5 du {window.index[0].date()} (heure locale de séance)")
    ax.set_title(
        f"{SCENARIO_LABELS[example['scenario']]} — "
        f"R réalisé {example['realised_r']:.2f}".replace(".", ",")
    )
    save_fig(fig, f"x10_fig12_trade_{SCENARIO_SLUG[example['scenario']]}")


def build_figures(bundle: dict[str, Any], indicator: Any, examples: list[dict[str, Any]]) -> None:
    returns = _analysed_returns()
    fig_grid_schema()
    fig_dollars_per_atr(bundle)
    fig_grid_heatmaps(bundle)
    fig_equity(returns)
    fig_underwater(returns)
    fig_expectancy_per_year(bundle)
    fig_scenario_session(bundle)
    fig_r_rule(bundle)
    fig_exit_reasons(bundle)
    fig_cost_sensitivity(bundle)
    fig_bootstrap(bundle)
    for example in examples:
        fig_trade_example(indicator, example)


# ═══════════════════════════════════════════════════════════════════════
# 8. ENTRÉE
# ═══════════════════════════════════════════════════════════════════════


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="ne rien écrire ; sortir non nul si une table publiée a dérivé",
    )
    parser.add_argument(
        "--tables-only",
        action="store_true",
        help="régénérer les tables sans rejouer le moteur ni les figures",
    )
    args = parser.parse_args(argv)

    if args.check:
        bundle = load_bundle()
        if bundle["feasibility"] is None:
            print(f"✗ {FEASIBILITY_JSON} absent : lancer le script sans --check.")
            return 1
        tables = build_tables(bundle, examples=None)
        drifted = []
        for name, content in tables.items():
            path = TBL_DIR / f"{name}.tex"
            if not path.is_file() or path.read_text(encoding="utf-8") != content:
                drifted.append(path.relative_to(_PROJECT_ROOT).as_posix())
        if drifted:
            print("✗ tables publiées différentes de ce qui serait généré :")
            for name in drifted:
                print(f"    {name}")
            return 1
        print(f"✓ {len(tables)} tables publiées conformes à leurs sources JSON.")
        return 0

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TBL_DIR.mkdir(parents=True, exist_ok=True)

    print("═" * 70)
    print("  Apogée Invest — stratégie 2 (XAUUSD x10) — assets du rapport")
    print("═" * 70)

    if not FEASIBILITY_JSON.is_file():
        print("\n[1/4] Table de faisabilité (10 $/ATR, descriptif)...")
        FEASIBILITY_JSON.write_text(
            json.dumps(build_feasibility_table(), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"  ✓ {FEASIBILITY_JSON.relative_to(_PROJECT_ROOT)}")
    else:
        print("\n[1/4] Table de faisabilité déjà produite — relue telle quelle.")

    bundle = load_bundle()

    indicator, examples = None, None
    if not args.tables_only:
        print("\n[2/4] Réexécution du centre de la grille (aucun essai logué)...")
        indicator = run_analysed_config()
        examples = pick_trade_examples(indicator)
        print(f"  ✓ {len(indicator.trades)} trades, {len(examples)} exemples retenus")

    print("\n[3/4] Tables...")
    for name, content in build_tables(bundle, examples).items():
        path = TBL_DIR / f"{name}.tex"
        path.write_text(content, encoding="utf-8")
        print(f"  ✓ {path.relative_to(_PROJECT_ROOT)}  ({len(content)} caractères)")

    if args.tables_only:
        print("\n[4/4] Figures ignorées (--tables-only).")
        return 0

    print("\n[4/4] Figures...")
    build_figures(bundle, indicator, examples)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
