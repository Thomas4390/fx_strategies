#!/usr/bin/env python3
"""build_setup_guide_tables — génère les tables de paramètres du guide client.

Le guide d'installation listait ses paramètres à la main. Il a dérivé deux fois :
en avril sur les allocations et la couche de risque, et le 2026-07-26 sur le
plafond de marge (`false`/`0.70` publié contre `true`/`0.50` compilé) et sur la
fenêtre de session du sleeve 1 (6-14 UTC publié contre 8-16 compilé). Un client
qui suivait le guide configurait une machine différente de celle qui a produit
les métriques annoncées.

Les **valeurs** viennent désormais de `write_default_preset.PRESET_LINES`, qui
`tests/test_mt5_preset_sync.py` asservit déjà aux défauts compilés du `.mq5`.
Seule la **prose** (le rôle de chaque paramètre) vit ici : c'est le seul contenu
qu'aucune source de vérité ne porte.

Usage :
    python scripts/build_setup_guide_tables.py
    python scripts/build_setup_guide_tables.py --check   # ne rien écrire
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SRC = _PROJECT_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from mt5.bridge.write_default_preset import PRESET_LINES  # noqa: E402

TBL_DIR = _PROJECT_ROOT / "reports" / "client_setup_guide" / "tables"


# ═══════════════════════════════════════════════════════════════════════
#  Lecture des valeurs — source unique
# ═══════════════════════════════════════════════════════════════════════

def preset_values() -> dict[str, str]:
    """`{Inp_Xxx: valeur littérale}` depuis PRESET_LINES.

    On garde le littéral tel quel : c'est exactement ce que MT5 reçoit, donc la
    seule chose défendable à publier. La forme étendue `v||def||min||max||N` est
    tronquée à sa première composante.
    """
    out: dict[str, str] = {}
    for line in PRESET_LINES:
        if line.startswith(";") or "=" not in line:
            continue
        key, _, raw = line.partition("=")
        out[key.strip()] = raw.split("||")[0].strip()
    return out


# ═══════════════════════════════════════════════════════════════════════
#  Mise en forme
# ═══════════════════════════════════════════════════════════════════════

def tex_escape(value: str) -> str:
    return value.replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")


def fmt_pairs(value: str) -> str:
    """`EURUSD,GBPUSD,USDCAD` -> `EUR, GBP, CAD` (la devise non-USD)."""
    out = []
    for sym in value.split(","):
        sym = sym.strip().upper()
        out.append(sym[3:] if sym.startswith("USD") else sym.replace("USD", ""))
    return ", ".join(out)


def n_pairs(value: str) -> int:
    return len([s for s in value.split(",") if s.strip()])


def fmt_value(name: str, value: str) -> str:
    if name.endswith("_Pairs"):
        return fmt_pairs(value)
    return tex_escape(value)


# ═══════════════════════════════════════════════════════════════════════
#  Prose — le seul contenu éditorial, tout le reste est dérivé
# ═══════════════════════════════════════════════════════════════════════
#
# `{v}` est remplacé par la valeur, `{n}` par le nombre de paires. Écrire le
# nombre en toutes lettres dans la prose est précisément ce qui a permis au
# guide d'annoncer « 4 paires » pour un univers qui en compte 3.

ROLES: dict[str, str] = {
    # --- 6.1 Allocation & risque global ---
    "Inp_AllocMRMacro": "Part de l'équity pilotée par le Sleeve~1 (MR Macro).",
    "Inp_AllocTSMomentum": "Part allouée au Sleeve~2 (TS Momentum).",
    "Inp_AllocRSIDaily": "Part allouée au Sleeve~3 (RSI Daily).",
    "Inp_AllocH1Momentum": "Part allouée au Sleeve~5 (H1 Momentum). À~{v}, la sleeve est compilée mais inactive~: elle ne prend aucune position.",
    "Inp_AllocGoldMomentum": "Part allouée au Sleeve~4 (Gold Momentum, XAU-USD).",
    "Inp_RiskScale": "Facteur d'échelle appliqué aux budgets de risque des quatre sleeves. \\textbf{C'est le paramètre qui dimensionne le risque}~: le rendement et le repli y répondent de façon quasi proportionnelle.",
    "Inp_EnableDDCap": "Circuit-breaker de \\emph{drawdown}. Désactivé (Phase~A 2026-05-04~: freinait 24\\,\\% des configurations in-sample sans bénéfice hors échantillon).",
    "Inp_DDCap": "Seuil de repli relatif, utilisé seulement si \\code{EnableDDCap=true}.",
    "Inp_ResetDDState": "Force la réinitialisation de l'état de repli persisté (diagnostic uniquement).",
    "Inp_EnableMarginCap": "Plafond d'utilisation de marge, \\textbf{actif}. Jamais déclenché sur le backtest de référence, mais il protège en conditions réelles.",
    "Inp_MarginCapPct": "Seuil d'utilisation de marge au-delà duquel le levier global est divisé par deux. Une fermeture forcée de toutes les positions intervient séparément à $85\\,\\%$.",
    # --- 6.2 Vol-targeting global ---
    "Inp_GlobalTargetVol": "Volatilité annualisée cible du portefeuille.",
    "Inp_GlobalMaxLeverage": "Plafond dur du levier brut.",
    "Inp_GlobalVolFloor": "Plancher sur la volatilité estimée~; évite un levier explosif quand la volatilité réalisée s'effondre.",
    # --- 6.3 Sleeve 1 — MR Macro ---
    "Inp_MR_Pairs": "Univers de {n}~paires equal-weight (cotées en USD).",
    "Inp_MR_BBWindow": "Fenêtre Bollinger sur la déviation au VWAP, en barres M1.",
    "Inp_MR_BBAlpha": "Largeur de bande en $\\sigma$~; entrée si $|z| > {v}\\sigma$.",
    "Inp_MR_TPStop": "Take-profit par transaction.",
    "Inp_MR_SLStop": "Stop-loss par transaction.",
    "Inp_MR_SessionStart": "Heure UTC d'ouverture de la fenêtre de trading.",
    "Inp_MR_SessionEnd": "Heure UTC de fermeture de la fenêtre~; aucune entrée au-delà.",
    "Inp_MR_TimeStopHours": "Time-stop~: clôture forcée après {v}\\,h sans take-profit ni stop-loss.",
    "Inp_MR_ForcedCloseHr": "Heure UTC du flatten intraday total (kill-switch quotidien).",
    "Inp_MR_SpreadThresh": "Seuil sur le spread 10Y--2Y pour le filtre macro.",
    "Inp_MR_SlippageBps": "Slippage modélisé en points de base.",
    "Inp_MR_DisableMacroFilter": "Neutralise le filtre macro (diagnostic uniquement).",
    "Inp_MR_NewsFilterEnabled": "Suspend les entrées autour des publications macro à fort impact.",
    # --- 6.4 Sleeve 2 — TS Momentum ---
    "Inp_TS_Pairs": "Univers de {n}~paires. USD/CAD est exclu (régime historiquement défavorable).",
    "Inp_TS_FastEMA": "EMA rapide du signal de tendance.",
    "Inp_TS_SlowEMA": "EMA lente~; position longue si EMA rapide $>$ EMA lente, courte sinon.",
    "Inp_TS_RSIPeriod": "Période du RSI servant de filtre.",
    "Inp_TS_RSILow": "Seuil RSI bas~; filtre les ventes en survente extrême.",
    "Inp_TS_RSIHigh": "Seuil RSI haut~; filtre les achats en surachat extrême.",
    "Inp_TS_TargetVol": "Volatilité cible par paire, avant la couche globale.",
    "Inp_TS_MaxLeverage": "Levier maximal interne à la sleeve (faible~: signal journalier).",
    "Inp_TS_SlippageBps": "Slippage modélisé en points de base.",
    # --- 6.5 Sleeve 3 — RSI Daily ---
    "Inp_RSI_Pairs": "Univers de {n}~paires equal-weight.",
    "Inp_RSI_Period": "Période du RSI (standard de Wilder).",
    "Inp_RSI_Oversold": "Seuil d'entrée longue~: RSI inférieur à {v}.",
    "Inp_RSI_Overbought": "Seuil d'entrée courte~: RSI supérieur à {v}.",
    "Inp_RSI_ExitMid": "Sortie quand le RSI repasse {v} (retour à la moyenne).",
    "Inp_RSI_SlippageBps": "Slippage modélisé en points de base.",
    "Inp_RSI_TimeStopDays": "Durée maximale de détention, en jours. Au-delà, le \\emph{swap} cumulé annule l'edge du signal.",
    # --- 6.6 Sleeve 4 — Gold Momentum ---
    "Inp_Gold_Symbols": "Liste d'instruments (CSV)~; le suffixe broker est ajouté automatiquement à chacun.",
    "Inp_Gold_LookbackA": "Horizons de tendance, en séances, dont les votes sont agrégés. Mettre un horizon à $0$ le désactive.",
    "Inp_Gold_AllowShort": "Positions courtes désactivées~: l'or présente une dérive haussière structurelle.",
    "Inp_Gold_TargetVol": "Volatilité cible propre à la sleeve, appliquée avant la couche globale.",
    "Inp_Gold_MaxLeverage": "Plafond de levier interne à la sleeve.",
    "Inp_Gold_SafetySL": "Stop de sécurité~; l'or est environ deux fois plus volatil que les paires FX.",
    "Inp_Gold_SlippageBps": "Spread CFD XAU-USD et commission, en points de base.",
    "Inp_Gold_Trace": "Trace journalière de réconciliation (diagnostic uniquement).",
    "Inp_Gold_TraceFile": "Fichier de la trace ci-dessus.",
    # --- 6.7 Opérationnel & source macro ---
    "Inp_SymbolSuffix": "Suffixe broker (\\textbf{critique}, voir~3.4).",
    "Inp_MagicMR": "Identifiant des transactions du Sleeve~1 (audit).",
    "Inp_MagicTS": "Identifiant des transactions du Sleeve~2.",
    "Inp_MagicRSI": "Identifiant des transactions du Sleeve~3.",
    "Inp_MagicH1": "Identifiant des transactions du Sleeve~5.",
    "Inp_MagicGold": "Identifiant des transactions du Sleeve~4.",
    "Inp_ExportDeals": "Écrit un CSV par transaction (avec magic et sleeve) en fin de backtest.",
    "Inp_LogVerbose": "Active les journaux \\emph{DEBUG}~; à laisser désactivé en production.",
    "Inp_LogToFile": "Écrit dans \\code{MQL5\\textbackslash Files\\textbackslash fx\\_log.csv}.",
    "Inp_DailyRecomputeHr": "Heure UTC du recalcul journalier (vol-targeting et sleeves D1).",
    "Inp_MacroSourceMode": "Mode de dispatch macro (FILE / NATIVE / HYBRID / HISTORY / AUTO). {v} = AUTO.",
    "Inp_MacroCacheFile": "Cache d'une ligne (mode FILE).",
    "Inp_MacroUseCommon": "Lecture dans \\code{Common\\textbackslash Files} plutôt que \\code{MQL5\\textbackslash Files}.",
    "Inp_MacroMaxAgeHours": "Fraîcheur maximale de la donnée macro~; au-delà, le Sleeve~1 est désactivé.",
    "Inp_FREDApiKeyFile": "Nom du fichier contenant la clé FRED.",
    "Inp_FREDKeyUseCommon": "Lit la clé dans \\code{Common\\textbackslash Files}.",
    "Inp_FREDSeriesId": "Série FRED utilisée (spread de courbe 10Y--2Y).",
    "Inp_MacroHistoryFile": "Fichier multi-lignes (mode HISTORY).",
    "Inp_MacroHistoryUseCommon": "Lit l'historique dans \\code{Common\\textbackslash Files}.",
}

# Inputs volontairement absents du guide client, avec la raison.
EXCLUDED: dict[str, str] = {
    "Inp_H1_Pairs": "Sleeve 5 non allouée en production",
    "Inp_H1_FastEMA": "Sleeve 5 non allouée en production",
    "Inp_H1_SlowEMA": "Sleeve 5 non allouée en production",
    "Inp_H1_RSIPeriod": "Sleeve 5 non allouée en production",
    "Inp_H1_RSILow": "Sleeve 5 non allouée en production",
    "Inp_H1_RSIHigh": "Sleeve 5 non allouée en production",
    "Inp_H1_ATRPeriod": "Sleeve 5 non allouée en production",
    "Inp_H1_ATRMultSL": "Sleeve 5 non allouée en production",
    "Inp_H1_TargetVol": "Sleeve 5 non allouée en production",
    "Inp_H1_MaxLeverage": "Sleeve 5 non allouée en production",
    "Inp_H1_SlippageBps": "Sleeve 5 non allouée en production",
    "Inp_Gold_LookbackB": "regroupé avec Inp_Gold_LookbackA",
    "Inp_Gold_LookbackC": "regroupé avec Inp_Gold_LookbackA",
    "Inp_Gold_LookbackD": "regroupé avec Inp_Gold_LookbackA",
    "Inp_CommissionBpsPerSide": "coûts d'exécution, traités en section 7",
    "Inp_SwapBpsPerNight": "coûts d'exécution, traités en section 7",
}

# Lignes regroupées : un seul rang pour plusieurs inputs.
GROUPED: dict[str, tuple[str, list[str]]] = {
    "Inp_Gold_LookbackA": (
        r"Inp\_Gold\_LookbackA..D",
        ["Inp_Gold_LookbackA", "Inp_Gold_LookbackB",
         "Inp_Gold_LookbackC", "Inp_Gold_LookbackD"],
    ),
}

# Une table par sous-section du guide. `spacer_before` insère un \addlinespace.
TABLES: list[tuple[str, str, list[str], set[str]]] = [
    ("params_alloc", "p{4.6cm}p{2.3cm}p{8.0cm}", [
        "Inp_AllocMRMacro", "Inp_AllocTSMomentum", "Inp_AllocRSIDaily",
        "Inp_AllocGoldMomentum", "Inp_AllocH1Momentum", "Inp_RiskScale",
        "Inp_EnableDDCap", "Inp_DDCap", "Inp_ResetDDState",
        "Inp_EnableMarginCap", "Inp_MarginCapPct",
    ], {"Inp_RiskScale", "Inp_EnableDDCap", "Inp_EnableMarginCap"}),
    ("params_voltarget", "p{4.6cm}p{2.3cm}p{8.0cm}", [
        "Inp_GlobalTargetVol", "Inp_GlobalMaxLeverage", "Inp_GlobalVolFloor",
    ], set()),
    ("params_sleeve_mr", "p{4.6cm}p{3.4cm}p{6.9cm}", [
        "Inp_MR_Pairs", "Inp_MR_BBWindow", "Inp_MR_BBAlpha", "Inp_MR_TPStop",
        "Inp_MR_SLStop", "Inp_MR_SessionStart", "Inp_MR_SessionEnd",
        "Inp_MR_TimeStopHours", "Inp_MR_ForcedCloseHr", "Inp_MR_SpreadThresh",
        "Inp_MR_SlippageBps", "Inp_MR_DisableMacroFilter",
        "Inp_MR_NewsFilterEnabled",
    ], set()),
    ("params_sleeve_ts", "p{4.6cm}p{3.4cm}p{6.9cm}", [
        "Inp_TS_Pairs", "Inp_TS_FastEMA", "Inp_TS_SlowEMA", "Inp_TS_RSIPeriod",
        "Inp_TS_RSILow", "Inp_TS_RSIHigh", "Inp_TS_TargetVol",
        "Inp_TS_MaxLeverage", "Inp_TS_SlippageBps",
    ], set()),
    ("params_sleeve_rsi", "p{4.6cm}p{3.4cm}p{6.9cm}", [
        "Inp_RSI_Pairs", "Inp_RSI_Period", "Inp_RSI_Oversold",
        "Inp_RSI_Overbought", "Inp_RSI_ExitMid", "Inp_RSI_TimeStopDays",
        "Inp_RSI_SlippageBps",
    ], set()),
    ("params_sleeve_gold", "p{4.6cm}p{3.4cm}p{6.9cm}", [
        "Inp_Gold_Symbols", "Inp_Gold_LookbackA", "Inp_Gold_AllowShort",
        "Inp_Gold_TargetVol", "Inp_Gold_MaxLeverage", "Inp_Gold_SafetySL",
        "Inp_Gold_SlippageBps", "Inp_Gold_Trace", "Inp_Gold_TraceFile",
    ], set()),
    ("params_ops", r"p{4.6cm}>{\footnotesize\ttfamily}p{3.4cm}p{6.9cm}", [
        "Inp_SymbolSuffix", "Inp_MagicMR", "Inp_MagicTS", "Inp_MagicRSI",
        "Inp_MagicGold", "Inp_MagicH1", "Inp_ExportDeals", "Inp_LogVerbose",
        "Inp_LogToFile", "Inp_DailyRecomputeHr",
        "Inp_MacroSourceMode", "Inp_MacroCacheFile", "Inp_MacroUseCommon",
        "Inp_MacroMaxAgeHours", "Inp_FREDApiKeyFile", "Inp_FREDKeyUseCommon",
        "Inp_FREDSeriesId", "Inp_MacroHistoryFile", "Inp_MacroHistoryUseCommon",
    ], {"Inp_MacroSourceMode"}),
]

HEADER = (
    "% Généré par scripts/build_setup_guide_tables.py — NE PAS ÉDITER À LA MAIN.\n"
    "% Valeurs dérivées de write_default_preset.PRESET_LINES.\n"
)


def render_row(name: str, values: dict[str, str]) -> str:
    if name in GROUPED:
        label, members = GROUPED[name]
        value = " / ".join(values[m] for m in members)
    else:
        label = tex_escape(name)
        value = fmt_value(name, values[name])

    # Substitution explicite : `str.format` buterait sur les accolades LaTeX
    # (\textbf{...}, \code{...}) que la prose contient légitimement.
    role = ROLES[name].replace("{v}", fmt_value(name, values[name]))
    if name.endswith("_Pairs"):
        role = role.replace("{n}", str(n_pairs(values[name])))
    return f"\\code{{{label}}} & \\metric{{{value}}} & {role} \\\\"


def render_table(col_spec: str, names: list[str], spacers: set[str],
                 values: dict[str, str]) -> str:
    rows = []
    for i, name in enumerate(names):
        if name in spacers and i:
            rows.append(r"\addlinespace")
        rows.append(render_row(name, values))
    body = "\n".join(rows)
    return (
        f"{HEADER}"
        "\\begin{center}\n\\small\n"
        f"\\begin{{tabular}}{{@{{}}{col_spec}@{{}}}}\n"
        "\\toprule\n"
        "\\textbf{Paramètre} & \\textbf{Défaut} & \\textbf{Rôle} \\\\\n"
        "\\midrule\n"
        f"{body}\n"
        "\\bottomrule\n"
        "\\end{tabular}\n\\end{center}\n"
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="Vérifie sans écrire ; sort en 1 si un fichier est périmé")
    args = ap.parse_args()

    values = preset_values()

    described = set(ROLES) | set(EXCLUDED)
    missing = sorted(set(values) - described)
    if missing:
        print("[erreur] inputs sans description ni exclusion explicite :", file=sys.stderr)
        for m in missing:
            print(f"  - {m}", file=sys.stderr)
        return 2
    unknown = sorted(described - set(values))
    if unknown:
        print("[erreur] décrits mais absents du preset :", file=sys.stderr)
        for u in unknown:
            print(f"  - {u}", file=sys.stderr)
        return 2

    TBL_DIR.mkdir(parents=True, exist_ok=True)
    stale = False
    for stem, col_spec, names, spacers in TABLES:
        content = render_table(col_spec, names, spacers, values)
        path = TBL_DIR / f"{stem}.tex"
        current = path.read_text(encoding="utf-8") if path.exists() else None
        if current == content:
            print(f"  = {path.relative_to(_PROJECT_ROOT)} ({len(names)} paramètres)")
            continue
        stale = True
        if args.check:
            print(f"  ! {path.relative_to(_PROJECT_ROOT)} PÉRIMÉ")
        else:
            path.write_text(content, encoding="utf-8")
            print(f"  ✓ {path.relative_to(_PROJECT_ROOT)} ({len(names)} paramètres)")

    covered = sum(len(n) for _, _, n, _ in TABLES)
    print(f"\n{covered} paramètres publiés, {len(EXCLUDED)} exclus, "
          f"{len(values)} au total dans le preset.")
    if args.check and stale:
        print("\n[périmé] relancer sans --check.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
