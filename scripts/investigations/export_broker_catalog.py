#!/usr/bin/env python3
"""Exporte le catalogue broker complet et l'archive dans le dépôt.

`FxListSymbols.mq5` écrit dans `MQL5/Files/`, hors dépôt : le catalogue du
courtier — 272 symboles, leurs spreads, leurs swaps, leurs plafonds de volume —
n'existait donc **nulle part** dans le repo. Résultat, chaque cycle de recherche
redécouvrait par backtest ce qu'une propriété statique de symbole dit
gratuitement : JPN225 non exécutable (plafond de volume face à un tick value
minuscule), indices sous cap, EM jamais qualifiés faute de connaître leur spread.

Ce script lance le catalogue en headless (même pattern que
`download_history.py`), copie le CSV dans `data/broker/` et applique les gates
d'éligibilité **avant** tout backtest, donc pour zéro essai consommé.

Les trois gates, dans l'ordre où ils tuent :

1. ``trade_mode != FULL`` — pas d'ouverture possible, ou close-only ;
2. ``spread >= 2x`` celui d'EURUSD — critère n°4 du plan d'expansion FX de
   2026-05-04, enfin applicable ;
Le plafond de volume, lui, n'est **pas** décidable ici : ``SYMBOL_TRADE_TICK_VALUE``
vaut 0 tant que le symbole n'est pas dans le MarketWatch (259 lignes sur 272 dans
cet export), donc le notionnel en dollars n'est pas calculable. La colonne
``max_units`` donne la taille brute ouvrable ; le cap d'exécution réel reste à
constater par un run, comme pour les indices du cycle précédent.

Usage :
    python scripts/investigations/export_broker_catalog.py
    python scripts/investigations/export_broker_catalog.py --no-run   # relit le CSV existant
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT / "scripts" / "investigations"))

from download_history import (  # noqa: E402
    PORTABLE,
    run_mt5_with_script,
)

RAW_CSV = PORTABLE / "MQL5" / "Files" / "symbols_catalog.csv"
OUT_DIR = _ROOT / "data" / "broker"

# SYMBOL_TRADE_MODE : 4 = FULL. Tout le reste (disabled, longonly, shortonly,
# closeonly) empêche la sleeve d'ouvrir librement.
TRADE_MODE_FULL = 4

REFERENCE_SYMBOL = "EURUSD.c"
SPREAD_MULTIPLE_MAX = 2.0


def export_catalog(timeout: int = 300) -> bool:
    """Lance MT5 headless sur FxListSymbols et attend le marqueur de fin."""
    return run_mt5_with_script(
        "FxListSymbols", "FxListSymbols done", timeout, time.time()
    )


def load_catalog(path: Path) -> pd.DataFrame:
    # FILE_ANSI côté MQL5 : le CSV sort dans la page de codes Windows, pas en
    # UTF-8. Les descriptions de symboles contiennent des espaces insécables
    # (0xA0) qui font échouer une lecture UTF-8 stricte.
    df = pd.read_csv(path, encoding="cp1252")
    numeric = [
        "spread_current", "swap_long", "swap_short", "contract_size",
        "volume_min", "volume_max", "volume_step", "tick_value", "tick_size",
        "margin_initial", "stops_level", "trade_mode",
    ]
    for col in numeric:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def classify(df: pd.DataFrame) -> pd.DataFrame:
    """Ajoute ``verdict`` et ``verdict_reason`` — gates statiques, zéro backtest."""
    ref = df.loc[df["name"] == REFERENCE_SYMBOL, "spread_current"]
    ref_spread = float(ref.iloc[0]) if len(ref) else float("nan")

    # Taille maximale ouvrable, en unités de l'actif sous-jacent.
    #
    # Le notionnel en dollars n'est PAS calculable ici, et il faut le dire
    # plutôt que de l'approcher : SYMBOL_TRADE_TICK_VALUE est une propriété
    # dynamique qui vaut 0 tant que le symbole n'est pas dans le MarketWatch —
    # elle est nulle sur 259 des 272 lignes de cet export. Une première version
    # de ce gate s'en servait et classait 207 symboles « plafonnés », dont
    # XAUUSD et USDJPY qui tradent en production : un gate qui rejette ce qui
    # tourne déjà se réfute tout seul.
    #
    # Le plafond d'exécution réel demande donc un run, ou un export enrichi
    # après SymbolSelect. Ce qui reste ici — swaps et spreads — est statique,
    # fiable, et suffit aux deux gates qui décident.
    df = df.assign(
        ref_spread=ref_spread,
        spread_ratio=df["spread_current"] / ref_spread if ref_spread else pd.NA,
        max_units=df["volume_max"] * df["contract_size"],
    )

    def verdict(row) -> tuple[str, str]:
        if row["trade_mode"] != TRADE_MODE_FULL:
            return "REJECT_NOT_TRADABLE", f"trade_mode={int(row['trade_mode'])}"
        if pd.notna(row["spread_ratio"]) and row["spread_ratio"] >= SPREAD_MULTIPLE_MAX:
            return "REJECT_SPREAD", f"spread {row['spread_ratio']:.1f}x EURUSD"
        return "ELIGIBLE", ""

    verdicts = df.apply(verdict, axis=1, result_type="expand")
    df["verdict"] = verdicts[0]
    df["verdict_reason"] = verdicts[1]
    return df


def carry_view(df: pd.DataFrame) -> pd.DataFrame:
    """Les éligibles, classés par portage long décroissant.

    C'est la table qui convertit la thèse du dossier — un moteur long-only lent
    doit préférer les instruments payés pour attendre — en un critère lisible
    avant tout backtest.
    """
    keep = df[df["verdict"] == "ELIGIBLE"].copy()
    return keep.sort_values("swap_long", ascending=False)[
        ["name", "swap_long", "swap_short", "spread_current", "spread_ratio",
         "max_units", "verdict"]
    ]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-run", action="store_true",
                    help="ne relance pas MT5, relit le CSV déjà exporté")
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--date", default="2026-07-28", help="suffixe du fichier archivé")
    args = ap.parse_args()

    if not args.no_run:
        print("[1/3] export du catalogue via MT5 headless...", flush=True)
        if not export_catalog(args.timeout):
            print("[!] marqueur de fin non détecté — le CSV peut être partiel")

    if not RAW_CSV.exists():
        print(f"[FAIL] {RAW_CSV} absent. Lancer sans --no-run.")
        return 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    archived = OUT_DIR / f"symbols_catalog_{args.date}.csv"
    shutil.copy2(RAW_CSV, archived)
    print(f"[2/3] archivé → {archived.relative_to(_ROOT)}")

    print("[3/3] gates d'éligibilité (statiques, zéro backtest)")
    df = classify(load_catalog(archived))
    df.to_csv(OUT_DIR / f"symbols_eligibility_{args.date}.csv", index=False)

    print(f"\n  {len(df)} symboles au catalogue")
    for verdict, sub in df.groupby("verdict"):
        print(f"    {verdict:<22} {len(sub):>4}")

    carry = carry_view(df)
    print(f"\n  Éligibles par portage long décroissant (top 20 sur {len(carry)}) :")
    print(carry.head(20).to_string(index=False))
    print(f"\n  {int((carry['swap_long'] > 0).sum())} éligibles ont un portage long positif.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
