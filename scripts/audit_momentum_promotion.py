#!/usr/bin/env python3
"""Validation rétrospective de la promotion du trio momentum à 20 %.

Applique au cycle momentum 2026-H2 le gate que la Phase 21 s'était donné et
n'a jamais appliqué : *promouvoir seulement si PBO < 0,5*
(`docs/research/phase21_2026-04-13_dsr_retrofit.md`). Trois mesures, aucune
configuration nouvelle — ce script **ne consomme aucun essai** et n'appelle donc
pas `trials.log_trials` : ré-évaluer des configurations déjà loguées n'est pas
un test nouveau.

Les conséquences de chaque issue sont écrites et commitées **avant** exécution
dans `docs/research/momentum_validation_2026H2.md` §1. Ce script ne décide de
rien : il produit les chiffres que cette table arbitre.

Ce qui est mesuré :

- **attribution** — d'où vient le résultat publié, par sleeve, par instrument et
  par position, scindé à la frontière du holdout (2026-01-01). Lecture d'une
  tranche gelée déjà publiée : aucune sélection, aucun classement ;
- **PBO-I** — la sélection de 3 instruments parmi les 21 classés. C'est le seul
  endroit du cycle où un choix réel a eu lieu, donc le seul PBO qui porte de
  l'information ;
- **PBO-W** — les 11 configurations du sweep de poids. Application littérale du
  gate, mais 10 colonnes sur 11 ne diffèrent que par un scalaire sur une sleeve
  commune : un CSCV sur une grille monotone unidimensionnelle rendra PBO ≈ 0
  quelle que soit la réalité. Calculé parce que c'est le gate promis, publié
  avec cette mise en garde ;
- **DSR** — déflaté par univers de trials nommé, parce que le résultat n'est pas
  uniforme : le Sharpe du portefeuille survit à un N correct, celui de chaque
  instrument non.

Le PBO sur les 3 compositions n'est pas calculé : 3 colonnes ⇒ le logit ne peut
prendre que 3 valeurs, la statistique est dégénérée.

Usage :
    python scripts/audit_momentum_promotion.py --selfcheck   # d'abord
    python scripts/audit_momentum_promotion.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import vectorbtpro as vbt

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "scripts"))

import sweep_momentum_weights as smw  # noqa: E402
from framework import holdout, trials  # noqa: E402
from framework.statistical_testing import (  # noqa: E402
    deflated_sharpe_ratio,
    probability_of_backtest_overfitting,
)
from screen_tsmom_universe import build_universe  # noqa: E402
from strategies.combined_portfolio import get_strategy_daily_returns  # noqa: E402
from strategies.combined_portfolio_v2 import (  # noqa: E402
    PRODUCTION_MAX_LEVERAGE,
    PRODUCTION_TARGET_VOL,
    build_combined_portfolio_v2,
)
from utils import apply_vbt_settings  # noqa: E402

ANN_DAYS = 252
YEAR_FREQ = pd.Timedelta(days=ANN_DAYS)

DEALS_PATH = _REPO / "reports" / "mt5" / "prod_ref_trio20_deals.csv"
SWEEP_CSV = _REPO / "reports" / "research" / "momentum_weights_sweep_2026H2.csv"
MT5_SWEEP_CSV = _REPO / "reports" / "mt5" / "tsmom_universe_sweep.csv"
OUT_DIR = _REPO / "reports" / "research"
# Consommée par scripts/build_latex_report_assets.py : le DSR et le PBO du
# rapport client doivent porter sur cette grille, pas sur les séries de sleeves.
WEIGHTS_RETURNS_CSV = OUT_DIR / "momentum_weights_returns_2026H2.csv"

# Le CSV de deals sort du tester en UTF-16, séparateur virgule.
DEALS_ENCODING = "utf-16"
_DEAL_TYPE_BALANCE = 2  # DEAL_TYPE_BALANCE — le dépôt initial, pas un trade

# n_bins impose C(n, n/2) splits : 10 → 252, 12 → 924, 16 → 12 870 (overkill).
PBO_BINS: tuple[int, ...] = (8, 10, 12, 16)
PBO_BINS_INSTRUMENTS: tuple[int, ...] = (8, 10, 12)

PROMOTED = ("XAUUSD", "USDJPY", "XAGUSD")


# ═══════════════════════════════════════════════════════════════════════
# ATTRIBUTION — d'où vient le résultat publié
# ═══════════════════════════════════════════════════════════════════════


def load_reference_deals() -> pd.DataFrame:
    """Deals du run de référence, avec le net par deal et la date de sortie.

    Le P&L d'une position est porté par son deal de sortie ; l'agréger par
    ``position_id`` évite le piège d'attribution du cycle précédent, où découper
    par date de sortie attribuait à la tranche gelée l'intégralité d'une
    position ouverte bien avant.
    """
    deals = pd.read_csv(DEALS_PATH, encoding=DEALS_ENCODING)
    deals["time_utc"] = pd.to_datetime(deals["time_utc"], format="%Y.%m.%d %H:%M:%S")
    deals["net"] = deals["profit"] + deals["commission"] + deals["swap"]
    # type=2 est l'écriture de balance (le dépôt de 10 000) : ce n'est pas un
    # trade, et la laisser gonfle le dénominateur de 15 % — toutes les parts
    # publiées s'en trouveraient sous-estimées.
    return deals[deals["type"] != _DEAL_TYPE_BALANCE].copy()


def positions(deals: pd.DataFrame) -> pd.DataFrame:
    """Une ligne par position : symbole, sleeve, ouverture, clôture, net.

    Le tri chronologique fait que ``first`` prend le deal d'entrée, donc le vrai
    magic de la position : les clôtures forcées de fin de test sortent avec
    magic 0 et ``sleeve=OTHER``, et seraient sinon détachées de leur sleeve.
    """
    grouped = deals.sort_values("time_utc").groupby("position_id").agg(
        symbol=("symbol", "first"),
        sleeve=("sleeve", "first"),
        magic=("magic", "first"),
        opened=("time_utc", "min"),
        closed=("time_utc", "max"),
        net=("net", "sum"),
    )
    grouped["frozen"] = grouped["closed"] >= holdout.HOLDOUT_START
    return grouped.sort_values("net", ascending=False)


def attribution_rows(pos: pd.DataFrame, net_total: float) -> list[dict[str, object]]:
    """Scission in-sample / holdout par sleeve puis par instrument, plus le top."""
    rows: list[dict[str, object]] = []

    for scope, key in (("sleeve", "sleeve"), ("instrument", "symbol")):
        for name, sub in pos.groupby(key):
            frozen = sub[sub["frozen"]]
            insample = sub[~sub["frozen"]]
            rows.append(
                dict(
                    scope=scope,
                    name=name,
                    n_positions=len(sub),
                    net_total=round(float(sub["net"].sum()), 2),
                    net_insample=round(float(insample["net"].sum()), 2),
                    net_frozen=round(float(frozen["net"].sum()), 2),
                    share_of_net=round(float(sub["net"].sum()) / net_total, 4),
                    opened=None,
                    closed=None,
                )
            )

    for position_id, row in pos.head(5).iterrows():
        rows.append(
            dict(
                scope="top_position",
                name=f"{row['symbol']} #{position_id}",
                n_positions=1,
                net_total=round(float(row["net"]), 2),
                net_insample=0.0 if row["frozen"] else round(float(row["net"]), 2),
                net_frozen=round(float(row["net"]), 2) if row["frozen"] else 0.0,
                share_of_net=round(float(row["net"]) / net_total, 4),
                opened=str(row["opened"].date()),
                closed=str(row["closed"].date()),
            )
        )
    return rows


# ═══════════════════════════════════════════════════════════════════════
# MATRICES
# ═══════════════════════════════════════════════════════════════════════


def instrument_matrix() -> pd.DataFrame:
    """Rendements quotidiens nets des 21 instruments classés, conventions du screen.

    Pas de ``fillna(0.0)`` : un instrument qui ne cote pas n'est pas une position
    plate, et un zéro écraserait sa volatilité, donc son rang OOS. Les colonnes
    sont laissées trouées ; c'est l'appelant qui choisit sa coupe.
    """
    series: dict[str, pd.Series] = {}
    for symbol, loader in build_universe():
        print(f"    {symbol:<9} [{loader}]", flush=True)
        ret = smw.instrument_returns(symbol, loader)
        series[symbol] = holdout.trim_insample(ret)
    return pd.DataFrame(series)


def sweep_context() -> tuple[dict[str, pd.Series], dict[str, float]]:
    """Le contexte dans lequel le sweep de poids a tourné — reconstruit, pas lu.

    Le sweep publié n'est **plus** reproductible depuis ``PRODUCTION_WEIGHTS`` et
    le cache, parce que la promotion a changé les deux : la baseline « or seul à
    0,10 » est devenue « trio à 0,20 », et ``cached["Gold_Momentum"]`` porte
    désormais le trio, dont l'historique remonte à 2000 par la série Yahoo de
    l'argent. L'intersection des quatre sleeves n'est donc plus bornée à gauche
    par l'or : la fenêtre passe de 1822 séances (2019-01-02) à 2081 (2018-01-02),
    et tous les Sharpe bougent.

    On refait donc la sleeve or seul et on reprend les poids d'alors. Le
    ``--selfcheck`` vérifie que cela redonne le CSV publié ; sans quoi le PBO-W
    porterait sur une grille qui n'est pas celle qui a produit la recommandation.
    """
    from strategies.combined_portfolio import backtest_momentum_sleeve

    cached = dict(get_strategy_daily_returns())
    cached["Gold_Momentum"] = backtest_momentum_sleeve(instruments=(("XAU-USD", None),))
    baseline_weights = {
        "MR_Macro": 0.72,
        **smw.FIXED_WEIGHTS,
        "Gold_Momentum": 0.10,
    }
    return cached, baseline_weights


def weight_matrix(
    cached: dict[str, pd.Series],
    baseline_weights: dict[str, float],
) -> tuple[pd.DataFrame, pd.Series]:
    """Rendements des 11 configurations du sweep de poids, et leurs Sharpe.

    Reconstruction à l'identique de ``sweep_momentum_weights`` : c'est le
    ``--selfcheck`` qui le prouve, en confrontant les Sharpe au CSV publié.
    """
    window = smw.selection_window(cached)
    columns: dict[str, pd.Series] = {}
    sharpes: dict[str, float] = {}

    def add(label: str, sleeves: dict[str, pd.Series], weights: dict[str, float]) -> None:
        res = build_combined_portfolio_v2(
            {k: sleeves[k].reindex(window) for k in weights},
            allocation="custom",
            custom_weights=weights,
            target_vol=PRODUCTION_TARGET_VOL,
            max_leverage=PRODUCTION_MAX_LEVERAGE,
            dd_cap_enabled=False,
        )
        columns[label] = res["portfolio_returns"]
        sharpes[label] = float(res["sharpe"])

    add(smw.BASELINE_LABEL, cached, baseline_weights)

    for name, instruments in smw.COMPOSITIONS.items():
        print(f"    {name}", flush=True)
        rets = {sym: smw.instrument_returns(sym, loader) for sym, loader in instruments}
        sleeve = smw.momentum_sleeve(rets).reindex(window).fillna(0.0)
        sleeves = {**cached, smw.MOMENTUM_KEY: sleeve}
        for w in smw.MOMENTUM_WEIGHTS:
            add(f"{name}_w{w:.3f}", sleeves, smw.portfolio_weights(w))

    return pd.DataFrame(columns).dropna(how="any"), pd.Series(sharpes)


# ═══════════════════════════════════════════════════════════════════════
# MESURES
# ═══════════════════════════════════════════════════════════════════════


def pbo_rows(
    matrix: pd.DataFrame,
    label: str,
    bins: tuple[int, ...],
    post_hoc: bool = False,
) -> list[dict]:
    """Une ligne par n_bins — la courbe entière, pas le point qui arrange.

    ``post_hoc`` marque une découpe décidée **après** avoir vu le résultat
    global : elle informe, elle ne décide pas. Sans ce drapeau, re-découper une
    matrice jusqu'à trouver le sous-ensemble qui passe serait exactement le
    biais que le pré-gel de la table de décision cherche à empêcher.
    """
    rows = []
    for n_bins in bins:
        if len(matrix) < n_bins * 2:
            continue
        res = probability_of_backtest_overfitting(matrix, n_bins=n_bins)
        rows.append(
            dict(
                matrix=label,
                post_hoc=post_hoc,
                n_configs=matrix.shape[1],
                n_bars=len(matrix),
                first_date=str(matrix.index.min().date()),
                last_date=str(matrix.index.max().date()),
                n_bins=n_bins,
                n_splits=res["n_splits"],
                pbo=round(float(res["pbo"]), 4),
                verdict="SAIN" if res["pbo"] < 0.5 else "OVERFIT",
            )
        )
    return rows


def weight_subsets(matrix: pd.DataFrame) -> dict[str, list[str]]:
    """Découpes post-hoc du sweep de poids, pour situer d'où vient le PBO.

    La matrice complète mélange deux décisions — quelle composition, et quel
    poids — là où les sous-ensembles à composition fixée n'en portent qu'une.
    """
    return {
        "weights_trio_only": [c for c in matrix.columns if "XAG" in c],
        "weights_duo_only": [c for c in matrix.columns if c.startswith("GOLD_JPY_w")],
    }


def dsr_rows(
    weight_matrix_df: pd.DataFrame,
    weight_sharpes: pd.Series,
    instruments: pd.DataFrame,
) -> list[dict[str, object]]:
    """DSR du portefeuille promu et de chaque instrument, univers de trials nommé.

    Le même Sharpe déflaté par deux univers différents donne deux verdicts : le
    tableau les publie côte à côte plutôt que d'en choisir un.
    """
    promoted = f"GOLD_JPY_XAG_w{max(smw.MOMENTUM_WEIGHTS):.3f}"
    n_distinct = trials.distinct_trials()
    n_raw = trials.total_trials()

    mt5 = pd.read_csv(MT5_SWEEP_CSV)
    mt5_sharpes = pd.to_numeric(mt5["sharpe"], errors="coerce").dropna()

    rows: list[dict[str, object]] = []

    def add(obj: str, returns: pd.Series, universe: str, n: int, sharpes) -> None:
        res = deflated_sharpe_ratio(
            returns,
            n_trials=n,
            trial_sharpes=np.asarray(sharpes, dtype=float),
            freq="1D",
            year_freq=YEAR_FREQ,
        )
        rows.append(
            dict(
                deflated_object=obj,
                trial_universe=universe,
                n_trials=n,
                sharpe=round(float(res["sharpe"]), 4),
                sharpe_std=round(float(res["sharpe_std"]), 4),
                expected_max_sharpe=round(float(res["expected_max_sharpe"]), 4),
                dsr=round(float(res["dsr"]), 4),
                verdict="PASS" if res["dsr"] >= 0.95 else "FAIL",
            )
        )

    top = weight_matrix_df[promoted]
    add(promoted, top, "sweep de poids (11 Sharpe portefeuille)",
        int(weight_sharpes.size), weight_sharpes.to_numpy())
    add(promoted, top, "registre, configurations distinctes",
        n_distinct, weight_sharpes.to_numpy())
    add(promoted, top, "registre brut, re-runs inclus",
        n_raw, weight_sharpes.to_numpy())

    for symbol in instruments.columns:
        if symbol.replace("-", "") not in PROMOTED:
            continue
        add(symbol, instruments[symbol].dropna(),
            "classement MT5 des 21 instruments", n_distinct, mt5_sharpes.to_numpy())

    return rows


# ═══════════════════════════════════════════════════════════════════════
# SELFCHECK
# ═══════════════════════════════════════════════════════════════════════


def selfcheck() -> int:
    """La reconstruction doit redonner les Sharpe publiés du sweep de poids.

    Si elle ne les redonne pas, la matrice du PBO-W n'est pas celle qui a
    produit la recommandation, et le chiffre ne vaut rien.
    """
    apply_vbt_settings()
    vbt.settings.returns.year_freq = YEAR_FREQ

    published = pd.read_csv(SWEEP_CSV).set_index("config")["sharpe"]
    print(f"  reconstruction des {len(published)} configurations du sweep…", flush=True)
    matrix, rebuilt = weight_matrix(*sweep_context())

    expected_bars = int(pd.read_csv(SWEEP_CSV)["n_bars"].iloc[0])
    if len(matrix) != expected_bars:
        print(f"  ÉCHEC : {len(matrix)} séances reconstruites contre {expected_bars} "
              "publiées — la fenêtre n'est pas celle du sweep.")
        return 1

    missing = set(published.index) - set(rebuilt.index)
    if missing:
        print(f"  ÉCHEC : configurations absentes de la reconstruction : {sorted(missing)}")
        return 1

    delta = (rebuilt[published.index] - published).abs()
    worst = float(delta.max())
    print(f"\n{'config':<24} {'publié':>10} {'reconstruit':>12} {'écart':>12}")
    for config in published.index:
        print(f"{config:<24} {published[config]:>10.6f} "
              f"{rebuilt[config]:>12.6f} {delta[config]:>12.2e}")

    tol = 1e-9
    if worst > tol:
        print(f"\n  ÉCHEC : écart max {worst:.2e} > {tol:.0e} — la matrice n'est pas "
              "celle du sweep publié, rien ne doit être publié sur cette base.")
        return 1
    print(f"\n  OK : écart max {worst:.2e} ≤ {tol:.0e}")
    return 0


# ═══════════════════════════════════════════════════════════════════════


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selfcheck", action="store_true",
                        help="vérifie la reconstruction contre le CSV publié, puis sort")
    args = parser.parse_args()

    if args.selfcheck:
        return selfcheck()

    apply_vbt_settings()
    vbt.settings.returns.year_freq = YEAR_FREQ
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 92}")
    print("  Validation rétrospective de la promotion du trio momentum")
    print("  table de décision : docs/research/momentum_validation_2026H2.md §1 (pré-gelée)")
    print(f"  trials : {trials.distinct_trials()} distincts / {trials.total_trials()} bruts")
    print(f"{'=' * 92}\n")

    # ── Attribution ────────────────────────────────────────────────────
    print("  [1/4] attribution du run de référence", flush=True)
    deals = load_reference_deals()
    pos = positions(deals)
    net_total = float(pos["net"].sum())
    attribution = pd.DataFrame(attribution_rows(pos, net_total))
    attribution.to_csv(OUT_DIR / "momentum_attribution_2026H2.csv", index=False)

    frozen_share = float(pos[pos["frozen"]]["net"].sum()) / net_total
    top_share = float(pos["net"].iloc[0]) / net_total
    print(f"        net des positions : {net_total:>12,.2f}")
    print(f"        part fermée dans la tranche gelée : {frozen_share:.1%}")
    print(f"        plus grosse position : {top_share:.1%} du net\n")

    # ── PBO-I ──────────────────────────────────────────────────────────
    print("  [2/4] PBO-I — sélection de 3 instruments parmi 21", flush=True)
    instruments = instrument_matrix()
    common = instruments.dropna(how="any")
    rows = pbo_rows(common, "instruments_common", PBO_BINS_INSTRUMENTS)

    deep = instruments.dropna(axis=1, thresh=int(len(instruments) * 0.8)).dropna(how="any")
    if deep.shape[1] >= 4 and len(deep) > len(common):
        rows += pbo_rows(deep, "instruments_deep", PBO_BINS_INSTRUMENTS)

    # ── PBO-W ──────────────────────────────────────────────────────────
    print("\n  [3/4] PBO-W — 11 configurations du sweep de poids", flush=True)
    weights_df, weight_sharpes = weight_matrix(*sweep_context())
    # Archivée : c'est la grille que le rapport client doit déflater et passer
    # au CSCV, à la place des 4-6 séries de sleeves qu'il utilisait.
    weights_df.to_csv(WEIGHTS_RETURNS_CSV)
    rows += pbo_rows(weights_df, "weights_sweep", PBO_BINS)
    for label, columns in weight_subsets(weights_df).items():
        rows += pbo_rows(weights_df[columns], label, PBO_BINS, post_hoc=True)

    pbo = pd.DataFrame(rows)
    pbo.to_csv(OUT_DIR / "momentum_pbo_2026H2.csv", index=False)

    # ── DSR ────────────────────────────────────────────────────────────
    print("\n  [4/4] DSR par univers de trials", flush=True)
    dsr = pd.DataFrame(dsr_rows(weights_df, weight_sharpes, instruments))
    dsr.to_csv(OUT_DIR / "momentum_dsr_2026H2.csv", index=False)

    print(f"\n{'─' * 92}\n  PBO\n{'─' * 92}")
    print(pbo.to_string(index=False))
    print(f"\n{'─' * 92}\n  DSR\n{'─' * 92}")
    print(dsr.to_string(index=False))
    print(f"\n{'─' * 92}\n  ATTRIBUTION (sleeves et instruments)\n{'─' * 92}")
    print(attribution[attribution["scope"] != "top_position"].to_string(index=False))
    print(f"\n{'─' * 92}\n  POSITIONS DOMINANTES\n{'─' * 92}")
    print(attribution[attribution["scope"] == "top_position"].to_string(index=False))

    print(f"\n  Écrit dans {OUT_DIR}/ : momentum_pbo, momentum_dsr, "
          "momentum_attribution _2026H2.csv")
    print("  La conséquence est celle du §1 de la note, sans rediscussion.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
