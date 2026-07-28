#!/usr/bin/env python3
"""Poids de la sleeve momentum multi-instruments dans le portefeuille 4-sleeves.

Suite de `scripts/stress_tsmom_survivors.py` : les survivants du cycle
(`docs/research/momentum_expansion_2026H2.md` §4.3/§4.5) sont portés par le même
moteur TSMOM, donc ils n'entrent pas en production comme trois sleeves de plus —
ils entrent comme **une** sleeve momentum dont le budget de risque se partage
entre instruments, exactement ce que fait le portage MQL5 (`sub_equity/n`).

Ce script est une **recommandation chiffrée**, pas une modification : rien de la
production n'est touché, `PRODUCTION_WEIGHTS` reste ce qu'il est.

Ce qui est balayé — grille fermée, 2 compositions × 5 poids + la baseline :

- compositions : `GOLD_JPY` = {XAU-USD, USD-JPY}, `GOLD_JPY_XAG` = + {XAG-USD} ;
- poids momentum w ∈ {0,10 · 0,125 · 0,15 · 0,175 · 0,20}, la réduction se
  faisant sur MR Macro seul (`0,82 − w`, TS et RSI figés à 0,09 chacun, somme
  vérifiée à 1,0) ;
- baseline : **or seul à 10 %, poids 0,72/0,09/0,09/0,10** — figés en clair.
  Cette baseline dérivait de `PRODUCTION_WEIGHTS` jusqu'au 2026-07-28, ce qui
  rendait le sweep non reproductible : la promotion du trio a changé les poids
  ET le contenu du cache `Gold_Momentum`, déplaçant la fenêtre de 1822 à 2081
  séances. Un sweep dont la référence suit la production se réécrit à chaque
  changement de production, donc ne peut plus servir à juger ce changement.

Conventions, toutes reprises du cycle :

- **rendements par instrument** : `tsmom.pipeline`, config or de production,
  `fill="next_open"`, demi-spread par symbole (`costs.yml`), swap-drag
  0,5 bp/nuit × |exposition| — le modèle de `scripts/screen_tsmom_universe.py` ;
- **source la plus longue par instrument**, comme le screening : export minute
  long pour USD-JPY (2018→) plutôt que le daily broker (2020-11→), sans quoi la
  sleeve serait muette sur le premier tiers de la fenêtre du portefeuille ;
- **agrégation** : moyenne équipondérée sur les *n configurés*, alignée sur la
  date calendaire (`normalize` + `groupby`, cf. `stress_tsmom_survivors`) ; un
  instrument sans séance ce jour-là compte 0, il ne redistribue pas son poids ;
- **assemblage** : `build_combined_portfolio_v2` en `custom`, tv 0,37, cap 31,
  DD-cap OFF — la mécanique exacte de `build_production_portfolio`, seuls les
  poids et la quatrième sleeve changent ;
- **fenêtre** : celle de la production (l'intersection des quatre sleeves du
  cache), coupée à 2025-12-31 (`framework.holdout`). Elle est imposée à *toutes*
  les configurations : la sleeve momentum remonte à 2018 par l'export minute
  USD-JPY, la comparer à la baseline sur sa propre fenêtre changerait deux
  choses à la fois ;
- **annualisation** : 252 séances (`year_freq`), convention du dépôt — sans quoi
  le Sharpe sort ×√(365/252) au-dessus des chiffres publiés.

Écart de méthode, à lire avant de comparer les colonnes (§ rapport) : la sleeve
or de la baseline est celle du cache (`Gold_Momentum`, fill au close décideur,
sans slippage ni swap), tandis que la composante or des compositions passe par
les conventions du cycle (next_open + coûts + swap). Le delta de Sharpe entre
une composition et la baseline mélange donc l'effet « multi-instruments » et
l'effet « conventions d'exécution », ce dernier valant ~−0,07 de Sharpe sur la
sleeve or isolée (§2 de la note de cycle).

Usage : python scripts/sweep_momentum_weights.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import vectorbtpro as vbt

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "scripts"))

from framework import holdout, trials  # noqa: E402
from framework.costs import cost_for  # noqa: E402
from strategies import tsmom  # noqa: E402
from strategies.combined_portfolio import get_strategy_daily_returns  # noqa: E402
from strategies.combined_portfolio_v2 import (  # noqa: E402
    PRODUCTION_MAX_LEVERAGE,
    PRODUCTION_TARGET_VOL,
    PRODUCTION_WEIGHTS,
    build_combined_portfolio_v2,
)
from update_data_manifest import assert_manifest_fresh  # noqa: E402
from utils import apply_vbt_settings  # noqa: E402

# (symbole, loader) — le loader est celui qui donne l'historique le plus long,
# comme dans le screening ; il n'est pas toujours le défaut du registre.
COMPOSITIONS: dict[str, tuple[tuple[str, str], ...]] = {
    "GOLD_JPY": (("XAU-USD", "qc"), ("USD-JPY", "fx_minute")),
    "GOLD_JPY_XAG": (
        ("XAU-USD", "qc"),
        ("USD-JPY", "fx_minute"),
        ("XAG-USD", "yahoo"),
    ),
}

MOMENTUM_WEIGHTS: tuple[float, ...] = (0.10, 0.125, 0.15, 0.175, 0.20)

# Les deux diversificateurs FX ne bougent pas : la réduction se fait sur MR Macro.
FIXED_WEIGHTS: dict[str, float] = {"TS_Momentum_3p": 0.09, "RSI_Daily_3p": 0.09}
MOMENTUM_KEY = "Momentum_Multi"
BASELINE_LABEL = "BASELINE_or_seul"

# Référence figée : la production d'avant le cycle momentum. Ne PAS la relier à
# PRODUCTION_WEIGHTS — voir le docstring.
BASELINE_WEIGHTS: dict[str, float] = {
    "MR_Macro": 0.72, "TS_Momentum_3p": 0.09,
    "RSI_Daily_3p": 0.09, "Gold_Momentum": 0.10,
}

SWAP_BPS_PER_NIGHT = 0.00005  # Inp_SwapBpsPerNight de l'EA, charge par séance
SELECTION_END = pd.Timestamp("2025-12-31")  # = HOLDOUT_START, écrit en clair
DD_FLAG_PP = 0.03  # au-delà de 3 pp de maxDD sous la baseline : ⚠️

OUTPUT_PATH = _REPO / "reports" / "research" / "momentum_weights_sweep_2026H2.csv"

CSV_COLUMNS: tuple[str, ...] = (
    "config", "composition", "n_instruments", "w_momentum", "w_mr_macro",
    "sharpe", "cagr", "maxdd", "vol", "risk_contrib_momentum",
    "corr_momentum_mr", "n_bars", "first_date", "last_date", "flag",
)


# ═══════════════════════════════════════════════════════════════════════
# SLEEVE MOMENTUM
# ═══════════════════════════════════════════════════════════════════════


def instrument_returns(symbol: str, loader: str) -> pd.Series:
    """Rendements quotidiens nets d'un instrument, indexés sur la date calendaire.

    Le swap-drag est soustrait sur la série entière, avec l'exposition de la
    séance à laquelle il appartient, avant toute coupe de fenêtre.
    """
    pf, _ = tsmom.pipeline(
        symbol,
        loader_override=loader,
        fill="next_open",
        slippage=cost_for(symbol),
    )
    ret = pf.returns
    exposure = (pf.asset_value / pf.value).reindex(ret.index).fillna(0.0)
    ret_net = ret + SWAP_BPS_PER_NIGHT * exposure.abs() * tsmom.carry_sign(symbol)
    # Les loaders n'ont pas la même convention d'horodatage (dates de session
    # pour l'or, timestamps du broker pour le minute long) : la date calendaire
    # est le seul axe sur lequel les trois séries sont comparables.
    return ret_net.set_axis(ret_net.index.normalize()).groupby(level=0).sum()


def momentum_sleeve(instrument_rets: dict[str, pd.Series]) -> pd.Series:
    """Sleeve combinée = moyenne équipondérée sur les *n configurés*.

    Un instrument sans séance ce jour-là rend 0 et son poids n'est pas
    redistribué : c'est le budget que le portage MQL5 lui alloue
    (``sub_equity/n``), qu'il trade ou non.
    """
    frame = pd.concat(instrument_rets, axis=1, sort=True).fillna(0.0)
    return frame.mean(axis=1).rename(MOMENTUM_KEY)


# ═══════════════════════════════════════════════════════════════════════
# PORTEFEUILLE
# ═══════════════════════════════════════════════════════════════════════


def portfolio_weights(w_momentum: float) -> dict[str, float]:
    """Poids du 4-sleeves pour un poids momentum donné. Somme = 1,0 exactement."""
    w_mr = round(1.0 - sum(FIXED_WEIGHTS.values()) - w_momentum, 10)
    weights = {"MR_Macro": w_mr, **FIXED_WEIGHTS, MOMENTUM_KEY: w_momentum}
    total = sum(weights.values())
    if abs(total - 1.0) > 1e-12:
        raise ValueError(f"poids w={w_momentum}: somme {total!r} != 1.0")
    return weights


def _insample(series: pd.Series) -> pd.Series:
    ret = holdout.trim_insample(series)
    return ret[ret.index <= SELECTION_END]


def baseline_context() -> tuple[dict[str, pd.Series], dict[str, float]]:
    """Le cache avec la sleeve **or seul**, et les poids d'avant le cycle.

    ``cached["Gold_Momentum"]`` porte le trio depuis la promotion, et sa série
    remonte à 2000 par le loader Yahoo de l'argent : l'utiliser comme référence
    changerait la fenêtre de comparaison en même temps que les poids.
    """
    from strategies.combined_portfolio import backtest_momentum_sleeve

    cached = dict(get_strategy_daily_returns())
    cached["Gold_Momentum"] = backtest_momentum_sleeve(instruments=(("XAU-USD", None),))
    return cached, dict(BASELINE_WEIGHTS)


def selection_window(cached: dict[str, pd.Series]) -> pd.DatetimeIndex:
    """Fenêtre de la production : intersection des quatre sleeves, coupée au holdout.

    C'est la sleeve or du cache qui la borne à gauche (2019). Les compositions
    momentum remontent plus haut ; les y laisser courir comparerait la baseline
    et les candidats sur deux échantillons différents.
    """
    frame = pd.DataFrame({k: _insample(cached[k]) for k in BASELINE_WEIGHTS})
    return frame.dropna().index


def evaluate(
    label: str,
    composition: str,
    n_instruments: int,
    sleeves: dict[str, pd.Series],
    weights: dict[str, float],
    momentum_key: str,
    window: pd.DatetimeIndex,
) -> dict[str, object]:
    """Une ligne du CSV : assemblage v2 puis métriques de la fenêtre de sélection."""
    filtered = {k: sleeves[k].reindex(window) for k in weights}
    res = build_combined_portfolio_v2(
        filtered,
        allocation="custom",
        custom_weights=weights,
        target_vol=PRODUCTION_TARGET_VOL,
        max_leverage=PRODUCTION_MAX_LEVERAGE,
        dd_cap_enabled=False,
    )

    common = res["component_returns"]
    holdout.assert_not_optimizing(common.index)

    # Contribution au risque : covariance de la jambe pondérée avec le
    # portefeuille non leviéré, rapportée à sa variance. La somme sur les
    # quatre sleeves vaut 1 par construction.
    base = res["port_rets_base"]
    weighted = common[momentum_key] * weights[momentum_key]
    risk_contrib = float(weighted.cov(base) / base.var())

    return dict(
        config=label,
        composition=composition,
        n_instruments=n_instruments,
        w_momentum=weights[momentum_key],
        w_mr_macro=weights["MR_Macro"],
        sharpe=res["sharpe"],
        cagr=res["annual_return"],
        maxdd=res["max_drawdown"],
        vol=res["annual_vol"],
        risk_contrib_momentum=risk_contrib,
        corr_momentum_mr=float(common[momentum_key].corr(common["MR_Macro"])),
        n_bars=len(common),
        first_date=str(common.index.min().date()),
        last_date=str(common.index.max().date()),
        flag="",
    )


def format_table(df: pd.DataFrame) -> str:
    """Tableau du sweep, baseline en tête puis les candidats par Sharpe."""
    shown = df.copy()
    for col, fmt in (
        ("w_momentum", "{:.3f}"), ("w_mr_macro", "{:.3f}"), ("sharpe", "{:.3f}"),
        ("cagr", "{:.2%}"), ("maxdd", "{:.2%}"), ("vol", "{:.2%}"),
        ("risk_contrib_momentum", "{:.1%}"), ("corr_momentum_mr", "{:+.3f}"),
    ):
        shown[col] = shown[col].map(lambda v, f=fmt: f.format(v))
    return shown.to_string(index=False)


def main() -> int:
    apply_vbt_settings()
    assert_manifest_fresh()
    vbt.settings.returns.year_freq = pd.Timedelta(days=252)

    print(f"\n{'=' * 92}")
    print("  Sweep du poids de la sleeve momentum multi-instruments — recommandation")
    print(f"  2 compositions × {len(MOMENTUM_WEIGHTS)} poids + baseline (or seul à 0,10), "
          f"fenêtre ≤ {SELECTION_END.date()}")
    print(f"  assemblage v2 custom, tv={PRODUCTION_TARGET_VOL}, "
          f"cap={PRODUCTION_MAX_LEVERAGE}, DD-cap OFF — production NON modifiée")
    print(f"{'=' * 92}\n")

    n_configs = 1 + len(COMPOSITIONS) * len(MOMENTUM_WEIGHTS)
    trials.log_trials(
        "integration_weights", n_configs,
        "baseline + 2 compositions x 5 poids de sleeve momentum",
        config_key="integration_weights:11cfg:gold_jpy_xag_grid",
    )

    cached, baseline_weights = baseline_context()

    # Contrôle : la production telle qu'elle tourne aujourd'hui, fenêtre pleine,
    # pour situer la baseline (coupée à 2025-12-31) contre le chiffre publié.
    prod_cache = get_strategy_daily_returns()
    full = build_combined_portfolio_v2(
        {k: prod_cache[k] for k in PRODUCTION_WEIGHTS},
        allocation="custom",
        custom_weights=PRODUCTION_WEIGHTS,
        target_vol=PRODUCTION_TARGET_VOL,
        max_leverage=PRODUCTION_MAX_LEVERAGE,
        dd_cap_enabled=False,
    )
    print(f"  contrôle production, fenêtre pleine : sharpe={full['sharpe']:.3f} "
          f"(référence publiée combined_portfolio_v2 : 1,084)")

    window = selection_window(cached)
    rows: list[dict[str, object]] = [
        evaluate(
            BASELINE_LABEL, "GOLD_ONLY", 1, cached, baseline_weights,
            momentum_key="Gold_Momentum", window=window,
        )
    ]
    baseline_dd = abs(float(rows[0]["maxdd"]))
    print(f"  baseline (or seul, w=0,10, ≤ {SELECTION_END.date()}) : "
          f"sharpe={rows[0]['sharpe']:.3f} — c'est elle qui fait foi pour les deltas\n")

    for name, instruments in COMPOSITIONS.items():
        print(f"  {name}", flush=True)
        rets = {}
        for symbol, loader in instruments:
            print(f"    {symbol:<9} [{loader}]", flush=True)
            rets[symbol] = instrument_returns(symbol, loader)
        # La sleeve est définie par la règle « absent = 0 » : la recaler sur la
        # fenêtre de production ne peut donc pas produire de trou.
        sleeve = momentum_sleeve(rets).reindex(window).fillna(0.0)
        sleeves = {**cached, MOMENTUM_KEY: sleeve}
        for w in MOMENTUM_WEIGHTS:
            rows.append(
                evaluate(
                    f"{name}_w{w:.3f}", name, len(instruments), sleeves,
                    portfolio_weights(w), momentum_key=MOMENTUM_KEY, window=window,
                )
            )

    baseline_row, candidates = rows[0], rows[1:]
    candidates.sort(key=lambda r: r["sharpe"], reverse=True)
    for row in candidates:
        if abs(float(row["maxdd"])) > baseline_dd + DD_FLAG_PP:
            row["flag"] = "⚠️"

    df = pd.DataFrame([baseline_row, *candidates], columns=list(CSV_COLUMNS))
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"\n{format_table(df)}\n")
    print(f"⚠️ = maxDD dégradé de plus de {DD_FLAG_PP:.0%} (pp) contre la baseline "
          f"({-baseline_dd:.2%})")
    print(f"\nWritten: {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
