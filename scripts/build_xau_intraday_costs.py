#!/usr/bin/env python3
"""Le spread XAUUSD **mesuré**, par heure de New York et par année.

`docs/specs/xau_x10_spec.md` §12 fait de la fourchette le premier terme de
l'équation : elle vaut de 5 % à 50 % d'un ATR M5 selon l'année. La campagne
in-sample a pourtant tourné à **0,29 $ constant**, faute de mesure — le terminal
ne joignait plus le serveur de démo du broker. Le dump M1 du run de référence
MT5 (`Inp_DumpBars`) porte enfin un `spread_points` par barre, du 2022-11-04 au
2025-12-30.

Ce que ce script écrit, et ce qu'il n'écrit pas :

* il écrit `costs_xau_intraday.yml` à la racine — médiane et p75 du spread en
  dollars, par **heure de New York × année**, `source: mt5_dump` pour les années
  couvertes par le dump, `source: extrapolated` pour 2019-2022 ;
* il ne touche **pas** `costs.yml` : la stratégie 1 ne doit rien voir de ce
  chantier (§12) ;
* il ne re-sélectionne **rien**. §1.1 de la table de décision gèle la règle : le
  spread mesuré est une sensibilité publiée, jamais un critère de choix.

**L'extrapolation d'avant 2022-11.** Il n'existe aucune mesure du spread du
broker sur cette période. La seule régularité disponible est que la fourchette
d'un métal suit grossièrement son prix : on applique donc à chaque heure le
ratio des prix médians annuels, `prix_median(année) / prix_median(2023)`, 2023
étant la première année pleinement couverte par le dump. C'est une hypothèse,
elle est fausse dans le détail, et c'est pour cela que chaque ligne concernée
porte `source: extrapolated` — un consommateur qui publierait ces chiffres sans
la mention publierait une mesure qui n'existe pas.

Usage
-----
    uv run python scripts/build_xau_intraday_costs.py \
        --dump data/XAU-USD_minute_mt5_x10dump.parquet \
        --out costs_xau_intraday.yml
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

#: §12 — `point = 0,001 $` pour XAUUSD.c, `data/broker/symbols_catalog_2026-07-28.csv`.
POINT = 0.001

#: §2 — toute l'horloge de la stratégie est celle de New York.
SESSION_TZ = "America/New_York"

#: Année de référence de l'extrapolation : la première entièrement couverte par
#: le dump. 2022 ne l'est pas (le run démarre le 4 novembre).
EXTRAPOLATION_BASE_YEAR = 2023

#: Années que la campagne in-sample couvre et que le dump ne couvre pas.
EXTRAPOLATED_YEARS: tuple[int, ...] = (2019, 2020, 2021, 2022)


def load_dump(path: Path) -> pd.DataFrame:
    """Le dump M1 du tester, index UTC, `spread_points` en points du symbole."""
    frame = pd.read_parquet(path)
    if frame.index.tz is None:
        frame.index = frame.index.tz_localize("UTC")
    missing = [c for c in ("close", "spread_points") if c not in frame.columns]
    if missing:
        raise SystemExit(f"[dump] colonnes manquantes {missing} dans {path}")
    return frame.sort_index()


def build_cost_table(
    spread_points: pd.Series,
    close: pd.Series,
    *,
    base_year: int = EXTRAPOLATION_BASE_YEAR,
    extrapolated_years: tuple[int, ...] = EXTRAPOLATED_YEARS,
) -> dict[str, Any]:
    """La table `heure New York × année` du spread, en dollars. Fonction pure.

    `spread_points` et `close` partagent un index UTC ; tout le reste se déduit.
    Aucune lecture de fichier, aucune écriture : c'est la partie qui se teste, et
    c'est elle que `tests/test_build_xau_intraday_costs.py` interroge.

    Les heures sans aucune barre n'apparaissent pas : une médiane sur zéro
    observation serait un `NaN` déguisé en mesure. Le consommateur qui tombe sur
    une heure absente doit se rabattre sur le repli déclaré, pas sur zéro.
    """
    local = spread_points.index.tz_convert(SESSION_TZ)
    frame = pd.DataFrame(
        {
            "spread_usd": spread_points.to_numpy(dtype=float) * POINT,
            "close": close.to_numpy(dtype=float),
            "year": local.year,
            "hour": local.hour,
        }
    )

    measured: dict[int, dict[int, dict[str, float]]] = {}
    for (year, hour), chunk in frame.groupby(["year", "hour"]):
        measured.setdefault(int(year), {})[int(hour)] = {
            "median": float(np.median(chunk["spread_usd"])),
            "p75": float(np.quantile(chunk["spread_usd"], 0.75)),
            "n_bars": int(len(chunk)),
        }

    price_by_year = frame.groupby("year")["close"].median()
    if base_year not in measured:
        raise SystemExit(
            f"[table] l'année de référence {base_year} est absente du dump ; "
            "l'extrapolation n'a pas de base."
        )

    hours: dict[str, dict[str, Any]] = {}
    # Le dump démarre en novembre 2022 et s'arrête le 30 décembre 2025 : la
    # première année n'est pas une année. Le dire dans la table, sinon un
    # consommateur lira « 2022 » comme une mesure de douze mois.
    full_year_bars = 200_000
    for year, by_hour in sorted(measured.items()):
        n_bars = sum(cell["n_bars"] for cell in by_hour.values())
        hours[str(year)] = {
            "source": "mt5_dump",
            "partial_year": bool(n_bars < full_year_bars),
            "n_bars": int(n_bars),
            "price_median_usd": float(price_by_year.loc[year]),
            "hours": {str(h): by_hour[h] for h in sorted(by_hour)},
        }

    # Avant la première mesure : même profil horaire, remis à l'échelle du prix.
    base_hours = measured[base_year]
    for year in extrapolated_years:
        if str(year) in hours:
            continue
        hours[str(year)] = {
            "source": "extrapolated",
            "extrapolated_from": base_year,
            "price_median_usd": None,
            "scale": None,
            "hours": {
                str(h): {
                    "median": base_hours[h]["median"],
                    "p75": base_hours[h]["p75"],
                    "n_bars": 0,
                }
                for h in sorted(base_hours)
            },
        }

    all_spreads = frame["spread_usd"].to_numpy(dtype=float)
    return {
        "symbol": "XAUUSD",
        "unit": "USD par once, fourchette complète (pas la demi-fourchette)",
        "point": POINT,
        "timezone": SESSION_TZ,
        "measured_window_utc": [
            str(spread_points.index[0]),
            str(spread_points.index[-1]),
        ],
        "overall": {
            "median": float(np.median(all_spreads)),
            "p75": float(np.quantile(all_spreads, 0.75)),
            "p95": float(np.quantile(all_spreads, 0.95)),
            "n_bars": int(len(all_spreads)),
        },
        "catalog_snapshot_usd": 0.29,
        "by_year": hours,
    }


def apply_price_scaling(table: dict[str, Any], price_by_year: dict[int, float]) -> None:
    """Renseigner le prix médian et l'échelle des années extrapolées, en place.

    Le prix médian d'une année que le dump ne couvre pas doit venir d'ailleurs —
    du parquet de la campagne. Sans lui, l'échelle reste `None` et la table dit
    honnêtement qu'elle recopie le profil de l'année de base sans le remettre à
    l'échelle.
    """
    base_year = None
    for _year, block in table["by_year"].items():
        if block["source"] == "extrapolated":
            base_year = block["extrapolated_from"]
            break
    if base_year is None:
        return
    base_price = table["by_year"][str(base_year)]["price_median_usd"]

    for year, block in table["by_year"].items():
        if block["source"] != "extrapolated":
            continue
        price = price_by_year.get(int(year))
        if price is None or not base_price:
            continue
        scale = float(price) / float(base_price)
        block["price_median_usd"] = float(price)
        block["scale"] = scale
        for hour in block["hours"].values():
            hour["median"] = round(hour["median"] * scale, 6)
            hour["p75"] = round(hour["p75"] * scale, 6)


def spread_per_m5_bar(dump: pd.DataFrame, m5_index: pd.DatetimeIndex) -> np.ndarray:
    """Spread en dollars pour chaque barre M5, médiane des M1 du bin (§12)."""
    owner = m5_index.get_indexer(dump.index.floor("5min"))
    if (owner < 0).any():
        raise SystemExit("[spread] des barres M1 tombent hors de la grille M5")
    values = dump["spread_points"].to_numpy(dtype=float) * POINT
    return (
        pd.Series(values).groupby(owner).median().reindex(range(len(m5_index))).to_numpy()
    )


def spread_from_table(
    table: dict[str, Any],
    index: pd.DatetimeIndex,
    statistic: str = "median",
    fallback: float = 0.29,
) -> np.ndarray:
    """Le spread de chaque barre, lu dans la table `année × heure New York`.

    `index` est l'index du moteur, c'est-à-dire l'horloge **New York naïve** de
    §2 : la table est indexée sur la même horloge, aucune conversion n'a lieu
    ici. Une case absente rend `fallback` — le repli déclaré de §12 — et jamais
    zéro : un spread nul est un backtest qui se ment.

    Fonction pure : c'est elle que le test interroge, pas le rejeu qui l'appelle.
    """
    years = index.year.to_numpy()
    hours = index.hour.to_numpy()
    lookup: dict[tuple[int, int], float] = {}
    for year, block in table["by_year"].items():
        for hour, cell in block["hours"].items():
            lookup[(int(year), int(hour))] = float(cell[statistic])

    out = np.full(len(index), float(fallback))
    for position, (year, hour) in enumerate(zip(years, hours, strict=True)):
        value = lookup.get((int(year), int(hour)))
        if value is not None:
            out[position] = value
    return out


def _run_variant(inputs: Any, spread: Any, label: str) -> dict[str, Any]:
    """Rejouer le CENTRE de la grille et n'en rendre que trois nombres.

    Trois nombres, et pas un de plus : nombre de trades, espérance en R, profit
    factor. C'est une **sensibilité** au sens de §1.1 de la table de décision —
    aucune grille, aucun `log_trials`, aucune re-sélection. La configuration
    rejouée est celle qui est déjà gelée.
    """
    import sys  # noqa: PLC0415

    sys.path.insert(0, str(_PROJECT_ROOT / "src"))
    from framework.x10_engine import (  # noqa: PLC0415
        RISK_FRAC,
        TRADE_COLUMNS,
        run_engine,
    )

    kwargs = inputs.kernel_kwargs(0.0)
    kwargs["spread"] = (
        np.full(len(inputs.m5), float(spread)) if np.isscalar(spread) else np.asarray(spread)
    )
    _, trades = run_engine(
        **kwargs, z=1.0, a_min=0.2, k_s=1.0, init_cash=10_000.0, risk_frac=RISK_FRAC
    )
    frame = pd.DataFrame(np.asarray(trades), columns=list(TRADE_COLUMNS))
    r = ((frame["equity_after"] - frame["equity_before"]) / frame["risk_amount"]).to_numpy()
    r = r[np.isfinite(r)]
    gains, losses = r[r > 0].sum(), -r[r <= 0].sum()
    return {
        "label": label,
        "spread_usd_mean": float(np.mean(kwargs["spread"])),
        "trades": int(len(frame)),
        "expectancy_r": float(r.mean()) if len(r) else float("nan"),
        "profit_factor": float(gains / losses) if losses > 0 else None,
        "final_equity": float(frame["equity_after"].iloc[-1]) if len(frame) else None,
    }


def cost_sensitivity(
    table: dict[str, Any],
    gold_parquet: Path,
    start: str,
    end: str,
    constant_spread: float = 0.29,
) -> dict[str, Any]:
    """La config du centre rejouée au spread mesuré, contre le spread constant.

    Trois passes sur les mêmes barres : le 0,29 $ de la sélection, puis la
    médiane mesurée par heure × année, puis son p75. La différence entre les
    trois est la sensibilité que §1.1 rend obligatoire de publier.
    """
    import sys  # noqa: PLC0415

    sys.path.insert(0, str(_PROJECT_ROOT / "src"))
    from strategies.xau_x10 import prepare_inputs  # noqa: PLC0415
    from utils import load_gold_data  # noqa: PLC0415

    raw, _ = load_gold_data(str(gold_parquet))
    raw = raw.loc[start:end]
    inputs = prepare_inputs(raw)

    variants = [
        _run_variant(inputs, constant_spread, f"constant_{constant_spread}"),
        _run_variant(
            inputs, spread_from_table(table, inputs.m5.index, "median"), "measured_median"
        ),
        _run_variant(
            inputs, spread_from_table(table, inputs.m5.index, "p75"), "measured_p75"
        ),
    ]
    baseline = variants[0]
    for variant in variants[1:]:
        variant["delta_expectancy_r_vs_constant"] = (
            variant["expectancy_r"] - baseline["expectancy_r"]
        )
        variant["delta_trades_vs_constant"] = variant["trades"] - baseline["trades"]

    return {
        "what": (
            "Sensibilité au spread MESURÉ de la configuration au centre de la "
            "grille (z=1, a_min=0,2, k_s=1). Aucune grille, aucune "
            "re-sélection : table de décision §1.1."
        ),
        "config": {"z": 1.0, "a_min": 0.2, "k_s": 1.0, "risk_frac": 0.005},
        "window": {"start": start, "end": end, "n_m5_bars": int(len(inputs.m5))},
        "gold_parquet": str(gold_parquet),
        "cost_table_source": "costs_xau_intraday.yml",
        "variants": {v["label"]: v for v in variants},
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--dump",
        type=Path,
        default=_PROJECT_ROOT / "data/XAU-USD_minute_mt5_x10dump.parquet",
        help="parquet M1 du dump broker (bid + spread_points)",
    )
    parser.add_argument(
        "--price-history",
        type=Path,
        default=_PROJECT_ROOT / "data/XAU-USD_minute_qc.parquet",
        help="parquet M1 de la campagne, d'où vient le prix médian des années "
        "antérieures au dump",
    )
    parser.add_argument(
        "--out", type=Path, default=_PROJECT_ROOT / "costs_xau_intraday.yml"
    )
    parser.add_argument(
        "--sensitivity-out",
        type=Path,
        default=None,
        help="écrire aussi la sensibilité de la config du CENTRE au spread mesuré",
    )
    parser.add_argument("--sensitivity-start", default="2019-01-01")
    parser.add_argument("--sensitivity-end", default="2025-12-31")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dump = load_dump(args.dump)
    table = build_cost_table(dump["spread_points"], dump["close"])

    if args.price_history.exists():
        history = pd.read_parquet(args.price_history)
        if history.index.tz is None:
            history.index = history.index.tz_localize("UTC")
        # §1 des règles de campagne : aucune donnée postérieure au 2025-12-31
        # n'entre dans un calcul, même pour une médiane de prix.
        history = history[history.index <= pd.Timestamp("2025-12-31", tz="UTC")]
        medians = history["close"].groupby(history.index.year).median()
        apply_price_scaling(table, {int(y): float(v) for y, v in medians.items()})

    args.out.write_text(
        "# Généré par scripts/build_xau_intraday_costs.py — ne pas éditer à la main.\n"
        "# Spread XAUUSD mesuré sur le dump M1 du run de référence MT5 (§12).\n"
        "# Sensibilité publiée, jamais un critère de sélection (table de décision §1.1).\n"
        + yaml.safe_dump(table, sort_keys=False, allow_unicode=True)
    )

    overall = table["overall"]
    print(f"[ok] → {args.out}")
    print(
        f"     spread global médian {overall['median']:.3f} $  "
        f"p75 {overall['p75']:.3f} $  p95 {overall['p95']:.3f} $  "
        f"sur {overall['n_bars']:,} barres M1"
    )
    for year, block in sorted(table["by_year"].items()):
        medians = [h["median"] for h in block["hours"].values()]
        worst = max(block["hours"].items(), key=lambda kv: kv[1]["median"])
        print(
            f"     {year} ({block['source']:<12}) médiane des heures "
            f"{np.median(medians):.3f} $  heure la plus chère {worst[0]}h → "
            f"{worst[1]['median']:.3f} $"
        )

    if args.sensitivity_out is not None:
        import json  # noqa: PLC0415

        report = cost_sensitivity(
            table,
            args.price_history,
            args.sensitivity_start,
            args.sensitivity_end,
        )
        args.sensitivity_out.parent.mkdir(parents=True, exist_ok=True)
        args.sensitivity_out.write_text(
            json.dumps(report, indent=2, ensure_ascii=False, default=str)
        )
        print(f"\n[ok] → {args.sensitivity_out}")
        for label, variant in report["variants"].items():
            print(
                f"     {label:<18} spread moyen {variant['spread_usd_mean']:.3f} $  "
                f"{variant['trades']:>4} trades  "
                f"espérance {variant['expectancy_r']:+.4f} R  "
                f"PF {variant['profit_factor']:.3f}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
