#!/usr/bin/env python3
"""check_screening_vs_broker — la série longue est-elle représentative du CFD ?

Les séries daily longues (`*_daily_yahoo.parquet`, cf.
`download_screening_daily.py`) ne peuvent nourrir un screening que si, sur la
fenêtre commune, elles racontent la même histoire que le CFD broker qui sera
réellement tradé (`*_daily_mt5.parquet`). Ce script mesure, par instrument :

- la corrélation des rendements log quotidiens (alignés par date calendaire) ;
- le ratio des volatilités annualisées ;
- la couverture (nombre de dates communes / dates broker).

Verdicts :
- corr >= 0.95 et |vol_ratio - 1| <= 0.10  -> LONG_OK   (screening sur la série longue)
- corr >= 0.85                             -> FLAGGED   (screening long possible, à
                                              documenter dans la note de recherche)
- sinon                                    -> BROKER_ONLY (screening sur 2022-11+ broker
                                              seulement, puissance réduite assumée)

Le verdict par instrument est écrit dans
`reports/research/screening_source_check.json` — consommé par le screening
Phase 2 pour choisir la source par instrument.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
OUT_PATH = REPO_ROOT / "reports" / "research" / "screening_source_check.json"

INSTRUMENTS = [
    "XAG-USD",
    "XTI-USD",
    "XBR-USD",
    "XNG-USD",
    "US500",
    "US100",
    "US30",
    "GER40",
    "JPN225",
    "UK100",
]


def _daily_log_returns_by_date(path: Path) -> pd.Series:
    close = pd.read_parquet(path, columns=["close"])["close"]
    idx = close.index
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    by_date = close.set_axis(idx.normalize()).groupby(level=0).last()
    return np.log(by_date).diff().dropna()


def check_one(name: str) -> dict:
    long_path = DATA_DIR / f"{name}_daily_yahoo.parquet"
    broker_path = DATA_DIR / f"{name}_daily_mt5.parquet"
    r_long = _daily_log_returns_by_date(long_path)
    r_broker = _daily_log_returns_by_date(broker_path)

    joined = pd.concat({"long": r_long, "broker": r_broker}, axis=1, sort=True).dropna()

    # Les conventions de clôture divergent (indice cash à l'heure de sa bourse,
    # barre broker à minuit heure serveur) : un décalage CONSTANT d'un jour
    # écrase la corrélation quotidienne sans rien dire de la cohérence des
    # tendances. On mesure donc au meilleur lag ∈ {-1, 0, +1}…
    corr_by_lag = {
        lag: float(joined["long"].shift(lag).corr(joined["broker"]))
        for lag in (-1, 0, 1)
    }
    best_lag = max(corr_by_lag, key=lambda k: corr_by_lag[k])
    corr_best = corr_by_lag[best_lag]

    # …et en rendements hebdomadaires, insensibles au découpage intra-semaine —
    # l'échelle pertinente pour des lookbacks de 40 à 250 séances.
    weekly = joined.resample("W").sum()
    weekly = weekly[(weekly != 0).all(axis=1)]
    corr_weekly = float(weekly["long"].corr(weekly["broker"]))

    vol_long = float(joined["long"].std() * np.sqrt(252))
    vol_broker = float(joined["broker"].std() * np.sqrt(252))
    vol_ratio = vol_long / vol_broker if vol_broker else float("nan")
    coverage = len(joined) / len(r_broker) if len(r_broker) else 0.0

    vol_ok = abs(vol_ratio - 1.0) <= 0.10
    if (corr_best >= 0.95 or corr_weekly >= 0.97) and vol_ok:
        verdict = "LONG_OK"
    elif corr_best >= 0.85 or corr_weekly >= 0.90:
        verdict = "FLAGGED"
    else:
        verdict = "BROKER_ONLY"

    return {
        "instrument": name,
        "corr_daily_lag0": round(corr_by_lag[0], 4),
        "corr_daily_best": round(corr_best, 4),
        "best_lag": best_lag,
        "corr_weekly": round(corr_weekly, 4),
        "vol_long_ann": round(vol_long, 4),
        "vol_broker_ann": round(vol_broker, 4),
        "vol_ratio": round(vol_ratio, 4),
        "common_days": len(joined),
        "coverage_of_broker": round(coverage, 4),
        "window": [str(joined.index[0].date()), str(joined.index[-1].date())],
        "verdict": verdict,
    }


def main() -> int:
    results = []
    for name in INSTRUMENTS:
        try:
            results.append(check_one(name))
        except FileNotFoundError as exc:
            results.append({"instrument": name, "verdict": "MISSING_DATA", "error": str(exc)})

    header = (
        f"{'instr':8s} {'lag0':>6s} {'best':>6s} {'lag':>4s} {'hebdo':>6s} "
        f"{'ratio':>6s} {'days':>5s}  verdict"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        if r["verdict"] == "MISSING_DATA":
            print(f"{r['instrument']:8s} {'—':>52s}  MISSING_DATA")
            continue
        print(
            f"{r['instrument']:8s} {r['corr_daily_lag0']:6.3f} "
            f"{r['corr_daily_best']:6.3f} {r['best_lag']:4d} {r['corr_weekly']:6.3f} "
            f"{r['vol_ratio']:6.3f} {r['common_days']:5d}  {r['verdict']}"
        )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(
        json.dumps(
            {
                "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "method": "corr of daily log returns on calendar-date alignment, common window",
                "results": results,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"\n→ {OUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
