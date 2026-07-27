#!/usr/bin/env python3
"""Passe 2 des survivants TSMOM — stabilité, DSR, corrélations croisées.

Trois vérifications sur les survivants du screening
(`docs/research/momentum_expansion_2026H2.md` §4.3), AUCUNE resélection de
paramètres :

1. **Stabilité** : le Sharpe vbt dans le voisinage de la config de production
   (lookbacks ±, target_vol ±) doit décrire un plateau, pas un pic. Les
   niveaux absolus vbt sont décalés du MT5 (cf. §2) mais la *forme* de la
   surface est informative.
2. **DSR** : Sharpe déflaté de chaque survivant avec le n_trials honnête du
   registre (`framework.trials.total_trials()`), les Sharpe rivaux étant ceux
   du classement MT5. Approximation documentée : la série de rendements est
   celle du moteur vbt (le tester n'exporte pas de rendements par barre).
3. **Corrélations candidat-candidat** (rendements quotidiens vbt, fenêtre de
   sélection) : les sleeves doivent être des paris distincts.

Usage : python scripts/stress_tsmom_survivors.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))

from framework import holdout, trials  # noqa: E402
from framework.statistical_testing import deflated_sharpe_ratio  # noqa: E402
from strategies import tsmom  # noqa: E402

SURVIVORS = ["USD-JPY", "XAG-USD", "US100"]
CONTROL = "XAU-USD"

LOOKBACK_GRIDS = [
    (30, 50, 100, 200),
    (40, 60, 120, 250),   # production
    (50, 80, 150, 300),
]
TARGET_VOLS = [0.40, 0.55, 0.70]

MT5_SWEEP = _REPO / "reports/mt5/tsmom_universe_sweep.csv"


def _insample_returns(pf) -> pd.Series:
    ret = holdout.trim_insample(pf.returns)
    ret = ret[ret.index <= pd.Timestamp("2025-12-31")]
    # Les loaders produisent des conventions d'horodatage différentes (dates de
    # session normalisées pour l'or, timestamps horaires pour le daily yahoo) :
    # aligner sur la date calendaire pour rendre les séries comparables.
    return ret.set_axis(ret.index.normalize()).groupby(level=0).sum()


def sharpe_252(ret: pd.Series) -> float:
    return float(ret.vbt.returns(freq="1D").sharpe_ratio())


def main() -> int:
    trials.log_trials(
        "tsmom_stability",
        len(LOOKBACK_GRIDS) * len(TARGET_VOLS) * (len(SURVIVORS) + 1),
        "stress de stabilité passe 2 — aucune resélection",
    )

    print("=== 1. Stabilité (Sharpe vbt net de coûts, fenêtre ≤ 2025-12-31) ===")
    rets_prod: dict[str, pd.Series] = {}
    for sym in [CONTROL, *SURVIVORS]:
        rows = []
        for lb in LOOKBACK_GRIDS:
            for tv in TARGET_VOLS:
                pf, _ = tsmom.pipeline(
                    sym, lookbacks=lb, target_vol=tv,
                    max_leverage=round(tv * 12.0, 2), fill="next_open",
                )
                ret = _insample_returns(pf)
                s = sharpe_252(ret)
                rows.append((f"{'/'.join(map(str, lb))}", tv, s))
                if lb == (40, 60, 120, 250) and tv == 0.55:
                    rets_prod[sym] = ret
        base = next(s for g, tv, s in rows if g == "40/60/120/250" and tv == 0.55)
        lo, hi = min(s for *_, s in rows), max(s for *_, s in rows)
        flag = "PLATEAU" if base - lo <= 0.30 else "PIC ⚠️"
        print(f"\n{sym}: prod={base:.3f}  min={lo:.3f}  max={hi:.3f}  → {flag}")
        for g, tv, s in rows:
            marker = " *" if (g == "40/60/120/250" and tv == 0.55) else ""
            print(f"    {g:<14} tv={tv:.2f}  sharpe={s:6.3f}{marker}")

    print("\n=== 2. DSR (n_trials du registre, rivaux = classement MT5) ===")
    n_trials = trials.total_trials()
    mt5 = pd.read_csv(MT5_SWEEP)
    trial_sharpes = mt5["sharpe"].dropna()
    for sym in SURVIVORS:
        ret = rets_prod[sym]
        dsr = deflated_sharpe_ratio(
            ret, n_trials=n_trials, trial_sharpes=trial_sharpes, freq="1D",
        )
        print(f"{sym}: {({k: round(v, 4) for k, v in dsr.items()})}  [n_trials={n_trials}]")

    print("\n=== 3. Corrélations quotidiennes (config production, in-sample) ===")
    aligned = pd.concat(rets_prod, axis=1, sort=True).dropna(how="all")
    print(aligned.corr().round(3).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
