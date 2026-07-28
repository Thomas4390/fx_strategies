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

Les trois sorties sont archivées en CSV depuis le 2026-07-28 : elles ne
vivaient que sur stdout, donc aucune de ces vérifications n'était consultable
après coup ni comparable d'un cycle à l'autre.

La table de stabilité porte aussi ``cap_share`` — la part de séances où le
levier bute sur ``max_leverage``. C'est la mesure de « le vol-targeting
mord-il réellement » : à la configuration de production, USD/JPY y est plat
plus d'une séance sur deux, ce qui ne se voit dans aucune métrique agrégée.

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
OUT_DIR = _REPO / "reports" / "research"
STABILITY_CSV = OUT_DIR / "tsmom_stability_2026H2.csv"
DSR_CSV = OUT_DIR / "tsmom_dsr_2026H2.csv"
CORR_CSV = OUT_DIR / "tsmom_corr_2026H2.csv"
PLATEAU_TOL = 0.30


def _insample_returns(pf) -> pd.Series:
    ret = holdout.trim_insample(pf.returns)
    ret = ret[ret.index <= pd.Timestamp("2025-12-31")]
    # Les loaders produisent des conventions d'horodatage différentes (dates de
    # session normalisées pour l'or, timestamps horaires pour le daily yahoo) :
    # aligner sur la date calendaire pour rendre les séries comparables.
    return ret.set_axis(ret.index.normalize()).groupby(level=0).sum()


def sharpe_252(ret: pd.Series) -> float:
    return float(ret.vbt.returns(freq="1D").sharpe_ratio())


def cap_share(indicator, max_leverage: float) -> float:
    """Part de séances où le levier bute sur son plafond.

    Un vol-targeting qui sature n'en est plus un : la sleeve y tourne à levier
    plat. La métrique ne se lit dans aucun agrégat — le plafond global ramène
    la vol réalisée sur sa cible quoi qu'il arrive — d'où son archivage ici.
    Le levier est lu sur l'indicateur, seul objet qui le porte par barre.
    """
    lev = pd.Series(indicator.leverage).squeeze().dropna()
    if lev.empty:
        return float("nan")
    return float((lev >= max_leverage - 1e-9).mean())


def main() -> int:
    trials.log_trials(
        "tsmom_stability",
        len(LOOKBACK_GRIDS) * len(TARGET_VOLS) * (len(SURVIVORS) + 1),
        "stress de stabilité passe 2 — aucune resélection",
        config_key="tsmom_stability:3lookback_grids_x_3tv_x_4instr",
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=== 1. Stabilité (Sharpe vbt net de coûts, fenêtre ≤ 2025-12-31) ===")
    rets_prod: dict[str, pd.Series] = {}
    stability: list[dict[str, object]] = []
    for sym in [CONTROL, *SURVIVORS]:
        rows = []
        for lb in LOOKBACK_GRIDS:
            for tv in TARGET_VOLS:
                cap = round(tv * 12.0, 2)
                pf, ind = tsmom.pipeline(
                    sym, lookbacks=lb, target_vol=tv,
                    max_leverage=cap, fill="next_open",
                )
                ret = _insample_returns(pf)
                sharpe = sharpe_252(ret)
                is_prod = lb == (40, 60, 120, 250) and tv == 0.55
                rows.append((f"{'/'.join(map(str, lb))}", tv, sharpe))
                stability.append(dict(
                    symbol=sym, lookbacks="/".join(map(str, lb)), target_vol=tv,
                    max_leverage=cap, sharpe=round(sharpe, 6),
                    cap_share=round(cap_share(ind, cap), 4),
                    is_production=is_prod, n_bars=len(ret),
                    first_date=str(ret.index.min().date()),
                    last_date=str(ret.index.max().date()),
                ))
                if is_prod:
                    rets_prod[sym] = ret
        base = next(s for g, tv, s in rows if g == "40/60/120/250" and tv == 0.55)
        lo, hi = min(s for *_, s in rows), max(s for *_, s in rows)
        flag = "PLATEAU" if base - lo <= PLATEAU_TOL else "PIC ⚠️"
        for row in stability:
            if row["symbol"] == sym:
                row["flag"] = flag
        print(f"\n{sym}: prod={base:.3f}  min={lo:.3f}  max={hi:.3f}  → {flag}")
        for g, tv, s in rows:
            marker = " *" if (g == "40/60/120/250" and tv == 0.55) else ""
            print(f"    {g:<14} tv={tv:.2f}  sharpe={s:6.3f}{marker}")
    pd.DataFrame(stability).to_csv(STABILITY_CSV, index=False)

    print("\n=== 2. DSR (n_trials du registre, rivaux = classement MT5) ===")
    n_trials = trials.total_trials()
    mt5 = pd.read_csv(MT5_SWEEP)
    trial_sharpes = mt5["sharpe"].dropna()
    dsr_rows: list[dict[str, object]] = []
    for sym in SURVIVORS:
        ret = rets_prod[sym]
        dsr = deflated_sharpe_ratio(
            ret, n_trials=n_trials, trial_sharpes=trial_sharpes, freq="1D",
        )
        print(f"{sym}: {({k: round(v, 4) for k, v in dsr.items()})}  [n_trials={n_trials}]")
        # dsr porte déjà n_trials : ne pas le redéclarer.
        dsr_rows.append(dict(
            symbol=sym,
            n_trials_distinct=trials.distinct_trials(),
            **{k: round(float(v), 6) for k, v in dsr.items()},
        ))
    pd.DataFrame(dsr_rows).to_csv(DSR_CSV, index=False)

    print("\n=== 3. Corrélations quotidiennes (config production, in-sample) ===")
    aligned = pd.concat(rets_prod, axis=1, sort=True).dropna(how="all")
    corr = aligned.corr()
    print(corr.round(3).to_string())
    corr.to_csv(CORR_CSV)

    print(f"\nÉcrit : {STABILITY_CSV.name}, {DSR_CSV.name}, {CORR_CSV.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
