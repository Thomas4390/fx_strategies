# Intégration de la sleeve momentum multi-instruments — cycle 2026-H2

> **Date** : 2026-07-27 · **Statut** : **PROMUE le 2026-07-27** (`fb6e949`, `2e939e4`,
> `e54c778`) à 0,20, puis **poids ramené à 0,15 le 2026-07-28** après échec du gate PBO —
> voir `momentum_validation_2026H2.md`. Les réserves du §4 ci-dessous n'avaient pas été
> levées au moment de la promotion ; la première (optimum au bord de grille) est
> précisément celle que le PBO a sanctionnée.
> **Holdout state** : LOCKED (frozen from 2026-01-01 until Phase 25 / 2026-12-31).
> Toute la sélection de ce document ferme au **2025-12-31**. Aucune lecture
> FROZEN_OOS supplémentaire consommée (les tranches USDJPY/XAGUSD l'ont été au
> cycle de recherche, cf. `momentum_expansion_2026H2.md` §4.8).

## 1. Objet

Suite directe du cycle de recherche (`momentum_expansion_2026H2.md`) : intégrer
les candidats validés (USD-JPY, XAG-USD) à la sleeve momentum et re-pondérer le
portefeuille. Périmètre validé par le propriétaire : les chiffres décident de
la composition ; sweep de poids 10-20 % ; **pas de republication des livrables
client dans ce cycle** — la promotion production reste une décision à prendre.

## 2. Exécution — la sleeve MQL5 devient multi-instruments

`FxSleeveGoldMomentum.mqh` refactoré : `Inp_Gold_Symbols` (CSV, défaut
`"XAUUSD"`), boucle par instrument (modèle `CSleeveTSMomentum`), budget
équipondéré `sub_equity / n configurés`, magic 835 conservé, constantes or
appliquées telles quelles (validées en mono sur chaque candidat).

- **Parité** : le binaire multi à `XAUUSD` seul reproduit la référence mono
  **au bit près** (CAGR 0,410999, dd 76,73, Sharpe 0,7627, 31 trades).
- **Défaut attrapé par la validation** (la revue statique ne l'avait pas vu) :
  l'absence d'historique D1 à l'Init désactivait l'instrument — or dans le
  tester, un symbole dont les données broker commencent dans la fenêtre a zéro
  barre à l'horloge simulée de l'Init. XAGUSD était éliminé silencieusement de
  tout run ouvert avant 2022-11 et le trio dégénérait en duo dilué (mêmes 62
  trades, CAGR ×2/3 — la signature qui a trahi le bug). Corrigé : l'historique
  manquant met l'instrument en warmup, seule l'irrésolvabilité du symbole
  l'écarte.

## 3. Recommandation — trio à 20 %

**Config recommandée** : sleeve momentum = {XAUUSD, USDJPY, XAGUSD}
équipondérée, poids portefeuille **0,20** (MR Macro 0,62, TS 0,09, RSI 0,09).

| Mesure (2021→2025-12-31) | Baseline (or 10 %) | Trio 20 % | Delta |
|---|---|---|---|
| vbt Sharpe (fenêtre sélection) | 1,034 | 1,185 | +0,15 |
| MT5 Sharpe (config prod, 10 k) | 1,062 | **1,288** | +0,23 |
| MT5 CAGR | 40,4 % | **55,0 %** | +14,6 pp |
| MT5 maxDD | 30,8 % | 29,3 % | −1,5 pp |
| MT5 trades | 812 | **861** | +49 |

Sleeve isolée MT5 (100 k, RiskScale=1,0) : or seul 0,763/76,7 %/31 trades ;
duo 1,204/46,7 %/62 ; trio 1,179/38,0 %/**80 trades**. Le trio paie en
drawdown (le mandat volumétrique passe de 35 à 80 trades sur la sleeve).
L'argent, faible seul (0,44), casse la dépendance à la trajectoire or — c'est
le poste qui divise le drawdown, sur les DEUX moteurs.

Sweep vbt complet : `reports/research/momentum_weights_sweep_2026H2.csv`
(baseline + 2 compositions × 5 poids ; budget n=11 logué). Monotonie
croissante jusqu'au bord de grille (0,20) sur les deux compositions.

## 4. Réserves (à instruire avant promotion)

1. **Optimum au bord** : 0,20 est le meilleur point TESTÉ, pas un maximum
   démontré. La contribution au risque de la sleeve y atteint 48 % du budget
   de variance pour 20 % du capital — c'est le vrai plafond à discuter.
2. **XAG reste le maillon fragile** (2 ans de CFD propre ; série longue à
   rolls côté vbt) — mais le tester MT5 sur données broker confirme
   indépendamment le gain de drawdown du trio.
3. **Le sample 2019-2026 aime le momentum** (bull or/argent, mégatrend yen) —
   le biais déjà documenté au cycle or (« l'optimum est au bord parce que le
   sample aime l'or ») s'applique.
4. Chiffres MT5 = majorants (model 1, OHLC M1) ; `RiskScale=4.5` de la config
   production éprouvé ici uniquement en backtest.
5. La promotion exigerait : décision du propriétaire sur le poids (0,15 est le
   cran conservateur : vbt 1,134, dd vbt −45 %), mise à jour de
   `PRODUCTION_WEIGHTS`/presets, backtest de référence complet 2021→2026-04,
   régénération de `mt5_reference.json` et republication des quatre livrables
   client — chantier séparé, non entamé.

## 5. Reproduire

```bash
# parité + compositions (sleeve isolée)
python scripts/sweep_tsmom_mt5.py --only XAUUSD          # référence mono
# binaire multi : --input Inp_Gold_Symbols=XAUUSD,USDJPY,XAGUSD sur run_backtest_cli
# portefeuille : --input Inp_AllocMRMacro=0.62 --input Inp_AllocGoldMomentum=0.20 \
#                --input Inp_Gold_Symbols=XAUUSD,USDJPY,XAGUSD
# sweep vbt
python scripts/sweep_momentum_weights.py
```
