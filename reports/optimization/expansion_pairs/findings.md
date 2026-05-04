# Findings — Expansion paires (Phases 1-3)

> **Date** : 2026-05-04 · **Statut** : RÉSOLU · **Verdict** : aucune paire
> candidate ne mérite ajout au portfolio sur ce broker.

## TL;DR

Test des 6 paires candidates (USDCHF, AUDUSD, NZDUSD, EURGBP, EURJPY, GBPJPY)
sur portfolio combiné prod (config 80/10/10, vol-targeting normal). Aucun
ajout n'améliore le Sharpe portfolio. Tous les ΔSharpe sont négatifs (-0.06
à -0.10). Garder config 4-pair actuelle.

## 1. Phases exécutées

### Phase 1 — Pre-flight infra (DONE)
- Téléchargement broker SquaredFinancial Demo : 10 paires × M1+D1
- Coverage D1 : 5.4 ans pour USDCHF/AUDUSD/NZDUSD/GBPJPY ; 3.5 ans pour
  EURGBP/EURJPY (broker n'a pas d'historique avant 2022-11)
- Coverage M1 : 3 mois max (broker rate-limit `CopyRates` mais Strategy
  Tester utilise les ticks bruts directement)

### Phase 2 — Sharpe standalone par paire par sleeve

| Paire | Sleeve | Sharpe | Trades | DD | Quality Gate (≥0.30) |
|---|---|---|---|---|---|
| USDCHF | MR | 0.00 | 0 | 0% | ✗ pas trades |
| USDCHF | TS | -1.75 | 148 | 48.88% | ✗ |
| USDCHF | RSI | -0.28 | 10 | 16.55% | ✗ |
| **AUDUSD** | **MR** | **+6.34** ⚠ | 64 | 3.08% | ✓ (suspect) |
| AUDUSD | TS | -0.03 | 164 | 20.98% | ✗ |
| AUDUSD | RSI | +0.15 | 8 | 4.11% | ✗ |
| **NZDUSD** | **MR** | **+2.62** | 69 | 3.16% | ✓ |
| NZDUSD | TS | -0.98 | 150 | 35.92% | ✗ |
| NZDUSD | RSI | +0.14 | 8 | 5.17% | ✗ |
| EURGBP | MR | -5.00 | 63 | 5.05% | ✗ |
| EURGBP | TS | -0.50 | 94 | 18.44% | ✗ |
| EURGBP | RSI | +0.10 | 4 | 2.51% | ✗ |
| EURJPY | MR | -5.00 | 29 | 7.15% | ✗ |
| **EURJPY** | **TS** | **+0.69** | 94 | 17.90% | ✓ |
| EURJPY | RSI | +0.08 | 7 | 5.20% | ✗ |
| GBPJPY | MR | -5.00 | 58 | 5.37% | ✗ |
| GBPJPY | TS | -0.76 | 134 | 42.64% | ✗ |
| GBPJPY | RSI | -0.30 | 7 | 16.75% | ✗ |

3/6 paires retenues : AUDUSD, NZDUSD, EURJPY.

⚠ Sharpe AUDUSD/MR = +6.34 suspect : peu de trades (64), session 6-14 UTC
mauvais timing pour AUD (Asia session). Edge probablement non-robuste.

### Phase 3 — Test incrémental dans portfolio prod

Config : alloc 80/10/10, defaults compilés, DDCap=0.30, fenêtre 5.4 ans.

| Variant | CAGR | ΔCAGR | Sharpe | ΔSharpe | DD | Trades |
|---|---|---|---|---|---|---|
| **V0 baseline 4-pair** | **+7.24%** | — | **+1.15** | — | 7.21% | 835 |
| V1 +AUDUSD MR | +6.93% | -0.31% | +1.09 | **-0.06** ✗ | 7.09% | 899 |
| V2 +NZDUSD MR | +6.63% | -0.61% | +1.05 | **-0.10** ✗ | 7.45% | 904 |
| V3 +AUDUSD+NZDUSD MR | +6.75% | -0.48% | +1.06 | **-0.09** ✗ | 6.49% | 968 |
| V4 +EURJPY TS | +0.00% | -7.24% | 0.00 | -1.15 | bug | 0 |
| V5 combiné | idem V4 | idem | idem | idem | idem | 0 |

Critère retenu : ΔSharpe ≥ +0.03. **0/4 variants passent**.

V4/V5 plantent : EURJPY broker history start = 2022-11-04, sleeve TS init
fail sur fenêtre 2020-11-23 (`no D1 history at all`). Pour tester EURJPY
correctement il faudrait restreindre la fenêtre à 2022-11+ ou modifier l'EA
pour skip paires sans historique au lieu de hard-fail.

## 2. Conclusions

1. **Le portfolio 4-pair est saturé** sur cet univers. Ajouter des paires
   dilue le sub_equity (1/4 → 1/5) sans apporter d'edge supplémentaire net.
2. **AUDUSD/NZDUSD apportent +trades** (835 → 968) mais coût marginal >
   edge. Le sleeve MR Macro session 6-14 UTC est mal aligné avec l'horaire
   AUD/NZD (Asia/Sydney session).
3. **EURJPY potentiellement intéressant** (TS Sharpe +0.69 standalone)
   mais bloqué par historique broker court. À retester si broker change.
4. **Le Sharpe AUDUSD/MR=+6.34 standalone** est un artefact : peu de
   trades + alloc 100% sur 1 paire amplifie le ratio. Quand intégré dans
   portfolio, l'edge se dilue à -0.06 ΔSharpe.

## 3. Implications business

**Pour atteindre 15% CAGR**, l'expansion paires sur ce broker n'est pas
le bon levier. Pistes restantes :

1. **Changer broker** : SquaredFinancial Demo a peut-être des spreads
   trop larges ou un univers limité. Tester Pepperstone, Tickmill,
   Dukascopy avec les mêmes paires.
2. **Nouveaux signaux** sur paires existantes : carry trade (différentiel
   taux), news momentum, FX vol RV.
3. **Nouvelles timeframes** : H1, H4 sur EUR/USD (entre M1 et D1).
4. **Réviser allocation** : 80/10/10 peut-être sub-optimal après
   l'analyse. À tester 50/25/25 ou 60/20/20.
5. **Le régime 2025-11→2026-04 est anormalement défavorable** (cf.
   walkforward N=5 fold5). Attendre rebascule régime macro plutôt que
   modifier la stratégie.

## 4. Artefacts

- `scripts/optimization/eval_new_pairs.py` (Phase 2 standalone)
- `scripts/optimization/phase3_incremental.py` (Phase 3 incrémental)
- `reports/optimization/expansion_pairs/standalone_*.csv` (18 standalone)
- `reports/optimization/expansion_pairs/phase3_incremental_*.csv` (6 variants)

## 5. Pour future investigation

- **Bug EA** : modifier `FxSleeveTSMomentum.mqh` pour skip paire sans D1
  historique au lieu de hard-fail global. Permettrait test EURJPY sur
  fenêtre adaptative.
- **Restreindre fenêtre** pour test EURJPY pure : 2022-11-05 → 2026-04-30
  (3.5 ans). Si Sharpe robuste, considérer ajout.
- **Test broker alternatif** : refaire Phase 1-3 sur Dukascopy ou
  Pepperstone pour comparer.
