# Phase 20A — Weight sweep (2026-04-13)

Suite directe du sweep Phase 19 (`docs/research/phase19_2026-04-13_refined_leverage.md`) qui avait établi un plateau Sharpe à 0.966 sur toute la région `(target_vol ≥ 0.22, max_leverage ≥ 14, DDoff)` avec les poids Phase 18 figés à 80/10/10. Phase 20A pose la question complémentaire : est-ce qu'une autre répartition des poids entre MR_Macro, TS_Momentum_3p et RSI_Daily_4p permet de sortir de ce plateau ?

Paramètres fixes du bloc CORE : `tv=0.25 / ml=14 / DDoff` (le point le plus conservateur du plateau Phase 19). Grille : 8 valeurs de `mr_weight` × 5 valeurs de `rsi_fraction` (part du budget diversifieur allouée à RSI vs TS) = 40 configs CORE, plus 10 configs ROBUST pour vérifier la stabilité à `tv ∈ {0.22, 0.28}` et 6 configs LIGHT_MR contrariennes avec `mr_weight ∈ {0.40, 0.50}`.

Baseline Phase 19 plateau : **BL-P19plateau** — Sharpe WF 0.966, CAGR 13.38%, MaxDD -18.41%.

Baseline Phase 18 prod : **BL-P18prod** — Sharpe WF 0.956, CAGR 13.11%, MaxDD -17.93%.

## Top 10 par Walk-Forward Sharpe

| Rank | ID | Config | Sharpe WF | CAGR | Vol | MaxDD | WF pos | ★ |
|------|----|--------|-----------|------|-----|-------|--------|----|
| 1 | `P20a-w75-10-15` | weights=75-10-15 / tv=0.25 ml=14 DDoff | **0.972** |   13.14% |   13.60% |  -17.35% | 6/7 | * |
| 2 | `P20a-w80-08-12` | weights=80-08-12 / tv=0.25 ml=14 DDoff | **0.971** |   12.72% |   13.43% |  -17.94% | 6/7 | * |
| 3 | `P20a-r80-10-10-tv28` | weights=80-10-10 / tv=0.28 ml=14 DDoff | **0.966** |   14.95% |   15.65% |  -20.57% | 6/7 | * |
| 4 | `P20a-w80-10-10` | weights=80-10-10 / tv=0.25 ml=14 DDoff | **0.966** |   13.38% |   13.97% |  -18.41% | 6/7 | * |
| 5 | `BL-P19plateau` | Phase 19 plateau (MR80/TS10/RSI10 tv=0.25 ml=14 DDoff) | **0.966** |   13.38% |   13.97% |  -18.41% | 6/7 | * |
| 6 | `P20a-r80-10-10-tv22` | weights=80-10-10 / tv=0.22 ml=14 DDoff | **0.966** |   11.80% |   12.30% |  -16.23% | 6/7 | * |
| 7 | `P20a-r70-12-18-tv28` | weights=70-12-18 / tv=0.28 ml=14 DDoff | **0.963** |   15.03% |   15.58% |  -19.26% | 6/7 |   |
| 8 | `P20a-w70-12-18` | weights=70-12-18 / tv=0.25 ml=14 DDoff | **0.963** |   13.45% |   13.91% |  -17.19% | 6/7 | * |
| 9 | `P20a-r70-12-18-tv22` | weights=70-12-18 / tv=0.22 ml=14 DDoff | **0.963** |   11.86% |   12.24% |  -15.12% | 6/7 | * |
| 10 | `P20a-w85-06-09` | weights=85-06-09 / tv=0.25 ml=14 DDoff | **0.962** |   12.23% |   13.42% |  -18.64% | 6/7 | * |

★ = CAGR ∈ [10%, 15%] AND MaxDD < 35%.

## Meilleure config par bloc

| Bloc | ID | Config | Sharpe WF | CAGR | MaxDD |
|------|----|--------|-----------|------|-------|
| CORE | `P20a-w75-10-15` | weights=75-10-15 / tv=0.25 ml=14 DDoff | **0.972** |   13.14% |  -17.35% |
| ROBUST | `P20a-r80-10-10-tv28` | weights=80-10-10 / tv=0.28 ml=14 DDoff | **0.966** |   14.95% |  -20.57% |
| LIGHT_MR | `P20a-l50-20-30` | weights=50-20-30 (light-MR) / tv=0.25 ml=14 DDoff | **0.893** |   14.33% |  -24.15% |
| BASELINE | `BL-P19plateau` | Phase 19 plateau (MR80/TS10/RSI10 tv=0.25 ml=14 DDoff) | **0.966** |   13.38% |  -18.41% |

## CORE grid — Sharpe WF par (MR, RSI fraction)

| MR \ RSI frac | 0.25 | 0.40 | 0.50 | 0.60 | 0.75 |
|---|---|---|---|---|---|
| 0.55 | 0.818 | 0.860 | 0.893 | 0.909 | 0.923 |
| 0.60 | 0.839 | 0.883 | 0.914 | 0.925 | 0.928 |
| 0.65 | 0.858 | 0.909 | 0.933 | 0.945 | 0.933 |
| 0.70 | 0.888 | 0.931 | 0.951 | 0.963 | 0.946 |
| 0.75 | 0.915 | 0.946 | 0.961 | 0.972 | 0.957 |
| 0.80 | 0.931 | 0.953 | 0.966 | 0.971 | 0.957 |
| 0.85 | 0.938 | 0.954 | 0.961 | 0.962 | 0.950 |
| 0.90 | 0.938 | 0.944 | 0.945 | 0.944 | 0.940 |

## Bootstrap stress-test (top-5, 500 × 20-day blocks)

| Rank | ID | CAGR P5 | CAGR P50 | MaxDD P5 | MaxDD P50 | Target hit |
|------|----|---------|----------|----------|-----------|-----------|
| 1 | `P20a-w75-10-15` | +5.71% | +13.23% | -30.06% | -18.58% | 38.6% |
| 2 | `P20a-w80-08-12` | +5.12% | +12.64% | -29.94% | -18.43% | 38.4% |
| 3 | `P20a-r80-10-10-tv28` | +6.38% | +14.95% | -34.45% | -21.25% | 31.4% |
| 4 | `P20a-w80-10-10` | +5.80% | +13.68% | -31.27% | -19.16% | 35.4% |
| 5 | `BL-P19plateau` | +5.80% | +13.68% | -31.27% | -19.16% | 35.4% |

## Conclusion

Meilleur point du sweep : **`P20a-w75-10-15`** — weights=75-10-15 / tv=0.25 ml=14 DDoff.  
Sharpe WF = **0.972** (vs plateau Phase 19 = 0.966, Δ = +0.007).

Un nouveau point **domine** le plateau Phase 19 — à valider sur le bootstrap et sur les points ROBUST avant de le promouvoir comme recommandation Phase 20.
