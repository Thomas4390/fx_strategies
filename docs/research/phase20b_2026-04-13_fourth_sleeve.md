# Phase 20B — 4th-sleeve sweep (2026-04-13)

Suite directe du sweep Phase 20A (`docs/research/phase20a_2026-04-13_weight_sweep.md`) qui avait montré que la recombinaison des poids entre les 3 sleeves P18 ne déplace le Sharpe WF que de quelques bps au-dessus du plateau Phase 19 (0.966). Phase 20B teste si l'ajout d'un 4ᵉ sleeve orthogonal — Composite FX Alpha, OU Mean Reversion ou un retour de XS Momentum — parvient à pousser le Sharpe au-delà de 0.97.

Paramètres fixes : `tv=0.25 / ml=14 / DDoff` (Phase 19 plateau sweet-spot). Chaque config prend un base split du trio P18 et réalloue une fraction ∈ {5%, 10%, 15%} au sleeve extra.

## Corrélations des sleeves extra vs trio P18

| Sleeve extra | Corr avec MR_Macro | Corr avec TS_3p | Corr avec RSI_4p |
|-------------|-------------------:|----------------:|-----------------:|
| Composite_FX_Alpha | +0.034 | +0.538 | -0.367 |
| OU_MR | +0.697 | -0.001 | -0.009 |
| XS_Momentum | +0.050 | +0.428 | -0.512 |

Baseline Phase 19 plateau : **BL-P19plateau** — Sharpe WF 0.966, CAGR 13.38%, MaxDD -18.41%.

Baseline Phase 20A top-1 : **BL-P20Atop** — Sharpe WF 0.972, CAGR 13.14%, MaxDD -17.35%.

## Top 10 par Walk-Forward Sharpe

| Rank | ID | Config | Sharpe WF | CAGR | Vol | MaxDD | WF pos | ★ |
|------|----|--------|-----------|------|-----|-------|--------|----|
| 1 | `BL-P20Atop` | Phase 20A top-1 (MR75/TS10/RSI15 tv=0.25 ml=14 DDoff) | **0.972** |   13.14% |   13.60% |  -17.35% | 6/7 | * |
| 2 | `BL-P19plateau` | Phase 19 plateau (MR80/TS10/RSI10 tv=0.25 ml=14 DDoff) | **0.966** |   13.38% |   13.97% |  -18.41% | 6/7 | * |
| 3 | `P20b-x75-10-15-X05` | +XS_Momentum 5% / base=75-10-15 / tv=0.25 ml=14 DDoff | **0.965** |   14.29% |   15.19% |  -22.63% | 6/7 | * |
| 4 | `P20b-x70-12-18-X05` | +XS_Momentum 5% / base=70-12-18 / tv=0.25 ml=14 DDoff | **0.965** |   14.64% |   15.49% |  -21.90% | 6/7 | * |
| 5 | `P20b-x80-08-12-X05` | +XS_Momentum 5% / base=80-8-12 / tv=0.25 ml=14 DDoff | **0.958** |   13.89% |   15.01% |  -23.33% | 6/7 | * |
| 6 | `BL-P18prod` | Phase 18 prod (MR80/TS10/RSI10 tv=0.28 ml=12 DDoff) | **0.956** |   13.11% |   13.73% |  -17.93% | 6/7 | * |
| 7 | `P20b-x80-10-10-X05` | +XS_Momentum 5% / base=80-10-10 / tv=0.25 ml=14 DDoff | **0.943** |   14.41% |   15.73% |  -23.78% | 6/7 | * |
| 8 | `P20b-x70-15-15-X05` | +XS_Momentum 5% / base=70-15-15 / tv=0.25 ml=14 DDoff | **0.938** |   15.37% |   16.75% |  -22.75% | 6/7 |   |
| 9 | `P20b-x75-15-10-X05` | +XS_Momentum 5% / base=75-15-10 / tv=0.25 ml=14 DDoff | **0.920** |   15.49% |   17.23% |  -23.91% | 6/7 |   |
| 10 | `P20b-c75-10-15-C05` | +Composite 5% / base=75-10-15 / tv=0.25 ml=14 DDoff | **0.901** |   12.30% |   13.38% |  -18.51% | 6/7 | * |

★ = CAGR ∈ [10%, 15%] AND MaxDD < 35%.

## Meilleure config par bloc

| Bloc | ID | Config | Sharpe WF | CAGR | MaxDD |
|------|----|--------|-----------|------|-------|
| COMPOSITE | `P20b-c75-10-15-C05` | +Composite 5% / base=75-10-15 / tv=0.25 ml=14 DDoff | **0.901** |   12.30% |  -18.51% |
| OU_MR | `P20b-o70-12-18-O05` | +OU_MR 5% / base=70-12-18 / tv=0.25 ml=14 DDoff | **0.859** |   12.62% |  -19.13% |
| XS_REVISIT | `P20b-x75-10-15-X05` | +XS_Momentum 5% / base=75-10-15 / tv=0.25 ml=14 DDoff | **0.965** |   14.29% |  -22.63% |
| BASELINE | `BL-P20Atop` | Phase 20A top-1 (MR75/TS10/RSI15 tv=0.25 ml=14 DDoff) | **0.972** |   13.14% |  -17.35% |

## Bootstrap stress-test (top-5, 500 × 20-day blocks)

| Rank | ID | CAGR P5 | CAGR P50 | MaxDD P5 | MaxDD P50 | Target hit |
|------|----|---------|----------|----------|-----------|-----------|
| 1 | `BL-P20Atop` | +5.71% | +13.23% | -30.06% | -18.58% | 38.6% |
| 2 | `BL-P19plateau` | +5.80% | +13.68% | -31.27% | -19.16% | 35.4% |
| 3 | `P20b-x75-10-15-X05` | +6.07% | +14.48% | -34.84% | -21.68% | 30.8% |
| 4 | `P20b-x70-12-18-X05` | +6.17% | +15.02% | -35.19% | -22.14% | 28.8% |
| 5 | `P20b-x80-08-12-X05` | +6.00% | +14.15% | -34.81% | -21.47% | 31.4% |

## Conclusion

Meilleur point du sweep : **`BL-P20Atop`** — Phase 20A top-1 (MR75/TS10/RSI15 tv=0.25 ml=14 DDoff).  
Sharpe WF = **0.972** (vs plateau P19 = 0.966, Δ = +0.007 ; vs P20A top = 0.972, Δ = +0.000).

L'ajout d'un 4ᵉ sleeve **ne dépasse pas** le top Phase 20A de façon significative. Le trio P18 + layer de leverage P19 reste la configuration recommandée.
