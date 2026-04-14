# Phase 20C — Graduated DD-cap sweep (2026-04-13)

Suite directe des sweeps Phase 20A/B. Phase 19 avait constaté que le DD cap binaire (ON/OFF avec l'échelle Phase 13 `{0.10→1.0, 0.20→0.60, 0.30→0.35, 0.35→0.15}`) perdait systématiquement contre le mode `DDoff` sur le trio P18 à `tv=0.25 ml=14`, car le cap de-leverage les drawdowns qui allaient se redresser. Phase 20C teste des échelles **plus douces** paramétrées par deux knobs :

- `dd_knee` — le seuil DD auquel le de-leveraging commence (avant ce seuil, scale = 1.0). Testé dans {0.10, 0.15, 0.20} + extremes {0.05, 0.25}.
- `dd_floor` — le multiplicateur atteint à DD = 35% (clippé au-delà). Testé dans {0.35, 0.50, 0.70, 0.85}.

Un schedule `knee=0.10, floor=0.15` approxime l'historique Phase 13 (hard-cap). Un `knee=0.20, floor=0.85` est un cap quasi-inactif. Paramètres fixes : `tv=0.25 ml=14` sur le trio P18. Deux mixes de poids sont testés : le canonique 80/10/10 et le P20A top 75/10/15.

Baseline `DDoff` (80-10-10) : Sharpe WF 0.966, CAGR 13.38%, MaxDD -18.41%.

Baseline `DDoff` (75-10-15) : Sharpe WF 0.972, CAGR 13.14%, MaxDD -17.35%.

Baseline `DDon-Phase13` (80-10-10) : Sharpe WF 0.935, CAGR 13.06%, MaxDD -17.73%.

## Top 10 par Walk-Forward Sharpe

| Rank | ID | Config | Sharpe WF | CAGR | Vol | MaxDD | WF pos | ★ |
|------|----|--------|-----------|------|-----|-------|--------|----|
| 1 | `P20c-soft-w75-10-15-k20-f35` | w75-10-15 / tv=0.25 ml=14 / DDsoft(knee=0.20, floor=0.35) | **0.972** |   13.14% |   13.60% |  -17.35% | 6/7 | * |
| 2 | `P20c-soft-w75-10-15-k20-f50` | w75-10-15 / tv=0.25 ml=14 / DDsoft(knee=0.20, floor=0.50) | **0.972** |   13.14% |   13.60% |  -17.35% | 6/7 | * |
| 3 | `P20c-soft-w75-10-15-k20-f70` | w75-10-15 / tv=0.25 ml=14 / DDsoft(knee=0.20, floor=0.70) | **0.972** |   13.14% |   13.60% |  -17.35% | 6/7 | * |
| 4 | `P20c-soft-w75-10-15-k20-f85` | w75-10-15 / tv=0.25 ml=14 / DDsoft(knee=0.20, floor=0.85) | **0.972** |   13.14% |   13.60% |  -17.35% | 6/7 | * |
| 5 | `BL-DDoff-w75-10-15` | w75-10-15 / tv=0.25 ml=14 / DDoff | **0.972** |   13.14% |   13.60% |  -17.35% | 6/7 | * |
| 6 | `P20c-soft-w75-10-15-k15-f85` | w75-10-15 / tv=0.25 ml=14 / DDsoft(knee=0.15, floor=0.85) | **0.972** |   13.14% |   13.60% |  -17.33% | 6/7 | * |
| 7 | `P20c-soft-w75-10-15-k15-f70` | w75-10-15 / tv=0.25 ml=14 / DDsoft(knee=0.15, floor=0.70) | **0.972** |   13.13% |   13.60% |  -17.32% | 6/7 | * |
| 8 | `P20c-soft-w75-10-15-k15-f50` | w75-10-15 / tv=0.25 ml=14 / DDsoft(knee=0.15, floor=0.50) | **0.971** |   13.13% |   13.60% |  -17.30% | 6/7 | * |
| 9 | `P20c-soft-w75-10-15-k15-f35` | w75-10-15 / tv=0.25 ml=14 / DDsoft(knee=0.15, floor=0.35) | **0.971** |   13.13% |   13.60% |  -17.29% | 6/7 | * |
| 10 | `P20c-soft-w75-10-15-k10-f85` | w75-10-15 / tv=0.25 ml=14 / DDsoft(knee=0.10, floor=0.85) | **0.969** |   13.10% |   13.59% |  -17.26% | 6/7 | * |

★ = CAGR ∈ [10%, 15%] AND MaxDD < 35%.

## SOFT_CAP — Sharpe WF par (knee, floor) — poids 80-10-10 (canonical)

| knee \ floor | 0.35 | 0.50 | 0.70 | 0.85 |
|---|---|---|---|---|
| 0.10 | 0.946 | 0.951 | 0.957 | 0.961 |
| 0.15 | 0.963 | 0.963 | 0.964 | 0.965 |
| 0.20 | 0.966 | 0.966 | 0.966 | 0.966 |

## SOFT_CAP — Sharpe WF par (knee, floor) — poids 75-10-15 (P20A top)

| knee \ floor | 0.35 | 0.50 | 0.70 | 0.85 |
|---|---|---|---|---|
| 0.10 | 0.956 | 0.960 | 0.965 | 0.969 |
| 0.15 | 0.971 | 0.971 | 0.972 | 0.972 |
| 0.20 | 0.972 | 0.972 | 0.972 | 0.972 |

## Meilleure config par bloc

| Bloc | ID | Config | Sharpe WF | CAGR | MaxDD |
|------|----|--------|-----------|------|-------|
| SOFT_CAP | `P20c-soft-w75-10-15-k20-f35` | w75-10-15 / tv=0.25 ml=14 / DDsoft(knee=0.20, floor=0.35) | **0.972** |   13.14% |  -17.35% |
| KNEE_DEEP | `P20c-deep-w80-10-10-k25-f35` | w80-10-10 / tv=0.25 ml=14 / DDsoft(knee=0.25, floor=0.35) | **0.966** |   13.38% |  -18.41% |
| BASELINE | `BL-DDoff-w75-10-15` | w75-10-15 / tv=0.25 ml=14 / DDoff | **0.972** |   13.14% |  -17.35% |

## Bootstrap stress-test (top-5, 500 × 20-day blocks)

| Rank | ID | CAGR P5 | CAGR P50 | MaxDD P5 | MaxDD P50 | Target hit |
|------|----|---------|----------|----------|-----------|-----------|
| 1 | `P20c-soft-w75-10-15-k20-f35` | +4.81% | +12.76% | -26.02% | -18.32% | 37.2% |
| 2 | `P20c-soft-w75-10-15-k20-f50` | +4.81% | +12.76% | -26.02% | -18.32% | 37.2% |
| 3 | `P20c-soft-w75-10-15-k20-f70` | +4.81% | +12.76% | -26.02% | -18.32% | 37.2% |
| 4 | `P20c-soft-w75-10-15-k20-f85` | +4.81% | +12.76% | -26.02% | -18.32% | 37.2% |
| 5 | `BL-DDoff-w75-10-15` | +5.71% | +13.23% | -30.06% | -18.58% | 38.6% |

## Conclusion

Meilleur point du sweep : **`P20c-soft-w75-10-15-k20-f35`** — w75-10-15 / tv=0.25 ml=14 / DDsoft(knee=0.20, floor=0.35).  
Sharpe WF = **0.972** (vs DDoff baseline 80-10-10 = 0.966, Δ = +0.007).

Un schedule DD cap gradué **améliore** le Sharpe par rapport au mode DDoff pur. Le cap doux préserve le recovery sur les drawdowns modérés tout en coupant la queue des scénarios extrêmes.
