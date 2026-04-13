# Phase 19 — Refined leverage sweep (2026-04-13)

Suite directe du sweep Phase 18 (`docs/research/sweep_2026-04-13_combinations.md`) qui avait révélé que le cluster `E1/E5/D5/D6` (P18 trio avec `dd_cap=ON` et max_leverage ∈ [3, 14]) dominait le Sharpe WF sans perdre le CAGR cible. Phase 19 fouille densément cette région avec un grid `target_vol × max_leverage × dd_cap` et teste aussi 7 variations de poids autour du 80/10/10 baseline, plus 9 points expérimentaux à haut levier (ml ∈ [18, 24]).

Baseline de comparaison : **Phase 18 prod** — Sharpe WF 0.956, CAGR 13.11%, MaxDD -17.93%.

Référence top-5 Phase 18 : **E5** (ml=14 DDon) — Sharpe WF 0.922, CAGR 14.43%, MaxDD -19.85%.

## Top 10 par Walk-Forward Sharpe

| Rank | ID | Config | Sharpe WF | CAGR | Vol | MaxDD | WF pos | ★ |
|------|----|--------|-----------|------|-----|-------|--------|----|
| 1 | `P19c-tv35-ml18-dd_OFF` | P18 weights / tv=0.35 ml=18 DDoff | **0.966** |   18.58% |   19.56% |  -25.57% | 6/7 |   |
| 2 | `P19c-tv25-ml14-dd_OFF` | P18 weights / tv=0.25 ml=14 DDoff | **0.966** |   13.38% |   13.97% |  -18.41% | 6/7 | * |
| 3 | `P19c-tv25-ml16-dd_OFF` | P18 weights / tv=0.25 ml=16 DDoff | **0.966** |   13.38% |   13.97% |  -18.41% | 6/7 | * |
| 4 | `P19c-tv25-ml18-dd_OFF` | P18 weights / tv=0.25 ml=18 DDoff | **0.966** |   13.38% |   13.97% |  -18.41% | 6/7 | * |
| 5 | `P19c-tv28-ml14-dd_OFF` | P18 weights / tv=0.28 ml=14 DDoff | **0.966** |   14.95% |   15.65% |  -20.57% | 6/7 | * |
| 6 | `P19c-tv32-ml16-dd_OFF` | P18 weights / tv=0.32 ml=16 DDoff | **0.966** |   17.03% |   17.89% |  -23.44% | 6/7 |   |
| 7 | `P19c-tv32-ml18-dd_OFF` | P18 weights / tv=0.32 ml=18 DDoff | **0.966** |   17.03% |   17.89% |  -23.44% | 6/7 |   |
| 8 | `P19c-tv30-ml16-dd_OFF` | P18 weights / tv=0.30 ml=16 DDoff | **0.966** |   15.99% |   16.77% |  -22.01% | 6/7 |   |
| 9 | `P19c-tv30-ml18-dd_OFF` | P18 weights / tv=0.30 ml=18 DDoff | **0.966** |   15.99% |   16.77% |  -22.01% | 6/7 |   |
| 10 | `P19c-tv28-ml16-dd_OFF` | P18 weights / tv=0.28 ml=16 DDoff | **0.966** |   14.95% |   15.65% |  -20.57% | 6/7 | * |

★ = CAGR ∈ [10%, 15%] AND MaxDD < 35%.

## Bootstrap stress-test (top-5, 500 × 20-day blocks)

| Rank | ID | CAGR P5 | CAGR P50 | MaxDD P5 | MaxDD P50 | Sharpe P5 | Sharpe P95 | Pos frac | Target hit |
|------|----|---------|----------|----------|-----------|-----------|-----------|----------|------------|
| 1 | `P19c-tv35-ml18-dd_OFF` |    7.60% |   18.29% |  -40.85% |  -25.94% |  +0.473 |  +1.490 |   99.6% |   15.4% |
| 2 | `P19c-tv25-ml14-dd_OFF` |    5.80% |   13.68% |  -31.27% |  -19.16% |  +0.473 |  +1.494 |   99.6% |   35.4% |
| 3 | `P19c-tv25-ml16-dd_OFF` |    5.80% |   13.68% |  -31.27% |  -19.23% |  +0.473 |  +1.492 |   99.6% |   35.2% |
| 4 | `P19c-tv25-ml18-dd_OFF` |    5.80% |   13.68% |  -31.27% |  -19.23% |  +0.473 |  +1.492 |   99.6% |   35.2% |
| 5 | `P19c-tv28-ml14-dd_OFF` |    6.38% |   14.95% |  -34.45% |  -21.25% |  +0.473 |  +1.506 |   99.6% |   31.4% |

## Best per block

### CORE — Dense tv × ml × dd grid autour du Phase 18 trio
- **96 configs**. Meilleure : `P19c-tv35-ml18-dd_OFF` (Sharpe WF **0.966**, CAGR 18.58%, MaxDD -25.57%, WF pos 6/7).
- vs Phase 18 prod : Δ Sharpe WF = +0.009, Δ CAGR = +5.47%.

### HLEV — High-leverage experimental (ml ∈ [18, 24])
- **9 configs**. Meilleure : `P19h-tv32-ml18` (Sharpe WF **0.902**, CAGR 16.14%, MaxDD -22.96%, WF pos 6/7).
- vs Phase 18 prod : Δ Sharpe WF = -0.054, Δ CAGR = +3.04%.

### WEIGHTS — Weights variations autour de 80/10/10
- **7 configs**. Meilleure : `P19w-75-10-15` (Sharpe WF **0.933**, CAGR 14.22%, MaxDD -18.91%, WF pos 6/7).
- vs Phase 18 prod : Δ Sharpe WF = -0.024, Δ CAGR = +1.11%.

### BASELINE — Références de comparaison
- **4 configs**. Meilleure : `BL-E1` (Sharpe WF **0.962**, CAGR 6.75%, MaxDD -9.09%, WF pos 6/7).
- vs Phase 18 prod : Δ Sharpe WF = +0.006, Δ CAGR = -6.36%.

## CORE grid — Sharpe WF heatmap (DDon only)

Ligne = target_vol, colonne = max_leverage. Valeurs = Sharpe WF (seulement `dd_cap=ON` pour la lisibilité).

| tv \ ml |   4 |   6 |   8 |  10 |  12 |  14 |  16 |  18 |
|---------|---------|---------|---------|---------|---------|---------|---------|---------|
| tv=0.22 | 0.962 | 0.962 | 0.957 | 0.948 | 0.948 | 0.948 | 0.948 | 0.948 |
| tv=0.25 | 0.962 | 0.962 | 0.959 | 0.943 | 0.936 | 0.935 | 0.935 | 0.935 |
| tv=0.28 | 0.962 | 0.962 | 0.960 | 0.947 | 0.928 | 0.922 | 0.922 | 0.922 |
| tv=0.30 | 0.962 | 0.962 | 0.960 | 0.948 | 0.927 | 0.917 | 0.912 | 0.912 |
| tv=0.32 | 0.962 | 0.962 | 0.960 | 0.949 | 0.930 | 0.912 | 0.902 | 0.902 |
| tv=0.35 | 0.962 | 0.962 | 0.960 | 0.950 | 0.932 | 0.909 | 0.895 | 0.889 |

## CORE grid — CAGR heatmap (DDon only)

| tv \ ml |   4 |   6 |   8 |  10 |  12 |  14 |  16 |  18 |
|---------|---------|---------|---------|---------|---------|---------|---------|---------|
| tv=0.22 |  +4.50%|  +6.75%|  +8.89%| +10.79%| +11.63%| +11.63%| +11.63%| +11.63%|
| tv=0.25 |  +4.50%|  +6.75%|  +8.93%| +10.89%| +12.67%| +13.06%| +13.06%| +13.06%|
| tv=0.28 |  +4.50%|  +6.75%|  +8.96%| +11.01%| +12.82%| +14.43%| +14.43%| +14.43%|
| tv=0.30 |  +4.50%|  +6.75%|  +8.96%| +11.03%| +12.87%| +14.58%| +15.31%| +15.31%|
| tv=0.32 |  +4.50%|  +6.75%|  +8.97%| +11.06%| +12.98%| +14.69%| +16.14%| +16.14%|
| tv=0.35 |  +4.50%|  +6.75%|  +8.97%| +11.10%| +13.03%| +14.76%| +16.34%| +17.33%|

## Tableau complet (trié par Sharpe WF)

| ID | Block | Config | Sharpe WF | CAGR | Vol | MaxDD | WF pos | ★ |
|----|-------|--------|-----------|------|-----|-------|--------|----|
| `P19c-tv35-ml18-dd_OFF` | CORE | P18 weights / tv=0.35 ml=18 DDoff | 0.966 |   18.58% |   19.56% |  -25.57% | 6/7 |   |
| `P19c-tv25-ml14-dd_OFF` | CORE | P18 weights / tv=0.25 ml=14 DDoff | 0.966 |   13.38% |   13.97% |  -18.41% | 6/7 | * |
| `P19c-tv25-ml16-dd_OFF` | CORE | P18 weights / tv=0.25 ml=16 DDoff | 0.966 |   13.38% |   13.97% |  -18.41% | 6/7 | * |
| `P19c-tv25-ml18-dd_OFF` | CORE | P18 weights / tv=0.25 ml=18 DDoff | 0.966 |   13.38% |   13.97% |  -18.41% | 6/7 | * |
| `P19c-tv28-ml14-dd_OFF` | CORE | P18 weights / tv=0.28 ml=14 DDoff | 0.966 |   14.95% |   15.65% |  -20.57% | 6/7 | * |
| `P19c-tv32-ml16-dd_OFF` | CORE | P18 weights / tv=0.32 ml=16 DDoff | 0.966 |   17.03% |   17.89% |  -23.44% | 6/7 |   |
| `P19c-tv32-ml18-dd_OFF` | CORE | P18 weights / tv=0.32 ml=18 DDoff | 0.966 |   17.03% |   17.89% |  -23.44% | 6/7 |   |
| `P19c-tv30-ml16-dd_OFF` | CORE | P18 weights / tv=0.30 ml=16 DDoff | 0.966 |   15.99% |   16.77% |  -22.01% | 6/7 |   |
| `P19c-tv30-ml18-dd_OFF` | CORE | P18 weights / tv=0.30 ml=18 DDoff | 0.966 |   15.99% |   16.77% |  -22.01% | 6/7 |   |
| `P19c-tv28-ml16-dd_OFF` | CORE | P18 weights / tv=0.28 ml=16 DDoff | 0.966 |   14.95% |   15.65% |  -20.57% | 6/7 | * |
| `P19c-tv28-ml18-dd_OFF` | CORE | P18 weights / tv=0.28 ml=18 DDoff | 0.966 |   14.95% |   15.65% |  -20.57% | 6/7 | * |
| `P19c-tv22-ml12-dd_OFF` | CORE | P18 weights / tv=0.22 ml=12 DDoff | 0.966 |   11.80% |   12.30% |  -16.23% | 6/7 | * |
| `P19c-tv22-ml14-dd_OFF` | CORE | P18 weights / tv=0.22 ml=14 DDoff | 0.966 |   11.80% |   12.30% |  -16.23% | 6/7 | * |
| `P19c-tv22-ml16-dd_OFF` | CORE | P18 weights / tv=0.22 ml=16 DDoff | 0.966 |   11.80% |   12.30% |  -16.23% | 6/7 | * |
| `P19c-tv22-ml18-dd_OFF` | CORE | P18 weights / tv=0.22 ml=18 DDoff | 0.966 |   11.80% |   12.30% |  -16.23% | 6/7 | * |
| `P19c-tv25-ml12-dd_OFF` | CORE | P18 weights / tv=0.25 ml=12 DDoff | 0.963 |   12.94% |   13.51% |  -17.76% | 6/7 | * |
| `P19c-tv28-ml08-dd_OFF` | CORE | P18 weights / tv=0.28 ml=8 DDoff | 0.962 |    8.98% |    9.24% |  -12.07% | 6/7 |   |
| `P19c-tv35-ml10-dd_OFF` | CORE | P18 weights / tv=0.35 ml=10 DDoff | 0.962 |   11.20% |   11.56% |  -15.01% | 6/7 | * |
| `P19c-tv22-ml06-ddON` | CORE | P18 weights / tv=0.22 ml=6 DDon | 0.962 |    6.75% |    6.94% |   -9.09% | 6/7 |   |
| `P19c-tv22-ml06-dd_OFF` | CORE | P18 weights / tv=0.22 ml=6 DDoff | 0.962 |    6.75% |    6.94% |   -9.09% | 6/7 |   |
| `P19c-tv32-ml08-dd_OFF` | CORE | P18 weights / tv=0.32 ml=8 DDoff | 0.962 |    8.98% |    9.25% |  -12.07% | 6/7 |   |
| `P19c-tv35-ml08-dd_OFF` | CORE | P18 weights / tv=0.35 ml=8 DDoff | 0.962 |    8.98% |    9.25% |  -12.07% | 6/7 |   |
| `P19c-tv25-ml06-ddON` | CORE | P18 weights / tv=0.25 ml=6 DDon | 0.962 |    6.75% |    6.94% |   -9.09% | 6/7 |   |
| `P19c-tv25-ml06-dd_OFF` | CORE | P18 weights / tv=0.25 ml=6 DDoff | 0.962 |    6.75% |    6.94% |   -9.09% | 6/7 |   |
| `P19c-tv28-ml06-ddON` | CORE | P18 weights / tv=0.28 ml=6 DDon | 0.962 |    6.75% |    6.94% |   -9.09% | 6/7 |   |
| `P19c-tv28-ml06-dd_OFF` | CORE | P18 weights / tv=0.28 ml=6 DDoff | 0.962 |    6.75% |    6.94% |   -9.09% | 6/7 |   |
| `P19c-tv30-ml06-ddON` | CORE | P18 weights / tv=0.30 ml=6 DDon | 0.962 |    6.75% |    6.94% |   -9.09% | 6/7 |   |
| `P19c-tv30-ml06-dd_OFF` | CORE | P18 weights / tv=0.30 ml=6 DDoff | 0.962 |    6.75% |    6.94% |   -9.09% | 6/7 |   |
| `P19c-tv32-ml06-ddON` | CORE | P18 weights / tv=0.32 ml=6 DDon | 0.962 |    6.75% |    6.94% |   -9.09% | 6/7 |   |
| `P19c-tv32-ml06-dd_OFF` | CORE | P18 weights / tv=0.32 ml=6 DDoff | 0.962 |    6.75% |    6.94% |   -9.09% | 6/7 |   |
| `P19c-tv35-ml06-ddON` | CORE | P18 weights / tv=0.35 ml=6 DDon | 0.962 |    6.75% |    6.94% |   -9.09% | 6/7 |   |
| `P19c-tv35-ml06-dd_OFF` | CORE | P18 weights / tv=0.35 ml=6 DDoff | 0.962 |    6.75% |    6.94% |   -9.09% | 6/7 |   |
| `BL-E1` | BASELINE | E1 (P18 weights / tv=0.28 ml=6 DDon) | 0.962 |    6.75% |    6.94% |   -9.09% | 6/7 |   |
| `P19c-tv22-ml04-ddON` | CORE | P18 weights / tv=0.22 ml=4 DDon | 0.962 |    4.50% |    4.62% |   -6.09% | 6/7 |   |
| `P19c-tv22-ml04-dd_OFF` | CORE | P18 weights / tv=0.22 ml=4 DDoff | 0.962 |    4.50% |    4.62% |   -6.09% | 6/7 |   |
| `P19c-tv25-ml04-ddON` | CORE | P18 weights / tv=0.25 ml=4 DDon | 0.962 |    4.50% |    4.62% |   -6.09% | 6/7 |   |
| `P19c-tv25-ml04-dd_OFF` | CORE | P18 weights / tv=0.25 ml=4 DDoff | 0.962 |    4.50% |    4.62% |   -6.09% | 6/7 |   |
| `P19c-tv28-ml04-ddON` | CORE | P18 weights / tv=0.28 ml=4 DDon | 0.962 |    4.50% |    4.62% |   -6.09% | 6/7 |   |
| `P19c-tv28-ml04-dd_OFF` | CORE | P18 weights / tv=0.28 ml=4 DDoff | 0.962 |    4.50% |    4.62% |   -6.09% | 6/7 |   |
| `P19c-tv30-ml04-ddON` | CORE | P18 weights / tv=0.30 ml=4 DDon | 0.962 |    4.50% |    4.62% |   -6.09% | 6/7 |   |
| `P19c-tv30-ml04-dd_OFF` | CORE | P18 weights / tv=0.30 ml=4 DDoff | 0.962 |    4.50% |    4.62% |   -6.09% | 6/7 |   |
| `P19c-tv32-ml04-ddON` | CORE | P18 weights / tv=0.32 ml=4 DDon | 0.962 |    4.50% |    4.62% |   -6.09% | 6/7 |   |
| `P19c-tv32-ml04-dd_OFF` | CORE | P18 weights / tv=0.32 ml=4 DDoff | 0.962 |    4.50% |    4.62% |   -6.09% | 6/7 |   |
| `P19c-tv35-ml04-ddON` | CORE | P18 weights / tv=0.35 ml=4 DDon | 0.962 |    4.50% |    4.62% |   -6.09% | 6/7 |   |
| `P19c-tv35-ml04-dd_OFF` | CORE | P18 weights / tv=0.35 ml=4 DDoff | 0.962 |    4.50% |    4.62% |   -6.09% | 6/7 |   |
| `BL-D1` | BASELINE | D1 (P18 weights / tv=0.10 ml=3 DDon) | 0.962 |    3.37% |    3.46% |   -4.57% | 6/7 |   |
| `P19c-tv30-ml08-dd_OFF` | CORE | P18 weights / tv=0.30 ml=8 DDoff | 0.962 |    8.98% |    9.25% |  -12.07% | 6/7 |   |
| `P19c-tv30-ml14-dd_OFF` | CORE | P18 weights / tv=0.30 ml=14 DDoff | 0.961 |   15.10% |   15.83% |  -20.72% | 6/7 |   |
| `P19c-tv32-ml10-dd_OFF` | CORE | P18 weights / tv=0.32 ml=10 DDoff | 0.961 |   11.16% |   11.54% |  -15.01% | 6/7 | * |
| `P19c-tv25-ml08-dd_OFF` | CORE | P18 weights / tv=0.25 ml=8 DDoff | 0.961 |    8.94% |    9.23% |  -12.07% | 6/7 |   |
| `P19c-tv30-ml10-dd_OFF` | CORE | P18 weights / tv=0.30 ml=10 DDoff | 0.960 |   11.14% |   11.53% |  -15.01% | 6/7 | * |
| `P19c-tv28-ml08-ddON` | CORE | P18 weights / tv=0.28 ml=8 DDon | 0.960 |    8.96% |    9.24% |  -12.02% | 6/7 |   |
| `P19c-tv35-ml12-dd_OFF` | CORE | P18 weights / tv=0.35 ml=12 DDoff | 0.960 |   13.32% |   13.84% |  -17.96% | 6/7 | * |
| `P19c-tv32-ml08-ddON` | CORE | P18 weights / tv=0.32 ml=8 DDon | 0.960 |    8.97% |    9.25% |  -12.02% | 6/7 |   |
| `P19c-tv35-ml08-ddON` | CORE | P18 weights / tv=0.35 ml=8 DDon | 0.960 |    8.97% |    9.25% |  -12.02% | 6/7 |   |
| `P19c-tv30-ml08-ddON` | CORE | P18 weights / tv=0.30 ml=8 DDon | 0.960 |    8.96% |    9.25% |  -12.02% | 6/7 |   |
| `P19c-tv35-ml16-dd_OFF` | CORE | P18 weights / tv=0.35 ml=16 DDoff | 0.960 |   17.24% |   18.15% |  -23.64% | 6/7 |   |
| `P19c-tv28-ml10-dd_OFF` | CORE | P18 weights / tv=0.28 ml=10 DDoff | 0.959 |   11.11% |   11.53% |  -15.01% | 6/7 | * |
| `P19c-tv22-ml10-dd_OFF` | CORE | P18 weights / tv=0.22 ml=10 DDoff | 0.959 |   10.88% |   11.35% |  -14.93% | 6/7 | * |
| `P19c-tv22-ml08-dd_OFF` | CORE | P18 weights / tv=0.22 ml=8 DDoff | 0.959 |    8.90% |    9.22% |  -12.07% | 6/7 |   |
| `P19c-tv25-ml08-ddON` | CORE | P18 weights / tv=0.25 ml=8 DDon | 0.959 |    8.93% |    9.23% |  -12.02% | 6/7 |   |
| `P19c-tv32-ml12-dd_OFF` | CORE | P18 weights / tv=0.32 ml=12 DDoff | 0.958 |   13.26% |   13.82% |  -17.96% | 6/7 | * |
| `P19c-tv22-ml08-ddON` | CORE | P18 weights / tv=0.22 ml=8 DDon | 0.957 |    8.89% |    9.22% |  -12.02% | 6/7 |   |
| `P19c-tv32-ml14-dd_OFF` | CORE | P18 weights / tv=0.32 ml=14 DDoff | 0.957 |   15.22% |   15.98% |  -20.83% | 6/7 |   |
| `P19c-tv28-ml12-dd_OFF` | CORE | P18 weights / tv=0.28 ml=12 DDoff | 0.956 |   13.11% |   13.73% |  -17.93% | 6/7 | * |
| `BL-P18prod` | BASELINE | Phase 18 prod (MR80/TS3p10/RSI10 tv=0.28 ml=12 DDoff) | 0.956 |   13.11% |   13.73% |  -17.93% | 6/7 | * |
| `P19c-tv30-ml12-dd_OFF` | CORE | P18 weights / tv=0.30 ml=12 DDoff | 0.955 |   13.16% |   13.79% |  -17.96% | 6/7 | * |
| `P19c-tv25-ml10-dd_OFF` | CORE | P18 weights / tv=0.25 ml=10 DDoff | 0.955 |   10.99% |   11.49% |  -15.01% | 6/7 | * |
| `P19c-tv35-ml14-dd_OFF` | CORE | P18 weights / tv=0.35 ml=14 DDoff | 0.955 |   15.30% |   16.09% |  -20.88% | 6/7 |   |
| `P19c-tv35-ml10-ddON` | CORE | P18 weights / tv=0.35 ml=10 DDon | 0.950 |   11.10% |   11.53% |  -14.72% | 6/7 | * |
| `P19c-tv32-ml10-ddON` | CORE | P18 weights / tv=0.32 ml=10 DDon | 0.949 |   11.06% |   11.52% |  -14.72% | 6/7 | * |
| `P19c-tv30-ml10-ddON` | CORE | P18 weights / tv=0.30 ml=10 DDon | 0.948 |   11.03% |   11.51% |  -14.72% | 6/7 | * |
| `P19c-tv22-ml10-ddON` | CORE | P18 weights / tv=0.22 ml=10 DDon | 0.948 |   10.79% |   11.33% |  -14.64% | 6/7 | * |
| `P19c-tv22-ml12-ddON` | CORE | P18 weights / tv=0.22 ml=12 DDon | 0.948 |   11.63% |   12.25% |  -15.81% | 6/7 | * |
| `P19c-tv22-ml14-ddON` | CORE | P18 weights / tv=0.22 ml=14 DDon | 0.948 |   11.63% |   12.25% |  -15.81% | 6/7 | * |
| `P19c-tv22-ml16-ddON` | CORE | P18 weights / tv=0.22 ml=16 DDon | 0.948 |   11.63% |   12.25% |  -15.81% | 6/7 | * |
| `P19c-tv22-ml18-ddON` | CORE | P18 weights / tv=0.22 ml=18 DDon | 0.948 |   11.63% |   12.25% |  -15.81% | 6/7 | * |
| `P19c-tv28-ml10-ddON` | CORE | P18 weights / tv=0.28 ml=10 DDon | 0.947 |   11.01% |   11.50% |  -14.72% | 6/7 | * |
| `P19c-tv25-ml10-ddON` | CORE | P18 weights / tv=0.25 ml=10 DDon | 0.943 |   10.89% |   11.47% |  -14.72% | 6/7 | * |
| `P19c-tv25-ml12-ddON` | CORE | P18 weights / tv=0.25 ml=12 DDon | 0.936 |   12.67% |   13.42% |  -17.16% | 6/7 | * |
| `P19c-tv25-ml14-ddON` | CORE | P18 weights / tv=0.25 ml=14 DDon | 0.935 |   13.06% |   13.86% |  -17.73% | 6/7 | * |
| `P19c-tv25-ml16-ddON` | CORE | P18 weights / tv=0.25 ml=16 DDon | 0.935 |   13.06% |   13.86% |  -17.73% | 6/7 | * |
| `P19c-tv25-ml18-ddON` | CORE | P18 weights / tv=0.25 ml=18 DDon | 0.935 |   13.06% |   13.86% |  -17.73% | 6/7 | * |
| `P19w-75-10-15` | WEIGHTS | weights=75-10-15 / tv=0.28 ml=14 DDon | 0.933 |   14.22% |   15.07% |  -18.91% | 6/7 | * |
| `P19c-tv35-ml12-ddON` | CORE | P18 weights / tv=0.35 ml=12 DDon | 0.932 |   13.03% |   13.74% |  -17.33% | 6/7 | * |
| `P19c-tv32-ml12-ddON` | CORE | P18 weights / tv=0.32 ml=12 DDon | 0.930 |   12.98% |   13.73% |  -17.33% | 6/7 | * |
| `P19c-tv28-ml12-ddON` | CORE | P18 weights / tv=0.28 ml=12 DDon | 0.928 |   12.82% |   13.63% |  -17.31% | 6/7 | * |
| `P19c-tv30-ml12-ddON` | CORE | P18 weights / tv=0.30 ml=12 DDon | 0.927 |   12.87% |   13.69% |  -17.33% | 6/7 | * |
| `P19c-tv28-ml14-ddON` | CORE | P18 weights / tv=0.28 ml=14 DDon | 0.922 |   14.43% |   15.45% |  -19.85% | 6/7 | * |
| `P19w-80-10-10` | WEIGHTS | weights=80-10-10 / tv=0.28 ml=14 DDon | 0.922 |   14.43% |   15.45% |  -19.85% | 6/7 | * |
| `BL-E5` | BASELINE | E5 (P18 weights / tv=0.28 ml=14 DDon) | 0.922 |   14.43% |   15.45% |  -19.85% | 6/7 | * |
| `P19c-tv28-ml16-ddON` | CORE | P18 weights / tv=0.28 ml=16 DDon | 0.922 |   14.43% |   15.45% |  -19.85% | 6/7 | * |
| `P19c-tv28-ml18-ddON` | CORE | P18 weights / tv=0.28 ml=18 DDon | 0.922 |   14.43% |   15.45% |  -19.85% | 6/7 | * |
| `P19c-tv30-ml14-ddON` | CORE | P18 weights / tv=0.30 ml=14 DDon | 0.917 |   14.58% |   15.62% |  -19.98% | 6/7 | * |
| `P19c-tv30-ml16-ddON` | CORE | P18 weights / tv=0.30 ml=16 DDon | 0.912 |   15.31% |   16.48% |  -21.42% | 6/7 |   |
| `P19c-tv30-ml18-ddON` | CORE | P18 weights / tv=0.30 ml=18 DDon | 0.912 |   15.31% |   16.48% |  -21.42% | 6/7 |   |
| `P19c-tv32-ml14-ddON` | CORE | P18 weights / tv=0.32 ml=14 DDon | 0.912 |   14.69% |   15.77% |  -20.08% | 6/7 | * |
| `P19c-tv35-ml14-ddON` | CORE | P18 weights / tv=0.35 ml=14 DDon | 0.909 |   14.76% |   15.87% |  -20.13% | 6/7 | * |
| `P19c-tv32-ml16-ddON` | CORE | P18 weights / tv=0.32 ml=16 DDon | 0.902 |   16.14% |   17.50% |  -22.96% | 6/7 |   |
| `P19c-tv32-ml18-ddON` | CORE | P18 weights / tv=0.32 ml=18 DDon | 0.902 |   16.14% |   17.50% |  -22.96% | 6/7 |   |
| `P19h-tv32-ml18` | HLEV | P18 weights / tv=0.32 ml=18 DDon (high-lev) | 0.902 |   16.14% |   17.50% |  -22.96% | 6/7 |   |
| `P19h-tv32-ml20` | HLEV | P18 weights / tv=0.32 ml=20 DDon (high-lev) | 0.902 |   16.14% |   17.50% |  -22.96% | 6/7 |   |
| `P19h-tv32-ml24` | HLEV | P18 weights / tv=0.32 ml=24 DDon (high-lev) | 0.902 |   16.14% |   17.50% |  -22.96% | 6/7 |   |
| `P19w-85-10-5` | WEIGHTS | weights=85-10-5 / tv=0.28 ml=14 DDon | 0.898 |   14.43% |   15.89% |  -21.03% | 6/7 | * |
| `P19c-tv35-ml16-ddON` | CORE | P18 weights / tv=0.35 ml=16 DDon | 0.895 |   16.34% |   17.75% |  -23.10% | 6/7 |   |
| `P19w-70-15-15` | WEIGHTS | weights=70-15-15 / tv=0.28 ml=14 DDon | 0.895 |   15.43% |   16.44% |  -22.42% | 6/7 |   |
| `P19c-tv35-ml18-ddON` | CORE | P18 weights / tv=0.35 ml=18 DDon | 0.889 |   17.33% |   18.98% |  -25.19% | 6/7 |   |
| `P19h-tv35-ml18` | HLEV | P18 weights / tv=0.35 ml=18 DDon (high-lev) | 0.889 |   17.33% |   18.98% |  -25.19% | 6/7 |   |
| `P19h-tv35-ml20` | HLEV | P18 weights / tv=0.35 ml=20 DDon (high-lev) | 0.889 |   17.33% |   18.98% |  -25.19% | 6/7 |   |
| `P19h-tv35-ml24` | HLEV | P18 weights / tv=0.35 ml=24 DDon (high-lev) | 0.889 |   17.33% |   18.98% |  -25.19% | 6/7 |   |
| `P19w-75-15-10` | WEIGHTS | weights=75-15-10 / tv=0.28 ml=14 DDon | 0.886 |   15.68% |   16.80% |  -23.30% | 6/7 |   |
| `P19h-tv40-ml18` | HLEV | P18 weights / tv=0.40 ml=18 DDon (high-lev) | 0.880 |   17.91% |   19.68% |  -26.15% | 6/7 |   |
| `P19h-tv40-ml20` | HLEV | P18 weights / tv=0.40 ml=20 DDon (high-lev) | 0.873 |   18.94% |   21.06% |  -28.97% | 6/7 |   |
| `P19h-tv40-ml24` | HLEV | P18 weights / tv=0.40 ml=24 DDon (high-lev) | 0.873 |   18.94% |   21.06% |  -28.97% | 6/7 |   |
| `P19w-80-15-5` | WEIGHTS | weights=80-15-5 / tv=0.28 ml=14 DDon | 0.863 |   15.67% |   17.19% |  -24.46% | 6/7 |   |
| `P19w-70-20-10` | WEIGHTS | weights=70-20-10 / tv=0.28 ml=14 DDon | 0.827 |   16.38% |   18.51% |  -26.37% | 5/7 |   |
