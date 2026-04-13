# Sweep combinations multi-stratégies — 2026-04-13

Sweep systématique de 37 configurations du combined portfolio v2, couvrant des variations non testées par `run_v2_benchmark` (alt-sleeve trios, allocations alternatives sur Phase 18, vol target sweep, DD cap réactivé, non-MR configs).

Baseline de comparaison : **Phase 18 prod** (`F1` ci-dessous) — Sharpe WF 0.956, CAGR 13.11%, MaxDD -17.93%.

## Top 5 par Walk-Forward Sharpe

| Rank | ID | Config | Sleeves | Sharpe WF | CAGR | Vol | MaxDD | WF pos | ★ |
|------|----|--------|---------|-----------|------|-----|-------|--------|----|
| 1 | B7 | All-4-sleeve / risk_parity / no lev | MR+TS_3p+RSI_4p+XS | **0.982** |    1.14% |    1.30% |   -1.99% | 6/7 |   |
| 2 | A1 | P18-sleeves / risk_parity / no lev | MR+TS_3p+RSI_4p | **0.981** |    1.09% |    1.14% |   -1.69% | 7/7 |   |
| 3 | A2 | P18-sleeves / risk_parity / tv=0.15 ml=5 DDon | MR+TS_3p+RSI_4p | **0.963** |    5.33% |    5.70% |   -8.33% | 6/7 |   |
| 4 | E1 | P18-prod / tv=0.28 ml=6 DDon | MR+TS_3p+RSI_4p | **0.962** |    6.75% |    6.94% |   -9.09% | 6/7 |   |
| 5 | D1 | P18-prod / tv=0.10 ml=3 DDon | MR+TS_3p+RSI_4p | **0.962** |    3.37% |    3.46% |   -4.57% | 6/7 |   |

★ = CAGR ∈ [10%, 15%] AND MaxDD < 35%.

## Bootstrap stress-test (top-5, 500 × 20-day blocks)

| Rank | ID | CAGR P5 | CAGR P50 | MaxDD P5 | MaxDD P50 | Sharpe P5 | Sharpe P95 | Pos frac | Target hit |
|------|----|---------|----------|----------|-----------|-----------|-----------|----------|------------|
| 1 | B7 |    0.49% |    1.16% |   -3.77% |   -2.08% |  +0.365 |  +1.381 |  100.0% |    0.0% |
| 2 | A1 |    0.45% |    1.03% |   -3.00% |   -1.79% |  +0.392 |  +1.372 |   99.8% |    0.0% |
| 3 | A2 |    2.09% |    5.07% |  -13.99% |   -8.79% |  +0.387 |  +1.383 |   99.8% |    0.2% |
| 4 | E1 |    2.92% |    6.72% |  -15.84% |   -9.73% |  +0.471 |  +1.494 |   99.6% |   10.4% |
| 5 | D1 |    1.52% |    3.37% |   -8.29% |   -4.92% |  +0.473 |  +1.492 |   99.6% |    0.0% |

## Tableau complet (37 configs, trié par Sharpe WF)

| ID | Block | Config | Sharpe WF | CAGR | Vol | MaxDD | WF pos | Full Sharpe | ★ |
|----|-------|--------|-----------|------|-----|-------|--------|-------------|----|
| B7 | B | All-4-sleeve / risk_parity / no lev | 0.982 |    1.14% |    1.30% |   -1.99% | 6/7 | 0.879 |   |
| A1 | A | P18-sleeves / risk_parity / no lev | 0.981 |    1.09% |    1.14% |   -1.69% | 7/7 | 0.956 |   |
| A2 | A | P18-sleeves / risk_parity / tv=0.15 ml=5 DDon | 0.963 |    5.33% |    5.70% |   -8.33% | 6/7 | 0.941 |   |
| E1 | E | P18-prod / tv=0.28 ml=6 DDon | 0.962 |    6.75% |    6.94% |   -9.09% | 6/7 | 0.976 |   |
| D1 | D | P18-prod / tv=0.10 ml=3 DDon | 0.962 |    3.37% |    3.46% |   -4.57% | 6/7 | 0.975 |   |
| D2 | D | P18-prod / tv=0.15 ml=4 DDon | 0.962 |    4.50% |    4.62% |   -6.09% | 6/7 | 0.976 |   |
| D3 | D | P18-prod / tv=0.18 ml=6 | 0.960 |    6.71% |    6.92% |   -9.09% | 6/7 | 0.973 |   |
| E2 | E | P18-prod / tv=0.28 ml=8 DDon | 0.960 |    8.96% |    9.24% |  -12.02% | 6/7 | 0.975 |   |
| D4 | D | P18-prod / tv=0.22 ml=8 | 0.959 |    8.90% |    9.22% |  -12.07% | 6/7 | 0.971 |   |
| F1 | F | Phase 18 prod (MR80/TS3p10/RSI10 tv=0.28 ml=12) | 0.956 |   13.11% |   13.73% |  -17.93% | 6/7 | 0.965 | * |
| D5 | D | P18-prod / tv=0.25 ml=10 | 0.955 |   10.99% |   11.49% |  -15.01% | 6/7 | 0.965 | * |
| A8 | A | P18-prod / tv=0.20 ml=8 DDon | 0.953 |    8.80% |    9.19% |  -12.02% | 6/7 | 0.964 |   |
| E3 | E | P18-prod / tv=0.28 ml=10 DDon | 0.947 |   11.01% |   11.50% |  -14.72% | 6/7 | 0.965 | * |
| A7 | A | P18-prod / tv=0.28 ml=12 DDon | 0.928 |   12.82% |   13.63% |  -17.31% | 6/7 | 0.953 | * |
| D6 | D | P18-prod / tv=0.28 ml=12 DDon | 0.928 |   12.82% |   13.63% |  -17.31% | 6/7 | 0.953 | * |
| E4 | E | P18-prod / tv=0.28 ml=12 DDon (dup A7) | 0.928 |   12.82% |   13.63% |  -17.31% | 6/7 | 0.953 | * |
| B5 | B | MR+TS_RSI(CAD)+RSI / 80-10-10 / tv=0.28 ml=12 | 0.924 |   11.97% |   13.40% |  -18.59% | 6/7 | 0.910 | * |
| E5 | E | P18-prod / tv=0.28 ml=14 DDon | 0.922 |   14.43% |   15.45% |  -19.85% | 6/7 | 0.950 | * |
| F2 | F | Phase 17 (MR90/TS3p10 tv=0.28 ml=12) | 0.915 |   13.53% |   14.87% |  -20.51% | 6/7 | 0.927 | * |
| F3 | F | Phase 16 (MR90/TS_RSI10 tv=0.28 ml=12) | 0.887 |   12.43% |   14.55% |  -20.66% | 5/7 | 0.877 | * |
| B4 | B | MR+TS3p / 90-10 / tv=0.28 ml=12 DDon (P17+DD) | 0.875 |   13.08% |   14.66% |  -19.90% | 6/7 | 0.911 | * |
| B3 | B | MR+RSI / 90-10 / tv=0.28 ml=12 | 0.836 |   10.28% |   13.26% |  -19.64% | 5/7 | 0.804 | * |
| B6 | B | All-4-sleeve / 70-10-10-10 / tv=0.28 ml=12 | 0.834 |   14.85% |   19.38% |  -28.55% | 6/7 | 0.811 | * |
| B1 | B | MR+XS+RSI / 80-10-10 / tv=0.28 ml=12 | 0.796 |   12.51% |   17.21% |  -29.39% | 5/7 | 0.771 | * |
| A3 | A | P18-sleeves / equal / no lev | 0.778 |    1.46% |    1.87% |   -3.45% | 6/7 | 0.785 |   |
| F4 | F | v1 risk_parity (MR+XS+TS_RSI) | 0.770 |    1.21% |    1.92% |   -4.27% | 6/7 | 0.636 |   |
| A4 | A | P18-sleeves / equal / tv=0.18 ml=6 | 0.766 |    8.38% |   11.20% |  -20.25% | 6/7 | 0.775 |   |
| B2 | B | MR+XS+RSI / 80-10-10 / tv=0.20 ml=8 DDon | 0.760 |    8.29% |   11.30% |  -19.74% | 5/7 | 0.761 |   |
| A5 | A | P18-sleeves / regime_adaptive / no lev | 0.750 |    1.26% |    1.62% |   -3.10% | 5/7 | 0.783 |   |
| A6 | A | P18-sleeves / regime_adaptive / tv=0.14 ml=4 DDon | 0.748 |    4.97% |    6.46% |  -11.90% | 5/7 | 0.783 |   |
| B8 | B | All-4-sleeve / equal / tv=0.18 ml=6 | 0.699 |   10.04% |   17.39% |  -22.61% | 6/7 | 0.637 | * |
| C3 | C | XS+TS3p+RSI / risk_parity / tv=0.18 ml=6 | 0.687 |    7.77% |   12.73% |  -15.05% | 7/7 | 0.651 |   |
| C2 | C | XS+TS3p+RSI / equal / tv=0.15 ml=5 DDon | 0.573 |    7.02% |   14.47% |  -19.58% | 6/7 | 0.541 |   |
| C5 | C | TS3p+RSI / 50-50 / tv=0.12 ml=4 DDon | 0.571 |    5.75% |   10.13% |  -18.35% | 6/7 | 0.602 |   |
| F5 | F | v1 mr_heavy (50/25/25) | 0.553 |    1.87% |    3.62% |   -5.46% | 6/7 | 0.530 |   |
| C1 | C | XS+TS3p+RSI / equal / no lev | 0.551 |    2.23% |    4.49% |   -6.88% | 6/7 | 0.514 |   |
| C4 | C | XS+RSI / 50-50 / tv=0.15 ml=5 DDon | 0.359 |    3.95% |   14.31% |  -23.06% | 6/7 | 0.342 |   |

## Lectures par bloc

### Bloc A — Phase 18 sleeves × allocations alternatives

- **8 configs** testées. Meilleure : `A1` (Sharpe WF 0.981, CAGR 1.09%, MaxDD -1.69%).
- **Pire** : `A6` (Sharpe WF 0.748, CAGR 4.97%).
- vs Phase 18 prod : Δ Sharpe WF = +0.025.

### Bloc B — Alt-sleeve trios (différentes compositions 2-4 sleeves)

- **8 configs** testées. Meilleure : `B7` (Sharpe WF 0.982, CAGR 1.14%, MaxDD -1.99%).
- **Pire** : `B8` (Sharpe WF 0.699, CAGR 10.04%).
- vs Phase 18 prod : Δ Sharpe WF = +0.026.

### Bloc C — Non-MR 'all-daily' (robustness check sans MR Macro)

- **5 configs** testées. Meilleure : `C3` (Sharpe WF 0.687, CAGR 7.77%, MaxDD -15.05%).
- **Pire** : `C4` (Sharpe WF 0.359, CAGR 3.95%).
- vs Phase 18 prod : Δ Sharpe WF = -0.269.

### Bloc D — Vol target sweep sur Phase 18 sleeves

- **6 configs** testées. Meilleure : `D1` (Sharpe WF 0.962, CAGR 3.37%, MaxDD -4.57%).
- **Pire** : `D6` (Sharpe WF 0.928, CAGR 12.82%).
- vs Phase 18 prod : Δ Sharpe WF = +0.005.

### Bloc E — DD cap reactivation sur Phase 18 (ml ∈ [6, 14])

- **5 configs** testées. Meilleure : `E1` (Sharpe WF 0.962, CAGR 6.75%, MaxDD -9.09%).
- **Pire** : `E5` (Sharpe WF 0.922, CAGR 14.43%).
- vs Phase 18 prod : Δ Sharpe WF = +0.006.

### Bloc F — Baselines (Phase 18, 17, 16, v1)

- **5 configs** testées. Meilleure : `F1` (Sharpe WF 0.956, CAGR 13.11%, MaxDD -17.93%).
- **Pire** : `F5` (Sharpe WF 0.553, CAGR 1.87%).
- vs Phase 18 prod : Δ Sharpe WF = +0.000.
