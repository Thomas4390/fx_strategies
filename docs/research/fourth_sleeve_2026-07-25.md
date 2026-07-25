# Fourth-sleeve evaluation sweep (2026-07-25)

Follow-up to the weight sweep which showed that recombining the 3-sleeve production trio weights only moves the WF Sharpe by a few basis points above the leverage plateau (0.966). This sweep tests whether adding a fourth orthogonal sleeve — Composite FX Alpha, OU Mean Reversion, or a restored XS Momentum — can push the Sharpe above 0.97.

Fixed parameters : `tv=0.25 / ml=14 / DDoff` (leverage plateau sweet-spot). Each config takes a base trio split and reallocates a fraction ∈ {5%, 10%, 15%} to the extra sleeve.

## Extra sleeve correlations vs the production trio

| Extra sleeve | Corr with MR_Macro | Corr with TS_3p | Corr with RSI_4p |
|-------------|-------------------:|----------------:|-----------------:|
| Composite_FX_Alpha | +0.033 | +0.560 | -0.370 |
| OU_MR | +0.460 | -0.004 | -0.003 |
| XS_Momentum | +0.005 | +0.425 | -0.405 |
| Gold_Momentum | -0.005 | +0.092 | -0.032 |

Leverage plateau baseline : **BL-plateau** — Sharpe WF 0.731, CAGR 17.23%, MaxDD -39.00%.

Weight-sweep top-1 baseline : **BL-weight-top** — Sharpe WF 0.802, CAGR 19.01%, MaxDD -38.63%.

## Top 10 by Walk-Forward Sharpe

| Rank | ID | Config | Sharpe WF | CAGR | Vol | MaxDD | WF pos | ★ |
|------|----|--------|-----------|------|-----|-------|--------|----|
| 1 | `SLV-g75-10-15-G15` | +Gold_Momentum 15% / base=75-10-15 / tv=0.25 ml=14 DDoff | **1.052** |   27.52% |   25.22% |  -38.32% | 6/7 |   |
| 2 | `SLV-g70-12-18-G15` | +Gold_Momentum 15% / base=70-12-18 / tv=0.25 ml=14 DDoff | **1.045** |   27.24% |   25.18% |  -38.42% | 6/7 |   |
| 3 | `SLV-g80-08-12-G15` | +Gold_Momentum 15% / base=80-8-12 / tv=0.25 ml=14 DDoff | **1.040** |   27.30% |   25.37% |  -38.18% | 6/7 |   |
| 4 | `SLV-g80-10-10-G15` | +Gold_Momentum 15% / base=80-10-10 / tv=0.25 ml=14 DDoff | **1.002** |   26.17% |   25.10% |  -34.20% | 6/7 |   |
| 5 | `SLV-g75-10-15-G10` | +Gold_Momentum 10% / base=75-10-15 / tv=0.25 ml=14 DDoff | **0.983** |   25.21% |   25.28% |  -38.39% | 6/7 |   |
| 6 | `SLV-g70-12-18-G10` | +Gold_Momentum 10% / base=70-12-18 / tv=0.25 ml=14 DDoff | **0.977** |   24.88% |   25.25% |  -38.25% | 6/7 |   |
| 7 | `SLV-g80-08-12-G10` | +Gold_Momentum 10% / base=80-8-12 / tv=0.25 ml=14 DDoff | **0.972** |   25.11% |   25.41% |  -38.43% | 6/7 |   |
| 8 | `SLV-g75-15-10-G15` | +Gold_Momentum 15% / base=75-15-10 / tv=0.25 ml=14 DDoff | **0.930** |   23.87% |   24.88% |  -33.73% | 6/7 |   |
| 9 | `SLV-g80-10-10-G10` | +Gold_Momentum 10% / base=80-10-10 / tv=0.25 ml=14 DDoff | **0.926** |   23.74% |   25.12% |  -34.73% | 6/7 |   |
| 10 | `SLV-g70-12-18-G05` | +Gold_Momentum 5% / base=70-12-18 / tv=0.25 ml=14 DDoff | **0.895** |   21.97% |   25.26% |  -38.07% | 6/7 |   |

★ = CAGR ∈ [10%, 15%] AND MaxDD < 35%.

## Best config per block

| Block | ID | Config | Sharpe WF | CAGR | MaxDD |
|-------|----|--------|-----------|------|-------|
| COMPOSITE | `SLV-c70-12-18-C05` | +Composite 5% / base=70-12-18 / tv=0.25 ml=14 DDoff | **0.803** |   18.96% |  -38.73% |
| OU_MR | `SLV-o70-12-18-O10` | +OU_MR 10% / base=70-12-18 / tv=0.25 ml=14 DDoff | **0.814** |   19.37% |  -38.44% |
| XS_REVISIT | `SLV-x70-12-18-X10` | +XS_Momentum 10% / base=70-12-18 / tv=0.25 ml=14 DDoff | **0.818** |   19.37% |  -37.82% |
| BASELINE | `BL-weight-top` | weight-sweep top-1 (MR75/TS10/RSI15 tv=0.25 ml=14 DDoff) | **0.802** |   19.01% |  -38.63% |

## Conclusion

Best point in the sweep : **`SLV-g75-10-15-G15`** — +Gold_Momentum 15% / base=75-10-15 / tv=0.25 ml=14 DDoff.  
Sharpe WF = **1.052** (vs leverage plateau = 0.731, Δ = +0.321 ; vs weight-sweep top = 0.802, Δ = +0.249).

Adding a fourth sleeve **materially improves** the Sharpe beyond what weight recombination could achieve. Validate on bootstrap before promoting.
