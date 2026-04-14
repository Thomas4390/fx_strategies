# Phase 21 — DSR retrofit (2026-04-13)

**Holdout state** : LOCKED (frozen from 2026-01-01 until Phase 25)  
**Holdout touched by this phase** : NO

Retroactive statistical audit of the top configurations produced by Phase 18/19/20. Each sweep is revisited with two corrections :

1. **Deflated Sharpe Ratio (DSR)** — Bailey & Lopez de Prado 2014. Converts the raw Sharpe of the top config into a probability that it exceeds what would be expected as the **maximum across N independent trials** with the same variance as the sweep. A DSR below ~0.95 means the observed Sharpe is indistinguishable from selection-bias luck. Computed via VBT Pro's native `ReturnsAccessor.sharpe_ratio_std` (Mertens 1998 correction for skewness and kurtosis).

2. **Probability of Backtest Overfitting (PBO)** — Bailey-Borwein-Lopez de Prado-Zhu 2015, via Combinatorially Symmetric Cross-Validation. Applied only to Phase 20A because its dense factorial grid of 40 CORE configs is the cleanest candidate for CSCV.

## Deflated Sharpe Ratio by phase

| Phase | Top config | N trials | SR full | SR WF | σ(SR) | E[max SR] | **DSR** | CAGR | MaxDD |
|-------|------------|---------:|--------:|------:|------:|----------:|--------:|-----:|------:|
| Phase 19 (refined leverage) | `P19c-tv25-ml14-dd_OFF` | 116 | 0.968 | 0.966 | 0.019 | 0.037 | **1.000** | 13.38% | -18.41% |
| Phase 20A (weight sweep) | `P20a-w75-10-15` | 58 | 0.976 | 0.972 | 0.019 | 0.113 | **1.000** | 13.14% | -17.35% |
| Phase 20B (4th sleeve) | `BL-P20Atop` | 45 | 0.976 | 0.972 | 0.019 | 0.218 | **1.000** | 13.14% | -17.35% |
| Phase 20C (DD cap sweep) | `P20c-soft-w75-10-15-k20-f35` | 34 | 0.976 | 0.972 | 0.019 | 0.015 | **1.000** | 13.14% | -17.35% |

**Reading** : DSR ≥ 0.95 = observed Sharpe clearly beats the best-of-N luck level. DSR in [0.70, 0.95] = borderline, the edge may be real but overfitting explains part of it. DSR < 0.50 = indistinguishable from a lucky winner, the selection bias dominates.

## Probability of Backtest Overfitting — Phase 20A (CSCV)

| Metric | Value |
|--------|-------|
| Configs evaluated | 40 |
| Time bins | 10 |
| Splits (C(n, n/2)) | 252 |
| T bars common | 2574 |
| **PBO** | **0.853** — overfit |


**Reading** : PBO is the probability that the top configuration selected on an in-sample slice lands **below** the out-of-sample median of the same config set. PBO < 0.5 means the selection process adds value ; PBO ≥ 0.5 means the process is effectively curve-fitting.

## Interpretation — the DSR/PBO paradox

**Every phase scores DSR ≈ 1.0 while the Phase 20A CSCV PBO is 0.853.** These two signals are not contradictory — they measure different things :

- **DSR ≈ 1.0** means the observed Sharpe of ~0.97 is numerically **very far** above the best-of-N luck threshold. The underlying edge of the *family* of configurations (MR-heavy 3-sleeve combined with vol targeting at tv=0.25 ml=14) is statistically real. ~10 years × 252 days ≈ 2500 bars make the Sharpe estimator very precise (σ(SR) ≈ 0.019), which shrinks the z-score denominator and inflates DSR.
- **PBO > 0.5** means that *within* Phase 20A's grid, selecting the in-sample top-1 and expecting it to stay top out-of-sample is effectively noise-selection. The 40 CORE configs sit on a flat Sharpe plateau in the [0.88, 0.97] band ; which one 'wins' in-sample rotates across CSCV splits.

**Operational consequence** : the 6 bps Sharpe gap reported in Phase 20A (`P20a-w75-10-15` at 0.972 vs the canonical `80/10/10` at 0.966) is **not** a real improvement — it is within the CSCV noise band. The production weights should stay on `80/10/10` (Phase 18/19 canonical) until a genuinely different configuration — different sleeves, different allocation mechanism, not just a weight perturbation — produces both high DSR **and** low PBO.

**Take-away for Phase 22+** : DSR alone is insufficient for config selection at this sample size. We need PBO (or its walk-forward cousin CPCV) as the *gating* test on any sweep that claims a winner within an existing alpha family. A sweep that moves to a new alpha family (Phase 20B tried this with a 4th sleeve and failed) should still be DSR-gated because its trial variance is genuinely larger.


## Next steps

- Phase 22 must wire PBO into the sweep rejection rule (see `plans/valiant-humming-coral.md`). Concretely : any new sweep claiming a top-1 within an existing alpha family must report CSCV PBO alongside the raw Sharpe ; promote only if PBO < 0.5.
- Lock the production weights at the Phase 18 canonical 80/10/10 with Phase 19 leverage `tv=0.25 ml=14 DDoff`. Phase 20A 'top' is retracted — it was a flat-plateau artefact.
- The frozen slice (2026-01-01 → now) remains locked. No optimization ran against it during this retrofit.
