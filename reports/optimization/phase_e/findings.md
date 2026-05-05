# Phase E — Refonte sleeves (findings — POSITIF, CAGR ≥ 10 %)

> **Date** : 2026-05-04 · **Plan source** :
> `docs/investigations/cagr_improvement_plan.md` (Phase E)

## TL;DR

Phase E livre **2 améliorations validées N=5 OOS** :
1. **Session MR Macro 8-16 UTC** au lieu de 6-14 (London full + early NY)
2. **RSI Daily sans USDJPY** (pairs = EURUSD, GBPUSD, USDCAD)

Le combined config bat le baseline pre-A sur tous les critères §3.1 :

| Métrique | Baseline pre-A (5.4y) | Phase E final (5.4y) | Δ |
|---|---|---|---|
| Sharpe Ratio | 1.15 | **1.44** | +0.29 (+25 %) |
| Net Profit | +4 615 USD | **+6 161 USD** | +1 546 (+33 %) |
| Profit Factor | 1.38 | **1.54** | +0.16 |
| Recovery Factor | 4.98 | **5.84** | +0.86 |
| MaxDD | 7.21 % | 7.77 % | +0.56 pp |
| CAGR | 7.24 % | **9.18 %** | +1.94 pp |

Walk-forward N=5 OOS (combined config) :

| Métrique | Baseline pre-A | Phase E combined | Δ |
|---|---|---|---|
| Sharpe_med | 1.50 | **1.83** | +0.33 |
| Sharpe_avg | 1.12 | **1.51** | +0.39 |
| Net_avg / fold | +793 | **+1 015** | +222 (+28 %) |
| MaxDD_max | 5.98 % | **5.20 %** | **-0.78 pp** |
| CAGR_avg ≈ Net_avg/10k | ~7.9 % | **~10.15 %** | +2.25 pp |

## E.1 — MR Macro session sweep

### Sweep 4 fenêtres UTC (full 5.4y)

| Session | Sharpe | Net | PF | DD % | Trades | Verdict |
|---|---|---|---|---|---|---|
| **8-16 (London full)** | **1.34** | +5 649 | 1.49 | 7.80 | 797 | ✓ retain |
| 6-14 (baseline) | 1.15 | +4 615 | 1.38 | 7.21 | 835 | — |
| 13-21 (NY full) | 1.14 | +4 848 | 1.46 | 12.38 | 732 | ✗ DD trop élevé |
| 0-23 (24h) | -0.43 | -655 | 0.93 | 15.13 | 955 | ✗ catastrophique |

### N=5 OOS validation 8-16 vs 6-14

| Cand | Sharpe_med | Sharpe_avg | Net_avg | DD_max |
|---|---|---|---|---|
| baseline_6_14 | 1.50 | 1.12 | 793 | 5.98 |
| **london_8_16** | **1.77** | 1.38 | 950 | 5.07 |

ΔSharpe_med = **+0.27**, ΔDD = **-0.91 pp** → ✓ RETAIN.

**Insight** : la session 8-16 capture l'overlap London + early NY (liquidité
maximale, mean-reversion plus efficace), évite la session asiatique
peu liquide.

## E.2 — TS Momentum EMA grid

### Grid 17 combos (full 5.4y)

Top 5 par Sharpe (avec session 8-16 déjà appliquée) :

| Fast | Slow | Sharpe | Net | DD % | Trades |
|---|---|---|---|---|---|
| 14 | 50 | 1.40 | +5 712 | 6.54 | 820 |
| 30 | 50 | 1.40 | +6 016 | 7.92 | 781 |
| 10 | 100 | 1.39 | +5 622 | 8.02 | 775 |
| **20 | 50** baseline | **1.34** | +5 649 | 7.80 | 797 |
| 14 | 100 | 1.33 | +5 467 | 8.34 | 765 |

### N=5 OOS validation top 2

| Cand | Sharpe_med | Sharpe_avg | Net_avg | DD_max |
|---|---|---|---|---|
| baseline 20/50 | 1.77 | 1.38 | 950 | 5.07 |
| ts_14_50 | 1.80 (+0.03) | 1.34 | 919 | 5.25 |
| ts_30_50 | 1.69 (-0.08) | **1.69** (+0.31) | 988 | 4.95 |

**Verdict** : ✗ NON retenu strict (ΔSharpe_med < 0.05 sur les deux). Note :
ts_30_50 a un Sharpe_avg significativement plus haut (+0.31) suggérant
plus de robustesse moyenne. À investiguer si on revient sur Phase E
ultérieurement. TS 20/50 baseline conservé.

## E.3 — RSI Daily seuils + pairs

### Grid 8 variants (full 5.4y, session 8-16 + TS 20/50)

| Variant | Pairs | Seuils | Sharpe | Net | Verdict |
|---|---|---|---|---|---|
| **no_jpy 25/75/50** | EUR/GBP/CAD | 25/75/50 | **1.44** | **+6 161** | ✓ retain |
| no_jpy_loose | EUR/GBP/CAD | 30/70/50 | 1.41 | +6 033 | ✓ |
| no_jpy_strict | EUR/GBP/CAD | 20/80/50 | 1.39 | +5 741 | ✗ borderline |
| baseline | EUR/GBP/JPY/CAD | 25/75/50 | 1.34 | +5 649 | — |

### N=5 OOS validation no_jpy

| Cand | Sharpe_med | Sharpe_avg | Net_avg | DD_max |
|---|---|---|---|---|
| baseline | 1.77 | 1.38 | 950 | 5.07 |
| **no_jpy** | **1.83** (+0.06) | 1.51 (+0.13) | 1015 (+64) | 5.20 (+0.13 pp) |

**Verdict** : ✓ RETAIN. ΔSharpe_med = +0.06, ΔNet = +64 USD/fold,
ΔDD = +0.13 pp.

**Insight Phase B confirmé** : RSI Daily sur USDJPY = drag -295 USD sur 5.4y.
Retirer la paire transforme un sleeve flat (PF 1.01) en sleeve productif
(Sharpe combiné 1.44 vs 1.34 baseline post-session).

## Combined config N=5 OOS (validation finale)

Per-fold OOS Sharpe (session 8-16 + RSI no_jpy) :

| Fold | Window | Sharpe | Net | DD % |
|---|---|---|---|---|
| fold1 | 2021-11→2022-10 | **2.44** | +1 833 | 5.20 |
| fold2 | 2022-11→2023-10 | 1.00 | +557 | 4.94 |
| fold3 | 2023-11→2024-10 | **2.23** | +1 631 | 4.16 |
| fold4 | 2024-11→2025-10 | 1.83 | +1 045 | 3.28 |
| fold5 | 2025-11→2026-04 | **0.04** | +8 | 2.92 |

**Aggregates** : Sharpe_med 1.83, Sharpe_avg 1.51, Net_avg +1 015, DD_max 5.20 %.

CAGR_avg_OOS ≈ 1015 / 10 000 = **+10.15 %** ≥ critère plan source.

**Faiblesse fold5** : Sharpe 0.04 confirmant l'hypothèse Phase B (filtre
macro bloque MR Macro sur ce régime). Le levier `Inp_MR_DisableMacroFilter=true`
sur fold5 améliore Sharpe de +0.51 mais pèse sur full 5.4y. Une
solution adaptative (filtre soft 50 % au lieu de cut binaire) reste un
candidat fort pour un E.4 futur.

## Décision Phase F

Plan source §3.3 : `Si plafond CAGR reste < 10 % après Phase E → Phase F`.

**CAGR_avg_OOS = 10.15 %** → marginal au-dessus du seuil 10 %. Décision :
**SKIP Phase F** (carry sleeve, 3-4 h, dépend FRED BoJ/RBA/RBNZ rates,
risque négatif comme Phase D).

Le levier dispo est plutôt :
- **Filtre macro adaptatif** (E.4 futur) pour débloquer fold5
- **Anti-overfit validation** (Phase G obligatoire) pour confirmer que
  le combined Phase E n'est pas data-mining

Phase G next.

## Défauts compilés mis à jour

```mql5
// FxMultiSleeve.mq5 (post-Phase A + D + E)
Inp_AllocMRMacro      = 0.80      // unchanged (Phase C)
Inp_AllocTSMomentum   = 0.10      // unchanged
Inp_AllocRSIDaily     = 0.10      // unchanged
Inp_AllocH1Momentum   = 0.0       // off (Phase D negative)
Inp_EnableDDCap       = false     // off (Phase A.1)
Inp_EnableMarginCap   = false     // off (Phase A.2)
Inp_DDCap             = 0.30      // up from 0.15
Inp_MR_SessionStart   = 8         // up from 6 (Phase E.1)
Inp_MR_SessionEnd     = 16        // up from 14 (Phase E.1)
Inp_RSI_Pairs         = "EURUSD,GBPUSD,USDCAD"  // USDJPY removed (Phase E.3)
Inp_MR_DisableMacroFilter = false // diagnostic flag (Phase B.4)
Inp_ExportDeals       = false     // diagnostic flag (Phase B.1)
```

## Artifacts

```
scripts/optimization/walkforward_session.py        # E.1
scripts/optimization/walkforward_session_n5.py     # E.1 N=5
scripts/optimization/walkforward_ema.py            # E.2
scripts/optimization/walkforward_ema_n5.py         # E.2 N=5
scripts/optimization/walkforward_rsi_thresh.py     # E.3
scripts/optimization/walkforward_rsi_n5.py         # E.3 N=5

reports/optimization/sessions/{session_sweep,n5_session}.csv
reports/optimization/ts_ema/{grid,n5}.csv
reports/optimization/rsi_thresh/{grid,n5}.csv
reports/optimization/phase_e/findings.md           # this file
```
