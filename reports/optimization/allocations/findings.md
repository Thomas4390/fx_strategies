# Phase C — Allocation sweep (findings)

> **Date** : 2026-05-04 · **Plan source** :
> `docs/investigations/cagr_improvement_plan.md` (Phase C)

## Méthodologie

Stage 1 : 6 allocations × 1 fenêtre 5.4 ans (screening rapide).
Stage 2 : top 2 par Sharpe + baseline × N=5 folds (validation OOS).

Allocations testées (sum = 1.0) :

| Label | MR | TS | RSI |
|---|---|---|---|
| conservative | 0.50 | 0.25 | 0.25 |
| balanced | 0.60 | 0.20 | 0.20 |
| **baseline** | **0.80** | **0.10** | **0.10** |
| mr_heavy | 0.90 | 0.05 | 0.05 |
| equal | 0.34 | 0.33 | 0.33 |
| ts_heavy | 0.40 | 0.50 | 0.10 |

## Stage 1 — Screening 5.4 ans

| Alloc | Sharpe | CAGR % | Net (USD) | PF | DD % |
|---|---|---|---|---|---|
| baseline | **1.15** | +7.24 | +4 615 | 1.38 | 7.21 |
| mr_heavy | 1.10 | +6.99 | +4 434 | 1.42 | 7.46 |
| balanced | 0.87 | +6.07 | +3 775 | 1.26 | 7.61 |
| conservative | 0.76 | +5.78 | +3 566 | 1.24 | 7.78 |
| ts_heavy | 0.63 | +7.10 | +4 517 | 1.23 | 9.95 |
| equal | 0.56 | +4.80 | +2 902 | 1.19 | 10.85 |

**Lecture** :
- 80/10/10 = top Sharpe. Toute dilution vers TS/RSI dégrade le risk-adjusted return.
- ts_heavy a CAGR proche du baseline mais DD presque 40 % plus élevé.
- equal alloc = pire de tout (Sharpe 0.56) car RSI flat dilue MR sans
  ajouter d'edge.

## Stage 2 — Walk-forward N=5 OOS

Top 2 par Sharpe = `baseline, mr_heavy` (les autres écartés direct).

| Candidate | Sharpe_med | Sharpe_avg | CAGR_avg % | MaxDD % |
|---|---|---|---|---|
| **baseline 80/10/10** | **1.50** | 1.12 | +7.76 | 5.98 |
| mr_heavy 90/5/5 | 1.37 | 1.01 | +8.60 | 6.00 |

## Verdict

| Critère §3.1 | mr_heavy vs baseline | Status |
|---|---|---|
| ΔSharpe_med ≥ +0.05 | -0.13 | ✗ |
| ΔCAGR_avg ≥ +1.0 pp | +0.84 pp | ✗ (proche) |
| ΔMaxDD ≤ +2.0 pp | +0.02 pp | ✓ |

**mr_heavy NON retenu** : trade-off Sharpe sacrifice trop important pour
+0.84 pp CAGR. Confirme que **80/10/10 est optimal** pour le portfolio
actuel de 4 paires.

## Insight

Cohérent avec findings Phase B :
- **MR Macro = edge réel et diversifié** (4 paires, PF 1.47) → mérite alloc max
- **TS Momentum = concentré sur USDJPY** (83 % du PnL sleeve) → augmenter
  son alloc augmente la concentration risk
- **RSI Daily = flat (PF 1.01)** → augmenter alloc dilue sans contribuer

→ Le portfolio est saturé en édges utiles. Le seul levier restant n'est
pas allocation mais **nouveaux édges** (Phase D timeframe H1, Phase E
refonte sleeves, Phase F carry).

## Implication critère §3.2

> Phase C : 80/10/10 reste optimal → SKIP Phase D-F, aller direct G+H.

Le critère du plan source dit de skip D-F. Cependant :
- Phase B révèle que TS Momentum est concentré → Phase E.2 (TS EMA grid)
  reste pertinente pour valider la robustesse.
- Phase D (H1 momentum) ajoute une dimension orthogonale (timeframe), pas
  une re-pondération → reste pertinente.
- Phase F (carry) ajoute un edge orthogonal → reste pertinente.

**Recommandation** : continuer Phase D + E (E.2 surtout), puis F conditionnel
si E n'atteint pas CAGR ≥ 10 %. Skip uniquement Phase C "alloc fine-tuning"
qui est terminée.

## Artifacts

- `reports/optimization/allocations/screening.csv`
- `reports/optimization/allocations/walkforward.csv` (30 lignes)
- `reports/optimization/allocations/walkforward_summary.csv`
- `scripts/optimization/walkforward_allocations.py`

## Reproduction

```bash
python scripts/optimization/walkforward_allocations.py
```

Durée : ~12 min sur Linux/Wine (36 backtests × 15-25 s chacun).
