# Findings — Sweep agressif visant CAGR ≥ 15% robuste

> **Date** : 2026-05-04 · **Statut** : NEGATIF (objectif non atteignable)
> **Scope** : recherche de configs portfolio combiné atteignant 15% CAGR
> robuste IS+OOS sur la fenêtre 2020-11 → 2026-04.

## TL;DR

**Objectif 15% CAGR robuste = non atteignable** dans la configuration
actuelle. Le plafond OOS observé sur la fenêtre 2024-11 → 2026-04 est
**+9.54%** (CAGR), atteint à `vt=2.0, lev=80`. Aucune config sur 637
ne combine CAGR_IS ≥ 15% **et** CAGR_OOS ≥ 10%.

8 configs passent le filtre relâché `CAGR_avg ≥ 15%` mais avec :
- `vt=2.0` (cible vol annualisée **200%** — non-physique sur FX majeurs)
- DD_max = 15.00% pile (= circuit-breaker `Inp_DDCap` artificiellement
  plafonné — DD réel sous-jacent probablement 20-30%)
- Sharpe_min = 0.62 (vs 1.10 pour défaut compilé)

## 1. Pourquoi le levier est plafonné

```mql5
// FxRiskManager.mqh:166-167
realized = MAX(σ21, σ63, vol_floor)
leverage = MIN(target_vol / realized, max_lev)
```

| target_vol | vol_floor | Levier max effectif |
|---|---|---|
| 0.28 (défaut) | 0.02 (défaut) | 14× |
| 0.50 | 0.04 | 12.5× |
| 1.00 | 0.02 | 50× |
| **2.00** | **0.075** | **26.7×** ← optimum agressif |

Le levier broker (1:80) n'est **pas exploitable** car `target_vol/vol_floor`
sature avant. Pour libérer 80×, il faudrait `vt > 6.0` ou `vfloor < 0.025`
combiné à `vt > 2.0` — toutes deux non-physiques.

## 2. Plafond OOS empirique

Stats globales du sweep (637 configs sur fenêtre OOS 2024-11 → 2026-04) :

| | min | max | médiane |
|---|---|---|---|
| CAGR_OOS | +2.15% | **+9.54%** | +5.20% |
| Sharpe_OOS | +0.48 | +0.83 | +0.66 |

**Le plafond +9.54% est physique** sur cette fenêtre courte. Aucune combo
de paramètres ne le dépasse. Causes probables :
- Régime EUR/USD consolidation post-Fed
- Divergence ECB/BoJ moins claire qu'en 2022-2023
- Sleeve MR Macro filtre macro plus restrictif (spread 10Y-2Y < 0.5)

## 3. Top 8 candidats (filtre relâché)

```
vt=2.00, vfloor=0.075, lev∈[32..80]:  CAGR_IS=+21.64%  CAGR_OOS=+9.22%
                                       DD_max=+15.00%  Sharpe_min=+0.63
vt=2.00, vfloor=0.080, lev∈[32..80]:  CAGR_IS=+21.26%  CAGR_OOS=+9.10%
                                       DD_max=+15.01%  Sharpe_min=+0.62
```

CAGR_avg = 15.43% mais **fragile** :
- Sharpe IS de 0.63 — l'agressivité ne paye pas en risk-adjusted
- DD pile au seuil DD-cap (le système freine artificiellement)
- Spearman ρ CAGR IS↔OOS = **+0.524** (vs +0.708 sweep modéré) — moins stable

## 4. Drapeaux rouges

1. **`vt=2.00`** = cible vol annualisée **200%**. La vol réelle FX majeurs
   est 7-12%. Le système est en **lev_max permanent à 26.7×** — il ne
   "respire" plus avec la vol réalisée.
2. **DD_max = 15.00%** = circuit-breaker `Inp_DDCap=0.15` activé. Sans
   ce plafond, DD réel observable serait probablement 20-30%.
3. **CAGR_IS jusqu'à +21.64%** mais **CAGR_OOS plafonne à +9.54%** — chute
   de 56%. C'est l'anti-pattern classique de paramètres à fort levier
   qui captent le bruit IS sans généraliser.

## 5. Conclusion

**Pour 15% CAGR robuste**, l'edge actuel ne suffit pas. Trois pistes
d'expansion (hors-scope ce sweep) :

1. **Ajouter des sleeves** (autres edges, paires alternatives)
2. **Élargir walk-forward N=5 fenêtres** glissantes pour confirmer si
   le plafond OOS 9.5% est constant ou un artefact 2024-2026
3. **Re-runner avec model=4 (real ticks)** : peut révéler que les coûts
   réels rendent même 9% non atteignable (cf. investigation RSI Daily où
   30-40 bps round-trip dégradent Sharpe de 0.46)

## 6. Recommandation finale

**Conserver les défauts compilés** (`vt=0.28, lev=12, vfloor=0.02`) qui
donnent ~6-7% CAGR robuste avec Sharpe 1.0+. Ne pas pousser vers 15%
sans expansion structurelle (nouveaux sleeves) — l'amélioration
paramétrique pure plafonne autour de 7-9% CAGR robuste.

Si 15% reste un objectif business, investir dans la R&D de nouveaux
edges (autres timeframes, autres paires, alternatives data) plutôt que
de tordre les paramètres existants.

## 7. Artefacts

- `reports/optimization/walkforward_aggressive/findings.md` (ce doc)
- `reports/optimization/walkforward_aggressive/scatter_*.png`
- `reports/optimization/walkforward_aggressive/merged_*.csv` (637 configs joinées)
- `reports/optimization/walkforward_aggressive/strict_15pct_*.csv` (8 candidats)
- `reports/optimization/agg_{full,is,oos}_*.csv|png` (3 sweeps bruts)
- `scripts/optimization/walkforward_aggressive.py`
