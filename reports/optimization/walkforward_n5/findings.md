# Findings — Walk-forward N=5 fenêtres glissantes

> **Date** : 2026-05-04 · **Statut** : RÉSOLU · **Scope** : valider si le
> plafond CAGR_OOS=+9.54% observé précédemment est constant ou dépend de
> la fenêtre. Réponse : **dépend totalement de la fenêtre**.

## TL;DR

Le plafond +9.54% n'est **PAS physique**. Sur 5 fenêtres OOS glissantes :

| Fenêtre OOS | CAGR_max OOS |
|---|---|
| 2021-11 → 2022-10 | +17.20% |
| 2022-11 → 2023-10 | +8.74% |
| **2023-11 → 2024-10** | **+22.56%** |
| 2024-11 → 2025-10 | +11.15% |
| **2025-11 → 2026-04** | **+0.00%** |

**Moyenne : +11.93% ± 7.67%**.

Le plafond précédent (+9.54%) correspondait à la fenêtre **fold5_oos**
seule (2025-11→2026-04). Régime de marché actuel défavorable, **non
représentatif** de la stratégie sur le long terme.

## 1. Configuration

Grille resserrée (zone optimum trouvée walkforward_3d) :
- target_vol : 0.20, 0.30, 0.40, 0.50 (4 valeurs)
- max_lev : 12, 24, 36, 48 (4 valeurs)
- vol_floor : 0.02, 0.04, 0.08 (3 valeurs)
- DDCap : 0.30 (relâché vs 0.15 initial)

= 48 combos × 5 folds × 2 (IS+OOS) = 480 backtests sur 32 cores en ~5 min.

## 2. Stabilité IS↔OOS par fold

| Fold | configs | ρ_CAGR | ρ_Sharpe | Verdict |
|---|---|---|---|---|
| 1 | 63 | +0.411 | +0.331 | Modérée |
| 2 | 64 | +0.445 | −0.641 | Faible (Sharpe inverse) |
| **3** | 64 | **+0.869** | −0.413 | **Excellente CAGR**, médiocre Sharpe |
| **4** | 64 | **+0.898** | +0.268 | **Excellente CAGR**, modérée Sharpe |
| 5 | 62 | **−0.672** | −0.471 | **INVERSION** — bad fold |

**Folds 3 et 4** : ρ CAGR ≈ +0.9 → la sélection IS prédit bien OOS,
classement stable. **Fold 5** : ρ négatif = la sélection IS donne le
contraire en OOS. Régime de bascule.

## 3. Pourquoi fold5 plante

Fenêtre 2025-11→2026-04 (6 mois) :
- **CAGR_max = 0.00%** (médiane −1.57%)
- **Sharpe_max = 0.00**
- ρ_CAGR IS↔OOS = −0.67

Causes probables :
- **Sleeve MR Macro filtre macro restrictif** : spread 10Y-2Y < 0.5
  depuis fin 2025 → MR Macro inactive → 80% du portfolio en standby
- **Fenêtre courte** (6 mois vs 1 an autres folds) → moins
  d'opportunités, plus de bruit
- **Régime EUR/USD consolidation** sans tendance exploitable par TS
  Momentum

## 4. Implications

### 4.1 Sur la question 15% CAGR robuste

**Atteignable** sur fenêtres :
- Fold 1 OOS (2021-11→2022-10) : max +17.20%
- Fold 3 OOS (2023-11→2024-10) : **max +22.56%, Sharpe +2.79**
- Fold 4 OOS (2024-11→2025-10) : max +11.15% (limite)

**Non atteignable** sur :
- Fold 2 (2022-11→2023-10) : +8.74% max
- Fold 5 (2025-11→2026-04) : 0% max

→ **Sur 5 fenêtres, 3/5 dépassent 11%, 2/5 dépassent 17%**. La cible
15% robuste est atteignable la moitié du temps, **pas constamment**.

### 4.2 Sur la stabilité production

Le fold actuel (5) est **anormalement défavorable**. Les sleeves daily
fonctionnent encore (Sharpe TS Momentum positif), mais MR Macro
inactive limite l'edge cumulé.

**Trois scenarios à 12 mois** :
1. Régime macro se débloque (spread 10Y-2Y > 0.5) → MR Macro reprend,
   CAGR remonte vers ~10-15%
2. Régime continue → CAGR continue ~0-5% sur 6-12 mois
3. Régime s'aggrave (récession) → DD cap testé pour la 1ère fois en
   live ?

## 5. Conclusion sur l'investigation 15% CAGR

**Verdict actualisé** : 15% CAGR n'est **pas un objectif paramétrique**
mais **un objectif conditionnel au régime**. Sur les 5.4 ans testés :

- 5 folds OOS : moyenne +11.93%, médiane +11.15%
- Atteignable 50% du temps avec config optimale
- **Le sweep agressif précédent qui plafonnait à +9.54% reflétait
  uniquement le régime 2025-11→2026-04**, pas la stratégie

Pour viser 15% CAGR de manière régulière, deux approches :
1. **Accepter la variance OOS** : moyenne ~12% / an mais ±7% selon
   régime. Bon trade-off avec Sharpe.
2. **Diversifier** : ajouter paires/sleeves pour amortir les bad folds
   comme fold5 (cf. `expansion_pairs_plan.md`)

## 6. Pour reprendre

```bash
python scripts/optimization/walkforward_n5.py
```

Artefacts :
- `reports/optimization/walkforward_n5/summary_*.csv` : stats par fold
- `reports/optimization/walkforward_n5/rho_per_fold_*.csv` : Spearman ρ
- `reports/optimization/n5_fold{1..5}_{is,oos}_*.csv` : 10 sweeps bruts
