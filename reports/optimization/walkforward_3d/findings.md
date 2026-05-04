# Findings — Walk-forward 3D sweep (target_vol × max_lev × vol_floor)

> **Date** : 2026-05-04 · **Statut** : RÉSOLU · **Scope** : analyse de
> sensibilité du portfolio combiné FxMultiSleeve aux paramètres de
> vol-targeting global, avec contrôle anti-overfit.

## TL;DR

Sweep 3D natif MT5 (32 cores, 720 combos × 3 fenêtres = **2 160 backtests
en ~5 minutes**) sur les 3 paramètres d'overlay vol-targeting, avec **3
best practices anti-overfit** appliquées : walk-forward IS/OOS,
identification de plateau, et stabilité du rang Spearman. Conclusion :
**il n'existe pas de configuration unique optimale** — trois régimes
distincts émergent selon l'objectif :

| Objectif | Configuration robuste | CAGR_avg | Sharpe_OOS | Risque |
|---|---|---|---|---|
| **Défensif** (Sharpe live max) | vt=0.10, vfloor=0.08 | +2.35% | **+1.66** | Faible |
| **Balanced** (top quartile IS&OOS) | vt=0.10-0.15, vfloor=0.02-0.03 | ~+3-4% | ~+1.0 | Modéré |
| **Agressif** (CAGR max robuste) | vt=0.50, vfloor=0.04 | **+7.42%** | +0.67 | Élevé |

Le défaut compilé (`vt=0.28, lev=12, vfloor=0.02`) se situe entre Balanced
et Agressif — **compromis raisonnable, à conserver pour l'instant**.

---

## 1. Méthodologie anti-overfit

### 1.1 Walk-forward IS/OOS

Split 74/26 :
- **IS** = 2020-11-23 → 2024-10-31 (4.0 ans, fenêtre d'optimisation)
- **OOS** = 2024-11-01 → 2026-04-30 (1.5 ans, fenêtre de validation)

### 1.2 Indicateurs de robustesse

| Indicateur | Définition | Valeur observée | Verdict |
|---|---|---|---|
| **Spearman ρ (CAGR IS↔OOS)** | Corrélation des rangs | **+0.708** (p<10⁻¹¹⁰) | ✅ Très stable |
| **Spearman ρ (Sharpe IS↔OOS)** | Idem sur Sharpe | **−0.247** | ⚠️ Inversion partielle |
| **PBO approché** | 1 − Pr(top10% IS au-dessus médiane OOS) | **0%** | ✅ Aucun overfit majeur |
| **Configs robustes CAGR** | Top quartile IS *et* OOS | 147 / 718 (20.5%) | ✅ Plateau large |
| **Configs robustes Sharpe** | Idem | 41 / 718 (5.7%) | ⚠️ Pic isolé |

### 1.3 Score de plateau

Pour chaque cellule de la grille, score = mean des métriques sur les 8
voisins immédiats. Préfère un plateau stable plutôt qu'un pic isolé.

---

## 2. Résultats par objectif

### 2.1 Sélection sur **Sharpe OOS** (priorité robustesse live)

```
TOP-10 par Sharpe OOS (filtré : Sharpe IS ≥ 0.5)
─────────────────────────────────────────────────────
vt=0.10, vfloor=0.08, lev∈[8..80]:
  CAGR IS=1.82%, CAGR OOS=2.87%
  DD  IS=2.99%, DD  OOS=1.42%
  Sharpe IS=0.93, Sharpe OOS=+1.66
```

→ **`vol_floor=0.08` agit comme régularisation** : il limite le levier
effectif à `target_vol/vol_floor = 0.10/0.08 = 1.25×` (très défensif).
Sharpe OOS exceptionnel (1.66), mais CAGR de 2.87% peut sembler peu
ambitieux.

**Effet de saturation** : à vol_floor=0.08, max_lev>1.25 n'a aucun effet
(le vol_floor est binding). Toutes les configs lev∈[8,80] donnent le
même résultat.

### 2.2 Sélection sur **CAGR moyen IS/OOS** (objectif rendement)

```
TOP-10 par CAGR moyen IS/OOS (robust optima : top quartile IS ET OOS)
─────────────────────────────────────────────────────────────────────
vt=0.50, vfloor=0.04, lev∈[16..80]:
  CAGR_IS=10.21%, CAGR_OOS=4.63%
  DD_IS=9.31%,    DD_OOS=5.51%
  Sharpe_IS=1.17, Sharpe_OOS=0.67
```

→ **CAGR maximum robuste = 7.42% moyen** (10.21 IS, 4.63 OOS). Mais
**Sharpe OOS de seulement 0.67** — la dégradation IS→OOS est marquée
(−0.50 sur Sharpe).

Saturation à `vol_floor=0.04` : levier effectif = 0.50/0.04 = 12.5×, donc
`max_lev > 12.5` n'a aucun effet. Toutes les configs `lev∈[16,80]` sont
équivalentes.

### 2.3 Configurations robustes Sharpe (top-quartile IS ET OOS)

41 configs sur 718 — distribution :
- **target_vol** : 0.10 (n=21), 0.15 (n=20)
- **vol_floor** : 0.02 (n=20), 0.03 (n=20), 0.01 (n=1)

→ **Zone "balanced"** : `vt ∈ [0.10, 0.15]` × `vfloor ∈ [0.02, 0.03]`.
Robuste sur les deux fenêtres, CAGR modéré (~3-4%), Sharpe ~1.0.

---

## 3. Découvertes inattendues

### 3.1 Le `vol_floor` a un rôle de régularisation

Le scatter plot `is_oos_scatter_*.png` révèle un gradient clair en
`vol_floor` (couleur de 0.01 violet à 0.08 jaune) :

| `vol_floor` | Sharpe IS | Sharpe OOS | Effet |
|---|---|---|---|
| 0.01 (permissif) | élevé (~1.2) | médiocre (~0.6) | Overfit |
| 0.04 (modéré) | moyen (~1.1) | moyen (~0.7) | Compromis |
| 0.08 (contraignant) | faible (~0.7-1.0) | excellent (~1.7) | Régularisé |

Le `vol_floor` plus élevé **brise** la relation IS↔OOS : il limite le
levier effectif et donc l'amplitude des prises de risque qui mèneraient à
un overfit.

### 3.2 Spearman Sharpe négatif (−0.247) — explication

Les configs maximisant le Sharpe IS sont à `vt=0.30-0.50` + `vfloor=0.01-0.02`
(beaucoup de levier, beaucoup de profit IS). Mais ces mêmes configs
**dégradent** en OOS car le régime 2024-2026 (consolidation EUR/USD,
divergence ECB/BoJ) ne récompense plus le levier élevé. À l'inverse, les
configs défensives (`vt=0.10`, `vfloor=0.08`) ont un Sharpe IS modeste mais
**captent les opportunités OOS** où l'edge net est plus faible.

### 3.3 Saturation du `max_lev`

Pour `vfloor=0.04` et `vt=0.50`, le levier effectif maximal possible est
`0.50/0.04 = 12.5`. Augmenter `max_lev` au-delà de 13 ne change rien.
**80 configs `lev∈[16, 80]` produisent le même résultat**, ce qui
explique pourquoi le top-10 est plein de doublons.

→ Le levier broker (jusqu'à 80) **n'est pas le facteur limitant** — c'est
le `vol_floor` qui définit le levier max effectif. **Le défaut compilé
`max_lev=12` est déjà suffisant** pour saturer le système avec
`vfloor=0.04`.

---

## 4. Recommandations

### 4.1 Pour la production (priorité : Sharpe live)

**Conserver ou réduire** les défauts compilés :
- `Inp_GlobalTargetVol` : `0.10` à `0.15` (vs défaut 0.28) — défensif
- `Inp_GlobalMaxLeverage` : `12` (déjà OK, saturation)
- `Inp_GlobalVolFloor` : **`0.08`** (vs défaut 0.02) — régularisation

CAGR attendu : 2-4% / an. Sharpe live cible : 1.0-1.5.

### 4.2 Pour ambition CAGR (acceptant overfit modéré)

- `Inp_GlobalTargetVol` : `0.30` à `0.50`
- `Inp_GlobalMaxLeverage` : `12` (saturation)
- `Inp_GlobalVolFloor` : `0.04`

CAGR attendu : 5-8% / an. Sharpe live cible : 0.7-1.0. **Risque OOS
significatif** (CAGR a chuté de 10% IS à 4.6% OOS sur la période testée).

### 4.3 Conserver le défaut compilé (statu quo informé)

`vt=0.28, lev=12, vfloor=0.02` est un **compromis acceptable** :
- CAGR 5.4 ans validé baseline = +6.13% / an
- Sharpe 1.02 (in-sample sur la fenêtre complète)
- Cohérent avec la philosophie "diversificateur multi-horizon"

**Le défaut actuel est valide en l'absence d'études OOS plus larges**
(plus de fenêtres glissantes nécessaires pour confirmer la stabilité du
compromis).

---

## 5. Limitations résiduelles

1. **Une seule paire OOS** : la fenêtre OOS 2024-11→2026-04 est
   spécifique. Une cassée par walk-forward N=5 fenêtres glissantes
   donnerait plus de confiance.
2. **Modèle 1-min OHLC** : pas de ticks réels. Les coûts de transaction
   sont sous-estimés (cf. investigation RSI Daily).
3. **Le levier broker à 80** n'est pas exploitable dans cette
   configuration — saturation `vfloor`-driven.
4. **Le DD-cap (`Inp_DDCap=0.15`) n'a pas été varié**. Pourrait faire
   l'objet d'un sweep 4D ultérieur.

---

## 6. Données et scripts produits

| Fichier | Rôle |
|---|---|
| `scripts/optimization/run_mt5_optimization.py` | Wrapper natif MT5 (32 cores) — modifié pour 3D |
| `scripts/optimization/walkforward_3d.py` | Orchestrateur full + IS + OOS + analyse |
| `reports/optimization/wf_full_*.csv|png` | Sweep 720 combos sur fenêtre 5.4 ans |
| `reports/optimization/wf_is_*.csv|png` | Sweep 720 combos sur IS (4 ans) |
| `reports/optimization/wf_oos_*.csv|png` | Sweep 720 combos sur OOS (1.5 ans) |
| `reports/optimization/walkforward_3d/merged_is_oos_*.csv` | Join IS↔OOS |
| `reports/optimization/walkforward_3d/robust_optima_*.csv` | 147 configs robustes |
| `reports/optimization/walkforward_3d/is_oos_scatter_*.png` | Visualisation Spearman |

## 7. Reproduction

```bash
# Walk-forward complet (full + IS + OOS + analyse) — ~10-15 min
python scripts/optimization/walkforward_3d.py

# Single sweep custom
python scripts/optimization/run_mt5_optimization.py \\
    --vt-start 0.10 --vt-stop 0.50 --vt-step 0.05 \\
    --lev-start 8 --lev-stop 80 --lev-step 8 \\
    --vfloor-grid 0.01,0.02,0.04,0.08 \\
    --from-date 2020.11.23 --to-date 2026.04.30
```

## 8. Pistes de prolongation

- **Sweep 4D** : ajouter `Inp_DDCap` (0.10, 0.15, 0.20, 0.25)
- **Walk-forward N=5** : 5 fenêtres glissantes de 1 an OOS chacune
- **Re-runner avec model=4** (ticks réels) pour l'optimum candidat
- **PSR (Probabilistic Sharpe Ratio)** au lieu du Sharpe nominal pour
  ajustement bruit d'estimation
- **Cross-validation** : k-fold purged temporal pour score moyen
