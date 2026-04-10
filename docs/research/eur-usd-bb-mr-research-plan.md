# Recherche FX — Journal de bord

**Date:** 2026-04-09
**Auteur:** Thomas / Claude

---

## Résumé exécutif

**Meilleure stratégie validée :** 1min BB(80,5.0) + macro filter (spread<0.5 + chômage stable) + VWAP anchor ~22h UTC.

| Configuration | Sharpe WF | Pos/7 | OOS 2025 | Statut |
|---------------|-----------|-------|----------|--------|
| 1min BB + macro + VWAP anchor 22h | **1.12** | **5/7** | **1.08** | **VALIDEE** |
| 1min BB + macro (baseline) | 0.94 | 4/7 | 0.77 | VALIDEE |
| 1min BB + macro + excl. Lun/Mar | 1.13 | 4/7 | 1.15 | VALIDEE |
| ~~1H BB(6,4.0) sans macro~~ | ~~3.10~~ | ~~7/7~~ | ~~2.81~~ | **INVALIDEE (look-ahead)** |

> **AVERTISSEMENT CRITIQUE :** Les résultats multi-timeframe (1H BB) initialement rapportés
> contenaient un **look-ahead bias** confirmé. Le bar horaire resampleé à `08:00` contient
> le close de `08:59`, rendant les bandes BB disponibles 59 minutes avant que les données
> soient réellement observables. Après correction (shift +1 période horaire), TOUTES les
> configurations 1H sont négatives (Sharpe -1.48). Voir Phase 2A-ERRATA ci-dessous.

**Leçons retenues :**
1. Le filtre macro reste l'alpha dominant pour le signal 1min
2. L'ancrage VWAP à 22h UTC (session FX) améliore la robustesse (+5ème année positive)
3. Les stratégies multi-timeframe avec resample+ffill doivent TOUJOURS être vérifiées pour le look-ahead
4. Un Sharpe > 2-3 en backtest FX intraday doit déclencher une investigation systématique

---

## Phase 0 : Résultats antérieurs (référence)

### Stratégie initiale : 1min BB(80,5.0) + macro filter

**Configuration :** BB sur déviation close-VWAP, session 6-14 UTC, SL=0.5%, TP=0.6%, td_stop=6h

| Config | Avg SR | Pos/7 | OOS 2025 | Trades | Profil |
|--------|--------|-------|----------|--------|--------|
| sp<0.5 + unemp_ok, td=6h | 0.94 | 4/7 | 0.77 | 103 | Meilleur Sharpe |
| sp<0.5 + unemp_ok, td=4h | 0.91 | 5/7 | 0.68 | 103 | Plus robuste |
| unemp_ok + fed_active | 0.74 | 6/7 | 0.76 | 103 | Le plus stable |
| No filter | 0.08 | 2/7 | -0.66 | 279 | Quasi-random |

**Insight clé Phase 0 :** Le filtre macro représente >90% de l'alpha pour le signal 1-minute.

### Tests qui ont échoué (Phase 0)

| Test | Résultat |
|------|----------|
| ATR dynamic SL/TP | Aucun impact |
| BBANDS %B filter | Tue tous les trades |
| Trailing stop | Pire que TP fixe |
| VWAP exit / Z-score exit | Pire que TP fixe |
| RSI filter sur 1min | Dégrade (trop de bruit) |
| SuperTrend, MACD, Stoch, IMS | Toutes négatives sur FX |

---

## Phase 1 : Macro Deep-Dive

### 1E. Variables macro individuelles

**Hypothèse :** D'autres variables macro ont un pouvoir prédictif sur le MR.

**Protocole :** Pour chaque variable, construire un filtre "< rolling median" et ">= rolling median", walk-forward 7 périodes.

**Résultats (top 5 et bottom 3) :**

| Variable & Direction | Sharpe | Pos/7 | OOS |
|---------------------|--------|-------|-----|
| core_cpi_yoy >= median | 0.47 | 3/7 | 0.88 |
| pce_yoy >= median | 0.47 | 3/7 | -0.40 |
| cpi_yoy >= median | 0.41 | 3/7 | -0.90 |
| dgs2_mom20 < median | 0.26 | 2/7 | -0.21 |
| fed_funds < median | 0.23 | 3/7 | 0.15 |
| ... | | | |
| pce_yoy < median | -0.23 | 2/7 | -0.58 |
| cpi_yoy < median | -0.19 | 2/7 | -0.16 |
| **fed_funds >= median** | **-0.54** | **1/7** | **-2.72** |

**Conclusion :** L'inflation élevée (core CPI, PCE >= médiane) est le meilleur signal individuel (Sharpe 0.47). Les taux élevés (fed_funds >= median) sont destructeurs (-0.54).

**Rationnel économique :** En environnement inflationniste, les banques centrales sont actives, créant de la volatilité directionnelle qui se traduit par des déviations de VWAP plus marquées et plus mean-revertantes. Les taux élevés induisent un carry trade fort qui crée du momentum (anti-MR).

### 1A. Seuils dynamiques (rolling percentile)

**Hypothèse :** `spread < 0.5` est un seuil fixe sur une variable dont le range est [-0.5, 2.5]. Un percentile roulant s'adapterait au régime courant.

**Résultat : REJETÉ.** Meilleur config : Sharpe 0.23 (vs baseline 0.94). Le rolling percentile produit trop peu de trades (4-31 vs 130) et perd la signification économique du seuil fixe.

**Conclusion :** Le seuil `spread < 0.5` capture un régime économique spécifique (courbe de taux plate/inversée). Le percentile roulant dilue ce signal en le rendant relatif.

### 1B. Score macro composite continu

**Hypothèse :** Un score pondéré z-scoré de toutes les variables macro est plus informatif que deux filtres binaires.

**Résultat : REJETÉ.** Meilleur config (`balanced q<40`): Sharpe 0.38, 4/7 positif. Le z-scoring détruit les seuils absolus économiquement significatifs.

**Conclusion :** Les filtres binaires avec seuils fixes ont une interprétation économique que les scores continus normalisés perdent. La courbe de taux inversée n'est pas la même chose que "spread dans son 30ème percentile".

### 1C. Lead/lag macro

**Hypothèse :** Les macro ont un effet retardé (publication avec délai, diffusion lente).

**Résultat : PARTIELLEMENT CONFIRMÉ.** `BEST_COMBO lag=20d` atteint Sharpe 0.38, 4/7 positif. Le signal persiste ~20 jours mais le lag dégrade la performance vs lag=0.

**Conclusion :** Le signal macro est le plus fort en temps réel. Le lag de 20j montre une persistance mais pas une amélioration.

### 1D. Clustering K-means

**Hypothèse :** Les régimes naturels dans les données macro ne correspondent pas aux seuils manuels.

**Résultat : REJETÉ.** Meilleur k=4 : Sharpe 0.21. Le filtre est trop lâche (65-70% des jours actifs) et le clustering non-supervisé ne maximise pas le signal stratégie.

### 1H. Filtres hybrides (base + inflation)

**Hypothèse :** Combiner le filtre existant avec le signal inflation via OR logique enrichit la couverture.

**Résultats :**

| Combo | Sharpe | Pos/7 | OOS |
|-------|--------|-------|-----|
| sp03+unemp OR core_cpi>=med | 0.94 | 4/7 | 1.11 |
| sp05+unemp OR core_cpi>=med | 0.90 | 4/7 | 0.86 |
| BASELINE sp05+unemp | 0.94 | 4/7 | 0.77 |
| sp05+unemp AND core_cpi>3% | 0.70 | 3/7 | 0.76 |

**Conclusion :** Le OR avec inflation ne dégrade pas le Sharpe et améliore l'OOS (1.11 vs 0.77). Cependant, l'amélioration est marginale. Le filtre macro original reste le plus simple et le plus robuste pour le signal 1-minute.

### Synthèse Phase 1

**Le filtre macro (spread<0.5 + chômage stable) est remarquablement difficile à battre.** Aucune des 100+ configurations testées ne le surpasse de manière significative. Sa force vient de sa simplicité et de sa signification économique directe.

Le filtre macro restera essentiel tant qu'une amélioration de signal est trouvée à la bonne fréquence.

---

## Phase 2 : Multi-Timeframe BB

### 2A. BB sur timeframe supérieur — **INVALIDEE (LOOK-AHEAD BIAS)**

> **ERRATA (2026-04-09) :** Les résultats ci-dessous contenaient un look-ahead bias fatal.
> Ils sont conservés pour documenter l'erreur et la méthodologie de détection.

**Hypothèse initiale :** Le BB sur 1-minute est dominé par le bruit. Calculer les BB sur 1H resampleé devrait améliorer le signal/bruit.

**Résultats initiaux (INVALIDES) :**

| Config | Sharpe | Note |
|--------|--------|------|
| ~~1H BB(4,3.0)~~ | ~~3.14~~ | INVALIDE |
| ~~1H BB(6,4.0)~~ | ~~3.10~~ | INVALIDE |
| BASELINE 1min | 0.94 | VALIDE |

**Investigation critique — Détection du look-ahead :**

1. **Observation suspecte :** Sharpe > 3 sur toutes les paires, toutes les années, même sans macro filter. C'est irréaliste pour du FX intraday.

2. **Analyse des exits :** 78.6% des trades sortent par time-stop (6h), seulement 17.8% par TP. La stratégie profite du drift 6h, pas de la convergence rapide.

3. **Test VWAP hourly vs minute :** Le VWAP approximatif horaire ne cause PAS de biais (Sharpe 3.05 vs 3.10). Fausse piste.

4. **Test décisif — shift des bandes :** En décalant les BB de +1h (pour n'utiliser que des données complétées), le Sharpe passe de **3.10 à -1.48**.

5. **Cause racine confirmée :** `close.resample('1h').last()` produit un bar à `08:00` contenant le close de `08:59`. Le forward-fill rend ce bar disponible à 08:00, soit **59 minutes avant que les données soient observables**.

```
Bar horaire '08:00' = close[08:59]  ← donnée future !
Forward-fill: disponible à 08:00    ← 59 min trop tôt
BB bands utilisant ce close → entrées à 08:01 basées sur le futur
```

**Résultats après correction (shift +1 période horaire) :**

| Config | Sharpe AVANT | Sharpe APRES correction | Trades |
|--------|-------------|------------------------|--------|
| 1H BB(4,3.0) | 3.14 | **-2.22** | 3582 |
| 1H BB(6,4.0) | 3.10 | **-1.48** | 1632 |
| 1H BB(8,5.0) | 1.32 | **-1.15** | 768 |
| 1H BB(12,5.0) | 1.40 | **-0.97** | 500 |

**TOUTES les configurations sont négatives après correction.** L'alpha multi-TF était entièrement un artefact du look-ahead.

**Leçon méthodologique :**
- Les stratégies multi-TF avec `resample().last() + reindex(ffill)` introduisent TOUJOURS un look-ahead d'une période.
- Le fix est `.shift(1)` sur les données resamplees AVANT le forward-fill.
- Un Sharpe > 2-3 en FX intraday doit systématiquement déclencher une vérification de look-ahead.
- Le test de détection : décaler les signaux d'une période et vérifier que la performance ne s'effondre pas.

### Session test (INVALIDE — conservé pour référence)

| Session | Sharpe | Trades |
|---------|--------|--------|
| 24/7 | 7.81 | 5152 |
| 0-14 | 6.85 | 3165 |
| 6-18 | 6.10 | 2400 |
| 8-16 | 5.92 | 1693 |
| 6-14 | 5.44 | 1720 |

La session 6-14 reste la plus conservative mais toutes les sessions sont profitables.

### 2B. RSI/Stochastic sur 1H comme confirmation

**Hypothèse :** RSI sur 1H pourrait confirmer les entrées (oversold pour longs).

**Résultat : REJETÉ.** Meilleur : RSI(14) 40/60 = Sharpe 0.31. Le filtre RSI tue trop de trades. Même pattern que RSI sur minute — les filtres de confirmation dégradent le MR.

**Conclusion :** Le BB seul est le meilleur signal d'entrée pour le MR. Les confirmations additionnelles réduisent le nombre de trades sans améliorer la qualité.

---

## Phase 3 : Timing & Qualité d'Entrée

### 3A. Jour de la semaine

**Résultats par jour (avec macro filter, 1min BB) :**

| Jour | Sharpe | Pos/7 | Trades |
|------|--------|-------|--------|
| Thu | 0.79 | 5/7 | 34 |
| Wed | 0.60 | 5/7 | 25 |
| Fri | 0.33 | 2/7 | 33 |
| Tue | 0.04 | 2/7 | 23 |
| Mon | 0.02 | 3/7 | 15 |

**Exclure Lun+Mar :** Sharpe 1.13, 4/7 positif, OOS 1.15.

**Rationnel économique :** Mercredi et jeudi concentrent les publications macro (FOMC, ECB, NFP) qui créent des déviations plus marquées puis des reversions. Lundi est un jour de continuation (digestion du weekend). Mardi manque de catalyseur.

### 3B. Ancrage VWAP — **AMÉLIORATION CONFIRMÉE**

**Hypothèse :** Le VWAP ancré à minuit UTC est arbitraire. Le marché FX recommence à 17h ET = 22h UTC.

| Anchor | Sharpe | Pos/7 | OOS |
|--------|--------|-------|-----|
| ~19:00 UTC (shift +5h) | **1.12** | **5/7** | **1.28** |
| ~22:00 UTC (shift +2h) | 1.12 | 5/7 | 1.08 |
| ~21:00 UTC (shift +3h) | 1.09 | 5/7 | 1.08 |
| Minuit (baseline) | 0.94 | 4/7 | 0.77 |

**Résultat : VALIDÉ.** L'anchor ~19-22h UTC améliore le Sharpe de 0.94 à 1.12 et gagne un 5ème année positive. Le VWAP ancré à la session FX réelle est plus pertinent.

**Conclusion :** L'amélioration est modérée (+0.18 Sharpe) mais robuste (5/7 vs 4/7). Validé par re-test indépendant.

### 3C. Filtre de vélocité

**Hypothèse :** Entrer seulement quand la déviation décélère (pas de couteau qui tombe).

**Résultat : NON CONCLUANT.** Meilleur config (lb=20, thr=0.0002) : Sharpe 0.90 vs 0.94 baseline. Le filtre réduit les trades sans améliorer la qualité.

---

## Phase 4 : Portefeuille Multi-Pair — **INVALIDEE**

> Les résultats de cette phase étaient basés sur le 1H BB invalidé (look-ahead).
> Le portefeuille multi-pair utilisant le signal 1min + macro est encore à tester.

---

## Phase 5 : Validation de Robustesse — **PARTIELLEMENT INVALIDEE**

### 5A. Monte Carlo Bootstrap — **INVALIDE** (basé sur 1H BB)

> Le bootstrap était basé sur les trades du 1H BB invalidé. Non applicable.

**P(Sharpe > 0) = 100%, P(Sharpe > 1) = 100%, P(Sharpe > 2) = 100%.**

Le 5ème percentile du Sharpe (2.72) est largement supérieur à zéro. Cela confirme que l'alpha n'est pas un artefact statistique.

---

## Synthèse & Leçons apprises

### Découvertes validées

1. **Le filtre macro (spread<0.5 + chômage stable) est l'alpha.** Pour le signal 1-minute, il représente >90% du Sharpe. Aucune des 100+ configurations macro testées ne le surpasse significativement.

2. **L'ancrage VWAP à la session FX (19-22h UTC) améliore la robustesse** du signal 1min : Sharpe 0.94 → 1.12, 4/7 → 5/7 années positives. Validé sans look-ahead.

3. **Mercredi et jeudi sont les meilleurs jours** pour le MR intraday (Sharpe 0.60-0.79 vs 0.02-0.04 pour Lun/Mar). Exclure Lun/Mar donne Sharpe 1.13.

4. **L'inflation élevée (core CPI >= médiane) est le meilleur signal macro individuel** (Sharpe 0.47). Combiné en OR avec le filtre base, il améliore l'OOS (1.11 vs 0.77).

5. **Les filtres de confirmation (RSI, velocity) dégradent systématiquement le MR.** Le BB seul est optimal.

### Erreur majeure et leçon méthodologique

**Les stratégies multi-TF avec resample+ffill avaient un look-ahead bias fatal.** Le bar horaire `08:00` contient le close de `08:59`, rendu disponible à `08:00` par le forward-fill. Après correction, toutes les configs 1H sont négatives.

**Checklist anti-look-ahead pour le multi-TF :**
- [ ] Le resample utilise-t-il `.shift(1)` avant le forward-fill ?
- [ ] Le test shift +1 période donne-t-il un Sharpe comparable ?
- [ ] Le Sharpe est-il < 2 (seuil de plausibilité pour FX intraday) ?
- [ ] La stratégie sans filtre donne-t-elle des résultats raisonnables ?

### Ce qui n'a pas fonctionné

- **Multi-TF BB (1H, 15min, 5min)** — INVALIDE (look-ahead bias)
- Seuils dynamiques (rolling percentile) — perd la signification économique
- Score composite z-scoré — détruit les seuils absolus
- Clustering K-means — trop lâche, pas optimisé pour la stratégie
- Lead/lag macro — le signal est meilleur en temps réel
- RSI/Stoch confirmation — tue les trades sur toutes les timeframes
- Filtre de vélocité — pas d'amélioration

### Risques et limitations de la stratégie validée

1. **Sharpe modeste (~1.0)** pour une stratégie avec ~130 trades/7 ans. L'edge est petit mais réel.
2. **Dépendance au régime macro :** le filtre spread+chômage peut ne pas capturer le prochain régime favorable.
3. **Faible nb de trades :** ~18/an, rendant les statistiques fragiles.
4. **Pas de multi-pair validé :** seul EUR-USD fonctionne avec le signal 1min + macro.

---

## Prochaines étapes

1. **Monte Carlo bootstrap sur stratégie 1min validée** — intervalle de confiance
2. **Multi-pair avec signal 1min + macro pair-spécifique** — tester d'autres paires
3. **Implémentation QuantConnect** — stratégie 1min + macro + VWAP anchor
4. **Slippage sensitivity sur stratégie validée** — confirmer marge de sécurité
5. **Exploration multi-TF CORRECTE** — utiliser `.shift(1)` sur les bandes resamplees

## Protocole de test utilisé

Pour chaque hypothèse :
1. Walk-forward sur 7 périodes (2019-2025), jamais de backtest full-sample pour la sélection
2. Métriques : Sharpe moyen, nombre d'années positives (/7), OOS 2025, nombre de trades
3. Seuil de validation : Sharpe avg > 0.5, >= 4/7 positif, OOS > 0
4. Vérification look-ahead obligatoire pour tout signal multi-timeframe
5. Documentation immédiate avec rationnel économique
