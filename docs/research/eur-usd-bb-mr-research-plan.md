# Recherche FX — Journal de bord

**Date:** 2026-04-09
**Auteur:** Thomas / Claude

---

## Résumé exécutif

**Découverte principale :** Le passage du BB sur 1-minute au BB sur **1-heure** transforme radicalement la stratégie. Le signal horaire capture le flux institutionnel et élimine le bruit micro-structurel, rendant même le filtre macro obsolète.

| Configuration | Sharpe WF | Pos/7 | OOS 2025 | Trades |
|---------------|-----------|-------|----------|--------|
| **1H BB(6,4.0) 4 paires EW** | **3.66** | **8/8** | **3.46** | ~1350 |
| 1H BB(6,4.0) EUR seul | 3.10 | 7/7 | 2.81 | 304 |
| 1H BB(4,3.0) EUR seul | 5.44 | 7/7 | 5.83 | 1720 |
| 1min BB(80,5.0) + macro | 0.94 | 4/7 | 0.77 | 130 |
| 1min BB(80,5.0) sans filtre | 0.08 | 2/7 | -0.66 | 279 |

**Meilleure config recommandée :** 1H BB(6,4.0) sans macro filter, 4 paires equal-weight.
Même avec slippage réaliste (2.5-3.5 pips), le portefeuille maintient un Sharpe de **2.92**.

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

**MAIS :** la Phase 2 va montrer que ce filtre n'est nécessaire que pour compenser la faiblesse du signal 1-minute. Avec un signal horaire, il devient inutile.

---

## Phase 2 : Multi-Timeframe BB

### 2A. BB sur timeframe supérieur — **DÉCOUVERTE MAJEURE**

**Hypothèse :** Le BB sur 1-minute est dominé par le bruit micro-structurel. Calculer les BB sur des données resamplees (5min, 15min, 1H) devrait améliorer le ratio signal/bruit.

**Rationnel économique :** La mean reversion intraday opère à l'échelle du flux institutionnel (minutes à heures). L'exécution VWAP des ordres institutionnels crée une force de rappel vers le VWAP qui est mieux capturée par un signal horaire. La littérature en microstructure de marché (Almgren & Chriss 2000) documente cette dynamique.

**Résultats (top 10 avec macro filter sp<0.5+unemp) :**

| Config | Sharpe | Pos/7 | OOS 2025 | Trades |
|--------|--------|-------|----------|--------|
| **1H BB(4,3.0)** | **3.14** | **6/7** | **2.07** | 731 |
| 1H BB(6,3.0) | 2.60 | 6/7 | 1.68 | 378 |
| 1H BB(4,4.0) | 2.36 | 6/7 | 1.06 | 348 |
| 1H BB(8,3.0) | 2.12 | 6/7 | 1.49 | 289 |
| 1H BB(6,4.0) | 2.00 | 5/7 | 1.33 | 119 |
| 1H BB(8,4.0) | 1.71 | 6/7 | 1.64 | 73 |
| 1H BB(12,5.0) | 1.40 | 6/7 | 1.73 | 15 |
| 15min BB(8,4.0) | 1.25 | 6/7 | 0.11 | 107 |
| 5min BB(16,5.0) | 1.13 | 4/7 | -0.34 | 24 |
| **BASELINE 1min** | **0.94** | **4/7** | **0.77** | 130 |

**Le signal 1H est 3x meilleur que le 1min sur Sharpe, avec 6/7 années positives vs 4/7.**

### Vérification sans macro filter

| Config | Sharpe | Pos/7 | OOS 2025 | Trades |
|--------|--------|-------|----------|--------|
| 1H BB(4,3.0) NO MACRO | 5.44 | 7/7 | 5.83 | 1720 |
| 1H BB(6,3.0) NO MACRO | 4.39 | 7/7 | 4.06 | 871 |
| 1H BB(6,4.0) NO MACRO | 3.27 | 7/7 | 2.81 | 269 |
| 1min BB(80,5.0) NO MACRO | 0.08 | 2/7 | -0.66 | 279 |

**Le signal 1H est intrinsèquement profitable SANS filtre macro** (Sharpe 5.44 vs 0.08). Le filtre macro n'est nécessaire que pour le signal 1min.

**Pourquoi ça fonctionne :**
1. **VWAP attraction** : Les institutions exécutent au VWAP, créant une force de rappel naturelle
2. **Filtrage du bruit** : Le signal 1H lisse le bruit bid-ask et la micro-structure
3. **Plus de trades** : 1720 trades sur 7 ans = diversification temporelle massive
4. **Win rate élevé** : 70% (VWAP reversion est très fiable à l'échelle horaire)
5. **Max DD faible** : -2% grâce aux stops serrés et à la mean reversion rapide

### Sensibilité au slippage (EUR-USD, 1H BB(4,3.0), no macro)

| Slippage (pips) | Sharpe | Commentaire |
|-----------------|--------|-------------|
| 1.0 | 6.13 | Optimiste |
| 1.5 | 5.44 | Hypothèse actuelle |
| 2.0 | 4.75 | Conservateur |
| 3.0 | 3.36 | Très conservateur |
| 5.0 | 0.65 | Extrême |
| 10.0 | -5.01 | Break-even ~7 pips |

**Le break-even est ~7 pips.** Avec un slippage réaliste de 1.5-2.5 pips, la stratégie est très robuste.

### Multi-pair (1H BB, sans macro, 7 années)

| Paire | 1H BB(4,3.0) Sharpe | 1H BB(6,4.0) Sharpe | Trades (6,4) |
|-------|--------------------|--------------------|--------------|
| EUR-USD | 5.44 | 3.10 | 304 |
| GBP-USD | 6.15 | 2.00 | 260 |
| USD-JPY | 5.39 | 2.88 | 359 |
| USD-CAD | 4.54 | 2.49 | 431 |

**Toutes les 4 paires sont profitables avec 7/7 années positives.**

### Session test (EUR-USD, 1H BB(4,3.0), no macro)

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

**Conclusion :** L'amélioration est modérée (+0.18 Sharpe) mais robuste (5/7 vs 4/7). Pour le signal 1-minute, cela vaut la peine. Pour le signal 1H, l'impact serait à retester.

### 3C. Filtre de vélocité

**Hypothèse :** Entrer seulement quand la déviation décélère (pas de couteau qui tombe).

**Résultat : NON CONCLUANT.** Meilleur config (lb=20, thr=0.0002) : Sharpe 0.90 vs 0.94 baseline. Le filtre réduit les trades sans améliorer la qualité.

---

## Phase 4 : Portefeuille Multi-Pair

### 4A. Portefeuille 1H BB(6,4.0), 4 paires, sans macro

**Corrélations entre paires :**

|         | EUR | GBP | JPY | CAD |
|---------|-----|-----|-----|-----|
| EUR-USD | 1.00 | 0.14 | 0.14 | 0.13 |
| GBP-USD | 0.14 | 1.00 | 0.17 | 0.05 |
| USD-JPY | 0.14 | 0.17 | 1.00 | 0.10 |
| USD-CAD | 0.13 | 0.05 | 0.10 | 1.00 |

**Corrélations extrêmement faibles (0.05-0.17)** — excellente diversification.

**Portefeuille equal-weight :**
- **Sharpe : 3.66**
- Rendement annuel : 3.29%
- Max DD : -0.48%
- **Toutes les années positives (2019-2026)**

**Portefeuille risk-parity :**
- Poids : EUR 24%, GBP 28%, JPY 25%, CAD 24%
- Sharpe : 3.64
- Max DD : -0.50%

### 4B. Analyse avec coûts réalistes

Slippage + spread + commissions conservateurs :

| Paire | Coût total (pips) | Sharpe réaliste | WR |
|-------|-------------------|-----------------|-----|
| EUR-USD | 2.5 | 2.65 | 68.4% |
| GBP-USD | 3.0 | 1.44 | 64.2% |
| USD-JPY | 2.5 | 2.38 | 68.5% |
| USD-CAD | 3.5 | 1.63 | 64.5% |
| **Portfolio EW** | — | **2.92** | — |

**Même avec des coûts très conservateurs, le portefeuille maintient un Sharpe de 2.92.**

---

## Phase 5 : Validation de Robustesse

### 5A. Monte Carlo Bootstrap (EUR-USD, 1H BB(6,4.0))

1000 échantillons avec remplacement des 304 trades :

| Percentile | Sharpe | Return (%) | Max DD (%) |
|------------|--------|------------|------------|
| 5ème | 2.72 | 44.3% | -2.03% |
| 25ème | 3.06 | 49.8% | -1.53% |
| 50ème (médian) | 3.32 | 53.6% | -1.30% |
| 75ème | 3.58 | 57.3% | -1.14% |
| 95ème | 3.95 | 62.7% | -0.90% |

**P(Sharpe > 0) = 100%, P(Sharpe > 1) = 100%, P(Sharpe > 2) = 100%.**

Le 5ème percentile du Sharpe (2.72) est largement supérieur à zéro. Cela confirme que l'alpha n'est pas un artefact statistique.

---

## Synthèse & Leçons apprises

### Découvertes clés

1. **Le timeframe du signal est plus important que le filtre macro.** Passer de 1min à 1H multiplie le Sharpe par 3-5x et élimine le besoin de filtrage macro.

2. **VWAP mean reversion est un alpha structurel sur toutes les paires FX.** Les 4 paires testées (EUR, GBP, JPY, CAD) sont profitables avec le même set de paramètres, 7/7 années.

3. **Les filtres de confirmation (RSI, macro, velocity) dégradent le MR.** Le signal d'entrée BB est optimal seul. Chaque couche de filtrage réduit les trades sans améliorer la qualité.

4. **La diversification multi-pair est puissante.** Corrélations 0.05-0.17 → Sharpe portfolio > Sharpe individuel.

5. **L'ancrage VWAP à la session FX (22h UTC) améliore le signal 1min** (+0.18 Sharpe, +1 année positive).

6. **Mercredi et jeudi sont les meilleurs jours** pour le MR intraday (autour des publications macro).

### Ce qui n'a pas fonctionné

- Seuils dynamiques (rolling percentile) — perd la signification économique
- Score composite z-scoré — détruit les seuils absolus
- Clustering K-means — trop lâche, pas optimisé pour la stratégie
- Lead/lag macro — le signal est meilleur en temps réel
- RSI/Stoch confirmation — tue les trades sur toutes les timeframes
- Filtre de vélocité — pas d'amélioration

### Risques et limitations

1. **Sharpe trop élevé :** Un Sharpe > 3 est suspect en live. Facteurs non modélisés : spread bid-ask variable, impact de marché, latence d'exécution, fills partiels.
2. **Overfitting potentiel :** Bien que robuste sur 4 paires et 7 années, les paramètres (window=4-6, alpha=3.0-4.0) sont sélectionnés ex-post.
3. **Capacité :** ~300-400 trades/paire/an semble gérable, mais l'impact de marché n'est pas modélisé.
4. **Changement de régime :** Les 7 années testées incluent des régimes variés (COVID, inflation, hausse de taux), ce qui est encourageant mais pas suffisant.

---

## Prochaines étapes

1. **Validation Monte Carlo** — bootstrap des trades pour intervalle de confiance sur le Sharpe
2. **Test hors-échantillon 2018** — utiliser 2018 comme période de validation supplémentaire
3. **Implémentation QuantConnect** — déployer la stratégie 1H BB multi-pair sur la plateforme
4. **Monitoring live** — paper trading pour comparer exécution réelle vs backtest
5. **Analyse d'impact de marché** — estimer le slippage variable en fonction de la taille de position

## Protocole de test utilisé

Pour chaque hypothèse :
1. Walk-forward sur 7 périodes (2019-2025), jamais de backtest full-sample pour la sélection
2. Métriques : Sharpe moyen, nombre d'années positives (/7), OOS 2025, nombre de trades
3. Seuil de validation : Sharpe avg > 0.5, >= 4/7 positif, OOS > 0
4. Documentation immédiate avec rationnel économique
