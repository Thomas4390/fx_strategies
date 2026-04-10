# EUR-USD BB MR — Recherche approfondie

**Date:** 2026-04-09
**Stratégie optimisée:** BB(80, 5.0) sur déviation close-VWAP, session 6-14 UTC, SL=0.5%, TP=0.6%, td_stop=6h, macro filter (spread<0.5 + unemp stable)
**Performance walk-forward (7 périodes 2019-2025):** Sharpe 0.94, 4/7 positif, OOS 2025 +0.77

## 0. Résultats finaux des tests

### Meilleures configurations (walk-forward 7 périodes)

| Config | Avg SR | Pos/7 | OOS 2025 | Trades | Profil |
|--------|--------|-------|----------|--------|--------|
| **sp<0.5 + unemp_ok, td=6h** | **0.94** | 4/7 | 0.77 | 103 | Meilleur Sharpe |
| **sp<0.5 + unemp_ok, td=4h** | **0.91** | **5/7** | 0.68 | 103 | Plus robuste |
| **unemp_ok + fed_active** | 0.74 | **6/7** | 0.76 | 103 | Le plus stable |
| sp<0.3 + unemp_ok, td=6h | 0.93 | 4/7 | 0.69 | 84 | Plus strict |
| spread < 0.3 seul | 0.61 | 4/7 | 1.16 | 147 | Simple, bon OOS |

### Tests Phase A : paramètres non-macro

| Test | Résultat |
|------|----------|
| **ATR dynamic SL/TP** | Aucun impact (trades sortent via TP/EOD avant SL) |
| **BBANDS bandwidth filter** | Non applicable (bandwidth négatif sur déviation) |
| **%B filter** | Tue tous les trades |
| **td_stop=6h** | **+0.03 Sharpe vs 4h** (meilleur holding period) |
| **dt_stop variations** | Aucun impact (trades sortent avant 18:00) |
| **Sessions 5-13 à 10-16** | 6-14 optimal, 6-12 quasi-identique |

### Tests Phase B : variables macro individuelles

| Variable macro | Sharpe seul | Impact |
|----------------|------------|--------|
| **spread < 0.3** | 0.61 | Fort positif |
| **unemp stable** | 0.60 | Fort positif |
| **fed not on hold** | 0.41 | Positif |
| cpi_yoy > 3% | 0.37 | Modéré |
| spread < 0.5 | 0.33 | Modéré |
| dgs10 falling | -0.06 | Aucun |
| dgs10 > 3% | 0.08 | Aucun |
| nfp rising | 0.21 | Faible |
| pce_yoy > 3% | 0.23 | Faible |
| No filter | 0.08 | Baseline |

### Insight clé : le filtre macro EST l'alpha
Sans filtre macro, la stratégie a un Sharpe de 0.08 (quasi-random). Avec filtre, 0.61-0.94. **Le filtre macro représente >90% de l'alpha.** La logique d'entrée BB/VWAP sélectionne les points d'entrée, mais c'est le filtre macro qui détermine SI on devrait trader du tout.

## 1. Ce qui a été testé

| Dimension | Valeurs testées | Meilleur |
|-----------|----------------|----------|
| BB Window | 30, 40, 60, 80, 100, 120 | 80 |
| BB Alpha | 3.0, 4.0, 5.0, 6.0, 7.0 | 5.0 |
| SL Stop | 0.003, 0.004, 0.005, 0.006, 0.007 | 0.005 |
| TP Stop | 0.004, 0.005, 0.006, 0.007, 0.008, 0.010, 0.015, 0.020 | 0.006 |
| Session UTC | (6,14), (7,15), (8,16) | (6,14) |
| Macro spread | 0.1, 0.2, 0.3, 0.4, 0.5, none | 0.5 |
| Macro unemp | on/off | on |
| Macro fed | on/off | off |
| dt_stop (EOD) | "21:00" seulement | — |
| td_stop (max hold) | "4h" seulement | — |
| Leverage | 1.0, 1.5, 2.0, 3.0 | 1.0 (Sharpe invariant) |
| Trailing stop | 0.3%, 0.4%, 0.5%, TTP combos | Pire que TP fixe |
| VWAP exit | Avec/sans | Pire que TP fixe |
| Z-score exit | 0.3, 0.5, 0.7, 1.0 | Pire que TP fixe |
| RSI filter | 25/75, 30/70, 35/65 | Dégrade (trop de bruit minute) |
| Paires | EUR, JPY, GBP, CAD | EUR seul |
| Strats alt | SuperTrend, MACD, Stoch, ATR Brk, IMS | Toutes négatives sur FX |

## 2. Ce qui n'a PAS été testé (gaps de recherche)

### Priorité haute — Impact probable fort

#### 2.1 ATR dynamique pour SL/TP
**Hypothèse:** Les stops fixes (0.5%/0.6%) sont sous-optimaux car la volatilité intraday varie. Un SL/TP calibré sur l'ATR daily devrait s'adapter aux régimes.
- `atr = data.run("talib:ATR", timeframe="1D", timeperiod=14)`
- SL = ATR × mult_sl, TP = ATR × mult_tp
- Sweep: mult_sl ∈ [0.5, 1.0, 1.5], mult_tp ∈ [0.8, 1.0, 1.5, 2.0]
- **Attendu:** Meilleur profit factor en régime volatile, potentiellement pire en régime calme

#### 2.2 BBANDS %B et Bandwidth comme filtres
**Hypothèse:** Entrer quand %B < 0 (sous la bande inférieure) ET bandwidth élevée (volatilité élevée) filtre les faux signaux.
- `bb.percent_b` — position normalisée dans les bandes [0,1], <0 = sous bande inf
- `bb.bandwidth` — largeur des bandes / SMA, proxy de volatilité
- Sweep: bandwidth_threshold ∈ [0.5, 1.0, 1.5] × median bandwidth
- **Attendu:** Moins de trades, meilleur win rate

#### 2.3 Macro variables individuelles (non testées)
**Hypothèse:** D'autres macro variables ont un pouvoir prédictif sur les régimes MR.
- **DGS10 (taux 10 ans):** niveau absolu + momentum (diff 20 jours)
- **DGS2 (taux 2 ans):** même analyse
- **CPI_CORE YoY:** inflation sous-jacente, potentiellement meilleur que CPI total
- **NFP change:** emploi = proxy de stress économique
- **PCE:** dépenses de consommation, signal de régime
- Test: corrélation de chaque variable avec la performance journalière de la stratégie
- Sweep: seuils sur chaque variable × combinaisons

#### 2.4 Session windows étendues
**Hypothèse:** La session 6-14 UTC capture le London AM mais manque le NY PM qui peut être profitable.
- Sessions à tester: (5,13), (6,12), (6,16), (8,14), (8,17), (9,15)
- Aussi: session par jour de la semaine (lundi vs vendredi)
- **Attendu:** Sessions plus courtes = moins de trades mais meilleure qualité

### Priorité moyenne — Impact probable modéré

#### 2.5 Holding period (td_stop)
**Hypothèse:** Certains trades profitent d'un holding plus long ou plus court que 4h.
- Sweep: td_stop ∈ ["1h", "2h", "3h", "4h", "6h", "8h", None]
- **Attendu:** Trade-off entre capturer les reversions lentes et éviter le bruit overnight

#### 2.6 EOD exit timing (dt_stop)
**Hypothèse:** Sortir à 21:00 UTC est arbitraire. Un exit plus tôt évite la volatilité de fin de journée.
- Sweep: dt_stop ∈ ["18:00", "19:00", "20:00", "21:00", "22:00"]
- **Attendu:** Impact modéré sur le DD

#### 2.7 VWAP anchor alternatifs
**Hypothèse:** Le VWAP reset à "D" (minuit UTC) est arbitraire pour FX. Un anchor à la session NY (17:00 ET = 22:00 UTC) serait plus pertinent.
- `vbt.VWAP.run(..., anchor="22h")` ou custom anchor
- **Attendu:** Légère amélioration si le marché FX structure ses sessions autour de 22h UTC

#### 2.8 Multi-timeframe BB
**Hypothèse:** BB calculé sur 5min ou 15min resamplé (pas 1min) filtre le bruit high-frequency.
- `deviation_5m = data.resample("5min").close - vwap_5m`
- `bb_5m = vbt.BBANDS.run(deviation_5m, window=60, alpha=5.0)`
- Realign signals to minute: `entries.vbt.realign(data.wrapper.index)`
- **Attendu:** Moins de trades, meilleur signal-to-noise

### Priorité basse — Exploratoire

#### 2.9 Day-of-week filter
**Hypothèse:** Certains jours sont plus favorables au MR (ex: lundi = continuation, mercredi = MR avant NFP/FOMC).
- Tester performance par jour de la semaine
- Filtrer les jours avec performance historique négative

#### 2.10 Slippage et fee sensitivity
**Hypothèse:** Les résultats sont sensibles au slippage car l'edge est petit (~1bp/trade).
- Tester avec slippage ∈ [0.0001, 0.00015, 0.0002, 0.0003]
- Ajouter fees fixes (0.5$ par lot)
- **Attendu:** Sharpe se dégrade significativement au-delà de 2 pips de slippage

#### 2.11 Combinaison BB + indicateur de confirmation
**Hypothèse:** Ajouter un filtre de confirmation (Stochastic, MACD histogram sign) améliore le win rate.
- **Attention:** Les tests précédents avec RSI sur minute ont DÉGRADÉ la performance. Tester sur 15min ou 1h resamplé.
- Stochastic(14) sur 1h: entry seulement si %K < 20 (oversold) pour longs
- MACD histogram positif sur 1h pour confirmer la direction

#### 2.12 Lead/lag macro
**Hypothèse:** Les variables macro ont un effet décalé sur les régimes FX.
- Tester le filtre macro avec lag de 1, 5, 10, 20 jours
- **Attendu:** Amélioration marginale si l'information macro met du temps à se diffuser

## 3. Protocole de test

Pour chaque hypothèse:
1. **Walk-forward sur 7 périodes** (2019-2025), jamais de backtest full-sample pour la sélection
2. **Métriques:** Sharpe moyen, nombre d'années positives (/7), OOS 2025, nombre de trades, PF, max DD
3. **Seuil de validation:** Sharpe avg > 0.5, ≥ 4/7 positif, OOS > 0
4. **Contrôle d'overfitting:** Le nombre de combos testés × degrés de liberté ne doit pas dépasser ~100 par dimension. Au-delà, appliquer un haircut de Bonferroni.
5. **Robustesse:** Toute amélioration doit être testée sur au moins 3 sous-périodes (early, mid, late)

## 4. Ordre d'exécution recommandé

```
Phase A — Diagnostic rapide (~5 min)
├── A1: ATR dynamic SL/TP sweep
├── A2: BBANDS %B + bandwidth filter
└── A3: Session windows étendues

Phase B — Macro deep-dive (~5 min)
├── B1: Corrélation macro individuelle vs perf stratégie
├── B2: Sweep seuils macro par variable
└── B3: Combinaisons macro optimales

Phase C — Fine-tuning (~5 min)
├── C1: td_stop + dt_stop sweep
├── C2: Multi-TF BB (5min, 15min)
└── C3: Day-of-week analysis

Phase D — Validation (~3 min)
├── D1: Slippage sensitivity
├── D2: Walk-forward final sur best combo
└── D3: Rapport final avec graphiques
```

## 5. Résultat attendu

Si les hypothèses hautes se confirment, on peut espérer :
- **ATR stops:** +0.1-0.2 Sharpe (meilleure adaptation aux régimes)
- **Bandwidth filter:** +0.1 Sharpe (trades plus sélectifs)
- **Macro individuelles:** +0.1-0.3 Sharpe (filtres plus fins)
- **Session/timing:** +0.05-0.1 Sharpe (optimisation marginale)

**Cible réaliste:** Sharpe walk-forward 1.0-1.3 avec macro filter optimisé.
**Plafond théorique:** Sharpe ~1.5 avec 50-80 trades/an de haute qualité.
