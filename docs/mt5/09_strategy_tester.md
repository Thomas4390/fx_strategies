# 09 — Strategy Tester

Backtester un EA dans MT5 — modes, multi-symbol, optimisation, walk-forward.

## Lancement

Terminal MT5 → `Vue → Testeur de stratégies` (Ctrl+R).

Onglets :
- **Réglages** : EA, symbole, timeframe, période, modèle de génération
- **Inputs** : valeurs des paramètres EA
- **Optimisation** : grid search ou génétique sur ranges d'inputs

## Modes de génération de ticks

| Mode | Précision | Vitesse | Quand utiliser |
|------|-----------|---------|----------------|
| `Every tick based on real ticks` | ⭐⭐⭐⭐⭐ | ⭐ | Stratégies intraday qui dépendent du flux tick exact |
| `Every tick` | ⭐⭐⭐⭐ | ⭐⭐ | Intraday sans tick-by-tick précis |
| `1 minute OHLC` | ⭐⭐⭐ | ⭐⭐⭐⭐ | Stratégies qui ne lisent que les bars M1 fermées |
| `Open prices only` | ⭐ | ⭐⭐⭐⭐⭐ | Optimisation rapide, swing trades |

**Pour ce projet** : `Every tick based on real ticks` sur EUR/USD M1. Les paires
secondaires (GBP/USD, USD/JPY, USD/CAD) sont resamplées depuis OHLC M1 — c'est
acceptable car les sleeves 2 et 3 sont en daily.

## Multi-symbol testing

Le Strategy Tester MT5 supporte le multi-symbol nativement depuis la v5 — pas besoin
de configuration spéciale. Mais :

1. L'EA doit appeler `SymbolSelect(symbol, true)` dans `OnInit` pour chaque paire
2. Les ticks des paires non-primaires sont **simulés** depuis les barres M1 OHLC
3. Pour des stratégies tick-précises sur 2+ symboles, utiliser le **Tick Modeling**
   du serveur cloud MT5 (payant)

## Période de test

Format : `Start date` / `End date` dans l'onglet Réglages.

Pour ce projet :
- **Validation rapide** : 2024-01 → 2024-12 (1 an)
- **Walk-forward** : 2019-01 → 2025-04 (~6 ans)
- **Stress 2023** : 2023-01 → 2023-12 (mauvaise année MR Macro selon le rapport)

## Délai (slippage simulé)

Onglet `Réglages` → `Délai` :
- `0 ms` : exécution instantanée (irréaliste)
- `1-50 ms` : retail typique
- `Random delay` : variable, plus réaliste

## Mode visuel vs non-visuel

- **Mode visuel** : EA dessine sur le chart pendant le test, on peut suivre tick-par-tick
- **Mode non-visuel** : 5-10× plus rapide, journal final seulement

## Logs et résultats

Onglets après test :
- **Résultats** : trade-by-trade
- **Graphique** : equity curve
- **Rapport** : Sharpe, Profit Factor, Max DD, Win Rate, etc.
- **Journal** : tous les `Print()` de l'EA

## Optimisation

Onglet `Réglages` → `Optimisation` :
- `Disabled` : test simple
- `Slow complete algorithm` : grid search exhaustif
- `Fast genetic algorithm` : sélection génétique (recommandé pour > 4 paramètres)
- `All symbols selected in MarketWatch` : multi-symbol opt
- `Forward` : split la période in-sample/out-of-sample (1/3, 1/2, 1/4)

### Inputs optimizables

```mql5
input int    Inp_FastEMA = 20; // 10, 30, 5    ← min, max, step
input double Inp_TPStop  = 0.006;
```

Cocher "Optimization" dans le panneau Inputs pour rendre éditable.

## Walk-forward analysis

MT5 propose un walk-forward simplifié via le `Forward` ratio :
- `Forward = 1/4` : 75% in-sample, 25% out-of-sample (test final automatique)
- Bouton "Forward" dans l'onglet Optimisation

Pour walk-forward plus avancé (multiple folds), il faut :
1. Lancer N optimisations sur N fenêtres in-sample
2. Tester chaque jeu de paramètres optimaux sur la fenêtre out-of-sample suivante
3. Agréger les résultats

→ Souvent fait hors de MT5 (Python qui appelle le tester en CLI).

## Limitations connues

1. **Multi-symbol tick precision** : symboles non-primaires resamplés depuis M1
2. **Macro data externe** : impossible de simuler un fichier `macro_cache.csv` qui
   évolue dans le temps. Solution : pré-loader un dump `macro_cache_history.csv` avec
   colonnes `(date, macro_ok)`, et que l'EA résolve la valeur historique en backtest
   (mode détecté via `MQLInfoInteger(MQL_TESTER)`)
3. **GlobalVariables** : isolées dans l'environnement de tester (n'affectent pas le
   terminal de prod) — comportement attendu et OK
4. **`OnTimer` minimum 1s** : si on a besoin de précision sub-seconde en backtest,
   passer par `OnTick` exclusivement

## Mode tester détecté

```mql5
bool IsTester() { return (bool)MQLInfoInteger(MQL_TESTER); }
bool IsOptimization() { return (bool)MQLInfoInteger(MQL_OPTIMIZATION); }
```

Utile pour :
- Désactiver les `Print` verbeux en optimisation
- Charger un dump historique macro vs un fichier live

## Quality of historical data

Avant de tester, vérifier la qualité de l'historique du symbole :

Terminal → Outils → Historique → choisir symbole + TF → vérifier le nombre de bars.

`Sortir le journal` :
```
2024.05.15 10:23:45.000   OnInit() Bars in M1: 1234567 (last: 2024.05.15 10:00)
```

## Compte démo équivalent

Une fois le backtest validé, **toujours** confirmer sur compte démo broker pendant
≥ 1 semaine avant prod. Le backtest n'inclut pas :
- Latence réseau réelle
- Re-quotes broker
- Slippage en news events
- Bugs OS/terminal

## Voir aussi

- [10_pitfalls.md](./10_pitfalls.md) — pièges du tester
- [11_porting_from_python.md](./11_porting_from_python.md) — comparer P&L vs vbt
