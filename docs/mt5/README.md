# Base de connaissance MQL5 — fx_strategies

Cette base de connaissance documente MQL5 (MetaTrader 5) dans le contexte spécifique
du portage de la stratégie FX Tri-Signaux (`src/strategies/*.py` → `src/mt5/`). Elle
sert de référence pour :

1. Comprendre rapidement les patterns MQL5 critiques (event handlers, trade ops, indicateurs).
2. Éviter les pièges connus (DST, suffixes broker, lots, retcodes).
3. Traduire les idiomes Python/vectorbtpro en MQL5.

## Comment lire ce dossier

| Si tu veux… | Lis… |
|------------|------|
| Démarrer un EA from scratch | [01_essentials.md](./01_essentials.md) → [02_event_handlers.md](./02_event_handlers.md) |
| Envoyer des ordres correctement | [03_trade_operations.md](./03_trade_operations.md) |
| Calculer un indicateur (natif ou custom) | [04_indicators_native.md](./04_indicators_native.md) → [05_indicators_custom.md](./05_indicators_custom.md) |
| Trader plusieurs symboles depuis un seul EA | [06_multi_symbol.md](./06_multi_symbol.md) |
| Gérer dates, sessions, history | [07_history_timeseries.md](./07_history_timeseries.md) |
| Lire un fichier ou persister un état | [08_file_io_globals.md](./08_file_io_globals.md) |
| Backtester correctement | [09_strategy_tester.md](./09_strategy_tester.md) |
| Comprendre pourquoi ça plante | [10_pitfalls.md](./10_pitfalls.md) → [troubleshooting.md](./troubleshooting.md) |
| Traduire du Python en MQL5 | [11_porting_from_python.md](./11_porting_from_python.md) |
| Trouver une source externe | [12_references.md](./12_references.md) |
| Accéder à des données macro depuis MQL5 (Calendar, WebRequest) | [13_native_data_sources.md](./13_native_data_sources.md) |
| Piloter MT5 en CLI sur Linux/Wine (compile, backtest headless) | [14_cli_backtest_linux.md](./14_cli_backtest_linux.md) |
| Investiguer l'écart RSI Daily VBT vs MT5 | [`../investigations/rsi_daily_vbt_vs_mt5.md`](../investigations/rsi_daily_vbt_vs_mt5.md) |
| Livrer la stratégie à un client (Windows MT5) | [`../../reports/client/guide_installation/main.pdf`](../../reports/client/guide_installation/main.pdf) |

## Conventions de code MQL5 utilisées dans ce repo

- Classes : `C` + PascalCase → `CSleeveBase`, `CMacroFilter`, `CRiskManager`
- Enums : `E` + PascalCase → `ESleeveID`
- Membres privés : préfixe `m_` → `m_anchor_day`
- Globaux statiques : préfixe `g_` → `g_pairs`
- Inputs EA : préfixe `Inp_` → `Inp_AllocMRMacro`
- Constantes `#define` : `MAGIC_*`, `FX_*` UPPER_SNAKE
- Magic numbers : `831` (MR Macro), `832` (TS Momentum), `833` (RSI Daily)
- Tous les timestamps en UTC via `TimeGMT()` (jamais `TimeCurrent()`)

## Contexte projet

Le code MQL5 cible reproduit la stratégie suivante :

| Sleeve | Allocation | Source Python | Symboles |
|--------|-----------|---------------|----------|
| MR Macro (intraday) | 80% | `src/strategies/mr_macro.py` | EUR/USD M1 |
| TS Momentum (daily) | 10% | `src/strategies/daily_momentum.py` | EUR/USD, GBP/USD, USD/JPY (D1) |
| RSI Daily | 10% | `src/strategies/rsi_daily.py` | EUR/USD, GBP/USD, USD/JPY, USD/CAD (D1) |

Overlay : `src/strategies/combined_portfolio_v2.py` — vol-targeting global 28%, max
leverage 12×, DD cap désactivé en production.

## Quand cette doc ne suffit pas

- **Doc officielle MetaQuotes** : `https://www.mql5.com/en/docs/`
- **Context7 (live docs)** : IDs `/websites/mql5docs_onrender` (5070 snippets, deep)
  et `/websites/mql5_en_book` (livre officiel, 185 snippets)
- **Articles MQL5** : `https://www.mql5.com/en/articles` — chercher "Multi-Currency",
  "Walk Forward", "CTrade tutorial"

## Maintenance

Cette base de connaissance est conçue pour évoluer :
- Ajouter un nouveau pitfall → `10_pitfalls.md` ou `troubleshooting.md`
- Découvrir un mapping Python/MQL5 → `11_porting_from_python.md`
- Trouver un article utile → `12_references.md`

Préférer mettre à jour les fichiers existants plutôt que d'en créer de nouveaux.
