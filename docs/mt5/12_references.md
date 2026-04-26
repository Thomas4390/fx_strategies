# 12 — Références

Sources externes utiles pour aller plus loin que cette base de connaissance.

## Documentation officielle MetaQuotes

### Hub principal

- `https://www.mql5.com/en/docs/` — Index global

### Sections les plus utilisées

| Section | URL | Quand consulter |
|---------|-----|-----------------|
| Event Handlers | `mql5.com/en/docs/event_handlers/` | Tous les `On*` |
| Trade Functions | `mql5.com/en/docs/trading/` | `OrderSend`, `PositionGet*`, `HistorySelect` |
| Standard Library — Trade | `mql5.com/en/docs/standardlibrary/tradeclasses/` | `CTrade`, `CSymbolInfo`, `CPositionInfo` |
| Series & Indicators | `mql5.com/en/docs/series/` | `CopyRates`, `CopyClose`, indicateurs |
| Account Information | `mql5.com/en/docs/account/` | `AccountInfoDouble`, marges |
| Constants | `mql5.com/en/docs/constants/` | Codes d'erreur, enums |
| Date and Time | `mql5.com/en/docs/dateandtime/` | `TimeGMT`, `TimeToStruct` |
| File Operations | `mql5.com/en/docs/files/` | `FileOpen`, flags |
| Global Variables | `mql5.com/en/docs/globalvariables/` | Persistance |
| Testing | `mql5.com/en/docs/runtime/testing/` | Strategy Tester |

### Codes de retour

`mql5.com/en/docs/constants/errorswarnings/enum_trade_return_codes` — `TRADE_RETCODE_*`
`mql5.com/en/docs/constants/errorswarnings/errorcodes` — codes `_LastError`

## Livre officiel

**MQL5 Programming for Traders** : `https://www.mql5.com/en/book`

Chapitres particulièrement utiles :
- `mql5.com/en/book/automation/experts/experts_ontick` — OnTick en profondeur
- `mql5.com/en/book/automation/experts/experts_ontimer` — OnTimer
- `mql5.com/en/book/standardlibrary/tradeclasses/trade` — `CTrade` complet

## Articles de référence (mql5.com/articles)

| ID | Titre | Pourquoi |
|----|-------|----------|
| 648 | MQL5 Cookbook : Multi-Currency Expert Advisor — Simple, Neat and Quick Approach | Pattern multi-currency canonique |
| 13008 | How to create a simple Multi-Currency EA Part 1 | ADX + Parabolic SAR multi-symbol |
| 13470 | Multi-Currency EA Part 2 | Multi-timeframe Parabolic SAR |
| 1428 | Standard Trade Library : CTrade tutorial | CTrade complet |
| 33 | Code Style Guide | Conventions de nommage |

URLs : `https://www.mql5.com/en/articles/<ID>`

## Forums

- `https://www.mql5.com/en/forum/automation` — questions EA / automation
- `https://www.mql5.com/en/forum/465680` — discussion best practices multi-currency

## Context7 — Live docs

Deux IDs Context7 à utiliser pour des questions ad-hoc :

| Library ID | Snippets | Quand utiliser |
|-----------|----------|----------------|
| `/websites/mql5docs_onrender` | 5070 | Recherche profonde dans la doc complète |
| `/websites/mql5_en_book` | 185 | Snippets du livre officiel |

Usage type :
```
Context7 query : "OnTradeTransaction example deal_add filter by magic"
```

## Outils complémentaires

- **MetaEditor** : IDE intégré, F7 pour compiler, F4 ou Ctrl+F8 pour profiler
- **MetaTrader 5 Terminal** : Strategy Tester (Ctrl+R), View → Toolbox pour journal
- **Python `MetaTrader5` package** : `pip install MetaTrader5` — pour scripter
  l'export d'historique ou les tests offline (mais pas pour exécuter en prod)

## Resources externes (non-officielles mais utiles)

- `darwinex.com/algorithmic-trading/mql5/` — articles tutoriels
- `www.earnforex.com/metatrader-expert-advisors/` — exemples EAs gratuits
- GitHub topic `mql5` — projets open-source

## Glossaire MQL5

| Terme | Définition |
|-------|-----------|
| **EA** | Expert Advisor — programme qui trade automatiquement |
| **Indicator** | Programme qui dessine sur le chart, ne trade pas |
| **Script** | Programme one-shot, lancé manuellement |
| **Service** | Programme background lancé au démarrage du terminal |
| **Handle** | Référence numérique vers un indicateur ou un fichier |
| **Tick** | Mise à jour de prix (Bid ou Ask change) |
| **Bar** | Barre OHLC sur un timeframe donné |
| **Point** | Plus petit incrément de prix (`_Point`) |
| **Pip** | Convention humaine, parfois 10× `_Point` (5-digit) |
| **Lot** | Volume — 1 lot standard = 100 000 unités base FX |
| **Magic** | Numéro identifiant un EA pour ses positions |
| **Slippage** | Différence prix demandé vs prix exécuté (en points) |
| **Deviation** | Synonyme de slippage acceptable dans `MqlTradeRequest` |

## Voir aussi

- [README.md](./README.md) — index local
- [troubleshooting.md](./troubleshooting.md) — diagnostic erreurs
