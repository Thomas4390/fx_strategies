# 06 — Multi-Symbol Expert Advisor

Trader plusieurs symboles depuis **un seul EA** attaché à un chart primaire. Pattern
recommandé pour les stratégies portefeuille (cf. notre stratégie : 4 paires FX).

## Pattern général

L'EA est attaché à un **symbole primaire** (ex. EUR/USD M1). `OnTick` ne se déclenche
que pour ce symbole. Les autres symboles (GBP/USD, USD/JPY, USD/CAD) sont accédés
via :
- `iMA(other_symbol, ...)`, `iRSI(other_symbol, ...)` — handles indicateurs cross-symbol
- `CopyRates(other_symbol, ...)`, `CopyClose(other_symbol, ...)` — données historiques
- `SymbolInfoDouble(other_symbol, SYMBOL_BID/ASK)` — prix courant
- `OrderSend` / `CTrade::Buy` avec `request.symbol = other_symbol`

## Étape 1 — `SymbolSelect` à `OnInit`

Un symbole doit être dans MarketWatch pour être tradable. La doc précise : *"Selects
or deselects a symbol in the MarketWatch window."*

```mql5
string g_pairs[] = {"EURUSD", "GBPUSD", "USDJPY", "USDCAD"};

int OnInit()
{
    for(int i = 0; i < ArraySize(g_pairs); i++)
    {
        if(!SymbolSelect(g_pairs[i], true))
        {
            PrintFormat("Cannot select %s in MarketWatch (err=%d)",
                        g_pairs[i], GetLastError());
            return INIT_FAILED;
        }
    }
    return INIT_SUCCEEDED;
}
```

## Étape 2 — Charger l'historique

Indispensable avant `iMA`/`iRSI` : forcer le téléchargement de l'historique.

```mql5
bool EnsureHistory(string symbol, ENUM_TIMEFRAMES tf, int min_bars)
{
    MqlRates rates[];
    int attempts = 0;
    int copied = 0;
    while(attempts < 25 && (copied = CopyRates(symbol, tf, 0, min_bars, rates)) < min_bars)
    {
        Sleep(100);
        attempts++;
    }
    if(copied < min_bars)
    {
        PrintFormat("History incomplete for %s on %s: got %d/%d bars",
                    symbol, EnumToString(tf), copied, min_bars);
        return false;
    }
    return true;
}
```

Source : pattern adapté du snippet officiel `CopyFromSymbolToBuffers` (mql5docs).

## Étape 3 — Suffixes broker

Beaucoup de brokers ajoutent un suffixe : `EURUSDm`, `EURUSD.r`, `EURUSD-Pro`, etc.

```mql5
input string Inp_SymbolSuffix = "";  // configurable

string MakeSymbol(string base)
{
    return base + Inp_SymbolSuffix;
}
```

Vérification supplémentaire :
```mql5
if(!SymbolInfoInteger(symbol, SYMBOL_SELECT))
{
    PrintFormat("Symbol %s not selected", symbol);
}
```

## Étape 4 — Lecture cross-symbol

```mql5
double ReadCloseShift1(string symbol, ENUM_TIMEFRAMES tf)
{
    double buf[];
    if(CopyClose(symbol, tf, 1, 1, buf) != 1)
    {
        PrintFormat("CopyClose failed for %s: %d", symbol, GetLastError());
        return 0.0;
    }
    return buf[0];
}

double ReadIndicator(int handle, int shift = 1)
{
    double buf[];
    if(CopyBuffer(handle, 0, shift, 1, buf) != 1) return 0.0;
    return buf[0];
}
```

## Étape 5 — Trade cross-symbol

```mql5
#include <Trade/Trade.mqh>
CTrade trade;

void OpenLongOnPair(string symbol, double lots, double sl_pct, double tp_pct)
{
    double price = SymbolInfoDouble(symbol, SYMBOL_ASK);
    double sl = price * (1.0 - sl_pct);
    double tp = price * (1.0 + tp_pct);
    trade.SetExpertMagicNumber(MAGIC_TS);
    if(!trade.Buy(lots, symbol, price, sl, tp, "TS Momentum"))
        PrintFormat("Buy %s failed: %d", symbol, trade.ResultRetcode());
}
```

## Détection de nouvelle barre cross-symbol

`OnTick` se déclenche **uniquement** pour le symbole du chart attaché. Pour détecter
les nouvelles barres D1 sur les autres paires, on utilise `OnTimer` :

```mql5
struct SPairBarTracker { string symbol; datetime last_bar; };
SPairBarTracker g_trackers[4];

void OnTimer()
{
    datetime now = TimeGMT();
    if((now % 86400) < 21*3600) return;  // attendre passage 21h UTC

    for(int i = 0; i < 4; i++)
    {
        datetime cur = iTime(g_trackers[i].symbol, PERIOD_D1, 0);
        if(cur != g_trackers[i].last_bar)
        {
            ProcessNewDailyBar(g_trackers[i].symbol);
            g_trackers[i].last_bar = cur;
        }
    }
}
```

## Strategy Tester multi-symbol

Le Strategy Tester MT5 supporte le multi-symbol nativement si :
1. `SymbolSelect(symbol, true)` est appelé dans `OnInit`
2. L'historique est chargé via `CopyRates` au démarrage

Limite importante : les symboles non-primaires sont **simulés depuis OHLC M1** —
pas tick-par-tick. Acceptable pour stratégies daily, problématique pour intraday.

## Pièges

| Piège | Mitigation |
|-------|-----------|
| Symbole pas dans MarketWatch | `SymbolSelect(symbol, true)` à `OnInit` |
| Suffixe broker inattendu | Input `Inp_SymbolSuffix` |
| Historique pas chargé | `EnsureHistory` retry pattern |
| Tick value différent (USDJPY a un point ≠ EURUSD) | Toujours lire `SYMBOL_TRADE_TICK_VALUE` du symbole concerné |
| Heures de cotation différentes | Vérifier `SymbolInfoSessionQuote` si trade tôt/tard |

## Voir aussi

- [03_trade_operations.md](./03_trade_operations.md) — CTrade pour multi-symbol
- [09_strategy_tester.md](./09_strategy_tester.md) — limitations multi-symbol en backtest
- Article MQL5 : "MQL5 Cookbook: Multi-Currency Expert Advisor" (`mql5.com/en/articles/648`)
- Article MQL5 : "How to create a simple Multi-Currency EA Part 1" (`articles/13008`)
