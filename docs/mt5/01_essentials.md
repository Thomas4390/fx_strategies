# 01 — MQL5 Essentials

Langage, types, structure d'un Expert Advisor (EA), cycle de vie.

## Types primitifs

| Type | Taille | Usage typique |
|------|--------|---------------|
| `bool` | 1 byte | flags |
| `char` / `uchar` | 1 byte | rare en EA |
| `short` / `ushort` | 2 bytes | rare |
| `int` / `uint` | 4 bytes | indices, counters |
| `long` / `ulong` | 8 bytes | tickets ordres/positions, magic |
| `float` | 4 bytes | jamais — toujours `double` pour les prix |
| `double` | 8 bytes | prix, volumes, indicateurs |
| `datetime` | 8 bytes | epoch UNIX en secondes |
| `string` | variable | libellés, comments, symboles |
| `color` | 4 bytes | UI |

**Règle** : pour les prix et calculs financiers, **toujours** `double`. Les volumes
sont aussi en `double` (pas en lot integer).

## Constantes prédéfinies

| Constante | Description |
|-----------|-------------|
| `_Symbol` | Symbole du chart attaché |
| `_Period` | Timeframe du chart attaché |
| `_Point` | Plus petit incrément de prix (ex. 0.00001 sur EUR/USD 5 chiffres) |
| `_Digits` | Nombre de décimales |
| `_LastError` | Dernier code d'erreur (équivalent à `GetLastError()`) |
| `INVALID_HANDLE` | -1, retour d'un `iMA`/`iRSI` raté |

## Enums clés

```mql5
ENUM_TIMEFRAMES        // PERIOD_M1, M5, M15, H1, H4, D1, W1, MN1
ENUM_ORDER_TYPE        // ORDER_TYPE_BUY, SELL, BUY_LIMIT, SELL_LIMIT, BUY_STOP, SELL_STOP
ENUM_TRADE_REQUEST_ACTIONS // TRADE_ACTION_DEAL, PENDING, MODIFY, REMOVE, CLOSE_BY
ENUM_POSITION_TYPE     // POSITION_TYPE_BUY, POSITION_TYPE_SELL
ENUM_ACCOUNT_INFO_DOUBLE  // ACCOUNT_BALANCE, ACCOUNT_EQUITY, ACCOUNT_MARGIN, ACCOUNT_FREEMARGIN
ENUM_SYMBOL_INFO_DOUBLE   // SYMBOL_BID, ASK, POINT, TRADE_TICK_VALUE, TRADE_TICK_SIZE,
                          // VOLUME_MIN, VOLUME_MAX, VOLUME_STEP
ENUM_SYMBOL_INFO_INTEGER  // SYMBOL_DIGITS, SYMBOL_TRADE_STOPS_LEVEL, SYMBOL_SELECT
```

## Structure d'un EA minimal

```mql5
//+------------------------------------------------------------------+
//| MyExpert.mq5                                                     |
//+------------------------------------------------------------------+
#property copyright "..."
#property version   "1.00"
#property strict

input int    Inp_FastEMA = 20;
input double Inp_RiskPct = 0.01;

#include <Trade/Trade.mqh>
CTrade g_trade;

int OnInit()
{
    g_trade.SetExpertMagicNumber(12345);
    EventSetTimer(60);   // timer 1 min
    return INIT_SUCCEEDED;
}

void OnDeinit(const int reason)
{
    EventKillTimer();
}

void OnTick()
{
    // Logique tick-by-tick (rarement utilisée directement — préférer NewBar pattern)
}

void OnTimer()
{
    // Logique 1×/min : refresh data, monitoring
}

void OnTradeTransaction(const MqlTradeTransaction &trans,
                        const MqlTradeRequest &req,
                        const MqlTradeResult &res)
{
    // Réagir aux fills/closes
}
```

## Includes standards

```mql5
#include <Trade/Trade.mqh>             // CTrade, CPositionInfo, COrderInfo
#include <Trade/SymbolInfo.mqh>        // CSymbolInfo
#include <Trade/AccountInfo.mqh>       // CAccountInfo
#include <Arrays/ArrayDouble.mqh>      // CArrayDouble
#include <ChartObjects/ChartObject.mqh> // dessins graphiques
#include <Indicators/Indicator.mqh>    // base indicateurs
```

Chemin physique : `<MQL5_root>/Include/...`. Les fichiers `.mqh` sont des en-têtes —
ils peuvent contenir code complet (pas seulement déclarations).

## Différences clés MQL4 → MQL5

| MQL4 | MQL5 |
|------|------|
| `OrderSend(symbol, type, lots, price, slippage, sl, tp, ...)` | `MqlTradeRequest` struct + `OrderSend(req, res)` |
| `OrdersTotal()` itère ordres+positions confondus | `OrdersTotal()` = pending only ; `PositionsTotal()` = positions ouvertes |
| `iMA(symbol, tf, period, shift, mode, applied, shift)` retourne directement la valeur | `iMA(...)` retourne un **handle** ; lire via `CopyBuffer(handle, buffer, start, count, dst[])` |
| Pas d'`OnTimer` natif | `EventSetTimer(seconds)` + `OnTimer()` |
| `IsTesting()` | `MQLInfoInteger(MQL_TESTER)` |

## Cycle de vie EA

```
[Compilation OK]
    ↓
OnInit()                ← retour INIT_SUCCEEDED, INIT_FAILED, INIT_PARAMETERS_INCORRECT
    ↓
[Loop]
    ├── OnTick()        ← chaque tick reçu sur le symbole du chart
    ├── OnTimer()       ← chaque N secondes si EventSetTimer()
    ├── OnChartEvent()  ← clics, touches clavier sur le chart
    ├── OnBookEvent()   ← market depth (si SymbolInfoInteger(SYMBOL_BOOK_DEPTH) > 0)
    └── OnTradeTransaction() ← chaque transaction du compte
    ↓
[Fin : retrait, recompil, terminal close]
    ↓
OnDeinit(reason)        ← raison via UninitializeReason()
```

Voir [02_event_handlers.md](./02_event_handlers.md) pour le détail de chaque handler.

## Codes de retour `OnInit`

```mql5
return INIT_SUCCEEDED;            // OK, EA tourne
return INIT_FAILED;               // erreur générale, EA ne démarre pas
return INIT_PARAMETERS_INCORRECT; // inputs invalides — l'utilisateur peut corriger
```

## Compilation

- IDE : MetaEditor (intégré au terminal MT5).
- Bouton **Compile** ou F7. Erreurs/warnings dans l'onglet "Errors".
- Sortie : fichier `.ex5` dans le même dossier que le `.mq5`.
- Path répertoire MQL5 : `Fichier → Ouvrir le dossier de données` dans le terminal.

## Bonnes pratiques

1. Toujours `#property strict`.
2. Toujours vérifier `_LastError` après une opération critique.
3. Ne jamais `Sleep` long dans `OnTick` (bloque le thread du terminal).
4. Stocker les handles d'indicateurs en variables membres, libérer dans `OnDeinit` via `IndicatorRelease`.
5. Préférer `CTrade` à `OrderSend` direct (gère les retries, normalisation prix).

## Voir aussi

- [02_event_handlers.md](./02_event_handlers.md) — détail des handlers
- [03_trade_operations.md](./03_trade_operations.md) — CTrade, OrderSend
- [10_pitfalls.md](./10_pitfalls.md) — pièges courants
