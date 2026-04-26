# 04 — Indicateurs natifs MQL5

MQL5 fournit ~30 indicateurs techniques natifs. Le pattern est uniforme :
1. `OnInit` : créer un **handle** via `iMA`, `iRSI`, `iBands`, etc.
2. À l'usage : `CopyBuffer(handle, buffer_index, start_pos, count, dst_array)`
3. `OnDeinit` : libérer via `IndicatorRelease(handle)`

## Indicateurs utilisés dans ce projet

| Indicateur | Fonction MQL5 | Buffers |
|-----------|---------------|---------|
| EMA | `iMA(symbol, tf, period, shift, MODE_EMA, applied_price)` | 0 = MA |
| SMA | `iMA(symbol, tf, period, shift, MODE_SMA, applied_price)` | 0 = MA |
| RSI | `iRSI(symbol, tf, period, applied_price)` | 0 = RSI |
| Bollinger | `iBands(symbol, tf, period, shift, deviation, applied_price)` | 0 = base, 1 = upper, 2 = lower |
| ATR | `iATR(symbol, tf, period)` | 0 = ATR |
| MACD | `iMACD(symbol, tf, fast, slow, signal, applied_price)` | 0 = main, 1 = signal |
| Stochastic | `iStochastic(...)` | 0 = main, 1 = signal |

## Pattern complet — création + lecture

```mql5
int g_h_ema20 = INVALID_HANDLE;

int OnInit()
{
    g_h_ema20 = iMA(_Symbol, PERIOD_D1, 20, 0, MODE_EMA, PRICE_CLOSE);
    if(g_h_ema20 == INVALID_HANDLE)
    {
        PrintFormat("iMA EMA20 failed: %d", GetLastError());
        return INIT_FAILED;
    }
    return INIT_SUCCEEDED;
}

void OnDeinit(const int reason)
{
    if(g_h_ema20 != INVALID_HANDLE) IndicatorRelease(g_h_ema20);
}

double ReadEMA20(int shift = 1)
{
    double buf[];
    if(CopyBuffer(g_h_ema20, 0, shift, 1, buf) != 1)
    {
        PrintFormat("CopyBuffer EMA20 failed: %d", GetLastError());
        return 0.0;
    }
    return buf[0];
}
```

## CopyBuffer — paramètres

```mql5
int CopyBuffer(int handle, int buf_idx, int start, int count, double &dst[]);
```

- `start = 0` : la bar courante (en formation, valeurs peuvent encore changer)
- `start = 1` : la dernière bar fermée (← **utilisé pour les signaux**, anti look-ahead)
- `count = N` : nombre de valeurs à copier
- Retour : nombre copié, ou `-1` en cas d'erreur (`GetLastError`)

**Toujours utiliser `start = 1`** pour les signaux qui doivent être causaux.

## Lecture multi-symbole

```mql5
struct SPairHandles { int ema20, ema50, rsi7; };
SPairHandles g_h[3];
string g_pairs[] = {"EURUSD", "GBPUSD", "USDJPY"};

int OnInit()
{
    for(int i = 0; i < 3; i++)
    {
        g_h[i].ema20 = iMA(g_pairs[i], PERIOD_D1, 20, 0, MODE_EMA, PRICE_CLOSE);
        g_h[i].ema50 = iMA(g_pairs[i], PERIOD_D1, 50, 0, MODE_EMA, PRICE_CLOSE);
        g_h[i].rsi7  = iRSI(g_pairs[i], PERIOD_D1, 7, PRICE_CLOSE);
        if(g_h[i].ema20 == INVALID_HANDLE ||
           g_h[i].ema50 == INVALID_HANDLE ||
           g_h[i].rsi7  == INVALID_HANDLE)
            return INIT_FAILED;
    }
    return INIT_SUCCEEDED;
}
```

## Wilder vs simple smoothing (RSI)

L'`iRSI` MQL5 utilise le **Wilder smoothing** standard (formule canonique de J. Welles
Wilder) — équivalent à un `pandas`-RSI avec `wilder=True` ou `vbt.RSI` par défaut. À
vérifier en pratique sur 250 barres avec un dump CSV.

Si un projet Python utilise une autre formule (RSI à moyenne simple), il faudra
recoder un RSI custom (cf. [05_indicators_custom.md](./05_indicators_custom.md)).

## Préfixe `_Symbol`, `_Period` pour le chart attaché

Raccourci pratique :
```mql5
int h = iMA(_Symbol, _Period, 20, 0, MODE_EMA, PRICE_CLOSE);
```

## Recover après `INVALID_HANDLE`

Causes typiques :
- Symbole non sélectionné dans MarketWatch (`SymbolSelect(symbol, true)` dans `OnInit`)
- Pas assez d'historique chargé
- Mauvais paramètre (`period <= 0`)

Pattern de retry :
```mql5
int CreateHandleWithRetry(string symbol, ENUM_TIMEFRAMES tf, int period)
{
    for(int attempt = 0; attempt < 5; attempt++)
    {
        int h = iMA(symbol, tf, period, 0, MODE_EMA, PRICE_CLOSE);
        if(h != INVALID_HANDLE) return h;
        Sleep(200);
    }
    return INVALID_HANDLE;
}
```

## Indicateurs custom

Si un indicateur natif n'existe pas (ex. VWAP daily-anchored), il faut l'implémenter
soi-même. Voir [05_indicators_custom.md](./05_indicators_custom.md).

## `BarsCalculated` — vérifier que l'indicateur a fini son calcul

```mql5
int ready = BarsCalculated(handle);
if(ready < period + 1)
{
    // Pas encore prêt — ne pas lire
    return;
}
```

## Voir aussi

- [05_indicators_custom.md](./05_indicators_custom.md) — VWAP, Bollinger custom, rolling stats
- [07_history_timeseries.md](./07_history_timeseries.md) — `CopyClose`, `ArraySetAsSeries`
- [11_porting_from_python.md](./11_porting_from_python.md) — équivalences pandas → MQL5
