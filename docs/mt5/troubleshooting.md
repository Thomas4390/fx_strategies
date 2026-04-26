# Troubleshooting MQL5 — diagnostics récurrents

Format : symptôme → cause probable → fix.

## Symptômes côté EA

### "EA does not start, nothing in journal"

Causes :
- Trading désactivé : terminal en mode "AutoTrading off" (bouton vert/rouge en haut)
- EA pas autorisé : Outils → Options → Expert Advisors → cocher "Allow algo trading"
- Symbole pas dans MarketWatch : `SymbolSelect` retourne `false`

Fix :
```mql5
if(!TerminalInfoInteger(TERMINAL_TRADE_ALLOWED))
{
    Print("Algo trading disabled in terminal");
    return INIT_FAILED;
}
if(!MQLInfoInteger(MQL_TRADE_ALLOWED))
{
    Print("Algo trading disabled in EA");
    return INIT_FAILED;
}
```

### "OnTick never fires"

Causes :
- Pas de ticks (weekend, marché fermé)
- EA en backtest sans ticks générés (vérifier mode tester)
- Symbol en arrêt (broker)

### "iMA returns INVALID_HANDLE"

Causes :
- Symbole pas dans MarketWatch
- Period invalide
- Pas assez d'historique chargé (la doc spécifie qu'il faut au moins `period + 1` bars)

Fix : appeler `EnsureHistory(symbol, tf, period + 50)` avant `iMA`.

## Erreurs OrderSend (`MqlTradeResult.retcode`)

### 10004 — `TRADE_RETCODE_REQUOTE`

Cause : prix bouge entre l'envoi et la réception broker.

Fix : augmenter `request.deviation` (slippage acceptable).
```mql5
req.deviation = 30;  // 30 points de slippage acceptable
```

### 10006 — `TRADE_RETCODE_REJECT`

Cause : règles broker (ex. trade hors heures, taille minimum).

Fix : vérifier `OrderCheck` avant. Lire `result.comment`.

### 10013 — `TRADE_RETCODE_INVALID`

Cause : champs manquants dans `MqlTradeRequest`.

Fix : initialiser via `request = {}` puis remplir tous les champs requis :
```mql5
req.action = TRADE_ACTION_DEAL;
req.symbol = ...;
req.volume = ...;
req.type   = ...;
req.price  = ...;
```

### 10014 — `TRADE_RETCODE_INVALID_VOLUME`

Cause : volume hors `[VOLUME_MIN, VOLUME_MAX]` ou pas multiple de `VOLUME_STEP`.

Fix : appliquer `NormalizeLots` (cf. [03_trade_operations.md](./03_trade_operations.md)).

### 10015 — `TRADE_RETCODE_INVALID_PRICE`

Cause : prix avec mauvais nombre de décimales.

Fix :
```mql5
req.price = NormalizeDouble(price, _Digits);
```

### 10016 — `TRADE_RETCODE_INVALID_STOPS`

Cause : SL ou TP trop proches du prix (< `SYMBOL_TRADE_STOPS_LEVEL`).

Fix :
```mql5
double level = SymbolInfoInteger(_Symbol, SYMBOL_TRADE_STOPS_LEVEL) * _Point;
if(MathAbs(price - sl) < level)
    sl = (type == ORDER_TYPE_BUY) ? price - level - _Point : price + level + _Point;
```

### 10018 — `TRADE_RETCODE_MARKET_CLOSED`

Cause : marché fermé (weekend, holiday, fin de session).

Fix : check `SymbolInfoSessionTrade` avant l'envoi, ou simplement skip et retry.

### 10019 — `TRADE_RETCODE_NO_MONEY`

Cause : marge insuffisante.

Fix : réduire les lots, ou refuser le trade. Toujours préférer `OrderCheck` avant.

### 10020 — `TRADE_RETCODE_PRICE_CHANGED`

Cause : prix a bougé pendant l'envoi.

Fix : retry avec nouveau prix.

### 10021 — `TRADE_RETCODE_PRICE_OFF`

Cause : pas de quotes (Bid ou Ask = 0).

Fix : check `SymbolInfoTick` retourne valide avant l'envoi.

### 10027 — `TRADE_RETCODE_TIMEOUT`

Cause : serveur broker timeout.

Fix : retry après `Sleep(500)`.

## Erreurs `_LastError` / `GetLastError`

| Code | Constante | Sens | Fix |
|------|-----------|------|-----|
| 4756 | `ERR_TRADE_SEND_FAILED` | OrderSend rejeté | check retcode |
| 4014 | `ERR_HISTORY_NOT_FOUND` | Pas d'historique | `CopyRates` retry |
| 4106 | `ERR_NOT_VISIBLE` | Symbole pas en MarketWatch | `SymbolSelect` |
| 4301 | `ERR_USER_ERROR_FIRST` | Erreur custom EA | check Print précédent |

## Symptômes en backtest

### "Trades opened but immediately closed"

Cause : SL/TP placés du mauvais côté du prix (ex. `sl > price` sur un BUY).

Fix : check signe :
```mql5
if(type == ORDER_TYPE_BUY)
{
    Assert(sl < price);
    Assert(tp > price);
}
```

### "Equity curve is flat"

Causes :
- Aucun trade ouvert (signal jamais déclenché → debug `Print` les conditions)
- `lots = 0` à cause de capital trop faible

### "Backtest results differ vastly from Python"

Causes :
- Spread bid/ask appliqué en MQL5 mais pas en Python (vbt utilise mid)
- Mode tester "Open prices only" trop grossier
- Indicateur custom mal porté (vérifier ddof, smoothing RSI)

Fix : valider d'abord les indicateurs unitairement (cf. `Scripts/FxIndicatorTest.mq5`).

## Symptômes runtime

### "Macro cache stale" log apparaît

Cause : le bridge Python n'a pas tourné depuis 24h.

Fix : vérifier le cron job, le path d'écriture du fichier, les logs Python.

### "Symbol EURUSD not selected"

Causes :
- Suffixe broker (broker utilise "EURUSDm" ou "EURUSD.r")
- Compte démo/test sans ce symbole

Fix : ajouter `Inp_SymbolSuffix` dans inputs, vérifier MarketWatch.

### "EA loops on 'CopyBuffer failed' in journal"

Cause : Indicateur pas encore prêt (pas assez de bars).

Fix :
```mql5
int ready = BarsCalculated(g_h);
if(ready < period + 1) return;  // attendre
```

## Diagnostic généraliste

1. **Activer les logs verbeux** : input `Inp_LogVerbose = true` → `Print` avant chaque
   décision.
2. **Vérifier le journal** dans la fenêtre "Journal" du terminal — tous les `Print`
   et erreurs s'y retrouvent.
3. **Inspecter les positions** : `View → Trade` montre les ordres + magic numbers.
4. **Backtest pas-à-pas** : Strategy Tester en mode visuel, mode "Pause", on observe
   tick par tick.
5. **Si tout échoue** : commenter progressivement le code de l'EA jusqu'à isoler la
   cause. En MQL5 on n'a pas de debugger interactif efficace.

## Bonnes pratiques pour faciliter le diagnostic

- Logger chaque OrderSend avec son retcode et tous les paramètres
- Logger chaque entrée de signal avec les valeurs des indicateurs
- Préfixer les Print par `[SLEEVE_NAME]` pour filtrer
- Garder un fichier `fx_log.csv` séparé pour les events critiques

## Voir aussi

- [10_pitfalls.md](./10_pitfalls.md) — pièges classiques
- [12_references.md](./12_references.md) — doc officielle codes d'erreur
