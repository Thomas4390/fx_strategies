# 03 — Trade Operations

Comment envoyer, modifier, fermer des ordres en MQL5. Pattern préféré : `CTrade`.

## CTrade — wrapper haut niveau

```mql5
#include <Trade/Trade.mqh>
CTrade trade;

int OnInit()
{
    trade.SetExpertMagicNumber(831);   // tag positions avec ce magic
    trade.SetDeviationInPoints(10);    // slippage acceptable en points
    trade.SetTypeFilling(ORDER_FILLING_FOK); // ou IOC, RETURN selon broker
    return INIT_SUCCEEDED;
}
```

### Buy / Sell market

```mql5
double price = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
double sl    = price * 0.995;  // -0.5%
double tp    = price * 1.006;  // +0.6%
double lots  = 0.10;

if(!trade.Buy(lots, _Symbol, price, sl, tp, "MR Macro long"))
{
    PrintFormat("Buy failed: retcode=%d desc=%s",
                trade.ResultRetcode(),
                trade.ResultRetcodeDescription());
}
```

`CTrade::Buy/Sell` appelle `OrderSend` en interne et stocke le résultat. Toujours
checker le retcode après.

### Fermer une position

```mql5
ulong ticket = PositionGetTicket(0);
if(!trade.PositionClose(ticket))
    PrintFormat("Close failed: %d", trade.ResultRetcode());
```

### Modifier SL/TP

```mql5
trade.PositionModify(ticket, new_sl, new_tp);
```

## OrderSend bas niveau (équivalent direct)

Utile quand `CTrade` ne suffit pas.

```mql5
MqlTradeRequest req = {};
MqlTradeResult  res = {};

req.action       = TRADE_ACTION_DEAL;
req.symbol       = _Symbol;
req.volume       = 0.10;
req.type         = ORDER_TYPE_BUY;
req.price        = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
req.sl           = req.price - 50 * _Point;
req.tp           = req.price + 60 * _Point;
req.deviation    = 10;
req.magic        = 831;
req.type_filling = ORDER_FILLING_FOK;
req.comment      = "MR Macro long";

if(!OrderSend(req, res))
{
    PrintFormat("OrderSend failed: error=%d", GetLastError());
    return;
}
if(res.retcode != TRADE_RETCODE_DONE)
{
    PrintFormat("Trade rejected: retcode=%d order=%I64u", res.retcode, res.order);
    return;
}
PrintFormat("OK deal=%I64u order=%I64u", res.deal, res.order);
```

## Codes de retour `MqlTradeResult.retcode`

Codes les plus fréquents :

| Code | Constante | Sens |
|------|-----------|------|
| 10009 | `TRADE_RETCODE_DONE` | Succès |
| 10004 | `TRADE_RETCODE_REQUOTE` | Prix invalide, retry avec nouveau prix |
| 10006 | `TRADE_RETCODE_REJECT` | Demande rejetée par le broker |
| 10013 | `TRADE_RETCODE_INVALID` | Demande invalide (champs manquants) |
| 10014 | `TRADE_RETCODE_INVALID_VOLUME` | Volume hors limites broker |
| 10015 | `TRADE_RETCODE_INVALID_PRICE` | Prix invalide |
| 10016 | `TRADE_RETCODE_INVALID_STOPS` | SL/TP trop proches du prix → cf. `SYMBOL_TRADE_STOPS_LEVEL` |
| 10017 | `TRADE_RETCODE_TRADE_DISABLED` | Trading désactivé |
| 10018 | `TRADE_RETCODE_MARKET_CLOSED` | Marché fermé |
| 10019 | `TRADE_RETCODE_NO_MONEY` | Marge insuffisante |
| 10020 | `TRADE_RETCODE_PRICE_CHANGED` | Prix a bougé pendant l'envoi |
| 10021 | `TRADE_RETCODE_PRICE_OFF` | Pas de quotes |
| 10027 | `TRADE_RETCODE_TIMEOUT` | Timeout serveur |

Liste complète : `https://www.mql5.com/en/docs/constants/errorswarnings/enum_trade_return_codes`.

## Validation `SYMBOL_TRADE_STOPS_LEVEL`

Distance minimale broker entre SL/TP et le prix courant. Si on viole → retcode 10016.

```mql5
double stops_level = SymbolInfoInteger(_Symbol, SYMBOL_TRADE_STOPS_LEVEL) * _Point;
if(MathAbs(price - sl) < stops_level)
    sl = (type == ORDER_TYPE_BUY) ? price - stops_level - _Point
                                  : price + stops_level + _Point;
```

## Lot normalization

```mql5
double NormalizeLots(string symbol, double raw_lots)
{
    double step = SymbolInfoDouble(symbol, SYMBOL_VOLUME_STEP);
    double minv = SymbolInfoDouble(symbol, SYMBOL_VOLUME_MIN);
    double maxv = SymbolInfoDouble(symbol, SYMBOL_VOLUME_MAX);
    double lots = MathFloor(raw_lots / step) * step;
    return MathMax(minv, MathMin(maxv, lots));
}
```

## Sizing par risque (pattern utilisé dans le projet)

```mql5
// risk_money = sub_equity * risk_pct * leverage
// lots       = risk_money / (sl_distance_in_points * tick_value)
double LotsForRisk(string symbol, double sub_equity, double risk_pct,
                   double leverage, double sl_distance_price)
{
    double risk_money = sub_equity * risk_pct * leverage;
    double tick_value = SymbolInfoDouble(symbol, SYMBOL_TRADE_TICK_VALUE);
    double tick_size  = SymbolInfoDouble(symbol, SYMBOL_TRADE_TICK_SIZE);
    double points     = sl_distance_price / tick_size;
    if(points <= 0 || tick_value <= 0) return 0.0;
    double raw_lots = risk_money / (points * tick_value);
    return NormalizeLots(symbol, raw_lots);
}
```

## Itération sur les positions

```mql5
int total = PositionsTotal();
for(int i = total - 1; i >= 0; i--)   // descending pour close-while-iterating
{
    ulong ticket = PositionGetTicket(i);
    if(ticket == 0) continue;
    if(PositionGetInteger(POSITION_MAGIC) != MY_MAGIC) continue;
    if(PositionGetString(POSITION_SYMBOL) != "EURUSD") continue;

    long pos_type = PositionGetInteger(POSITION_TYPE); // POSITION_TYPE_BUY/SELL
    double volume = PositionGetDouble(POSITION_VOLUME);
    double open   = PositionGetDouble(POSITION_PRICE_OPEN);
    datetime opened = (datetime)PositionGetInteger(POSITION_TIME);

    // Logique fermeture conditionnelle
    if((TimeGMT() - opened) > 6 * 3600)
        trade.PositionClose(ticket);
}
```

## Hedging vs Netting

- **Netting** (compte unique sur EU brokers) : une seule position nette par symbole.
  `Buy 0.1` puis `Sell 0.05` → reste `Buy 0.05`. `PositionsTotal` = nombre de symboles
  avec position non nulle.
- **Hedging** (FX brokers offshore) : positions opposées coexistent. `Buy 0.1` puis
  `Sell 0.05` → 2 positions. `PositionsTotal` = total positions.

Pour fermer en netting : `OrderSend` opposé du même volume.
Pour fermer en hedging : `trade.PositionClose(ticket)` cible le ticket précis.

`CTrade::PositionClose(ticket)` fonctionne dans les deux modes.

Détection :
```mql5
ENUM_ACCOUNT_MARGIN_MODE mode = (ENUM_ACCOUNT_MARGIN_MODE)AccountInfoInteger(ACCOUNT_MARGIN_MODE);
// ACCOUNT_MARGIN_MODE_RETAIL_NETTING ou ACCOUNT_MARGIN_MODE_RETAIL_HEDGING
```

## OrderCheck — pré-validation

Avant d'envoyer un ordre coûteux, vérifier qu'il passera :
```mql5
MqlTradeCheckResult check = {};
if(!OrderCheck(req, check))
    PrintFormat("Pre-check failed: %d %s", check.retcode, check.comment);
```

## Voir aussi

- [10_pitfalls.md](./10_pitfalls.md) — pièges sur OrderSend
- [troubleshooting.md](./troubleshooting.md) — diagnostic codes 10004-10027
- [06_multi_symbol.md](./06_multi_symbol.md) — trader plusieurs symboles
