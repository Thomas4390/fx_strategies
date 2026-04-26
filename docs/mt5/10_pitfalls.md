# 10 — Pitfalls (pièges courants)

Liste numérotée de pièges fréquents en MQL5, avec code "wrong" / "right".

---

## 1. `OnTick` qui bloque trop longtemps

**Wrong** :
```mql5
void OnTick()
{
    Sleep(2000);  // ← bloque le thread terminal
    HeavyCalculation();
}
```

**Right** :
```mql5
void OnTick()
{
    static datetime last_calc = 0;
    if(TimeGMT() - last_calc < 60) return;
    HeavyCalculation();
    last_calc = TimeGMT();
}
```

Limite pratique : `OnTick` doit retourner en < 200ms.

---

## 2. `TimeCurrent()` au lieu de `TimeGMT()`

**Wrong** :
```mql5
MqlDateTime tm;
TimeToStruct(TimeCurrent(), tm);
if(tm.hour >= 6 && tm.hour < 14) /* session */;  // ← drift DST
```

**Right** :
```mql5
TimeToStruct(TimeGMT(), tm);
if(tm.hour >= 6 && tm.hour < 14) /* session */;
```

`TimeCurrent` retourne l'heure broker — variable selon DST.

---

## 3. Lire `shift = 0` (la barre courante en formation)

**Wrong** :
```mql5
double buf[];
CopyBuffer(h_ema20, 0, 0, 1, buf);  // ← shift=0, valeur change à chaque tick
```

**Right** :
```mql5
CopyBuffer(h_ema20, 0, 1, 1, buf);  // ← shift=1, dernière barre fermée
```

---

## 4. `OrderSend` sans check du retcode

**Wrong** :
```mql5
OrderSend(req, res);
Print("Order sent");  // ← peut être un échec silencieux
```

**Right** :
```mql5
if(!OrderSend(req, res) || res.retcode != TRADE_RETCODE_DONE)
{
    PrintFormat("OrderSend failed: code=%d desc=%s",
                res.retcode, _LastError);
    return;
}
```

---

## 5. SL/TP trop proches du prix → retcode 10016

**Wrong** :
```mql5
req.sl = price - 1 * _Point;  // ← 1 point seulement, broker rejette
```

**Right** :
```mql5
double stops_level = SymbolInfoInteger(_Symbol, SYMBOL_TRADE_STOPS_LEVEL) * _Point;
double sl_dist = MathMax(stops_level + _Point, desired_sl_distance);
req.sl = price - sl_dist;
```

---

## 6. `PositionsTotal` confondu avec `OrdersTotal`

`OrdersTotal` = ordres pendants (non exécutés).
`PositionsTotal` = positions ouvertes.

**Wrong** :
```mql5
if(OrdersTotal() == 0)  // ← teste les pendants, pas les positions
    OpenLong();
```

**Right** :
```mql5
if(PositionsTotal() == 0)
    OpenLong();
```

---

## 7. Suffix broker ignoré

**Wrong** :
```mql5
SymbolSelect("EURUSD", true);  // ← échoue si broker = "EURUSDm"
```

**Right** :
```mql5
input string Inp_SymbolSuffix = "";
SymbolSelect("EURUSD" + Inp_SymbolSuffix, true);
```

---

## 8. Lots non normalisés (multiple de `VOLUME_STEP`)

**Wrong** :
```mql5
double lots = risk_money / sl_in_points / tick_value;  // ← peut donner 0.123456
trade.Buy(lots, ...);  // ← rejected: invalid volume
```

**Right** :
```mql5
double step = SymbolInfoDouble(symbol, SYMBOL_VOLUME_STEP);
lots = MathFloor(lots / step) * step;
double minv = SymbolInfoDouble(symbol, SYMBOL_VOLUME_MIN);
double maxv = SymbolInfoDouble(symbol, SYMBOL_VOLUME_MAX);
lots = MathMax(minv, MathMin(maxv, lots));
```

---

## 9. Itération `i++` pendant fermeture de positions

**Wrong** :
```mql5
for(int i = 0; i < PositionsTotal(); i++)  // ← total change à chaque close
{
    ulong t = PositionGetTicket(i);
    trade.PositionClose(t);
}
```

**Right** :
```mql5
for(int i = PositionsTotal() - 1; i >= 0; i--)
{
    ulong t = PositionGetTicket(i);
    trade.PositionClose(t);
}
```

---

## 10. `IndicatorRelease` oublié

**Wrong** :
```mql5
int OnInit() { g_h = iMA(...); return INIT_SUCCEEDED; }
// pas de OnDeinit → handle leak après chaque rechargement
```

**Right** :
```mql5
void OnDeinit(const int reason)
{
    if(g_h != INVALID_HANDLE) IndicatorRelease(g_h);
}
```

---

## 11. ddof=0 vs ddof=1 dans rolling std

`pandas.std()` → ddof=1 (n-1).
`numpy.std()` → ddof=0 (n).

**Pas un piège MQL5** mais un piège de portage Python → MQL5 : vérifier que ta formule
match celle utilisée côté Python.

---

## 12. ArraySetAsSeries inverse l'indexation

**Wrong** :
```mql5
double buf[];
CopyClose(_Symbol, PERIOD_D1, 0, 100, buf);
ArraySetAsSeries(buf, true);
double yesterday = buf[0];  // ← non, c'est aujourd'hui (en formation)
```

**Right (consistent)** :
```mql5
// Convention : ne PAS utiliser ArraySetAsSeries — laisser ordre chronologique
double buf[];
CopyClose(_Symbol, PERIOD_D1, 1, 100, buf);
double oldest    = buf[0];
double yesterday = buf[ArraySize(buf) - 1];
```

---

## 13. `FileOpen` sans `FILE_COMMON` mais fichier dans `Common`

**Wrong** :
```mql5
int h = FileOpen("macro_cache.csv", FILE_READ | FILE_CSV);  // cherche dans MQL5/Files
```

**Right** :
```mql5
int h = FileOpen("macro_cache.csv", FILE_READ | FILE_CSV | FILE_COMMON, ',');
```

---

## 14. `EventSetTimer(0)` ou < 1s

**Wrong** :
```mql5
EventSetTimer(0);  // ← invalide
```

**Right** :
```mql5
EventSetTimer(60);
// ou pour haute fréquence :
EventSetMillisecondTimer(500);
```

---

## 15. Magic number partagé entre EAs

Si plusieurs EAs tournent sur le même compte avec le même magic, ils vont se voler
les positions.

**Right** : magic distinct par sleeve, par EA, par stratégie.

```mql5
#define MAGIC_MR_MACRO     831
#define MAGIC_TS_MOMENTUM  832
#define MAGIC_RSI_DAILY    833
```

---

## 16. `iMA` `applied_price` mismatch

**Wrong** :
```mql5
iMA(_Symbol, PERIOD_D1, 20, 0, MODE_EMA, PRICE_OPEN);  // ← lit l'ouverture, pas le close
```

**Right** :
```mql5
iMA(_Symbol, PERIOD_D1, 20, 0, MODE_EMA, PRICE_CLOSE);
```

---

## 17. Spread bid/ask non pris en compte dans le sizing

`SymbolInfoDouble(_Symbol, SYMBOL_BID)` ≠ `SymbolInfoDouble(_Symbol, SYMBOL_ASK)`.
- Buy : on paye `ASK`
- Sell : on reçoit `BID`

Le SL/TP doit tenir compte du sens.

---

## 18. `NormalizeDouble` oublié

**Wrong** :
```mql5
req.price = price + 0.000001;  // ← 7 décimales sur EURUSD 5 chiffres → invalid price
```

**Right** :
```mql5
req.price = NormalizeDouble(price + diff, _Digits);
```

---

## 19. `Sleep` dans le testeur

`Sleep` est ignoré dans le Strategy Tester (pas de temps réel). Donc tout retry loop
basé sur `Sleep` ne fait rien. Solution : utiliser `OnTick` repeated checks.

---

## 20. Capital trop faible → lots = MIN_VOLUME surdimensionne

Si `risk_money / sl_distance / tick_value` retourne `0.005` mais `SYMBOL_VOLUME_MIN`
est `0.01`, on est forcé à `0.01` qui correspond à 2× le risk visé.

**Mitigation** : refuser le trade si `lots > 1.5 * raw_lots` (signal de capital trop faible).

---

## Voir aussi

- [troubleshooting.md](./troubleshooting.md) — codes d'erreur diagnostiqués
- [03_trade_operations.md](./03_trade_operations.md) — patterns OrderSend
