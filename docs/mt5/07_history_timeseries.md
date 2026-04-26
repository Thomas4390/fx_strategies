# 07 — History & Time Series

Accéder aux barres historiques, gérer le temps (UTC, DST, sessions), détecter de
nouvelles barres.

## Fonctions de copie

```mql5
int CopyRates(string symbol, ENUM_TIMEFRAMES tf, int start_pos, int count, MqlRates &dst[]);
int CopyRates(string symbol, ENUM_TIMEFRAMES tf, datetime from, datetime to, MqlRates &dst[]);
int CopyRates(string symbol, ENUM_TIMEFRAMES tf, datetime from, int count, MqlRates &dst[]);

int CopyClose(string symbol, ENUM_TIMEFRAMES tf, int start_pos, int count, double &dst[]);
int CopyOpen / CopyHigh / CopyLow / CopyTickVolume / CopyTime / CopySpread (idem)
```

Retour : nombre copié, ou `-1` si erreur.

## MqlRates struct

```mql5
struct MqlRates
{
    datetime time;          // open time of the bar (UTC seconds since epoch)
    double   open, high, low, close;
    long     tick_volume;   // number of ticks in the bar
    int      spread;        // average spread in points
    long     real_volume;   // 0 sur retail FX
};
```

## Pattern NewBar — détecter une nouvelle barre fermée

```mql5
bool IsNewBar(ENUM_TIMEFRAMES tf, datetime &last_bar_inout)
{
    datetime current = iTime(_Symbol, tf, 0);
    if(current == last_bar_inout) return false;
    last_bar_inout = current;
    return true;
}

void OnTick()
{
    static datetime last_m1 = 0;
    if(!IsNewBar(PERIOD_M1, last_m1)) return;
    // Ici la barre M1 précédente vient de fermer — lire shift=1 pour signal causal
}
```

## Lire la dernière barre **fermée**

```mql5
MqlRates last[];
if(CopyRates(_Symbol, PERIOD_M1, 1, 1, last) == 1)
{
    double close = last[0].close;
    // ...
}
```

`start_pos = 1` : la dernière barre **fermée**.
`start_pos = 0` : la barre courante (encore en formation, valeurs changent).

## ArraySetAsSeries — ordre chronologique

Par défaut, les arrays MQL5 sont indexés du **plus ancien au plus récent**
(`buf[0]` = bar la plus ancienne).

Pour inverser (style "iCustom" historique, `buf[0]` = bar la plus récente) :
```mql5
double buf[];
ArraySetAsSeries(buf, true);
CopyClose(_Symbol, PERIOD_D1, 0, 100, buf);
// Maintenant buf[0] = aujourd'hui, buf[1] = hier, etc.
```

**Convention recommandée** : `ArraySetAsSeries(arr, false)` (l'ordre par défaut), c'est
plus intuitif et cohérent avec `MqlRates`.

## Time : UTC vs broker time vs local

| Fonction | Description |
|----------|-------------|
| `TimeGMT()` | Temps UTC réel (corrigé DST côté serveur OS) — **à utiliser pour les sessions** |
| `TimeCurrent()` | Temps du dernier tick reçu — peut être en retard, et exprimé en heure broker |
| `TimeLocal()` | Temps local de la machine — déconseillé |
| `TimeTradeServer()` | Temps serveur broker |

**Règle** : pour toute logique de session UTC (ex. ouvrir/fermer entre 6h–14h), utiliser
`TimeGMT()`. Cela évite les drifts DST.

## Décomposer un datetime

```mql5
datetime now = TimeGMT();
MqlDateTime st;
TimeToStruct(now, st);
// st.year, st.mon, st.day, st.hour, st.min, st.sec, st.day_of_week, st.day_of_year
if(st.hour >= 6 && st.hour < 14)
{
    // Session active
}
```

## Floor à minuit UTC

```mql5
datetime now = TimeGMT();
datetime midnight_utc = now - (now % 86400);
```

## Calcul de durée

```mql5
datetime opened = (datetime)PositionGetInteger(POSITION_TIME);
datetime now    = TimeGMT();
int hours_open  = (int)((now - opened) / 3600);
if(hours_open >= 6) trade.PositionClose(ticket);
```

## HistorySelect — pour les deals passés

Pour itérer sur les deals (équivalent "trade history") :

```mql5
HistorySelect(TimeGMT() - 86400 * 90, TimeGMT());  // 90 derniers jours
int deals = HistoryDealsTotal();
for(int i = 0; i < deals; i++)
{
    ulong ticket = HistoryDealGetTicket(i);
    if(ticket == 0) continue;
    long magic = HistoryDealGetInteger(ticket, DEAL_MAGIC);
    double profit = HistoryDealGetDouble(ticket, DEAL_PROFIT);
    datetime time = (datetime)HistoryDealGetInteger(ticket, DEAL_TIME);
    // ...
}
```

Utilisé dans le projet pour reconstruire les returns daily de l'equity (vol-targeting global).

## Annualization factors

| TF | Bars / an | Annualization |
|----|-----------|---------------|
| D1 | 252 | `sqrt(252)` |
| H1 | 252 × 24 | `sqrt(252 × 24)` |
| M1 (24h) | 252 × 1440 | `sqrt(252 × 1440)` |
| M1 (FX 6.5h trading) | 252 × 6.5 × 60 = 100,920 | `sqrt(100920)` |

Le code Python utilise `FX_MINUTE_ANN_FACTOR = 100920` pour le sleeve 1.

## DST shifts

Brokers MT5 sont souvent en GMT+2 hiver / GMT+3 été. Cela cause :
- Des barres D1 décalées (close à 21:00 ou 22:00 selon la saison)
- Des heures de session qui drift

**Solution** : `TimeGMT()` partout, et exprimer les sessions en UTC pur.

## Voir aussi

- [02_event_handlers.md](./02_event_handlers.md) — pattern NewBar dans OnTick
- [10_pitfalls.md](./10_pitfalls.md) — DST, ArraySetAsSeries
- [05_indicators_custom.md](./05_indicators_custom.md) — buffer circulaire
