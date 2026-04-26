# 13 — Sources de données natives MQL5

Comment se passer d'un bridge externe (Python, scraper) en accédant aux données
macro directement depuis MQL5. Trois APIs principales :

| API | Utilité | Limite |
|-----|---------|--------|
| `Calendar*` | Événements économiques publiés (NFP, CPI, Unemployment Rate, etc.) | Uniquement les indicateurs publiés à des dates précises |
| `WebRequest` | Requêtes HTTP/HTTPS arbitraires vers une API externe | Synchrone bloquant, URL whitelistée manuellement |
| Custom Symbols | Créer un symbole tradable alimenté par une source externe | Setup complexe, peu de cas d'usage hors price data |

## 1. Calendar économique MT5

MT5 maintient un calendar économique synchronisé avec le serveur MetaQuotes. Les
événements sont taggués par pays, importance, et type. C'est **idéal pour les
publications datées** comme :
- US Unemployment Rate
- Non-Farm Payrolls (NFP)
- CPI (inflation)
- Fed Funds Rate decisions
- ECB rate decisions

### Lister les événements d'un pays

```mql5
MqlCalendarEvent events[];
int n = CalendarEventByCountry("US", events);
for(int i = 0; i < n; i++)
    PrintFormat("id=%I64u name=%s impact=%d",
                events[i].id, events[i].name, events[i].importance);
```

Champs intéressants de `MqlCalendarEvent` :
- `id` (ulong) — identifiant unique de l'événement
- `name` (string) — ex. "Unemployment Rate"
- `country_id` (ulong) — code pays
- `importance` (`ENUM_CALENDAR_EVENT_IMPORTANCE`) — Low / Moderate / High
- `frequency` (`ENUM_CALENDAR_EVENT_FREQUENCY`) — Monthly, Weekly, etc.

### Récupérer les valeurs publiées

```mql5
MqlCalendarValue values[];
datetime from = TimeGMT() - 86400 * 30 * 18;  // 18 mois
datetime to   = TimeGMT();
int n = CalendarValueHistoryByEvent(event_id, values, from, to);
```

`MqlCalendarValue` clés :
- `actual_value` (long) — valeur publiée × 1e6 (ex. 3.7% → `3700000`)
- `prev_value`, `revised_prev_value`, `forecast_value` (long, × 1e6)
- `time` (datetime) — moment de publication
- `period` (datetime) — période concernée
- `impact_type` (`ENUM_CALENDAR_EVENT_IMPACT`)
- Méthode `HasActualValue()` → true si la valeur a été publiée (sinon `LONG_MAX`)

Exemple complet — variation 3 mois du taux de chômage US :

```mql5
ulong event_id = ResolveUnemploymentEventId();  // cf. CalendarEventByCountry
MqlCalendarValue vals[];
CalendarValueHistoryByEvent(event_id, vals, TimeGMT() - 86400*30*18, TimeGMT());

double last4[4]; int k = 0;
for(int i = ArraySize(vals) - 1; i >= 0 && k < 4; i--)
{
    if(!vals[i].HasActualValue()) continue;
    last4[3 - k] = (double)vals[i].actual_value / 1e6;
    k++;
}
if(k == 4)
{
    bool unemp_rising = (last4[3] - last4[0]) > 0.0;
}
```

### Pré-requis

Aucun. Le calendar est natif. Synchronisation automatique au démarrage du
terminal (peut prendre quelques secondes la première fois).

### Limitations

- Pas tous les pays / indicateurs disponibles selon la version MT5
- Les données historiques peuvent être tronquées (selon MetaQuotes)
- En **Strategy Tester**, le calendar peut être vide ou figé — vérifier si
  c'est un blocker pour le backtest

## 2. WebRequest

API HTTP/HTTPS pour appeler n'importe quelle API JSON/REST externe.

### Pré-requis CRITIQUE — URL whitelist

Doc : `https://www.mql5.com/en/docs/network/webrequest`.

Avant qu'un EA puisse appeler `WebRequest` vers `https://api.example.com`, **un
opérateur humain doit ajouter l'URL** dans :

```
Terminal MT5 → Outils → Options → Onglet "Expert Advisors"
  → cocher "Allow WebRequest for listed URL"
  → ajouter : https://api.stlouisfed.org
```

Sans ça, `WebRequest` retourne `-1` avec `_LastError = ERR_FUNCTION_NOT_CONFIRMED (4014)`.

### Signature

```mql5
int WebRequest(
    const string method,         // "GET" ou "POST"
    const string url,
    const string cookie,         // ou NULL
    const string referer,        // ou NULL
    int          timeout_ms,
    const char  &data[],         // body POST (vide pour GET)
    int          data_size,
    char        &result[],       // body de la réponse
    string      &result_headers
);
// Retour : code HTTP (200, 404…), ou -1 en erreur
```

Variante avec headers personnalisés :

```mql5
int WebRequest(
    const string method,
    const string url,
    const string headers,        // "Content-Type: application/json\r\n..."
    int          timeout_ms,
    const char  &data[],
    char        &result[],
    string      &result_headers
);
```

### Exemple : FRED API

```mql5
string url = "https://api.stlouisfed.org/fred/series/observations"
             "?series_id=T10Y2Y&api_key=YOUR_KEY"
             "&file_type=json&limit=1&sort_order=desc";

char post[], result[];
string headers;
int code = WebRequest("GET", url, NULL, NULL, 5000, post, 0, result, headers);
if(code == 200)
{
    string body = CharArrayToString(result, 0, WHOLE_ARRAY, CP_UTF8);
    // Parser le JSON manuellement (pas de json natif en MQL5)
}
else if(code == -1)
{
    PrintFormat("WebRequest err=%d (URL whitelistée ?)", GetLastError());
}
```

### Parsing JSON sans librairie native

MQL5 n'a **pas** de parser JSON natif (à mai 2026). Pour les besoins simples :

```mql5
// Cherche "value":"X.XX" dans le body
int idx = StringFind(body, "\"value\":\"");
if(idx >= 0)
{
    int start = idx + StringLen("\"value\":\"");
    int end   = StringFind(body, "\"", start);
    string v = StringSubstr(body, start, end - start);
    double val = StringToDouble(v);
}
```

Pour des JSONs complexes, utiliser une librairie tierce
(ex. `JAson.mqh` distribuée sur la marketplace MQL5) ou un bridge externe.

### Limitations

- **Synchrone bloquant** : `WebRequest` bloque le thread EA jusqu'à réponse ou
  timeout. **Jamais à appeler dans `OnTick`** — uniquement dans `OnTimer`.
- Pas de WebSocket, pas de streaming, pas de SSE.
- Pas de TLS client cert custom (mTLS non supporté).
- En **Strategy Tester**, `WebRequest` retourne `-1` — pas d'accès réseau en
  backtest.

## 3. Custom Symbols (mention pour mémoire)

`SymbolCreate`, `CustomRatesUpdate`, etc. permettent de créer des symboles
synthétiques alimentés depuis l'extérieur. Utile si on veut tradeur un
synthétique (panier de devises, spread synthétique). Pas adapté pour stocker des
séries macro consommées en lecture seule.

## Architecture utilisée dans ce projet

`CMacroFilter` supporte 3 modes via `EMacroSourceMode` :

| Mode | Comportement |
|------|--------------|
| `MACRO_SOURCE_FILE` | Lit `macro_cache.csv` produit par `bridge/fx_macro_bridge.py` |
| `MACRO_SOURCE_NATIVE` | Calendar (chômage US) + WebRequest FRED (spread T10Y2Y) |
| `MACRO_SOURCE_HYBRID` | Tente NATIVE, fallback FILE en cas d'échec |

Le mode `HYBRID` est le plus robuste : il préfère les sources natives quand
elles sont disponibles, sans abandonner la stratégie en cas de panne API ou
backtest.

### Stockage de la clé API FRED

**Ne jamais** mettre la clé en `input` visible — exposée à quiconque peut
inspecter l'EA. Convention :

1. Créer un fichier texte `fred_api_key.txt` contenant uniquement la clé
2. Le placer dans `Common/Files/` (par défaut, partagé entre terminaux)
3. L'EA le lit via `FileOpen(name, FILE_READ | FILE_TXT | FILE_COMMON)`

### Choix du mode selon le contexte

| Contexte | Mode recommandé |
|----------|-----------------|
| Production live | `HYBRID` (sécurité maxi) |
| Backtest Strategy Tester | `FILE` (WebRequest indispo) |
| Démo broker | `NATIVE` (validation chemin) |
| Pas d'API key FRED | `FILE` (le bridge Python suffit) |

## Voir aussi

- [doc officielle Calendar](https://www.mql5.com/en/docs/calendar)
- [doc officielle WebRequest](https://www.mql5.com/en/docs/network/webrequest)
- [src/mt5/Include/FxMacroSourceNative.mqh](../../src/mt5/Include/FxMacroSourceNative.mqh)
- [src/mt5/Include/FxMacroFilter.mqh](../../src/mt5/Include/FxMacroFilter.mqh)
- [src/mt5/bridge/fx_macro_bridge.py](../../src/mt5/bridge/fx_macro_bridge.py) — bridge Python (mode FILE)
