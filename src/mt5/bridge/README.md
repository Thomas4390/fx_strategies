# Bridge macro Python → MT5

Le sleeve MR Macro (80% allocation) dépend de deux séries macro publiques :
1. **Treasury yield spread 10Y − 2Y** (daily) — règle : `spread < 0.5`
2. **US unemployment rate** (mensuel) — règle : variation 3m **non haussière**

MetaTrader 5 ne sait pas accéder à ces séries (ni FRED API ni parquet). Ce bridge
sert de relais : il calcule offline le filtre macro et écrit un fichier CSV que
l'EA `FxMultiSleeve` lit via `FileOpen(FILE_COMMON)`.

## Pré-requis

- Python 3.10+
- Les parquets `data/SPREAD_10Y2Y_daily.parquet` et
  `data/UNEMPLOYMENT_monthly.parquet` à jour (cf. `src/data/` et le pipeline
  d'ingestion FRED du repo)

## Installation

```bash
pip install pandas pyarrow
```

## Usage manuel

```bash
python src/mt5/bridge/fx_macro_bridge.py
```

Sortie attendue :

```
OK wrote /home/.../MetaQuotes/Terminal/Common/Files/macro_cache.csv: 2026-04-24T18:00:00Z,0.3520,0,0.5000,1
```

## Schéma du fichier `macro_cache.csv`

```
timestamp_utc,spread_10y2y,unemp_rising,spread_threshold,macro_ok
2026-04-24T18:00:00Z,0.352000,0,0.5000,1
```

| Colonne | Type | Description |
|---------|------|-------------|
| `timestamp_utc` | ISO 8601 UTC | Moment de génération |
| `spread_10y2y` | float | Dernière valeur du spread |
| `unemp_rising` | 0/1 | 1 si `unemployment[-1] − unemployment[-4] > 0` |
| `spread_threshold` | float | Seuil utilisé (par défaut 0.5) |
| `macro_ok` | 0/1 | `(spread < threshold) AND NOT unemp_rising` |

## Schedule cron (Linux/Wine)

```cron
# Run hourly at minute 0
0 * * * * /usr/bin/python3 /chemin/vers/fx_strategies/src/mt5/bridge/fx_macro_bridge.py >> /tmp/fx_macro_bridge.log 2>&1
```

## Schedule Task Scheduler (Windows)

1. Action : `Démarrer un programme`
2. Programme : `python.exe`
3. Arguments : `C:\chemin\vers\fx_strategies\src\mt5\bridge\fx_macro_bridge.py`
4. Déclencheur : `Quotidien` toutes les 1 heure

## Comportement en cas d'erreur

- Si la lecture parquet échoue → exit 1 sans écraser le fichier (l'EA verra le cache
  existant comme stale au-delà de 24h)
- L'EA détecte un cache vieux de plus de `Inp_MacroMaxAgeHours` (24h par défaut) et :
  - Refuse les nouvelles entrées MR Macro
  - Ferme les positions MR Macro existantes
  - Émet `Alert("FX MR Macro: macro cache stale")`

## Override de la destination

Par défaut le bridge cherche `~/.wine/drive_c/users/<u>/AppData/Roaming/MetaQuotes/Terminal/Common/Files/`
(Linux/Wine) ou l'équivalent Windows. Pour forcer un chemin :

```bash
python fx_macro_bridge.py --output /chemin/explicite/macro_cache.csv
```

L'EA correspondant doit pointer vers le même chemin (input `Inp_MacroCacheFile`,
flag `FILE_COMMON` ou non selon où on écrit).

## Tester sans MT5

```bash
python fx_macro_bridge.py --output /tmp/macro_cache.csv
cat /tmp/macro_cache.csv
```
