# 14 — CLI Backtest sur Linux/Wine

> Comment piloter MT5 entièrement en CLI depuis un agent ou un script (ouvrir, compiler, backtester, fermer) sur **Ubuntu / Wine**. Complète [09_strategy_tester.md](./09_strategy_tester.md) qui couvre le flux GUI.

## Pourquoi ce doc

- Le poste de travail courant tourne **Ubuntu 26 / Wine 10.0** (réinstallé 2026-05-02), pas Windows. Le `src/mt5/CLAUDE.md` est encore Windows-centric.
- Pour automatiser **walk-forward analysis** (cf. TODO #9 de `src/mt5/SESSION_NOTES.md`), un script Python doit pouvoir lancer N backtests sans interaction GUI.
- Un agent (Claude Code, etc.) doit pouvoir ouvrir, fermer, compiler, et backtester sans assistance humaine.

État vérifié le **2026-05-02** :

| Étape | Statut | Preuve |
|---|---|---|
| Wine + MT5 binaires | ✅ | `wine-10.0`, `terminal64.exe` build 5836 |
| Ouverture / fermeture sans GUI | ✅ | `pgrep` voit MT5, `pkill -f terminal64.exe` ferme |
| Compilation `.mq5` → `.ex5` | ✅ | 0 errors, 0 warnings, 773 ms sur `FxMultiSleeve.mq5` |
| Lancement Strategy Tester via `/config:` | ✅ | EA chargé, rapport HTML généré, `ShutdownTerminal=1` ferme proprement |
| Backtest avec résultats numériques | ⚠️ | Pipeline OK mais `EURUSD.c: history check timeout` — historique pas pré-téléchargé (cf. § *Pré-requis données*) |

## Environnement Linux/Wine

### Versions

```bash
$ wine --version
wine-10.0 (Ubuntu 10.0~repack-12ubuntu1)
$ # MT5 build : 5836 (visible dans logs/<date>.log)
```

### Mode portable activé

⚠️ **Sur ce poste, MT5 tourne en mode portable** (présence de `portable.txt`). Conséquence : tout vit sous `/home/thomas/.mt5/drive_c/Program Files/MetaTrader 5/` et **PAS** sous `~/.mt5/drive_c/users/thomas/AppData/Roaming/MetaQuotes/Terminal/<HASH>/`.

```bash
ls /home/thomas/.mt5/drive_c/Program\ Files/MetaTrader\ 5/portable.txt   # exists → portable mode
```

Pour vérifier où MT5 écrit ses données :

```bash
find /home/thomas/.mt5 -maxdepth 5 -name "logs" -type d
# Quand portable : /home/thomas/.mt5/drive_c/Program Files/MetaTrader 5/{logs,MQL5/logs,Tester/logs}
# Sinon          : /home/thomas/.mt5/drive_c/users/thomas/AppData/Roaming/MetaQuotes/Terminal/<HASH>/{logs,...}
```

### Chemins critiques (mode portable)

| Rôle | Chemin Linux | Chemin Wine (Windows-style) |
|---|---|---|
| Binaire principal | `~/.mt5/drive_c/Program Files/MetaTrader 5/terminal64.exe` | `C:\Program Files\MetaTrader 5\terminal64.exe` |
| Compilateur | `~/.mt5/drive_c/Program Files/MetaTrader 5/MetaEditor64.exe` | `C:\Program Files\MetaTrader 5\MetaEditor64.exe` |
| MQL5 (EAs/Includes) | `~/.mt5/drive_c/Program Files/MetaTrader 5/MQL5/{Experts,Include,Scripts}/` | `C:\Program Files\MetaTrader 5\MQL5\...` |
| Common Files (FILE_COMMON) | `~/.mt5/drive_c/Program Files/MetaTrader 5/Common/Files/` | `C:\Program Files\MetaTrader 5\Common\Files\` |
| Log Terminal | `~/.mt5/drive_c/Program Files/MetaTrader 5/logs/YYYYMMDD.log` | `C:\Program Files\MetaTrader 5\logs\...` |
| Log EA (live) | `~/.mt5/drive_c/Program Files/MetaTrader 5/MQL5/logs/YYYYMMDD.log` | `…\MQL5\logs\…` |
| Log Tester | `~/.mt5/drive_c/Program Files/MetaTrader 5/Tester/logs/YYYYMMDD.log` | `…\Tester\logs\…` |
| `accounts.dat` (broker) | `~/.mt5/drive_c/Program Files/MetaTrader 5/Config/accounts.dat` | `C:\Program Files\MetaTrader 5\Config\accounts.dat` |
| Bases (historique) | `~/.mt5/drive_c/Program Files/MetaTrader 5/Bases/<ServerName>/` | `…\Bases\<ServerName>\` |
| Profiles tester preset | `~/.mt5/drive_c/Program Files/MetaTrader 5/MQL5/Profiles/Tester/<EA>.set` | `…\MQL5\Profiles\Tester\…` |
| Source repo | `/home/thomas/Documents_Thomas/11_CodingProjects/fx_strategies/fx_strategies/src/mt5/` | (accessible via `Z:\home\thomas\…`) |

### Symlinks `fx_strategies` requis dans le MQL5 portable

Le tester cherche `Experts\fx_strategies\FxMultiSleeve.ex5`. Sans le symlink → `Experts\fx_strategies\FxMultiSleeve.ex5 not found`. Créer une fois :

```bash
PORTABLE_MQL5="/home/thomas/.mt5/drive_c/Program Files/MetaTrader 5/MQL5"
SRC="/home/thomas/Documents_Thomas/11_CodingProjects/fx_strategies/fx_strategies/src/mt5"
ln -sfn "$SRC/Experts" "$PORTABLE_MQL5/Experts/fx_strategies"
ln -sfn "$SRC/Include" "$PORTABLE_MQL5/Include/fx_strategies"
ln -sfn "$SRC/Scripts" "$PORTABLE_MQL5/Scripts/fx_strategies"
```

> Note : `src/mt5/CLAUDE.md` documente d'anciens symlinks dans `…/AppData/Roaming/…/MQL5/`. Ceux-là servent quand MT5 tourne **sans** `/portable`. Les deux peuvent coexister.

## Flags CLI officiels (MetaQuotes)

| Flag | Effet | Source |
|---|---|---|
| `/config:<path>` | Charge un fichier de config au démarrage. Pour le Tester : section `[Tester]` requise. | [metatrader5.com](https://www.metatrader5.com/en/terminal/help/start_advanced/start) |
| `/portable` | Force le data folder à `<install>` au lieu de `AppData`. Crée `portable.txt`. | idem |
| `/profile:<name>` | Charge un profil chart prédéfini. | idem |
| `/login:<account>` | Pré-remplit le numéro de compte broker. Le password reste géré via `accounts.dat`. | idem |
| `MetaEditor64.exe /compile:<path>` | Compile un `.mq5`. Ajouter `/log` pour générer un `<name>.log` UTF-16 à côté du source. | testé OK |

## Format `tester.ini` (vérifié)

**Encodage critique** : UTF-16 LE **avec BOM** + terminaisons **CRLF**. Sans cela, MT5 affiche `cannot load config "..." at start` et lance le terminal en mode normal au lieu de tester.

```ini
[Tester]
Expert=fx_strategies\FxMultiSleeve.ex5
Symbol=EURUSD.c
Period=M1
Model=1
FromDate=2024.06.01
ToDate=2024.06.30
Deposit=10000
Currency=USD
Leverage=1:30
Optimization=0
Visual=0
ShutdownTerminal=1
Report=fx_cli_smoke_report
ReplaceReport=1

[TesterInputs]
Inp_SymbolSuffix=.c
Inp_MacroSourceMode=3
Inp_LogVerbose=true
```

| Clé | Type | Valeurs | Note |
|---|---|---|---|
| `Expert` | path | path relatif depuis `MQL5\Experts\` | ⚠️ chemins Windows `\` requis |
| `Symbol` | str | ex: `EURUSD.c` | doit exister dans MarketWatch broker |
| `Period` | str | `M1`, `M5`, `M15`, `M30`, `H1`, `H4`, `D1`, `W1`, `MN1` | défaut `H1` si absent |
| `Model` | int | 0=Every tick, 1=1 minute OHLC, 2=Open prices only, 3=Math, 4=Real ticks | 1 ou 2 pour smoke ; 4 pour final |
| `FromDate` / `ToDate` | YYYY.MM.DD | période de test | |
| `Deposit` | int | dépôt initial | |
| `Currency` | str | `USD`, `EUR`, … | |
| `Leverage` | str | `1:30`, `1:100` | |
| `Optimization` | int | 0=disabled, 1=Slow complete, 2=Fast genetic, 3=All MarketWatch | |
| `Visual` | int | 0=non-visuel (rapide), 1=visuel | |
| `ShutdownTerminal` | int | 0=garde le terminal ouvert, 1=ferme MT5 après test | **mettre à 1** pour CLI |
| `Report` | str | nom (sans extension) du rapport HTML | écrit dans le data folder |
| `ReplaceReport` | int | 1 pour overwrite | |

`[TesterInputs]` accepte `key=value` ou `key=value\|\|value\|\|lo\|\|hi\|\|N` (forme étendue pour optimisation, cf. `src/mt5/bridge/reset_tester_preset.py:84`).

## Commandes vérifiées (copier-coller)

### 0) Pré-requis : data + symlinks

```bash
WINEPREFIX=/home/thomas/.mt5
PORTABLE="$WINEPREFIX/drive_c/Program Files/MetaTrader 5"

# Une seule fois — symlinks fx_strategies dans MQL5 portable
ln -sfn "$REPO/src/mt5/Experts" "$PORTABLE/MQL5/Experts/fx_strategies"
ln -sfn "$REPO/src/mt5/Include" "$PORTABLE/MQL5/Include/fx_strategies"
ln -sfn "$REPO/src/mt5/Scripts" "$PORTABLE/MQL5/Scripts/fx_strategies"
```

### 1) Compiler un `.mq5`

```bash
WINEPREFIX=/home/thomas/.mt5 wine \
  "/home/thomas/.mt5/drive_c/Program Files/MetaTrader 5/MetaEditor64.exe" \
  /compile:"Z:\\home\\thomas\\Documents_Thomas\\11_CodingProjects\\fx_strategies\\fx_strategies\\src\\mt5\\Experts\\FxMultiSleeve.mq5" \
  /log
# Le log est écrit à côté du .mq5 en UTF-16 LE.
# Lecture : iconv -f UTF-16LE -t UTF-8 …/FxMultiSleeve.log | tail -3
# Cible : "Result: 0 errors, 0 warnings, … ms"
```

> Notes :
> - `Z:\` est le drive Wine pointant vers `/`. Donc `Z:\home\…` ↔ `/home/…`.
> - `/compile:` accepte uniquement les chemins Windows-style avec `\` (ou `/`), JAMAIS `\\` doublé.

### 2) Ouvrir / fermer MT5

```bash
# Lancement (foreground bloquant — utiliser & pour background)
WINEPREFIX=/home/thomas/.mt5 wine \
  "/home/thomas/.mt5/drive_c/Program Files/MetaTrader 5/terminal64.exe" /portable &

# Vérifier
pgrep -af terminal64.exe

# Fermer proprement
pkill -f terminal64.exe
```

### 3) Lancer un backtest CLI

⚠️ **Le `/config:` est très sensible aux quotes** quand le path contient des espaces. Méthode robuste : copier le `.ini` dans un chemin sans espaces (`C:\<name>.ini`).

```bash
# Copie du .ini dans drive_c root (chemin sans espaces)
cp "$PORTABLE/Config/fx_cli_smoke.ini" /home/thomas/.mt5/drive_c/fxc.ini

# Lancement (60-90s pour 1 mois D1, plusieurs minutes en M1 real ticks)
timeout 360 env WINEPREFIX=/home/thomas/.mt5 wine \
  "/home/thomas/.mt5/drive_c/Program Files/MetaTrader 5/terminal64.exe" \
  /portable "/config:C:\fxc.ini"
# exit 0 si test OK, exit 124 si timeout, exit 189 si erreur fatale (broker manquant, etc.)
```

### 4) Lire les résultats

```bash
# Logs UTF-16 LE
iconv -f UTF-16LE -t UTF-8 \
  "$PORTABLE/Tester/logs/$(date +%Y%m%d).log" | tail -30

# Rapport HTML (généré dans le data folder, à la racine portable)
ls "$PORTABLE/fx_cli_smoke_report"*
# fx_cli_smoke_report.htm
# fx_cli_smoke_report-hst.png
# fx_cli_smoke_report-holding.png
# fx_cli_smoke_report-mfemae.png
```

## Pré-requis données : pré-téléchargement de l'historique

Le smoke test du 2026-05-02 a échoué à produire des trades parce que `EURUSD.c: history check timeout` : le broker n'avait pas encore poussé l'historique M1 dans `Bases/`. Solutions :

1. **Manuel (UI)** : ouvrir MT5, `Outils → Historique → EURUSD.c → M1 → Download`. À faire une fois.
2. **Pré-téléchargement programmatique** : avec un script MQL5 utilisant `CopyRates(symbol, PERIOD_M1, time_from, time_to, rates)` sur la période de test, exécuté avant le tester. Voir `src/mt5/Scripts/FxPreflight.mq5` qui fait ça pour les 4 paires.
3. **Backtest plus tolérant** : passer `Period=D1` et `Model=2` (Open prices only) — D1 nécessite ~250 bars seulement, souvent déjà dans le cache broker.

## Encoding `tester.ini` — pourquoi UTF-16 LE BOM CRLF

MT5 lit les `.ini` en UTF-16 LE strict (héritage Windows). En écrivant depuis Python sur Linux, il faut :

```python
content = "[Tester]\nExpert=fx_strategies\\FxMultiSleeve.ex5\n…"
data = ("﻿" + content.replace("\n", "\r\n")).encode("utf-16-le")
ini_path.write_bytes(data)   # ⚠️ write_bytes, PAS write_text (universal newlines)
```

Le pattern existe déjà dans `src/mt5/bridge/reset_tester_preset.py:104` (corrigé d'un bug `\r\r\n` 2026-04-30, commit `11c5d83`). Ne pas inventer un nouveau writer — réutiliser ce pattern.

Vérification de l'encoding écrit :

```bash
$ file fx_cli_smoke.ini
fx_cli_smoke.ini: Unicode text, UTF-16, little-endian text, with CRLF line terminators
```

## Pièges rencontrés (vécus le 2026-05-02)

| Symptôme | Cause | Fix |
|---|---|---|
| `cannot load config "….ini""` (DEUX guillemets) au log Terminal | Bash a injecté un `"` dans l'argument `/config:"…"` quand le path contient des espaces | Copier le `.ini` à `C:\<name>.ini` (drive_c root) et utiliser `/config:C:\name.ini` sans quotes Windows |
| `Experts\fx_strategies\FxMultiSleeve.ex5 not found` au log Tester | Symlink `fx_strategies` absent du MQL5 *portable* (mais présent dans MQL5 user profile) | `ln -sfn` vers `$PORTABLE/MQL5/Experts/fx_strategies` |
| `EURUSD.c: history check timeout` puis `no history data, stop testing` | Bases broker pas peuplées (M1 historique pas encore poussé) | Pré-télécharger via UI ou script `CopyRates` |
| MT5 démarre, n'écrit aucun log, exit 189 en 2s | Pas de connexion broker active (pas de login dans `accounts.dat`) | Login manuel UI une fois → MT5 sauvegarde les credentials |
| `Macro initial load failed (mode=…)` au log live | `macro_history.csv` absent dans `Common/Files/` ou format invalide | `python src/mt5/bridge/fx_macro_history.py` (génère 1800+ lignes 2019-2026) — nécessite `FRED_API_KEY` dans `<repo>/.env` |
| Logs Wine bruyants (`ToolbarWindowProc unknown msg`, `DATETIME_WindowProc`) | Bug Wine 10 sur widgets MT5 | Ignorer — bénin, MT5 fonctionne |
| Window MT5 trop petit (HiDPI) | `xwayland-native-scaling` GNOME Mutter | `gsettings set org.gnome.mutter experimental-features "['scale-monitor-framebuffer']"` (sans `xwayland-native-scaling`) — cf. mémoire S1516 |

## Pipeline reproductible (squelette Python)

À ajouter dans un script type `src/mt5/bridge/run_backtest_cli.py` (futur) :

```python
import subprocess, time
from pathlib import Path

PORTABLE = Path("/home/thomas/.mt5/drive_c/Program Files/MetaTrader 5")
WINEPREFIX = "/home/thomas/.mt5"

def write_tester_ini(dst: Path, **params) -> None:
    """Write a tester.ini in UTF-16 LE BOM with CRLF."""
    content = "[Tester]\n" + "\n".join(f"{k}={v}" for k, v in params.items())
    data = ("﻿" + content.replace("\n", "\r\n") + "\r\n").encode("utf-16-le")
    dst.write_bytes(data)

def run_backtest(ini_simple_path: Path, timeout: int = 360) -> int:
    """Launch MT5 with /config:, return process exit code."""
    wine_path = "C:\\" + ini_simple_path.name  # drive_c root
    return subprocess.call([
        "env", f"WINEPREFIX={WINEPREFIX}", "wine",
        str(PORTABLE / "terminal64.exe"),
        "/portable", f"/config:{wine_path}",
    ], timeout=timeout)

def parse_html_report(htm: Path) -> dict:
    """Parse <td> values from the Strategy Tester HTML (UTF-16 LE)."""
    text = htm.read_text(encoding="utf-16")
    # TODO: extract Sharpe, Max DD, Total Trades, Profit Factor via regex/BeautifulSoup
    ...
```

## Export historique MT5 → Parquet (vérifié 2026-05-02)

Pipeline pour récupérer l'OHLCV brut du broker dans `data/` au format Parquet, naming convention `<PAIR-DASH>_<period>_mt5.parquet` (le suffixe `_mt5` flag la provenance, distinct des Parquets pré-existants `<PAIR-DASH>_<period>.parquet`).

### Composants

- **`src/mt5/Scripts/FxExportRates.mq5`** — script MQL5 qui itère sur (symbols × timeframes) et écrit un CSV par combinaison dans `MQL5\Files\exports\<SYMBOL>_<TF>.csv`. Inputs : `Inp_SymbolsCSV`, `Inp_PeriodsCSV`, `Inp_FromDate`, `Inp_ToDate`, `Inp_SymbolSuffix`. Idempotent (écrase les CSV existants par défaut).
- **`src/mt5/bridge/import_mt5_rates.py`** — convertit les CSV en Parquet `data/<PAIR-DASH>_<period>_mt5.parquet`. Utilise `pandas` + `pyarrow`. Snappy compression. Index = `time` (UTC, datetime64).

### Lancement automatique en CLI (vérifié)

Astuce : la section `[StartUp]` du config MT5 accepte `Script=` qui charge un script MQL5 sur un chart au démarrage du terminal. Combiné avec un `pkill` après délai, on obtient un export 100% headless :

```bash
# 1) Écrire un config minimal qui auto-lance le script
python3 - <<'PY'
from pathlib import Path
content = """[Charts]
PreloadCharts=1

[StartUp]
Profile=Default
Script=fx_strategies\\FxExportRates
Symbol=EURUSD.c
Period=D1
"""
data = ("﻿" + content.replace("\n", "\r\n")).encode("utf-16-le")
Path("/home/thomas/.mt5/drive_c/fxexp.ini").write_bytes(data)
PY

# 2) Préparer le dossier de sortie côté MT5
mkdir -p "/home/thomas/.mt5/drive_c/Program Files/MetaTrader 5/MQL5/Files/exports"

# 3) Lancer MT5 en background, attendre que le script écrive les CSV, fermer
WINEPREFIX=/home/thomas/.mt5 wine \
  "/home/thomas/.mt5/drive_c/Program Files/MetaTrader 5/terminal64.exe" \
  /portable "/config:C:\fxexp.ini" >/tmp/mt5_export.log 2>&1 &
sleep 50                                      # 50s suffisent pour 4 paires × 2 TF
pkill -f terminal64.exe

# 4) Convertir les CSV en Parquet dans data/
cd /path/to/fx_strategies
python3 src/mt5/bridge/import_mt5_rates.py
```

### Résultat 2026-05-02 (broker SquaredFinancialSC-MT5 Demo)

| Fichier | Bars | Période |
|---|---|---|
| `data/EUR-USD_daily_mt5.parquet` | 30 | 2026-03-29 → 2026-05-01 |
| `data/EUR-USD_minute_mt5.parquet` | 34 071 | 2026-03-29 → 2026-05-01 |
| `data/GBP-USD_daily_mt5.parquet` | 1 697 | 2020-11-22 → 2026-05-01 |
| `data/GBP-USD_minute_mt5.parquet` | 98 746 | 2026-01-23 → 2026-05-01 |
| `data/USD-CAD_daily_mt5.parquet` | 1 697 | 2020-11-22 → 2026-05-01 |
| `data/USD-CAD_minute_mt5.parquet` | 98 746 | 2026-01-23 → 2026-05-01 |
| `data/USD-JPY_daily_mt5.parquet` | 1 697 | 2020-11-22 → 2026-05-01 |
| `data/USD-JPY_minute_mt5.parquet` | 98 746 | 2026-01-23 → 2026-05-01 |

> ⚠️ EUR-USD a beaucoup moins d'historique que les autres paires sur ce broker — symbole probablement réintroduit récemment côté SquaredFinancial. Pour EUR-USD historique long, voir les Parquets pré-existants `data/EUR-USD_minute.parquet` (généré par un autre pipeline, FRED/Dukascopy/etc.).

### Format Parquet produit

```
columns: open, high, low, close, tick_volume, spread, real_volume
index  : time (datetime64[ns, UTC])
dtype  : float64 (OHLC), int64 (volumes/spread)
```

### Personnaliser l'export

Modifier les inputs du script — soit en éditant `src/mt5/Scripts/FxExportRates.mq5` et recompilant, soit (mieux) en passant un preset `.set` qui surcharge les valeurs par défaut. Voir le pattern dans `src/mt5/bridge/reset_tester_preset.py` pour générer des `.set` UTF-16 LE.

Exemple : pour exporter aussi H1 et H4 sur EUR/USD seul :
```
Inp_SymbolsCSV = EURUSD
Inp_PeriodsCSV = M1,M5,M15,M30,H1,H4,D1
```

## Pour le prochain agent — to-do prioritaire

1. **Pré-télécharger l'historique** EURUSD.c M1 sur 2024-2026 (via `FxPreflight.mq5` lancé en script depuis le terminal, ou via UI). Sans ça, les backtests CLI sont vides.
2. **Régénérer `macro_history.csv`** : `python src/mt5/bridge/fx_macro_history.py` (clé FRED dans `<repo>/.env`, déjà déployée le 2026-05-02). Sortie attendue : 1800+ lignes 2019-2026, copiée auto dans `Common/Files/`.
3. **Walk-forward CLI** (TODO #9 de SESSION_NOTES) : utiliser le squelette ci-dessus pour piloter N backtests sur N fenêtres, agréger Sharpe IS/OOS.
4. **Vérifier que la connexion broker survit** à un `pkill` + relance CLI — testé OK sur SquaredFinancialSC-MT5 Demo le 2026-05-02 (re-auth automatique via `accounts.dat`).

## Voir aussi

- [09_strategy_tester.md](./09_strategy_tester.md) — flux GUI Strategy Tester (manuel)
- [10_pitfalls.md](./10_pitfalls.md) — pièges génériques MQL5/Tester
- [`src/mt5/CLAUDE.md`](../../src/mt5/CLAUDE.md) — env Windows, broker, codes erreur EA
- [`src/mt5/SESSION_NOTES.md`](../../src/mt5/SESSION_NOTES.md) — état projet, TODO
- [`src/mt5/bridge/reset_tester_preset.py`](../../src/mt5/bridge/reset_tester_preset.py) — référence encoding UTF-16 LE pour `.set/.ini`
- [Doc MetaQuotes officielle — Platform Start](https://www.metatrader5.com/en/terminal/help/start_advanced/start)
- [MQL5 forum — CLI backtesting thread](https://www.mql5.com/en/forum/462397)
