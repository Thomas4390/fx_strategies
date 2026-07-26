# CLAUDE.md — `src/mt5/` (FxMultiSleeve)

> **Stratégie de trading FX algorithmique pour MetaTrader 5** : 3 sleeves (mean-reversion macro intraday + trend-following + RSI mean-reversion daily).
> **Référence stratégie/risque/maths** : voir `README.md` (388 lignes, exhaustif).
> **Ce fichier** : informations opérationnelles Windows (chemins absolus, broker, debug).
>
> 🐧 **Pour l'environnement Linux/Wine** (pilotage CLI, mode portable, symlinks) : voir [`docs/mt5/14_cli_backtest_linux.md`](../../docs/mt5/14_cli_backtest_linux.md).

---

## 🚀 Démarrage rapide pour agent fresh

Si tu reprends une nouvelle session, lis dans cet ordre :

1. **Ce fichier** — environnement Windows + procédure drag-and-drop + codes d'erreur.
2. [`SESSION_NOTES.md`](./SESSION_NOTES.md) — état d'avancement, baselines numériques, todo prioritaire, mise à jour 2026-05-04 sur l'infra Linux/Wine.
3. [`docs/mt5/14_cli_backtest_linux.md`](../../docs/mt5/14_cli_backtest_linux.md) — pipeline CLI Wine, format `.ini` UTF-16, pièges connus.
4. [`README.md`](./README.md) — théorie complète (3 sleeves, risk management, vol-targeting).

**Investigations ouvertes** :
- [`docs/investigations/rsi_daily_vbt_vs_mt5.md`](../../docs/investigations/rsi_daily_vbt_vs_mt5.md) — écart entre la référence VBT~Pro et le port MQL5 du sleeve RSI Daily, plan complet d'enquête (8 hypothèses).

**Outils CLI clés** (Linux/Wine, mais transposable Windows) :
- `bridge/run_backtest_cli.py` — backtest 5.4 ans en 22 s (Sharpe 1.15 baseline).
- `bridge/write_default_preset.py` — régénère `FxMultiSleeve_Default.set` depuis les défauts compilés.
- `bridge/reset_tester_preset.py` — patch les `.set` cachés MT5 quand on change un défaut compilé.
- `bridge/fx_macro_history.py` — régénère `macro_history.csv` (FRED API, à faire mensuellement).

**Livrables client** :
- [`reports/client_setup_guide/main.pdf`](../../reports/client_setup_guide/main.pdf) — guide client 11 pages plug-and-play.
- [`reports/latex_report/main_executive.pdf`](../../reports/latex_report/main_executive.pdf) — synthèse exécutive 10 pages (rapport investissement).

---

## Environnement de déploiement (Windows 11)

| Type | Chemin absolu |
|---|---|
| Source projet | `C:\Users\vaude\Documents\Coding_Project\src\mt5\` |
| Instance MT5 | `D0E8209F77C8CF37AD8BF550E51FF075` |
| Data Folder MT5 | `C:\Users\vaude\AppData\Roaming\MetaQuotes\Terminal\D0E8209F77C8CF37AD8BF550E51FF075\MQL5\` |
| Common Files | `C:\Users\vaude\AppData\Roaming\MetaQuotes\Terminal\Common\Files\` |
| Logs MT5 | `…\D0E8209F77C8CF37AD8BF550E51FF075\MQL5\logs\YYYYMMDD.log` (UTF-16) |

**Pour déployer** : copier `Experts/*.mq5`, `Include/*.mqh`, `Scripts/*.mq5` dans les sous-dossiers correspondants du Data Folder MT5, puis F7 dans MetaEditor pour compiler `FxMultiSleeve.mq5`.

## Broker — IMPORTANT

**Suffixe broker actif : `.c`** (compte ECN/Raw). Les symboles s'appellent `EURUSD.c`, `GBPUSD.c`, `USDJPY.c`, `USDCAD.c`.

**Le défaut compilé est désormais `Inp_SymbolSuffix=".c"`** dans `FxMultiSleeve.mq5:74` et `FxPreflight.mq5:16` — donc le drag-and-drop fonctionne sans aucun tweak des Inputs sur ce broker.

Si on change de broker plus tard : vérifier les noms de symboles dans MarketWatch et changer le défaut dans les 2 sources puis recompiler (peut être `""`, `"m"`, `".raw"`, `"-pro"`, etc.).

## Procédure drag-and-drop standard

Avec les défauts compilés actuels (`Inp_SymbolSuffix=".c"` et `Inp_MacroSourceMode=MACRO_SOURCE_AUTO`), le drag-and-drop est maintenant zero-config en live ET en backtest :

1. Ouvrir un chart **`EURUSD.c` M1** (timeframe critique pour Sleeve 1)
2. Vérifier `Outils → Options → Expert Advisors` :
   - ✅ `Allow Algo Trading` activé
   - ✅ `Allow WebRequest for listed URL` activé avec `https://api.stlouisfed.org` listé
3. Glisser `Navigator → Expert Advisors → FxMultiSleeve` sur le chart
4. Onglet **Common** : `Allow Algo Trading` coché
5. OK (ne rien toucher dans Inputs)
6. Vérifier 🙂 dans le coin du chart + Journal `[INIT][INFO] EA ready`

**Lancer `FxPreflight` d'abord** (même procédure) pour valider l'environnement avant l'EA.

> ⚠️ Si l'EA était déjà attaché à un chart avec d'anciennes valeurs sauvegardées : `clic-droit chart → Expert Advisors → Remove`, puis re-drag pour bénéficier des nouveaux défauts.

## Modes macro — 5 valeurs de `EMacroSourceMode`

| Mode | Use case | Source de données | Fichiers requis |
|---|---|---|---|
| `MACRO_SOURCE_FILE` | Live legacy (cron Python horaire) | `macro_cache.csv` (1 ligne) | `bridge/fx_macro_bridge.py` planifié |
| `MACRO_SOURCE_NATIVE` | Live autonome (recommandé live) | FRED `WebRequest` + MT5 Calendar | `fred_api_key.txt` + URL whitelist |
| `MACRO_SOURCE_HYBRID` | Live robuste | tente NATIVE, fallback FILE | les deux ci-dessus |
| `MACRO_SOURCE_HISTORY` | Backtest (Strategy Tester) | `macro_history.csv` time-indexed | `bridge/fx_macro_history.py` lancé une fois |
| `MACRO_SOURCE_AUTO` | **Défaut** | Détecte `MQLInfoInteger(MQL_TESTER)` → HISTORY en tester, NATIVE en live | les fichiers requis du mode résolu |

**Mode AUTO = best of both worlds** : un seul `.ex5` compilé tourne sans modification d'inputs en live ET en backtest. Le dispatch se fait à chaque `Refresh()` via `CMacroFilter::ResolveEffectiveMode()`.

### Sources implémentées

- `FxMacroSourceNative.mqh` — `CMacroSourceCalendar` (chômage via MT5 Calendar) + `CMacroSourceFRED` (spread via WebRequest)
- `FxMacroSourceHistory.mqh` — `CMacroSourceHistory` (CSV multi-lignes en mémoire + binary search par `TimeCurrent()`)

### Fichiers requis sur ce poste

| Fichier | Mode(s) | Statut local |
|---|---|---|
| `Common\Files\fred_api_key.txt` | NATIVE / HYBRID / AUTO-live | ✅ déployé — ⚠️ voir le piège de chemin ci-dessous |
| `Common\Files\macro_cache.csv` | FILE / HYBRID-fallback | non requis (NATIVE par défaut) |
| `Common\Files\macro_history.csv` | HISTORY / AUTO-tester | ✅ généré pour 2019-2026 (1833 lignes) |
| URL whitelist `https://api.stlouisfed.org` | NATIVE / HYBRID / AUTO-live | ✅ activé dans MT5 |

**Pour obtenir une clé FRED (gratuit)** : https://fredaccount.stlouisfed.org/apikeys

## Backtest dans Strategy Tester

### 1) Régénérer (ou rafraîchir) `macro_history.csv`

```bash
python src/mt5/bridge/fx_macro_history.py
# ou avec une fenêtre custom :
python src/mt5/bridge/fx_macro_history.py --start 2019-01-01 --end 2026-04-30
```

Le script :
- Lit `FRED_API_KEY` dans `<repo>/.env` (gitignoré)
- Fetch `T10Y2Y` daily et `UNRATE` monthly via FRED API
- Écrit les parquets bruts dans `data/` (utilisés aussi par `fx_macro_bridge.py`)
- Calcule `unemp_rising` (delta 3m sur UNRATE) et `macro_ok` pour chaque date
- Écrit `Common\Files\macro_history.csv` (multi-lignes, ASCII, ~1800 lignes pour 2019-2026)

### 2) Lancer le tester

1. `View → Strategy Tester` (Ctrl+R)
2. **Expert** : `FxMultiSleeve`, **Symbol** : `EURUSD.c`, **Period** : `M1`
3. **Modeling** : `Every tick based on real ticks` (le plus précis)
4. **Date** : 2019-01-01 → 2026-04-30 (ou sous-fenêtre)
5. Onglet **Inputs** : ne rien toucher (AUTO détecte le tester et bascule en HISTORY)
6. **Start**

### 3) Vérifier la sortie

Dans le Journal du tester, chercher au démarrage :
```
CMacroSourceHistory: loaded 1833 rows from macro_history.csv [2019.01.02 ... 2026.04.30]
```
Et au fil du backtest, chaque refresh log : `Macro source=history spread=… macro_ok=…` avec une valeur qui change au cours du temps simulé (preuve que le binary search marche).

### 4) Refresh régulier

À refaire toutes les semaines/mois pour ajouter les nouvelles obs FRED. Idempotent : ré-exécuter le script écrase les parquets et le CSV avec les données fraîches.

## Secrets / clé FRED — où vit la clé sur ce poste

⚠️ **Ne jamais coller la clé littérale dans ce CLAUDE.md, dans un commit, ou un message qui sortirait du poste.** La clé est personnelle (compte `vaudescal.t@gmail.com`) et bien que FRED soit gratuit, la fuite gaspillerait le quota et tracerait l'identité.

**Emplacements actifs sur ce poste** (gitignorés ou hors-repo) :

| Fichier | Rôle | Statut git |
|---|---|---|
| `<repo-root>/.env` | Source de vérité Python/dev — `FRED_API_KEY=…` | gitignoré (`.gitignore` ligne 30) |
| `C:\…\Terminal\Common\Files\fred_api_key.txt` | Lu par MT5 via `FileOpen(…, FILE_COMMON)` | hors du repo |

> ⚠️ **Piège de chemin `FILE_COMMON` en mode portable — diagnostiqué le 2026-07-26.**
> Symptôme : le journal live répète `CMacroSourceFRED: no API key configured` puis
> `CMacroFilter::NATIVE: FRED fetch failed`, une fois par minute, indéfiniment. Le message
> se lit comme une clé expirée ; **il signifie fichier introuvable**. La clé, elle, était
> valide (HTTP 200 sur `T10Y2Y`).
>
> Sous Wine avec `/portable`, `FILE_COMMON` ne résout PAS vers la racine portable
> `…/MetaTrader 5/Common/Files/` mais vers :
>
> ```
> ~/.mt5/drive_c/users/thomas/AppData/Roaming/MetaQuotes/Terminal/Common/Files/
> ```
>
> C'est le même piège que celui déjà documenté pour `macro_history.csv`, jamais appliqué à
> la clé : elle n'existait que dans la racine portable. Vérification rapide — le répertoire
> ACTIF est celui où l'EA écrit ses propres sorties (`deals_*.csv`, `optim_results.csv`) :
>
> ```bash
> ls ~/.mt5/drive_c/users/thomas/AppData/Roaming/MetaQuotes/Terminal/Common/Files/
> ```
>
> Tout fichier lu via `FILE_COMMON` doit être déposé **là**, pas dans la racine portable.
> N'affecte que le live : en tester, `MACRO_SOURCE_AUTO` bascule sur `HISTORY` et n'appelle
> jamais FRED — ce qui explique que les backtests n'aient jamais rien signalé.

**Pour récupérer la clé en future session Claude** : `Read C:\Users\vaude\Documents\Coding_Project\.env` (le fichier est local, jamais committé).

**Si on doit régénérer la clé** (compromission, rotation) : régénérer sur https://fredaccount.stlouisfed.org/apikeys puis mettre à jour les 2 fichiers ci-dessus.

## Codes d'erreur MQL5 rencontrés (avec fix)

| Code | Constante | Symptôme typique | Fix |
|---|---|---|---|
| **4305** | `ERR_MARKET_SELECT_ERROR` | `EnsureSymbolSelected: cannot add EURUSD` | Définir `Inp_SymbolSuffix` au bon suffixe broker |
| **5004** | `ERR_CANNOT_OPEN_FILE` | `cannot open macro_cache.csv` | Soit générer le CSV via bridge Python, soit passer en `MACRO_SOURCE_NATIVE` |
| **reason=8** | `REASON_INITFAILED` | `[DEINIT][INFO] EA stopped reason=8` | OnInit() a retourné non-zéro — remonter dans les logs pour la cause précise |

**Référence officielle** : https://www.mql5.com/en/docs/constants/errorswarnings/errorcodes

## Lecture des logs MT5

```
C:\Users\vaude\AppData\Roaming\MetaQuotes\Terminal\D0E8209F77C8CF37AD8BF550E51FF075\MQL5\logs\YYYYMMDD.log
```

**Encodage** : UTF-16 LE avec BOM. Le tool `Read` de Claude Code affiche des espaces entre les caractères — c'est normal, lire en filtrant ou utiliser PowerShell `Get-Content -Encoding Unicode`.

**Ce qu'on cherche** :
- `[INIT][INFO] EA ready` → succès
- `[INIT][ERROR]` → cause de l'échec d'init sur la ligne suivante ou précédente
- `[DEINIT][INFO] EA stopped reason=N` → l'EA s'est détaché ; reason=8 = init failed

## Architecture du code (rappel rapide)

**EA principal** : `Experts/FxMultiSleeve.mq5` (268 lignes)
- Orchestre 3 sleeves via `g_sleeve_mr`, `g_sleeve_ts`, `g_sleeve_rsi`
- Risk management global via `CRiskManager` (`FxRiskManager.mqh`)
- Macro filter via `CMacroFilter` (`FxMacroFilter.mqh`)

**Allocations strictes** (somme = 1.0, validée à 1e-6) :
- 0.80 → Sleeve 1 MR Macro (M1 intraday, 4 paires, fenêtre 6h-14h UTC)
- 0.10 → Sleeve 2 TS Momentum (D1, 3 paires)
- 0.10 → Sleeve 3 RSI Daily (D1, 4 paires)

**Includes (13 fichiers `.mqh`)** :
```
FxCommon.mqh                  Constantes, enums (EMacroSourceMode), helpers
FxLogger.mqh                  Print + CSV logging
FxRiskManager.mqh             Vol-targeting, sub-equity, circuit-breaker DD, marge cap
FxMacroFilter.mqh             Orchestrateur 5 modes (FILE / NATIVE / HYBRID / HISTORY / AUTO)
FxMacroSourceNative.mqh       Calendar MT5 + WebRequest FRED (live)
FxMacroSourceHistory.mqh      CSV time-indexed binary search (backtest)
FxIndicatorVWAP.mqh           VWAP daily-anchor
FxIndicatorBBDeviation.mqh    Bollinger Bands sur déviation VWAP
FxSleeveBase.mqh              Interface abstraite des sleeves
FxSleeveMRMacro.mqh           Sleeve 1
FxSleeveTSMomentum.mqh        Sleeve 2
FxSleeveRSIDaily.mqh          Sleeve 3
FxTradeHelpers.mqh            CTrade wrappers, sizing, stop level
```

**Scripts** :
- `Scripts/FxPreflight.mq5` — vérif environnement (symboles, history, macro) avant déploiement
- `Scripts/FxIndicatorTest.mq5` — tests unitaires VWAP / BBDeviation

**Bridges Python** (séparation par responsabilité) :
- `bridge/fx_macro_bridge.py` — live : 1 ligne CSV → `Common\Files\macro_cache.csv` (cron horaire si on veut le mode FILE)
- `bridge/fx_macro_history.py` — backtest : N lignes CSV → `Common\Files\macro_history.csv` (one-shot, à relancer périodiquement)

Les 2 scripts partagent le même schéma CSV ; la seule différence est le nombre de lignes (1 vs ~1800).

## Inputs critiques de référence

```
// Allocations (somme strict = 1.0)
Inp_AllocMRMacro      = 0.80
Inp_AllocTSMomentum   = 0.10
Inp_AllocRSIDaily     = 0.10

// Risk (Phase I 2026-05-05 leverage uplift)
Inp_GlobalTargetVol   = 0.75      // 75% annualisé (vs 28% pré-Phase I)
Inp_GlobalMaxLeverage = 64.0      // (vs 12.0 pré-Phase I)
Inp_EnableDDCap       = false     // Désactivé Phase A (pas de bénéfice OOS)
Inp_DDCap             = 0.30
Inp_EnableMarginCap   = false     // Désactivé Phase A (jamais touché en backtest)
Inp_MarginCapPct      = 0.70

// Broker (CRITIQUE)
Inp_SymbolSuffix      = ".c"      // ⚠️ adapter au broker

// Macro
Inp_MacroSourceMode       = MACRO_SOURCE_AUTO        // tester→HISTORY, live→NATIVE
Inp_MacroCacheFile        = "macro_cache.csv"        // utilisé par FILE / HYBRID-fallback
Inp_MacroHistoryFile      = "macro_history.csv"      // utilisé par HISTORY / AUTO-tester
Inp_MacroUseCommon        = true
Inp_MacroHistoryUseCommon = true
Inp_MacroMaxAgeHours      = 168

// Logging
Inp_LogVerbose        = false
Inp_LogToFile         = true      // → MQL5\Files\fx_log.csv
```

## Workflow de debug type

Quand quelque chose ne marche pas en drag-and-drop :

1. **Lire le log du jour** : `…\MQL5\logs\YYYYMMDD.log`
2. **Localiser la dernière séquence FxMultiSleeve** (ou FxPreflight)
3. **Chercher `[INIT][ERROR]`** ou `[DEINIT] reason=` → cause directe
4. **Si err=4305** → suffixe broker mal réglé (`Inp_SymbolSuffix`)
5. **Si err=5004** → fichier manquant (clé FRED ou macro_cache.csv selon le mode)
6. **Si compilation échoue** → vérifier que tous les `.mqh` sont dans `Include/` directement (pas de sous-dossier)
7. **Si EA tourne mais ne trade pas** → vérifier `Inp_AllocMRMacro` ≠ 0, fenêtre horaire (Sleeve 1 ne trade que 6-14 UTC), macro filter actif

**Tester avec `FxPreflight` AVANT chaque déploiement** sur un chart vierge — il valide les 4 paires + history M1/D1 + macro source.

## Notes diverses

- **Aucune DLL** dans le code (pur MQL5, portable)
- **Aucun indicateur custom** déployé (tout est embarqué dans les .mqh)
- **Pas de redistribution FRED** : la clé est personnelle, fichier local uniquement (jamais committer `fred_api_key.txt` dans git)
- **Encodage source** : les `.mq5`/`.mqh` sont en UTF-8 BOM (standard MetaEditor) — ne pas resauvegarder en autre encoding
