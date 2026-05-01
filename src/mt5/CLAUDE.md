# CLAUDE.md — `src/mt5/` (FxMultiSleeve)

> **Stratégie de trading FX algorithmique pour MetaTrader 5** : 3 sleeves (mean-reversion macro intraday + trend-following + RSI mean-reversion daily).
> **Référence stratégie/risque/maths** : voir `README.md` (388 lignes, exhaustif).
> **Ce fichier** : informations opérationnelles spécifiques à ce déploiement (chemins absolus, broker, debug).

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

Avec les défauts compilés actuels (`Inp_SymbolSuffix=".c"` et `Inp_MacroSourceMode=MACRO_SOURCE_NATIVE`), le drag-and-drop est maintenant zero-config :

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

## Mode macro NATIVE (FRED API + MT5 Calendar)

Utilisé à la place du bridge Python `fx_macro_bridge.py` (mode FILE) — autonome et plus simple.

**Sources de données** :
- **Spread Treasury 10Y-2Y** : série FRED `T10Y2Y` via `WebRequest` (`FxMacroSourceNative.mqh::CMacroSourceFRED`)
- **Taux de chômage US** : `CalendarValueHistoryByEvent("Unemployment Rate", "US")` natif MT5

**Fichiers requis** :
- `Common\Files\fred_api_key.txt` : la clé API FRED (32 chars hex), une seule ligne, ANSI (déjà créé sur ce poste)
- URL whitelist : `https://api.stlouisfed.org` dans `Outils → Options → Expert Advisors`

**Pour obtenir une clé FRED (gratuit)** : https://fredaccount.stlouisfed.org/apikeys

## Secrets / clé FRED — où vit la clé sur ce poste

⚠️ **Ne jamais coller la clé littérale dans ce CLAUDE.md, dans un commit, ou un message qui sortirait du poste.** La clé est personnelle (compte `vaudescal.t@gmail.com`) et bien que FRED soit gratuit, la fuite gaspillerait le quota et tracerait l'identité.

**Emplacements actifs sur ce poste** (gitignorés ou hors-repo) :

| Fichier | Rôle | Statut git |
|---|---|---|
| `<repo-root>/.env` | Source de vérité Python/dev — `FRED_API_KEY=…` | gitignoré (`.gitignore` ligne 30) |
| `C:\…\Terminal\Common\Files\fred_api_key.txt` | Lu par MT5 via `FileOpen(…, FILE_COMMON)` | hors du repo |

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

**Includes (12 fichiers `.mqh`)** :
```
FxCommon.mqh                  Constantes, enums, helpers (EnsureSymbolSelected, EnsureHistory, MakeSymbolWithSuffix, SplitCsv)
FxLogger.mqh                  Print + CSV logging
FxRiskManager.mqh             Vol-targeting, sub-equity, circuit-breaker DD, marge cap
FxMacroFilter.mqh             Filtre macro 2-étages (FILE / NATIVE / HYBRID)
FxMacroSourceNative.mqh       Calendar MT5 + WebRequest FRED
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

**Bridge Python (mode FILE — non utilisé en NATIVE)** :
- `bridge/fx_macro_bridge.py` génère `Common\Files\macro_cache.csv` toutes les heures via cron/Task Scheduler

## Inputs critiques de référence

```
// Allocations (somme strict = 1.0)
Inp_AllocMRMacro      = 0.80
Inp_AllocTSMomentum   = 0.10
Inp_AllocRSIDaily     = 0.10

// Risk
Inp_GlobalTargetVol   = 0.28      // 28% annualisé
Inp_GlobalMaxLeverage = 12.0
Inp_EnableDDCap       = true      // Circuit-breaker à -15%
Inp_DDCap             = 0.15
Inp_EnableMarginCap   = true
Inp_MarginCapPct      = 0.70

// Broker (CRITIQUE)
Inp_SymbolSuffix      = ".c"      // ⚠️ adapter au broker

// Macro
Inp_MacroSourceMode   = MACRO_SOURCE_NATIVE
Inp_MacroUseCommon    = true
Inp_MacroMaxAgeHours  = 168

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
