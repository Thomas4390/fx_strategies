# FxMultiSleeve — Portage MQL5 de la stratégie FX Tri-Signaux

Implémentation MetaTrader 5 (MQL5) de la stratégie documentée dans
`reports/latex_report/main.tex` et codée en Python/vectorbtpro dans
`src/strategies/`.

## Vue d'ensemble

Un seul EA (`Experts/FxMultiSleeve.mq5`) attaché à n'importe quel chart
(typiquement EUR/USD M1) orchestre les trois sleeves :

| Sleeve | Allocation | Symboles | TF | Logique |
|--------|-----------|----------|----|---------|
| MR Macro | 80 % | EUR/USD, GBP/USD, USD/JPY, USD/CAD | M1 | VWAP daily-anchor + Bollinger 5σ 80p, session 6-14 UTC, filtre macro 2-étages |
| TS Momentum | 10 % | EUR/USD, GBP/USD, USD/JPY | D1 | EMA 20/50 + RSI(7) ; vol-target 10 % par paire |
| RSI Daily | 10 % | EUR/USD, GBP/USD, USD/JPY, USD/CAD | D1 | RSI(14) crossings 25/75 |

Overlay portfolio (cf. `combined_portfolio_v2.py`) :
- Vol-targeting global : `lev = min(0.28 / max(σ21, σ63, 0.02), 12.0)` (shift -1)
- Allocation statique 80/10/10 (pas de regime-adaptive)
- **Circuit-breaker DD activé à 15 %** (LaTeX § 13.3)
- **Cap marge à 70 %** avec auto-deleverage (LaTeX § 13.2)
- **Freshness macro 7 jours** (LaTeX § 13.1)
- Slippage paramétré : 15 bps Sleeve 1, 10 bps Sleeves 2-3

## Structure des fichiers

```
src/mt5/
├── Experts/FxMultiSleeve.mq5         # EA principal (orchestrateur)
├── Include/                           # Headers .mqh (compilés via #include)
│   ├── FxCommon.mqh                   # Constantes, helpers, magic numbers
│   ├── FxLogger.mqh                   # Print + fichier CSV
│   ├── FxRiskManager.mqh              # Sub-equity, vol-target, DD, cap marge
│   ├── FxMacroFilter.mqh              # Filtre macro 2-étages
│   ├── FxMacroSourceNative.mqh       # Calendar US + WebRequest FRED
│   ├── FxIndicatorVWAP.mqh            # CVWAPDaily (cumulative reset 00h UTC)
│   ├── FxIndicatorBBDeviation.mqh    # CBBDeviation (rolling 80p, 5σ)
│   ├── FxSleeveBase.mqh               # Classe abstraite
│   ├── FxSleeveMRMacro.mqh            # Sleeve 1 (4 paires)
│   ├── FxSleeveTSMomentum.mqh         # Sleeve 2 (3 paires)
│   ├── FxSleeveRSIDaily.mqh           # Sleeve 3 (4 paires)
│   └── FxTradeHelpers.mqh             # CTrade wrappers, sizing
├── Scripts/
│   ├── FxIndicatorTest.mq5            # Tests unitaires VWAP/BB
│   └── FxPreflight.mq5                # Vérif environnement avant déploiement
└── bridge/
    ├── fx_macro_bridge.py             # Cron horaire → macro_cache.csv
    └── README.md
```

---

## Installation pas-à-pas dans MetaEditor

L'objectif : prendre le code de ce dépôt, le déposer dans le bon dossier MT5,
ouvrir MetaEditor, compiler et obtenir un EA `.ex5` directement utilisable.

### 1) Localiser le dossier MQL5 du terminal

MetaTrader 5 stocke tous les fichiers compilables dans un dossier appelé
**Data Folder** (un par installation). Pour le trouver depuis le terminal :

> `Fichier → Ouvrir le répertoire de données`

Ce dossier contient un sous-dossier `MQL5/` avec la structure suivante :

```
<Data Folder>/MQL5/
├── Experts/
├── Include/
├── Scripts/
├── Indicators/
└── Files/
```

#### Chemins typiques

| Plateforme | Chemin |
|---|---|
| Windows natif | `C:\Users\<USER>\AppData\Roaming\MetaQuotes\Terminal\<HASH>\MQL5\` |
| Linux + Wine | `~/.wine/drive_c/users/<USER>/AppData/Roaming/MetaQuotes/Terminal/<HASH>/MQL5/` |
| macOS + Wine (PlayOnMac) | `~/Library/Application Support/<wrapper>/drive_c/users/<USER>/AppData/Roaming/MetaQuotes/Terminal/<HASH>/MQL5/` |

`<HASH>` est un identifiant unique généré par MT5. Il y a aussi un dossier
**Common** partagé entre toutes les instances MT5 sous
`...\MetaQuotes\Terminal\Common\Files\` — c'est là qu'on dépose le CSV macro
et la clé API FRED.

### 2) Copier les fichiers source

Depuis la racine du dépôt :

```bash
# Variables (à adapter à votre installation)
MQL5_DIR="$HOME/.wine/drive_c/users/$USER/AppData/Roaming/MetaQuotes/Terminal/<HASH>/MQL5"
COMMON_DIR="$HOME/.wine/drive_c/users/$USER/AppData/Roaming/MetaQuotes/Terminal/Common/Files"

# Création des cibles si elles n'existent pas
mkdir -p "$MQL5_DIR/Experts" "$MQL5_DIR/Include" "$MQL5_DIR/Scripts" "$COMMON_DIR"

# Copie du code
cp src/mt5/Experts/FxMultiSleeve.mq5     "$MQL5_DIR/Experts/"
cp src/mt5/Include/Fx*.mqh                "$MQL5_DIR/Include/"
cp src/mt5/Scripts/Fx*.mq5                "$MQL5_DIR/Scripts/"
```

Sur Windows natif, copier manuellement via l'Explorateur — la structure cible
doit être :

```
<Data Folder>/MQL5/
├── Experts/FxMultiSleeve.mq5
├── Include/FxCommon.mqh
├── Include/FxLogger.mqh
├── Include/FxRiskManager.mqh
├── Include/FxMacroFilter.mqh
├── Include/FxMacroSourceNative.mqh
├── Include/FxIndicatorVWAP.mqh
├── Include/FxIndicatorBBDeviation.mqh
├── Include/FxSleeveBase.mqh
├── Include/FxSleeveMRMacro.mqh
├── Include/FxSleeveTSMomentum.mqh
├── Include/FxSleeveRSIDaily.mqh
├── Include/FxTradeHelpers.mqh
├── Scripts/FxIndicatorTest.mq5
└── Scripts/FxPreflight.mq5
```

> **Important** : tous les `.mqh` doivent atterrir dans `Include/` (pas dans
> un sous-dossier), car le `#include "..\Include\..."` de l'EA est relatif.

### 3) Ouvrir MetaEditor et compiler

Dans le terminal MT5 :

1. Appuyer sur **F4** (ou `Outils → MetaQuotes Language Editor`).
2. Dans le panneau **Navigator** (gauche), naviguer vers `Experts/FxMultiSleeve.mq5`.
3. Double-cliquer pour ouvrir le fichier.
4. Appuyer sur **F7** pour compiler.

La compilation doit afficher dans l'onglet **Errors** :

```
0 errors, 0 warnings
```

Si une erreur d'inclusion apparaît (`'FxLogger.mqh' not found`), vérifier que
les `.mqh` sont bien dans le dossier `Include/` du Data Folder, pas ailleurs.

> **Astuce** : si MetaEditor est déjà ouvert pendant la copie, fermer-rouvrir
> pour qu'il rafraîchisse l'arbre Navigator.

Compiler aussi les deux scripts utilitaires :
- `Scripts/FxIndicatorTest.mq5` (F7) → produit `FxIndicatorTest.ex5`
- `Scripts/FxPreflight.mq5` (F7) → produit `FxPreflight.ex5`

### 4) Configurer la source macro

L'EA supporte trois modes via l'input `Inp_MacroSourceMode` :

| Mode | Comportement | Quand l'utiliser |
|------|--------------|------------------|
| `MACRO_SOURCE_FILE` | Lit `macro_cache.csv` (bridge Python) | Backtest, simplicité opérationnelle |
| `MACRO_SOURCE_NATIVE` | Calendar MT5 (chômage) + WebRequest FRED (spread) | Pas de Python disponible |
| `MACRO_SOURCE_HYBRID` | NATIVE puis fallback FILE | **Recommandé en production live** |

#### 4a) Mode FILE — bridge Python horaire

Le script `bridge/fx_macro_bridge.py` lit les parquets locaux du dépôt et
écrit un CSV qui sera lu par l'EA.

```bash
# Test ponctuel (vérifie que ça écrit bien dans Common/Files)
python src/mt5/bridge/fx_macro_bridge.py

# Production : cron horaire (Linux/Mac)
( crontab -l 2>/dev/null; \
  echo "0 * * * * /usr/bin/python3 $(pwd)/src/mt5/bridge/fx_macro_bridge.py" \
) | crontab -

# Windows : Planificateur de tâches → Action: python.exe ; Argument: chemin du script
```

Vérifier ensuite que `<Common>/Files/macro_cache.csv` est créé avec un contenu :

```csv
timestamp_utc,spread_10y2y,unemp_rising,spread_threshold,macro_ok
2026-04-26T14:00:00Z,0.3520,0,0.50,1
```

#### 4b) Mode NATIVE / HYBRID — Calendar + FRED

1. **Whitelister l'URL FRED** dans le terminal (étape obligatoire MQL5) :
   `Outils → Options → Conseillers experts (Expert Advisors)` →
   cocher *"Allow WebRequest for the following URL"* et ajouter
   `https://api.stlouisfed.org` puis **OK**. Le terminal prend en compte la
   nouvelle whitelist immédiatement.

2. **Obtenir une clé API FRED** (gratuite, instantanée) :
   https://fred.stlouisfed.org/docs/api/api_key.html

3. **Stocker la clé** dans un fichier sandbox MT5 (jamais en input visible) :

   ```bash
   echo "VOTRE_CLE_API" > "$COMMON_DIR/fred_api_key.txt"
   ```

   Sur Windows : créer manuellement le fichier `fred_api_key.txt` dans
   `<Data Folder>/Common/Files/` avec la clé en texte brut sur une ligne.

4. **Configurer l'EA** : passer
   - `Inp_MacroSourceMode = MACRO_SOURCE_HYBRID`
   - `Inp_FREDApiKeyFile = "fred_api_key.txt"`
   - `Inp_FREDKeyUseCommon = true`

> Note : en backtest Strategy Tester, `WebRequest` retourne -1 (pas de réseau).
> Le mode HYBRID basculera automatiquement sur le fichier `macro_cache.csv` —
> donc même en backtest il faut un CSV valide.

### 5) Lancer le preflight

Avant tout déploiement live, valider l'environnement :

1. Dans le terminal MT5, ouvrir un chart **EUR/USD M1**.
2. Dans le **Navigator** (Ctrl+N), section *Scripts*, glisser-déposer
   `FxPreflight` sur le chart.
3. Lire l'onglet **Experts** ou **Journal** : tous les checks doivent être
   en `PASS`. Vérifier notamment :
   - Les 4 paires (EURUSD, GBPUSD, USDJPY, USDCAD) sont disponibles
   - L'historique D1 est chargeable
   - `macro_cache.csv` est lisible (en mode FILE)
   - Les permissions WebRequest sont actives (en mode NATIVE)

### 6) Tests unitaires des indicateurs

Glisser-déposer `Scripts/FxIndicatorTest` sur un chart EUR/USD M1. Le script
vérifie :
- BBDeviation sur cas connus (constante 0.0, suite [0..79]) → `Pass`
- VWAP daily-anchor reset à minuit UTC → `Pass`

### 7) Backtest Strategy Tester

1. Ouvrir le tester : **View → Strategy Tester** (Ctrl+R).
2. Choisir :
   - Expert : `FxMultiSleeve`
   - Symbol : `EURUSD`
   - Period : `M1`
   - Date : 2024-01-01 → 2024-12-31 (validation rapide)
   - Modeling : **Every tick based on real ticks** (le plus précis)
3. Onglet **Inputs** : adapter `Inp_SymbolSuffix` au broker (ex. `"m"` pour
   "EURUSDm") si applicable.
4. **Start**. Vérifier ensuite l'onglet **Backtest report** :
   - Sharpe ≈ 0.85–1.05 (Python ref ≈ 0.94 IS / 1.44 OOS)
   - Max DD ≈ -15 à -20 %
   - Trades sur les 4 paires (pas seulement EURUSD)

> Pour valider la conformité long terme : 2019-01 → 2025-12 en walk-forward.

### 8) Déploiement compte démo (≥ 1 semaine recommandé)

1. Ouvrir un chart EUR/USD M1.
2. Glisser-déposer `Experts/FxMultiSleeve` sur le chart.
3. Onglet **Common** : cocher
   - "Allow Algorithmic trading"
   - "Allow modification of Signal settings" (optionnel)
4. Onglet **Inputs** : configurer selon ci-dessous.
5. **OK** — l'EA s'attache, un sourire 🙂 apparaît dans le coin du chart.

Vérifier dans le **Journal** :
```
[INIT][INFO] FxMultiSleeve start build ...
[INIT][INFO] Macro source=file spread=0.3520 unemp_rising=0 macro_ok=1
[MR_Macro][INFO] Init OK 4 pairs (EURUSD,GBPUSD,USDJPY,USDCAD)
[TS_Momentum][INFO] Init OK 3 pairs
[RSI_Daily][INFO] Init OK 4 pairs
[INIT][INFO] EA ready
```

Surveillance recommandée pendant la phase démo :
- Magic numbers cohérents (831/832/833 par sleeve)
- Macro cache rafraîchi régulièrement (`[MACRO][INFO]`)
- Aucun retcode d'erreur trade > 10003 récurrent

---

## Inputs notables (configuration)

### Allocation & risk

| Input | Défaut | Effet |
|-------|--------|-------|
| `Inp_AllocMRMacro/TS/RSI` | 0.80 / 0.10 / 0.10 | Allocation entre sleeves (somme = 1) |
| `Inp_GlobalTargetVol` | 0.28 | Vol cible 28 % annualisé |
| `Inp_GlobalMaxLeverage` | 12.0 | Plafond levier global |
| `Inp_EnableDDCap` | **true** | Circuit-breaker DD (LaTeX § 13.3) |
| `Inp_DDCap` | 0.15 | Seuil DD (-15 %) |
| `Inp_EnableMarginCap` | **true** | Cap marge / auto-deleverage |
| `Inp_MarginCapPct` | 0.70 | Seuil margin/equity (70 %) |

### Sleeves

| Input | Défaut | Effet |
|-------|--------|-------|
| `Inp_MR_Pairs` | "EURUSD,GBPUSD,USDJPY,USDCAD" | Univers Sleeve 1 (4 paires) |
| `Inp_MR_BBWindow` / `Inp_MR_BBAlpha` | 80 / 5.0 | Bollinger Bands sur deviation VWAP |
| `Inp_MR_SLStop` / `Inp_MR_TPStop` | 0.005 / 0.006 | SL/TP du sleeve 1 (-0.5% / +0.6%) |
| `Inp_TS_Pairs` | "EURUSD,GBPUSD,USDJPY" | Univers Sleeve 2 (3 paires, USDCAD exclu) |
| `Inp_RSI_Pairs` | "EURUSD,GBPUSD,USDJPY,USDCAD" | Univers Sleeve 3 (4 paires) |
| `Inp_*_SlippageBps` | 15 / 10 / 10 | Slippage attendu par sleeve (LaTeX § 13.5) |

### Opérationnel

| Input | Défaut | Effet |
|-------|--------|-------|
| `Inp_SymbolSuffix` | "" | Suffixe broker ("m", ".r", etc.) |
| `Inp_MacroSourceMode` | `MACRO_SOURCE_FILE` | Choix de source macro |
| `Inp_MacroUseCommon` | true | Lit macro_cache.csv depuis Common/Files |
| `Inp_MacroMaxAgeHours` | **168** | Freshness macro 7 jours (LaTeX § 13.1) |
| `Inp_DailyRecomputeHr` | 21 | Heure UTC de recompute daily |

## Magic numbers

- `831` MR Macro
- `832` TS Momentum
- `833` RSI Daily

## Garde-fous opérationnels intégrés

| Garde-fou | LaTeX | Implémentation |
|---|---|---|
| Alerte DD ≥ 15 % | § 13.3 | `CRiskManager::CheckDDCircuitBreaker` (à chaque tick) |
| Cap marge 70 % | § 13.2 | `CRiskManager::CheckMarginCap` (toutes les 30 s) |
| Freshness macro 7 j | § 13.1 | `Inp_MacroMaxAgeHours = 168` |
| Slippage simulé | § 13.5 | `Inp_*_SlippageBps` injecté dans sizing |
| Recompute vol-target daily | § 06 | `OnTimer` à 21h UTC, σ21/σ63 sur 80 j |

Garde-fous **opérationnels humains** (à automatiser hors EA) :
- Réserve cash 30 % outside broker (LaTeX § 13.5 — accès < 24h)
- Kill switch < 10 min (procédure documentée, drill mensuel)
- Audit weekly des logs d'exécution

## Tests

### Test unitaire indicateurs

`Scripts/FxIndicatorTest` — vérifie BBDeviation sur cas connus (constante 0.0,
suite [0..79]) et VWAP daily-anchor.

### Strategy Tester

Sur EUR/USD M1, mode **Every tick based on real ticks**, période 2024-01 →
2024-12 (validation rapide), puis walk-forward 2019 → 2025.

Note : les paires non-primaires (GBP/USD, USD/JPY, USD/CAD) sont resamplées
depuis OHLC M1 — P&L Sleeves 2/3 moins précise mais acceptable car daily.

### Compte démo

Avant prod : ≥ 1 semaine sur compte démo. Vérifier dans le journal :
- Les 3 sleeves sont actifs
- Trades sur les 4 paires (Sleeve 1) — pas seulement EURUSD
- Magic numbers cohérents
- Macro cache rafraîchi (`[MACRO][INFO]` régulier)
- Pas de DD anormal vs walk-forward Python

## Limitations connues

1. **P&L MT5 ≠ Python** : spreads bid/ask, lots discrets, latence broker. Objectif :
   fidélité du signal et ordre de grandeur Sharpe (±20 % du Sharpe Python).
2. **Macro stale → MR Macro arrêté** : si la source macro tombe et le cache > 7j,
   le sleeve 1 se ferme automatiquement. *By design*.
3. **Tick volume au lieu de real volume** sur le VWAP — cohérent avec le code
   Python qui utilise aussi du tick volume sur les parquets.
4. **Vol-targeting daily, pas tick** — recompute à 21:00 UTC, appliqué pour les
   trades du jour suivant (équivalent à `shift(1)` Python).
5. **Strategy Tester multi-symbol** — MT5 backteste un EA principal sur un
   symbole de référence ; les ordres sur les autres paires sont exécutés mais
   moins précisément modélisés que le symbole du chart.

## Pour aller plus loin

- [docs/mt5/](../../docs/mt5/) — base de connaissance MQL5 (14 fichiers)
- [docs/mt5/11_porting_from_python.md](../../docs/mt5/11_porting_from_python.md) — mappings Python → MQL5
- [docs/mt5/10_pitfalls.md](../../docs/mt5/10_pitfalls.md) — pièges connus
- [docs/mt5/troubleshooting.md](../../docs/mt5/troubleshooting.md) — diagnostics
- [docs/mt5/13_native_data_sources.md](../../docs/mt5/13_native_data_sources.md) — détails sources NATIVE/HYBRID
