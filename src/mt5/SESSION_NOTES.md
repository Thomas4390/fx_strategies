# Notes de session — Port MT5 FxMultiSleeve (2026-04-30 / 05-01)

> **Document de transition fin de session.** Résumé de ce qui a été fait, état du système, et todo list priorisée pour la prochaine session. Pour le détail opérationnel (chemins, codes d'erreur, debug) voir [`CLAUDE.md`](CLAUDE.md). Pour la stratégie voir [`README.md`](README.md).

---

## TL;DR

Le port MQL5 de la stratégie 3-sleeve **est validé en backtest** : Sharpe 1.02 (réf Python 0.94), Max DD -7% (réf -15 à -20%), 838 trades sur 5.4 ans, 4 paires tradées. **Drag-and-drop fonctionne**, **Strategy Tester fonctionne** (mode AUTO bascule live ↔ tester sans config), infra macro (FRED API live + CSV historique backtest) en place. Reste à faire : walk-forward Sharpe OOS, optimisation 32-cores, démo live ≥1 mois, monitoring.

---

## Statut actuel du système

| Couche | Statut | Détail |
|---|---|---|
| **EA compilé** | ✅ | `FxMultiSleeve.ex5` — 0 errors / 0 warnings, 3 sleeves init OK |
| **Drag-and-drop live** | ✅ | EURUSD.c M1 → 🙂 + `[INIT][INFO] EA ready` |
| **Strategy Tester** | ✅ | Run 2020-11 → 2026-04 complet en 7.4 sec |
| **Macro NATIVE (live)** | ✅ | FRED T10Y2Y + MT5 Calendar Unemployment |
| **Macro HISTORY (backtest)** | ✅ | 1833 lignes 2019-01-02 → 2026-04-30, time-indexed lookup |
| **Mode AUTO** | ✅ | Dispatch via `MQLInfoInteger(MQL_TESTER)` → zero-config tester ↔ live |
| **Risk management** | ✅ | Vol-target + margin cap testés (auto-deleverage observé en 2022) |
| **CLAUDE.md docs** | ✅ | Chemins, broker, codes d'erreur, debug workflow |
| **Walk-forward Sharpe IS/OOS** | ⏳ | Pas encore lancé |
| **Optimisation 32-cores** | ⏳ | Pas encore lancé |
| **Démo live long-run** | ⏳ | Pas commencé |

---

## Avancées de la session (chronologique, par commit)

| Commit | Sujet | Pourquoi |
|---|---|---|
| `523e083` | `fix(mt5): default to .c suffix and NATIVE macro source for ECN broker` | Drag-and-drop échouait : broker SquaredFinancial utilise `.c`, défaut était `""`. Aussi NATIVE plutôt que FILE pour éviter la dépendance Python. |
| `a41e371` | `feat(mt5): backtest macro infrastructure with HISTORY + AUTO source modes` | WebRequest bloqué dans Tester → ajout `CMacroSourceHistory` (CSV multi-lignes binary search), nouveau script `bridge/fx_macro_history.py`, nouveaux modes `MACRO_SOURCE_HISTORY` et `MACRO_SOURCE_AUTO`. |
| `241427a` | `chore(mt5): log raw + resolved macro mode at init for diagnostic` | L'EA logguait `Macro source=native` mais ne disait pas si c'était `NATIVE` brut ou `AUTO` résolu. Ajout d'un log explicite. |
| `a003847` | `chore(mt5): add reset_tester_preset.py utility` | MT5 cache les inputs précédents dans `.set` → override les défauts compilés. Utilitaire pour patcher les `.set/.ini` quand on change les défauts. |
| `11c5d83` | `fix(mt5): preserve CRLF in tester preset writer + add multi-chart script` | Bug `\r\r\n` dans le `.set` (Windows newline translation par `Path.write_text`) → fix avec `write_bytes` + auto-détection. Bonus : `Scripts/FxOpenAllCharts.mq5` pour ouvrir les 4 paires en 1 clic. |
| `ec532ff` | `feat(mt5): graceful M1 history warmup in Sleeve MR Macro` | EnsureHistory(M1, 1500) hard-fail au démarrage du backtest (broker M1 commence 2020-11-22 avec 109 bars). Soft-floor à `BBWindow+20`, BB warmup au fil de l'eau. |
| `cc0d24a` | `feat(mt5): graceful D1 warmup in TS Momentum and RSI Daily sleeves` | Même pb sur D1 : TS exigeait 250 bars, RSI 100. Soft-floor à 1 bar (CopyBuffer skip gracieusement, indicateurs warmup au fil des bars D1). |

**13 fichiers touchés** : .gitignore, CLAUDE.md, FxMultiSleeve.mq5, FxCommon.mqh, FxMacroFilter.mqh, FxMacroSourceHistory.mqh (nouveau), FxSleeveMRMacro.mqh, FxSleeveRSIDaily.mqh, FxSleeveTSMomentum.mqh, FxOpenAllCharts.mq5 (nouveau), FxPreflight.mq5, fx_macro_history.py (nouveau), reset_tester_preset.py (nouveau).

---

## Résultats du backtest validé

**Configuration** : EURUSD.c M1, 2020-11-23 → 2026-04-29, real ticks (auto-fallback OHLC pour < 2025), defaults compilés.

```
Total Net Profit       +3825.60 USD  (+38.26% sur 5.43 ans)
Annualized return      +6.13% / an
Sharpe Ratio           1.02              ← réf Python IS = 0.94
Max Drawdown (eq)      -7.06%            ← réf Python -15 à -20%
Profit Factor          1.33
Recovery Factor        4.48
LR Correlation         0.96              (R² = 0.92, equity quasi-linéaire)
Win rate               58.95%
Total Trades           838
   - MR Macro          352 (M1 intraday, 4 pairs)
   - TS Momentum       441 (D1, 3 pairs)
   - RSI Daily          45 (D1, 4 pairs)
```

**Verdict** : ✅ Cohérent avec la référence Python VBT Pro (Sharpe légèrement supérieur, DD nettement inférieur — probablement dû à la période différente + risk management actif).

---

## Architecture finale (rappel rapide)

```
                         MACRO_SOURCE_AUTO (défaut compilé)
                                  │
            MQLInfoInteger(MQL_TESTER) → dispatch
            ┌─────────yes─────────┘└─────────no──────┐
            ▼                                         ▼
       HISTORY                                    NATIVE
   (binary search                            (FRED WebRequest
    macro_history.csv,                        + MT5 Calendar,
    1833 rows 2019→2026)                      live only)
            ▲                                         ▲
            │                                         │
   bridge/fx_macro_history.py             fred_api_key.txt + URL whitelist
   (one-shot, refresh hebdo)              (Common\Files\, gitignored .env source)
```

**3 sleeves** depuis 1 chart :
- **MR Macro** (M1 intraday 6-14h UTC, 4 paires, magic 831) — 80% allocation
- **TS Momentum** (D1, 3 paires, magic 832) — 10% allocation
- **RSI Daily** (D1, 4 paires, magic 833) — 10% allocation

---

## Reste à faire — priorisé

### 🔴 Haute priorité (avant déploiement live sérieux)

1. **Walk-forward Sharpe IS/OOS** (~30 min)
   - Strategy Tester → onglet Settings → champ `Forward` à `1/3` ou `30%`
   - Permet de calculer Sharpe sur 70% IS + 30% OOS séparément
   - Cible : OOS Sharpe ≥ 1.0 (réf Python OOS = 1.44)
   - Si OOS << IS : signal d'overfit, à investiguer

2. **Optimisation 32-cores sur paramètres clés** (~1h pour génétique, ~6h pour exhaustif)
   - Strategy Tester → `Optimization: Fast genetic based`
   - Cocher dans Inputs et configurer ranges :
     - `Inp_MR_BBAlpha` : start=3.0 step=0.5 stop=7.0
     - `Inp_MR_BBWindow` : start=40 step=20 stop=120
     - `Inp_MR_TPStop` : start=0.004 step=0.001 stop=0.010
     - `Inp_MR_SLStop` : start=0.003 step=0.001 stop=0.008
   - Surveiller `Tools → Options → Network` pour confirmer 32 agents actifs
   - Critère robustesse : Sharpe > 0.8 sur ≥ 80% des combos = stratégie pas overfit

3. **Démo live ≥ 1 mois sur compte démo SquaredFinancial**
   - Vérifier slippages réels vs simulés
   - Vérifier que macro NATIVE (FRED API) refresh OK chaque heure
   - Vérifier que le DD live reste sous 5% (vs simulé 7%)

### 🟡 Moyenne priorité (qualité / monitoring)

4. **Investiguer pourquoi RSI Daily ne fait que 8 trades/an**
   - Comparer avec la version Python : est-ce le même comportement ?
   - Peut-être que `Inp_RSI_Oversold=25 / Overbought=75` est trop strict en pratique
   - Tester avec 30/70 (plus permissif) si Python ref l'utilisait

5. **Profile par sleeve**
   - MT5 ne ventile pas Sharpe/PF/DD par magic number
   - Idée : exporter le deal log MT5 (CSV via `File → Save Deals`) puis script Python pour ventiler
   - Permettrait de valider que chaque sleeve performe individuellement

6. **Setup monitoring live**
   - `FxOpenAllCharts` script existe déjà — utiliser pour dashboard 4-pairs
   - Sauvegarder en Profile MT5 pour 1-clic
   - Considérer : MCP MetaTrader (ariadng/metatrader-mcp-server) pour query positions depuis Claude

### 🟢 Basse priorité (nice-to-have)

7. **Étendre la fenêtre de backtest jusqu'à 2019** (pour comparer 1:1 au ref Python)
   - Limite actuelle : broker SquaredFinancial M1 commence 2020-11-22
   - Solutions : changer de broker (Dukascopy, Pepperstone, Tickmill ont souvent + d'historique) ou rester sur 2021+

8. **Documenter la procédure de refresh hebdo de `macro_history.csv`**
   - Ajouter Task Scheduler Windows pour `python fx_macro_history.py` (ex: chaque dimanche)
   - Sinon le CSV vieillira (les nouvelles obs FRED ne seront pas captées)

9. **Setup walk-forward analysis automatisé**
   - MT5 a un mode "Forward" mais pas de WFA itérative
   - Idée : script Python qui pilote `terminal64.exe /config:tester.ini` pour W-F sur N fenêtres glissantes

10. **CI / tests**
    - `Scripts/FxIndicatorTest.mq5` existe pour BB/VWAP — l'étendre aux autres composants ?
    - GitHub Actions pour valider compilation MQL5 (avec wine-mt5 si dispo, ou skip)

---

## Quick reference

### Chemins clés
```
Source :               C:\Users\vaude\Documents\Coding_Project\src\mt5\
MT5 deploy :           C:\Users\vaude\AppData\Roaming\MetaQuotes\Terminal\D0E8209F77C8CF37AD8BF550E51FF075\MQL5\
MT5 Common Files :     C:\Users\vaude\AppData\Roaming\MetaQuotes\Terminal\Common\Files\
                       (fred_api_key.txt + macro_history.csv)
MT5 Tester logs :      ...\D0E8209F77C8CF37AD8BF550E51FF075\Tester\logs\YYYYMMDD.log
MT5 live logs :        ...\D0E8209F77C8CF37AD8BF550E51FF075\MQL5\logs\YYYYMMDD.log
.env (FRED key) :      C:\Users\vaude\Documents\Coding_Project\.env  (gitignored)
```

### Commandes utiles
```bash
# Régénérer la macro history (à faire chaque semaine)
python src/mt5/bridge/fx_macro_history.py

# Reset des presets MT5 Tester si on change un défaut compilé
# (MT5 doit être FERMÉ avant)
python src/mt5/bridge/reset_tester_preset.py
python src/mt5/bridge/reset_tester_preset.py --check    # dry-run

# Recompiler après modif source (sans ouvrir MetaEditor)
'/c/Program Files/MetaTrader 5/MetaEditor64.exe' /compile:"<path-to-.mq5>"
```

### Procédure standard de backtest
1. Fermer MT5 → relancer (vide le cache .set en mémoire)
2. `Ctrl+R` → Strategy Tester
3. `Expert: FxMultiSleeve, Symbol: EURUSD.c, Period: M1`
4. `Modeling: Every tick based on real ticks` (ou `1 minute OHLC` pour speed)
5. `Date: 2020-11-23 → 2026-04-30` (ou + court pour smoke test)
6. `Inputs` : ne rien toucher (AUTO + .c défauts compilés)
7. **Start** → vérifier au journal `[INIT][INFO] EA ready`

### En cas de souci rapide
- **Symboles non trouvés** : `Inp_SymbolSuffix` ne matche pas le broker → modifier compile-time ou Inputs runtime
- **Macro source=file en tester** : `.set` cache l'ancien défaut → fermer MT5 + `python reset_tester_preset.py` + rouvrir
- **`EnsureHistory ... only X/Y bars`** : warning normal si on backtest près de la limite broker, sinon attendre que MT5 télécharge l'historique

---

## Session next : où reprendre

**Suggestion d'ordre pour la prochaine session** :
1. Walk-forward (point #1 ci-dessus) — 30 min, donne la métrique manquante (Sharpe OOS) pour valider 100%
2. Optimisation génétique 32-cores (point #2) — fire-and-forget pendant qu'on fait autre chose
3. Pendant que l'optim tourne : investiguer RSI Daily faible turnover (point #4)
4. Setup démo live (point #3) avec le profil 4-charts (point #6)
