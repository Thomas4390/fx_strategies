# Plan — Amélioration CAGR + cleanup features + qualité résultats

> **Date création** : 2026-05-04 · **Cible** : agent fresh ou Thomas en
> session suivante · **Statut** : OUVERT
>
> Plan structuré pour améliorer le CAGR du portfolio FxMultiSleeve au-delà
> du plafond actuel ~7-12%, supprimer les features qui n'apportent pas de
> valeur (DDCap notamment), inspecter rigoureusement la qualité des trades
> et des chiffres, et limiter l'overfit via méthodes statistiques avancées.

---

## 0. Contexte essentiel — état avant ce plan

### 0.1 Findings cumulés sessions précédentes

| Investigation | Verdict | Ref |
|---|---|---|
| RSI Daily VBT vs MT5 | Coûts broker 30-40 bps explainent l'écart | `docs/investigations/rsi_daily_vbt_vs_mt5.md` |
| Sweep 2D (vol_target × max_lev) | Sharpe pic ~1.10 sur diagonale haute | `reports/optimization/optim_complete_*.png` |
| Sweep 3D walkforward IS/OOS | ρ_CAGR=+0.71 stable, 3 régimes (defensif/balanced/aggressif) | `reports/optimization/walkforward_3d/findings.md` |
| Sweep 4D agressif (vt jusqu'à 2.0, lev 80) | Plafond OOS +9.54% sur fenêtre 2024-2026 | `reports/optimization/walkforward_aggressive/findings.md` |
| **DDCap utility test** | **Inutile à 0.15, redondant** | `commit 7150bce` |
| **Walkforward N=5** | **Plafond OOS variable [+0%, +22%]**, fold5 défavorable | `reports/optimization/walkforward_n5/findings.md` |
| **Expansion paires (6 candidates)** | **0/6 améliore portfolio** | `reports/optimization/expansion_pairs/findings.md` |

### 0.2 Config actuelle (commit `dfceb85`)

```
Allocations : 80/10/10 (MR Macro / TS Momentum / RSI Daily)
TargetVol   : 0.28 / MaxLev : 12 / VolFloor : 0.02
DDCap       : 0.30 (était 0.15, relâché)
MarginCap   : 0.70 (jamais activé en pratique)
Paires      : EUR/USD, GBP/USD, USD/JPY, USD/CAD
Broker      : SquaredFinancial Demo (.c suffix)
Baseline    : Sharpe 1.15, CAGR ~7%, MaxDD ~7% (5.4 ans)
```

### 0.3 Bugs/limitations connus

1. **`Inp_*_Pairs` hard-fail si paire sans D1 history** — empêche test
   EURJPY (broker start 2022-11-04). Cf. `FxSleeveTSMomentum.mqh:67-73`.
2. **Sharpe -5.00 capped** observé sur configs MR Macro qui plantent —
   valeur sentinelle MT5 quand TesterStatistics retourne -inf.
3. **`OnTester().years` calculé via deals** — incorrect si 0 deal. Hardcoder
   à `(to_date - from_date)/365.25` plus robuste.
4. **`DDCap=0.15` (initial) freinait 24% configs** sans bénéfice OOS.
5. **MarginCap=0.70** jamais activé sur 5.4 ans testés. Possiblement à
   retirer aussi.
6. **`vol_target/vol_floor` sature avant `max_lev`** — broker leverage 80
   inexploitable sans pousser vt à ~6 (non-physique).

---

## 1. Objectifs et priorités

### 1.1 Objectifs primaires

1. **CAGR robuste ≥ 10% moyen sur N=5 walk-forward** (au lieu de 7-12%
   variable actuel).
2. **Sharpe robuste ≥ 1.0** sur tous les folds (pas juste 4/5).
3. **DD réel sous 15%** sans circuit-breaker artificiel (= edge intrinsèque).

### 1.2 Objectifs secondaires

- Code cleanup : retirer DDCap si vraiment inutile, fix bugs identifiés.
- **Qualité de mesure** : per-trade analysis, distribution PnL, identifier
  trades aberrants.
- **Robustesse statistique** : PSR (Probabilistic Sharpe Ratio), bootstrap
  CI, DSR (Deflated Sharpe Ratio).

### 1.3 Hors scope

- Changement broker (suppose même broker SquaredFinancial)
- Réécriture sleeves from scratch
- Live trading deployment

---

## 2. Phases du plan (ordonnées)

### Phase A — Cleanup code et fix bugs (1-2h)

**But** : éliminer le bruit avant d'optimiser.

#### A.1 Confirmer DDCap inutile et le retirer (ou désactiver par défaut)

```mql5
// Option 1 : Retrait complet
// Supprimer Inp_DDCap, Inp_EnableDDCap, et CheckDDCircuitBreaker()
// dans FxRiskManager.mqh + FxMultiSleeve.mq5

// Option 2 : Désactiver par défaut (plus prudent, ré-activable)
input bool   Inp_EnableDDCap = false;   // était true
input double Inp_DDCap       = 0.30;
```

**Choix recommandé** : Option 2 (désactivé par défaut, ré-activable). Garde
le code comme tail-risk insurance optionnelle.

**Test** : Re-run baseline 4-pair → vérifier CAGR/Sharpe identiques (DD-cap
pas activé en config normale, donc no-op).

#### A.2 Évaluer si MarginCap aussi inutile

```bash
python scripts/optimization/run_mt5_optimization.py \
    --vt-start 0.28 --vt-stop 0.28 --vt-step 0.01 \
    --lev-start 12 --lev-stop 12 --lev-step 1 \
    --vfloor-grid 0.02 \
    --fixed-input Inp_EnableMarginCap=false \
    --fixed-input Inp_EnableDDCap=false \
    --from-date 2020.11.23 --to-date 2026.04.30
```

→ Si résultat identique au baseline (Sharpe 1.15, CAGR 7.24%) : MarginCap
inutile aussi. Désactiver par défaut.

#### A.3 Fix `OnTester().years` (mauvais quand 0 deal)

`FxMultiSleeve.mq5:OnTester()` utilise `HistoryDealGetInteger` pour
détecter la période. Si 0 deal → years=1.0 par défaut, fausse le CAGR.

```mql5
// Plus robuste : utiliser TesterStatistics(STAT_TESTER_FROM/TO_DATE)
// OU calculer depuis les inputs date de la session
double years = (TesterStatistics(STAT_TESTER_TO_DATE) -
                TesterStatistics(STAT_TESTER_FROM_DATE)) / 31557600.0;
```

#### A.4 Fix `FxSleeveTSMomentum` hard-fail sans D1 history

```mql5
// Au lieu de :
if(!EnsureHistory(m_pairs[i], PERIOD_D1, 1))
{
    g_logger.Error(m_name, "no D1 history at all; sleeve disabled");
    return false;  // ← hard-fail
}

// Faire :
if(!EnsureHistory(m_pairs[i], PERIOD_D1, 1))
{
    g_logger.Warn(m_name, StringFormat(
        "%s: skipped (no D1 history)", m_pairs[i]));
    m_pairs[i] = "";  // marquer comme skipped, ne pas trader
    continue;
}
// Skip toutes les pairs avec m_pairs[i] == "" dans OnNewBarD1
```

**Bénéfice** : permet de tester EURJPY (history start 2022-11) sans
restreindre toute la fenêtre.

#### A.5 Investigate Sharpe -5.00 floor

Reproduire un run MR Macro standalone sur paire `EURGBP` qui donne
Sharpe=-5.00. Inspecter `TesterStatistics(STAT_SHARPE_RATIO)` raw value.
Si MT5 cappe à -5, documenter dans `OnTester` ou utiliser une métrique
alternative (Sortino, Calmar).

**Output Phase A** : commits atomiques avec compile checks. EA recompilé.

---

### Phase B — Inspection qualité trades (2-3h)

**But** : comprendre OÙ vient l'edge, identifier trades aberrants/lucky.

#### B.1 Extract deal log par sleeve

Modifier `OnTester()` ou `OnDeinit()` pour exporter en CSV un per-trade :

```mql5
// FILE_COMMON CSV : deals_<run_id>.csv
HistorySelect(0, TimeCurrent());
for(int i = 0; i < HistoryDealsTotal(); i++)
{
    ulong tk = HistoryDealGetTicket(i);
    FileWrite(h,
        TimeToString((datetime)HistoryDealGetInteger(tk, DEAL_TIME)),
        HistoryDealGetString(tk, DEAL_SYMBOL),
        HistoryDealGetInteger(tk, DEAL_MAGIC),
        HistoryDealGetInteger(tk, DEAL_TYPE),
        HistoryDealGetDouble(tk, DEAL_VOLUME),
        HistoryDealGetDouble(tk, DEAL_PRICE),
        HistoryDealGetDouble(tk, DEAL_PROFIT),
        HistoryDealGetDouble(tk, DEAL_COMMISSION),
        HistoryDealGetDouble(tk, DEAL_SWAP));
}
```

#### B.2 Analyse Python — `scripts/analysis/inspect_trades.py`

Pour chaque sleeve (par magic number) :

| Métrique | Formule | Cible |
|---|---|---|
| **Win rate** | `wins/total` | > 50% pour MR, > 35% pour TS |
| **Avg win/avg loss** | `mean(P>0) / mean(P<0)` | > 1.0 |
| **Profit factor** | `sum(P>0) / abs(sum(P<0))` | > 1.3 |
| **Distribution PnL** | histogramme | pas de fat tail unique |
| **Top 5 trades** | sort | < 30% du PnL total (pas dépendant d'1 lucky shot) |
| **Bottom 5 trades** | sort | < 30% du loss total |
| **Holding time** | `exit_time - entry_time` | aligné avec timeframe sleeve |
| **Entry hour distribution** | histogramme par heure UTC | MR : 6-14h, TS/RSI : 21h |
| **PnL par jour de la semaine** | groupby weekday | pas de bias jour spécifique |
| **PnL par mois** | groupby month | pas de saisonnalité forte |
| **PnL par paire** | groupby symbol | pas dépendant d'1 paire seule |

#### B.3 Detection trades aberrants

- **Trade > 5σ du PnL moyen** : flag, vérifier date/news event
- **Holding > 3× médian** : flag, vérifier si SL touché ou exit normal
- **Slippage anormal** : `(executed_price - signal_price)` vs spread moyen
- **Trades simultanés** : 4 paires open en même temps → margin saturée ?

#### B.4 Inspection du filtre macro

`MR_Macro` filtre via `macro_ok` (spread 10Y-2Y > 0.5 + chômage non-rising).
Mesurer :
- Combien de signaux **bloqués** par le filtre macro (vs quand `macro_ok=true`) ?
- Sur la fenêtre fold5 (2025-11→2026-04), `macro_ok` est-il bloqué tout le temps ?
- Performance théorique **sans** filtre macro vs **avec** sur chaque fold ?

```bash
# Test : MR Macro sans filtre macro (Inp_MacroSourceMode=FILE avec macro_cache.csv
# qui contient toujours macro_ok=1)
python src/mt5/bridge/run_backtest_cli.py \
    --input Inp_MacroSourceMode=0 \
    --input Inp_MR_SpreadThresh=0.0 \
    --report-name no_macro_filter
```

**Hypothèse** : si fold5 est bad parce que macro filter rejette tout,
relâcher le seuil pourrait débloquer.

**Output Phase B** : `reports/analysis/trade_inspection_<ts>.html`
(per-sleeve breakdown), `reports/analysis/macro_filter_impact.csv`.

---

### Phase C — Allocation sweep (1h)

**But** : tester si 80/10/10 est sub-optimal.

#### C.1 Grid allocations

Bornes (sum=1.0 imposé par EA) :

| Allocation | MR | TS | RSI |
|---|---|---|---|
| Conservative | 0.50 | 0.25 | 0.25 |
| Balanced | 0.60 | 0.20 | 0.20 |
| Current | 0.80 | 0.10 | 0.10 |
| MR-heavy | 0.90 | 0.05 | 0.05 |
| Equal | 0.34 | 0.33 | 0.33 |
| TS-heavy | 0.40 | 0.50 | 0.10 |

= 6 variants × N=5 folds = 30 backtests.

```python
# Réutiliser walkforward_n5.py avec bornes vt/lev fixes
# et ajouter sweep allocations via Inp_Alloc*
```

#### C.2 Risk-parity dynamique

Si `combined_portfolio.py` Python a déjà `_compute_weights_ts(allocation="risk_parity")`,
porter ce calcul vers MT5 via une routine `OnTimer()` quotidienne qui :
1. Appelle `risk.BuildDailyEquityReturns()` par sleeve (séparer en 3)
2. Calcule inverse-vol weights avec lookback 63j shift 1
3. Override `m_alloc_*` du `CRiskManager`

**Bénéfice attendu** : amortit fold5 en réduisant alloc MR si sa vol
récente est élevée vs ses returns.

**Output Phase C** : `reports/optimization/allocations_<ts>.csv` + heatmap.

---

### Phase D — Expansion timeframe (H1/H4) (2-3h)

**But** : trouver edge entre M1 (MR Macro) et D1 (TS/RSI).

#### D.1 Création nouveau sleeve `H1_Momentum` (refonte)

Comme TS Momentum mais sur H1 :
- EMA 20/50 sur H1 → trend regime
- ATR(14) H1 → SL dynamic
- Entry sur H1 close, exit sur opposite EMA cross
- Pairs = EUR/USD, GBP/USD, USD/JPY (mêmes que TS Momentum)
- Allocation : 5% (reprendre sur RSI ou MR)

#### D.2 Test isolé H1_Momentum

Sharpe standalone sur 5.4 ans. Si > 0.5 → ajout au portfolio.

#### D.3 Walk-forward H1_Momentum incrémental

Comparer V0 (4 paires actuelles) vs V_new (avec H1_Momentum 5%) sur N=5 folds.

**Décision** : retient si ΔSharpe ≥ +0.05 sur médian fold.

**Output Phase D** : nouveau `FxSleeveH1Momentum.mqh` + tests + findings.md.

---

### Phase E — Refonte sleeves existants (2-3h)

**But** : améliorer chaque sleeve individuellement.

#### E.1 MR Macro : ajuster session

Tester fenêtres alternatives :
- 6-14h UTC (actuel)
- 8-16h UTC (London + early NY)
- 13-21h UTC (NY full)
- 0-23h UTC (24h)

Mesurer Sharpe sleeve isolé par session.

#### E.2 TS Momentum : EMA optimization

Grid search sur EMA fast/slow :
- Fast : 10, 14, 20, 30, 50
- Slow : 30, 50, 100, 200

20 combos × 5 folds = 100 backtests. Filtre PBO < 50%.

#### E.3 RSI Daily : seuils oversold/overbought

Cf. findings RSI Daily — réduire fréquence trades pour amortir coûts.

Grid : (oversold, overbought, exit_mid)
- (20, 80, 50)
- (25, 75, 50) actuel
- (30, 70, 50)
- (25, 75, 55)
- (20, 80, 60)

**Output Phase E** : trois `findings_<sleeve>.md` avec params optimaux
robustes.

---

### Phase F — Nouveau sleeve carry trade (3-4h, optionnel)

**But** : edge orthogonal aux 3 sleeves existants.

#### F.1 Hypothèse

Carry trade FX = profit du différentiel de taux entre 2 devises. Sur
G10 :
- AUD/USD long si RBA > Fed (taux directeurs)
- NZD/USD long si RBNZ > Fed
- USD/JPY long quasi-permanent (BoJ < Fed)

Pas dépendant des signaux techniques → décorrélé de MR/TS/RSI.

#### F.2 Implementation

Nouveau sleeve `FxSleeveCarry.mqh` :
- Inputs : taux directeurs FRED (FEDFUNDS, ECB MRO, BoE Bank Rate, BoJ
  policy rate)
- Update mensuel : recompute carry per pair
- Open : long pair high-yield vs USD si carry > 200 bps annualized
- Risk : vol-target par pair 5%

**Pre-flight** : vérifier que les taux directeurs sont accessibles via
FRED (FxMacroSourceFRED) ou hardcoder via `macro_history.csv`.

#### F.3 Allocation

Ajouter sleeve 4 avec 5-10% allocation. Réduire MR Macro à 70-75% en
contrepartie.

**Output Phase F** : nouveau sleeve + tests + walkforward + findings.

---

### Phase G — Validation anti-overfit avancée (2h)

**But** : confirmer que les améliorations Phase A-F ne sont pas du
data-mining.

#### G.1 Probabilistic Sharpe Ratio (PSR)

Pour chaque config retenue, calculer :

```python
# scipy.stats normal CDF
PSR(SR_observed, SR_threshold=0)
  = Phi( (SR_observed - SR_threshold) * sqrt(N-1) /
         sqrt(1 - skew*SR + (kurt-1)/4 * SR^2) )
```

Critère : **PSR(SR > 0) ≥ 95%** = la config a un edge statistiquement
significatif.

#### G.2 Deflated Sharpe Ratio (DSR)

Corrige pour le nombre de trials testés (multiple comparison).

```python
DSR = Phi( (SR_observed - E[SR_max | N_trials]) * ... )
# E[SR_max] = expected max under null hypothesis with N independent trials
```

Si on a testé 720 configs et le best Sharpe = 1.20, DSR ajuste pour
sélection.

#### G.3 Bootstrap CI

Block bootstrap sur les daily returns, B=1000 iterations.

```python
# Pour chaque iteration :
# 1. Sample blocks de 21 jours avec remplacement
# 2. Compute Sharpe / CAGR / MaxDD sur la trajectoire bootstrapped
# 3. Aggréger : P5/P50/P95 des métriques

# Critère : P5(Sharpe) > 0 ET P5(CAGR) > 0
```

#### G.4 Reality Check (White)

Test si la meilleure config est statistiquement meilleure que zéro,
en tenant compte de toutes les configs testées.

**Output Phase G** : `reports/anti_overfit/<ts>/{psr,dsr,bootstrap}.csv`
+ verdict pour chaque config retenue Phase A-F.

---

### Phase H — Synthèse et recommandation finale (1h)

#### H.1 Tableau comparatif

| Config | CAGR_avg | Sharpe_med | MaxDD | PSR | DSR | Verdict |
|---|---|---|---|---|---|---|
| Baseline (V0) | 7-12% | 1.0-1.6 | 7-9% | 90% | 65% | OK |
| Phase A cleanup | ? | ? | ? | ? | ? | ? |
| Phase C alloc | ? | ? | ? | ? | ? | ? |
| Phase D +H1 | ? | ? | ? | ? | ? | ? |
| Phase E refonte | ? | ? | ? | ? | ? | ? |
| Phase F +carry | ? | ? | ? | ? | ? | ? |
| **Combiné final** | **TARGET ≥ 10%** | **TARGET ≥ 1.0** | **< 15%** | **≥ 95%** | **≥ 80%** | ? |

#### H.2 Updates recommended

- [ ] Modif défauts compilés `FxMultiSleeve.mq5`
- [ ] Mise à jour `src/mt5/SESSION_NOTES.md` + `CLAUDE.md`
- [ ] Update `reports/client_setup_guide/main.tex` avec nouveau baseline

---

## 3. Critères de succès / arrêt

### 3.1 Critères de succès cumulatifs

Une nouvelle config est retenue ssi **TOUS les critères suivants sont
remplis** :

1. **CAGR_avg** ≥ baseline + 1pp (statistiquement significatif)
2. **ΔSharpe_med** ≥ +0.05 sur N=5 folds
3. **PSR** ≥ 95% (edge significatif)
4. **DSR** ≥ 80% (pas data-mining)
5. **MaxDD** ≤ baseline + 2pp
6. **Spearman ρ CAGR IS↔OOS** ≥ +0.50 sur tous les folds

### 3.2 Critères d'arrêt précoce

- **Phase A** échoue → STOP (cleanup pas safe, garder existant)
- **Phase B** révèle trade unique > 30% PnL total → ALERT (lucky, pas robuste)
- **Phase C** allocation 80/10/10 reste optimal → SKIP Phase D-F
- **Phase G** rejette toutes les configs Phase D-F → REVERT au baseline

### 3.3 Critères pivot

Si plafond CAGR reste < 10% après Phase E :
- Pivot vers Phase F (carry) en priorité
- Sinon, considérer changement broker (hors-scope)

---

## 4. Anti-overfit guards (transversal)

Appliquer **à chaque phase** :

1. **Walk-forward IS/OOS** systématique (pas IS-only)
2. **Spearman ρ** logué pour chaque test (rang stable)
3. **PBO** calculé pour les sweeps > 20 configs
4. **Hold-out fenêtre** : garder fold5 (2025-11→2026-04) pour validation
   finale uniquement, ne PAS optimiser dessus
5. **Param parsimony** : préférer config avec moins de degrés de
   liberté à performance égale
6. **Plateau detection** : valoriser configs avec voisins similaires (±1
   step) plutôt que pic isolé

---

## 5. Inspection qualité chiffres (transversal)

Pour chaque test :

1. **TesterStatistics raw** : extraire toutes les valeurs MT5 (pas juste
   les parsed) pour cross-check
2. **Compare html report** : sanity check que CAGR/Sharpe/DD écrits dans
   l'HTML sont cohérents avec OnTester custom output
3. **Per-trade reconciliation** : `sum(deal.profit) == html.netProfit` ?
4. **Slippage realized** : `mean(executed_price - requested_price)` vs
   `Inp_*_SlippageBps` configuré
5. **Marge utilisée max** : `max(margin/equity)` durant le run vs cap 70%
6. **Pas de NaN/Inf** : valider TOUTES les métriques exportées

---

## 6. Code et scripts à créer

```
scripts/
├── analysis/
│   ├── inspect_trades.py          # Phase B
│   ├── macro_filter_impact.py     # Phase B.4
│   └── per_sleeve_pnl.py          # Phase B
├── optimization/
│   ├── walkforward_allocations.py # Phase C
│   ├── walkforward_h1.py          # Phase D (nouveau sleeve)
│   ├── walkforward_session.py     # Phase E.1 (MR session)
│   ├── walkforward_ema.py         # Phase E.2 (TS EMA)
│   └── walkforward_rsi_thresh.py  # Phase E.3
├── anti_overfit/
│   ├── psr_dsr.py                 # Phase G.1, G.2
│   ├── bootstrap_ci.py            # Phase G.3
│   └── reality_check.py           # Phase G.4
└── synthesis/
    └── final_comparison.py        # Phase H

src/mt5/
├── Experts/FxMultiSleeve.mq5      # Phase A.1, A.3, A.4
├── Include/FxRiskManager.mqh      # Phase A.1, A.2
├── Include/FxSleeveTSMomentum.mqh # Phase A.4
├── Include/FxSleeveH1Momentum.mqh # Phase D (nouveau)
└── Include/FxSleeveCarry.mqh      # Phase F (nouveau, optionnel)
```

---

## 7. Reproduction commands (squelette pour future session)

### Setup environnement
```bash
cd /home/thomas/Documents_Thomas/11_CodingProjects/fx_strategies/fx_strategies
git pull origin main
git log --oneline -10  # confirmer commits cleanup d'aujourd'hui

# Vérifier MT5 dispo
pgrep -af terminal64.exe || echo "MT5 OK to launch"
```

### Phase A — cleanup
```bash
# A.1 : DDCap default off
# Edit src/mt5/Experts/FxMultiSleeve.mq5 ligne 30 : Inp_EnableDDCap = false
# Recompile
WINEPREFIX=/home/thomas/.mt5 wine \
  "/home/thomas/.mt5/drive_c/Program Files/MetaTrader 5/MetaEditor64.exe" \
  /compile:"Z:\\<path>\\FxMultiSleeve.mq5" /log

# Test baseline
python src/mt5/bridge/run_backtest_cli.py --report-name baseline_no_dd
# Vérifier Sharpe ≈ 1.15 (config inchangée fonctionnellement)

git commit -am "feat(mt5): disable DDCap by default (no-op in normal regime)"
```

### Phase B — trade inspection
```bash
# B.1 : run baseline avec deal log enabled (modifier OnDeinit pour FileWrite deals)
python src/mt5/bridge/run_backtest_cli.py --report-name with_deals
# Le CSV deals_<ts>.csv sera dans Common/Files/

# B.2 : analyse Python
python scripts/analysis/inspect_trades.py \
    --deals reports/mt5/deals_*.csv \
    --output reports/analysis/trade_inspection.html

# B.4 : test sans filtre macro
python src/mt5/bridge/run_backtest_cli.py \
    --input Inp_MR_SpreadThresh=-1.0 \
    --report-name no_macro_filter
```

### Phase C — allocations
```bash
python scripts/optimization/walkforward_allocations.py
# Output : reports/optimization/allocations/findings.md
```

### Phase D — H1 momentum
```bash
# Compile new sleeve
WINEPREFIX=/home/thomas/.mt5 wine ... MetaEditor64.exe /compile:FxMultiSleeve.mq5

python scripts/optimization/walkforward_h1.py
```

### Phase E — refonte sleeves
```bash
python scripts/optimization/walkforward_session.py    # MR Macro session
python scripts/optimization/walkforward_ema.py        # TS Momentum EMA
python scripts/optimization/walkforward_rsi_thresh.py # RSI Daily seuils
```

### Phase F — carry (optionnel)
```bash
# Compile new carry sleeve
# Run incremental + walkforward
python scripts/optimization/walkforward_carry.py
```

### Phase G — anti-overfit
```bash
python scripts/anti_overfit/psr_dsr.py --input <best_config>
python scripts/anti_overfit/bootstrap_ci.py --input <best_config>
python scripts/anti_overfit/reality_check.py --all-tested-configs
```

### Phase H — synthèse
```bash
python scripts/synthesis/final_comparison.py --baseline V0 --candidates *
# Génère reports/optimization/cagr_improvement_final.md
```

---

## 8. Estimations temps total

| Phase | Heure | Auto/Manuel |
|---|---|---|
| A — cleanup | 1-2h | M (code review) |
| B — trade inspection | 2-3h | A+M (long: extract + analyze) |
| C — allocations | 1h | A (sweep auto) |
| D — H1 sleeve | 2-3h | M+A (code+test) |
| E — refonte sleeves | 2-3h | A (3 sweeps) |
| F — carry (optionnel) | 3-4h | M+A |
| G — anti-overfit | 2h | A |
| H — synthèse | 1h | M |
| **Total core (A-E+G+H)** | **9-12h** | |
| **Avec F** | **12-16h** | |

Réaliste sur 2-3 sessions de travail.

---

## 9. Pour reprendre en nouvelle session

Lire dans cet ordre :

1. `src/mt5/SESSION_NOTES.md` (état projet)
2. `src/mt5/CLAUDE.md` (env Windows, broker)
3. **Ce document** (cagr_improvement_plan.md)
4. `reports/optimization/walkforward_n5/findings.md` (constat plafond
   variable)
5. `reports/optimization/expansion_pairs/findings.md` (paires écartées)

Puis exécuter dans l'ordre Phase A → B → C ... → H.

**Blocage prévisible** : Phase A.4 (fix TS Momentum hard-fail) demande
modif EA + recompile + retest. Si bug introduit, revert atomique via git.

---

## 10. Annexe — métriques à logger systématiquement

Pour chaque backtest exécuté dans les phases :

```python
{
  "config_id": str,
  "phase": "A|B|C|D|E|F|G|H",
  "from_date": str, "to_date": str,
  "fold_id": int,         # 1-5 si walk-forward N=5
  "is_or_oos": "is|oos",
  # Performance
  "cagr_pct": float,
  "sharpe": float,
  "sortino": float,
  "calmar": float,
  "max_dd_pct": float,
  "balance_dd_pct": float,
  "total_return_pct": float,
  # Trades
  "total_trades": int,
  "wins": int, "losses": int,
  "win_rate": float,
  "avg_win": float, "avg_loss": float,
  "profit_factor": float,
  "recovery_factor": float,
  # Risk
  "max_margin_used_pct": float,
  "avg_lev_realized": float,
  "max_lev_realized": float,
  # Robustness
  "spearman_is_oos": float,
  "psr_zero": float,
  "dsr_n_trials": float,
  "bootstrap_p5_sharpe": float,
  "bootstrap_p5_cagr": float,
  # Inputs
  "inputs_changed": dict,
}
```

Stocker en `reports/optimization/<phase>/<ts>.csv` pour reproductibilité
totale.
