# Audit critique réalisme code MQL5 — FxMultiSleeve (2026-05-06)

> **Auteur**: Thomas Vaudescal (assisté Claude)
> **Date**: 2026-05-06
> **Scope**: code MQL5 uniquement (`src/mt5/Experts/FxMultiSleeve.mq5` + 13 includes)
> **Méthode**: lecture critique fichier-par-fichier + recherches Internet pour valider les hypothèses sur les conventions broker, le tester MT5 et MQL5
> **Statut**: audit préliminaire complet — patches NON appliqués, propositions uniquement
> **Baseline auditée**: `Sharpe 0.98 / CAGR 10.79% / MaxDD 19.73% / 774 trades` (post-Phase M.7, `reports/qc_phase_i/mt5_c1_phase_m7_final.json`)

---

## 0. Synthèse exécutive

**Total findings : 60+ (5 CRITIQUES, 17 HAUTES, 18 MOYENNES, ~20 BASSES)**

Trois familles de problèmes identifiées, classées par impact estimé sur le réalisme :

1. **Tester MT5 fondamentalement biaisé** (CRITIQUE). Le backtest tourne en `Model=1` (1-minute OHLC) qui interpole les SL/TP à l'intérieur de la barre avec des fills "parfaits" — les sources MT5 et plusieurs études confirment que ce mode produit systématiquement de meilleurs résultats que la réalité tick-level. Le mode rigoureux est `Model=4` ("real ticks"). De plus, la documentation de la Phase M.6 inverse la nomenclature `Model=1`/`Model=2` par rapport aux docs MT5 officielles, ce qui rend la conclusion « tick-mode plus conservateur que OHLC » **invalide** : la phase a comparé OHLC (1.01) vs Open prices only (1.05), pas every-tick vs OHLC.
2. **Modélisation des coûts non alignée avec un broker concret** (CRITIQUE). `Inp_CommissionBpsPerSide=2.0` ne correspond ni à OANDA Standard (commission=0, spread-only) ni à OANDA Core/IC Markets Raw (≈ 3-5 bps/side). Le swap rollover overnight n'est pas modélisé explicitement (laissé à la discrétion du broker tester). Les spreads réels sont « floating » mais limités à la résolution minute — pas de spread spike news (NFP/CPI/FOMC = +8-10 pips momentanés).
3. **Vol-targeting + sizing approximatifs** (HAUTE). `BuildDailyEquityReturns()` approxime les rendements daily en divisant le P&L par l'equity COURANTE (pas l'equity au jour J), ignore le P&L flottant des positions ouvertes, et compte 80 jours calendaires (≈ 56 trading days, pas 80). La σ21/σ63 est donc biaisée pendant les périodes de DD important ou de positions long-held. La leverage TS Momentum mélange `lev_pair × GlobalLeverage()` ce qui double-leverage potentiellement (3× × 64× = 192× max théorique sur une paire seule).

Recommandation court terme :
- **Re-run Sharpe 0.98 avec `Model=4` (real ticks)** avant toute autre modification — les 5.4 ans MT5 broker `.c` ont les real ticks dispo. Sharpe attendu inférieur (~0.7-0.9 estimé).
- **Patcher `Inp_CommissionBpsPerSide` en fonction du broker cible** (0 si OANDA Standard, 3-5 si Core/Raw) avec input reflétant la réalité broker.
- **Audit `BuildDailyEquityReturns()`** pour corriger l'approximation vol et inclure les positions ouvertes.

---

## 1. Findings par fichier

> Format: `ID | Sévérité | Fichier:ligne | Problème | Fix proposé | Réf`. Réf = ✓ si validé Internet (URL en § 7).

### 1.1 `src/mt5/Experts/FxMultiSleeve.mq5` (EA orchestrateur)

| ID | Sév | Fichier:ligne | Problème | Fix proposé | Réf |
|---|---|---|---|---|---|
| **EA-01** | CRITIQUE | bridge:65 (`run_backtest_cli.py`) | `DEFAULT_MODEL = 1` (1-min OHLC). Le mode OHLC interpole les fills SL/TP à l'intérieur des bars avec des entrées « parfaites », surestimant le Sharpe selon les études tickstory/forex factory. Mode rigoureux = `Model=4` (real ticks). | Changer `DEFAULT_MODEL=4` ou exposer comme arg CLI `--model`, vérifier broker `.c` dispose des real ticks 2020-11→2026-04. Re-run baseline complet, attendu Sharpe ↓ -0.05 à -0.20. | ✓ |
| **EA-02** | CRITIQUE | docs/investigations/phase_m4_m7_realism_corrections.md:124 | « Diagnostic Model=2 OHLC vs Model=1 every-tick » est inversé : MT5 docs officielles disent `Model=0 = every tick simulé, Model=1 = 1-min OHLC, Model=2 = Open prices only, Model=3 = math, Model=4 = real ticks`. La Phase M.6 a comparé OHLC (Sharpe 1.01) vs Open prices only (1.05). Conclusion « tick mode plus conservateur que OHLC » → **invalide**. | Re-documenter Phase M.6 avec nomenclature correcte. Refaire diagnostic Model=4 (real ticks) vs Model=1 (OHLC) pour mesurer le vrai « impact intra-bar ». | ✓ |
| EA-03 | HAUTE | mq5:266-291 | `OnTick()` itère sur 4 paires MR Macro à chaque tick (`g_sleeve_mr.OnNewBarM1`). En mode real ticks, ~31M ticks total → 124M bar-detection checks. Performance OK mais le check `iTime != m_last_m1_bar[i]` peut fire avant la *vraie* close M1 (race avec le générateur de ticks). | Tester avec `Model=4` real ticks pour confirmer si l'effet est mineur. Sinon, ajouter un buffer `>=5s past minute boundary` comme initialement prévu en Phase M.6. | — |
| EA-04 | HAUTE | mq5:295-329 (OnTimer) | Selon plusieurs threads MQL5, `OnTimer` ne tire **PAS** dans le Strategy Tester en MT5 ancienne version, mais peut être appelé via OnTick fallback. À vérifier en debug : la `Recompute Daily 21h` sleeves passe-t-elle vraiment via OnTimer ou est-elle accidentellement déclenchée via le first-tick-after-21h dans OnTick ? Logs `[DAILY][INFO] Daily recompute done at hour=21 UTC` confirment que ça tire mais le mécanisme exact n'est pas documenté. | Ajouter assertion explicite `IsStrategyTester() ? "via OnTick" : "via OnTimer"` dans le log. Tester si OnTimer firing déterministe en backtest avec `Model=4`. | ✓ |
| EA-05 | HAUTE | mq5:268-270 | `CheckDDCircuitBreaker()` appelé sur **chaque tick** (~31M en real ticks). Performance OK (équivaut à 31M `AccountInfoDouble` calls), mais sur un EA live à 100Hz pendant news = ~360k checks/heure. Pas un bug, juste à mentioner. | Réduire fréquence si problème : check toutes les 1s via timestamp diff. Pas urgent. | — |
| EA-06 | MOYENNE | mq5:181 | `g_session_start = TimeCurrent()` capturé en OnInit. En tester, c'est `tester_start_date`. OK. En live, c'est l'heure actuelle au load — donc Sharpe live computation `years` part du moment où l'EA est attaché, pas du début du compte. Cohérent intent. | OK as-is. | — |
| EA-07 | MOYENNE | mq5:317-318 | `if(today != g_last_d1_bar && hour_utc >= Inp_DailyRecomputeHr)` — si l'EA est attaché à 22h UTC le jour J, le first OnTimer fires immédiatement le DailyRecompute du jour J. Si attaché à 20h UTC, attendra jour J+1 21h. Edge case startup mais pas crash. | Documenter ou ajouter input `Inp_SkipFirstDailyRecompute=true` si on veut éviter le recompute immédiat au startup. | — |
| EA-08 | BASSE | mq5:393-397 | Log `[OPTIM]` produit dans OnTester pour un test simple (pas optim) — utile pour parsing automatique. OK. | — | — |

### 1.2 `src/mt5/Include/FxRiskManager.mqh`

| ID | Sév | Fichier:ligne | Problème | Fix proposé | Réf |
|---|---|---|---|---|---|
| **RM-01** | CRITIQUE | FxRiskManager.mqh:118-154 | `BuildDailyEquityReturns()` approxime les returns daily en divisant le P&L par l'equity **courante** (`equity = AccountInfoDouble(ACCOUNT_EQUITY)` au moment du recompute, pas au jour J). Pendant un grand DD, l'equity courante est BASSE → returns artificiellement gonflés → σ surestimée → leverage sous-estimé → sizing trop petit. Pendant un rally, inversement. Comments du code mentionnent explicitement « approximation : pas d'equity par-jour disponible nativement ». | Implémenter equity-curve par jour via `HistoryDealsTotal` cumulatif depuis tester_start. Coût compute : ~O(n_deals × n_days) ≈ 1700×80 = 136k ops, négligeable. | — |
| **RM-02** | CRITIQUE | FxRiskManager.mqh:118-154 | `daily_pnl[day_idx] += profit + comm + swap` — n'inclut QUE les *closed* deals (`HistoryDealsTotal` retourne deals fermés). Le P&L flottant des positions ouvertes pendant 6h-21h MR Macro ou plusieurs jours pour TS/RSI est ignoré. Sur trending periods, σ peut être largement sous-estimée → leverage trop élevé. | Ajouter le P&L flottant via `PositionGetDouble(POSITION_PROFIT)` pour chaque position ouverte, pondéré par le temps (mark-to-market journalier). | ✓ |
| RM-03 | HAUTE | FxRiskManager.mqh:118-128 | `from = now - lookback_days * 86400` = 80 jours **calendaires**, pas trading. Sur 80 calendar days = ~56 trading days. La σ21 trading est calculée sur 21 calendar slots dont ~6 sont weekends (rets[]=0). Effectif: σ sur 15 trading days réels. | Filtrer weekends (`if(day_idx is weekend) continue`) ou augmenter lookback à 110 calendar = 80 trading days. | — |
| RM-04 | HAUTE | FxRiskManager.mqh:169-175 | `realized = MathMax(MathMax(σ21, σ63), m_vol_floor)` prend le **plus haut** de σ21 et σ63. Conservative mais peut être inverted intent : si vol récente (σ21) est BASSE et vol long-terme (σ63) HAUTE, on cap au max → leverage très bas. Si user veut leverage adaptatif réactif, devrait prendre σ21 seul. | Préciser intent dans config (input `Inp_VolMode=max | recent | avg`) ou documenter explicitement. | — |
| RM-05 | HAUTE | FxRiskManager.mqh:184-214 | `CheckDDCircuitBreaker` lit `GV_PEAK_EQUITY` global persisted entre EA reload. **PROBLÈME** : si user déploie EA sur **nouveau compte** sans flag `Inp_ResetDDState=true`, peak_equity ancien (potentiellement 0 ou inflated) déclenche DD breaker immédiat. | Reset auto si nouveau compte détecté via `AccountInfoString(ACCOUNT_NAME)` ou `ACCOUNT_LOGIN` change. Sinon documenter strictement la procédure de reset. | — |
| RM-06 | HAUTE | FxRiskManager.mqh:230-260 | `CheckMarginCap` à seuil 50% margin/equity → réduit leverage de 50%. Idempotent mais cumulatif : si le breaker fire 3 fois en 1h, leverage peut tomber à 0.125× du base. Pas de timeout/cooldown. | Ajouter cooldown 1h entre triggers ou ne déclencher qu'1 fois par jour. | — |
| RM-07 | MOYENNE | FxRiskManager.mqh:107-114 | `LotsFor()` fait `risk_money = sub_eq * risk_pct * GlobalLeverage() * extra_lev` puis appelle `LotsForRisk(symbol, risk_money, sl_distance_price)`. Le « risk_pct » est 1% MR Macro / 5% TS/RSI. Avec sub_eq=0.8×equity et lev=64×, MR risk_money par trade = 51% de l'equity total. Si SL hit avec slippage, perte = 51% × SL distance%. **À cross-check avec `daily_momentum.py` / `mr_macro.py` pour s'assurer que les valeurs python correspondent**. | Audit comparé MQL5 vs python sizing. | — |
| RM-08 | MOYENNE | FxRiskManager.mqh:32-40 | Constructor defaults `m_target_vol = FX_VOL_TARGET_GLOBAL = 0.28` (pre-Phase I). Inputs override → 0.75 post-Phase I. OK fonctionnel mais source de confusion si docs lookup les défauts. | Update defaults match Phase I (0.75) ou supprimer constructor defaults pour forcer Init() params. | — |

### 1.3 `src/mt5/Include/FxSleeveMRMacro.mqh`

| ID | Sév | Fichier:ligne | Problème | Fix proposé | Réf |
|---|---|---|---|---|---|
| MR-01 | HAUTE | mqh:101-115 | `OnNewBarM1()` détecte nouvelle bar M1 via `iTime(symbol, PERIOD_M1, 0) != m_last_m1_bar[i]`. En `Model=1` OHLC tester, le tick "open M1" arrive immédiatement à HH:MM:00 → trigger sur first tick after open. **Mais**: le code lit `CopyRates(symbol, PERIOD_M1, 1, 1, last)` (shift=1) = bar fermée précédente. Donc shift=1 OK. Le risque restant est si le *generator de ticks Model=1* place le first tick before bar.high or bar.low — fills auto-injection. | Re-test avec `Model=4` (real ticks) pour valider que les fills SL/TP sont bien plausibles. | — |
| MR-02 | HAUTE | mqh:117-152 | `CheckIntradayExits()` appelé sur **chaque tick**. Pour les exits time-stop / EOD-21h, `m_trade.PositionClose(ticket)` n'applique PAS le slippage shift. **C'est la limitation `Phase M.4` documentée** : seuls les SL/TP triggered exits paient slip. Time-stop = 6h, EOD = 21h. | Wrapper `m_trade.PositionClose` qui pré-pay slippage cost via `risk_money` dampening au moment de l'OPEN (déjà fait via SL/TP shift) — mais pour exits non-SL/TP, le slip réel sera là live mais pas en backtest. À acter comme limitation tester ou patcher avec un fee_per_close hook. | — |
| MR-03 | MOYENNE | mqh:181 | `if(CountSleevePositions(m_magic, m_symbols[idx]) > 0) return;` — pas de pyramiding. Cohérent intent avec `mr_macro.py` (1 position par paire max). | OK. | — |
| MR-04 | MOYENNE | mqh:204-218 | `WarmupBBFromHistory()` reconstitue BB sur `bb_window+20=100` bars M1. Utilise un VWAP **local** qui reset à minuit UTC. Si le warmup demarre EN MILIEU de session UTC, le VWAP local commence cumul à 00:00 UTC mais avec data bars partial → cumul tronqué. CLAUDE.md mentionne « bug 'buffer rempli de zéros' fixé » — vérifier code actuellement OK. | Inspection: ligne 211-217 `warmup_vwap.OnNewBarM1(rates[i])` qui reset auto à minuit UTC interne. **OK fonctionnel**. | — |
| MR-05 | MOYENNE | mqh:241 | `slip_pct = (Inp_MR_SlippageBps + Inp_CommissionBpsPerSide) / 10000.0` = (15+2)/10000 = 0.17%. SL shift = 0.5% + 0.17% = 0.67%. TP = 0.6% - 0.17% = 0.43%. Donc **TP MOINS de profit que SL si triggered**. Coherent slip absorption. | OK Phase M.4 logic. | — |
| MR-06 | MOYENNE | mqh:255 | `slip_pts = (int)MathCeil(slip_price / point)` — convert slippage to broker points. Pour EUR/USD à 1.10, slip 17 bps = 0.0019 = 19 pips = 190 points (5 décimales). `SetDeviationInPoints(MathMax(slip_pts, 5))`. **Très large** : 190 points = 19 pips. Si live spread 1.5 pips, fill se fera ; mais tolérance 19 pips peut accepter de très mauvais fills. | Réduire deviation à ~1.5× spread typique paire (3-5 pips = 30-50 points). | — |
| MR-07 | BASSE | mqh:96 | `m_trade.SetDeviationInPoints(20)` global default = 2 pips. Override per-trade dans OpenPosition. OK. | — | — |
| MR-08 | BASSE | mqh:104 | Comments lines 117-118 confirment intent : "L'EA principal appelle cette fonction sur chaque tick". | — | — |

### 1.4 `src/mt5/Include/FxSleeveTSMomentum.mqh`

| ID | Sév | Fichier:ligne | Problème | Fix proposé | Réf |
|---|---|---|---|---|---|
| **TS-01** | HAUTE | mqh:78-82 | `iMA(pair, PERIOD_D1, fast_ema=20, 0, MODE_EMA, PRICE_CLOSE)` crée handle indicateur D1. Le 4ème paramètre (`shift`) = 0 = pas de décalage handle. **MAIS**: lecture via `ReadShift1(handle, 1, 1)` lit shift=1 du buffer. Donc handle non-shifted + read shift=1 = bar fermée hier. **OK** selon MQL5 docs (« indices décroissent vers le passé »). Aucun look-ahead. | Documenter explicitement dans le code que handle shift=0 ≠ buffer shift=1. | ✓ |
| TS-02 | HAUTE | mqh:141-142 | `bool long_signal = (ema_fast > ema_slow) && (rsi < Inp_TS_RSIHigh=60)`. RSI < 60 = pas en territoire surachat. Cohérent vbt `daily_momentum.py`. | OK. | — |
| **TS-03** | HAUTE | mqh:246-249 | `risk_money = sub_eq * 0.05 * lev_pair * risk.GlobalLeverage() * slip_drag`. **Double leverage**: `lev_pair` (cap 3×) × `GlobalLeverage()` (cap 64×) = 192× max. Vraisemblement bug — vbt `daily_momentum.py` n'utilise QUE leverage=10 fixed (Phase M.1 alignment). Peut expliquer divergence MT5 ↔ vbt. | Vérifier intent : si TS Momentum doit hériter du global leverage via `extra_lev` paramètre du `risk.LotsFor()`, supprimer multiplication par `lev_pair` ici. Sinon supprimer × `GlobalLeverage()`. | — |
| TS-04 | HAUTE | mqh:233 | `sl_dist_safety = price * (0.05 + slip_pct)` = 5% + 0.12% = 5.12%. Très large (vs SL distance ~0.5% MR Macro). C'est le `safety` SL pour cas extrême — le sleeve TS sort sur signal flip, pas SL. Mais 5% fixed peut blow-up si overnight gap > 5% (rare en majors mais possible). | Réduire à ATR-based ou 2% conservative. | — |
| TS-05 | MOYENNE | mqh:122-126 | `OnNewBarD1` itère `m_n_pairs` (3 paires post-Phase E). Pas de filtre macro (TS standalone). OK. | — | — |
| TS-06 | MOYENNE | mqh:182-198 | `ComputePairSigma21` : log returns sur 22 closes → 21 returns. `var = (s2 - 21*mean²)/20` (Welford). Annualisation `sqrt(252)`. Math correct. | — | — |
| TS-07 | MOYENNE | mqh:246-247 | `slip_drag = 1.0 - 2.0 * slip_pct` = 1 - 0.0024 = 0.9976. Sizing dampened ~0.24%. **Hypothèse**: position détenue plusieurs jours, frais cumulatifs swap+commission ≈ 12 bps round-trip. Slip_drag couvre PARTIELLEMENT cette assomption. | Ajouter swap modeling explicite ou augmenter slip_drag pour overnight holdings. | — |
| TS-08 | BASSE | mqh:255-258 | `m_trade.Buy(lots, symbol, price, sl, 0.0, ...)` — TP=0.0 = pas de TP. Position fermée par signal flip (logic OK, vbt même intent). | — | — |
| TS-09 | BASSE | mqh:65-77 | Graceful degradation : si pair n'a pas de D1 history (ex EURJPY pre-2022), skip silently. OK. | — | — |

### 1.5 `src/mt5/Include/FxSleeveRSIDaily.mqh`

| ID | Sév | Fichier:ligne | Problème | Fix proposé | Réf |
|---|---|---|---|---|---|
| RSI-01 | HAUTE | mqh:111-119 | Détection RSI cross via `(rsi_prev >= OS) && (rsi_now < OS)`. shift=1=hier, shift=2=avant-hier. **OK shift logic** (vbt rsi_daily.py même convention). | — | — |
| RSI-02 | HAUTE | mqh:125-139 | Exits sur RSI cross retour à 50 (`exit_long`/`exit_short`). Pas de time-stop. Position peut rester ouverte 30+ jours si RSI oscille. Implication : **swap rollover cumulatif** non modélisé sur ces longues détentions. Phase M.4 dampening 12 bps round-trip ne couvre que entry+exit, pas les ~30 nuits × 0.5 bps swap = 15 bps drag. | Ajouter time-stop maximum (ex 21 jours) ou modeliser swap explicitement via input `Inp_SwapBpsPerNight`. | ✓ |
| RSI-03 | HAUTE | mqh:38-46 | Validation `Inp_RSI_Oversold < ExitMid < Overbought` mais pas de validation que la stratégie est rentable post-Phase M.5/M.7. Sensibilité paramètres OS=25, OB=75 figée. | Sensibilité ±5pts à mesurer en walk-forward. | — |
| RSI-04 | MOYENNE | mqh:194-196 | `risk_money = sub_eq * 0.05 * GlobalLeverage() * slip_drag`. Pas de `lev_pair` (RSI Daily natif lev=1 selon comments). **Différent de TS-03 double-leverage** — RSI Daily semble correctement designed. | OK contrast TS-03 confirme TS-03 = bug. | — |
| RSI-05 | MOYENNE | mqh:185-186 | `sl_dist = price * (0.05 + slip_pct)` = 5.12% safety SL. Idem TS-04, large. | Idem TS-04. | — |
| RSI-06 | BASSE | mqh:96-101 | `OnNewBarD1` pas de filtre macro. OK (RSI standalone, pas concerné par UNRATE). | — | — |
| RSI-07 | BASSE | mqh:115 | `entry_long` détecté quand RSI cross **DESCENDS** below OS. Convention vbt MR : achat quand survendu = correct. | — | — |

### 1.6 `src/mt5/Include/FxIndicatorVWAP.mqh` + `FxIndicatorBBDeviation.mqh`

| ID | Sév | Fichier:ligne | Problème | Fix proposé | Réf |
|---|---|---|---|---|---|
| **IND-01** | HAUTE | VWAP:48 | VWAP utilise `bar.tick_volume` — proxy volume forex (pas real volume disponible en spot OTC). Études : tick_volume corrélation ~90% avec real volume. Acceptable mais pas parfait. **vbt `mr_macro.py` n'appelle même PAS VWAP explicitement** dans le code grep (lignes 257-260 plotting only). À cross-check : la stratégie vbt utilise-t-elle vraiment VWAP, ou juste un proxy mid-price ? Si différence, divergence MT5 ↔ vbt. | Audit vbt: chercher `vbt.VWAP` ou rolling weighted price formule. Aligner si différent. | ✓ |
| IND-02 | MOYENNE | VWAP:47 | `tp = (high + low + close) / 3.0` — typical price standard. OK. | — | — |
| IND-03 | BASSE | BB:49-58 | BB ddof=1 `var = (s2 - s²/n)/(n-1)` Welford. OK math match pandas/vbt. | — | — |
| IND-04 | BASSE | BB:23-32 | Buffer circulaire 80 valeurs. Init zeros. Si Compute called avant warmup → returns false. OK. | — | — |
| IND-05 | BASSE | VWAP:62-79 | `Warmup` utilise CopyRates depuis midnight UTC → now. Reset propre. OK. | — | — |

### 1.7 `src/mt5/Include/FxMacroFilter.mqh` + sources

| ID | Sév | Fichier:ligne | Problème | Fix proposé | Réf |
|---|---|---|---|---|---|
| **MAC-01** | HAUTE | FxMacroSourceNative.mqh:166-194 | `CMacroSourceFRED::FetchLatest()` use `?limit=1&sort_order=desc` SANS `realtime_start` param. Le mode NATIVE live retourne donc le **dernier value (révision incluse)** indexé par `period_date`, PAS `release_date`. **Look-ahead potentiel en live** sur T10Y2Y si FRED publie révision rétroactive (rare mais possible). UNRATE non concerné (lu via Calendar MT5 séparément en NATIVE). | Ajouter `&realtime_start=YYYY-MM-DD` au URL FRED dans NATIVE pour symétrie avec ALFRED endpoint Phase M.5. | ✓ |
| **MAC-02** | HAUTE | FxMacroSourceNative.mqh:114 | `out_values[take - 1 - written] = (double)values[i].actual_value / 1e6`. Hardcode `1e6` divisor. Selon docs MQL5 `MqlCalendarValue.actual_value`, cette valeur est en `event.multiplier` units qui peut être 1, 1000, 1e6 selon event type. UNRATE event = 1e6 (correct), mais hardcode fragile. | Lire `event.multiplier` via `CalendarEventById(event_id, event)` et utiliser `actual_value / event.multiplier`. | — |
| MAC-03 | MOYENNE | FxMacroSourceNative.mqh:178 | `WebRequest` timeout 5000ms. Si FRED API ralentit (2-3s parfois), fail silencieux. Le mode HYBRID fallback FILE peut couvrir mais AUTO-live (NATIVE) n'a pas de fallback. | Ajouter retry 1× sur timeout, ou fallback FILE en NATIVE. | — |
| MAC-04 | MOYENNE | FxMacroSourceNative.mqh:201-230 | `ParseLatestObservation` use `StringFind("\"value\":\"...")` minimal JSON parser. Fragile si FRED schema change ou whitespace différent. | Utiliser proper JSON parser (`MqlJson` ou bibliothèque externe). | — |
| MAC-05 | BASSE | FxMacroFilter.mqh:222-251 | `RefreshFromHistory` use `TimeCurrent()` pour binary search. En tester = temps simulé OK. En live AUTO routes vers NATIVE pas HISTORY. OK. | — | — |
| MAC-06 | BASSE | FxMacroSourceHistory.mqh:158-171 | Binary search `lo < hi - 1` invariant correct. OK math. | — | — |

### 1.8 `src/mt5/Include/FxTradeHelpers.mqh` + `FxCommon.mqh`

| ID | Sév | Fichier:ligne | Problème | Fix proposé | Réf |
|---|---|---|---|---|---|
| HLP-01 | MOYENNE | FxTradeHelpers.mqh:30-68 | `EnforceStopLevel` utilise `SYMBOL_TRADE_STOPS_LEVEL` (souvent 0 sur ECN) pour borner SL/TP. Si broker passe à un strict niveau (ex 5 pips) après update, anciens SL/TP peuvent être rejected. Pas de retry logic. | Ajouter retry avec SL/TP réajusté en cas de retcode `INVALID_STOPS`. | — |
| HLP-02 | MOYENNE | FxTradeHelpers.mqh:76-85 | `LotsForRisk` divise par `points × tick_value`. tick_value en account currency (USD ici). OK. Si compte EUR avec USDJPY trade nécessiterait `SYMBOL_TRADE_TICK_VALUE_PROFIT/LOSS` selon side. À vérifier. | Audit pour comptes non-USD. | — |
| HLP-03 | BASSE | FxTradeHelpers.mqh:14-24 | `NormalizeLots` use `MathFloor` → arrondi vers le bas. Si raw_lots < min, force min. OK comportement. | — | — |
| CMN-01 | BASSE | FxCommon.mqh:42-45 | `FX_VOL_TARGET_GLOBAL = 0.28` defaults pre-Phase I. Inputs override. OK. | — | — |
| CMN-02 | BASSE | FxCommon.mqh:108-113 | `IsInUTCSession(t, start_h, end_h)` use `st.hour` exclusivement. Boundaries entiers. OK. | — | — |
| CMN-03 | BASSE | FxCommon.mqh:82-95 | `EnsureHistory` retry 25× × 100ms = 2.5s timeout. OK. | — | — |

### 1.9 `src/mt5/bridge/run_backtest_cli.py`

| ID | Sév | Fichier:ligne | Problème | Fix proposé | Réf |
|---|---|---|---|---|---|
| **CLI-01** | CRITIQUE | run_backtest_cli.py:65 | `DEFAULT_MODEL = 1` (1-min OHLC). Voir EA-01 ci-dessus. | Voir EA-01. | ✓ |
| **CLI-02** | CRITIQUE | run_backtest_cli.py:194-216 | `build_tester_ini()` ne configure PAS `Spread=` dans l'INI. MT5 utilise donc le mode default = "current" (spread courant terminal au moment du run) ou "floating" (spread historique tick par tick). **Pour `Model=1` OHLC**, le spread = bar's historical spread (résolution minute = pas de spike news intra-minute). | Ajouter `Spread=` config explicit. Pour réalisme : `Spread=0` (use historical bar spread) ou hardcode élevé pour worst-case (`Spread=20` = 2 pips fixed). | ✓ |
| CLI-03 | HAUTE | run_backtest_cli.py:68 | `DEFAULT_LEVERAGE = "1:100"` hardcoded. EA cap leverage à 64×, donc OK headroom. Mais si broker live = 1:30 ESMA cap, leverage 64× requested = rejected. Compte tester ≠ live. | Documenter clearly broker target (US/Pro 1:100+). | — |
| CLI-04 | MOYENNE | run_backtest_cli.py:75-80 | `DEFAULT_TESTER_INPUTS` minimaliste (4 inputs). Le reste utilise les defaults compilés `.ex5`. Mais defaults Phase M.7 sont `Inp_EnableDDCap=true, Inp_DDCap=0.20, ...`. Si user veut override pour test sans protections, doit utiliser `--input`. OK CLI design. | — | — |
| CLI-05 | MOYENNE | run_backtest_cli.py:347-365 | `parse_html_report` utilise regex maison sur `<td>Label:</td><td>Value</td>`. Fragile si MT5 schema change. | Utiliser BeautifulSoup ou parser HTML structuré. | — |

---

## 2. Findings cross-cutting (multi-fichiers)

| ID | Sév | Description | Fix proposé |
|---|---|---|---|
| **X-01** | CRITIQUE | **Commission 2 bps/side ne correspond à AUCUN broker réel**. OANDA Standard = 0 commission (spread-only). OANDA Core/Raw = `$5/$100k/side` ≈ 5 bps EUR/USD à 1.10 ($5/110k = 4.5 bps). IC Markets Raw = `$3.5/$100k/side` ≈ 3.2 bps. Donc 2 bps est trop bas pour Core/Raw, trop haut pour Standard. | Choisir broker target explicite et patcher : `Inp_CommissionBpsPerSide=0` (Standard) OU `=5` (Core/Raw). Documenter dans CLAUDE.md. |
| **X-02** | CRITIQUE | **Spread spike news (NFP/CPI/FOMC) non modélisé**. EUR/USD typical 1 pip → 8-10 pips pendant NFP. Sur 5.4 ans = ~65 NFP releases. Si stratégie trade pendant ces fenêtres (MR Macro session 8-16 UTC inclut le 8h30 EST = 12h30 UTC NFP release), drag estimé : 65 events × 5 pips × ~3 trades MR Macro/event = ~1000 pips drag = ~$1000 sur 10k = ~10% net profit hit. | Ajouter filter `is_news_window` via Calendar MT5 high-impact USD events, skip MR Macro entries dans `±15 min` autour de release. |
| **X-03** | CRITIQUE | **Swap rollover overnight non modélisé explicitement**. Phase M.4 dampening (`slip_drag`) couvre entry+exit slippage uniquement. Pour TS Momentum / RSI Daily qui détiennent positions multi-jours, swap cumulatif = ~5-10 pips/nuit selon paire. Sur 5.4 ans × ~100 overnight holds par sleeve = ~5000 pips ignorés. | Vérifier que le tester MT5 applique automatiquement les swaps depuis Symbol properties (réponse Internet : oui mais avec swap rates récents, pas historiques). Sinon ajouter `Inp_SwapBpsPerNight` explicite. |
| X-04 | HAUTE | **`SetDeviationInPoints` inutile sur OANDA / NDD brokers** (Market Execution policy). Selon Internet research, SetDeviationInPoints ne marche QUE sur Instant Execution (Dealing Desk). Sur ECN/STP/NDD, c'est ignoré → fill au market price quel qu'il soit. Le code l'utilise pour cosmétique mais aucune protection effective. | Documenter dans CLAUDE.md : SetDeviation = no-op sur OANDA / IC Markets / Pepperstone. Garder le code (pas dommageable) mais ne pas se reposer dessus pour limit slippage. |
| X-05 | HAUTE | **vbt sync UNRATE post-Phase M.5** : parquets régénérés ALFRED-indexed via `realtime_start` (Phase M.5 OK). Mais `mr_macro.py:133` utilise `unemp.diff(3)` 3-month change. **Si parquet indexed by release_date alors `diff(3)` calcule sur 3 publications espacées ~30 jours mais avec gaps potentiels** (publications skipped, holidays). Comportement consistant entre vbt et MT5 ? | Audit cross : compare unemp_rising trace MT5 (via macro_history.csv) vs vbt (via realign_opening). Doit être identique. |
| X-06 | HAUTE | **Phase I anti-overfit (PSR 100%, DSR 94.5%) calculé sur baseline pré-M corrections** (Sharpe 1.385). Tous les tests stat sont sur returns artificiellement gonflés. Doit être **REFAIT** sur baseline post-M.7 (Sharpe 0.98). | Re-run `scripts/anti_overfit/psr_dsr_bootstrap.py --deals reports/mt5/deals_phase_m7.csv`. Critère pass : PSR ≥ 95% au seuil SR > 0.5, Bootstrap P5(Sharpe) > 0. |
| X-07 | MOYENNE | **Documentation interne inverse Model=1/Model=2 nomenclature**. Voir EA-02. À jour rapidement. | Update `docs/investigations/phase_m4_m7_realism_corrections.md` § 4 avec nomenclature correcte. |
| X-08 | MOYENNE | **Inputs slippage 15 / 10 bps figés**, pas sweepés. Sensibilité paramètres slippage non testée. Si broker live a slip réel +20% supérieur (ex 18 bps MR), Sharpe dropdown additionnel. | Sensibilité walk-forward sur `Inp_MR_SlippageBps ∈ [10, 15, 20]`. |

---

## 3. Recommandations priorisées

### 3.1 Court terme (1-3 jours, impact CRITIQUE)

1. **Re-run baseline avec `Model=4` (real ticks)**. Mesure Sharpe réaliste vs Model=1 OHLC. Patch `run_backtest_cli.py:65` ou via `--model 4` arg.
2. **Aligner commission au broker cible**. Choisir OANDA Standard (commission=0, spread déjà absorbé in-bar) ou Core/Raw (commission=5 bps). Patch `Inp_CommissionBpsPerSide=0` ou `=5`.
3. **Audit `BuildDailyEquityReturns()`**. Implémenter equity-curve par jour + inclure floating P&L positions ouvertes.
4. **Refaire anti-overfit** (PSR/DSR/Bootstrap) sur baseline post-M.7. Critère pass PSR ≥ 95% sur SR > 0.5.

### 3.2 Moyen terme (3-7 jours, impact HAUT)

5. **Spread spike news filter**. Ajouter `is_news_window` via Calendar MT5 high-impact, skip MR Macro `±15 min` autour de release.
6. **Patch double-leverage TS Momentum** (`TS-03`). Supprimer `× lev_pair` ou `× GlobalLeverage()` dans `risk_money` calc.
7. **Documentation Model nomenclature** (EA-02). Update `phase_m4_m7_realism_corrections.md`.
8. **Audit cross-check vbt ↔ MT5** post Phase M.5 (unemp_rising trace identique).

### 3.3 Long terme (1-2 semaines, impact MOYEN)

9. **Sensibilité paramètres** (vol_target, max_lev, BB period, RSI thresholds, slippage bps). Walk-forward N=10 sweeps ±25%.
10. **Time-stop RSI Daily** pour limiter swap drag positions long-held.
11. **JSON parser proper** pour FRED native mode + Spread INI config explicit.

---

## 4. Hors-scope (non-modifié dans cet audit)

- vbt / QuantConnect re-validation
- Paper trade démo OANDA
- Stress test 2018-2020 (data MT5 broker limited 2020-11+)
- Refactor signal logic MR Macro / TS / RSI (logiques OK structurellement)
- Application des patches (audit statique, lecture + raisonnement uniquement)

---

## 5. Annexe — Synthèse impact estimé sur Sharpe

> Impact cumulé estimé en ré-évaluant la baseline 0.98 selon les patches CRITIQUES :

| Patch | Δ Sharpe estimé | Δ CAGR | Justification |
|---|---:|---:|---|
| EA-01 (Model=4 real ticks) | -0.05 à -0.20 | -2 à -5% | OHLC interpolation perfect entries → real ticks pas le luxe |
| X-01 (commission OANDA Standard 0 vs M.7 2 bps) | +0.02 à +0.05 | +0.5 à +1% | Commission baisse de 2→0 bps |
| X-01 (commission Core 5 vs M.7 2 bps) | -0.05 à -0.10 | -1 à -2% | Commission monte de 2→5 bps |
| X-02 (news spike spread) | -0.10 à -0.20 | -2 à -5% | MR Macro intraday touché |
| X-03 (swap rollover modélé) | -0.05 à -0.10 | -1 à -2% | TS/RSI overnight holdings |
| RM-01/RM-02 (vol-target accurate) | ±0.05 | ±1% | Direction inconnue (peut booster ou réduire selon période) |
| **Cumul estimé worst-case** | **-0.30 à -0.55** | **-7 à -15%** | Sharpe ~0.45-0.70 / CAGR ~3-5% |

**Conclusion** : la stratégie reste plausible (Sharpe > 0.5 borderline mandate Apogee implicit) mais avec un Sharpe réaliste live attendu **dans la zone 0.5-0.8**, pas 0.98 actuel.

---

## 6. Verification end-to-end (post-patches)

- Compilation MT5 : `wine MetaEditor64.exe /compile:...` returns 0 errors.
- Re-run avec `Model=4` real ticks → reports/mt5/run_<ts>.json
- Re-run avec `Inp_CommissionBpsPerSide=0` (Standard) ou `=5` (Core)
- `scripts/anti_overfit/psr_dsr_bootstrap.py` post-patches avec PSR ≥ 95% au seuil SR > 0.5
- Audit cross : trace unemp_rising MT5 vs vbt sur 50 dates samples

---

## 7. Annexe — Sources Internet consultées

- [MT5 Strategy Tester — Official Docs (tick generation modes)](https://www.metatrader5.com/en/terminal/help/algotrading/testing) — Confirme nomenclature Model=0..4
- [MT5 Strategy Tester — Real Ticks vs Generated Ticks](https://www.mql5.com/en/blogs/post/762517) — "Not all 99% backtests are equal" — Model=1 OHLC overestimates vs real ticks
- [MQL5 Forum — Every Tick vs Real Ticks variance](https://www.mql5.com/en/forum/451232) — confirme accuracy hierarchy
- [MQL5 Forum — Slippage / Deviation in Points](https://www.mql5.com/en/forum/74571) — `SetDeviationInPoints` no-op sur Market Execution NDD
- [MQL5 Forum — Slippage Setting Best Practice](https://www.mql5.com/en/forum/40162) — Realistic slippage tolerance
- [OANDA Pricing — Core vs Standard Account](https://www.oanda.com/us-en/trading/our-pricing/) — Confirme structure 0 vs $5/$100k commission
- [OANDA Core Pricing PDF](https://www.oanda.com/assets/documents/566/OANDA-CC-Pricing.pdf) — Détails commission per side
- [FX news spike spread — FXEmpire](https://www.fxempire.com/education/article/news-driven-fx-trading-how-to-trade-events-like-the-fomc-cpi-and-nfp-1549791) — EUR/USD spread × 8-10 sur NFP
- [Forex Factory Calendar 2025](https://eplanetbrokers.com/en-US/training/forex-factory-calendar) — High-impact event timing
- [MQL5 Docs — iMA / iRSI / CopyBuffer shift convention](https://www.mql5.com/en/docs/series/copybuffer) — shift=1 pour bar fermée
- [MT5 Backtest Spread Modeling — PuPrime](https://www.puprime.com/how-to-add-spread-to-mt5-strategy-tester-a-guide-for-accurate-backtesting/) — Spread INI options
- [MQL5 Forum — Swap in Strategy Tester](https://www.mql5.com/en/forum/364242) — Swap auto-applied via Symbol properties (mais current values, pas historical)
- [MQL5 Forum — DEAL_PROFIT / DEAL_COMMISSION / DEAL_SWAP](https://www.mql5.com/en/forum/277701) — Confirme metrics séparés (pas double-count)
- [MT5 Backtest Multi-Currency Settings — MQL5 Blog 2026-04](https://www.mql5.com/en/blogs/post/769442) — Match leverage live broker
- [Forex VWAP tick_volume vs real_volume — OANDA Trade Tap](https://www.oanda.com/us-en/trade-tap-blog/trading-knowledge/volume-indicators-enhance-technical-analysis-trading-strategies/) — 90% corrélation tick vs real volume forex
- [Forex Swap Rates — TastyFX Overnight Funding](https://www.tastyfx.com/markets/overnight-funding-rates/) — Rollover cost benchmarks
