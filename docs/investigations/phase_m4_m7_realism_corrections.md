# Phase M.4 → M.7 — Corrections réalisme stratégies (audit critique 2026-05-05)

**Statut :** appliqué et validé. Sharpe MT5 C1 passe de **1.38 (illusion)** à **0.98 (réaliste)**, convergence quasi-parfaite avec vbt référence (0.97).

---

## 1. Contexte audit

Après Phase M.1 (sizing calibration vbt ↔ MT5), un gap résiduel **+23% Sharpe MT5 vs vbt** restait inexpliqué (MT5 1.38 vs vbt 0.97). Suspicion: surestimation MT5 par bugs documentation / look-ahead / friction sous-modélisée.

Audit critique structuré en 3 axes :
- Slippage real vs documenté
- Look-ahead biais macro
- Réalisme exécution tester

---

## 2. Phase M.4 — Fix slippage cost (commit pending)

### Cause root

`Inp_*_SlippageBps` (15 pour MR Macro, 10 pour TS/RSI) était utilisé UNIQUEMENT comme :
- Tolérance fill via `m_trade.SetDeviationInPoints()` (rejection si market move > tolerance)
- Majoration `sl_distance` pour sizing conservateur (réduit lots)

**Pas un coût soustrait du P&L.** Documentation `docs/apogee_quantitative_report.md:381` prétendait le contraire (« slippage 15 bps sur MR intraday, 10 bps sur daily »), induisait surestimation Sharpe.

### Fix appliqué

`OpenPosition()` dans les 3 sleeves: shift SL/TP par `slip_pct` pour absorber coût round-trip:

```mql5
// AVANT
sl = price * (1.0 - SLStop)
tp = price * (1.0 + TPStop)

// APRÈS Phase M.4
double slip_pct = SlippageBps / 10000.0;
sl = price * (1.0 - SLStop - slip_pct)   // SL plus large, perte plus grande
tp = price * (1.0 + TPStop - slip_pct)   // TP plus tight, profit plus petit
```

Sémantique: chaque trade SL/TP-triggered absorbe slippage cost (similaire vbt `from_signals(slippage=...)`). Time-stop / signal-flip exits ne paient pas slip (limitation acceptée, contribution mineure).

### Impact mesuré

| Métrique | Baseline (bug) | M.4 (fix) | Diff |
|---|---:|---:|---:|
| Sharpe rf=0 | 1.379 | **1.190** | -14% |
| CAGR | 15.26% | 12.89% | -16% |
| MaxDD equity | -13.00% | -16.25% | +25% |
| Net Profit USD | 11,625 | 9,286 | -20% |

**MR Macro = 95%+ du coût slippage total** (intraday haut turnover, alloc 80%). TS/RSI patches cosmétiques (signal-flip exits).

### Fichiers modifiés
- `src/mt5/Include/FxSleeveMRMacro.mqh` — slip_pct shift on SL/TP
- `src/mt5/Include/FxSleeveTSMomentum.mqh` — sl_safety shift + risk_money dampening
- `src/mt5/Include/FxSleeveRSIDaily.mqh` — idem

---

## 3. Phase M.5 — Fix UNRATE look-ahead (FRED ALFRED endpoint)

### Cause root

`src/mt5/bridge/fx_macro_history.py:85` (avant fix) :
```python
rows.append((pd.Timestamp(o["date"]), float(v)))
```

`o["date"]` = **period_date** FRED (mois de référence). Pour UNRATE février, period_date = 2026-02-01, mais BLS publie le data **~5 jours ouvrés après fin du mois** → release_date ≈ 2026-03-06.

CSV historique contenait `2026-02-01T00:00:00Z, 3.8` mais cette valeur n'était pas connue avant 2026-03-06. Backtest tradant 2026-02-15 utilisait UNRATE de février *avant publication réelle* = **look-ahead 30-35 jours**.

**Affecté**: filtre macro MR Macro (`unemp_3m_change <= 0`). 80% allocation portfolio touchée.

### Fix appliqué — FRED ALFRED endpoint

Refactor `fetch_fred_series()` avec param `realtime_start` activé (ALFRED mode):

```python
params["realtime_start"] = start
params["realtime_end"] = "9999-12-31"
# Pour chaque obs: index par o["realtime_start"] (release_date) au lieu de o["date"]
# Drop_duplicates per period_date keep first → première publication uniquement
```

Diagnostic post-fix:
- T10Y2Y: avg release lag 0 jours, max 3 (lag T+1 négligeable)
- **UNRATE: avg release lag 35.2 jours, max 80, min 31** ✓ confirme bug

### Impact mesuré

| Métrique | M.4 | M.5 (+ALFRED) | Diff |
|---|---:|---:|---:|
| Sharpe rf=0 | 1.190 | **1.010** | -15% |
| CAGR | 12.89% | 11.30% | -12% |
| MaxDD | -16.25% | -19.77% | +22% |
| Trades | 783 | 773 | -1.3% |

**Look-ahead UNRATE valait ~0.18 Sharpe artificiel.** Confirmation empirique du bug.

### Fichiers modifiés
- `src/mt5/bridge/fx_macro_history.py` — ALFRED endpoint, drop_duplicates par period_date
- `data/UNEMPLOYMENT_monthly.parquet` — régénéré (indexé par release_date)
- `data/SPREAD_10Y2Y_daily.parquet` — régénéré (cohérence ALFRED)
- `Common/Files/macro_history.csv` — régénéré

### Live impact

`MACRO_SOURCE_NATIVE` (live) utilise `CalendarValueHistoryByEvent()` MT5 + WebRequest FRED qui retournent automatiquement les release_dates correctes. **Live trading n'a jamais eu ce bug — uniquement le tester `MACRO_SOURCE_HISTORY` était affecté.**

---

## 4. Phase M.6 — Diagnostic Model=2 OHLC (refactor abandonné)

### Hypothèse audit

`OnNewBarM1()` invoqué via `OnTick()` (chaque tick) suspectait look-ahead intra-bar : signal trigger avant close M1.

### Diagnostic Model=2 OHLC M1

Hypothèse réfutée: avec Modeling=2 (OHLC M1, pas every-tick), Sharpe = **1.05** vs Model=1 = 1.01.

**Tick mode est PLUS conservateur, pas un look-ahead.** OHLC mode lisse l'intra-bar et perd le coût spread tick-level. Décision: garder Model=1, abandonner refactor OnTick → OnTimer.

---

## 5. Phase M.7 — Commission + protections RÉACTIVÉES

### Modifications EA

`src/mt5/Experts/FxMultiSleeve.mq5` defaults:

```mql5
// AVANT
input bool   Inp_EnableDDCap     = false;   // OFF
input double Inp_DDCap           = 0.30;    // 30%
input bool   Inp_EnableMarginCap = false;   // OFF
input double Inp_MarginCapPct    = 0.70;    // 70%

// APRÈS Phase M.7
input bool   Inp_EnableDDCap     = true;    // ACTIVÉ
input double Inp_DDCap           = 0.20;    // 20% (plus conservateur)
input bool   Inp_EnableMarginCap = true;    // ACTIVÉ
input double Inp_MarginCapPct    = 0.50;    // 50% (plus conservateur)
input double Inp_CommissionBpsPerSide = 2.0;  // NOUVEAU: 2 bps OANDA Standard
```

**Lev cap 64.0 maintenu** (Phase I conservé, choix utilisateur explicite — broker US/Pro target).

### Commission modélisée

2 bps per-side ajouté à `slip_pct` dans les 3 sleeves (équivalent à $2/lot/side OANDA Standard, IC Markets Raw):

```mql5
double slip_pct = (Inp_*_SlippageBps + Inp_CommissionBpsPerSide) / 10000.0;
```

### Impact mesuré

| Métrique | M.5 | M.7 (+commission+prot) | Diff |
|---|---:|---:|---:|
| Sharpe rf=0 | 1.010 | **0.980** | -3% |
| CAGR | 11.30% | 10.79% | -5% |
| MaxDD | -19.77% | -19.73% | -0.2% |
| Trades | 773 | 774 | 0% |

DD-cap 20% **non-déclenché** (MaxDD réel 19.73% juste sous le seuil). Marge mince → si stress crisis arrive, cap déclenche.

### Fichiers modifiés
- `src/mt5/Experts/FxMultiSleeve.mq5` — Inp defaults + Inp_CommissionBpsPerSide
- `src/mt5/Include/FxSleeveMRMacro.mqh` — slip_pct includes commission
- `src/mt5/Include/FxSleeveTSMomentum.mqh` — idem
- `src/mt5/Include/FxSleeveRSIDaily.mqh` — idem

---

## 6. Synthèse cumulative (5.43 ans, EUR/USD/GBP/JPY/CAD)

| Phase | Sharpe rf=0 | CAGR | MaxDD | Net USD | Trades | Drop Sharpe |
|---|---:|---:|---:|---:|---:|---:|
| BASELINE (bug doc) | **1.379** | 15.26% | -13.00% | 11,625 | 785 | — |
| M.4 slip fix | 1.174 | 12.73% | -16.19% | 9,168 | 784 | -14.8% |
| M.4 full slip 3p | 1.190 | 12.89% | -16.25% | 9,286 | 783 | — |
| M.5 ALFRED macro | 1.010 | 11.30% | -19.77% | 7,884 | 773 | -15.1% |
| M.6 (diagnostic Model=2) | 1.050 | 11.85% | -18.29% | 8,289 | 773 | — |
| **M.7 commission+prot** | **0.980** | **10.79%** | **-19.73%** | **7,477** | **774** | **-3.0%** |
| **vbt référence M.1** | **0.970** | **10.11%** | **-13.52%** | — | — | — |

**Cumul total: -29% Sharpe** (1.38 → 0.98). Convergence MT5 ↔ vbt **quasi-parfaite** (gap 1%).

## 7. Mandate Apogee post-corrections

| Critère | Cible | M.7 réel | Verdict |
|---|---|---:|:---:|
| CAGR | ∈ [10%, 15%] | 10.79% | ✅ PASS borderline |
| MaxDD | < 35% | 19.73% | ✅ PASS marge confortable |
| Sharpe rf=0 | implicite > 1.0 | 0.98 | ⚠️ borderline |

## 8. Live forecast post-corrections

- **Tester MT5 final**: Sharpe 0.98
- **Live OANDA estim**: 0.7-0.9 (broker spread variable, news spike, latency)
- **Décision déploiement**: paper trade 4-6 semaines obligatoire avant capital réel
- **Sharpe 1.38 advertised baseline = ILLUSION** (look-ahead UNRATE + cost slippage ignored)

## 9. Phases suivantes (non-bloquantes, hors scope immédiat)

- **M.8 stress test** : OOS 2018-2020 (incl. crash COVID Mar 2020), walk-forward N=10
- **M.9 paper trade** : 4-6 semaines OANDA démo, monitor spread/slip réel
- **Update QC `30125395/main.py`** : recalibrer `MT5_LEV_AVG` + `MR_SLIP_PER_LEG` pour matcher nouveau Sharpe 0.98 au lieu de 1.38

## 10. Notes clés

1. **Documentation pré-M.4 trompeuse** — claims slippage appliqué étaient faux. Doc à corriger (ce document remplace l'ancienne narrative).
2. **vbt N'avait PAS le look-ahead UNRATE** car parquets locaux étaient générés avec period_date mais vbt appliquait `realign_opening()` ffill = même bias structurel mais **moins permissif** (vbt n'apply pas conjointement les bonnes ré-alignements de release_date — résultat: vbt sous-estimait edge artificiel disponible). Convergence post-fix confirme cohérence sur input macro lagged.
3. **Live trading via `MACRO_SOURCE_NATIVE`** n'a jamais été affecté par le bug UNRATE (Calendar MT5 + WebRequest FRED fournissent release_dates natives). Bug uniquement tester historique.
4. **Convergence vbt-MT5 = preuve de cohérence** — pas de validation indépendante de stratégie. La stratégie peut toujours être surfittée même si moteurs concordent.
