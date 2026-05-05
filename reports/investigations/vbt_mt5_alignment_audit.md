# Audit Alignement vbt pro vs MT5 C1 (champion)

> **Date** : 2026-05-05
> **Source MT5 C1** : `src/mt5/Experts/FxMultiSleeve.mq5` (commit 750a0d2 Phase I)
> **Stratégies vbt** : `src/strategies/{combined_portfolio_v2.py, mr_macro.py, daily_momentum.py, rsi_daily.py, combined_portfolio.py}`

## TL;DR

Audit param-by-param identifie 4 écarts critiques entre stratégies vbt pro et configuration MT5 C1 déployée :

1. **Vol-targeting global** : vbt vt=0.28/lev=12 vs MT5 vt=0.75/lev=64 (Phase I)
2. **MR Macro session** : vbt 6-14 UTC vs MT5 8-16 UTC (Phase E.1)
3. **RSI Daily pairs** : vbt 4 paires (incl USDJPY) vs MT5 3 paires (no USDJPY, Phase E.3)
4. **PRODUCTION_WEIGHTS keys** : `RSI_Daily_4p` doit devenir `RSI_Daily_3p`

Les autres params (BB, TP/SL, EMA, RSI thresholds, slippage) match MT5 C1.

## Tableau Détaillé

### Global Vol-Targeting

| Param | MT5 C1 | vbt actuel | Fichier vbt | Action |
|---|---|---|---|---|
| `target_vol` | **0.75** | 0.28 | `combined_portfolio_v2.py:74` | UPDATE |
| `max_leverage` | **64.0** | 12.0 | `combined_portfolio_v2.py:75` | UPDATE |
| `vol_floor` | 0.02 | n/a (vbt utilise vol-target sans floor explicite) | — | DOCUMENT |
| `EnableDDCap` | false | n/a | `combined_portfolio_v2.py:compute_dd_cap_scale` | OK (vbt has equivalent off-by-default) |
| `EnableMarginCap` | false | n/a | n/a | OK |

### MR Macro

| Param | MT5 C1 | vbt actuel | Fichier vbt:line | Action |
|---|---|---|---|---|
| Pairs | EUR/GBP/JPY/CAD (4) | DEFAULT_PAIRS = same ✅ | `mr_macro.py:45` | OK |
| BB window | 80 | 80 ✅ | `mr_macro.py:299` | OK |
| BB alpha | 5.0 | 5.0 ✅ | `mr_macro.py:300` | OK |
| TP stop | 0.006 | 0.006 ✅ | `mr_macro.py:302` | OK |
| SL stop | 0.005 | 0.005 ✅ | `mr_macro.py:301` | OK |
| **session_start UTC** | **8** | **6** ❌ | `mr_macro.py:303` | **UPDATE** |
| **session_end UTC** | **16** | **14** ❌ | `mr_macro.py:304` | **UPDATE** |
| spread_threshold | 0.5 | 0.5 ✅ (pipeline default) | `mr_macro.py:305` | OK |
| spread_threshold (load_macro_filters) | 0.5 (ref) | 0.3 ❌ (helper default) | `mr_macro.py:182` | UPDATE (cosmetic, pipeline override prevails) |
| dt_stop | "21:00" | "21:00" ✅ | `mr_macro.py:306` | OK |
| td_stop (time-stop) | 6h | "6h" ✅ | `mr_macro.py:307` | OK |
| Slippage intraday | 15 bps | PROJECT_CONFIG["slippage_intraday"] | `mr_macro.py:309` | VERIFY value=0.0015 |
| Macro filter | gate sur entries | gate sur entries ✅ | `mr_macro.py:355,375` | OK |
| Risk per trade | 1% sub-equity | n/a (vbt sizing différent — fraction equity via Portfolio.from_signals) | — | DOCUMENT divergence |

### TS Momentum

| Param | MT5 C1 | vbt actuel | Fichier vbt:line | Action |
|---|---|---|---|---|
| Pairs | EUR/GBP/JPY (3 no CAD) | `closes_3p[["EUR-USD","GBP-USD","USD-JPY"]]` ✅ | `combined_portfolio.py:119` | OK (Phase 17) |
| Fast EMA | 20 | 20 ✅ | `daily_momentum.py:116,146` | OK |
| Slow EMA | 50 | 50 ✅ | `daily_momentum.py:117,147` | OK |
| RSI period | 7 | 7 ✅ | `daily_momentum.py:118,148` | OK |
| RSI low | 40 | 40 ✅ | `daily_momentum.py:119,149` | OK |
| RSI high | 60 | 60 ✅ | `daily_momentum.py:120,150` | OK |
| target_vol per pair | 0.10 | 0.10 ✅ | `daily_momentum.py:121,151` | OK |
| max_lev per pair | 3.0 | 3.0 ✅ | `daily_momentum.py:140` | OK |
| Slippage daily | 10 bps | n/a (sleeve-level pas exposé) | — | DOCUMENT |
| Risk per trade | 5% sub-equity | n/a | — | DOCUMENT |

### RSI Daily

| Param | MT5 C1 | vbt actuel | Fichier vbt:line | Action |
|---|---|---|---|---|
| **Pairs** | **EUR/GBP/CAD (3 no JPY)** Phase E.3 | **EUR/GBP/JPY/CAD (4)** ❌ | `combined_portfolio.py:41` | **UPDATE RSI_DAILY_PAIRS** |
| RSI period | 14 | 14 ✅ | `rsi_daily.py:85` | OK |
| Oversold | 25 | 25 ✅ | `rsi_daily.py:86` | OK |
| Overbought | 75 | 75 ✅ | `rsi_daily.py:87` | OK |
| Exit mid | 50 | 50 ✅ | `rsi_daily.py:88` | OK |
| Logique entries | crossed_below(25) | crossed_below(25) ✅ | `rsi_daily.py:114` | OK |
| Logique exits | crossed_above(50) | crossed_above(50) ✅ | `rsi_daily.py:115` | OK |
| Slippage daily | 10 bps | None (pas par défaut) | `rsi_daily.py:91` | OPTIONAL SET |
| Risk per trade | 5% sub-equity | n/a | — | DOCUMENT |

### Production Weights

| Key | MT5 sleeves alloué | vbt actuel | Action |
|---|---|---|---|
| MR_Macro | 0.80 | 0.80 ✅ | OK |
| TS_Momentum_3p | 0.10 | 0.10 ✅ | OK |
| **RSI_Daily_4p** | n/a | 0.10 | **RENAME → RSI_Daily_3p** |

`combined_portfolio.py:41` `RSI_DAILY_PAIRS` doit retirer USD-JPY → automatiquement, key `RSI_Daily_4p` devient mismatched. Renommer la clé + variable interne.

## Divergences Inhérentes (non-bloquantes)

Différences architecturales entre vbt (Python) et MT5 (MQL5) qu'on **ne peut pas réconcilier sans refacto majeur** :

1. **Sizing model** : MT5 = lots discrets calculés via `risk_pct × sub_equity / SL_distance` ; vbt = fraction continue equity via `Portfolio.from_signals(size=...)`. Sur petites fenêtres (<2 mois) MT5 lots arrondit à 0.01 (artifact stress C1/C2 W2/W3).

2. **Sub-equity virtuelle** : MT5 calcule `sub_equity = equity × allocation_pct` par sleeve indépendamment ; vbt aggreggate daily returns par sleeve puis applique weights via `from_optimizer`. Mathematically équivalent en absence de leverage mais peut diverger sous vol-targeting agressif.

3. **Slippage application** : MT5 applique slippage sur ordre execution + ajuste SL distance ; vbt applique slippage uniformément via param Portfolio. Léger biais si SL distances varient.

4. **Macro filter timing** : MT5 dispatche signaux à barre M1 close, vérifie macro à open trade ; vbt utilise mask broadcast row-wise (vectorisé). Equivalent en pratique mais ordre exact peut varier.

5. **Vol recompute** : MT5 recompute `realized_vol = max(σ21, σ63)` à 21:00 UTC daily, applique le lendemain ; vbt utilise `vol_target_leverage` avec `shift(1)`. Causal semantics identique mais window definition peut différer (rolling sample size).

## Tolerances Acceptables (post-patch)

Pour valider parité après patches :

| Métrique | MT5 C1 | Tolerance vbt | Acceptable range |
|---|---|---|---|
| Sharpe full 5.4y | 1.38 | ±0.10 | [1.28, 1.48] |
| CAGR full | 22.79% | ±2pp | [20.79%, 24.79%] |
| MaxDD full | 13.00% | ±2pp | [11.0%, 15.0%] |
| Profit Factor | 1.50 | ±0.10 | [1.40, 1.60] |
| Trades count | 785 | ±10% | [707, 864] |

Si écart > tolerance → investigation cause (lots discretization, slippage, vol recompute timing).

## Patches à Appliquer (B.2)

```python
# 1. src/strategies/combined_portfolio_v2.py:74-75
PRODUCTION_TARGET_VOL: float = 0.75   # Phase I (was 0.28)
PRODUCTION_MAX_LEVERAGE: float = 64.0 # Phase I (was 12.0)

# 2. src/strategies/combined_portfolio_v2.py:69-73
PRODUCTION_WEIGHTS: dict[str, float] = {
    "MR_Macro": 0.80,
    "TS_Momentum_3p": 0.10,
    "RSI_Daily_3p": 0.10,    # was RSI_Daily_4p (Phase E.3 retire USDJPY)
}

# 3. src/strategies/mr_macro.py:303-304
session_start: int = 8,   # Phase E.1 (was 6)
session_end: int = 16,    # Phase E.1 (was 14)

# 4. src/strategies/mr_macro.py:182
spread_threshold: float = 0.5,   # Phase H (was 0.3, cosmetic — pipeline default override)

# 5. src/strategies/combined_portfolio.py:41
RSI_DAILY_PAIRS = ("EUR-USD", "GBP-USD", "USD-CAD")   # Phase E.3 (was incl USD-JPY)

# 6. src/strategies/combined_portfolio.py (rename refs)
# rsi_daily_4p → rsi_daily_3p in _compute_strategy_daily_returns
# "RSI_Daily_4p" → "RSI_Daily_3p" in dict key
```
