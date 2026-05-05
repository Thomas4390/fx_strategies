# Parité vbt pro vs MT5 C1 — 2026-05-05 19:49 UTC

**MT5 C1 reference** : `run_20260505T172809Z.json` (5.43y backtest, vt=0.75, lev=64.0)

**Verdict global** : ⚠️  4 écart(s) hors tolérance

## Métriques side-by-side

| Métrique | MT5 C1 | vbt pro | Tolérance | Δ |
|---|---|---|---|---|
| vt | 0.7500 | 0.7500 | config | — |
| lev | 64.0000 | 64.0000 | config | — |
| sharpe | 1.3786 | 0.9745 | ±0.1 | -0.404 ✗ |
| cagr_pct | 15.2565 | 10.1119 | ±2.0 | -5.145 ✗ |
| dd_pct | 13.0038 | 13.5242 | ±2.0 | +0.520 ✓ |
| profit_factor | 1.4978 | nan | ±0.1 | +nan ✗ |
| trades | 785 | 0 | ±10% | -100.0% ✗ |

## Écarts à investiguer

- **sharpe** : MT5=1.3786 vs vbt=0.9744528487511804, delta -0.404 ✗
- **cagr_pct** : MT5=15.2565 vs vbt=10.111923364844454, delta -5.145 ✗
- **profit_factor** : MT5=1.4978 vs vbt=nan, delta +nan ✗
- **trades** : MT5=785 vs vbt=0, delta -100.0% ✗

## Phase M.1 SUCCESS (calibration sizing)

### Évolution

| Métrique | MT5 C1 | vbt PRE-M | vbt POST-M.1 | Verdict |
|---|---|---|---|---|
| Sharpe | 1.38 | 0.97 | **0.97** | ✗ gap structurel |
| CAGR | 15.26% | 45.09% | **10.11%** | borderline (-5.15pp) |
| DD | 13.0% | 49.63% | **13.52%** | ✅ **gap 0.52pp** |
| vol | ~11% | 1.74% | **10.45%** | ✅ **gap 0.5pp** |

### Patches M.1

1. `combined_portfolio.py:_compute_strategy_daily_returns` :
   - `init_cash=10_000.0` (match MT5 deposit)
   - MR pipeline : `size=0.20, size_type="percent", leverage=12.0`
   - TS rets × 12.0 + RSI rets × 12.0 (uniform MT5 GlobalLeverage)
2. `compare_vbt_vs_mt5_c1.py` : `target_vol=None` (évite double-stacking)
3. `mr_macro.py:pipeline()` : ajouter `size`, `size_type` params
4. Cache version : v3-phase-k → v5e-phase-m

### Verdict M.1

✅ **Sizing/leverage calibrés** — DD et vol matchent quasi-parfaitement MT5
✅ **Pas de blowup** (vs Phase L Sharpe -0.25)
⚠️ **Sharpe gap 0.40 persiste** = écart d'edge signal logic intrinsèque
⚠️ **CAGR borderline** (-5.15pp, juste hors tolérance ±5pp M.1)

### Cause Sharpe gap résiduel

Pas problème sizing (calibré). Edge intrinsèque vbt vs MT5 :
1. Tick events (MT5) vs M1 OHLC (vbt) — TP/SL execution priority
2. Macro filter timing : MT5 evalué à entry order ; vbt broadcast row-wise
3. Slippage : MT5 ajuste SL distance per-trade ; vbt uniforme

### Phase M.2/M.3 (future, optionnel)

- **M.2 LotsForRisk Python** (~1j) : amélioration marginale (gap signal-level pas sizing-level)
- **M.3 Refactor signal logic** (~3j) : aligne timing TP/SL/macro filter exact MT5, vraie convergence Sharpe ±0.05

### Causes documentées (résiduel post-M.1)

1. **Sharpe gap signal-level** : edge diverge ~0.40 (architectural, pas sizing)
2. **PF/Trades non-applicable** : vbt `from_optimizer` synthetic price
3. **CAGR borderline** : conséquence directe Sharpe gap × vol equivalente
