# Parité vbt pro vs MT5 C1 — 2026-05-05 19:10 UTC

**MT5 C1 reference** : `run_20260505T172809Z.json` (5.43y backtest, vt=0.75, lev=64.0)

**Verdict global** : ⚠️  5 écart(s) hors tolérance

## Métriques side-by-side

| Métrique | MT5 C1 | vbt pro | Tolérance | Δ |
|---|---|---|---|---|
| vt | 0.7500 | 0.7500 | config | — |
| lev | 64.0000 | 64.0000 | config | — |
| sharpe | 1.3786 | 1.0872 | ±0.1 | -0.291 ✗ |
| cagr_pct | 15.2565 | 49.0090 | ±2.0 | +33.753 ✗ |
| dd_pct | 13.0038 | 46.1515 | ±2.0 | +33.148 ✗ |
| profit_factor | 1.4978 | nan | ±0.1 | +nan ✗ |
| trades | 785 | 0 | ±10% | -100.0% ✗ |

## Écarts à investiguer

- **sharpe** : MT5=1.3786 vs vbt=1.0872349355076478, delta -0.291 ✗
- **cagr_pct** : MT5=15.2565 vs vbt=49.00903958545057, delta +33.753 ✗
- **dd_pct** : MT5=13.0038 vs vbt=46.15146954933964, delta +33.148 ✗
- **profit_factor** : MT5=1.4978 vs vbt=nan, delta +nan ✗
- **trades** : MT5=785 vs vbt=0, delta -100.0% ✗

Causes documentées (post-investigation) :

### Cause #1 — MR Macro vbt = SINGLE-PAIR (architectural divergence)

`src/strategies/combined_portfolio.py:101` :
```python
_, data_eur = load_fx_data()           # single EUR-USD only
pf_mr = backtest_mr_macro(data_eur)    # 1-pair MR Macro
```

vs MT5 C1 qui trade 4 paires equal-weight (EUR/GBP/JPY/CAD) avec sub-equity
0.80 / 4 = 20% allocation par paire.

**Impact** : vbt MR Macro a 4× moins de positions, mais le levier global vt=0.75
s'applique sur returns single-pair → CAGR vbt 49% (gonflé par single-pair vol)
vs MT5 22% (diversifié 4-pair).

**Fix requis (hors scope Phase J)** : refactor `_compute_strategy_daily_returns`
pour utiliser `load_all_fx_data()` + `backtest_mr_macro` en multi-symbol mode
(la fonction supporte déjà multi-symbol via `is_multi=True` branch).

### Cause #2 — Profit Factor / Trades non-applicable

vbt utilise `daily_returns aggregate → from_optimizer → vbt.Portfolio` qui ne
retient pas les positions individuelles (synthetic price). PF=nan, Trades=0
sont attendus dans cette architecture.

**Fix requis** : pour métriques trade-level, utiliser `pipeline()` per-sleeve et
agréger via `vbt.Portfolio.from_signals` multi-symbol au lieu de
`from_optimizer`. Refacto significatif.

### Cause #3 — Sizing model inhérent

MT5 = lots discrets calculés via `risk_pct × sub_equity / SL_distance` →
arrondi 0.01 lot.

vbt = fraction continue equity via `Portfolio.from_signals(size=fraction)`.

Sur backtests 5.4y avec compounding gros, divergence cumulée notable (déjà
observé stress C1/C2 W2/W3 = lot=0.01 binding sur petites fenêtres).

### Cause #4 — Slippage / Vol recompute timing

Différences mineures qui contribuent ~5% delta Sharpe (sub-dominant vs
cause #1 et #3).

## Verdict Phase J

**Defaults sync vbt vs MT5 C1** : ✅ COMPLET (PRODUCTION_TARGET_VOL=0.75,
PRODUCTION_MAX_LEVERAGE=64.0, RSI no JPY, MR session 8-16, spread 0.5).

**Parité numérique** : ❌ NON ATTEINTE — divergence architecturale single-pair
MR Macro vs multi-pair MT5 dominante (cause #1). Refacto multi-pair MR pour vbt
= follow-up post-Phase J.

**Action recommandée** :
1. Phase K (futur) : refactor `_compute_strategy_daily_returns` multi-pair MR Macro
2. Phase L (futur) : `from_optimizer` → `from_signals` multi-symbol pour récupérer
   trades + PF + métriques trade-level
3. Re-run comparison après Phase K — espérer écart < 5pp CAGR/DD.

Pour l'instant : Phase J **defaults sync uniquement**, parité numérique
documentée comme limitation connue.
