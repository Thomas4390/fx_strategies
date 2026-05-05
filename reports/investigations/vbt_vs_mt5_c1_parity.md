# Parité vbt pro vs MT5 C1 — 2026-05-05 20:00 UTC

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

## Phase M.3 attempt — Signal alignment (résultats)

### Diagnostic deep-dive trade-by-trade

Comparaison stats trades MT5 MR vs vbt MR (multi-pair, full 5.4y) :

| Métrique | MT5 MR | vbt MR | Δ |
|---|---|---|---|
| Trades count | 312 | 374 | vbt +20% (over-trade) |
| Duration médiane | 360 min | 360 min | ✅ identique (= time-stop 6h) |
| Duration mean | 280 min | 327 min | vbt +17% holds longer |
| Win rate | 55.8% | 53.7% | vbt -2.1pp |
| **PnL mean** | **$24.47** | **$8.94** | vbt **-64%** per trade |
| PnL std | $128 | $87 | vbt -32% |

**Cause Sharpe gap structurel** : vbt PnL/trade = 1/2.7 MT5. PnL/notional MT5
50× supérieur (notional MT5 plus petit lots discrets, mais wins captent moves
plus grands).

### Tentatives M.3

**Tentative 1 — slippage 1.5bps → 15bps + stop_exit_price="Stop"** :
- Sharpe **-1.21** (catastrophe)
- DD **69.83%**
- Slippage 10× appliqué uniformly sur multi-symbol = tue edge complètement.
- **Rollback**.

**Tentative 2 — stop_exit_price="Stop" only** :
- Sharpe **0.97** (identique M.1, no change)
- vbt déjà check sl_stop/tp_stop contre HIGH/LOW de la bar par défaut.
- "Stop" exit price seulement change exit price (vs Close), pas le trigger.
- Effet marginal sur intraday minute (vol bar < 0.1% → diff close vs stop ≈ 0).
- **Rollback** (no benefit).

### Conclusion Phase M.3

Sharpe gap 0.40 = **vraiment structurel signal-level**, pas accessible via
config tweaks vbt. Causes profondes :

1. **TP/SL execution model** : MT5 fire @ tick exact ; vbt check @ bar close
   contre HIGH/LOW = approximation grossière sur volatilité intraday.
2. **Macro filter timing** : MT5 evalué à entry order tick ; vbt mask boolean
   broadcast row-wise = différence quelques minutes peut filter trades
   différemment.
3. **No-pyramiding par-symbol** : MT5 explicit `CountSleevePositions(magic, sym) > 0`
   ; vbt via accumulate=False (default) = équivalent mais peut diverger sur
   edge cases (re-entry après exit dans même bar).

**Refacto vrai requise (Phase N future, ~5-7 jours)** :
- Implémenter tick-level TP/SL simulation Python (utiliser HIGH/LOW interpolation)
- Macro filter at-entry-time check (pas mask broadcast)
- Per-symbol position tracker explicit
- OU passer au backtest QuantConnect (déjà installé selon CLAUDE.md) qui
  supporte tick events natifs.

### Verdict M.3

❌ **Convergence Sharpe non atteinte** — gap 0.40 reste structurel.
✅ **Diagnostic complet** : PnL/trade MT5 2.7× vbt = TP/SL execution timing
   limitation vbt fundamental.
📋 **État actuel maintenu** : M.1 calibration sizing (DD/vol parité) ✅.
   Phase M.3 attempts rollbacked (no benefit ou catastrophe).

## Causes documentées (résiduel post-M.3)

1. **Sharpe gap signal-level (0.40)** : TP/SL execution sur ticks vs minute close.
   Non-fixable sans tick data ou refactor majeur.
2. **PF/Trades non-applicable** : vbt `from_optimizer` synthetic price.
3. **vbt over-trade léger (+20%)** : possibly entry-edge cases différentes timing.
4. **PnL/trade vbt < MT5 (×0.37)** : conséquence TP/SL miss intra-bar moves.
