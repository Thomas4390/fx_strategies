# Parité vbt pro vs MT5 C1 — 2026-05-05 19:16 UTC

**MT5 C1 reference** : `run_20260505T172809Z.json` (5.43y backtest, vt=0.75, lev=64.0)

**Verdict global** : ⚠️  5 écart(s) hors tolérance

## Métriques side-by-side

| Métrique | MT5 C1 | vbt pro | Tolérance | Δ |
|---|---|---|---|---|
| vt | 0.7500 | 0.7500 | config | — |
| lev | 64.0000 | 64.0000 | config | — |
| sharpe | 1.3786 | 0.9660 | ±0.1 | -0.413 ✗ |
| cagr_pct | 15.2565 | 45.0929 | ±2.0 | +29.836 ✗ |
| dd_pct | 13.0038 | 49.6293 | ±2.0 | +36.626 ✗ |
| profit_factor | 1.4978 | nan | ±0.1 | +nan ✗ |
| trades | 785 | 0 | ±10% | -100.0% ✗ |

## Écarts à investiguer

- **sharpe** : MT5=1.3786 vs vbt=0.9660264168781456, delta -0.413 ✗
- **cagr_pct** : MT5=15.2565 vs vbt=45.09292058743117, delta +29.836 ✗
- **dd_pct** : MT5=13.0038 vs vbt=49.62933950793218, delta +36.626 ✗
- **profit_factor** : MT5=1.4978 vs vbt=nan, delta +nan ✗
- **trades** : MT5=785 vs vbt=0, delta -100.0% ✗

## Évolution post-Phase K (multi-pair MR Macro)

| Métrique | MT5 C1 | vbt PRE-K (single-pair MR) | vbt POST-K (4-pair MR) |
|---|---|---|---|
| Sharpe | 1.38 | 1.09 | **0.97** (pire) |
| CAGR | 15.26% | 49.01% | 45.09% (-4pp) |
| DD | 13.00% | 46.15% | **49.63%** (pire) |

Phase K (multi-pair MR via `load_all_fx_data` + `cash_sharing=True`) a réduit
légèrement CAGR mais Sharpe/DD se sont **dégradés**. Cause : leverage stacking.

### Cause persistante post-Phase K — Leverage Stacking

**MT5 architecture** :
```
MR Macro signals → RiskManager.GlobalLeverage scale LOTS → trades exécutés
```
Lots scaled par `min(target_vol/realized_vol, max_lev)`, applied PER-TRADE.
Stops fixes (TP 0.6%, SL 0.5%) + slippage (15 bps) limitent perte par trade
même avec lev=64.

**vbt architecture** :
```
MR Macro pipeline → returns daily (100% base capital, no leverage)
  → combined_portfolio_v2.vol_target_leverage(returns) → returns × leverage
```
Returns scaled POST-FACT. Quand returns × 64, gains ET pertes amplifiés
linéairement. Stops dans returns sont DÉJÀ inclus à 1:1 levier — multiplier
par 64 amplifie les pertes au-delà de ce que MT5 vivrait avec lots scaled.

### Verdict Phase K

✅ **Multi-pair MR Macro implémenté** (refactor `_compute_strategy_daily_returns`).
❌ **Parité numérique non atteinte** — leverage stacking dégrade Sharpe.

### Phase L recommandée (future)

Pour vraie parité, refactor combined_portfolio_v2 :
1. Remplacer `from_optimizer(synthetic_price)` par `from_signals(close=multi_symbol)`
2. Passer `leverage` au niveau Portfolio (vbt natif applique sur position sizing
   pas sur returns)
3. Ou alternativement : passer `size_type=SizeType.Percent100` + `leverage`
   dans pipeline MR Macro pour que vbt scale lots (équivalent MT5)
4. Désactiver layer vol-targeting global de combined_portfolio_v2 si chaque
   sleeve a son propre vol-targeting (équivalent MT5 RiskManager)

**Estimation** : 1-2 jours refactor + tests. Espérer Sharpe ∈ [1.20, 1.45],
CAGR ∈ [12%, 18%], DD ∈ [10%, 16%] (parité ±10%).

## Causes documentées (résiduel)

1. **Leverage stacking** (cause #1 post-K) : returns × leverage au lieu de
   lots × leverage. Fix Phase L via from_signals.
2. **Sizing model inhérent** : MT5 lots discrets 0.01 vs vbt fraction continue.
3. **Sub-equity calculation** : MT5 sub_equity par sleeve = equity × alloc ;
   vbt utilise weights post-aggregation.
4. **Slippage application** : MT5 ajuste SL distance per-trade ; vbt uniforme
   bps sur returns.
5. **Vol recompute timing** : MT5 21:00 UTC daily ; vbt rolling shift(1).
6. **Profit Factor / Trades non-applicable** : vbt `from_optimizer` synthetic
   price ne retient pas trades individuels.
