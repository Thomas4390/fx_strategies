# Parité vbt pro vs MT5 C1 — 2026-05-05 19:25 UTC

**MT5 C1 reference** : `run_20260505T172809Z.json` (5.43y backtest, vt=0.75, lev=64.0)

**Verdict global** : ⚠️  5 écart(s) hors tolérance

## Métriques side-by-side

| Métrique | MT5 C1 | vbt pro | Tolérance | Δ |
|---|---|---|---|---|
| vt | 0.7500 | 0.7500 | config | — |
| lev | 64.0000 | 64.0000 | config | — |
| sharpe | 1.3786 | -0.2517 | ±0.1 | -1.630 ✗ |
| cagr_pct | 15.2565 | 25.2798 | ±2.0 | +10.023 ✗ |
| dd_pct | 13.0038 | 111.8563 | ±2.0 | +98.853 ✗ |
| profit_factor | 1.4978 | nan | ±0.1 | +nan ✗ |
| trades | 785 | 0 | ±10% | -100.0% ✗ |

## Écarts à investiguer

- **sharpe** : MT5=1.3786 vs vbt=-0.25168433928618505, delta -1.630 ✗
- **cagr_pct** : MT5=15.2565 vs vbt=25.27975534225966, delta +10.023 ✗
- **dd_pct** : MT5=13.0038 vs vbt=111.85632206267559, delta +98.853 ✗
- **profit_factor** : MT5=1.4978 vs vbt=nan, delta +nan ✗
- **trades** : MT5=785 vs vbt=0, delta -100.0% ✗

## Phase L attempt (2026-05-05) — ÉCHEC documenté

### Tentative : pipeline-level leverage=10 + multi-pair MR

Test pragmatique : passer `leverage=10.0` à `backtest_mr_macro(data_4p, leverage=10.0)`
pour matcher MT5 sub_equity sizing convention (1 lot ≈ 10× notional/equity).

**Résultat catastrophique** :
- Sharpe : -0.25 (vs MT5 1.38)
- DD : **111.86%** (blowup)
- CAGR : 25.28%

**Cause** : double leverage stacking. Pipeline lev=10 produit returns deja amplifiés
10×, puis combined_portfolio_v2 applique `vol_target_leverage` sur ces returns
amplifiés (vol portfolio "augmentée" → calculé lev encore plus haut). DD explose.

→ Rollback effectué (back to lev=1 in pipeline).

### Diagnostic root-cause

Test isolé `MR_Macro` standalone montre vol divergente :

| Source | vol_annualisée portfolio combiné |
|---|---|
| vbt (sleeves lev=1) | **1.74%** |
| MT5 C1 implicit (Sharpe 1.38, CAGR 15%) | **~11%** |

Ratio 6.3×. vbt sleeve "lev=1" = fraction continue capital = 1× notional/equity.
MT5 sleeve trade 1 lot = 100K notional sur 10K equity = **10× notional/equity inhérent**.

Donc vbt sous-représente size de ~6-10× vs MT5. Quand combined applique vol-target
pour atteindre vt=0.75 sur vbt vol 1.74%, lev calculé = **35×** (cap 64). MT5 vol
~11%, lev calculé = ~7×.

### Pourquoi naive scale échoue

Naïvement multiplier returns vbt × 10 ne marche pas car :
1. **Drawdowns ne sont pas linéaires** au-delà de 1× — variance compose, lev=10 sur
   DD 27% standalone donne DD 100%+ en compound.
2. **vol_target_leverage adapte dynamiquement** : si returns sont déjà × 10, vol
   est × 10, mais target inchangé → leverage_ts ÷ 10. Net : returns_scaled × leverage_ts
   = returns_original × 10 × 1/10 = returns_original (no-op visible). Mais DD reste
   compound × 10 dans certaines fenêtres.

### Solution propre nécessite refactor profond (Phase M future)

Options :

**Option M.1 — vbt scale lots natif** :
- Refactor pipeline MR Macro pour passer `size_type=SizeType.Amount` + size en
  lots discrets (compatible `vbt.Portfolio.from_signals` natif).
- Réplique exact MT5 lot-based sizing.
- **Effort** : 2-3 jours, refactor signal pipelines.

**Option M.2 — désactiver vol-target Python, utiliser leverage natif** :
- Pipelines pass `leverage=lev_ts` (time-series) calculée depuis MT5 RiskManager logic.
- combined_portfolio_v2 sans vol_target_leverage layer.
- **Effort** : 1-2 jours, mais signal/entry timing peut diverger.

**Option M.3 — calibration empirique** :
- Mesurer ratio `MT5_vol / vbt_vol = K` empiriquement par sleeve.
- Multiplier returns vbt par K constant avant aggregate (= renormalisation).
- Vol-target Python recalcule lev cohérent.
- **Effort** : 0.5 jour mais pas physically meaningful, sensible aux régimes.

### Verdict Phase L

❌ **Échec** — approche pragmatique pipeline leverage cause double-leverage stacking.
✅ **Diagnostic root-cause complet** : vbt vol baseline 6-10× plus basse que MT5.
📋 **Recommandation** : Phase M.1 (lot-based sizing) pour parité véritable.

Pour l'instant, vbt strategies utilisables comme **prototypage relatif** (compare
configs entre elles, walk-forward, anti-overfit) mais **PAS en absolu** vs MT5.

## Causes documentées (résiduel post-K)

1. **vol baseline 6× divergente** : vbt fraction continue vs MT5 lot 100K notional.
2. **Sizing model** : MT5 lots discrets 0.01 vs vbt fraction continue.
3. **Sub-equity calculation** : MT5 sub_equity par sleeve = equity × alloc ;
   vbt utilise weights post-aggregation.
4. **Slippage application** : MT5 ajuste SL distance per-trade ; vbt uniforme bps.
5. **Vol recompute timing** : MT5 21:00 UTC daily ; vbt rolling shift(1).
6. **Profit Factor / Trades non-applicable** : vbt `from_optimizer` synthetic
   price ne retient pas trades individuels.
