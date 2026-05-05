# Phase D — H1 Momentum sleeve (findings — NEGATIF, skip retenu)

> **Date** : 2026-05-04 · **Plan source** :
> `docs/investigations/cagr_improvement_plan.md` (Phase D)

## TL;DR

Sleeve H1 Momentum **construit, intégré, testé en standalone et abandonné**.
Sharpe standalone = -3.98 (params plan), -3.12 (params alternatifs).
Critère §3.2 plan source : `H1 standalone Sharpe < 0.5 → skip D`. Respecté.

Le code reste committé en off-by-default (`Inp_AllocH1Momentum=0.0`) pour
une éventuelle refonte ultérieure.

## Construction

| Composant | Statut |
|---|---|
| `FxSleeveH1Momentum.mqh` (clone TS sur H1, ATR-based SL) | ✅ |
| `FxCommon.mqh` enum `SLEEVE_H1_MOMENTUM` + `MAGIC_H1_MOMENTUM=834` | ✅ |
| `FxRiskManager` 4-alloc (sum=1.0) + SubEquity case + DD/Margin close | ✅ |
| `FxMultiSleeve.mq5` inputs (`Inp_H1_*`) + Init/OnTick guarded by alloc>0 | ✅ |
| Compile clean (0/0) | ✅ |
| Baseline V0 inchangé (Sharpe 1.15, +4615 USD, DD 7.21%, 835 trades) | ✅ |

## Tests standalone (alloc 0/0/0/1.0, 5.4 ans)

| Config | EMA fast/slow | RSI lo/hi | ATR mult | Sharpe | Net | PF | DD | Trades |
|---|---|---|---|---|---|---|---|---|
| **v1 (plan)** | 20 / 50 | 40 / 60 | 2.0 | **-3.98** | -1 493 | 0.41 | 15.05 % | 301 |
| **v2 (longer)** | 50 / 200 | 30 / 70 | 3.0 | **-3.12** | -1 079 | 0.64 | 15.42 % | 173 |

Tests sur EUR/USD, GBP/USD, USD/JPY (mêmes paires que TS Momentum).

## Diagnostic

Inspection logs : trade durations très courtes (1-7 h) → whipsaw constant.
- v1 EMA 20/50 H1 = signaux trop fréquents, EMA bouge sur bruit intraday
- Spread + slippage = ~24 bps round-trip, mange l'edge
- Pas de filtre macro / session restriction → trade 24/24

v2 EMA 50/200 réduit la fréquence (173 vs 301 trades) mais reste négatif :
le timeframe H1 sur ces paires majeures n'a pas d'edge momentum simple
exploitable.

## Pourquoi H1 ne marche pas (hypothèses)

1. **Bruit microstructure** : H1 sur EUR/GBP/JPY = signal-to-noise trop
   faible pour EMA crossover sans filtre lourd.
2. **Coûts proportionnellement plus grands** que sur D1 (où TS fonctionne)
   ou M1 (où MR Macro fonctionne avec filtre macro).
3. **Pas de filtre régime** appliqué (contrairement à MR Macro qui a le
   filtre macro et la fenêtre 6-14h).
4. **Vol-target par paire = 0.10** identique à TS daily, mais la vol H1
   réalisée est plus élevée → leverage trop agressif.

## Pistes pour future refonte (non implémentées)

Si on revient sur H1 plus tard :
- **Restriction session** London + NY (8-21h UTC), skip Asia roll
- **Filtre régime** ADX(14) > 25 sur H1 (trade que les marchés trending)
- **Confirmation D1** : long H1 seulement si EMA20 D1 > EMA50 D1 (trend
  daily aligné avec signal H1)
- **Vol-target ajusté** 0.05 (vs 0.10 sur D1)
- **Coûts réduits** : choisir paires à spread plus bas (EUR/USD prioritaire)
- **Soft signal** : confirmer avec MACD ou ATR breakout en plus du EMA cross

Ces pistes nécessiteraient une investigation séparée. Pas dans le scope
actuel.

## Décision

- **Skip Phase D** confirmé.
- **Phase E** (refonte sleeves existants) reste GO — focus sur amélioration
  des 3 sleeves qui ont déjà un edge plutôt que d'ajouter du bruit.
- **Phase F** (carry) reste conditionnelle (E < 10 % CAGR).

Code H1 reste en place avec `Inp_AllocH1Momentum=0.0` par défaut →
no-op total, aucune régression.

## Artifacts

- `src/mt5/Include/FxSleeveH1Momentum.mqh` (300 lignes)
- `src/mt5/Include/FxCommon.mqh` (+ `SLEEVE_H1_MOMENTUM`)
- `src/mt5/Include/FxRiskManager.mqh` (4-alloc support)
- `src/mt5/Experts/FxMultiSleeve.mq5` (inputs + wiring guarded)
- `reports/mt5/run_20260505T004036Z.json` (v1 standalone)
- `reports/mt5/run_20260505T004141Z.json` (v2 standalone)

## Reproduction

```bash
# v1 plan
python src/mt5/bridge/run_backtest_cli.py \
    --report-name d_h1_standalone \
    --input Inp_AllocMRMacro=0.0 --input Inp_AllocTSMomentum=0.0 \
    --input Inp_AllocRSIDaily=0.0 --input Inp_AllocH1Momentum=1.0

# v2 EMA 50/200 RSI 30/70 ATR 3x
python src/mt5/bridge/run_backtest_cli.py \
    --report-name d_h1_v2 \
    --input Inp_AllocMRMacro=0.0 --input Inp_AllocTSMomentum=0.0 \
    --input Inp_AllocRSIDaily=0.0 --input Inp_AllocH1Momentum=1.0 \
    --input Inp_H1_FastEMA=50 --input Inp_H1_SlowEMA=200 \
    --input Inp_H1_RSILow=30 --input Inp_H1_RSIHigh=70 \
    --input Inp_H1_ATRMultSL=3.0
```
