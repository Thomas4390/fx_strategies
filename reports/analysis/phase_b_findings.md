# Phase B — Trade quality inspection (findings)

> **Date** : 2026-05-04 · **Scope** : 5.4 ans baseline (2020-11-23 → 2026-04-30) +
> fold5 (2025-11-01 → 2026-04-30) · **Plan source** :
> `docs/investigations/cagr_improvement_plan.md` (Phase B)

## Métriques per-sleeve (baseline 5.4 ans, 832 trades)

| Sleeve | Trades | Win % | PF | Net (USD) | Top-5 wins / total wins |
|---|---|---|---|---|---|
| MR_MACRO | 350 | 51.1 | 1.47 | +3 067 | ~25 % (sain) |
| TS_MOMENTUM | 437 | 64.5 | 1.31 | +1 521 | ~25 % (sain) |
| RSI_DAILY | 45 | 62.2 | 1.01 | +6.82 | n/a (sleeve flat) |

**Verdict edge** :
- **MR Macro = sain** : edge réel et **diversifié** sur 4 paires
  (EUR 44 %, GBP 28 %, JPY 15 %, CAD 13 %), win rates 46-56 % cohérents
  avec stratégie mean-reversion intraday.
- **TS Momentum = concentré** : edge à **83 % sur USDJPY** (1260 USD sur
  1521). EURUSD breakeven (-15 USD), GBPUSD marginal (+275 USD). Si trend
  JPY meurt (BoJ pivot), TS Momentum disparaît.
- **RSI Daily = flat (no edge)** : Net +6.82 USD sur 5.4 ans, PF 1.01.
  EUR/CAD/GBP positifs (+302 USD combinés) mais USDJPY -295 USD mange
  presque tous les gains. Sleeve paie ses coûts mais pas plus.

## Concentration risk

| Sleeve | Paire dominante | Part du PnL net |
|---|---|---|
| MR_MACRO | EURUSD | 44 % |
| TS_MOMENTUM | **USDJPY** | **83 %** |
| RSI_DAILY | (USDJPY drag) | -∞ (paire négative dominante) |

**Action** : Phase E.2 (TS Momentum EMA grid) doit valider que le profil
USDJPY-heavy n'est pas du data-mining — tester params robustes sur les 3 paires.

## Macro filter impact (B.4)

Test : `Inp_MR_DisableMacroFilter=true` (bypass complet, force `MacroOk()=true`).

| Fenêtre | Filter ON | Filter OFF | Δ |
|---|---|---|---|
| Full 5.4 ans | Sharpe 1.15, +4 615, DD 7.21 % | Net **-995**, DD **15.00 %** | **-5 610 USD, +7.8 pp DD** |
| Fold5 (6 mois) | Sharpe -0.49, -90, 47 trades | Sharpe +0.02, +8, 94 trades | **+0.51 Sharpe** |

**Diagnostic** : sur fold5, `spread=0.51, unemp_rising=1, macro_ok=0`. C'est
le canal **chômage** qui bloque (pas le spread 10Y-2Y). Le sleeve MR fait
**0 trade** sur fold5 quand le filtre est ON.

**Verdict** :
- Filtre macro = **net protecteur sur 5.4 ans** (économise 5 610 USD,
  réduit DD de 7.8 pp). Indispensable.
- Sur fold5 spécifiquement, trop conservateur : `unemp_rising` se déclenche
  même quand le marché ne fait pas de stress (régime "soft landing"
  potentiel).

**Recommandations Phase E** :
1. **Filtre adaptatif** : au lieu de couper à 0, réduire position size
   de 50 % quand `macro_ok=false` (compromis).
2. **Confirmation multi-signaux** : ne bloquer que si spread inversé
   ET chômage en hausse (et pas l'un ou l'autre).
3. **Tester sur autres folds** : confirmer que fold5 est bien atypique
   et pas un signal de fragilité plus large.

## Alerts levées (vs critères §3.2 du plan source)

| Critère STOP | Mesure | Statut |
|---|---|---|
| Top-5 trades > 30 % PnL total | Top-5 wins ≤ 25 % par sleeve | ✅ OK |
| 1 paire > 60 % PnL d'1 sleeve | TS_MOMENTUM USDJPY = 83 % | ⚠️ ALERT |
| Outliers > 5σ | 0 sur tous sleeves | ✅ OK |
| Holding > 3× médian | flagged dans inspect_trades, non bloquant | ✅ OK |

**Décision** : pas de STOP catastrophique. **GO Phase C** (allocation sweep)
avec garde TS Momentum à valider en Phase E.2.

## Artifacts produits

- `reports/analysis/trade_inspection_phase_b.html` — rapport HTML détaillé
- `reports/analysis/macro_filter_impact.csv` — table comparison fold5
- `reports/mt5/deals_phase_b_baseline.csv` — deal log brut (1671 deals)
- `scripts/analysis/inspect_trades.py` — script per-sleeve breakdown
- `scripts/analysis/macro_filter_impact.py` — script comparison runner

## Reproduction

```bash
# 1. Activer deal export et lancer baseline
python src/mt5/bridge/run_backtest_cli.py \
    --input Inp_ExportDeals=true \
    --report-name baseline_with_deals

# 2. Copier le CSV depuis MT5 Common/Files
cp ~/.mt5/drive_c/users/thomas/AppData/Roaming/MetaQuotes/Terminal/Common/Files/deals_*.csv \
   reports/mt5/

# 3. Inspecter les trades
python scripts/analysis/inspect_trades.py \
    --deals reports/mt5/deals_<ts>.csv \
    --output reports/analysis/trade_inspection_<ts>.html

# 4. Tester l'impact du filtre macro sur fold5
python scripts/analysis/macro_filter_impact.py
```
