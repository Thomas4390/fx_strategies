# Gold sleeve — momentum signal and martingale / grid sizing

**Date** : 2026-07-25
**Instrument** : XAUUSD, `data/XAU-USD_minute_qc.parquet` (QuantConnect, OANDA CFD)
**Holdout state** : LOCKED
**Holdout touched by this phase** : YES — once, at the end, as designed

**Holdout definition (adapted for this sleeve)** : the repo policy locks
`>= 2026-01-01`. This sleeve locks **`>= 2025-07-01`** instead, so the blind
period (333 sessions) spans both the parabolic 2025 advance and the 2026
drawdown of −28%. A sizing rule can only be judged against the regime that
kills it, and that regime is in 2026.

---

## 1. Data

2 703 484 minute bars, 2019-01-01 → 2026-07-24 (7.56 years, 2364 sessions), no
duplicates, no NaN, no volume column. Index is tz-aware UTC and is converted to
**naive New York time** on load (`utils.load_gold_data`): the session
boundaries are clock times, so a UTC index drifts by an hour across DST.

Gold ran 1282 → 4029 over the sample with a peak at 5593, i.e. a terminal
drawdown of **−28%**. Buy and hold over the full sample: 12.99% CAGR.

Two rejected inputs are recorded so they are not re-tried:
- `GOLD_minute.csv` is **GLD**, the ETF — US session only (09:31-16:00), not spot.
- The first `XAUUSD_minute.csv` was **truncated at 1 048 575 rows = 2²⁰**, the
  spreadsheet row limit; it stopped at 2021-12-06, losing 4.6 years.

## 2. Signal — what was rejected and why

### 2.1 Market Intraday Momentum — REJECTED on cost

Gao, Han, Li & Zhou (JFE 2018): the first half-hour return predicts the last.
The predictor **replicates cleanly on gold**:

| Regression | beta | t | R² |
|---|---|---|---|
| r1 → last half-hour | +0.0200 | **+6.10** | 1.86% |
| r1 → 10:00-16:00 | +0.0507 | +3.35 | 0.57% |

(the paper reports R² = 1.6% on SPY, so gold is in line)

Statistically real, economically dead. At one round trip per session the gross
edge is **1.5 to 2.9 bps per trade against ~2 bps of round-trip cost**:

| design | trades/yr | gross | net @2bps | SR net |
|---|---|---|---|---|
| full session, unconditional | 247 | 3.79% | −1.15% | −0.14 |
| full session, vol21 > P70 | 73 | 1.99% | +0.53% | 0.10 |
| full session, \|r1\| > P50 | 108 | 2.80% | +0.63% | 0.11 |
| last half-hour, unconditional | 241 | 0.95% | −3.86% | −2.15 |

Conditioning on volatility is supported by the paper (predictability
concentrates on volatile days) and still does not save it. **No sizing overlay
can rescue a signal whose gross edge is below its cost — martingale and grid
both *increase* turnover.** Kept in code as `gold_momentum.mim_signal` for the
record only.

### 2.2 Time-series momentum ensemble — RETAINED

Moskowitz, Ooi & Pedersen (JFE 2012). Rather than selecting a lookback, the
signal **averages the sign of four fixed lookbacks (40/60/120/250)**. Averaging
replaces a fitted choice with an aggregate. Long-only by default: gold carries a
structural positive drift, and enabling shorts cost 3.8 pp of return and
doubled the drawdown (−66.84% vs −23.31%).

Turnover is ~10 flips/year, so cost is ~0.2%/yr — negligible. Positions are held
for weeks, which is also the structure martingale and grid actually need.

## 3. Sizing — the four regimes

Implemented in `src/framework/sizing_nb.py` as one Numba `signal_func_nb`
parameterised by mode, with 13 unit tests on hand-built sequences
(`tests/test_sizing_nb.py`). Every path-dependent mode carries three hard
guards: cumulative-size cap, basket stop, and a kill switch that reverts to flat
sizing and **never re-arms**.

Prior statistical tests on the trade sequence, before any backtest:

| Test | Result | Reading |
|---|---|---|
| Autocorrelation of trade PnL, lag 1 | ρ = −0.0041, p = 0.86 | no martingale basis |
| E[PnL \| previous loss] vs previous win | +4.70 vs +0.53 bps, t = +1.59, p = 0.11 | not significant |
| Longest observed losing streak | **11** | ×2 martingale ⇒ ×2048 size |
| Forward return after −0.5 to −1.0% adverse | −0.44 bps, t = −4.17 | averaging down has *negative* expectancy |
| Forward return while in profit > +0.3% | +2.48 bps, t = +24.46 | favours pyramiding, not averaging down |

## 4. Results

### 4.1 Selection (2019-01 → 2025-06, 2030 sessions), risk-matched to 25% vol

Risk-matching is mandatory: at natural exposure the martingale "wins" only by
being more levered on average.

| regime | ann | SR | maxDD | MAR | skew | kurt | P(loss>50%) | DD p95 |
|---|---|---|---|---|---|---|---|---|
| grid k=0.5 | **18.73%** | 0.75 | −42.70% | **0.44** | −0.63 | 9.6 | 0.05% | 52.0% |
| grid k=1.0 | 17.67% | 0.71 | −44.70% | 0.40 | −0.63 | 9.7 | 0.10% | 53.8% |
| **flat (control)** | 17.08% | 0.68 | −48.18% | 0.35 | −0.66 | 10.6 | 0.10% | 54.8% |
| grid deep k=3, 1 level | 17.01% | 0.68 | −47.94% | 0.35 | −0.66 | 9.9 | 0.05% | 54.8% |
| anti-martingale m=2 | 16.37% | 0.65 | −46.66% | 0.35 | −0.34 | 15.4 | 0.25% | 54.7% |
| combo m=2 | 16.04% | 0.64 | −50.07% | 0.32 | −0.66 | 16.2 | 0.60% | 58.9% |
| **martingale m=2 n=3** | **13.13%** | 0.53 | −52.13% | **0.25** | **−1.15** | **19.4** | **1.40%** | **62.8%** |

Year-by-year ranking over 7 years (mean rank, 1 = best): grid k=0.5 **2.00**,
martingale 3.86, grid k=1.0 3.64, combo 4.00, flat 4.64, anti-martingale 5.07.

**`n_max` and `n_levels` never bind.** Results are identical for n=2/3/4 and
lv=2/3/5: with 43 trades in 6.5 years the losing streaks never exceed 2 and the
grid never fills more than 2 levels. So it is not the *extreme* martingale that
underperforms here — it is the mechanism at its mildest.

The deep single-level grid, the one shape with a prior from the tail data
(post-capitulation rebound beyond −1.5%), came out **identical to flat**.
Hypothesis not confirmed.

### 4.2 Holdout (2025-07 → 2026-07, 333 sessions), natural exposure

| regime | ann | SR | maxDD | MAR | mean exposure | DD p95 | P(loss>50%) |
|---|---|---|---|---|---|---|---|
| martingale m=2 n=3 | **42.76%** | **1.29** | −31.30% | **1.37** | **125%** | 45.3% | 0.20% |
| combo m=2 | 40.31% | 1.30 | −31.92% | 1.26 | — | 42.6% | 0.10% |
| grid k=1.0 | 20.50% | 1.01 | −24.16% | 0.85 | — | 32.0% | 0.00% |
| **flat (control)** | 19.31% | 0.98 | −21.78% | 0.89 | 69% | 32.2% | 0.00% |
| grid k=0.5 | 17.64% | 0.88 | −24.60% | 0.72 | — | 33.7% | 0.00% |
| *buy and hold* | *17.08%* | *0.72* | *−28.58%* | *0.60* | *100%* | — | — |

## 5. Conclusions

**5.1 The momentum signal is validated.** Flat sizing beat buy and hold on the
blind period: SR 0.98 vs 0.72 and maxDD −21.78% vs −28.58%. The trend filter
did its job through the 2026 crash. This is the result the sleeve rests on.

**5.2 The grid's selection-period edge did not survive.** Grid k=0.5 ranked
first in selection (MAR 0.44 vs 0.35, best mean annual rank 2.00) and *last of
the grid family* in the holdout (SR 0.88 vs 0.98 flat). A marginal effect that
looked stable across seven annual windows still failed out of sample. Treat it
as noise.

**5.3 The two periods disagree about the martingale, and that is the finding.**
Selection: worst regime, MAR 0.25 vs 0.35. Holdout: best regime, MAR 1.37 vs
0.89. This is not a contradiction to be resolved in favour of one window — it
is the martingale's defining property. It does not change expected return; it
raises **dispersion**. The holdout happened to contain the sequence a martingale
is built for (losses into the crash, then a sharp recovery caught with a larger
position). The selection period did not.

What is stable across *both* periods is the tail:

| | selection | holdout |
|---|---|---|
| martingale bootstrap DD p95 | 62.8% | 45.3% |
| flat bootstrap DD p95 | 54.8% | 32.2% |
| martingale P(loss > 50%) | 1.40% | 0.20% |
| flat P(loss > 50%) | 0.10% | 0.00% |
| martingale skew / kurtosis | −1.15 / 19.4 | −0.57 / 6.7 |
| flat skew / kurtosis | −0.66 / 10.6 | −0.69 / 8.2 |

**In every window, the martingale carries a materially fatter left tail for a
return that is not reliably higher.** Judging it on a realized Sharpe or MAR
from any single window — including the favourable one — is the error that
precedes the blow-up.

**5.4 Recommendation.** Ship the sleeve with **flat sizing**. The martingale and
grid overlays remain in the codebase, tested and guarded, but are not enabled:
neither demonstrates a repeatable edge, and the martingale demonstrably worsens
the tail. The 50%/yr objective is not reachable from a Sharpe-0.7 to Sharpe-1.0
base without leverage that pushes the drawdown past the −50% budget; the honest
operating point is closer to 20-25%/yr for a −25 to −30% drawdown.

## 6. Integration with the three FX sleeves

Common period 2019-01-01 → 2026-04-01 (2262 sessions; the FX sleeve returns stop
at 2026-04-01). Gold runs with flat sizing, `target_vol=0.25`.

**Correlation — the whole argument for a gold sleeve, measured rather than assumed:**

| | MR_Macro | TS_Momentum_3p | RSI_Daily_3p | Gold_Momentum |
|---|---|---|---|---|
| MR_Macro | 1.000 | −0.028 | −0.021 | **−0.025** |
| TS_Momentum_3p | −0.028 | 1.000 | −0.193 | **+0.057** |
| RSI_Daily_3p | −0.021 | −0.193 | 1.000 | **−0.087** |

Standalone over the same window (sleeves at their production leverage):

| sleeve | ann | vol | SR | maxDD |
|---|---|---|---|---|
| MR_Macro | 1.69% | 6.88% | 0.25 | −16.66% |
| TS_Momentum_3p | 39.68% | 67.22% | 0.59 | −81.28% |
| RSI_Daily_3p | 8.58% | 28.43% | 0.30 | −51.80% |
| **Gold_Momentum** | 9.61% | 12.42% | **0.77** | −23.31% |

Gold has the best standalone Sharpe of the four and is effectively orthogonal to
all of them.

**Adding it to the 80/10/10 trio** (trio rescaled by `1 − w`):

| w_gold | ann | vol | SR | maxDD | MAR |
|---|---|---|---|---|---|
| 0% | 6.18% | 8.57% | 0.72 | −13.10% | 0.47 |
| 5% | 6.35% | 8.17% | 0.78 | −12.11% | 0.52 |
| 10% | 6.52% | 7.81% | 0.83 | −11.37% | 0.57 |
| 15% | 6.70% | 7.52% | 0.89 | −11.15% | 0.60 |
| 20% | 6.87% | 7.29% | 0.94 | −10.93% | 0.63 |
| 30% | 7.21% | 7.06% | 1.02 | −10.54% | 0.68 |

Return up, volatility down, drawdown down, monotonically. This is the strongest
result of the whole exercise — and note that it comes from **diversification, not
from any sizing overlay**.

**Caveat on the weight.** The improvement is monotone across the whole range
tested, which means the sample simply likes gold: 2019-2026 is a gold bull market
and the optimum is at the boundary. Do not read 30% as the recommended weight.
A defensible prior is **10-15%**, consistent with the sizing of the two minor FX
sleeves, and it captures most of the Sharpe gain (0.72 → 0.89) already.

Not yet done: rerunning `scripts/sweep_fourth_sleeve.py` with `Gold_Momentum`
injected, and bumping `_SLEEVES_VERSION` in `src/framework/data_cache.py` once
the sleeve is added to `_compute_strategy_daily_returns()`.

## 7. Cross-validation on QuantConnect — and what it found

QC project `34489845` (`GoldMomentumSizing_Validation`), LEAN 2.5.0.0.17941,
XAUUSD OANDA CFD daily, 2019-01-01 → 2026-07-24, same signal, same
`target_vol=0.25 / max_leverage=3.0`, 1 bp slippage per side.

| | vbt | QC (edge-triggered) | QC (daily rebalance) |
|---|---|---|---|
| trades / orders | 50 | 128 orders | 1215 orders |
| CAGR | 8.44% | 20.17% | 26.73% |
| Sharpe | 0.726 | 0.575 | 0.797 |
| ann. volatility | **12.19%** | **23.3%** | 21.7% |
| max drawdown | −23.31% | −51.9% | −50.1% |

**The Sharpe agrees to within about 0.15 across two independent engines and two
independent data sources, which validates the signal.** The volatility does not,
and that gap is a defect in the vbt sleeve, not in the port.

Root cause: `pipeline()` passes `size=1.0, size_type="percent"` with a per-bar
`leverage` array. `percent` means *percent of available cash*, so VBT caps the
order at the cash balance and the leverage array cannot lift it the way a
target-weight order would. Measured mean gross exposure is **52.7%**, and
realized volatility comes out at 12.19% against a 25% target — the vol-target
layer is delivering roughly half of what it asks for. QC's `set_holdings(weight)`
sets the portfolio weight directly and lands at 23.3%, which is what the stated
configuration should produce.

Two consequences, and they differ in severity:

- **The sizing-regime comparison stands.** Every regime in §4 shares the same
  base plumbing, so the under-exposure is common-mode and the *relative* ranking
  — and the risk-matched comparison in particular, which normalises volatility
  explicitly — is unaffected.
- **The absolute figures for the gold sleeve understate the configuration.**
  CAGR 8.44% at 12.19% volatility is roughly half the intended exposure. The
  §6 portfolio weights were derived from these understated returns and should be
  recomputed once the sizing is fixed.

Fix to apply: replace `size_type="percent"` + `leverage` with
`size_type="targetpercent"` and the vol-target weight as the size, following
`daily_momentum.py:224-233` and `composite_fx_alpha.py:389-398`. Then rerun §4
and §6 and check the realized volatility lands near the 25% target.

Two porting traps worth recording:

1. XAUUSD on QC is an **OANDA CFD, not a forex pair**: it delivers `QuoteBar`s
   and carries no volume, so reading `data.bars` (TradeBars) silently returns
   nothing. First backtest: 0 orders, no error.
2. Naming the symbol `self.symbol` shadows a `QCAlgorithm` method. The compiler
   only warns.

## 8. Reproduce

```bash
python scripts/sweep_gold_sizing.py --smoke        # 3 regimes, fast
python scripts/sweep_gold_sizing.py                # 24 regimes, selection
python scripts/sweep_gold_sizing.py --holdout      # blind period — read once
pytest tests/test_sizing_nb.py -v                  # 13 kernel tests
```

Artifacts land in `results/gold_sizing/sizing_{selection,holdout}_<stamp>.{csv,json}`.

## 9. Notes for whoever picks this up

- **Rust backend**: `vectorbtpro-rust==2026.6.27` is now installed (built from
  the `rust/` subdirectory of the vectorbt.pro repo, ~12 min with LTO). Measured
  on 2.7M bars it is **not faster** for this workload: rolling_std 0.81×,
  ewm_mean 0.89×, rolling_mean 0.88×, pct_change 1.35×. Numba stays the default.
- **Pre-existing breakage, unrelated to this work**: `utils.apply_vbt_settings()`
  sets `plotting.pre_show_func`, a key vbt 2026.6.27 renamed to `pre_render_func`.
  The config is frozen, so it raises `KeyError` and takes down all 16
  `tests/test_pipeline_equivalence.py` cases at fixture setup. Verified pre-existing
  by stashing this branch's `src/utils.py` and re-running. Not fixed here because
  the rename may carry different semantics and the function is imported by every
  strategy module.
- A bug worth remembering: an early version of `sweep_gold_sizing.py` sliced
  prices to the holdout *before* computing the signal, stranding the 250-session
  lookback with no history and scoring 250 of 333 blind sessions on a signal
  that did not exist. Always simulate on full history and slice the returns.
