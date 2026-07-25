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

### 4.1 Selection (2019-01 → 2025-06), risk-matched to 25% vol

Risk-matching is mandatory: at natural exposure the martingale "wins" only by
being more levered on average.

> **Superseded on 2026-07-25 — the session boundary was wrong, and the ranking
> moves.** Sessions were cut at midnight instead of the 17:00 New York CFD
> close, which turned every Sunday evening into a session of its own: 392 of
> them, ~356 minutes each against 1375 for a real one. Session count was
> inflated 20% (2363 vs 1971) and every lookback shortened by the same
> proportion. Details in `docs/specs/gold_momentum_spec.md` §2.
>
> This matters more here than anywhere else in the study, because **every
> overlay in this table is path-dependent**: martingale sizes on the previous
> trade's outcome, the grid on adverse excursions. Feeding them a run of
> six-hour pseudo-sessions corrupts precisely the sequence they key on.

Re-run on correct sessions:

| regime | ann | SR | maxDD | MAR | skew | kurt | P(loss>50%) | DD p95 |
|---|---|---|---|---|---|---|---|---|
| **anti-martingale m=1.5** | **20.25%** | **0.81** | −46.00% | **0.44** | **+0.01** | 10.4 | 0.25% | 49.5% |
| anti-martingale m=2.0 | 19.46% | 0.78 | −48.41% | 0.40 | −0.07 | 10.6 | 0.25% | 51.7% |
| grid k=2.0 | 16.15% | 0.65 | −48.11% | 0.34 | −0.53 | 6.1 | 0.30% | 54.3% |
| grid k=0.5 | 15.66% | 0.63 | −47.84% | 0.33 | −0.53 | 5.5 | 0.45% | 56.1% |
| grid deep k=3, 1 level | 15.37% | 0.61 | −52.98% | 0.29 | −0.55 | 7.1 | 0.55% | 56.5% |
| **flat (control)** | 15.06% | 0.60 | −54.25% | 0.28 | −0.57 | 7.3 | 0.60% | 57.1% |
| **martingale m=2 n=3** | 9.49% | 0.38 | −59.96% | 0.16 | −0.80 | 7.5 | **4.70%** | 69.2% |
| combo m=2 k=1.0 | 7.34% | 0.29 | −63.07% | 0.12 | −0.84 | 6.9 | **8.05%** | 72.7% |

**What changed, and it is not a detail.** Anti-martingale goes from 5th (SR
0.65) to **1st (0.81)**; flat falls from 3rd (0.68) to **7th (0.60)**; grid
k=0.5, the previous winner at 0.75, drops to 0.63.

The direction is mechanically sensible rather than lucky. Anti-martingale adds
after a win and cuts after a loss, which is the same bet the momentum signal is
already making — it only works if "the previous trade" means something, and a
Sunday-evening stub is not a trade. It is also the only regime that fixes the
distribution shape: **skew +0.01 against −0.57 for flat**, which was the stated
theoretical case for it all along.

What did *not* change is the verdict on martingale, which degrades further:
P(loss>50%) rises from 1.40% to **4.70%**, and combo to 8.05%. The one
conclusion that survives untouched is the one that mattered most.

⚠️ **This is in-sample selection over 24 regimes.** It reopens the "flat sizing"
decision, it does not settle it — see §5.

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

> ⚠️ **Figures below are stale — computed on the wrong session boundary — and
> they have deliberately NOT been recomputed.** The holdout has already been
> read once. Re-running it after changing the specification would spend it a
> second time on a decision it was frozen to arbitrate, which is the exact
> failure `HOLDOUT_POLICY.md` exists to prevent. Whether to spend it is a
> methodological call for the project owner, not a step to take in passing.

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
independent data sources, which validates the signal.**

> **Correction, 2026-07-25 (later session).** The vbt column above was measured
> with the vol-target layer inactive, and the "sizing defect" diagnosed from it
> does not exist. Everything below the table supersedes the original reading;
> the table's vbt column is kept only as the record of the faulty measurement.

Re-measured with `pipeline(data)` at its documented defaults — `target_vol=0.25`,
`max_leverage=3.0`, the same window:

| | vbt (as published above) | vbt (re-measured) | QC (edge-triggered) |
|---|---|---|---|
| CAGR | 8.44% | **18.65%** | 20.17% |
| Sharpe | 0.726 | 0.700 | 0.575 |
| ann. volatility | 12.19% | **23.74%** | 23.3% |
| max drawdown | −23.31% | **−46.51%** | −51.9% |
| trades | 50 | 50 | 128 orders |

The volatility gap that motivated the whole diagnosis is largely gone: 23.74%
against QC's 23.3%, for a 25% target. Volatility was cross-checked three ways
(`std(rets)·√252`, `returns_acc.annualized_volatility`, log-returns) — 23.74 /
23.74 / 23.86%.

**Why the original figures were wrong.** Passing `target_vol=None`, which sizes
flat at 1×, reproduces them: 12.46% volatility, −24.47% drawdown, 54.13% mean
gross exposure — against the 12.19% / −23.31% / 52.7% published above. The vbt
column was a 1× run compared against a 25% target.

The exposure arithmetic closes the case. Mean gross exposure is 102.66% across
all bars, but the sleeve is in position only 54.1% of the time; **while in
position it averages 189.67%**, against a median vol-target leverage of 2.007.
`size_type="percent"` is not capping anything. The published 52.7% is simply
100% × 54.1% — a flat run measured over all bars.

Consequences:

- **The sizing-regime comparison in §4 stands**, as it did before — every regime
  shares the same plumbing, and the risk-matched comparison normalises volatility
  explicitly.
- **§6 portfolio weights need no correction on this account.** They were derived
  from sleeve returns that are internally consistent; what was wrong was the
  cross-engine reading, not the sleeve.
- **No sizing fix is to be applied.** The previously prescribed change —
  `size_type="targetpercent"` — is not merely unnecessary, it is **rejected by
  the engine**: `from_signals` raises `ValueError: Target size types are not
  supported`. The two patterns cited as precedent (`daily_momentum.py:224-233`,
  `composite_fx_alpha.py:389-398`) both use `from_orders`, which does support
  target sizing. The gold sleeve needs `from_signals` for its edge-triggered
  entries/exits, its `sl_stop`, and the `signal_func_nb` seam the sizing overlays
  plug into, so the two are not interchangeable.

What remains genuinely open for vbt ↔ QC is the drawdown (−46.51% vs −51.9%) and
CAGR (18.65% vs 20.17%), both consistent with the known fill-timing difference —
vbt fills at the signal bar's close, QC at the T+1 open.

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
- **Pre-existing breakage — fixed on 2026-07-25 (commit `c3f1809`).**
  `utils.apply_vbt_settings()` set `plotting.pre_show_func`, a key vbt 2026.6.27
  renamed to `pre_render_func`; the frozen config raised `KeyError` and took down
  all 17 `tests/test_pipeline_equivalence.py` cases at fixture setup. The key is
  now selected from the installed version.

  Worth knowing what it hid: the `KeyError` fired on the line *before*
  `vbt.settings.returns.year_freq = 252 days`, so that setting never applied — any
  annualised metric read in an affected session used vbt's 365-day default. It also
  masked 9 stale snapshots, re-baselined in `23c02e3`. The environment had drifted
  from `uv.lock` on 18 of 35 packages, pandas 2.3.3 → 3.0.5 among them, and
  `vectorbtpro` is not covered by the lock at all. **Pin the environment before
  trusting any cross-engine comparison** — a backtest that is not reproducible
  month to month cannot be reconciled against anything.
- A bug worth remembering: an early version of `sweep_gold_sizing.py` sliced
  prices to the holdout *before* computing the signal, stranding the 250-session
  lookback with no history and scoring 250 of 333 blind sessions on a signal
  that did not exist. Always simulate on full history and slice the returns.
