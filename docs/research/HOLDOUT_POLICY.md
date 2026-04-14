# HOLDOUT POLICY — strict OOS lock-in

**Effective date** : 2026-04-13 (Phase 21)
**Status** : active
**Review due** : after Phase 25 or 2026-12-31, whichever comes first

## Problem statement

Between Phase 1 and Phase 20C the project ran 20 successive research
phases on **the same historical window 2019-01-01 → 2026-04-01**.
Across the 5 large sweeps alone (`sweep_combinations.py` 37 configs,
`sweep_phase19.py` 116 configs, `sweep_phase20a_weights.py` 58 configs,
`sweep_phase20b_fourth_sleeve.py` 45 configs, `sweep_phase20c_dd_cap.py`
34 configs) we tested **290+ strategy configurations** against the same
7 walk-forward windows. Each phase inherited the best points of the
previous phase, amplifying the effective test count (tree search with
memory of past winners).

No formal correction for multiple testing was applied. Phase 21
introduces Deflated Sharpe Ratio and Probability of Backtest Overfitting
(see `src/framework/statistical_testing.py`) but **the statistical tests
are only a safety net**. The empirically robust protection is a
hold-out : a slice of data that the research process has never been
allowed to touch.

## The rule

### Frozen slice

**All data on or after `2026-01-01` is locked out of model selection
until at least Phase 25 or 2026-12-31.**

- Date of freeze : **2026-01-01 00:00:00 UTC**.
- Locked columns in `WF_PERIODS` : the window `("2025-01-01",
  "2026-04-01")` is **split** into a training tail `2025-01-01 →
  2025-12-31` and a frozen OOS tail `2026-01-01 → 2026-04-01`.
- The frozen tail is ~3 months of business days (~65 bars). It will
  grow as new data lands — when 2027 data arrives, the frozen tail
  automatically covers `2026-01-01 → 2027-06-30` (18 months).

### What is forbidden

During the lock-in period, the following operations are **prohibited**
on the frozen slice :

1. Reading the slice as input to any `@vbt.parameterized` or
   `@vbt.cv_split` decorated function used for parameter selection.
2. Computing a Sharpe, CAGR, MaxDD or any ranking metric on the slice
   and using it to sort, filter, or select configurations.
3. Including the slice in `WF_PERIODS` when computing walk-forward
   averages used as an optimization objective.
4. Running any sweep script (`scripts/sweep_phase*.py`) that touches
   the slice.
5. Posting metric comparisons on the slice in `docs/research/*.md`
   before the lock-in is released.

### What is allowed

- **Single inference pass** : once per phase, run the best candidate
  from the locked-in research region on the frozen slice and report
  the raw result alongside the in-sample result. The report must be
  marked `FROZEN_OOS_RESULT` and is the only number from the slice
  that can appear in a doc.
- **Visualization** : plotting the equity curve of a final selected
  strategy across the full period (including the frozen tail) in a
  single tearsheet, for human sanity check, is allowed as long as it
  is not used to re-select.
- **Data quality check** : running a data-level sanity test (NaN
  count, gaps, index monotonicity) on the frozen slice.

### Release conditions

The lock-in is released when **all** of the following are satisfied :

1. Phase 25 is closed (roadmap in `plans/valiant-humming-coral.md`).
2. A `docs/research/HOLDOUT_RELEASE.md` document is produced that
   lists : the candidate configurations promoted to production, the
   raw frozen-tail Sharpe / CAGR / MaxDD for each, and a final DSR
   accounting for every sweep run during the lock-in.
3. The user signs off explicitly on the release.

Alternatively, **automatic release on 2026-12-31** — after which point
the slice is no longer treated as frozen, but the cumulated DSR
penalty continues to grow and is published in every subsequent
decision.

## Enforcement

### Code-level

`src/framework/holdout.py` (to be created in Phase 22) will expose two
helpers :

```python
from framework.holdout import HOLDOUT_START, is_in_holdout, assert_not_optimizing

HOLDOUT_START = pd.Timestamp("2026-01-01", tz="UTC")

def is_in_holdout(ts: pd.Timestamp) -> bool: ...
def assert_not_optimizing(index: pd.Index, context: str) -> None:
    """Raise RuntimeError if ``index`` touches the frozen slice."""
```

Every sweep script in `scripts/sweep_phase*.py` is expected to call
`assert_not_optimizing(common_all.index, context="phase22_sweep")` at
the top of `main()`. The assertion reads `HOLDOUT_START` and inspects
the index of the data actually used for ranking. Presence of any
timestamp `>= HOLDOUT_START` triggers an immediate failure.

Until the helper exists, the rule is honored manually :
**do not update `WF_PERIODS` to extend past 2025-12-31 until the
lock-in is released**.

### Documentation-level

All Phase 21+ research reports (`docs/research/phase21*.md` and later)
must include the following header :

```
**Holdout state** : LOCKED  (frozen from 2026-01-01 until Phase 25)
**Holdout touched by this phase** : NO
```

If a phase does produce a single-pass frozen-tail result, the header
instead reads :

```
**Holdout state** : LOCKED  (frozen from 2026-01-01 until Phase 25)
**Holdout touched by this phase** : YES — single inference pass,
    config <id>, no re-selection
```

## Rationale — why this matters

Bailey-Borwein-Lopez de Prado-Zhu (2015) show that when ``N`` strategy
trials are performed under the same historical window and the top one
is selected, the published Sharpe ratio overstates the true Sharpe by
a factor that grows with ``√ln N``. For ``N = 300`` and an observed
Sharpe of 1.0, the deflated estimate drops to ``~0.7`` purely from
selection bias — before any other correction. The only way to avoid
this inflation empirically is to evaluate the chosen strategy on data
it has not seen.

We also inherit the risk that minor decisions made during the phases
— which sleeves to load, which metric to maximize, which weights to
test — were themselves subtly influenced by observed behavior on the
2025-2026 tail. A frozen slice is the cheapest way to disprove that
influence : if the frozen-tail Sharpe matches the in-sample estimate
within a reasonable confidence band, the selection was robust ; if
the frozen-tail Sharpe falls apart, we know the optimization was
nearer to curve-fitting than we believed.

## Historical record (locked-in configs under watch)

The following configurations were produced by the research phases
preceding this policy and will be the first ones evaluated on the
frozen tail when the lock-in is released :

| Phase | Config ID | Weights | tv / ml / DD | Sharpe WF (IS) |
|-------|-----------|---------|--------------|----------------|
| 18    | Phase 18 prod | 80/10/10 | 0.28 / 12 / off | 0.956 |
| 19    | P19c-tv25-ml14-dd_OFF | 80/10/10 | 0.25 / 14 / off | 0.966 |
| 20A   | P20a-w75-10-15 | 75/10/15 | 0.25 / 14 / off | **0.972** |
| 20C   | P20c-soft-w75-10-15-k20-f35 | 75/10/15 | 0.25 / 14 / soft knee=0.20 | **0.972** |

All four sit within 16 bps of each other on the in-sample WF Sharpe.
The DSR retrofit (Phase 21) will quantify how much of that spread is
statistically distinguishable from selection bias.

## References

- Bailey, Borwein, Lopez de Prado, Zhu (2015). *The Probability of
  Backtest Overfitting*. [davidhbailey.com](https://www.davidhbailey.com/dhbpapers/backtest-prob.pdf)
- Bailey, Lopez de Prado (2014). *The Deflated Sharpe Ratio*.
  [davidhbailey.com](https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf)
- Lopez de Prado (2018). *Advances in Financial Machine Learning*,
  Chapter 7 (Cross-Validation in Finance) and Chapter 11 (The Dangers
  of Backtesting).
