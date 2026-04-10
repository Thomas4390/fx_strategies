# FX Strategies — ims_pipeline architecture

VBT Pro intraday & daily FX mean reversion / momentum strategies, built
on the `pipeline / pipeline_nb / create_cv_pipeline` pattern (see
`example/ims_pipeline.py` for the reference template).

## Layout

```
src/
├── framework/
│   ├── pipeline_utils.py    # METRIC constants, compute_metric_nb,
│   │                        # analyze_portfolio, plot_cv_heatmap/volume/splitter
│   └── plotting.py          # low-level plot helpers used by analyze_portfolio
├── strategies/
│   ├── mr_turbo.py          # Intraday VWAP MR — pipeline / run_grid / create_cv_pipeline
│   ├── mr_macro.py          # MR + macro regime (yield spread, unemployment)
│   ├── rsi_daily.py         # Daily RSI mean reversion
│   ├── daily_momentum.py    # XS + TS momentum (two pipelines: _xs and _ts)
│   ├── ou_mean_reversion.py # VWAP MR + vol-targeted dynamic leverage
│   ├── composite_fx_alpha.py# Multi-factor daily rebalance
│   └── combined_portfolio.py# Returns aggregator — imports the returns helpers
├── research/                # Sweep / walk-forward / reporting scripts
│                            # (consume the backtest_* shims — stable API)
└── utils.py                 # load_fx_data, Numba helpers (VWAP, vol, leverage)
```

## Running a strategy

Every strategy exposes a `__main__` with three modes:

```bash
# Single run with analyze_portfolio report (tearsheet, stats, overlay)
python src/strategies/mr_turbo.py --mode single --show

# Parameter grid search via vbt.Param + @vbt.parameterized (threadpool)
python src/strategies/mr_turbo.py --mode grid --show

# Walk-forward CV via @vbt.cv_split — produces (grid_perf, best_perf)
python src/strategies/mr_turbo.py --mode cv --n-folds 15 --show
```

Programmatic use:

```python
from strategies.mr_turbo import pipeline, run_grid, create_cv_pipeline
import vectorbtpro as vbt

# 1. Investigation
pf, ind = pipeline(data, bb_window=80, bb_alpha=5.0, sl_stop=0.005, tp_stop=0.006)
print(pf.stats())

# 2. Grid search (returns pd.Series multi-indexed by swept params)
from framework.pipeline_utils import SHARPE_RATIO
grid = run_grid(
    data,
    bb_window=[40, 60, 80, 120],
    bb_alpha=[4.0, 5.0, 6.0],
    sl_stop=[0.004, 0.005, 0.006],
    tp_stop=[0.004, 0.006, 0.008],
    metric_type=SHARPE_RATIO,
)
grid.vbt.heatmap(x_level="bb_window", y_level="bb_alpha", slider_level="sl_stop").show()

# 3. Walk-forward CV
splitter = vbt.Splitter.from_purged_walkforward(
    data.index, n_folds=15, n_test_folds=1, purge_td="1 day", min_train_folds=3,
)
cv = create_cv_pipeline(splitter, metric_type=SHARPE_RATIO)
grid_perf, best_perf = cv(
    data,
    bb_window=vbt.Param([40, 60, 80, 120]),
    bb_alpha=vbt.Param([4.0, 5.0, 6.0]),
    sl_stop=0.005,
    tp_stop=0.006,
)
print(best_perf)  # best per (split, set) with in-sample and out-of-sample Sharpe
```

## Tests

```bash
pytest tests/                      # 46 tests, ~4 min wall-clock
pytest tests/test_pipeline_utils.py -v          # metric dispatch unit tests
pytest tests/test_pipeline_equivalence.py -v    # rtol=1e-10 vs frozen snapshots
```

The equivalence tests replay `pipeline(data, **params)` against baseline
`stats()` snapshots generated from the legacy `backtest_*` functions
before they were migrated. Any numerical drift triggers a failure.

## Legacy compatibility

Each strategy keeps a thin `backtest_*` shim that delegates to
`pipeline(data, **kwargs)[0]`. This preserves the `src/research/`
scripts unchanged. New code should import `pipeline` directly.

## Metric constants

`framework.pipeline_utils` defines integer metric IDs (Numba-safe) for
`compute_metric_nb` dispatch:

```
TOTAL_RETURN, SHARPE_RATIO, CALMAR_RATIO, SORTINO_RATIO, OMEGA_RATIO,
ANNUALIZED_RETURN, MAX_DRAWDOWN, PROFIT_FACTOR, VALUE_AT_RISK,
TAIL_RATIO, ANNUALIZED_VOLATILITY, INFORMATION_RATIO, DOWNSIDE_RISK,
COND_VALUE_AT_RISK
```

Metrics where "lower is better" (drawdown, VaR, volatility, CVaR,
downside risk) have their sign flipped so the CV selection logic can
always maximize.

## Annualization

- FX minute data (24h market): `FX_MINUTE_ANN_FACTOR = 24 * 60 * 252 = 362880`
- Daily data: `DAILY_ANN_FACTOR = 252`
- Stock intraday (NYSE, 6.5h): `STOCK_MINUTE_ANN_FACTOR = 6.5 * 60 * 252 = 98280`
