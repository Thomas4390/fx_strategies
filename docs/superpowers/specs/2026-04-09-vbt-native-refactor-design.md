# VBT Pro Native Refactoring — Design Spec

**Date**: 2026-04-09
**Scope**: `src/` framework + strategies
**Goal**: Replace manual Numba kernels with VBT Pro native functions where possible, fix threadpool parallelism, add best-combination visualization.

## 1. Architecture: `prepare_fn` Hook

### Spec Changes (`spec.py`)

Add one optional field to `StrategySpec`:

```python
prepare_fn: Callable[[pd.DataFrame, vbt.Data], dict[str, np.ndarray]] | None = None
```

- Returns `dict[str, ndarray]` of pre-computed native VBT arrays
- Keys matching `indicator.input_names` are injected as kernel inputs
- Keys accessible via `"pre.<name>"` prefix in `signal_args_map`
- Add `"pre"` to valid source prefixes in `__post_init__`
- Backwards compatible: strategies without `prepare_fn` are unchanged

### Runner Changes (`runner.py`)

Add `_run_prepare(raw, data) -> dict` called before indicator and portfolio runs.

Resolution chain:
1. `_run_prepare()` produces `prepared: dict[str, ndarray]`
2. `_run_indicator()` receives `prepared`, injects matching keys as kernel inputs
3. `_run_portfolio()` receives `prepared`, resolves `"pre.*"` sources in signal_args_map
4. CV loop calls `_run_prepare()` per split (native indicators must be recomputed on each data slice)

### Input Resolution in `_build_input_kwargs`

```python
# Existing resolution: DEFAULT_INPUT_MAP -> raw columns
# New resolution: if name in prepared -> use prepared[name]
for name in input_names:
    if name in prepared:
        kwargs[name] = prepared[name]
    elif ...  # existing logic
```

### Source Resolution in `_resolve_source`

```python
if prefix == "pre":
    return prepared[name]
```

## 2. Threading Fixes

### Fix 1: Portfolio parallelism (CRITICAL)

`jitted=dict(parallel=True)` does NOT work with `from_signal_func_nb` — confirmed by VBT maintainer. The Numba prange loop has multiple exit points which prevent parallelization.

```python
# runner.py _run_portfolio()
# BEFORE (broken):
pf_kwargs["jitted"] = {"parallel": True}

# AFTER (correct):
if n_cols > 1:
    pf_kwargs["chunked"] = "threadpool"
```

### Fix 2: Monthly heatmap GIL blocker

```python
# BEFORE (lambda blocks GIL):
monthly = rets.resample("ME").apply(lambda x: (1 + x).prod() - 1)

# AFTER (native VBT):
pf_daily = pf.resample("1D")
mo_rets = pf_daily.returns_acc.resample("M").get()
```

### Fix 3: Indicator factory parallelism

Already correct (`execute_kwargs={"engine": "threadpool"}`), no changes needed.

### Threading compatibility of `prepare_fn`

`prepare_fn` runs ONCE before the parallel phase. It produces plain `ndarray` objects that are:
- Passed to `broadcast_named_args` (auto-tiled by VBT to match column count)
- Accessed by `signal_func_nb` via `vbt.pf_nb.select_nb(c, arr)` (column-aware slicing)
- Fully compatible with `chunked="threadpool"` (each thread gets its own column slice)

No GIL interaction — the arrays are pre-computed numpy, not Python objects.

## 3. Native Indicator Migration

### Removed from `utils.py` (~180 lines)

| Function | Replacement |
|----------|-------------|
| `compute_adx_nb` (50 lines) | `data.run("talib:ADX", timeframe="1D", timeperiod=14)` |
| `compute_daily_adx_broadcast_nb` (40 lines) | Native realignment in `data.run(..., timeframe="1D")` |
| `compute_adx_regime_nb` (12 lines) | `np.where(adx < threshold, 1.0, 0.0)` in `prepare_fn` |
| `compute_intraday_twap_nb` (25 lines) | `vbt.VWAP.run(high, low, close, volume, anchor="D")` |
| `compute_mr_base_indicators_nb` (15 lines) | Decomposed: VWAP/ADX native, bands in simplified kernel |

### Kept in `utils.py`

| Function | Reason |
|----------|--------|
| `find_day_boundaries_nb` | Used by session-resetting rolling functions, no native equivalent |
| `compute_intraday_rolling_std_nb` | Session-resetting std, no native equivalent |
| `compute_intraday_zscore_nb` | Session-resetting zscore, no native equivalent |
| `compute_deviation_nb` | Simple but used inside `@njit` kernel chains |
| `compute_intraday_bands_nb` | Combines std + deviation around anchor |
| `compute_daily_rolling_volatility_nb` | Custom close-to-close vol broadcast |
| `compute_leverage_nb` | Vol-targeting, trivial but reused |
| `mr_band_signal_nb` | Signal function, unchanged |

### New shared utility

```python
@njit(nogil=True)
def compute_mr_bands_nb(index_ns, close, vwap, lookback, band_width):
    """Session-resetting bands around pre-computed VWAP."""
    deviation = compute_deviation_nb(close, vwap)
    zscore = compute_intraday_zscore_nb(index_ns, deviation, lookback)
    upper, lower = compute_intraday_bands_nb(index_ns, vwap, deviation, lookback, band_width)
    return zscore, upper, lower
```

## 4. Pilot Strategy: `ou_mean_reversion.py`

### `prepare_fn`

```python
def prepare_ou_mr(raw, data):
    vwap_ind = vbt.VWAP.run(raw["high"], raw["low"], raw["close"],
                             raw.get("volume", 1.0), anchor="D")
    adx = data.run("talib:ADX", timeframe="1D", timeperiod=14)
    regime_ok = np.where(np.isnan(adx.real.values) | (adx.real.values < 30.0), 1.0, 0.0)
    return {"vwap": vwap_ind.vwap.values, "regime_ok": regime_ok}
```

### Simplified kernel

```python
@njit(nogil=True)
def compute_ou_indicators_nb(index_ns, close, vwap,
                              lookback, band_width, vol_window,
                              sigma_target, max_leverage):
    zscore, upper_band, lower_band = compute_mr_bands_nb(
        index_ns, close, vwap, lookback, band_width)
    rolling_vol = compute_daily_rolling_volatility_nb(index_ns, close, vol_window)
    leverage = compute_leverage_nb(rolling_vol, sigma_target, max_leverage)
    return zscore, upper_band, lower_band, leverage
```

### Updated `IndicatorSpec`

- `input_names`: `("index_ns", "close_minute", "vwap")` — vwap injected from `prepared`
- `param_names`: `("lookback", "band_width", "vol_window", "sigma_target", "max_leverage")` — adx_period/adx_threshold removed
- `output_names`: `("zscore", "upper_band", "lower_band", "leverage")` — twap/regime_ok removed (now pre-computed)

### Updated `signal_args_map`

VWAP and regime_ok sourced via `"pre.*"` instead of `"ind.*"`.

## 5. Other Strategy Migration

Same pattern applied to:
- `mr_v1.py` through `mr_v4.py`: All share `compute_mr_base_indicators_nb` which embeds ADX+TWAP. Each gets a `prepare_fn` that pre-computes VWAP+ADX natively, simplified kernel for bands only.
- `donchian_breakout.py`: Uses VWAP in its kernel. Gets `prepare_fn` for native VWAP.
- `kalman_trend.py`: Uses VWAP. Gets `prepare_fn` for native VWAP.
- `composite_fx_alpha.py`: Daily strategy, no TWAP/ADX. No `prepare_fn` needed.

## 6. Visualization Additions (`plotting.py`)

### Corrected existing plots

1. **Monthly heatmap**: Native VBT `pf.resample("M").returns` + `vbt.ts_heatmap()` or pivot + `vbt.heatmap()`
2. **Portfolio summary**: Already native, no changes needed

### New analysis plots

1. **`plot_equity_top_n(pf_sweep, n=5)`** — Overlay equity curves of top N parameter combos by Sharpe
2. **`plot_cv_stability(grid_perf, best_params)`** — Bar chart of Sharpe per CV fold for best combo
3. **`plot_rolling_sharpe(pf)`** — Native VBT rolling Sharpe (252-day window)
4. **`plot_partial_dependence(sweep_results, param_grid)`** — For each swept param, marginal mean Sharpe
5. **`plot_train_vs_test(grid_perf)`** — Scatter: train Sharpe (x) vs test Sharpe (y) per combo
6. **`plot_best_combination_report(pf_train, pf_test, grid_perf, best_params)`** — Composite figure combining equity, monthly heatmap, CV stability, and stats comparison

### Integration in runner

`_save_results()` calls the new plots and saves as HTML alongside existing outputs.

## 7. Testing

| Test file | What it tests | Data size |
|-----------|--------------|-----------|
| `test_prepare_fn.py` | VWAP/ADX native produce valid arrays, shapes match raw | 1 week |
| `test_backtest.py` | Single backtest with prepare_fn, trades > 0, stats not NaN | 1 week |
| `test_threading.py` | `chunked="threadpool"` sweep runs without error | 1 week, 2x2 grid |
| `test_cv_pipeline.py` | `full_pipeline()` structure: opt_params, pf_train, pf_test | 1 month, 2x2 grid, 3 folds |
| `test_plotting.py` | Each plot function returns `go.Figure` without error | 1 week |
| `test_strategy_registry.py` | All 8 strategies instantiate and backtest without error | 1 week |

All tests use a small data slice for fast execution (< 30s total).

## 8. File Change Summary

| File | Action |
|------|--------|
| `src/framework/spec.py` | Add `prepare_fn` field, `"pre"` prefix |
| `src/framework/runner.py` | Add `_run_prepare()`, fix threadpool, update source resolution |
| `src/framework/plotting.py` | Fix monthly heatmap, add 6 new plot functions |
| `src/utils.py` | Remove ADX/TWAP/MR-base (~180 lines), add `compute_mr_bands_nb` |
| `src/strategies/ou_mean_reversion.py` | Add `prepare_fn`, simplified kernel, updated spec |
| `src/strategies/mr_v1.py` | Add `prepare_fn`, simplified kernel |
| `src/strategies/mr_v2.py` | Add `prepare_fn`, simplified kernel |
| `src/strategies/mr_v3.py` | Add `prepare_fn`, simplified kernel |
| `src/strategies/mr_v4.py` | Add `prepare_fn`, simplified kernel |
| `src/strategies/donchian_breakout.py` | Add `prepare_fn` for VWAP |
| `src/strategies/kalman_trend.py` | Add `prepare_fn` for VWAP |
| `src/strategies/composite_fx_alpha.py` | No changes |
| `tests/test_*.py` | 6 new test files |
