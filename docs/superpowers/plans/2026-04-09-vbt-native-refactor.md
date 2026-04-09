# VBT Pro Native Refactoring — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace manual Numba kernels (ADX, TWAP) with native VBT Pro functions, fix broken threadpool parallelism, add best-combination visualization with browser rendering.

**Architecture:** Add optional `prepare_fn` hook to `StrategySpec` for pre-computing native VBT indicators (VWAP, ADX). Runner injects these arrays into kernels and signal functions. Fix `from_signals` to use `chunked="threadpool"` instead of broken `jitted(parallel=True)`. All plots render to browser via `renderer="browser"`.

**Tech Stack:** vectorbtpro (VWAP, talib:ADX, Splitter, plotting), numba (remaining session-resetting kernels), plotly (browser rendering), pytest

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `src/framework/spec.py` | Modify | Add `prepare_fn` field, `"pre"` prefix validation |
| `src/framework/runner.py` | Modify | Add `_run_prepare()`, fix threadpool, resolve `"pre.*"` sources |
| `src/framework/plotting.py` | Modify | Fix monthly heatmap, add 6 new analysis plot functions, browser rendering |
| `src/utils.py` | Modify | Remove ADX/TWAP (~180 lines), add `compute_mr_bands_nb`, add volume dummy in `load_fx_data` |
| `src/strategies/ou_mean_reversion.py` | Modify | Add `prepare_fn`, simplified kernel |
| `src/strategies/mr_v1.py` | Modify | Add `prepare_fn`, use simplified kernel |
| `src/strategies/mr_v2.py` | Modify | Add `prepare_fn`, use simplified kernel |
| `src/strategies/mr_v3.py` | Modify | Add `prepare_fn`, use simplified kernel |
| `src/strategies/mr_v4.py` | Modify | Add `prepare_fn`, use simplified kernel |
| `src/strategies/donchian_breakout.py` | Modify | Add `prepare_fn` for VWAP, remove VWAP from kernel |
| `src/strategies/kalman_trend.py` | Modify | Add `prepare_fn` for VWAP, remove VWAP from kernel |
| `src/strategies/composite_fx_alpha.py` | No change | Daily strategy, no TWAP/ADX |
| `tests/test_prepare_fn.py` | Create | Test native VWAP/ADX pre-computation |
| `tests/test_backtest.py` | Create | Test single backtest produces valid results |
| `tests/test_threading.py` | Create | Test chunked threadpool sweep runs correctly |
| `tests/test_plotting.py` | Create | Test all plot functions return figures |
| `tests/test_registry.py` | Create | Test all strategies backtest without error |
| `tests/conftest.py` | Create | Shared fixtures (mini data slice) |

---

### Task 1: Test fixtures and data loading fix

**Files:**
- Create: `tests/conftest.py`
- Modify: `src/utils.py:75-118`

- [ ] **Step 1: Create test fixtures with mini data**

```python
# tests/conftest.py
"""Shared test fixtures — 1-week data slice for fast tests."""

import numpy as np
import pandas as pd
import pytest
import vectorbtpro as vbt


@pytest.fixture(scope="session")
def raw_and_data():
    """Load 1 week of EUR-USD data for all tests."""
    from utils import load_fx_data

    raw, data = load_fx_data("data/EUR-USD.parquet")
    # Take last 5 trading days (~6300 bars at ~21h/day * 60min)
    raw_mini = raw.iloc[-6300:].copy()
    index_ns = vbt.dt.to_ns(raw_mini.index)
    return raw_mini, data, index_ns


@pytest.fixture(scope="session")
def raw(raw_and_data):
    return raw_and_data[0]


@pytest.fixture(scope="session")
def data(raw_and_data):
    return raw_and_data[1]


@pytest.fixture(scope="session")
def index_ns(raw_and_data):
    return raw_and_data[2]
```

- [ ] **Step 2: Add volume dummy column in `load_fx_data`**

In `src/utils.py`, after building `raw` (line ~113), add volume if missing:

```python
def load_fx_data(
    path: str = "data/EUR-USD.parquet",
    shift_hours: int = 0,
) -> tuple[pd.DataFrame, vbt.Data]:
    # ... existing parquet loading code ...

    # Raw DataFrame with lowercase columns for Numba kernels
    raw = df.copy()
    raw.columns = [c.lower() for c in raw.columns]

    # Add dummy volume for FX data (required by vbt.VWAP.run)
    if "volume" not in raw.columns:
        raw["volume"] = 1.0

    # VBT Data wrapper with capitalized columns for native functions
    df_cap = raw.copy()
    df_cap.columns = [c.capitalize() for c in df_cap.columns]
    data = vbt.Data.from_data({symbol: df_cap}, tz_localize=False, tz_convert=False)
    return raw, data
```

Note: the `df` variable used to build `data` must now include the Volume column too, so we build `df_cap` from `raw` instead.

- [ ] **Step 3: Run fixture test**

Run: `cd /home/thomas/Documents_Thomas/11_CodingProjects/fx_strategies/fx_strategies && python -m pytest tests/conftest.py --collect-only`
Expected: conftest collected, no errors

- [ ] **Step 4: Commit**

```bash
git add tests/conftest.py src/utils.py
git commit -m "feat: add test fixtures and volume dummy in load_fx_data"
```

---

### Task 2: Add `prepare_fn` to `StrategySpec`

**Files:**
- Modify: `src/framework/spec.py:1-213`

- [ ] **Step 1: Write the test**

```python
# tests/test_prepare_fn.py
"""Test prepare_fn hook and native VBT pre-computation."""

import numpy as np
import vectorbtpro as vbt

from framework.spec import StrategySpec, IndicatorSpec, ParamDef, PortfolioConfig


def test_spec_accepts_prepare_fn():
    """StrategySpec with prepare_fn should construct without error."""
    def dummy_prepare(raw, data):
        return {"vwap": np.ones(len(raw))}

    # Minimal valid spec with prepare_fn
    spec = StrategySpec(
        name="Test",
        indicator=IndicatorSpec(
            class_name="T", short_name="t",
            input_names=("close_minute",),
            param_names=(),
            output_names=("out",),
            kernel_func=lambda close: close,
        ),
        signal_func=lambda c: (False, False, False, False),
        signal_args_map=(("close_arr", "data.close"),),
        params={},
        prepare_fn=dummy_prepare,
    )
    assert spec.prepare_fn is not None


def test_spec_without_prepare_fn():
    """StrategySpec without prepare_fn should still work (backwards compat)."""
    spec = StrategySpec(
        name="Test",
        indicator=IndicatorSpec(
            class_name="T", short_name="t",
            input_names=("close_minute",),
            param_names=(),
            output_names=("out",),
            kernel_func=lambda close: close,
        ),
        signal_func=lambda c: (False, False, False, False),
        signal_args_map=(("close_arr", "data.close"),),
        params={},
    )
    assert spec.prepare_fn is None


def test_pre_prefix_valid_in_signal_args_map():
    """signal_args_map with 'pre.*' prefix should pass validation."""
    def dummy_prepare(raw, data):
        return {"vwap": np.ones(10)}

    spec = StrategySpec(
        name="Test",
        indicator=IndicatorSpec(
            class_name="T", short_name="t",
            input_names=("close_minute",),
            param_names=(),
            output_names=("out",),
            kernel_func=lambda close: close,
        ),
        signal_func=lambda c: (False, False, False, False),
        signal_args_map=(
            ("close_arr", "data.close"),
            ("vwap_arr", "pre.vwap"),
        ),
        params={},
        prepare_fn=dummy_prepare,
    )
    assert ("vwap_arr", "pre.vwap") in spec.signal_args_map
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_prepare_fn.py -v`
Expected: `test_pre_prefix_valid_in_signal_args_map` FAILS (ValueError: Unknown source prefix 'pre')

- [ ] **Step 3: Implement changes in spec.py**

In `src/framework/spec.py`, add `prepare_fn` field and `"pre"` prefix:

```python
# At the top of the file, add import:
from collections.abc import Callable

# In StrategySpec dataclass, add field after plot_config:
    prepare_fn: Callable | None = None

# In __post_init__, update valid_prefixes:
        valid_prefixes = {"data", "ind", "extra", "param", "eval", "pre"}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_prepare_fn.py -v`
Expected: All 3 PASS

- [ ] **Step 5: Commit**

```bash
git add src/framework/spec.py tests/test_prepare_fn.py
git commit -m "feat: add prepare_fn hook to StrategySpec for native VBT pre-computation"
```

---

### Task 3: Wire `prepare_fn` into the runner

**Files:**
- Modify: `src/framework/runner.py`

- [ ] **Step 1: Write the test**

```python
# tests/test_backtest.py
"""Test single backtest with prepare_fn produces valid results."""

import numpy as np
import pandas as pd
import vectorbtpro as vbt

from framework.runner import StrategyRunner


def test_backtest_ou_mr_with_prepare_fn(raw, index_ns, data):
    """OU MR strategy with prepare_fn should produce trades."""
    from strategies.ou_mean_reversion import spec

    runner = StrategyRunner(spec, raw, data)
    pf, ind = runner.backtest()

    assert pf.trades.count() > 0
    assert not np.isnan(pf.sharpe_ratio)
    assert hasattr(ind, "upper_band")
    assert hasattr(ind, "lower_band")


def test_backtest_mr_v1_with_prepare_fn(raw, index_ns, data):
    """MR V1 strategy with prepare_fn should produce trades."""
    from strategies.mr_v1 import spec

    runner = StrategyRunner(spec, raw, data)
    pf, ind = runner.backtest()

    assert pf.trades.count() > 0
```

- [ ] **Step 2: Implement `_run_prepare` and wire it in**

In `src/framework/runner.py`:

1. Add `_run_prepare` method:

```python
def _run_prepare(
    self,
    raw: pd.DataFrame,
    data: vbt.Data | None,
) -> dict[str, Any]:
    """Run prepare_fn if defined, return pre-computed arrays."""
    if self.spec.prepare_fn is None:
        return {}
    return self.spec.prepare_fn(raw, data)
```

2. Update `backtest()` to call `_run_prepare` and pass `prepared`:

```python
def backtest(self, **overrides: Any) -> tuple[vbt.Portfolio, Any]:
    params = self._resolve_params(overrides)
    prepared = self._run_prepare(self.raw, self.data)
    ind = self._run_indicator(self.raw, self.index_ns, params, prepared=prepared)
    pf = self._run_portfolio(self.raw, self.index_ns, ind, params, prepared=prepared)
    return pf, ind
```

3. Update `_run_indicator` signature to accept `prepared` and inject matching keys:

```python
def _run_indicator(
    self, raw, index_ns, params, *, parallel=True, prepared=None,
) -> Any:
    if prepared is None:
        prepared = {}
    # ... existing factory setup ...

    # Build input kwargs, injecting prepared arrays
    input_kwargs = self._build_input_kwargs(raw, index_ns, ispec.input_names, prepared)
    # ... rest unchanged ...
```

4. Update `_build_input_kwargs` to check `prepared` first:

```python
def _build_input_kwargs(
    self, raw, index_ns, input_names, prepared=None,
) -> dict[str, Any]:
    if prepared is None:
        prepared = {}
    kwargs: dict[str, Any] = {}
    for name in input_names:
        # Check prepared arrays first
        if name in prepared:
            kwargs[name] = prepared[name]
            continue
        # ... existing DEFAULT_INPUT_MAP resolution ...
    return kwargs
```

5. Update `_run_portfolio` to accept `prepared` and resolve `"pre.*"` sources:

```python
def _run_portfolio(
    self, raw, index_ns, ind_result, params, *, parallel=True, prepared=None,
) -> vbt.Portfolio:
    if prepared is None:
        prepared = {}
    # ... existing code ...
    # In the loop resolving broadcast_named_args:
    for rep_name, source in self.spec.signal_args_map:
        signal_args.append(vbt.Rep(rep_name))
        if source.startswith("eval:"):
            extra_top_level[rep_name] = vbt.RepEval(source[5:])
        else:
            broadcast_named_args[rep_name] = self._resolve_source(
                source, raw, index_ns, ind_result, params, prepared,
            )
```

6. Update `_resolve_source` to handle `"pre"`:

```python
@staticmethod
def _resolve_source(source, raw, index_ns, ind_result, params, prepared=None):
    if prepared is None:
        prepared = {}
    prefix, _, name = source.partition(".")
    if prefix == "data":
        return raw[name]
    if prefix == "ind":
        return getattr(ind_result, name).values
    if prefix == "extra":
        if name == "index_ns":
            return index_ns
        raise ValueError(f"Unknown extra source: {name}")
    if prefix == "param":
        return params[name]
    if prefix == "pre":
        return prepared[name]
    raise ValueError(f"Unknown source prefix: {prefix!r} in {source!r}")
```

7. Update `cv()` to call `_run_prepare` per split:

```python
# Inside the split loop, after getting raw_split:
prepared_split = self._run_prepare(raw_split, None)
# ... pass prepared=prepared_split to _run_indicator and _run_portfolio
```

Note: `data` is `None` for CV splits because the `vbt.Data` wrapper is for the full dataset. The `prepare_fn` must handle `data=None` by computing from `raw` directly. Update the spec accordingly (see Task 5).

8. Update `full_pipeline` and `save_backtest_plots` similarly — pass `prepared` through all paths.

- [ ] **Step 3: Run tests (they will fail until Task 5 updates the strategies)**

Run: `python -m pytest tests/test_backtest.py -v`
Expected: FAIL initially (strategies not yet migrated)

- [ ] **Step 4: Commit framework changes**

```bash
git add src/framework/runner.py
git commit -m "feat: wire prepare_fn into runner — pre-compute, resolve, CV support"
```

---

### Task 4: Fix threadpool parallelism

**Files:**
- Modify: `src/framework/runner.py:404-513`

- [ ] **Step 1: Write the test**

```python
# tests/test_threading.py
"""Test chunked threadpool parallelism works correctly."""

import vectorbtpro as vbt

from framework.runner import StrategyRunner


def test_sweep_with_threadpool(raw, index_ns, data):
    """Parameter sweep with chunked=threadpool should run without error."""
    from strategies.ou_mean_reversion import spec

    runner = StrategyRunner(spec, raw, data)
    params = spec.default_params()
    # Small 2x2 grid
    params["lookback"] = [40, 60]
    params["band_width"] = [2.0, 2.5]

    prepared = runner._run_prepare(raw, data)
    ind = runner._run_indicator(raw, index_ns, params, parallel=True, prepared=prepared)
    pf = runner._run_portfolio(raw, index_ns, ind, params, parallel=True, prepared=prepared)

    # Should have 4 columns (2x2 product)
    assert pf.wrapper.shape_2d[1] == 4
    assert pf.sharpe_ratio.notna().any()
```

- [ ] **Step 2: Fix the parallelism bug**

In `src/framework/runner.py`, `_run_portfolio` method, replace the parallel block:

```python
# BEFORE (BROKEN):
if parallel:
    pf_kwargs["jitted"] = {"parallel": True}

# AFTER (CORRECT):
if parallel and n_cols > 1:
    pf_kwargs["chunked"] = "threadpool"
```

- [ ] **Step 3: Run test**

Run: `python -m pytest tests/test_threading.py -v`
Expected: PASS (after Task 5 migrates the strategy)

- [ ] **Step 4: Commit**

```bash
git add src/framework/runner.py
git commit -m "fix: use chunked=threadpool instead of broken jitted(parallel=True)"
```

---

### Task 5: Migrate `utils.py` — remove ADX/TWAP, add `compute_mr_bands_nb`

**Files:**
- Modify: `src/utils.py`

- [ ] **Step 1: Add `compute_mr_bands_nb` function**

Add after `compute_intraday_bands_nb`:

```python
@njit(nogil=True)
def compute_mr_bands_nb(
    index_ns: np.ndarray,
    close: np.ndarray,
    vwap: np.ndarray,
    lookback: int,
    band_width: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Session-resetting Bollinger bands around pre-computed VWAP.

    Returns (zscore, upper_band, lower_band).
    """
    deviation = compute_deviation_nb(close, vwap)
    zscore = compute_intraday_zscore_nb(index_ns, deviation, lookback)
    upper, lower = compute_intraday_bands_nb(
        index_ns, vwap, deviation, lookback, band_width
    )
    return zscore, upper, lower
```

- [ ] **Step 2: Add shared `prepare_mr` function**

Add at end of utils.py (before signal function):

```python
def prepare_mr(
    raw: pd.DataFrame,
    data: vbt.Data | None,
    adx_period: int = 14,
    adx_threshold: float = 30.0,
) -> dict[str, np.ndarray]:
    """Pre-compute VWAP (native) and ADX regime filter (native talib).

    Works with both full vbt.Data and raw-only (CV splits where data=None).
    """
    # Native VWAP — session-anchored, resets daily
    vwap_ind = vbt.VWAP.run(
        raw["high"], raw["low"], raw["close"], raw["volume"], anchor="D"
    )
    vwap = vwap_ind.vwap.values

    # Native ADX on daily timeframe with anti-look-ahead realignment
    if data is not None:
        adx_result = data.run("talib:ADX", timeframe="1D", timeperiod=adx_period)
        adx_values = adx_result.real.values
    else:
        # CV split: build temporary Data wrapper for talib
        df_cap = raw.copy()
        df_cap.columns = [c.capitalize() for c in df_cap.columns]
        temp_data = vbt.Data.from_data(
            {"tmp": df_cap}, tz_localize=False, tz_convert=False
        )
        adx_result = temp_data.run("talib:ADX", timeframe="1D", timeperiod=adx_period)
        adx_values = adx_result.real.values

    # Regime filter: 1.0 = MR allowed (ADX below threshold), 0.0 = trending
    regime_ok = np.where(
        np.isnan(adx_values) | (adx_values < adx_threshold), 1.0, 0.0
    )

    return {"vwap": vwap, "regime_ok": regime_ok}
```

- [ ] **Step 3: Remove deprecated functions**

Remove these functions from `utils.py`:
- `compute_adx_nb` (lines ~239-294)
- `compute_daily_adx_broadcast_nb` (lines ~297-351)
- `compute_adx_regime_nb` (lines ~508-525)
- `compute_intraday_twap_nb` (lines ~359-386)
- `compute_mr_base_indicators_nb` (lines ~528-550)

Keep all other functions intact.

- [ ] **Step 4: Verify no import errors**

Run: `cd /home/thomas/Documents_Thomas/11_CodingProjects/fx_strategies/fx_strategies && python -c "from utils import compute_mr_bands_nb, prepare_mr, mr_band_signal_nb, compute_daily_rolling_volatility_nb, compute_leverage_nb; print('OK')"`
Expected: `OK`

- [ ] **Step 5: Commit**

```bash
git add src/utils.py
git commit -m "refactor: replace ADX/TWAP kernels with native VBT, add compute_mr_bands_nb and prepare_mr"
```

---

### Task 6: Migrate `ou_mean_reversion.py`

**Files:**
- Modify: `src/strategies/ou_mean_reversion.py`

- [ ] **Step 1: Rewrite the strategy**

```python
"""OU Mean Reversion: Vol-Targeted Leverage, VWAP Anchor.

Native VBT VWAP + talib ADX, simplified Numba kernel for bands + leverage.
"""

import numpy as np
from numba import njit

from framework.spec import (
    IndicatorSpec,
    OverlayLine,
    ParamDef,
    PlotConfig,
    PortfolioConfig,
    StrategySpec,
)
from utils import (
    compute_daily_rolling_volatility_nb,
    compute_leverage_nb,
    compute_mr_bands_nb,
    mr_band_signal_nb,
    prepare_mr,
)


# ═══════════════════════════════════════════════════════════════════════
# INDICATOR KERNEL (simplified — VWAP & ADX pre-computed natively)
# ═══════════════════════════════════════════════════════════════════════


@njit(nogil=True)
def compute_ou_indicators_nb(
    index_ns: np.ndarray,
    close: np.ndarray,
    vwap: np.ndarray,
    lookback: int,
    band_width: float,
    vol_window: int,
    sigma_target: float,
    max_leverage: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Bands around VWAP + vol-targeted leverage."""
    zscore, upper_band, lower_band = compute_mr_bands_nb(
        index_ns, close, vwap, lookback, band_width
    )
    rolling_vol = compute_daily_rolling_volatility_nb(index_ns, close, vol_window)
    leverage = compute_leverage_nb(rolling_vol, sigma_target, max_leverage)
    return zscore, upper_band, lower_band, leverage


# ═══════════════════════════════════════════════════════════════════════
# PREPARE FUNCTION (native VBT: VWAP + ADX)
# ═══════════════════════════════════════════════════════════════════════


def prepare_ou_mr(raw, data):
    return prepare_mr(raw, data, adx_period=14, adx_threshold=30.0)


# ═══════════════════════════════════════════════════════════════════════
# STRATEGY SPECIFICATION
# ═══════════════════════════════════════════════════════════════════════

spec = StrategySpec(
    name="OU Mean Reversion (Vol-Targeted)",
    indicator=IndicatorSpec(
        class_name="IntradayMR",
        short_name="imr",
        input_names=("index_ns", "close_minute", "vwap"),
        param_names=(
            "lookback", "band_width", "vol_window", "sigma_target", "max_leverage",
        ),
        output_names=("zscore", "upper_band", "lower_band", "leverage"),
        kernel_func=compute_ou_indicators_nb,
    ),
    signal_func=mr_band_signal_nb,
    signal_args_map=(
        ("close_arr", "data.close"),
        ("upper_arr", "ind.upper_band"),
        ("lower_arr", "ind.lower_band"),
        ("twap_arr", "pre.vwap"),
        ("regime_ok_arr", "pre.regime_ok"),
        ("index_arr", "extra.index_ns"),
        ("eod_hour", "param.eod_hour"),
        ("eod_minute", "param.eod_minute"),
        ("eval_freq", "param.eval_freq"),
    ),
    params={
        "lookback": ParamDef(60, sweep=[20, 40, 60, 120, 240]),
        "band_width": ParamDef(2.0, sweep=[1.5, 2.0, 2.5, 3.0, 3.5]),
        "vol_window": ParamDef(20),
        "sigma_target": ParamDef(0.01),
        "max_leverage": ParamDef(3.0),
        "eod_hour": ParamDef(21),
        "eod_minute": ParamDef(0),
        "eval_freq": ParamDef(5),
        "sl_stop": ParamDef(0.005, sweep=[0.002, 0.003, 0.005]),
    },
    portfolio_config=PortfolioConfig(leverage="ind.leverage", sl_stop=0.005),
    plot_config=PlotConfig(
        overlays=(
            OverlayLine("pre.vwap", "VWAP", color="#FF9800", dash="dash"),
            OverlayLine("ind.upper_band", "Upper Band", color="#E91E63", dash="dot"),
            OverlayLine("ind.lower_band", "Lower Band", color="#E91E63", dash="dot"),
        ),
    ),
    prepare_fn=prepare_ou_mr,
)


if __name__ == "__main__":
    from framework.runner import run_strategy

    run_strategy(spec, mode="full")
```

Note: `signal_args_map` uses `"pre.vwap"` for the TWAP/VWAP arg (the signal function param is still called `twap_arr` for backwards compat with the signal function signature, which doesn't need to change).

- [ ] **Step 2: Update overlay resolution for `"pre.*"` sources**

In `src/framework/plotting.py`, `resolve_overlays` function, add support for `"pre"` prefix. The runner must pass `prepared` dict to the plotting functions. Update `resolve_overlays`:

```python
def resolve_overlays(
    spec, raw, ind_result, prepared=None,
) -> dict[str, tuple[pd.Series, str | None, str | None]]:
    if prepared is None:
        prepared = {}
    result = {}
    for overlay in spec.plot_config.overlays:
        prefix, _, name = overlay.source.partition(".")
        if prefix == "ind":
            values = getattr(ind_result, name).values
        elif prefix == "data":
            values = raw[name].values
        elif prefix == "pre":
            values = prepared[name]
        else:
            continue
        series = pd.Series(values, index=raw.index, name=overlay.label)
        result[overlay.label] = (series, overlay.color, overlay.dash)
    return result
```

Update all call sites in `runner.py` to pass `prepared` to `resolve_overlays`.

- [ ] **Step 3: Run backtest test**

Run: `python -m pytest tests/test_backtest.py::test_backtest_ou_mr_with_prepare_fn -v`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add src/strategies/ou_mean_reversion.py src/framework/plotting.py
git commit -m "feat: migrate ou_mean_reversion to native VWAP + ADX via prepare_fn"
```

---

### Task 7: Migrate MR V1-V4

**Files:**
- Modify: `src/strategies/mr_v1.py`, `mr_v2.py`, `mr_v3.py`, `mr_v4.py`

- [ ] **Step 1: Migrate `mr_v1.py`**

MR V1 currently uses `compute_mr_base_indicators_nb` (now removed). Replace with a kernel that takes pre-computed VWAP and computes only bands:

```python
"""MR V1: No Leverage, Stop-Loss Only.

Intraday VWAP mean reversion without vol-targeted sizing.
Entry on band breach, exit on VWAP crossback or SL, EOD forced exit.
"""

import numpy as np
from numba import njit

from framework.spec import (
    IndicatorSpec,
    OverlayLine,
    ParamDef,
    PlotConfig,
    PortfolioConfig,
    StrategySpec,
)
from utils import compute_mr_bands_nb, mr_band_signal_nb, prepare_mr


MR_BAND_PLOT_CONFIG = PlotConfig(
    overlays=(
        OverlayLine("pre.vwap", "VWAP", color="#FF9800", dash="dash"),
        OverlayLine("ind.upper_band", "Upper Band", color="#E91E63", dash="dot"),
        OverlayLine("ind.lower_band", "Lower Band", color="#E91E63", dash="dot"),
    ),
)


@njit(nogil=True)
def compute_mr_v1_indicators_nb(
    index_ns: np.ndarray,
    close: np.ndarray,
    vwap: np.ndarray,
    lookback: int,
    band_width: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bands around VWAP — no leverage."""
    return compute_mr_bands_nb(index_ns, close, vwap, lookback, band_width)


def prepare_mr_v1(raw, data):
    return prepare_mr(raw, data, adx_period=14, adx_threshold=30.0)


spec = StrategySpec(
    name="MR V1: No Leverage",
    indicator=IndicatorSpec(
        class_name="MR_V1",
        short_name="mr_v1",
        input_names=("index_ns", "close_minute", "vwap"),
        param_names=("lookback", "band_width"),
        output_names=("zscore", "upper_band", "lower_band"),
        kernel_func=compute_mr_v1_indicators_nb,
    ),
    signal_func=mr_band_signal_nb,
    signal_args_map=(
        ("close_arr", "data.close"),
        ("upper_arr", "ind.upper_band"),
        ("lower_arr", "ind.lower_band"),
        ("twap_arr", "pre.vwap"),
        ("regime_ok_arr", "pre.regime_ok"),
        ("index_arr", "extra.index_ns"),
        ("eod_hour", "param.eod_hour"),
        ("eod_minute", "param.eod_minute"),
        ("eval_freq", "param.eval_freq"),
    ),
    params={
        "lookback": ParamDef(60, sweep=[20, 40, 60, 120, 240]),
        "band_width": ParamDef(2.0, sweep=[1.5, 2.0, 2.5, 3.0]),
        "eod_hour": ParamDef(21),
        "eod_minute": ParamDef(0),
        "eval_freq": ParamDef(5),
        "sl_stop": ParamDef(0.005, sweep=[0.001, 0.002, 0.003, 0.005]),
    },
    portfolio_config=PortfolioConfig(leverage=1.0, sl_stop=0.005),
    plot_config=MR_BAND_PLOT_CONFIG,
    prepare_fn=prepare_mr_v1,
)


if __name__ == "__main__":
    from framework.runner import run_strategy

    run_strategy(spec, mode="full")
```

- [ ] **Step 2: Migrate `mr_v2.py`, `mr_v3.py`, `mr_v4.py`**

Apply the same pattern: add `prepare_fn=prepare_mr_vN`, replace kernel to accept `vwap` input instead of computing TWAP+ADX internally, source VWAP and regime_ok from `"pre.*"`. Each V2/V3/V4 has its own signal function and unique kernel logic — keep those, just remove the ADX/TWAP parts.

For `mr_v2.py`: kernel computes zscore only (uses VWAP from prepare_fn), signal uses zscore thresholds.
For `mr_v3.py`: kernel adds session filter, uses VWAP from prepare_fn.
For `mr_v4.py`: kernel uses EWM std instead of rolling std, uses VWAP from prepare_fn.

- [ ] **Step 3: Run all MR tests**

Run: `python -m pytest tests/test_backtest.py -v`
Expected: All PASS

- [ ] **Step 4: Commit**

```bash
git add src/strategies/mr_v1.py src/strategies/mr_v2.py src/strategies/mr_v3.py src/strategies/mr_v4.py
git commit -m "feat: migrate MR V1-V4 to native VWAP + ADX via prepare_fn"
```

---

### Task 8: Migrate Donchian and Kalman strategies

**Files:**
- Modify: `src/strategies/donchian_breakout.py`, `src/strategies/kalman_trend.py`

- [ ] **Step 1: Migrate `donchian_breakout.py`**

The kernel currently calls `vbt.indicators.nb.vwap_1d_nb` directly inside `@njit`. Move VWAP to `prepare_fn`, keep channel computation in kernel:

```python
def prepare_donchian(raw, data):
    """Pre-compute VWAP natively."""
    vwap_ind = vbt.VWAP.run(
        raw["high"], raw["low"], raw["close"], raw["volume"], anchor="D"
    )
    return {"vwap": vwap_ind.vwap.values}
```

Update kernel to remove `volume` input and `vwap_1d_nb` call, output only channels. VWAP goes through `"pre.vwap"` in `signal_args_map`.

- [ ] **Step 2: Migrate `kalman_trend.py`**

Same pattern: VWAP to `prepare_fn`, kernel keeps Kalman + EMA logic:

```python
def prepare_kalman(raw, data):
    """Pre-compute VWAP natively."""
    vwap_ind = vbt.VWAP.run(
        raw["high"], raw["low"], raw["close"], raw["volume"], anchor="D"
    )
    return {"vwap": vwap_ind.vwap.values}
```

Kernel removes `volume` input and `vwap_1d_nb`, outputs only Kalman + EMA arrays. VWAP sourced from `"pre.vwap"`.

- [ ] **Step 3: Run tests**

Run: `python -m pytest tests/test_backtest.py -v -k "donchian or kalman"`
Add quick tests in `test_backtest.py`:

```python
def test_backtest_donchian(raw, index_ns, data):
    from strategies.donchian_breakout import spec
    runner = StrategyRunner(spec, raw, data)
    pf, ind = runner.backtest()
    assert pf.trades.count() >= 0  # May have 0 trades on 1 week


def test_backtest_kalman(raw, index_ns, data):
    from strategies.kalman_trend import spec
    runner = StrategyRunner(spec, raw, data)
    pf, ind = runner.backtest()
    assert pf.trades.count() >= 0
```

- [ ] **Step 4: Commit**

```bash
git add src/strategies/donchian_breakout.py src/strategies/kalman_trend.py tests/test_backtest.py
git commit -m "feat: migrate Donchian and Kalman to native VWAP via prepare_fn"
```

---

### Task 9: Fix plotting — native monthly heatmap + browser rendering

**Files:**
- Modify: `src/framework/plotting.py`

- [ ] **Step 1: Fix `plot_monthly_heatmap` to use native VBT**

Replace the entire function:

```python
def plot_monthly_heatmap(
    pf: vbt.Portfolio,
    title: str = "Monthly Returns (%)",
) -> go.Figure:
    """Create a year x month heatmap of portfolio returns using native VBT."""
    import calendar

    pf_daily = pf.resample("1D")
    mo_rets = pf_daily.resample("ME").returns
    mo_matrix = pd.Series(
        mo_rets.values,
        index=pd.MultiIndex.from_arrays(
            [mo_rets.index.year, mo_rets.index.month],
            names=["year", "month"],
        ),
    ).unstack("month")
    mo_matrix.columns = [calendar.month_abbr[m] for m in mo_matrix.columns]
    fig = mo_matrix.vbt.heatmap(
        is_x_category=True,
        trace_kwargs=dict(
            zmid=0,
            colorscale="RdYlGn",
            text=np.round(mo_matrix.values * 100, 1),
            texttemplate="%{text}%",
        ),
    )
    fig.update_layout(title=title, height=400)
    return fig
```

- [ ] **Step 2: Add browser rendering utility**

```python
def show_browser(fig: go.Figure) -> None:
    """Show a plotly figure in the default web browser."""
    fig.show(renderer="browser")
```

- [ ] **Step 3: Update `runner.py` — open plots in browser with `show(renderer="browser")`**

Replace all `self._open_html(path)` calls with direct `fig.show(renderer="browser")` where appropriate. Keep HTML saving for archival, but show key plots live:

In `_save_results`, after writing each HTML:

```python
# Show key plots in browser
fig.show(renderer="browser")  # replaces webbrowser.open(...)
```

Remove the `_open_html` static method and `import webbrowser`.

- [ ] **Step 4: Write plot test**

```python
# tests/test_plotting.py
"""Test all plot functions return valid figures."""

import plotly.graph_objects as go
import vectorbtpro as vbt

from framework.runner import StrategyRunner


def test_plot_monthly_heatmap(raw, data):
    from strategies.ou_mean_reversion import spec
    from framework.plotting import plot_monthly_heatmap

    runner = StrategyRunner(spec, raw, data)
    pf, _ = runner.backtest()
    fig = plot_monthly_heatmap(pf)
    assert isinstance(fig, go.Figure)


def test_plot_portfolio_summary(raw, data):
    from strategies.ou_mean_reversion import spec
    from framework.plotting import plot_portfolio_summary

    runner = StrategyRunner(spec, raw, data)
    pf, _ = runner.backtest()
    fig = plot_portfolio_summary(pf)
    assert isinstance(fig, go.Figure)


def test_plot_trade_analysis(raw, data):
    from strategies.ou_mean_reversion import spec
    from framework.plotting import plot_trade_analysis

    runner = StrategyRunner(spec, raw, data)
    pf, _ = runner.backtest()
    fig = plot_trade_analysis(pf)
    assert isinstance(fig, go.Figure)
```

- [ ] **Step 5: Commit**

```bash
git add src/framework/plotting.py src/framework/runner.py tests/test_plotting.py
git commit -m "fix: native VBT monthly heatmap, browser rendering for all plots"
```

---

### Task 10: Add best-combination analysis plots

**Files:**
- Modify: `src/framework/plotting.py`
- Modify: `src/framework/runner.py`

- [ ] **Step 1: Add `plot_equity_top_n`**

```python
def plot_equity_top_n(
    pf_sweep: vbt.Portfolio,
    n: int = 5,
    title: str = "Top Parameter Combos — Equity",
) -> go.Figure:
    """Overlay equity curves of the top N combos by Sharpe."""
    sharpes = pf_sweep.sharpe_ratio
    top_idx = sharpes.nlargest(n).index
    top_pf = pf_sweep[top_idx]
    fig = top_pf.value.vbt.plot()
    fig.update_layout(title=title, height=500)
    return fig
```

- [ ] **Step 2: Add `plot_cv_stability`**

```python
def plot_cv_stability(
    grid_perf: pd.Series,
    title: str = "CV Stability — Sharpe per Fold",
) -> go.Figure:
    """Bar chart of best-combo Sharpe per CV fold."""
    if "split" not in grid_perf.index.names:
        fig = go.Figure()
        fig.add_annotation(text="No CV splits available", showarrow=False)
        return fig

    # Get best combo across train splits
    train_mask = grid_perf.index.get_level_values("set").isin(["train", "set_0", 0])
    train = grid_perf[train_mask]
    sweep_levels = [n for n in train.index.names if n not in ("split", "set")]

    if sweep_levels:
        best_combo = train.groupby(sweep_levels).mean().idxmax()
        if isinstance(best_combo, tuple):
            mask = True
            for level, val in zip(sweep_levels, best_combo):
                mask &= train.index.get_level_values(level) == val
            fold_sharpes = train[mask]
        else:
            fold_sharpes = train.xs(best_combo, level=sweep_levels[0])
    else:
        fold_sharpes = train

    split_vals = fold_sharpes.index.get_level_values("split")
    fig = go.Figure(data=go.Bar(x=[f"Fold {s}" for s in split_vals], y=fold_sharpes.values))
    fig.add_hline(y=fold_sharpes.mean(), line_dash="dash", line_color="red",
                  annotation_text=f"Mean: {fold_sharpes.mean():.2f}")
    fig.update_layout(title=title, height=400, yaxis_title="Sharpe Ratio")
    return fig
```

- [ ] **Step 3: Add `plot_rolling_sharpe`**

```python
def plot_rolling_sharpe(
    pf: vbt.Portfolio,
    window: int = 252,
    title: str = "Rolling 1-Year Sharpe Ratio",
) -> go.Figure:
    """Native VBT rolling Sharpe on daily-resampled portfolio."""
    pf_daily = pf.resample("1D")
    rets = pf_daily.returns
    rolling_sr = rets.rolling(window).apply(
        lambda x: x.mean() / x.std() * np.sqrt(252) if x.std() > 0 else 0
    )
    fig = rolling_sr.vbt.plot()
    fig.add_hline(y=0, line_color="gray")
    fig.add_hline(y=1, line_dash="dot", line_color="green",
                  annotation_text="Sharpe=1")
    fig.update_layout(title=title, height=400)
    return fig
```

- [ ] **Step 4: Add `plot_partial_dependence`**

```python
def plot_partial_dependence(
    sweep_sharpes: pd.Series,
    param_grid: dict[str, list],
    title: str = "Parameter Sensitivity",
) -> go.Figure:
    """Marginal mean Sharpe for each swept parameter."""
    from plotly.subplots import make_subplots

    params = list(param_grid.keys())
    n = len(params)
    if n == 0:
        return go.Figure()

    fig = make_subplots(rows=1, cols=n, subplot_titles=params)
    idx_names = list(sweep_sharpes.index.names)

    for i, param in enumerate(params):
        matched = param
        for name in idx_names:
            if name and name.endswith(param):
                matched = name
                break
        if matched not in idx_names:
            continue
        marginal = sweep_sharpes.groupby(matched).mean()
        fig.add_trace(
            go.Bar(x=[str(v) for v in marginal.index], y=marginal.values, name=param),
            row=1, col=i + 1,
        )

    fig.update_layout(title=title, height=350, showlegend=False)
    return fig
```

- [ ] **Step 5: Add `plot_train_vs_test`**

```python
def plot_train_vs_test(
    grid_perf: pd.Series,
    title: str = "Train vs Test Sharpe (Overfitting Check)",
) -> go.Figure:
    """Scatter: train Sharpe (x) vs test Sharpe (y) per param combo."""
    if "set" not in grid_perf.index.names:
        return go.Figure()

    train_mask = grid_perf.index.get_level_values("set").isin(["train", "set_0", 0])
    test_mask = grid_perf.index.get_level_values("set").isin(["test", "set_1", 1])
    train = grid_perf[train_mask]
    test = grid_perf[test_mask]

    sweep_levels = [n for n in train.index.names if n not in ("split", "set")]
    if not sweep_levels:
        return go.Figure()

    train_avg = train.groupby(sweep_levels).mean()
    test_avg = test.groupby(sweep_levels).mean()
    common = train_avg.index.intersection(test_avg.index)

    if len(common) == 0:
        return go.Figure()

    fig = go.Figure(data=go.Scatter(
        x=train_avg.loc[common].values,
        y=test_avg.loc[common].values,
        mode="markers",
        text=[str(c) for c in common],
    ))
    max_val = max(train_avg.max(), test_avg.max()) * 1.1
    min_val = min(train_avg.min(), test_avg.min()) * 0.9
    fig.add_trace(go.Scatter(
        x=[min_val, max_val], y=[min_val, max_val],
        mode="lines", line=dict(dash="dash", color="gray"), name="y=x",
    ))
    fig.update_layout(
        title=title, height=500,
        xaxis_title="Train Sharpe", yaxis_title="Test Sharpe",
    )
    return fig
```

- [ ] **Step 6: Wire new plots into `_save_results` in runner.py**

After existing plots in `_save_results`, add:

```python
# Best-combination analysis plots
from framework.plotting import (
    plot_cv_stability,
    plot_partial_dependence,
    plot_rolling_sharpe,
    plot_train_vs_test,
    show_browser,
)

fig_cv_stab = plot_cv_stability(grid_perf, f"{name} — CV Stability")
fig_cv_stab.write_html(f"{results_dir}/cv_stability.html")
show_browser(fig_cv_stab)

fig_pd = plot_partial_dependence(
    grid_perf[grid_perf.index.get_level_values("set").isin(["train", "set_0", 0])],
    self.spec.sweep_grid(),
    f"{name} — Parameter Sensitivity",
)
fig_pd.write_html(f"{results_dir}/partial_dependence.html")
show_browser(fig_pd)

fig_tvt = plot_train_vs_test(grid_perf, f"{name} — Overfitting Check")
fig_tvt.write_html(f"{results_dir}/train_vs_test.html")
show_browser(fig_tvt)

for label, pf in [("train", pf_train), ("test", pf_test)]:
    fig_rs = plot_rolling_sharpe(pf, title=f"{name} — {label.title()} Rolling Sharpe")
    fig_rs.write_html(f"{results_dir}/rolling_sharpe_{label}.html")
    show_browser(fig_rs)
```

- [ ] **Step 7: Add plot tests**

Add to `tests/test_plotting.py`:

```python
def test_plot_cv_stability():
    import pandas as pd
    from framework.plotting import plot_cv_stability

    idx = pd.MultiIndex.from_tuples(
        [(0, "train", 60, 2.0), (1, "train", 60, 2.0)],
        names=["split", "set", "lookback", "band_width"],
    )
    grid = pd.Series([1.5, 1.2], index=idx)
    fig = plot_cv_stability(grid)
    assert isinstance(fig, go.Figure)


def test_plot_partial_dependence():
    import pandas as pd
    from framework.plotting import plot_partial_dependence

    idx = pd.MultiIndex.from_tuples(
        [(0, "train", 40, 2.0), (0, "train", 60, 2.0),
         (0, "train", 40, 2.5), (0, "train", 60, 2.5)],
        names=["split", "set", "lookback", "band_width"],
    )
    grid = pd.Series([1.0, 1.5, 1.2, 1.8], index=idx)
    fig = plot_partial_dependence(grid, {"lookback": [40, 60], "band_width": [2.0, 2.5]})
    assert isinstance(fig, go.Figure)
```

- [ ] **Step 8: Commit**

```bash
git add src/framework/plotting.py src/framework/runner.py tests/test_plotting.py
git commit -m "feat: add best-combination analysis plots with browser rendering"
```

---

### Task 11: Registry test — all strategies backtest

**Files:**
- Create: `tests/test_registry.py`

- [ ] **Step 1: Write the test**

```python
# tests/test_registry.py
"""Smoke test: every registered strategy backtests without error."""

import pytest
from strategies import REGISTRY
from framework.runner import StrategyRunner


@pytest.mark.parametrize("name", list(REGISTRY.keys()))
def test_strategy_backtests(name, raw, data):
    """Each strategy should construct, backtest, and return a Portfolio."""
    spec = REGISTRY[name]
    runner = StrategyRunner(spec, raw, data)
    pf, ind = runner.backtest()

    # Portfolio object is valid
    assert pf is not None
    assert pf.wrapper.shape_2d[0] > 0

    # Stats don't error
    stats = pf.stats()
    assert stats is not None
```

- [ ] **Step 2: Run all tests**

Run: `python -m pytest tests/ -v --tb=short`
Expected: All PASS. Total time < 60s.

- [ ] **Step 3: Commit**

```bash
git add tests/test_registry.py
git commit -m "test: add registry smoke test — all strategies backtest successfully"
```

---

### Task 12: Final validation — run full pipeline on OU MR

**Files:** No new files — validation only.

- [ ] **Step 1: Run OU MR full pipeline**

```bash
cd /home/thomas/Documents_Thomas/11_CodingProjects/fx_strategies/fx_strategies
python -m run_strategy ou_mean_reversion --mode=full --holdout_ratio=0.2 --n_folds=5
```

Expected:
- Prints core/thread info
- Runs 5-fold walk-forward CV with parameter sweep
- Selects best parameters
- Shows train + test stats
- Opens 10+ browser tabs: portfolio summary, monthly heatmap, trade signals, trade analysis, CV stability, partial dependence, train-vs-test, rolling sharpe (for both train and test)
- Total runtime < 10 minutes on the full dataset

- [ ] **Step 2: Run full test suite with timing**

```bash
python -m pytest tests/ -v --tb=short --durations=10
```

Expected: All PASS, total < 60s, no test > 15s.

- [ ] **Step 3: Final commit**

```bash
git add -A
git commit -m "chore: final validation — full pipeline + test suite passing"
```
