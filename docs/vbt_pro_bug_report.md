# VBT Pro Bug Report: pandas 3.x Compatibility Issues

**VBT Pro version:** 2026.3.1  
**pandas version:** 3.0.2  
**Python version:** 3.13.5  
**OS:** Linux (Ubuntu)

---

## Bug 1: `offset_to_timedelta` does not handle `pd.offsets.Day` (pandas 3.x breaking change)

### Summary

In pandas 3.x, `pd.offsets.Day` no longer inherits from `Tick`. It now directly extends `SingleConstructorOffset` → `BaseOffset`. This causes `pd.Timedelta(Day(1))` to raise `ValueError`, which breaks VBT's `offset_to_timedelta` function.

### MRO comparison (pandas 3.x)

```python
>>> [c.__name__ for c in pd.offsets.Day.__mro__]
['Day', 'SingleConstructorOffset', 'BaseOffset', 'object']

>>> [c.__name__ for c in pd.offsets.Hour.__mro__]
['Hour', 'Tick', 'SingleConstructorOffset', 'BaseOffset', 'object']
```

`Hour` and `Minute` still inherit from `Tick` and work fine with `pd.Timedelta()`. `Day` does not.

### Reproduction

```python
import pandas as pd
pd.Timedelta(pd.offsets.Day(1))
# ValueError: Value must be Timedelta, string, integer, float, timedelta or convertible, not Day
```

### Where it triggers in VBT

The error manifests when computing `sharpe_ratio` (or any metric requiring `ann_factor`) inside a `cv_split` pipeline. The `get_ann_factor` method calls `to_timedelta(freq)`, which calls `offset_to_timedelta(freq)` with a `Day` offset inferred from the portfolio wrapper.

```
File "vectorbtpro/returns/accessors.py", line 727, in get_ann_factor
    return dt.to_timedelta(year_freq, approximate=True) / dt.to_timedelta(freq, approximate=True)
File "vectorbtpro/utils/datetime_.py", line 563, in to_timedelta
    freq = offset_to_timedelta(freq)
File "vectorbtpro/utils/datetime_.py", line 513, in offset_to_timedelta
    return pd.Timedelta(offset)
ValueError: Value must be Timedelta, string, integer, float, timedelta or convertible, not Day
```

### Root cause

In `vectorbtpro/utils/datetime_.py`, `offset_to_timedelta` handles `BusinessDay`, `CustomBusinessDay`, `Week`, `Month*`, `Quarter*`, and `Year*` offsets, but **not** `pd.offsets.Day`:

```python
def offset_to_timedelta(offset: BaseOffset) -> pd.Timedelta:
    if isinstance(offset, (pd.offsets.BusinessDay, pd.offsets.CustomBusinessDay)):
        return pd.Timedelta(nb.d_td * offset.n)
    # ... other offset types handled ...
    return pd.Timedelta(offset)  # ← fails for Day in pandas 3.x
```

### Suggested fix

Add a `Day` handler before the fallback:

```python
if isinstance(offset, pd.offsets.Day):
    return pd.Timedelta(nb.d_td * offset.n)
```

### Current workaround

```python
import vectorbtpro.utils.datetime_ as vbt_dt

_original = vbt_dt.offset_to_timedelta

def _patched(offset):
    if isinstance(offset, pd.offsets.Day):
        return pd.Timedelta(days=offset.n)
    return _original(offset)

vbt_dt.offset_to_timedelta = _patched
```

---

## Bug 2: `arr_to_timedelta` OverflowError on large minute-bar datasets

### Summary

When calling `pf.stats()` or `pf.deep_getattr("sharpe_ratio")` on a portfolio with 3M+ minute bars, the `max_dd_duration` and related metrics overflow because `duration_in_bars * freq` exceeds pandas Timedelta's int64 nanosecond precision limit.

### Reproduction

```python
import vectorbtpro as vbt
import pandas as pd

# Load large minute-bar FX dataset (~3M bars)
# data = ...
pf = vbt.Portfolio.from_signals(close=close, freq='1min', ...)
pf.stats()  # OverflowError
```

### Error

```
File "vectorbtpro/base/accessors.py", line 662, in arr_to_timedelta
    out = a * freq
File "pandas/_libs/tslibs/timedeltas.pyx", line 2455, in Timedelta.__mul__
OverflowError: Python int too large to convert to C long
```

### Context

- Dataset: ~3,055,756 minute bars (EUR/USD, 2018-2026)
- The overflow occurs in `max_duration` / `max_dd_duration` metrics
- `pf.stats(settings=dict(to_timedelta=False))` does **not** prevent this because the overflow happens inside the cached property (`get_max_duration`), not in the stats builder's own timedelta conversion
- `pf.stats(settings=dict(to_timedelta=False, fill_wrap_kwargs=True))` also does not help because the property is resolved via `getattr(obj, attr)` in `_getattr_func`, bypassing `wrap_kwargs` injection

### Known Discord discussion

This was previously reported on Discord by another user with 300k datapoints. The maintainer suggested `pf.stats(settings=dict(to_timedelta=False))`, but this workaround is insufficient when the overflow occurs inside the property itself (not the stats builder's post-processing).

### Suggested fix

In `BaseIDXAccessor.arr_to_timedelta`, catch `OverflowError` and return the raw integer value (bar count) as a fallback:

```python
def arr_to_timedelta(self, a, to_pd=None, silence_warnings=None):
    freq = self.freq
    # ... existing code ...
    try:
        out = a * freq
    except OverflowError:
        if not silence_warnings:
            warnings.warn(
                "Duration exceeds pandas Timedelta precision. "
                "Returning raw bar count instead.",
                stacklevel=2,
            )
        return a
    return out
```

Alternatively, the `fill_wrap_kwargs=True` + `to_timedelta=False` path could be made to propagate into cached property computations triggered by the stats builder (e.g., by passing `wrap_kwargs` through `getattr_func` context).

### Current workaround

```python
import vectorbtpro.base.accessors as vbt_acc

_original = vbt_acc.BaseIDXAccessor.arr_to_timedelta

def _safe(self, a, to_pd=None, silence_warnings=None):
    try:
        return _original(self, a, to_pd, silence_warnings)
    except OverflowError:
        return a

vbt_acc.BaseIDXAccessor.arr_to_timedelta = _safe
```
