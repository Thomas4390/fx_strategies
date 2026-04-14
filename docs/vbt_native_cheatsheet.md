# VBT Pro Native Cheatsheet — fx_strategies

**Purpose.** Quick reference table: raw pandas/numpy → VBT Pro native
equivalent. Use this when writing or refactoring code under
`src/strategies/` or `src/framework/` to keep the codebase idiomatic.

**Sources.** Patterns observed in the three exemplary files
(`mr_turbo.py`, `ou_mean_reversion.py`, `rsi_daily.py`) and the VBT Pro
documentation (via MCP `vectorbtpro__search`).

**Golden rule.** Every data transformation should flow through a VBT
accessor (`.vbt.*`) or a VBT namespaced function (`vbt.*`) **unless**
you are inside a `@njit` kernel, in which case plain numpy is optimal.

---

## Quick reference table

| Category | Raw pandas/numpy | VBT native | Notes |
|---|---|---|---|
| **Indicators — RSI** | manual | `vbt.RSI.run(close, window=14)` or `vbt.talib('RSI').run(...)` | `vbt.IF.list_indicators("RSI*")` to see all 5 implementations |
| **Indicators — Bollinger** | manual | `vbt.BBANDS.run(close, window=n, alpha=k)` | Returns `.upper`, `.lower`, `.middle` |
| **Indicators — VWAP** | cumsum manual | `vbt.VWAP.run(h, l, c, v, anchor='D')` | `anchor` controls session reset |
| **Indicators — ATR** | manual | `vbt.ATR.run(h, l, c, window=n)` | Returns `.atr`, `.tr` |
| **Rolling mean** | `.rolling(n).mean()` | `.vbt.rolling_mean(n, minp=n)` | Numba-compiled |
| **Rolling std** | `.rolling(n).std(ddof=1)` | `.vbt.rolling_std(n, minp=n, ddof=1)` | Numba-compiled |
| **Rolling sum** | `.rolling(n).sum()` | `.vbt.rolling_sum(n, minp=n)` | Numba-compiled |
| **EWM mean** | `.ewm(span=n).mean()` | `.vbt.ewm_mean(span=n, minp=n, adjust=True)` | |
| **EWM std** | `.ewm(span=n).std()` | `.vbt.ewm_std(span=n, minp=n, adjust=True)` | |
| **Expanding std** | `.expanding(minp).std()` | `.vbt.expanding_std(minp=minp)` | |
| **pct_change** | `.pct_change()` | `.vbt.pct_change()` | Chainable with other accessors |
| **Log returns** | `np.log(c / c.shift(n))` | `np.log1p(c.vbt.pct_change(n))` | Bit-equivalent. Keep raw inside `@njit`. |
| **Crossover (up)** | `(a > b) & (a.shift(1) <= b.shift(1))` | `a.vbt.crossed_above(b)` | |
| **Crossover (down)** | `(a < b) & (a.shift(1) >= b.shift(1))` | `a.vbt.crossed_below(b)` | |
| **Causal shift** | `.shift(n).fillna(v)` | `.vbt.fshift(n, fill_value=v)` | `n` must be **positive** — negative = look-ahead |
| **Forward shift (PFO)** | `.shift(-n).fillna(v)` | keep raw | PFO semantic quirk, not a causal violation |
| **Cross-freq resample (down)** | `.resample('1D').last()` | `.vbt.resample_apply('1D', vbt.nb.last_reduce_nb)` | Reducers in `vbt.nb.*_reduce_nb` |
| **Cross-freq realign (up)** | `.reindex(...).ffill()` | `series.vbt.realign_closing(resampler, ffill=True)` | With `vbt.Resampler(src_idx, tgt_idx, ...)` |
| **Cross-freq realign (open)** | `.reindex(...).ffill()` on session-open | `series.vbt.realign_opening(resampler, ffill=True)` | For forward-looking references (prev week's ATR, yesterday's macro) |
| **Data loading** | `pd.read_parquet(path)` | `vbt.ParquetData.pull(path)` | Only if you need OHLCV; overkill for plain Series |
| **Parameter sweep** | `for p in ps: ...` | `vbt.Param([...])` + `@vbt.parameterized` | |
| **Random sweep** | `np.random.choice(...)` loop | `vbt.Param([...])` + `broadcast_kwargs={'random_subset': N}` | |
| **Custom indicator** | raw function | `vbt.IF.with_apply_func(func)` or `vbt.IF.from_expr(...)` | |
| **Portfolio — signals** | manual | `vbt.Portfolio.from_signals(close, entries, exits, sl_stop=..., tp_stop=..., td_stop=..., dt_stop=...)` | Use **native stops** not manual exits |
| **Portfolio — orders** | manual | `vbt.Portfolio.from_orders(close, size, size_type='targetpercent', ...)` | For weight-driven rebalancing |
| **Portfolio — optimizer** | manual | `vbt.Portfolio.from_optimizer(prices, pfo, pf_method='from_orders', ...)` | For `PortfolioOptimizer.from_filled_allocations` |
| **Multi-asset basket** | separate portfolios | `cash_sharing=True, group_by=True` | Shared margin pool across columns |
| **Sharpe / AR / MDD** | `qs.stats.*` / manual | `pf.sharpe_ratio`, `pf.annualized_return`, `pf.max_drawdown` | Native Numba, respects grouping |
| **Walk-forward CV** | manual loop | `@vbt.cv_split(...)` + `splitter` | |
| **Chunked grid** | manual chunking | `@vbt.chunked(chunk_len=N)` + `vbt.Chunked(...)` | RAM-bounded sweeps |

---

## Patterns with full examples

### 1. Indicators — always use `vbt.IF` entry points

```python
# GOOD (from mr_turbo.py / ou_mean_reversion.py)
vwap = vbt.VWAP.run(data.high, data.low, data.close, data.volume, anchor="D").vwap
bb   = vbt.BBANDS.run(close - vwap, window=80, alpha=5.0)
upper, lower = vwap + bb.upper, vwap + bb.lower

# BAD — manual cumsum, rolling std, manual band arithmetic
# (don't)
```

List all available implementations of an indicator:
```python
vbt.IF.list_indicators("RSI*")
# ['vbt:RSI', 'talib:RSI', 'pandas_ta:RSI', 'ta:RSIIndicator', 'technical:RSI']
```

Rule of thumb: **TA-Lib for speed, VBT for speed+plotting, others for
features you can't find elsewhere.**

### 2. Rolling operations — always on `.vbt.*`

```python
# GOOD
vol_21 = returns.vbt.rolling_std(21, minp=21, ddof=1) * np.sqrt(252)
ma_50  = close.vbt.rolling_mean(50, minp=50)
ewm    = close.vbt.ewm_mean(span=20, minp=20, adjust=True)

# BAD
vol_21 = returns.rolling(21, min_periods=21).std(ddof=1) * np.sqrt(252)
ma_50  = close.rolling(50).mean()
```

### 3. Causal shift — `.vbt.fshift`

```python
# GOOD
lev = (target_vol / vol.clip(lower=0.01)).clip(upper=5.0).vbt.fshift(1, fill_value=1.0)

# BAD (old pattern, still common in this codebase)
lev = (target_vol / vol.clip(lower=0.01)).clip(upper=5.0).shift(1).fillna(1.0)
```

**Important.** Negative `n` in `pd.Series.shift(-n)` is a **forward
lookup**. Never use it on data that should be causally isolated — you
will leak look-ahead information. The PFO pattern `aligned.shift(-1)`
in `combined_core.py:129` is a PFO-semantic quirk (weights at `t` apply
to returns at `t+1`), not a causal violation; it's a legitimate
exception.

### 4. Cross-frequency alignment — `vbt.Resampler` + `realign_*`

Minute-level data with daily macro filters, as in
`mr_macro.py::_get_aligned_macro`:

```python
# GOOD — canonical pattern
resampler = vbt.Resampler(
    source_index=macro_series.index,   # daily
    target_index=minute_index,         # 1-minute
    source_freq="D",
    target_freq="1min",
)
macro_min = macro_series.vbt.realign_opening(resampler, ffill=True)
# Use realign_opening for forward-looking references (previous day's value)
# Use realign_closing for backward-looking references (close-of-bar)

# BAD
macro_min = macro_series.reindex(minute_index, method='ffill')
```

Down-sampling (minute → daily) is the inverse direction, use
`.vbt.resample_apply`:
```python
daily_close = minute_close.vbt.resample_apply("1D", vbt.nb.last_reduce_nb)
daily_high  = minute_high.vbt.resample_apply("1D", vbt.nb.max_reduce_nb)
daily_low   = minute_low.vbt.resample_apply("1D", vbt.nb.min_reduce_nb)
daily_open  = minute_open.vbt.resample_apply("1D", vbt.nb.first_reduce_nb)
```

### 5. Portfolio with native stops — `from_signals`

```python
# GOOD (from mr_turbo.py / ou_mean_reversion.py)
pf = vbt.Portfolio.from_signals(
    data,                          # or pass close=...
    entries=long_mask,
    exits=False,                   # let stops handle exits
    short_entries=short_mask,
    short_exits=False,
    sl_stop=0.005,                 # 50 bps stop-loss
    tp_stop=0.006,                 # 60 bps take-profit
    dt_stop="21:00",               # session-end absolute time stop
    td_stop="6h",                  # relative time-delta stop
    leverage=leverage_array,       # per-bar leverage (broadcast to columns)
    slippage=0.00015,
    init_cash=1_000_000.0,
    freq="1min",
)

# BAD — manual SL/TP/EOD loops in Python
```

Always prefer the native stops (`sl_stop`, `tp_stop`, `dt_stop`,
`td_stop`) over manually computing exit signals. VBT handles them in
the Numba simulation loop, which is orders of magnitude faster and
bit-identical to a correct manual implementation.

### 6. Multi-asset with shared margin — `cash_sharing=True, group_by=True`

```python
# GOOD (from mr_macro.py:388 and combined_core.py:138)
pf_kwargs = dict(entries=..., short_entries=..., sl_stop=..., tp_stop=..., ...)
if is_multi:
    pf_kwargs["cash_sharing"] = True
    pf_kwargs["group_by"] = True
pf = vbt.Portfolio.from_signals(data_multi, **pf_kwargs)
```

Without `cash_sharing`, each column runs its own cash pool (isolated
backtest per pair). With `cash_sharing=True, group_by=True`, all
columns share one pool — the correct semantics for a multi-asset
basket or a multi-strategy portfolio where margin is fungible.

See also the [cash_sharing audit](../plans/bubbly-wondering-storm.md) in the
session plan file for the full context.

### 7. Parameter sweeps — `vbt.Param` + `@vbt.parameterized`

```python
# GOOD (from ou_mean_reversion.py::pipeline_nb)
@vbt.parameterized(
    merge_func="concat",
    execute_kwargs=make_execute_kwargs("OU MR grid"),
)
def pipeline_nb(data, bb_window, bb_alpha, sl_stop=0.005, tp_stop=0.006, ...) -> float:
    pf, _ = pipeline(data, bb_window=bb_window, bb_alpha=bb_alpha, ...)
    return float(pf.sharpe_ratio)

# Call with vbt.Param sequences — broadcasts the grid automatically
result = pipeline_nb(
    data,
    bb_window=vbt.Param([60, 80, 100, 120]),
    bb_alpha=vbt.Param([3.0, 4.0, 5.0]),
)
```

For RAM-bounded sweeps (when the grid is too big to hold in memory),
add `@vbt.chunked` on top — see `pipeline_utils.py` for the chunking
helper used in `mr_macro` grids.

### 8. Metrics — always through the Portfolio object

```python
# GOOD
sr   = float(pf.sharpe_ratio)
ar   = float(pf.annualized_return)
mdd  = float(pf.max_drawdown)
stats = pf.stats()   # full dataframe

# BAD — hand-rolled Sharpe from returns
# sr = returns.mean() / returns.std() * np.sqrt(252)
```

`pf.sharpe_ratio` respects `cash_sharing` and grouping. A hand-rolled
Sharpe on per-column returns will give wrong answers for a grouped
portfolio.

---

## Reference files — idiomatic exemplars

When in doubt, copy the style of these three files. They have
**0 raw pandas/numpy findings** in the audit:

| File | Why it's exemplary |
|------|---------------------|
| `src/strategies/mr_turbo.py` | Uses `vbt.VWAP`, `vbt.BBANDS`, `Portfolio.from_signals` with 4 native stops, `@vbt.parameterized` for the grid, and the investigation/grid-search/CV trio pattern. |
| `src/strategies/ou_mean_reversion.py` | Same as mr_turbo + per-bar leverage array. The Numba kernel for vol-target leverage is a legitimate custom because VBT doesn't expose a direct minute→daily rolling-std helper. |
| `src/strategies/rsi_daily.py` | Uses `vbt.RSI.run`, `.vbt.resample_apply`, native stops. |

---

## Gaps — things VBT does NOT provide (keep raw)

Document these in PRs when you encounter them so we don't waste time
searching:

1. **Minute→daily rolling vol** with specific session boundary. Used
   in `ou_mean_reversion.py` via `utils.compute_daily_rolling_volatility_nb`.
   VBT has `resample_apply` + `rolling_std` but the naive composition
   changes the warmup window semantics — refactor only with a
   bit-equivalence test.

2. **K-overlapping sub-portfolios (Jegadeesh-Titman)**. The
   `sub_portfolio_weights_nb` kernel in `composite_fx_alpha.py:128-149`
   implements K=5 rotating sub-portfolios. No direct VBT equivalent.

3. **State-machine drawdown hysteresis**. `drawdown_control_nb` in
   `composite_fx_alpha.py:96-125` is a 3-state machine (NORMAL →
   REDUCED → FLAT) with soft/hard/recovery thresholds. Keep as-is.

4. **Risk-parity with NaN fallback to equal weights**. Implemented in
   `combined_portfolio.py:_compute_weights_ts` L214-230. Contains a
   documented bug fix (weights summing to 1.40 on 2018-03-16 without
   the fallback) that's stricter than `vbt.PortfolioOptimizer`'s
   generic risk-parity objective.

5. **`.pct_change()` with `fill_value=0.0` in one call**. VBT's
   accessor doesn't accept a `fill_value` argument, so
   `.vbt.pct_change().fillna(0.0)` remains the correct idiom.

---

## Meta — patterns the audit scanner can miss

The AST-based scan in `scripts/audit_vbt_native.py` catches high-confidence
anti-patterns but misses:

- Semantic intent behind generic calls (e.g., `reindex` for column
  alignment vs cross-frequency alignment — the scanner flags both,
  only the latter is a real finding).
- Intentional `shift(-n)` used for PFO forward-semantics (not causal).
- `np.broadcast_to` as zero-copy const-DataFrame construction (correct)
  vs as explicit mask-expand (cosmetic).

For these, trust the manual review in
`reports/audits/vbt_native_audit.md` over the CSV finding list.

---

## Where to learn more

- **MCP search** : `mcp__vectorbtpro__search <topic>` inside Claude
  Code — live access to VBT Pro docs, forum threads, and source.
- **Upstream docs** : https://vectorbt.pro/pvt_74d2a4ff/documentation/
- **Reference files** : the 3 exemplars listed above.
- **Audit report** : `reports/audits/vbt_native_audit.md` for the
  current state of raw pandas/numpy usage in `src/strategies/`.
