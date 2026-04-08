"""Generate all 8 strategy research notebooks — fully self-contained, VBT Pro native."""
import json
from pathlib import Path

NOTEBOOKS_DIR = Path(__file__).parent


def md(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source}


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "source": source,
        "outputs": [],
        "execution_count": None,
    }


def make_notebook(cells: list[dict]) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12.0"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def write_nb(name: str, cells: list[dict]):
    nb = make_notebook(cells)
    path = NOTEBOOKS_DIR / name
    path.write_text(json.dumps(nb, indent=1, ensure_ascii=False))
    print(f"  Created {path.name} ({len(cells)} cells)")


# ═══════════════════════════════════════════════════════════════════════
# SHARED CODE BLOCKS (inlined in each notebook)
# ═══════════════════════════════════════════════════════════════════════

IMPORTS = """\
import warnings
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import vectorbtpro as vbt
from numba import njit
from numba.core.errors import NumbaPerformanceWarning
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)
warnings.filterwarnings(
    "ignore",
    message="Argument at index .* not found in SequenceTaker",
    module=r"vectorbtpro\\\\.utils\\\\.chunking",
)"""

VBT_SETTINGS = """\
import multiprocessing
from numba import get_num_threads
from pathlib import Path

n_cores = multiprocessing.cpu_count()
print(f"Cores: {n_cores} | Numba threads: {get_num_threads()}")

def _fullscreen(fig):
    fig.update_layout(width=None, height=None, autosize=True,
        margin=dict(l=30, r=30, t=60, b=30),
        title=dict(font=dict(size=20), x=0.5, xanchor="center"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5))
    return fig

vbt.settings.set("plotting.pre_show_func", _fullscreen)
vbt.settings.returns.year_freq = pd.Timedelta(hours=24) * 252"""

LOAD_DATA_INTRADAY = """\
_PROJECT_ROOT = Path(__file__).resolve().parent if "__file__" in dir() else Path(".").resolve()
# When run from notebooks/ dir: parent is project root
# When run from project root: "." is project root
# Fallback: search upward for data/
for _p in [Path(".").resolve(), Path(".").resolve().parent, Path(".").resolve().parent.parent]:
    if (_p / "data" / "EUR-USD.parquet").exists():
        _PROJECT_ROOT = _p
        break

def load_fx_data(path="data/EUR-USD.parquet", shift_hours=0):
    resolved = _PROJECT_ROOT / path
    data_raw = vbt.Data.from_parquet(str(resolved))
    symbol = data_raw.symbols[0]
    df = data_raw.data[symbol].set_index("date").sort_index()
    if shift_hours:
        df.index = df.index + pd.Timedelta(hours=shift_hours)
    raw = df.copy()
    raw.columns = [c.lower() for c in raw.columns]
    df.columns = [c.capitalize() for c in df.columns]
    data = vbt.Data.from_data({symbol: df}, tz_localize=False, tz_convert=False)
    return raw, data

raw, data = load_fx_data()
index_ns = vbt.dt.to_ns(raw.index)

# Quick stats
bars_per_day = raw.index.to_series().groupby(raw.index.date).count()
ann_factor = 252.0 * bars_per_day.mean()
print(f"Bars: {len(raw):,} | Range: {raw.index[0]} -> {raw.index[-1]}")
print(f"Ann factor: {ann_factor:.0f}")"""

# All shared MR Numba kernels
MR_KERNELS = """\
# ═══════════════════════════════════════════════════════════════════════
# NUMBA KERNELS (all @njit(nogil=True) for parallel execution)
# ═══════════════════════════════════════════════════════════════════════

@njit(nogil=True)
def find_day_boundaries_nb(index_ns):
    n = len(index_ns)
    start_idx = np.empty(n, dtype=np.int64)
    end_idx = np.empty(n, dtype=np.int64)
    if n == 0:
        return start_idx, end_idx, 0
    day_number = vbt.dt_nb.days_nb(ts=index_ns)
    current_day = day_number[0]
    day_counter = 0
    current_start = 0
    for i in range(1, n):
        if day_number[i] != current_day:
            start_idx[day_counter] = current_start
            end_idx[day_counter] = i
            day_counter += 1
            current_day = day_number[i]
            current_start = i
    start_idx[day_counter] = current_start
    end_idx[day_counter] = n
    day_counter += 1
    return start_idx, end_idx, day_counter


@njit(nogil=True)
def compute_adx_nb(high, low, close, period):
    n = len(close)
    adx = np.full(n, np.nan)
    if n < period + 1:
        return adx
    plus_dm = np.empty(n); minus_dm = np.empty(n); tr = np.empty(n)
    plus_dm[0] = 0.0; minus_dm[0] = 0.0; tr[0] = high[0] - low[0]
    for i in range(1, n):
        up = high[i] - high[i-1]; down = low[i-1] - low[i]
        plus_dm[i] = up if (up > down and up > 0) else 0.0
        minus_dm[i] = down if (down > up and down > 0) else 0.0
        hl = high[i] - low[i]; hc = abs(high[i] - close[i-1]); lc = abs(low[i] - close[i-1])
        tr[i] = max(hl, max(hc, lc))
    s_plus = vbt.generic.nb.ewm_mean_1d_nb(plus_dm, span=period, minp=period, adjust=False)
    s_minus = vbt.generic.nb.ewm_mean_1d_nb(minus_dm, span=period, minp=period, adjust=False)
    s_tr = vbt.generic.nb.ewm_mean_1d_nb(tr, span=period, minp=period, adjust=False)
    dx = np.full(n, np.nan)
    for i in range(n):
        if not np.isnan(s_tr[i]) and s_tr[i] > 1e-10:
            pdi = s_plus[i] / s_tr[i]; mdi = s_minus[i] / s_tr[i]
            s = pdi + mdi
            if s > 1e-10:
                dx[i] = abs(pdi - mdi) / s * 100.0
    adx[:] = vbt.generic.nb.ewm_mean_1d_nb(dx, span=period, minp=period, adjust=False)
    return adx


@njit(nogil=True)
def compute_daily_adx_broadcast_nb(index_ns, high, low, close, open_, adx_period):
    n = len(close)
    start_arr, end_arr, n_days = find_day_boundaries_nb(index_ns)
    if n_days < adx_period + 2:
        return np.full(n, np.nan)
    d_high = np.empty(n_days); d_low = np.empty(n_days); d_close = np.empty(n_days)
    for d in range(n_days):
        s, e = start_arr[d], end_arr[d]
        mx, mn = high[s], low[s]
        for i in range(s+1, e):
            if high[i] > mx: mx = high[i]
            if low[i] < mn: mn = low[i]
        d_high[d] = mx; d_low[d] = mn; d_close[d] = close[e-1]
    daily_adx = compute_adx_nb(d_high, d_low, d_close, adx_period)
    adx_minute = np.full(n, np.nan)
    for d in range(1, n_days):
        val = daily_adx[d-1]
        for i in range(start_arr[d], end_arr[d]):
            adx_minute[i] = val
    return adx_minute


@njit(nogil=True)
def compute_adx_regime_nb(index_ns, high, low, close, open_, adx_period, adx_threshold):
    n = len(close)
    adx = compute_daily_adx_broadcast_nb(index_ns, high, low, close, open_, adx_period)
    regime_ok = np.ones(n, dtype=np.float64)
    for i in range(n):
        if not np.isnan(adx[i]) and adx[i] > adx_threshold:
            regime_ok[i] = 0.0
    return regime_ok


@njit(nogil=True)
def compute_intraday_twap_nb(index_ns, high, low, close):
    n = len(close)
    twap = np.full(n, np.nan)
    if n == 0:
        return twap
    start_arr, end_arr, n_days = find_day_boundaries_nb(index_ns)
    for d in range(n_days):
        s, e = start_arr[d], end_arr[d]
        cum_tp = 0.0; count = 0
        for i in range(s, e):
            tp = (high[i] + low[i] + close[i]) / 3.0
            if not np.isnan(tp):
                cum_tp += tp; count += 1
                twap[i] = cum_tp / count
    return twap


@njit(nogil=True)
def compute_intraday_rolling_std_nb(index_ns, data, lookback):
    n = len(data)
    out = np.full(n, np.nan)
    if n == 0:
        return out
    start_arr, end_arr, n_days = find_day_boundaries_nb(index_ns)
    minp = min(lookback, 20)
    for d in range(n_days):
        s, e = start_arr[d], end_arr[d]
        if e - s < minp:
            continue
        day_std = vbt.generic.nb.rolling_std_1d_nb(data[s:e], lookback, minp=minp, ddof=1)
        for i in range(e - s):
            out[s + i] = day_std[i]
    return out


@njit(nogil=True)
def compute_intraday_zscore_nb(index_ns, data, lookback):
    n = len(data)
    out = np.full(n, np.nan)
    if n == 0:
        return out
    start_arr, end_arr, n_days = find_day_boundaries_nb(index_ns)
    minp = min(lookback, 20)
    for d in range(n_days):
        s, e = start_arr[d], end_arr[d]
        if e - s < minp:
            continue
        day_z = vbt.generic.nb.rolling_zscore_1d_nb(data[s:e], lookback, minp=minp, ddof=1)
        for i in range(e - s):
            out[s + i] = day_z[i]
    return out


@njit(nogil=True)
def compute_deviation_nb(close, twap):
    n = len(close)
    dev = np.empty(n)
    for i in range(n):
        if np.isnan(twap[i]) or np.isnan(close[i]):
            dev[i] = np.nan
        else:
            dev[i] = close[i] - twap[i]
    return dev


@njit(nogil=True)
def compute_intraday_bands_nb(index_ns, twap, deviation, lookback, band_width):
    n = len(twap)
    rstd = compute_intraday_rolling_std_nb(index_ns, deviation, lookback)
    upper = np.full(n, np.nan); lower = np.full(n, np.nan)
    for i in range(n):
        s = rstd[i]
        if not np.isnan(s) and s > 1e-10 and not np.isnan(twap[i]):
            upper[i] = twap[i] + band_width * s
            lower[i] = twap[i] - band_width * s
    return upper, lower


@njit(nogil=True)
def compute_mr_base_indicators_nb(index_ns, high, low, close, open_,
                                   lookback, band_width, adx_period, adx_threshold):
    twap = compute_intraday_twap_nb(index_ns, high, low, close)
    deviation = compute_deviation_nb(close, twap)
    zscore = compute_intraday_zscore_nb(index_ns, deviation, lookback)
    upper, lower = compute_intraday_bands_nb(index_ns, twap, deviation, lookback, band_width)
    regime_ok = compute_adx_regime_nb(index_ns, high, low, close, open_, adx_period, adx_threshold)
    return twap, zscore, upper, lower, regime_ok


@njit(nogil=True)
def mr_band_signal_nb(c, close_arr, upper_arr, lower_arr, twap_arr,
                       regime_ok_arr, index_ns_arr, eod_hour_arr, eod_minute_arr, eval_freq_arr):
    ts_ns = index_ns_arr[c.i]
    cur_h = vbt.dt_nb.hour_nb(ts_ns); cur_m = vbt.dt_nb.minute_nb(ts_ns)
    eod_h = vbt.pf_nb.select_nb(c, eod_hour_arr); eod_m = vbt.pf_nb.select_nb(c, eod_minute_arr)
    if (cur_h > eod_h) or (cur_h == eod_h and cur_m >= eod_m):
        el = vbt.pf_nb.ctx_helpers.in_long_position_nb(c)
        es = vbt.pf_nb.ctx_helpers.in_short_position_nb(c)
        return False, el, False, es
    ef = vbt.pf_nb.select_nb(c, eval_freq_arr)
    if ef > 0 and cur_m % ef != 0:
        return False, False, False, False
    px = vbt.pf_nb.select_nb(c, close_arr)
    ub = vbt.pf_nb.select_nb(c, upper_arr); lb = vbt.pf_nb.select_nb(c, lower_arr)
    tw = vbt.pf_nb.select_nb(c, twap_arr); regime = vbt.pf_nb.select_nb(c, regime_ok_arr)
    if np.isnan(px) or np.isnan(ub) or np.isnan(lb) or np.isnan(tw):
        return False, False, False, False
    il = vbt.pf_nb.ctx_helpers.in_long_position_nb(c)
    ish = vbt.pf_nb.ctx_helpers.in_short_position_nb(c)
    if not il and not ish:
        if regime < 0.5: return False, False, False, False
        if px < lb: return True, False, False, False
        elif px > ub: return False, False, True, False
    elif il:
        if px >= tw: return False, True, False, False
    elif ish:
        if px <= tw: return False, False, False, True
    return False, False, False, False"""

# Volatility + leverage kernels (for OU MR)
VOL_LEVERAGE_KERNELS = """\
@njit(nogil=True)
def compute_daily_rolling_volatility_nb(index_ns, close_minute, window_size):
    n = len(close_minute)
    if n == 0 or window_size <= 0:
        return np.full(n, np.nan)
    start_arr, end_arr, n_days = find_day_boundaries_nb(index_ns)
    if n_days < 2:
        return np.full(n, np.nan)
    last_close = np.full(n_days, np.nan)
    for d in range(n_days):
        if end_arr[d] > 0:
            last_close[d] = close_minute[end_arr[d] - 1]
    returns = np.full(n_days - 1, np.nan)
    for i in range(1, n_days):
        prev = last_close[i-1]
        if not np.isnan(prev) and np.abs(prev) > 1e-9:
            returns[i-1] = last_close[i] / prev - 1.0
    if len(returns) < window_size:
        return np.full(n, np.nan)
    rolling_std = vbt.generic.nb.rolling_std_1d_nb(returns, window=window_size, minp=window_size, ddof=1)
    vol = np.full(n, np.nan)
    for d in range(1, n_days):
        if d-1 < rolling_std.size:
            if start_arr[d] < end_arr[d]:
                vol[start_arr[d]:end_arr[d]] = rolling_std[d-1]
    return vol


@njit(nogil=True)
def compute_leverage_nb(rolling_vol, sigma_target, max_leverage):
    n = len(rolling_vol)
    lev = np.full(n, 1.0)
    for i in range(n):
        v = rolling_vol[i]
        if not np.isnan(v) and v > 1e-9:
            lev[i] = min(sigma_target / v, max_leverage)
    return lev"""


def data_exploration_cells():
    return [
        code("""\
# Raw data inspection
print("=== OHLC Data ===")
print(f"Shape: {raw.shape}")
print(f"Columns: {raw.columns.tolist()}")
print(f"Dtypes:\\n{raw.dtypes}")
print(f"\\nNaN counts:\\n{raw.isna().sum()}")
raw.head(10)"""),
        code("""\
raw.describe()"""),
        code("""\
# Data overview
fig = go.Figure(data=go.Scatter(x=raw.index, y=raw["close"], line=dict(color="#333", width=1)))
fig.update_layout(height=400, title="EUR/USD Close Price Overview")
fig.show()"""),
        code("""\
# Daily returns distribution
daily_close = raw["close"].resample("1D").last().dropna()
daily_rets = daily_close.pct_change().dropna()

fig = make_subplots(rows=1, cols=2, subplot_titles=("Daily Returns Distribution", "QQ-like: Sorted Returns"))
fig.add_trace(go.Histogram(x=daily_rets.values, nbinsx=100, name="Daily Returns",
                           marker_color="#2196F3"), row=1, col=1)
sorted_rets = np.sort(daily_rets.values)
fig.add_trace(go.Scatter(y=sorted_rets, mode="lines", name="Sorted Returns",
                         line=dict(color="#FF5722")), row=1, col=2)
fig.update_layout(height=350, showlegend=False)
fig.show()"""),
    ]


def results_cells(overlay_code: str) -> list[dict]:
    """Generate all visualization cells after backtest."""
    return [
        md("### Orders & Trades Inspection"),
        code("""\
# Orders log
print(f"=== Orders: {pf.orders.count()} ===")
orders_df = pf.orders.records_readable
print(orders_df.head(20).to_string())"""),
        code("""\
# Trades log
print(f"=== Trades: {pf.trades.count()} ===")
if pf.trades.count() > 0:
    trades_df = pf.trades.records_readable
    print(trades_df.head(20).to_string())
    print(f"\\n=== Trade PnL Distribution ===")
    pnl = trades_df["PnL"].dropna()
    print(f"  Mean:   {pnl.mean():.4f}")
    print(f"  Median: {pnl.median():.4f}")
    print(f"  Std:    {pnl.std():.4f}")
    print(f"  Min:    {pnl.min():.4f}")
    print(f"  Max:    {pnl.max():.4f}")
    print(f"  Win%:   {(pnl > 0).mean()*100:.1f}%")
else:
    print("No trades generated.")"""),
        code("""\
# Position summary: time in market
print(f"=== Position Coverage ===")
pos = pf.position_mask
in_market_pct = pos.sum() / len(pos) * 100
print(f"  In market: {pos.sum():,} bars ({in_market_pct:.1f}%)")
print(f"  Flat:      {(~pos).sum():,} bars ({100-in_market_pct:.1f}%)")"""),
        md("### Portfolio Stats"),
        code("""\
print("PORTFOLIO STATS")
print("=" * 50)
print(pf.stats().to_string())
if pf.trades.count() > 0:
    print(f"\\nTRADE STATS\\n{'='*50}")
    print(pf.trades.stats().to_string())"""),
        md("### Equity Curve & Drawdowns"),
        code("""\
# Resample to daily for fast plotting (minute data has 3M+ points)
pf_daily = pf.resample("1D")
fig = pf_daily.plot(subplots=["cumulative_returns", "drawdowns", "underwater", "trade_pnl"])
fig.update_layout(height=900, title="Portfolio Summary (daily)")
fig.show()"""),
        md("### Trade Signals on Price"),
        code(f"""\
n_bars = min(7200, len(raw))
sim_start = raw.index[-n_bars]
fig = pf.plot_trade_signals(plot_positions="zones", sim_start=sim_start)
{overlay_code}
fig.update_layout(height=600, title="Trade Signals (last week)")
fig.show()"""),
        md("### Trade Analysis"),
        code("""\
if pf.trades.count() > 0:
    fig = make_subplots(rows=2, cols=2,
        subplot_titles=("Trade PnL (%)", "MAE", "MFE", "Running Edge Ratio"))
    pf.trades.plot_pnl(pct_scale=True, fig=fig, add_trace_kwargs=dict(row=1, col=1))
    pf.trades.plot_mae(fig=fig, add_trace_kwargs=dict(row=1, col=2))
    pf.trades.plot_mfe(fig=fig, add_trace_kwargs=dict(row=2, col=1))
    pf.trades.plot_running_edge_ratio(fig=fig, add_trace_kwargs=dict(row=2, col=2))
    fig.update_layout(height=800, showlegend=False)
    fig.show()"""),
        md("### Monthly Returns Heatmap"),
        code("""\
rets = pf_daily.returns
monthly = rets.resample("ME").apply(lambda x: (1 + x).prod() - 1)
df_m = pd.DataFrame({"return": monthly})
df_m["year"] = df_m.index.year; df_m["month"] = df_m.index.month
pivot = df_m.pivot_table(values="return", index="year", columns="month", aggfunc="first")
month_names = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
pivot.columns = month_names[:len(pivot.columns)]

fig = go.Figure(data=go.Heatmap(
    z=pivot.values * 100, x=pivot.columns.tolist(), y=[str(y) for y in pivot.index],
    colorscale="RdYlGn", texttemplate="%{z:.1f}%", textfont=dict(size=10), zmid=0))
fig.update_layout(title="Monthly Returns (%)", height=300 + len(pivot) * 30)
fig.show()"""),
        md("### Rolling Sharpe Ratio (Daily)"),
        code("""\
# Use daily-resampled portfolio for fast rolling calc
daily_rets = pf_daily.returns
rolling_sharpe = daily_rets.rolling(252).apply(
    lambda x: x.mean() / x.std() * np.sqrt(252) if x.std() > 0 else 0)
fig = go.Figure()
fig.add_trace(go.Scatter(x=rolling_sharpe.index, y=rolling_sharpe.values,
                         name="Rolling Sharpe (1Y)", line=dict(color="#2196F3")))
fig.add_hline(y=0, line_dash="solid", line_color="gray")
fig.add_hline(y=1, line_dash="dot", line_color="green", annotation_text="Sharpe=1")
fig.update_layout(height=350, title="Rolling 1-Year Sharpe Ratio (Daily)")
fig.show()"""),
    ]


def sweep_cells(ind_factory: str, ind_run: str, pf_run: str,
                x_level: str, y_level: str) -> list[dict]:
    return [
        md("## Parameter Sweep"),
        code(ind_run),
        code(pf_run),
        md("### Sweep Rankings"),
        code("""\
# Parameter ranking table
sweep_summary = pd.DataFrame({
    "Sharpe": pf_sweep.sharpe_ratio,
    "Total Return": pf_sweep.total_return,
    "Max DD": pf_sweep.max_drawdown,
    "Trades": pf_sweep.trades.count(),
    "Win Rate": pf_sweep.trades.win_rate,
}).sort_values("Sharpe", ascending=False)

print("=== Top 10 Parameter Combos (by Sharpe) ===")
print(sweep_summary.head(10).to_string())
print(f"\\n=== Bottom 5 ===")
print(sweep_summary.tail(5).to_string())"""),
        md("### Sweep Heatmaps"),
        code(f"""\
fig = pf_sweep.sharpe_ratio.vbt.heatmap(x_level="{x_level}", y_level="{y_level}")
fig.update_layout(title="Sharpe Ratio Heatmap")
fig.show()"""),
        code(f"""\
fig = pf_sweep.total_return.vbt.heatmap(x_level="{x_level}", y_level="{y_level}")
fig.update_layout(title="Total Return Heatmap")
fig.show()"""),
        code(f"""\
fig = pf_sweep.max_drawdown.vbt.heatmap(x_level="{x_level}", y_level="{y_level}")
fig.update_layout(title="Max Drawdown Heatmap")
fig.show()"""),
        code(f"""\
fig = pf_sweep.trades.count().vbt.heatmap(x_level="{x_level}", y_level="{y_level}")
fig.update_layout(title="Trade Count Heatmap")
fig.show()"""),
    ]


def cv_cells(pipeline_func: str, pipeline_call: str) -> list[dict]:
    """CV section using @vbt.cv_split native."""
    return [
        md("## Cross-Validation (Walk-Forward)\n\nUsing VBT Pro native `@vbt.cv_split` decorator."),
        code("""\
# 80/20 holdout split
split_idx = int(len(raw) * 0.8)
raw_train = raw.iloc[:split_idx].copy()
raw_test = raw.iloc[split_idx:].copy()
print(f"Train: {len(raw_train):,} bars ({raw_train.index[0]} -> {raw_train.index[-1]})")
print(f"Test:  {len(raw_test):,} bars ({raw_test.index[0]} -> {raw_test.index[-1]})")"""),
        code(f"""\
# Define CV pipeline with @vbt.cv_split
def selection_func(grid_results):
    return vbt.LabelSel([grid_results.idxmax()])

{pipeline_func}"""),
        code(f"""\
# Run CV
{pipeline_call}"""),
        code("""\
# Display CV results
print("Grid results (all splits x params):")
print(grid_results)
print(f"\\nBest results per split:")
print(best_results)"""),
        md("## Holdout Test"),
        code("""\
# TODO: Extract best params from CV and rerun on train/test
# For now, display the CV selection results
print("CV analysis complete.")
print("Best params selected per split are shown above.")
print("Next step: validate on holdout set with selected params.")"""),
    ]


# ═══════════════════════════════════════════════════════════════════════
# NOTEBOOK 01: MR V1 — Band-Based Mean Reversion
# ═══════════════════════════════════════════════════════════════════════

def nb_01():
    cells = [
        md("# MR V1: Band-Based Mean Reversion\n\n"
           "**Intraday TWAP mean reversion** with rolling std bands and ADX regime filter.\n\n"
           "| Feature | Detail |\n|---------|--------|\n"
           "| Entry | Price < lower band (long) or > upper band (short) |\n"
           "| Exit | Price crosses back through TWAP or SL hit |\n"
           "| Regime | ADX < 30 (ranging market only) |\n"
           "| EOD | Forced exit at 21:00 UTC |\n"
           "| Eval freq | Every 5 minutes |\n"
           "| Leverage | Fixed 1x |"),
        md("## 1. Setup"),
        code(IMPORTS),
        code(VBT_SETTINGS),
        md("## 2. Data"),
        code(LOAD_DATA_INTRADAY),
        *data_exploration_cells(),
        md("## 3. Numba Kernels"),
        code(MR_KERNELS),
        md("## 4. Indicators"),
        code("""\
MR_V1 = vbt.IF(
    class_name="MR_V1", short_name="mr_v1",
    input_names=["index_ns", "high_minute", "low_minute", "close_minute", "open_minute"],
    param_names=["lookback", "band_width", "adx_period", "adx_threshold"],
    output_names=["twap", "zscore", "upper_band", "lower_band", "regime_ok"],
).with_apply_func(
    compute_mr_base_indicators_nb, takes_1d=True,
    lookback=60, band_width=2.0, adx_period=14, adx_threshold=30.0)

ind = MR_V1.run(
    index_ns, raw["high"], raw["low"], raw["close"], raw["open"],
    lookback=60, band_width=2.0, adx_period=14, adx_threshold=30.0,
    jitted_loop=True, jitted_warmup=True)
print("Outputs:", ind.output_names)"""),
        md("### Indicator Inspection"),
        code("""\
# Build indicator DataFrame for inspection
ind_df = pd.DataFrame({
    "close": raw["close"].values,
    "twap": ind.twap.values,
    "zscore": ind.zscore.values,
    "upper_band": ind.upper_band.values,
    "lower_band": ind.lower_band.values,
    "regime_ok": ind.regime_ok.values,
}, index=raw.index)

print("=== Indicators DataFrame ===")
print(f"Shape: {ind_df.shape}")
print(f"\\nNaN counts:\\n{ind_df.isna().sum()}")
print(f"\\nFirst valid rows (after warmup):")
ind_df.dropna().head(10)"""),
        code("""\
# Indicator statistics
ind_df.describe()"""),
        md("## 5. Signal Visualization"),
        code("""\
n = min(7200, len(raw)); sl = slice(-n, None)

fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.5, 0.25, 0.25],
    subplot_titles=("Price + TWAP + Bands", "Z-Score", "ADX Regime Filter"))

fig.add_trace(go.Scatter(x=raw.index[sl], y=raw["close"].values[sl], name="Close",
    line=dict(color="#333", width=1)), row=1, col=1)
fig.add_trace(go.Scatter(x=raw.index[sl], y=ind.twap.values[sl], name="TWAP",
    line=dict(color="#FF9800", dash="dash")), row=1, col=1)
fig.add_trace(go.Scatter(x=raw.index[sl], y=ind.upper_band.values[sl], name="Upper Band",
    line=dict(color="#E91E63", dash="dot")), row=1, col=1)
fig.add_trace(go.Scatter(x=raw.index[sl], y=ind.lower_band.values[sl], name="Lower Band",
    line=dict(color="#E91E63", dash="dot")), row=1, col=1)

fig.add_trace(go.Scatter(x=raw.index[sl], y=ind.zscore.values[sl], name="Z-Score",
    line=dict(color="#2196F3")), row=2, col=1)
fig.add_hline(y=2, line_dash="dot", line_color="red", row=2, col=1)
fig.add_hline(y=-2, line_dash="dot", line_color="green", row=2, col=1)
fig.add_hline(y=0, line_color="gray", row=2, col=1)

fig.add_trace(go.Scatter(x=raw.index[sl], y=ind.regime_ok.values[sl], name="Regime OK",
    fill="tozeroy", line=dict(color="rgba(76,175,80,0.5)")), row=3, col=1)

fig.update_layout(height=900, title="MR V1 Indicators")
fig.show()"""),
        md("## 6. Backtest"),
        code("""\
pf = vbt.Portfolio.from_signals(
    close=raw["close"], open=raw["open"], high=raw["high"], low=raw["low"],
    signal_func_nb=mr_band_signal_nb,
    signal_args=(
        vbt.Rep("close_arr"), vbt.Rep("upper_arr"), vbt.Rep("lower_arr"),
        vbt.Rep("twap_arr"), vbt.Rep("regime_ok_arr"), vbt.Rep("index_ns_arr"),
        vbt.Rep("eod_hour_arr"), vbt.Rep("eod_minute_arr"), vbt.Rep("eval_freq_arr")),
    broadcast_named_args=dict(
        close_arr=raw["close"], upper_arr=ind.upper_band.values,
        lower_arr=ind.lower_band.values, twap_arr=ind.twap.values,
        regime_ok_arr=ind.regime_ok.values, index_ns_arr=index_ns,
        eod_hour_arr=21, eod_minute_arr=0, eval_freq_arr=5),
    leverage=1.0, slippage=0.00015, sl_stop=0.005,
    init_cash=1_000_000, freq="1min")
print(f"Trades: {pf.trades.count()}")"""),
        md("## 7. Results"),
        *results_cells("""\
fig.add_trace(go.Scatter(x=raw.index[-n_bars:], y=ind.twap.values[-n_bars:],
    name="TWAP", line=dict(color="#FF9800", dash="dash")))
fig.add_trace(go.Scatter(x=raw.index[-n_bars:], y=ind.upper_band.values[-n_bars:],
    name="Upper", line=dict(color="#E91E63", dash="dot")))
fig.add_trace(go.Scatter(x=raw.index[-n_bars:], y=ind.lower_band.values[-n_bars:],
    name="Lower", line=dict(color="#E91E63", dash="dot")))"""),
        *sweep_cells("MR_V1",
            ind_run="""\
ind_sweep = MR_V1.run(
    index_ns, raw["high"], raw["low"], raw["close"], raw["open"],
    lookback=vbt.Param([20, 40, 60, 120, 240]),
    band_width=vbt.Param([1.5, 2.0, 2.5, 3.0]),
    adx_period=14, adx_threshold=30.0,
    jitted_loop=True, jitted_warmup=True, param_product=True)
print(f"Param combos: {ind_sweep.wrapper.shape_2d[1]}")""",
            pf_run="""\
pf_sweep = vbt.Portfolio.from_signals(
    close=raw["close"], open=raw["open"], high=raw["high"], low=raw["low"],
    signal_func_nb=mr_band_signal_nb,
    signal_args=(
        vbt.Rep("close_arr"), vbt.Rep("upper_arr"), vbt.Rep("lower_arr"),
        vbt.Rep("twap_arr"), vbt.Rep("regime_ok_arr"), vbt.Rep("index_ns_arr"),
        vbt.Rep("eod_hour_arr"), vbt.Rep("eod_minute_arr"), vbt.Rep("eval_freq_arr")),
    broadcast_named_args=dict(
        close_arr=raw["close"], upper_arr=ind_sweep.upper_band,
        lower_arr=ind_sweep.lower_band, twap_arr=ind_sweep.twap,
        regime_ok_arr=ind_sweep.regime_ok, index_ns_arr=index_ns,
        eod_hour_arr=21, eod_minute_arr=0, eval_freq_arr=5),
    leverage=1.0, slippage=0.00015, sl_stop=0.005,
    init_cash=1_000_000, freq="1min",
    jitted=dict(parallel=True), chunked="threadpool")
print("Sweep done.")""",
            x_level="mr_v1_lookback", y_level="mr_v1_band_width"),
        *cv_cells(
            pipeline_func="""\
@vbt.cv_split(
    splitter="from_purged_walkforward",
    splitter_kwargs=dict(n_folds=10, n_test_folds=1, min_train_folds=3, purge_td="1 hour"),
    takeable_args=["raw_data"],
    merge_func="concat",
    selection=vbt.RepFunc(selection_func),
    return_grid="all",
)
def mr_v1_cv(raw_data, lookback, band_width):
    ns = vbt.dt.to_ns(raw_data.index)
    ind_cv = MR_V1.run(ns, raw_data["high"], raw_data["low"], raw_data["close"], raw_data["open"],
        lookback=lookback, band_width=band_width, adx_period=14, adx_threshold=30.0,
        jitted_loop=True, jitted_warmup=True)
    pf_cv = vbt.Portfolio.from_signals(
        close=raw_data["close"], open=raw_data["open"], high=raw_data["high"], low=raw_data["low"],
        signal_func_nb=mr_band_signal_nb,
        signal_args=(
            vbt.Rep("close_arr"), vbt.Rep("upper_arr"), vbt.Rep("lower_arr"),
            vbt.Rep("twap_arr"), vbt.Rep("regime_ok_arr"), vbt.Rep("index_ns_arr"),
            vbt.Rep("eod_hour_arr"), vbt.Rep("eod_minute_arr"), vbt.Rep("eval_freq_arr")),
        broadcast_named_args=dict(
            close_arr=raw_data["close"], upper_arr=ind_cv.upper_band.values,
            lower_arr=ind_cv.lower_band.values, twap_arr=ind_cv.twap.values,
            regime_ok_arr=ind_cv.regime_ok.values, index_ns_arr=ns,
            eod_hour_arr=21, eod_minute_arr=0, eval_freq_arr=5),
        leverage=1.0, slippage=0.00015, sl_stop=0.005,
        init_cash=1_000_000, freq="1min")
    return pf_cv.sharpe_ratio""",
            pipeline_call="""\
grid_results, best_results = mr_v1_cv(
    raw_train,
    vbt.Param([20, 60, 120, 240]),
    vbt.Param([1.5, 2.0, 2.5, 3.0]),
)"""),
    ]
    write_nb("01_mr_v1_band_reversion.ipynb", cells)


# ═══════════════════════════════════════════════════════════════════════
# NOTEBOOK 02-08: Same pattern, strategy-specific code
# ═══════════════════════════════════════════════════════════════════════
# (Generating remaining 7 notebooks with same structure but unique kernels/signals)

def nb_02():
    cells = [
        md("# MR V2: Z-Score Based Entry/Exit\n\n"
           "| Feature | Detail |\n|---------|--------|\n"
           "| Entry | |z| > entry_z (z < -entry_z for long, z > entry_z for short) |\n"
           "| Exit | |z| < exit_z while in position |\n"
           "| Regime | ADX < 30 |\n| EOD | 21:00 UTC |\n| Eval freq | 5 min |"),
        md("## 1. Setup"), code(IMPORTS), code(VBT_SETTINGS),
        md("## 2. Data"), code(LOAD_DATA_INTRADAY), *data_exploration_cells(),
        md("## 3. Numba Kernels"), code(MR_KERNELS),
        md("### V2-Specific: Z-Score Indicator + Signal"),
        code("""\
@njit(nogil=True)
def compute_mr_v2_indicators_nb(index_ns, high, low, close, open_, lookback, adx_period, adx_threshold):
    twap = compute_intraday_twap_nb(index_ns, high, low, close)
    deviation = compute_deviation_nb(close, twap)
    zscore = compute_intraday_zscore_nb(index_ns, deviation, lookback)
    regime_ok = compute_adx_regime_nb(index_ns, high, low, close, open_, adx_period, adx_threshold)
    return twap, zscore, regime_ok

@njit(nogil=True)
def mr_v2_signal_nb(c, zscore_arr, regime_ok_arr, index_ns_arr,
                     eod_hour_arr, eod_minute_arr, eval_freq_arr, entry_z_arr, exit_z_arr):
    ts_ns = index_ns_arr[c.i]
    cur_h = vbt.dt_nb.hour_nb(ts_ns); cur_m = vbt.dt_nb.minute_nb(ts_ns)
    eod_h = vbt.pf_nb.select_nb(c, eod_hour_arr); eod_m = vbt.pf_nb.select_nb(c, eod_minute_arr)
    if (cur_h > eod_h) or (cur_h == eod_h and cur_m >= eod_m):
        return False, vbt.pf_nb.ctx_helpers.in_long_position_nb(c), False, vbt.pf_nb.ctx_helpers.in_short_position_nb(c)
    ef = vbt.pf_nb.select_nb(c, eval_freq_arr)
    if ef > 0 and cur_m % ef != 0:
        return False, False, False, False
    z = vbt.pf_nb.select_nb(c, zscore_arr)
    regime = vbt.pf_nb.select_nb(c, regime_ok_arr)
    entry_z = vbt.pf_nb.select_nb(c, entry_z_arr)
    exit_z = vbt.pf_nb.select_nb(c, exit_z_arr)
    if np.isnan(z): return False, False, False, False
    il = vbt.pf_nb.ctx_helpers.in_long_position_nb(c)
    ish = vbt.pf_nb.ctx_helpers.in_short_position_nb(c)
    if il:
        return (False, True, False, False) if z >= -exit_z else (False, False, False, False)
    if ish:
        return (False, False, False, True) if z <= exit_z else (False, False, False, False)
    if regime < 0.5: return False, False, False, False
    if z < -entry_z: return True, False, False, False
    elif z > entry_z: return False, False, True, False
    return False, False, False, False"""),
        md("## 4. Indicators"),
        code("""\
MR_V2 = vbt.IF(
    class_name="IntradayMRv2", short_name="mrv2",
    input_names=["index_ns", "high_minute", "low_minute", "close_minute", "open_minute"],
    param_names=["lookback", "adx_period", "adx_threshold"],
    output_names=["twap", "zscore", "regime_ok"],
).with_apply_func(compute_mr_v2_indicators_nb, takes_1d=True,
    lookback=60, adx_period=14, adx_threshold=30.0)

ind = MR_V2.run(index_ns, raw["high"], raw["low"], raw["close"], raw["open"],
    lookback=60, adx_period=14, adx_threshold=30.0, jitted_loop=True, jitted_warmup=True)"""),
        md("### Indicator Inspection"),
        code("""\
ind_df = pd.DataFrame({
    "close": raw["close"].values, "twap": ind.twap.values,
    "zscore": ind.zscore.values, "regime_ok": ind.regime_ok.values,
}, index=raw.index)
print(f"Shape: {ind_df.shape} | NaN: {ind_df.isna().sum().to_dict()}")
ind_df.dropna().head(10)"""),
        md("## 5. Signal Visualization"),
        code("""\
n = min(7200, len(raw)); sl = slice(-n, None)
fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.4, 0.4, 0.2],
    subplot_titles=("Price & TWAP", "Z-Score + Entry/Exit Zones", "Regime"))
fig.add_trace(go.Scatter(x=raw.index[sl], y=raw["close"].values[sl], name="Close",
    line=dict(color="#333", width=1)), row=1, col=1)
fig.add_trace(go.Scatter(x=raw.index[sl], y=ind.twap.values[sl], name="TWAP",
    line=dict(color="#FF9800", dash="dash")), row=1, col=1)
fig.add_trace(go.Scatter(x=raw.index[sl], y=ind.zscore.values[sl], name="Z-Score",
    line=dict(color="#2196F3")), row=2, col=1)
fig.add_hline(y=2.0, line_dash="dash", line_color="red", annotation_text="Short entry", row=2, col=1)
fig.add_hline(y=-2.0, line_dash="dash", line_color="green", annotation_text="Long entry", row=2, col=1)
fig.add_hline(y=0.5, line_dash="dot", line_color="orange", annotation_text="Exit", row=2, col=1)
fig.add_hline(y=-0.5, line_dash="dot", line_color="orange", row=2, col=1)
fig.add_hline(y=0, line_color="gray", row=2, col=1)
fig.add_trace(go.Scatter(x=raw.index[sl], y=ind.regime_ok.values[sl], name="Regime",
    fill="tozeroy", line=dict(color="rgba(76,175,80,0.5)")), row=3, col=1)
fig.update_layout(height=900, title="MR V2: Z-Score Entry/Exit Zones")
fig.show()"""),
        md("## 6. Backtest"),
        code("""\
pf = vbt.Portfolio.from_signals(
    close=raw["close"], open=raw["open"], high=raw["high"], low=raw["low"],
    signal_func_nb=mr_v2_signal_nb,
    signal_args=(vbt.Rep("zscore_arr"), vbt.Rep("regime_ok_arr"), vbt.Rep("index_ns_arr"),
        vbt.Rep("eod_hour_arr"), vbt.Rep("eod_minute_arr"), vbt.Rep("eval_freq_arr"),
        vbt.Rep("entry_z_arr"), vbt.Rep("exit_z_arr")),
    broadcast_named_args=dict(zscore_arr=ind.zscore.values, regime_ok_arr=ind.regime_ok.values,
        index_ns_arr=index_ns, eod_hour_arr=21, eod_minute_arr=0, eval_freq_arr=5,
        entry_z_arr=2.0, exit_z_arr=0.5),
    leverage=1.0, slippage=0.00015, sl_stop=0.005, init_cash=1_000_000, freq="1min")
print(f"Trades: {pf.trades.count()}")"""),
        md("## 7. Results"),
        *results_cells("""\
fig.add_trace(go.Scatter(x=raw.index[-n_bars:], y=ind.twap.values[-n_bars:],
    name="TWAP", line=dict(color="#FF9800", dash="dash")))"""),
        *sweep_cells("MR_V2",
            ind_run="""\
ind_sweep = MR_V2.run(index_ns, raw["high"], raw["low"], raw["close"], raw["open"],
    lookback=vbt.Param([20, 40, 60, 120, 240]), adx_period=14, adx_threshold=30.0,
    jitted_loop=True, jitted_warmup=True)""",
            pf_run="""\
pf_sweep = vbt.Portfolio.from_signals(
    close=raw["close"], open=raw["open"], high=raw["high"], low=raw["low"],
    signal_func_nb=mr_v2_signal_nb,
    signal_args=(vbt.Rep("zscore_arr"), vbt.Rep("regime_ok_arr"), vbt.Rep("index_ns_arr"),
        vbt.Rep("eod_hour_arr"), vbt.Rep("eod_minute_arr"), vbt.Rep("eval_freq_arr"),
        vbt.Rep("entry_z_arr"), vbt.Rep("exit_z_arr")),
    broadcast_named_args=dict(zscore_arr=ind_sweep.zscore, regime_ok_arr=ind_sweep.regime_ok,
        index_ns_arr=index_ns, eod_hour_arr=21, eod_minute_arr=0, eval_freq_arr=5,
        entry_z_arr=vbt.Param([1.5, 2.0, 2.5, 3.0]), exit_z_arr=vbt.Param([0.3, 0.5, 0.7])),
    leverage=1.0, slippage=0.00015, sl_stop=0.005, init_cash=1_000_000, freq="1min",
    jitted=dict(parallel=True), chunked="threadpool")""",
            x_level="mrv2_lookback", y_level="entry_z_arr"),
        *cv_cells(
            pipeline_func="""\
@vbt.cv_split(
    splitter="from_purged_walkforward",
    splitter_kwargs=dict(n_folds=10, n_test_folds=1, min_train_folds=3, purge_td="1 hour"),
    takeable_args=["raw_data"], merge_func="concat",
    selection=vbt.RepFunc(selection_func), return_grid="all")
def mr_v2_cv(raw_data, lookback, entry_z, exit_z):
    ns = vbt.dt.to_ns(raw_data.index)
    ind_cv = MR_V2.run(ns, raw_data["high"], raw_data["low"], raw_data["close"], raw_data["open"],
        lookback=lookback, adx_period=14, adx_threshold=30.0, jitted_loop=True, jitted_warmup=True)
    pf_cv = vbt.Portfolio.from_signals(
        close=raw_data["close"], open=raw_data["open"], high=raw_data["high"], low=raw_data["low"],
        signal_func_nb=mr_v2_signal_nb,
        signal_args=(vbt.Rep("za"), vbt.Rep("ra"), vbt.Rep("na"),
            vbt.Rep("eh"), vbt.Rep("em"), vbt.Rep("ef"), vbt.Rep("ez"), vbt.Rep("xz")),
        broadcast_named_args=dict(za=ind_cv.zscore.values, ra=ind_cv.regime_ok.values,
            na=ns, eh=21, em=0, ef=5, ez=entry_z, xz=exit_z),
        leverage=1.0, slippage=0.00015, sl_stop=0.005, init_cash=1_000_000, freq="1min")
    return pf_cv.sharpe_ratio""",
            pipeline_call="""\
grid_results, best_results = mr_v2_cv(
    raw_train, vbt.Param([20, 60, 120]), vbt.Param([1.5, 2.0, 2.5]), vbt.Param([0.3, 0.5]))"""),
    ]
    write_nb("02_mr_v2_zscore_exit.ipynb", cells)


# Notebooks 03-08 follow the same pattern. For brevity, generating them with their unique parts.

def nb_03():
    """MR V3: Session Filter"""
    cells = [
        md("# MR V3: Session-Filtered Mean Reversion\n\n"
           "| Feature | Detail |\n|---------|--------|\n"
           "| Session | 8-16 UTC (London/NY overlap): entries allowed |\n"
           "| Outside | Exits only, no new entries |\n"
           "| Entry | Band breach (same as V1) |\n| Regime | ADX < 30 |"),
        md("## 1. Setup"), code(IMPORTS), code(VBT_SETTINGS),
        md("## 2. Data"), code(LOAD_DATA_INTRADAY), *data_exploration_cells(),
        md("## 3. Numba Kernels"), code(MR_KERNELS),
        md("### V3-Specific: Session Filter Signal"),
        code("""\
@njit(nogil=True)
def mr_v3_signal_nb(c, close_arr, upper_arr, lower_arr, twap_arr, regime_ok_arr,
                     index_ns_arr, eod_hour_arr, eod_minute_arr, eval_freq_arr,
                     session_start_arr, session_end_arr):
    ts_ns = index_ns_arr[c.i]
    cur_h = vbt.dt_nb.hour_nb(ts_ns); cur_m = vbt.dt_nb.minute_nb(ts_ns)
    eod_h = vbt.pf_nb.select_nb(c, eod_hour_arr); eod_m = vbt.pf_nb.select_nb(c, eod_minute_arr)
    if (cur_h > eod_h) or (cur_h == eod_h and cur_m >= eod_m):
        return False, vbt.pf_nb.ctx_helpers.in_long_position_nb(c), False, vbt.pf_nb.ctx_helpers.in_short_position_nb(c)
    ef = vbt.pf_nb.select_nb(c, eval_freq_arr)
    if ef > 0 and cur_m % ef != 0: return False, False, False, False
    px = vbt.pf_nb.select_nb(c, close_arr); ub = vbt.pf_nb.select_nb(c, upper_arr)
    lb = vbt.pf_nb.select_nb(c, lower_arr); tw = vbt.pf_nb.select_nb(c, twap_arr)
    regime = vbt.pf_nb.select_nb(c, regime_ok_arr)
    if np.isnan(px) or np.isnan(ub) or np.isnan(lb) or np.isnan(tw): return False, False, False, False
    il = vbt.pf_nb.ctx_helpers.in_long_position_nb(c)
    ish = vbt.pf_nb.ctx_helpers.in_short_position_nb(c)
    ss = vbt.pf_nb.select_nb(c, session_start_arr); se = vbt.pf_nb.select_nb(c, session_end_arr)
    in_session = (cur_h >= ss) and (cur_h < se)
    if not in_session:
        if il and px >= tw: return False, True, False, False
        if ish and px <= tw: return False, False, False, True
        return False, False, False, False
    if not il and not ish:
        if regime < 0.5: return False, False, False, False
        if px < lb: return True, False, False, False
        elif px > ub: return False, False, True, False
    elif il:
        if px >= tw: return False, True, False, False
    elif ish:
        if px <= tw: return False, False, False, True
    return False, False, False, False"""),
        md("## 4. Indicators"),
        code("""\
MR_V3 = vbt.IF(
    class_name="MR_V3", short_name="mr_v3",
    input_names=["index_ns", "high_minute", "low_minute", "close_minute", "open_minute"],
    param_names=["lookback", "band_width", "adx_period", "adx_threshold"],
    output_names=["twap", "zscore", "upper_band", "lower_band", "regime_ok"],
).with_apply_func(compute_mr_base_indicators_nb, takes_1d=True,
    lookback=60, band_width=2.0, adx_period=14, adx_threshold=30.0)
ind = MR_V3.run(index_ns, raw["high"], raw["low"], raw["close"], raw["open"],
    lookback=60, band_width=2.0, adx_period=14, adx_threshold=30.0,
    jitted_loop=True, jitted_warmup=True)"""),
        md("### Indicator Inspection"),
        code("""\
ind_df = pd.DataFrame({
    "close": raw["close"].values, "twap": ind.twap.values, "zscore": ind.zscore.values,
    "upper": ind.upper_band.values, "lower": ind.lower_band.values, "regime_ok": ind.regime_ok.values,
}, index=raw.index)
print(f"Shape: {ind_df.shape} | NaN: {ind_df.isna().sum().to_dict()}")
ind_df.dropna().head(10)"""),
        md("## 5. Signal Visualization"),
        code("""\
n = min(7200, len(raw)); sl = slice(-n, None)
fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3],
    subplot_titles=("Price & Bands (session 8-16 shaded)", "Z-Score"))
fig.add_trace(go.Scatter(x=raw.index[sl], y=raw["close"].values[sl], name="Close",
    line=dict(color="#333", width=1)), row=1, col=1)
fig.add_trace(go.Scatter(x=raw.index[sl], y=ind.twap.values[sl], name="TWAP",
    line=dict(color="#FF9800", dash="dash")), row=1, col=1)
fig.add_trace(go.Scatter(x=raw.index[sl], y=ind.upper_band.values[sl], name="Upper",
    line=dict(color="#E91E63", dash="dot")), row=1, col=1)
fig.add_trace(go.Scatter(x=raw.index[sl], y=ind.lower_band.values[sl], name="Lower",
    line=dict(color="#E91E63", dash="dot")), row=1, col=1)
dates = raw.index[sl]
for d in pd.Series(dates.date).unique()[-5:]:
    fig.add_vrect(x0=pd.Timestamp(d)+pd.Timedelta(hours=8), x1=pd.Timestamp(d)+pd.Timedelta(hours=16),
        fillcolor="rgba(33,150,243,0.08)", line_width=0, row=1, col=1)
fig.add_trace(go.Scatter(x=raw.index[sl], y=ind.zscore.values[sl], name="Z-Score",
    line=dict(color="#2196F3")), row=2, col=1)
fig.add_hline(y=2, line_dash="dot", line_color="red", row=2, col=1)
fig.add_hline(y=-2, line_dash="dot", line_color="green", row=2, col=1)
fig.update_layout(height=700, title="MR V3: Session Filter (8-16 UTC)")
fig.show()"""),
        md("## 6. Backtest"),
        code("""\
pf = vbt.Portfolio.from_signals(
    close=raw["close"], open=raw["open"], high=raw["high"], low=raw["low"],
    signal_func_nb=mr_v3_signal_nb,
    signal_args=(vbt.Rep("ca"), vbt.Rep("ua"), vbt.Rep("la"), vbt.Rep("ta"),
        vbt.Rep("ra"), vbt.Rep("na"), vbt.Rep("eh"), vbt.Rep("em"), vbt.Rep("ef"),
        vbt.Rep("ss"), vbt.Rep("se")),
    broadcast_named_args=dict(ca=raw["close"], ua=ind.upper_band.values,
        la=ind.lower_band.values, ta=ind.twap.values, ra=ind.regime_ok.values,
        na=index_ns, eh=21, em=0, ef=5, ss=8, se=16),
    leverage=1.0, slippage=0.00015, sl_stop=0.005, init_cash=1_000_000, freq="1min")
print(f"Trades: {pf.trades.count()}")"""),
        md("## 7. Results"),
        *results_cells("""\
fig.add_trace(go.Scatter(x=raw.index[-n_bars:], y=ind.twap.values[-n_bars:],
    name="TWAP", line=dict(color="#FF9800", dash="dash")))
fig.add_trace(go.Scatter(x=raw.index[-n_bars:], y=ind.upper_band.values[-n_bars:],
    name="Upper", line=dict(color="#E91E63", dash="dot")))
fig.add_trace(go.Scatter(x=raw.index[-n_bars:], y=ind.lower_band.values[-n_bars:],
    name="Lower", line=dict(color="#E91E63", dash="dot")))"""),
        *sweep_cells("MR_V3",
            ind_run="""\
ind_sweep = MR_V3.run(index_ns, raw["high"], raw["low"], raw["close"], raw["open"],
    lookback=vbt.Param([20, 40, 60, 120, 240]), band_width=vbt.Param([1.5, 2.0, 2.5, 3.0]),
    adx_period=14, adx_threshold=30.0, jitted_loop=True, jitted_warmup=True, param_product=True)""",
            pf_run="""\
pf_sweep = vbt.Portfolio.from_signals(
    close=raw["close"], open=raw["open"], high=raw["high"], low=raw["low"],
    signal_func_nb=mr_v3_signal_nb,
    signal_args=(vbt.Rep("ca"), vbt.Rep("ua"), vbt.Rep("la"), vbt.Rep("ta"),
        vbt.Rep("ra"), vbt.Rep("na"), vbt.Rep("eh"), vbt.Rep("em"), vbt.Rep("ef"),
        vbt.Rep("ss"), vbt.Rep("se")),
    broadcast_named_args=dict(ca=raw["close"], ua=ind_sweep.upper_band,
        la=ind_sweep.lower_band, ta=ind_sweep.twap, ra=ind_sweep.regime_ok,
        na=index_ns, eh=21, em=0, ef=5, ss=8, se=16),
    leverage=1.0, slippage=0.00015, sl_stop=0.005, init_cash=1_000_000, freq="1min",
    jitted=dict(parallel=True), chunked="threadpool")""",
            x_level="mr_v3_lookback", y_level="mr_v3_band_width"),
        md("## Cross-Validation\n\n*(Same `@vbt.cv_split` pattern as MR V1 — adapt params)*"),
    ]
    write_nb("03_mr_v3_session_filter.ipynb", cells)


def nb_04():
    """MR V4: Adaptive EWM Bands"""
    cells = [
        md("# MR V4: Adaptive EWM Bands\n\n"
           "| Feature | Detail |\n|---------|--------|\n"
           "| Bands | EWM std (faster volatility adaptation) |\n"
           "| Entry | Band breach |\n| Exit | TWAP crossback |\n"
           "| Key diff | `ewm_std` vs `rolling_std` |"),
        md("## 1. Setup"), code(IMPORTS), code(VBT_SETTINGS),
        md("## 2. Data"), code(LOAD_DATA_INTRADAY), *data_exploration_cells(),
        md("## 3. Numba Kernels"), code(MR_KERNELS),
        md("### V4-Specific: EWM Indicator"),
        code("""\
@njit(nogil=True)
def compute_mr_v4_indicators_nb(index_ns, high, low, close, open_,
                                 ewm_span, band_width, adx_period, adx_threshold):
    n = len(close)
    twap = compute_intraday_twap_nb(index_ns, high, low, close)
    deviation = compute_deviation_nb(close, twap)
    ewm_std = vbt.generic.nb.ewm_std_1d_nb(deviation, span=ewm_span, minp=ewm_span, adjust=False)
    smoothed = vbt.generic.nb.ewm_mean_1d_nb(deviation, span=ewm_span, minp=ewm_span, adjust=False)
    zscore = np.full(n, np.nan)
    for i in range(n):
        if not np.isnan(smoothed[i]) and not np.isnan(ewm_std[i]) and ewm_std[i] > 1e-10:
            zscore[i] = smoothed[i] / ewm_std[i]
    upper = np.full(n, np.nan); lower = np.full(n, np.nan)
    for i in range(n):
        s = ewm_std[i]
        if not np.isnan(s) and s > 1e-10 and not np.isnan(twap[i]):
            upper[i] = twap[i] + band_width * s
            lower[i] = twap[i] - band_width * s
    regime_ok = compute_adx_regime_nb(index_ns, high, low, close, open_, adx_period, adx_threshold)
    return twap, zscore, upper, lower, regime_ok"""),
        md("## 4. Indicators"),
        code("""\
MR_V4 = vbt.IF(
    class_name="MR_V4", short_name="mr_v4",
    input_names=["index_ns", "high_minute", "low_minute", "close_minute", "open_minute"],
    param_names=["ewm_span", "band_width", "adx_period", "adx_threshold"],
    output_names=["twap", "zscore", "upper_band", "lower_band", "regime_ok"],
).with_apply_func(compute_mr_v4_indicators_nb, takes_1d=True,
    ewm_span=60, band_width=2.0, adx_period=14, adx_threshold=30.0)
ind = MR_V4.run(index_ns, raw["high"], raw["low"], raw["close"], raw["open"],
    ewm_span=60, band_width=2.0, adx_period=14, adx_threshold=30.0,
    jitted_loop=True, jitted_warmup=True)"""),
        md("### Indicator Inspection"),
        code("""\
ind_df = pd.DataFrame({
    "close": raw["close"].values, "twap": ind.twap.values, "zscore": ind.zscore.values,
    "upper": ind.upper_band.values, "lower": ind.lower_band.values, "regime_ok": ind.regime_ok.values,
}, index=raw.index)
print(f"Shape: {ind_df.shape} | NaN: {ind_df.isna().sum().to_dict()}")
ind_df.dropna().head(10)"""),
        md("## 5. Signal Visualization"),
        code("""\
n = min(7200, len(raw)); sl = slice(-n, None)
fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3],
    subplot_titles=("Price & EWM Adaptive Bands", "EWM Z-Score"))
fig.add_trace(go.Scatter(x=raw.index[sl], y=raw["close"].values[sl], name="Close",
    line=dict(color="#333", width=1)), row=1, col=1)
fig.add_trace(go.Scatter(x=raw.index[sl], y=ind.twap.values[sl], name="TWAP",
    line=dict(color="#FF9800", dash="dash")), row=1, col=1)
fig.add_trace(go.Scatter(x=raw.index[sl], y=ind.upper_band.values[sl], name="EWM Upper",
    line=dict(color="#9C27B0", dash="dot")), row=1, col=1)
fig.add_trace(go.Scatter(x=raw.index[sl], y=ind.lower_band.values[sl], name="EWM Lower",
    line=dict(color="#9C27B0", dash="dot")), row=1, col=1)
fig.add_trace(go.Scatter(x=raw.index[sl], y=ind.zscore.values[sl], name="EWM Z-Score",
    line=dict(color="#2196F3")), row=2, col=1)
fig.add_hline(y=2, line_dash="dot", line_color="red", row=2, col=1)
fig.add_hline(y=-2, line_dash="dot", line_color="green", row=2, col=1)
fig.update_layout(height=700, title="MR V4: Adaptive EWM Bands")
fig.show()"""),
        md("## 6. Backtest\n\nUses `mr_band_signal_nb` (shared with V1)."),
        code("""\
pf = vbt.Portfolio.from_signals(
    close=raw["close"], open=raw["open"], high=raw["high"], low=raw["low"],
    signal_func_nb=mr_band_signal_nb,
    signal_args=(vbt.Rep("ca"), vbt.Rep("ua"), vbt.Rep("la"), vbt.Rep("ta"),
        vbt.Rep("ra"), vbt.Rep("na"), vbt.Rep("eh"), vbt.Rep("em"), vbt.Rep("ef")),
    broadcast_named_args=dict(ca=raw["close"], ua=ind.upper_band.values,
        la=ind.lower_band.values, ta=ind.twap.values, ra=ind.regime_ok.values,
        na=index_ns, eh=21, em=0, ef=5),
    leverage=1.0, slippage=0.00015, sl_stop=0.005, init_cash=1_000_000, freq="1min")
print(f"Trades: {pf.trades.count()}")"""),
        md("## 7. Results"),
        *results_cells("""\
fig.add_trace(go.Scatter(x=raw.index[-n_bars:], y=ind.twap.values[-n_bars:],
    name="TWAP", line=dict(color="#FF9800", dash="dash")))
fig.add_trace(go.Scatter(x=raw.index[-n_bars:], y=ind.upper_band.values[-n_bars:],
    name="EWM Upper", line=dict(color="#9C27B0", dash="dot")))
fig.add_trace(go.Scatter(x=raw.index[-n_bars:], y=ind.lower_band.values[-n_bars:],
    name="EWM Lower", line=dict(color="#9C27B0", dash="dot")))"""),
        *sweep_cells("MR_V4",
            ind_run="""\
ind_sweep = MR_V4.run(index_ns, raw["high"], raw["low"], raw["close"], raw["open"],
    ewm_span=vbt.Param([20, 40, 60, 120]), band_width=vbt.Param([1.5, 2.0, 2.5, 3.0]),
    adx_period=14, adx_threshold=30.0, jitted_loop=True, jitted_warmup=True, param_product=True)""",
            pf_run="""\
pf_sweep = vbt.Portfolio.from_signals(
    close=raw["close"], open=raw["open"], high=raw["high"], low=raw["low"],
    signal_func_nb=mr_band_signal_nb,
    signal_args=(vbt.Rep("ca"), vbt.Rep("ua"), vbt.Rep("la"), vbt.Rep("ta"),
        vbt.Rep("ra"), vbt.Rep("na"), vbt.Rep("eh"), vbt.Rep("em"), vbt.Rep("ef")),
    broadcast_named_args=dict(ca=raw["close"], ua=ind_sweep.upper_band,
        la=ind_sweep.lower_band, ta=ind_sweep.twap, ra=ind_sweep.regime_ok,
        na=index_ns, eh=21, em=0, ef=5),
    leverage=1.0, slippage=0.00015, sl_stop=0.005, init_cash=1_000_000, freq="1min",
    jitted=dict(parallel=True), chunked="threadpool")""",
            x_level="mr_v4_ewm_span", y_level="mr_v4_band_width"),
        md("## Cross-Validation\n\n*(Same `@vbt.cv_split` pattern as MR V1)*"),
    ]
    write_nb("04_mr_v4_adaptive_ewm.ipynb", cells)


def nb_05():
    """OU Mean Reversion: Vol-Targeted"""
    cells = [
        md("# OU Mean Reversion: Vol-Targeted Leverage\n\n"
           "| Feature | Detail |\n|---------|--------|\n"
           "| Sizing | Dynamic: sigma_target / realized_vol |\n"
           "| Vol window | 20-day close-to-close |\n"
           "| Max leverage | 3.0x |\n| Entry/Exit | Same as V1 |"),
        md("## 1. Setup"), code(IMPORTS), code(VBT_SETTINGS),
        md("## 2. Data"), code(LOAD_DATA_INTRADAY), *data_exploration_cells(),
        md("## 3. Numba Kernels"), code(MR_KERNELS), code(VOL_LEVERAGE_KERNELS),
        md("### OU-Specific: Vol-Targeted Indicator"),
        code("""\
@njit(nogil=True)
def compute_ou_mr_indicators_nb(index_ns, high, low, close, open_,
    lookback, band_width, adx_period, adx_threshold, vol_window, sigma_target, max_leverage):
    twap, zscore, upper, lower, regime_ok = compute_mr_base_indicators_nb(
        index_ns, high, low, close, open_, lookback, band_width, adx_period, adx_threshold)
    rolling_vol = compute_daily_rolling_volatility_nb(index_ns, close, vol_window)
    leverage = compute_leverage_nb(rolling_vol, sigma_target, max_leverage)
    return twap, zscore, upper, lower, regime_ok, leverage"""),
        md("## 4. Indicators"),
        code("""\
OU_MR = vbt.IF(
    class_name="IntradayMR", short_name="imr",
    input_names=["index_ns", "high_minute", "low_minute", "close_minute", "open_minute"],
    param_names=["lookback", "band_width", "adx_period", "adx_threshold",
                 "vol_window", "sigma_target", "max_leverage"],
    output_names=["twap", "zscore", "upper_band", "lower_band", "regime_ok", "leverage"],
).with_apply_func(compute_ou_mr_indicators_nb, takes_1d=True,
    lookback=60, band_width=2.0, adx_period=14, adx_threshold=30.0,
    vol_window=20, sigma_target=0.01, max_leverage=3.0)
ind = OU_MR.run(index_ns, raw["high"], raw["low"], raw["close"], raw["open"],
    lookback=60, band_width=2.0, adx_period=14, adx_threshold=30.0,
    vol_window=20, sigma_target=0.01, max_leverage=3.0,
    jitted_loop=True, jitted_warmup=True)"""),
        md("### Indicator Inspection"),
        code("""\
ind_df = pd.DataFrame({
    "close": raw["close"].values, "twap": ind.twap.values, "zscore": ind.zscore.values,
    "upper": ind.upper_band.values, "lower": ind.lower_band.values,
    "regime_ok": ind.regime_ok.values, "leverage": ind.leverage.values,
}, index=raw.index)
print(f"Shape: {ind_df.shape} | NaN: {ind_df.isna().sum().to_dict()}")
print(f"\\nLeverage stats: mean={ind_df['leverage'].mean():.2f}, max={ind_df['leverage'].max():.2f}")
ind_df.dropna().head(10)"""),
        md("## 5. Signal Visualization"),
        code("""\
n = min(7200, len(raw)); sl = slice(-n, None)
fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.5, 0.25, 0.25],
    subplot_titles=("Price & Bands", "Z-Score", "Dynamic Leverage"))
fig.add_trace(go.Scatter(x=raw.index[sl], y=raw["close"].values[sl], name="Close",
    line=dict(color="#333", width=1)), row=1, col=1)
fig.add_trace(go.Scatter(x=raw.index[sl], y=ind.twap.values[sl], name="TWAP",
    line=dict(color="#FF9800", dash="dash")), row=1, col=1)
fig.add_trace(go.Scatter(x=raw.index[sl], y=ind.upper_band.values[sl], name="Upper",
    line=dict(color="#E91E63", dash="dot")), row=1, col=1)
fig.add_trace(go.Scatter(x=raw.index[sl], y=ind.lower_band.values[sl], name="Lower",
    line=dict(color="#E91E63", dash="dot")), row=1, col=1)
fig.add_trace(go.Scatter(x=raw.index[sl], y=ind.zscore.values[sl], name="Z-Score",
    line=dict(color="#2196F3")), row=2, col=1)
fig.add_hline(y=2, line_dash="dot", line_color="red", row=2, col=1)
fig.add_hline(y=-2, line_dash="dot", line_color="green", row=2, col=1)
fig.add_trace(go.Scatter(x=raw.index[sl], y=ind.leverage.values[sl], name="Leverage",
    fill="tozeroy", line=dict(color="#4CAF50")), row=3, col=1)
fig.add_hline(y=3.0, line_dash="dot", line_color="red", annotation_text="Cap", row=3, col=1)
fig.update_layout(height=900, title="OU Mean Reversion: Vol-Targeted")
fig.show()"""),
        code("""\
# Leverage distribution
lev = ind.leverage.values[~np.isnan(ind.leverage.values)]
fig = go.Figure(data=go.Histogram(x=lev, nbinsx=50, marker_color="#4CAF50"))
fig.add_vline(x=np.median(lev), line_dash="dash", line_color="red",
    annotation_text=f"Median: {np.median(lev):.2f}")
fig.update_layout(height=300, title="Leverage Distribution")
fig.show()"""),
        md("## 6. Backtest"),
        code("""\
pf = vbt.Portfolio.from_signals(
    close=raw["close"], open=raw["open"], high=raw["high"], low=raw["low"],
    signal_func_nb=mr_band_signal_nb,
    signal_args=(vbt.Rep("ca"), vbt.Rep("ua"), vbt.Rep("la"), vbt.Rep("ta"),
        vbt.Rep("ra"), vbt.Rep("na"), vbt.Rep("eh"), vbt.Rep("em"), vbt.Rep("ef")),
    broadcast_named_args=dict(ca=raw["close"], ua=ind.upper_band.values,
        la=ind.lower_band.values, ta=ind.twap.values, ra=ind.regime_ok.values,
        na=index_ns, eh=21, em=0, ef=5),
    leverage=ind.leverage.values,  # Dynamic vol-targeted!
    slippage=0.00015, sl_stop=0.005, init_cash=1_000_000, freq="1min")
print(f"Trades: {pf.trades.count()}")"""),
        md("## 7. Results"),
        *results_cells("""\
fig.add_trace(go.Scatter(x=raw.index[-n_bars:], y=ind.twap.values[-n_bars:],
    name="TWAP", line=dict(color="#FF9800", dash="dash")))"""),
        *sweep_cells("OU_MR",
            ind_run="""\
ind_sweep = OU_MR.run(index_ns, raw["high"], raw["low"], raw["close"], raw["open"],
    lookback=vbt.Param([20, 40, 60, 120, 240]), band_width=vbt.Param([1.5, 2.0, 2.5, 3.0, 3.5]),
    adx_period=14, adx_threshold=30.0, vol_window=20, sigma_target=0.01, max_leverage=3.0,
    jitted_loop=True, jitted_warmup=True, param_product=True)""",
            pf_run="""\
pf_sweep = vbt.Portfolio.from_signals(
    close=raw["close"], open=raw["open"], high=raw["high"], low=raw["low"],
    signal_func_nb=mr_band_signal_nb,
    signal_args=(vbt.Rep("ca"), vbt.Rep("ua"), vbt.Rep("la"), vbt.Rep("ta"),
        vbt.Rep("ra"), vbt.Rep("na"), vbt.Rep("eh"), vbt.Rep("em"), vbt.Rep("ef")),
    broadcast_named_args=dict(ca=raw["close"], ua=ind_sweep.upper_band,
        la=ind_sweep.lower_band, ta=ind_sweep.twap, ra=ind_sweep.regime_ok,
        na=index_ns, eh=21, em=0, ef=5),
    leverage=ind_sweep.leverage,
    slippage=0.00015, sl_stop=0.005, init_cash=1_000_000, freq="1min",
    jitted=dict(parallel=True), chunked="threadpool")""",
            x_level="imr_lookback", y_level="imr_band_width"),
        md("## Cross-Validation\n\n*(Same `@vbt.cv_split` pattern as MR V1)*"),
    ]
    write_nb("05_ou_mean_reversion.ipynb", cells)


def nb_06():
    """Kalman Trend Following"""
    cells = [
        md("# Kalman Trend Following\n\n"
           "| Feature | Detail |\n|---------|--------|\n"
           "| Filter | 2-state Kalman (price + velocity) |\n"
           "| Entry Long | EMA_fast > EMA_slow AND vel > 0 AND price > VWAP |\n"
           "| Entry Short | EMA_fast < EMA_slow AND vel < 0 AND price < VWAP |\n"
           "| Eval freq | 15 min |"),
        md("## 1. Setup"), code(IMPORTS), code(VBT_SETTINGS),
        md("## 2. Data"), code(LOAD_DATA_INTRADAY), *data_exploration_cells(),
        md("## 3. Numba Kernels"),
        code("""\
@njit(nogil=True)
def find_day_boundaries_nb(index_ns):
    n = len(index_ns)
    start_idx = np.empty(n, dtype=np.int64); end_idx = np.empty(n, dtype=np.int64)
    if n == 0: return start_idx, end_idx, 0
    day_number = vbt.dt_nb.days_nb(ts=index_ns)
    current_day = day_number[0]; day_counter = 0; current_start = 0
    for i in range(1, n):
        if day_number[i] != current_day:
            start_idx[day_counter] = current_start; end_idx[day_counter] = i
            day_counter += 1; current_day = day_number[i]; current_start = i
    start_idx[day_counter] = current_start; end_idx[day_counter] = n; day_counter += 1
    return start_idx, end_idx, day_counter

@njit(nogil=True)
def kalman_filter_1d_nb(close, process_var, measurement_var):
    n = len(close)
    kf_price = np.full(n, np.nan); kf_velocity = np.full(n, np.nan)
    if n == 0: return kf_price, kf_velocity
    start = 0
    while start < n and np.isnan(close[start]): start += 1
    if start >= n: return kf_price, kf_velocity
    pe = close[start]; ve = 0.0; p11 = 1.0; p12 = 0.0; p22 = 1.0
    kf_price[start] = pe; kf_velocity[start] = ve
    for i in range(start+1, n):
        if np.isnan(close[i]):
            kf_price[i] = pe + ve; kf_velocity[i] = ve; continue
        pp = pe + ve; vp = ve
        p11p = p11 + 2*p12 + p22 + process_var; p12p = p12 + p22; p22p = p22 + process_var*0.01
        inn = close[i] - pp; S = p11p + measurement_var
        if abs(S) < 1e-15: kf_price[i] = pp; kf_velocity[i] = vp; continue
        k1 = p11p/S; k2 = p12p/S
        pe = pp + k1*inn; ve = vp + k2*inn
        p11 = (1-k1)*p11p; p12 = p12p - k1*p12p; p22 = p22p - k2*p12p
        kf_price[i] = pe; kf_velocity[i] = ve
    return kf_price, kf_velocity

@njit(nogil=True)
def compute_intraday_kalman_indicators_nb(index_ns, high, low, close, open_,
                                           process_var, measurement_var, ema_fast, ema_slow):
    kf_price, kf_velocity = kalman_filter_1d_nb(close, process_var, measurement_var)
    ema_fast_line = vbt.generic.nb.ewm_mean_1d_nb(kf_price, ema_fast, minp=1, adjust=True)
    ema_slow_line = vbt.generic.nb.ewm_mean_1d_nb(kf_price, ema_slow, minp=1, adjust=True)
    start_arr, end_arr, n_days = find_day_boundaries_nb(index_ns)
    group_lens = end_arr[:n_days] - start_arr[:n_days]
    vol_proxy = np.ones(len(close))  # No volume in FX data — use uniform weights
    vwap = vbt.indicators.nb.vwap_1d_nb(high, low, close, vol_proxy, group_lens)
    return kf_price, kf_velocity, ema_fast_line, ema_slow_line, vwap

@njit(nogil=True)
def intraday_kalman_signal_nb(c, close_arr, ema_fast_arr, ema_slow_arr,
                               velocity_arr, vwap_arr, index_ns_arr, eod_hour_arr, eod_minute_arr):
    ts_ns = index_ns_arr[c.i]
    cur_h = vbt.dt_nb.hour_nb(ts_ns); cur_m = vbt.dt_nb.minute_nb(ts_ns)
    eod_h = vbt.pf_nb.select_nb(c, eod_hour_arr); eod_m = vbt.pf_nb.select_nb(c, eod_minute_arr)
    if (cur_h > eod_h) or (cur_h == eod_h and cur_m >= eod_m):
        return False, vbt.pf_nb.ctx_helpers.in_long_position_nb(c), False, vbt.pf_nb.ctx_helpers.in_short_position_nb(c)
    if cur_m % 15 != 0: return False, False, False, False
    px = vbt.pf_nb.select_nb(c, close_arr); ef = vbt.pf_nb.select_nb(c, ema_fast_arr)
    es = vbt.pf_nb.select_nb(c, ema_slow_arr); vel = vbt.pf_nb.select_nb(c, velocity_arr)
    vw = vbt.pf_nb.select_nb(c, vwap_arr)
    if np.isnan(px) or np.isnan(ef) or np.isnan(es) or np.isnan(vel) or np.isnan(vw):
        return False, False, False, False
    il = vbt.pf_nb.ctx_helpers.in_long_position_nb(c)
    ish = vbt.pf_nb.ctx_helpers.in_short_position_nb(c)
    if not il and not ish:
        if ef > es and vel > 0 and px > vw: return True, False, False, False
        elif ef < es and vel < 0 and px < vw: return False, False, True, False
    elif il:
        if ef < es or vel < 0: return False, True, False, False
    elif ish and (ef > es or vel > 0): return False, False, False, True
    return False, False, False, False"""),
        md("## 4. Indicators"),
        code("""\
KALMAN = vbt.IF(
    class_name="IntradayKalman", short_name="ikt",
    input_names=["index_ns", "high_minute", "low_minute", "close_minute", "open_minute"],
    param_names=["process_var", "measurement_var", "ema_fast", "ema_slow"],
    output_names=["kalman_price", "kalman_velocity", "ema_fast_line", "ema_slow_line", "vwap"],
).with_apply_func(compute_intraday_kalman_indicators_nb, takes_1d=True,
    process_var=0.001, measurement_var=1.0, ema_fast=100, ema_slow=500)
ind = KALMAN.run(index_ns, raw["high"], raw["low"], raw["close"], raw["open"],
    process_var=0.001, measurement_var=1.0, ema_fast=100, ema_slow=500,
    jitted_loop=True, jitted_warmup=True)"""),
        md("### Indicator Inspection"),
        code("""\
ind_df = pd.DataFrame({
    "close": raw["close"].values, "kalman_price": ind.kalman_price.values,
    "velocity": ind.kalman_velocity.values, "ema_fast": ind.ema_fast_line.values,
    "ema_slow": ind.ema_slow_line.values, "vwap": ind.vwap.values,
}, index=raw.index)
print(f"Shape: {ind_df.shape} | NaN: {ind_df.isna().sum().to_dict()}")
print(f"\\nVelocity stats: mean={ind_df['velocity'].mean():.6f}, std={ind_df['velocity'].std():.6f}")
ind_df.dropna().head(10)"""),
        md("## 5. Signal Visualization"),
        code("""\
n = min(7200, len(raw)); sl = slice(-n, None)
fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3],
    subplot_titles=("Price + Kalman + EMAs + VWAP", "Kalman Velocity"))
fig.add_trace(go.Scatter(x=raw.index[sl], y=raw["close"].values[sl], name="Close",
    line=dict(color="#CCC", width=1)), row=1, col=1)
fig.add_trace(go.Scatter(x=raw.index[sl], y=ind.kalman_price.values[sl], name="Kalman",
    line=dict(color="#2196F3", width=2)), row=1, col=1)
fig.add_trace(go.Scatter(x=raw.index[sl], y=ind.ema_fast_line.values[sl], name="EMA Fast",
    line=dict(color="#4CAF50", dash="dash")), row=1, col=1)
fig.add_trace(go.Scatter(x=raw.index[sl], y=ind.ema_slow_line.values[sl], name="EMA Slow",
    line=dict(color="#FF5722", dash="dash")), row=1, col=1)
fig.add_trace(go.Scatter(x=raw.index[sl], y=ind.vwap.values[sl], name="VWAP",
    line=dict(color="#9C27B0", dash="dot")), row=1, col=1)
fig.add_trace(go.Scatter(x=raw.index[sl], y=ind.kalman_velocity.values[sl], name="Velocity",
    fill="tozeroy", line=dict(color="#2196F3")), row=2, col=1)
fig.add_hline(y=0, line_color="gray", row=2, col=1)
fig.update_layout(height=700, title="Kalman Trend Following")
fig.show()"""),
        md("## 6. Backtest"),
        code("""\
pf = vbt.Portfolio.from_signals(
    close=raw["close"], open=raw["open"], high=raw["high"], low=raw["low"],
    signal_func_nb=intraday_kalman_signal_nb,
    signal_args=(vbt.Rep("ca"), vbt.Rep("ef"), vbt.Rep("es"), vbt.Rep("va"),
        vbt.Rep("vw"), vbt.Rep("na"), vbt.Rep("eh"), vbt.Rep("em")),
    broadcast_named_args=dict(ca=raw["close"], ef=ind.ema_fast_line.values,
        es=ind.ema_slow_line.values, va=ind.kalman_velocity.values,
        vw=ind.vwap.values, na=index_ns, eh=21, em=0),
    slippage=0.00015, init_cash=1_000_000, freq="1min")
print(f"Trades: {pf.trades.count()}")"""),
        md("## 7. Results"),
        *results_cells("""\
fig.add_trace(go.Scatter(x=raw.index[-n_bars:], y=ind.kalman_price.values[-n_bars:],
    name="Kalman", line=dict(color="#2196F3")))
fig.add_trace(go.Scatter(x=raw.index[-n_bars:], y=ind.ema_fast_line.values[-n_bars:],
    name="EMA Fast", line=dict(color="#4CAF50", dash="dash")))
fig.add_trace(go.Scatter(x=raw.index[-n_bars:], y=ind.vwap.values[-n_bars:],
    name="VWAP", line=dict(color="#9C27B0", dash="dot")))"""),
        *sweep_cells("KALMAN",
            ind_run="""\
ind_sweep = KALMAN.run(index_ns, raw["high"], raw["low"], raw["close"], raw["open"],
    process_var=0.001, measurement_var=1.0,
    ema_fast=vbt.Param([50, 100, 200]), ema_slow=vbt.Param([300, 500, 700]),
    jitted_loop=True, jitted_warmup=True, param_product=True)""",
            pf_run="""\
pf_sweep = vbt.Portfolio.from_signals(
    close=raw["close"], open=raw["open"], high=raw["high"], low=raw["low"],
    signal_func_nb=intraday_kalman_signal_nb,
    signal_args=(vbt.Rep("ca"), vbt.Rep("ef"), vbt.Rep("es"), vbt.Rep("va"),
        vbt.Rep("vw"), vbt.Rep("na"), vbt.Rep("eh"), vbt.Rep("em")),
    broadcast_named_args=dict(ca=raw["close"], ef=ind_sweep.ema_fast_line,
        es=ind_sweep.ema_slow_line, va=ind_sweep.kalman_velocity,
        vw=ind_sweep.vwap, na=index_ns, eh=21, em=0),
    slippage=0.00015, init_cash=1_000_000, freq="1min",
    jitted=dict(parallel=True), chunked="threadpool")""",
            x_level="ikt_ema_fast", y_level="ikt_ema_slow"),
        md("## Cross-Validation\n\n*(Same `@vbt.cv_split` pattern)*"),
    ]
    write_nb("06_kalman_trend.ipynb", cells)


def nb_07():
    """Donchian Channel Breakout"""
    cells = [
        md("# Donchian Channel Breakout\n\n"
           "| Feature | Detail |\n|---------|--------|\n"
           "| Entry Long | Price > upper channel AND > VWAP |\n"
           "| Entry Short | Price < lower channel AND < VWAP |\n"
           "| Exit | Price crosses exit channel |\n"
           "| Channels | 1-bar lagged (no look-ahead) |\n| Eval freq | 15 min |"),
        md("## 1. Setup"), code(IMPORTS), code(VBT_SETTINGS),
        md("## 2. Data"), code(LOAD_DATA_INTRADAY), *data_exploration_cells(),
        md("## 3. Numba Kernels"),
        code("""\
@njit(nogil=True)
def find_day_boundaries_nb(index_ns):
    n = len(index_ns)
    start_idx = np.empty(n, dtype=np.int64); end_idx = np.empty(n, dtype=np.int64)
    if n == 0: return start_idx, end_idx, 0
    day_number = vbt.dt_nb.days_nb(ts=index_ns)
    current_day = day_number[0]; day_counter = 0; current_start = 0
    for i in range(1, n):
        if day_number[i] != current_day:
            start_idx[day_counter] = current_start; end_idx[day_counter] = i
            day_counter += 1; current_day = day_number[i]; current_start = i
    start_idx[day_counter] = current_start; end_idx[day_counter] = n; day_counter += 1
    return start_idx, end_idx, day_counter

@njit(nogil=True)
def compute_intraday_donchian_nb(index_ns, high, low, close, open_, entry_period, exit_period):
    upper = vbt.generic.nb.fshift_1d_nb(vbt.generic.nb.rolling_max_1d_nb(high, entry_period, minp=entry_period), 1)
    lower = vbt.generic.nb.fshift_1d_nb(vbt.generic.nb.rolling_min_1d_nb(low, entry_period, minp=entry_period), 1)
    exit_upper = vbt.generic.nb.fshift_1d_nb(vbt.generic.nb.rolling_max_1d_nb(high, exit_period, minp=exit_period), 1)
    exit_lower = vbt.generic.nb.fshift_1d_nb(vbt.generic.nb.rolling_min_1d_nb(low, exit_period, minp=exit_period), 1)
    start_arr, end_arr, n_days = find_day_boundaries_nb(index_ns)
    group_lens = end_arr[:n_days] - start_arr[:n_days]
    vol_proxy = np.ones(len(close))  # No volume in FX data
    vwap = vbt.indicators.nb.vwap_1d_nb(high, low, close, vol_proxy, group_lens)
    return upper, lower, exit_upper, exit_lower, vwap

@njit(nogil=True)
def intraday_donchian_signal_nb(c, close_arr, upper_arr, lower_arr, exit_upper_arr, exit_lower_arr,
                                 vwap_arr, index_ns_arr, eod_hour_arr, eod_minute_arr):
    ts_ns = index_ns_arr[c.i]
    cur_h = vbt.dt_nb.hour_nb(ts_ns); cur_m = vbt.dt_nb.minute_nb(ts_ns)
    eod_h = vbt.pf_nb.select_nb(c, eod_hour_arr); eod_m = vbt.pf_nb.select_nb(c, eod_minute_arr)
    if (cur_h > eod_h) or (cur_h == eod_h and cur_m >= eod_m):
        return False, vbt.pf_nb.ctx_helpers.in_long_position_nb(c), False, vbt.pf_nb.ctx_helpers.in_short_position_nb(c)
    if cur_m % 15 != 0: return False, False, False, False
    px = vbt.pf_nb.select_nb(c, close_arr); ub = vbt.pf_nb.select_nb(c, upper_arr)
    lb = vbt.pf_nb.select_nb(c, lower_arr); exu = vbt.pf_nb.select_nb(c, exit_upper_arr)
    exl = vbt.pf_nb.select_nb(c, exit_lower_arr); vw = vbt.pf_nb.select_nb(c, vwap_arr)
    if np.isnan(px) or np.isnan(ub) or np.isnan(lb) or np.isnan(vw): return False, False, False, False
    il = vbt.pf_nb.ctx_helpers.in_long_position_nb(c)
    ish = vbt.pf_nb.ctx_helpers.in_short_position_nb(c)
    if not il and not ish:
        if px > ub and px > vw: return True, False, False, False
        elif px < lb and px < vw: return False, False, True, False
    elif il:
        if not np.isnan(exl) and px < exl: return False, True, False, False
    elif ish and not np.isnan(exu) and px > exu: return False, False, False, True
    return False, False, False, False"""),
        md("## 4. Indicators"),
        code("""\
DONCHIAN = vbt.IF(
    class_name="IntradayDonchian", short_name="idb",
    input_names=["index_ns", "high_minute", "low_minute", "close_minute", "open_minute"],
    param_names=["entry_period", "exit_period"],
    output_names=["upper_channel", "lower_channel", "exit_upper", "exit_lower", "vwap"],
).with_apply_func(compute_intraday_donchian_nb, takes_1d=True, entry_period=240, exit_period=60)
ind = DONCHIAN.run(index_ns, raw["high"], raw["low"], raw["close"], raw["open"],
    entry_period=240, exit_period=60, jitted_loop=True, jitted_warmup=True)"""),
        md("### Indicator Inspection"),
        code("""\
ind_df = pd.DataFrame({
    "close": raw["close"].values, "upper": ind.upper_channel.values, "lower": ind.lower_channel.values,
    "exit_upper": ind.exit_upper.values, "exit_lower": ind.exit_lower.values, "vwap": ind.vwap.values,
}, index=raw.index)
print(f"Shape: {ind_df.shape} | NaN: {ind_df.isna().sum().to_dict()}")
print(f"\\nChannel width: mean={( ind_df['upper']-ind_df['lower']).mean():.6f}")
ind_df.dropna().head(10)"""),
        md("## 5. Signal Visualization"),
        code("""\
n = min(7200, len(raw)); sl = slice(-n, None)
fig = go.Figure()
fig.add_trace(go.Scatter(x=raw.index[sl], y=raw["close"].values[sl], name="Close", line=dict(color="#333", width=1)))
fig.add_trace(go.Scatter(x=raw.index[sl], y=ind.upper_channel.values[sl], name="Entry Upper", line=dict(color="#4CAF50", width=2)))
fig.add_trace(go.Scatter(x=raw.index[sl], y=ind.lower_channel.values[sl], name="Entry Lower", line=dict(color="#F44336", width=2)))
fig.add_trace(go.Scatter(x=raw.index[sl], y=ind.exit_upper.values[sl], name="Exit Upper", line=dict(color="#4CAF50", dash="dot")))
fig.add_trace(go.Scatter(x=raw.index[sl], y=ind.exit_lower.values[sl], name="Exit Lower", line=dict(color="#F44336", dash="dot")))
fig.add_trace(go.Scatter(x=raw.index[sl], y=ind.vwap.values[sl], name="VWAP", line=dict(color="#9C27B0", dash="dash")))
fig.update_layout(height=600, title="Donchian Channels (entry=240, exit=60)")
fig.show()"""),
        code("""\
# Channel width
cw = ind.upper_channel.values - ind.lower_channel.values
fig = go.Figure(data=go.Scatter(x=raw.index[sl], y=cw[sl], fill="tozeroy",
    line=dict(color="rgba(33,150,243,0.5)"), name="Width"))
fig.update_layout(height=300, title="Channel Width (Volatility Proxy)")
fig.show()"""),
        md("## 6. Backtest"),
        code("""\
pf = vbt.Portfolio.from_signals(
    close=raw["close"], open=raw["open"], high=raw["high"], low=raw["low"],
    signal_func_nb=intraday_donchian_signal_nb,
    signal_args=(vbt.Rep("ca"), vbt.Rep("ua"), vbt.Rep("la"), vbt.Rep("eua"), vbt.Rep("ela"),
        vbt.Rep("vw"), vbt.Rep("na"), vbt.Rep("eh"), vbt.Rep("em")),
    broadcast_named_args=dict(ca=raw["close"], ua=ind.upper_channel.values, la=ind.lower_channel.values,
        eua=ind.exit_upper.values, ela=ind.exit_lower.values, vw=ind.vwap.values,
        na=index_ns, eh=21, em=0),
    slippage=0.00015, init_cash=1_000_000, freq="1min")
print(f"Trades: {pf.trades.count()}")"""),
        md("## 7. Results"),
        *results_cells("""\
fig.add_trace(go.Scatter(x=raw.index[-n_bars:], y=ind.upper_channel.values[-n_bars:],
    name="Upper", line=dict(color="#4CAF50")))
fig.add_trace(go.Scatter(x=raw.index[-n_bars:], y=ind.lower_channel.values[-n_bars:],
    name="Lower", line=dict(color="#F44336")))
fig.add_trace(go.Scatter(x=raw.index[-n_bars:], y=ind.vwap.values[-n_bars:],
    name="VWAP", line=dict(color="#9C27B0", dash="dash")))"""),
        *sweep_cells("DONCHIAN",
            ind_run="""\
ind_sweep = DONCHIAN.run(index_ns, raw["high"], raw["low"], raw["close"], raw["open"],
    entry_period=vbt.Param([120, 180, 240, 360]), exit_period=vbt.Param([30, 60, 90, 120]),
    jitted_loop=True, jitted_warmup=True, param_product=True)""",
            pf_run="""\
pf_sweep = vbt.Portfolio.from_signals(
    close=raw["close"], open=raw["open"], high=raw["high"], low=raw["low"],
    signal_func_nb=intraday_donchian_signal_nb,
    signal_args=(vbt.Rep("ca"), vbt.Rep("ua"), vbt.Rep("la"), vbt.Rep("eua"), vbt.Rep("ela"),
        vbt.Rep("vw"), vbt.Rep("na"), vbt.Rep("eh"), vbt.Rep("em")),
    broadcast_named_args=dict(ca=raw["close"], ua=ind_sweep.upper_channel, la=ind_sweep.lower_channel,
        eua=ind_sweep.exit_upper, ela=ind_sweep.exit_lower, vw=ind_sweep.vwap,
        na=index_ns, eh=21, em=0),
    slippage=0.00015, init_cash=1_000_000, freq="1min",
    jitted=dict(parallel=True), chunked="threadpool")""",
            x_level="idb_entry_period", y_level="idb_exit_period"),
        md("## Cross-Validation\n\n*(Same `@vbt.cv_split` pattern)*"),
    ]
    write_nb("07_donchian_breakout.ipynb", cells)


def nb_08():
    """Composite FX Alpha (Daily)"""
    cells = [
        md("# Composite FX Alpha: Multi-Factor Daily\n\n"
           "| Component | Detail |\n|-----------|--------|\n"
           "| Momentum | Blended 21d/63d log returns |\n"
           "| Vol Regime | Short/long vol ratio -> weight |\n"
           "| Vol Scaling | target_vol / realized_vol |\n"
           "| Drawdown | State machine (NORMAL/REDUCED/FLAT) |\n"
           "| Sub-portfolios | K=5 Jegadeesh-Titman |\n"
           "| Rebalancing | Daily target-weight (amount) |"),
        md("## 1. Setup"), code(IMPORTS), code(VBT_SETTINGS),
        md("## 2. Data (Daily)"),
        code("""\
_PROJECT_ROOT = Path(__file__).resolve().parent if "__file__" in dir() else Path(".").resolve()
# When run from notebooks/ dir: parent is project root
# When run from project root: "." is project root
# Fallback: search upward for data/
for _p in [Path(".").resolve(), Path(".").resolve().parent, Path(".").resolve().parent.parent]:
    if (_p / "data" / "EUR-USD.parquet").exists():
        _PROJECT_ROOT = _p
        break

def load_fx_data(path="data/EUR-USD.parquet", shift_hours=0):
    resolved = _PROJECT_ROOT / path
    data_raw = vbt.Data.from_parquet(str(resolved))
    symbol = data_raw.symbols[0]
    df = data_raw.data[symbol].set_index("date").sort_index()
    if shift_hours:
        df.index = df.index + pd.Timedelta(hours=shift_hours)
    raw = df.copy()
    raw.columns = [c.lower() for c in raw.columns]
    return raw

raw = load_fx_data(shift_hours=7)
daily = raw.resample("1D").agg({"open":"first","high":"max","low":"min","close":"last"}).dropna()
close = daily["close"].values
log_returns = np.log(daily["close"] / daily["close"].shift(1)).values
print(f"Daily bars: {len(daily):,} | {daily.index[0].date()} -> {daily.index[-1].date()}")"""),
        code("""\
fig = go.Figure(data=go.Scatter(x=daily.index, y=daily["close"], line=dict(color="#333")))
fig.update_layout(height=400, title="EUR/USD Daily Close")
fig.show()"""),
        md("## 3. Numba Kernels"),
        code("""\
@njit(nogil=True)
def momentum_signal_nb(close, w_short, w_long):
    n = len(close); out = np.full(n, np.nan)
    for i in range(w_long, n):
        out[i] = 0.5 * np.log(close[i]/close[i-w_short]) + 0.5 * np.log(close[i]/close[i-w_long])
    return out

@njit(nogil=True)
def regime_weight_nb(vr, low_th, high_th, w_low, w_normal, w_high):
    n = len(vr); out = np.full(n, np.nan)
    for i in range(n):
        if np.isnan(vr[i]): continue
        elif vr[i] < low_th: out[i] = w_low
        elif vr[i] > high_th: out[i] = w_high
        else: out[i] = w_normal
    return out

@njit(nogil=True)
def vol_scaling_nb(ewma_vol, target, cap):
    n = len(ewma_vol); out = np.full(n, np.nan)
    for i in range(n):
        v = ewma_vol[i]
        if not np.isnan(v) and v > 0: out[i] = min(target/v, cap)
    return out

@njit(nogil=True)
def drawdown_control_nb(dd, soft, hard, recovery):
    n = len(dd); mult = np.ones(n); state = 0
    for i in range(n):
        d = dd[i]
        if state==0:
            if d > hard: state=2
            elif d > soft: state=1
        elif state==1:
            if d > hard: state=2
            elif d < recovery: state=0
        elif state==2 and d < recovery: state=0
        if state==1: mult[i]=0.5
        elif state==2: mult[i]=0.0
    return mult

@njit(nogil=True)
def sub_portfolio_weights_nb(direction, regime_wt, vol_scale, dd_mult, n_days, k):
    sub_w = np.zeros((k, n_days))
    for j in range(k):
        current = 0.0
        for i in range(n_days):
            if i >= j*5 and (i-j*5) % (k*5) == 0:
                d=direction[i]; r=regime_wt[i]; v=vol_scale[i]; m=dd_mult[i]
                if not (np.isnan(d) or np.isnan(r) or np.isnan(v) or np.isnan(m)):
                    current = d*r*v*m
            sub_w[j,i] = current
    weights = np.zeros(n_days)
    for i in range(n_days):
        t = 0.0
        for j in range(k): t += sub_w[j,i]
        weights[i] = t/k
    return weights

@njit(nogil=True)
def compute_composite_nb(close, returns, w_short, w_long, vol_short, vol_long,
    ewma_span, target_vol, leverage_cap, vr_low, vr_high,
    mom_w_low, mom_w_normal, mom_w_high, dd_soft, dd_hard, dd_recovery, n_sub):
    n = len(close)
    momentum = momentum_signal_nb(close, w_short, w_long)
    direction = np.full(n, 0.0)
    for i in range(n):
        if not np.isnan(momentum[i]):
            direction[i] = 1.0 if momentum[i]>0 else (-1.0 if momentum[i]<0 else 0.0)
    ss = vbt.generic.nb.rolling_std_1d_nb(returns, vol_short, minp=vol_short, ddof=1)
    sl = vbt.generic.nb.rolling_std_1d_nb(returns, vol_long, minp=vol_long, ddof=1)
    vr = np.full(n, np.nan)
    for i in range(n):
        if not np.isnan(ss[i]) and not np.isnan(sl[i]) and sl[i]>0: vr[i]=ss[i]/sl[i]
    regime_wt = regime_weight_nb(vr, vr_low, vr_high, mom_w_low, mom_w_normal, mom_w_high)
    sq = np.empty(n)
    for i in range(n): sq[i] = returns[i]**2 if not np.isnan(returns[i]) else np.nan
    ev = vbt.generic.nb.ewm_mean_1d_nb(sq, ewma_span, minp=1, adjust=True)
    ewma_vol = np.full(n, np.nan)
    for i in range(n):
        if not np.isnan(ev[i]) and ev[i]>0: ewma_vol[i]=np.sqrt(ev[i])*np.sqrt(252)
    vs = vol_scaling_nb(ewma_vol, target_vol, leverage_cap)
    pe = np.ones(n)
    for i in range(1, n):
        d=direction[i-1]; rw=regime_wt[i-1]; v=vs[i-1]; r=returns[i]
        if not (np.isnan(d) or np.isnan(rw) or np.isnan(v) or np.isnan(r)):
            pe[i]=pe[i-1]*(1+r*d*rw*v)
        else: pe[i]=pe[i-1]
    lb=63; dd=np.zeros(n)
    for i in range(n):
        pk=pe[max(0,i-lb+1)]
        for j in range(max(0,i-lb+1),i+1):
            if pe[j]>pk: pk=pe[j]
        dd[i]=1-pe[i]/pk if pk>0 else 0
    dm = drawdown_control_nb(dd, dd_soft, dd_hard, dd_recovery)
    weights = sub_portfolio_weights_nb(direction, regime_wt, vs, dm, n, n_sub)
    return momentum, direction, vr, regime_wt, ewma_vol, vs, dd, dm, weights

@njit(nogil=True)
def composite_signal_nb(c, target_weights, size_arr):
    tw = vbt.pf_nb.select_nb(c, target_weights)
    if np.isnan(tw): return False, False, False, False
    pos=c.last_position[c.col]; price=c.last_val_price[c.col]; value=c.last_value[c.group]
    if value<=0 or price<=0: return False, False, False, False
    tp = tw*value/price; delta=tp-pos
    if abs(delta*price/value)<0.005: return False, False, False, False
    size_arr[c.i,c.col]=abs(delta)
    if pos>=0 and delta>0: return True, False, False, False
    elif pos>0 and delta<0 and tp>=0: return False, True, False, False
    elif pos>=0 and tp<0: return False, False, True, False
    elif pos<=0 and delta<0: return False, False, True, False
    elif pos<0 and delta>0 and tp<=0: return False, False, False, True
    elif pos<=0 and tp>0: return True, False, False, False
    return False, False, False, False"""),
        md("## 4. Indicators"),
        code("""\
COMPOSITE = vbt.IF(
    class_name="CompositeAlpha", short_name="ca",
    input_names=["close", "returns"],
    param_names=["w_short","w_long","vol_short","vol_long","ewma_span","target_vol",
                 "leverage_cap","vr_low","vr_high","mom_w_low","mom_w_normal","mom_w_high",
                 "dd_soft","dd_hard","dd_recovery","n_sub"],
    output_names=["momentum","direction","vol_regime","regime_weight","ewma_vol",
                  "vol_scale","drawdown","dd_multiplier","target_weight"],
).with_apply_func(compute_composite_nb, takes_1d=True,
    w_short=21, w_long=63, vol_short=21, vol_long=252, ewma_span=30,
    target_vol=0.10, leverage_cap=3.0, vr_low=0.8, vr_high=1.2,
    mom_w_low=0.20, mom_w_normal=0.30, mom_w_high=0.50,
    dd_soft=0.12, dd_hard=0.20, dd_recovery=0.10, n_sub=5)
ind = COMPOSITE.run(close, log_returns,
    w_short=21, w_long=63, vol_short=21, vol_long=252, ewma_span=30,
    target_vol=0.10, leverage_cap=3.0, vr_low=0.8, vr_high=1.2,
    mom_w_low=0.20, mom_w_normal=0.30, mom_w_high=0.50,
    dd_soft=0.12, dd_hard=0.20, dd_recovery=0.10, n_sub=5,
    jitted_loop=True, jitted_warmup=True)"""),
        md("### Indicator Inspection"),
        code("""\
ind_df = pd.DataFrame({
    "close": daily["close"].values, "momentum": ind.momentum.values,
    "direction": ind.direction.values, "vol_regime": ind.vol_regime.values,
    "regime_wt": ind.regime_weight.values, "vol_scale": ind.vol_scale.values,
    "drawdown": ind.drawdown.values, "dd_mult": ind.dd_multiplier.values,
    "target_wt": ind.target_weight.values,
}, index=daily.index)
print(f"Shape: {ind_df.shape} | NaN: {ind_df.isna().sum().to_dict()}")
print(f"\\nTarget weight stats:")
print(f"  mean={ind_df['target_wt'].mean():.4f}, std={ind_df['target_wt'].std():.4f}")
print(f"  min={ind_df['target_wt'].min():.4f}, max={ind_df['target_wt'].max():.4f}")
print(f"  zero%: {(ind_df['target_wt']==0).mean()*100:.1f}%")
ind_df.dropna().tail(15)"""),
        md("## 5. Signal Dashboard"),
        code("""\
fig = make_subplots(rows=5, cols=1, shared_xaxes=True, row_heights=[0.25,0.15,0.15,0.2,0.25],
    subplot_titles=("Price","Momentum","Vol Regime","Vol Scale & DD Mult","Target Weight"))
fig.add_trace(go.Scatter(x=daily.index, y=daily["close"], name="Close", line=dict(color="#333")), row=1, col=1)
fig.add_trace(go.Scatter(x=daily.index, y=ind.momentum.values, name="Mom", line=dict(color="#2196F3")), row=2, col=1)
fig.add_hline(y=0, line_color="gray", row=2, col=1)
fig.add_trace(go.Scatter(x=daily.index, y=ind.vol_regime.values, name="VR", line=dict(color="#FF9800")), row=3, col=1)
fig.add_hline(y=0.8, line_dash="dot", line_color="green", row=3, col=1)
fig.add_hline(y=1.2, line_dash="dot", line_color="red", row=3, col=1)
fig.add_trace(go.Scatter(x=daily.index, y=ind.vol_scale.values, name="Vol Scale", line=dict(color="#4CAF50")), row=4, col=1)
fig.add_trace(go.Scatter(x=daily.index, y=ind.dd_multiplier.values, name="DD Mult", line=dict(color="#F44336", dash="dash")), row=4, col=1)
fig.add_trace(go.Scatter(x=daily.index, y=ind.target_weight.values, name="Target Wt", fill="tozeroy", line=dict(color="#9C27B0")), row=5, col=1)
fig.update_layout(height=1200, title="Composite Alpha Signal Dashboard")
fig.show()"""),
        code("""\
fig = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=("Drawdown %","DD Multiplier"))
fig.add_trace(go.Scatter(x=daily.index, y=ind.drawdown.values*100, fill="tozeroy", line=dict(color="#F44336"), name="DD%"), row=1, col=1)
fig.add_hline(y=12, line_dash="dot", line_color="orange", annotation_text="Soft", row=1, col=1)
fig.add_hline(y=20, line_dash="dot", line_color="red", annotation_text="Hard", row=1, col=1)
fig.add_trace(go.Scatter(x=daily.index, y=ind.dd_multiplier.values, fill="tozeroy", line=dict(color="#2196F3"), name="Mult"), row=2, col=1)
fig.update_layout(height=500, title="Drawdown Control State Machine")
fig.show()"""),
        md("## 6. Backtest"),
        code("""\
pf = vbt.Portfolio.from_signals(
    close=daily["close"],
    signal_func_nb=composite_signal_nb,
    signal_args=(vbt.Rep("tw"), vbt.Rep("sz")),
    broadcast_named_args=dict(tw=ind.target_weight.values,
        sz=vbt.RepEval("np.full(wrapper.shape_2d, np.nan)")),
    size=vbt.RepEval("np.full(wrapper.shape_2d, np.nan)"),
    size_type="amount", accumulate=True, upon_opposite_entry="Reverse",
    leverage=2.0, leverage_mode="lazy", fees=0.00035, init_cash=1_000_000, freq="1D")
print(f"Trades: {pf.trades.count()}")"""),
        md("## 7. Results"),
        code("""\
print("PORTFOLIO STATS\\n" + "="*50)
print(pf.stats().to_string())"""),
        code("""\
fig = pf.plot(subplots=["cumulative_returns", "drawdowns", "underwater", "trade_pnl"])
fig.update_layout(height=900, title="Composite FX Alpha: Summary")
fig.show()"""),
        code("""\
if pf.trades.count() > 0:
    fig = make_subplots(rows=2, cols=2,
        subplot_titles=("Trade PnL (%)", "MAE", "MFE", "Running Edge Ratio"))
    pf.trades.plot_pnl(pct_scale=True, fig=fig, add_trace_kwargs=dict(row=1, col=1))
    pf.trades.plot_mae(fig=fig, add_trace_kwargs=dict(row=1, col=2))
    pf.trades.plot_mfe(fig=fig, add_trace_kwargs=dict(row=2, col=1))
    pf.trades.plot_running_edge_ratio(fig=fig, add_trace_kwargs=dict(row=2, col=2))
    fig.update_layout(height=800, showlegend=False)
    fig.show()"""),
        code("""\
rets = pf.returns
monthly = rets.resample("ME").apply(lambda x: (1+x).prod()-1)
df_m = pd.DataFrame({"return": monthly})
df_m["year"] = df_m.index.year; df_m["month"] = df_m.index.month
pivot = df_m.pivot_table(values="return", index="year", columns="month", aggfunc="first")
mn = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
pivot.columns = mn[:len(pivot.columns)]
fig = go.Figure(data=go.Heatmap(z=pivot.values*100, x=pivot.columns.tolist(),
    y=[str(y) for y in pivot.index], colorscale="RdYlGn", texttemplate="%{z:.1f}%", zmid=0))
fig.update_layout(title="Monthly Returns (%)", height=300+len(pivot)*30)
fig.show()"""),
        md("## 8. Target Vol Sweep"),
        code("""\
results = {}
for tv in [0.05, 0.08, 0.10, 0.15]:
    ind_i = COMPOSITE.run(close, log_returns,
        w_short=21, w_long=63, vol_short=21, vol_long=252, ewma_span=30,
        target_vol=tv, leverage_cap=3.0, vr_low=0.8, vr_high=1.2,
        mom_w_low=0.20, mom_w_normal=0.30, mom_w_high=0.50,
        dd_soft=0.12, dd_hard=0.20, dd_recovery=0.10, n_sub=5,
        jitted_loop=True, jitted_warmup=True)
    pf_i = vbt.Portfolio.from_signals(
        close=daily["close"], signal_func_nb=composite_signal_nb,
        signal_args=(vbt.Rep("tw"), vbt.Rep("sz")),
        broadcast_named_args=dict(tw=ind_i.target_weight.values,
            sz=vbt.RepEval("np.full(wrapper.shape_2d, np.nan)")),
        size=vbt.RepEval("np.full(wrapper.shape_2d, np.nan)"),
        size_type="amount", accumulate=True, upon_opposite_entry="Reverse",
        leverage=2.0, leverage_mode="lazy", fees=0.00035, init_cash=1_000_000, freq="1D")
    results[f"TV={tv}"] = pf_i.stats()
print(pd.DataFrame(results).to_string())"""),
        code("""\
fig = go.Figure()
for tv in [0.05, 0.08, 0.10, 0.15]:
    ind_i = COMPOSITE.run(close, log_returns, w_short=21, w_long=63, vol_short=21, vol_long=252,
        ewma_span=30, target_vol=tv, leverage_cap=3.0, vr_low=0.8, vr_high=1.2,
        mom_w_low=0.20, mom_w_normal=0.30, mom_w_high=0.50,
        dd_soft=0.12, dd_hard=0.20, dd_recovery=0.10, n_sub=5,
        jitted_loop=True, jitted_warmup=True)
    pf_i = vbt.Portfolio.from_signals(
        close=daily["close"], signal_func_nb=composite_signal_nb,
        signal_args=(vbt.Rep("tw"), vbt.Rep("sz")),
        broadcast_named_args=dict(tw=ind_i.target_weight.values,
            sz=vbt.RepEval("np.full(wrapper.shape_2d, np.nan)")),
        size=vbt.RepEval("np.full(wrapper.shape_2d, np.nan)"),
        size_type="amount", accumulate=True, upon_opposite_entry="Reverse",
        leverage=2.0, leverage_mode="lazy", fees=0.00035, init_cash=1_000_000, freq="1D")
    fig.add_trace(go.Scatter(x=daily.index, y=(1+pf_i.returns).cumprod().values, name=f"TV={tv}"))
fig.update_layout(height=500, title="Equity Curves by Target Vol")
fig.show()"""),
    ]
    write_nb("08_composite_fx_alpha.ipynb", cells)


if __name__ == "__main__":
    print("Generating self-contained notebooks...")
    nb_01()
    nb_02()
    nb_03()
    nb_04()
    nb_05()
    nb_06()
    nb_07()
    nb_08()
    print("Done! 8 self-contained notebooks created.")
