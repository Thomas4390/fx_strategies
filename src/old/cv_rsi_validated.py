#!/usr/bin/env python
"""
Multi-Strategy Walk-Forward CV — VBT Pro Native Pipeline

Architecture:
- splitter.apply() + @vbt.parameterized for idiomatic VBT CV
- Multi-metric returns (Sharpe, Sortino, Calmar, PF, max DD, return, trades)
- prange batch for ultra-fast full-sample sweeps
- slider_labels with dates on all heatmaps
- Fullscreen browser plots
"""

import os
import sys
import time
import warnings

import numba
import numpy as np
import pandas as pd
import vectorbtpro as vbt
from numba import njit, prange

warnings.filterwarnings("ignore")
os.environ["NUMBA_DISABLE_PERFORMANCE_WARNINGS"] = "1"

sys.path.insert(0, os.path.dirname(__file__))
from utils import load_fx_data

RESULTS_DIR = "results/exploration/cv"
SLIPPAGE = 0.00008
FEES = 0.0001
INIT_CASH = 1_000_000


def fullscreen(fig, title="", height=900):
    fig.update_layout(
        width=None,
        height=height,
        autosize=True,
        title={"text": title, "font": {"size": 22}, "x": 0.5, "xanchor": "center"},
        margin={"l": 40, "r": 40, "t": 80, "b": 40},
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "x": 0.5,
            "xanchor": "center",
        },
        template="plotly_white",
    )
    return fig


def show_save(fig, name, title, h=800):
    fig = fullscreen(fig, title, h)
    fig.write_html(f"{RESULTS_DIR}/{name}.html")
    fig.show(renderer="browser")
    print(f"  {name} → browser")


def split_date_labels(splitter, set_label="train"):
    """Build date range labels for each split."""
    bounds = splitter.get_bounds(index_bounds=True)
    labels = []
    for i in range(len(splitter.splits)):
        row = bounds.loc[(i, set_label)]
        s = str(pd.Timestamp(row["start"]).date())
        e = str(pd.Timestamp(row["end"]).date())
        labels.append(f"{s} → {e}")
    return labels


# ═══════════════════════════════════════════════════════════════════════
# 1. NUMBA PRANGE KERNELS (fast full-sample sweeps)
# ═══════════════════════════════════════════════════════════════════════


@njit(nogil=True)
def rsi_signal_nb(c, rsi_arr, entry_th_arr, exit_th_arr):
    rsi_val = vbt.pf_nb.select_nb(c, rsi_arr)
    entry_th = vbt.pf_nb.select_nb(c, entry_th_arr)
    exit_th = vbt.pf_nb.select_nb(c, exit_th_arr)
    if np.isnan(rsi_val) or c.i == 0:
        return False, False, False, False
    prev = rsi_arr[c.i - 1, c.col] if rsi_arr.ndim > 1 else rsi_arr[c.i - 1]
    if np.isnan(prev):
        return False, False, False, False
    return (
        prev >= entry_th and rsi_val < entry_th,
        prev <= 50.0 and rsi_val > 50.0,
        prev <= exit_th and rsi_val > exit_th,
        prev >= 50.0 and rsi_val < 50.0,
    )


@njit(nogil=True)
def _run_rsi_sim(close_arr, rsi_window, entry_th, fees, slippage):
    ts = close_arr.shape
    rsi = vbt.indicators.nb.rsi_1d_nb(close_arr[:, 0], window=rsi_window)
    gl = np.full(ts[1], 1)
    so = vbt.pf_nb.from_signal_func_nb(
        target_shape=ts,
        group_lens=gl,
        init_cash=1000000.0,
        cash_sharing=False,
        close=close_arr,
        signal_func_nb=rsi_signal_nb,
        signal_args=(
            rsi.reshape(-1, 1),
            np.full(ts, entry_th),
            np.full(ts, 100.0 - entry_th),
        ),
        slippage=np.full(ts, slippage),
        fees=np.full(ts, fees),
        post_segment_func_nb=vbt.pf_nb.save_post_segment_func_nb,
        in_outputs=vbt.pf_nb.init_FSInOutputs_nb(ts, gl, cash_sharing=False),
    )
    return vbt.ret_nb.sharpe_ratio_nb(returns=so.in_outputs.returns, ann_factor=252.0)[
        0
    ]


@njit(parallel=True, nogil=True)
def rsi_batch(close_arr, rsi_windows, entry_ths, fees, slippage):
    n_rw, n_eth = len(rsi_windows), len(entry_ths)
    results = np.empty(n_rw * n_eth)
    for idx in prange(n_rw * n_eth):
        results[idx] = _run_rsi_sim(
            close_arr, rsi_windows[idx // n_eth], entry_ths[idx % n_eth], fees, slippage
        )
    return results


def batch_to_series(results, names_vals):
    names = [nv[0] for nv in names_vals]
    vals = [nv[1] for nv in names_vals]
    return pd.Series(results, index=pd.MultiIndex.from_product(vals, names=names))


# ═══════════════════════════════════════════════════════════════════════
# 2. VBT NATIVE CV PIPELINE (multi-metric, splitter.apply)
# ═══════════════════════════════════════════════════════════════════════


def rsi_multi_metrics(close, rsi_window, entry_th):
    """Single backtest returning multiple metrics as a Series."""
    exit_th = 100 - entry_th
    rsi = vbt.RSI.run(close, window=rsi_window)
    el = rsi.rsi_crossed_below(entry_th)
    xl = rsi.rsi_crossed_above(50)
    es = rsi.rsi_crossed_above(exit_th)
    xs = rsi.rsi_crossed_below(50)
    pf = vbt.PF.from_signals(
        close,
        long_entries=el,
        long_exits=xl,
        short_entries=es,
        short_exits=xs,
        slippage=SLIPPAGE,
        fees=FEES,
        init_cash=INIT_CASH,
        freq="1d",
    )
    stats = pf.stats()
    return pd.Series(
        {
            "sharpe": pf.sharpe_ratio,
            "sortino": pf.sortino_ratio,
            "calmar": pf.calmar_ratio,
            "total_return": pf.total_return,
            "max_dd": pf.max_drawdown,
            "profit_factor": stats.get("Profit Factor", np.nan),
            "win_rate": stats.get("Win Rate [%]", np.nan),
            "trades": stats.get("Total Trades", 0),
        }
    )


# Create parameterized wrapper (once)
param_rsi_multi = vbt.parameterized(rsi_multi_metrics, merge_func="concat")
param_rsi_sharpe = vbt.parameterized(
    lambda close, rsi_window, entry_th: rsi_multi_metrics(close, rsi_window, entry_th)[
        "sharpe"
    ],
    merge_func="concat",
)


def run_native_cv(close, rsi_windows, entry_ths, n_splits=6, window_length=600):
    """VBT native walk-forward CV with multi-metric output.

    Returns:
        splitter: vbt.Splitter instance
        train_metrics: multi-metric Series (split × params × metric)
        test_metrics: multi-metric Series
        train_sharpe: sharpe-only Series (for heatmaps)
        test_sharpe: sharpe-only Series
    """
    splitter = vbt.Splitter.from_n_rolling(
        close.index,
        n=n_splits,
        length=window_length,
        split=0.5,
        set_labels=["train", "test"],
    )

    print(f"  Splitter: {n_splits} splits × {window_length} bars")

    # Multi-metric CV via splitter.apply
    t0 = time.time()
    train_metrics = splitter.apply(
        param_rsi_multi,
        vbt.Takeable(close),
        vbt.Param(rsi_windows),
        vbt.Param(entry_ths),
        set_="train",
        merge_func="concat",
        execute_kwargs={"show_progress": True},
    )
    t1 = time.time()
    print(f"  Train: {t1 - t0:.3f}s, {train_metrics.shape[0]} results")

    t0 = time.time()
    test_metrics = splitter.apply(
        param_rsi_multi,
        vbt.Takeable(close),
        vbt.Param(rsi_windows),
        vbt.Param(entry_ths),
        set_="test",
        merge_func="concat",
        execute_kwargs={"show_progress": True},
    )
    t1 = time.time()
    print(f"  Test:  {t1 - t0:.3f}s")

    # Extract Sharpe for heatmaps
    metric_level = (
        train_metrics.index.names.index(None)
        if None in train_metrics.index.names
        else -1
    )
    train_sharpe = train_metrics.xs("sharpe", level=metric_level)
    test_sharpe = test_metrics.xs("sharpe", level=metric_level)

    return splitter, train_metrics, test_metrics, train_sharpe, test_sharpe


# ═══════════════════════════════════════════════════════════════════════
# 3. BEST PARAM SELECTION + OOS VALIDATION
# ═══════════════════════════════════════════════════════════════════════


def select_best_and_validate(
    splitter, train_sharpe, test_sharpe, train_metrics, test_metrics
):
    """Select best params per split on train, validate on test."""
    n_splits = len(splitter.splits)
    metric_level = (
        train_metrics.index.names.index(None)
        if None in train_metrics.index.names
        else -1
    )

    rows = []
    for split_i in range(n_splits):
        tr = train_sharpe.xs(split_i, level="split")
        best_idx = tr.idxmax()
        rw, eth = best_idx

        te = test_sharpe.xs(split_i, level="split")
        test_val = te.loc[best_idx] if best_idx in te.index else np.nan

        # Get all metrics for best params
        tr_all = (
            train_metrics.xs(split_i, level="split")
            .xs(rw, level="rsi_window")
            .xs(eth, level="entry_th")
        )
        te_all = (
            test_metrics.xs(split_i, level="split")
            .xs(rw, level="rsi_window")
            .xs(eth, level="entry_th")
        )

        row = {"split": split_i, "rsi_window": rw, "entry_th": eth}
        for metric in [
            "sharpe",
            "sortino",
            "calmar",
            "total_return",
            "max_dd",
            "profit_factor",
            "win_rate",
            "trades",
        ]:
            row[f"train_{metric}"] = (
                tr_all.get(metric, np.nan) if metric in tr_all.index else np.nan
            )
            row[f"test_{metric}"] = (
                te_all.get(metric, np.nan) if metric in te_all.index else np.nan
            )
        rows.append(row)

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("MULTI-STRATEGY CV — VBT Pro Native + prange")
    print(f"Numba threads: {numba.config.NUMBA_NUM_THREADS}")
    print("=" * 60)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    _, data = load_fx_data()
    close_d = data.resample("1d").close.dropna()
    arr_d = vbt.to_2d_array(close_d)
    print(
        f"Daily: {len(close_d)} bars ({close_d.index[0].date()} → {close_d.index[-1].date()})"
    )

    rsi_ws = list(range(3, 35, 2))
    entry_ts = [float(x) for x in range(15, 46, 2)]

    # ── A. PRANGE FULL-SAMPLE SWEEP ──────────────────────────────
    print("\n" + "=" * 60)
    print("A. FULL-SAMPLE SWEEP (prange multicore)")
    print("=" * 60)
    rsi_ws_np = np.array(rsi_ws, dtype=np.int64)
    entry_ts_np = np.array(entry_ts)
    _run_rsi_sim(arr_d, 14, 30.0, FEES, SLIPPAGE)  # warmup
    rsi_batch(arr_d, rsi_ws_np[:1], entry_ts_np[:1], FEES, SLIPPAGE)

    t0 = time.time()
    full_res = rsi_batch(arr_d, rsi_ws_np, entry_ts_np, FEES, SLIPPAGE)
    print(f"  {len(full_res)} combos in {time.time() - t0:.3f}s (prange)")
    full_sr = batch_to_series(
        full_res, [("rsi_window", rsi_ws), ("entry_th", entry_ts)]
    )
    best = full_sr.idxmax()
    print(
        f"  Best: RSI({best[0]}) {best[1]}/{100 - best[1]} → Sharpe {full_sr.max():.4f}"
    )

    # ── B. NATIVE CV (multi-metric) ──────────────────────────────
    print("\n" + "=" * 60)
    print("B. VBT NATIVE CV (multi-metric, splitter.apply)")
    print("=" * 60)
    splitter, train_m, test_m, train_sh, test_sh = run_native_cv(
        close_d,
        rsi_ws,
        entry_ts,
        n_splits=6,
        window_length=600,
    )

    # ── C. BEST PARAMS + OOS VALIDATION ──────────────────────────
    print("\n" + "=" * 60)
    print("C. BEST PARAMS + OOS VALIDATION")
    print("=" * 60)
    best_df = select_best_and_validate(splitter, train_sh, test_sh, train_m, test_m)

    print(
        best_df[
            [
                "split",
                "rsi_window",
                "entry_th",
                "train_sharpe",
                "test_sharpe",
                "train_sortino",
                "test_sortino",
                "train_total_return",
                "test_total_return",
                "train_max_dd",
                "test_max_dd",
                "train_profit_factor",
                "test_profit_factor",
                "train_trades",
                "test_trades",
            ]
        ]
        .round(4)
        .to_string(index=False)
    )

    print(
        f"\n  Train Sharpe: mean={best_df['train_sharpe'].mean():.4f} min={best_df['train_sharpe'].min():.4f}"
    )
    print(
        f"  Test Sharpe:  mean={best_df['test_sharpe'].mean():.4f} min={best_df['test_sharpe'].min():.4f}"
    )
    print(f"  Test Sortino: mean={best_df['test_sortino'].mean():.4f}")
    print(f"  Test PF:      mean={best_df['test_profit_factor'].mean():.4f}")

    # ── D. HEATMAPS ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("D. HEATMAPS → browser")
    print("=" * 60)

    train_labels = split_date_labels(splitter, "train")
    test_labels = split_date_labels(splitter, "test")

    # 1. Full-sample 2D Sharpe
    show_save(
        full_sr.vbt.heatmap(
            x_level="rsi_window",
            y_level="entry_th",
            trace_kwargs={"colorscale": "RdYlGn", "zmid": 0},
        ),
        "full_sample_sharpe",
        "RSI Full Sample — Sharpe (prange 32-core)",
    )

    # 2. Splitter visualization
    fig_split = splitter.plots()
    show_save(fig_split, "splitter_plot", "Walk-Forward Splits", h=400)

    # 3. Train Sharpe — slider by split (dates)
    show_save(
        train_sh.vbt.heatmap(
            x_level="rsi_window",
            y_level="entry_th",
            slider_level="split",
            slider_labels=train_labels,
            trace_kwargs={"colorscale": "RdYlGn", "zmid": 0},
        ),
        "cv_train_sharpe",
        "CV Train Sharpe — per split (dates)",
    )

    # 4. Test Sharpe — slider by split (dates)
    show_save(
        test_sh.vbt.heatmap(
            x_level="rsi_window",
            y_level="entry_th",
            slider_level="split",
            slider_labels=test_labels,
            trace_kwargs={"colorscale": "RdYlGn", "zmid": 0},
        ),
        "cv_test_sharpe",
        "CV Test Sharpe — per split (dates)",
    )

    # 5. Multi-metric train — slider by metric
    # Rename the unnamed metric level, then aggregate across splits
    train_m_named = train_m.copy()
    train_m_named.index = train_m_named.index.set_names("metric", level=-1)
    train_mean = train_m_named.groupby(
        level=["rsi_window", "entry_th", "metric"]
    ).mean()
    show_save(
        train_mean.vbt.heatmap(
            x_level="rsi_window",
            y_level="entry_th",
            slider_level="metric",
            trace_kwargs={"colorscale": "RdYlGn", "zmid": 0},
        ),
        "cv_train_multi_metric",
        "CV Train Mean — slider by metric",
    )

    # 6. Test multi-metric — slider by metric
    test_m_named = test_m.copy()
    test_m_named.index = test_m_named.index.set_names("metric", level=-1)
    test_mean = test_m_named.groupby(level=["rsi_window", "entry_th", "metric"]).mean()
    show_save(
        test_mean.vbt.heatmap(
            x_level="rsi_window",
            y_level="entry_th",
            slider_level="metric",
            trace_kwargs={"colorscale": "RdYlGn", "zmid": 0},
        ),
        "cv_test_multi_metric",
        "CV Test Mean — slider by metric",
    )

    # 7. Combined train vs test Sharpe — slider by set
    combined = pd.concat([train_sh, test_sh], keys=["train", "test"], names=["set"])
    show_save(
        combined.vbt.heatmap(
            x_level="rsi_window",
            y_level="entry_th",
            slider_level="set",
            trace_kwargs={"colorscale": "RdYlGn", "zmid": 0},
        ),
        "cv_combined_sharpe",
        "CV Sharpe — slider train vs test",
    )

    # ── E. INSPECT BEST ──────────────────────────────────────────
    print("\n" + "=" * 60)
    rw, eth = int(best[0]), int(best[1])
    print(f"E. INSPECT BEST: RSI({rw}) {eth}/{100 - eth}")
    print("=" * 60)

    rsi_ind = vbt.RSI.run(close_d, window=rw)
    pf = vbt.PF.from_signals(
        close_d,
        long_entries=rsi_ind.rsi_crossed_below(eth),
        long_exits=rsi_ind.rsi_crossed_above(50),
        short_entries=rsi_ind.rsi_crossed_above(100 - eth),
        short_exits=rsi_ind.rsi_crossed_below(50),
        slippage=SLIPPAGE,
        fees=FEES,
        init_cash=INIT_CASH,
        freq="1d",
    )
    print(pf.stats().to_string())

    trades = pf.trades.records_readable
    if len(trades) > 0:
        pnl, ret, w = trades["PnL"], trades["Return"], trades["PnL"] > 0
        print(f"\n  Trades: {len(trades)}, Win: {w.sum()} ({w.mean():.1%})")
        print(f"  Avg win: {ret[w].mean():.4%}, Avg loss: {ret[~w].mean():.4%}")
        t = trades.copy()
        t["year"] = pd.to_datetime(t["Entry Index"]).dt.year
        print(
            t.groupby("year")
            .agg(
                n=("PnL", "count"),
                pnl=("PnL", "sum"),
                wr=("PnL", lambda x: (x > 0).mean()),
            )
            .round(4)
            .to_string()
        )

    show_save(
        pf.plot(
            subplots=["cumulative_returns", "drawdowns", "underwater", "trade_pnl"]
        ),
        "best_portfolio",
        f"RSI({rw}) {eth}/{100 - eth} — Best Portfolio",
        h=1000,
    )

    # Save best_df
    best_df.to_csv(f"{RESULTS_DIR}/cv_best_params.csv", index=False)

    print(f"\n{'=' * 60}")
    print(f"DONE — {RESULTS_DIR}/")
    print(f"{'=' * 60}")
