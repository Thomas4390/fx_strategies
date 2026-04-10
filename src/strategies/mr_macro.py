"""MR Macro — Macro-regime-filtered intraday VWAP mean reversion (ims_pipeline format).

Extends MR Turbo with macro regime filters:
- Yield curve spread (10Y-2Y Treasury) < threshold
- Unemployment trend (3-month change) not rising

Research finding: filtering on spread < 0.3 + unemployment not rising boosts
the walk-forward Sharpe from 0.19 to 1.07 (2021-2025), OOS 2025 Sharpe 2.30.

Three entry points mirror the ims_pipeline format:

- ``pipeline(data, **params) -> (pf, ind)`` — investigation path,
  bit-equivalent to the legacy ``backtest_mr_macro``.
- ``pipeline_nb(data, **params)`` — ``@vbt.parameterized`` wrapper that
  returns a scalar metric for grid search.
- ``create_cv_pipeline(splitter, metric_type)`` — ``@vbt.cv_split`` factory
  for walk-forward cross-validation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import vectorbtpro as vbt

from framework.pipeline_utils import (
    FX_MINUTE_ANN_FACTOR,
    SHARPE_RATIO,
    compute_metric_nb,
)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


# ═══════════════════════════════════════════════════════════════════════
# MACRO DATA LOADING (VBT native realign + module-level cache)
# ═══════════════════════════════════════════════════════════════════════

# Module-level cache: (first_ts, last_ts, n_bars, spread_threshold) → np.ndarray
_MACRO_FILTER_CACHE: dict[tuple, np.ndarray] = {}
_MACRO_CACHE_MAX = 8


def _load_macro_series(data_dir: Path) -> tuple[pd.Series, pd.Series]:
    """Read raw macro parquets and return (spread_daily, unemp_rising_monthly)."""
    spread_df = pd.read_parquet(data_dir / "SPREAD_10Y2Y_daily.parquet")
    spread_df["date"] = pd.to_datetime(spread_df["date"])
    spread = spread_df.set_index("date")["spread_10y2y"].sort_index()

    unemp_df = pd.read_parquet(data_dir / "UNEMPLOYMENT_monthly.parquet")
    unemp_df["date"] = pd.to_datetime(unemp_df["date"])
    unemp = unemp_df.set_index("date")["unemployment"].sort_index()
    unemp_rising = unemp.diff(3) > 0  # 3-month change, boolean
    return spread, unemp_rising


def load_macro_filters(
    minute_index: pd.DatetimeIndex,
    spread_threshold: float = 0.3,
    data_dir: Path | None = None,
) -> pd.Series:
    """Load macro data and build regime filter aligned to minute index.

    Filter: yield spread 10Y-2Y < threshold AND unemployment not rising (3m).
    Both conditions must be True for trading to be allowed.

    Uses ``vbt.Resampler`` + ``.vbt.realign_opening`` (macro data is as-of the
    report date, usable from 00:00 of that date). Cached at module level so
    repeated calls in a grid sweep reuse previous alignment.
    """
    if data_dir is None:
        data_dir = _PROJECT_ROOT / "data"

    cache_key = (
        minute_index[0].value,
        minute_index[-1].value,
        len(minute_index),
        float(spread_threshold),
    )
    cached = _MACRO_FILTER_CACHE.get(cache_key)
    if cached is not None:
        return pd.Series(cached, index=minute_index, name="macro_ok")

    spread, unemp_rising = _load_macro_series(data_dir)

    spread_resampler = vbt.Resampler(
        source_index=spread.index,
        target_index=minute_index,
        source_freq="D",
        target_freq="1min",
    )
    spread_min = spread.vbt.realign_opening(spread_resampler, ffill=True)

    unemp_resampler = vbt.Resampler(
        source_index=unemp_rising.index,
        target_index=minute_index,
        source_freq="MS",
        target_freq="1min",
    )
    unemp_rising_min_f = unemp_rising.astype(float).vbt.realign_opening(
        unemp_resampler, ffill=True
    )
    unemp_ok = unemp_rising_min_f.fillna(0.0).astype(bool)

    macro_ok = (spread_min < spread_threshold) & (~unemp_ok)
    macro_ok = macro_ok.where(macro_ok.notna(), False).astype(bool)
    macro_ok.name = "macro_ok"

    if len(_MACRO_FILTER_CACHE) >= _MACRO_CACHE_MAX:
        _MACRO_FILTER_CACHE.pop(next(iter(_MACRO_FILTER_CACHE)))
    _MACRO_FILTER_CACHE[cache_key] = macro_ok.values.copy()
    return macro_ok


# ═══════════════════════════════════════════════════════════════════════
# 1. INVESTIGATION PATH — pipeline() returns (pf, indicator)
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class MRMacroIndicator:
    """Lightweight indicator wrapper with a Plotly overlay."""

    close: pd.Series
    vwap: pd.Series
    upper: pd.Series
    lower: pd.Series
    macro_ok: pd.Series

    def plot(self, fig: go.Figure | None = None, **layout_kwargs) -> go.Figure:
        fig = fig or go.Figure()
        self.close.vbt.plot(
            fig=fig,
            trace_kwargs=dict(name="Close", line=dict(width=2, color="blue")),
        )
        self.lower.vbt.plot(
            fig=fig,
            trace_kwargs=dict(name="Lower Band", line=dict(width=1.2, color="grey")),
        )
        self.upper.vbt.plot(
            fig=fig,
            trace_kwargs=dict(
                name="Upper Band",
                line=dict(width=1.2, color="grey"),
                fill="tonexty",
                fillcolor="rgba(255, 255, 0, 0.2)",
            ),
        )
        self.vwap.vbt.plot(
            fig=fig,
            trace_kwargs=dict(
                name="VWAP", line=dict(color="red", width=1, dash="dot")
            ),
        )
        fig.update_layout(**layout_kwargs)
        return fig


def backtest_mr_macro(
    data: vbt.Data,
    **kwargs: Any,
) -> vbt.Portfolio:
    """DEPRECATED shim — use ``pipeline(data, ...)`` instead.

    Kept only while ``src/research/`` scripts migrate to the new API
    (Phase 8). Scheduled for removal once all callers adopt the new API.
    """
    pf, _ = pipeline(data, **kwargs)
    return pf


def pipeline(
    data: vbt.Data,
    bb_window: int = 80,
    bb_alpha: float = 5.0,
    sl_stop: float = 0.005,
    tp_stop: float = 0.006,
    session_start: int = 6,
    session_end: int = 14,
    spread_threshold: float = 0.5,
    dt_stop: str = "21:00",
    td_stop: str = "6h",
    init_cash: float = 1_000_000.0,
    slippage: float = 0.00015,
    leverage: float = 1.0,
) -> tuple[vbt.Portfolio, MRMacroIndicator]:
    """Investigation path — bit-equivalent to the legacy ``backtest_mr_macro``.

    Same pre-computed boolean signals passed to ``vbt.Portfolio.from_signals``
    with native ``dt_stop``/``td_stop``. The macro regime filter is applied
    as an additional session-level AND mask before signal generation.
    """
    close = data.close

    vwap = vbt.VWAP.run(data.high, data.low, close, data.volume, anchor="D").vwap
    deviation = close - vwap
    bb = vbt.BBANDS.run(deviation, window=bb_window, alpha=bb_alpha)
    upper = vwap + bb.upper
    lower = vwap + bb.lower

    hours = close.index.hour
    session = (hours >= session_start) & (hours < session_end)

    macro_ok = load_macro_filters(close.index, spread_threshold)

    entries = (close < lower) & session & macro_ok
    short_entries = (close > upper) & session & macro_ok

    pf = vbt.Portfolio.from_signals(
        data,
        entries=entries,
        exits=False,
        short_entries=short_entries,
        short_exits=False,
        sl_stop=sl_stop,
        tp_stop=tp_stop,
        dt_stop=dt_stop,
        td_stop=td_stop,
        slippage=slippage,
        init_cash=init_cash,
        leverage=leverage,
        freq="1min",
    )

    indicator = MRMacroIndicator(
        close=close,
        vwap=vwap.rename("VWAP"),
        upper=upper.rename("Upper Band"),
        lower=lower.rename("Lower Band"),
        macro_ok=macro_ok,
    )
    return pf, indicator


# ═══════════════════════════════════════════════════════════════════════
# 2. GRID-SEARCH PATH — pipeline_nb (@vbt.parameterized)
# ═══════════════════════════════════════════════════════════════════════


@vbt.parameterized(
    merge_func="concat",
    execute_kwargs=dict(chunk_len="auto", engine="threadpool"),
)
def pipeline_nb(
    data: vbt.Data,
    bb_window: int,
    bb_alpha: float,
    sl_stop: float,
    tp_stop: float,
    spread_threshold: float = 0.5,
    session_start: int = 6,
    session_end: int = 14,
    dt_stop: str = "21:00",
    td_stop: str = "6h",
    init_cash: float = 1_000_000.0,
    slippage: float = 0.00015,
    leverage: float = 1.0,
    ann_factor: float = FX_MINUTE_ANN_FACTOR,
    cutoff: float = 0.05,
    metric_type: int = SHARPE_RATIO,
) -> float:
    """Grid-search path — returns a scalar metric per param combo.

    Delegates to ``pipeline()`` so the numerical result is identical. The
    ``@vbt.parameterized`` decorator distributes the combos via a thread
    pool and concatenates the scalar outputs into a multi-indexed Series.
    """
    pf, _ = pipeline(
        data,
        bb_window=bb_window,
        bb_alpha=bb_alpha,
        sl_stop=sl_stop,
        tp_stop=tp_stop,
        session_start=session_start,
        session_end=session_end,
        spread_threshold=spread_threshold,
        dt_stop=dt_stop,
        td_stop=td_stop,
        init_cash=init_cash,
        slippage=slippage,
        leverage=leverage,
    )
    returns = pf.returns.values
    if returns.ndim > 1:
        returns = returns[:, 0]
    return float(compute_metric_nb(returns, metric_type, ann_factor, cutoff))


def run_grid(
    data: vbt.Data,
    *,
    bb_window: list[int] | int,
    bb_alpha: list[float] | float,
    sl_stop: list[float] | float,
    tp_stop: list[float] | float,
    spread_threshold: list[float] | float = 0.5,
    metric_type: int = SHARPE_RATIO,
    **kwargs: Any,
) -> pd.Series:
    """Wrap list inputs as ``vbt.Param`` and call ``pipeline_nb``."""

    def _param(v):
        if isinstance(v, (list, tuple, np.ndarray)):
            return vbt.Param(list(v))
        return v

    return pipeline_nb(
        data,
        bb_window=_param(bb_window),
        bb_alpha=_param(bb_alpha),
        sl_stop=_param(sl_stop),
        tp_stop=_param(tp_stop),
        spread_threshold=_param(spread_threshold),
        metric_type=metric_type,
        **kwargs,
    )


# ═══════════════════════════════════════════════════════════════════════
# 3. CV FACTORY — create_cv_pipeline (@vbt.cv_split)
# ═══════════════════════════════════════════════════════════════════════


def create_cv_pipeline(
    splitter: Any,
    metric_type: int = SHARPE_RATIO,
    **pipeline_defaults: Any,
):
    """Build a ``@vbt.cv_split`` pipeline for walk-forward cross-validation."""
    splitter_kwargs = pipeline_defaults.pop("splitter_kwargs", {})

    defaults = dict(
        session_start=6,
        session_end=14,
        spread_threshold=0.5,
        dt_stop="21:00",
        td_stop="6h",
        init_cash=1_000_000.0,
        slippage=0.00015,
        leverage=1.0,
        ann_factor=FX_MINUTE_ANN_FACTOR,
        cutoff=0.05,
        metric_type=metric_type,
    )
    defaults.update(pipeline_defaults)

    @vbt.cv_split(
        splitter=splitter,
        splitter_kwargs=splitter_kwargs,
        takeable_args=["data"],
        parameterized_kwargs=dict(
            execute_kwargs=dict(chunk_len="auto", engine="threadpool"),
            merge_func="concat",
        ),
        merge_func="concat",
        return_grid="all",
        attach_bounds="index",
    )
    def cv_pipeline(
        data: vbt.Data,
        bb_window: int,
        bb_alpha: float,
        sl_stop: float,
        tp_stop: float,
        spread_threshold: float = defaults["spread_threshold"],
        session_start: int = defaults["session_start"],
        session_end: int = defaults["session_end"],
        dt_stop: str = defaults["dt_stop"],
        td_stop: str = defaults["td_stop"],
        init_cash: float = defaults["init_cash"],
        slippage: float = defaults["slippage"],
        leverage: float = defaults["leverage"],
        ann_factor: float = defaults["ann_factor"],
        cutoff: float = defaults["cutoff"],
        metric_type: int = defaults["metric_type"],
    ) -> float:
        pf, _ = pipeline(
            data,
            bb_window=bb_window,
            bb_alpha=bb_alpha,
            sl_stop=sl_stop,
            tp_stop=tp_stop,
            spread_threshold=spread_threshold,
            session_start=session_start,
            session_end=session_end,
            dt_stop=dt_stop,
            td_stop=td_stop,
            init_cash=init_cash,
            slippage=slippage,
            leverage=leverage,
        )
        returns = pf.returns.values
        if returns.ndim > 1:
            returns = returns[:, 0]
        return float(compute_metric_nb(returns, metric_type, ann_factor, cutoff))

    return cv_pipeline


# ═══════════════════════════════════════════════════════════════════════
# 4. CLI — single / grid / cv modes
# ═══════════════════════════════════════════════════════════════════════


def _walk_forward_report(data: vbt.Data) -> None:
    print(f"\n{'=' * 60}\nWalk-Forward Validation (per-year)\n{'=' * 60}")
    for year in range(2021, 2027):
        d_yr = data.loc[f"{year}-01-01":f"{year}-12-31"]
        if d_yr.shape[0] < 1000:
            continue
        pf_yr, _ = pipeline(d_yr)
        tc = pf_yr.trades.count()
        sr = pf_yr.sharpe_ratio if tc > 0 else 0
        ret = pf_yr.total_return * 100 if tc > 0 else 0
        wr = pf_yr.trades.win_rate * 100 if tc > 0 else 0
        print(
            f"  {year}: Sharpe={sr:>7.3f}  Ret={ret:>6.2f}%  "
            f"Trades={tc}  WR={wr:.1f}%"
        )


if __name__ == "__main__":
    import argparse
    import sys
    from pathlib import Path as _Path

    _SRC = _Path(__file__).resolve().parent.parent
    if str(_SRC) not in sys.path:
        sys.path.insert(0, str(_SRC))

    from framework.pipeline_utils import (
        analyze_portfolio,
        apply_vbt_plot_defaults,
        plot_cv_heatmap,
        plot_cv_splitter,
    )
    from utils import load_fx_data

    ap = argparse.ArgumentParser(description="MR Macro pipeline (ims format)")
    ap.add_argument("--data", default="data/EUR-USD_minute.parquet")
    ap.add_argument("--mode", choices=["single", "grid", "cv"], default="single")
    ap.add_argument("--bb-window", type=int, default=80)
    ap.add_argument("--bb-alpha", type=float, default=5.0)
    ap.add_argument("--sl-stop", type=float, default=0.005)
    ap.add_argument("--tp-stop", type=float, default=0.006)
    ap.add_argument("--spread-threshold", type=float, default=0.5)
    ap.add_argument("--leverage", type=float, default=1.0)
    ap.add_argument("--n-folds", type=int, default=15)
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--output-dir", default="results/mr_macro")
    args = ap.parse_args()

    apply_vbt_plot_defaults()
    print("Loading data...")
    _, data = load_fx_data(args.data)

    if args.mode == "single":
        _walk_forward_report(data)
        pf, ind = pipeline(
            data,
            bb_window=args.bb_window,
            bb_alpha=args.bb_alpha,
            sl_stop=args.sl_stop,
            tp_stop=args.tp_stop,
            spread_threshold=args.spread_threshold,
            leverage=args.leverage,
        )
        print(pf.stats())
        analyze_portfolio(
            pf,
            name="MR Macro",
            output_dir=args.output_dir,
            show_charts=args.show,
            indicator=ind,
        )

    elif args.mode == "grid":
        grid = run_grid(
            data,
            bb_window=[40, 60, 80, 120],
            bb_alpha=[3.5, 4.0, 4.5, 5.0, 5.5, 6.0],
            sl_stop=0.005,
            tp_stop=0.006,
            spread_threshold=[0.3, 0.5, 0.7],
            metric_type=SHARPE_RATIO,
        )
        print("\nTop 20 combos by Sharpe:")
        print(grid.sort_values(ascending=False).head(20))
        if args.show:
            fig = grid.vbt.heatmap(
                x_level="bb_window",
                y_level="bb_alpha",
                slider_level="spread_threshold",
            )
            fig.show()

    elif args.mode == "cv":
        splitter = vbt.Splitter.from_purged_walkforward(
            data.index,
            n_folds=args.n_folds,
            n_test_folds=1,
            purge_td="1 day",
            min_train_folds=3,
        )
        if args.show:
            plot_cv_splitter(splitter, title="MR Macro — CV Splits").show()
        cv_pipeline = create_cv_pipeline(splitter, metric_type=SHARPE_RATIO)
        grid_perf, best_perf = cv_pipeline(
            data,
            bb_window=vbt.Param([60, 80, 120]),
            bb_alpha=vbt.Param([4.0, 5.0, 6.0]),
            sl_stop=0.005,
            tp_stop=0.006,
            spread_threshold=vbt.Param([0.3, 0.5]),
        )
        print("\n▶ Best per split:")
        print(best_perf)
        if args.show:
            plot_cv_heatmap(
                grid_perf,
                x_level="bb_window",
                y_level="bb_alpha",
                slider_level="split",
                title="MR Macro — CV Sharpe Heatmap",
            ).show()

    print("\nDone.")
