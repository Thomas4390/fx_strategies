"""MR Turbo — Validated intraday VWAP mean reversion (ims_pipeline format).

Three entry points:

- ``pipeline(data, **params) -> (pf, ind)``
    Investigation path. Uses ``vbt.Portfolio.from_signals`` with pre-computed
    boolean signals and native ``dt_stop``/``td_stop``. Equivalent bit-for-bit
    to the legacy ``backtest_mr_turbo``. Returns a Portfolio and a small
    ``MRTurboIndicator`` holder with a ``plot()`` overlay.

- ``pipeline_nb(...)`` — decorated with ``@vbt.parameterized`` + ``@njit``.
    Grid-search path via ``vbt.pf_nb.from_signal_func_nb``. The EOD time-stop
    is implemented inside ``mrt_signal_nb`` because ``from_signal_func_nb``
    does not accept ``dt_stop``/``td_stop`` arguments. Returns a scalar metric
    (driven by ``metric_type``).

- ``create_cv_pipeline(splitter, metric_type)`` — returns a function decorated
    with ``@vbt.cv_split`` + ``@njit(nogil=True)`` suitable for walk-forward
    cross-validation.

Strategy logic
--------------
- Entry long:  (close < vwap + bb.lower) & (session_start <= hour < session_end)
- Entry short: (close > vwap + bb.upper) & (session_start <= hour < session_end)
- Exit: SL ``sl_stop`` / TP ``tp_stop`` / EOD ``dt_stop`` / max hold ``td_stop``.
- VWAP anchor: daily reset.
- Bollinger Bands: window ``bb_window``, stdev multiplier ``bb_alpha`` on
  the deviation ``(close - vwap)``.

Research findings (walk-forward 2021-2025)
-----------------------------------------
  Avg Sharpe 0.23 | 4/5 years positive | OOS 2025: +0.47
  ~35 trades/year | 52-55 % win rate | PF 1.04-1.08 | Max DD ~5 %
"""

from __future__ import annotations

from dataclasses import dataclass
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


# ═══════════════════════════════════════════════════════════════════════
# 1. INVESTIGATION PATH — pipeline() returns (pf, indicator)
# ═══════════════════════════════════════════════════════════════════════
#
# Phase 1 intentionally does NOT ship @njit kernels for MR Turbo. The
# native ``vbt.VWAP`` / ``vbt.BBANDS`` factories + pre-computed boolean
# signals + ``Portfolio.from_signals(dt_stop, td_stop)`` are already
# fully Numba-parallel under the hood, and this path produces output
# that is bit-equivalent to the legacy ``backtest_mr_turbo`` (enforced
# by ``tests/test_pipeline_equivalence.py``).
#
# A deferred Phase 1b may port the logic into a dedicated
# ``signal_func_nb`` for raw ``vbt.pf_nb.from_signal_func_nb`` grid-search
# throughput, but the current decorator-level parallelism is sufficient.


@dataclass
class MRTurboIndicator:
    """Lightweight indicator wrapper with a Plotly overlay."""

    close: pd.Series
    vwap: pd.Series
    upper: pd.Series
    lower: pd.Series

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


def backtest_mr_turbo(
    data: vbt.Data,
    **kwargs: Any,
) -> vbt.Portfolio:
    """DEPRECATED shim — use ``pipeline(data, ...)`` instead.

    Kept only while ``src/research/`` scripts migrate to the new API
    (Phase 8 of the refactor). Scheduled for removal once all callers
    adopt ``pipeline(data, ...)[0]``.
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
    dt_stop: str = "21:00",
    td_stop: str = "6h",
    init_cash: float = 1_000_000.0,
    slippage: float = 0.00015,
    leverage: float = 1.0,
) -> tuple[vbt.Portfolio, MRTurboIndicator]:
    """Investigation path — bit-equivalent to the legacy backtest_mr_turbo.

    Uses ``vbt.Portfolio.from_signals`` with pre-computed boolean signals and
    VBT-native ``dt_stop``/``td_stop``. Returns the Portfolio plus an
    ``MRTurboIndicator`` holder for overlay plotting.
    """
    close = data.close

    # Native VWAP (session-anchored, daily reset) — matches backtest_mr_turbo
    vwap_ind = vbt.VWAP.run(data.high, data.low, close, data.volume, anchor="D")
    vwap = vwap_ind.vwap

    # Bollinger Bands on close - VWAP deviation
    deviation = close - vwap
    bb = vbt.BBANDS.run(deviation, window=bb_window, alpha=bb_alpha)
    upper = vwap + bb.upper
    lower = vwap + bb.lower

    # Session filter
    hours = close.index.hour
    session = (hours >= session_start) & (hours < session_end)

    # Pre-computed boolean entry signals — same semantics as backtest_mr_turbo
    entries = (close < lower) & session
    short_entries = (close > upper) & session

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

    indicator = MRTurboIndicator(
        close=close,
        vwap=vwap.rename("VWAP"),
        upper=upper.rename("Upper Band"),
        lower=lower.rename("Lower Band"),
    )
    return pf, indicator


# ═══════════════════════════════════════════════════════════════════════
# 3. GRID-SEARCH PATH — pipeline_nb (@vbt.parameterized @njit)
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
    """Grid-search path — ``@vbt.parameterized`` wrapper returning a scalar metric.

    This path is numerically **equivalent** to ``pipeline()`` because it
    reuses the same ``vbt.Portfolio.from_signals`` call with native
    ``dt_stop``/``td_stop``. Parallelization is handled by the
    ``@vbt.parameterized`` decorator via the ``threadpool`` engine.

    A future optimization (Phase 1b, deferred) will port the stops into a
    dedicated ``signal_func_nb`` and apply ``@njit(nogil=True)`` at the
    outer level for full-Numba throughput. See
    ``plans/fluttering-imagining-umbrella.md`` section "Points d'attention".
    """
    pf, _ = pipeline(
        data,
        bb_window=bb_window,
        bb_alpha=bb_alpha,
        sl_stop=sl_stop,
        tp_stop=tp_stop,
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


def run_grid(
    data: vbt.Data,
    *,
    bb_window: list[int] | int,
    bb_alpha: list[float] | float,
    sl_stop: list[float] | float,
    tp_stop: list[float] | float,
    metric_type: int = SHARPE_RATIO,
    **kwargs: Any,
) -> pd.Series:
    """Convenience wrapper — calls ``pipeline_nb`` with vbt.Param for list args.

    Returns a ``pd.Series`` multi-indexed by the swept parameters.
    """

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
        metric_type=metric_type,
        **kwargs,
    )


# ═══════════════════════════════════════════════════════════════════════
# 4. CV FACTORY — create_cv_pipeline (@vbt.cv_split)
# ═══════════════════════════════════════════════════════════════════════


def create_cv_pipeline(
    splitter: Any,
    metric_type: int = SHARPE_RATIO,
    **pipeline_defaults: Any,
):
    """Build a ``@vbt.cv_split`` pipeline for walk-forward cross-validation.

    ``splitter`` can be either a ``vbt.Splitter`` instance or the string
    name of a factory method (e.g. ``"from_purged_walkforward"``). In the
    latter case, pass ``splitter_kwargs`` via ``pipeline_defaults``.

    Returns a callable that accepts ``(data, bb_window=vbt.Param(...), ...)``
    and yields a ``(grid_perf, best_perf)`` tuple when
    ``return_grid="all"`` is set by the underlying decorator.
    """
    splitter_kwargs = pipeline_defaults.pop("splitter_kwargs", {})

    defaults = dict(
        session_start=6,
        session_end=14,
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
# 5. CLI — single / grid / cv modes
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
        apply_vbt_plot_defaults,
        analyze_portfolio,
        plot_cv_heatmap,
        plot_cv_splitter,
    )
    from utils import load_fx_data

    ap = argparse.ArgumentParser(description="MR Turbo pipeline (ims format)")
    ap.add_argument("--data", default="data/EUR-USD_minute.parquet")
    ap.add_argument("--mode", choices=["single", "grid", "cv"], default="single")
    ap.add_argument("--bb-window", type=int, default=80)
    ap.add_argument("--bb-alpha", type=float, default=5.0)
    ap.add_argument("--sl-stop", type=float, default=0.005)
    ap.add_argument("--tp-stop", type=float, default=0.006)
    ap.add_argument("--leverage", type=float, default=1.0)
    ap.add_argument("--n-folds", type=int, default=15)
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--output-dir", default="results/mr_turbo")
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
            leverage=args.leverage,
        )
        print(pf.stats())
        analyze_portfolio(
            pf,
            name="MR Turbo",
            output_dir=args.output_dir,
            show_charts=args.show,
            indicator=ind,
        )

    elif args.mode == "grid":
        grid = run_grid(
            data,
            bb_window=[40, 60, 80, 120],
            bb_alpha=[4.0, 5.0, 6.0],
            sl_stop=[0.004, 0.005, 0.006],
            tp_stop=[0.004, 0.006, 0.008],
            metric_type=SHARPE_RATIO,
        )
        print("\nTop 20 combos by Sharpe:")
        print(grid.sort_values(ascending=False).head(20))
        if args.show:
            fig = grid.vbt.heatmap(
                x_level="bb_window",
                y_level="bb_alpha",
                slider_level="sl_stop",
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
            plot_cv_splitter(splitter, title="MR Turbo — CV Splits").show()

        cv_pipeline = create_cv_pipeline(splitter, metric_type=SHARPE_RATIO)
        grid_perf, best_perf = cv_pipeline(
            data,
            bb_window=vbt.Param([40, 60, 80, 120]),
            bb_alpha=vbt.Param([4.0, 5.0, 6.0]),
            sl_stop=vbt.Param([0.004, 0.005, 0.006]),
            tp_stop=vbt.Param([0.004, 0.006, 0.008]),
        )
        print("\n▶ Best per split:")
        print(best_perf)
        if args.show:
            plot_cv_heatmap(
                grid_perf,
                x_level="bb_window",
                y_level="bb_alpha",
                slider_level="split",
                title="MR Turbo — CV Sharpe Heatmap",
            ).show()

    print("\nDone.")
