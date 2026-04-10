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
    make_execute_kwargs,
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
        """Draw Close + VWAP + filled BB band as plain ``go.Scatter`` traces.

        Lower and Upper bands are added back-to-back so ``fill="tonexty"``
        on the Upper trace reliably fills between the two bands (Plotly
        fills to the trace IMMEDIATELY before in the traces list).
        """
        fig = fig or go.Figure()
        # Close (drawn first so it is UNDERNEATH the band fill, still visible)
        fig.add_trace(
            go.Scatter(
                x=self.close.index, y=self.close.values,
                mode="lines", name="Close",
                line=dict(width=2, color="royalblue"),
            )
        )
        # VWAP
        fig.add_trace(
            go.Scatter(
                x=self.vwap.index, y=self.vwap.values,
                mode="lines", name="VWAP",
                line=dict(color="crimson", width=1, dash="dot"),
            )
        )
        # Lower band — must be added directly BEFORE the upper band so
        # `fill="tonexty"` on the upper resolves to this trace.
        fig.add_trace(
            go.Scatter(
                x=self.lower.index, y=self.lower.values,
                mode="lines", name="Lower Band",
                line=dict(width=1.1, color="rgba(110,110,110,0.85)"),
                showlegend=True,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=self.upper.index, y=self.upper.values,
                mode="lines", name="Upper Band",
                line=dict(width=1.1, color="rgba(110,110,110,0.85)"),
                fill="tonexty",
                fillcolor="rgba(255,215,0,0.18)",
                showlegend=True,
            )
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
    execute_kwargs=make_execute_kwargs("MR Turbo grid"),
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
            execute_kwargs=make_execute_kwargs(
                "MR Turbo combos", pbar_kwargs=dict(leave=False)
            ),
            merge_func="concat",
        ),
        execute_kwargs=make_execute_kwargs("MR Turbo CV splits"),
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
    import sys
    from pathlib import Path as _Path

    _SRC = _Path(__file__).resolve().parent.parent
    if str(_SRC) not in sys.path:
        sys.path.insert(0, str(_SRC))

    from framework.pipeline_utils import (
        METRIC_LABELS,
        analyze_portfolio,
        apply_vbt_plot_defaults,
        plot_cv_heatmap,
        plot_cv_splitter,
        plot_cv_volume,
        plot_grid_heatmap,
        plot_grid_surface,
        plot_grid_volume,
    )
    from framework.plotting import print_cv_results, print_grid_results
    from utils import load_fx_data

    # ─────────────────────────────────────────────────────────────────
    # CONFIGURATION — edit these defaults to change the run behaviour
    # ─────────────────────────────────────────────────────────────────
    DATA_PATH = "data/EUR-USD_minute.parquet"
    OUTPUT_DIR = "results/mr_turbo"
    SHOW_CHARTS = True
    N_FOLDS = 15

    # Single-run parameters
    SINGLE_PARAMS: dict[str, Any] = dict(
        bb_window=80,
        bb_alpha=5.0,
        sl_stop=0.005,
        tp_stop=0.006,
        leverage=1.0,
    )
    # Grid-search sweep
    GRID_PARAMS: dict[str, list] = dict(
        bb_window=[40, 60, 80, 120],
        bb_alpha=[4.0, 5.0, 6.0],
        sl_stop=[0.004, 0.005, 0.006],
        tp_stop=[0.004, 0.006, 0.008],
    )
    _METRIC_NAME = METRIC_LABELS[SHARPE_RATIO]

    def _header(label: str) -> None:
        bar = "█" * 78
        print(f"\n{bar}")
        print(f"██  {label.ljust(72)}  ██")
        print(f"{bar}\n")

    apply_vbt_plot_defaults()
    print("Loading data...")
    _, data = load_fx_data(DATA_PATH)

    # ─────────────────────────────────────────────────────────────────
    # 1) SINGLE RUN — full report + all individual plots
    # ─────────────────────────────────────────────────────────────────
    _header("MR TURBO  ·  SINGLE RUN")
    _walk_forward_report(data)
    pf, ind = pipeline(data, **SINGLE_PARAMS)
    analyze_portfolio(
        pf,
        name="MR Turbo",
        output_dir=OUTPUT_DIR,
        show_charts=SHOW_CHARTS,
        indicator=ind,
    )

    # ─────────────────────────────────────────────────────────────────
    # 2) GRID SEARCH — parameter sweep + heatmap / volume plots
    # ─────────────────────────────────────────────────────────────────
    _header("MR TURBO  ·  GRID SEARCH")
    grid = run_grid(data, metric_type=SHARPE_RATIO, **GRID_PARAMS)
    print_grid_results(
        grid,
        title="MR Turbo — Grid Search",
        metric_name=_METRIC_NAME,
        top_n=20,
    )
    if SHOW_CHARTS:
        plot_grid_heatmap(
            grid,
            x_level="bb_window",
            y_level="bb_alpha",
            slider_level="sl_stop",
            title=f"MR Turbo — {_METRIC_NAME} heatmap (slider: sl_stop)",
            metric_name=_METRIC_NAME,
        ).show()
        plot_grid_heatmap(
            grid,
            x_level="bb_window",
            y_level="bb_alpha",
            slider_level=None,
            title=f"MR Turbo — {_METRIC_NAME} heatmap (aggregated)",
            metric_name=_METRIC_NAME,
        ).show()
        plot_grid_volume(
            grid,
            x_level="bb_window",
            y_level="bb_alpha",
            z_level="sl_stop",
            slider_level="tp_stop",
            title=f"MR Turbo — {_METRIC_NAME} volume (slider: tp_stop)",
            metric_name=_METRIC_NAME,
        ).show()
        plot_grid_surface(
            grid,
            x_level="bb_window",
            y_level="bb_alpha",
            slider_level="sl_stop",
            title=f"MR Turbo — {_METRIC_NAME} surface (slider: sl_stop)",
            metric_name=_METRIC_NAME,
        ).show()
        plot_grid_surface(
            grid,
            x_level="bb_window",
            y_level="bb_alpha",
            slider_level=None,
            title=f"MR Turbo — {_METRIC_NAME} surface (aggregated)",
            metric_name=_METRIC_NAME,
        ).show()

    # ─────────────────────────────────────────────────────────────────
    # 3) WALK-FORWARD CROSS-VALIDATION — per-fold + aggregated plots
    # ─────────────────────────────────────────────────────────────────
    _header("MR TURBO  ·  WALK-FORWARD CV")
    # Build the splitter on a DAILY-resampled index so the train/test
    # plot has ~2k points instead of ~3M minute bars, and splits align
    # on day boundaries (no mid-day cut-offs).
    daily_index = data.close.vbt.resample_apply("1D", "last").dropna().index
    splitter = vbt.Splitter.from_purged_walkforward(
        daily_index,
        n_folds=N_FOLDS,
        n_test_folds=1,
        purge_td="1 day",
        min_train_folds=3,
    )
    if SHOW_CHARTS:
        plot_cv_splitter(splitter, title="MR Turbo — CV Splits").show()

    cv_pipeline = create_cv_pipeline(splitter, metric_type=SHARPE_RATIO)
    grid_perf, best_perf = cv_pipeline(
        data,
        bb_window=vbt.Param(GRID_PARAMS["bb_window"]),
        bb_alpha=vbt.Param(GRID_PARAMS["bb_alpha"]),
        sl_stop=vbt.Param(GRID_PARAMS["sl_stop"]),
        tp_stop=vbt.Param(GRID_PARAMS["tp_stop"]),
    )
    print_cv_results(
        grid_perf,
        best_perf,
        splitter=splitter,
        title="MR Turbo — Walk-Forward CV",
        metric_name=_METRIC_NAME,
        top_n=10,
    )
    if SHOW_CHARTS:
        plot_cv_heatmap(
            grid_perf,
            x_level="bb_window",
            y_level="bb_alpha",
            slider_level="split",
            splitter=splitter,
            title=f"MR Turbo — CV {_METRIC_NAME} heatmap (per split)",
            metric_name=_METRIC_NAME,
        ).show()
        plot_cv_heatmap(
            grid_perf,
            x_level="bb_window",
            y_level="bb_alpha",
            slider_level=None,
            splitter=splitter,
            title=f"MR Turbo — CV {_METRIC_NAME} heatmap (mean across splits)",
            metric_name=_METRIC_NAME,
        ).show()
        plot_cv_volume(
            grid_perf,
            x_level="bb_window",
            y_level="bb_alpha",
            z_level="sl_stop",
            slider_level="split",
            splitter=splitter,
            title=f"MR Turbo — CV {_METRIC_NAME} volume (per split)",
            metric_name=_METRIC_NAME,
        ).show()
        plot_cv_volume(
            grid_perf,
            x_level="bb_window",
            y_level="bb_alpha",
            z_level="sl_stop",
            slider_level=None,
            splitter=splitter,
            title=f"MR Turbo — CV {_METRIC_NAME} volume (mean across splits)",
            metric_name=_METRIC_NAME,
        ).show()
        plot_grid_surface(
            grid_perf,
            x_level="bb_window",
            y_level="bb_alpha",
            slider_level="split",
            splitter=splitter,
            title=f"MR Turbo — CV {_METRIC_NAME} surface (per split)",
            metric_name=_METRIC_NAME,
        ).show()
        plot_grid_surface(
            grid_perf,
            x_level="bb_window",
            y_level="bb_alpha",
            slider_level=None,
            splitter=splitter,
            title=f"MR Turbo — CV {_METRIC_NAME} surface (mean across splits)",
            metric_name=_METRIC_NAME,
        ).show()

    print("\nAll modes done.")
