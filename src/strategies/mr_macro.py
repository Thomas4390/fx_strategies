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
    make_execute_kwargs,
)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Default 4-pair universe for MR Macro multi-pair runs. EUR-USD stays the
# primary pair for backwards compatibility (single-symbol ``vbt.Data`` is
# still fully supported by :func:`pipeline`); the other three are added
# only when the caller explicitly loads them via :func:`load_all_fx_data`.
DEFAULT_PAIRS: tuple[str, ...] = ("EUR-USD", "GBP-USD", "USD-JPY", "USD-CAD")


def load_all_fx_data(
    pairs: tuple[str, ...] | list[str] | None = None,
    data_dir: Path | None = None,
) -> vbt.Data:
    """Load multiple FX pairs into a single multi-symbol ``vbt.Data``.

    The per-pair parquets are loaded through :func:`utils.load_fx_data`
    (lazy import to avoid circular deps with strategies importing from
    ``utils``), then **strictly intersected** on their minute index — no
    ``ffill`` between pairs — before being re-wrapped into a
    multi-symbol ``vbt.Data``. A ``ValueError`` is raised if the strict
    intersection is empty or shrinks by more than 5% relative to the
    largest input index (a signal that one pair has a gap that would
    silently bias the backtest).

    Parameters
    ----------
    pairs : tuple of str, optional
        Pair codes like ``"EUR-USD"``. Defaults to :data:`DEFAULT_PAIRS`.
    data_dir : Path, optional
        Override for the project ``data/`` directory.
    """
    from utils import load_fx_data  # lazy to avoid circular import at module load

    if pairs is None:
        pairs = DEFAULT_PAIRS
    if data_dir is None:
        data_dir = _PROJECT_ROOT / "data"

    per_pair: dict[str, pd.DataFrame] = {}
    for pair in pairs:
        _, data_single = load_fx_data(str(data_dir / f"{pair}_minute.parquet"))
        symbol = data_single.symbols[0]
        df = data_single.data[symbol]
        per_pair[pair] = df

    # Strict index intersection — ffill between pairs is forbidden.
    indexes = [df.index for df in per_pair.values()]
    common_index = indexes[0]
    for idx in indexes[1:]:
        common_index = common_index.intersection(idx)
    if len(common_index) == 0:
        raise ValueError(
            f"load_all_fx_data: empty index intersection across pairs {pairs}"
        )
    largest = max(len(idx) for idx in indexes)
    if len(common_index) < 0.95 * largest:
        raise ValueError(
            f"load_all_fx_data: intersection shrinks to {len(common_index)} "
            f"bars vs largest {largest} (>5% loss) — pair gaps detected."
        )

    aligned = {p: df.reindex(common_index) for p, df in per_pair.items()}
    return vbt.Data.from_data(aligned, tz_localize=False, tz_convert=False)


# ═══════════════════════════════════════════════════════════════════════
# MACRO DATA LOADING (VBT native realign + module-level cache)
# ═══════════════════════════════════════════════════════════════════════

# Aligned macro series (minute-frequency) cached at module level so
# walk-forward CV does not re-realign the spread / unemployment series
# on every call. Keyed by ``(first_ts, last_ts, n_bars)`` — the
# spread_threshold does NOT enter the key because we cache the raw
# ``spread_min`` / ``unemp_ok`` arrays and apply the threshold on the
# fly in :func:`load_macro_filters`.
_ALIGNED_MACRO_CACHE: dict[tuple, tuple[np.ndarray, np.ndarray]] = {}
_ALIGNED_MACRO_CACHE_MAX = 4  # plenty for a single strategy run


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


def _get_aligned_macro(
    minute_index: pd.DatetimeIndex,
    data_dir: Path,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (spread_min_vals, unemp_ok_vals) aligned to ``minute_index``.

    Results are cached so that repeated calls with the same index reuse
    the expensive ``vbt.Resampler`` work. This is the hot path in
    ``@vbt.cv_split`` where the same split ranges are hit many times.
    """
    cache_key = (
        minute_index[0].value,
        minute_index[-1].value,
        len(minute_index),
    )
    cached = _ALIGNED_MACRO_CACHE.get(cache_key)
    if cached is not None:
        return cached

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

    spread_vals = np.asarray(spread_min.values, dtype=float)
    unemp_vals = np.asarray(unemp_ok.values, dtype=bool)

    if len(_ALIGNED_MACRO_CACHE) >= _ALIGNED_MACRO_CACHE_MAX:
        _ALIGNED_MACRO_CACHE.pop(next(iter(_ALIGNED_MACRO_CACHE)))
    _ALIGNED_MACRO_CACHE[cache_key] = (spread_vals, unemp_vals)
    return spread_vals, unemp_vals


def load_macro_filters(
    minute_index: pd.DatetimeIndex,
    spread_threshold: float = 0.3,
    data_dir: Path | None = None,
) -> pd.Series:
    """Return the boolean macro regime filter aligned to ``minute_index``.

    Filter: yield spread 10Y-2Y < ``spread_threshold`` AND unemployment
    not rising (3-month diff). The expensive realign step is cached in
    ``_ALIGNED_MACRO_CACHE`` keyed by the target index only, so sweeping
    ``spread_threshold`` across a grid is essentially free after the
    first call.
    """
    if data_dir is None:
        data_dir = _PROJECT_ROOT / "data"

    spread_vals, unemp_vals = _get_aligned_macro(minute_index, data_dir)
    mask = (spread_vals < float(spread_threshold)) & (~unemp_vals)
    # NaN spread values (possible at the very start before the first
    # observation) propagate as False.
    mask = np.where(np.isnan(spread_vals), False, mask)
    return pd.Series(mask, index=minute_index, name="macro_ok")


# ═══════════════════════════════════════════════════════════════════════
# 1. INVESTIGATION PATH — pipeline() returns (pf, indicator)
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class MRMacroIndicator:
    """Lightweight indicator wrapper with a Plotly overlay.

    Single-symbol runs hold ``pd.Series`` for price/VWAP/bands (the
    legacy ``.plot()`` renders a single-pair overlay). Multi-symbol runs
    hold ``pd.DataFrame`` — ``.plot()`` then raises a clear error: use
    ``analyze_portfolio`` on the ``pf`` for per-pair breakdowns instead.
    """

    close: pd.Series | pd.DataFrame
    vwap: pd.Series | pd.DataFrame
    upper: pd.Series | pd.DataFrame
    lower: pd.Series | pd.DataFrame
    macro_ok: pd.Series

    def plot(self, fig: go.Figure | None = None, **layout_kwargs) -> go.Figure:
        """Draw Close + VWAP + filled BB band as plain ``go.Scatter`` traces.

        Lower and Upper bands are added back-to-back so ``fill="tonexty"``
        on the Upper trace reliably fills between the two bands (Plotly
        fills to the trace IMMEDIATELY before in the traces list, so
        inserting Close/VWAP between them would break the fill).
        """
        if isinstance(self.close, pd.DataFrame):
            raise ValueError(
                "MRMacroIndicator.plot() only supports single-symbol runs. "
                "For multi-pair runs use pipeline_utils.analyze_portfolio(pf) "
                "to get per-pair tearsheets."
            )
        fig = fig or go.Figure()
        fig.add_trace(
            go.Scatter(
                x=self.close.index,
                y=self.close.values,
                mode="lines",
                name="Close",
                line=dict(width=2, color="royalblue"),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=self.vwap.index,
                y=self.vwap.values,
                mode="lines",
                name="VWAP",
                line=dict(color="crimson", width=1, dash="dot"),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=self.lower.index,
                y=self.lower.values,
                mode="lines",
                name="Lower Band",
                line=dict(width=1.1, color="rgba(110,110,110,0.85)"),
                showlegend=True,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=self.upper.index,
                y=self.upper.values,
                mode="lines",
                name="Upper Band",
                line=dict(width=1.1, color="rgba(110,110,110,0.85)"),
                fill="tonexty",
                fillcolor="rgba(255,215,0,0.18)",
                showlegend=True,
            )
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

    Supports both single-symbol ``vbt.Data`` (``close`` is a ``pd.Series``,
    legacy behavior — bit-equivalent to snapshots) and multi-symbol
    ``vbt.Data`` (``close`` is a ``pd.DataFrame``). In the multi-symbol
    case ``cash_sharing=True, group_by=True`` is set so the 4-pair
    MR Macro shares a single cash pool, and the session/macro masks are
    broadcast column-wise.
    """
    n_symbols = len(data.symbols) if hasattr(data, "symbols") else 1
    is_multi = n_symbols > 1

    close = data.close

    vwap = vbt.VWAP.run(data.high, data.low, close, data.volume, anchor="D").vwap
    deviation = close - vwap
    bb = vbt.BBANDS.run(deviation, window=bb_window, alpha=bb_alpha)
    if is_multi:
        # bb.upper/lower carry extra MultiIndex levels (bb_window, bb_alpha)
        # from the BBANDS indicator — pandas strict arithmetic rejects
        # adding them to ``vwap`` (simple column index). Drop to numpy,
        # then rebuild a DataFrame with the pair columns preserved.
        upper = pd.DataFrame(
            vwap.values + bb.upper.values,
            index=close.index,
            columns=close.columns,
        )
        lower = pd.DataFrame(
            vwap.values + bb.lower.values,
            index=close.index,
            columns=close.columns,
        )
    else:
        upper = vwap + bb.upper
        lower = vwap + bb.lower

    hours = close.index.hour
    session_1d = (hours >= session_start) & (hours < session_end)
    macro_ok_1d = load_macro_filters(close.index, spread_threshold)

    if is_multi:
        # Row-wise broadcasting via numpy column vectors (T, 1) → (T, K).
        # Replaces the previous pd.DataFrame(np.broadcast_to(...)) pattern:
        # same element-wise booleans, no intermediate 2D DataFrames.
        session_col = session_1d[:, None]
        macro_col = macro_ok_1d.values[:, None]
        entries = pd.DataFrame(
            (close.values < lower.values) & session_col & macro_col,
            index=close.index,
            columns=close.columns,
        )
        short_entries = pd.DataFrame(
            (close.values > upper.values) & session_col & macro_col,
            index=close.index,
            columns=close.columns,
        )
    else:
        session = pd.Series(session_1d, index=close.index)
        entries = (close < lower) & session & macro_ok_1d
        short_entries = (close > upper) & session & macro_ok_1d

    pf_kwargs: dict[str, Any] = dict(
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
    if is_multi:
        pf_kwargs["cash_sharing"] = True
        pf_kwargs["group_by"] = True

    pf = vbt.Portfolio.from_signals(data, **pf_kwargs)

    indicator = MRMacroIndicator(
        close=close,
        vwap=vwap.rename("VWAP") if isinstance(vwap, pd.Series) else vwap,
        upper=upper.rename("Upper Band") if isinstance(upper, pd.Series) else upper,
        lower=lower.rename("Lower Band") if isinstance(lower, pd.Series) else lower,
        macro_ok=macro_ok_1d,
    )
    return pf, indicator


# ═══════════════════════════════════════════════════════════════════════
# 2. GRID-SEARCH PATH — pipeline_nb (@vbt.parameterized)
# ═══════════════════════════════════════════════════════════════════════


# MR Macro runs on 3M-bar minute data, so each in-flight Portfolio is
# ~400 MB. Cap the parallel chunk at 4 combos (≈1.6 GB peak) and flush
# the GC + VBT caches between chunks — otherwise the default 8 would
# peak around 3.2 GB for the live batch AND the 24 merged-but-not-yet-
# released Portfolios linger in VBT's internal caches until the end.
@vbt.parameterized(
    merge_func="concat",
    execute_kwargs=make_execute_kwargs("MR Macro grid", chunk_len=4, flush_every=1),
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
            execute_kwargs=make_execute_kwargs(
                "MR Macro combos", chunk_len=4, pbar_kwargs=dict(leave=False)
            ),
            merge_func="concat",
        ),
        execute_kwargs=make_execute_kwargs("MR Macro CV splits"),
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
        d_yr = data.loc[f"{year}-01-01" : f"{year}-12-31"]
        if d_yr.shape[0] < 1000:
            continue
        pf_yr, _ = pipeline(d_yr)
        tc = pf_yr.trades.count()
        sr = pf_yr.sharpe_ratio if tc > 0 else 0
        ret = pf_yr.total_return * 100 if tc > 0 else 0
        wr = pf_yr.trades.win_rate * 100 if tc > 0 else 0
        print(
            f"  {year}: Sharpe={sr:>7.3f}  Ret={ret:>6.2f}%  Trades={tc}  WR={wr:.1f}%"
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
    # CONFIGURATION
    # ─────────────────────────────────────────────────────────────────
    DATA_PATH = "data/EUR-USD_minute.parquet"
    OUTPUT_DIR = "results/mr_macro"
    SHOW_CHARTS = True
    N_FOLDS = 15

    SINGLE_PARAMS: dict[str, Any] = dict(
        bb_window=80,
        bb_alpha=5.0,
        sl_stop=0.005,
        tp_stop=0.006,
        spread_threshold=0.5,
        leverage=1.0,
    )
    GRID_PARAMS: dict[str, Any] = dict(
        bb_window=[40, 60, 80, 120],
        bb_alpha=[3.5, 4.0, 4.5, 5.0, 5.5, 6.0],
        sl_stop=0.005,
        tp_stop=0.006,
        spread_threshold=[0.3, 0.5, 0.7],
    )
    CV_PARAMS: dict[str, Any] = dict(
        bb_window=[60, 80, 120],
        bb_alpha=[4.0, 5.0, 6.0],
        spread_threshold=[0.3, 0.5],
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
    # 1) SINGLE RUN
    # ─────────────────────────────────────────────────────────────────
    _header("MR MACRO  ·  SINGLE RUN")
    _walk_forward_report(data)
    pf, ind = pipeline(data, **SINGLE_PARAMS)
    analyze_portfolio(
        pf,
        name="MR Macro",
        output_dir=OUTPUT_DIR,
        show_charts=SHOW_CHARTS,
        indicator=ind,
    )

    # ─────────────────────────────────────────────────────────────────
    # 2) GRID SEARCH
    # ─────────────────────────────────────────────────────────────────
    _header("MR MACRO  ·  GRID SEARCH")
    grid = run_grid(data, metric_type=SHARPE_RATIO, **GRID_PARAMS)
    print_grid_results(
        grid,
        title="MR Macro — Grid Search",
        metric_name=_METRIC_NAME,
        top_n=20,
    )
    if SHOW_CHARTS:
        plot_grid_heatmap(
            grid,
            x_level="bb_window",
            y_level="bb_alpha",
            slider_level="spread_threshold",
            title=f"MR Macro — {_METRIC_NAME} heatmap (slider: spread_threshold)",
            metric_name=_METRIC_NAME,
        ).show()
        plot_grid_heatmap(
            grid,
            x_level="bb_window",
            y_level="bb_alpha",
            slider_level=None,
            title=f"MR Macro — {_METRIC_NAME} heatmap (aggregated)",
            metric_name=_METRIC_NAME,
        ).show()
        plot_grid_volume(
            grid,
            x_level="bb_window",
            y_level="bb_alpha",
            z_level="spread_threshold",
            title=f"MR Macro — {_METRIC_NAME} volume",
            metric_name=_METRIC_NAME,
        ).show()
        plot_grid_surface(
            grid,
            x_level="bb_window",
            y_level="bb_alpha",
            slider_level="spread_threshold",
            title=f"MR Macro — {_METRIC_NAME} surface (slider: spread_threshold)",
            metric_name=_METRIC_NAME,
        ).show()
        plot_grid_surface(
            grid,
            x_level="bb_window",
            y_level="bb_alpha",
            slider_level=None,
            title=f"MR Macro — {_METRIC_NAME} surface (aggregated)",
            metric_name=_METRIC_NAME,
        ).show()

    # ─────────────────────────────────────────────────────────────────
    # 3) WALK-FORWARD CROSS-VALIDATION
    # ─────────────────────────────────────────────────────────────────
    _header("MR MACRO  ·  WALK-FORWARD CV")
    # Daily-resampled index for the splitter so plot_cv_splitter renders
    # fast and the train/test bounds align on day boundaries.
    daily_index = data.close.vbt.resample_apply("1D", "last").dropna().index
    splitter = vbt.Splitter.from_purged_walkforward(
        daily_index,
        n_folds=N_FOLDS,
        n_test_folds=1,
        purge_td="1 day",
        min_train_folds=3,
    )
    if SHOW_CHARTS:
        plot_cv_splitter(splitter, title="MR Macro — CV Splits").show()

    cv_pipeline = create_cv_pipeline(splitter, metric_type=SHARPE_RATIO)
    grid_perf, best_perf = cv_pipeline(
        data,
        bb_window=vbt.Param(CV_PARAMS["bb_window"]),
        bb_alpha=vbt.Param(CV_PARAMS["bb_alpha"]),
        sl_stop=0.005,
        tp_stop=0.006,
        spread_threshold=vbt.Param(CV_PARAMS["spread_threshold"]),
    )
    print_cv_results(
        grid_perf,
        best_perf,
        splitter=splitter,
        title="MR Macro — Walk-Forward CV",
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
            title=f"MR Macro — CV {_METRIC_NAME} heatmap (per split)",
            metric_name=_METRIC_NAME,
        ).show()
        plot_cv_heatmap(
            grid_perf,
            x_level="bb_window",
            y_level="bb_alpha",
            slider_level=None,
            splitter=splitter,
            title=f"MR Macro — CV {_METRIC_NAME} heatmap (mean across splits)",
            metric_name=_METRIC_NAME,
        ).show()
        plot_cv_volume(
            grid_perf,
            x_level="bb_window",
            y_level="bb_alpha",
            z_level="spread_threshold",
            slider_level="split",
            splitter=splitter,
            title=f"MR Macro — CV {_METRIC_NAME} volume (per split)",
            metric_name=_METRIC_NAME,
        ).show()
        plot_cv_volume(
            grid_perf,
            x_level="bb_window",
            y_level="bb_alpha",
            z_level="spread_threshold",
            slider_level=None,
            splitter=splitter,
            title=f"MR Macro — CV {_METRIC_NAME} volume (mean across splits)",
            metric_name=_METRIC_NAME,
        ).show()
        plot_grid_surface(
            grid_perf,
            x_level="bb_window",
            y_level="bb_alpha",
            slider_level="split",
            splitter=splitter,
            title=f"MR Macro — CV {_METRIC_NAME} surface (per split)",
            metric_name=_METRIC_NAME,
        ).show()
        plot_grid_surface(
            grid_perf,
            x_level="bb_window",
            y_level="bb_alpha",
            slider_level=None,
            splitter=splitter,
            title=f"MR Macro — CV {_METRIC_NAME} surface (mean across splits)",
            metric_name=_METRIC_NAME,
        ).show()

    print("\nAll modes done.")
