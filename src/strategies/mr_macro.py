"""MR Macro: Macro-regime-filtered intraday VWAP mean reversion.

Extends MR Turbo with macro regime filters derived from:
- Yield curve spread (10Y-2Y Treasury)
- Unemployment trend (3-month change)

Research finding: filtering on spread<0.3 + unemployment not rising
boosts Sharpe from 0.19 to 1.07 (walk-forward 2021-2025), with OOS 2025 Sharpe 2.30.

100% VBT Pro native approach — pre-computed boolean signals.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import vectorbtpro as vbt


_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


# ═══════════════════════════════════════════════════════════════════════
# MACRO DATA LOADING (VBT native realign + module-level cache)
# ═══════════════════════════════════════════════════════════════════════

# Module-level cache: (first_ts, last_ts, n_bars, spread_threshold) → np.ndarray
# Keeps the most recent filter computation so repeated calls in a grid
# sweep hit the cache instead of recomputing. Capped at 8 entries.
_MACRO_FILTER_CACHE: dict[tuple, np.ndarray] = {}
_MACRO_CACHE_MAX = 8


def _load_macro_series(data_dir: Path) -> tuple[pd.Series, pd.Series]:
    """Read raw macro parquets and return (spread_daily, unemp_rising_monthly).

    Keeps the original date stamps unchanged — VBT ``Resampler`` takes
    the source frequency as a separate kwarg so we don't need to call
    ``asfreq`` which would shift business-day timestamps to standard
    calendar boundaries and break the 3-month diff semantics.
    """
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

    **Uses VBT native `series.vbt.realign_closing`** (via
    :class:`vbt.Resampler`) instead of pandas ``resample + reindex`` to
    align the daily spread and monthly unemployment series onto the
    minute-frequency FX index. Results are cached at module level so
    the same (index, threshold) signature reuses previous work — this
    is the key speedup for sequential parameter sweeps where every
    combination would otherwise re-do the realignment.
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

    # VBT native realign — use *realign_opening* because macro data is
    # "as-of" the report date: the value for date D is available to use
    # starting at 00:00:00 of D, not at the close of D. ``realign_closing``
    # would right-bound on the source and delay each value by one period.
    spread_resampler = vbt.Resampler(
        source_index=spread.index,
        target_index=minute_index,
        source_freq="D",
        target_freq="1min",
    )
    spread_min = spread.vbt.realign_opening(spread_resampler, ffill=True)

    # Unemployment rising flag → minute index (cast to float for realign,
    # cast back to bool). Same opening-bound semantics.
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

    # FIFO eviction: keep cache small
    if len(_MACRO_FILTER_CACHE) >= _MACRO_CACHE_MAX:
        _MACRO_FILTER_CACHE.pop(next(iter(_MACRO_FILTER_CACHE)))
    _MACRO_FILTER_CACHE[cache_key] = macro_ok.values.copy()
    return macro_ok


# ═══════════════════════════════════════════════════════════════════════
# BACKTEST (100% VBT native)
# ═══════════════════════════════════════════════════════════════════════


def backtest_mr_macro(
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
    init_cash: float = 1_000_000,
    slippage: float = 0.00015,
    leverage: float = 1.0,
) -> vbt.Portfolio:
    """Run macro-filtered MR backtest using 100% VBT native functions."""
    close = data.close

    # Native VWAP
    vwap = vbt.VWAP.run(data.high, data.low, close, data.volume, anchor="D").vwap

    # Bollinger Bands on deviation
    deviation = close - vwap
    bb = vbt.BBANDS.run(deviation, window=bb_window, alpha=bb_alpha)
    upper = vwap + bb.upper
    lower = vwap + bb.lower

    # Session filter
    hours = close.index.hour
    session = (hours >= session_start) & (hours < session_end)

    # Macro regime filter
    macro_ok = load_macro_filters(close.index, spread_threshold)

    # Entries: BB breach + session + macro regime
    entries = (close < lower) & session & macro_ok
    short_entries = (close > upper) & session & macro_ok

    return vbt.Portfolio.from_signals(
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


# ═══════════════════════════════════════════════════════════════════════
# MULTI-PARAM VARIANT (fully Numba-parallel via VBT native broadcasting)
# ═══════════════════════════════════════════════════════════════════════


def backtest_mr_macro_multi(
    data: vbt.Data,
    bb_window: list | int = 80,
    bb_alpha: list | float = 5.0,
    sl_stop: float = 0.005,
    tp_stop: float = 0.006,
    session_start: int = 6,
    session_end: int = 14,
    spread_threshold: float = 0.5,
    dt_stop: str = "21:00",
    td_stop: str = "6h",
    init_cash: float = 1_000_000,
    slippage: float = 0.00015,
    leverage: float = 1.0,
    param_product: bool = True,
) -> vbt.Portfolio:
    """Multi-column MR Macro — one broadcasted portfolio for the grid.

    Same as :func:`backtest_mr_macro` but accepts lists for
    ``bb_window`` / ``bb_alpha`` and runs the whole pipeline as a
    single multi-column portfolio with
    ``Portfolio.from_signals(chunked="threadpool")``. Entries happen
    in parallel Numba kernels across parameter combinations.
    """
    close = data.close

    vwap = vbt.VWAP.run(data.high, data.low, close, data.volume, anchor="D").vwap
    deviation = close - vwap

    bb = vbt.BBANDS.run(
        deviation,
        window=bb_window,
        alpha=bb_alpha,
        param_product=param_product,
    )
    upper = vwap.values[:, None] + bb.upper.values
    lower = vwap.values[:, None] + bb.lower.values

    # Session + macro filters broadcast to match column count
    hours = close.index.hour.values
    session_1d = (hours >= session_start) & (hours < session_end)
    macro_ok_1d = load_macro_filters(close.index, spread_threshold).values
    combined_1d = session_1d & macro_ok_1d
    combined = np.broadcast_to(combined_1d[:, None], upper.shape)

    close_2d = np.broadcast_to(close.values[:, None], upper.shape)
    entries = pd.DataFrame(
        (close_2d < lower) & combined,
        index=close.index,
        columns=bb.upper.columns,
    )
    short_entries = pd.DataFrame(
        (close_2d > upper) & combined,
        index=close.index,
        columns=bb.upper.columns,
    )

    return vbt.Portfolio.from_signals(
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
        chunked="threadpool",
    )


# ═══════════════════════════════════════════════════════════════════════
# CLI ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════

def _walk_forward_report(data: vbt.Data) -> None:
    """Per-year walk-forward diagnostic table."""
    print(f"\n{'=' * 60}\nWalk-Forward Validation (per-year)\n{'=' * 60}")
    for year in range(2021, 2027):
        d_yr = data.loc[f"{year}-01-01":f"{year}-12-31"]
        if d_yr.shape[0] < 1000:
            continue
        pf_yr = backtest_mr_macro(d_yr)
        tc = pf_yr.trades.count()
        sr = pf_yr.sharpe_ratio if tc > 0 else 0
        ret = pf_yr.total_return * 100 if tc > 0 else 0
        wr = pf_yr.trades.win_rate * 100 if tc > 0 else 0
        print(f"  {year}: Sharpe={sr:>7.3f}  Ret={ret:>6.2f}%  Trades={tc}  WR={wr:.1f}%")


if __name__ == "__main__":
    import argparse
    import sys
    from pathlib import Path as _Path

    # Allow running directly: `python src/strategies/mr_macro.py`
    _SRC = _Path(__file__).resolve().parent.parent
    if str(_SRC) not in sys.path:
        sys.path.insert(0, str(_SRC))

    from framework.plotting import generate_standalone_report
    from utils import apply_vbt_settings, load_fx_data

    ap = argparse.ArgumentParser(description="MR Macro standalone report")
    ap.add_argument("--data", default="data/EUR-USD_minute.parquet")
    ap.add_argument("--leverage", type=float, default=1.0,
                    help="Fixed leverage for the single run")
    ap.add_argument("--no-grid", action="store_true",
                    help="Skip parameter grid sweep (faster)")
    ap.add_argument("--no-show", action="store_true",
                    help="Do not open plots in the browser")
    ap.add_argument("--output-dir", default="results/mr_standalone",
                    help="Directory for fullscreen HTML plots")
    args = ap.parse_args()

    apply_vbt_settings()
    print("Loading data...")
    raw, data = load_fx_data(args.data)

    _walk_forward_report(data)

    # Only broadcast-native params (bb_window × bb_alpha) → fully
    # Numba-parallel via backtest_mr_macro_multi. For SL/TP sweeps,
    # run the script multiple times with different --sl-stop values or
    # pass --no-grid.
    param_grid = None if args.no_grid else {
        "bb_window": [40, 60, 80, 120],
        "bb_alpha": [3.5, 4.0, 4.5, 5.0, 5.5, 6.0],
    }

    generate_standalone_report(
        backtest_fn=backtest_mr_macro,
        backtest_multi_fn=backtest_mr_macro_multi,  # Numba-parallel path
        data=data,
        name="MR Macro",
        param_grid=param_grid,
        fixed_params={"leverage": args.leverage},
        output_dir=args.output_dir,
        show=not args.no_show,
    )
    print("\nDone.")
