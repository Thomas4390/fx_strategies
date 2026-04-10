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
# MACRO DATA LOADING
# ═══════════════════════════════════════════════════════════════════════


def load_macro_filters(
    minute_index: pd.DatetimeIndex,
    spread_threshold: float = 0.3,
    data_dir: Path | None = None,
) -> pd.Series:
    """Load macro data and build regime filter aligned to minute index.

    Filter: yield spread 10Y-2Y < threshold AND unemployment not rising (3m).
    Both conditions must be True for trading to be allowed.

    Returns a boolean Series aligned to minute_index.
    """
    if data_dir is None:
        data_dir = _PROJECT_ROOT / "data"

    # Yield spread (daily)
    spread_df = pd.read_parquet(data_dir / "SPREAD_10Y2Y_daily.parquet")
    spread_df["date"] = pd.to_datetime(spread_df["date"])
    spread = spread_df.set_index("date")["spread_10y2y"]

    # Unemployment (monthly)
    unemp_df = pd.read_parquet(data_dir / "UNEMPLOYMENT_monthly.parquet")
    unemp_df["date"] = pd.to_datetime(unemp_df["date"])
    unemp = unemp_df.set_index("date")["unemployment"]
    unemp_rising = unemp.diff(3) > 0  # 3-month change

    # Forward-fill to daily, then to minute
    spread_min = spread.resample("1D").ffill().reindex(minute_index, method="ffill")
    unemp_rising_min = (
        unemp_rising.resample("1D").ffill().reindex(minute_index, method="ffill")
    )

    # Regime: spread < threshold AND unemployment not rising
    unemp_ok = unemp_rising_min.fillna(False).astype(bool)
    macro_ok = (spread_min < spread_threshold) & (~unemp_ok)
    return macro_ok.fillna(False)


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
        freq="1min",
    )


# ═══════════════════════════════════════════════════════════════════════
# CLI ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import time
    from framework.plotting import (
        plot_monthly_heatmap,
        plot_portfolio_summary,
        plot_rolling_sharpe,
        plot_trade_analysis,
        show_browser,
    )
    from utils import load_fx_data

    vbt.settings.returns.year_freq = pd.Timedelta(hours=24) * 252

    print("Loading data...")
    raw, data = load_fx_data()

    # ── Full dataset backtest ──
    print(f"\n{'=' * 60}")
    print("MR MACRO — Full Dataset (macro-filtered)")
    print(f"{'=' * 60}")

    t0 = time.time()
    pf = backtest_mr_macro(data)
    t1 = time.time()
    print(f"Backtest: {t1 - t0:.1f}s")
    print(pf.stats().to_string())

    if pf.trades.count() > 0:
        print(f"\nTrades: {pf.trades.count()}")
        print(pf.trades.stats().to_string())

    # ── Walk-forward ──
    print(f"\n{'=' * 60}")
    print("Walk-Forward Validation (per-year)")
    print(f"{'=' * 60}")

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

    # ── Plots ──
    print("\nGenerating plots...")
    fig_summary = plot_portfolio_summary(pf, "MR Macro — Full Dataset")
    show_browser(fig_summary)

    fig_monthly = plot_monthly_heatmap(pf, "MR Macro — Monthly Returns")
    show_browser(fig_monthly)

    fig_trades = plot_trade_analysis(pf, "MR Macro — Trade Analysis")
    show_browser(fig_trades)

    fig_sharpe = plot_rolling_sharpe(pf, title="MR Macro — Rolling Sharpe")
    show_browser(fig_sharpe)

    # ── Comparison with unfiltered ──
    from strategies.mr_turbo import backtest_mr_turbo

    pf_base = backtest_mr_turbo(data)

    print(f"\n{'=' * 60}")
    print("COMPARISON: MR Macro vs MR Turbo (unfiltered)")
    print(f"{'=' * 60}")
    print(f"{'Metric':<30} {'MR Macro':>15} {'MR Turbo':>15}")
    print("-" * 60)
    for metric in [
        "Total Return [%]",
        "Sharpe Ratio",
        "Max Drawdown [%]",
        "Win Rate [%]",
        "Profit Factor",
        "Total Trades",
    ]:
        v1 = pf.stats().get(metric, "N/A")
        v2 = pf_base.stats().get(metric, "N/A")
        if isinstance(v1, float):
            print(f"  {metric:<30} {v1:>14.3f} {v2:>14.3f}")
        else:
            print(f"  {metric:<30} {str(v1):>15} {str(v2):>15}")

    print("\nDone.")
