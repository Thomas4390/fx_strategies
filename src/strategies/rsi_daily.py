"""RSI Mean Reversion (Daily).

Single-pair daily RSI mean reversion:
- Long when RSI < oversold threshold
- Short when RSI > overbought threshold
- Flat in the middle band

Uses ``vbt.Portfolio.from_signals`` so all framework plots (orders on
price, trades on price, exposure, heatmaps, volumes, etc.) work out of
the box.

Direct execution:
    python src/strategies/rsi_daily.py
    python src/strategies/rsi_daily.py --pair GBP-USD --leverage 3
    python src/strategies/rsi_daily.py --no-grid --no-show
"""

from __future__ import annotations

import vectorbtpro as vbt


# ═══════════════════════════════════════════════════════════════════════
# BACKTEST — single-pair daily RSI mean reversion, returns vbt.Portfolio
# ═══════════════════════════════════════════════════════════════════════


def backtest_rsi_daily(
    data: vbt.Data,
    rsi_period: int = 14,
    oversold: float = 25.0,
    overbought: float = 75.0,
    exit_mid: float = 50.0,
    leverage: float = 1.0,
    init_cash: float = 1_000_000.0,
    slippage: float = 0.0001,
) -> vbt.Portfolio:
    """RSI mean-reversion backtest on daily-resampled close.

    Entries/exits use band crossings, not always-in-market conditions:
    - Enter long when RSI crosses below *oversold*
    - Exit long when RSI crosses above *exit_mid*
    - Enter short when RSI crosses above *overbought*
    - Exit short when RSI crosses below *exit_mid*
    """
    # Daily resample from whatever the source frequency is
    if hasattr(data, "close"):
        close_any = data.close
    else:
        close_any = data  # already a Series/DataFrame
    close_daily = close_any.resample("1D").last().dropna()

    rsi = vbt.RSI.run(close_daily, window=rsi_period).rsi

    entries = rsi.vbt.crossed_below(oversold)
    exits = rsi.vbt.crossed_above(exit_mid)
    short_entries = rsi.vbt.crossed_above(overbought)
    short_exits = rsi.vbt.crossed_below(exit_mid)

    return vbt.Portfolio.from_signals(
        close=close_daily,
        entries=entries,
        exits=exits,
        short_entries=short_entries,
        short_exits=short_exits,
        init_cash=init_cash,
        leverage=leverage,
        slippage=slippage,
        freq="1D",
    )


# ═══════════════════════════════════════════════════════════════════════
# CLI ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════


if __name__ == "__main__":
    import argparse
    import sys
    from pathlib import Path as _Path

    # Allow running directly: `python src/strategies/rsi_daily.py`
    _SRC = _Path(__file__).resolve().parent.parent
    if str(_SRC) not in sys.path:
        sys.path.insert(0, str(_SRC))

    from framework.plotting import generate_standalone_report
    from utils import apply_vbt_settings, load_fx_data

    ap = argparse.ArgumentParser(description="RSI Daily MR standalone report")
    ap.add_argument("--pair", default="EUR-USD",
                    choices=["EUR-USD", "GBP-USD", "USD-JPY", "USD-CAD"])
    ap.add_argument("--leverage", type=float, default=1.0)
    ap.add_argument("--rsi-period", type=int, default=14)
    ap.add_argument("--oversold", type=float, default=25.0)
    ap.add_argument("--overbought", type=float, default=75.0)
    ap.add_argument("--no-grid", action="store_true")
    ap.add_argument("--no-show", action="store_true")
    ap.add_argument("--output-dir", default=None,
                    help="Default: results/rsi_daily_<pair>")
    args = ap.parse_args()

    apply_vbt_settings()
    data_path = f"data/{args.pair}_minute.parquet"
    print(f"Loading {args.pair} ...")
    _, data = load_fx_data(data_path)

    output_dir = args.output_dir or f"results/rsi_daily_{args.pair.lower()}"

    param_grid = None if args.no_grid else {
        "rsi_period": [7, 10, 14, 21],
        "oversold": [20.0, 25.0, 30.0],
        "overbought": [70.0, 75.0, 80.0],
    }

    fixed_params = {
        "leverage": args.leverage,
        "rsi_period": args.rsi_period,
        "oversold": args.oversold,
        "overbought": args.overbought,
    }

    generate_standalone_report(
        backtest_fn=backtest_rsi_daily,
        data=data,
        name=f"RSI Daily MR ({args.pair})",
        param_grid=param_grid,
        fixed_params=fixed_params if args.no_grid else {"leverage": args.leverage},
        output_dir=output_dir,
        show=not args.no_show,
    )
    print("\nDone.")
