"""Build PNG signals charts (price + indicators + trades) for each
of the three report sleeves, using the framework's existing
``plot_trade_signals()`` Plotly figure exported via kaleido.

Run:
    python scripts/build_sleeve_signals_figures.py

Outputs:
    reports/latex_report/figures/sleeve_mr_macro_signals.png
    reports/latex_report/figures/sleeve_ts_momentum_3p_signals.png
    reports/latex_report/figures/sleeve_rsi_daily_3p_signals.png

Multi-pair sleeves (TS Momentum 3p, RSI Daily 4p) are represented by
their anchor pair — GBP-USD and EUR-USD respectively — since
``plot_trade_signals`` operates on a single (portfolio, indicator)
pair. The chart is selected via ``_find_featured_trade_window`` which
picks a representative trade whose indicator values AND close price
are fully populated over the entry→exit window (see the price-NaN
guard added in ``framework/plotting/_core.py``).
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SRC = _PROJECT_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

FIG_DIR = _PROJECT_ROOT / "reports" / "latex_report" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

PNG_WIDTH = 1600
PNG_HEIGHT = 800
PNG_SCALE = 2


def _save(fig, stem: str) -> Path:
    out_path = FIG_DIR / f"{stem}.png"
    fig.write_image(
        str(out_path),
        width=PNG_WIDTH,
        height=PNG_HEIGHT,
        scale=PNG_SCALE,
    )
    print(f"    saved → {out_path.relative_to(_PROJECT_ROOT)}")
    return out_path


def build_mr_macro_signals() -> None:
    from framework.plotting._trades import plot_trade_signals
    from framework.project_config import PROJECT_CONFIG, data_path
    from strategies.mr_macro import pipeline as mr_pipeline
    from utils import load_fx_data

    pair = PROJECT_CONFIG["default_pair"]
    print(f"  [MR Macro] loading {pair} minute data...")
    _, data = load_fx_data(str(data_path(pair)))

    print("  [MR Macro] running backtest...")
    pf, ind = mr_pipeline(
        data,
        bb_window=80,
        bb_alpha=5.0,
        sl_stop=0.005,
        tp_stop=0.006,
        spread_threshold=0.5,
    )

    print("  [MR Macro] rendering signals chart...")
    fig = plot_trade_signals(
        pf,
        title=f"MR Macro ({pair}) — Signals + Indicators",
        indicator=ind,
        height=PNG_HEIGHT,
    )
    _save(fig, "sleeve_mr_macro_signals")


def build_ts_momentum_3p_signals() -> None:
    """Anchor pair = GBP-USD (flagged as the TS-best pair per the 2021-2025
    per-pair decomposition in daily_momentum.py)."""
    from framework.plotting._trades import plot_trade_signals
    from strategies.daily_momentum import load_daily_closes, pipeline_ts

    anchor_pair = "GBP-USD"
    print(f"  [TS Momentum 3p] loading daily closes (anchor: {anchor_pair})...")
    closes = load_daily_closes()
    close_daily = closes[anchor_pair]

    print(f"  [TS Momentum 3p] running backtest on {anchor_pair}...")
    pf, ind = pipeline_ts(close_daily, leverage=1.0)

    print("  [TS Momentum 3p] rendering signals chart...")
    fig = plot_trade_signals(
        pf,
        title=f"TS Momentum 3p (anchor: {anchor_pair}) — Signals + Indicators",
        indicator=ind,
        height=PNG_HEIGHT,
    )
    _save(fig, "sleeve_ts_momentum_3p_signals")


def build_rsi_daily_3p_signals() -> None:
    """Anchor pair = EUR-USD (default pair, first in the RSI_Daily_3p
    equal-weight basket). Uses the per-pair rsi_daily.pipeline()."""
    from framework.plotting._trades import plot_trade_signals
    from framework.project_config import data_path
    from strategies.rsi_daily import pipeline as rsi_pipeline
    from utils import load_fx_data

    anchor_pair = "EUR-USD"
    print(f"  [RSI Daily 4p] loading {anchor_pair} minute data...")
    _, data = load_fx_data(str(data_path(anchor_pair)))

    print(f"  [RSI Daily 4p] running backtest on {anchor_pair}...")
    pf, ind = rsi_pipeline(
        data,
        rsi_period=14,
        oversold=25.0,
        overbought=75.0,
    )

    print("  [RSI Daily 4p] rendering signals chart...")
    fig = plot_trade_signals(
        pf,
        title=f"RSI Daily 4p (anchor: {anchor_pair}) — Signals + Indicators",
        indicator=ind,
        height=PNG_HEIGHT,
    )
    _save(fig, "sleeve_rsi_daily_3p_signals")


def main() -> None:
    print("═" * 70)
    print("  Sleeve signals charts — PNG builder")
    print("═" * 70)

    print("\n[1/3] MR Macro")
    build_mr_macro_signals()

    print("\n[2/3] TS Momentum 3p")
    build_ts_momentum_3p_signals()

    print("\n[3/3] RSI Daily 4p")
    build_rsi_daily_3p_signals()

    print("\nDone.")


if __name__ == "__main__":
    main()
