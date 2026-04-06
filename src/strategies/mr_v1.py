"""MR V1: No Leverage, Stop-Loss Only.

Intraday TWAP mean reversion without vol-targeted sizing.
Entry on band breach, exit on TWAP crossback or SL, EOD forced exit.
"""

from framework.spec import (
    IndicatorSpec,
    OverlayLine,
    ParamDef,
    PlotConfig,
    PortfolioConfig,
    StrategySpec,
)
from utils import compute_mr_base_indicators_nb, mr_band_signal_nb

MR_BAND_PLOT_CONFIG = PlotConfig(
    overlays=(
        OverlayLine("ind.twap", "TWAP", color="#FF9800", dash="dash"),
        OverlayLine("ind.upper_band", "Upper Band", color="#E91E63", dash="dot"),
        OverlayLine("ind.lower_band", "Lower Band", color="#E91E63", dash="dot"),
    ),
)


# ═══════════════════════════════════════════════════════════════════════
# STRATEGY SPECIFICATION
# ═══════════════════════════════════════════════════════════════════════

spec = StrategySpec(
    name="MR V1: No Leverage",
    indicator=IndicatorSpec(
        class_name="MR_V1",
        short_name="mr_v1",
        input_names=(
            "index_ns",
            "high_minute",
            "low_minute",
            "close_minute",
            "open_minute",
        ),
        param_names=("lookback", "band_width", "adx_period", "adx_threshold"),
        output_names=("twap", "zscore", "upper_band", "lower_band", "regime_ok"),
        kernel_func=compute_mr_base_indicators_nb,
    ),
    signal_func=mr_band_signal_nb,
    signal_args_map=(
        ("close_arr", "data.close"),
        ("upper_arr", "ind.upper_band"),
        ("lower_arr", "ind.lower_band"),
        ("twap_arr", "ind.twap"),
        ("regime_ok_arr", "ind.regime_ok"),
        ("index_arr", "extra.index_ns"),
        ("eod_hour", "param.eod_hour"),
        ("eod_minute", "param.eod_minute"),
        ("eval_freq", "param.eval_freq"),
    ),
    params={
        "lookback": ParamDef(60, sweep=[20, 40, 60, 120, 240]),
        "band_width": ParamDef(2.0, sweep=[1.5, 2.0, 2.5, 3.0]),
        "adx_period": ParamDef(14),
        "adx_threshold": ParamDef(30.0),
        "eod_hour": ParamDef(21),
        "eod_minute": ParamDef(0),
        "eval_freq": ParamDef(5),
        "sl_stop": ParamDef(0.005, sweep=[0.001, 0.002, 0.003, 0.005]),
    },
    portfolio_config=PortfolioConfig(leverage=1.0, sl_stop=0.005),
    plot_config=MR_BAND_PLOT_CONFIG,
)


if __name__ == "__main__":
    from framework.runner import run_strategy

    run_strategy(spec, mode="full")
