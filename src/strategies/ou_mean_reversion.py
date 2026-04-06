"""OU Mean Reversion: Vol-Targeted Leverage, TWAP Anchor.

Original intraday TWAP mean reversion with volatility-targeted position sizing.
"""

import numpy as np
from numba import njit

from framework.spec import (
    IndicatorSpec,
    OverlayLine,
    ParamDef,
    PlotConfig,
    PortfolioConfig,
    StrategySpec,
)
from utils import (
    compute_daily_rolling_volatility_nb,
    compute_leverage_nb,
    compute_mr_base_indicators_nb,
    mr_band_signal_nb,
)

# ═══════════════════════════════════════════════════════════════════════
# INDICATOR KERNEL (thin wrapper adding leverage to shared MR base)
# ═══════════════════════════════════════════════════════════════════════


@njit(nogil=True)
def compute_ou_mr_indicators_nb(
    index_ns: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    open_: np.ndarray,
    lookback: int,
    band_width: float,
    adx_period: int,
    adx_threshold: float,
    vol_window: int,
    sigma_target: float,
    max_leverage: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """MR base indicators + vol-targeted leverage."""
    twap, zscore, upper_band, lower_band, regime_ok = compute_mr_base_indicators_nb(
        index_ns,
        high,
        low,
        close,
        open_,
        lookback,
        band_width,
        adx_period,
        adx_threshold,
    )
    rolling_vol = compute_daily_rolling_volatility_nb(index_ns, close, vol_window)
    leverage = compute_leverage_nb(rolling_vol, sigma_target, max_leverage)
    return twap, zscore, upper_band, lower_band, regime_ok, leverage


# ═══════════════════════════════════════════════════════════════════════
# STRATEGY SPECIFICATION
# ═══════════════════════════════════════════════════════════════════════

spec = StrategySpec(
    name="OU Mean Reversion (Vol-Targeted)",
    indicator=IndicatorSpec(
        class_name="IntradayMR",
        short_name="imr",
        input_names=(
            "index_ns",
            "high_minute",
            "low_minute",
            "close_minute",
            "open_minute",
        ),
        param_names=(
            "lookback",
            "band_width",
            "adx_period",
            "adx_threshold",
            "vol_window",
            "sigma_target",
            "max_leverage",
        ),
        output_names=(
            "twap",
            "zscore",
            "upper_band",
            "lower_band",
            "regime_ok",
            "leverage",
        ),
        kernel_func=compute_ou_mr_indicators_nb,
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
        "band_width": ParamDef(2.0, sweep=[1.5, 2.0, 2.5, 3.0, 3.5]),
        "adx_period": ParamDef(14),
        "adx_threshold": ParamDef(30.0),
        "vol_window": ParamDef(20),
        "sigma_target": ParamDef(0.01),
        "max_leverage": ParamDef(3.0),
        "eod_hour": ParamDef(21),
        "eod_minute": ParamDef(0),
        "eval_freq": ParamDef(5),
        "sl_stop": ParamDef(0.005, sweep=[0.002, 0.003, 0.005]),
    },
    portfolio_config=PortfolioConfig(leverage="ind.leverage", sl_stop=0.005),
    plot_config=PlotConfig(
        overlays=(
            OverlayLine("ind.twap", "TWAP", color="#FF9800", dash="dash"),
            OverlayLine("ind.upper_band", "Upper Band", color="#E91E63", dash="dot"),
            OverlayLine("ind.lower_band", "Lower Band", color="#E91E63", dash="dot"),
        ),
    ),
)


if __name__ == "__main__":
    from framework.runner import run_strategy

    run_strategy(spec, mode="full")
