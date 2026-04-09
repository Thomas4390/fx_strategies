"""OU Mean Reversion: Vol-Targeted Leverage, VWAP Anchor.

Native VBT VWAP + talib ADX, simplified Numba kernel for bands + leverage.
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
    compute_mr_bands_nb,
    mr_band_signal_nb,
    prepare_mr,
)


# ═══════════════════════════════════════════════════════════════════════
# INDICATOR KERNEL (simplified — VWAP & ADX pre-computed natively)
# ═══════════════════════════════════════════════════════════════════════


@njit(nogil=True)
def compute_ou_indicators_nb(
    index_ns: np.ndarray,
    close: np.ndarray,
    vwap: np.ndarray,
    lookback: int,
    band_width: float,
    vol_window: int,
    sigma_target: float,
    max_leverage: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Bands around VWAP + vol-targeted leverage."""
    zscore, upper_band, lower_band = compute_mr_bands_nb(
        index_ns, close, vwap, lookback, band_width
    )
    rolling_vol = compute_daily_rolling_volatility_nb(index_ns, close, vol_window)
    leverage = compute_leverage_nb(rolling_vol, sigma_target, max_leverage)
    return zscore, upper_band, lower_band, leverage


# ═══════════════════════════════════════════════════════════════════════
# PREPARE FUNCTION (native VBT: VWAP + ADX)
# ═══════════════════════════════════════════════════════════════════════


def prepare_ou_mr(raw, data):
    return prepare_mr(raw, data, adx_period=14, adx_threshold=30.0)


# ═══════════════════════════════════════════════════════════════════════
# STRATEGY SPECIFICATION
# ═══════════════════════════════════════════════════════════════════════

spec = StrategySpec(
    name="OU Mean Reversion (Vol-Targeted)",
    indicator=IndicatorSpec(
        class_name="IntradayMR",
        short_name="imr",
        input_names=("index_ns", "close_minute", "vwap"),
        param_names=(
            "lookback",
            "band_width",
            "vol_window",
            "sigma_target",
            "max_leverage",
        ),
        output_names=("zscore", "upper_band", "lower_band", "leverage"),
        kernel_func=compute_ou_indicators_nb,
    ),
    signal_func=mr_band_signal_nb,
    signal_args_map=(
        ("close_arr", "data.close"),
        ("upper_arr", "ind.upper_band"),
        ("lower_arr", "ind.lower_band"),
        ("twap_arr", "pre.vwap"),
        ("regime_ok_arr", "pre.regime_ok"),
        ("index_arr", "extra.index_ns"),
        ("eod_hour", "param.eod_hour"),
        ("eod_minute", "param.eod_minute"),
        ("eval_freq", "param.eval_freq"),
    ),
    params={
        "lookback": ParamDef(60, sweep=[20, 40, 60, 120, 240]),
        "band_width": ParamDef(2.0, sweep=[1.5, 2.0, 2.5, 3.0, 3.5]),
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
            OverlayLine("pre.vwap", "VWAP", color="#FF9800", dash="dash"),
            OverlayLine("ind.upper_band", "Upper Band", color="#E91E63", dash="dot"),
            OverlayLine("ind.lower_band", "Lower Band", color="#E91E63", dash="dot"),
        ),
    ),
    prepare_fn=prepare_ou_mr,
)


if __name__ == "__main__":
    from framework.runner import run_strategy

    run_strategy(spec, mode="full")
