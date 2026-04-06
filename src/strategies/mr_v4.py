"""MR V4: Adaptive EWM Bands.

EWM std replaces rolling std for faster volatility adaptation.
"""

import numpy as np
import vectorbtpro as vbt
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
    compute_adx_regime_nb,
    compute_deviation_nb,
    compute_intraday_twap_nb,
    mr_band_signal_nb,
)

# ══════���════════════════════════════════════════════════════════════════
# INDICATOR KERNEL
# ════��══════════════════════════════════════════════════════════════════


@njit(nogil=True)
def compute_mr_v4_indicators_nb(
    index_ns: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    open_: np.ndarray,
    ewm_span: int,
    band_width: float,
    adx_period: int,
    adx_threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute TWAP, EWM z-score, adaptive bands, and ADX filter."""
    n = len(close)

    twap = compute_intraday_twap_nb(index_ns, high, low, close)
    deviation = compute_deviation_nb(close, twap)

    ewm_std = vbt.generic.nb.ewm_std_1d_nb(
        deviation, span=ewm_span, minp=ewm_span, adjust=False
    )
    smoothed_deviation = vbt.generic.nb.ewm_mean_1d_nb(
        deviation, span=ewm_span, minp=ewm_span, adjust=False
    )

    zscore = np.full(n, np.nan)
    for i in range(n):
        sd = smoothed_deviation[i]
        es = ewm_std[i]
        if not np.isnan(sd) and not np.isnan(es) and es > 1e-10:
            zscore[i] = sd / es

    upper_band = np.full(n, np.nan)
    lower_band = np.full(n, np.nan)
    for i in range(n):
        s = ewm_std[i]
        if not np.isnan(s) and s > 1e-10 and not np.isnan(twap[i]):
            upper_band[i] = twap[i] + band_width * s
            lower_band[i] = twap[i] - band_width * s

    regime_ok = compute_adx_regime_nb(
        index_ns, high, low, close, open_, adx_period, adx_threshold
    )

    return twap, zscore, upper_band, lower_band, regime_ok


# ═══════════════════════════════════════════════════════════════════════
# STRATEGY SPECIFICATION
# ════════════════════════════════════════��══════════════════════════════

spec = StrategySpec(
    name="MR V4: Adaptive EWM Bands",
    indicator=IndicatorSpec(
        class_name="MR_V4",
        short_name="mr_v4",
        input_names=(
            "index_ns",
            "high_minute",
            "low_minute",
            "close_minute",
            "open_minute",
        ),
        param_names=("ewm_span", "band_width", "adx_period", "adx_threshold"),
        output_names=("twap", "zscore", "upper_band", "lower_band", "regime_ok"),
        kernel_func=compute_mr_v4_indicators_nb,
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
        "ewm_span": ParamDef(60, sweep=[20, 40, 60, 120]),
        "band_width": ParamDef(2.0, sweep=[1.5, 2.0, 2.5, 3.0]),
        "adx_period": ParamDef(14),
        "adx_threshold": ParamDef(30.0),
        "eod_hour": ParamDef(21),
        "eod_minute": ParamDef(0),
        "eval_freq": ParamDef(5),
        "sl_stop": ParamDef(0.005, sweep=[0.001, 0.002, 0.003, 0.005]),
    },
    portfolio_config=PortfolioConfig(leverage=1.0, sl_stop=0.005),
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
