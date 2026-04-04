"""Composite FX Alpha: Momentum + Volatility Timing (Daily).

Multi-factor strategy with drawdown control and Jegadeesh-Titman sub-portfolios.
Uses target-weight rebalancing (size_type=amount), not simple entry/exit signals.
"""

import numpy as np
import vectorbtpro as vbt
from numba import njit

from framework.spec import IndicatorSpec, ParamDef, PortfolioConfig, StrategySpec


# ═══════════════════════════════════════════════════════════════════════
# NUMBA KERNELS
# ═══════════════════════════════════════════════════════════════════════


@njit
def momentum_signal_nb(close: np.ndarray, w_short: int, w_long: int) -> np.ndarray:
    """Blended momentum: 0.5 * log_return(21d) + 0.5 * log_return(63d)."""
    n = len(close)
    out = np.full(n, np.nan)
    for i in range(w_long, n):
        r_short = np.log(close[i] / close[i - w_short])
        r_long = np.log(close[i] / close[i - w_long])
        out[i] = 0.5 * r_short + 0.5 * r_long
    return out


@njit
def regime_weight_nb(
    vr: np.ndarray,
    low_th: float,
    high_th: float,
    w_low: float,
    w_normal: float,
    w_high: float,
) -> np.ndarray:
    """Map volatility regime ratio to momentum weight."""
    n = len(vr)
    out = np.full(n, np.nan)
    for i in range(n):
        if np.isnan(vr[i]):
            continue
        elif vr[i] < low_th:
            out[i] = w_low
        elif vr[i] > high_th:
            out[i] = w_high
        else:
            out[i] = w_normal
    return out


@njit
def vol_scaling_nb(ewma_vol: np.ndarray, target: float, cap: float) -> np.ndarray:
    """lambda_t = min(sigma_target / sigma_Pt, lambda_max)."""
    n = len(ewma_vol)
    out = np.full(n, np.nan)
    for i in range(n):
        v = ewma_vol[i]
        if not np.isnan(v) and v > 0:
            out[i] = min(target / v, cap)
    return out


@njit
def drawdown_control_nb(
    dd: np.ndarray,
    soft: float,
    hard: float,
    recovery: float,
) -> np.ndarray:
    """State machine: NORMAL(1.0) -> REDUCED(0.5) -> FLAT(0.0) with hysteresis."""
    n = len(dd)
    mult = np.ones(n)
    state = 0
    for i in range(n):
        d = dd[i]
        if state == 0:
            if d > hard:
                state = 2
            elif d > soft:
                state = 1
        elif state == 1:
            if d > hard:
                state = 2
            elif d < recovery:
                state = 0
        elif state == 2 and d < recovery:
            state = 0
        if state == 1:
            mult[i] = 0.5
        elif state == 2:
            mult[i] = 0.0
    return mult


@njit
def sub_portfolio_weights_nb(
    direction: np.ndarray,
    regime_wt: np.ndarray,
    vol_scale: np.ndarray,
    dd_mult: np.ndarray,
    n_days: int,
    k: int,
) -> np.ndarray:
    """K=5 overlapping sub-portfolio weights (Jegadeesh-Titman)."""
    sub_w = np.zeros((k, n_days))
    for j in range(k):
        current = 0.0
        for i in range(n_days):
            if i >= j * 5 and (i - j * 5) % (k * 5) == 0:
                d = direction[i]
                r = regime_wt[i]
                v = vol_scale[i]
                m = dd_mult[i]
                if not (np.isnan(d) or np.isnan(r) or np.isnan(v) or np.isnan(m)):
                    current = d * r * v * m
            sub_w[j, i] = current

    weights = np.zeros(n_days)
    for i in range(n_days):
        total = 0.0
        for j in range(k):
            total += sub_w[j, i]
        weights[i] = total / k
    return weights


@njit
def compute_composite_nb(
    close: np.ndarray,
    returns: np.ndarray,
    w_short: int,
    w_long: int,
    vol_short: int,
    vol_long: int,
    ewma_span: int,
    target_vol: float,
    leverage_cap: float,
    vr_low: float,
    vr_high: float,
    mom_w_low: float,
    mom_w_normal: float,
    mom_w_high: float,
    dd_soft: float,
    dd_hard: float,
    dd_recovery: float,
    n_sub: int,
) -> tuple:
    """Master kernel: compute all signals -> daily target weights."""
    n = len(close)

    momentum = momentum_signal_nb(close, w_short, w_long)
    direction = np.full(n, 0.0)
    for i in range(n):
        if not np.isnan(momentum[i]):
            direction[i] = (
                1.0 if momentum[i] > 0 else (-1.0 if momentum[i] < 0 else 0.0)
            )

    sigma_short = vbt.generic.nb.rolling_std_1d_nb(
        returns, vol_short, minp=vol_short, ddof=1
    )
    sigma_long = vbt.generic.nb.rolling_std_1d_nb(
        returns, vol_long, minp=vol_long, ddof=1
    )
    vr = np.full(n, np.nan)
    for i in range(n):
        if (
            not np.isnan(sigma_short[i])
            and not np.isnan(sigma_long[i])
            and sigma_long[i] > 0
        ):
            vr[i] = sigma_short[i] / sigma_long[i]

    regime_wt = regime_weight_nb(
        vr, vr_low, vr_high, mom_w_low, mom_w_normal, mom_w_high
    )

    sq_returns = np.empty(n)
    for i in range(n):
        sq_returns[i] = returns[i] ** 2 if not np.isnan(returns[i]) else np.nan
    ewma_var = vbt.generic.nb.ewm_mean_1d_nb(sq_returns, ewma_span, minp=1, adjust=True)
    ewma_vol = np.full(n, np.nan)
    for i in range(n):
        if not np.isnan(ewma_var[i]) and ewma_var[i] > 0:
            ewma_vol[i] = np.sqrt(ewma_var[i]) * np.sqrt(252)
    vol_scale = vol_scaling_nb(ewma_vol, target_vol, leverage_cap)

    proxy_equity = np.ones(n)
    for i in range(1, n):
        d, v, r = direction[i - 1], vol_scale[i - 1], returns[i]
        if not (np.isnan(d) or np.isnan(v) or np.isnan(r)):
            proxy_equity[i] = proxy_equity[i - 1] * (1 + r * d * v)
        else:
            proxy_equity[i] = proxy_equity[i - 1]

    lookback = 63
    dd = np.zeros(n)
    for i in range(n):
        peak = proxy_equity[max(0, i - lookback + 1)]
        for j in range(max(0, i - lookback + 1), i + 1):
            if proxy_equity[j] > peak:
                peak = proxy_equity[j]
        dd[i] = 1.0 - proxy_equity[i] / peak if peak > 0 else 0.0

    dd_mult = drawdown_control_nb(dd, dd_soft, dd_hard, dd_recovery)

    weights = sub_portfolio_weights_nb(
        direction, regime_wt, vol_scale, dd_mult, n, n_sub
    )

    return momentum, direction, vr, regime_wt, ewma_vol, vol_scale, dd, dd_mult, weights


# ═══════════════════════════════════════════════════════════════════════
# SIGNAL FUNCTION
# ═══════════════════════════════════════════════════════════════════════


@njit
def composite_signal_nb(c, target_weights, size_arr):
    """Rebalance to target weight each bar using delta sizing."""
    tw = vbt.pf_nb.select_nb(c, target_weights)
    if np.isnan(tw):
        return False, False, False, False

    pos = c.last_position[c.col]
    price = c.last_val_price[c.col]
    value = c.last_value[c.group]

    if value <= 0 or price <= 0:
        return False, False, False, False

    target_pos = tw * value / price
    delta = target_pos - pos

    if abs(delta * price / value) < 0.005:
        return False, False, False, False

    size_arr[c.i, c.col] = abs(delta)

    if pos >= 0 and delta > 0:
        return True, False, False, False
    elif pos > 0 and delta < 0 and target_pos >= 0:
        return False, True, False, False
    elif pos >= 0 and target_pos < 0:
        return False, False, True, False
    elif pos <= 0 and delta < 0:
        return False, False, True, False
    elif pos < 0 and delta > 0 and target_pos <= 0:
        return False, False, False, True
    elif pos <= 0 and target_pos > 0:
        return True, False, False, False

    return False, False, False, False


# ═══════════════════════════════════════════════════════════════════════
# STRATEGY SPECIFICATION
# ═══════════════════════════════════════════════════════════════════════

spec = StrategySpec(
    name="Composite FX Alpha",
    indicator=IndicatorSpec(
        class_name="CompositeAlpha",
        short_name="ca",
        input_names=("close", "returns"),
        param_names=(
            "w_short",
            "w_long",
            "vol_short",
            "vol_long",
            "ewma_span",
            "target_vol",
            "leverage_cap",
            "vr_low",
            "vr_high",
            "mom_w_low",
            "mom_w_normal",
            "mom_w_high",
            "dd_soft",
            "dd_hard",
            "dd_recovery",
            "n_sub",
        ),
        output_names=(
            "momentum",
            "direction",
            "vol_regime",
            "regime_weight",
            "ewma_vol",
            "vol_scale",
            "drawdown",
            "dd_multiplier",
            "target_weight",
        ),
        kernel_func=compute_composite_nb,
    ),
    signal_func=composite_signal_nb,
    signal_args_map=(
        ("target_weights", "ind.target_weight"),
        ("size", "eval:np.full(wrapper.shape_2d, np.nan)"),
    ),
    params={
        "w_short": ParamDef(21),
        "w_long": ParamDef(63),
        "vol_short": ParamDef(21),
        "vol_long": ParamDef(252),
        "ewma_span": ParamDef(30),
        "target_vol": ParamDef(0.10, sweep=[0.05, 0.08, 0.10, 0.15]),
        "leverage_cap": ParamDef(3.0),
        "vr_low": ParamDef(0.8),
        "vr_high": ParamDef(1.2),
        "mom_w_low": ParamDef(0.20),
        "mom_w_normal": ParamDef(0.30),
        "mom_w_high": ParamDef(0.50),
        "dd_soft": ParamDef(0.12),
        "dd_hard": ParamDef(0.20),
        "dd_recovery": ParamDef(0.10),
        "n_sub": ParamDef(5),
    },
    portfolio_config=PortfolioConfig(
        slippage=0.0,
        fixed_fees=0.0,
        init_cash=1_000_000.0,
        freq="1D",
        leverage=2.0,
        size_type="amount",
        accumulate=True,
        upon_opposite_entry="Reverse",
        extra_kwargs={"fees": 0.00035, "leverage_mode": "lazy"},
    ),
    takeable_args=("close_arr",),
)


if __name__ == "__main__":
    from framework.runner import run_strategy

    run_strategy(spec, mode="full")
