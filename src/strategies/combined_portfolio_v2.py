"""Combined Multi-Horizon FX Portfolio — v2 (regime-adaptive + global leverage).

Layers a global vol-targeting + drawdown cap on top of the v1 combined
portfolio and introduces a ``regime_adaptive`` allocation alongside the
existing ``risk_parity / equal / mr_heavy`` modes. Designed for the
CAGR 10-15% / Max DD < 35% objective set in Phase 13 of the research
plan, where the bottleneck was identified as the absence of global
leverage in v1 rather than the sub-strategy alphas themselves.

Key pieces
----------
- :func:`compute_vol_regime` — 20/252-day volatility ratio, shift(1).
- :func:`compute_trend_score` — % of strategies with positive 63-day
  drift, shift(1).
- :data:`DEFAULT_REGIME_WEIGHTS_3STRAT` — **hard-coded priors** (6 regime
  cells × 3 strategies). These are not to be fitted / grid-searched.
- :func:`compute_global_leverage` — vol targeting using ``max(vol_21,
  vol_63)`` (pessimistic) with a leverage cap, shift(1).
- :func:`compute_dd_cap_scale` — lagged DD cap that de-leverages
  progressively as the prelev drawdown deepens, shift(1) guarantees no
  look-ahead.
- :func:`build_combined_portfolio_v2` — orchestrator returning the same
  dict contract as v1 for drop-in comparability, plus extra diagnostic
  series (``leverage_ts``, ``dd_scale_ts``, ``regime_ts``).

Equivalence with v1
-------------------
When called with ``allocation="risk_parity"``, ``target_vol=None`` and
``dd_cap_enabled=False`` this module produces **bit-identical** returns
and ``pf.stats()`` vs :func:`combined_portfolio.build_combined_portfolio`.
This is guarded by ``tests/test_combined_portfolio_v2.py``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import vectorbtpro as vbt

from strategies.combined_core import (
    build_native_combined,
    combined_returns_from_pf,
    window_metrics,
)
from strategies.combined_portfolio import (
    WF_PERIODS,
    _INIT_CASH,
    _compute_weights_ts,
    get_strategy_daily_returns,
)


# ═══════════════════════════════════════════════════════════════════════
# PRODUCTION CONFIGURATION (MR-heavy + 2 orthogonal diversifiers)
# ═══════════════════════════════════════════════════════════════════════


# Production allocation: the recommended live-trading configuration.
# MR Macro carries the bulk of the alpha (intraday VWAP mean reversion
# filtered by US macro regime, Sharpe 0.82 standalone), while
# TS Momentum 3-pair (trend following on EUR/GBP/JPY, no USD-CAD) and
# RSI Daily 4-pair (daily mean reversion on 4 pairs) act as
# near-orthogonal diversifiers whose 2019/2023 alpha plugs the two
# weak years of the MR Macro standalone.
PRODUCTION_WEIGHTS: dict[str, float] = {
    "MR_Macro": 0.80,
    "TS_Momentum_3p": 0.10,
    "RSI_Daily_4p": 0.10,
}
PRODUCTION_TARGET_VOL: float = 0.28
PRODUCTION_MAX_LEVERAGE: float = 12.0


def build_production_portfolio(
    strategy_returns: dict[str, pd.Series] | None = None,
    target_vol: float = PRODUCTION_TARGET_VOL,
    max_leverage: float = PRODUCTION_MAX_LEVERAGE,
) -> dict[str, Any]:
    """Build the recommended production combined portfolio in one call.

    Thin wrapper around :func:`build_combined_portfolio_v2` that pins
    the recipe to the production allocation (MR 80% / TS3p 10% / RSI 10%),
    static weights, ``dd_cap_enabled=False`` and the pair
    ``target_vol=0.28`` / ``max_leverage=12.0`` that passes every IS,
    OOS and bootstrap gate. Intended as the canonical entry point for
    reproduction scripts and the pedagogical report — callers that
    need to tweak the config should call
    ``build_combined_portfolio_v2`` directly.

    Parameters
    ----------
    strategy_returns : dict[str, pd.Series] or None
        If ``None``, loads fresh component returns via
        :func:`get_strategy_daily_returns` (slow, ~1 minute). Pass an
        existing dict to skip the data loading when the caller has
        already computed it (useful in tests and benchmarks).
    target_vol : float
        Annualized vol target for the global vol targeting layer.
    max_leverage : float
        Cap on the global leverage multiplier.

    Returns
    -------
    dict
        Identical contract to :func:`build_combined_portfolio_v2`.
    """
    if strategy_returns is None:
        strategy_returns = get_strategy_daily_returns()
    filtered = {k: strategy_returns[k] for k in PRODUCTION_WEIGHTS}
    return build_combined_portfolio_v2(
        filtered,
        allocation="custom",
        custom_weights=PRODUCTION_WEIGHTS,
        target_vol=target_vol,
        max_leverage=max_leverage,
        dd_cap_enabled=False,
    )


# ═══════════════════════════════════════════════════════════════════════
# LEVERAGE VARIANTS — conservative and aggressive tunings of PRODUCTION
# ═══════════════════════════════════════════════════════════════════════


# The conservative and aggressive variants re-use the exact same sleeves
# and weights as PRODUCTION — only the leverage layer changes.
#
# Key empirical finding (116-combo ``target_vol × max_leverage × dd_cap``
# sweep): PRODUCTION's ``max_leverage=12`` was hitting the cap often
# enough that the realized leverage distribution was clipped, costing
# ~0.010 Sharpe WF. Raising ``max_leverage`` to 14+ removes the binding
# constraint and unlocks a stable Sharpe plateau at **0.966** across a
# wide region of the ``(target_vol, max_leverage)`` space with
# ``dd_cap_enabled=False``. The DD cap consistently *hurts* the Sharpe
# on this trio because it de-leverages drawdowns that would have
# recovered anyway — hence ``dd_cap_enabled=False`` on both variants.
#
# 1. CONSERVATIVE — low-risk upgrade over PRODUCTION.
#    ``target_vol=0.25, max_leverage=14, dd_cap_enabled=False``.
#    CAGR 13.38% (+0.27% vs PRODUCTION), MaxDD -18.41% (essentially flat
#    vs PRODUCTION's -17.93%), Sharpe WF 0.966 (+0.010 vs 0.956). This is
#    the recommended drop-in replacement for the PRODUCTION preset: same
#    risk profile, free Sharpe lift.
#
# 2. AGGRESSIVE — higher-CAGR variant for callers that want to push the
#    vol target up. ``target_vol=0.35, max_leverage=18, dd_cap_enabled=False``.
#    CAGR 18.58% (+5.47% vs PRODUCTION), MaxDD -25.57% (+7.64% worse),
#    Sharpe WF 0.966 (same plateau as CONSERVATIVE). Bootstrap P5 CAGR
#    +7.60%, target-hit rate 15.4% (the target band [10%, 15%] is below
#    the expected CAGR — use a higher CAGR target band when evaluating
#    this config).


CONSERVATIVE_WEIGHTS: dict[str, float] = dict(PRODUCTION_WEIGHTS)
CONSERVATIVE_TARGET_VOL: float = 0.25
CONSERVATIVE_MAX_LEVERAGE: float = 14.0
CONSERVATIVE_DD_CAP_ENABLED: bool = False


AGGRESSIVE_WEIGHTS: dict[str, float] = dict(PRODUCTION_WEIGHTS)
AGGRESSIVE_TARGET_VOL: float = 0.35
AGGRESSIVE_MAX_LEVERAGE: float = 18.0
AGGRESSIVE_DD_CAP_ENABLED: bool = False


def build_conservative_portfolio(
    strategy_returns: dict[str, pd.Series] | None = None,
    target_vol: float = CONSERVATIVE_TARGET_VOL,
    max_leverage: float = CONSERVATIVE_MAX_LEVERAGE,
) -> dict[str, Any]:
    """Build the *conservative* combined portfolio.

    Drop-in replacement for :func:`build_production_portfolio` that uses
    ``target_vol=0.25, max_leverage=14, dd_cap_enabled=False`` instead of
    the production ``(0.28, 12, False)`` triple. On the same sleeves and
    weights this produces a slightly higher Sharpe WF (~0.966 vs 0.956)
    for an essentially identical risk profile — the full
    ``(target_vol, max_leverage)`` grid validates the Sharpe plateau at
    0.966 for ``max_leverage >= 14`` across ``target_vol ∈ [0.22, 0.28]``.
    """
    if strategy_returns is None:
        strategy_returns = get_strategy_daily_returns()
    filtered = {k: strategy_returns[k] for k in CONSERVATIVE_WEIGHTS}
    return build_combined_portfolio_v2(
        filtered,
        allocation="custom",
        custom_weights=CONSERVATIVE_WEIGHTS,
        target_vol=target_vol,
        max_leverage=max_leverage,
        dd_cap_enabled=CONSERVATIVE_DD_CAP_ENABLED,
    )


def build_aggressive_portfolio(
    strategy_returns: dict[str, pd.Series] | None = None,
    target_vol: float = AGGRESSIVE_TARGET_VOL,
    max_leverage: float = AGGRESSIVE_MAX_LEVERAGE,
) -> dict[str, Any]:
    """Build the *aggressive* combined portfolio.

    Higher-CAGR variant with ``target_vol=0.35, max_leverage=18``,
    keeping the same sleeves and weights as PRODUCTION and CONSERVATIVE.
    Sharpe WF stays on the 0.966 plateau but CAGR rises to ~18.58% at
    the cost of MaxDD ~-25.57% (vs PRODUCTION's -17.93%). The bootstrap
    tail risk is markedly higher (P5 MaxDD ~-40%) — do NOT deploy this
    config without a broker-level margin gate and an explicit risk
    budget that accepts the increased drawdown profile.
    """
    if strategy_returns is None:
        strategy_returns = get_strategy_daily_returns()
    filtered = {k: strategy_returns[k] for k in AGGRESSIVE_WEIGHTS}
    return build_combined_portfolio_v2(
        filtered,
        allocation="custom",
        custom_weights=AGGRESSIVE_WEIGHTS,
        target_vol=target_vol,
        max_leverage=max_leverage,
        dd_cap_enabled=AGGRESSIVE_DD_CAP_ENABLED,
    )


# ═══════════════════════════════════════════════════════════════════════
# REGIME DETECTION
# ═══════════════════════════════════════════════════════════════════════


# Regime weights are **hard-coded priors** derived from the 3-strategy
# Phase 13 analysis (MR Macro Sharpe 1.07 dominates, XS and TS around
# 0.70). The matrix tilts further toward MR Macro in high-vol regimes
# (MR is intraday and thrives on chop) and gives momentum more room in
# low-vol trending regimes. DO NOT grid-search these values — they are
# priors, not hyperparameters.
DEFAULT_REGIME_WEIGHTS_3STRAT: dict[tuple[str, str], dict[str, float]] = {
    ("low", "up"): {"MR_Macro": 0.40, "XS_Momentum": 0.35, "TS_Momentum_RSI": 0.25},
    ("low", "down"): {"MR_Macro": 0.55, "XS_Momentum": 0.25, "TS_Momentum_RSI": 0.20},
    ("normal", "up"): {"MR_Macro": 0.45, "XS_Momentum": 0.30, "TS_Momentum_RSI": 0.25},
    ("normal", "down"): {
        "MR_Macro": 0.60,
        "XS_Momentum": 0.20,
        "TS_Momentum_RSI": 0.20,
    },
    ("high", "up"): {"MR_Macro": 0.60, "XS_Momentum": 0.25, "TS_Momentum_RSI": 0.15},
    ("high", "down"): {"MR_Macro": 0.75, "XS_Momentum": 0.15, "TS_Momentum_RSI": 0.10},
}


def compute_vol_regime(
    proxy_returns: pd.Series,
    short_window: int = 20,
    long_window: int = 252,
    low_ratio: float = 0.8,
    high_ratio: float = 1.2,
) -> pd.Series:
    """Return a string regime label per bar: ``low`` / ``normal`` / ``high``.

    Regime is defined by the ratio of the short-window realized vol to
    the long-window realized vol, both shifted by 1 day so the regime
    at ``t`` depends only on returns strictly before ``t``. The 0.8 and
    1.2 thresholds are the standard TAA cuts and are not tuned.
    """
    vol_short = proxy_returns.vbt.rolling_std(
        short_window, minp=short_window // 2, ddof=1
    ) * np.sqrt(252)
    vol_long = proxy_returns.vbt.rolling_std(
        long_window, minp=long_window // 4, ddof=1
    ) * np.sqrt(252)
    ratio = (vol_short / vol_long.replace(0, np.nan)).shift(1)

    regime = pd.Series("normal", index=proxy_returns.index, dtype=object)
    regime[ratio < low_ratio] = "low"
    regime[ratio > high_ratio] = "high"
    # NaN ratio (warmup) falls back to "normal".
    return regime


def compute_trend_score(
    strategy_rets: pd.DataFrame,
    lookback: int = 63,
    min_periods: int = 30,
) -> pd.Series:
    """Return the fraction of strategies with a positive rolling-mean return.

    The rolling mean is computed on each strategy's daily returns over
    ``lookback`` bars, then ``.shift(1)`` ensures no look-ahead. The
    output is in ``[0, 1]`` with NaN during the warmup window.
    """
    mean_rets = strategy_rets.vbt.rolling_mean(lookback, minp=min_periods).shift(1)
    positive = mean_rets > 0
    return positive.mean(axis=1)


def compute_regime_adaptive_weights(
    common: pd.DataFrame,
    regime_weights: dict[tuple[str, str], dict[str, float]] | None = None,
    trend_threshold: float = 0.5,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Time-varying weights from vol regime × trend regime.

    Returns
    -------
    weights_ts : DataFrame
        Allocation per bar, columns = ``common.columns``. Rows before
        the regime detectors have enough warmup fall back to equal
        weights (``1/N``) so no row is all-NaN.
    vol_regime : Series of str
        ``low`` / ``normal`` / ``high`` per bar (for diagnostics).
    trend_score : Series of float
        Fraction of strategies in positive trend per bar (for diagnostics).
    """
    if regime_weights is None:
        regime_weights = DEFAULT_REGIME_WEIGHTS_3STRAT

    proxy_rets = common.mean(axis=1)
    vol_regime = compute_vol_regime(proxy_rets)
    trend_score = compute_trend_score(common)

    # trend_label: "up" when >= threshold fraction of strats are trending,
    # "down" otherwise. NaN trend_score (warmup) → "down" by convention.
    trend_label = pd.Series("down", index=common.index, dtype=object)
    trend_label[trend_score >= trend_threshold] = "up"

    weights_ts = pd.DataFrame(
        np.nan, index=common.index, columns=common.columns, dtype=float
    )
    for (vol_key, trend_key), weight_dict in regime_weights.items():
        mask = (vol_regime == vol_key) & (trend_label == trend_key)
        if not mask.any():
            continue
        for strat, w in weight_dict.items():
            if strat in weights_ts.columns:
                weights_ts.loc[mask, strat] = float(w)

    # Rows with any NaN fall back to equal weights — happens only during
    # warmup or if a regime weights dict omits some strategies.
    n_cols = len(common.columns)
    eq_w = 1.0 / n_cols
    weights_ts = weights_ts.fillna(eq_w)

    # Renormalize (guard against user-supplied rows summing != 1).
    row_sum = weights_ts.sum(axis=1)
    weights_ts = weights_ts.div(row_sum.where(row_sum > 0, 1.0), axis=0)

    return weights_ts, vol_regime, trend_score


# ═══════════════════════════════════════════════════════════════════════
# GLOBAL LEVERAGE AND DD CAP
# ═══════════════════════════════════════════════════════════════════════


def compute_global_leverage(
    port_rets_base: pd.Series,
    target_vol: float,
    max_leverage: float = 3.0,
    min_vol_floor: float = 0.02,
) -> pd.Series:
    """Vol-targeted leverage series, shifted by 1 bar (no look-ahead).

    Uses ``max(vol_21, vol_63)`` as the realized vol estimate to stay
    pessimistic during regime transitions (the short window reacts
    faster, the long window is more stable). Annualized via ``sqrt(252)``.
    """
    vol_21 = port_rets_base.vbt.rolling_std(21, minp=10, ddof=1) * np.sqrt(252)
    vol_63 = port_rets_base.vbt.rolling_std(63, minp=30, ddof=1) * np.sqrt(252)
    realized_vol = pd.concat([vol_21, vol_63], axis=1).max(axis=1)
    leverage = (
        (target_vol / realized_vol.clip(lower=min_vol_floor))
        .clip(upper=max_leverage)
        .shift(1)
        .fillna(1.0)
    )
    return leverage


# De-leveraging schedule: abs(DD) → leverage multiplier.
# DD in [0, 0.10): full leverage.
# DD in [0.10, 0.20): taper from 1.0 to 0.6.
# DD in [0.20, 0.30): taper from 0.6 to 0.35.
# DD in [0.30, 0.35]: taper from 0.35 to 0.15.
# DD > 0.35: clipped at 0.15 (soft bottom, not a full flatten so we can
# participate in the recovery).
_DD_BREAKPOINTS = np.array([0.0, 0.10, 0.20, 0.30, 0.35])
_DD_LEV_SCALES = np.array([1.0, 1.0, 0.6, 0.35, 0.15])


def compute_dd_cap_scale(
    port_rets_prelev: pd.Series,
    breakpoints: np.ndarray | tuple[float, ...] | None = None,
    scales: np.ndarray | tuple[float, ...] | None = None,
) -> pd.Series:
    """Lagged drawdown-based leverage scaling.

    Computes running drawdown on the **pre-DD-cap** equity (i.e. the
    equity that would have resulted from just applying the global
    leverage, before the DD cap itself modifies returns). The scale for
    bar ``t`` depends only on drawdown known strictly before ``t`` via
    ``.shift(1)``, which breaks the circular dependency between the
    DD cap and the equity it modifies.

    Parameters
    ----------
    breakpoints, scales : array-like, optional
        Override the default de-leveraging schedule. Both must be
        monotonically ordered and the same length. When None (default)
        the module-level ``_DD_BREAKPOINTS`` / ``_DD_LEV_SCALES``
        schedule is used — the historical Phase 13 schedule.
    """
    bps = np.asarray(breakpoints) if breakpoints is not None else _DD_BREAKPOINTS
    scl = np.asarray(scales) if scales is not None else _DD_LEV_SCALES
    if bps.shape != scl.shape:
        raise ValueError(
            f"DD schedule length mismatch: breakpoints={bps.shape}, scales={scl.shape}"
        )
    equity = (1.0 + port_rets_prelev.fillna(0.0)).cumprod()
    running_max = equity.expanding().max()
    dd = (equity / running_max - 1.0).shift(1).fillna(0.0)
    dd_abs = (-dd).clip(lower=0.0)
    scale_values = np.interp(dd_abs.values, bps, scl)
    return pd.Series(scale_values, index=port_rets_prelev.index)


# ═══════════════════════════════════════════════════════════════════════
# ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════


def build_combined_portfolio_v2(
    strategy_returns: dict[str, pd.Series],
    allocation: str = "regime_adaptive",
    target_vol: float | None = 0.12,
    max_leverage: float = 3.0,
    dd_cap_enabled: bool = True,
    custom_weights: dict[str, float] | None = None,
    regime_weights: dict[tuple[str, str], dict[str, float]] | None = None,
) -> dict[str, Any]:
    """Build the v2 combined portfolio.

    Parameters
    ----------
    strategy_returns : dict[str, pd.Series]
        Daily returns per sub-strategy, as produced by
        :func:`combined_portfolio.get_strategy_daily_returns`.
    allocation : str
        One of ``risk_parity`` / ``equal`` / ``mr_heavy`` / ``custom``
        (delegates to v1's :func:`combined_portfolio._compute_weights_ts`)
        or ``regime_adaptive`` (uses vol × trend regime detection).
    target_vol : float or None
        Annualized vol target for the global vol targeting layer. Set
        to ``None`` to disable the vol targeting entirely (leverage=1);
        this is the v1-equivalence mode.
    max_leverage : float
        Cap on the global leverage multiplier.
    dd_cap_enabled : bool
        If ``False``, the DD cap is bypassed (dd_scale=1). Combined
        with ``target_vol=None`` and ``allocation`` in the v1 modes,
        this reproduces v1 bit-identically.
    custom_weights : dict
        Passed through to v1 ``_compute_weights_ts`` when
        ``allocation="custom"``.
    regime_weights : dict
        Override for :data:`DEFAULT_REGIME_WEIGHTS_3STRAT`.

    Returns
    -------
    dict
        Same keys as v1's result plus ``leverage_ts``, ``dd_scale_ts``,
        ``vol_regime_ts``, ``trend_score_ts``, ``port_rets_base``,
        and ``port_rets_prelev`` for diagnostics.
    """
    all_rets = pd.DataFrame(strategy_returns)
    common = all_rets.dropna()

    if allocation == "regime_adaptive":
        weights_ts, vol_regime_ts, trend_score_ts = compute_regime_adaptive_weights(
            common, regime_weights=regime_weights
        )
    else:
        weights_ts = _compute_weights_ts(common, allocation, custom_weights)
        vol_regime_ts = pd.Series("normal", index=common.index)
        trend_score_ts = pd.Series(np.nan, index=common.index)

    # Base (unlevered) proxy returns — kept as a pandas Series because
    # compute_global_leverage and compute_dd_cap_scale need a 1-D driver
    # series to derive the leverage/DD scales. These are intermediate
    # diagnostics, not the final portfolio returns.
    port_rets_base = (common * weights_ts).sum(axis=1)

    # Optional vol targeting.
    if target_vol is None:
        leverage_ts = pd.Series(1.0, index=common.index)
    else:
        leverage_ts = compute_global_leverage(
            port_rets_base, target_vol=target_vol, max_leverage=max_leverage
        )
    port_rets_prelev = port_rets_base * leverage_ts

    # Optional DD cap.
    if dd_cap_enabled:
        dd_scale_ts = compute_dd_cap_scale(port_rets_prelev)
    else:
        dd_scale_ts = pd.Series(1.0, index=common.index)

    # Bake leverage + DD cap into the allocations and route through the
    # native vbt PFO + from_optimizer pipeline. ``leverage_cap`` must be
    # >= max row sum of the final allocations, which is bounded by
    # ``max_leverage × 1.0`` (dd_scale_ts <= 1). Add a safety margin.
    final_allocations = weights_ts.mul(leverage_ts * dd_scale_ts, axis=0)
    leverage_cap = max(float(max_leverage) * 1.5, 10.0)
    pf_combined, _, _ = build_native_combined(
        common,
        final_allocations,
        init_cash=_INIT_CASH,
        leverage_cap=leverage_cap,
    )
    port_rets_v2 = combined_returns_from_pf(pf_combined)

    # Walk-forward per-year metrics — native window Sharpe/TR/MDD.
    wf_sharpes: list[float] = []
    wf_returns: list[float] = []
    wf_max_dd: list[float] = []
    for start, end in WF_PERIODS:
        m = window_metrics(pf_combined, start, end)
        wf_sharpes.append(m["sharpe"])
        wf_returns.append(m["total_return"])
        wf_max_dd.append(m["max_drawdown"])

    full_sr = float(pf_combined.sharpe_ratio)
    if np.isnan(full_sr):
        full_sr = 0.0

    return {
        "weights_ts": weights_ts,
        "weights": weights_ts.mean().to_dict(),
        "allocation": allocation,
        "target_vol": target_vol,
        "max_leverage": max_leverage,
        "dd_cap_enabled": dd_cap_enabled,
        "portfolio_returns": port_rets_v2,
        "port_rets_base": port_rets_base,
        "port_rets_prelev": port_rets_prelev,
        "leverage_ts": leverage_ts,
        "dd_scale_ts": dd_scale_ts,
        "vol_regime_ts": vol_regime_ts,
        "trend_score_ts": trend_score_ts,
        "component_returns": common,
        "correlation": common.corr(),
        "pf_combined": pf_combined,
        "sharpe": full_sr,
        "annual_return": float(pf_combined.annualized_return),
        "annual_vol": float(pf_combined.annualized_volatility),
        "max_drawdown": float(pf_combined.max_drawdown),
        "wf_sharpes": wf_sharpes,
        "wf_returns": wf_returns,
        "wf_max_dd": wf_max_dd,
        "wf_avg_sharpe": float(np.mean(wf_sharpes)),
        "wf_pos_years": sum(1 for s in wf_sharpes if s > 0),
    }


# ═══════════════════════════════════════════════════════════════════════
# BENCHMARK HELPERS
# ═══════════════════════════════════════════════════════════════════════


def _format_pct(x: float) -> str:
    return f"{x * 100:>7.2f}%"


def _collect_v1_baselines(
    strat_rets: dict[str, pd.Series],
) -> dict[str, dict[str, Any]]:
    """Run the three v1 allocations as reference baselines."""
    from strategies.combined_portfolio import build_combined_portfolio

    out: dict[str, dict[str, Any]] = {}
    for alloc in ("risk_parity", "equal", "mr_heavy"):
        out[f"v1/{alloc}"] = build_combined_portfolio(strat_rets, allocation=alloc)
    return out


def _collect_v2_no_leverage(
    strat_rets: dict[str, pd.Series],
) -> dict[str, dict[str, Any]]:
    """Run v2 with leverage disabled — two variants for the equivalence check."""
    return {
        # Must match v1 risk_parity bit-for-bit.
        "v2_nolev/risk_parity": build_combined_portfolio_v2(
            strat_rets,
            allocation="risk_parity",
            target_vol=None,
            dd_cap_enabled=False,
        ),
        # Pure regime-allocation effect (no leverage layer).
        "v2_nolev/regime_adaptive": build_combined_portfolio_v2(
            strat_rets,
            allocation="regime_adaptive",
            target_vol=None,
            dd_cap_enabled=False,
        ),
    }


def _collect_v2_regime_sweep(
    strat_rets: dict[str, pd.Series],
) -> dict[str, dict[str, Any]]:
    """Conservative regime-adaptive configs with DD cap ON (max_leverage=3)."""
    out: dict[str, dict[str, Any]] = {}
    for target_vol in (0.08, 0.10, 0.12, 0.14):
        key = f"v2_regime/target_vol={target_vol:.2f}_ml=3_DDcap=ON"
        out[key] = build_combined_portfolio_v2(
            strat_rets,
            allocation="regime_adaptive",
            target_vol=target_vol,
            max_leverage=3.0,
            dd_cap_enabled=True,
        )
    return out


# Aggressive configs rely on DD cap OFF because on this combined the
# soft-cap actually hurts the Sharpe (Phase 14 finding) — it de-leverages
# drawdowns that would have recovered. DO NOT run these in production
# without a broker-level margin check: 12-15x notional on a daily-rebalanced
# combined requires ~7-10% margin available at all times.
_MR_HEAVY_WEIGHTS: dict[str, float] = {
    "MR_Macro": 0.80,
    "XS_Momentum": 0.10,
    "TS_Momentum_RSI": 0.10,
}
_MR_TS_WEIGHTS: dict[str, float] = {  # Phase 16
    "MR_Macro": 0.90,
    "TS_Momentum_RSI": 0.10,
}
_MR_TS3P_WEIGHTS: dict[str, float] = {  # Phase 17
    "MR_Macro": 0.90,
    "TS_Momentum_3p": 0.10,
}


def _collect_v2_mr_heavy_sweep(
    strat_rets: dict[str, pd.Series],
) -> dict[str, dict[str, Any]]:
    """Aggressive MR80/XS10/TS10 configs — Phase 14 aggressive target."""
    out: dict[str, dict[str, Any]] = {}
    for target_vol in (0.20, 0.22, 0.25, 0.28):
        key = f"v2_MR80/tv={target_vol:.2f}_ml=10_DDcap=OFF"
        out[key] = build_combined_portfolio_v2(
            strat_rets,
            allocation="custom",
            custom_weights=_MR_HEAVY_WEIGHTS,
            target_vol=target_vol,
            max_leverage=10.0,
            dd_cap_enabled=False,
        )
    return out


def _collect_v2_phase16_sweep(
    strat_rets: dict[str, pd.Series],
) -> dict[str, dict[str, Any]]:
    """Phase 16 — drop XS Momentum, keep MR Macro + TS Momentum RSI."""
    filtered = {k: strat_rets[k] for k in ("MR_Macro", "TS_Momentum_RSI")}
    out: dict[str, dict[str, Any]] = {}
    for target_vol, max_lev in [(0.28, 10), (0.28, 12), (0.28, 15)]:
        key = f"v2_MR90_TS10/tv={target_vol:.2f}_ml={max_lev}_DDcap=OFF"
        out[key] = build_combined_portfolio_v2(
            filtered,
            allocation="custom",
            custom_weights=_MR_TS_WEIGHTS,
            target_vol=target_vol,
            max_leverage=float(max_lev),
            dd_cap_enabled=False,
        )
    return out


def _collect_v2_phase17_sweep(
    strat_rets: dict[str, pd.Series],
) -> dict[str, dict[str, Any]]:
    """Phase 17 — drop USD-CAD from TS Momentum (worst pair 2019/2022/2023)."""
    filtered = {k: strat_rets[k] for k in ("MR_Macro", "TS_Momentum_3p")}
    out: dict[str, dict[str, Any]] = {}
    for target_vol, max_lev in [(0.28, 10), (0.28, 12), (0.28, 15)]:
        key = f"v2_MR90_TS3p10/tv={target_vol:.2f}_ml={max_lev}_DDcap=OFF"
        out[key] = build_combined_portfolio_v2(
            filtered,
            allocation="custom",
            custom_weights=_MR_TS3P_WEIGHTS,
            target_vol=target_vol,
            max_leverage=float(max_lev),
            dd_cap_enabled=False,
        )
    return out


def _collect_v2_phase18_sweep(
    strat_rets: dict[str, pd.Series],
) -> dict[str, dict[str, Any]]:
    """Phase 18 — add RSI Daily 4-pair as third orthogonal sleeve."""
    filtered = {
        k: strat_rets[k] for k in ("MR_Macro", "TS_Momentum_3p", "RSI_Daily_4p")
    }
    out: dict[str, dict[str, Any]] = {}
    for target_vol, max_lev in [(0.28, 10), (0.28, 12), (0.28, 14)]:
        key = f"v2_MR80_TS3p10_RSI10/tv={target_vol:.2f}_ml={max_lev}_DDcap=OFF"
        out[key] = build_combined_portfolio_v2(
            filtered,
            allocation="custom",
            custom_weights=PHASE18_WEIGHTS,
            target_vol=target_vol,
            max_leverage=float(max_lev),
            dd_cap_enabled=False,
        )
    return out


def _print_summary_table(results: dict[str, dict[str, Any]]) -> None:
    """Print the CAGR/Vol/MaxDD/Sharpe table. ★ marks in-target configs."""
    headers = (
        f"{'Config':<45} "
        f"{'CAGR':>9} "
        f"{'Vol':>8} "
        f"{'MaxDD':>9} "
        f"{'Sharpe':>8} "
        f"{'WF avg':>8} "
        f"{'Pos':>5}"
    )
    print("\n" + headers)
    print("-" * len(headers))
    for key, res in results.items():
        cagr = res["annual_return"]
        dd = res["max_drawdown"]
        in_target = (0.10 <= cagr <= 0.15) and (abs(dd) < 0.35)
        mark = "★" if in_target else " "
        print(
            f"{mark} {key:<43} "
            f"{_format_pct(cagr)} "
            f"{_format_pct(res['annual_vol'])} "
            f"{_format_pct(dd)} "
            f"{res['sharpe']:>7.3f}  "
            f"{res['wf_avg_sharpe']:>7.3f}  "
            f"{res['wf_pos_years']}/7"
        )


def _print_recommended_notes() -> None:
    """Static notes about the Phase 18 recommended configuration."""
    print("\nTarget: CAGR ∈ [10%, 15%] AND Max DD < 35% (rows marked with ★)")
    print(
        "Recommended (Phase 18 — add RSI Daily 4-pair as third sleeve): "
        "MR80/TS3p10/RSI10 tv=0.28 ml=12 DDcap=OFF"
    )
    print("  → IS CAGR 13.33%, MaxDD -17.93%, Sharpe 0.94, 6/7 WF positive")
    print(
        "  → OOS 2025-04+ CAGR 11.52%, MaxDD -6.27%, Sharpe 1.44 "
        "(vs Phase 17 OOS Sharpe 1.24 — +16% out-of-sample lift)"
    )
    print(
        "  → Bootstrap tail risk: P5 MaxDD = -30.68% (vs -33.98% P17, "
        "-47.46% P15), P5 CAGR 5.54%, positive fraction 99.8%, "
        "target hit 39.0% (vs 33.4% P17)."
    )
    print(
        "  → RSI Daily is near-zero correlated with MR Macro (+0.056) "
        "and slightly anti-correlated with TS Momentum (-0.251) — "
        "a textbook diversifier."
    )


def run_v2_benchmark(output_dir: str = "results/combined_v2") -> dict[str, Any]:
    """Run v1 and v2 back-to-back and print a comparison table.

    Thin orchestrator that composes the per-phase sweep helpers and
    prints the summary. See :func:`_collect_v2_phase18_sweep` for the
    currently recommended production config.
    """
    import os

    os.makedirs(output_dir, exist_ok=True)
    print("=" * 72)
    print("  Combined Portfolio v2 — Benchmark vs v1")
    print("=" * 72)

    strat_rets = get_strategy_daily_returns()

    results: dict[str, Any] = {}
    results.update(_collect_v1_baselines(strat_rets))
    results.update(_collect_v2_no_leverage(strat_rets))
    results.update(_collect_v2_regime_sweep(strat_rets))
    results.update(_collect_v2_mr_heavy_sweep(strat_rets))
    results.update(_collect_v2_phase16_sweep(strat_rets))
    results.update(_collect_v2_phase17_sweep(strat_rets))
    results.update(_collect_v2_phase18_sweep(strat_rets))

    _print_summary_table(results)
    _print_recommended_notes()
    return results


# ═══════════════════════════════════════════════════════════════════════
# BACK-COMPAT ALIASES
# ═══════════════════════════════════════════════════════════════════════
# These keep the historic PHASE18_*/PHASE19_*/build_phase18/build_phase19
# names alive so tests, scripts, and LaTeX report pseudocode that import
# by the old names continue to work unchanged. Remove once all callers
# (see git grep for PHASE18_WEIGHTS / build_phase19_balanced_portfolio)
# have migrated to the semantic names.

PHASE18_WEIGHTS = PRODUCTION_WEIGHTS
PHASE18_TARGET_VOL = PRODUCTION_TARGET_VOL
PHASE18_MAX_LEVERAGE = PRODUCTION_MAX_LEVERAGE
build_phase18_portfolio = build_production_portfolio

PHASE19_BALANCED_WEIGHTS = CONSERVATIVE_WEIGHTS
PHASE19_BALANCED_TARGET_VOL = CONSERVATIVE_TARGET_VOL
PHASE19_BALANCED_MAX_LEVERAGE = CONSERVATIVE_MAX_LEVERAGE
PHASE19_BALANCED_DD_CAP_ENABLED = CONSERVATIVE_DD_CAP_ENABLED
build_phase19_balanced_portfolio = build_conservative_portfolio

PHASE19_AGGRESSIVE_WEIGHTS = AGGRESSIVE_WEIGHTS
PHASE19_AGGRESSIVE_TARGET_VOL = AGGRESSIVE_TARGET_VOL
PHASE19_AGGRESSIVE_MAX_LEVERAGE = AGGRESSIVE_MAX_LEVERAGE
PHASE19_AGGRESSIVE_DD_CAP_ENABLED = AGGRESSIVE_DD_CAP_ENABLED
build_phase19_aggressive_portfolio = build_aggressive_portfolio


# ═══════════════════════════════════════════════════════════════════════
# CLI — single production run + full benchmark suite
# ═══════════════════════════════════════════════════════════════════════


if __name__ == "__main__":
    import sys
    from pathlib import Path as _Path

    _SRC = _Path(__file__).resolve().parent.parent
    if str(_SRC) not in sys.path:
        sys.path.insert(0, str(_SRC))

    from framework.pipeline_utils import (
        METRIC_LABELS,
        SHARPE_RATIO,
        analyze_portfolio,
        apply_vbt_plot_defaults,
    )
    from utils import apply_vbt_settings

    # ─────────────────────────────────────────────────────────────────
    # CONFIGURATION
    # ─────────────────────────────────────────────────────────────────
    OUTPUT_DIR = "results/combined_v2"
    SHOW_CHARTS = True
    RUN_BENCHMARK = True  # set False to only run the SINGLE mode
    _METRIC_NAME = METRIC_LABELS[SHARPE_RATIO]

    def _header(label: str) -> None:
        bar = "█" * 78
        print(f"\n{bar}")
        print(f"██  {label.ljust(72)}  ██")
        print(f"{bar}\n")

    apply_vbt_settings()
    apply_vbt_plot_defaults()
    vbt.settings.returns.year_freq = pd.Timedelta(days=252)

    # 1) SINGLE RUN — the recommended production config
    _header("COMBINED PORTFOLIO LEVERED  ·  PRODUCTION SINGLE RUN")
    res_prod = build_production_portfolio()
    analyze_portfolio(
        res_prod["pf_combined"],
        name="Combined Production Portfolio (MR 80% + TS3p 10% + RSI 10%)",
        output_dir=OUTPUT_DIR,
        show_charts=SHOW_CHARTS,
    )

    # 2) BENCHMARK — full baseline/variant sweep comparison
    if RUN_BENCHMARK:
        _header("COMBINED PORTFOLIO LEVERED  ·  BENCHMARK")
        run_v2_benchmark(output_dir=OUTPUT_DIR)

    print("\nAll modes done.")
