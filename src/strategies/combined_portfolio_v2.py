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
This is guarded by ``tests/test_combined_v2_equivalence.py``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import vectorbtpro as vbt

from strategies.combined_portfolio import (
    WF_PERIODS,
    _INIT_CASH,
    _compute_weights_ts,
    get_strategy_daily_returns,
    returns_to_pf,
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
    ("low", "up"):      {"MR_Macro": 0.40, "XS_Momentum": 0.35, "TS_Momentum_RSI": 0.25},
    ("low", "down"):    {"MR_Macro": 0.55, "XS_Momentum": 0.25, "TS_Momentum_RSI": 0.20},
    ("normal", "up"):   {"MR_Macro": 0.45, "XS_Momentum": 0.30, "TS_Momentum_RSI": 0.25},
    ("normal", "down"): {"MR_Macro": 0.60, "XS_Momentum": 0.20, "TS_Momentum_RSI": 0.20},
    ("high", "up"):     {"MR_Macro": 0.60, "XS_Momentum": 0.25, "TS_Momentum_RSI": 0.15},
    ("high", "down"):   {"MR_Macro": 0.75, "XS_Momentum": 0.15, "TS_Momentum_RSI": 0.10},
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
    vol_short = proxy_returns.rolling(short_window, min_periods=short_window // 2).std() * np.sqrt(252)
    vol_long = proxy_returns.rolling(long_window, min_periods=long_window // 4).std() * np.sqrt(252)
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
    mean_rets = strategy_rets.rolling(lookback, min_periods=min_periods).mean().shift(1)
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
    vol_21 = port_rets_base.rolling(21, min_periods=10).std() * np.sqrt(252)
    vol_63 = port_rets_base.rolling(63, min_periods=30).std() * np.sqrt(252)
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


def compute_dd_cap_scale(port_rets_prelev: pd.Series) -> pd.Series:
    """Lagged drawdown-based leverage scaling.

    Computes running drawdown on the **pre-DD-cap** equity (i.e. the
    equity that would have resulted from just applying the global
    leverage, before the DD cap itself modifies returns). The scale for
    bar ``t`` depends only on drawdown known strictly before ``t`` via
    ``.shift(1)``, which breaks the circular dependency between the
    DD cap and the equity it modifies.
    """
    equity = (1.0 + port_rets_prelev.fillna(0.0)).cumprod()
    running_max = equity.expanding().max()
    dd = (equity / running_max - 1.0).shift(1).fillna(0.0)
    dd_abs = (-dd).clip(lower=0.0)
    scale_values = np.interp(dd_abs.values, _DD_BREAKPOINTS, _DD_LEV_SCALES)
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

    # Base (unlevered) portfolio returns.
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
    port_rets_v2 = port_rets_prelev * dd_scale_ts

    pf_combined = returns_to_pf(port_rets_v2)

    # Walk-forward per-year metrics — one VBT portfolio per window.
    wf_sharpes: list[float] = []
    wf_returns: list[float] = []
    wf_max_dd: list[float] = []
    for start, end in WF_PERIODS:
        p = port_rets_v2.loc[start:end]
        if len(p) < 20:
            wf_sharpes.append(0.0)
            wf_returns.append(0.0)
            wf_max_dd.append(0.0)
            continue
        pf_w = returns_to_pf(p)
        sr = float(pf_w.sharpe_ratio)
        tr = float(pf_w.total_return)
        mdd = float(pf_w.max_drawdown)
        wf_sharpes.append(0.0 if np.isnan(sr) else sr)
        wf_returns.append(0.0 if np.isnan(tr) else tr)
        wf_max_dd.append(0.0 if np.isnan(mdd) else mdd)

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
# CLI
# ═══════════════════════════════════════════════════════════════════════


def _format_pct(x: float) -> str:
    return f"{x * 100:>7.2f}%"


def run_v2_benchmark(output_dir: str = "results/combined_v2") -> dict[str, Any]:
    """Run v1 and v2 back-to-back and print a comparison table."""
    import os

    from strategies.combined_portfolio import build_combined_portfolio

    os.makedirs(output_dir, exist_ok=True)

    print("=" * 72)
    print("  Combined Portfolio v2 — Benchmark vs v1")
    print("=" * 72)

    strat_rets = get_strategy_daily_returns()

    results: dict[str, Any] = {}

    # v1 baselines
    for alloc in ("risk_parity", "equal", "mr_heavy"):
        res = build_combined_portfolio(strat_rets, allocation=alloc)
        results[f"v1/{alloc}"] = res

    # v2 risk_parity no-leverage → must match v1 risk_parity
    results["v2_nolev/risk_parity"] = build_combined_portfolio_v2(
        strat_rets, allocation="risk_parity", target_vol=None, dd_cap_enabled=False
    )

    # v2 regime_adaptive no leverage (pure regime allocation effect)
    results["v2_nolev/regime_adaptive"] = build_combined_portfolio_v2(
        strat_rets,
        allocation="regime_adaptive",
        target_vol=None,
        dd_cap_enabled=False,
    )

    # v2 conservative configs (safe default — max_leverage=3)
    for target_vol in (0.08, 0.10, 0.12, 0.14):
        key = f"v2_regime/target_vol={target_vol:.2f}_ml=3_DDcap=ON"
        results[key] = build_combined_portfolio_v2(
            strat_rets,
            allocation="regime_adaptive",
            target_vol=target_vol,
            max_leverage=3.0,
            dd_cap_enabled=True,
        )

    # v2 aggressive configs tuned for the CAGR 10-15% target. These rely
    # on DD cap OFF because on this combined the soft-cap actually hurts
    # the Sharpe (Phase 14 finding) — the DD cap de-leverages drawdowns
    # that would have recovered. DO NOT run these in production without
    # a broker-level margin check: 15x notional on a daily-rebalanced
    # combined requires ~6-7% margin available at all times.
    mr_heavy_weights = {
        "MR_Macro": 0.80,
        "XS_Momentum": 0.10,
        "TS_Momentum_RSI": 0.10,
    }
    for target_vol in (0.18, 0.20, 0.22, 0.25):
        key = f"v2_MR80/tv={target_vol:.2f}_ml=15_DDcap=OFF"
        results[key] = build_combined_portfolio_v2(
            strat_rets,
            allocation="custom",
            custom_weights=mr_heavy_weights,
            target_vol=target_vol,
            max_leverage=15.0,
            dd_cap_enabled=False,
        )

    # Summary table
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
        # Mark configs that hit the 10-15% CAGR / < 35% DD target.
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

    # Highlight the currently recommended config for the target.
    print("\nTarget: CAGR ∈ [10%, 15%] AND Max DD < 35% (rows marked with ★)")
    print(
        "Recommended (Phase 15 post-stress-test): "
        "MR80/tv=0.28_ml=10_DDcap=OFF"
    )
    print(
        "  → CAGR ~12.4%, MaxDD ~-26%, Sharpe 0.75, 6/7 WF positive"
    )
    print(
        "  → max_leverage=10 is safer for retail FX (30:1 EU cap) "
        "than the tv=0.22/ml=15 alternative and has a BETTER "
        "risk-adjusted profile."
    )
    print(
        "  → Bootstrap tail risk: P5 Max DD = -47% on 1000 resamples, "
        "so real margin headroom > 30% is mandatory in production."
    )

    return results


if __name__ == "__main__":
    import sys
    from pathlib import Path as _Path

    _SRC = _Path(__file__).resolve().parent.parent
    if str(_SRC) not in sys.path:
        sys.path.insert(0, str(_SRC))

    from utils import apply_vbt_settings

    apply_vbt_settings()
    vbt.settings.returns.year_freq = pd.Timedelta(days=252)
    run_v2_benchmark()
