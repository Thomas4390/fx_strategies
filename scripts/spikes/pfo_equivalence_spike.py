"""Numerical equivalence spike: pandas-aggregation vs vbt PortfolioOptimizer.

Goal
----
Measure the numerical gap between the current combined-portfolio pattern

    port_ret_t = sum_i( w_i_t * ret_i_t )

and the vbt-native ``Portfolio.from_optimizer`` / ``from_filled_allocations``
pattern that rebalances daily with ``targetpercent`` sizing and zero fees.

If the gap is negligible (< 1e-6 on annualized Sharpe for a smoke test), we
can proceed with the full refonte. If not, we need to either relax the
equivalence tolerance or find a semantic mapping that matches exactly.

Run
---
    .venv/bin/python scripts/spikes/pfo_equivalence_spike.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import vectorbtpro as vbt


def synthetic_strategy_returns(
    seed: int, n: int = 1500, scale: float = 0.008
) -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    return pd.Series(rng.normal(0.0002, scale, size=n), index=idx)


def returns_to_price(rets: pd.Series, base: float = 1000.0) -> pd.Series:
    return (1.0 + rets.fillna(0.0)).cumprod() * base


def legacy_aggregate(
    strat_rets: dict[str, pd.Series],
    weights: dict[str, float] | pd.DataFrame,
) -> pd.Series:
    """Current pattern — instantaneous weighted sum of returns."""
    df = pd.DataFrame(strat_rets).dropna()
    if isinstance(weights, pd.DataFrame):
        w = weights.reindex(index=df.index, columns=df.columns).fillna(0.0)
    else:
        w = pd.Series(weights).reindex(df.columns).fillna(0.0)
    return (df * w).sum(axis=1)


def native_from_optimizer(
    strat_rets: dict[str, pd.Series],
    weights: dict[str, float] | pd.DataFrame,
    init_cash: float = 1_000_000.0,
    shift_allocations: bool = True,
) -> vbt.Portfolio:
    """PFO-based pattern — synthetic prices + filled allocations + from_optimizer.

    With ``shift_allocations=True`` the allocations are shifted forward by 1
    bar so that the target set at bar t-1 drives the position held during
    bar t, matching the legacy ``(weights * returns).sum()`` semantics where
    ``weights`` was already causal (shift(1) applied upstream).
    """
    rets_df = pd.DataFrame(strat_rets).dropna()
    prices = pd.DataFrame(
        {c: returns_to_price(rets_df[c]) for c in rets_df.columns},
        index=rets_df.index,
    )

    if isinstance(weights, pd.DataFrame):
        allocations = weights.reindex(index=prices.index, columns=prices.columns).fillna(0.0)
    else:
        w_series = pd.Series(weights).reindex(prices.columns).fillna(0.0)
        allocations = pd.DataFrame(
            np.broadcast_to(w_series.values, prices.shape),
            index=prices.index,
            columns=prices.columns,
        )

    if shift_allocations:
        allocations = allocations.shift(-1).fillna(0.0)

    pfo = vbt.PortfolioOptimizer.from_filled_allocations(
        allocations,
        valid_only=False,
        nonzero_only=False,
        unique_only=False,
    )
    pf = vbt.Portfolio.from_optimizer(
        prices,
        pfo,
        pf_method="from_orders",
        size_type="targetpercent",
        cash_sharing=True,
        group_by=True,
        init_cash=init_cash,
        fees=0.0,
        slippage=0.0,
        leverage=10.0,
        leverage_mode="eager",
        freq="1D",
    )
    return pf


def compare_metrics(
    legacy_rets: pd.Series, native_pf: vbt.Portfolio, label: str
) -> dict[str, float]:
    # Legacy Sharpe via synthetic portfolio to match the existing code path.
    legacy_price = returns_to_price(legacy_rets)
    legacy_pf = vbt.Portfolio.from_holding(
        close=legacy_price, init_cash=1_000_000.0, freq="1D"
    )

    legacy_sr = float(legacy_pf.sharpe_ratio)
    native_sr = float(native_pf.sharpe_ratio)
    legacy_tr = float(legacy_pf.total_return)
    native_tr = float(native_pf.total_return)
    legacy_mdd = float(legacy_pf.max_drawdown)
    native_mdd = float(native_pf.max_drawdown)

    # Return-series comparison.
    legacy_rs = legacy_pf.returns.fillna(0.0)
    native_rs = native_pf.returns.fillna(0.0)
    common_idx = legacy_rs.index.intersection(native_rs.index)
    legacy_rs = legacy_rs.loc[common_idx]
    native_rs = native_rs.loc[common_idx]
    max_abs = float((legacy_rs - native_rs).abs().max())

    print(f"\n=== {label} ===")
    print(f"  Sharpe      legacy={legacy_sr:+.8f}  native={native_sr:+.8f}  "
          f"Δ={native_sr - legacy_sr:+.2e}")
    print(f"  TotalReturn legacy={legacy_tr:+.8f}  native={native_tr:+.8f}  "
          f"Δ={native_tr - legacy_tr:+.2e}")
    print(f"  MaxDD       legacy={legacy_mdd:+.8f}  native={native_mdd:+.8f}  "
          f"Δ={native_mdd - legacy_mdd:+.2e}")
    print(f"  max|Δ returns_t|={max_abs:.2e}")

    return {
        "sharpe_delta": native_sr - legacy_sr,
        "total_return_delta": native_tr - legacy_tr,
        "max_dd_delta": native_mdd - legacy_mdd,
        "max_abs_return_diff": max_abs,
    }


def main() -> None:
    print("Spike: pandas aggregation vs vbt Portfolio.from_optimizer")
    print("=" * 64)

    # 3 uncorrelated synthetic strategies.
    strat_rets = {
        "A": synthetic_strategy_returns(seed=1, scale=0.006),
        "B": synthetic_strategy_returns(seed=2, scale=0.008),
        "C": synthetic_strategy_returns(seed=3, scale=0.010),
    }

    # Case 1 — equal weight.
    weights_eq = {"A": 1 / 3, "B": 1 / 3, "C": 1 / 3}
    legacy_rets = legacy_aggregate(strat_rets, weights_eq)
    native_pf = native_from_optimizer(strat_rets, weights_eq)
    compare_metrics(legacy_rets, native_pf, "Equal weight (1/3 each)")

    # Case 2 — skewed weights.
    weights_sk = {"A": 0.70, "B": 0.20, "C": 0.10}
    legacy_rets = legacy_aggregate(strat_rets, weights_sk)
    native_pf = native_from_optimizer(strat_rets, weights_sk)
    compare_metrics(legacy_rets, native_pf, "Skewed 70/20/10")

    # Case 3 — single strategy.
    weights_s = {"A": 1.0, "B": 0.0, "C": 0.0}
    legacy_rets = legacy_aggregate(strat_rets, weights_s)
    native_pf = native_from_optimizer(strat_rets, weights_s)
    compare_metrics(legacy_rets, native_pf, "Single strategy (A=1.0)")

    # Case 4 — time-varying weights (simulates risk-parity/regime output).
    rets_df = pd.DataFrame(strat_rets).dropna()
    rng = np.random.default_rng(42)
    raw = rng.uniform(0.1, 0.9, size=rets_df.shape)
    dyn_weights = pd.DataFrame(
        raw / raw.sum(axis=1, keepdims=True),
        index=rets_df.index,
        columns=rets_df.columns,
    )
    legacy_rets = legacy_aggregate(strat_rets, dyn_weights)
    native_pf = native_from_optimizer(strat_rets, dyn_weights)
    compare_metrics(legacy_rets, native_pf, "Time-varying weights")

    # Case 5 — leveraged allocations (sum > 1, simulates vol targeting).
    dyn_weights_lev = dyn_weights * 2.5
    legacy_rets = legacy_aggregate(strat_rets, dyn_weights_lev)
    native_pf = native_from_optimizer(strat_rets, dyn_weights_lev)
    compare_metrics(legacy_rets, native_pf, "Leveraged (sum=2.5)")


if __name__ == "__main__":
    main()
