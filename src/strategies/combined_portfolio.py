"""Combined Multi-Horizon FX Portfolio.

Combines three orthogonal strategies into a single portfolio:
1. MR Macro (intraday minute, EUR-USD) — Sharpe 0.94
2. XS Momentum (daily, 4 pairs) — Sharpe 0.72
3. TS Momentum + RSI (daily, 4 pairs) — Sharpe 0.70

Correlation between MR and daily strategies: 0.04-0.05 (quasi-independent).
Risk parity allocation: MR ~78%, XS ~9%, TS ~14%.

Research findings (walk-forward 2019-2025):
  Risk Parity: Sharpe 0.67, 6/7 years positive
  Equal weight MR50/XS25/TS25: Sharpe 0.49, 6/7 years positive
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import vectorbtpro as vbt

from strategies.daily_momentum import (
    backtest_ts_momentum_portfolio,
    backtest_xs_momentum,
    load_daily_closes,
)
from strategies.mr_macro import backtest_mr_macro
from utils import apply_vbt_settings, load_fx_data


_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

WF_PERIODS = [
    (f"{y}-01-01", f"{y}-12-31") for y in range(2019, 2025)
] + [("2025-01-01", "2026-04-01")]


# ===================================================================
# PORTFOLIO CONSTRUCTION
# ===================================================================


def get_strategy_daily_returns() -> dict[str, pd.Series]:
    """Compute daily returns for each component strategy."""
    print("  Loading EUR-USD minute data...")
    _, data_eur = load_fx_data()

    print("  Running MR Macro backtest...")
    pf_mr = backtest_mr_macro(data_eur)
    mr_rets = pf_mr.daily_returns

    print("  Loading daily closes (4 pairs)...")
    closes = load_daily_closes()

    print("  Running XS Momentum...")
    xs_rets = backtest_xs_momentum(closes)

    print("  Running TS Momentum + RSI...")
    ts_rets = backtest_ts_momentum_portfolio(closes)

    return {
        "MR_Macro": mr_rets,
        "XS_Momentum": xs_rets,
        "TS_Momentum_RSI": ts_rets,
    }


def build_combined_portfolio(
    strategy_returns: dict[str, pd.Series],
    allocation: str = "risk_parity",
    custom_weights: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Build combined portfolio from strategy daily returns.

    Allocations:
      risk_parity: weight inversely proportional to volatility
      equal: 1/N equal weight
      mr_heavy: MR 50%, XS 25%, TS 25%
      custom: use custom_weights dict
    """
    # Align to common index
    all_rets = pd.DataFrame(strategy_returns)
    common = all_rets.dropna()

    # Compute weights
    if allocation == "risk_parity":
        vols = common.std()
        inv_vol = 1.0 / vols
        weights = inv_vol / inv_vol.sum()
    elif allocation == "equal":
        n = len(common.columns)
        weights = pd.Series(1.0 / n, index=common.columns)
    elif allocation == "mr_heavy":
        weights = pd.Series({
            "MR_Macro": 0.50,
            "XS_Momentum": 0.25,
            "TS_Momentum_RSI": 0.25,
        })
    elif allocation == "custom" and custom_weights:
        weights = pd.Series(custom_weights)
    else:
        weights = pd.Series(1.0 / len(common.columns), index=common.columns)

    # Normalize weights
    weights = weights.reindex(common.columns).fillna(0)
    weights = weights / weights.sum()

    # Combined returns
    port_rets = (common * weights).sum(axis=1)

    # Walk-forward analysis
    wf_sharpes = []
    for start, end in WF_PERIODS:
        p = port_rets.loc[start:end]
        if len(p) < 20:
            wf_sharpes.append(0.0)
            continue
        sr = p.mean() / p.std() * np.sqrt(252) if p.std() > 0 else 0.0
        wf_sharpes.append(float(sr) if not np.isnan(sr) else 0.0)

    # Full-period stats
    full_sr = (
        port_rets.mean() / port_rets.std() * np.sqrt(252)
        if port_rets.std() > 0 else 0.0
    )
    ann_ret = port_rets.mean() * 252
    ann_vol = port_rets.std() * np.sqrt(252)
    cum = (1 + port_rets).cumprod()
    max_dd = (cum / cum.cummax() - 1).min()

    # Correlation matrix
    corr = common.corr()

    return {
        "weights": weights.to_dict(),
        "allocation": allocation,
        "portfolio_returns": port_rets,
        "component_returns": common,
        "correlation": corr,
        "sharpe": float(full_sr),
        "annual_return": float(ann_ret),
        "annual_vol": float(ann_vol),
        "max_drawdown": float(max_dd),
        "wf_sharpes": wf_sharpes,
        "wf_avg_sharpe": float(np.mean(wf_sharpes)),
        "wf_pos_years": sum(1 for s in wf_sharpes if s > 0),
    }


# ===================================================================
# FULL ANALYSIS & REPORTING
# ===================================================================


def run_full_analysis(output_dir: str = "results/combined") -> None:
    """Run complete combined portfolio analysis with full reporting."""
    from framework.plotting import (
        generate_html_tearsheet,
        plot_drawdown_analysis,
        plot_monthly_heatmap,
        plot_multi_strategy_equity,
        plot_returns_distribution,
        plot_rolling_sharpe,
        show_browser,
    )

    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("  Combined Multi-Horizon FX Portfolio Analysis")
    print("=" * 60)

    # Get strategy returns
    strat_rets = get_strategy_daily_returns()

    # Test multiple allocations
    allocations = ["risk_parity", "equal", "mr_heavy"]
    results = {}

    for alloc in allocations:
        print(f"\n--- Allocation: {alloc} ---")
        res = build_combined_portfolio(strat_rets, allocation=alloc)
        results[alloc] = res

        w = res["weights"]
        print(f"  Weights: {', '.join(f'{k}={v:.1%}' for k, v in w.items())}")
        print(f"  Full-period Sharpe: {res['sharpe']:.2f}")
        print(f"  Annual Return: {res['annual_return']*100:.2f}%")
        print(f"  Annual Vol: {res['annual_vol']*100:.2f}%")
        print(f"  Max Drawdown: {res['max_drawdown']*100:.2f}%")

        wf_detail = " ".join(f"{s:>6.2f}" for s in res["wf_sharpes"])
        print(
            f"  Walk-forward: avg={res['wf_avg_sharpe']:.2f}"
            f" pos={res['wf_pos_years']}/7 | {wf_detail}"
        )

    # Correlation matrix
    best_alloc = max(results, key=lambda k: results[k]["wf_avg_sharpe"])
    best = results[best_alloc]

    print(f"\n--- Correlation Matrix ---")
    print(best["correlation"].round(3).to_string())

    print(f"\n--- Best allocation: {best_alloc} ---")
    print(f"  WF Sharpe: {best['wf_avg_sharpe']:.2f}, {best['wf_pos_years']}/7 positive")

    # Individual strategy walk-forward for comparison
    print(f"\n--- Individual Strategy Walk-Forward ---")
    for name, rets in strat_rets.items():
        sharpes = []
        for start, end in WF_PERIODS:
            p = rets.loc[start:end].dropna()
            if len(p) < 20:
                sharpes.append(0.0)
                continue
            sr = p.mean() / p.std() * np.sqrt(252) if p.std() > 0 else 0
            sharpes.append(float(sr) if not np.isnan(sr) else 0.0)
        detail = " ".join(f"{s:>6.2f}" for s in sharpes)
        avg = np.mean(sharpes)
        pos = sum(1 for s in sharpes if s > 0)
        print(f"  {name:<25} avg={avg:>5.2f} pos={pos}/7 | {detail}")

    # Generate portfolio from best allocation returns
    port_rets = best["portfolio_returns"]

    # Create a VBT portfolio from returns for native plotting
    pf_combined = vbt.Portfolio.from_returns(
        port_rets,
        init_cash=1_000_000,
        freq="1D",
    )

    # Also create individual strategy portfolios for comparison
    pf_components = {}
    for name, rets in strat_rets.items():
        pf_components[name] = vbt.Portfolio.from_returns(
            rets, init_cash=1_000_000, freq="1D",
        )
    pf_components[f"Combined ({best_alloc})"] = pf_combined

    # HTML tearsheet
    print("\nGenerating HTML tearsheet...")
    generate_html_tearsheet(
        pf_combined,
        os.path.join(output_dir, "combined_tearsheet.html"),
        f"Combined Portfolio ({best_alloc})",
    )

    # Plots
    print("Generating plots...")

    # Multi-strategy equity comparison
    fig_eq = plot_multi_strategy_equity(
        pf_components,
        "Strategy Comparison — Normalized Equity",
    )
    show_browser(fig_eq)

    # Portfolio summary plots
    fig_monthly = plot_monthly_heatmap(pf_combined, "Combined — Monthly Returns")
    show_browser(fig_monthly)

    fig_dist = plot_returns_distribution(pf_combined, "Combined — Returns Distribution")
    show_browser(fig_dist)

    fig_dd = plot_drawdown_analysis(pf_combined, "Combined — Drawdowns")
    show_browser(fig_dd)

    fig_rs = plot_rolling_sharpe(pf_combined, title="Combined — Rolling Sharpe")
    show_browser(fig_rs)

    print("\nDone. All reports saved to:", output_dir)


# ===================================================================
# CLI
# ===================================================================

if __name__ == "__main__":
    apply_vbt_settings()
    vbt.settings.returns.year_freq = pd.Timedelta(days=252)
    run_full_analysis()
