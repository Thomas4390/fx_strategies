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
from typing import Any

import numpy as np
import pandas as pd
import vectorbtpro as vbt

from strategies.combined_core import (
    build_native_combined,
    combined_returns_from_pf,
    sharpe_for_window,
)
from strategies.daily_momentum import (
    backtest_ts_momentum_portfolio,
    backtest_xs_momentum,
    load_daily_closes,
)
from strategies.mr_macro import backtest_mr_macro
from strategies.rsi_daily import pipeline as rsi_daily_pipeline
from utils import apply_vbt_settings, load_fx_data

_RSI_DAILY_PAIRS = ("EUR-USD", "GBP-USD", "USD-JPY", "USD-CAD")


def backtest_rsi_daily_portfolio(
    pairs: tuple[str, ...] = _RSI_DAILY_PAIRS,
    rsi_period: int = 14,
    oversold: float = 25.0,
    overbought: float = 75.0,
) -> pd.Series:
    """Equal-weight RSI Daily across ``pairs`` — daily returns Series.

    Phase 18: RSI Daily multi-pair is the orthogonal 3rd-sleeve of the
    combined portfolio. Per-year standalone it's positive in the very
    years the other sleeves lose (2019 Sharpe +0.95, 2023 +0.92,
    2026 YTD +3.54) while the full-period Sharpe stays low because
    trades are sparse — exactly the diversifier profile we want at a
    10% weight, not a 50% one.
    """
    per_pair: list[pd.Series] = []
    for pair in pairs:
        _, data = load_fx_data(f"data/{pair}_minute.parquet")
        pf, _ = rsi_daily_pipeline(
            data,
            rsi_period=rsi_period,
            oversold=oversold,
            overbought=overbought,
        )
        per_pair.append(pf.daily_returns)
    # skipna=True so the first pair's warmup doesn't drag the mean.
    return pd.concat(per_pair, axis=1).mean(axis=1, skipna=True)


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

    # Phase 17: TS Momentum restricted to 3 pairs (drop USD-CAD).
    # Per-year decomposition showed USD-CAD is consistently the worst
    # pair for the 20/50 EMA + RSI(7) signal: -8.94% in 2019, -7.32%
    # in 2022, -5.72% in 2023, -0.83% in 2026 YTD. Removing it lifts
    # the full-period Sharpe from 0.44 to 0.57 (+30%) and restores
    # 2023 to positive in the combined v2 walk-forward.
    closes_3p = closes[["EUR-USD", "GBP-USD", "USD-JPY"]]
    ts_rets_3p = backtest_ts_momentum_portfolio(closes_3p)

    # Phase 18: RSI Daily 4-pair as a third orthogonal sleeve.
    # Near-zero correlation with MR Macro (+0.056) and slightly
    # negative correlation with TS Momentum (-0.251) — positive in
    # 2019 and 2023 when the other two sleeves lose. At 10% weight
    # it boosts bootstrap P5 Max DD from -33.98% to -30.68% and OOS
    # Sharpe from 1.24 to 1.44.
    print("  Running RSI Daily 4-pair...")
    rsi_daily_4p = backtest_rsi_daily_portfolio()

    return {
        "MR_Macro": mr_rets,
        "XS_Momentum": xs_rets,
        "TS_Momentum_RSI": ts_rets,
        "TS_Momentum_3p": ts_rets_3p,
        "RSI_Daily_4p": rsi_daily_4p,
    }


_RISK_PARITY_WARMUP = 63  # ~3 months — warmup before expanding-window vol is trusted
_INIT_CASH = 1_000_000
_SYNTHETIC_BASE_PRICE = 1000.0  # kept for backwards-compat with returns_to_pf


def returns_to_pf(
    rets: pd.Series,
    init_cash: float = _INIT_CASH,
) -> vbt.Portfolio:
    """Wrap a daily returns Series in a single-column ``vbt.Portfolio``.

    Thin ``from_holding`` wrapper retained for walk-forward window analysis
    where we want a stand-alone VBT portfolio over a slice of returns. The
    main ``build_combined_portfolio`` path no longer uses this helper — it
    routes through ``strategies.combined_core.build_native_combined`` which
    uses ``Portfolio.from_optimizer`` with a ``PortfolioOptimizer`` built
    from filled allocations. See ``combined_core`` for the equivalence
    argument and the ``scripts/spikes/pfo_equivalence_spike.py`` numerical
    check (bit-identical at 1e-14 on 5 test cases).
    """
    rets_clean = rets.fillna(0.0)
    price = (1.0 + rets_clean).cumprod() * _SYNTHETIC_BASE_PRICE
    return vbt.Portfolio.from_holding(
        close=price, init_cash=init_cash, freq="1D"
    )


def _compute_weights_ts(
    common: pd.DataFrame,
    allocation: str,
    custom_weights: dict[str, float] | None,
) -> pd.DataFrame:
    """Return a time-varying weights DataFrame aligned to ``common``.

    Static allocations broadcast a single Series; ``risk_parity`` uses an
    expanding-window inverse-vol estimate with ``.shift(1)`` so weights at
    time ``t`` depend only on returns strictly before ``t`` (no look-ahead).
    Before ``_RISK_PARITY_WARMUP`` observations, equal weights are used.
    """
    n_cols = len(common.columns)
    eq_w = 1.0 / n_cols

    if allocation == "risk_parity":
        vol = common.vbt.expanding_std(minp=_RISK_PARITY_WARMUP).shift(1)
        inv_vol = vol.where(vol > 0).rdiv(1.0)  # 1 / vol, NaN where vol is 0
        row_sum = inv_vol.sum(axis=1)
        weights_ts = inv_vol.div(row_sum.where(row_sum > 0), axis=0)
        # Warmup rows (before min_periods) fall back to equal weights.
        return weights_ts.fillna(eq_w)

    if allocation == "equal":
        static = pd.Series(eq_w, index=common.columns)
    elif allocation == "mr_heavy":
        static = pd.Series(
            {"MR_Macro": 0.50, "XS_Momentum": 0.25, "TS_Momentum_RSI": 0.25}
        )
    elif allocation == "custom" and custom_weights:
        static = pd.Series(custom_weights)
    else:
        static = pd.Series(eq_w, index=common.columns)

    static = static.reindex(common.columns).fillna(0.0)
    total = static.sum()
    static = static / total if total > 0 else static
    return pd.DataFrame(
        np.broadcast_to(static.values, common.shape),
        index=common.index,
        columns=common.columns,
    )


def build_combined_portfolio(
    strategy_returns: dict[str, pd.Series],
    allocation: str = "risk_parity",
    custom_weights: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Build combined portfolio from strategy daily returns.

    Allocations:
      risk_parity: expanding-window inverse-vol, shifted by 1 bar (no look-ahead)
      equal: 1/N equal weight
      mr_heavy: MR 50%, XS 25%, TS 25%
      custom: use custom_weights dict

    The returned ``pf_combined`` is built via ``vbt.Portfolio.from_returns`` on
    the component-net-of-fees daily returns. It therefore reflects fees already
    paid by each sub-strategy but does NOT charge any rebalancing/turnover cost
    for the combined allocation.
    """
    # Align to common index
    all_rets = pd.DataFrame(strategy_returns)
    common = all_rets.dropna()

    weights_ts = _compute_weights_ts(common, allocation, custom_weights)

    # Native multi-strategy portfolio via PFO + from_optimizer. Equivalent
    # to the previous ``(common * weights_ts).sum(axis=1) → returns_to_pf``
    # pipeline at machine precision — see combined_core.build_native_combined
    # docstring and scripts/spikes/pfo_equivalence_spike.py.
    pf_combined, _, _ = build_native_combined(
        common,
        weights_ts,
        init_cash=_INIT_CASH,
    )
    port_rets = combined_returns_from_pf(pf_combined)

    # Walk-forward analysis — native per-window Sharpe via combined_core.
    wf_sharpes = [
        sharpe_for_window(pf_combined, start, end) for start, end in WF_PERIODS
    ]

    full_sr = float(pf_combined.sharpe_ratio)
    if np.isnan(full_sr):
        full_sr = 0.0

    return {
        "weights_ts": weights_ts,
        "weights": weights_ts.mean().to_dict(),  # average allocation across time
        "allocation": allocation,
        "portfolio_returns": port_rets,
        "component_returns": common,
        "correlation": common.corr(),
        "pf_combined": pf_combined,
        "sharpe": full_sr,
        "annual_return": float(pf_combined.annualized_return),
        "annual_vol": float(pf_combined.annualized_volatility),
        "max_drawdown": float(pf_combined.max_drawdown),
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
            # Individual strategy walk-forward — these are raw return
            # series not tied to a multi-strategy pf, so we still build a
            # single-column from_holding via returns_to_pf for a native
            # per-window Sharpe.
            sr = float(returns_to_pf(p).sharpe_ratio)
            sharpes.append(sr if not np.isnan(sr) else 0.0)
        detail = " ".join(f"{s:>6.2f}" for s in sharpes)
        avg = np.mean(sharpes)
        pos = sum(1 for s in sharpes if s > 0)
        print(f"  {name:<25} avg={avg:>5.2f} pos={pos}/7 | {detail}")

    # Reuse the VBT portfolio built inside build_combined_portfolio —
    # avoids a duplicate from_returns pass.
    pf_combined = best["pf_combined"]

    # Individual strategy portfolios for comparison
    pf_components = {}
    for name, rets in strat_rets.items():
        pf_components[name] = returns_to_pf(rets)
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
