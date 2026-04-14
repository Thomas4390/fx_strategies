"""Generate all Phase 18 report artifacts into results/phase18/.

Produces:
  results/phase18/sleeves/<name>/<name>_*.html   (tearsheet per sleeve)
  results/phase18/combined/Phase18_Combined_*.html  (tearsheet combined)
  results/phase18/figures/equity_comparison.html     (log-scale equity)
  results/phase18/figures/rolling_correlation.html   (63d rolling corr)
  results/phase18/figures/bootstrap_scatter.html     (CAGR × MaxDD cloud)
  results/phase18/figures/per_sleeve_monthly.html    (stacked contribution)
  results/phase18/stress_test_report.json            (bootstrap / scenarios)
  results/phase18/summary.txt                        (IS/OOS metrics dump)

Run:
    python scripts/generate_phase18_report_artifacts.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SRC = _PROJECT_ROOT / "src"
_SCRIPTS = _PROJECT_ROOT / "scripts"
for p in (_SRC, _SCRIPTS):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


OUTPUT_ROOT = _PROJECT_ROOT / "results" / "production_report"
OOS_SPLIT_DATE = "2025-04-01"


# ═══════════════════════════════════════════════════════════════════════
# Sleeve tearsheets
# ═══════════════════════════════════════════════════════════════════════


def generate_sleeve_tearsheets(
    strat_rets: dict[str, pd.Series],
) -> None:
    """Build a vbt.Portfolio per sleeve and call analyze_portfolio on each."""
    from framework.pipeline_utils import analyze_portfolio
    from strategies.combined_portfolio import returns_to_pf
    from strategies.combined_portfolio_v2 import PRODUCTION_WEIGHTS

    sleeves_dir = OUTPUT_ROOT / "sleeves"
    sleeves_dir.mkdir(parents=True, exist_ok=True)

    for sleeve_name in PRODUCTION_WEIGHTS:
        rets = strat_rets[sleeve_name]
        pf = returns_to_pf(rets)
        out_dir = sleeves_dir / sleeve_name
        print(f"\n→ Tearsheet for sleeve: {sleeve_name}")
        analyze_portfolio(
            pf,
            name=sleeve_name,
            output_dir=str(out_dir),
            show_charts=False,
            save_excel=False,
            max_plot_points=20_000,
        )


def generate_combined_tearsheet(production_result: dict[str, Any]) -> None:
    from framework.pipeline_utils import analyze_portfolio

    combined_dir = OUTPUT_ROOT / "combined"
    combined_dir.mkdir(parents=True, exist_ok=True)

    pf = production_result["pf_combined"]
    print("\n→ Tearsheet for Phase 18 combined portfolio")
    analyze_portfolio(
        pf,
        name="Phase18_Combined",
        output_dir=str(combined_dir),
        show_charts=False,
        save_excel=False,
        max_plot_points=20_000,
    )


# ═══════════════════════════════════════════════════════════════════════
# Custom figures
# ═══════════════════════════════════════════════════════════════════════


def figure_equity_comparison(
    strat_rets: dict[str, pd.Series],
    production_result: dict[str, Any],
) -> go.Figure:
    """Log-scale equity base 100 for the 3 sleeves and the combined."""
    fig = go.Figure()

    colors = {
        "MR_Macro": "#1f77b4",
        "TS_Momentum_3p": "#ff7f0e",
        "RSI_Daily_4p": "#2ca02c",
        "Phase18 Combined": "#d62728",
    }

    for name, rets in strat_rets.items():
        cum = (1.0 + rets.fillna(0.0)).cumprod() * 100.0
        fig.add_trace(
            go.Scatter(
                x=cum.index,
                y=cum.values,
                mode="lines",
                name=name,
                line=dict(color=colors.get(name, "gray"), width=1.5),
            )
        )

    combined = production_result["portfolio_returns"]
    cum_combined = (1.0 + combined.fillna(0.0)).cumprod() * 100.0
    fig.add_trace(
        go.Scatter(
            x=cum_combined.index,
            y=cum_combined.values,
            mode="lines",
            name="Phase18 Combined",
            line=dict(color=colors["Phase18 Combined"], width=3),
        )
    )

    fig.update_layout(
        title="Phase 18 — Cumulative Equity (base 100, log scale)",
        xaxis_title="Date",
        yaxis_title="Equity (log)",
        yaxis_type="log",
        template="plotly_white",
        height=650,
    )
    return fig


def figure_rolling_correlation(
    strat_rets: dict[str, pd.Series],
    window: int = 63,
) -> go.Figure:
    """63-day rolling correlation between the 3 sleeve pairs."""
    df = pd.DataFrame(strat_rets).dropna()
    mr = df["MR_Macro"]
    ts = df["TS_Momentum_3p"]
    rsi = df["RSI_Daily_4p"]

    roll = pd.DataFrame(
        {
            "MR / TS_3p": mr.rolling(window).corr(ts),
            "MR / RSI_4p": mr.rolling(window).corr(rsi),
            "TS_3p / RSI_4p": ts.rolling(window).corr(rsi),
        }
    ).dropna()

    fig = go.Figure()
    for col in roll.columns:
        fig.add_trace(
            go.Scatter(x=roll.index, y=roll[col].values, mode="lines", name=col)
        )
    fig.add_hline(y=0.0, line_dash="dash", line_color="gray")
    fig.update_layout(
        title=f"Phase 18 — Rolling {window}d correlation between sleeves",
        xaxis_title="Date",
        yaxis_title="Correlation",
        yaxis=dict(range=[-1, 1]),
        template="plotly_white",
        height=500,
    )
    return fig


def figure_bootstrap_scatter(
    production_strat_rets: dict[str, pd.Series],
    production_result: dict[str, Any],
    n_runs: int = 500,
    block_size: int = 20,
) -> tuple[go.Figure, dict[str, float]]:
    """Bootstrap 500 paths, scatter CAGR × MaxDD with Phase 18 point overlay."""
    from stress_test_combined import block_bootstrap_indices
    from strategies.combined_portfolio_v2 import (
        PRODUCTION_MAX_LEVERAGE,
        PRODUCTION_TARGET_VOL,
        PRODUCTION_WEIGHTS,
        build_combined_portfolio_v2,
    )

    df = pd.DataFrame(production_strat_rets).dropna()
    n_bars = len(df)
    rng = np.random.default_rng(20260413)

    cagrs: list[float] = []
    dds: list[float] = []
    sharpes: list[float] = []

    for i in range(n_runs):
        idx = block_bootstrap_indices(n_bars, block_size, rng)
        resampled = df.iloc[idx].reset_index(drop=True)
        resampled.index = pd.bdate_range("2000-01-03", periods=n_bars)
        strat_resampled = {col: resampled[col] for col in df.columns}

        try:
            res = build_combined_portfolio_v2(
                strat_resampled,
                allocation="custom",
                custom_weights=PRODUCTION_WEIGHTS,
                target_vol=PRODUCTION_TARGET_VOL,
                max_leverage=PRODUCTION_MAX_LEVERAGE,
                dd_cap_enabled=False,
            )
        except Exception:
            continue

        cagrs.append(res["annual_return"])
        dds.append(res["max_drawdown"])
        sr = res["sharpe"]
        sharpes.append(0.0 if np.isnan(sr) else sr)

    cagr_arr = np.array(cagrs)
    dd_arr = np.array(dds)

    # Real Phase 18 point
    real_cagr = production_result["annual_return"]
    real_dd = production_result["max_drawdown"]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=dd_arr * 100,
            y=cagr_arr * 100,
            mode="markers",
            marker=dict(size=5, color="lightblue", opacity=0.6),
            name=f"Bootstrap ({len(cagrs)} runs)",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[real_dd * 100],
            y=[real_cagr * 100],
            mode="markers+text",
            marker=dict(size=20, color="red", symbol="star"),
            text=["Phase 18 real"],
            textposition="top center",
            name="Phase 18 in-sample",
        )
    )
    fig.add_vline(
        x=-35.0,
        line_dash="dash",
        line_color="red",
        annotation_text="−35% cap",
        annotation_position="top right",
    )
    fig.add_hrect(
        y0=10.0,
        y1=15.0,
        fillcolor="green",
        opacity=0.1,
        line_width=0,
        annotation_text="Target CAGR band",
        annotation_position="top left",
    )
    fig.update_layout(
        title=f"Phase 18 — Bootstrap {len(cagrs)} paths (block={block_size}d)",
        xaxis_title="Max Drawdown (%)",
        yaxis_title="CAGR (%)",
        template="plotly_white",
        height=650,
    )

    summary = {
        "n_runs": len(cagrs),
        "cagr_mean": float(cagr_arr.mean()),
        "cagr_p05": float(np.percentile(cagr_arr, 5)),
        "cagr_p50": float(np.percentile(cagr_arr, 50)),
        "cagr_p95": float(np.percentile(cagr_arr, 95)),
        "dd_p05": float(np.percentile(dd_arr, 5)),
        "dd_p50": float(np.percentile(dd_arr, 50)),
        "dd_p95": float(np.percentile(dd_arr, 95)),
        "pos_fraction": float((cagr_arr > 0).mean()),
    }
    return fig, summary


def figure_per_sleeve_monthly(production_result: dict[str, Any]) -> go.Figure:
    """Stacked monthly contribution of each sleeve to the combined."""
    common = production_result["component_returns"]
    weights_ts = production_result["weights_ts"]
    leverage_ts = production_result.get("leverage_ts")
    dd_scale_ts = production_result.get("dd_scale_ts")
    if leverage_ts is None:
        leverage_ts = pd.Series(1.0, index=common.index)
    if dd_scale_ts is None:
        dd_scale_ts = pd.Series(1.0, index=common.index)

    scale = leverage_ts * dd_scale_ts
    contrib = common.multiply(weights_ts, axis=0).multiply(scale, axis=0)
    monthly = contrib.resample("ME").apply(lambda x: (1 + x).prod() - 1)

    fig = go.Figure()
    for col in monthly.columns:
        fig.add_trace(
            go.Bar(
                x=monthly.index,
                y=monthly[col].values * 100,
                name=col,
            )
        )
    fig.update_layout(
        barmode="relative",
        title="Phase 18 — Monthly contribution per sleeve (leveraged)",
        xaxis_title="Month",
        yaxis_title="Contribution (%)",
        template="plotly_white",
        height=550,
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════
# IS / OOS summary + stress test
# ═══════════════════════════════════════════════════════════════════════


def dump_is_oos_summary(production_result: dict[str, Any]) -> dict[str, Any]:
    port = production_result["portfolio_returns"]
    split = pd.Timestamp(OOS_SPLIT_DATE)
    is_rets = port.loc[: split - pd.Timedelta(days=1)]
    oos_rets = port.loc[split:]

    def metrics(rets: pd.Series) -> dict[str, float]:
        if len(rets) < 20 or rets.std() == 0:
            return {"n": int(len(rets))}
        ann_ret = (1 + rets).prod() ** (252 / len(rets)) - 1
        ann_vol = rets.std() * np.sqrt(252)
        sharpe = rets.mean() / rets.std() * np.sqrt(252)
        cum = (1 + rets).cumprod()
        mdd = float((cum / cum.expanding().max() - 1).min())
        total = float((1 + rets).prod() - 1)
        return {
            "n": int(len(rets)),
            "cagr": float(ann_ret),
            "vol": float(ann_vol),
            "sharpe": float(sharpe),
            "max_dd": mdd,
            "total_return": total,
        }

    return {
        "split_date": OOS_SPLIT_DATE,
        "in_sample": metrics(is_rets),
        "out_of_sample": metrics(oos_rets),
        "wf_sharpes": production_result.get("wf_sharpes", []),
        "wf_pos_years": production_result.get("wf_pos_years", 0),
        "wf_avg_sharpe": production_result.get("wf_avg_sharpe", float("nan")),
    }


def run_stress_test(production_strat_rets: dict[str, pd.Series]) -> dict[str, Any]:
    """Re-run the full stress test suite on the Phase 18 config."""
    import stress_test_combined

    # Override the module-level RECOMMENDED_CONFIG so the helpers use
    # Phase 18. We restore the previous value on exit to keep the
    # script re-entrant.
    prev_config = stress_test_combined.RECOMMENDED_CONFIG
    try:
        from strategies.combined_portfolio_v2 import (
            PRODUCTION_MAX_LEVERAGE,
            PRODUCTION_TARGET_VOL,
            PRODUCTION_WEIGHTS,
        )

        stress_test_combined.RECOMMENDED_CONFIG = {
            "allocation": "custom",
            "custom_weights": dict(PRODUCTION_WEIGHTS),
            "target_vol": PRODUCTION_TARGET_VOL,
            "max_leverage": PRODUCTION_MAX_LEVERAGE,
            "dd_cap_enabled": False,
        }
        print("\n→ Running full stress test suite (this is slow)...")
        boot = stress_test_combined.run_block_bootstrap(
            production_strat_rets, n_runs=1000, block_size=20, seed=20260413
        )
        scenarios = stress_test_combined.run_scenario_replay(production_strat_rets)
        is_oos = stress_test_combined.run_is_oos_split(production_strat_rets)
        sensitivity = stress_test_combined.run_parameter_sensitivity(
            production_strat_rets
        )
    finally:
        stress_test_combined.RECOMMENDED_CONFIG = prev_config

    return {
        "config": {
            "allocation": "custom",
            "weights": dict(stress_test_combined.RECOMMENDED_CONFIG["custom_weights"])
            if False
            else {"MR_Macro": 0.80, "TS_Momentum_3p": 0.10, "RSI_Daily_4p": 0.10},
            "target_vol": 0.28,
            "max_leverage": 12.0,
            "dd_cap_enabled": False,
        },
        "bootstrap": boot.to_dict(),
        "scenarios": scenarios,
        "is_oos": is_oos,
        "sensitivity": sensitivity,
    }


# ═══════════════════════════════════════════════════════════════════════
# Orchestration
# ═══════════════════════════════════════════════════════════════════════


def main() -> None:
    from strategies.combined_portfolio import get_strategy_daily_returns
    from strategies.combined_portfolio_v2 import (
        PRODUCTION_WEIGHTS,
        build_production_portfolio,
    )
    from utils import apply_vbt_settings
    import vectorbtpro as vbt

    apply_vbt_settings()
    vbt.settings.returns.year_freq = pd.Timedelta(days=252)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    figures_dir = OUTPUT_ROOT / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  Phase 18 — Generating report artifacts")
    print("=" * 70)

    print("\n[1/6] Loading strategy daily returns...")
    strat_rets = get_strategy_daily_returns()
    production_strat_rets = {k: strat_rets[k] for k in PRODUCTION_WEIGHTS}

    print("\n[2/6] Building Phase 18 combined portfolio...")
    production_result = build_production_portfolio(production_strat_rets)

    print(
        f"  IS metrics  : CAGR={production_result['annual_return'] * 100:.2f}% "
        f"MaxDD={production_result['max_drawdown'] * 100:.2f}% "
        f"Sharpe={production_result['sharpe']:.3f} "
        f"Pos years={production_result['wf_pos_years']}/7"
    )

    print("\n[3/6] Generating per-sleeve tearsheets...")
    generate_sleeve_tearsheets(production_strat_rets)

    print("\n[4/6] Generating combined tearsheet...")
    generate_combined_tearsheet(production_result)

    print("\n[5/6] Generating custom figures...")
    fig_equity = figure_equity_comparison(production_strat_rets, production_result)
    fig_equity.write_html(str(figures_dir / "equity_comparison.html"))
    print("  ✓ equity_comparison.html")

    fig_corr = figure_rolling_correlation(production_strat_rets)
    fig_corr.write_html(str(figures_dir / "rolling_correlation.html"))
    print("  ✓ rolling_correlation.html")

    fig_scatter, scatter_summary = figure_bootstrap_scatter(
        production_strat_rets, production_result, n_runs=500
    )
    fig_scatter.write_html(str(figures_dir / "bootstrap_scatter.html"))
    print(
        f"  ✓ bootstrap_scatter.html "
        f"(P5 CAGR {scatter_summary['cagr_p05'] * 100:.2f}%, "
        f"P5 DD {scatter_summary['dd_p05'] * 100:.2f}%)"
    )

    fig_stack = figure_per_sleeve_monthly(production_result)
    fig_stack.write_html(str(figures_dir / "per_sleeve_monthly.html"))
    print("  ✓ per_sleeve_monthly.html")

    print("\n[6/6] Running full stress test suite...")
    stress_report = run_stress_test(production_strat_rets)
    stress_report["bootstrap_scatter_summary"] = scatter_summary
    stress_report["is_oos_summary"] = dump_is_oos_summary(production_result)

    json_path = OUTPUT_ROOT / "stress_test_report.json"
    with open(json_path, "w") as fh:
        json.dump(stress_report, fh, indent=2, default=str)
    print(f"  ✓ {json_path.relative_to(_PROJECT_ROOT)}")

    # Summary text dump for the report
    summary_path = OUTPUT_ROOT / "summary.txt"
    is_oos = stress_report["is_oos_summary"]
    boot = stress_report["bootstrap"]
    with open(summary_path, "w") as fh:
        fh.write("Phase 18 — Final Strategy Summary\n")
        fh.write("=" * 45 + "\n\n")
        fh.write("Config: MR80 / TS_Momentum_3p 10 / RSI_Daily_4p 10\n")
        fh.write("        target_vol=0.28, max_leverage=12, DDcap=OFF\n\n")
        fh.write("In-sample (2019 → 2025-04):\n")
        _is = is_oos["in_sample"]
        fh.write(
            f"  CAGR   : {_is.get('cagr', 0) * 100:.2f}%\n"
            f"  Vol    : {_is.get('vol', 0) * 100:.2f}%\n"
            f"  MaxDD  : {_is.get('max_dd', 0) * 100:.2f}%\n"
            f"  Sharpe : {_is.get('sharpe', 0):.3f}\n"
            f"  Bars   : {_is.get('n', 0)}\n\n"
        )
        fh.write("Out-of-sample (2025-04 → 2026-04):\n")
        _oos = is_oos["out_of_sample"]
        fh.write(
            f"  CAGR   : {_oos.get('cagr', 0) * 100:.2f}%\n"
            f"  Vol    : {_oos.get('vol', 0) * 100:.2f}%\n"
            f"  MaxDD  : {_oos.get('max_dd', 0) * 100:.2f}%\n"
            f"  Sharpe : {_oos.get('sharpe', 0):.3f}\n"
            f"  Bars   : {_oos.get('n', 0)}\n\n"
        )
        fh.write("Bootstrap 1000 runs, block=20d:\n")
        fh.write(
            f"  CAGR mean: {boot['cagr_mean'] * 100:.2f}%  "
            f"P5 {boot['cagr_p05'] * 100:.2f}%  "
            f"P50 {boot['cagr_p50'] * 100:.2f}%  "
            f"P95 {boot['cagr_p95'] * 100:.2f}%\n"
            f"  MaxDD mean: {boot['max_dd_mean'] * 100:.2f}%  "
            f"P5 {boot['max_dd_p05'] * 100:.2f}%  "
            f"P50 {boot['max_dd_p50'] * 100:.2f}%  "
            f"P95 {boot['max_dd_p95'] * 100:.2f}%\n"
            f"  Sharpe mean: {boot['sharpe_mean']:.3f}\n"
            f"  Positive CAGR fraction: {boot['pos_fraction'] * 100:.1f}%\n"
            f"  Target hit: {boot['target_hit_fraction'] * 100:.1f}%\n\n"
        )
        fh.write(
            f"WF per-year Sharpe: {is_oos.get('wf_sharpes', [])}\n"
            f"WF positive years: {is_oos.get('wf_pos_years', 0)}/7\n"
        )
    print(f"  ✓ {summary_path.relative_to(_PROJECT_ROOT)}")

    print("\n" + "=" * 70)
    print("  DONE — artifacts in results/phase18/")
    print("=" * 70)


if __name__ == "__main__":
    main()
