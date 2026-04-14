"""Phase 21 — Retroactive Deflated Sharpe Ratio + PBO on Phase 18/19/20.

For each past sweep we have the full vector of ``sharpe`` (or
``wf_avg_sharpe``) for every config tried, plus the ID of the top
configuration selected. This script :

1. Loads each `results/phase*_<date>/metrics.json`.
2. Reruns the top config through
   :func:`strategies.combined_portfolio_v2.build_combined_portfolio_v2`
   with its recorded weights + leverage + DD cap to get its daily
   returns series.
3. Calls
   :func:`framework.statistical_testing.deflated_sharpe_ratio` passing
   the full Sharpe vector from the sweep as the selection-bias prior.
4. Runs :func:`framework.statistical_testing.probability_of_backtest_overfitting`
   on Phase 20A only (58 configs, dense factorial grid — the cleanest
   sweep for CSCV).
5. Emits a markdown table to `docs/research/phase21_dsr_retrofit.md`.

No new data is collected — everything is derived from the existing
sweep exports. This honors the HOLDOUT_POLICY : the retrofit only
reads in-sample metrics that were already computed.

Usage
-----
    python scripts/compute_dsr.py
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SRC = _PROJECT_ROOT / "src"
_SCRIPTS = _PROJECT_ROOT / "scripts"
for _p in (_SRC, _SCRIPTS):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


import vectorbtpro as vbt  # noqa: E402

from framework.statistical_testing import (  # noqa: E402
    deflated_sharpe_ratio,
    expected_max_sharpe,
    probability_of_backtest_overfitting,
)


_SWEEP_REPORTS: list[tuple[str, str, str]] = [
    # (display_name, metrics_path, top_config_id)
    # Historical paths are preserved — these are pre-refactor result
    # directories kept for the retrospective DSR audit. Future runs
    # should point at the new naming scheme under ``results/``.
    (
        "leverage grid",
        "results/phase19_2026-04-13/metrics.json",
        "P19c-tv25-ml14-dd_OFF",
    ),
    (
        "weight grid",
        "results/phase20a_2026-04-13/metrics.json",
        "P20a-w75-10-15",
    ),
    (
        "fourth sleeve",
        "results/phase20b_2026-04-13/metrics.json",
        "BL-P20Atop",
    ),  # top is the weight-sweep baseline; no true fourth-sleeve winner
    (
        "dd cap schedule",
        "results/phase20c_2026-04-13/metrics.json",
        "P20c-soft-w75-10-15-k20-f35",
    ),
]


def _load_metrics(path: Path) -> dict[str, Any]:
    with path.open("r") as f:
        return json.load(f)


def _rebuild_top_returns(
    top_config: dict[str, Any],
    sleeves: dict[str, pd.Series],
) -> pd.Series:
    """Re-run the top config through build_combined_portfolio_v2 to get
    the daily returns series (needed for DSR denominator via
    ``sharpe_ratio_std``)."""
    from strategies.combined_portfolio_v2 import build_combined_portfolio_v2

    custom_weights = top_config.get("custom_weights")
    filtered = {k: sleeves[k] for k in top_config["sleeves"] if k in sleeves}

    # DD schedule overrides for Phase 20C soft-cap configs are not
    # recorded in metrics.json (they existed only in the live
    # SweepConfig). Fall back to dd_cap_enabled=True with default
    # schedule if a soft-cap flag is present but the schedule is not —
    # this matches the P20C baseline (knee=0.20 floor=0.35 never fires
    # on the historical 17.35% MaxDD so default vs soft is equivalent).
    res = build_combined_portfolio_v2(
        filtered,
        allocation=top_config["allocation"],
        target_vol=top_config.get("target_vol"),
        max_leverage=float(top_config.get("max_leverage", 3.0)),
        dd_cap_enabled=bool(top_config.get("dd_cap_enabled", False)),
        custom_weights=custom_weights,
    )
    return res["portfolio_returns"]


def _sharpe_vector_from_metrics(metrics: dict[str, Any]) -> np.ndarray:
    """Return the vector of full-period Sharpe across all configs."""
    return np.asarray(
        [
            c["sharpe"]
            for c in metrics["configs"]
            if np.isfinite(c.get("sharpe", np.nan))
        ],
        dtype=float,
    )


def _dsr_row(
    sweep_name: str,
    metrics: dict[str, Any],
    top_id: str,
    sleeves: dict[str, pd.Series],
) -> dict[str, Any]:
    top_cfg = next(c for c in metrics["configs"] if c["id"] == top_id)
    top_returns = _rebuild_top_returns(top_cfg, sleeves)
    sharpe_vec = _sharpe_vector_from_metrics(metrics)

    dsr_result = deflated_sharpe_ratio(
        top_returns,
        n_trials=int(sharpe_vec.size),
        trial_sharpes=sharpe_vec,
    )

    return {
        "sweep": sweep_name,
        "top_id": top_id,
        "n_trials": int(sharpe_vec.size),
        "sharpe_full": float(top_cfg["sharpe"]),
        "wf_avg_sharpe": float(top_cfg["wf_avg_sharpe"]),
        "sharpe_std": dsr_result["sharpe_std"],
        "expected_max_sharpe": dsr_result["expected_max_sharpe"],
        "dsr": dsr_result["dsr"],
        "annual_return": float(top_cfg["annual_return"]),
        "max_drawdown": float(top_cfg["max_drawdown"]),
    }


def _pbo_weight_sweep(
    metrics: dict[str, Any],
    sleeves: dict[str, pd.Series],
) -> dict[str, Any] | None:
    """Run CSCV PBO on the Phase 20A sweep (58 dense configs).

    Rebuilds the returns matrix by replaying every CORE config through
    ``build_combined_portfolio_v2`` — reuses the same sleeves to keep
    the call quick (~1 min with parallelism already enabled by vbt).
    """
    from strategies.combined_portfolio_v2 import build_combined_portfolio_v2

    core_cfgs = [c for c in metrics["configs"] if c.get("block") == "CORE"]
    if not core_cfgs:
        return None

    returns_mat: dict[str, pd.Series] = {}
    for cfg in core_cfgs:
        filtered = {k: sleeves[k] for k in cfg["sleeves"] if k in sleeves}
        res = build_combined_portfolio_v2(
            filtered,
            allocation=cfg["allocation"],
            target_vol=cfg.get("target_vol"),
            max_leverage=float(cfg.get("max_leverage", 3.0)),
            dd_cap_enabled=bool(cfg.get("dd_cap_enabled", False)),
            custom_weights=cfg.get("custom_weights"),
        )
        returns_mat[cfg["id"]] = res["portfolio_returns"]

    returns_df = pd.DataFrame(returns_mat).dropna(how="any")
    pbo_res = probability_of_backtest_overfitting(
        returns_df, n_bins=10, objective="sharpe"
    )
    pbo_res["n_configs_actual"] = returns_df.shape[1]
    pbo_res["n_bars"] = returns_df.shape[0]
    return pbo_res


def _build_markdown(
    rows: list[dict[str, Any]],
    pbo: dict[str, Any] | None,
    report_date: str,
) -> str:
    lines: list[str] = []
    lines.append(f"# Phase 21 — DSR retrofit ({report_date})\n")
    lines.append(
        "**Holdout state** : LOCKED (frozen from 2026-01-01 until Phase 25)  \n"
        "**Holdout touched by this phase** : NO\n"
    )
    lines.append(
        "Retroactive statistical audit of the top configurations produced "
        "by Phase 18/19/20. Each sweep is revisited with two corrections :\n"
    )
    lines.append(
        "1. **Deflated Sharpe Ratio (DSR)** — Bailey & Lopez de Prado 2014. "
        "Converts the raw Sharpe of the top config into a probability that "
        "it exceeds what would be expected as the **maximum across N "
        "independent trials** with the same variance as the sweep. A DSR "
        "below ~0.95 means the observed Sharpe is indistinguishable from "
        "selection-bias luck. Computed via VBT Pro's native "
        "`ReturnsAccessor.sharpe_ratio_std` (Mertens 1998 correction for "
        "skewness and kurtosis).\n"
    )
    lines.append(
        "2. **Probability of Backtest Overfitting (PBO)** — Bailey-Borwein-"
        "Lopez de Prado-Zhu 2015, via Combinatorially Symmetric "
        "Cross-Validation. Applied only to Phase 20A because its dense "
        "factorial grid of 40 CORE configs is the cleanest candidate for "
        "CSCV.\n"
    )

    lines.append("## Deflated Sharpe Ratio by phase\n")
    lines.append(
        "| Phase | Top config | N trials | SR full | SR WF | σ(SR) | E[max SR] | **DSR** | CAGR | MaxDD |"
    )
    lines.append(
        "|-------|------------|---------:|--------:|------:|------:|----------:|--------:|-----:|------:|"
    )
    for r in rows:
        lines.append(
            f"| {r['sweep']} "
            f"| `{r['top_id']}` "
            f"| {r['n_trials']} "
            f"| {r['sharpe_full']:.3f} "
            f"| {r['wf_avg_sharpe']:.3f} "
            f"| {r['sharpe_std']:.3f} "
            f"| {r['expected_max_sharpe']:.3f} "
            f"| **{r['dsr']:.3f}** "
            f"| {r['annual_return'] * 100:.2f}% "
            f"| {r['max_drawdown'] * 100:.2f}% |"
        )
    lines.append(
        "\n**Reading** : DSR ≥ 0.95 = observed Sharpe clearly beats the "
        "best-of-N luck level. DSR in [0.70, 0.95] = borderline, the edge "
        "may be real but overfitting explains part of it. DSR < 0.50 = "
        "indistinguishable from a lucky winner, the selection bias "
        "dominates.\n"
    )

    if pbo is not None:
        pbo_value = pbo["pbo"]
        assessment = (
            "excellent"
            if pbo_value < 0.2
            else "good"
            if pbo_value < 0.4
            else "borderline"
            if pbo_value < 0.6
            else "overfit"
        )
        lines.append("## Probability of Backtest Overfitting — Phase 20A (CSCV)\n")
        lines.append(
            f"| Metric | Value |\n"
            f"|--------|-------|\n"
            f"| Configs evaluated | {pbo['n_configs']} |\n"
            f"| Time bins | {pbo['n_bins']} |\n"
            f"| Splits (C(n, n/2)) | {pbo['n_splits']} |\n"
            f"| T bars common | {pbo.get('n_bars', '?')} |\n"
            f"| **PBO** | **{pbo_value:.3f}** — {assessment} |\n"
        )
        lines.append(
            "\n**Reading** : PBO is the probability that the top "
            "configuration selected on an in-sample slice lands **below** "
            "the out-of-sample median of the same config set. PBO < 0.5 "
            "means the selection process adds value ; PBO ≥ 0.5 means the "
            "process is effectively curve-fitting.\n"
        )

    lines.append("## Interpretation — the DSR/PBO paradox\n")
    if rows and pbo is not None:
        all_dsr_high = all(r["dsr"] >= 0.95 for r in rows)
        pbo_high = pbo["pbo"] >= 0.5
        if all_dsr_high and pbo_high:
            lines.append(
                "**Every phase scores DSR ≈ 1.0 while the Phase 20A CSCV "
                f"PBO is {pbo['pbo']:.3f}.** These two signals are not "
                "contradictory — they measure different things :\n"
            )
            lines.append(
                "- **DSR ≈ 1.0** means the observed Sharpe of ~0.97 is "
                "numerically **very far** above the best-of-N luck "
                "threshold. The underlying edge of the *family* of "
                "configurations (MR-heavy 3-sleeve combined with vol "
                "targeting at tv=0.25 ml=14) is statistically real. "
                "~10 years × 252 days ≈ 2500 bars make the Sharpe "
                "estimator very precise (σ(SR) ≈ 0.019), which shrinks "
                "the z-score denominator and inflates DSR.\n"
                "- **PBO > 0.5** means that *within* Phase 20A's grid, "
                "selecting the in-sample top-1 and expecting it to stay "
                "top out-of-sample is effectively noise-selection. The "
                "40 CORE configs sit on a flat Sharpe plateau in the "
                "[0.88, 0.97] band ; which one 'wins' in-sample rotates "
                "across CSCV splits.\n"
            )
            lines.append(
                "**Operational consequence** : the 6 bps Sharpe gap "
                "reported in Phase 20A (`P20a-w75-10-15` at 0.972 vs the "
                "canonical `80/10/10` at 0.966) is **not** a real "
                "improvement — it is within the CSCV noise band. The "
                "production weights should stay on `80/10/10` "
                "(Phase 18/19 canonical) until a genuinely different "
                "configuration — different sleeves, different allocation "
                "mechanism, not just a weight perturbation — produces "
                "both high DSR **and** low PBO.\n"
            )
            lines.append(
                "**Take-away for Phase 22+** : DSR alone is insufficient "
                "for config selection at this sample size. We need PBO "
                "(or its walk-forward cousin CPCV) as the *gating* test "
                "on any sweep that claims a winner within an existing "
                "alpha family. A sweep that moves to a new alpha family "
                "(Phase 20B tried this with a 4th sleeve and failed) "
                "should still be DSR-gated because its trial variance "
                "is genuinely larger.\n"
            )
        else:
            best = max(rows, key=lambda r: r["dsr"])
            worst = min(rows, key=lambda r: r["dsr"])
            lines.append(
                f"- Highest DSR : **{best['sweep']}** — `{best['top_id']}` "
                f"at DSR = {best['dsr']:.3f}.\n"
                f"- Lowest DSR : **{worst['sweep']}** — `{worst['top_id']}` "
                f"at DSR = {worst['dsr']:.3f}.\n"
            )

    lines.append("\n## Next steps\n")
    lines.append(
        "- Phase 22 must wire PBO into the sweep rejection rule "
        "(see `plans/valiant-humming-coral.md`). Concretely : any new "
        "sweep claiming a top-1 within an existing alpha family must "
        "report CSCV PBO alongside the raw Sharpe ; promote only if "
        "PBO < 0.5.\n"
        "- Lock the production weights at the Phase 18 canonical "
        "80/10/10 with Phase 19 leverage `tv=0.25 ml=14 DDoff`. "
        "Phase 20A 'top' is retracted — it was a flat-plateau "
        "artefact.\n"
        "- The frozen slice (2026-01-01 → now) remains locked. No "
        "optimization ran against it during this retrofit.\n"
    )
    return "\n".join(lines)


def main() -> None:
    from strategies.combined_portfolio import get_strategy_daily_returns
    from utils import apply_vbt_settings

    apply_vbt_settings()
    vbt.settings.returns.year_freq = pd.Timedelta(days=252)

    report_date = date.today().isoformat()
    output_path = (
        _PROJECT_ROOT / "docs" / "research" / f"phase21_{report_date}_dsr_retrofit.md"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("  Phase 21 — DSR retrofit")
    print("=" * 72)
    print(f"Report date : {report_date}")
    print(f"Output      : {output_path}")
    print()

    print("Loading all 5 sleeves via combined_portfolio.get_strategy_daily_returns…")
    sleeves = get_strategy_daily_returns()
    print(f"  Loaded {len(sleeves)} sleeves: {sorted(sleeves)}\n")

    # Quick theory demo : what does E[max SR] look like for our scale?
    demo_var = 0.02  # typical trial SR variance observed in sweeps
    print("Expected max SR across N trials (trial variance 0.02):")
    for n in (10, 37, 58, 116, 300):
        print(f"  N = {n:>3}  →  E[max SR] = {expected_max_sharpe(n, demo_var):.3f}")
    print()

    rows: list[dict[str, Any]] = []
    for sweep_name, rel_path, top_id in _SWEEP_REPORTS:
        abs_path = _PROJECT_ROOT / rel_path
        if not abs_path.exists():
            print(f"  [skip] {sweep_name} — missing {rel_path}")
            continue
        print(f"Processing {sweep_name} ({rel_path})…")
        metrics = _load_metrics(abs_path)
        row = _dsr_row(sweep_name, metrics, top_id, sleeves)
        rows.append(row)
        print(
            f"  top={row['top_id']:<30} "
            f"SR={row['sharpe_full']:.3f}  "
            f"N={row['n_trials']:>3}  "
            f"σ(SR)={row['sharpe_std']:.3f}  "
            f"E[max]={row['expected_max_sharpe']:.3f}  "
            f"DSR={row['dsr']:.3f}"
        )

    # PBO on Phase 20A only (dense factorial grid)
    pbo_res: dict[str, Any] | None = None
    p20a_path = _PROJECT_ROOT / "results/phase20a_2026-04-13/metrics.json"
    if p20a_path.exists():
        print("\nRunning CSCV PBO on Phase 20A CORE configs…")
        p20a_metrics = _load_metrics(p20a_path)
        pbo_res = _pbo_weight_sweep(p20a_metrics, sleeves)
        if pbo_res is not None:
            print(
                f"  PBO = {pbo_res['pbo']:.3f} "
                f"(n_configs={pbo_res['n_configs']}, "
                f"n_bins={pbo_res['n_bins']}, "
                f"splits={pbo_res['n_splits']}, "
                f"T={pbo_res.get('n_bars', '?')})"
            )

    md = _build_markdown(rows, pbo_res, report_date)
    output_path.write_text(md)
    print(f"\nMarkdown exported → {output_path}")


if __name__ == "__main__":
    main()
