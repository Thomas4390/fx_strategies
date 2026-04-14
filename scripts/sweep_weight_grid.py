"""Phase 20A — Dense weight sweep around the Phase 18/19 trio.

Follow-up to ``scripts/sweep_phase19.py``. Phase 19 pinned the weights to
``MR80 / TS3p10 / RSI10`` and only swept the leverage layer; it reached a
Sharpe plateau at 0.966 across any ``(target_vol >= 0.22, max_leverage
>= 14, DDoff)`` point. Phase 20A asks the complementary question: does a
*different* weight mix break out of that plateau when fed through the
same leverage layer?

Grid (~60 configs)
------------------
**Core weight grid** (tv=0.25 ml=14 DDoff — the Phase 19 plateau sweet
spot):
  - ``mr_weight`` ∈ {0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90} (8)
  - ``rsi_fraction`` ∈ {0.25, 0.40, 0.50, 0.60, 0.75} — the share of the
    diversifier budget (= 1 - mr) routed to ``RSI_Daily_4p`` vs
    ``TS_Momentum_3p`` (5)
  - Total: 8 × 5 = **40 configs**

**Leverage robustness cross-check** (best-5 weight mixes replayed on the
two adjacent plateau points ``tv=0.22 ml=14`` and ``tv=0.28 ml=14``):
  - 5 weight mixes × 2 (tv, ml) pairs × 1 DDoff = **10 configs**

**Light-MR contrarian** (drop MR below 0.55 — stress-test the "MR alpha
is the only alpha" assumption from Phase 13):
  - ``mr_weight`` ∈ {0.40, 0.50}, ``rsi_fraction`` ∈ {0.40, 0.50, 0.60}
  - (tv=0.25, ml=14, DDoff)
  - Total: **6 configs**

**Baselines** (exact P18 prod + the Phase 19 plateau top point):
  - ``BL-P18prod`` — MR80/TS10/RSI10 tv=0.28 ml=12 DDoff
  - ``BL-P19plateau`` — MR80/TS10/RSI10 tv=0.25 ml=14 DDoff
  - Total: **2 configs**

Grand total: 40 + 10 + 6 + 2 = **58 configs**

Usage
-----
    python scripts/sweep_phase20a_weights.py             # full run
    python scripts/sweep_phase20a_weights.py --smoke     # 4 configs
    python scripts/sweep_phase20a_weights.py --no-bootstrap
"""

from __future__ import annotations

import argparse
import json
import sys
import time
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


from sweep_combinations import (  # noqa: E402
    SweepConfig,
    _target_hit_mark,
    bootstrap_config,
    generate_top_n_artifacts,
    native_parallel_sweep,
    rows_from_native_metrics,
    sanitize_result_for_json,
)


_P18_SLEEVES: tuple[str, ...] = ("MR_Macro", "TS_Momentum_3p", "RSI_Daily_4p")
_P18_WEIGHTS: dict[str, float] = {
    "MR_Macro": 0.80,
    "TS_Momentum_3p": 0.10,
    "RSI_Daily_4p": 0.10,
}

# Phase 19 plateau sweet-spot: tv=0.25, ml=14, DDoff.
_PLATEAU_TV: float = 0.25
_PLATEAU_ML: float = 14.0
_PLATEAU_DD: bool = False


# ═══════════════════════════════════════════════════════════════════════
# Grid builder
# ═══════════════════════════════════════════════════════════════════════


def _make_weights(mr: float, rsi_fraction: float) -> dict[str, float]:
    """Split the 1-MR diversifier budget between TS and RSI.

    ``rsi_fraction`` is the share routed to RSI (so TS gets 1-rsi_fraction
    of the diversifier budget). Returns a normalized weights dict that
    always sums to 1.0.
    """
    div_budget = 1.0 - mr
    rsi_w = div_budget * rsi_fraction
    ts_w = div_budget * (1.0 - rsi_fraction)
    return {
        "MR_Macro": round(mr, 4),
        "TS_Momentum_3p": round(ts_w, 4),
        "RSI_Daily_4p": round(rsi_w, 4),
    }


def _weight_tag(mr: float, rsi_fraction: float) -> str:
    w = _make_weights(mr, rsi_fraction)
    mr_pct = int(round(w["MR_Macro"] * 100))
    ts_pct = int(round(w["TS_Momentum_3p"] * 100))
    rsi_pct = int(round(w["RSI_Daily_4p"] * 100))
    return f"{mr_pct:02d}-{ts_pct:02d}-{rsi_pct:02d}"


def build_phase20a_grid() -> list[SweepConfig]:
    """Return the full Phase 20A sweep grid (~58 configurations)."""
    cfgs: list[SweepConfig] = []

    # ─── Block CORE — tv=0.25 ml=14 DDoff, dense weight grid (40) ──
    MR_WEIGHTS = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
    RSI_FRACTIONS = [0.25, 0.40, 0.50, 0.60, 0.75]
    for mr in MR_WEIGHTS:
        for rsi_f in RSI_FRACTIONS:
            w = _make_weights(mr, rsi_f)
            tag = _weight_tag(mr, rsi_f)
            cfgs.append(
                SweepConfig(
                    id=f"P20a-w{tag}",
                    block="CORE",
                    name=f"weights={tag} / tv=0.25 ml=14 DDoff",
                    sleeves=_P18_SLEEVES,
                    allocation="custom",
                    target_vol=_PLATEAU_TV,
                    max_leverage=_PLATEAU_ML,
                    dd_cap_enabled=_PLATEAU_DD,
                    custom_weights=w,
                )
            )

    # ─── Block ROBUST — top-5 weight mixes × 2 adjacent plateau pts ─
    # We don't know the top-5 yet at build time, so we replay the 5
    # weight mixes that theory says are most likely to win (MR heavy
    # but not extreme, reasonable RSI share) across 2 alternate
    # (tv, ml) points. This gives 10 robustness checks for free.
    ROBUST_MIXES = [
        (0.80, 0.50),  # 80/10/10 (P18 canonical)
        (0.75, 0.50),  # 75/12.5/12.5
        (0.70, 0.50),  # 70/15/15
        (0.80, 0.40),  # 80/12/8 (slightly TS-heavy)
        (0.70, 0.60),  # 70/12/18 (slightly RSI-heavy)
    ]
    ROBUST_PLATEAU_POINTS = [
        (0.22, 14.0),
        (0.28, 14.0),
    ]
    for mr, rsi_f in ROBUST_MIXES:
        for tv, ml in ROBUST_PLATEAU_POINTS:
            w = _make_weights(mr, rsi_f)
            tag = _weight_tag(mr, rsi_f)
            tv_t = f"{int(tv * 100):02d}"
            cfgs.append(
                SweepConfig(
                    id=f"P20a-r{tag}-tv{tv_t}",
                    block="ROBUST",
                    name=f"weights={tag} / tv={tv:.2f} ml={int(ml)} DDoff",
                    sleeves=_P18_SLEEVES,
                    allocation="custom",
                    target_vol=tv,
                    max_leverage=ml,
                    dd_cap_enabled=False,
                    custom_weights=w,
                )
            )

    # ─── Block LIGHT_MR — contrarian MR < 0.55 (6) ──────────────────
    LIGHT_MRS = [0.40, 0.50]
    LIGHT_RSI_FRACS = [0.40, 0.50, 0.60]
    for mr in LIGHT_MRS:
        for rsi_f in LIGHT_RSI_FRACS:
            w = _make_weights(mr, rsi_f)
            tag = _weight_tag(mr, rsi_f)
            cfgs.append(
                SweepConfig(
                    id=f"P20a-l{tag}",
                    block="LIGHT_MR",
                    name=f"weights={tag} (light-MR) / tv=0.25 ml=14 DDoff",
                    sleeves=_P18_SLEEVES,
                    allocation="custom",
                    target_vol=_PLATEAU_TV,
                    max_leverage=_PLATEAU_ML,
                    dd_cap_enabled=_PLATEAU_DD,
                    custom_weights=w,
                )
            )

    # ─── Block BASELINE — reference configs (2) ─────────────────────
    cfgs += [
        SweepConfig(
            id="BL-P18prod",
            block="BASELINE",
            name="Phase 18 prod (MR80/TS10/RSI10 tv=0.28 ml=12 DDoff)",
            sleeves=_P18_SLEEVES,
            allocation="custom",
            target_vol=0.28,
            max_leverage=12.0,
            dd_cap_enabled=False,
            custom_weights=dict(_P18_WEIGHTS),
        ),
        SweepConfig(
            id="BL-P19plateau",
            block="BASELINE",
            name="Phase 19 plateau (MR80/TS10/RSI10 tv=0.25 ml=14 DDoff)",
            sleeves=_P18_SLEEVES,
            allocation="custom",
            target_vol=_PLATEAU_TV,
            max_leverage=_PLATEAU_ML,
            dd_cap_enabled=_PLATEAU_DD,
            custom_weights=dict(_P18_WEIGHTS),
        ),
    ]

    return cfgs


# ═══════════════════════════════════════════════════════════════════════
# Reporting
# ═══════════════════════════════════════════════════════════════════════


def _fmt_pct(x: float, width: int = 8) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "".rjust(width)
    return f"{x * 100:>{width - 1}.2f}%"


def build_phase20a_markdown(
    rows: list[dict[str, Any]],
    bootstrap_by_id: dict[str, dict[str, float]],
    report_date: str,
) -> str:
    sorted_rows = sorted(rows, key=lambda r: r["wf_avg_sharpe"], reverse=True)
    top10 = sorted_rows[:10]
    p18_row = next((r for r in rows if r["id"] == "BL-P18prod"), None)
    p19_row = next((r for r in rows if r["id"] == "BL-P19plateau"), None)

    lines: list[str] = []
    lines.append(f"# Phase 20A — Weight sweep ({report_date})\n")
    lines.append(
        "Suite directe du sweep Phase 19 "
        "(`docs/research/phase19_2026-04-13_refined_leverage.md`) qui avait "
        "établi un plateau Sharpe à 0.966 sur toute la région "
        "`(target_vol ≥ 0.22, max_leverage ≥ 14, DDoff)` avec les poids Phase 18 "
        "figés à 80/10/10. Phase 20A pose la question complémentaire : est-ce "
        "qu'une autre répartition des poids entre MR_Macro, TS_Momentum_3p et "
        "RSI_Daily_4p permet de sortir de ce plateau ?\n"
    )
    lines.append(
        "Paramètres fixes du bloc CORE : `tv=0.25 / ml=14 / DDoff` (le point "
        "le plus conservateur du plateau Phase 19). Grille : 8 valeurs de "
        "`mr_weight` × 5 valeurs de `rsi_fraction` (part du budget "
        "diversifieur allouée à RSI vs TS) = 40 configs CORE, plus 10 configs "
        "ROBUST pour vérifier la stabilité à `tv ∈ {0.22, 0.28}` et 6 configs "
        "LIGHT_MR contrariennes avec `mr_weight ∈ {0.40, 0.50}`.\n"
    )
    if p19_row:
        lines.append(
            f"Baseline Phase 19 plateau : **{p19_row['id']}** — Sharpe WF "
            f"{p19_row['wf_avg_sharpe']:.3f}, CAGR "
            f"{p19_row['annual_return'] * 100:.2f}%, "
            f"MaxDD {p19_row['max_drawdown'] * 100:.2f}%.\n"
        )
    if p18_row:
        lines.append(
            f"Baseline Phase 18 prod : **{p18_row['id']}** — Sharpe WF "
            f"{p18_row['wf_avg_sharpe']:.3f}, CAGR "
            f"{p18_row['annual_return'] * 100:.2f}%, "
            f"MaxDD {p18_row['max_drawdown'] * 100:.2f}%.\n"
        )

    # ── Top-10 ──────────────────────────────────────────────────────
    lines.append("## Top 10 par Walk-Forward Sharpe\n")
    lines.append("| Rank | ID | Config | Sharpe WF | CAGR | Vol | MaxDD | WF pos | ★ |")
    lines.append(
        "|------|----|--------|-----------|------|-----|-------|--------|----|"
    )
    for i, r in enumerate(top10, 1):
        mark = _target_hit_mark(r["annual_return"], r["max_drawdown"])
        lines.append(
            f"| {i} | `{r['id']}` | {r['name']} "
            f"| **{r['wf_avg_sharpe']:.3f}** "
            f"| {_fmt_pct(r['annual_return'])} "
            f"| {_fmt_pct(r['annual_vol'])} "
            f"| {_fmt_pct(r['max_drawdown'])} "
            f"| {r['wf_pos_years']}/7 | {mark} |"
        )
    lines.append("\n★ = CAGR ∈ [10%, 15%] AND MaxDD < 35%.\n")

    # ── Per-block best ─────────────────────────────────────────────
    lines.append("## Meilleure config par bloc\n")
    lines.append("| Bloc | ID | Config | Sharpe WF | CAGR | MaxDD |")
    lines.append("|------|----|--------|-----------|------|-------|")
    for block in ("CORE", "ROBUST", "LIGHT_MR", "BASELINE"):
        block_rows = [r for r in sorted_rows if r["block"] == block]
        if not block_rows:
            continue
        r = block_rows[0]
        lines.append(
            f"| {block} | `{r['id']}` | {r['name']} "
            f"| **{r['wf_avg_sharpe']:.3f}** "
            f"| {_fmt_pct(r['annual_return'])} "
            f"| {_fmt_pct(r['max_drawdown'])} |"
        )
    lines.append("")

    # ── CORE grid heatmap (MR × RSI fraction) ──────────────────────
    lines.append("## CORE grid — Sharpe WF par (MR, RSI fraction)\n")
    core_rows = [r for r in rows if r["block"] == "CORE"]
    core_by_id = {r["id"]: r for r in core_rows}
    mr_vals = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
    rsi_vals = [0.25, 0.40, 0.50, 0.60, 0.75]
    hdr = "| MR \\ RSI frac | " + " | ".join(f"{rf:.2f}" for rf in rsi_vals) + " |"
    sep = "|" + "---|" * (len(rsi_vals) + 1)
    lines.append(hdr)
    lines.append(sep)
    for mr in mr_vals:
        cells: list[str] = [f"{mr:.2f}"]
        for rsi_f in rsi_vals:
            tag = _weight_tag(mr, rsi_f)
            cid = f"P20a-w{tag}"
            row = core_by_id.get(cid)
            if row is None:
                cells.append("—")
            else:
                cells.append(f"{row['wf_avg_sharpe']:.3f}")
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")

    # ── Bootstrap top-5 ─────────────────────────────────────────────
    if bootstrap_by_id:
        lines.append("## Bootstrap stress-test (top-5, 500 × 20-day blocks)\n")
        lines.append(
            "| Rank | ID | CAGR P5 | CAGR P50 | MaxDD P5 | MaxDD P50 | Target hit |"
        )
        lines.append(
            "|------|----|---------|----------|----------|-----------|-----------|"
        )
        for rank, r in enumerate(sorted_rows[:5], 1):
            stats = bootstrap_by_id.get(r["id"])
            if stats is None:
                continue
            lines.append(
                f"| {rank} | `{r['id']}` "
                f"| {stats['cagr_p05'] * 100:+.2f}% "
                f"| {stats['cagr_p50'] * 100:+.2f}% "
                f"| {stats['max_dd_p05'] * 100:+.2f}% "
                f"| {stats['max_dd_p50'] * 100:+.2f}% "
                f"| {stats['target_hit_fraction'] * 100:.1f}% |"
            )
        lines.append("")

    # ── Narrative / conclusion ─────────────────────────────────────
    lines.append("## Conclusion\n")
    if not sorted_rows:
        lines.append("Aucun résultat collecté.\n")
    else:
        top = sorted_rows[0]
        plateau_sharpe = p19_row["wf_avg_sharpe"] if p19_row else 0.966
        gap = top["wf_avg_sharpe"] - plateau_sharpe
        sign = "+" if gap >= 0 else ""
        lines.append(
            f"Meilleur point du sweep : **`{top['id']}`** — {top['name']}.  \n"
            f"Sharpe WF = **{top['wf_avg_sharpe']:.3f}** "
            f"(vs plateau Phase 19 = {plateau_sharpe:.3f}, "
            f"Δ = {sign}{gap:.3f})."
        )
        if abs(gap) < 0.005:
            lines.append(
                "\nLe plateau Phase 19 **résiste** au changement de poids : "
                "aucune recombinaison des poids MR/TS/RSI ne fait bouger "
                "le Sharpe WF de plus de 5 millipoints. Le plateau est donc "
                "défini conjointement par (i) la composition des sleeves et "
                "(ii) la layer de leverage — pas par la répartition des "
                "poids elle-même."
            )
        elif gap > 0:
            lines.append(
                "\nUn nouveau point **domine** le plateau Phase 19 — à "
                "valider sur le bootstrap et sur les points ROBUST avant "
                "de le promouvoir comme recommandation Phase 20."
            )
        else:
            lines.append(
                "\nAucune configuration de poids ne dépasse le plateau "
                "Phase 19 ; le 80/10/10 reste la meilleure répartition "
                "parmi celles testées."
            )
    lines.append("")
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="Run 4 configs only")
    parser.add_argument(
        "--no-bootstrap",
        action="store_true",
        help="Skip bootstrap stress-test on top-5",
    )
    parser.add_argument("--bootstrap-runs", type=int, default=500)
    args = parser.parse_args()

    import vectorbtpro as vbt

    from strategies.combined_portfolio import get_strategy_daily_returns
    from utils import apply_vbt_settings

    apply_vbt_settings()
    vbt.settings.returns.year_freq = pd.Timedelta(days=252)

    report_date = date.today().isoformat()
    if args.smoke:
        output_root = Path("/tmp") / f"phase20a_smoke_{report_date}"
        md_path = output_root / "phase20a.md"
    else:
        output_root = _PROJECT_ROOT / "results" / f"phase20a_{report_date}"
        md_path = (
            _PROJECT_ROOT
            / "docs"
            / "research"
            / f"phase20a_{report_date}_weight_sweep.md"
        )
    output_root.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("  Phase 20A — Weight sweep")
    print("=" * 72)
    print(f"Report date : {report_date}")
    print(f"Output root : {output_root}")
    print(f"Markdown    : {md_path}")
    print()

    print("Loading sleeves…")
    sleeves = get_strategy_daily_returns()
    sleeves_p20 = {k: sleeves[k] for k in _P18_SLEEVES}
    print(f"  Loaded {len(sleeves_p20)} sleeves: {sorted(sleeves_p20)}")
    print()

    grid = build_phase20a_grid()
    if args.smoke:
        smoke_ids = {
            "P20a-w80-10-10",
            "P20a-w70-15-15",
            "BL-P18prod",
            "BL-P19plateau",
        }
        grid = [c for c in grid if c.id in smoke_ids]
    print(f"Running {len(grid)} configurations via native parallel sweep…")

    t0 = time.perf_counter()
    pf_all, metrics = native_parallel_sweep(grid, sleeves_p20)
    elapsed = time.perf_counter() - t0
    print(
        f"  Native sweep completed in {elapsed:.1f}s "
        f"({elapsed / max(len(grid), 1):.2f}s per combo)"
    )

    rows = rows_from_native_metrics(grid, metrics, pf_all)
    sorted_rows = sorted(rows, key=lambda r: r["wf_avg_sharpe"], reverse=True)

    for r in sorted_rows[:20]:
        mark = _target_hit_mark(r["annual_return"], r["max_drawdown"])
        print(
            f"  [{r['block']:>8}] {r['id']:<28} "
            f"sharpe={r['wf_avg_sharpe']:+.3f} "
            f"CAGR={r['annual_return'] * 100:>+6.2f}% "
            f"DD={r['max_drawdown'] * 100:>+6.2f}% {mark}"
        )
    if len(sorted_rows) > 20:
        print(f"  … and {len(sorted_rows) - 20} more (see markdown report)")

    # ── Bootstrap top-5 ────────────────────────────────────────────
    bootstrap_by_id: dict[str, dict[str, float]] = {}
    if not args.smoke and not args.no_bootstrap:
        print(f"\nBootstrapping top-5 ({args.bootstrap_runs} runs × 20-day blocks)…")
        for rank, row in enumerate(sorted_rows[:5], 1):
            cfg = next(c for c in grid if c.id == row["id"])
            t0 = time.perf_counter()
            try:
                stats = bootstrap_config(cfg, sleeves_p20, n_runs=args.bootstrap_runs)
                bootstrap_by_id[row["id"]] = stats
                el = time.perf_counter() - t0
                print(
                    f"  [top{rank}] {row['id']}: "
                    f"CAGR P5={stats['cagr_p05'] * 100:+.2f}% "
                    f"MaxDD P5={stats['max_dd_p05'] * 100:+.2f}% "
                    f"target_hit={stats['target_hit_fraction'] * 100:.1f}% "
                    f"({el:.0f}s, {stats['n_runs']}/{args.bootstrap_runs})"
                )
            except Exception as exc:
                print(f"  [top{rank}] {row['id']} FAILED: {exc}")

    # ── Export JSON ────────────────────────────────────────────────
    json_path = output_root / "metrics.json"
    json_data = {
        "report_date": report_date,
        "n_configs": len(rows),
        "sleeves_loaded": sorted(sleeves_p20),
        "configs": [sanitize_result_for_json(r) for r in sorted_rows],
        "bootstrap_top5": bootstrap_by_id,
    }
    json_path.write_text(json.dumps(json_data, indent=2, default=str))
    print(f"\nJSON exported → {json_path}")

    # ── Export markdown ────────────────────────────────────────────
    md = build_phase20a_markdown(rows, bootstrap_by_id, report_date)
    md_path.write_text(md)
    print(f"Markdown exported → {md_path}")

    # ── Top-N artifacts ────────────────────────────────────────────
    if not args.smoke:
        print("\nGenerating top-N artifacts (tearsheets + mix plots)…")
        generate_top_n_artifacts(sorted_rows, grid, sleeves_p20, output_root)

    # ── Final summary ──────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  Top 5 by Walk-Forward Sharpe")
    print("=" * 72)
    for i, r in enumerate(sorted_rows[:5], 1):
        print(
            f"  {i}. {r['id']:<30} sharpe={r['wf_avg_sharpe']:+.3f}  "
            f"CAGR={r['annual_return'] * 100:>+6.2f}%  "
            f"MaxDD={r['max_drawdown'] * 100:>+6.2f}%"
        )
    p19_plateau_row = next((r for r in rows if r["id"] == "BL-P19plateau"), None)
    if p19_plateau_row:
        rank = next(
            i for i, r in enumerate(sorted_rows, 1) if r["id"] == "BL-P19plateau"
        )
        print(
            f"\n  Phase 19 plateau baseline: rank #{rank}, "
            f"sharpe={p19_plateau_row['wf_avg_sharpe']:+.3f}"
        )


if __name__ == "__main__":
    main()
