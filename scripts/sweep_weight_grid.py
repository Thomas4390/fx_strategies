"""Weight sweep — dense factorial grid on the production trio.

Follow-up to ``scripts/sweep_leverage_grid.py``. The leverage sweep
pinned the weights to ``MR80 / TS3p10 / RSI10`` and only swept the
leverage layer; it reached a Sharpe plateau at 0.966 across any
``(target_vol >= 0.22, max_leverage >= 14, DDoff)`` point. This sweep
asks the complementary question : does a *different* weight mix break
out of that plateau when fed through the same leverage layer ?

Grid (~60 configs)
------------------
**Core weight grid** (tv=0.25 ml=14 DDoff — the leverage plateau sweet
spot):
  - ``mr_weight`` ∈ {0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90} (8)
  - ``rsi_fraction`` ∈ {0.25, 0.40, 0.50, 0.60, 0.75} — the share of the
    diversifier budget (= 1 - mr) routed to ``RSI_Daily_4p`` vs
    ``TS_Momentum_3p`` (5)
  - Total: 8 × 5 = **40 configs**

**Leverage robustness cross-check** (best-5 weight mixes replayed on the
two adjacent plateau points ``tv=0.22 ml=14`` and ``tv=0.28 ml=14``):
  - 5 weight mixes × 2 (tv, ml) pairs × 1 DDoff = **10 configs**

**Light-MR contrarian** (drop MR below 0.55 — stress-test the
"MR alpha is the only alpha" assumption):
  - ``mr_weight`` ∈ {0.40, 0.50}, ``rsi_fraction`` ∈ {0.40, 0.50, 0.60}
  - (tv=0.25, ml=14, DDoff)
  - Total: **6 configs**

**Baselines** (production prod + leverage plateau top point):
  - ``BL-prod`` — MR80/TS10/RSI10 tv=0.28 ml=12 DDoff
  - ``BL-plateau`` — MR80/TS10/RSI10 tv=0.25 ml=14 DDoff
  - Total: **2 configs**

Grand total: 40 + 10 + 6 + 2 = **58 configs**

Usage
-----
    python scripts/sweep_weight_grid.py             # full run
    python scripts/sweep_weight_grid.py --smoke     # 4 configs
    python scripts/sweep_weight_grid.py --no-bootstrap
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


_BASELINE_SLEEVES: tuple[str, ...] = (
    "MR_Macro",
    "TS_Momentum_3p",
    "RSI_Daily_4p",
)
_BASELINE_WEIGHTS: dict[str, float] = {
    "MR_Macro": 0.80,
    "TS_Momentum_3p": 0.10,
    "RSI_Daily_4p": 0.10,
}

# Leverage plateau sweet-spot: tv=0.25, ml=14, DDoff.
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


def build_weight_grid() -> list[SweepConfig]:
    """Return the full weight sweep grid (~58 configurations)."""
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
                    id=f"WGT-w{tag}",
                    block="CORE",
                    name=f"weights={tag} / tv=0.25 ml=14 DDoff",
                    sleeves=_BASELINE_SLEEVES,
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
        (0.80, 0.50),  # 80/10/10 (canonical production)
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
                    id=f"WGTR-{tag}-tv{tv_t}",
                    block="ROBUST",
                    name=f"weights={tag} / tv={tv:.2f} ml={int(ml)} DDoff",
                    sleeves=_BASELINE_SLEEVES,
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
                    id=f"WGTL-{tag}",
                    block="LIGHT_MR",
                    name=f"weights={tag} (light-MR) / tv=0.25 ml=14 DDoff",
                    sleeves=_BASELINE_SLEEVES,
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
            id="BL-prod",
            block="BASELINE",
            name="production (MR80/TS10/RSI10 tv=0.28 ml=12 DDoff)",
            sleeves=_BASELINE_SLEEVES,
            allocation="custom",
            target_vol=0.28,
            max_leverage=12.0,
            dd_cap_enabled=False,
            custom_weights=dict(_BASELINE_WEIGHTS),
        ),
        SweepConfig(
            id="BL-plateau",
            block="BASELINE",
            name="leverage plateau (MR80/TS10/RSI10 tv=0.25 ml=14 DDoff)",
            sleeves=_BASELINE_SLEEVES,
            allocation="custom",
            target_vol=_PLATEAU_TV,
            max_leverage=_PLATEAU_ML,
            dd_cap_enabled=_PLATEAU_DD,
            custom_weights=dict(_BASELINE_WEIGHTS),
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


def build_weight_markdown(
    rows: list[dict[str, Any]],
    bootstrap_by_id: dict[str, dict[str, float]],
    report_date: str,
) -> str:
    sorted_rows = sorted(rows, key=lambda r: r["wf_avg_sharpe"], reverse=True)
    top10 = sorted_rows[:10]
    prod_row = next((r for r in rows if r["id"] == "BL-prod"), None)
    plateau_row = next((r for r in rows if r["id"] == "BL-plateau"), None)

    lines: list[str] = []
    lines.append(f"# Weight sweep — dense factorial ({report_date})\n")
    lines.append(
        "Follow-up to the leverage sweep which established a Sharpe plateau "
        "at 0.966 across the region `(target_vol ≥ 0.22, max_leverage ≥ 14, "
        "DDoff)` with production weights fixed at 80/10/10. This sweep asks "
        "the complementary question : does a different weight allocation "
        "across MR_Macro, TS_Momentum_3p and RSI_Daily_4p break out of that "
        "plateau ?\n"
    )
    lines.append(
        "Fixed parameters for the CORE block : `tv=0.25 / ml=14 / DDoff` "
        "(the most conservative point on the plateau). Grid : 8 values of "
        "`mr_weight` × 5 values of `rsi_fraction` (share of the diversifier "
        "budget allocated to RSI vs TS) = 40 CORE configs, plus 10 ROBUST "
        "configs to check stability at `tv ∈ {0.22, 0.28}` and 6 contrarian "
        "LIGHT_MR configs with `mr_weight ∈ {0.40, 0.50}`.\n"
    )
    if plateau_row:
        lines.append(
            f"Leverage plateau baseline : **{plateau_row['id']}** — Sharpe WF "
            f"{plateau_row['wf_avg_sharpe']:.3f}, CAGR "
            f"{plateau_row['annual_return'] * 100:.2f}%, "
            f"MaxDD {plateau_row['max_drawdown'] * 100:.2f}%.\n"
        )
    if prod_row:
        lines.append(
            f"Production baseline : **{prod_row['id']}** — Sharpe WF "
            f"{prod_row['wf_avg_sharpe']:.3f}, CAGR "
            f"{prod_row['annual_return'] * 100:.2f}%, "
            f"MaxDD {prod_row['max_drawdown'] * 100:.2f}%.\n"
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
    lines.append("## Best config per block\n")
    lines.append("| Block | ID | Config | Sharpe WF | CAGR | MaxDD |")
    lines.append("|-------|----|--------|-----------|------|-------|")
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
            cid = f"WGT-w{tag}"
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
        lines.append("No results collected.\n")
    else:
        top = sorted_rows[0]
        plateau_sharpe = plateau_row["wf_avg_sharpe"] if plateau_row else 0.966
        gap = top["wf_avg_sharpe"] - plateau_sharpe
        sign = "+" if gap >= 0 else ""
        lines.append(
            f"Best point in the sweep : **`{top['id']}`** — {top['name']}.  \n"
            f"Sharpe WF = **{top['wf_avg_sharpe']:.3f}** "
            f"(vs leverage plateau = {plateau_sharpe:.3f}, "
            f"Δ = {sign}{gap:.3f})."
        )
        if abs(gap) < 0.005:
            lines.append(
                "\nThe leverage plateau **resists** weight recombination : "
                "no mix of MR/TS/RSI weights moves the WF Sharpe by more "
                "than 5 basis points. The plateau is therefore defined "
                "jointly by (i) the sleeve composition and (ii) the "
                "leverage layer — not by the weight split itself."
            )
        elif gap > 0:
            lines.append(
                "\nA new point **dominates** the leverage plateau — validate "
                "against bootstrap and ROBUST points before promoting it "
                "as the next production recommendation."
            )
        else:
            lines.append(
                "\nNo weight configuration exceeds the leverage plateau ; "
                "80/10/10 remains the best allocation among those tested."
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
        output_root = Path("/tmp") / f"weight_grid_smoke_{report_date}"
        md_path = output_root / "weight_grid.md"
    else:
        output_root = _PROJECT_ROOT / "results" / f"weight_grid_{report_date}"
        md_path = _PROJECT_ROOT / "docs" / "research" / f"weight_grid_{report_date}.md"
    output_root.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("  Weight sweep — dense factorial")
    print("=" * 72)
    print(f"Report date : {report_date}")
    print(f"Output root : {output_root}")
    print(f"Markdown    : {md_path}")
    print()

    print("Loading sleeves…")
    sleeves = get_strategy_daily_returns()
    sleeves_trio = {k: sleeves[k] for k in _BASELINE_SLEEVES}
    print(f"  Loaded {len(sleeves_trio)} sleeves: {sorted(sleeves_trio)}")
    print()

    grid = build_weight_grid()
    if args.smoke:
        smoke_ids = {
            "WGT-w80-10-10",
            "WGT-w70-15-15",
            "BL-prod",
            "BL-plateau",
        }
        grid = [c for c in grid if c.id in smoke_ids]
    print(f"Running {len(grid)} configurations via native parallel sweep…")

    t0 = time.perf_counter()
    pf_all, metrics = native_parallel_sweep(grid, sleeves_trio)
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
                stats = bootstrap_config(cfg, sleeves_trio, n_runs=args.bootstrap_runs)
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
        "sleeves_loaded": sorted(sleeves_trio),
        "configs": [sanitize_result_for_json(r) for r in sorted_rows],
        "bootstrap_top5": bootstrap_by_id,
    }
    json_path.write_text(json.dumps(json_data, indent=2, default=str))
    print(f"\nJSON exported → {json_path}")

    # ── Export markdown ────────────────────────────────────────────
    md = build_weight_markdown(rows, bootstrap_by_id, report_date)
    md_path.write_text(md)
    print(f"Markdown exported → {md_path}")

    # ── Top-N artifacts ────────────────────────────────────────────
    if not args.smoke:
        print("\nGenerating top-N artifacts (tearsheets + mix plots)…")
        generate_top_n_artifacts(sorted_rows, grid, sleeves_trio, output_root)

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
    plateau_rank_row = next((r for r in rows if r["id"] == "BL-plateau"), None)
    if plateau_rank_row:
        rank = next(i for i, r in enumerate(sorted_rows, 1) if r["id"] == "BL-plateau")
        print(
            f"\n  Leverage plateau baseline: rank #{rank}, "
            f"sharpe={plateau_rank_row['wf_avg_sharpe']:+.3f}"
        )


if __name__ == "__main__":
    main()
