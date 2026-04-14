"""Drawdown cap schedule sweep — graduated soft caps.

Follow-up to the weight and fourth-sleeve sweeps. The leverage sweep
established that the binary DD cap (ON with the original hard schedule
vs OFF entirely) has a clear winner : OFF dominates the Sharpe plateau
at 0.966 because the hard schedule de-leverages recoveries that would
have come back. This sweep asks : is there a **softer** DD schedule
that keeps most of the upside (Sharpe ≥ 0.96) while improving the tail
risk (better P5 MaxDD on bootstrap) ?

Methodology
-----------
Uses the leverage plateau sweet-spot ``tv=0.25 ml=14`` as the fixed
leverage layer and sweeps 8 alternate DD schedules alongside the two
baselines (``OFF`` and ``ON-default``). Each alternate schedule is
parameterized by two knobs :

- **``dd_knee``** — the DD level at which de-leveraging starts
  (below this, scale = 1.0). Tested at {0.10, 0.15, 0.20}.
- **``dd_floor``** — the minimum scale reached at DD = 35% (max DD
  clipped). Tested at {0.35, 0.50, 0.70, 0.85}.

The schedule is linear between ``(dd_knee, 1.0)`` and ``(0.35, dd_floor)``
then clipped to ``dd_floor`` beyond 35% DD. A ``knee=0.10, floor=0.15``
approximates the original hard schedule ; a ``knee=0.20, floor=0.85``
is a very soft cap that barely touches the leverage.

Grid (~34 configs)
------------------
**Block SOFT_CAP** (~24 configs):
  ``dd_knee`` ∈ {0.10, 0.15, 0.20} × ``dd_floor`` ∈ {0.35, 0.50, 0.70, 0.85}
  × weights ∈ {MR80/TS10/RSI10, MR75/TS10/RSI15} × tv=0.25 ml=14.
  = 24 configs.

**Block KNEE_DEEP** (6 configs):
  Very deep knees for comparison: ``dd_knee`` ∈ {0.05, 0.25} × 3 floors.
  Single weight config (canonical 80/10/10).

**Block BASELINE** (4 configs):
  OFF / ON-default at each of the 2 weight mixes.

Grand total: 24 + 6 + 4 = **34 configs**

Usage
-----
    python scripts/sweep_dd_cap_schedule.py
    python scripts/sweep_dd_cap_schedule.py --smoke
    python scripts/sweep_dd_cap_schedule.py --no-bootstrap
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
_PLATEAU_TV: float = 0.25
_PLATEAU_ML: float = 14.0

_CANONICAL_WEIGHTS: dict[str, float] = {
    "MR_Macro": 0.80,
    "TS_Momentum_3p": 0.10,
    "RSI_Daily_4p": 0.10,
}
_WEIGHT_TOP_WEIGHTS: dict[str, float] = {
    "MR_Macro": 0.75,
    "TS_Momentum_3p": 0.10,
    "RSI_Daily_4p": 0.15,
}


def _make_soft_schedule(
    knee: float,
    floor: float,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Build a 4-point soft DD schedule.

    Schedule semantics (linear interpolation via ``np.interp``):
        DD ∈ [0, knee]          → scale = 1.0
        DD ∈ [knee, 0.35]       → linear taper 1.0 → floor
        DD > 0.35               → clipped at floor (np.interp extrapolation)
    """
    bps = (0.0, knee, 0.35, 1.0)
    scl = (1.0, 1.0, floor, floor)
    return bps, scl


def _schedule_tag(knee: float, floor: float) -> str:
    return f"k{int(round(knee * 100)):02d}-f{int(round(floor * 100)):02d}"


# ═══════════════════════════════════════════════════════════════════════
# Grid builder
# ═══════════════════════════════════════════════════════════════════════


def build_dd_cap_grid() -> list[SweepConfig]:
    cfgs: list[SweepConfig] = []

    # ─── Block SOFT_CAP — 3 knees × 4 floors × 2 weight mixes ───────
    KNEES = [0.10, 0.15, 0.20]
    FLOORS = [0.35, 0.50, 0.70, 0.85]
    WEIGHT_MIXES = [
        ("w80-10-10", _CANONICAL_WEIGHTS),
        ("w75-10-15", _WEIGHT_TOP_WEIGHTS),
    ]
    for weight_tag, w in WEIGHT_MIXES:
        for knee in KNEES:
            for floor in FLOORS:
                bps, scl = _make_soft_schedule(knee, floor)
                sched_tag = _schedule_tag(knee, floor)
                cfgs.append(
                    SweepConfig(
                        id=f"DDC-soft-{weight_tag}-{sched_tag}",
                        block="SOFT_CAP",
                        name=(
                            f"{weight_tag} / tv=0.25 ml=14 / "
                            f"DDsoft(knee={knee:.2f}, floor={floor:.2f})"
                        ),
                        sleeves=_BASELINE_SLEEVES,
                        allocation="custom",
                        target_vol=_PLATEAU_TV,
                        max_leverage=_PLATEAU_ML,
                        dd_cap_enabled=True,
                        custom_weights=dict(w),
                        dd_breakpoints=bps,
                        dd_scales=scl,
                    )
                )

    # ─── Block KNEE_DEEP — extreme knees, canonical weights only ────
    EXTRA_KNEES = [0.05, 0.25]
    EXTRA_FLOORS = [0.35, 0.50, 0.70]
    for knee in EXTRA_KNEES:
        for floor in EXTRA_FLOORS:
            bps, scl = _make_soft_schedule(knee, floor)
            sched_tag = _schedule_tag(knee, floor)
            cfgs.append(
                SweepConfig(
                    id=f"DDC-deep-w80-10-10-{sched_tag}",
                    block="KNEE_DEEP",
                    name=(
                        f"w80-10-10 / tv=0.25 ml=14 / "
                        f"DDsoft(knee={knee:.2f}, floor={floor:.2f})"
                    ),
                    sleeves=_BASELINE_SLEEVES,
                    allocation="custom",
                    target_vol=_PLATEAU_TV,
                    max_leverage=_PLATEAU_ML,
                    dd_cap_enabled=True,
                    custom_weights=dict(_CANONICAL_WEIGHTS),
                    dd_breakpoints=bps,
                    dd_scales=scl,
                )
            )

    # ─── Block BASELINE — OFF + ON-default at each weight mix ───────
    for weight_tag, w in WEIGHT_MIXES:
        cfgs.append(
            SweepConfig(
                id=f"BL-DDoff-{weight_tag}",
                block="BASELINE",
                name=f"{weight_tag} / tv=0.25 ml=14 / DDoff",
                sleeves=_BASELINE_SLEEVES,
                allocation="custom",
                target_vol=_PLATEAU_TV,
                max_leverage=_PLATEAU_ML,
                dd_cap_enabled=False,
                custom_weights=dict(w),
            )
        )
        cfgs.append(
            SweepConfig(
                id=f"BL-DDon-{weight_tag}",
                block="BASELINE",
                name=f"{weight_tag} / tv=0.25 ml=14 / DDon (default schedule)",
                sleeves=_BASELINE_SLEEVES,
                allocation="custom",
                target_vol=_PLATEAU_TV,
                max_leverage=_PLATEAU_ML,
                dd_cap_enabled=True,
                custom_weights=dict(w),
                # dd_breakpoints / dd_scales = None → use module defaults
                # (the original hard de-leveraging schedule).
            )
        )

    return cfgs


# ═══════════════════════════════════════════════════════════════════════
# Reporting
# ═══════════════════════════════════════════════════════════════════════


def _fmt_pct(x: float, width: int = 8) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "".rjust(width)
    return f"{x * 100:>{width - 1}.2f}%"


def build_dd_cap_markdown(
    rows: list[dict[str, Any]],
    bootstrap_by_id: dict[str, dict[str, float]],
    report_date: str,
) -> str:
    sorted_rows = sorted(rows, key=lambda r: r["wf_avg_sharpe"], reverse=True)
    top10 = sorted_rows[:10]
    off_80 = next((r for r in rows if r["id"] == "BL-DDoff-w80-10-10"), None)
    off_75 = next((r for r in rows if r["id"] == "BL-DDoff-w75-10-15"), None)
    on_80 = next((r for r in rows if r["id"] == "BL-DDon-w80-10-10"), None)

    lines: list[str] = []
    lines.append(f"# Drawdown cap schedule sweep ({report_date})\n")
    lines.append(
        "Follow-up to the weight and fourth-sleeve sweeps. The binary DD "
        "cap (ON with the original hard schedule "
        "`{0.10→1.0, 0.20→0.60, 0.30→0.35, 0.35→0.15}` vs OFF entirely) "
        "showed that `DDoff` systematically beats the default-ON schedule "
        "at `tv=0.25 ml=14` because the cap de-leverages drawdowns that "
        "recover. This sweep tests **softer** schedules parameterized by "
        "two knobs :\n"
    )
    lines.append(
        "- `dd_knee` — the DD level at which de-leveraging starts (scale "
        "= 1.0 below). Tested in {0.10, 0.15, 0.20} + extremes {0.05, 0.25}.\n"
        "- `dd_floor` — the scale reached at DD = 35% (clipped beyond). "
        "Tested in {0.35, 0.50, 0.70, 0.85}.\n"
    )
    lines.append(
        "A schedule `knee=0.10, floor=0.15` approximates the original hard "
        "cap ; `knee=0.20, floor=0.85` is nearly inactive. Fixed parameters "
        ": `tv=0.25 ml=14` on the production trio. Two weight mixes tested "
        ": the canonical 80/10/10 and the weight-sweep top 75/10/15.\n"
    )

    if off_80:
        lines.append(
            f"Baseline `DDoff` (80-10-10) : Sharpe WF "
            f"{off_80['wf_avg_sharpe']:.3f}, CAGR "
            f"{off_80['annual_return'] * 100:.2f}%, "
            f"MaxDD {off_80['max_drawdown'] * 100:.2f}%.\n"
        )
    if off_75:
        lines.append(
            f"Baseline `DDoff` (75-10-15) : Sharpe WF "
            f"{off_75['wf_avg_sharpe']:.3f}, CAGR "
            f"{off_75['annual_return'] * 100:.2f}%, "
            f"MaxDD {off_75['max_drawdown'] * 100:.2f}%.\n"
        )
    if on_80:
        lines.append(
            f"Baseline `DDon-default` (80-10-10) : Sharpe WF "
            f"{on_80['wf_avg_sharpe']:.3f}, CAGR "
            f"{on_80['annual_return'] * 100:.2f}%, "
            f"MaxDD {on_80['max_drawdown'] * 100:.2f}%.\n"
        )

    # ── Top-10 ─────────────────────────────────────────────────────
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

    # ── SOFT_CAP heatmap per weight mix ────────────────────────────
    soft_by_id = {r["id"]: r for r in rows if r["block"] == "SOFT_CAP"}
    for weight_tag, mix_name in [
        ("w80-10-10", "80-10-10 (canonical)"),
        ("w75-10-15", "75-10-15 (weight-sweep top)"),
    ]:
        lines.append(
            f"## SOFT_CAP — Sharpe WF par (knee, floor) — weights {mix_name}\n"
        )
        knees = [0.10, 0.15, 0.20]
        floors = [0.35, 0.50, 0.70, 0.85]
        hdr = "| knee \\ floor | " + " | ".join(f"{f:.2f}" for f in floors) + " |"
        sep = "|" + "---|" * (len(floors) + 1)
        lines.append(hdr)
        lines.append(sep)
        for k in knees:
            cells: list[str] = [f"{k:.2f}"]
            for f in floors:
                sched_tag = _schedule_tag(k, f)
                cid = f"DDC-soft-{weight_tag}-{sched_tag}"
                r = soft_by_id.get(cid)
                if r is None:
                    cells.append("—")
                else:
                    cells.append(f"{r['wf_avg_sharpe']:.3f}")
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")

    # ── Per-block best ─────────────────────────────────────────────
    lines.append("## Best config per block\n")
    lines.append("| Block | ID | Config | Sharpe WF | CAGR | MaxDD |")
    lines.append("|-------|----|--------|-----------|------|-------|")
    for block in ("SOFT_CAP", "KNEE_DEEP", "BASELINE"):
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

    # ── Bootstrap top-5 ────────────────────────────────────────────
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

    # ── Narrative ──────────────────────────────────────────────────
    lines.append("## Conclusion\n")
    if not sorted_rows:
        lines.append("No results collected.\n")
    else:
        top = sorted_rows[0]
        off_sharpe = off_80["wf_avg_sharpe"] if off_80 else 0.966
        gap_off = top["wf_avg_sharpe"] - off_sharpe
        sign = "+" if gap_off >= 0 else ""
        lines.append(
            f"Best point in the sweep : **`{top['id']}`** — {top['name']}.  \n"
            f"Sharpe WF = **{top['wf_avg_sharpe']:.3f}** "
            f"(vs DDoff baseline 80-10-10 = {off_sharpe:.3f}, "
            f"Δ = {sign}{gap_off:.3f})."
        )
        if gap_off > 0.005:
            lines.append(
                "\nA graduated DD cap schedule **improves** the Sharpe over "
                "pure DDoff mode. The soft cap preserves recoveries on "
                "moderate drawdowns while trimming the tail of extreme "
                "scenarios."
            )
        elif abs(gap_off) <= 0.005:
            lines.append(
                "\n`DDoff` remains **equivalent** to the best graduated "
                "schedule : in-sample, the cap does not fire often enough "
                "to make a difference. The real impact shows up in the "
                "bootstrap tails — see the P5 MaxDD table above."
            )
        else:
            lines.append(
                "\nNo graduated schedule exceeds `DDoff` on the WF Sharpe. "
                "The cap remains disabled in production."
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
        output_root = Path("/tmp") / f"dd_cap_schedule_smoke_{report_date}"
        md_path = output_root / "dd_cap_schedule.md"
    else:
        output_root = _PROJECT_ROOT / "results" / f"dd_cap_schedule_{report_date}"
        md_path = (
            _PROJECT_ROOT / "docs" / "research" / f"dd_cap_schedule_{report_date}.md"
        )
    output_root.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("  Drawdown cap schedule sweep")
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

    grid = build_dd_cap_grid()
    if args.smoke:
        smoke_ids = {
            "DDC-soft-w80-10-10-k15-f50",
            "BL-DDoff-w80-10-10",
            "BL-DDon-w80-10-10",
            "DDC-soft-w75-10-15-k20-f70",
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

    for r in sorted_rows[:25]:
        mark = _target_hit_mark(r["annual_return"], r["max_drawdown"])
        print(
            f"  [{r['block']:>9}] {r['id']:<42} "
            f"sharpe={r['wf_avg_sharpe']:+.3f} "
            f"CAGR={r['annual_return'] * 100:>+6.2f}% "
            f"DD={r['max_drawdown'] * 100:>+6.2f}% {mark}"
        )
    if len(sorted_rows) > 25:
        print(f"  … and {len(sorted_rows) - 25} more (see markdown report)")

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
    md = build_dd_cap_markdown(rows, bootstrap_by_id, report_date)
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
            f"  {i}. {r['id']:<42} sharpe={r['wf_avg_sharpe']:+.3f}  "
            f"CAGR={r['annual_return'] * 100:>+6.2f}%  "
            f"MaxDD={r['max_drawdown'] * 100:>+6.2f}%"
        )


if __name__ == "__main__":
    main()
