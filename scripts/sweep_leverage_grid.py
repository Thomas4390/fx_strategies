"""Leverage sweep — refined grid around the production trio.

Follow-up to ``scripts/sweep_combinations.py``. Drills into the promising
region of the canonical production trio
(`MR_Macro + TS_Momentum_3p + RSI_Daily_4p`) with a much denser grid on
``target_vol × max_leverage × dd_cap_enabled``, plus a small
weights-variation sub-block around the canonical 80/10/10 baseline.

Grid (~116 configs)
-------------------
**Core dense grid** (production weights fixed at 80/10/10):
  - ``target_vol``  ∈ {0.22, 0.25, 0.28, 0.30, 0.32, 0.35} (6)
  - ``max_leverage`` ∈ {4, 6, 8, 10, 12, 14, 16, 18} (8)
  - ``dd_cap_enabled`` ∈ {ON, OFF} (2)
  - Total: 6 × 8 × 2 = **96 configs**

**High-leverage experimental**:
  - ``target_vol`` ∈ {0.32, 0.35, 0.40} × ``max_leverage`` ∈ {18, 20, 24}
  - ``dd_cap_enabled`` = ON (mandatory at this leverage)
  - Total: **9 configs**

**Weights variations** (best observed tv/ml/dd from above, applied to 7
alternative production-trio weight mixes):
  - 85/10/5, 80/15/5, 80/10/10 (dup for reference), 75/15/10, 75/10/15,
    70/20/10, 70/15/15
  - Total: **7 configs**

**Baselines** (exact production prod and top-3 from previous sweep):
  - prod (MR80/TS3p10/RSI10 tv=0.28 ml=12 DDoff)
  - E5 (ml=14 DDon), E1 (ml=6 DDon), D1 (ml=3 DDon)
  - Total: **4 configs**

Grand total: 96 + 9 + 7 + 4 = **116 configs**

Native parallel execution via ``native_parallel_sweep`` — one
``Portfolio.from_optimizer`` call on a 116 × 3 = 348-column wide
DataFrame grouped by ``config_id``.

Usage
-----
    .venv/bin/python scripts/sweep_leverage_grid.py               # full run
    .venv/bin/python scripts/sweep_leverage_grid.py --no-bootstrap
    .venv/bin/python scripts/sweep_leverage_grid.py --smoke       # 4 combos only
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
for p in (_SRC, _SCRIPTS):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


from sweep_combinations import (  # noqa: E402
    SweepConfig,
    _target_hit_mark,
    bootstrap_config,
    generate_top_n_artifacts,
    native_parallel_sweep,
    rows_from_native_metrics,
    sanitize_result_for_json,
)


_BASELINE_WEIGHTS = {"MR_Macro": 0.80, "TS_Momentum_3p": 0.10, "RSI_Daily_4p": 0.10}
_BASELINE_SLEEVES = ("MR_Macro", "TS_Momentum_3p", "RSI_Daily_4p")


# ═══════════════════════════════════════════════════════════════════════
# Grid builder
# ═══════════════════════════════════════════════════════════════════════


def build_leverage_grid() -> list[SweepConfig]:
    """Return the full leverage sweep grid (~116 configurations)."""
    cfgs: list[SweepConfig] = []

    # ─── Block CORE — dense tv × ml × dd grid (96 configs) ────────
    TARGET_VOLS = [0.22, 0.25, 0.28, 0.30, 0.32, 0.35]
    MAX_LEVS = [4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0]
    DD_CAPS = [True, False]
    for tv in TARGET_VOLS:
        for ml in MAX_LEVS:
            for dd in DD_CAPS:
                tv_tag = f"{int(tv * 100):02d}"
                ml_tag = f"{int(ml):02d}"
                dd_tag = "ddON" if dd else "dd_OFF"
                cid = f"LEV-tv{tv_tag}-ml{ml_tag}-{dd_tag}"
                cfgs.append(
                    SweepConfig(
                        id=cid,
                        block="CORE",
                        name=f"baseline weights / tv={tv:.2f} ml={int(ml)} {'DDon' if dd else 'DDoff'}",
                        sleeves=_BASELINE_SLEEVES,
                        allocation="custom",
                        target_vol=tv,
                        max_leverage=ml,
                        dd_cap_enabled=dd,
                        custom_weights=dict(_BASELINE_WEIGHTS),
                    )
                )

    # ─── Block HLEV — high-leverage experimental (9 configs) ──────
    HLEV_GRID = [
        (0.32, 18.0),
        (0.32, 20.0),
        (0.32, 24.0),
        (0.35, 18.0),
        (0.35, 20.0),
        (0.35, 24.0),
        (0.40, 18.0),
        (0.40, 20.0),
        (0.40, 24.0),
    ]
    for tv, ml in HLEV_GRID:
        cfgs.append(
            SweepConfig(
                id=f"LEVH-tv{int(tv * 100):02d}-ml{int(ml):02d}",
                block="HLEV",
                name=f"baseline weights / tv={tv:.2f} ml={int(ml)} DDon (high-lev)",
                sleeves=_BASELINE_SLEEVES,
                allocation="custom",
                target_vol=tv,
                max_leverage=ml,
                dd_cap_enabled=True,
                custom_weights=dict(_BASELINE_WEIGHTS),
            )
        )

    # ─── Block WEIGHTS — production-trio weight variations (7 configs)
    # Applied to a plausible "good-looking" core point (tv=0.28 ml=14 DDon).
    # The question asked here: does a different weight mix around that
    # sweet spot outperform the canonical 80/10/10?
    WEIGHT_VARIATIONS = [
        ("85-10-5", {"MR_Macro": 0.85, "TS_Momentum_3p": 0.10, "RSI_Daily_4p": 0.05}),
        ("80-15-5", {"MR_Macro": 0.80, "TS_Momentum_3p": 0.15, "RSI_Daily_4p": 0.05}),
        ("80-10-10", {"MR_Macro": 0.80, "TS_Momentum_3p": 0.10, "RSI_Daily_4p": 0.10}),
        ("75-15-10", {"MR_Macro": 0.75, "TS_Momentum_3p": 0.15, "RSI_Daily_4p": 0.10}),
        ("75-10-15", {"MR_Macro": 0.75, "TS_Momentum_3p": 0.10, "RSI_Daily_4p": 0.15}),
        ("70-20-10", {"MR_Macro": 0.70, "TS_Momentum_3p": 0.20, "RSI_Daily_4p": 0.10}),
        ("70-15-15", {"MR_Macro": 0.70, "TS_Momentum_3p": 0.15, "RSI_Daily_4p": 0.15}),
    ]
    for tag, w in WEIGHT_VARIATIONS:
        cfgs.append(
            SweepConfig(
                id=f"LEVW-{tag}",
                block="WEIGHTS",
                name=f"weights={tag} / tv=0.28 ml=14 DDon",
                sleeves=_BASELINE_SLEEVES,
                allocation="custom",
                target_vol=0.28,
                max_leverage=14.0,
                dd_cap_enabled=True,
                custom_weights=w,
            )
        )

    # ─── Block BASELINE — reference configs (4 configs) ───────────
    cfgs += [
        SweepConfig(
            "BL-prod",
            "BASELINE",
            "production (MR80/TS3p10/RSI10 tv=0.28 ml=12 DDoff)",
            _BASELINE_SLEEVES,
            "custom",
            0.28,
            12.0,
            False,
            custom_weights=dict(_BASELINE_WEIGHTS),
        ),
        SweepConfig(
            "BL-E5",
            "BASELINE",
            "E5 (baseline weights / tv=0.28 ml=14 DDon)",
            _BASELINE_SLEEVES,
            "custom",
            0.28,
            14.0,
            True,
            custom_weights=dict(_BASELINE_WEIGHTS),
        ),
        SweepConfig(
            "BL-E1",
            "BASELINE",
            "E1 (baseline weights / tv=0.28 ml=6 DDon)",
            _BASELINE_SLEEVES,
            "custom",
            0.28,
            6.0,
            True,
            custom_weights=dict(_BASELINE_WEIGHTS),
        ),
        SweepConfig(
            "BL-D1",
            "BASELINE",
            "D1 (baseline weights / tv=0.10 ml=3 DDon)",
            _BASELINE_SLEEVES,
            "custom",
            0.10,
            3.0,
            True,
            custom_weights=dict(_BASELINE_WEIGHTS),
        ),
    ]

    return cfgs


# ═══════════════════════════════════════════════════════════════════════
# Reporting (custom — uses block-aware narratives)
# ═══════════════════════════════════════════════════════════════════════


def _fmt_pct(x: float, width: int = 8) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "".rjust(width)
    return f"{x * 100:>{width - 1}.2f}%"


def build_leverage_markdown(
    rows: list[dict[str, Any]],
    bootstrap_by_id: dict[str, dict[str, float]],
    report_date: str,
) -> str:
    sorted_rows = sorted(rows, key=lambda r: r["wf_avg_sharpe"], reverse=True)
    top10 = sorted_rows[:10]
    prod_row = next((r for r in rows if r["id"] == "BL-prod"), None)
    e5_row = next((r for r in rows if r["id"] == "BL-E5"), None)

    lines: list[str] = []
    lines.append(f"# Leverage sweep — refined grid ({report_date})\n")
    lines.append(
        "Follow-up to ``sweep_combinations.py`` which revealed that a cluster "
        "(E1/E5/D5/D6 on the production trio with ``dd_cap=ON`` and "
        "``max_leverage`` ∈ [3, 14]) dominated the walk-forward Sharpe without "
        "sacrificing the target CAGR band. This sweep drills densely into "
        "that region with a ``target_vol × max_leverage × dd_cap`` grid, plus "
        "7 weight variations around the 80/10/10 baseline and 9 high-leverage "
        "experimental points (ml ∈ [18, 24]).\n"
    )
    if prod_row:
        lines.append(
            f"Baseline de comparaison : **production** — Sharpe WF "
            f"{prod_row['wf_avg_sharpe']:.3f}, CAGR {prod_row['annual_return'] * 100:.2f}%, "
            f"MaxDD {prod_row['max_drawdown'] * 100:.2f}%.\n"
        )
    if e5_row:
        lines.append(
            f"Top-5 reference : **E5** (ml=14 DDon) — Sharpe WF "
            f"{e5_row['wf_avg_sharpe']:.3f}, CAGR {e5_row['annual_return'] * 100:.2f}%, "
            f"MaxDD {e5_row['max_drawdown'] * 100:.2f}%.\n"
        )

    # ── Top-10 summary ────────────────────────────────────────────
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

    # ── Bootstrap top-5 ───────────────────────────────────────────
    if bootstrap_by_id:
        lines.append("## Bootstrap stress-test (top-5, 500 × 20-day blocks)\n")
        lines.append(
            "| Rank | ID | CAGR P5 | CAGR P50 | MaxDD P5 | MaxDD P50 | "
            "Sharpe P5 | Sharpe P95 | Pos frac | Target hit |"
        )
        lines.append(
            "|------|----|---------|----------|----------|-----------|-----------|-----------|----------|------------|"
        )
        for rank, r in enumerate(sorted_rows[:5], 1):
            b = bootstrap_by_id.get(r["id"])
            if b is None:
                continue
            lines.append(
                f"| {rank} | `{r['id']}` "
                f"| {_fmt_pct(b['cagr_p05'])} "
                f"| {_fmt_pct(b['cagr_p50'])} "
                f"| {_fmt_pct(b['max_dd_p05'])} "
                f"| {_fmt_pct(b['max_dd_p50'])} "
                f"| {b['sharpe_p05']:>+7.3f} "
                f"| {b['sharpe_p95']:>+7.3f} "
                f"| {b['pos_fraction'] * 100:>6.1f}% "
                f"| {b['target_hit_fraction'] * 100:>6.1f}% |"
            )
        lines.append("")

    # ── Best per block (CORE / HLEV / WEIGHTS / BASELINE) ─────────
    lines.append("## Best per block\n")
    for block_id, desc in (
        ("CORE", "Dense tv × ml × dd grid around the production trio"),
        ("HLEV", "High-leverage experimental (ml ∈ [18, 24])"),
        ("WEIGHTS", "Weights variations around 80/10/10"),
        ("BASELINE", "Reference points"),
    ):
        block_rows = [r for r in sorted_rows if r["block"] == block_id]
        if not block_rows:
            continue
        best = max(block_rows, key=lambda r: r["wf_avg_sharpe"])
        lines.append(f"### {block_id} — {desc}")
        lines.append(
            f"- **{len(block_rows)} configs**. Best : `{best['id']}` "
            f"(Sharpe WF **{best['wf_avg_sharpe']:.3f}**, "
            f"CAGR {best['annual_return'] * 100:.2f}%, "
            f"MaxDD {best['max_drawdown'] * 100:.2f}%, "
            f"WF pos {best['wf_pos_years']}/7)."
        )
        if prod_row:
            d = best["wf_avg_sharpe"] - prod_row["wf_avg_sharpe"]
            lines.append(
                f"- vs production : Δ Sharpe WF = {'+' if d >= 0 else ''}{d:.3f}, "
                f"Δ CAGR = {'+' if best['annual_return'] >= prod_row['annual_return'] else ''}"
                f"{(best['annual_return'] - prod_row['annual_return']) * 100:.2f}%."
            )
        lines.append("")

    # ── CORE grid visualisation (pivot table: tv rows × ml cols) ──
    core_rows = [r for r in rows if r["block"] == "CORE"]
    if core_rows:
        lines.append("## CORE grid — Sharpe WF heatmap (DDon only)\n")
        lines.append(
            "Ligne = target_vol, colonne = max_leverage. Valeurs = Sharpe WF "
            "(seulement `dd_cap=ON` pour la lisibilité)."
        )
        lines.append("")
        ddon = [r for r in core_rows if r["dd_cap_enabled"]]
        tvs = sorted({r["target_vol"] for r in ddon})
        mls = sorted({r["max_leverage"] for r in ddon})
        header = "| tv \\ ml |" + "|".join(f" {int(ml):>3} " for ml in mls) + "|"
        sep = "|" + "|".join(["-" * 9] * (len(mls) + 1)) + "|"
        lines.append(header)
        lines.append(sep)
        for tv in tvs:
            row = f"| tv={tv:.2f} |"
            for ml in mls:
                match = next(
                    (
                        r
                        for r in ddon
                        if r["target_vol"] == tv and r["max_leverage"] == ml
                    ),
                    None,
                )
                if match:
                    row += f" {match['wf_avg_sharpe']:.3f} |"
                else:
                    row += "       |"
            lines.append(row)
        lines.append("")

        lines.append("## CORE grid — CAGR heatmap (DDon only)\n")
        lines.append(header)
        lines.append(sep)
        for tv in tvs:
            row = f"| tv={tv:.2f} |"
            for ml in mls:
                match = next(
                    (
                        r
                        for r in ddon
                        if r["target_vol"] == tv and r["max_leverage"] == ml
                    ),
                    None,
                )
                if match:
                    row += f" {match['annual_return'] * 100:>+6.2f}%|"
                else:
                    row += "       |"
            lines.append(row)
        lines.append("")

    # ── Full table ────────────────────────────────────────────────
    lines.append("## Tableau complet (trié par Sharpe WF)\n")
    lines.append(
        "| ID | Block | Config | Sharpe WF | CAGR | Vol | MaxDD | WF pos | ★ |"
    )
    lines.append(
        "|----|-------|--------|-----------|------|-----|-------|--------|----|"
    )
    for r in sorted_rows:
        mark = _target_hit_mark(r["annual_return"], r["max_drawdown"])
        lines.append(
            f"| `{r['id']}` | {r['block']} | {r['name']} "
            f"| {r['wf_avg_sharpe']:.3f} "
            f"| {_fmt_pct(r['annual_return'])} "
            f"| {_fmt_pct(r['annual_vol'])} "
            f"| {_fmt_pct(r['max_drawdown'])} "
            f"| {r['wf_pos_years']}/7 | {mark} |"
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
        output_root = Path("/tmp") / f"leverage_grid_smoke_{report_date}"
        md_path = output_root / "leverage_grid.md"
    else:
        output_root = _PROJECT_ROOT / "results" / f"leverage_grid_{report_date}"
        md_path = (
            _PROJECT_ROOT / "docs" / "research" / f"leverage_grid_{report_date}.md"
        )
    output_root.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("  Leverage sweep — refined grid")
    print("=" * 72)
    print(f"Report date : {report_date}")
    print(f"Output root : {output_root}")
    print(f"Markdown    : {md_path}")
    print()

    print("Loading sleeves…")
    sleeves = get_strategy_daily_returns()
    # Drop sleeves that are not in the production trio (we only need
    # MR_Macro + TS_Momentum_3p + RSI_Daily_4p for this sweep).
    sleeves_trio = {k: sleeves[k] for k in _BASELINE_SLEEVES}
    print(f"  Loaded {len(sleeves_trio)} sleeves: {sorted(sleeves_trio)}")
    print()

    grid = build_leverage_grid()
    if args.smoke:
        smoke_ids = {
            "LEV-tv28-ml12-dd_OFF",
            "LEV-tv28-ml14-ddON",
            "BL-prod",
            "BL-E5",
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

    # Summary line per config (grouped by block for readability)
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

    # ── Bootstrap top-5 ───────────────────────────────────────────
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

    # ── Export JSON ───────────────────────────────────────────────
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

    # ── Export markdown ───────────────────────────────────────────
    md = build_leverage_markdown(rows, bootstrap_by_id, report_date)
    md_path.write_text(md)
    print(f"Markdown exported → {md_path}")

    # ── Top-N artifacts ───────────────────────────────────────────
    if not args.smoke:
        print("\nGenerating top-N artifacts (tearsheets + mix plots)…")
        generate_top_n_artifacts(sorted_rows, grid, sleeves_trio, output_root)

    # ── Final summary ────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  Top 5 by Walk-Forward Sharpe")
    print("=" * 72)
    for i, r in enumerate(sorted_rows[:5], 1):
        print(
            f"  {i}. {r['id']:<30} sharpe={r['wf_avg_sharpe']:+.3f}  "
            f"CAGR={r['annual_return'] * 100:>+6.2f}%  "
            f"MaxDD={r['max_drawdown'] * 100:>+6.2f}%"
        )
    prod_rank_row = next((r for r in rows if r["id"] == "BL-prod"), None)
    if prod_rank_row:
        rank = next(i for i, r in enumerate(sorted_rows, 1) if r["id"] == "BL-prod")
        print(
            f"\n  production (BL-prod) ranks #{rank} of {len(sorted_rows)}  "
            f"(sharpe={prod_rank_row['wf_avg_sharpe']:+.3f})"
        )
    print()


if __name__ == "__main__":
    main()
