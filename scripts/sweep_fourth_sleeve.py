"""Fourth-sleeve evaluation sweep.

Follow-up to ``scripts/sweep_weight_grid.py``. The weight sweep confirmed
that on the 3-sleeve production trio (MR_Macro / TS_Momentum_3p /
RSI_Daily_4p) the walk-forward Sharpe plateau is essentially flat
between 0.96 and 0.97 regardless of the weight split — the alpha is
dominated by MR_Macro and adding more weight to either diversifier
only moves the Sharpe by a few basis points. This sweep asks the next
question: **does adding a fourth orthogonal sleeve push the Sharpe
above 0.97 ?**

Candidate 4th sleeves
---------------------
- **Composite FX Alpha** (``strategies.composite_fx_alpha.pipeline``) —
  daily multi-factor trend + vol + drawdown on EUR-USD. Defaults are
  ``w_short=21, w_long=63, target_vol=0.10, leverage=2.0``.
- **OU Mean Reversion** (``strategies.ou_mean_reversion.pipeline``) —
  intraday VWAP mean reversion with vol-targeted dynamic leverage.
  Defaults are ``bb_window=80, sigma_target=0.10, max_leverage=3``.
- **XS Momentum** (existing sleeve) — revalidates the earlier decision
  to drop it. Added at small weights.

Grid (~50 configs)
------------------
**Block COMPOSITE** (15 configs):
  Add ``composite_fx_alpha`` at weight ∈ {0.05, 0.10, 0.15} with the
  3-sleeve base rescaled to ``1 - w_composite``. Tests 5 base splits.

**Block OU_MR** (15 configs):
  Same grid with ``ou_mean_reversion`` instead of composite_fx_alpha.

**Block XS_REVISIT** (12 configs):
  Add XS_Momentum at weight ∈ {0.05, 0.10} with 6 base splits.

**Block BASELINE** (3 configs):
  production prod, leverage plateau, weight-sweep top-1.

Grand total: 15 + 15 + 12 + 3 = **45 configs**

All configs fixed at ``tv=0.25 ml=14 DDoff`` (the leverage plateau
sweet spot). Native parallel execution via ``native_parallel_sweep``.

Usage
-----
    python scripts/sweep_fourth_sleeve.py
    python scripts/sweep_fourth_sleeve.py --smoke
    python scripts/sweep_fourth_sleeve.py --no-bootstrap
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


_PLATEAU_TV: float = 0.25
_PLATEAU_ML: float = 14.0
_PLATEAU_DD: bool = False


# ═══════════════════════════════════════════════════════════════════════
# Extra-sleeve loaders
# ═══════════════════════════════════════════════════════════════════════


def compute_composite_fx_alpha_returns() -> pd.Series:
    """Run composite_fx_alpha with default params and return daily returns."""
    from strategies.composite_fx_alpha import pipeline as composite_pipeline
    from utils import load_fx_data

    print("  Running Composite FX Alpha (EUR-USD, defaults)…")
    _, data = load_fx_data("data/EUR-USD_minute.parquet")
    pf, _ = composite_pipeline(data)
    rets = pf.daily_returns
    if hasattr(rets, "ndim") and rets.ndim > 1:
        rets = rets.iloc[:, 0]
    return rets.rename("Composite_FX_Alpha")


def compute_ou_mr_returns() -> pd.Series:
    """Run ou_mean_reversion with default params and return daily returns."""
    from strategies.ou_mean_reversion import pipeline as ou_pipeline
    from utils import load_fx_data

    print("  Running OU Mean Reversion (EUR-USD, defaults)…")
    _, data = load_fx_data("data/EUR-USD_minute.parquet")
    pf, _ = ou_pipeline(data)
    rets = pf.daily_returns
    if hasattr(rets, "ndim") and rets.ndim > 1:
        rets = rets.iloc[:, 0]
    return rets.rename("OU_MR")


# ═══════════════════════════════════════════════════════════════════════
# Grid builder
# ═══════════════════════════════════════════════════════════════════════


# Base 3-sleeve splits. Each tuple is (MR, TS_Momentum_3p, RSI_Daily_4p)
# for the base trio. The grid rescales these by (1 - w_extra) and adds
# the 4th sleeve at w_extra.
_BASE_SPLITS_3: list[tuple[float, float, float]] = [
    (0.80, 0.10, 0.10),  # canonical production
    (0.75, 0.10, 0.15),  # weight-sweep top-1
    (0.80, 0.08, 0.12),  # weight-sweep top-2
    (0.75, 0.15, 0.10),  # TS-heavy
    (0.70, 0.12, 0.18),  # RSI-heavy light-MR
]

_W_EXTRA_CORE: list[float] = [0.05, 0.10, 0.15]
_W_EXTRA_XS: list[float] = [0.05, 0.10]


def _weight_tag_4(mr: float, ts: float, rsi: float, extra: float, label: str) -> str:
    return (
        f"{int(round(mr * 100)):02d}-"
        f"{int(round(ts * 100)):02d}-"
        f"{int(round(rsi * 100)):02d}-"
        f"{label}{int(round(extra * 100)):02d}"
    )


def _make_weights_4(
    mr: float,
    ts: float,
    rsi: float,
    extra: float,
    extra_key: str,
) -> dict[str, float]:
    """Rescale base trio by (1 - extra) and append the 4th sleeve."""
    scale = 1.0 - extra
    weights = {
        "MR_Macro": round(mr * scale, 4),
        "TS_Momentum_3p": round(ts * scale, 4),
        "RSI_Daily_4p": round(rsi * scale, 4),
        extra_key: round(extra, 4),
    }
    # Safety renorm against rounding drift.
    total = sum(weights.values())
    if total > 0:
        weights = {k: round(v / total, 4) for k, v in weights.items()}
    return weights


def build_fourth_sleeve_grid() -> list[SweepConfig]:
    cfgs: list[SweepConfig] = []

    # ─── Block COMPOSITE ────────────────────────────────────────────
    for mr, ts, rsi in _BASE_SPLITS_3:
        for w_extra in _W_EXTRA_CORE:
            w = _make_weights_4(mr, ts, rsi, w_extra, "Composite_FX_Alpha")
            tag = _weight_tag_4(mr, ts, rsi, w_extra, "C")
            cfgs.append(
                SweepConfig(
                    id=f"SLV-c{tag}",
                    block="COMPOSITE",
                    name=(
                        f"+Composite {int(w_extra * 100)}% "
                        f"/ base={int(mr * 100)}-{int(ts * 100)}-{int(rsi * 100)} "
                        f"/ tv=0.25 ml=14 DDoff"
                    ),
                    sleeves=(
                        "MR_Macro",
                        "TS_Momentum_3p",
                        "RSI_Daily_4p",
                        "Composite_FX_Alpha",
                    ),
                    allocation="custom",
                    target_vol=_PLATEAU_TV,
                    max_leverage=_PLATEAU_ML,
                    dd_cap_enabled=_PLATEAU_DD,
                    custom_weights=w,
                )
            )

    # ─── Block OU_MR ────────────────────────────────────────────────
    for mr, ts, rsi in _BASE_SPLITS_3:
        for w_extra in _W_EXTRA_CORE:
            w = _make_weights_4(mr, ts, rsi, w_extra, "OU_MR")
            tag = _weight_tag_4(mr, ts, rsi, w_extra, "O")
            cfgs.append(
                SweepConfig(
                    id=f"SLV-o{tag}",
                    block="OU_MR",
                    name=(
                        f"+OU_MR {int(w_extra * 100)}% "
                        f"/ base={int(mr * 100)}-{int(ts * 100)}-{int(rsi * 100)} "
                        f"/ tv=0.25 ml=14 DDoff"
                    ),
                    sleeves=(
                        "MR_Macro",
                        "TS_Momentum_3p",
                        "RSI_Daily_4p",
                        "OU_MR",
                    ),
                    allocation="custom",
                    target_vol=_PLATEAU_TV,
                    max_leverage=_PLATEAU_ML,
                    dd_cap_enabled=_PLATEAU_DD,
                    custom_weights=w,
                )
            )

    # ─── Block XS_REVISIT — 12 configs ──────────────────────────────
    XS_BASE_SPLITS = [
        (0.80, 0.10, 0.10),
        (0.75, 0.10, 0.15),
        (0.80, 0.08, 0.12),
        (0.75, 0.15, 0.10),
        (0.70, 0.12, 0.18),
        (0.70, 0.15, 0.15),
    ]
    for mr, ts, rsi in XS_BASE_SPLITS:
        for w_extra in _W_EXTRA_XS:
            w = _make_weights_4(mr, ts, rsi, w_extra, "XS_Momentum")
            tag = _weight_tag_4(mr, ts, rsi, w_extra, "X")
            cfgs.append(
                SweepConfig(
                    id=f"SLV-x{tag}",
                    block="XS_REVISIT",
                    name=(
                        f"+XS_Momentum {int(w_extra * 100)}% "
                        f"/ base={int(mr * 100)}-{int(ts * 100)}-{int(rsi * 100)} "
                        f"/ tv=0.25 ml=14 DDoff"
                    ),
                    sleeves=(
                        "MR_Macro",
                        "TS_Momentum_3p",
                        "RSI_Daily_4p",
                        "XS_Momentum",
                    ),
                    allocation="custom",
                    target_vol=_PLATEAU_TV,
                    max_leverage=_PLATEAU_ML,
                    dd_cap_enabled=_PLATEAU_DD,
                    custom_weights=w,
                )
            )

    # ─── Block BASELINE — 3 configs ─────────────────────────────────
    cfgs += [
        SweepConfig(
            id="BL-prod",
            block="BASELINE",
            name="production (MR80/TS10/RSI10 tv=0.28 ml=12 DDoff)",
            sleeves=("MR_Macro", "TS_Momentum_3p", "RSI_Daily_4p"),
            allocation="custom",
            target_vol=0.28,
            max_leverage=12.0,
            dd_cap_enabled=False,
            custom_weights={
                "MR_Macro": 0.80,
                "TS_Momentum_3p": 0.10,
                "RSI_Daily_4p": 0.10,
            },
        ),
        SweepConfig(
            id="BL-plateau",
            block="BASELINE",
            name="leverage plateau (MR80/TS10/RSI10 tv=0.25 ml=14 DDoff)",
            sleeves=("MR_Macro", "TS_Momentum_3p", "RSI_Daily_4p"),
            allocation="custom",
            target_vol=_PLATEAU_TV,
            max_leverage=_PLATEAU_ML,
            dd_cap_enabled=_PLATEAU_DD,
            custom_weights={
                "MR_Macro": 0.80,
                "TS_Momentum_3p": 0.10,
                "RSI_Daily_4p": 0.10,
            },
        ),
        SweepConfig(
            id="BL-weight-top",
            block="BASELINE",
            name="weight-sweep top-1 (MR75/TS10/RSI15 tv=0.25 ml=14 DDoff)",
            sleeves=("MR_Macro", "TS_Momentum_3p", "RSI_Daily_4p"),
            allocation="custom",
            target_vol=_PLATEAU_TV,
            max_leverage=_PLATEAU_ML,
            dd_cap_enabled=_PLATEAU_DD,
            custom_weights={
                "MR_Macro": 0.75,
                "TS_Momentum_3p": 0.10,
                "RSI_Daily_4p": 0.15,
            },
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


def build_fourth_sleeve_markdown(
    rows: list[dict[str, Any]],
    bootstrap_by_id: dict[str, dict[str, float]],
    report_date: str,
    correlations: dict[str, float],
) -> str:
    sorted_rows = sorted(rows, key=lambda r: r["wf_avg_sharpe"], reverse=True)
    top10 = sorted_rows[:10]
    plateau_row = next((r for r in rows if r["id"] == "BL-plateau"), None)
    weight_top_row = next((r for r in rows if r["id"] == "BL-weight-top"), None)

    lines: list[str] = []
    lines.append(f"# Fourth-sleeve evaluation sweep ({report_date})\n")
    lines.append(
        "Follow-up to the weight sweep which showed that recombining the "
        "3-sleeve production trio weights only moves the WF Sharpe by a "
        "few basis points above the leverage plateau (0.966). This sweep "
        "tests whether adding a fourth orthogonal sleeve — Composite FX "
        "Alpha, OU Mean Reversion, or a restored XS Momentum — can push "
        "the Sharpe above 0.97.\n"
    )
    lines.append(
        "Fixed parameters : `tv=0.25 / ml=14 / DDoff` (leverage plateau "
        "sweet-spot). Each config takes a base trio split and reallocates "
        "a fraction ∈ {5%, 10%, 15%} to the extra sleeve.\n"
    )

    # ── Correlations vs the base trio ──────────────────────────────
    if correlations:
        lines.append("## Extra sleeve correlations vs the production trio\n")
        lines.append(
            "| Extra sleeve | Corr with MR_Macro | Corr with TS_3p | Corr with RSI_4p |"
        )
        lines.append(
            "|-------------|-------------------:|----------------:|-----------------:|"
        )
        for k, v in correlations.items():
            if isinstance(v, dict):
                lines.append(
                    f"| {k} | {v['MR_Macro']:+.3f} "
                    f"| {v['TS_Momentum_3p']:+.3f} "
                    f"| {v['RSI_Daily_4p']:+.3f} |"
                )
        lines.append("")

    if plateau_row:
        lines.append(
            f"Leverage plateau baseline : **{plateau_row['id']}** — Sharpe WF "
            f"{plateau_row['wf_avg_sharpe']:.3f}, CAGR "
            f"{plateau_row['annual_return'] * 100:.2f}%, "
            f"MaxDD {plateau_row['max_drawdown'] * 100:.2f}%.\n"
        )
    if weight_top_row:
        lines.append(
            f"Weight-sweep top-1 baseline : **{weight_top_row['id']}** — Sharpe WF "
            f"{weight_top_row['wf_avg_sharpe']:.3f}, CAGR "
            f"{weight_top_row['annual_return'] * 100:.2f}%, "
            f"MaxDD {weight_top_row['max_drawdown'] * 100:.2f}%.\n"
        )

    # ── Top-10 ─────────────────────────────────────────────────────
    lines.append("## Top 10 by Walk-Forward Sharpe\n")
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
    for block in ("COMPOSITE", "OU_MR", "XS_REVISIT", "BASELINE"):
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
        plateau_sharpe = plateau_row["wf_avg_sharpe"] if plateau_row else 0.966
        weight_top_sharpe = weight_top_row["wf_avg_sharpe"] if weight_top_row else 0.972
        gap_plateau = top["wf_avg_sharpe"] - plateau_sharpe
        gap_weight_top = top["wf_avg_sharpe"] - weight_top_sharpe
        sign_p = "+" if gap_plateau >= 0 else ""
        sign_a = "+" if gap_weight_top >= 0 else ""
        lines.append(
            f"Best point in the sweep : **`{top['id']}`** — {top['name']}.  \n"
            f"Sharpe WF = **{top['wf_avg_sharpe']:.3f}** "
            f"(vs leverage plateau = {plateau_sharpe:.3f}, "
            f"Δ = {sign_p}{gap_plateau:.3f} ; "
            f"vs weight-sweep top = {weight_top_sharpe:.3f}, "
            f"Δ = {sign_a}{gap_weight_top:.3f})."
        )
        if gap_weight_top > 0.005:
            lines.append(
                "\nAdding a fourth sleeve **materially improves** the Sharpe "
                "beyond what weight recombination could achieve. Validate "
                "on bootstrap before promoting."
            )
        elif abs(gap_weight_top) <= 0.005:
            lines.append(
                "\nAdding a fourth sleeve **does not exceed** the weight-sweep "
                "top meaningfully. The 3-sleeve production trio + leverage "
                "layer remains the recommended configuration."
            )
        else:
            lines.append(
                "\nNo fourth sleeve tested exceeds the weight-sweep top ; "
                "adding a non-orthogonal sleeve degrades the ratio by "
                "diluting the MR_Macro alpha."
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
        output_root = Path("/tmp") / f"fourth_sleeve_smoke_{report_date}"
        md_path = output_root / "fourth_sleeve.md"
    else:
        output_root = _PROJECT_ROOT / "results" / f"fourth_sleeve_{report_date}"
        md_path = (
            _PROJECT_ROOT / "docs" / "research" / f"fourth_sleeve_{report_date}.md"
        )
    output_root.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("  Fourth-sleeve evaluation sweep")
    print("=" * 72)
    print(f"Report date : {report_date}")
    print(f"Output root : {output_root}")
    print(f"Markdown    : {md_path}")
    print()

    print("Loading base 3 sleeves (production trio)…")
    sleeves = get_strategy_daily_returns()
    base_keys = ("MR_Macro", "TS_Momentum_3p", "RSI_Daily_4p", "XS_Momentum")
    sleeves_all: dict[str, pd.Series] = {k: sleeves[k] for k in base_keys}

    print("Loading extra sleeves…")
    sleeves_all["Composite_FX_Alpha"] = compute_composite_fx_alpha_returns()
    sleeves_all["OU_MR"] = compute_ou_mr_returns()

    print(f"  Loaded {len(sleeves_all)} sleeves: {sorted(sleeves_all)}")

    # ── Compute correlations of extra sleeves vs the production trio
    common_idx = None
    for s in sleeves_all.values():
        idx = s.dropna().index
        common_idx = idx if common_idx is None else common_idx.intersection(idx)
    correlations: dict[str, Any] = {}
    for extra in ("Composite_FX_Alpha", "OU_MR", "XS_Momentum"):
        corr_row: dict[str, float] = {}
        for base in ("MR_Macro", "TS_Momentum_3p", "RSI_Daily_4p"):
            a = sleeves_all[extra].loc[common_idx]
            b = sleeves_all[base].loc[common_idx]
            corr_row[base] = float(a.corr(b))
        correlations[extra] = corr_row
        print(
            f"  corr({extra:<20}) vs trio : "
            f"MR={corr_row['MR_Macro']:+.3f}  "
            f"TS={corr_row['TS_Momentum_3p']:+.3f}  "
            f"RSI={corr_row['RSI_Daily_4p']:+.3f}"
        )
    print()

    grid = build_fourth_sleeve_grid()
    if args.smoke:
        smoke_ids = {
            "SLV-c80-10-10-C10",
            "SLV-o80-10-10-O10",
            "SLV-x80-10-10-X10",
            "BL-weight-top",
        }
        grid = [c for c in grid if c.id in smoke_ids]
    print(f"Running {len(grid)} configurations via native parallel sweep…")

    t0 = time.perf_counter()
    pf_all, metrics = native_parallel_sweep(grid, sleeves_all)
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
            f"  [{r['block']:>9}] {r['id']:<28} "
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
                stats = bootstrap_config(cfg, sleeves_all, n_runs=args.bootstrap_runs)
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
        "sleeves_loaded": sorted(sleeves_all),
        "correlations": correlations,
        "configs": [sanitize_result_for_json(r) for r in sorted_rows],
        "bootstrap_top5": bootstrap_by_id,
    }
    json_path.write_text(json.dumps(json_data, indent=2, default=str))
    print(f"\nJSON exported → {json_path}")

    # ── Export markdown ────────────────────────────────────────────
    md = build_fourth_sleeve_markdown(rows, bootstrap_by_id, report_date, correlations)
    md_path.write_text(md)
    print(f"Markdown exported → {md_path}")

    # ── Top-N artifacts ────────────────────────────────────────────
    if not args.smoke:
        print("\nGenerating top-N artifacts (tearsheets + mix plots)…")
        generate_top_n_artifacts(sorted_rows, grid, sleeves_all, output_root)

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
    weight_top = next((r for r in rows if r["id"] == "BL-weight-top"), None)
    if weight_top:
        rank = next(
            i for i, r in enumerate(sorted_rows, 1) if r["id"] == "BL-weight-top"
        )
        print(
            f"\n  Weight-sweep top-1 baseline: rank #{rank}, "
            f"sharpe={weight_top['wf_avg_sharpe']:+.3f}"
        )


if __name__ == "__main__":
    main()
