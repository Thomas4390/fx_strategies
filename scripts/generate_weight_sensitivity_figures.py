"""Weight sensitivity figures and tables for the LaTeX report appendix.

Sweeps the allocation (w_mr, w_ts, w_rsi) across:
  1. 8 named allocations (production, pure MR, equal, risk-parity, ...)
  2. A 1D line sweep on w_mr with w_ts = w_rsi
  3. A 2D simplex grid (231 valid points, step 0.05)

Risk layers are pinned to the production config (target_vol=0.28,
max_leverage=12, dd_cap_enabled=False) so only the weight effect is
isolated.

Outputs:
  reports/latex_report/figures/weight_sensitivity_named_bars.png
  reports/latex_report/figures/weight_sensitivity_1d_mr.png
  reports/latex_report/figures/weight_sensitivity_simplex_ternary.png
  reports/latex_report/figures/weight_sensitivity_pareto_frontier.png
  reports/latex_report/tables/weight_sensitivity_named.tex
  reports/latex_report/tables/weight_sensitivity_extremes.tex

Usage:
    python scripts/generate_weight_sensitivity_figures.py             # full
    python scripts/generate_weight_sensitivity_figures.py --smoke     # fast
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SRC = _PROJECT_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

FIG_DIR = _PROJECT_ROOT / "reports" / "latex_report" / "figures"
TBL_DIR = _PROJECT_ROOT / "reports" / "latex_report" / "tables"
FIG_DIR.mkdir(parents=True, exist_ok=True)
TBL_DIR.mkdir(parents=True, exist_ok=True)

# Production risk layer — fixed across the entire sweep.
TARGET_VOL: float = 0.28
MAX_LEVERAGE: float = 12.0

# Sleeves in canonical order.
SLEEVE_KEYS: tuple[str, str, str] = ("MR_Macro", "TS_Momentum_3p", "RSI_Daily_4p")
SLEEVE_LABELS: dict[str, str] = {
    "MR_Macro": "MR",
    "TS_Momentum_3p": "TS",
    "RSI_Daily_4p": "RSI",
}

# Palette aligned with build_latex_report_assets.py.
PALETTE = {
    "primary": "#0B2545",
    "accent": "#CC6B2F",
    "mr": "#1F5582",
    "ts": "#CC6B2F",
    "rsi": "#2E8B57",
    "combined": "#8E1616",
    "bg_grid": "#E6E8EB",
    "text": "#1A1A1A",
    "gold": "#B08D3C",
}

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "axes.edgecolor": "#4A4A4A",
        "axes.labelcolor": PALETTE["text"],
        "xtick.color": "#4A4A4A",
        "ytick.color": "#4A4A4A",
        "axes.grid": True,
        "grid.color": PALETTE["bg_grid"],
        "grid.linewidth": 0.6,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "legend.frameon": False,
        "legend.fontsize": 9,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
    }
)


# ═══════════════════════════════════════════════════════════════════════
# Data containers
# ═══════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class WeightPoint:
    """One allocation plus its computed metrics."""

    label: str
    w_mr: float
    w_ts: float
    w_rsi: float
    sharpe: float
    cagr: float
    vol: float
    max_dd: float  # negative number
    calmar: float


def make_weights(w_mr: float, w_ts: float, w_rsi: float) -> dict[str, float]:
    total = w_mr + w_ts + w_rsi
    if total <= 0:
        raise ValueError("Weights must sum to a positive number.")
    return {
        "MR_Macro": w_mr / total,
        "TS_Momentum_3p": w_ts / total,
        "RSI_Daily_4p": w_rsi / total,
    }


# ═══════════════════════════════════════════════════════════════════════
# Backtest driver
# ═══════════════════════════════════════════════════════════════════════


def compute_metrics(
    sleeves: dict[str, pd.Series],
    weights: dict[str, float],
    label: str,
    build_fn,
) -> WeightPoint:
    """Run one v2 portfolio and extract the 5 metrics."""
    result = build_fn(
        {k: sleeves[k] for k in SLEEVE_KEYS},
        allocation="custom",
        custom_weights=weights,
        target_vol=TARGET_VOL,
        max_leverage=MAX_LEVERAGE,
        dd_cap_enabled=False,
    )
    sharpe = float(result["wf_avg_sharpe"])
    cagr = float(result["annual_return"])
    vol = float(result["annual_vol"])
    max_dd = float(result["max_drawdown"])
    calmar = cagr / abs(max_dd) if max_dd != 0 else np.nan
    return WeightPoint(
        label=label,
        w_mr=weights["MR_Macro"],
        w_ts=weights["TS_Momentum_3p"],
        w_rsi=weights["RSI_Daily_4p"],
        sharpe=sharpe,
        cagr=cagr,
        vol=vol,
        max_dd=max_dd,
        calmar=calmar,
    )


def run_named_allocations(
    sleeves: dict[str, pd.Series],
    build_fn,
) -> list[WeightPoint]:
    """Run the 8 named allocations — production, extremes, equal, risk-parity."""
    named = [
        ("80 / 10 / 10 (production)", 0.80, 0.10, 0.10),
        ("100 / 0 / 0 (pur MR)", 1.00, 0.00, 0.00),
        ("0 / 50 / 50 (sans MR)", 0.00, 0.50, 0.50),
        ("70 / 15 / 15", 0.70, 0.15, 0.15),
        ("60 / 20 / 20", 0.60, 0.20, 0.20),
        ("50 / 25 / 25", 0.50, 0.25, 0.25),
        ("33 / 33 / 33 (équipondéré)", 1 / 3, 1 / 3, 1 / 3),
    ]
    points: list[WeightPoint] = []
    for label, w_mr, w_ts, w_rsi in named:
        print(f"  • {label}…", end="", flush=True)
        t0 = time.perf_counter()
        pt = compute_metrics(
            sleeves, make_weights(w_mr, w_ts, w_rsi), label, build_fn
        )
        print(f" Sharpe={pt.sharpe:+.3f}  ({time.perf_counter() - t0:.1f}s)")
        points.append(pt)

    # Risk-parity is allocation="risk_parity", no custom_weights — use a
    # dedicated call that delegates to _compute_weights_ts.
    print("  • risk-parity (inverse-vol)…", end="", flush=True)
    t0 = time.perf_counter()
    result = build_fn(
        {k: sleeves[k] for k in SLEEVE_KEYS},
        allocation="risk_parity",
        target_vol=TARGET_VOL,
        max_leverage=MAX_LEVERAGE,
        dd_cap_enabled=False,
    )
    avg_w = result["weights_ts"].mean()
    rp_point = WeightPoint(
        label="risk-parity",
        w_mr=float(avg_w.get("MR_Macro", 0.0)),
        w_ts=float(avg_w.get("TS_Momentum_3p", 0.0)),
        w_rsi=float(avg_w.get("RSI_Daily_4p", 0.0)),
        sharpe=float(result["wf_avg_sharpe"]),
        cagr=float(result["annual_return"]),
        vol=float(result["annual_vol"]),
        max_dd=float(result["max_drawdown"]),
        calmar=float(result["annual_return"]) / abs(float(result["max_drawdown"])),
    )
    print(f" Sharpe={rp_point.sharpe:+.3f}  ({time.perf_counter() - t0:.1f}s)")
    points.append(rp_point)
    return points


def run_1d_sweep(
    sleeves: dict[str, pd.Series],
    build_fn,
    mr_values: np.ndarray,
) -> list[WeightPoint]:
    """Sweep w_mr with w_ts = w_rsi = (1 - w_mr)/2."""
    points: list[WeightPoint] = []
    for w_mr in mr_values:
        w_div = (1.0 - w_mr) / 2.0
        if w_mr == 0.0 and w_div == 0.0:
            continue
        weights = make_weights(w_mr, w_div, w_div)
        label = f"mr={w_mr:.2f}"
        pt = compute_metrics(sleeves, weights, label, build_fn)
        points.append(pt)
    return points


def run_simplex_grid(
    sleeves: dict[str, pd.Series],
    build_fn,
    step: float = 0.05,
) -> list[WeightPoint]:
    """Sweep the full 2-simplex with a regular grid of step `step`."""
    n = int(round(1.0 / step)) + 1
    grid_1d = np.linspace(0.0, 1.0, n)
    points: list[WeightPoint] = []
    total = 0
    for w_mr in grid_1d:
        for w_ts in grid_1d:
            w_rsi = 1.0 - w_mr - w_ts
            if w_rsi < -1e-9:
                continue
            w_rsi = max(w_rsi, 0.0)
            if w_mr + w_ts + w_rsi < 1e-9:
                continue
            total += 1
    count = 0
    t0 = time.perf_counter()
    for w_mr in grid_1d:
        for w_ts in grid_1d:
            w_rsi = 1.0 - w_mr - w_ts
            if w_rsi < -1e-9:
                continue
            w_rsi = max(w_rsi, 0.0)
            if w_mr + w_ts + w_rsi < 1e-9:
                continue
            count += 1
            weights = make_weights(w_mr, w_ts, w_rsi)
            pt = compute_metrics(
                sleeves, weights, f"s_{count:04d}", build_fn
            )
            points.append(pt)
            if count % 25 == 0 or count == total:
                elapsed = time.perf_counter() - t0
                print(
                    f"    simplex {count:3d}/{total}  "
                    f"({elapsed:.1f}s, {elapsed / count:.2f}s/pt)"
                )
    return points


# ═══════════════════════════════════════════════════════════════════════
# Plot helpers
# ═══════════════════════════════════════════════════════════════════════


def save_fig(fig, name: str) -> None:
    path = FIG_DIR / f"{name}.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    size_kb = path.stat().st_size / 1024
    print(f"  ✓ {path.relative_to(_PROJECT_ROOT)}  ({size_kb:.0f} KB)")


def plot_named_bars(points: list[WeightPoint]) -> None:
    """Grouped bar chart: Sharpe, CAGR, |MaxDD| per allocation."""
    labels = [p.label for p in points]
    sharpes = np.array([p.sharpe for p in points])
    cagrs = np.array([p.cagr * 100.0 for p in points])
    dds = np.array([abs(p.max_dd) * 100.0 for p in points])

    fig, (ax_sr, ax_cagr, ax_dd) = plt.subplots(3, 1, figsize=(10.5, 9.0), sharex=True)

    # Highlight production bar.
    colors = [
        PALETTE["accent"] if "production" in lab else PALETTE["primary"]
        for lab in labels
    ]

    x = np.arange(len(labels))
    ax_sr.bar(x, sharpes, color=colors, edgecolor="white")
    ax_sr.set_ylabel("Sharpe WF")
    ax_sr.set_title("Sharpe walk-forward", loc="left")
    for xi, v in zip(x, sharpes):
        ax_sr.text(xi, v + 0.02, f"{v:+.2f}", ha="center", fontsize=8)
    ax_sr.axhline(0, color="#4A4A4A", linewidth=0.6)
    ax_sr.set_ylim(min(sharpes.min() - 0.1, -0.1), sharpes.max() + 0.15)

    ax_cagr.bar(x, cagrs, color=colors, edgecolor="white")
    ax_cagr.set_ylabel("CAGR (%)")
    ax_cagr.set_title("CAGR annualisé", loc="left")
    for xi, v in zip(x, cagrs):
        ax_cagr.text(xi, v + 0.3, f"{v:+.1f}%", ha="center", fontsize=8)
    ax_cagr.axhline(0, color="#4A4A4A", linewidth=0.6)
    ax_cagr.set_ylim(min(cagrs.min() - 1.5, -1.5), cagrs.max() + 2.5)

    ax_dd.bar(x, dds, color=colors, edgecolor="white")
    ax_dd.set_ylabel("|Max DD| (%)")
    ax_dd.set_title("Max Drawdown (valeur absolue)", loc="left")
    for xi, v in zip(x, dds):
        ax_dd.text(xi, v + 0.4, f"{v:.1f}%", ha="center", fontsize=8)
    ax_dd.set_xticks(x)
    ax_dd.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
    ax_dd.set_ylim(0, dds.max() + 5)

    fig.suptitle(
        "Sensibilité à l'allocation — comparaison d'allocations nommées",
        fontsize=13,
        fontweight="bold",
        color=PALETTE["primary"],
    )
    fig.tight_layout()
    save_fig(fig, "weight_sensitivity_named_bars")


def plot_1d_sweep(points: list[WeightPoint]) -> None:
    """2x2 panel: Sharpe, CAGR, Max DD, Calmar vs w_mr."""
    mr = np.array([p.w_mr for p in points])
    sharpe = np.array([p.sharpe for p in points])
    cagr = np.array([p.cagr * 100 for p in points])
    dd = np.array([p.max_dd * 100 for p in points])
    calmar = np.array([p.calmar for p in points])

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.0))

    def _panel(ax, y, title, ylabel, color, is_pct=True, prod_label=True):
        ax.plot(mr * 100, y, color=color, linewidth=2.0, marker="o", markersize=4)
        ax.axvline(80, color=PALETTE["accent"], linestyle="--", linewidth=1.2)
        ax.set_title(title, loc="left")
        ax.set_xlabel("Poids MR Macro (%)")
        ax.set_ylabel(ylabel)
        ax.set_xlim(-2, 102)
        # Mark the production point.
        prod_idx = int(np.argmin(np.abs(mr - 0.80)))
        ax.plot(
            mr[prod_idx] * 100,
            y[prod_idx],
            marker="*",
            markersize=16,
            color=PALETTE["accent"],
            markeredgecolor=PALETTE["primary"],
            markeredgewidth=0.8,
            zorder=10,
            label="80/10/10 (prod)" if prod_label else None,
        )
        if prod_label:
            ax.legend(loc="best")

    _panel(axes[0, 0], sharpe, "Sharpe WF", "Sharpe", PALETTE["primary"], is_pct=False)
    _panel(axes[0, 1], cagr, "CAGR", "CAGR (%)", PALETTE["mr"], prod_label=False)
    _panel(
        axes[1, 0], dd, "Max Drawdown", "Max DD (%)", PALETTE["combined"], prod_label=False
    )
    _panel(
        axes[1, 1],
        calmar,
        "Calmar = CAGR / |Max DD|",
        "Calmar",
        PALETTE["rsi"],
        is_pct=False,
        prod_label=False,
    )

    fig.suptitle(
        "Balayage 1D — part MR de 0 % à 100 % (TS = RSI = (1 − MR)/2)",
        fontsize=13,
        fontweight="bold",
        color=PALETTE["primary"],
    )
    fig.tight_layout()
    save_fig(fig, "weight_sensitivity_1d_mr")


def _barycentric_to_cartesian(
    w_mr: np.ndarray, w_ts: np.ndarray, w_rsi: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Project (w_mr, w_ts, w_rsi) from the 2-simplex to a flat triangle.

    Vertices: MR=(0,0), TS=(1,0), RSI=(0.5, sqrt(3)/2).
    """
    x = w_ts + 0.5 * w_rsi
    y = (np.sqrt(3) / 2.0) * w_rsi
    return x, y


def plot_simplex_ternary(points: list[WeightPoint]) -> None:
    """Triangular heatmap of Sharpe on the simplex."""
    w_mr = np.array([p.w_mr for p in points])
    w_ts = np.array([p.w_ts for p in points])
    w_rsi = np.array([p.w_rsi for p in points])
    sharpe = np.array([p.sharpe for p in points])

    x, y = _barycentric_to_cartesian(w_mr, w_ts, w_rsi)

    fig, ax = plt.subplots(figsize=(9.5, 8.0))

    # Triangulate and fill.
    tri = mtri.Triangulation(x, y)
    levels = np.linspace(max(-0.2, sharpe.min() - 0.05), sharpe.max() + 0.02, 18)
    cf = ax.tricontourf(tri, sharpe, levels=levels, cmap="viridis")
    ax.tricontour(
        tri,
        sharpe,
        levels=[0.0, 0.3, 0.6, 0.8, 0.9, 0.95],
        colors="white",
        linewidths=0.8,
        alpha=0.85,
    )

    # Triangle outline.
    tri_outline = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.5, np.sqrt(3) / 2.0],
            [0.0, 0.0],
        ]
    )
    ax.plot(tri_outline[:, 0], tri_outline[:, 1], color=PALETTE["primary"], linewidth=1.5)

    # Vertex labels.
    ax.text(-0.04, -0.04, "MR 100%", fontsize=11, fontweight="bold", ha="right")
    ax.text(1.04, -0.04, "TS 100%", fontsize=11, fontweight="bold", ha="left")
    ax.text(
        0.5,
        np.sqrt(3) / 2.0 + 0.035,
        "RSI 100%",
        fontsize=11,
        fontweight="bold",
        ha="center",
    )

    # Mark the production point 80/10/10.
    px, py = _barycentric_to_cartesian(
        np.array([0.80]), np.array([0.10]), np.array([0.10])
    )
    ax.plot(
        px,
        py,
        marker="*",
        markersize=22,
        color=PALETTE["accent"],
        markeredgecolor="white",
        markeredgewidth=1.5,
        zorder=10,
    )
    ax.annotate(
        "80 / 10 / 10",
        xy=(px[0], py[0]),
        xytext=(px[0] + 0.10, py[0] + 0.07),
        fontsize=10,
        fontweight="bold",
        color=PALETTE["primary"],
        arrowprops=dict(arrowstyle="-", color=PALETTE["primary"], linewidth=0.8),
    )

    ax.set_aspect("equal")
    ax.set_xlim(-0.18, 1.18)
    ax.set_ylim(-0.12, np.sqrt(3) / 2.0 + 0.12)
    ax.axis("off")

    cbar = fig.colorbar(cf, ax=ax, shrink=0.75, pad=0.02)
    cbar.set_label("Sharpe WF", fontsize=10)
    cbar.ax.tick_params(labelsize=8)

    ax.set_title(
        "Carte de chaleur ternaire — Sharpe WF sur le simplex des poids",
        fontsize=13,
        fontweight="bold",
        color=PALETTE["primary"],
        loc="center",
        pad=12,
    )

    fig.tight_layout()
    save_fig(fig, "weight_sensitivity_simplex_ternary")


def plot_pareto_frontier(points: list[WeightPoint]) -> None:
    """Scatter Vol × CAGR with Sharpe as color, Pareto envelope overlaid."""
    vol = np.array([p.vol * 100 for p in points])
    cagr = np.array([p.cagr * 100 for p in points])
    sharpe = np.array([p.sharpe for p in points])

    fig, ax = plt.subplots(figsize=(10.0, 7.0))
    sc = ax.scatter(
        vol,
        cagr,
        c=sharpe,
        cmap="viridis",
        s=40,
        edgecolor="white",
        linewidth=0.3,
    )

    # Pareto envelope: sort by vol, keep running max of cagr.
    order = np.argsort(vol)
    vol_s, cagr_s = vol[order], cagr[order]
    running_max = np.maximum.accumulate(cagr_s)
    mask = cagr_s == running_max
    ax.plot(
        vol_s[mask],
        cagr_s[mask],
        color=PALETTE["combined"],
        linewidth=2.0,
        linestyle="--",
        label="Frontière efficiente empirique",
    )

    # Production star.
    prod_idx = int(
        np.argmin(
            (np.abs([p.w_mr - 0.80 for p in points]))
            + (np.abs([p.w_ts - 0.10 for p in points]))
            + (np.abs([p.w_rsi - 0.10 for p in points]))
        )
    )
    ax.plot(
        vol[prod_idx],
        cagr[prod_idx],
        marker="*",
        markersize=22,
        color=PALETTE["accent"],
        markeredgecolor=PALETTE["primary"],
        markeredgewidth=1.2,
        zorder=10,
        label="80 / 10 / 10 (production)",
    )

    ax.set_xlabel("Volatilité annualisée (%)")
    ax.set_ylabel("CAGR (%)")
    ax.set_title(
        "Frontière efficiente empirique — 231 allocations du simplex",
        fontsize=13,
        fontweight="bold",
        color=PALETTE["primary"],
        loc="left",
    )
    ax.legend(loc="lower right")

    cbar = fig.colorbar(sc, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label("Sharpe WF", fontsize=10)
    cbar.ax.tick_params(labelsize=8)

    fig.tight_layout()
    save_fig(fig, "weight_sensitivity_pareto_frontier")


# ═══════════════════════════════════════════════════════════════════════
# Table writers
# ═══════════════════════════════════════════════════════════════════════


def save_tex(name: str, content: str) -> None:
    path = TBL_DIR / f"{name}.tex"
    path.write_text(content)
    print(f"  ✓ {path.relative_to(_PROJECT_ROOT)}  ({len(content)} chars)")


def _fmt_pct(x: float, digits: int = 2) -> str:
    if pd.isna(x):
        return "--"
    return f"{x * 100:.{digits}f}\\,\\%"


def _fmt_num(x: float, digits: int = 2) -> str:
    if pd.isna(x):
        return "--"
    return f"{x:.{digits}f}"


def _escape(s: str) -> str:
    return s.replace("_", "\\_").replace("%", "\\%").replace("&", "\\&")


def write_named_table(points: list[WeightPoint]) -> None:
    """8-row table: label + 5 metrics."""
    rows = []
    for p in points:
        rows.append(
            f"{_escape(p.label)} & "
            f"{_fmt_num(p.sharpe, 3)} & "
            f"{_fmt_pct(p.cagr)} & "
            f"{_fmt_pct(p.vol)} & "
            f"{_fmt_pct(p.max_dd)} & "
            f"{_fmt_num(p.calmar)} \\\\"
        )
    body = "\n".join(rows)
    tex = (
        "\\begin{table}[H]\n"
        "\\centering\n"
        "\\caption{Comparaison de 8 allocations nommées sous le régime de risque "
        "production ($\\mathrm{tv}=0.28$, $\\mathrm{ml}=12$, DD-cap désactivé). "
        "Sharpe WF = moyenne sur les 7~fenêtres annuelles 2019--2025.}\n"
        "\\label{tbl:weight_sensitivity_named}\n"
        "\\begin{tabular}{lrrrrr}\n"
        "\\toprule\n"
        "Allocation & Sharpe WF & CAGR & Volatilité & Max DD & Calmar \\\\\n"
        "\\midrule\n"
        f"{body}\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table}\n"
    )
    save_tex("weight_sensitivity_named", tex)


def write_extremes_table(points: list[WeightPoint]) -> None:
    """Top-3 and bottom-3 allocations from the simplex grid."""
    sorted_pts = sorted(points, key=lambda p: p.sharpe, reverse=True)
    top3 = sorted_pts[:3]
    bot3 = sorted_pts[-3:]

    def _row(p: WeightPoint) -> str:
        alloc = (
            f"{int(round(p.w_mr * 100))}/{int(round(p.w_ts * 100))}/"
            f"{int(round(p.w_rsi * 100))}"
        )
        return (
            f"{alloc} & "
            f"{_fmt_num(p.sharpe, 3)} & "
            f"{_fmt_pct(p.cagr)} & "
            f"{_fmt_pct(p.vol)} & "
            f"{_fmt_pct(p.max_dd)} & "
            f"{_fmt_num(p.calmar)} \\\\"
        )

    top_rows = "\n".join(_row(p) for p in top3)
    bot_rows = "\n".join(_row(p) for p in bot3)

    tex = (
        "\\begin{table}[H]\n"
        "\\centering\n"
        "\\caption{Top-3 et bottom-3 des 231 allocations du simplex "
        "(pas de 0.05), classées par Sharpe WF. Le top-3 est serré "
        "autour de la zone MR-lourde ; le bottom-3 correspond aux "
        "allocations sans exposition MR.}\n"
        "\\label{tbl:weight_sensitivity_extremes}\n"
        "\\begin{tabular}{lrrrrr}\n"
        "\\toprule\n"
        "Allocation MR/TS/RSI & Sharpe WF & CAGR & Volatilité & Max DD & Calmar \\\\\n"
        "\\midrule\n"
        "\\multicolumn{6}{l}{\\textbf{Top 3}} \\\\\n"
        f"{top_rows}\n"
        "\\midrule\n"
        "\\multicolumn{6}{l}{\\textbf{Bottom 3}} \\\\\n"
        f"{bot_rows}\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table}\n"
    )
    save_tex("weight_sensitivity_extremes", tex)


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Reduced grid (4 named + 5 1D + coarse simplex) for quick tests",
    )
    args = parser.parse_args()

    print("═" * 72)
    print("  Weight sensitivity — figure and table generator")
    print("═" * 72)

    import vectorbtpro as vbt

    from strategies.combined_portfolio import get_strategy_daily_returns
    from strategies.combined_portfolio_v2 import build_combined_portfolio_v2
    from utils import apply_vbt_settings

    apply_vbt_settings()
    vbt.settings.returns.year_freq = pd.Timedelta(days=252)

    print("\n[1/5] Loading cached sleeves…")
    t0 = time.perf_counter()
    sleeves_all = get_strategy_daily_returns()
    sleeves = {k: sleeves_all[k] for k in SLEEVE_KEYS}
    print(
        f"  Loaded {len(sleeves)} sleeves in {time.perf_counter() - t0:.1f}s: "
        f"{sorted(sleeves)}"
    )

    print("\n[2/5] Running named allocations…")
    named_pts = run_named_allocations(sleeves, build_combined_portfolio_v2)

    print("\n[3/5] Running 1D MR sweep…")
    if args.smoke:
        mr_vals = np.array([0.0, 0.25, 0.50, 0.80, 1.0])
    else:
        mr_vals = np.linspace(0.0, 1.0, 21)
    sweep_pts = run_1d_sweep(sleeves, build_combined_portfolio_v2, mr_vals)
    print(f"  {len(sweep_pts)} points computed.")

    print("\n[4/5] Running 2D simplex grid…")
    step = 0.20 if args.smoke else 0.05
    simplex_pts = run_simplex_grid(sleeves, build_combined_portfolio_v2, step=step)
    print(f"  {len(simplex_pts)} simplex points computed.")

    print("\n[5/5] Rendering figures and tables…")
    plot_named_bars(named_pts)
    plot_1d_sweep(sweep_pts)
    plot_simplex_ternary(simplex_pts)
    plot_pareto_frontier(simplex_pts)
    write_named_table(named_pts)
    write_extremes_table(simplex_pts)

    print("\nDone.")


if __name__ == "__main__":
    main()
