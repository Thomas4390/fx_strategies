#!/usr/bin/env python3
"""Figures pédagogiques du dimensionnement progressif de la sleeve or.

Trois figures, toutes sur données réelles XAU-USD (``data/XAU-USD_minute_qc.parquet``)
et sur la séquence de transactions réellement produite par le signal de production :

1. ``gold_sizing_martingale.png`` — la martingale sur la plus longue série de
   pertes réellement observée : taille exigée à chaque transaction, contre le
   dimensionnement plat.
2. ``gold_sizing_grid.png`` — la grille sur une position réelle qui part contre :
   paliers d'ajout, prix de revient moyen, exposition cumulée.
3. ``gold_sizing_outcome.png`` — équité comparée des quatre régimes à risque égal,
   sur la fenêtre de sélection puis sur la fenêtre aveugle, qui montre l'inversion
   du classement.

Les règles reproduites sont celles de ``framework.sizing_nb`` :
    martingale : size = base * mult ** min(loss_streak, n_max)
    grille     : ajout de base * grid_mult**(level+1) dès que l'excursion adverse
                 dépasse (level+1) * grid_k * ATR / prix d'ancrage

Usage :
    python scripts/build_gold_sizing_figures.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import vectorbtpro as vbt  # noqa: E402

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SRC = _PROJECT_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from framework.sizing_nb import (  # noqa: E402
    MODE_ANTI_MART,
    MODE_FLAT,
    MODE_GRID,
    MODE_MARTINGALE,
    build_overlay_kwargs,
    make_params,
)
from strategies.gold_momentum import pipeline, session_dates  # noqa: E402
from utils import load_gold_data  # noqa: E402

FIG_DIR = _PROJECT_ROOT / "reports" / "client" / "rapport_technique" / "figures"

HOLDOUT_START = pd.Timestamp("2025-07-01")
INIT_CASH = 100_000.0
TARGET_VOL = 0.25
SLIPPAGE = 0.0001
ATR_WINDOW = 14

PALETTE = {
    "flat": "#0B2545",
    "martingale": "#8E1616",
    "grid": "#CC6B2F",
    "anti": "#2E8B57",
    "price": "#4A4A4A",
    "gold": "#B08D3C",
    "grid_bg": "#E6E8EB",
    "text": "#1A1A1A",
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
        "grid.color": PALETTE["grid_bg"],
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


def save_fig(fig, name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / f"{name}.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {path.relative_to(_PROJECT_ROOT)}  ({path.stat().st_size / 1024:.0f} KB)")


# ────────────────────────────────────────────────────────────────────────
# Données — mêmes bornes de séance que la sleeve (clôture 17h New York)
# ────────────────────────────────────────────────────────────────────────
def load_daily() -> tuple[pd.DataFrame, np.ndarray]:
    raw, _ = load_gold_data()
    sessions = session_dates(raw.index)
    daily = pd.DataFrame(
        {
            "high": raw.high.groupby(sessions).max(),
            "low": raw.low.groupby(sessions).min(),
            "close": raw.close.groupby(sessions).last(),
        }
    ).dropna()
    atr = (
        vbt.ATR.run(daily.high, daily.low, daily.close, window=ATR_WINDOW)
        .atr.bfill()
        .to_numpy()
        .reshape(-1, 1)
    )
    return daily, atr


def run_regime(daily: pd.DataFrame, atr: np.ndarray, mode: int, **params):
    """Simule un régime sur l'historique complet ; renvoie le portefeuille vbt."""
    memory: dict = {}
    p = make_params(mode, base_size=1.0, max_total=4.0, **params)
    pf, _ = pipeline(
        daily.close,
        target_vol=TARGET_VOL,
        init_cash=INIT_CASH,
        fees=0.0,
        slippage=SLIPPAGE,
        **build_overlay_kwargs(p, atr, memory=memory),
    )
    return pf


# ────────────────────────────────────────────────────────────────────────
# Figure 1 — la martingale sur la vraie séquence de pertes
# ────────────────────────────────────────────────────────────────────────
def _martingale_sizes(pnl: np.ndarray, mult: float, n_max: int, cap: float):
    """Taille exigée à chaque transaction : brute et telle qu'implémentée."""
    capped, uncapped, streak = [], [], 0
    for x in pnl:
        capped.append(min(mult ** min(streak, n_max), cap))
        uncapped.append(mult**streak)
        streak = 0 if x > 0 else streak + 1
    return np.array(capped), np.array(uncapped)


def figure_martingale(trades: pd.DataFrame, mult: float = 2.0, n_max: int = 3) -> dict:
    """Planche martingale : vue d'ensemble sur tout l'historique, puis zoom mécanique."""
    pnl = trades["PnL"].to_numpy()
    dates = pd.to_datetime(trades["Entry Index"])
    capped_all, uncapped_all = _martingale_sizes(pnl, mult, n_max, cap=4.0)

    # plus longue série de pertes réellement observée
    best_start, best_len, cur_start, cur_len = 0, 0, 0, 0
    for i, w in enumerate(pnl > 0):
        if w:
            cur_len, cur_start = 0, i + 1
        else:
            cur_len += 1
            if cur_len > best_len:
                best_len, best_start = cur_len, cur_start

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(9.6, 7.4), gridspec_kw={"height_ratios": [1, 1.15]}
    )

    # ── (a) toute la période ────────────────────────────────────────────
    ax1.step(dates, uncapped_all, where="post", color=PALETTE["martingale"],
             linewidth=1.6, label=f"Martingale ×{mult:g} — taille exigée")
    ax1.fill_between(dates, 1.0, uncapped_all, step="post",
                     color=PALETTE["martingale"], alpha=0.13)
    ax1.axhline(1.0, color=PALETTE["flat"], linewidth=2.0,
                label="Dimensionnement plat (livré) — constant à 1×")
    ax1.set_yscale("log", base=2)
    ax1.set_ylabel("Taille de position\n(× la mise initiale)")
    ax1.set_title(
        f"(a) Sur tout l'historique — {len(pnl)} transactions du moteur or, 2019–2026",
        fontsize=11,
    )
    ax1.legend(loc="upper left")
    span = dates.iloc[best_start], dates.iloc[min(len(pnl) - 1, best_start + best_len)]
    ax1.axvspan(span[0], span[1], color=PALETTE["gold"], alpha=0.22, zorder=0)
    ax1.annotate("zoom (b)", xy=(span[0], uncapped_all.max()), xytext=(6, -4),
                 textcoords="offset points", fontsize=9, color="#7a5c1e",
                 fontweight="bold")
    ax1.tick_params(axis="x", rotation=20)

    # ── (b) zoom sur la plus longue série de pertes ─────────────────────
    hi = min(len(pnl), best_start + best_len + 1)
    seq_pnl = pnl[best_start:hi]
    capped, uncapped = _martingale_sizes(seq_pnl, mult, n_max, cap=4.0)
    x = np.arange(1, len(seq_pnl) + 1)

    ax2.step(x, uncapped, where="mid", color=PALETTE["martingale"], linewidth=2.0,
             label=f"Martingale ×{mult:g} — exigence brute")
    ax2.step(x, capped, where="mid", color=PALETTE["grid"], linewidth=1.8,
             linestyle="--", label="Martingale telle qu'implémentée (plafond 4×)")
    ax2.axhline(1.0, color=PALETTE["flat"], linewidth=2.0, label="Dimensionnement plat")
    ax2.set_yscale("log", base=2)
    ax2.set_ylim(0.7, uncapped.max() * 2.6)
    ax2.set_ylabel("Taille de position\n(× la mise initiale)")
    ax2.set_xlabel("Transactions successives, de la 1ʳᵉ perte au gain qui clôt la série")
    ax2.set_xticks(x)
    ax2.set_title(
        f"(b) Zoom sur la plus longue série de pertes réelle — {best_len} pertes consécutives",
        fontsize=11,
    )
    ax2.legend(loc="upper left", ncol=1)

    # résultat de chaque transaction, en couleur, sous l'axe
    for xi, p in zip(x, seq_pnl):
        ax2.annotate(
            f"{p / 1000:+.1f}k$",
            xy=(xi, 0.78), ha="center", fontsize=8,
            color=PALETTE["anti"] if p > 0 else PALETTE["martingale"],
        )

    peak = float(uncapped.max())
    ax2.annotate(
        f"×{peak:g} la mise initiale",
        xy=(float(np.argmax(uncapped)) + 1, peak), xytext=(-8, 14),
        textcoords="offset points", fontsize=9.5,
        color=PALETTE["martingale"], fontweight="bold", ha="right",
    )

    fig.suptitle(
        "Martingale : la taille double après chaque perte",
        fontsize=12.5, fontweight="bold", y=0.995,
    )
    fig.tight_layout()
    save_fig(fig, "gold_sizing_martingale")
    return {
        "streak": int(best_len),
        "peak": peak,
        "n_trades": int(len(pnl)),
        "n_above_cap": int((uncapped_all > 4.0).sum()),
        "max_all": float(uncapped_all.max()),
    }


# ────────────────────────────────────────────────────────────────────────
# Figure 2 — la grille sur une position réelle qui part contre
# ────────────────────────────────────────────────────────────────────────
def _simulate_grid(seg: pd.Series, atr_at_entry: float, grid_k: float,
                   n_levels: int, max_total: float = 4.0):
    """Applique la règle de grille de ``sizing_nb`` au chemin de prix réel."""
    anchor = float(seg.iloc[0])
    spacing = atr_at_entry / anchor * grid_k
    level, total = 0, 1.0
    fills = [(seg.index[0], anchor)]
    for ts, px in seg.items():
        adv = 1.0 - float(px) / anchor
        while (
            level < n_levels
            and spacing > 0
            and adv >= (level + 1) * spacing
            and total + 1.0 <= max_total
        ):
            level += 1
            total += 1.0
            fills.append((ts, float(px)))
    return fills, level, total, spacing, anchor


def figure_grid(
    daily: pd.DataFrame,
    atr: np.ndarray,
    trades: pd.DataFrame,
    grid_k: float = 0.5,
    n_levels: int = 3,
) -> dict:
    """Planche grille : vue d'ensemble sur tout l'historique, puis zoom mécanique."""
    atr_s = pd.Series(atr.ravel(), index=daily.index)

    # simule la grille sur chaque position réelle
    rows = []
    for _, tr in trades.iterrows():
        entry, exit_ = pd.Timestamp(tr["Entry Index"]), pd.Timestamp(tr["Exit Index"])
        seg = daily.close.loc[entry:exit_]
        if len(seg) < 2:
            continue
        fills, level, total, spacing, anchor = _simulate_grid(
            seg, float(atr_s.loc[entry]), grid_k, n_levels
        )
        rows.append({
            "entry": entry, "exit": exit_, "seg": seg, "fills": fills,
            "level": level, "total": total, "spacing": spacing, "anchor": anchor,
            "n_days": len(seg), "dd": float(1.0 - seg.min() / anchor),
        })

    # épisode de démonstration : le plus de paliers, puis le plus lisible (durée)
    max_level = max(r["level"] for r in rows)
    cands = [r for r in rows if r["level"] == max_level]
    ep = max(cands, key=lambda r: r["n_days"])

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(9.6, 7.8), gridspec_kw={"height_ratios": [1, 1.5]}
    )

    # ── (a) toute la période : exposition atteinte position par position ──
    xs = [r["entry"] for r in rows]
    ys = [r["total"] for r in rows]
    cols = [PALETTE["grid"] if r["level"] else "#B8C2CC" for r in rows]
    ax1.bar(xs, ys, width=26, color=cols)
    ax1.axhline(1.0, color=PALETTE["flat"], linewidth=2.0,
                label="Dimensionnement plat (livré) — constant à 1×")
    ax1.axhline(4.0, color=PALETTE["martingale"], linewidth=1.1, linestyle=":",
                label="Plafond d'exposition cumulée (4×)")
    n_triggered = sum(1 for r in rows if r["level"])
    n_capped = sum(1 for r in rows if r["total"] >= 4.0)
    ax1.set_ylabel("Exposition maximale\n(× la mise initiale)")
    ax1.set_ylim(0, 4.6)
    ax1.set_title(
        f"(a) Sur tout l'historique — la grille ajoute au moins un palier sur "
        f"{n_triggered} des {len(rows)} positions, et sature à 4× sur {n_capped}",
        fontsize=11,
    )
    ax1.legend(loc="upper left")
    ax1.axvspan(ep["entry"], ep["exit"], color=PALETTE["gold"], alpha=0.30, zorder=0)
    ax1.annotate("zoom (b)", xy=(ep["entry"], 4.3), xytext=(6, 0),
                 textcoords="offset points", fontsize=9, color="#7a5c1e",
                 fontweight="bold")
    ax1.tick_params(axis="x", rotation=20)

    # ── (b) zoom mécanique sur un épisode réel ──────────────────────────
    seg, fills, spacing, anchor = ep["seg"], ep["fills"], ep["spacing"], ep["anchor"]
    avg_price = np.cumsum([f[1] for f in fills]) / np.arange(1, len(fills) + 1)

    ax2.plot(seg.index, seg.to_numpy(), color=PALETTE["price"], linewidth=1.5,
             label="Cours de l'once XAU-USD")
    for i in range(1, ep["level"] + 1):
        ax2.axhline(anchor * (1 - i * spacing), color=PALETTE["grid"],
                    linewidth=0.9, linestyle=":", zorder=1)
        ax2.annotate(f"seuil palier {i}", xy=(seg.index[0], anchor * (1 - i * spacing)),
                     xytext=(2, 3), textcoords="offset points", fontsize=7.5,
                     color=PALETTE["grid"])
    ax2.scatter([fills[0][0]], [fills[0][1]], color=PALETTE["flat"], s=70, zorder=6,
                label="Entrée du signal (1×)")
    if len(fills) > 1:
        ax2.scatter([f[0] for f in fills[1:]], [f[1] for f in fills[1:]],
                    color=PALETTE["grid"], s=60, zorder=6,
                    label=f"Ajouts de la grille ({len(fills) - 1} paliers)")
    ax2.step(list(seg.index[seg.index >= fills[0][0]])[:1] + [f[0] for f in fills] + [seg.index[-1]],
             [avg_price[0]] + list(avg_price) + [avg_price[-1]],
             where="post", color=PALETTE["martingale"], linewidth=1.6, linestyle="--",
             label="Prix de revient moyen")
    ax2.set_ylabel("Prix de l'once ($)")
    ax2.set_xlabel("")
    ax2.set_title(
        f"(b) Zoom mécanique — position du {ep['entry']:%d %b %Y} au {ep['exit']:%d %b %Y} "
        f"({ep['n_days']} séances, excursion adverse de {ep['dd']:.1%})",
        fontsize=11,
    )
    ax2.legend(loc="best", fontsize=8.5)
    ax2.tick_params(axis="x", rotation=20)

    ax2b = ax2.twinx()
    steps_x = [f[0] for f in fills] + [seg.index[-1]]
    steps_y = list(np.arange(1.0, len(fills) + 1.0)) + [float(len(fills))]
    ax2b.step(steps_x, steps_y, where="post", color=PALETTE["grid"],
              linewidth=1.6, alpha=0.55)
    ax2b.fill_between(steps_x, 0, steps_y, step="post",
                      color=PALETTE["grid"], alpha=0.10)
    ax2b.set_ylabel("Exposition cumulée (×)", color=PALETTE["grid"])
    ax2b.set_ylim(0, 4.6)
    ax2b.grid(False)

    fig.suptitle(
        "Grille : on rachète par paliers à mesure que le prix descend contre la position",
        fontsize=12.5, fontweight="bold", y=0.995,
    )
    fig.tight_layout()
    save_fig(fig, "gold_sizing_grid")
    return {
        "drawdown": float(ep["dd"]), "levels": int(ep["level"]),
        "entry": f"{ep['entry']:%Y-%m-%d}", "exit": f"{ep['exit']:%Y-%m-%d}",
        "n_days": int(ep["n_days"]), "n_triggered": n_triggered,
        "n_positions": len(rows), "n_capped": n_capped,
    }


# ────────────────────────────────────────────────────────────────────────
# Figure 3 — l'inversion du classement entre les deux fenêtres
# ────────────────────────────────────────────────────────────────────────
def figure_outcome(daily: pd.DataFrame, atr: np.ndarray) -> None:
    regimes = [
        ("Plat (livré)", MODE_FLAT, {}, PALETTE["flat"], 2.2),
        ("Martingale ×2", MODE_MARTINGALE, dict(mult=2.0, n_max=3), PALETTE["martingale"], 1.5),
        ("Grille k=0,5", MODE_GRID, dict(grid_k=0.5, n_levels=3), PALETTE["grid"], 1.5),
        ("Anti-martingale ×1,5", MODE_ANTI_MART, dict(mult=1.5, n_max=3), PALETTE["anti"], 1.5),
    ]
    series = {}
    for label, mode, params, _, _ in regimes:
        pf = run_regime(daily, atr, mode, **params)
        series[label] = pf.returns
        print(f"    · {label} simulé")

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.3))
    windows = [
        ("Fenêtre de sélection (2019 – juin 2025)", lambda s: s[s.index < HOLDOUT_START]),
        ("Fenêtre aveugle (juillet 2025 – 2026)", lambda s: s[s.index >= HOLDOUT_START]),
    ]
    for ax, (title, slicer) in zip(axes, windows):
        sharpes = {}
        for label, _, _, color, lw in regimes:
            r = slicer(series[label]).copy()
            # même volatilité réalisée pour toutes : la comparaison porte sur la
            # forme, pas sur qui a pris le plus de levier
            vol = r.std() * np.sqrt(252)
            if vol > 0:
                r = r * (0.25 / vol)
            sharpes[label] = float(r.mean() / r.std() * np.sqrt(252)) if r.std() > 0 else 0.0
            ax.plot(r.index, (1.0 + r).cumprod().to_numpy(),
                    color=color, linewidth=lw, label=label)
        ax.set_title(title, fontsize=11)
        ax.set_ylabel("Équité (base 1, à risque égal)")
        ax.tick_params(axis="x", rotation=25)
        # classement lisible directement sur le graphique
        ranked = sorted(sharpes.items(), key=lambda kv: -kv[1])
        txt = "\n".join(f"{i + 1}. {lbl} — Sharpe {s:.2f}" for i, (lbl, s) in enumerate(ranked))
        ax.text(0.03, 0.97, txt, transform=ax.transAxes, va="top", ha="left",
                fontsize=8.2, color=PALETTE["text"],
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                          edgecolor="#C8CDD2", linewidth=0.7, alpha=0.94))
    axes[0].legend(loc="lower right", fontsize=8.2)
    fig.suptitle(
        "Mêmes signaux, quatre règles de taille : le classement s'inverse d'une fenêtre à l'autre",
        fontsize=12, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    save_fig(fig, "gold_sizing_outcome")


def main() -> None:
    print("Chargement des données XAU-USD…")
    daily, atr = load_daily()
    print(f"  {len(daily)} séances, {daily.index[0]:%Y-%m-%d} → {daily.index[-1]:%Y-%m-%d}")

    print("Simulation du moteur en dimensionnement plat (séquence de transactions réelle)…")
    pf_flat = run_regime(daily, atr, MODE_FLAT)
    trades = pf_flat.trades.records_readable
    print(f"  {len(trades)} transactions")

    print("Figure 1 — martingale sur la série de pertes réelle…")
    m = figure_martingale(trades)
    print("Figure 2 — grille sur une position réelle…")
    g = figure_grid(daily, atr, trades)
    print("Figure 3 — inversion du classement…")
    figure_outcome(daily, atr)

    print("\nChiffres à citer dans le texte :")
    print(f"  transactions du moteur (flat)        : {m['n_trades']}")
    print(f"  plus longue série de pertes          : {m['streak']}")
    print(f"  pic martingale sur cette série       : ×{m['peak']:g}")
    print(f"  pic martingale sur tout l'historique : ×{m['max_all']:g}")
    if g:
        print(f"  grille — positions avec ≥1 palier    : {g['n_triggered']}/{g['n_positions']}")
        print(f"  grille — positions saturées à 4×     : {g['n_capped']}")
        print(f"  épisode zoom                         : {g['entry']} → {g['exit']} "
              f"({g['n_days']} séances, excursion {g['drawdown']:.1%}, {g['levels']} paliers)")


if __name__ == "__main__":
    main()
