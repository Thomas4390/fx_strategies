#!/usr/bin/env python3
"""Figures et tables du rapport trade par trade du candidat USD/JPY.

Consomme ``reports/mt5/usdjpy_trades_research.csv`` (figé par
``scripts/extract_usdjpy_trades.py``). Sortie :
``reports/client/rapport_technique/figures/usdjpy_trades_*.png`` et
``reports/client/rapport_technique/tables/usdjpy_trades_*.tex``.

Le document décrit un **run de recherche**, pas un backtest de production : la
sleeve TSMOM (moteur or, ``Inp_Gold_Symbol=USDJPY``) y est jouée seule, à
``Alloc=1.0`` et ``Inp_RiskScale=1.0``, sur un dépôt de 100 000 $. Les montants
qui suivent décrivent donc ce sous-compte isolé et ne sont comparables ni au
portefeuille client ni aux chiffres de la sleeve or.

Comme pour l'or, le document est **adossé au moteur d'exécution** : le module de
stratégie n'est utilisé que pour recalculer l'indicateur — cours de clôture de
séance, score d'ensemble, levier cible — qui sert de contexte aux figures.
Aucun résultat de backtest Python n'entre dans le rapport.

Trois précautions de mesure sont câblées ici, parce que les ignorer produit des
chiffres plausibles et faux :

* **On ne publie aucun ratio de Sharpe reconstruit.** L'équité tirée des deals
  est une *balance* : elle ne bouge qu'à la clôture d'une position. Le seul
  Sharpe cité dans le document est celui du simulateur (0,66).
* **Le rendement de prix d'un long USD/JPY n'est pas ``sortie/entrée - 1``.**
  Le dollar est ici la devise de *base* : le résultat en dollars vaut
  ``notionnel × (1 - entrée/sortie)``. C'est cette définition qui recolle
  exactement au ``profit`` du courtier, et c'est elle qui est publiée.
* **Le swap est une recette, pas un coût.** Un long USD/JPY encaisse le
  différentiel de taux ; toutes les légendes qui parlent de portage sont
  écrites dans ce sens et calculent leur part plutôt que de la supposer.

Usage:
    python scripts/build_usdjpy_trades_figures.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SCRIPTS = Path(__file__).resolve().parent
for _path in (str(_SCRIPTS), str(_PROJECT_ROOT / "src")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

# Style, palette et écriture des assets viennent du générateur du rapport : une
# cinquième copie du bloc rcParams finirait par dériver des quatre autres.
from build_latex_report_assets import (  # noqa: E402
    PALETTE,
    save_fig,
    save_tex,
)

TRADES_CSV = _PROJECT_ROOT / "reports/mt5/usdjpy_trades_research.csv"

# Bleu de la sleeve momentum (preamble.tex) plutôt que le doré de l'or : les
# deux jeux de figures doivent rester distinguables au premier coup d'œil.
YEN = "#1F5582"
WIN = "#2E8B57"
LOSS = "#8E1616"
NEUTRAL = "#4A4A4A"

# Lot FX standard : 100 000 unités de la devise de base, ici le dollar. Le
# notionnel d'une position est donc directement en dollars, sans conversion.
UNITS_PER_LOT = 100_000.0
LOT_STEP = 0.01
# Inp_Gold_SlippageBps côté MT5, en points de base par côté.
SLIPPAGE_BPS_PER_SIDE = 2.0
# Dépôt du run de recherche en sleeve isolée — pas celui du portefeuille client.
MT5_INITIAL_DEPOSIT = 100_000.0
TARGET_VOL_PCT = 55.0
MAX_LEVERAGE = 6.6


# ---------------------------------------------------------------------------
# Chargement
# ---------------------------------------------------------------------------


def load_mt5_trades() -> pd.DataFrame:
    trades = _read_trades()
    trades["notional"] = trades["lots"] * UNITS_PER_LOT
    # Le dollar est la devise de base : le profit en dollars d'un long vaut
    # notionnel × (1 - entrée/sortie). Vérifié ligne à ligne contre le CSV.
    trades["price_return"] = trades["profit"] / trades["notional"]
    trades["win"] = trades["net"] > 0
    # La balance ne bouge qu'à la clôture : l'équité se cumule dans l'ordre des
    # sorties, pas des entrées.
    by_exit = trades.sort_values("exit_time")
    trades["balance_after"] = (
        MT5_INITIAL_DEPOSIT + by_exit["net"].cumsum()
    ).reindex(trades.index)
    return trades


def _read_trades() -> pd.DataFrame:
    if not TRADES_CSV.exists():
        raise SystemExit(
            f"{TRADES_CSV.relative_to(_PROJECT_ROOT)} manque.\n"
            "Lancer d'abord : python scripts/extract_usdjpy_trades.py"
        )
    trades = pd.read_csv(TRADES_CSV, parse_dates=["entry_time", "exit_time"])
    drift = (
        trades["profit"]
        - trades["lots"] * UNITS_PER_LOT * (1.0 - trades["entry_price"] / trades["exit_price"])
    ).abs()
    if drift.max() > 0.02:
        raise SystemExit(
            "Le résultat en dollars ne se reconstruit pas depuis les prix "
            f"(écart max {drift.max():.2f} $). La convention de contrat "
            "retenue est fausse — ne pas publier."
        )
    return trades


def load_signal_context() -> dict[str, Any]:
    """Recalculer l'indicateur du moteur : cours de séance, score, levier cible.

    Seul l'indicateur est utilisé — ni le portefeuille ni les trades produits
    par ce chemin n'entrent dans le rapport, qui est adossé au seul journal
    d'exécution. ``strategies.tsmom`` porte la même sleeve que l'or, avec les
    conventions de données d'USD/JPY ; ``fill="next_open"`` est la convention
    d'exécution du simulateur.
    """
    from strategies.tsmom import pipeline
    from utils import apply_vbt_settings

    apply_vbt_settings()
    _, ind = pipeline("USD-JPY", fill="next_open")

    return {
        "close": ind.close,
        "score": ind.score,
        "leverage": ind.leverage,
    }


# ---------------------------------------------------------------------------
# Chapitre 1 — le moteur en une page
# ---------------------------------------------------------------------------


def fig_timeline(mt5: pd.DataFrame, ctx: dict[str, Any]) -> None:
    close, score, lev = ctx["close"], ctx["score"], ctx["leverage"]
    lo = mt5["entry_time"].min().normalize()
    close, score, lev = close[close.index >= lo], score[score.index >= lo], lev[lev.index >= lo]

    fig, axes = plt.subplots(
        3, 1, figsize=(9.5, 7.6), sharex=True, gridspec_kw={"height_ratios": [3, 1.2, 1.2]}
    )

    ax = axes[0]
    ax.plot(close.index, close.values, color=NEUTRAL, linewidth=1.0, zorder=3)
    for _, t in mt5.iterrows():
        ax.axvspan(
            t["entry_time"], t["exit_time"],
            color=WIN if t["net"] > 0 else LOSS, alpha=0.30, linewidth=0, zorder=1,
        )
    ax.set_ylabel("USDJPY (yens par dollar)")
    ax.set_title(
        f"Les {len(mt5)} trades du candidat USD/JPY dans leur contexte de prix",
        color=PALETTE["primary"],
    )
    ax.legend(
        handles=[
            Patch(facecolor=WIN, alpha=0.30, label="trade gagnant"),
            Patch(facecolor=LOSS, alpha=0.30, label="trade perdant"),
        ],
        loc="upper left",
    )

    ax = axes[1]
    ax.fill_between(score.index, 0, score.values, color=YEN, alpha=0.55, linewidth=0)
    ax.axhline(0, color=NEUTRAL, linewidth=0.8)
    ax.set_ylabel("score")
    ax.set_ylim(-1.05, 1.05)

    ax = axes[2]
    ax.plot(lev.index, lev.values, color=PALETTE["mr"], linewidth=1.0)
    ax.scatter(
        mt5["entry_time"], mt5["leverage"],
        s=22, color=YEN, edgecolor="white", linewidth=0.5, zorder=4,
        label="levier MT5 à l'entrée",
    )
    ax.set_ylabel("levier")
    ax.set_xlabel("")
    ax.legend(loc="lower left", fontsize=8)

    fig.align_ylabels(axes)
    save_fig(fig, "usdjpy_trades_timeline")


# ---------------------------------------------------------------------------
# Chapitre 2 — le catalogue
# ---------------------------------------------------------------------------


def fig_gantt(mt5: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 8.2))
    # Les volumes vont de 8 à 25 lots : c'est le rapport entre eux qui porte
    # l'information, pas leur valeur absolue, sinon les barres se chevauchent.
    thickness = 0.30 + 0.60 * mt5["lots"] / mt5["lots"].max()
    for i, (_, t) in enumerate(mt5.iterrows()):
        span = (t["exit_time"] - t["entry_time"]).total_seconds() / 86400.0
        ax.barh(
            i, max(span, 0.6), left=t["entry_time"],
            height=thickness.iloc[i],
            color=WIN if t["net"] > 0 else LOSS, alpha=0.85,
        )
        if abs(t["net"]) > 20_000:
            ax.text(
                t["exit_time"] + pd.Timedelta(days=25), i, f"{t['net']:+,.0f}",
                va="center", fontsize=7, color=NEUTRAL,
            )
    ax.set_yticks(range(len(mt5)))
    ax.set_yticklabels(
        [f"{i + 1:>2}  {d:%Y-%m-%d}" for i, d in enumerate(mt5["entry_time"])],
        fontsize=7,
    )
    ax.invert_yaxis()
    ax.set_xlabel("")
    ax.set_title(
        "Durée de chaque trade — épaisseur proportionnelle aux lots engagés",
        color=PALETTE["primary"],
    )
    ax.legend(
        handles=[
            Patch(facecolor=WIN, label="gagnant"),
            Patch(facecolor=LOSS, label="perdant"),
        ],
        loc="upper right",
    )
    save_fig(fig, "usdjpy_trades_gantt")


def fig_waterfall(mt5: pd.DataFrame) -> None:
    ordered = mt5.sort_values("exit_time", ignore_index=True)
    running = MT5_INITIAL_DEPOSIT + ordered["net"].cumsum()
    starts = running.shift(1).fillna(MT5_INITIAL_DEPOSIT)

    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    for i, (_, t) in enumerate(ordered.iterrows()):
        ax.bar(
            i, t["net"], bottom=starts.iloc[i], width=0.72,
            color=WIN if t["net"] > 0 else LOSS, alpha=0.9,
        )
    ax.plot(range(len(ordered)), running.values, color=PALETTE["primary"], linewidth=1.4)
    ax.axhline(MT5_INITIAL_DEPOSIT, color=NEUTRAL, linestyle="--", linewidth=0.9)
    ax.text(
        0, MT5_INITIAL_DEPOSIT, f"  dépôt {MT5_INITIAL_DEPOSIT:,.0f}",
        fontsize=8, color=NEUTRAL, va="bottom",
    )
    ax.set_xticks(range(0, len(ordered), 2))
    ax.set_xticklabels(
        [f"{d:%m/%y}" for d in ordered["exit_time"].iloc[::2]], fontsize=7, rotation=45
    )
    ax.set_ylabel("balance du sous-compte (USD)")
    ax.set_title(
        "Cascade du résultat, trade par trade, dans l'ordre des clôtures",
        color=PALETTE["primary"],
    )
    save_fig(fig, "usdjpy_trades_waterfall")


def fig_scatter_duration(mt5: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    ax.scatter(
        mt5["duration_days"], mt5["net"],
        s=mt5["lots"] * 12, alpha=0.75,
        c=[WIN if n > 0 else LOSS for n in mt5["net"]],
        edgecolor="white", linewidth=0.6,
    )
    ax.axhline(0, color=NEUTRAL, linewidth=0.9)
    ax.set_xscale("symlog", linthresh=1.0)
    # Aucun trade ne dure moins d'une journée : sans borne, symlog déroule une
    # demi-échelle négative vide.
    ax.set_xlim(0.85, mt5["duration_days"].max() * 1.6)
    ax.set_xlabel("durée du trade (jours, échelle log)")
    ax.set_ylabel("résultat net (USD)")
    ax.set_title(
        "Le résultat vient des tenues longues — la taille du point est le lot engagé",
        color=PALETTE["primary"],
    )
    for _, t in mt5.nlargest(3, "net").iterrows():
        ax.annotate(
            f"{t['entry_time']:%b %Y}\n{t['net']:+,.0f}",
            (t["duration_days"], t["net"]), textcoords="offset points",
            xytext=(-8, -34), fontsize=7.5, color=NEUTRAL, ha="right",
        )
    save_fig(fig, "usdjpy_trades_scatter_duration")


def fig_distribution(mt5: pd.DataFrame) -> None:
    fig, axes = plt.subplots(
        1, 2, figsize=(9.5, 4.2), gridspec_kw={"width_ratios": [3, 1]}
    )
    ax = axes[0]
    bins = np.histogram_bin_edges(mt5["net"], bins=18)
    ax.hist(
        mt5.loc[mt5["net"] <= 0, "net"], bins=bins, color=LOSS, alpha=0.85,
        label=f"perdants ({(~mt5['win']).sum()})",
    )
    ax.hist(
        mt5.loc[mt5["net"] > 0, "net"], bins=bins, color=WIN, alpha=0.85,
        label=f"gagnants ({mt5['win'].sum()})",
    )
    ax.axvline(0, color=NEUTRAL, linewidth=0.9)
    ax.axvline(
        mt5["net"].mean(), color=YEN, linestyle="--", linewidth=1.4,
        label=f"moyenne {mt5['net'].mean():+,.0f}",
    )
    ax.set_xlabel("résultat net par trade (USD)")
    ax.set_ylabel("nombre de trades")
    ax.set_title(
        "Un trade contre trente-quatre : la distribution est le moteur",
        color=PALETTE["primary"],
    )
    ax.legend()

    ax = axes[1]
    ax.boxplot(
        mt5["net"], widths=0.5, orientation="vertical",
        patch_artist=True,
        boxprops={"facecolor": YEN, "alpha": 0.5},
        medianprops={"color": PALETTE["primary"], "linewidth": 1.6},
        flierprops={"marker": "o", "markersize": 4, "markerfacecolor": YEN},
    )
    ax.axhline(0, color=NEUTRAL, linewidth=0.9)
    ax.set_xticks([])
    ax.set_ylabel("USD")
    ax.set_title("dispersion", color=PALETTE["primary"], fontsize=10)
    # L'axe du boxplot monte à 190 k : sans mise en page, ses graduations
    # débordent sur l'histogramme.
    fig.tight_layout()
    save_fig(fig, "usdjpy_trades_distribution")


# ---------------------------------------------------------------------------
# Chapitre 3 — anatomie
# ---------------------------------------------------------------------------


def _plot_trade_panel(ax, trade: pd.Series, close: pd.Series) -> None:
    pad = pd.Timedelta(days=max(6, trade["duration_days"] * 0.35))
    lo, hi = trade["entry_time"] - pad, trade["exit_time"] + pad
    window = close[(close.index >= lo) & (close.index <= hi)]
    colour = WIN if trade["net"] > 0 else LOSS

    ax.plot(window.index, window.values, color="#9AA0A6", linewidth=1.0, zorder=2)
    ax.axvspan(trade["entry_time"], trade["exit_time"], color=colour, alpha=0.12, linewidth=0)

    # Le trade tel qu'il a été exécuté : segment entre les deux prix du broker.
    ax.plot(
        [trade["entry_time"], trade["exit_time"]],
        [trade["entry_price"], trade["exit_price"]],
        color=colour, linewidth=2.0, zorder=4, solid_capstyle="round",
    )
    ax.scatter([trade["entry_time"]], [trade["entry_price"]], marker="^", s=64,
               color=WIN, edgecolor="white", linewidth=0.7, zorder=5)
    ax.scatter([trade["exit_time"]], [trade["exit_price"]], marker="v", s=64,
               color=LOSS, edgecolor="white", linewidth=0.7, zorder=5)

    if trade["safety_stop"]:
        tag = "  — stop de sécurité"
    elif trade["forced_close"]:
        tag = "  — liquidation de fin de test"
    else:
        tag = ""
    ax.set_title(
        f"{trade['entry_time']:%d %b %Y} → {trade['exit_time']:%d %b %Y}  "
        f"({trade['duration_days']:.0f} j, {trade['lots']:.2f} lot){tag}\n"
        f"net {trade['net']:+,.0f} USD   ·   prix {trade['price_return'] * 100:+.1f} %"
        f"   ·   portage {trade['swap']:+,.0f}",
        fontsize=9, color=PALETTE["primary"],
    )
    ax.tick_params(labelsize=7)
    ax.xaxis.set_major_locator(plt.MaxNLocator(4))
    for label in ax.get_xticklabels():
        label.set_rotation(20)
        label.set_horizontalalignment("right")


def fig_best_worst(mt5: pd.DataFrame, close: pd.Series) -> None:
    subtitle = (
        "trait gris : cours de clôture quotidien d'USD/JPY · "
        "trait coloré : le trade tel qu'exécuté, d'un prix à l'autre"
    )
    for name, subset, title in (
        ("usdjpy_trades_best4", mt5.nlargest(4, "net"),
         "Les quatre trades qui ont fait le résultat"),
        ("usdjpy_trades_worst4", mt5.nsmallest(4, "net"),
         "Les quatre trades les plus coûteux"),
    ):
        fig, axes = plt.subplots(2, 2, figsize=(9.5, 6.8))
        for ax, (_, trade) in zip(axes.ravel(), subset.iterrows()):
            _plot_trade_panel(ax, trade, close)
        fig.suptitle(
            title, color=PALETTE["primary"], fontsize=12, fontweight="bold", y=0.995
        )
        fig.text(0.5, 0.947, subtitle, ha="center", fontsize=8, color=NEUTRAL)
        fig.tight_layout(rect=(0, 0, 1, 0.925))
        save_fig(fig, name)


# ---------------------------------------------------------------------------
# Chapitre 4 — séquences, risque, régimes
# ---------------------------------------------------------------------------


def _streaks(flags: pd.Series) -> list[tuple[bool, int]]:
    out: list[tuple[bool, int]] = []
    for flag in flags:
        if out and out[-1][0] == flag:
            out[-1] = (flag, out[-1][1] + 1)
        else:
            out.append((flag, 1))
    return out


def fig_streaks(mt5: pd.DataFrame) -> tuple[int, int]:
    ordered = mt5.sort_values("exit_time")
    runs = _streaks(ordered["win"])
    worst_loss = max((n for w, n in runs if not w), default=0)
    best_win = max((n for w, n in runs if w), default=0)

    fig, ax = plt.subplots(figsize=(9.5, 4.0))
    cursor = 0
    for is_win, length in runs:
        ax.barh(
            0, length, left=cursor, height=0.55,
            color=WIN if is_win else LOSS, alpha=0.9,
            edgecolor="white", linewidth=0.8,
        )
        if length >= 3:
            ax.text(
                cursor + length / 2, 0, str(length),
                ha="center", va="center", fontsize=8, color="white", fontweight="bold",
            )
        cursor += length
    ax.set_yticks([])
    ax.set_xlabel("trades, dans l'ordre des clôtures")
    ax.set_xlim(0, len(ordered))
    ax.set_title(
        f"Séries consécutives — au pire {worst_loss} pertes d'affilée, "
        f"au mieux {best_win} gains",
        color=PALETTE["primary"],
    )
    ax.legend(
        handles=[Patch(facecolor=WIN, label="gains"), Patch(facecolor=LOSS, label="pertes")],
        loc="upper right", ncol=2,
    )
    save_fig(fig, "usdjpy_trades_streaks")
    return worst_loss, best_win


def fig_equity_dd(mt5: pd.DataFrame) -> float:
    ordered = mt5.sort_values("exit_time", ignore_index=True)
    balance = pd.Series(
        (MT5_INITIAL_DEPOSIT + ordered["net"].cumsum()).to_numpy(),
        index=ordered["exit_time"],
    )
    balance = pd.concat(
        [pd.Series([MT5_INITIAL_DEPOSIT], index=[mt5["entry_time"].min()]), balance]
    )
    underwater = (balance / balance.cummax() - 1.0) * 100
    max_dd = float(-underwater.min())

    fig, axes = plt.subplots(
        2, 1, figsize=(9.5, 5.6), sharex=True, gridspec_kw={"height_ratios": [2, 1]}
    )
    ax = axes[0]
    ax.step(balance.index, balance.values, where="post", color=YEN, linewidth=1.6)
    ax.axhline(MT5_INITIAL_DEPOSIT, color=NEUTRAL, linestyle="--", linewidth=0.9)
    ax.set_ylabel("balance (USD)")
    ax.set_title(
        "Balance du sous-compte USD/JPY et repli — la balance ne bouge "
        "qu'aux clôtures",
        color=PALETTE["primary"],
    )

    # Les trois pires creux tombent à quelques semaines les uns des autres :
    # sans décalage vertical, leurs étiquettes se superposent.
    worst = underwater.nsmallest(3)
    for rank, (date, value) in enumerate(worst.items()):
        ax.annotate(
            f"{value:.0f} %", (date, balance.loc[date]),
            textcoords="offset points", xytext=(4, -12 - 13 * rank),
            fontsize=7.5, color=LOSS,
        )

    ax = axes[1]
    ax.fill_between(underwater.index, underwater.values, 0, color=LOSS, alpha=0.35, step="post")
    ax.set_ylabel("repli (%)")
    ax.set_xlabel("")
    save_fig(fig, "usdjpy_trades_equity_dd")
    return max_dd


def fig_yearly(mt5: pd.DataFrame) -> pd.DataFrame:
    ordered = mt5.sort_values("exit_time")
    grouped = ordered.groupby(ordered["exit_time"].dt.year)
    yearly = pd.DataFrame(
        {
            "net": grouped["net"].sum(),
            "trades": grouped["net"].size(),
            "wins": grouped["win"].sum(),
            "swap": grouped["swap"].sum(),
        }
    )

    fig, ax = plt.subplots(figsize=(9.5, 4.0))
    colours = [WIN if v > 0 else LOSS for v in yearly["net"]]
    ax.bar(yearly.index.astype(str), yearly["net"], color=colours, alpha=0.9, width=0.6)
    ax.axhline(0, color=NEUTRAL, linewidth=0.9)
    for x, (_, row) in enumerate(yearly.iterrows()):
        offset = 12 if row["net"] > 0 else -18
        ax.annotate(
            f"{row['net']:+,.0f}\n{int(row['wins'])}/{int(row['trades'])}",
            (x, row["net"]), ha="center", textcoords="offset points",
            xytext=(0, offset), fontsize=8, color=NEUTRAL,
        )
    ax.set_ylabel("résultat net (USD)")
    ax.set_title(
        "Résultat par année de clôture (gagnants / total)", color=PALETTE["primary"]
    )
    margin = max(abs(yearly["net"].min()), yearly["net"].max()) * 0.25
    ax.set_ylim(yearly["net"].min() - margin, yearly["net"].max() + margin)
    save_fig(fig, "usdjpy_trades_yearly")
    return yearly


# ---------------------------------------------------------------------------
# Chapitre 5 — d'où vient l'argent
# ---------------------------------------------------------------------------


def cost_attribution(mt5: pd.DataFrame) -> dict[str, float]:
    """Décomposer le résultat net en postes **mesurés**, plus une estimation.

    Les trois premiers postes viennent du CSV et se recollent exactement au net
    du rapport MT5. Le slippage est reconstruit depuis ``Inp_Gold_SlippageBps``:
    il est déjà contenu dans les prix d'exécution, ce n'est donc pas un poste
    additionnel mais une part du résultat de prix — la figure le dit.
    """
    gross = float(mt5["profit"].sum())
    swap = float(mt5["swap"].sum())
    commission = float(mt5["commission"].sum())
    net = float(mt5["net"].sum())

    slippage_est = float(
        (mt5["notional"] * SLIPPAGE_BPS_PER_SIDE / 10_000.0 * 2).sum()
    )
    forced = mt5[mt5["forced_close"]]
    # Le portage rapporté au notionnel effectivement porté : c'est le taux de
    # carry quotidien, la grandeur qui se compare d'un instrument à l'autre.
    carry_bp_day = swap / float((mt5["notional"] * mt5["duration_days"]).sum()) * 10_000
    return {
        "gross_price": gross,
        "swap": swap,
        "commission": commission,
        "net": net,
        "slippage_estimate": slippage_est,
        "safety_stop_net": float(mt5.loc[mt5["safety_stop"], "net"].sum()),
        "safety_stop_count": int(mt5["safety_stop"].sum()),
        "forced_close_net": float(forced["net"].sum()),
        "forced_close_count": int(len(forced)),
        "notional_total": float(mt5["notional"].sum()),
        "carry_bp_day": carry_bp_day,
        "swap_share_of_net": swap / net * 100,
        "flipped_by_swap": int(((mt5["profit"] < 0) & (mt5["net"] > 0)).sum()),
    }


def fig_attribution(mt5: pd.DataFrame, att: dict[str, float]) -> None:
    steps = [
        ("Résultat de prix", att["gross_price"], YEN),
        ("Portage (swap)", att["swap"], WIN),
        ("Commission", att["commission"], NEUTRAL),
    ]
    fig, ax = plt.subplots(figsize=(9.5, 4.8))

    cursor = 0.0
    for i, (label, value, colour) in enumerate(steps):
        ax.bar(i, value, bottom=cursor, width=0.62, color=colour, alpha=0.9)
        ax.annotate(
            f"{value:+,.0f}", (i, cursor + value),
            ha="center", va="bottom" if value > 0 else "top",
            textcoords="offset points", xytext=(0, 6 if value > 0 else -14),
            fontsize=9, color=NEUTRAL,
        )
        cursor += value
    ax.bar(len(steps), cursor, width=0.62, color=PALETTE["primary"], alpha=0.9)
    ax.annotate(
        f"{cursor:+,.0f}", (len(steps), cursor), ha="center", va="bottom",
        textcoords="offset points", xytext=(0, 6), fontsize=9,
        color=PALETTE["primary"], fontweight="bold",
    )

    ax.set_xticks(range(len(steps) + 1))
    ax.set_xticklabels([s[0] for s in steps] + ["Résultat net"], fontsize=9)
    ax.axhline(0, color=NEUTRAL, linewidth=0.9)
    ax.set_ylabel("USD")
    ax.set_ylim(0, att["net"] * 1.18)
    ax.set_title(
        "Du résultat de prix au résultat net — le portage apporte "
        f"{att['swap_share_of_net']:.0f} % du net",
        color=PALETTE["primary"], pad=14,
    )
    # Placé au-dessus de la colonne « Commission », qui est vide : ailleurs, la
    # note se superpose à une barre et devient illisible.
    ax.text(
        0.62, 0.32,
        f"Slippage estimé : {att['slippage_estimate']:,.0f} USD\n"
        f"({SLIPPAGE_BPS_PER_SIDE:.0f} bps par côté sur "
        f"{att['notional_total'] / 1e6:.1f} M USD\nde notionnel cumulé) — déjà "
        "compris\ndans le résultat de prix.",
        transform=ax.transAxes, fontsize=7.5, color=NEUTRAL, ha="center", va="center",
    )
    save_fig(fig, "usdjpy_trades_attribution")


def fig_costs(mt5: pd.DataFrame) -> None:
    ordered = mt5.sort_values("exit_time", ignore_index=True)
    fig, axes = plt.subplots(
        1, 2, figsize=(9.5, 4.2), gridspec_kw={"width_ratios": [1.7, 1]}
    )

    ax = axes[0]
    ax.plot(
        ordered["exit_time"], ordered["profit"].cumsum(),
        color=YEN, linewidth=1.8, label="résultat de prix cumulé",
    )
    ax.plot(
        ordered["exit_time"], ordered["net"].cumsum(),
        color=PALETTE["primary"], linewidth=1.8, label="résultat net cumulé",
    )
    ax.fill_between(
        ordered["exit_time"], ordered["profit"].cumsum(), ordered["net"].cumsum(),
        color=WIN, alpha=0.18, label="portage cumulé",
    )
    ax.axhline(0, color=NEUTRAL, linewidth=0.9)
    ax.set_ylabel("USD")
    ax.set_title("Ce que le portage a rapporté", color=PALETTE["primary"], fontsize=11)
    ax.legend(fontsize=8)
    ax.xaxis.set_major_locator(plt.MaxNLocator(5))
    for label in ax.get_xticklabels():
        label.set_rotation(20)
        label.set_horizontalalignment("right")

    ax = axes[1]
    ax.scatter(
        ordered["duration_days"], ordered["swap"],
        s=26, color=WIN, alpha=0.75, edgecolor="white", linewidth=0.5,
    )
    ax.set_xscale("symlog", linthresh=1.0)
    ax.set_yscale("symlog", linthresh=100.0)
    ax.set_xlabel("durée (jours, log)")
    ax.set_ylabel("portage encaissé (USD, log)")
    ax.set_title("Le portage est un loyer de temps", color=PALETTE["primary"], fontsize=11)
    fig.tight_layout()
    save_fig(fig, "usdjpy_trades_costs")


# ---------------------------------------------------------------------------
# Chapitre 6 — le sizing
# ---------------------------------------------------------------------------


def fig_sizing(mt5: pd.DataFrame, ctx: dict[str, Any]) -> dict[str, float]:
    close = ctx["close"]
    realized = (
        close.vbt.pct_change().vbt.rolling_std(21, minp=21, ddof=1) * np.sqrt(252)
    )
    sessions = mt5["entry_time"].dt.normalize()
    vol_at_entry = realized.reindex(sessions, method="ffill").to_numpy() * 100
    window = realized[realized.index >= sessions.min()] * 100

    fig, axes = plt.subplots(
        2, 1, figsize=(9.5, 6.0), sharex=True, gridspec_kw={"height_ratios": [1, 1]}
    )

    ax = axes[0]
    ax.plot(mt5["entry_time"], mt5["leverage"], marker="o", markersize=4,
            color=PALETTE["mr"], linewidth=1.1, label="levier appliqué")
    ax.axhline(MAX_LEVERAGE, color=LOSS, linestyle="--", linewidth=1.0,
               label=f"plafond {MAX_LEVERAGE}×".replace(".", ","))
    ax.set_ylabel("levier")
    ax.set_title(
        "Le vol-targeting en pratique : levier appliqué et volatilité mesurée",
        color=PALETTE["primary"],
    )
    ax.legend(fontsize=8, ncol=2)

    ax = axes[1]
    ax.plot(mt5["entry_time"], vol_at_entry, marker="s", markersize=3.5,
            color=YEN, linewidth=1.1, label="volatilité réalisée 21 j (annualisée)")
    ax.axhline(TARGET_VOL_PCT / MAX_LEVERAGE, color=PALETTE["primary"], linestyle=":",
               linewidth=1.0,
               label=f"seuil de saturation {TARGET_VOL_PCT / MAX_LEVERAGE:.1f} %".replace(".", ","))
    ax.set_ylabel("volatilité (%)")
    ax.set_xlabel("")
    ax.legend(fontsize=8, ncol=2)

    fig.align_ylabels(axes)
    save_fig(fig, "usdjpy_trades_sizing")

    # Écart d'arrondi : le pas de lot de 0.01 ne peut pas exprimer le levier
    # exact. Sur des volumes de 8 à 25 lots il est négligeable — c'est l'inverse
    # du cas de l'or, et le chapitre le dit.
    ideal = mt5["lots"] / LOT_STEP
    rounding = (ideal - ideal.round()).abs()
    if rounding.max() > 1e-6:
        print(f"[i] lots non multiples du pas {LOT_STEP} : max écart {rounding.max():.4f}")

    return {
        "vol_min": float(np.nanmin(window)),
        "vol_max": float(np.nanmax(window)),
        "vol_median": float(np.nanmedian(window)),
        "vol_entry_median": float(np.nanmedian(vol_at_entry)),
        "sessions_at_cap_pct": float((ctx["leverage"][ctx["leverage"].index >= sessions.min()]
                                     >= MAX_LEVERAGE - 1e-9).mean() * 100),
        "entries_at_cap": int((mt5["leverage"] >= MAX_LEVERAGE - 1e-9).sum()),
        "lot_step_share_max_pct": float(LOT_STEP / mt5["lots"].min() * 100),
        "lot_step_share_min_pct": float(LOT_STEP / mt5["lots"].max() * 100),
    }


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------


def _tex_num(value: float, digits: int = 0, signed: bool = False) -> str:
    """Séparateur de milliers typographique, sans toucher au reste de la ligne.

    Un ``replace(",", "\\,")`` appliqué à la ligne entière corrompt les macros
    qui contiennent déjà une virgule — ``\\,`` devient ``\\\\,``, c'est-à-dire
    un saut de ligne, et la ligne du tableau se scinde en deux.
    """
    text = f"{value:+,.{digits}f}" if signed else f"{value:,.{digits}f}"
    return text.replace(",", "\\,")


def table_full(mt5: pd.DataFrame) -> None:
    rows = []
    for i, (_, t) in enumerate(mt5.iterrows(), start=1):
        if t["safety_stop"]:
            flag = r"\,$\dagger$"
        elif t["forced_close"]:
            flag = r"\,$\ddagger$"
        else:
            flag = ""
        rows.append(
            f"{i} & {t['entry_time']:%Y-%m-%d} & {t['exit_time']:%Y-%m-%d}{flag} & "
            f"{t['duration_days']:.0f} & {t['lots']:.2f} & {t['leverage']:.1f} & "
            f"{t['entry_price']:.3f} & {t['exit_price']:.3f} & "
            f"{t['price_return'] * 100:+.2f} & {_tex_num(t['swap'], signed=True)} & "
            f"{_tex_num(t['net'], signed=True)} \\\\"
        )
    body = "\n".join(rows)
    content = (
        "% Généré par scripts/build_usdjpy_trades_figures.py — NE PAS ÉDITER À LA MAIN.\n"
        "\\begin{longtable}{@{}rllrrrrrrrr@{}}\n"
        "\\caption{Les 35 trades du candidat USD/JPY, run MT5 de recherche}\\\\\n"
        "\\toprule\n"
        "\\# & Entrée & Sortie & j & Lots & Lev. & Prix in & Prix out & "
        "\\% prix & Portage & Net \\\\\n\\midrule\n\\endfirsthead\n"
        "\\toprule\n"
        "\\# & Entrée & Sortie & j & Lots & Lev. & Prix in & Prix out & "
        "\\% prix & Portage & Net \\\\\n\\midrule\n\\endhead\n"
        f"{body}\n\\bottomrule\n"
        "\\multicolumn{11}{@{}l@{}}{\\footnotesize $\\dagger$ sortie déclenchée "
        "par le stop de sécurité, hors borne de séance \\quad $\\ddagger$ position "
        "liquidée par le testeur à la fin de la fenêtre.}\n"
        "\\end{longtable}\n"
    )
    save_tex("usdjpy_trades_full", content)


def table_summary(
    mt5: pd.DataFrame, att: dict[str, float], max_dd: float
) -> None:
    wins = mt5[mt5["win"]]
    losses = mt5[~mt5["win"]]
    lines = [
        ("Trades", f"{len(mt5)}"),
        ("Gagnants", f"{len(wins)} ({len(wins) / len(mt5) * 100:.1f}\\,\\%)"),
        ("Gagnants avant portage", f"{int((mt5['profit'] > 0).sum())}"),
        ("Résultat net", f"{att['net']:+,.0f}\\,\\$"),
        ("Résultat de prix", f"{att['gross_price']:+,.0f}\\,\\$"),
        ("Portage encaissé", f"{att['swap']:+,.0f}\\,\\$"),
        ("Gain moyen", f"{wins['net'].mean():+,.0f}\\,\\$"),
        ("Perte moyenne", f"{losses['net'].mean():+,.0f}\\,\\$"),
        ("Meilleur trade", f"{mt5['net'].max():+,.0f}\\,\\$"),
        ("Pire trade", f"{mt5['net'].min():+,.0f}\\,\\$"),
        ("Résultat médian", f"{mt5['net'].median():+,.0f}\\,\\$"),
        ("Rendement de prix moyen", f"{mt5['price_return'].mean() * 100:+.2f}\\,\\%"),
        ("Durée médiane", f"{mt5['duration_days'].median():.0f}\\,j"),
        ("Durée maximale", f"{mt5['duration_days'].max():.0f}\\,j"),
        ("Repli maximal de balance", f"{max_dd:.1f}\\,\\%"),
        ("Stops de sécurité", f"{att['safety_stop_count']}"),
    ]
    body = "\n".join(f"{a} & {b} \\\\" for a, b in lines)
    content = (
        "% Généré par scripts/build_usdjpy_trades_figures.py — NE PAS ÉDITER À LA MAIN.\n"
        "\\begin{table}[H]\n\\centering\n\\begin{tabular}{@{}lr@{}}\n\\toprule\n"
        "\\textbf{Métrique} & \\textbf{Run MT5 de recherche} \\\\\n\\midrule\n"
        f"{body}\n\\bottomrule\n\\end{{tabular}}\n"
        "\\caption{Le candidat USD/JPY en chiffres, sleeve isolée sur un dépôt de "
        "100\\,000\\,\\$, période 2021--2026.}\n"
        "\\end{table}\n"
    )
    save_tex("usdjpy_trades_summary", content)


def table_attribution(att: dict[str, float], yearly: pd.DataFrame) -> None:
    price_share = att["gross_price"] / att["net"] * 100
    lines = [
        ("Résultat de prix (mesuré)", f"{att['gross_price']:+,.0f}",
         f"{price_share:.1f}\\,\\% du net"),
        ("Portage (mesuré)", f"{att['swap']:+,.0f}",
         f"{att['swap_share_of_net']:.1f}\\,\\% du net"),
        ("Commission (mesurée)", f"{att['commission']:+,.0f}", "nulle sur ce compte"),
        ("\\textbf{Résultat net}", f"\\textbf{{{att['net']:+,.0f}}}",
         "recolle au rapport MT5"),
    ]
    body = "\n".join(f"{a} & {b} & {c} \\\\" for a, b, c in lines)
    extra = (
        "\\multicolumn{3}{@{}p{0.92\\linewidth}@{}}{\\footnotesize "
        f"Le portage vaut {att['carry_bp_day']:.2f}\\,pb par jour de notionnel "
        f"porté et retourne le signe de {att['flipped_by_swap']} trades, perdants "
        "sur le prix et gagnants une fois le carry compté. Slippage estimé à "
        f"{att['slippage_estimate']:,.0f}\\,\\$ "
        f"({SLIPPAGE_BPS_PER_SIDE:.0f}\\,bps par côté sur "
        f"{att['notional_total'] / 1e6:.1f}\\,M\\$ de notionnel cumulé) : il est "
        "déjà contenu dans les prix d'exécution, ce n'est pas un poste "
        f"additionnel. La dernière position a été liquidée par le testeur en fin "
        f"de fenêtre pour {att['forced_close_net']:+,.0f}\\,\\$ : ce n'est pas une "
        "sortie décidée par le signal.}\n"
    )
    content = (
        "% Généré par scripts/build_usdjpy_trades_figures.py — NE PAS ÉDITER À LA MAIN.\n"
        "\\begin{table}[H]\n\\centering\n\\begin{tabular}{@{}lrl@{}}\n\\toprule\n"
        "Poste & USD & Part \\\\\n\\midrule\n"
        f"{body}\n\\bottomrule\n\\addlinespace[0.4em]\n{extra}"
        "\\end{tabular}\n"
        "\\caption{Décomposition du résultat : ce qui est mesuré, ce qui est estimé}\n"
        "\\end{table}\n"
    )
    save_tex("usdjpy_trades_attribution", content)

    rows = "\n".join(
        f"{year} & {int(r['trades'])} & {int(r['wins'])} & "
        f"{r['swap']:+,.0f} & {r['net']:+,.0f} \\\\"
        for year, r in yearly.iterrows()
    )
    content = (
        "% Généré par scripts/build_usdjpy_trades_figures.py — NE PAS ÉDITER À LA MAIN.\n"
        "\\begin{table}[H]\n\\centering\n\\begin{tabular}{@{}lrrrr@{}}\n\\toprule\n"
        "Année & Trades & Gagnants & Portage & Net \\\\\n\\midrule\n"
        f"{rows}\n\\bottomrule\n\\end{{tabular}}\n"
        "\\caption{Résultat par année de clôture}\n\\end{table}\n"
    )
    save_tex("usdjpy_trades_yearly", content)


# ---------------------------------------------------------------------------


def main() -> None:
    print("[1/4] Chargement des trades MT5 figés...")
    mt5 = load_mt5_trades()
    print(
        f"      {len(mt5)} trades  "
        f"{mt5['entry_time'].min():%Y-%m-%d} → {mt5['exit_time'].max():%Y-%m-%d}"
    )

    print("[2/4] Recalcul de l'indicateur (cours, score, levier)...")
    ctx = load_signal_context()
    print(f"      {len(ctx['close'])} séances de contexte")

    print("[3/4] Figures...")
    fig_timeline(mt5, ctx)
    fig_gantt(mt5)
    fig_waterfall(mt5)
    fig_scatter_duration(mt5)
    fig_distribution(mt5)
    fig_best_worst(mt5, ctx["close"])
    fig_streaks(mt5)
    max_dd = fig_equity_dd(mt5)
    yearly = fig_yearly(mt5)
    att = cost_attribution(mt5)
    fig_attribution(mt5, att)
    fig_costs(mt5)
    sizing = fig_sizing(mt5, ctx)

    print("[4/4] Tables...")
    table_full(mt5)
    table_summary(mt5, att, max_dd)
    table_attribution(att, yearly)

    print(
        f"\nnet {att['net']:+,.0f}  ·  prix {att['gross_price']:+,.0f}  ·  "
        f"portage {att['swap']:+,.0f} ({att['swap_share_of_net']:.0f} % du net)  ·  "
        f"repli max de balance {max_dd:.1f} %"
    )
    print(
        f"volatilité réalisée {sizing['vol_min']:.1f}–{sizing['vol_max']:.1f} % "
        f"(médiane {sizing['vol_median']:.1f} %)  ·  "
        f"plafond de levier atteint sur {sizing['sessions_at_cap_pct']:.0f} % des "
        f"séances et {sizing['entries_at_cap']} entrées sur {len(mt5)}"
    )
    print(
        f"pas de lot : {sizing['lot_step_share_min_pct']:.3f}–"
        f"{sizing['lot_step_share_max_pct']:.3f} % de la position"
    )


if __name__ == "__main__":
    main()
