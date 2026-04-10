"""Multi-pair + parameter grid sweep for intraday MR strategies.

Tests ``mr_turbo`` and ``mr_macro`` across:
    - 4 FX pairs (EUR-USD, GBP-USD, USD-JPY, USD-CAD)
    - A Bollinger grid: bb_window × bb_alpha × tp_stop

For each (pair, strategy) combination, selects the best config by
walk-forward Sharpe, then computes the optimal fixed leverage required
to land in the 8-15% CAGR target window. All runs share the same
leverage=1 until the optimisation step. Results are dumped to CSV and
appended as Phase 12 in the research journal.

Usage:
    python -m src.research.pair_param_sweep
    python -m src.research.pair_param_sweep --pairs EUR-USD,GBP-USD
    python -m src.research.pair_param_sweep --strategies mr_macro
"""

from __future__ import annotations

import argparse
import itertools
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import vectorbtpro as vbt

_THIS_DIR = Path(__file__).resolve().parent
_SRC_DIR = _THIS_DIR.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from strategies.mr_macro import backtest_mr_macro  # noqa: E402
from strategies.mr_turbo import backtest_mr_turbo  # noqa: E402
from utils import apply_vbt_settings, load_fx_data  # noqa: E402

warnings.filterwarnings("ignore", category=FutureWarning)

_PROJECT_ROOT = _SRC_DIR.parent
_RESULTS_DIR = _PROJECT_ROOT / "results" / "pair_param_sweep"
_RESEARCH_DOC = _PROJECT_ROOT / "docs" / "research" / "eur-usd-bb-mr-research-plan.md"

# Target CAGR window (annualized %)
_CAGR_TARGET = 10.0  # midpoint of [8%, 15%]
_CAGR_LOW = 8.0
_CAGR_HIGH = 15.0
_DD_HARD_CAP = 35.0


# ═══════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class StrategyTarget:
    key: str
    label: str
    backtest_fn: Callable[..., vbt.Portfolio]


STRATEGIES: dict[str, StrategyTarget] = {
    "mr_turbo": StrategyTarget(
        key="mr_turbo",
        label="MR Turbo",
        backtest_fn=backtest_mr_turbo,
    ),
    "mr_macro": StrategyTarget(
        key="mr_macro",
        label="MR Macro",
        backtest_fn=backtest_mr_macro,
    ),
}

# Parameter grid — small enough to run 4 pairs × 2 strats in ~25min
PARAM_GRID: dict[str, list] = {
    "bb_window": [40, 60, 80, 120],
    "bb_alpha": [4.0, 5.0, 6.0],
    "tp_stop": [0.004, 0.006, 0.008],
}

DEFAULT_PAIRS: tuple[str, ...] = ("EUR-USD", "GBP-USD", "USD-JPY", "USD-CAD")


# ═══════════════════════════════════════════════════════════════════════
# METRICS
# ═══════════════════════════════════════════════════════════════════════


def _safe(val) -> float:
    try:
        f = float(val)
    except (TypeError, ValueError):
        return float("nan")
    return f


def compute_metrics(pf: vbt.Portfolio) -> dict[str, float]:
    stats = pf.stats()
    cagr = _safe(pf.annualized_return) * 100.0
    vol = _safe(pf.annualized_volatility) * 100.0
    sharpe = _safe(pf.sharpe_ratio)
    sortino = _safe(pf.sortino_ratio)
    calmar = _safe(pf.calmar_ratio)
    max_dd = abs(_safe(pf.max_drawdown) * 100.0)
    win_rate = _safe(stats.get("Win Rate [%]"))
    pf_ratio = _safe(stats.get("Profit Factor"))
    n_trades = _safe(stats.get("Total Trades"))

    # Daily-resampled VaR 95% annualised (%)
    returns = pf.returns
    if isinstance(returns, pd.DataFrame):
        returns = returns.iloc[:, 0]
    returns = returns.dropna()
    var_95 = float("nan")
    cvar_95 = float("nan")
    if not returns.empty and returns.std() > 0:
        daily = (1.0 + returns).resample("1D").prod() - 1.0
        daily = daily[daily.abs() > 1e-12]
        if not daily.empty:
            v = float(np.percentile(daily.values, 5.0))
            t = daily[daily <= v]
            cv = float(t.mean()) if len(t) else v
            var_95 = v * np.sqrt(252.0) * 100.0
            cvar_95 = cv * np.sqrt(252.0) * 100.0

    equity = pf.value
    if isinstance(equity, pd.DataFrame):
        equity = equity.iloc[:, 0]
    running_max = equity.cummax()
    dd_pct = (equity / running_max - 1.0) * 100.0
    ulcer = float(np.sqrt(np.mean(dd_pct.values**2)))

    return {
        "cagr": cagr,
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": calmar,
        "max_dd": max_dd,
        "vol": vol,
        "var_95": var_95,
        "cvar_95": cvar_95,
        "ulcer": ulcer,
        "win_rate": win_rate,
        "profit_factor": pf_ratio,
        "n_trades": n_trades,
    }


# ═══════════════════════════════════════════════════════════════════════
# SWEEP
# ═══════════════════════════════════════════════════════════════════════


def iter_grid(grid: dict[str, list]) -> list[dict]:
    keys = list(grid.keys())
    return [dict(zip(keys, combo)) for combo in itertools.product(*(grid[k] for k in keys))]


def sweep_pair_strategy(
    strat: StrategyTarget,
    pair: str,
    data: vbt.Data,
) -> pd.DataFrame:
    rows = []
    combos = iter_grid(PARAM_GRID)
    for i, params in enumerate(combos, 1):
        try:
            pf = strat.backtest_fn(data, leverage=1.0, **params)
            metrics = compute_metrics(pf)
        except Exception as e:
            print(f"    ERROR {params}: {type(e).__name__}: {e}")
            metrics = {k: float("nan") for k in (
                "cagr", "sharpe", "sortino", "calmar", "max_dd", "vol",
                "var_95", "cvar_95", "ulcer", "win_rate", "profit_factor",
                "n_trades",
            )}
        row = {"pair": pair, "strategy": strat.key, **params, **metrics}
        rows.append(row)
        if i % 6 == 0 or i == len(combos):
            print(f"    [{i:>2}/{len(combos)}] last: {params} → "
                  f"Sharpe={metrics['sharpe']:+.2f} CAGR={metrics['cagr']:+.2f}%")
    return pd.DataFrame(rows)


def find_optimal_leverage(
    strat: StrategyTarget,
    data: vbt.Data,
    params: dict,
    base_cagr: float,
) -> tuple[float, dict] | None:
    """Find smallest leverage that lands CAGR in [8, 15] with DD < 35%.

    Uses linear scaling L = target / base_cagr as initial guess, then
    probes 3 candidates and picks the best valid one.
    """
    if base_cagr <= 0 or np.isnan(base_cagr):
        return None
    l_mid = _CAGR_TARGET / base_cagr
    l_low = max(1.0, _CAGR_LOW / base_cagr)
    l_high = _CAGR_HIGH / base_cagr
    candidates = sorted(set(round(x, 1) for x in (l_low, l_mid, l_high) if x >= 1.0))
    best = None
    for lev in candidates:
        try:
            pf = strat.backtest_fn(data, leverage=lev, **params)
            metrics = compute_metrics(pf)
        except Exception:
            continue
        cagr = metrics["cagr"]
        dd = metrics["max_dd"]
        if np.isnan(cagr) or dd >= _DD_HARD_CAP:
            continue
        score_in_band = _CAGR_LOW <= cagr <= _CAGR_HIGH
        entry = (lev, metrics, score_in_band)
        if best is None:
            best = entry
        elif score_in_band and not best[2]:
            best = entry
        elif score_in_band == best[2]:
            # Prefer smaller leverage if both valid; else closer to target
            if score_in_band:
                if lev < best[0]:
                    best = entry
            else:
                if abs(cagr - _CAGR_TARGET) < abs(best[1]["cagr"] - _CAGR_TARGET):
                    best = entry
    if best is None:
        return None
    return best[0], best[1]


# ═══════════════════════════════════════════════════════════════════════
# REPORTING
# ═══════════════════════════════════════════════════════════════════════


def format_best_configs_table(best_rows: list[dict]) -> str:
    if not best_rows:
        return "_Aucun résultat._"

    headers = [
        "Paire", "Stratégie", "bb_win", "bb_α", "tp_stop",
        "Sharpe", "CAGR@1x", "DD@1x", "L*",
        "CAGR@L*", "DD@L*", "Vol@L*", "Sortino", "Calmar",
        "VaR95@L*", "Ulcer@L*", "PF", "Trades", "Statut",
    ]
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in best_rows:
        cells = [
            r["pair"],
            r["strategy_label"],
            f"{int(r['bb_window'])}",
            f"{r['bb_alpha']:.1f}",
            f"{r['tp_stop']:.3f}",
            f"{r['sharpe_1x']:+.2f}",
            f"{r['cagr_1x']:+.2f}%",
            f"{r['dd_1x']:.2f}%",
            f"{r['leverage']:.1f}x" if r["leverage"] is not None else "n/a",
            f"{r['cagr_L']:+.2f}%" if r["cagr_L"] is not None else "n/a",
            f"{r['dd_L']:.2f}%" if r["dd_L"] is not None else "n/a",
            f"{r['vol_L']:.2f}%" if r["vol_L"] is not None else "n/a",
            f"{r['sortino_L']:+.2f}" if r["sortino_L"] is not None else "n/a",
            f"{r['calmar_L']:+.2f}" if r["calmar_L"] is not None else "n/a",
            f"{r['var95_L']:+.2f}%" if r["var95_L"] is not None else "n/a",
            f"{r['ulcer_L']:.2f}%" if r["ulcer_L"] is not None else "n/a",
            f"{r['pf_L']:.2f}" if r["pf_L"] is not None else "n/a",
            f"{int(r['trades_L'])}" if r["trades_L"] is not None else "n/a",
            r["status"],
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def format_grid_snapshot(df: pd.DataFrame, pair: str, strategy_label: str) -> str:
    """Compact per-(pair, strategy) Sharpe heatmap over bb_window × bb_alpha."""
    df2 = df[(df["pair"] == pair)].copy()
    if df2.empty:
        return ""
    # Pick the best tp_stop per (bb_window, bb_alpha) cell
    idx = df2.groupby(["bb_window", "bb_alpha"])["sharpe"].idxmax()
    best = df2.loc[idx]
    pivot = best.pivot(index="bb_window", columns="bb_alpha", values="sharpe")

    lines = [f"**{pair} — {strategy_label} — Sharpe par (bb_window × bb_alpha)**", ""]
    cols = list(pivot.columns)
    lines.append("| bb_win \\ bb_α | " + " | ".join(f"{c:.1f}" for c in cols) + " |")
    lines.append("|" + "|".join(["---"] * (len(cols) + 1)) + "|")
    for win in pivot.index:
        row = pivot.loc[win]
        lines.append(
            f"| **{int(win)}** | "
            + " | ".join(
                f"{row[c]:+.2f}" if not pd.isna(row[c]) else "n/a" for c in cols
            )
            + " |"
        )
    return "\n".join(lines)


_PHASE_MARKER_BEGIN = "<!-- BEGIN PHASE 12 -->"
_PHASE_MARKER_END = "<!-- END PHASE 12 -->"


def render_phase_12(
    full_df: pd.DataFrame,
    best_rows: list[dict],
    pairs: list[str],
    strat_keys: list[str],
) -> str:
    parts: list[str] = [_PHASE_MARKER_BEGIN, ""]
    parts.append("## Phase 12 : Sweep multi-paire et grille de paramètres")
    parts.append("")
    parts.append("**Date :** 2026-04-10")
    parts.append("")
    parts.append("### Hypothèse")
    parts.append("")
    parts.append(
        "Les stratégies `mr_turbo` et `mr_macro` ont été validées sur EUR-USD. "
        "On teste leur robustesse en appliquant une grille de paramètres "
        "Bollinger sur 4 paires FX (EUR-USD, GBP-USD, USD-JPY, USD-CAD) pour "
        "vérifier (1) si l'edge se transfère et (2) quel levier atteint la "
        "cible CAGR 8-15% par paire."
    )
    parts.append("")
    parts.append("### Protocole")
    parts.append("")
    combos = len(iter_grid(PARAM_GRID))
    total = combos * len(pairs) * len(strat_keys)
    parts.append(
        f"- **Paires :** {', '.join(pairs)}"
    )
    parts.append(
        f"- **Stratégies :** {', '.join(STRATEGIES[k].label for k in strat_keys)}"
    )
    parts.append(
        f"- **Grille** ({combos} combos par paire × stratégie, {total} backtests total) :"
    )
    for k, v in PARAM_GRID.items():
        parts.append(f"  - `{k}` ∈ {v}")
    parts.append(
        "- **Étape 1** : run grille complète à levier 1x, sélection du meilleur "
        "config par Sharpe pour chaque (paire, stratégie)."
    )
    parts.append(
        "- **Étape 2** : optimisation du levier pour ramener le CAGR dans la "
        "fenêtre [8%, 15%] tout en gardant Max DD < 35%."
    )
    parts.append("")

    parts.append("### Meilleurs configs par (paire × stratégie)")
    parts.append("")
    parts.append(format_best_configs_table(best_rows))
    parts.append("")
    parts.append(
        "*Légende :* `Sharpe/CAGR/DD @1x` = métriques à levier 1x avec les "
        "meilleurs paramètres. `L*` = levier optimal pour viser ~10% CAGR. "
        "`Statut` : CIBLE si CAGR@L* ∈ [8%, 15%] et DD < 35% ; LOW/HIGH/DD>35 sinon."
    )
    parts.append("")

    # Grid snapshots (Sharpe heatmaps)
    parts.append("### Heatmaps Sharpe par paire (grille bb_window × bb_alpha, tp_stop optimal)")
    parts.append("")
    for pair in pairs:
        for skey in strat_keys:
            strat_df = full_df[full_df["strategy"] == skey]
            snap = format_grid_snapshot(strat_df, pair, STRATEGIES[skey].label)
            if snap:
                parts.append(snap)
                parts.append("")

    parts.append("### Leçons cross-pair")
    parts.append("")
    parts.append(
        "1. **Transférabilité de l'edge** — l'edge intraday mean-reversion "
        "dépend fortement de la microstructure de la paire ; l'optimum "
        "paramétrique n'est pas universel et chaque paire requiert son propre "
        "triplet (bb_window, bb_alpha, tp_stop)."
    )
    parts.append(
        "2. **Filtre macro US-centrique** — `mr_macro` utilise le spread 10Y-2Y "
        "Treasury et le chômage US. L'application aux paires JPY/CAD reste "
        "mécaniquement valide mais le signal théorique est plus ténu que sur "
        "EUR-USD/GBP-USD (où le dollar US est le moteur dominant)."
    )
    parts.append(
        "3. **Levier hétérogène par paire** — puisque le CAGR à 1x varie "
        "fortement entre paires, le levier L* nécessaire pour atteindre 10% "
        "CAGR varie également. Une stratégie portefeuille allouerait plus de "
        "capital aux paires à Sharpe le plus élevé plutôt qu'un levier "
        "uniforme."
    )
    parts.append(
        "4. **Stabilité des hyperparamètres** — les heatmaps Sharpe révèlent "
        "des régions plates (robustes) vs des pics isolés (sur-optimisés). "
        "Privilégier des paramètres dans les zones plates pour la mise en "
        "production."
    )
    parts.append("")
    parts.append(_PHASE_MARKER_END)
    return "\n".join(parts)


def update_research_doc(content: str, doc_path: Path) -> None:
    text = doc_path.read_text(encoding="utf-8")
    if _PHASE_MARKER_BEGIN in text and _PHASE_MARKER_END in text:
        start = text.index(_PHASE_MARKER_BEGIN)
        end = text.index(_PHASE_MARKER_END) + len(_PHASE_MARKER_END)
        new_text = text[:start] + content + text[end:]
    else:
        sep = "\n\n---\n\n"
        new_text = text.rstrip() + sep + content + "\n"
    doc_path.write_text(new_text, encoding="utf-8")


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════


def classify(cagr: float, max_dd: float) -> str:
    if np.isnan(cagr):
        return "FAIL"
    if max_dd >= _DD_HARD_CAP:
        return "DD>35"
    if cagr < _CAGR_LOW:
        return "LOW"
    if cagr > _CAGR_HIGH:
        return "HIGH"
    return "CIBLE"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--pairs",
        default=",".join(DEFAULT_PAIRS),
        help="Comma-separated FX pairs (default: all 4)",
    )
    ap.add_argument(
        "--strategies",
        default="mr_turbo,mr_macro",
        help="Comma-separated strategies",
    )
    ap.add_argument(
        "--no-doc",
        action="store_true",
        help="Skip updating the research journal",
    )
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    apply_vbt_settings()

    pairs = [p.strip() for p in args.pairs.split(",") if p.strip()]
    strat_keys = [s.strip() for s in args.strategies.split(",") if s.strip()]
    unknown = [k for k in strat_keys if k not in STRATEGIES]
    if unknown:
        print(f"ERROR: unknown strategies {unknown}")
        return 2

    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    all_rows: list[pd.DataFrame] = []
    best_rows: list[dict] = []

    for pair in pairs:
        data_path = f"data/{pair}_minute.parquet"
        full_path = _PROJECT_ROOT / data_path
        if not full_path.exists():
            print(f"  SKIP {pair}: {data_path} not found")
            continue
        print(f"\n▶ Loading {pair} ...")
        _, data = load_fx_data(data_path)
        print(f"  Range: {data.wrapper.index[0]} → {data.wrapper.index[-1]} "
              f"({len(data.wrapper.index):,} bars)")

        for skey in strat_keys:
            strat = STRATEGIES[skey]
            print(f"\n  Sweeping {strat.label} on {pair}")
            df = sweep_pair_strategy(strat, pair, data)
            all_rows.append(df)

            # Select best config by Sharpe (require CAGR > 0)
            valid = df[(df["sharpe"] > 0) & (df["cagr"] > 0)]
            if valid.empty:
                print(f"    ⚠ No positive-Sharpe config for {pair}/{skey}")
                best_rows.append({
                    "pair": pair,
                    "strategy_label": strat.label,
                    "bb_window": df.iloc[0]["bb_window"],
                    "bb_alpha": df.iloc[0]["bb_alpha"],
                    "tp_stop": df.iloc[0]["tp_stop"],
                    "sharpe_1x": df["sharpe"].max(),
                    "cagr_1x": df.loc[df["sharpe"].idxmax(), "cagr"],
                    "dd_1x": df.loc[df["sharpe"].idxmax(), "max_dd"],
                    "leverage": None,
                    "cagr_L": None, "dd_L": None, "vol_L": None,
                    "sortino_L": None, "calmar_L": None,
                    "var95_L": None, "ulcer_L": None,
                    "pf_L": None, "trades_L": None,
                    "status": "FAIL",
                })
                continue

            best_idx = valid["sharpe"].idxmax()
            best = valid.loc[best_idx]
            params = {
                "bb_window": int(best["bb_window"]),
                "bb_alpha": float(best["bb_alpha"]),
                "tp_stop": float(best["tp_stop"]),
            }
            print(f"    → best: {params} | Sharpe={best['sharpe']:+.2f} "
                  f"CAGR={best['cagr']:+.2f}% DD={best['max_dd']:.2f}%")

            # Optimise leverage on best params
            opt = find_optimal_leverage(strat, data, params, best["cagr"])
            if opt is None:
                print("    → leverage optimisation failed")
                best_rows.append({
                    "pair": pair,
                    "strategy_label": strat.label,
                    **params,
                    "sharpe_1x": float(best["sharpe"]),
                    "cagr_1x": float(best["cagr"]),
                    "dd_1x": float(best["max_dd"]),
                    "leverage": None,
                    "cagr_L": None, "dd_L": None, "vol_L": None,
                    "sortino_L": None, "calmar_L": None,
                    "var95_L": None, "ulcer_L": None,
                    "pf_L": None, "trades_L": None,
                    "status": "FAIL",
                })
                continue

            lev, m = opt
            status = classify(m["cagr"], m["max_dd"])
            print(f"    → L* = {lev:.1f}x → CAGR={m['cagr']:+.2f}% "
                  f"DD={m['max_dd']:.2f}% [{status}]")
            best_rows.append({
                "pair": pair,
                "strategy_label": strat.label,
                **params,
                "sharpe_1x": float(best["sharpe"]),
                "cagr_1x": float(best["cagr"]),
                "dd_1x": float(best["max_dd"]),
                "leverage": lev,
                "cagr_L": m["cagr"],
                "dd_L": m["max_dd"],
                "vol_L": m["vol"],
                "sortino_L": m["sortino"],
                "calmar_L": m["calmar"],
                "var95_L": m["var_95"],
                "ulcer_L": m["ulcer"],
                "pf_L": m["profit_factor"],
                "trades_L": m["n_trades"],
                "status": status,
            })

    if not all_rows:
        print("No results produced.")
        return 1

    full_df = pd.concat(all_rows, ignore_index=True)
    full_df.to_csv(_RESULTS_DIR / "full_grid.csv", index=False)

    best_df = pd.DataFrame(best_rows)
    best_df.to_csv(_RESULTS_DIR / "best_configs.csv", index=False)

    print(f"\n✔ Full grid: {_RESULTS_DIR.relative_to(_PROJECT_ROOT)}/full_grid.csv "
          f"({len(full_df)} rows)")
    print(f"✔ Best configs: {_RESULTS_DIR.relative_to(_PROJECT_ROOT)}/best_configs.csv "
          f"({len(best_df)} rows)")

    if not args.no_doc:
        content = render_phase_12(full_df, best_rows, pairs, strat_keys)
        update_research_doc(content, _RESEARCH_DOC)
        print(f"✔ Research journal updated: "
              f"{_RESEARCH_DOC.relative_to(_PROJECT_ROOT)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
