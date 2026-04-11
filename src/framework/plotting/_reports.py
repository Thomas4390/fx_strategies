"""Text-based trade reports, extended-stats printing, and terminal
pretty-print helpers for grid-search / CV results.

Extracted from the monolithic ``_core.py`` to keep each plotting module
under the 800-line rule. All functions are self-contained and only
depend on ``numpy``, ``pandas``, ``vectorbtpro``, and the local
``tabulate`` import inside each function body.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import vectorbtpro as vbt


# ═══════════════════════════════════════════════════════════════════════
# TRADE REPORT (text-based stats)
# ═══════════════════════════════════════════════════════════════════════


def _stats_to_rows(stats: pd.Series) -> list[list[str]]:
    """Convert a ``pd.Series`` of stats to ``[[label, value], ...]`` rows
    with nicely formatted values (durations, percentages, floats).
    """
    rows: list[list[str]] = []
    for label, val in stats.items():
        if val is None:
            rows.append([str(label), "—"])
            continue
        if isinstance(val, pd.Timedelta):
            total_min = int(val.total_seconds() / 60)
            if total_min < 60:
                rows.append([str(label), f"{total_min} min"])
            elif total_min < 1440:
                rows.append([str(label), f"{total_min / 60:.1f} h"])
            else:
                rows.append([str(label), f"{total_min / 1440:.1f} d"])
            continue
        if isinstance(val, pd.Timestamp):
            rows.append([str(label), val.strftime("%Y-%m-%d %H:%M")])
            continue
        try:
            f = float(val)
        except (TypeError, ValueError):
            rows.append([str(label), str(val)])
            continue
        if np.isnan(f):
            rows.append([str(label), "—"])
            continue
        if abs(f) >= 10_000:
            rows.append([str(label), f"{f:,.2f}"])
        else:
            rows.append([str(label), f"{f:.2f}"])
    return rows


def _box_title(title: str, width: int = 78) -> str:
    """Return a Unicode-boxed title header line block."""
    title = title.strip()
    inner = f"  {title}  "
    pad = max(width - len(inner) - 2, 0)
    top = "╔" + "═" * (width - 2) + "╗"
    mid = "║" + inner + " " * pad + "║"
    bot = "╠" + "═" * (width - 2) + "╣"
    return f"{top}\n{mid}\n{bot}"


def build_trade_report(pf: vbt.Portfolio, name: str = "Strategy") -> str:
    """Build a text report with portfolio stats and trade stats,
    rendered as aligned tabulate boxes with section headers.

    ``pf.returns_stats()`` is skipped to avoid the heavy returns
    computation.
    """
    from tabulate import tabulate

    sections: list[str] = []
    sections.append(_box_title(f"{name} — Backtest Report"))

    # ---- Portfolio stats ----
    try:
        stats = pf.stats()
        if isinstance(stats, pd.DataFrame):
            stats = stats.iloc[:, 0]
        rows = _stats_to_rows(stats)
        sections.append("\n  ── Portfolio Stats ──")
        sections.append(
            tabulate(rows, headers=["Metric", "Value"], tablefmt="rounded_outline")
        )
    except Exception as e:
        sections.append(f"\n  [portfolio stats failed: {e}]")

    # ---- Trade stats ----
    trade_count = pf.trades.count()
    if isinstance(trade_count, pd.Series):
        has_trades = bool((trade_count > 0).any())
    else:
        has_trades = trade_count > 0

    if has_trades:
        try:
            ts = pf.trades.stats()
            if isinstance(ts, pd.DataFrame):
                ts = ts.iloc[:, 0]
            rows = _stats_to_rows(ts)
            sections.append("\n  ── Trade Stats ──")
            sections.append(
                tabulate(rows, headers=["Metric", "Value"], tablefmt="rounded_outline")
            )
        except Exception as e:
            sections.append(f"\n  [trade stats failed: {e}]")

        # Quick win/loss summary directly from trade records
        try:
            pnls = np.asarray(pf.trades.pnl.values).ravel()
            pnls = pnls[~np.isnan(pnls)]
            if pnls.size:
                wins = pnls[pnls > 0]
                losses = pnls[pnls < 0]
                extra = [
                    ["Mean PnL", f"{pnls.mean():,.2f}"],
                    ["Median PnL", f"{np.median(pnls):,.2f}"],
                    ["Std PnL", f"{pnls.std():,.2f}"],
                    ["Largest Win", f"{pnls.max():,.2f}"],
                    ["Largest Loss", f"{pnls.min():,.2f}"],
                    ["Avg Win", f"{wins.mean():,.2f}" if wins.size else "—"],
                    ["Avg Loss", f"{losses.mean():,.2f}" if losses.size else "—"],
                    ["Win / Loss ratio",
                     f"{abs(wins.mean() / losses.mean()):.2f}"
                     if wins.size and losses.size else "—"],
                ]
                sections.append("\n  ── Trade Distribution ──")
                sections.append(
                    tabulate(extra, headers=["Metric", "Value"], tablefmt="rounded_outline")
                )
        except Exception:
            pass

    return "\n".join(sections)


# ═══════════════════════════════════════════════════════════════════════
# EXTENDED STATS (text output)
# ═══════════════════════════════════════════════════════════════════════


def _total_trade_count(pf: vbt.Portfolio) -> int:
    """Return total number of trades across all columns/groups."""
    cnt = pf.trades.count()
    if hasattr(cnt, "sum"):
        try:
            return int(cnt.sum())
        except Exception:
            pass
    try:
        return int(cnt)
    except Exception:
        return 0


def print_extended_stats(pf: vbt.Portfolio, name: str = "Strategy") -> None:
    """Print comprehensive stats: portfolio, returns, trades, drawdowns.

    Handles multi-column / grouped portfolios: uses ``.sum()`` on
    trade counts and shows the per-column table rather than crashing
    on ambiguous Series truthiness.
    """
    print(f"\n{'=' * 60}")
    print(f"  {name} — Extended Statistics")
    print(f"{'=' * 60}")

    print(f"\n--- Portfolio Stats ---")
    print(pf.stats().to_string())

    print(f"\n--- Returns Stats ---")
    print(pf.returns_stats().to_string())

    n_trades = _total_trade_count(pf)
    if n_trades > 0:
        print(f"\n--- Trade Stats ---")
        print(pf.trades.stats().to_string())

        # Per-column aggregated trade metrics (works for single or multi-col)
        try:
            pnls = np.asarray(pf.trades.pnl.values).ravel()
            pnls = pnls[~np.isnan(pnls)]
            if pnls.size:
                print(f"\n--- Trade Distribution ---")
                print(f"  Mean PnL: {pnls.mean():.2f}")
                print(f"  Median PnL: {np.median(pnls):.2f}")
                print(f"  Skew: {pd.Series(pnls).skew():.2f}")
                print(f"  Kurtosis: {pd.Series(pnls).kurtosis():.2f}")
        except Exception as e:
            print(f"  (trade distribution skipped: {e})")

    try:
        dd = pf.drawdowns
        dd_count = dd.count()
        if hasattr(dd_count, "sum"):
            dd_count = int(dd_count.sum())
        if dd_count > 0:
            print(f"\n--- Drawdown Stats ---")
            print(dd.stats().to_string())
    except Exception:
        pass


# ═══════════════════════════════════════════════════════════════════════
# TERMINAL PRETTY-PRINT HELPERS (grid-search + CV results)
# ═══════════════════════════════════════════════════════════════════════


def _format_metric_header(metric_name: str) -> str:
    """Convert a snake_case metric name into a Title Case header."""
    return metric_name.replace("_", " ").title()


def _format_cell(val: Any) -> str:
    """Always format numeric values with 2 decimal places."""
    if val is None:
        return "—"
    try:
        f = float(val)
    except (TypeError, ValueError):
        return str(val)
    if np.isnan(f):
        return "—"
    if abs(f) >= 10_000:
        return f"{f:,.2f}"
    return f"{f:.2f}"


def print_grid_results(
    grid: pd.Series,
    *,
    title: str = "Grid Search Results",
    metric_name: str = "metric",
    top_n: int = 20,
    ascending: bool = False,
) -> None:
    """Pretty-print a grid-search ``pd.Series`` to the terminal.

    Uses ``tabulate`` for a clean aligned layout. The index levels
    become the left-hand columns and the metric becomes the right-hand
    column. Shows a header with summary statistics (best/median/worst).
    """
    from tabulate import tabulate

    if not isinstance(grid, pd.Series) or len(grid) == 0:
        print(f"\n(empty {title})")
        return

    sorted_grid = grid.sort_values(ascending=ascending)
    head = sorted_grid.head(top_n)
    df = head.reset_index()
    metric_col = df.columns[-1]
    df.rename(columns={metric_col: _format_metric_header(metric_name)}, inplace=True)
    for col in df.columns[:-1]:
        df[col] = df[col].apply(
            lambda v: f"{v:g}" if isinstance(v, (int, float, np.floating)) else str(v)
        )
    df[df.columns[-1]] = df[df.columns[-1]].apply(_format_cell)

    bar = "═" * 78
    print(f"\n{bar}")
    print(f"  {title}")
    print(f"{bar}")
    print(
        f"  Combos: {len(grid)}   "
        f"Best: {_format_cell(sorted_grid.iloc[0])}   "
        f"Median: {_format_cell(sorted_grid.median())}   "
        f"Worst: {_format_cell(sorted_grid.iloc[-1])}"
    )
    print(f"  Top {min(top_n, len(grid))} combos by {metric_name}:")
    print()
    print(tabulate(df, headers="keys", tablefmt="rounded_outline", showindex=False))
    print()


def print_cv_results(
    grid_perf: pd.Series,
    best_perf: pd.Series | None = None,
    splitter: Any | None = None,
    *,
    title: str = "Cross-Validation Results",
    metric_name: str = "metric",
    top_n: int = 5,
) -> None:
    """Pretty-print CV grid + best-per-split results.

    Displays two sections:
      1. Best combo per (split, set) — typically best on train per fold.
      2. Aggregated ranking across splits (mean metric per combo).
    """
    from tabulate import tabulate

    if not isinstance(grid_perf, pd.Series) or len(grid_perf) == 0:
        print(f"\n(empty {title})")
        return

    names = list(grid_perf.index.names)
    has_split = "split" in names
    has_set = "set" in names

    date_labels: dict[int, str] = {}
    if splitter is not None and has_split:
        try:
            bounds = splitter.index_bounds
            for (split_i, set_i), row in bounds.iterrows():
                if set_i in ("test", 1):
                    start = pd.Timestamp(row["start"]).strftime("%Y-%m-%d")
                    end = pd.Timestamp(row["end"]).strftime("%Y-%m-%d")
                    date_labels[split_i] = f"{start} → {end}"
        except Exception:
            pass

    bar = "═" * 78
    print(f"\n{bar}")
    print(f"  {title}")
    print(f"{bar}")
    print(
        f"  Splits: {len(date_labels) or '?'}   "
        f"Combos: {len(grid_perf)}   "
        f"Metric: {metric_name}"
    )

    # -------- Section 1: best per split (from best_perf if provided) --------
    if best_perf is not None and len(best_perf) > 0 and has_split:
        print(f"\n  ── Best combo per split (test range) ──")
        rows = []
        bp = best_perf
        if has_set:
            try:
                bp = bp.xs("test", level="set")
            except (KeyError, ValueError):
                try:
                    bp = bp.xs(1, level="set")
                except Exception:
                    pass
        for idx, val in bp.items():
            if not isinstance(idx, tuple):
                idx = (idx,)
            split_val = idx[0]
            param_vals = idx[1:]
            param_names = [n for n in bp.index.names if n != "split"]
            date_str = date_labels.get(split_val, f"Split {split_val}")
            params_str = ", ".join(
                f"{n}={v}" for n, v in zip(param_names, param_vals)
            )
            rows.append([date_str, params_str, _format_cell(val)])
        print(
            tabulate(
                rows,
                headers=["Test Range", "Best Params", _format_metric_header(metric_name)],
                tablefmt="rounded_outline",
            )
        )

    # -------- Section 2: aggregated ranking across splits --------
    sweep_levels = [
        n for n in names if n not in ("split", "set")
    ]
    if sweep_levels:
        test_perf = grid_perf
        if has_set:
            try:
                test_perf = test_perf.xs("test", level="set")
            except (KeyError, ValueError):
                try:
                    test_perf = test_perf.xs(1, level="set")
                except Exception:
                    pass
        agg = test_perf.groupby(sweep_levels).agg(["mean", "std", "min", "max"])
        agg = agg.sort_values("mean", ascending=False).head(top_n)
        df = agg.reset_index()
        for col in sweep_levels:
            df[col] = df[col].apply(lambda v: f"{v:g}" if isinstance(v, (int, float, np.floating)) else str(v))
        for col in ["mean", "std", "min", "max"]:
            df[col] = df[col].apply(_format_cell)
        print(f"\n  ── Top {top_n} combos by mean test {metric_name} (across all splits) ──")
        print(tabulate(df, headers="keys", tablefmt="rounded_outline", showindex=False))
    print()
