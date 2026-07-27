"""Generate/refresh ``costs.yml`` from the empirical MT5 broker spreads.

Every ``data/<PAIR>_daily_mt5.parquet`` carries a ``spread`` column in
**broker points** (integer, one value per daily bar). Backtests need a
price-relative cost instead, so this script converts points to a fraction
of price and writes one entry per symbol to ``costs.yml`` at the project
root. Consumers read it through ``framework.costs.cost_for`` to replace
the single global ``slippage: 0.0001`` of ``vbt.yml`` by a per-symbol
figure.

Two conventions worth stating explicitly :

- **Per-bar median, not ratio of medians.** ``spread_frac_median`` is
  ``median(spread * point / close)`` computed bar by bar, so a bar with a
  wide spread is weighted at the price that actually prevailed on that
  bar. The cruder ``median(spread) * point / median(close)`` gives a
  slightly different number (1.00 bp vs 0.96 bp on EUR-USD) because
  spread and price co-move.
- **Half spread as the per-fill cost.** A round trip pays the full
  spread; a single fill pays half of it. VBT applies ``slippage`` per
  fill, hence ``half_spread_frac = spread_frac_median / 2``.

Swap and commission are deliberately out of scope: this table is a
screening-level friction model only.

Usage
-----
    python scripts/build_cost_table.py           # refresh in place
    python scripts/build_cost_table.py --check   # non-zero exit if drift
    python scripts/build_cost_table.py --dry-run # print, no write
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DATA_DIR = _PROJECT_ROOT / "data"
_COSTS_PATH = _PROJECT_ROOT / "costs.yml"

# Point size per symbol, tabulated on purpose rather than inferred from the
# quote magnitude — a magnitude heuristic silently breaks on any symbol the
# broker reprices. The Phase 1 broker catalogue will verify these values
# against the MT5 ``SYMBOL_POINT`` field.
_POINT_BY_SYMBOL: dict[str, float] = {
    # 5-digit majors / crosses
    "EUR-USD": 1e-5,
    "GBP-USD": 1e-5,
    "USD-CAD": 1e-5,
    "USD-CHF": 1e-5,
    "AUD-USD": 1e-5,
    "NZD-USD": 1e-5,
    "EUR-GBP": 1e-5,
    # 3-digit, JPY-quoted
    "USD-JPY": 1e-3,
    "EUR-JPY": 1e-3,
    "GBP-JPY": 1e-3,
    # 3-digit metals and energies — verified against the broker catalogue
    # (symbols_catalog.csv, SYMBOL_POINT) on 2026-07-27.
    "XAU-USD": 1e-3,
    "XAG-USD": 1e-3,
    "XTI-USD": 1e-3,
    "XBR-USD": 1e-3,
    "XNG-USD": 1e-3,
    # 2-digit cash index CFDs
    "US500": 1e-2,
    "US100": 1e-2,
    "US30": 1e-2,
    "GER40": 1e-2,
    "JPN225": 1e-2,
    "UK100": 1e-2,
}

# Gold has no ``_mt5`` parquet: the spec fixes the friction at 1 bp per order.
# See docs/specs/gold_momentum_spec.md §8.
_SPEC_ENTRIES: dict[str, dict[str, Any]] = {
    "XAU-USD": {
        "half_spread_frac": 0.0001,
        "source": "spec",
    },
}

# Significant digits kept in the YAML. Rounding on *significant* digits and
# not decimals matters: the fractions live around 1e-4, so ``round(x, 10)``
# would keep only six meaningful digits and break the
# ``half_spread_frac == spread_frac_median / 2`` identity.
_ROUND_SIGNIF = 10

_HEADER = (
    "# GÉNÉRÉ par scripts/build_cost_table.py — ne pas éditer.\n"
    "# Coûts de transaction par symbole, dérivés des spreads broker MT5\n"
    "# (colonne `spread` en points des parquets data/<PAIR>_daily_mt5.parquet).\n"
    "# `half_spread_frac` est le coût par fill à passer en slippage VBT.\n"
    "# Ni swap ni commission ici — modèle de friction de screening uniquement.\n"
)


def _symbol_from_path(path: Path) -> str:
    """``data/EUR-USD_daily_mt5.parquet`` → ``EUR-USD``."""
    return path.name.split("_")[0]


def _round_signif(value: float) -> float:
    return float(f"{value:.{_ROUND_SIGNIF}g}")


def _symbol_entry(path: Path, point: float) -> dict[str, Any]:
    df = pd.read_parquet(path, columns=["close", "spread"])
    spread_frac = df["spread"] * point / df["close"]
    frac_median = float(spread_frac.median())
    return {
        "spread_points_median": float(df["spread"].median()),
        "spread_frac_median": _round_signif(frac_median),
        "spread_frac_p75": _round_signif(float(spread_frac.quantile(0.75))),
        "half_spread_frac": _round_signif(frac_median / 2),
        "n_bars": int(len(df)),
        "from": str(df.index.min().date()),
        "to": str(df.index.max().date()),
    }


def build_cost_table() -> dict[str, dict[str, Any]]:
    """Walk ``data/*_daily_mt5.parquet`` and return a fresh cost table."""
    table: dict[str, dict[str, Any]] = {}
    for path in sorted(_DATA_DIR.glob("*_daily_mt5.parquet")):
        symbol = _symbol_from_path(path)
        point = _POINT_BY_SYMBOL.get(symbol)
        if point is None:
            raise KeyError(
                f"No point size tabulated for {symbol!r} — add it to "
                "_POINT_BY_SYMBOL in scripts/build_cost_table.py."
            )
        table[symbol] = _symbol_entry(path, point)
    for symbol, entry in _SPEC_ENTRIES.items():
        table[symbol] = dict(entry)
    return dict(sorted(table.items()))


def load_cost_table() -> dict[str, dict[str, Any]] | None:
    """Read ``costs.yml`` if present."""
    if not _COSTS_PATH.exists():
        return None
    with _COSTS_PATH.open("r") as fh:
        return yaml.safe_load(fh)


def save_cost_table(table: dict[str, dict[str, Any]]) -> None:
    with _COSTS_PATH.open("w") as fh:
        fh.write(_HEADER)
        yaml.safe_dump(table, fh, sort_keys=False, default_flow_style=False)


def diff_tables(
    old: dict[str, dict[str, Any]], new: dict[str, dict[str, Any]]
) -> list[str]:
    """Return a flat list of human-readable differences."""
    diffs: list[str] = []
    for symbol in sorted(set(new) - set(old)):
        diffs.append(f"+ {symbol}  (new symbol)")
    for symbol in sorted(set(old) - set(new)):
        diffs.append(f"- {symbol}  (removed)")
    for symbol in sorted(set(old) & set(new)):
        for key in sorted(set(old[symbol]) | set(new[symbol])):
            old_val = old[symbol].get(key)
            new_val = new[symbol].get(key)
            if old_val != new_val:
                diffs.append(f"~ {symbol}.{key}: {old_val} → {new_val}")
    return diffs


def _fmt(value: float | int | None, digits: int) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def _fmt_bp(value: float | None) -> str:
    """Fraction of price → basis points, for display only."""
    return "—" if value is None else f"{value * 1e4:.2f}"


def format_table(table: dict[str, dict[str, Any]]) -> str:
    """Render the cost table as an aligned text block (bp units)."""
    header = (
        f"{'symbol':<9}{'pts':>7}{'spread bp':>11}{'p75 bp':>9}"
        f"{'half bp':>9}{'bars':>7}  range"
    )
    lines = [header, "-" * len(header)]
    for symbol, entry in table.items():
        pts = _fmt(entry.get("spread_points_median"), 1)
        frac = _fmt_bp(entry.get("spread_frac_median"))
        p75 = _fmt_bp(entry.get("spread_frac_p75"))
        half = _fmt_bp(entry.get("half_spread_frac"))
        bars = _fmt(entry.get("n_bars"), 0)
        rng = (
            f"{entry['from']} → {entry['to']}"
            if "from" in entry
            else f"source: {entry.get('source', 'n/a')}"
        )
        lines.append(
            f"{symbol:<9}{pts:>7}{frac:>11}{p75:>9}{half:>9}{bars:>7}  {rng}"
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit non-zero if costs.yml is out of date.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the cost table instead of writing it.",
    )
    args = parser.parse_args()

    new_table = build_cost_table()

    if args.check:
        old = load_cost_table()
        if old is None:
            print("ERROR: costs.yml is missing — run without --check to create.")
            return 2
        diffs = diff_tables(old, new_table)
        if diffs:
            print("Cost table drift detected:")
            for d in diffs:
                print(f"  {d}")
            return 1
        print(f"OK — costs.yml up to date ({len(new_table)} symbols).")
        return 0

    if args.dry_run:
        print(format_table(new_table))
        return 0

    old = load_cost_table()
    save_cost_table(new_table)
    print(format_table(new_table))
    print(f"\nCost table written → {_COSTS_PATH}  ({len(new_table)} symbols)")
    if old is not None:
        diffs = diff_tables(old, new_table)
        if diffs:
            print("\nDiff vs previous table:")
            for d in diffs:
                print(f"  {d}")
        else:
            print("\nNo changes vs previous table.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
