"""La table de coûts par symbole ne doit pas dériver de ses parquets sources.

`costs.yml` remplace le `slippage: 0.0001` global de `vbt.yml` par un
demi-spread empirique par symbole. Deux façons de se tromper en silence :
un point size faux (un facteur 100 sur les paires JPY passerait inaperçu
dans un Sharpe) et un `costs.yml` régénéré à la main. Ces tests ancrent
l'ordre de grandeur sur deux oracles vérifiés (EUR-USD ≈ 1 bp de spread,
USD-JPY du même ordre) et vérifient que `--check` voit la dérive.
"""
from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
_SCRIPT = _ROOT / "scripts" / "build_cost_table.py"
_COSTS = _ROOT / "costs.yml"

_FX_PAIRS = (
    "EUR-USD",
    "GBP-USD",
    "USD-JPY",
    "USD-CAD",
    "USD-CHF",
    "AUD-USD",
    "NZD-USD",
    "EUR-GBP",
    "EUR-JPY",
    "GBP-JPY",
)


def _load_builder():
    if not _SCRIPT.exists():  # pragma: no cover
        pytest.skip(f"{_SCRIPT} absent")
    spec = importlib.util.spec_from_file_location("build_cost_table", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def builder():
    return _load_builder()


@pytest.fixture(scope="module")
def table(builder):
    return builder.build_cost_table()


def test_table_covers_the_ten_pairs_plus_gold(table):
    # Subset, not equality: the table also carries whatever non-FX symbols
    # have been exported from the broker (silver, energies, cash indices).
    assert set(_FX_PAIRS) | {"XAU-USD"} <= set(table)


def test_fx_entries_have_the_full_schema(table):
    expected = {
        "spread_points_median",
        "spread_frac_median",
        "spread_frac_p75",
        "half_spread_frac",
        "n_bars",
        "from",
        "to",
    }
    for pair in _FX_PAIRS:
        assert set(table[pair]) == expected, pair


def test_gold_entry_comes_from_the_spec(table):
    """Pas de parquet _mt5 pour l'or : 1 bp par ordre, gold_momentum_spec §8."""
    assert table["XAU-USD"] == {"half_spread_frac": 0.0001, "source": "spec"}


def test_eurusd_matches_the_empirical_spread_oracle(table):
    """Spread médian 11 points à ~1.0979 → ~1 bp, donc ~0.5 bp par fill."""
    entry = table["EUR-USD"]
    assert entry["spread_points_median"] == 11.0
    assert 0.3e-4 <= entry["half_spread_frac"] <= 0.7e-4
    assert entry["spread_frac_p75"] >= entry["spread_frac_median"]


def test_usdjpy_point_size_is_the_three_digit_one(table):
    """Un point à 1e-5 au lieu de 1e-3 ferait tomber le coût d'un facteur 100."""
    entry = table["USD-JPY"]
    assert entry["spread_points_median"] == 13.0
    assert 0.3e-4 <= entry["half_spread_frac"] <= 0.7e-4


def test_half_spread_is_half_the_median(table):
    for pair in _FX_PAIRS:
        entry = table[pair]
        assert entry["half_spread_frac"] == pytest.approx(
            entry["spread_frac_median"] / 2, rel=1e-6
        )


def test_cost_for_reads_the_generated_file():
    from framework.costs import cost_for, load_cost_table

    load_cost_table.cache_clear()
    assert cost_for("EUR-USD") > 0
    assert cost_for("XAU-USD") == 0.0001
    assert cost_for("EUR-USD", stat="spread_frac_p75") > cost_for("EUR-USD")


def test_cost_for_unknown_symbol_points_at_the_generator():
    from framework.costs import cost_for

    with pytest.raises(KeyError, match="scripts/build_cost_table.py"):
        cost_for("ZZZ")


def test_check_mode_is_green_right_after_generation():
    assert _COSTS.exists(), "lance scripts/build_cost_table.py d'abord"
    proc = subprocess.run(
        [sys.executable, str(_SCRIPT), "--check"],
        cwd=_ROOT,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
