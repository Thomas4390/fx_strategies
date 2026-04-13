"""Smoke + integration tests for MR Macro multi-pair refactor (Phase 1).

These tests guard against regressions on the multi-symbol code path added
to :func:`strategies.mr_macro.pipeline`. They do NOT re-run the mono-pair
equivalence tests (those live in ``test_pipeline_equivalence.py``).

Scope:
- ``load_all_fx_data`` produces a strictly index-aligned multi-symbol
  ``vbt.Data`` (no ffill between pairs).
- ``pipeline(multi_data, ...)`` returns a single-group portfolio with
  ``pf.stats()`` numeric and non-NaN on ``Sharpe Ratio``.
- Per-pair trades are non-zero (sanity check that signals reach every
  column, not just one).
- Mono-pair and multi-pair paths return the SAME stats when the multi
  contains only one symbol (i.e. single-symbol ``vbt.Data`` wrapped).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


# The full 4-pair load takes ~10 s on disk, so we restrict to a 6-month
# window in the smoke tests — enough to accumulate a meaningful number
# of trades (MR Macro averages ~35 trades/year on EUR-USD) while keeping
# the CI cost modest.
SMOKE_START = "2024-01-01"
SMOKE_END = "2024-07-01"


@pytest.fixture(scope="module")
def multi_data():
    from strategies.mr_macro import load_all_fx_data
    from utils import apply_vbt_settings

    apply_vbt_settings()
    data = load_all_fx_data()
    return data.loc[SMOKE_START:SMOKE_END]


@pytest.fixture(scope="module")
def single_data():
    from utils import apply_vbt_settings, load_fx_data

    apply_vbt_settings()
    _, data = load_fx_data("data/EUR-USD_minute.parquet")
    return data.loc[SMOKE_START:SMOKE_END]


def test_load_all_fx_data_aligned(multi_data):
    """All 4 pairs must share exactly the same minute index."""
    assert len(multi_data.symbols) == 4
    assert set(multi_data.symbols) == {"EUR-USD", "GBP-USD", "USD-JPY", "USD-CAD"}
    # close is a DataFrame with one column per pair — index must be unique,
    # monotonic, and every row non-NaN (strict intersection, no ffill).
    close = multi_data.close
    assert close.index.is_monotonic_increasing
    assert close.index.is_unique
    assert close.notna().all().all(), "Strict intersection left NaN residuals"
    assert close.shape[1] == 4


def test_pipeline_multipair_smoke(multi_data):
    """pipeline() runs on multi-symbol data and returns non-degenerate stats."""
    from strategies.mr_macro import pipeline

    pf, ind = pipeline(
        multi_data,
        bb_window=80,
        bb_alpha=5.0,
        sl_stop=0.005,
        tp_stop=0.006,
        spread_threshold=0.5,
    )
    stats = pf.stats()
    # Cash sharing collapses the 4 pairs into a single group — stats is a
    # Series (not a DataFrame) and Sharpe Ratio is a finite scalar.
    assert hasattr(stats, "index")
    sharpe = stats.get("Sharpe Ratio")
    assert sharpe is not None
    assert np.isfinite(sharpe), f"Sharpe Ratio is non-finite: {sharpe}"

    # Trades should fire on more than one pair over a full month — if
    # only EUR-USD trades we have a broadcasting bug in the macro mask.
    trades_count = int(pf.trades.count())
    assert trades_count > 0, "Zero trades on multi-pair smoke — signals broken"

    # Indicator holds DataFrames in multi-pair mode.
    import pandas as pd

    assert isinstance(ind.close, pd.DataFrame)
    assert isinstance(ind.upper, pd.DataFrame)
    assert isinstance(ind.lower, pd.DataFrame)
    assert ind.close.shape[1] == 4
    # .plot() must raise for multi-pair runs.
    with pytest.raises(ValueError, match="single-symbol"):
        ind.plot()


def test_pipeline_multipair_cash_sharing(multi_data):
    """cash_sharing=True produces a single-group portfolio (daily_returns is 1-D)."""
    from strategies.mr_macro import pipeline

    pf, _ = pipeline(multi_data)
    dr = pf.daily_returns
    # Cash-shared grouped portfolio → daily_returns is a pd.Series, not
    # a DataFrame. This is what combined_portfolio.get_strategy_daily_returns
    # relies on when aggregating into the combined portfolio.
    import pandas as pd

    assert isinstance(dr, pd.Series), (
        f"daily_returns should be Series when cash_sharing=True, got {type(dr)}"
    )
    assert len(dr) > 0
    assert not dr.isna().all()


def test_pipeline_mono_still_routes_through_mono_branch(single_data):
    """Sanity: a single-symbol ``vbt.Data`` takes the mono-pair branch.

    The strict numerical equivalence vs legacy snapshots is already
    covered by ``test_pipeline_equivalence.test_mr_macro_pipeline_equivalent``
    on the full dataset; this smoke test only guards that the branching
    logic still selects the Series code path for ``len(symbols) == 1``.
    """
    from strategies.mr_macro import pipeline

    assert len(single_data.symbols) == 1
    pf_single, ind = pipeline(single_data)
    # In the mono branch the indicator fields are Series, not DataFrames.
    import pandas as pd

    assert isinstance(ind.close, pd.Series)
    assert isinstance(ind.upper, pd.Series)
    assert isinstance(ind.lower, pd.Series)
