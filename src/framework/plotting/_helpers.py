"""Tiny zero-dependency helpers shared between plotting submodules.

Keeping these here (rather than in ``_core``) avoids a circular import
between ``_core`` and ``_trades``: ``_core.plot_portfolio_summary`` needs
``_pick_first_column`` and ``_trades.plot_trade_signals`` needs the
same helper.
"""

from __future__ import annotations

import vectorbtpro as vbt


def _slice_pf_last(pf: vbt.Portfolio, max_bars: int) -> vbt.Portfolio:
    """Return a portfolio sliced to the last *max_bars* rows.

    Essential on minute-frequency data: embedding a 3M-bar price series
    in an HTML plot produces hundred-megabyte files. We slice the
    portfolio itself rather than passing ``sim_start`` so every
    downstream accessor (``orders``, ``trades``) operates on the
    reduced window.
    """
    if len(pf.wrapper.index) <= max_bars:
        return pf
    start = pf.wrapper.index[-max_bars]
    return pf.loc[start:]


def _pick_first_column(pf: vbt.Portfolio) -> tuple[vbt.Portfolio, str | None]:
    """For multi-column portfolios, return a single-column slice.

    VBT's trade / order / MAE / MFE plots only accept a single column.
    When faced with a grouped / multi-asset portfolio, this helper
    falls back to the first column so the plot still works. Returns
    ``(pf_single, label)`` where *label* is the selected column name
    to append to the chart title (``None`` if the input was already
    single-column).
    """
    try:
        n_cols = pf.wrapper.shape_2d[1]
    except Exception:
        return pf, None
    if n_cols <= 1:
        return pf, None

    columns = pf.wrapper.columns
    first_col = columns[0]
    # Try to ungroup first if grouped (cash_sharing portfolios need this)
    try:
        pf_ungrouped = pf.regroup(False)
    except Exception:
        pf_ungrouped = pf
    try:
        return pf_ungrouped[first_col], str(first_col)
    except Exception:
        try:
            return pf_ungrouped.iloc[:, 0], str(first_col)
        except Exception:
            return pf, None
