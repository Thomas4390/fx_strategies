"""Vol-targeting leverage helpers for strategy pipelines.

Single source of truth for the ``target_vol / realized_vol`` pattern
used across ``daily_momentum.py`` (4 sites) and duplicated in
``combined_portfolio_v2.py``. See ``reports/audits/vbt_native_audit.md``
finding P07 and the R-REFACTOR backlog.

Design note. We deliberately keep the pandas ``.shift(1).fillna(default)``
composition instead of ``.vbt.fshift(1, fill_value=default)``. The two
are NOT strictly equivalent when the input contains NaNs in positions
other than index 0: pandas fillna replaces every NaN with ``default``,
while VBT's fshift only fills positions vacated by the shift. The
existing call sites rely on the pandas semantics (warmup bars before
the rolling-std window fills get leverage = 1.0), so we preserve it
here for bit-equivalence with the frozen snapshots under
``tests/snapshots/``.
"""

from __future__ import annotations

import pandas as pd


def vol_target_leverage(
    realized_vol: pd.Series,
    target_vol: float,
    *,
    max_leverage: float,
    vol_floor: float = 0.01,
    default: float = 1.0,
) -> pd.Series:
    """Compute a causally-shifted, clipped vol-target leverage series.

    Equivalent to the inline expression::

        (target_vol / realized_vol.clip(lower=vol_floor))
            .clip(upper=max_leverage)
            .shift(1)
            .fillna(default)

    Used by the cross-sectional / time-series momentum sleeves in
    ``strategies.daily_momentum`` and the global leverage layer in
    ``strategies.combined_portfolio_v2``.

    Parameters
    ----------
    realized_vol
        Annualized rolling volatility (the caller handles the
        ``rolling_std * sqrt(ann_factor)`` step, keeping the helper
        agnostic to the window / annualization choice).
    target_vol
        Annualized volatility target (e.g. 0.10 for 10%).
    max_leverage
        Upper bound clip. Keyword-only because it varies by sleeve
        (3.0 for TS momentum, 5.0 for XS momentum in the current
        production config) and we want call sites to be explicit.
    vol_floor
        Lower bound clip applied to ``realized_vol`` before division,
        to avoid a divide-by-zero and cap the leverage at
        ``target_vol / vol_floor``.
    default
        Fallback leverage value used for positions where the input
        is NaN after the causal shift (typically the first bar plus
        any rolling-std warmup).

    Returns
    -------
    A pandas Series of leverage multipliers, same index as
    ``realized_vol``, causally shifted by one bar.
    """
    return (
        (target_vol / realized_vol.clip(lower=vol_floor))
        .clip(upper=max_leverage)
        .shift(1)
        .fillna(default)
    )
