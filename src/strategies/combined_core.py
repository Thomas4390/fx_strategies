"""Native vbt-pro multi-strategy aggregation layer.

Replaces the pandas weighted-sum + synthetic ``from_holding`` workaround used
by the sibling portfolio builders (``combined_portfolio.py`` and the levered
variant) with a single ``vbt.Portfolio.from_optimizer`` call driven by a
``PortfolioOptimizer`` built from filled allocations.

Numerical equivalence with the legacy pattern
---------------------------------------------
The legacy pattern was::

    port_t = sum_i( weights_ts[i, t] * strategy_rets[i, t] )
    pf     = from_holding(cumprod(1 + port_t) * base)

where ``weights_ts`` was already causal (shift(1) applied upstream). Running
``scripts/spikes/pfo_equivalence_spike.py`` against the native pattern below
shows bit-identical Sharpe / total return / max drawdown at ~1e-14 (machine
precision) across 5 test cases (equal, skewed, single-strategy, time-varying,
leveraged sum>1). The exact recipe that preserves equivalence:

1. ``allocations.shift(-1)`` — a PFO allocation set at bar t drives the
   position held during bar t+1, while the legacy pattern applied
   ``weights_ts[t]`` to ``rets[t]`` directly. Shifting forward by one bar
   aligns the two semantics.
2. ``valid_only=False, nonzero_only=False, unique_only=False`` — force a
   rebalance on every bar so positions do not drift (the default filters
   skip identical/zero/NaN rows which caused the ~0.2 Sharpe gap observed
   in the initial spike).
3. ``leverage=<cap>, leverage_mode="eager"`` — allows allocations whose row
   sums exceed 1 (needed for the v2 global vol-targeting layer that scales
   weights by a leverage series that can reach 12-15x).
4. ``fees=0.0, slippage=0.0`` — the legacy code did not charge rebalancing
   costs at the combined level (sub-strategies pay their own fees inside
   their individual portfolios); we preserve this to maintain equivalence.
"""

from __future__ import annotations


import numpy as np
import pandas as pd
import vectorbtpro as vbt


SYNTHETIC_BASE_PRICE: float = 1000.0
DEFAULT_INIT_CASH: float = 1_000_000.0
DEFAULT_LEVERAGE_CAP: float = 20.0


def returns_to_synthetic_prices(
    strategy_returns: dict[str, pd.Series] | pd.DataFrame,
    base_price: float = SYNTHETIC_BASE_PRICE,
) -> pd.DataFrame:
    """Convert a dict of daily-return Series to a synthetic multi-asset price frame.

    Each strategy becomes a synthetic asset whose price is the cumulative
    product of its returns starting from ``base_price``. Missing values in
    individual strategy series are filled with zero *before* the cumulative
    product so that the resulting price never jumps to NaN (which would
    break ``vbt.Portfolio.from_optimizer``).
    """
    if isinstance(strategy_returns, dict):
        df = pd.DataFrame(strategy_returns)
    else:
        df = strategy_returns.copy()
    df = df.dropna(how="all")
    filled = df.fillna(0.0)
    prices = (1.0 + filled).cumprod() * base_price
    return prices


def build_native_combined(
    strategy_returns: dict[str, pd.Series] | pd.DataFrame,
    allocations: pd.DataFrame,
    init_cash: float = DEFAULT_INIT_CASH,
    leverage_cap: float = DEFAULT_LEVERAGE_CAP,
    freq: str = "1D",
) -> tuple[vbt.Portfolio, pd.DataFrame, pd.DataFrame]:
    """Build a native combined ``vbt.Portfolio`` from per-strategy returns and allocations.

    Parameters
    ----------
    strategy_returns :
        Dict (or DataFrame) mapping strategy name -> daily return Series.
    allocations :
        Time-varying target weights, already causal (i.e. shift(1) applied
        upstream when the weights depend on prior returns). Columns must
        match the strategy names. Rows where allocations are non-causal are
        the caller's responsibility — this function does not apply any
        additional look-ahead guard beyond the ``.shift(-1)`` alignment
        required for PFO semantics.
    init_cash :
        Starting cash for the combined portfolio. Purely notional — the
        output metrics (Sharpe, annualized return, max drawdown) do not
        depend on this value as long as it is large enough to fund the
        initial allocation without cash-out.
    leverage_cap :
        Upper bound for per-bar allocation sums; passed to the underlying
        ``from_orders`` as ``leverage=leverage_cap`` with
        ``leverage_mode="eager"``. Set generously (>= max expected row sum
        of ``allocations``).
    freq :
        Bar frequency passed through to ``vbt.Portfolio.from_optimizer``
        so annualized metrics are correct.

    Returns
    -------
    pf : vbt.Portfolio
        Grouped single-column portfolio exposing native ``.sharpe_ratio``,
        ``.annualized_return``, ``.max_drawdown``, ``.stats()``, etc.
    prices : DataFrame
        The synthetic per-strategy price frame actually simulated (useful
        for diagnostics and plotting).
    aligned_allocations : DataFrame
        The allocations DataFrame reindexed to ``prices.index`` and with
        columns matching ``prices.columns``, NaN-filled with 0. Useful for
        callers that want to expose the final weights.
    """
    prices = returns_to_synthetic_prices(strategy_returns)

    aligned = allocations.reindex(index=prices.index, columns=prices.columns).fillna(
        0.0
    )

    # Shift forward by 1: PFO applies an allocation at bar t to the
    # position held during bar t+1, while the legacy pattern multiplied
    # weights[t] by rets[t]. The shift aligns the two semantics. The last
    # row is filled with 0 (no trade at the very end, no return to apply).
    allocations_shifted = aligned.shift(-1).fillna(0.0)

    pfo = vbt.PortfolioOptimizer.from_filled_allocations(
        allocations_shifted,
        valid_only=False,
        nonzero_only=False,
        unique_only=False,
    )

    pf = vbt.Portfolio.from_optimizer(
        prices,
        pfo,
        pf_method="from_orders",
        size_type="targetpercent",
        cash_sharing=True,
        group_by=True,
        init_cash=init_cash,
        fees=0.0,
        slippage=0.0,
        leverage=leverage_cap,
        leverage_mode="eager",
        freq=freq,
    )
    return pf, prices, aligned


def combined_returns_from_pf(pf: vbt.Portfolio) -> pd.Series:
    """Return the combined portfolio's daily returns as a 1-D Series.

    ``Portfolio.from_optimizer`` with ``group_by=True`` yields a portfolio
    whose ``.returns`` is a 1-column DataFrame. Downstream code (walk-forward
    analysis, plotting, regime analysis) expects a plain ``pd.Series``.
    """
    rets = pf.returns
    if isinstance(rets, pd.DataFrame):
        if rets.shape[1] != 1:
            raise ValueError(
                f"Expected 1-column returns from grouped portfolio, "
                f"got shape {rets.shape}"
            )
        return rets.iloc[:, 0]
    return rets


def sharpe_for_window(
    pf: vbt.Portfolio,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    min_bars: int = 20,
) -> float:
    """Native walk-forward Sharpe ratio for a date window on a combined portfolio.

    Slices the combined portfolio's returns to ``[start, end]``, rewraps them
    with a trivial ``from_holding`` wrapper so the native ``ReturnsAccessor``
    machinery computes the annualized Sharpe, and returns 0.0 if the window
    contains fewer than ``min_bars`` observations or if the standard deviation
    vanishes. This matches the legacy walk-forward behavior inherited from
    ``combined_portfolio.build_combined_portfolio``.
    """
    combined = combined_returns_from_pf(pf)
    window = combined.loc[start:end]
    if len(window) < min_bars:
        return 0.0
    price = (1.0 + window.fillna(0.0)).cumprod() * SYNTHETIC_BASE_PRICE
    pf_window = vbt.Portfolio.from_holding(
        close=price, init_cash=DEFAULT_INIT_CASH, freq="1D"
    )
    sr = float(pf_window.sharpe_ratio)
    return 0.0 if np.isnan(sr) else sr


def window_metrics(
    pf: vbt.Portfolio,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    min_bars: int = 20,
) -> dict[str, float]:
    """Return (sharpe, total_return, max_drawdown) for a date window."""
    combined = combined_returns_from_pf(pf)
    window = combined.loc[start:end]
    if len(window) < min_bars:
        return {"sharpe": 0.0, "total_return": 0.0, "max_drawdown": 0.0}
    price = (1.0 + window.fillna(0.0)).cumprod() * SYNTHETIC_BASE_PRICE
    pf_window = vbt.Portfolio.from_holding(
        close=price, init_cash=DEFAULT_INIT_CASH, freq="1D"
    )
    sr = float(pf_window.sharpe_ratio)
    tr = float(pf_window.total_return)
    mdd = float(pf_window.max_drawdown)
    return {
        "sharpe": 0.0 if np.isnan(sr) else sr,
        "total_return": 0.0 if np.isnan(tr) else tr,
        "max_drawdown": 0.0 if np.isnan(mdd) else mdd,
    }
