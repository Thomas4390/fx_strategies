"""Monte Carlo resampling at the trade level.

The :mod:`framework.bootstrap` module resamples *return bars* which
preserves the time-domain structure but mixes trades and idle periods.
For a trading strategy we often want a complementary perspective : take
the observed trade sequence and ask "what would the equity curve look
like if the *order* of trades had been different?".

Two schemes are implemented :

- **Shuffle** (Build Alpha / StrategyQuant style) — randomly permute
  the trade order without replacement. Total PnL is preserved, only
  the path changes. Exposes the *sequence risk* : how much of the
  observed max drawdown and underwater duration is a function of
  ordering luck.
- **Resample** — draw trades with replacement. Breaks the total PnL
  invariant but models the sampling uncertainty over the finite set
  of observed trades (what if we had drawn a different sample of
  length N from the same "trade distribution"?).

Both schemes produce a distribution of equity curves from which we
extract :

- max drawdown distribution (primary risk metric),
- longest underwater period distribution,
- terminal equity distribution (resample mode only — shuffle has a
  constant terminal value).

The kernel uses VBT's native ``pf.trades.returns`` accessor so it
works transparently on any ``vbt.Portfolio`` produced by the existing
strategies in ``src/strategies/``.

Thread safety
-------------
The Numba kernels seed NumPy's **global** RNG via ``np.random.seed``.
Running them concurrently in multiple threads is **not** safe and
will produce non-reproducible results. Invoke sequentially from a
single Python thread.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import vectorbtpro as vbt
from numba import njit


# ═══════════════════════════════════════════════════════════════════════
# Numba kernels
# ═══════════════════════════════════════════════════════════════════════


@njit(nogil=True, cache=True)
def _max_drawdown_from_equity_nb(equity: np.ndarray) -> float:
    """Compute max drawdown of a 1D equity curve in fractional units."""
    peak = equity[0]
    mdd = 0.0
    for i in range(1, equity.shape[0]):
        if equity[i] > peak:
            peak = equity[i]
        else:
            dd = (peak - equity[i]) / peak
            if dd > mdd:
                mdd = dd
    return mdd


@njit(nogil=True, cache=True)
def _longest_underwater_nb(equity: np.ndarray) -> int:
    """Length (in trades) of the longest underwater stretch."""
    peak = equity[0]
    longest = 0
    current = 0
    for i in range(1, equity.shape[0]):
        if equity[i] >= peak:
            peak = equity[i]
            if current > longest:
                longest = current
            current = 0
        else:
            current += 1
    if current > longest:
        longest = current
    return longest


@njit(nogil=True, cache=True)
def mc_trade_shuffle_nb(
    trade_returns: np.ndarray,
    n_sim: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Permute the trade order ``n_sim`` times and return distributions.

    Parameters
    ----------
    trade_returns
        1D array of per-trade fractional returns (e.g.
        ``pf.trades.returns.values``). A 0.015 entry means the trade
        multiplied equity by 1.015.
    n_sim
        Number of random permutations.
    seed
        Integer seed.

    Returns
    -------
    tuple
        - ``mdds`` : array of max drawdowns (shape ``(n_sim,)``)
        - ``uw_durations`` : array of longest underwater lengths
        - ``terminals`` : array of terminal equities (all equal for
          shuffle since total return is permutation-invariant, but we
          return the vector for API symmetry with ``mc_trade_resample_nb``).
    """
    np.random.seed(seed)
    n = trade_returns.shape[0]
    mdds = np.empty(n_sim, dtype=np.float64)
    uw_durations = np.empty(n_sim, dtype=np.int64)
    terminals = np.empty(n_sim, dtype=np.float64)
    eq = np.empty(n + 1, dtype=np.float64)
    for s in range(n_sim):
        perm = np.random.permutation(n)
        eq[0] = 1.0
        for i in range(n):
            eq[i + 1] = eq[i] * (1.0 + trade_returns[perm[i]])
        mdds[s] = _max_drawdown_from_equity_nb(eq)
        uw_durations[s] = _longest_underwater_nb(eq)
        terminals[s] = eq[n]
    return mdds, uw_durations, terminals


@njit(nogil=True, cache=True)
def mc_trade_resample_nb(
    trade_returns: np.ndarray,
    n_sim: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Draw the trade sequence with replacement ``n_sim`` times.

    Same return shapes as :func:`mc_trade_shuffle_nb` but the terminal
    equity varies across simulations because trades are drawn with
    replacement.
    """
    np.random.seed(seed)
    n = trade_returns.shape[0]
    mdds = np.empty(n_sim, dtype=np.float64)
    uw_durations = np.empty(n_sim, dtype=np.int64)
    terminals = np.empty(n_sim, dtype=np.float64)
    eq = np.empty(n + 1, dtype=np.float64)
    for s in range(n_sim):
        eq[0] = 1.0
        for i in range(n):
            j = np.random.randint(0, n)
            eq[i + 1] = eq[i] * (1.0 + trade_returns[j])
        mdds[s] = _max_drawdown_from_equity_nb(eq)
        uw_durations[s] = _longest_underwater_nb(eq)
        terminals[s] = eq[n]
    return mdds, uw_durations, terminals


@njit(nogil=True, cache=True)
def mc_trade_shuffle_paths_nb(
    trade_returns: np.ndarray,
    n_paths: int,
    seed: int,
) -> np.ndarray:
    """Return a ``(n_trades+1, n_paths)`` matrix of shuffled equity curves.

    Used as the raw material for a per-trade fan chart, complementary
    to the bar-level fan chart in :mod:`framework.bootstrap`.
    """
    np.random.seed(seed)
    n = trade_returns.shape[0]
    out = np.empty((n + 1, n_paths), dtype=np.float64)
    for s in range(n_paths):
        perm = np.random.permutation(n)
        out[0, s] = 1.0
        for i in range(n):
            out[i + 1, s] = out[i, s] * (1.0 + trade_returns[perm[i]])
    return out


# ═══════════════════════════════════════════════════════════════════════
# Python API
# ═══════════════════════════════════════════════════════════════════════


def _extract_trade_returns(pf: vbt.Portfolio) -> np.ndarray:
    """Return a 1D float64 array of per-trade fractional returns.

    Uses ``pf.trades.returns`` — the VBT-native accessor that yields
    the per-trade P&L as a fraction of the entry notional. Falls back
    to dividing ``pnl`` by ``entry_price * size`` if the accessor is
    unavailable.
    """
    try:
        ret = pf.trades.returns.values  # VBT native — preferred path
    except Exception:
        rec = pf.trades.records_readable
        if "Return" in rec.columns:
            ret = rec["Return"].to_numpy(dtype=np.float64)
        else:
            # Fallback when the native accessor is unavailable. Use
            # abs(notional) so shorts (negative size in VBT) produce
            # a positive denominator and the return sign matches pnl.
            pnl = rec["PnL"].to_numpy(dtype=np.float64)
            size = rec["Size"].to_numpy(dtype=np.float64)
            entry = rec["Avg Entry Price"].to_numpy(dtype=np.float64)
            denom = np.abs(size * entry)
            denom[denom == 0.0] = np.nan
            ret = pnl / denom
    arr = np.asarray(ret, dtype=np.float64).ravel()
    arr = arr[np.isfinite(arr)]
    return np.ascontiguousarray(arr)


def mc_max_drawdown_distribution(
    pf: vbt.Portfolio,
    *,
    n_sim: int = 2000,
    mode: str = "shuffle",
    seed: int = 42,
) -> dict[str, Any]:
    """Distribution of max drawdown under permutation / resampling.

    Parameters
    ----------
    pf
        Full-resolution portfolio.
    n_sim
        Number of Monte Carlo replicates.
    mode
        ``"shuffle"`` (no replacement, total PnL preserved) or
        ``"resample"`` (with replacement, models sampling noise).
    seed
        Integer seed.

    Returns
    -------
    dict
        ``{observed_mdd, mode, mdd_samples, mdd_p50, mdd_p95, mdd_p99,
        uw_samples, uw_p50, uw_p95, terminal_samples, n_trades, seed}``.
        ``observed_mdd`` is the drawdown of the *trade-equity* curve
        built by compounding the observed trade returns in order — it
        is the correct reference for the shuffle distribution. Note
        that this is NOT the same as ``pf.max_drawdown`` which runs on
        the bar-level equity curve and includes idle periods.
    """
    if mode not in ("shuffle", "resample"):
        raise ValueError(f"mode must be 'shuffle' or 'resample', got {mode!r}")

    tr = _extract_trade_returns(pf)
    if tr.size < 2:
        raise ValueError(f"Need at least 2 trades, got {tr.size}")

    # Observed trade-equity curve (no shuffle).
    observed_eq = np.empty(tr.size + 1, dtype=np.float64)
    observed_eq[0] = 1.0
    for i, r in enumerate(tr):
        observed_eq[i + 1] = observed_eq[i] * (1.0 + r)
    observed_mdd = float(_max_drawdown_from_equity_nb(observed_eq))
    observed_uw = int(_longest_underwater_nb(observed_eq))

    kernel = mc_trade_shuffle_nb if mode == "shuffle" else mc_trade_resample_nb
    mdds, uws, terminals = kernel(tr, int(n_sim), int(seed))

    p50, p95, p99 = np.quantile(mdds, [0.50, 0.95, 0.99])
    return {
        "mode": mode,
        "n_trades": int(tr.size),
        "n_sim": int(n_sim),
        "seed": int(seed),
        "observed_mdd": observed_mdd,
        "observed_underwater": observed_uw,
        "observed_terminal": float(observed_eq[-1]),
        "mdd_samples": mdds,
        "mdd_p50": float(p50),
        "mdd_p95": float(p95),
        "mdd_p99": float(p99),
        "mdd_mean": float(np.mean(mdds)),
        "mdd_std": float(np.std(mdds, ddof=1)),
        "uw_samples": uws,
        "uw_p50": float(np.quantile(uws, 0.50)),
        "uw_p95": float(np.quantile(uws, 0.95)),
        "terminal_samples": terminals,
    }


def mc_sequence_risk_report(
    pf: vbt.Portfolio,
    *,
    n_sim: int = 2000,
    seed: int = 42,
) -> dict[str, Any]:
    """Quantify how much of the observed drawdown is a function of ordering.

    Computes the shuffle-mode distribution and returns a small summary
    dict suitable for inclusion in a text report.

    Key output : ``sequence_luck`` — ``(observed_mdd - median_mdd) /
    std_mdd``. Large negative values (observed MDD well below median)
    mean the realised ordering was lucky ; large positive values mean
    the realised ordering was unlucky.
    """
    dist = mc_max_drawdown_distribution(pf, n_sim=n_sim, mode="shuffle", seed=seed)
    std = max(dist["mdd_std"], 1e-12)
    luck = (dist["observed_mdd"] - dist["mdd_p50"]) / std
    pct_worse = float(np.mean(dist["mdd_samples"] > dist["observed_mdd"]))
    return {
        "observed_mdd": dist["observed_mdd"],
        "mdd_median": dist["mdd_p50"],
        "mdd_p95": dist["mdd_p95"],
        "sequence_luck_zscore": float(luck),
        "pct_shuffles_worse_than_observed": pct_worse,
        "n_sim": dist["n_sim"],
    }


def mc_trade_equity_paths(
    pf: vbt.Portfolio,
    *,
    n_paths: int = 100,
    seed: int = 42,
) -> pd.DataFrame:
    """Return a DataFrame of ``n_paths`` shuffled trade-equity curves.

    Columns labelled ``path_0000`` .. ``path_{n_paths-1}``, rows
    indexed by trade number starting at 0 (initial equity = 1.0).
    """
    tr = _extract_trade_returns(pf)
    if tr.size < 2:
        raise ValueError(f"Need at least 2 trades, got {tr.size}")
    mat = mc_trade_shuffle_paths_nb(tr, int(n_paths), int(seed))
    cols = [f"path_{i:04d}" for i in range(int(n_paths))]
    idx = pd.RangeIndex(mat.shape[0], name="trade")
    return pd.DataFrame(mat, index=idx, columns=cols)


__all__ = [
    "mc_trade_shuffle_nb",
    "mc_trade_resample_nb",
    "mc_trade_shuffle_paths_nb",
    "mc_max_drawdown_distribution",
    "mc_sequence_risk_report",
    "mc_trade_equity_paths",
]
