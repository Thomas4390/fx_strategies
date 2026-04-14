"""Statistical testing helpers — DSR, PSR and PBO.

Implements three tools for correcting backtest performance claims in the
presence of multiple testing and non-normal returns:

- **Probabilistic Sharpe Ratio (PSR)** — Bailey & Lopez de Prado 2012.
  Probability that the observed Sharpe exceeds a benchmark SR given the
  skewness and kurtosis of returns. Computed natively via
  ``returns.vbt.returns().prob_sharpe_ratio()`` — we only wrap the call
  and expose the benchmark argument.
- **Deflated Sharpe Ratio (DSR)** — Bailey & Lopez de Prado 2014.
  PSR with the benchmark set to the *expected maximum Sharpe across N
  independent trials*, closed-form using the Euler-Mascheroni constant.
  Answers: "after trying N strategies, how likely is it that the observed
  top Sharpe would have happened by chance alone?".
- **Probability of Backtest Overfitting (PBO)** — Bailey-Borwein-Lopez
  de Prado-Zhu 2015, via Combinatorially Symmetric Cross-Validation
  (CSCV). Estimates how often the in-sample top strategy lands below
  the median out-of-sample. Returns a probability in [0, 1].

Design choices
--------------
- The DSR/PSR path delegates to VBT Pro's native
  ``ReturnsAccessor.prob_sharpe_ratio`` and ``sharpe_ratio_std`` — we do
  not reimplement the skew/kurtosis correction (VBT uses the Mertens
  1998 formula already).
- The CSCV PBO is a pure-numpy implementation because VBT Pro has no
  native equivalent (confirmed via ``mcp__vectorbtpro__search``). The
  algorithm is ~40 lines, no external dependency.

References
----------
- Bailey, Lopez de Prado (2012). "The Sharpe Ratio Efficient Frontier".
- Bailey, Lopez de Prado (2014). "The Deflated Sharpe Ratio: Correcting
  for Selection Bias, Backtest Overfitting and Non-Normality".
- Bailey, Borwein, Lopez de Prado, Zhu (2015). "The Probability of
  Backtest Overfitting".
"""

from __future__ import annotations

from math import sqrt
from typing import Any

import numpy as np
import pandas as pd
import vectorbtpro  # noqa: F401 — registers the `.vbt` pandas accessor
from scipy.stats import norm


# ═══════════════════════════════════════════════════════════════════════
# Probabilistic Sharpe Ratio (PSR)
# ═══════════════════════════════════════════════════════════════════════


def probabilistic_sharpe_ratio(
    returns: pd.Series,
    sr_benchmark: float = 0.0,
    freq: str | None = "1D",
    year_freq: str | pd.Timedelta | None = None,
    ddof: int = 1,
    bias: bool = True,
) -> float:
    """Compute the Probabilistic Sharpe Ratio vs a benchmark SR.

    Thin wrapper around ``returns.vbt.returns(...).prob_sharpe_ratio(...)``.
    PSR is ``Φ((SR - SR*) / σ(SR))`` where ``σ(SR)`` is the Mertens
    standard error (uses skewness and kurtosis of the return series)
    and ``SR*`` is the benchmark Sharpe threshold to beat.

    Parameters
    ----------
    returns : pd.Series
        Per-period return series (e.g. daily returns from
        ``pf.daily_returns``).
    sr_benchmark : float
        Benchmark (annualized) Sharpe ratio to compare against. Default
        ``0.0`` means "probability the strategy has any Sharpe edge".
    freq, year_freq : str | pd.Timedelta | None
        Return accessor annualization. ``year_freq`` defaults to the
        global vbt settings (``vbt.settings.returns.year_freq``).
    ddof : int
        Delta degrees of freedom passed to the Mertens std.
    bias : bool
        Passed to the Mertens std (bias correction for skew/kurtosis).

    Returns
    -------
    float
        PSR ∈ [0, 1]. A PSR of 0.95 means 95% confidence that the true
        Sharpe exceeds ``sr_benchmark``.
    """
    ret_acc = returns.vbt.returns(freq=freq, year_freq=year_freq)
    # VBT's prob_sharpe_ratio uses a benchmark returns series, but we
    # want a flat scalar benchmark. Re-derive the z-score manually from
    # the native sharpe_ratio + sharpe_ratio_std (both VBT-native).
    sr_hat = float(ret_acc.sharpe_ratio())
    sr_std = float(ret_acc.sharpe_ratio_std(ddof=ddof, bias=bias))
    if sr_std <= 0 or not np.isfinite(sr_std):
        return float("nan")
    z = (sr_hat - sr_benchmark) / sr_std
    return float(norm.cdf(z))


# ═══════════════════════════════════════════════════════════════════════
# Deflated Sharpe Ratio (DSR)
# ═══════════════════════════════════════════════════════════════════════


_EULER_MASCHERONI = 0.5772156649015329


def expected_max_sharpe(n_trials: int, trial_sr_var: float) -> float:
    """Closed-form E[max SR] over ``n_trials`` independent Sharpe draws.

    From Bailey & Lopez de Prado 2014, eq. (5). Approximates the
    expected maximum of ``n_trials`` iid ``N(0, σ_SR)`` draws using the
    two-term asymptotic expansion involving the Euler-Mascheroni
    constant and the inverse-normal of ``1 - 1/(e*N)``.

    Parameters
    ----------
    n_trials : int
        Number of independent strategy trials that produced a Sharpe.
    trial_sr_var : float
        Variance of the ``n_trials`` Sharpe estimates (σ²_SR). When
        there is only one group of configs, use ``np.var(sharpes, ddof=1)``.

    Returns
    -------
    float
        Expected maximum Sharpe across the ``n_trials`` draws, in the
        same units as the input Sharpes (typically annualized).
    """
    if n_trials <= 1 or trial_sr_var <= 0:
        return 0.0
    n = float(n_trials)
    z1 = norm.ppf(1.0 - 1.0 / n)
    z2 = norm.ppf(1.0 - 1.0 / (n * np.e))
    expected_z = (1.0 - _EULER_MASCHERONI) * z1 + _EULER_MASCHERONI * z2
    return float(sqrt(trial_sr_var) * expected_z)


def deflated_sharpe_ratio(
    returns: pd.Series,
    n_trials: int,
    trial_sharpes: np.ndarray | pd.Series | list[float] | None = None,
    trial_sr_var: float | None = None,
    freq: str | None = "1D",
    year_freq: str | pd.Timedelta | None = None,
    ddof: int = 1,
    bias: bool = True,
) -> dict[str, float]:
    """Compute the Deflated Sharpe Ratio for a single top strategy.

    DSR is the PSR of the selected strategy vs ``E[max SR]`` over the
    ``n_trials`` attempted during selection. It penalizes Sharpes that
    look good only because many strategies were tested.

    You can either pass the full vector of ``trial_sharpes`` (preferred —
    the variance is computed internally) or pass ``trial_sr_var``
    directly when the vector is not available.

    Parameters
    ----------
    returns : pd.Series
        Return series of the *selected* (top) strategy.
    n_trials : int
        Total number of strategies tested during selection. This is
        the "multiple testing" count.
    trial_sharpes, trial_sr_var : optional
        Either the raw vector of trial Sharpes (more accurate) or the
        variance of those Sharpes. Exactly one must be provided.
    freq, year_freq, ddof, bias : see :func:`probabilistic_sharpe_ratio`.

    Returns
    -------
    dict
        ``{sharpe, sharpe_std, expected_max_sharpe, dsr, n_trials}``.
        ``dsr`` is in [0, 1] — PSR vs the selection-adjusted threshold.
        A DSR below ~0.95 is a red flag: the observed Sharpe is not
        statistically distinguishable from the best-of-N luck level.
    """
    if (trial_sharpes is None) == (trial_sr_var is None):
        raise ValueError("Pass exactly one of trial_sharpes or trial_sr_var.")
    if trial_sharpes is not None:
        arr = np.asarray(list(trial_sharpes), dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size < 2:
            raise ValueError(f"Need at least 2 finite trial Sharpes, got {arr.size}")
        var = float(np.var(arr, ddof=1))
    else:
        var = float(trial_sr_var)

    ret_acc = returns.vbt.returns(freq=freq, year_freq=year_freq)
    sr_hat = float(ret_acc.sharpe_ratio())
    sr_std = float(ret_acc.sharpe_ratio_std(ddof=ddof, bias=bias))
    sr_threshold = expected_max_sharpe(n_trials, var)

    if sr_std <= 0 or not np.isfinite(sr_std):
        dsr = float("nan")
    else:
        z = (sr_hat - sr_threshold) / sr_std
        dsr = float(norm.cdf(z))

    return {
        "sharpe": sr_hat,
        "sharpe_std": sr_std,
        "expected_max_sharpe": sr_threshold,
        "dsr": dsr,
        "n_trials": int(n_trials),
    }


# ═══════════════════════════════════════════════════════════════════════
# Probability of Backtest Overfitting — CSCV
# ═══════════════════════════════════════════════════════════════════════


def _cscv_combinations(n_bins: int) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
    """Return all combinations (IS, OOS) for CSCV with ``n_bins`` bins.

    Bins are split in half : ``n_bins//2`` for IS, ``n_bins//2`` for OOS.
    Uses ``itertools.combinations`` via numpy.
    """
    from itertools import combinations

    all_bins = tuple(range(n_bins))
    half = n_bins // 2
    combos: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
    for is_bins in combinations(all_bins, half):
        oos_bins = tuple(b for b in all_bins if b not in is_bins)
        combos.append((is_bins, oos_bins))
    return combos


def probability_of_backtest_overfitting(
    returns_matrix: pd.DataFrame,
    n_bins: int = 16,
    objective: str = "sharpe",
) -> dict[str, Any]:
    """Estimate PBO via Combinatorially Symmetric Cross-Validation (CSCV).

    Implements Bailey-Borwein-Lopez de Prado-Zhu 2015 (section 3). For a
    wide DataFrame of ``n_configs`` strategies observed over ``T`` bars,
    CSCV :

    1. Splits the ``T`` rows into ``n_bins`` equal-size contiguous chunks.
    2. Enumerates all ``C(n_bins, n_bins/2)`` ways to choose half the
       bins as IS and the remaining half as OOS.
    3. For each split, finds the config ``n*`` that maximizes the
       ``objective`` (Sharpe or mean) in IS, then measures its
       performance on OOS relative to all other configs.
    4. Computes the log-odds of the OOS rank of ``n*`` and aggregates.
    5. PBO is the fraction of splits where the IS-best config lands
       **below** the OOS median (i.e. logit < 0).

    Parameters
    ----------
    returns_matrix : pd.DataFrame
        Shape ``(T, n_configs)``. Each column is the return series of
        one strategy trial over the common time index.
    n_bins : int
        Number of time-slice bins (must be even). Default 16 → C(16,8)
        = 12,870 splits, which is overkill. Use 10 or 12 for speed.
    objective : {"sharpe", "mean"}
        Metric maximized in IS and evaluated in OOS. ``"sharpe"`` uses
        ``mean / std`` (ddof=1) per column. ``"mean"`` is the raw mean.

    Returns
    -------
    dict
        - ``pbo``: probability in [0, 1]. PBO < 0.5 is a good signal.
        - ``logits``: np.ndarray of shape ``(n_splits,)`` — the log-odds
          of the OOS rank for the IS-best config, one per split.
        - ``n_splits``: int, number of splits evaluated.
        - ``n_bins``: int, the chosen bin count.
        - ``n_configs``: int, columns in the input matrix.
    """
    if n_bins % 2 != 0 or n_bins < 2:
        raise ValueError(f"n_bins must be an even positive int, got {n_bins}")
    if objective not in ("sharpe", "mean"):
        raise ValueError(f"objective must be 'sharpe' or 'mean', got {objective}")

    mat = returns_matrix.dropna(how="any").to_numpy(dtype=float)
    if mat.size == 0:
        raise ValueError("returns_matrix is empty after dropna")
    n_rows, n_configs = mat.shape
    if n_configs < 2:
        raise ValueError(f"Need at least 2 configs, got {n_configs}")

    # Trim to a multiple of n_bins, then split equally.
    trimmed = (n_rows // n_bins) * n_bins
    mat = mat[:trimmed]
    bin_size = trimmed // n_bins
    # Shape: (n_bins, bin_size, n_configs).
    binned = mat.reshape(n_bins, bin_size, n_configs)

    def _score(chunks: np.ndarray) -> np.ndarray:
        """Per-column score on a stacked IS or OOS slice."""
        stacked = chunks.reshape(-1, n_configs)
        mean = stacked.mean(axis=0)
        if objective == "mean":
            return mean
        std = stacked.std(axis=0, ddof=1)
        # Guard against zero-variance columns (no trades).
        return np.where(std > 0, mean / std, -np.inf)

    combos = _cscv_combinations(n_bins)
    logits = np.empty(len(combos), dtype=float)
    ranks_oos = np.empty(len(combos), dtype=float)

    for i, (is_bins, oos_bins) in enumerate(combos):
        is_score = _score(binned[list(is_bins)])
        oos_score = _score(binned[list(oos_bins)])

        # n* = argmax IS score.
        n_star = int(np.argmax(is_score))

        # Rank of n* on OOS ("bigger = better"). Convert to a relative
        # rank in (0, 1) so the logit is bounded.
        order = np.argsort(oos_score)  # ascending
        rank = int(np.where(order == n_star)[0][0]) + 1  # 1-indexed
        w_bar = rank / (n_configs + 1)  # in (0, 1)
        ranks_oos[i] = w_bar
        logits[i] = np.log(w_bar / (1.0 - w_bar))

    pbo = float((logits < 0).sum()) / float(len(logits))
    return {
        "pbo": pbo,
        "logits": logits,
        "ranks_oos": ranks_oos,
        "n_splits": len(logits),
        "n_bins": n_bins,
        "n_configs": n_configs,
    }


# ═══════════════════════════════════════════════════════════════════════
# Helper: bulk DSR for a top-1 config from a sweep
# ═══════════════════════════════════════════════════════════════════════


def dsr_for_sweep_top(
    top_returns: pd.Series,
    sweep_sharpes: pd.Series,
    freq: str | None = "1D",
    year_freq: str | pd.Timedelta | None = None,
) -> dict[str, float]:
    """Shortcut : given the top config's returns and the full vector of
    sweep Sharpes, return its DSR + supporting numbers.

    This is the common call-site inside the Phase 21 retrofit script —
    one call per (phase, top_config) pair.
    """
    return deflated_sharpe_ratio(
        top_returns,
        n_trials=int(sweep_sharpes.size),
        trial_sharpes=sweep_sharpes.to_numpy(dtype=float),
        freq=freq,
        year_freq=year_freq,
    )


__all__ = [
    "probabilistic_sharpe_ratio",
    "expected_max_sharpe",
    "deflated_sharpe_ratio",
    "probability_of_backtest_overfitting",
    "dsr_for_sweep_top",
]
