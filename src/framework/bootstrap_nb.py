"""Numba kernels for block bootstrap resampling of return series.

Implements two resampling schemes :

- **Stationary bootstrap** (Politis & Romano 1994, JASA 89(428)) — blocks
  of random geometric length with mean ``block_len_mean``. Preserves
  weak dependence without breaking stationarity. The default choice for
  financial returns.
- **Moving block bootstrap** (Kunsch 1989) — blocks of fixed length
  ``block_len``. Slightly lower bias near the edges but the resampled
  series is non-stationary at the boundaries.

Metric computation is dispatched to
:func:`framework.pipeline_utils.compute_metric_nb` so the 14 existing
VBT-native metrics are reused without duplication.

Reproducibility
---------------
Every kernel accepts a ``seed`` argument that is passed to
``np.random.seed`` inside the ``@njit`` scope. Re-running with the same
seed produces identical outputs.

Performance notes
-----------------
- All kernels are ``nogil=True`` so they release the GIL and can run
  under ``vbt.execute`` threadpools alongside the existing pipeline.
- ``cache=True`` persists the AOT-compiled artefact between interpreter
  restarts.
- ``bootstrap_all_metrics_nb`` computes the 14 metrics in a single pass
  over each resampled vector to amortise the resampling cost.

Thread safety
-------------
These kernels seed NumPy's **global** RNG inside each Numba function
via ``np.random.seed(seed)``. Running them concurrently in multiple
threads (e.g. under ``vbt.execute`` with ``threadpool``) will cause
the threads to race on that global state and the results become
non-reproducible. Call them sequentially from a single Python thread,
or wrap the call site in an external ``threading.Lock``. A future
upgrade path is to switch to ``numba.np.random.Generator`` (Numba
0.57+) which exposes a per-call RNG state.
"""

from __future__ import annotations

import numpy as np
from numba import njit

from framework.pipeline_utils import (
    FX_MINUTE_ANN_FACTOR,
    compute_metric_nb,
)

# Number of metric ids exposed by compute_metric_nb (0..13).
N_METRICS: int = 14


# ═══════════════════════════════════════════════════════════════════════
# Index generators
# ═══════════════════════════════════════════════════════════════════════


@njit(nogil=True, cache=True)
def stationary_bootstrap_indices_nb(
    n: int,
    block_len_mean: float,
    seed: int,
) -> np.ndarray:
    """Generate ``n`` indices using the Politis-Romano stationary bootstrap.

    Parameters
    ----------
    n
        Length of the bootstrap sample (typically = len(returns)).
    block_len_mean
        Average block length. The probability of starting a new block
        at each step is ``p = 1.0 / block_len_mean``. ``block_len_mean=1``
        reduces to iid bootstrap.
    seed
        Integer seed for reproducibility.

    Returns
    -------
    np.ndarray[int64] of shape (n,)
        Indices into the original return vector. Wraps around modulo
        ``n`` to keep the generator stationary.
    """
    np.random.seed(seed)
    p = 1.0 / max(block_len_mean, 1.0)
    out = np.empty(n, dtype=np.int64)
    out[0] = np.random.randint(0, n)
    for i in range(1, n):
        if np.random.random() < p:
            out[i] = np.random.randint(0, n)
        else:
            out[i] = (out[i - 1] + 1) % n
    return out


@njit(nogil=True, cache=True)
def moving_block_indices_nb(
    n: int,
    block_len: int,
    seed: int,
) -> np.ndarray:
    """Generate ``n`` indices using fixed-length moving block bootstrap.

    Parameters
    ----------
    n
        Length of the bootstrap sample.
    block_len
        Fixed block length. Blocks are picked uniformly from
        ``[0, n - block_len + 1)`` and concatenated until ``n`` indices
        are produced; the final block is truncated.
    seed
        Integer seed.

    Returns
    -------
    np.ndarray[int64] of shape (n,)
    """
    np.random.seed(seed)
    if block_len < 1:
        block_len = 1
    if block_len > n:
        block_len = n
    out = np.empty(n, dtype=np.int64)
    filled = 0
    last_start = n - block_len + 1
    while filled < n:
        start = np.random.randint(0, last_start)
        take = block_len
        if filled + take > n:
            take = n - filled
        for j in range(take):
            out[filled + j] = start + j
        filled += take
    return out


# ═══════════════════════════════════════════════════════════════════════
# Bootstrap metric loops
# ═══════════════════════════════════════════════════════════════════════


@njit(nogil=True, cache=True)
def bootstrap_metric_nb(
    returns: np.ndarray,
    metric_type: int,
    n_boot: int,
    block_len_mean: float,
    seed: int,
    ann_factor: float = FX_MINUTE_ANN_FACTOR,
    cutoff: float = 0.05,
) -> np.ndarray:
    """Bootstrap distribution of a single metric via stationary bootstrap.

    Parameters
    ----------
    returns
        1D array of per-period returns.
    metric_type
        One of the metric constants defined in ``pipeline_utils``
        (``SHARPE_RATIO = 1``, ``CALMAR_RATIO = 2``, ...).
    n_boot
        Number of bootstrap replicates (typically 2000-5000).
    block_len_mean
        Mean block length for the stationary bootstrap.
    seed
        Integer seed. Each replicate uses ``seed + i`` to decorrelate
        the internal PRNG streams while keeping the run reproducible.
    ann_factor, cutoff
        Forwarded to ``compute_metric_nb``.

    Returns
    -------
    np.ndarray[float64] of shape (n_boot,)
    """
    n = returns.shape[0]
    samples = np.empty(n_boot, dtype=np.float64)
    buf = np.empty(n, dtype=np.float64)
    for b in range(n_boot):
        idx = stationary_bootstrap_indices_nb(n, block_len_mean, seed + b)
        for k in range(n):
            buf[k] = returns[idx[k]]
        samples[b] = compute_metric_nb(buf, metric_type, ann_factor, cutoff)
    return samples


@njit(nogil=True, cache=True)
def bootstrap_all_metrics_nb(
    returns: np.ndarray,
    n_boot: int,
    block_len_mean: float,
    seed: int,
    ann_factor: float = FX_MINUTE_ANN_FACTOR,
    cutoff: float = 0.05,
) -> np.ndarray:
    """Bootstrap distribution of all 14 metrics in a single resampling pass.

    Amortises the resampling cost : for each of ``n_boot`` iterations
    a single resampled vector is produced and the 14 metrics are
    evaluated on it before moving on.

    Returns
    -------
    np.ndarray[float64] of shape ``(n_boot, 14)`` — columns ordered by
    the metric constants in ``framework.pipeline_utils``.
    """
    n = returns.shape[0]
    out = np.empty((n_boot, N_METRICS), dtype=np.float64)
    buf = np.empty(n, dtype=np.float64)
    for b in range(n_boot):
        idx = stationary_bootstrap_indices_nb(n, block_len_mean, seed + b)
        for k in range(n):
            buf[k] = returns[idx[k]]
        for m in range(N_METRICS):
            out[b, m] = compute_metric_nb(buf, m, ann_factor, cutoff)
    return out


@njit(nogil=True, cache=True)
def bootstrap_returns_matrix_nb(
    returns: np.ndarray,
    n_sim: int,
    block_len_mean: float,
    seed: int,
) -> np.ndarray:
    """Return a ``(n, n_sim)`` matrix of resampled return series.

    Used as the raw material for equity fan charts : each column is an
    alternative return path produced by the stationary bootstrap.

    Parameters
    ----------
    returns
        1D array of per-period returns.
    n_sim
        Number of alternative paths to generate.
    block_len_mean
        Mean block length for the stationary bootstrap.
    seed
        Integer seed.
    """
    n = returns.shape[0]
    out = np.empty((n, n_sim), dtype=np.float64)
    for s in range(n_sim):
        idx = stationary_bootstrap_indices_nb(n, block_len_mean, seed + s)
        for k in range(n):
            out[k, s] = returns[idx[k]]
    return out


__all__ = [
    "N_METRICS",
    "stationary_bootstrap_indices_nb",
    "moving_block_indices_nb",
    "bootstrap_metric_nb",
    "bootstrap_all_metrics_nb",
    "bootstrap_returns_matrix_nb",
]
