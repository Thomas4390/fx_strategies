"""High-level Python API for block-bootstrap return resampling.

Thin, side-effect free wrappers around :mod:`framework.bootstrap_nb`
that convert the Numba outputs into pandas objects keyed by the
metric constants defined in :mod:`framework.pipeline_utils`.

Typical usage
-------------
>>> from framework.bootstrap import bootstrap_all_metrics
>>> rets = pf.returns.squeeze()
>>> df = bootstrap_all_metrics(rets, n_boot=2000, block_len_mean=50)
>>> df.loc["sharpe_ratio"]
observed    1.27
mean        1.24
median      1.25
std         0.18
ci_low      0.89
ci_high     1.60
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import vectorbtpro as vbt  # noqa: F401 — registers the ``.vbt`` accessor

from framework.bootstrap_nb import (
    bootstrap_all_metrics_nb,
    bootstrap_metric_nb,
    bootstrap_returns_matrix_nb,
)
from framework.pipeline_utils import (
    FX_MINUTE_ANN_FACTOR,
    METRIC_LABELS,
    METRIC_NAMES,
    compute_metric_nb,
)


# ═══════════════════════════════════════════════════════════════════════
# Internal helpers
# ═══════════════════════════════════════════════════════════════════════


def _as_1d_float(returns: pd.Series | np.ndarray) -> np.ndarray:
    """Coerce an input to a contiguous 1D float64 array of finite values."""
    arr = np.asarray(returns, dtype=np.float64)
    if arr.ndim > 1:
        arr = arr[:, 0] if arr.shape[1] == 1 else arr.ravel()
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        raise ValueError("returns is empty after dropping non-finite values")
    return np.ascontiguousarray(arr)


# ═══════════════════════════════════════════════════════════════════════
# Single-metric bootstrap
# ═══════════════════════════════════════════════════════════════════════


def bootstrap_metric(
    returns: pd.Series | np.ndarray,
    metric_id: int,
    *,
    n_boot: int = 5000,
    block_len_mean: float = 50.0,
    ci: tuple[float, float] = (0.025, 0.975),
    seed: int = 42,
    ann_factor: float = FX_MINUTE_ANN_FACTOR,
    cutoff: float = 0.05,
) -> dict[str, Any]:
    """Bootstrap confidence interval for a single metric.

    Parameters
    ----------
    returns
        Per-period return series.
    metric_id
        Integer metric id from :mod:`framework.pipeline_utils` (e.g.
        ``SHARPE_RATIO = 1``).
    n_boot, block_len_mean, seed, ann_factor, cutoff
        Forwarded to the Numba kernel.
    ci
        Lower/upper quantiles of the bootstrap distribution.

    Returns
    -------
    dict
        ``{observed, mean, median, std, ci_low, ci_high, samples,
        metric_id, metric_name}``.
    """
    arr = _as_1d_float(returns)
    observed = float(compute_metric_nb(arr, int(metric_id), ann_factor, cutoff))
    samples = bootstrap_metric_nb(
        arr,
        int(metric_id),
        int(n_boot),
        float(block_len_mean),
        int(seed),
        float(ann_factor),
        float(cutoff),
    )
    q_low, q_high = np.quantile(samples, ci)
    return {
        "observed": observed,
        "mean": float(np.mean(samples)),
        "median": float(np.median(samples)),
        "std": float(np.std(samples, ddof=1)),
        "ci_low": float(q_low),
        "ci_high": float(q_high),
        "samples": samples,
        "metric_id": int(metric_id),
        "metric_name": METRIC_NAMES.get(int(metric_id), str(metric_id)),
    }


# ═══════════════════════════════════════════════════════════════════════
# All metrics (single pass)
# ═══════════════════════════════════════════════════════════════════════


def bootstrap_all_metrics(
    returns: pd.Series | np.ndarray,
    *,
    n_boot: int = 5000,
    block_len_mean: float = 50.0,
    ci: tuple[float, float] = (0.025, 0.975),
    seed: int = 42,
    ann_factor: float = FX_MINUTE_ANN_FACTOR,
    cutoff: float = 0.05,
    return_samples: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, np.ndarray]:
    """Bootstrap confidence intervals for all 14 metrics in one pass.

    Parameters
    ----------
    return_samples
        When ``True``, also return the raw ``(n_boot, 14)`` sample matrix
        for downstream plotting.

    Returns
    -------
    pd.DataFrame
        Rows = metric name (snake_case), columns = ``[observed, mean,
        median, std, ci_low, ci_high, label]``. ``label`` is the human
        display name from ``METRIC_LABELS``.
    np.ndarray, optional
        Raw ``(n_boot, 14)`` sample matrix when ``return_samples=True``.
    """
    arr = _as_1d_float(returns)
    samples = bootstrap_all_metrics_nb(
        arr,
        int(n_boot),
        float(block_len_mean),
        int(seed),
        float(ann_factor),
        float(cutoff),
    )
    # Observed values in the same order as the sample matrix columns.
    observed = np.array(
        [
            float(compute_metric_nb(arr, m, ann_factor, cutoff))
            for m in range(samples.shape[1])
        ],
        dtype=np.float64,
    )

    q_low, q_high = np.quantile(samples, ci, axis=0)
    medians = np.median(samples, axis=0)
    rows = []
    for m in range(samples.shape[1]):
        col = samples[:, m]
        name = METRIC_NAMES.get(m, str(m))
        label = METRIC_LABELS.get(m, name)
        rows.append(
            dict(
                metric=name,
                label=label,
                observed=float(observed[m]),
                mean=float(np.mean(col)),
                median=float(medians[m]),
                std=float(np.std(col, ddof=1)),
                ci_low=float(q_low[m]),
                ci_high=float(q_high[m]),
            )
        )
    df = pd.DataFrame(rows).set_index("metric")
    if return_samples:
        return df, samples
    return df


# ═══════════════════════════════════════════════════════════════════════
# Equity fan charts
# ═══════════════════════════════════════════════════════════════════════


def bootstrap_equity_paths(
    returns: pd.Series | np.ndarray,
    *,
    n_sim: int = 200,
    block_len_mean: float = 50.0,
    seed: int = 42,
    initial_value: float = 1.0,
) -> pd.DataFrame:
    """Generate a matrix of alternative equity curves via stationary bootstrap.

    Each column is an alternative path built by resampling the input
    return series under the Politis-Romano scheme and compounding via
    ``(1 + r).cumprod()``.

    Parameters
    ----------
    returns
        Per-period return series. When passed as a ``pd.Series`` its
        index is reused for the output DataFrame.
    n_sim
        Number of simulated paths.
    block_len_mean
        Mean block length for the stationary bootstrap.
    seed
        Integer seed.
    initial_value
        Starting value of each simulated equity curve.

    Returns
    -------
    pd.DataFrame
        Shape ``(n, n_sim)``. Columns are labelled ``sim_0000`` ..
        ``sim_{n_sim-1}``. Suitable input for
        ``plot_equity_fan_chart``.
    """
    arr = _as_1d_float(returns)
    mat = bootstrap_returns_matrix_nb(
        arr,
        int(n_sim),
        float(block_len_mean),
        int(seed),
    )
    # Compound returns along the time axis.
    equity = initial_value * np.cumprod(1.0 + mat, axis=0)
    if isinstance(returns, pd.Series):
        # Filter by finiteness (not just NaN) to match ``_as_1d_float``
        # so a Series containing ``inf`` still aligns to the filtered
        # index instead of silently falling back to RangeIndex.
        finite = returns[np.isfinite(returns.to_numpy(dtype=np.float64))]
        if len(finite) == arr.size:
            index = finite.index
        else:
            index = pd.RangeIndex(arr.size, name="bar")
    else:
        index = pd.RangeIndex(arr.size, name="bar")
    cols = [f"sim_{i:04d}" for i in range(int(n_sim))]
    return pd.DataFrame(equity, index=index, columns=cols)


def equity_fan_percentiles(
    equity_paths: pd.DataFrame,
    percentiles: tuple[float, ...] = (5.0, 25.0, 50.0, 75.0, 95.0),
) -> pd.DataFrame:
    """Reduce a matrix of equity curves to per-bar percentile bands.

    Parameters
    ----------
    equity_paths
        Output of :func:`bootstrap_equity_paths`.
    percentiles
        Percentiles to compute. Default 5/25/50/75/95 maps directly to
        the 90% / 50% fan bands plus the median.

    Returns
    -------
    pd.DataFrame
        Rows indexed by the same time index as ``equity_paths``,
        columns named ``p05``, ``p25``, ``p50``, ``p75``, ``p95``.
    """
    arr = equity_paths.to_numpy(dtype=np.float64)
    perc = np.percentile(arr, list(percentiles), axis=1)
    cols = [f"p{int(round(p)):02d}" for p in percentiles]
    return pd.DataFrame(perc.T, index=equity_paths.index, columns=cols)


__all__ = [
    "bootstrap_metric",
    "bootstrap_all_metrics",
    "bootstrap_equity_paths",
    "equity_fan_percentiles",
]
