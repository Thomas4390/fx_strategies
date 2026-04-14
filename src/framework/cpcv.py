"""Combinatorially Purged Cross-Validation (CPCV).

Implements López de Prado's *Advances in Financial Machine Learning*
(2018) ch. 12 CPCV scheme. Compared to the walk-forward splitter
already wired into ``create_cv_pipeline`` (via
``vbt.Splitter.from_purged_walkforward``), CPCV enumerates every way
to choose ``n_test_groups`` groups out of ``n_groups`` — producing
``C(N, k)`` splits that can be recombined into multiple distinct
OOS paths.

Design
------
- **Splitter construction** uses ``vbt.Splitter.from_splits`` with
  explicit boolean masks so the decorator ``@vbt.cv_split`` can consume
  it transparently. Existing strategies switch from walk-forward to
  CPCV by swapping the splitter and passing it to
  ``create_cv_pipeline`` — no other change.
- **Purging** drops training samples whose timestamp is within
  ``purge_td`` of any test sample (López de Prado ch. 7).
- **Embargo** drops training samples landing in the
  ``embargo_pct * n_total`` bars that immediately follow each test
  group (avoids leakage through short-horizon serial correlation).
- **OOS distribution** : after the CV run, :func:`cpcv_oos_distribution`
  turns the ``(params, split, set)``-indexed grid_perf Series into a
  per-config summary (median, std, fraction of positive splits).
  Full CPCV path reconstruction (López de Prado AFML ch. 12 exact
  recipe) is intentionally left out of this first increment and can
  be added later once a concrete need arises — the current aggregation
  is already sufficient for Monte Carlo PBO downstream.
"""

from __future__ import annotations

from itertools import combinations
from math import comb
from typing import Any

import numpy as np
import pandas as pd
import vectorbtpro as vbt


# ═══════════════════════════════════════════════════════════════════════
# Splitter construction
# ═══════════════════════════════════════════════════════════════════════


def _group_bounds(n: int, n_groups: int) -> list[tuple[int, int]]:
    """Return ``n_groups`` contiguous ``(start, end)`` bounds covering ``n``.

    The last group absorbs the remainder when ``n`` is not a multiple
    of ``n_groups``.
    """
    size = n // n_groups
    out: list[tuple[int, int]] = []
    for g in range(n_groups):
        start = g * size
        end = (g + 1) * size if g < n_groups - 1 else n
        out.append((start, end))
    return out


def _purge_mask(
    index: pd.DatetimeIndex,
    test_mask: np.ndarray,
    purge_td: pd.Timedelta,
) -> np.ndarray:
    """Return a boolean mask of indices within ``purge_td`` of any test sample.

    Used to drop leakage-prone train samples near test boundaries.
    """
    if purge_td <= pd.Timedelta(0):
        return np.zeros_like(test_mask, dtype=bool)
    test_times = index[test_mask]
    if test_times.empty:
        return np.zeros_like(test_mask, dtype=bool)
    # A train sample is purged if its distance to any test sample is
    # smaller than purge_td. We approximate this by checking each test
    # group's contiguous [start - purge, end + purge] window.
    purge = np.zeros(len(index), dtype=bool)
    # Identify contiguous runs in test_mask.
    diffs = np.diff(test_mask.astype(np.int8))
    run_starts = np.where(diffs == 1)[0] + 1
    run_ends = np.where(diffs == -1)[0] + 1
    if test_mask[0]:
        run_starts = np.insert(run_starts, 0, 0)
    if test_mask[-1]:
        run_ends = np.append(run_ends, len(test_mask))
    for start, end in zip(run_starts, run_ends):
        t_start = index[start] - purge_td
        t_end = index[end - 1] + purge_td
        left = index.searchsorted(t_start, side="left")
        right = index.searchsorted(t_end, side="right")
        purge[left:right] = True
    return purge & ~test_mask


def _embargo_mask(
    test_mask: np.ndarray,
    embargo_bars: int,
) -> np.ndarray:
    """Drop ``embargo_bars`` samples after each contiguous test run."""
    if embargo_bars <= 0:
        return np.zeros_like(test_mask, dtype=bool)
    emb = np.zeros_like(test_mask, dtype=bool)
    diffs = np.diff(test_mask.astype(np.int8))
    run_ends = np.where(diffs == -1)[0] + 1
    if test_mask[-1]:
        run_ends = np.append(run_ends, len(test_mask))
    n = len(test_mask)
    for end in run_ends:
        emb[end : min(end + embargo_bars, n)] = True
    return emb & ~test_mask


def build_cpcv_splitter(
    index: pd.DatetimeIndex,
    *,
    n_groups: int = 6,
    n_test_groups: int = 2,
    purge_td: str | pd.Timedelta = "1 day",
    embargo_pct: float = 0.01,
) -> vbt.Splitter:
    """Build a ``vbt.Splitter`` implementing CPCV.

    Parameters
    ----------
    index
        Time index the splits operate on (typically a resampled daily
        index, same as walk-forward).
    n_groups
        Total number of time groups ``N``. Must satisfy ``N > n_test_groups``.
    n_test_groups
        Number of groups ``k`` used for OOS testing in each split.
        Total splits = ``C(N, k)``.
    purge_td
        Time distance to purge around each test group (López de Prado
        ch. 7). ``"1 day"`` is the typical value for FX daily.
    embargo_pct
        Fraction of the full index length used as embargo after each
        test group. ``0.01`` = 1% of the total bars.

    Returns
    -------
    vbt.Splitter
        Built via ``vbt.Splitter.from_splits`` with one entry per
        combination, each containing a boolean train/test mask.
    """
    if n_test_groups >= n_groups or n_test_groups < 1:
        raise ValueError(
            f"Need 1 <= n_test_groups < n_groups, got "
            f"n_test_groups={n_test_groups}, n_groups={n_groups}"
        )

    n = len(index)
    if n < n_groups * 2:
        raise ValueError(
            f"Index too short for CPCV: {n} bars, need ≥ {n_groups * 2}."
        )

    bounds = _group_bounds(n, n_groups)
    purge = pd.Timedelta(purge_td) if not isinstance(purge_td, pd.Timedelta) else purge_td
    embargo_bars = int(round(embargo_pct * n))

    split_tuples: list[tuple[np.ndarray, np.ndarray]] = []
    split_labels: list[str] = []

    for combo in combinations(range(n_groups), n_test_groups):
        test_mask = np.zeros(n, dtype=bool)
        for g in combo:
            s, e = bounds[g]
            test_mask[s:e] = True

        purge_m = _purge_mask(index, test_mask, purge)
        emb_m = _embargo_mask(test_mask, embargo_bars)
        train_mask = ~test_mask & ~purge_m & ~emb_m

        # vbt.Splitter.from_splits expects integer positions per set.
        train_pos = np.nonzero(train_mask)[0]
        test_pos = np.nonzero(test_mask)[0]
        split_tuples.append((train_pos, test_pos))
        split_labels.append("+".join(str(g) for g in combo))

    splitter = vbt.Splitter.from_splits(
        index,
        splits=split_tuples,
        split_labels=pd.Index(split_labels, name="split"),
        set_labels=["train", "test"],
    )
    return splitter


# ═══════════════════════════════════════════════════════════════════════
# Post-CV aggregation
# ═══════════════════════════════════════════════════════════════════════


def _select_set(grid_perf: pd.Series, set_name: str = "test") -> pd.Series:
    """Extract the subset of ``grid_perf`` matching the given ``set``."""
    if "set" not in (grid_perf.index.names or []):
        return grid_perf
    try:
        return grid_perf.xs(set_name, level="set")
    except (KeyError, ValueError):
        try:
            idx = 1 if set_name == "test" else 0
            return grid_perf.xs(idx, level="set")
        except Exception:
            return grid_perf


def cpcv_oos_distribution(
    grid_perf: pd.Series,
    *,
    metric_label: str = "Sharpe",
) -> pd.DataFrame:
    """Summarise the OOS distribution of each configuration across splits.

    Parameters
    ----------
    grid_perf
        Output of a ``@vbt.cv_split`` pipeline run on a CPCV splitter,
        with index levels ``(param1, param2, ..., split, set)``.
    metric_label
        Human label used in the description of the summary stats.

    Returns
    -------
    pd.DataFrame
        One row per configuration, columns ``{mean, median, std,
        min, max, q05, q95, pct_positive, n_splits}``. Sorted by
        ``median`` descending.
    """
    oos = _select_set(grid_perf, set_name="test")
    if not isinstance(oos.index, pd.MultiIndex):
        raise ValueError("grid_perf must have a MultiIndex with a 'split' level")

    param_levels = [n for n in (oos.index.names or []) if n != "split"]
    if not param_levels:
        raise ValueError("grid_perf has no parameter levels beyond 'split'")

    # pandas ≥2.2 : pass a scalar level when there is only one to keep
    # keys as scalars (avoids the FutureWarning about tuple keys).
    group_level: Any = param_levels[0] if len(param_levels) == 1 else param_levels
    grouped = oos.groupby(level=group_level)
    rows = []
    for key, values in grouped:
        arr = values.to_numpy(dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            continue
        rows.append(
            dict(
                key=key,
                mean=float(np.mean(arr)),
                median=float(np.median(arr)),
                std=float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
                min=float(np.min(arr)),
                max=float(np.max(arr)),
                q05=float(np.quantile(arr, 0.05)),
                q95=float(np.quantile(arr, 0.95)),
                pct_positive=float(np.mean(arr > 0)),
                n_splits=int(arr.size),
            )
        )
    df = pd.DataFrame(rows).set_index("key")
    df.attrs["metric_label"] = metric_label
    return df.sort_values("median", ascending=False)


def cpcv_summary(
    grid_perf: pd.Series,
    *,
    n_groups: int,
    n_test_groups: int,
) -> dict[str, Any]:
    """One-line summary of the full CPCV run suitable for a report."""
    dist = cpcv_oos_distribution(grid_perf)
    n_splits_expected = comb(n_groups, n_test_groups)
    n_paths = comb(n_groups - 1, n_test_groups - 1)
    return {
        "n_groups": int(n_groups),
        "n_test_groups": int(n_test_groups),
        "n_splits_expected": int(n_splits_expected),
        "n_reconstructed_paths": int(n_paths),
        "n_configs": int(dist.shape[0]),
        "best_config_key": dist.index[0] if not dist.empty else None,
        "best_median": float(dist["median"].iloc[0]) if not dist.empty else float("nan"),
        "best_pct_positive": float(dist["pct_positive"].iloc[0]) if not dist.empty else float("nan"),
        "best_q05": float(dist["q05"].iloc[0]) if not dist.empty else float("nan"),
    }


__all__ = [
    "build_cpcv_splitter",
    "cpcv_oos_distribution",
    "cpcv_summary",
]
