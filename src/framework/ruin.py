"""Tail-risk metrics for path-dependent sizing regimes.

Sharpe and hit rate are actively misleading for martingale and grid systems.
Both convert a roughly symmetric return distribution into "many small wins, a
rare catastrophic loss": the mean and variance barely move while the left tail
grows without bound. A martingale can post a high Sharpe and a 90% win rate
right up to the trade that ends it.

What actually separates these regimes is the shape of the loss distribution and
the probability of not surviving. This module measures that:

- terminal wealth percentiles, from a stationary block bootstrap
- probability of ruin and of losing more than a given fraction of capital
- skewness, excess kurtosis, and the worst single trade
- MAR (CAGR / max drawdown) and the longest time to recover a peak

The bootstrap resamples *blocks* of returns (Politis-Romano) rather than
individual days, which preserves the volatility clustering and serial structure
that drive drawdowns. Resampling i.i.d. would understate exactly the risk these
regimes carry.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from numba import njit

DEFAULT_N_BOOT: int = 2_000
DEFAULT_BLOCK_MEAN: int = 21
DEFAULT_ANN_FACTOR: float = 252.0

#: Equity fraction below which a path is treated as ruined. Not zero: an account
#: down 90% is finished in practice — margin no longer supports the strategy,
#: and the 10x recovery it would need is not a realistic proposition.
RUIN_THRESHOLD: float = 0.10


@njit(nogil=True, cache=True)
def _block_bootstrap_paths_nb(
    returns: np.ndarray,
    n_boot: int,
    n_steps: int,
    block_mean: int,
    seed: int,
) -> np.ndarray:
    """Stationary block bootstrap of equity paths.

    Block lengths are geometric with mean ``block_mean``; the start of each
    block is uniform over the sample, with wrap-around. Returns an
    ``(n_boot, n_steps)`` array of equity multiples starting from 1.0.
    """
    np.random.seed(seed)
    n = returns.shape[0]
    p = 1.0 / block_mean
    out = np.empty((n_boot, n_steps), dtype=np.float64)

    for b in range(n_boot):
        equity = 1.0
        idx = np.random.randint(0, n)
        for t in range(n_steps):
            equity *= 1.0 + returns[idx]
            if equity < 0.0:
                equity = 0.0
            out[b, t] = equity
            if np.random.random() < p:
                idx = np.random.randint(0, n)
            else:
                idx += 1
                if idx >= n:
                    idx = 0
    return out


@njit(nogil=True, cache=True)
def _path_stats_nb(paths: np.ndarray, ruin_threshold: float) -> tuple:
    """Per-path terminal equity, max drawdown, and ruin flag."""
    n_boot, n_steps = paths.shape
    terminal = np.empty(n_boot, dtype=np.float64)
    max_dd = np.empty(n_boot, dtype=np.float64)
    ruined = np.zeros(n_boot, dtype=np.int64)

    for b in range(n_boot):
        peak = 1.0
        worst = 0.0
        hit = False
        for t in range(n_steps):
            eq = paths[b, t]
            if eq > peak:
                peak = eq
            dd = 1.0 - eq / peak if peak > 0.0 else 1.0
            if dd > worst:
                worst = dd
            if eq <= ruin_threshold:
                hit = True
        terminal[b] = paths[b, n_steps - 1]
        max_dd[b] = worst
        ruined[b] = 1 if hit else 0
    return terminal, max_dd, ruined


@dataclass
class RuinReport:
    """Tail-risk profile of one sizing regime."""

    label: str
    ann_return: float
    ann_vol: float
    sharpe: float
    max_drawdown: float
    mar: float
    skew: float
    excess_kurtosis: float
    worst_trade: float
    longest_recovery_days: int
    p_ruin: float
    p_loss_50: float
    terminal_p5: float
    terminal_p50: float
    terminal_p95: float
    boot_dd_p95: float
    peak_exposure: float = float("nan")
    n_addons: int = 0
    n_kills: int = 0
    extras: dict = field(default_factory=dict)

    def as_row(self) -> dict:
        return {
            "regime": self.label,
            "ann_return": self.ann_return,
            "ann_vol": self.ann_vol,
            "sharpe": self.sharpe,
            "max_dd": self.max_drawdown,
            "MAR": self.mar,
            "skew": self.skew,
            "exc_kurt": self.excess_kurtosis,
            "worst_trade": self.worst_trade,
            "recovery_days": self.longest_recovery_days,
            "P(ruin)": self.p_ruin,
            "P(loss>50%)": self.p_loss_50,
            "terminal_p5": self.terminal_p5,
            "terminal_p50": self.terminal_p50,
            "terminal_p95": self.terminal_p95,
            "boot_dd_p95": self.boot_dd_p95,
            "peak_exposure": self.peak_exposure,
            "n_addons": self.n_addons,
            "n_kills": self.n_kills,
        }


def longest_recovery(returns: pd.Series) -> int:
    """Longest run of periods spent below a previous equity peak."""
    equity = (1.0 + returns.fillna(0.0)).cumprod()
    peak = equity.cummax()
    underwater = equity < peak
    longest = run = 0
    for flag in underwater.to_numpy():
        run = run + 1 if flag else 0
        longest = max(longest, run)
    return int(longest)


def ruin_report(
    returns: pd.Series,
    *,
    label: str,
    worst_trade: float = float("nan"),
    peak_exposure: float = float("nan"),
    n_addons: int = 0,
    n_kills: int = 0,
    n_boot: int = DEFAULT_N_BOOT,
    block_mean: int = DEFAULT_BLOCK_MEAN,
    horizon: int | None = None,
    ann_factor: float = DEFAULT_ANN_FACTOR,
    ruin_threshold: float = RUIN_THRESHOLD,
    seed: int = 42,
) -> RuinReport:
    """Full tail-risk profile of a return series.

    ``horizon`` is the bootstrap path length; it defaults to the sample length,
    i.e. "what could a run of this duration have looked like".
    """
    r = returns.dropna()
    if len(r) < block_mean * 3:
        raise ValueError(f"{label}: {len(r)} returns is too short to bootstrap")

    arr = np.ascontiguousarray(r.to_numpy(dtype=np.float64))
    steps = int(horizon) if horizon is not None else len(arr)

    paths = _block_bootstrap_paths_nb(arr, n_boot, steps, block_mean, seed)
    terminal, dd, ruined = _path_stats_nb(paths, ruin_threshold)

    ann = float(r.mean() * ann_factor)
    vol = float(r.std() * np.sqrt(ann_factor))
    equity = (1.0 + r).cumprod()
    realized_dd = float((equity / equity.cummax() - 1.0).min())

    return RuinReport(
        label=label,
        ann_return=ann,
        ann_vol=vol,
        sharpe=ann / vol if vol > 0 else 0.0,
        max_drawdown=realized_dd,
        mar=ann / abs(realized_dd) if realized_dd < 0 else float("nan"),
        skew=float(r.skew()),
        excess_kurtosis=float(r.kurtosis()),
        worst_trade=worst_trade,
        longest_recovery_days=longest_recovery(r),
        p_ruin=float(ruined.mean()),
        p_loss_50=float((terminal < 0.50).mean()),
        terminal_p5=float(np.percentile(terminal, 5)),
        terminal_p50=float(np.percentile(terminal, 50)),
        terminal_p95=float(np.percentile(terminal, 95)),
        boot_dd_p95=float(np.percentile(dd, 95)),
        peak_exposure=peak_exposure,
        n_addons=n_addons,
        n_kills=n_kills,
    )


def compare_regimes(reports: list[RuinReport]) -> pd.DataFrame:
    """Reports as a table, ranked by MAR."""
    df = pd.DataFrame([rep.as_row() for rep in reports])
    return df.sort_values("MAR", ascending=False).reset_index(drop=True)


def format_comparison(df: pd.DataFrame) -> str:
    """Fixed-width rendering of :func:`compare_regimes` for logs and reports."""
    head = (
        f"{'regime':30s} {'ann':>7s} {'SR':>6s} {'maxDD':>8s} {'MAR':>6s} "
        f"{'skew':>7s} {'kurt':>7s} {'P(ruine)':>9s} {'P(-50%)':>8s} "
        f"{'W_p5':>6s} {'W_p50':>6s} {'W_p95':>6s} {'DD_p95':>7s}"
    )
    lines = [head, "-" * len(head)]
    for _, row in df.iterrows():
        lines.append(
            f"{row['regime']:30s} {row['ann_return'] * 100:6.2f}% {row['sharpe']:6.2f} "
            f"{row['max_dd'] * 100:7.2f}% {row['MAR']:6.2f} {row['skew']:7.2f} "
            f"{row['exc_kurt']:7.1f} {row['P(ruin)'] * 100:8.2f}% "
            f"{row['P(loss>50%)'] * 100:7.2f}% {row['terminal_p5']:6.2f} "
            f"{row['terminal_p50']:6.2f} {row['terminal_p95']:6.2f} "
            f"{row['boot_dd_p95'] * 100:6.1f}%"
        )
    return "\n".join(lines)
