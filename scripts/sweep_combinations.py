"""Systematic sweep of multi-strategy combined-portfolio configurations.

Runs ~37 combinations of sleeves × allocation × target_vol × max_leverage ×
dd_cap_enabled through ``strategies.combined_portfolio_v2.build_combined_portfolio_v2``,
ranks by walk-forward average Sharpe, and exports:

1. ``docs/research/sweep_<YYYY-MM-DD>_combinations.md`` — human-readable
   markdown report with exec summary, full ranked table and per-block
   narratives.
2. ``results/sweep_<YYYY-MM-DD>/metrics.json`` — machine-readable dump
   of every combo's metrics (including bootstrap results for the top-5).
3. ``results/sweep_<YYYY-MM-DD>/tearsheets/top{1..5}.html`` — QuantStats
   tearsheets for the five best combos.
4. ``results/sweep_<YYYY-MM-DD>/mix_plots/top{1..3}/*.html`` — portfolio-mix
   plots (weights over time, contributions, rolling correlation) for the
   three best combos.

Usage
-----
    # Full run (~1h30 with bootstrap on top-5)
    .venv/bin/python scripts/sweep_combinations.py

    # Sweep only (no bootstrap, ~3 min)
    .venv/bin/python scripts/sweep_combinations.py --no-bootstrap

    # Smoke test (3 combos, no bootstrap, writes to /tmp)
    .venv/bin/python scripts/sweep_combinations.py --smoke

Design notes
------------
- Sleeves are loaded exactly once via ``get_strategy_daily_returns`` and
  reused across every combo — the intraday MR Macro load dominates the
  cost (~1-2 min).
- Block-bootstrap uses an in-script helper (``bootstrap_config``) that
  accepts an arbitrary config kwargs dict instead of hard-coding
  Phase 18 like the legacy ``stress_test_combined.run_block_bootstrap``.
- ``regime_adaptive`` on the Phase 18 trio is supported via the
  ``P18_REGIME_WEIGHTS`` mapping below (same structure as
  ``DEFAULT_REGIME_WEIGHTS_3STRAT`` but keyed on MR/TS3p/RSI).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SRC = _PROJECT_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
if str(_PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "scripts"))


# ═══════════════════════════════════════════════════════════════════════
# Regime weights templates
# ═══════════════════════════════════════════════════════════════════════


# Legacy v1 trio (MR_Macro + XS_Momentum + TS_Momentum_RSI) — used by
# DEFAULT_REGIME_WEIGHTS_3STRAT in combined_portfolio_v2.
#
# Phase 18 trio (MR_Macro + TS_Momentum_3p + RSI_Daily_4p) — same
# structural pattern but substitutes TS3p for XS (both trend-ish) and
# RSI_4p for TS_RSI (both mean-reverting).
P18_REGIME_WEIGHTS: dict[tuple[str, str], dict[str, float]] = {
    ("low",    "up"):   {"MR_Macro": 0.40, "TS_Momentum_3p": 0.35, "RSI_Daily_4p": 0.25},
    ("low",    "down"): {"MR_Macro": 0.55, "TS_Momentum_3p": 0.25, "RSI_Daily_4p": 0.20},
    ("normal", "up"):   {"MR_Macro": 0.45, "TS_Momentum_3p": 0.30, "RSI_Daily_4p": 0.25},
    ("normal", "down"): {"MR_Macro": 0.60, "TS_Momentum_3p": 0.20, "RSI_Daily_4p": 0.20},
    ("high",   "up"):   {"MR_Macro": 0.60, "TS_Momentum_3p": 0.25, "RSI_Daily_4p": 0.15},
    ("high",   "down"): {"MR_Macro": 0.75, "TS_Momentum_3p": 0.15, "RSI_Daily_4p": 0.10},
}


# ═══════════════════════════════════════════════════════════════════════
# Sweep configuration dataclass
# ═══════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class SweepConfig:
    id: str
    block: str
    name: str
    sleeves: tuple[str, ...]
    allocation: str  # risk_parity / equal / mr_heavy / custom / regime_adaptive
    target_vol: float | None
    max_leverage: float
    dd_cap_enabled: bool
    custom_weights: dict[str, float] | None = None
    regime_weights: dict[tuple[str, str], dict[str, float]] | None = None
    # Optional DD cap schedule overrides (Phase 20C). Both tuples must
    # be the same length. When None the module defaults are used (the
    # historical Phase 13 schedule).
    dd_breakpoints: tuple[float, ...] | None = None
    dd_scales: tuple[float, ...] | None = None

    def describe(self) -> str:
        tv = f"tv={self.target_vol:.2f}" if self.target_vol is not None else "tv=None"
        ml = f"ml={self.max_leverage:.0f}" if self.max_leverage is not None else ""
        dd = "ddON" if self.dd_cap_enabled else "ddOFF"
        return f"{self.allocation} / {tv} / {ml} / {dd}"


_P18 = {"MR_Macro": 0.80, "TS_Momentum_3p": 0.10, "RSI_Daily_4p": 0.10}
_P18_SLEEVES = ("MR_Macro", "TS_Momentum_3p", "RSI_Daily_4p")
_V1_SLEEVES = ("MR_Macro", "XS_Momentum", "TS_Momentum_RSI")


def build_grid() -> list[SweepConfig]:
    """Return the full ordered list of sweep configurations."""
    cfgs: list[SweepConfig] = []

    # ─── Bloc A — Phase 18 sleeves × allocation variations ──────────
    cfgs += [
        SweepConfig("A1", "A", "P18-sleeves / risk_parity / no lev",
                    _P18_SLEEVES, "risk_parity", None, 5.0, False),
        SweepConfig("A2", "A", "P18-sleeves / risk_parity / tv=0.15 ml=5 DDon",
                    _P18_SLEEVES, "risk_parity", 0.15, 5.0, True),
        SweepConfig("A3", "A", "P18-sleeves / equal / no lev",
                    _P18_SLEEVES, "equal", None, 5.0, False),
        SweepConfig("A4", "A", "P18-sleeves / equal / tv=0.18 ml=6",
                    _P18_SLEEVES, "equal", 0.18, 6.0, False),
        SweepConfig("A5", "A", "P18-sleeves / regime_adaptive / no lev",
                    _P18_SLEEVES, "regime_adaptive", None, 5.0, False,
                    regime_weights=P18_REGIME_WEIGHTS),
        SweepConfig("A6", "A", "P18-sleeves / regime_adaptive / tv=0.14 ml=4 DDon",
                    _P18_SLEEVES, "regime_adaptive", 0.14, 4.0, True,
                    regime_weights=P18_REGIME_WEIGHTS),
        SweepConfig("A7", "A", "P18-prod / tv=0.28 ml=12 DDon",
                    _P18_SLEEVES, "custom", 0.28, 12.0, True,
                    custom_weights=dict(_P18)),
        SweepConfig("A8", "A", "P18-prod / tv=0.20 ml=8 DDon",
                    _P18_SLEEVES, "custom", 0.20, 8.0, True,
                    custom_weights=dict(_P18)),
    ]

    # ─── Bloc B — alt-sleeve trios ─────────────────────────────────
    MR_XS_RSI = ("MR_Macro", "XS_Momentum", "RSI_Daily_4p")
    MR_RSI = ("MR_Macro", "RSI_Daily_4p")
    MR_TS3P = ("MR_Macro", "TS_Momentum_3p")
    MR_TSRSI_RSI = ("MR_Macro", "TS_Momentum_RSI", "RSI_Daily_4p")
    ALL4 = ("MR_Macro", "TS_Momentum_3p", "RSI_Daily_4p", "XS_Momentum")
    cfgs += [
        SweepConfig("B1", "B", "MR+XS+RSI / 80-10-10 / tv=0.28 ml=12",
                    MR_XS_RSI, "custom", 0.28, 12.0, False,
                    custom_weights={"MR_Macro": 0.80, "XS_Momentum": 0.10, "RSI_Daily_4p": 0.10}),
        SweepConfig("B2", "B", "MR+XS+RSI / 80-10-10 / tv=0.20 ml=8 DDon",
                    MR_XS_RSI, "custom", 0.20, 8.0, True,
                    custom_weights={"MR_Macro": 0.80, "XS_Momentum": 0.10, "RSI_Daily_4p": 0.10}),
        SweepConfig("B3", "B", "MR+RSI / 90-10 / tv=0.28 ml=12",
                    MR_RSI, "custom", 0.28, 12.0, False,
                    custom_weights={"MR_Macro": 0.90, "RSI_Daily_4p": 0.10}),
        SweepConfig("B4", "B", "MR+TS3p / 90-10 / tv=0.28 ml=12 DDon (P17+DD)",
                    MR_TS3P, "custom", 0.28, 12.0, True,
                    custom_weights={"MR_Macro": 0.90, "TS_Momentum_3p": 0.10}),
        SweepConfig("B5", "B", "MR+TS_RSI(CAD)+RSI / 80-10-10 / tv=0.28 ml=12",
                    MR_TSRSI_RSI, "custom", 0.28, 12.0, False,
                    custom_weights={"MR_Macro": 0.80, "TS_Momentum_RSI": 0.10, "RSI_Daily_4p": 0.10}),
        SweepConfig("B6", "B", "All-4-sleeve / 70-10-10-10 / tv=0.28 ml=12",
                    ALL4, "custom", 0.28, 12.0, False,
                    custom_weights={"MR_Macro": 0.70, "TS_Momentum_3p": 0.10,
                                    "RSI_Daily_4p": 0.10, "XS_Momentum": 0.10}),
        SweepConfig("B7", "B", "All-4-sleeve / risk_parity / no lev",
                    ALL4, "risk_parity", None, 5.0, False),
        SweepConfig("B8", "B", "All-4-sleeve / equal / tv=0.18 ml=6",
                    ALL4, "equal", 0.18, 6.0, False),
    ]

    # ─── Bloc C — non-MR "all-daily" ───────────────────────────────
    XS_TS3P_RSI = ("XS_Momentum", "TS_Momentum_3p", "RSI_Daily_4p")
    XS_RSI = ("XS_Momentum", "RSI_Daily_4p")
    TS3P_RSI = ("TS_Momentum_3p", "RSI_Daily_4p")
    cfgs += [
        SweepConfig("C1", "C", "XS+TS3p+RSI / equal / no lev",
                    XS_TS3P_RSI, "equal", None, 5.0, False),
        SweepConfig("C2", "C", "XS+TS3p+RSI / equal / tv=0.15 ml=5 DDon",
                    XS_TS3P_RSI, "equal", 0.15, 5.0, True),
        SweepConfig("C3", "C", "XS+TS3p+RSI / risk_parity / tv=0.18 ml=6",
                    XS_TS3P_RSI, "risk_parity", 0.18, 6.0, False),
        SweepConfig("C4", "C", "XS+RSI / 50-50 / tv=0.15 ml=5 DDon",
                    XS_RSI, "custom", 0.15, 5.0, True,
                    custom_weights={"XS_Momentum": 0.50, "RSI_Daily_4p": 0.50}),
        SweepConfig("C5", "C", "TS3p+RSI / 50-50 / tv=0.12 ml=4 DDon",
                    TS3P_RSI, "custom", 0.12, 4.0, True,
                    custom_weights={"TS_Momentum_3p": 0.50, "RSI_Daily_4p": 0.50}),
    ]

    # ─── Bloc D — Vol target sweep on P18 ──────────────────────────
    cfgs += [
        SweepConfig("D1", "D", "P18-prod / tv=0.10 ml=3 DDon",
                    _P18_SLEEVES, "custom", 0.10, 3.0, True, custom_weights=dict(_P18)),
        SweepConfig("D2", "D", "P18-prod / tv=0.15 ml=4 DDon",
                    _P18_SLEEVES, "custom", 0.15, 4.0, True, custom_weights=dict(_P18)),
        SweepConfig("D3", "D", "P18-prod / tv=0.18 ml=6",
                    _P18_SLEEVES, "custom", 0.18, 6.0, False, custom_weights=dict(_P18)),
        SweepConfig("D4", "D", "P18-prod / tv=0.22 ml=8",
                    _P18_SLEEVES, "custom", 0.22, 8.0, False, custom_weights=dict(_P18)),
        SweepConfig("D5", "D", "P18-prod / tv=0.25 ml=10",
                    _P18_SLEEVES, "custom", 0.25, 10.0, False, custom_weights=dict(_P18)),
        SweepConfig("D6", "D", "P18-prod / tv=0.28 ml=12 DDon",
                    _P18_SLEEVES, "custom", 0.28, 12.0, True, custom_weights=dict(_P18)),
    ]

    # ─── Bloc E — DD cap reactivation on P18 ───────────────────────
    cfgs += [
        SweepConfig("E1", "E", "P18-prod / tv=0.28 ml=6 DDon",
                    _P18_SLEEVES, "custom", 0.28, 6.0, True, custom_weights=dict(_P18)),
        SweepConfig("E2", "E", "P18-prod / tv=0.28 ml=8 DDon",
                    _P18_SLEEVES, "custom", 0.28, 8.0, True, custom_weights=dict(_P18)),
        SweepConfig("E3", "E", "P18-prod / tv=0.28 ml=10 DDon",
                    _P18_SLEEVES, "custom", 0.28, 10.0, True, custom_weights=dict(_P18)),
        SweepConfig("E4", "E", "P18-prod / tv=0.28 ml=12 DDon (dup A7)",
                    _P18_SLEEVES, "custom", 0.28, 12.0, True, custom_weights=dict(_P18)),
        SweepConfig("E5", "E", "P18-prod / tv=0.28 ml=14 DDon",
                    _P18_SLEEVES, "custom", 0.28, 14.0, True, custom_weights=dict(_P18)),
    ]

    # ─── Bloc F — Baselines ────────────────────────────────────────
    cfgs += [
        SweepConfig("F1", "F", "Phase 18 prod (MR80/TS3p10/RSI10 tv=0.28 ml=12)",
                    _P18_SLEEVES, "custom", 0.28, 12.0, False, custom_weights=dict(_P18)),
        SweepConfig("F2", "F", "Phase 17 (MR90/TS3p10 tv=0.28 ml=12)",
                    MR_TS3P, "custom", 0.28, 12.0, False,
                    custom_weights={"MR_Macro": 0.90, "TS_Momentum_3p": 0.10}),
        SweepConfig("F3", "F", "Phase 16 (MR90/TS_RSI10 tv=0.28 ml=12)",
                    ("MR_Macro", "TS_Momentum_RSI"), "custom", 0.28, 12.0, False,
                    custom_weights={"MR_Macro": 0.90, "TS_Momentum_RSI": 0.10}),
        SweepConfig("F4", "F", "v1 risk_parity (MR+XS+TS_RSI)",
                    _V1_SLEEVES, "risk_parity", None, 5.0, False),
        SweepConfig("F5", "F", "v1 mr_heavy (50/25/25)",
                    _V1_SLEEVES, "mr_heavy", None, 5.0, False),
    ]

    return cfgs


# ═══════════════════════════════════════════════════════════════════════
# Native vbt parallel sweep: one Portfolio.from_optimizer call for all combos
# ═══════════════════════════════════════════════════════════════════════


# Walk-forward windows (same as combined_portfolio.WF_PERIODS)
WF_PERIODS: list[tuple[str, str]] = [
    (f"{y}-01-01", f"{y}-12-31") for y in range(2019, 2025)
] + [("2025-01-01", "2026-04-01")]


def subset_sleeves(
    sleeves_all: dict[str, pd.Series], names: tuple[str, ...]
) -> dict[str, pd.Series]:
    missing = [n for n in names if n not in sleeves_all]
    if missing:
        raise KeyError(
            f"Sleeves {missing} not available. "
            f"Available: {sorted(sleeves_all)}"
        )
    return {n: sleeves_all[n] for n in names}


def compute_final_allocations(
    cfg: SweepConfig,
    common_all: pd.DataFrame,
) -> pd.DataFrame:
    """Compute the final (post-leverage, post-DD-cap) allocations for one config.

    Returns a DataFrame indexed by ``common_all.index`` with columns equal
    to the 5-sleeve superset. Sleeves the config does not use get zero
    weight. The output is ready to be stacked into the wide MultiIndex
    allocations frame consumed by the native parallel sweep.
    """
    from strategies.combined_portfolio import _compute_weights_ts
    from strategies.combined_portfolio_v2 import (
        compute_dd_cap_scale,
        compute_global_leverage,
        compute_regime_adaptive_weights,
    )

    subset = common_all[list(cfg.sleeves)]

    # 1. Base weights.
    if cfg.allocation == "regime_adaptive":
        weights_ts, _, _ = compute_regime_adaptive_weights(
            subset, regime_weights=cfg.regime_weights
        )
    else:
        weights_ts = _compute_weights_ts(subset, cfg.allocation, cfg.custom_weights)

    # 2. Base portfolio returns (proxy for leverage / DD cap drivers).
    port_rets_base = (subset * weights_ts).sum(axis=1)

    # 3. Optional vol targeting.
    if cfg.target_vol is None:
        leverage_ts = pd.Series(1.0, index=subset.index)
    else:
        leverage_ts = compute_global_leverage(
            port_rets_base,
            target_vol=cfg.target_vol,
            max_leverage=cfg.max_leverage,
        )
    port_rets_prelev = port_rets_base * leverage_ts

    # 4. Optional DD cap — with optional per-config schedule override.
    if cfg.dd_cap_enabled:
        dd_scale_ts = compute_dd_cap_scale(
            port_rets_prelev,
            breakpoints=cfg.dd_breakpoints,
            scales=cfg.dd_scales,
        )
    else:
        dd_scale_ts = pd.Series(1.0, index=subset.index)

    # 5. Final allocations = weights × leverage × dd_scale.
    final_alloc = weights_ts.mul(leverage_ts * dd_scale_ts, axis=0)

    # 6. Expand to the 5-sleeve superset (zero for unused sleeves).
    expanded = pd.DataFrame(
        0.0, index=common_all.index, columns=common_all.columns, dtype=float
    )
    expanded.loc[:, final_alloc.columns] = final_alloc.values
    return expanded


def native_parallel_sweep(
    grid: list[SweepConfig],
    sleeves_all: dict[str, pd.Series],
    init_cash: float = 1_000_000.0,
) -> tuple[Any, pd.DataFrame]:
    """Execute the whole grid in a single native vbt ``from_optimizer`` call.

    Builds a wide MultiIndex-columns allocations DataFrame (``config_id`` ×
    sleeve), assembles a matching price frame by broadcasting the synthetic
    per-sleeve prices across configs, then routes everything through
    ``vbt.PortfolioOptimizer.from_filled_allocations`` +
    ``vbt.Portfolio.from_optimizer`` with ``group_by="config"``. The result
    is a single grouped portfolio that exposes all 37 combos in parallel —
    ``pf.sharpe_ratio``, ``pf.annualized_return`` etc. all return Series
    indexed by ``config_id``.

    Returns
    -------
    pf_all : vbt.Portfolio
        Grouped portfolio (one group per config).
    metrics_df : DataFrame
        Per-config metrics (sharpe, annual_return, annual_vol,
        max_drawdown, wf_avg_sharpe, wf_pos_years, wf_sharpes) as a
        tidy DataFrame indexed by config_id.
    """
    import vectorbtpro as vbt

    # Common intersection index across all 5 sleeves (consistent apples-to-
    # apples comparison across configs).
    common_all = pd.DataFrame(sleeves_all).dropna()
    all_sleeve_names = list(common_all.columns)

    # 1. Per-config final allocations on the superset.
    alloc_frames: dict[str, pd.DataFrame] = {}
    for cfg in grid:
        alloc_frames[cfg.id] = compute_final_allocations(cfg, common_all)

    # 2. Wide MultiIndex-columns frame: (config_id, sleeve).
    wide_alloc = pd.concat(alloc_frames, axis=1)
    wide_alloc.columns.names = ["config", "sleeve"]

    # 3. Matching price frame: for each (config_id, sleeve) column, use the
    # sleeve's synthetic cumulative-return price (repeated across configs).
    synthetic_prices = (1.0 + common_all.fillna(0.0)).cumprod() * 1000.0
    # Broadcast: pick the price by the inner 'sleeve' level.
    inner_sleeve = wide_alloc.columns.get_level_values("sleeve")
    wide_prices = synthetic_prices[inner_sleeve].copy()
    wide_prices.columns = wide_alloc.columns

    # 4. Temporal shift: PFO applies allocation at bar t to the position
    # held during bar t+1. The legacy pattern applies weights[t] to ret[t]
    # directly. Shift forward by 1 to align.
    wide_alloc_shifted = wide_alloc.shift(-1).fillna(0.0)

    # Compute an adaptive leverage cap: max row sum in any group × safety
    # margin. This must be >= the highest per-row total that any config
    # ever reaches (leverage_ts * dd_scale_ts can exceed 1).
    max_row_sum_per_config = (
        wide_alloc_shifted.T.groupby(level="config").sum().T.max().max()
    )
    leverage_cap = max(float(max_row_sum_per_config) * 1.5, 10.0)

    # 5. Native parallel build.
    pfo = vbt.PortfolioOptimizer.from_filled_allocations(
        wide_alloc_shifted,
        valid_only=False,
        nonzero_only=False,
        unique_only=False,
    )
    pf_all = vbt.Portfolio.from_optimizer(
        wide_prices,
        pfo,
        pf_method="from_orders",
        size_type="targetpercent",
        cash_sharing=True,
        group_by="config",
        init_cash=init_cash,
        fees=0.0,
        slippage=0.0,
        leverage=leverage_cap,
        leverage_mode="eager",
        freq="1D",
    )

    # 6. Extract per-config metrics (all native, parallelized by vbt).
    sharpe_series = pf_all.sharpe_ratio
    cagr_series = pf_all.annualized_return
    vol_series = pf_all.annualized_volatility
    mdd_series = pf_all.max_drawdown

    # Fix for vbt returning a single float when groups have only one entry
    # (shouldn't happen here — we have 37 groups).
    config_ids = [cfg.id for cfg in grid]
    sharpe_series = pd.Series(sharpe_series).reindex(config_ids)
    cagr_series = pd.Series(cagr_series).reindex(config_ids)
    vol_series = pd.Series(vol_series).reindex(config_ids)
    mdd_series = pd.Series(mdd_series).reindex(config_ids)

    # 7. Walk-forward per config — native per-column Sharpe via vbt
    # ReturnsAccessor on the grouped returns DataFrame. Each window
    # produces a Series[config_id], stacked into a matrix.
    grouped_returns = pf_all.returns  # DataFrame, columns=config_id
    if isinstance(grouped_returns, pd.Series):
        grouped_returns = grouped_returns.to_frame()

    wf_sharpe_matrix = []
    for start, end in WF_PERIODS:
        window = grouped_returns.loc[start:end]
        if len(window) < 20:
            wf_sharpe_matrix.append(pd.Series(0.0, index=config_ids))
            continue
        # Wrap each column in a synthetic from_holding portfolio (same
        # convention as combined_core.sharpe_for_window) and call
        # .sharpe_ratio — vectorized across columns.
        synthetic_window_price = (1.0 + window.fillna(0.0)).cumprod() * 1000.0
        pf_window = vbt.Portfolio.from_holding(
            close=synthetic_window_price,
            init_cash=init_cash,
            freq="1D",
        )
        sr_window = pd.Series(pf_window.sharpe_ratio).reindex(config_ids).fillna(0.0)
        wf_sharpe_matrix.append(sr_window)

    wf_df = pd.concat(wf_sharpe_matrix, axis=1)
    wf_df.columns = [f"wf_{i}" for i in range(len(WF_PERIODS))]

    metrics = pd.DataFrame({
        "sharpe": sharpe_series,
        "annual_return": cagr_series,
        "annual_vol": vol_series,
        "max_drawdown": mdd_series,
        "wf_avg_sharpe": wf_df.mean(axis=1),
        "wf_pos_years": (wf_df > 0).sum(axis=1),
    })
    for col in wf_df.columns:
        metrics[col] = wf_df[col]

    return pf_all, metrics


def rows_from_native_metrics(
    grid: list[SweepConfig],
    metrics: pd.DataFrame,
    pf_all: Any,
) -> list[dict[str, Any]]:
    """Assemble row dicts from the native parallel metrics DataFrame."""
    rows: list[dict[str, Any]] = []
    n_wf = len(WF_PERIODS)
    for cfg in grid:
        if cfg.id not in metrics.index:
            continue
        m = metrics.loc[cfg.id]
        wf_sharpes = [float(m[f"wf_{i}"]) for i in range(n_wf)]
        rows.append({
            "id": cfg.id,
            "block": cfg.block,
            "name": cfg.name,
            "sleeves": list(cfg.sleeves),
            "allocation": cfg.allocation,
            "target_vol": cfg.target_vol,
            "max_leverage": cfg.max_leverage,
            "dd_cap_enabled": cfg.dd_cap_enabled,
            "custom_weights": cfg.custom_weights,
            "sharpe": float(m["sharpe"]) if not np.isnan(m["sharpe"]) else 0.0,
            "annual_return": float(m["annual_return"]) if not np.isnan(m["annual_return"]) else 0.0,
            "annual_vol": float(m["annual_vol"]) if not np.isnan(m["annual_vol"]) else 0.0,
            "max_drawdown": float(m["max_drawdown"]) if not np.isnan(m["max_drawdown"]) else 0.0,
            "wf_avg_sharpe": float(m["wf_avg_sharpe"]),
            "wf_pos_years": int(m["wf_pos_years"]),
            "wf_sharpes": wf_sharpes,
            # Per-config sub-portfolio view for tearsheets (native vbt slice)
            "_pf_sub": pf_all[cfg.id] if cfg.id in pf_all.wrapper.grouper.get_index() else None,
        })
    return rows


# ═══════════════════════════════════════════════════════════════════════
# Bootstrap (arbitrary config) — inlined copy of stress_test_combined
# ═══════════════════════════════════════════════════════════════════════


def block_bootstrap_indices(
    n: int, block_size: int, rng: np.random.Generator
) -> np.ndarray:
    if block_size <= 0 or block_size > n:
        raise ValueError(f"block_size must be in (0, {n}], got {block_size}")
    n_blocks = n // block_size + (1 if n % block_size else 0)
    starts = rng.integers(0, n - block_size + 1, size=n_blocks)
    return np.concatenate(
        [np.arange(s, s + block_size) for s in starts]
    )[:n]


def bootstrap_config(
    cfg: SweepConfig,
    sleeves_all: dict[str, pd.Series],
    n_runs: int = 500,
    block_size: int = 20,
    seed: int = 20260413,
) -> dict[str, float]:
    """Block-bootstrap the given sweep config. Returns summary statistics."""
    from strategies.combined_portfolio_v2 import build_combined_portfolio_v2

    strat_rets = subset_sleeves(sleeves_all, cfg.sleeves)
    df = pd.DataFrame(strat_rets).dropna()
    n_bars = len(df)
    rng = np.random.default_rng(seed)

    kwargs: dict[str, Any] = dict(
        allocation=cfg.allocation,
        target_vol=cfg.target_vol,
        max_leverage=cfg.max_leverage,
        dd_cap_enabled=cfg.dd_cap_enabled,
    )
    if cfg.custom_weights is not None:
        kwargs["custom_weights"] = cfg.custom_weights
    if cfg.regime_weights is not None:
        kwargs["regime_weights"] = cfg.regime_weights

    cagrs: list[float] = []
    dds: list[float] = []
    sharpes: list[float] = []

    for i in range(n_runs):
        idx = block_bootstrap_indices(n_bars, block_size, rng)
        resampled = df.iloc[idx].reset_index(drop=True)
        resampled.index = pd.bdate_range("2000-01-03", periods=n_bars)
        strat_resampled = {c: resampled[c] for c in df.columns}
        try:
            res = build_combined_portfolio_v2(strat_resampled, **kwargs)
        except Exception:
            continue
        cagrs.append(res["annual_return"])
        dds.append(res["max_drawdown"])
        sr = res["sharpe"]
        sharpes.append(0.0 if np.isnan(sr) else sr)

    cagr_arr = np.asarray(cagrs)
    dd_arr = np.asarray(dds)
    sharpe_arr = np.asarray(sharpes)
    target_hit = (cagr_arr >= 0.10) & (cagr_arr <= 0.15) & (dd_arr > -0.35)

    return {
        "n_runs": int(len(cagrs)),
        "cagr_p05": float(np.percentile(cagr_arr, 5)),
        "cagr_p50": float(np.percentile(cagr_arr, 50)),
        "cagr_p95": float(np.percentile(cagr_arr, 95)),
        "max_dd_p05": float(np.percentile(dd_arr, 5)),
        "max_dd_p50": float(np.percentile(dd_arr, 50)),
        "max_dd_p95": float(np.percentile(dd_arr, 95)),
        "sharpe_p05": float(np.percentile(sharpe_arr, 5)),
        "sharpe_p95": float(np.percentile(sharpe_arr, 95)),
        "pos_fraction": float((cagr_arr > 0).mean()),
        "target_hit_fraction": float(target_hit.mean()),
    }


# ═══════════════════════════════════════════════════════════════════════
# Reporting helpers
# ═══════════════════════════════════════════════════════════════════════


def _fmt_pct(x: float, width: int = 8) -> str:
    if x is None or np.isnan(x):
        return "".rjust(width)
    return f"{x * 100:>{width - 1}.2f}%"


def _target_hit_mark(cagr: float, mdd: float) -> str:
    hit = (0.10 <= cagr <= 0.15) and (abs(mdd) < 0.35)
    return "*" if hit else " "


def build_markdown_report(
    rows: list[dict[str, Any]],
    bootstrap_by_id: dict[str, dict[str, float]],
    report_date: str,
) -> str:
    """Return the full markdown report as a string."""
    # Sort by wf_avg_sharpe descending.
    sorted_rows = sorted(rows, key=lambda r: r["wf_avg_sharpe"], reverse=True)
    top5 = sorted_rows[:5]
    p18_row = next((r for r in rows if r["id"] == "F1"), None)

    lines: list[str] = []
    lines.append(f"# Sweep combinations multi-stratégies — {report_date}\n")
    lines.append(
        "Sweep systématique de 37 configurations du combined portfolio v2, "
        "couvrant des variations non testées par `run_v2_benchmark` "
        "(alt-sleeve trios, allocations alternatives sur Phase 18, vol target "
        "sweep, DD cap réactivé, non-MR configs).\n"
    )
    lines.append(
        f"Baseline de comparaison : **Phase 18 prod** "
        f"(`F1` ci-dessous) — Sharpe WF "
        f"{p18_row['wf_avg_sharpe']:.3f}, "
        f"CAGR {p18_row['annual_return']*100:.2f}%, "
        f"MaxDD {p18_row['max_drawdown']*100:.2f}%.\n"
        if p18_row
        else ""
    )

    # ── Top-5 résumé ───────────────────────────────────────────────
    lines.append("## Top 5 par Walk-Forward Sharpe\n")
    lines.append(
        "| Rank | ID | Config | Sleeves | Sharpe WF | CAGR | Vol | MaxDD | WF pos | ★ |"
    )
    lines.append("|------|----|--------|---------|-----------|------|-----|-------|--------|----|")
    for i, r in enumerate(top5, 1):
        mark = _target_hit_mark(r["annual_return"], r["max_drawdown"])
        sleeves_short = "+".join(s.replace("_Momentum", "").replace("_Macro", "").replace("_Daily", "") for s in r["sleeves"])
        lines.append(
            f"| {i} | {r['id']} | {r['name']} | {sleeves_short} "
            f"| **{r['wf_avg_sharpe']:.3f}** "
            f"| {_fmt_pct(r['annual_return'])} "
            f"| {_fmt_pct(r['annual_vol'])} "
            f"| {_fmt_pct(r['max_drawdown'])} "
            f"| {r['wf_pos_years']}/7 | {mark} |"
        )
    lines.append("\n★ = CAGR ∈ [10%, 15%] AND MaxDD < 35%.\n")

    # ── Bootstrap summary for top-5 ───────────────────────────────
    if bootstrap_by_id:
        lines.append("## Bootstrap stress-test (top-5, 500 × 20-day blocks)\n")
        lines.append(
            "| Rank | ID | CAGR P5 | CAGR P50 | MaxDD P5 | MaxDD P50 | Sharpe P5 | Sharpe P95 | Pos frac | Target hit |"
        )
        lines.append(
            "|------|----|---------|----------|----------|-----------|-----------|-----------|----------|------------|"
        )
        for i, r in enumerate(top5, 1):
            b = bootstrap_by_id.get(r["id"])
            if b is None:
                continue
            lines.append(
                f"| {i} | {r['id']} "
                f"| {_fmt_pct(b['cagr_p05'])} "
                f"| {_fmt_pct(b['cagr_p50'])} "
                f"| {_fmt_pct(b['max_dd_p05'])} "
                f"| {_fmt_pct(b['max_dd_p50'])} "
                f"| {b['sharpe_p05']:>+7.3f} "
                f"| {b['sharpe_p95']:>+7.3f} "
                f"| {b['pos_fraction']*100:>6.1f}% "
                f"| {b['target_hit_fraction']*100:>6.1f}% |"
            )
        lines.append("")

    # ── Tableau complet ───────────────────────────────────────────
    lines.append("## Tableau complet (37 configs, trié par Sharpe WF)\n")
    lines.append(
        "| ID | Block | Config | Sharpe WF | CAGR | Vol | MaxDD | WF pos | Full Sharpe | ★ |"
    )
    lines.append("|----|-------|--------|-----------|------|-----|-------|--------|-------------|----|")
    for r in sorted_rows:
        mark = _target_hit_mark(r["annual_return"], r["max_drawdown"])
        lines.append(
            f"| {r['id']} | {r['block']} | {r['name']} "
            f"| {r['wf_avg_sharpe']:.3f} "
            f"| {_fmt_pct(r['annual_return'])} "
            f"| {_fmt_pct(r['annual_vol'])} "
            f"| {_fmt_pct(r['max_drawdown'])} "
            f"| {r['wf_pos_years']}/7 "
            f"| {r['sharpe']:.3f} | {mark} |"
        )
    lines.append("")

    # ── Narratifs par bloc ────────────────────────────────────────
    lines.append("## Lectures par bloc\n")
    blocks_desc = {
        "A": "Phase 18 sleeves × allocations alternatives",
        "B": "Alt-sleeve trios (différentes compositions 2-4 sleeves)",
        "C": "Non-MR 'all-daily' (robustness check sans MR Macro)",
        "D": "Vol target sweep sur Phase 18 sleeves",
        "E": "DD cap reactivation sur Phase 18 (ml ∈ [6, 14])",
        "F": "Baselines (Phase 18, 17, 16, v1)",
    }
    for block_id, desc in blocks_desc.items():
        block_rows = [r for r in sorted_rows if r["block"] == block_id]
        if not block_rows:
            continue
        best = max(block_rows, key=lambda r: r["wf_avg_sharpe"])
        worst = min(block_rows, key=lambda r: r["wf_avg_sharpe"])
        lines.append(f"### Bloc {block_id} — {desc}\n")
        lines.append(
            f"- **{len(block_rows)} configs** testées. Meilleure : `{best['id']}` "
            f"(Sharpe WF {best['wf_avg_sharpe']:.3f}, CAGR "
            f"{best['annual_return']*100:.2f}%, MaxDD "
            f"{best['max_drawdown']*100:.2f}%)."
        )
        lines.append(
            f"- **Pire** : `{worst['id']}` (Sharpe WF "
            f"{worst['wf_avg_sharpe']:.3f}, CAGR "
            f"{worst['annual_return']*100:.2f}%)."
        )
        if p18_row:
            delta_sharpe = best["wf_avg_sharpe"] - p18_row["wf_avg_sharpe"]
            sign = "+" if delta_sharpe >= 0 else ""
            lines.append(
                f"- vs Phase 18 prod : Δ Sharpe WF = {sign}{delta_sharpe:.3f}."
            )
        lines.append("")

    return "\n".join(lines)


def sanitize_result_for_json(row: dict[str, Any]) -> dict[str, Any]:
    """Drop non-serializable keys from a sweep row."""
    return {k: v for k, v in row.items() if not k.startswith("_")}


# ═══════════════════════════════════════════════════════════════════════
# Top-N artifacts (tearsheets + mix plots)
# ═══════════════════════════════════════════════════════════════════════


def generate_top_n_artifacts(
    sorted_rows: list[dict[str, Any]],
    grid: list[SweepConfig],
    sleeves_all: dict[str, pd.Series],
    output_dir: Path,
    n_tearsheets: int = 5,
    n_mix_plots: int = 3,
) -> None:
    """Rebuild top-N portfolios individually (for tearsheet access) + mix plots.

    The native parallel sweep returns a grouped portfolio where per-config
    slicing exposes summary metrics but not everything a QuantStats tearsheet
    needs. For the handful of top-N combos, we re-run ``build_combined_portfolio_v2``
    to get the full per-config result dict — this is cheap (5-8 extra combos)
    and keeps the parallel sweep lean.
    """
    from framework.plotting import (
        generate_html_tearsheet,
        generate_portfolio_mix_plots,
        save_fullscreen_html,
    )
    from strategies.combined_portfolio_v2 import build_combined_portfolio_v2

    tear_dir = output_dir / "tearsheets"
    mix_dir = output_dir / "mix_plots"
    tear_dir.mkdir(parents=True, exist_ok=True)
    mix_dir.mkdir(parents=True, exist_ok=True)

    n_total = max(n_tearsheets, n_mix_plots)
    for rank, row in enumerate(sorted_rows[:n_total], 1):
        cfg = next(c for c in grid if c.id == row["id"])
        strat_rets = subset_sleeves(sleeves_all, cfg.sleeves)
        kwargs: dict[str, Any] = dict(
            allocation=cfg.allocation,
            target_vol=cfg.target_vol,
            max_leverage=cfg.max_leverage,
            dd_cap_enabled=cfg.dd_cap_enabled,
        )
        if cfg.custom_weights is not None:
            kwargs["custom_weights"] = cfg.custom_weights
        if cfg.regime_weights is not None:
            kwargs["regime_weights"] = cfg.regime_weights

        try:
            res = build_combined_portfolio_v2(strat_rets, **kwargs)
        except Exception as exc:
            print(f"  [top{rank} {row['id']}] rebuild FAILED: {exc}")
            continue

        if rank <= n_tearsheets:
            path = tear_dir / f"top{rank}_{row['id']}.html"
            try:
                generate_html_tearsheet(res["pf_combined"], str(path), title=f"Top {rank} — {row['name']}")
                print(f"  [tearsheet top{rank}] {path}")
            except Exception as exc:
                print(f"  [tearsheet {row['id']}] SKIPPED: {exc}")

        if rank <= n_mix_plots:
            sub = mix_dir / f"top{rank}_{row['id']}"
            sub.mkdir(parents=True, exist_ok=True)
            try:
                figs = generate_portfolio_mix_plots(res, show=False)
                for name, fig in figs.items():
                    save_fullscreen_html(fig, str(sub / f"{name}.html"))
                print(f"  [mix plots top{rank}] {len(figs)} figures in {sub}")
            except Exception as exc:
                print(f"  [mix plots {row['id']}] SKIPPED: {exc}")


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true",
                        help="Run 3 configs only, no bootstrap, write to /tmp")
    parser.add_argument("--no-bootstrap", action="store_true",
                        help="Skip the bootstrap stress-test on top-5")
    parser.add_argument("--bootstrap-runs", type=int, default=500,
                        help="Bootstrap runs per top-5 combo (default 500)")
    args = parser.parse_args()

    import pandas as pd
    import vectorbtpro as vbt

    from strategies.combined_portfolio import get_strategy_daily_returns
    from utils import apply_vbt_settings

    apply_vbt_settings()
    vbt.settings.returns.year_freq = pd.Timedelta(days=252)

    report_date = date.today().isoformat()
    if args.smoke:
        output_root = Path("/tmp") / f"sweep_smoke_{report_date}"
        md_path = output_root / "sweep.md"
    else:
        output_root = _PROJECT_ROOT / "results" / f"sweep_{report_date}"
        md_path = _PROJECT_ROOT / "docs" / "research" / f"sweep_{report_date}_combinations.md"
    output_root.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("  Sweep combinations multi-stratégies FX")
    print("=" * 72)
    print(f"Report date : {report_date}")
    print(f"Output root : {output_root}")
    print(f"Markdown    : {md_path}")
    print()

    print("Loading sleeves (this will take 1-2 min)…")
    sleeves = get_strategy_daily_returns()
    print(f"  Loaded {len(sleeves)} sleeves: {sorted(sleeves)}")
    print()

    grid = build_grid()
    if args.smoke:
        # Pick 1 from each of A, D, F blocks for a quick sanity check.
        smoke_ids = {"A3", "D1", "F1"}
        grid = [c for c in grid if c.id in smoke_ids]
    print(f"Running {len(grid)} configurations via native parallel sweep…")
    t0 = time.perf_counter()
    pf_all, metrics = native_parallel_sweep(grid, sleeves)
    elapsed = time.perf_counter() - t0
    print(f"  Native sweep completed in {elapsed:.1f}s ({elapsed/len(grid):.2f}s per combo)")

    rows = rows_from_native_metrics(grid, metrics, pf_all)

    # Print per-config summary line.
    for i, row in enumerate(rows, 1):
        mark = _target_hit_mark(row["annual_return"], row["max_drawdown"])
        print(
            f"  [{i:>2}/{len(grid)}] {row['id']} ({row['block']}) "
            f"sharpe={row['wf_avg_sharpe']:+.3f} "
            f"CAGR={row['annual_return']*100:>+6.2f}% "
            f"vol={row['annual_vol']*100:>5.2f}% "
            f"DD={row['max_drawdown']*100:>+6.2f}% "
            f"{mark}"
        )

    if not rows:
        print("No configs succeeded, aborting.")
        return

    sorted_rows = sorted(rows, key=lambda r: r["wf_avg_sharpe"], reverse=True)

    # ── Bootstrap top-5 ───────────────────────────────────────────
    bootstrap_by_id: dict[str, dict[str, float]] = {}
    if not args.smoke and not args.no_bootstrap:
        print(f"\nBootstrapping top-5 ({args.bootstrap_runs} runs × 20-day blocks each, ~15 min/combo)…")
        for rank, row in enumerate(sorted_rows[:5], 1):
            cfg = next(c for c in grid if c.id == row["id"])
            t0 = time.perf_counter()
            try:
                stats = bootstrap_config(cfg, sleeves, n_runs=args.bootstrap_runs)
                bootstrap_by_id[row["id"]] = stats
                elapsed = time.perf_counter() - t0
                print(
                    f"  [top{rank}] {row['id']}: "
                    f"CAGR P5={stats['cagr_p05']*100:+.2f}% "
                    f"MaxDD P5={stats['max_dd_p05']*100:+.2f}% "
                    f"target_hit={stats['target_hit_fraction']*100:.1f}% "
                    f"({elapsed:.0f}s, {stats['n_runs']}/{args.bootstrap_runs} runs)"
                )
            except Exception as exc:
                print(f"  [top{rank}] {row['id']} FAILED: {exc}")

    # ── Export JSON ───────────────────────────────────────────────
    json_path = output_root / "metrics.json"
    json_data = {
        "report_date": report_date,
        "n_configs": len(rows),
        "sleeves_loaded": sorted(sleeves),
        "configs": [sanitize_result_for_json(r) for r in sorted_rows],
        "bootstrap_top5": bootstrap_by_id,
    }
    json_path.write_text(json.dumps(json_data, indent=2, default=str))
    print(f"\nJSON exported → {json_path}")

    # ── Export markdown ───────────────────────────────────────────
    md = build_markdown_report(rows, bootstrap_by_id, report_date)
    md_path.write_text(md)
    print(f"Markdown exported → {md_path}")

    # ── Top-N artifacts ──────────────────────────────────────────
    if not args.smoke:
        print("\nGenerating top-N artifacts (tearsheets + mix plots)…")
        generate_top_n_artifacts(sorted_rows, grid, sleeves, output_root)

    # ── Final console summary ────────────────────────────────────
    print("\n" + "=" * 72)
    print("  Top 5 by Walk-Forward Sharpe")
    print("=" * 72)
    for i, r in enumerate(sorted_rows[:5], 1):
        print(
            f"  {i}. {r['id']:<4} {r['name'][:48]:<50} "
            f"sharpe={r['wf_avg_sharpe']:+.3f}  "
            f"CAGR={r['annual_return']*100:+.2f}%  "
            f"MaxDD={r['max_drawdown']*100:+.2f}%"
        )
    p18 = next((r for r in rows if r["id"] == "F1"), None)
    if p18 is not None:
        p18_rank = next(i for i, r in enumerate(sorted_rows, 1) if r["id"] == "F1")
        print(
            f"\n  Phase 18 baseline (F1) ranks #{p18_rank} of {len(sorted_rows)}  "
            f"(sharpe={p18['wf_avg_sharpe']:+.3f})"
        )
    print()


if __name__ == "__main__":
    main()
