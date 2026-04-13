"""Phase 5 — Stress tests for combined_portfolio_v2 recommended config.

Runs four diagnostic suites on the recommended v2 configuration
(``allocation='custom'`` with MR80/XS10/TS10 weights, ``target_vol=0.22``,
``max_leverage=15``, ``dd_cap_enabled=False``) and prints a consolidated
report. Intended to be run manually before any paper-trading or
production deployment.

Suites
------
1. **Block-bootstrap** — resample the 2019-2025 component daily returns
   in 20-day blocks, rebuild the portfolio 1000 times, and tabulate
   the distribution of CAGR, Max DD, and Sharpe. Provides an empirical
   tail-risk estimate that the single-path backtest cannot.
2. **Scenario replay** — report the v2 performance metrics restricted
   to well-known stress periods (2019 full year, 2020 Q1 Covid,
   2022 Q3 GBP crisis, 2023 rate hikes). Small-sample so the Sharpe
   is noisy but the Max DD is meaningful.
3. **In-sample / out-of-sample split** — split at 2025-04-01. Report
   metrics on each side separately to detect over-fitting.
4. **Parameter sensitivity** — sweep target_vol and max_leverage
   around the recommended config and report monotonicity of CAGR and
   Max DD. A non-monotonic response would be a red flag that the
   recommended point is a local cliff rather than a stable plateau.

The script is side-effect free — it only prints to stdout and writes
an optional JSON artifact for downstream analysis.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SRC = _PROJECT_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


# ═══════════════════════════════════════════════════════════════════════
# Recommended configuration (from Phase 16)
# ═══════════════════════════════════════════════════════════════════════


# Phase 16 config — MR Macro + TS Momentum only, XS Momentum dropped
# because it was the biggest contributor to the 2019 loss (-3.33% at
# 11% vol). This config cuts Max DD by ~6pp and lifts Sharpe by ~17%
# vs the Phase 15 MR80/XS10/TS10 mix at comparable CAGR.
RECOMMENDED_CONFIG: dict[str, Any] = {
    "allocation": "custom",
    "custom_weights": {
        "MR_Macro": 0.90,
        "TS_Momentum_RSI": 0.10,
    },
    "target_vol": 0.28,
    "max_leverage": 12.0,
    "dd_cap_enabled": False,
}


OOS_SPLIT_DATE = "2025-04-01"


# ═══════════════════════════════════════════════════════════════════════
# Result dataclasses for structured output
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class BootstrapStats:
    n_runs: int
    cagr_mean: float
    cagr_p05: float
    cagr_p50: float
    cagr_p95: float
    max_dd_mean: float
    max_dd_p05: float
    max_dd_p50: float
    max_dd_p95: float
    sharpe_mean: float
    sharpe_p05: float
    sharpe_p95: float
    pos_fraction: float  # fraction of runs with CAGR > 0
    target_hit_fraction: float  # CAGR in [0.10, 0.15] AND MaxDD > -0.35

    def to_dict(self) -> dict[str, float]:
        return {k: float(v) for k, v in self.__dict__.items()}


# ═══════════════════════════════════════════════════════════════════════
# Block-bootstrap
# ═══════════════════════════════════════════════════════════════════════


def block_bootstrap_indices(
    n: int,
    block_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample ``n`` indices as concatenated blocks of length ``block_size``.

    Uses moving-block bootstrap (Künsch 1989): pick random start
    positions uniformly in ``[0, n - block_size]`` and concatenate
    until the resampled index length reaches ``n``. The last block is
    truncated if it would overflow.
    """
    if block_size <= 0 or block_size > n:
        raise ValueError(f"block_size must be in (0, {n}], got {block_size}")

    n_blocks = n // block_size + (1 if n % block_size else 0)
    starts = rng.integers(0, n - block_size + 1, size=n_blocks)
    indices = np.concatenate(
        [np.arange(s, s + block_size) for s in starts]
    )[:n]
    return indices


def run_block_bootstrap(
    strategy_returns: dict[str, pd.Series],
    n_runs: int = 1000,
    block_size: int = 20,
    seed: int = 20260413,
) -> BootstrapStats:
    """Resample the component returns and rebuild v2 ``n_runs`` times."""
    from strategies.combined_portfolio_v2 import build_combined_portfolio_v2

    # Align to a common index first so every resample preserves the
    # joint structure between strategies.
    df = pd.DataFrame(strategy_returns).dropna()
    n_bars = len(df)
    rng = np.random.default_rng(seed)

    cagrs: list[float] = []
    max_dds: list[float] = []
    sharpes: list[float] = []

    for i in range(n_runs):
        idx = block_bootstrap_indices(n_bars, block_size, rng)
        resampled = df.iloc[idx].reset_index(drop=True)
        # Re-attach a synthetic business-day index so downstream vol
        # targeting windows see the right calendar spacing.
        resampled.index = pd.bdate_range("2000-01-03", periods=n_bars)
        strat_resampled = {col: resampled[col] for col in df.columns}

        try:
            res = build_combined_portfolio_v2(strat_resampled, **RECOMMENDED_CONFIG)
        except Exception as exc:
            # Some resamples may have degenerate vol → skip rather
            # than crash the whole sweep.
            print(f"  [run {i}] skipped: {exc}")
            continue

        cagrs.append(res["annual_return"])
        max_dds.append(res["max_drawdown"])
        sr = res["sharpe"]
        sharpes.append(0.0 if np.isnan(sr) else sr)

    cagr_arr = np.asarray(cagrs)
    dd_arr = np.asarray(max_dds)
    sharpe_arr = np.asarray(sharpes)

    target_hit = (cagr_arr >= 0.10) & (cagr_arr <= 0.15) & (dd_arr > -0.35)

    return BootstrapStats(
        n_runs=len(cagrs),
        cagr_mean=float(cagr_arr.mean()),
        cagr_p05=float(np.percentile(cagr_arr, 5)),
        cagr_p50=float(np.percentile(cagr_arr, 50)),
        cagr_p95=float(np.percentile(cagr_arr, 95)),
        max_dd_mean=float(dd_arr.mean()),
        max_dd_p05=float(np.percentile(dd_arr, 5)),
        max_dd_p50=float(np.percentile(dd_arr, 50)),
        max_dd_p95=float(np.percentile(dd_arr, 95)),
        sharpe_mean=float(sharpe_arr.mean()),
        sharpe_p05=float(np.percentile(sharpe_arr, 5)),
        sharpe_p95=float(np.percentile(sharpe_arr, 95)),
        pos_fraction=float((cagr_arr > 0).mean()),
        target_hit_fraction=float(target_hit.mean()),
    )


# ═══════════════════════════════════════════════════════════════════════
# Scenario replay
# ═══════════════════════════════════════════════════════════════════════


SCENARIOS: list[tuple[str, str, str]] = [
    ("2019 full year",        "2019-01-01", "2019-12-31"),
    ("2020 Q1 Covid",         "2020-02-20", "2020-04-30"),
    ("2020 full year",        "2020-01-01", "2020-12-31"),
    ("2022 Q3 GBP crisis",    "2022-08-01", "2022-11-30"),
    ("2023 rate hikes year",  "2023-01-01", "2023-12-31"),
    ("2024 full year",        "2024-01-01", "2024-12-31"),
]


def run_scenario_replay(
    strategy_returns: dict[str, pd.Series],
) -> list[dict[str, Any]]:
    """Restrict the component returns to each scenario and rebuild v2."""
    from strategies.combined_portfolio_v2 import build_combined_portfolio_v2

    results: list[dict[str, Any]] = []
    for name, start, end in SCENARIOS:
        sliced = {
            k: v.loc[start:end] for k, v in strategy_returns.items()
        }
        min_len = min(len(s) for s in sliced.values())
        if min_len < 10:
            results.append({
                "scenario": name, "start": start, "end": end,
                "n_bars": min_len, "skipped": True,
            })
            continue

        res = build_combined_portfolio_v2(sliced, **RECOMMENDED_CONFIG)
        results.append({
            "scenario": name, "start": start, "end": end,
            "n_bars": min_len, "skipped": False,
            "cagr": float(res["annual_return"]),
            "vol": float(res["annual_vol"]),
            "max_dd": float(res["max_drawdown"]),
            "sharpe": float(res["sharpe"]),
            "total_return": float((1 + res["portfolio_returns"]).prod() - 1),
        })
    return results


# ═══════════════════════════════════════════════════════════════════════
# IS / OOS split
# ═══════════════════════════════════════════════════════════════════════


def run_is_oos_split(
    strategy_returns: dict[str, pd.Series],
    split_date: str = OOS_SPLIT_DATE,
) -> dict[str, dict[str, float]]:
    """Split the component returns at ``split_date`` and rebuild each side."""
    from strategies.combined_portfolio_v2 import build_combined_portfolio_v2

    split_ts = pd.Timestamp(split_date)
    in_sample = {
        k: v.loc[: split_ts - pd.Timedelta(days=1)]
        for k, v in strategy_returns.items()
    }
    out_sample = {k: v.loc[split_ts:] for k, v in strategy_returns.items()}

    def summary(rets_dict: dict[str, pd.Series]) -> dict[str, float]:
        min_len = min(len(s) for s in rets_dict.values())
        if min_len < 20:
            return {"n_bars": min_len, "skipped": True}
        res = build_combined_portfolio_v2(rets_dict, **RECOMMENDED_CONFIG)
        return {
            "n_bars": min_len,
            "skipped": False,
            "cagr": float(res["annual_return"]),
            "vol": float(res["annual_vol"]),
            "max_dd": float(res["max_drawdown"]),
            "sharpe": float(res["sharpe"]),
            "total_return": float((1 + res["portfolio_returns"]).prod() - 1),
        }

    return {
        "in_sample": summary(in_sample),
        "out_of_sample": summary(out_sample),
        "split_date": split_date,
    }


# ═══════════════════════════════════════════════════════════════════════
# Parameter sensitivity
# ═══════════════════════════════════════════════════════════════════════


def run_parameter_sensitivity(
    strategy_returns: dict[str, pd.Series],
) -> list[dict[str, Any]]:
    """Sweep target_vol × max_leverage around the recommended config."""
    from strategies.combined_portfolio_v2 import build_combined_portfolio_v2

    sweeps: list[dict[str, Any]] = []
    for target_vol in (0.15, 0.18, 0.20, 0.22, 0.25, 0.28):
        for max_lev in (10.0, 15.0, 20.0):
            config = {**RECOMMENDED_CONFIG, "target_vol": target_vol, "max_leverage": max_lev}
            res = build_combined_portfolio_v2(strategy_returns, **config)
            sweeps.append({
                "target_vol": target_vol,
                "max_leverage": max_lev,
                "cagr": float(res["annual_return"]),
                "vol": float(res["annual_vol"]),
                "max_dd": float(res["max_drawdown"]),
                "sharpe": float(res["sharpe"]),
            })
    return sweeps


# ═══════════════════════════════════════════════════════════════════════
# Formatting helpers
# ═══════════════════════════════════════════════════════════════════════


def _pct(x: float, width: int = 7) -> str:
    return f"{x * 100:>{width}.2f}%"


def _print_header(title: str) -> None:
    bar = "═" * 78
    print(f"\n{bar}\n  {title}\n{bar}")


def print_bootstrap_stats(stats: BootstrapStats) -> None:
    _print_header("Block-Bootstrap Stress Test (1000 runs, 20-day blocks)")
    print(f"  Successful runs: {stats.n_runs}")
    print(f"\n  CAGR    : mean={_pct(stats.cagr_mean)}  p05={_pct(stats.cagr_p05)}  p50={_pct(stats.cagr_p50)}  p95={_pct(stats.cagr_p95)}")
    print(f"  Max DD  : mean={_pct(stats.max_dd_mean)}  p05={_pct(stats.max_dd_p05)}  p50={_pct(stats.max_dd_p50)}  p95={_pct(stats.max_dd_p95)}")
    print(f"  Sharpe  : mean={stats.sharpe_mean:>6.3f}  p05={stats.sharpe_p05:>6.3f}  p95={stats.sharpe_p95:>6.3f}")
    print(f"\n  Positive CAGR fraction   : {stats.pos_fraction:.1%}")
    print(f"  Target hit (10-15%, <35%): {stats.target_hit_fraction:.1%}")


def print_scenarios(scenarios: list[dict[str, Any]]) -> None:
    _print_header("Scenario Replay (recommended config)")
    print(f"  {'Scenario':<25} {'Start':<12} {'End':<12} {'N':>6} {'TotalRet':>10} {'CAGR':>10} {'MaxDD':>10} {'Sharpe':>8}")
    print("  " + "-" * 93)
    for s in scenarios:
        if s.get("skipped"):
            print(f"  {s['scenario']:<25} {s['start']:<12} {s['end']:<12} {s['n_bars']:>6} skipped (too short)")
            continue
        print(
            f"  {s['scenario']:<25} {s['start']:<12} {s['end']:<12} {s['n_bars']:>6} "
            f"{_pct(s['total_return'], 9)} {_pct(s['cagr'], 9)} {_pct(s['max_dd'], 9)} {s['sharpe']:>7.3f}"
        )


def print_is_oos(split: dict[str, Any]) -> None:
    _print_header(f"In-Sample / Out-of-Sample Split at {split['split_date']}")
    for label in ("in_sample", "out_of_sample"):
        s = split[label]
        print(f"\n  {label.replace('_', ' ').title()}:")
        if s.get("skipped"):
            print(f"    n_bars={s['n_bars']} — skipped")
            continue
        print(f"    n_bars={s['n_bars']}")
        print(f"    Total Return : {_pct(s['total_return'])}")
        print(f"    CAGR         : {_pct(s['cagr'])}")
        print(f"    Vol          : {_pct(s['vol'])}")
        print(f"    Max DD       : {_pct(s['max_dd'])}")
        print(f"    Sharpe       : {s['sharpe']:>6.3f}")


def print_sensitivity(sweeps: list[dict[str, Any]]) -> None:
    _print_header("Parameter Sensitivity — target_vol × max_leverage")
    print(f"  {'target_vol':>10} {'max_lev':>10} {'CAGR':>10} {'Vol':>10} {'MaxDD':>10} {'Sharpe':>10}")
    print("  " + "-" * 66)
    for s in sweeps:
        in_target = (0.10 <= s["cagr"] <= 0.15) and (s["max_dd"] > -0.35)
        mark = "★" if in_target else " "
        print(
            f"{mark} {s['target_vol']:>10.2f} {s['max_leverage']:>10.1f} "
            f"{_pct(s['cagr'], 9)} {_pct(s['vol'], 9)} "
            f"{_pct(s['max_dd'], 9)} {s['sharpe']:>9.3f}"
        )


# ═══════════════════════════════════════════════════════════════════════
# Orchestrator
# ═══════════════════════════════════════════════════════════════════════


def run_stress_tests(
    n_bootstrap: int = 1000,
    block_size: int = 20,
    save_json: str | None = None,
) -> dict[str, Any]:
    """Run all four suites and return a structured report."""
    from strategies.combined_portfolio import get_strategy_daily_returns
    from utils import apply_vbt_settings
    import vectorbtpro as vbt

    apply_vbt_settings()
    vbt.settings.returns.year_freq = pd.Timedelta(days=252)

    print("Loading strategy returns (2019-2025)...")
    strat_rets = get_strategy_daily_returns()

    # Filter to only the strategies actually used by the recommended
    # config. Passing XS Momentum returns when the custom_weights dict
    # omits it would still let the bootstrap dropna() shrink the common
    # index to the XS availability window, which is not what we want
    # for a 2-strategy config.
    active = set(RECOMMENDED_CONFIG["custom_weights"].keys())
    strat_rets = {k: v for k, v in strat_rets.items() if k in active}
    print(f"Active strategies: {sorted(strat_rets.keys())}")

    report: dict[str, Any] = {
        "config": {
            k: v for k, v in RECOMMENDED_CONFIG.items() if k != "custom_weights"
        },
        "custom_weights": RECOMMENDED_CONFIG["custom_weights"],
        "n_bootstrap": n_bootstrap,
        "block_size": block_size,
    }

    print(f"\nRunning block-bootstrap ({n_bootstrap} resamples, block={block_size})...")
    boot_stats = run_block_bootstrap(strat_rets, n_runs=n_bootstrap, block_size=block_size)
    print_bootstrap_stats(boot_stats)
    report["bootstrap"] = boot_stats.to_dict()

    print("\nRunning scenario replay...")
    scenarios = run_scenario_replay(strat_rets)
    print_scenarios(scenarios)
    report["scenarios"] = scenarios

    print("\nRunning in-sample / out-of-sample split...")
    split = run_is_oos_split(strat_rets)
    print_is_oos(split)
    report["is_oos"] = split

    print("\nRunning parameter sensitivity sweep...")
    sweeps = run_parameter_sensitivity(strat_rets)
    print_sensitivity(sweeps)
    report["sensitivity"] = sweeps

    if save_json:
        os.makedirs(os.path.dirname(save_json) or ".", exist_ok=True)
        with open(save_json, "w") as fh:
            json.dump(report, fh, indent=2, default=str)
        print(f"\nReport saved to {save_json}")

    return report


if __name__ == "__main__":
    run_stress_tests(
        n_bootstrap=1000,
        block_size=20,
        save_json="results/combined_v2/stress_test_report.json",
    )
