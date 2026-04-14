"""High-level robustness reporting orchestrator.

Brings together the bootstrap module, the trade Monte Carlo, and the
statistical testing helpers (PSR/DSR/PBO/Haircut/SPA/StepM) into a
single call site.

Typical usage inside a strategy runner
--------------------------------------
>>> from framework.robustness import robustness_report, print_robustness_report
>>> report = robustness_report(
...     pf,
...     grid_sharpes=grid_series,     # optional — unlocks DSR/PBO/Haircut
...     cv_returns_matrix=cv_mat,     # optional — unlocks SPA/StepM
...     n_boot=2000,
...     n_mc=1000,
... )
>>> print_robustness_report(report, name="MR Macro (EURUSD)")

Every sub-result is placed under a named key so callers can pick
only the pieces they need (e.g. only bootstrap CIs for a quick
analysis, or the full report for a final sign-off).

The orchestrator is deliberately forgiving : if any optional input
is missing, the corresponding section is marked ``None`` in the
returned dict instead of raising. This lets the reporting flow
degrade gracefully when, for example, only a single run is available
(no grid sweep, no CV paths).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import vectorbtpro as vbt

from framework.bootstrap import (
    bootstrap_all_metrics,
    bootstrap_equity_paths,
)
from framework.mc_trades import (
    mc_max_drawdown_distribution,
    mc_sequence_risk_report,
)
from framework.pipeline_utils import (
    FX_MINUTE_ANN_FACTOR,
    METRIC_LABELS,
    METRIC_NAMES,
    SHARPE_RATIO,
    resolve_ann_factor,
)
from framework.statistical_testing import (
    deflated_sharpe_ratio,
    haircut_sharpe_ratio,
    minimum_backtest_length,
    probability_of_backtest_overfitting,
    probabilistic_sharpe_ratio,
    reality_check_via_arch,
    stepm_romano_wolf,
)


ROBUSTNESS_DEFAULTS: dict[str, Any] = dict(
    n_boot=2000,
    n_mc=1000,
    n_equity_paths=200,
    block_len_mean=50.0,
    seed=42,
    haircut_correction="BHY",
)


# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════


def _get_returns(pf: vbt.Portfolio) -> pd.Series:
    """Return a 1D Series of per-bar strategy returns from a Portfolio."""
    r = pf.returns
    if isinstance(r, pd.DataFrame):
        if r.shape[1] == 1:
            r = r.iloc[:, 0]
        else:
            r = r.iloc[:, 0]
    return r.dropna()


def _sanitize_grid_sharpes(grid_sharpes: Any) -> np.ndarray | None:
    if grid_sharpes is None:
        return None
    if isinstance(grid_sharpes, pd.Series):
        arr = grid_sharpes.to_numpy(dtype=np.float64)
    elif isinstance(grid_sharpes, pd.DataFrame):
        arr = grid_sharpes.to_numpy(dtype=np.float64).ravel()
    else:
        arr = np.asarray(grid_sharpes, dtype=np.float64).ravel()
    arr = arr[np.isfinite(arr)]
    return arr if arr.size >= 2 else None


def _log_err(section: str, exc: Exception) -> None:
    """Uniform error log used by every section sub-helper."""
    print(f"  [robustness] {section} failed: {exc}")


# ═══════════════════════════════════════════════════════════════════════
# Section helpers — one per robustness report block
# ═══════════════════════════════════════════════════════════════════════


def _section_bootstrap(
    returns: pd.Series,
    *,
    n_boot: int,
    block_len_mean: float,
    seed: int,
    ann_factor: float,
) -> tuple[Any, Any]:
    """Run the bootstrap over all 14 metrics. Returns ``(df, samples)``."""
    try:
        return bootstrap_all_metrics(
            returns,
            n_boot=n_boot,
            block_len_mean=block_len_mean,
            seed=seed,
            ann_factor=ann_factor,
            return_samples=True,
        )
    except Exception as exc:  # pragma: no cover
        _log_err("bootstrap_all_metrics", exc)
        return None, None


def _section_equity_paths(
    returns: pd.Series,
    *,
    n_equity_paths: int,
    block_len_mean: float,
    seed: int,
) -> tuple[Any, Any]:
    """Generate the bootstrap equity fan chart material."""
    try:
        paths = bootstrap_equity_paths(
            returns,
            n_sim=n_equity_paths,
            block_len_mean=block_len_mean,
            seed=seed,
        )
        curve = (1.0 + returns).cumprod()
        return paths, curve
    except Exception as exc:  # pragma: no cover
        _log_err("bootstrap_equity_paths", exc)
        return None, None


def _section_psr(returns: pd.Series) -> dict[str, Any] | None:
    try:
        return {"psr": probabilistic_sharpe_ratio(returns, sr_benchmark=0.0)}
    except Exception as exc:
        _log_err("PSR", exc)
        return None


def _section_dsr_haircut(
    returns: pd.Series,
    sharpes_arr: np.ndarray,
    *,
    haircut_correction: str,
    ann_factor: float,
) -> tuple[Any, Any, Any]:
    """Return ``(dsr, haircut, min_backtest_length)`` triple.

    Each slot is ``None`` on failure ; the three are computed in
    dependency order so a missing DSR still lets Haircut/MinBTL run
    by falling back to a VBT-native Sharpe.
    """
    dsr = haircut = min_btl = None
    try:
        dsr = deflated_sharpe_ratio(
            returns,
            n_trials=int(sharpes_arr.size),
            trial_sharpes=sharpes_arr,
        )
    except Exception as exc:
        _log_err("DSR", exc)

    try:
        sr_obs = float(dsr["sharpe"]) if dsr else float("nan")
        if not np.isfinite(sr_obs):
            sr_obs = float(returns.vbt.returns().sharpe_ratio())
        haircut = haircut_sharpe_ratio(
            sr_obs,
            n_trials=int(sharpes_arr.size),
            sample_length=int(returns.size),
            correction=haircut_correction,
            ann_factor=ann_factor,
        )
    except Exception as exc:
        _log_err("Haircut Sharpe", exc)

    try:
        sr_obs = float(haircut["sharpe_obs"]) if haircut else float("nan")
        if np.isfinite(sr_obs) and sr_obs > 0:
            min_btl = {
                "years": minimum_backtest_length(sr_obs, int(sharpes_arr.size)),
                "sharpe_target": sr_obs,
                "n_trials": int(sharpes_arr.size),
            }
    except Exception as exc:
        _log_err("MinBTL", exc)

    return dsr, haircut, min_btl


def _pick_pbo_n_bins(n_rows: int, n_configs: int) -> int:
    """Choose an even bin count compatible with the sample size, ≤16."""
    safe_bins = (n_rows // max(1, n_configs // 4 + 1)) // 2 * 2
    n_bins = max(4, min(16, safe_bins))
    return n_bins if n_bins % 2 == 0 else n_bins - 1


def _section_pbo(grid_returns_matrix: pd.DataFrame) -> dict[str, Any] | None:
    try:
        n_configs = grid_returns_matrix.shape[1]
        n_rows = grid_returns_matrix.dropna(how="any").shape[0]
        n_bins = _pick_pbo_n_bins(n_rows, n_configs)
        return probability_of_backtest_overfitting(
            grid_returns_matrix,
            n_bins=n_bins,
            objective="sharpe",
        )
    except Exception as exc:
        _log_err("PBO", exc)
        return None


def _section_mc_trades(
    pf: vbt.Portfolio, *, n_mc: int, seed: int
) -> tuple[Any, Any]:
    mc_trades = sequence_risk = None
    try:
        mc_trades = mc_max_drawdown_distribution(
            pf, n_sim=n_mc, mode="shuffle", seed=seed
        )
    except Exception as exc:
        _log_err("MC trades", exc)
    try:
        sequence_risk = mc_sequence_risk_report(pf, n_sim=n_mc, seed=seed)
    except Exception as exc:
        _log_err("sequence risk", exc)
    return mc_trades, sequence_risk


def _section_arch_tests(
    grid_returns_matrix: pd.DataFrame,
    benchmark_returns: pd.Series,
    *,
    n_boot: int,
    block_len_mean: float,
    seed: int,
) -> tuple[Any, Any]:
    spa = stepm = None
    effective_boot = min(n_boot, 1000)
    try:
        spa = reality_check_via_arch(
            grid_returns_matrix,
            benchmark_returns,
            n_boot=effective_boot,
            block_size=int(block_len_mean),
            seed=seed,
        )
    except Exception as exc:
        _log_err("SPA", exc)
    try:
        stepm = stepm_romano_wolf(
            grid_returns_matrix,
            benchmark_returns,
            n_boot=effective_boot,
            block_size=int(block_len_mean),
            seed=seed,
        )
    except Exception as exc:
        _log_err("StepM", exc)
    return spa, stepm


# ═══════════════════════════════════════════════════════════════════════
# Orchestrator
# ═══════════════════════════════════════════════════════════════════════


def robustness_report(
    pf: vbt.Portfolio,
    *,
    grid_sharpes: pd.Series | np.ndarray | None = None,
    grid_returns_matrix: pd.DataFrame | None = None,
    benchmark_returns: pd.Series | None = None,
    n_boot: int = ROBUSTNESS_DEFAULTS["n_boot"],
    n_mc: int = ROBUSTNESS_DEFAULTS["n_mc"],
    n_equity_paths: int = ROBUSTNESS_DEFAULTS["n_equity_paths"],
    block_len_mean: float = ROBUSTNESS_DEFAULTS["block_len_mean"],
    seed: int = ROBUSTNESS_DEFAULTS["seed"],
    haircut_correction: str = ROBUSTNESS_DEFAULTS["haircut_correction"],
    ann_factor: float | None = None,
    include_equity_paths: bool = True,
    include_mc_trades: bool = True,
) -> dict[str, Any]:
    """Compute the full robustness report for a strategy run.

    Parameters
    ----------
    pf
        Full-resolution VBT portfolio.
    grid_sharpes
        Optional vector of Sharpes from a parameter sweep. Unlocks
        DSR, PBO (when ``grid_returns_matrix`` is also given), and
        Haircut Sharpe.
    grid_returns_matrix
        Optional ``(T, n_configs)`` DataFrame of sweep per-bar returns.
        Unlocks PBO via CSCV and SPA/StepM via ``arch``.
    benchmark_returns
        Optional benchmark return series for SPA/StepM.
    n_boot
        Number of bootstrap replicates for metric CIs.
    n_mc
        Number of Monte Carlo trade shuffles for the MDD distribution.
    n_equity_paths
        Number of bootstrap equity paths for the fan chart.
    block_len_mean
        Mean block length for the stationary bootstrap.
    seed
        Integer seed propagated to every stochastic sub-routine.
    haircut_correction
        ``"Bonferroni"``, ``"Holm"``, or ``"BHY"``. Default ``"BHY"``.
    ann_factor
        Annualization factor. If None, resolved from the portfolio
        index via :func:`framework.pipeline_utils.resolve_ann_factor`.
    include_equity_paths
        Skip the equity fan chart generation (saves memory on long
        minute-frequency histories).
    include_mc_trades
        Skip the trade-level MC (useful when the strategy has few
        trades and the distribution would be uninformative).

    Returns
    -------
    dict
        Named sections : ``bootstrap_df``, ``equity_paths``,
        ``equity_curve``, ``psr``, ``dsr``, ``pbo``, ``haircut``,
        ``min_backtest_length``, ``mc_trades``, ``sequence_risk``,
        ``spa``, ``stepm``, ``config``, ``metric_labels``.
    """
    returns = _get_returns(pf)
    if returns.empty:
        raise ValueError("portfolio has no returns")

    if ann_factor is None:
        try:
            ann_factor = resolve_ann_factor(returns.index)
        except Exception:
            ann_factor = FX_MINUTE_ANN_FACTOR

    report: dict[str, Any] = {
        "config": dict(
            n_boot=int(n_boot),
            n_mc=int(n_mc),
            n_equity_paths=int(n_equity_paths),
            block_len_mean=float(block_len_mean),
            seed=int(seed),
            haircut_correction=haircut_correction,
            ann_factor=float(ann_factor),
        ),
        "metric_labels": dict(METRIC_LABELS),
    }

    # 1. Bootstrap CI on every metric + 2. equity fan chart material.
    bdf, bsamples = _section_bootstrap(
        returns,
        n_boot=n_boot,
        block_len_mean=block_len_mean,
        seed=seed,
        ann_factor=ann_factor,
    )
    report["bootstrap_df"] = bdf
    report["bootstrap_samples"] = bsamples

    if include_equity_paths:
        paths, curve = _section_equity_paths(
            returns,
            n_equity_paths=n_equity_paths,
            block_len_mean=block_len_mean,
            seed=seed,
        )
    else:
        paths, curve = None, None
    report["equity_paths"] = paths
    report["equity_curve"] = curve

    # 3. PSR vs flat-zero benchmark.
    report["psr"] = _section_psr(returns)

    # 4. DSR + Haircut + MinBTL (require sweep sharpes).
    sharpes_arr = _sanitize_grid_sharpes(grid_sharpes)
    if sharpes_arr is not None:
        dsr, haircut, minbtl = _section_dsr_haircut(
            returns,
            sharpes_arr,
            haircut_correction=haircut_correction,
            ann_factor=ann_factor,
        )
    else:
        dsr = haircut = minbtl = None
    report["dsr"] = dsr
    report["haircut"] = haircut
    report["min_backtest_length"] = minbtl

    # 5. PBO (CSCV) — needs the full per-config return matrix.
    have_grid_matrix = (
        isinstance(grid_returns_matrix, pd.DataFrame)
        and grid_returns_matrix.shape[1] >= 2
    )
    report["pbo"] = _section_pbo(grid_returns_matrix) if have_grid_matrix else None

    # 6. Monte Carlo trade shuffle.
    if include_mc_trades:
        mc_trades, seq_risk = _section_mc_trades(pf, n_mc=n_mc, seed=seed)
    else:
        mc_trades, seq_risk = None, None
    report["mc_trades"] = mc_trades
    report["sequence_risk"] = seq_risk

    # 7. Hansen SPA / Romano-Wolf StepM via the arch package.
    if have_grid_matrix and benchmark_returns is not None:
        spa, stepm = _section_arch_tests(
            grid_returns_matrix,
            benchmark_returns,
            n_boot=n_boot,
            block_len_mean=block_len_mean,
            seed=seed,
        )
    else:
        spa, stepm = None, None
    report["spa"] = spa
    report["stepm"] = stepm

    return report


# ═══════════════════════════════════════════════════════════════════════
# Printing
# ═══════════════════════════════════════════════════════════════════════


def _fmt(val: Any, spec: str = ".4f") -> str:
    if val is None:
        return "—"
    try:
        return format(float(val), spec)
    except (TypeError, ValueError):
        return str(val)


def print_robustness_report(report: dict[str, Any], name: str = "Strategy") -> None:
    """Pretty-print the robustness report to stdout."""
    bar = "═" * 78
    print(f"\n{bar}")
    print(f"  ROBUSTNESS REPORT — {name}")
    print(bar)

    cfg = report.get("config", {})
    print(
        f"  config: n_boot={cfg.get('n_boot')} · n_mc={cfg.get('n_mc')} · "
        f"block={cfg.get('block_len_mean')} · seed={cfg.get('seed')}"
    )

    # Bootstrap CIs
    df = report.get("bootstrap_df")
    if df is not None and not df.empty:
        print("\n  Bootstrap 95% CIs (stationary block bootstrap)")
        print("  " + "─" * 74)
        header = f"  {'metric':<22} {'observed':>12} {'mean':>12} {'ci_low':>12} {'ci_high':>12}"
        print(header)
        for metric, row in df.iterrows():
            print(
                f"  {row.get('label', metric):<22} "
                f"{_fmt(row['observed']):>12} "
                f"{_fmt(row['mean']):>12} "
                f"{_fmt(row['ci_low']):>12} "
                f"{_fmt(row['ci_high']):>12}"
            )

    # PSR / DSR / Haircut
    psr = report.get("psr")
    dsr = report.get("dsr")
    haircut = report.get("haircut")
    minbtl = report.get("min_backtest_length")
    print("\n  Overfitting checks")
    print("  " + "─" * 74)
    if psr:
        print(f"  Probabilistic Sharpe Ratio (vs 0) : {_fmt(psr.get('psr'), '.4f')}")
    if dsr:
        print(
            f"  Deflated Sharpe Ratio             : {_fmt(dsr.get('dsr'), '.4f')} "
            f"(sharpe={_fmt(dsr.get('sharpe'), '.3f')}, "
            f"E[max|N={dsr.get('n_trials')}]={_fmt(dsr.get('expected_max_sharpe'), '.3f')})"
        )
    if haircut:
        print(
            f"  Haircut Sharpe [{haircut.get('correction', 'BHY')}]          : "
            f"{_fmt(haircut.get('haircut_sharpe'), '.3f')} "
            f"(ratio={_fmt(haircut.get('haircut_ratio'), '.2%')}, "
            f"adj p={_fmt(haircut.get('adjusted_pvalue'), '.4f')})"
        )
    if minbtl:
        print(
            f"  Min. backtest length (years)      : {_fmt(minbtl.get('years'), '.2f')} "
            f"for SR target {_fmt(minbtl.get('sharpe_target'), '.2f')}"
        )

    # PBO
    pbo = report.get("pbo")
    if pbo:
        verdict = "HEALTHY" if pbo.get("pbo", 1.0) < 0.5 else "OVERFIT"
        print(
            f"  PBO (CSCV, n_bins={pbo.get('n_bins')})         : "
            f"{_fmt(pbo.get('pbo'), '.3f')}  [{verdict}]"
        )

    # MC trades
    mc = report.get("mc_trades")
    seq = report.get("sequence_risk")
    if mc:
        print("\n  Monte Carlo trade-shuffle (sequence risk)")
        print("  " + "─" * 74)
        print(
            f"  n_trades={mc.get('n_trades')} · n_sim={mc.get('n_sim')} · "
            f"mode={mc.get('mode')}"
        )
        print(
            f"  observed MDD                      : {_fmt(mc.get('observed_mdd'), '.3%')}  "
            f"(underwater={mc.get('observed_underwater')} trades)"
        )
        print(
            f"  MDD distribution                  : "
            f"p50={_fmt(mc.get('mdd_p50'), '.3%')} · "
            f"p95={_fmt(mc.get('mdd_p95'), '.3%')} · "
            f"p99={_fmt(mc.get('mdd_p99'), '.3%')}"
        )
        if seq:
            print(
                f"  sequence luck (z-score)           : {_fmt(seq.get('sequence_luck_zscore'), '+.2f')}  "
                f"({_fmt(seq.get('pct_shuffles_worse_than_observed'), '.1%')} of shuffles worse)"
            )

    # SPA / StepM
    spa = report.get("spa")
    stepm = report.get("stepm")
    if spa or stepm:
        print("\n  Multiple testing (arch.bootstrap)")
        print("  " + "─" * 74)
    if spa:
        print(
            f"  Hansen SPA p-values               : "
            f"lower={_fmt(spa.get('pvalue_lower'), '.4f')} · "
            f"consistent={_fmt(spa.get('pvalue_consistent'), '.4f')} · "
            f"upper={_fmt(spa.get('pvalue_upper'), '.4f')}"
        )
    if stepm:
        print(
            f"  Romano-Wolf StepM (α={stepm.get('alpha')})       : "
            f"{stepm.get('n_significant')}/{stepm.get('n_strategies')} strategies dominate benchmark"
        )

    print(bar + "\n")


# ═══════════════════════════════════════════════════════════════════════
# Figure builder — used by analyze_portfolio
# ═══════════════════════════════════════════════════════════════════════


def build_robustness_figures(
    report: dict[str, Any],
    *,
    name: str = "Strategy",
    returns: pd.Series | None = None,
) -> dict[str, Any]:
    """Produce a dict of Plotly figures from a ``robustness_report`` result.

    Safe to call with a partial report : missing sections are skipped.
    Used by :func:`framework.pipeline_utils.analyze_portfolio` when
    ``robustness=True``.
    """
    from framework.plotting._robustness import (
        plot_bootstrap_distribution,
        plot_cpcv_distribution,
        plot_equity_fan_chart,
        plot_mdd_distribution,
        plot_metric_ci_forest,
        plot_pbo_logits,
        plot_rolling_metric_stability,
        plot_spa_pvalues,
    )

    figs: dict[str, Any] = {}

    df = report.get("bootstrap_df")
    samples_mat = report.get("bootstrap_samples")
    if df is not None and samples_mat is not None:
        try:
            figs["metric_ci_forest"] = plot_metric_ci_forest(
                df,
                title=f"{name} — Bootstrap 95% CI Forest",
                include_metrics=[
                    "sharpe_ratio",
                    "sortino_ratio",
                    "calmar_ratio",
                    "omega_ratio",
                    "profit_factor",
                    "tail_ratio",
                ],
            )
        except Exception as exc:
            print(f"  [robustness] forest plot failed: {exc}")

        try:
            # Sharpe distribution as the headline bootstrap chart.
            sharpe_samples = samples_mat[:, SHARPE_RATIO]
            sharpe_row = df.loc[METRIC_NAMES[SHARPE_RATIO]]
            figs["bootstrap_sharpe"] = plot_bootstrap_distribution(
                sharpe_samples,
                observed=float(sharpe_row["observed"]),
                metric_label="Sharpe Ratio",
                ci_low=float(sharpe_row["ci_low"]),
                ci_high=float(sharpe_row["ci_high"]),
                title=f"{name} — Bootstrap Sharpe Distribution",
            )
        except Exception as exc:
            print(f"  [robustness] bootstrap sharpe plot failed: {exc}")

    eq_paths = report.get("equity_paths")
    eq_curve = report.get("equity_curve")
    if eq_paths is not None and not eq_paths.empty:
        try:
            figs["equity_fan_chart"] = plot_equity_fan_chart(
                eq_paths,
                observed_equity=eq_curve,
                title=f"{name} — Bootstrap Equity Fan Chart",
            )
        except Exception as exc:
            print(f"  [robustness] fan chart failed: {exc}")

    mc = report.get("mc_trades")
    if mc:
        try:
            figs["mdd_distribution"] = plot_mdd_distribution(
                mc["mdd_samples"],
                observed_mdd=float(mc["observed_mdd"]),
                title=f"{name} — MC MaxDD Distribution",
            )
        except Exception as exc:
            print(f"  [robustness] mdd plot failed: {exc}")

    pbo = report.get("pbo")
    if pbo and "logits" in pbo:
        try:
            figs["pbo_logits"] = plot_pbo_logits(
                pbo["logits"],
                pbo=float(pbo.get("pbo", float("nan"))),
                title=f"{name} — PBO (CSCV) Logits",
            )
        except Exception as exc:
            print(f"  [robustness] pbo plot failed: {exc}")

    spa = report.get("spa")
    if spa:
        try:
            figs["spa_pvalues"] = plot_spa_pvalues(
                spa, title=f"{name} — Hansen SPA p-values"
            )
        except Exception as exc:
            print(f"  [robustness] spa plot failed: {exc}")

    if returns is not None and not returns.empty:
        try:
            window = "365D" if isinstance(returns.index, pd.DatetimeIndex) else 500
            figs["rolling_stability"] = plot_rolling_metric_stability(
                returns,
                window=window,
                ann_factor=report.get("config", {}).get("ann_factor", 252.0),
                title=f"{name} — Rolling Metric Stability",
            )
        except Exception as exc:
            print(f"  [robustness] rolling stability failed: {exc}")

    return figs


__all__ = [
    "ROBUSTNESS_DEFAULTS",
    "robustness_report",
    "print_robustness_report",
    "build_robustness_figures",
]
