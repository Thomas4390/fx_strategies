"""Tests for framework.robustness end-to-end orchestrator.

Scope:
- :func:`robustness_report` on a toy VBT portfolio without grid
  matrix : verifies that bootstrap/PSR/MC sections are populated,
  overfitting sections are ``None``.
- With ``grid_sharpes`` provided : DSR, Haircut, MinBTL become
  populated.
- With ``grid_returns_matrix`` + benchmark : PBO, SPA, StepM populate.
- :func:`build_robustness_figures` returns a non-empty dict of
  Plotly figures when given a full report.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


import vectorbtpro as vbt  # noqa: E402

from framework.robustness import (  # noqa: E402
    build_robustness_figures,
    print_robustness_report,
    robustness_report,
)


# ═══════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════


@pytest.fixture(scope="module")
def toy_pf() -> vbt.Portfolio:
    rng = np.random.default_rng(11)
    n = 800
    idx = pd.date_range("2022-01-01", periods=n, freq="h")
    drift = 0.00005
    close = pd.Series(
        100.0 * np.cumprod(1.0 + rng.normal(drift, 0.004, n)),
        index=idx,
        name="close",
    )
    entries = pd.Series(False, index=idx)
    exits = pd.Series(False, index=idx)
    for i in range(0, n, 40):
        entries.iloc[i] = True
        if i + 20 < n:
            exits.iloc[i + 20] = True
    return vbt.Portfolio.from_signals(
        close=close, entries=entries, exits=exits, init_cash=10_000.0
    )


@pytest.fixture(scope="module")
def fake_grid_matrix(toy_pf) -> pd.DataFrame:
    """Create a small ``(T, n_configs)`` matrix of synthetic returns."""
    rng = np.random.default_rng(21)
    idx = toy_pf.wrapper.index
    cols = [f"cfg_{i}" for i in range(6)]
    data = rng.normal(loc=0.00003, scale=0.002, size=(len(idx), len(cols)))
    return pd.DataFrame(data, index=idx, columns=cols)


# ═══════════════════════════════════════════════════════════════════════
# Baseline : just the portfolio
# ═══════════════════════════════════════════════════════════════════════


def test_report_minimal_sections(toy_pf):
    report = robustness_report(
        toy_pf,
        n_boot=100,
        n_mc=100,
        n_equity_paths=20,
    )
    # Bootstrap is always present.
    assert report["bootstrap_df"] is not None
    assert report["bootstrap_samples"] is not None
    assert report["equity_paths"] is not None
    assert report["psr"] is not None
    # Sections that need a grid remain None.
    assert report["dsr"] is None
    assert report["haircut"] is None
    assert report["min_backtest_length"] is None
    assert report["pbo"] is None
    assert report["spa"] is None
    assert report["stepm"] is None
    # MC trades should populate because the portfolio has trades.
    assert report["mc_trades"] is not None
    assert 0.0 <= report["mc_trades"]["observed_mdd"] <= 1.0


# ═══════════════════════════════════════════════════════════════════════
# With grid_sharpes : DSR / Haircut / MinBTL
# ═══════════════════════════════════════════════════════════════════════


def test_report_with_grid_sharpes(toy_pf):
    grid = np.array([0.2, 0.3, 0.15, 0.25, 0.1], dtype=np.float64)
    report = robustness_report(
        toy_pf,
        grid_sharpes=grid,
        n_boot=100,
        n_mc=100,
        n_equity_paths=20,
    )
    assert report["dsr"] is not None
    assert 0.0 <= report["dsr"]["dsr"] <= 1.0
    assert report["haircut"] is not None
    assert report["haircut"]["n_trials"] == 5
    assert report["min_backtest_length"] is not None
    assert report["min_backtest_length"]["years"] > 0


# ═══════════════════════════════════════════════════════════════════════
# Full grid + benchmark : PBO / SPA / StepM populate
# ═══════════════════════════════════════════════════════════════════════


def test_report_full_with_benchmark(toy_pf, fake_grid_matrix):
    benchmark = pd.Series(0.0, index=fake_grid_matrix.index)
    grid_sharpes = pd.Series(
        [0.1, 0.2, 0.15, 0.25, 0.05, 0.3],
        index=fake_grid_matrix.columns,
    )
    report = robustness_report(
        toy_pf,
        grid_sharpes=grid_sharpes,
        grid_returns_matrix=fake_grid_matrix,
        benchmark_returns=benchmark,
        n_boot=150,
        n_mc=100,
        n_equity_paths=20,
    )
    assert report["pbo"] is not None
    assert 0.0 <= report["pbo"]["pbo"] <= 1.0
    # SPA / StepM should run (arch is an installed dep).
    assert report["spa"] is not None
    assert set(report["spa"].keys()) >= {
        "pvalue_lower", "pvalue_consistent", "pvalue_upper"
    }
    assert report["stepm"] is not None


# ═══════════════════════════════════════════════════════════════════════
# Figure builder
# ═══════════════════════════════════════════════════════════════════════


def test_build_robustness_figures_returns_plotly(toy_pf):
    report = robustness_report(
        toy_pf, n_boot=80, n_mc=80, n_equity_paths=20,
    )
    returns = toy_pf.returns
    if isinstance(returns, pd.DataFrame):
        returns = returns.iloc[:, 0]
    figs = build_robustness_figures(report, name="Toy", returns=returns.dropna())
    assert len(figs) >= 4  # forest, sharpe, fan chart, mdd
    for name, fig in figs.items():
        assert hasattr(fig, "to_dict"), f"figure {name} is not a Plotly figure"


def test_print_robustness_report_no_raise(toy_pf, capsys):
    report = robustness_report(toy_pf, n_boot=50, n_mc=50, n_equity_paths=10)
    print_robustness_report(report, name="Toy")
    captured = capsys.readouterr().out
    assert "ROBUSTNESS REPORT" in captured
    assert "Bootstrap 95% CIs" in captured
