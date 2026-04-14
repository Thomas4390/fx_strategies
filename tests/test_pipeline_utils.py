"""Tests for src/framework/pipeline_utils.py — metric dispatch, analyzer API."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import vectorbtpro as vbt

from framework.pipeline_utils import (
    ANNUALIZED_RETURN,
    ANNUALIZED_VOLATILITY,
    CALMAR_RATIO,
    COND_VALUE_AT_RISK,
    DOWNSIDE_RISK,
    FX_MINUTE_ANN_FACTOR,
    MAX_DRAWDOWN,
    METRIC_NAMES,
    OMEGA_RATIO,
    PROFIT_FACTOR,
    SHARPE_RATIO,
    SORTINO_RATIO,
    STOCK_MINUTE_ANN_FACTOR,
    TAIL_RATIO,
    TOTAL_RETURN,
    VALUE_AT_RISK,
    analyze_portfolio,
    assert_pf_equivalent,
    compute_metric_nb,
    resolve_ann_factor,
    slugify_for_filename,
)


@pytest.fixture
def sample_returns():
    """Reproducible synthetic returns stream with positive drift + noise."""
    rng = np.random.default_rng(seed=42)
    n = 10_000
    return 0.0001 + 0.002 * rng.standard_normal(n)


class TestMetricConstants:
    def test_all_ids_unique(self):
        ids = list(METRIC_NAMES.keys())
        assert len(ids) == len(set(ids)), "duplicate metric ids"

    def test_ids_contiguous_from_zero(self):
        assert sorted(METRIC_NAMES.keys()) == list(range(len(METRIC_NAMES)))

    def test_fx_ann_factor_matches_year_freq(self):
        # 24h market × 252 trading days × 60 minutes/hour
        assert FX_MINUTE_ANN_FACTOR == pytest.approx(24.0 * 60.0 * 252.0)

    def test_stock_ann_factor(self):
        assert STOCK_MINUTE_ANN_FACTOR == pytest.approx(6.5 * 60.0 * 252.0)


class TestComputeMetricDispatch:
    """Verify each dispatch branch returns a finite float matching vbt.ret_nb."""

    def test_total_return(self, sample_returns):
        val = compute_metric_nb(sample_returns, TOTAL_RETURN)
        expected = vbt.ret_nb.total_return_1d_nb(returns=sample_returns)
        assert val == pytest.approx(expected)
        assert np.isfinite(val)

    def test_sharpe_ratio(self, sample_returns):
        val = compute_metric_nb(sample_returns, SHARPE_RATIO, FX_MINUTE_ANN_FACTOR)
        expected = vbt.ret_nb.sharpe_ratio_1d_nb(
            returns=sample_returns, ann_factor=FX_MINUTE_ANN_FACTOR
        )
        assert val == pytest.approx(expected)
        assert np.isfinite(val)

    def test_sortino_ratio(self, sample_returns):
        val = compute_metric_nb(sample_returns, SORTINO_RATIO, FX_MINUTE_ANN_FACTOR)
        expected = vbt.ret_nb.sortino_ratio_1d_nb(
            returns=sample_returns, ann_factor=FX_MINUTE_ANN_FACTOR
        )
        assert val == pytest.approx(expected)

    def test_calmar_ratio(self, sample_returns):
        val = compute_metric_nb(sample_returns, CALMAR_RATIO, FX_MINUTE_ANN_FACTOR)
        expected = vbt.ret_nb.calmar_ratio_1d_nb(
            returns=sample_returns, ann_factor=FX_MINUTE_ANN_FACTOR
        )
        assert val == pytest.approx(expected)

    def test_omega_ratio(self, sample_returns):
        val = compute_metric_nb(sample_returns, OMEGA_RATIO)
        expected = vbt.ret_nb.omega_ratio_1d_nb(returns=sample_returns)
        assert val == pytest.approx(expected)

    def test_annualized_return(self, sample_returns):
        val = compute_metric_nb(
            sample_returns, ANNUALIZED_RETURN, FX_MINUTE_ANN_FACTOR
        )
        expected = vbt.ret_nb.annualized_return_1d_nb(
            returns=sample_returns, ann_factor=FX_MINUTE_ANN_FACTOR
        )
        assert val == pytest.approx(expected)

    def test_max_drawdown_sign_flipped(self, sample_returns):
        """Max drawdown is returned negated so higher == better for CV selection."""
        val = compute_metric_nb(sample_returns, MAX_DRAWDOWN)
        raw = vbt.ret_nb.max_drawdown_1d_nb(returns=sample_returns)
        assert val == pytest.approx(-raw)

    def test_profit_factor(self, sample_returns):
        val = compute_metric_nb(sample_returns, PROFIT_FACTOR)
        expected = vbt.ret_nb.profit_factor_1d_nb(returns=sample_returns)
        assert val == pytest.approx(expected)

    def test_value_at_risk_sign_flipped(self, sample_returns):
        val = compute_metric_nb(sample_returns, VALUE_AT_RISK, cutoff=0.05)
        raw = vbt.ret_nb.value_at_risk_1d_nb(returns=sample_returns, cutoff=0.05)
        assert val == pytest.approx(-raw)

    def test_tail_ratio(self, sample_returns):
        val = compute_metric_nb(sample_returns, TAIL_RATIO)
        expected = vbt.ret_nb.tail_ratio_1d_nb(returns=sample_returns)
        assert val == pytest.approx(expected)

    def test_annualized_volatility_sign_flipped(self, sample_returns):
        val = compute_metric_nb(
            sample_returns, ANNUALIZED_VOLATILITY, FX_MINUTE_ANN_FACTOR
        )
        raw = vbt.ret_nb.annualized_volatility_1d_nb(
            returns=sample_returns, ann_factor=FX_MINUTE_ANN_FACTOR
        )
        assert val == pytest.approx(-raw)

    def test_downside_risk_sign_flipped(self, sample_returns):
        val = compute_metric_nb(
            sample_returns, DOWNSIDE_RISK, FX_MINUTE_ANN_FACTOR
        )
        raw = vbt.ret_nb.downside_risk_1d_nb(
            returns=sample_returns, ann_factor=FX_MINUTE_ANN_FACTOR
        )
        assert val == pytest.approx(-raw)

    def test_cond_var_sign_flipped(self, sample_returns):
        val = compute_metric_nb(sample_returns, COND_VALUE_AT_RISK, cutoff=0.05)
        raw = vbt.ret_nb.cond_value_at_risk_1d_nb(
            returns=sample_returns, cutoff=0.05
        )
        assert val == pytest.approx(-raw)

    def test_unknown_metric_falls_back_to_total_return(self, sample_returns):
        val = compute_metric_nb(sample_returns, 999)
        expected = vbt.ret_nb.total_return_1d_nb(returns=sample_returns)
        assert val == pytest.approx(expected)


class TestResolveAnnFactor:
    def test_default_is_fx_minute(self):
        assert resolve_ann_factor(None) == FX_MINUTE_ANN_FACTOR

    def test_from_index_minute(self):
        # 3 full 24h days of minute bars → ~1440 bars/day × 252 ≈ FX_MINUTE
        idx = pd.date_range("2024-01-01", periods=3 * 24 * 60, freq="1min")
        f = resolve_ann_factor(idx)
        assert 1000.0 < f < 1_000_000.0
        # Should be close to 252 * 1440 since we have 3 full days
        assert f == pytest.approx(252 * 1440, rel=0.01)


class TestAssertPfEquivalent:
    """Ensure the equivalence helper detects divergence correctly."""

    def _make_portfolio(self, seed: int) -> vbt.Portfolio:
        rng = np.random.default_rng(seed=seed)
        n = 500
        close = pd.Series(
            100.0 + np.cumsum(rng.standard_normal(n) * 0.1),
            index=pd.date_range("2024-01-01", periods=n, freq="1min"),
            name="close",
        )
        entries = pd.Series(False, index=close.index)
        entries.iloc[50] = True
        exits = pd.Series(False, index=close.index)
        exits.iloc[150] = True
        return vbt.Portfolio.from_signals(close, entries=entries, exits=exits)

    def test_equivalent_portfolios_pass(self):
        pf_a = self._make_portfolio(seed=42)
        pf_b = self._make_portfolio(seed=42)
        assert_pf_equivalent(pf_a, pf_b)

    def test_divergent_portfolios_raise(self):
        pf_a = self._make_portfolio(seed=42)
        pf_b = self._make_portfolio(seed=1)
        with pytest.raises(AssertionError):
            assert_pf_equivalent(pf_a, pf_b)


class TestSlugifyForFilename:
    def test_replaces_slashes_with_underscore(self):
        assert slugify_for_filename("MR80 / TS10 / RSI10") == "MR80_TS10_RSI10"

    def test_replaces_spaces(self):
        assert slugify_for_filename("Combined Portfolio") == "Combined_Portfolio"

    def test_collapses_runs_of_unsafe_chars(self):
        assert slugify_for_filename("a  /  b") == "a_b"

    def test_strips_leading_trailing_underscore(self):
        assert slugify_for_filename("  leading and trailing  ") == "leading_and_trailing"

    def test_preserves_unicode_dash_and_parentheses(self):
        out = slugify_for_filename("Combined — (MR + TS)")
        assert out == "Combined_—_(MR_+_TS)"

    def test_replaces_all_windows_reserved(self):
        # <>:"/\|?* are all forbidden on Windows, plus whitespace
        out = slugify_for_filename('a<b>c:d"e/f\\g|h?i*j')
        assert out == "a_b_c_d_e_f_g_h_i_j"


class TestAnalyzePortfolioHandlesUnsafeNames:
    def test_name_with_slashes_writes_into_output_dir(self, tmp_path):
        """Regression: titles containing '/' previously caused FileNotFoundError
        because ``name.replace(' ', '_')`` left the slashes intact and
        ``pathlib.Path`` interpreted them as directory separators."""
        rng = np.random.default_rng(seed=7)
        n = 500
        close = pd.Series(
            100.0 + np.cumsum(rng.standard_normal(n) * 0.1),
            index=pd.date_range("2024-01-01", periods=n, freq="1min"),
            name="close",
        )
        entries = pd.Series(False, index=close.index)
        entries.iloc[50] = True
        exits = pd.Series(False, index=close.index)
        exits.iloc[150] = True
        pf = vbt.Portfolio.from_signals(close, entries=entries, exits=exits)

        # Title includes slashes — must not crash, must not escape tmp_path.
        analyze_portfolio(
            pf,
            name="Combined v2 — Phase 18 (MR80 / TS10 / RSI10)",
            output_dir=str(tmp_path),
            show_charts=False,
            save_excel=False,
        )

        written = sorted(p.name for p in tmp_path.iterdir())
        assert written, "analyze_portfolio wrote nothing into output_dir"
        # Nothing escaped into a subdirectory named after a slashed token.
        assert not any(p.is_dir() for p in tmp_path.iterdir())
        # All outputs use a single safe stem with underscores instead of '/'.
        stem = "Combined_v2_—_Phase_18_(MR80_TS10_RSI10)"
        assert any(f.startswith(stem) for f in written), (
            f"expected stem {stem!r} in {written}"
        )
