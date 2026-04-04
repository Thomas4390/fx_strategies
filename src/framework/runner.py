"""Strategy execution engine.

Consumes a ``StrategySpec`` and data, providing three execution modes:
- ``backtest()``  — single run with concrete parameters
- ``cv()``        — cross-validation with parameter grid + splitter
- ``full_pipeline()`` — complete holdout → CV → best params → rerun → test → save
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import pandas as pd
import vectorbtpro as vbt

from framework.spec import DEFAULT_INPUT_MAP, StrategySpec
from utils import apply_vbt_settings, compute_ann_factor, load_fx_data


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class StrategyRunner:
    """Execute a strategy in any mode from a single ``StrategySpec``."""

    def __init__(
        self,
        spec: StrategySpec,
        raw: pd.DataFrame,
        data: vbt.Data | None = None,
    ) -> None:
        self.spec = spec
        self.raw = raw
        self.data = data
        self.index_ns = vbt.dt.to_ns(raw.index)
        self.ann_factor = compute_ann_factor(raw.index)

    # -- Single run ---------------------------------------------------------

    def backtest(self, **overrides: Any) -> tuple[vbt.Portfolio, Any]:
        """Run a single backtest with concrete parameters.

        Returns ``(portfolio, indicator_result)``.
        """
        params = self._resolve_params(overrides)
        ind = self._run_indicator(self.raw, self.index_ns, params)
        pf = self._run_portfolio(self.raw, self.index_ns, ind, params)
        return pf, ind

    # -- Cross-validation ---------------------------------------------------

    def cv(
        self,
        param_grid: dict[str, list[Any]],
        splitter: Any,
        metric: str = "sharpe_ratio",
        *,
        raw: pd.DataFrame | None = None,
        index_ns: np.ndarray | None = None,
    ) -> tuple[Any, Any]:
        """Run cross-validation over *param_grid* using *splitter*.

        Returns ``(grid_perf, best_perf)`` — the full grid and best-per-split.
        Pass *raw* / *index_ns* to run on a subset (e.g. train split).
        """
        raw = raw if raw is not None else self.raw
        index_ns = index_ns if index_ns is not None else vbt.dt.to_ns(raw.index)

        cv_func = self._build_cv_func(splitter)
        vbt_params = {k: vbt.Param(v) for k, v in param_grid.items()}
        fixed = self._fixed_params(param_grid)

        takeable = self._takeable_data(raw, index_ns)

        return cv_func(
            **takeable,
            **vbt_params,
            **fixed,
            metric=metric,
            _return_grid="all",
            _index=raw.index,
        )

    # -- Full pipeline ------------------------------------------------------

    def full_pipeline(
        self,
        param_grid: dict[str, list[Any]] | None = None,
        holdout_ratio: float = 0.2,
        n_folds: int = 10,
        min_train_folds: int = 3,
        purge_td: str = "1 hour",
        metric: str = "sharpe_ratio",
        results_dir: str | None = None,
    ) -> dict[str, Any]:
        """Complete pipeline: holdout → CV → best params → rerun → test → save.

        Returns a dict with keys: ``opt_params``, ``pf_train``, ``pf_test``,
        ``grid_perf``, ``best_perf``, ``comparison``.
        """
        if param_grid is None:
            param_grid = self.spec.sweep_grid()
        if results_dir is None:
            results_dir = f"results/{self.spec.indicator.short_name}"
        os.makedirs(results_dir, exist_ok=True)

        # -- Holdout split --------------------------------------------------
        split_idx = int(len(self.raw) * (1 - holdout_ratio))
        holdout_date = self.raw.index[split_idx]
        raw_train = self.raw.loc[:holdout_date]
        raw_test = self.raw.loc[holdout_date:]
        ns_train = vbt.dt.to_ns(raw_train.index)

        print(
            f"  Train: {len(raw_train)} bars ({raw_train.index[0]} → {raw_train.index[-1]})"
        )
        print(
            f"  Test:  {len(raw_test)} bars ({raw_test.index[0]} → {raw_test.index[-1]})"
        )

        # -- Walk-forward CV on train ---------------------------------------
        splitter = vbt.Splitter.from_purged_walkforward(
            raw_train.index,
            n_folds=n_folds,
            n_test_folds=1,
            min_train_folds=min_train_folds,
            purge_td=purge_td,
        )

        n_combos = 1
        for v in param_grid.values():
            n_combos *= len(v)
        n_splits = len(splitter.splits)
        print(
            f"\n  Grid: {n_combos} combos × {n_splits} splits = {n_combos * n_splits} backtests"
        )

        grid_perf, best_perf = self.cv(
            param_grid,
            splitter,
            metric,
            raw=raw_train,
            index_ns=ns_train,
        )

        # -- Best parameter selection ---------------------------------------
        opt_params = self._extract_best_params(best_perf, param_grid)
        print(f"  Best params: {opt_params}")

        # -- Re-run on train with best params -------------------------------
        train_runner = StrategyRunner(self.spec, raw_train, self.data)
        pf_train, _ = train_runner.backtest(**opt_params)

        print(f"\n{'=' * 60}")
        print(f"OPTIMIZED — TRAIN ({self.spec.name})")
        print(f"{'=' * 60}")
        print(pf_train.stats().to_string())

        # -- Holdout test ---------------------------------------------------
        test_runner = StrategyRunner(self.spec, raw_test, self.data)
        pf_test, _ = test_runner.backtest(**opt_params)

        print(f"\n{'=' * 60}")
        print(f"HOLD-OUT TEST ({self.spec.name})")
        print(f"{'=' * 60}")
        print(pf_test.stats().to_string())

        # -- Comparison -----------------------------------------------------
        comparison = pd.DataFrame(
            {
                "Optimized (train)": pf_train.stats(),
                "Hold-out (test)": pf_test.stats(),
            }
        )
        print(f"\n{'=' * 60}")
        print("COMPARISON")
        print(f"{'=' * 60}")
        print(comparison.to_string())

        # -- Save results ---------------------------------------------------
        self._save_results(
            results_dir,
            opt_params,
            pf_train,
            pf_test,
            comparison,
            grid_perf,
        )

        return {
            "opt_params": opt_params,
            "pf_train": pf_train,
            "pf_test": pf_test,
            "grid_perf": grid_perf,
            "best_perf": best_perf,
            "comparison": comparison,
        }

    # -----------------------------------------------------------------------
    # Private helpers
    # -----------------------------------------------------------------------

    def _resolve_params(self, overrides: dict[str, Any]) -> dict[str, Any]:
        """Merge spec defaults with caller overrides."""
        params = self.spec.default_params()
        params.update(overrides)
        return params

    def _run_indicator(
        self,
        raw: pd.DataFrame,
        index_ns: np.ndarray,
        params: dict[str, Any],
    ) -> Any:
        """Build ``vbt.IF``, run it, return the indicator result."""
        ispec = self.spec.indicator

        # Collect indicator-level params (those declared in param_names)
        ind_params = {k: params[k] for k in ispec.param_names if k in params}

        factory = vbt.IF(
            class_name=ispec.class_name,
            short_name=ispec.short_name,
            input_names=list(ispec.input_names),
            param_names=list(ispec.param_names),
            output_names=list(ispec.output_names),
        ).with_apply_func(
            ispec.kernel_func,
            takes_1d=True,
            **ind_params,
        )

        # Build input kwargs from input_names → raw columns
        input_kwargs = self._build_input_kwargs(raw, index_ns, ispec.input_names)

        return factory.run(
            **input_kwargs,
            **ind_params,
            jitted_loop=True,
            jitted_warmup=True,
            execute_kwargs={"engine": "threadpool", "n_chunks": "auto"},
        )

    def _run_portfolio(
        self,
        raw: pd.DataFrame,
        index_ns: np.ndarray,
        ind_result: Any,
        params: dict[str, Any],
    ) -> vbt.Portfolio:
        """Resolve signal mapping and call ``Portfolio.from_signals``."""
        pcfg = self.spec.portfolio_config

        # Build signal_args tuple and broadcast_named_args dict
        signal_args = []
        broadcast_named_args = {}
        extra_top_level: dict[str, Any] = {}  # for RepEval args
        for rep_name, source in self.spec.signal_args_map:
            signal_args.append(vbt.Rep(rep_name))
            if source.startswith("eval:"):
                # RepEval: passed as top-level kwarg, not in broadcast_named_args
                extra_top_level[rep_name] = vbt.RepEval(source[5:])
            else:
                broadcast_named_args[rep_name] = self._resolve_source(
                    source,
                    raw,
                    index_ns,
                    ind_result,
                    params,
                )

        # Resolve portfolio-level params
        leverage = self._resolve_portfolio_ref(pcfg.leverage, ind_result, params)
        sl_stop = self._resolve_portfolio_ref(pcfg.sl_stop, ind_result, params)
        tp_stop = self._resolve_portfolio_ref(pcfg.tp_stop, ind_result, params)

        # Build from_signals kwargs
        pf_kwargs: dict[str, Any] = {
            "close": raw["close"],
            "jitted": {"parallel": True},
            "chunked": "threadpool",
            "signal_func_nb": self.spec.signal_func,
            "signal_args": tuple(signal_args),
            "broadcast_named_args": broadcast_named_args,
            "leverage": leverage,
            "slippage": pcfg.slippage,
            "fixed_fees": pcfg.fixed_fees,
            "init_cash": pcfg.init_cash,
            "freq": pcfg.freq,
        }

        # Pass OHLC if available (needed for stop-loss simulation)
        for col in ("open", "high", "low"):
            if col in raw.columns:
                pf_kwargs[col] = raw[col]

        if sl_stop is not None:
            pf_kwargs["sl_stop"] = sl_stop
        if tp_stop is not None:
            pf_kwargs["tp_stop"] = tp_stop
        if pcfg.size_type is not None:
            pf_kwargs["size_type"] = pcfg.size_type
        if pcfg.accumulate:
            pf_kwargs["accumulate"] = True
        if pcfg.upon_opposite_entry is not None:
            pf_kwargs["upon_opposite_entry"] = pcfg.upon_opposite_entry

        # RepEval top-level args (e.g., size=RepEval("np.full(...)"))
        pf_kwargs.update(extra_top_level)

        # Strategy-specific extra kwargs (e.g., leverage_mode, fees)
        if pcfg.extra_kwargs:
            pf_kwargs.update(pcfg.extra_kwargs)

        return vbt.Portfolio.from_signals(**pf_kwargs)

    def _build_cv_func(self, splitter: Any) -> Any:
        """Build the ``vbt.cv_split``-wrapped function for this strategy."""
        spec = self.spec

        def _run_pipeline(
            *,
            metric: str = "sharpe_ratio",
            **kwargs: Any,
        ) -> Any:
            # Separate takeable data from params
            takeable_names = set(spec.takeable_args)
            data_kw = {k: kwargs[k] for k in takeable_names if k in kwargs}
            param_kw = {k: v for k, v in kwargs.items() if k not in takeable_names}

            # Reconstruct raw-like DataFrame from takeable arrays
            raw_cv = self._arrays_to_raw(data_kw)
            idx_ns = data_kw.get("idx_ns", vbt.dt.to_ns(raw_cv.index))
            if idx_ns.ndim > 1:
                idx_ns = idx_ns[:, 0]

            # Resolve params: merge defaults with what cv_split passes
            params = spec.default_params()
            params.update(param_kw)

            # Run indicator
            ind = self._run_indicator(raw_cv, idx_ns, params)

            # Run portfolio
            pf = self._run_portfolio(raw_cv, idx_ns, ind, params)

            return pf.deep_getattr(metric)

        return vbt.cv_split(
            _run_pipeline,
            splitter=splitter,
            takeable_args=list(spec.takeable_args),
            parameterized_kwargs={"engine": "threadpool", "chunk_len": "auto"},
            merge_func="concat",
        )

    # -- Source resolution --------------------------------------------------

    @staticmethod
    def _resolve_source(
        source: str,
        raw: pd.DataFrame,
        index_ns: np.ndarray,
        ind_result: Any,
        params: dict[str, Any],
    ) -> Any:
        """Resolve a ``signal_args_map`` source reference to a concrete value."""
        prefix, _, name = source.partition(".")
        if prefix == "data":
            return raw[name]
        if prefix == "ind":
            return getattr(ind_result, name).values
        if prefix == "extra":
            if name == "index_ns":
                return index_ns
            raise ValueError(f"Unknown extra source: {name}")
        if prefix == "param":
            return params[name]
        raise ValueError(f"Unknown source prefix: {prefix!r} in {source!r}")

    @staticmethod
    def _resolve_portfolio_ref(
        value: Any,
        ind_result: Any,
        params: dict[str, Any],
    ) -> Any:
        """Resolve a ``PortfolioConfig`` field that may be a string reference."""
        if not isinstance(value, str):
            return value
        prefix, _, name = value.partition(".")
        if prefix == "ind":
            return getattr(ind_result, name).values
        if prefix == "param":
            return params[name]
        raise ValueError(f"Unknown portfolio ref: {value!r}")

    # -- Input wiring -------------------------------------------------------

    @staticmethod
    def _build_input_kwargs(
        raw: pd.DataFrame,
        index_ns: np.ndarray,
        input_names: tuple[str, ...],
    ) -> dict[str, Any]:
        """Map ``IndicatorSpec.input_names`` to concrete data arrays."""
        kwargs: dict[str, Any] = {}
        for name in input_names:
            mapped = DEFAULT_INPUT_MAP.get(name)
            if mapped == "__index_ns__":
                kwargs[name] = index_ns
            elif mapped == "__returns__":
                kwargs[name] = np.log(raw["close"] / raw["close"].shift(1))
            elif mapped is not None and mapped in raw.columns:
                kwargs[name] = raw[mapped]
            elif name in raw.columns:
                kwargs[name] = raw[name]
            else:
                raise ValueError(
                    f"Cannot map indicator input {name!r}: "
                    f"not in DEFAULT_INPUT_MAP and not a column of raw"
                )
        return kwargs

    # -- CV helpers ---------------------------------------------------------

    @staticmethod
    def _arrays_to_raw(data_kw: dict[str, Any]) -> pd.DataFrame:
        """Convert CV-sliced arrays back to a DataFrame for the runner."""

        def _col(arr: np.ndarray) -> np.ndarray:
            return arr[:, 0] if arr.ndim > 1 else arr

        cols: dict[str, Any] = {}
        key_map = {
            "high_arr": "high",
            "low_arr": "low",
            "close_arr": "close",
            "open_arr": "open",
            "volume_arr": "volume",
        }
        for arr_key, col_name in key_map.items():
            if arr_key in data_kw:
                cols[col_name] = _col(data_kw[arr_key])

        return pd.DataFrame(cols)

    def _takeable_data(
        self,
        raw: pd.DataFrame,
        index_ns: np.ndarray,
    ) -> dict[str, Any]:
        """Prepare the takeable arrays for ``vbt.cv_split``."""
        takeable: dict[str, Any] = {}
        key_map = {
            "high_arr": "high",
            "low_arr": "low",
            "close_arr": "close",
            "open_arr": "open",
            "volume_arr": "volume",
        }
        for name in self.spec.takeable_args:
            if name == "idx_ns":
                takeable[name] = index_ns
            elif name in key_map:
                takeable[name] = vbt.to_2d_array(raw[key_map[name]])
            else:
                raise ValueError(f"Unknown takeable arg: {name!r}")
        return takeable

    def _fixed_params(self, param_grid: dict[str, Any]) -> dict[str, Any]:
        """Return non-swept params (defaults for params NOT in param_grid)."""
        defaults = self.spec.default_params()
        return {k: v for k, v in defaults.items() if k not in param_grid}

    # -- Result extraction --------------------------------------------------

    @staticmethod
    def _extract_best_params(
        best_perf: Any,
        param_grid: dict[str, Any],
    ) -> dict[str, Any]:
        """Extract best parameter combination from CV results."""
        if not isinstance(best_perf.index, pd.MultiIndex):
            return {}

        best_row = best_perf[best_perf == best_perf.max()]
        level_names = best_perf.index.names
        opt_params: dict[str, Any] = {}
        for name in param_grid:
            if name in level_names:
                val = best_row.index.get_level_values(name)[0]
                # Convert numpy types to Python native for clean display
                opt_params[name] = val.item() if hasattr(val, "item") else val
        return opt_params

    # -- Saving results -----------------------------------------------------

    def _save_results(
        self,
        results_dir: str,
        opt_params: dict[str, Any],
        pf_train: vbt.Portfolio,
        pf_test: vbt.Portfolio,
        comparison: pd.DataFrame,
        grid_perf: Any,
    ) -> None:
        """Save plots and summary to *results_dir*."""
        from framework.plotting import plot_monthly_heatmap

        name = self.spec.name

        # Portfolio plots
        for label, pf in [("train", pf_train), ("test", pf_test)]:
            fig = pf.plot(subplots=["cumulative_returns", "drawdowns", "underwater"])
            fig.update_layout(title=f"{name} — {label.title()}", height=900)
            fig.write_html(f"{results_dir}/portfolio_{label}.html")

            fig_m = plot_monthly_heatmap(
                pf, f"{name} — {label.title()} Monthly Returns (%)"
            )
            fig_m.write_html(f"{results_dir}/monthly_{label}.html")

        # CV heatmap (pick first two sweep params for axes)
        sweep_keys = list(self.spec.sweep_grid().keys())
        if len(sweep_keys) >= 2:
            try:
                fig_cv = grid_perf.vbt.heatmap(
                    x_level=sweep_keys[0],
                    y_level=sweep_keys[1],
                    slider_level="split",
                )
                fig_cv.write_html(f"{results_dir}/cv_heatmap.html")
            except Exception:
                pass  # heatmap may fail with >2 sweep dims

        # Text summary
        summary_path = f"{results_dir}/summary.txt"
        with open(summary_path, "w") as f:
            f.write(f"{name}\n{'=' * 60}\n\n")
            f.write("BEST PARAMETERS (walk-forward CV)\n")
            f.write("-" * 40 + "\n")
            for k, v in opt_params.items():
                f.write(f"  {k}: {v}\n")
            f.write(f"\nTRAIN SET STATS\n{'-' * 40}\n")
            f.write(pf_train.stats().to_string() + "\n\n")
            f.write(f"HOLD-OUT SET STATS\n{'-' * 40}\n")
            f.write(pf_test.stats().to_string() + "\n\n")
            f.write(f"COMPARISON\n{'-' * 40}\n")
            f.write(comparison.to_string() + "\n")

        print(f"\n  Results saved to {results_dir}/")


# ---------------------------------------------------------------------------
# Module-level convenience
# ---------------------------------------------------------------------------


def run_strategy(
    spec: StrategySpec,
    data_path: str = "data/EUR-USD.parquet",
    shift_hours: int = 0,
    mode: str = "full",
    **kwargs: Any,
) -> dict[str, Any] | tuple[vbt.Portfolio, Any]:
    """Convenience function: load data and run a strategy in the given mode.

    Modes: ``"backtest"`` (single run), ``"full"`` (full pipeline with CV).
    """
    apply_vbt_settings()
    raw, data = load_fx_data(data_path, shift_hours)
    runner = StrategyRunner(spec, raw, data)

    print(f"  {len(raw)} bars: {raw.index[0]} → {raw.index[-1]}")
    print(f"  Annualization factor: {runner.ann_factor:.0f}")

    if mode == "backtest":
        return runner.backtest(**kwargs)
    if mode == "full":
        return runner.full_pipeline(**kwargs)

    raise ValueError(f"Unknown mode: {mode!r}")
