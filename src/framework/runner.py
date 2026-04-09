"""Strategy execution engine.

Consumes a ``StrategySpec`` and data, providing three execution modes:
- ``backtest()``  — single run with concrete parameters
- ``cv()``        — cross-validation with parameter grid + splitter
- ``full_pipeline()`` — complete holdout -> CV -> best params -> rerun -> test -> save
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import vectorbtpro as vbt

from framework.spec import DEFAULT_INPUT_MAP, StrategySpec

# Project root: three levels up from this file (src/framework/runner.py -> project/)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

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
        # Pre-compute log returns (avoids recalculation per indicator call)
        self._returns = np.log(raw["close"] / raw["close"].shift(1))

    # -- Single run ---------------------------------------------------------

    def backtest(self, **overrides: Any) -> tuple[vbt.Portfolio, Any]:
        """Run a single backtest with concrete parameters.

        Returns ``(portfolio, indicator_result)``.
        """
        params = self._resolve_params(overrides)
        prepared = self._run_prepare(self.raw, self.data)
        ind = self._run_indicator(self.raw, self.index_ns, params, prepared=prepared)
        pf = self._run_portfolio(self.raw, self.index_ns, ind, params, prepared=prepared)
        return pf, ind

    def _run_prepare(
        self,
        raw: pd.DataFrame,
        data: vbt.Data | None,
    ) -> dict[str, Any]:
        """Run prepare_fn if defined, return pre-computed arrays."""
        if self.spec.prepare_fn is None:
            return {}
        return self.spec.prepare_fn(raw, data)

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

        Runs ALL parameter combinations at once per split via VBT native
        broadcasting, enabling true multicore parallelization through
        ``chunked="threadpool"``.

        Returns ``(grid_perf, best_perf)`` — the full grid and best-per-split.
        """
        raw = raw if raw is not None else self.raw

        # Merge param grid with defaults (arrays for swept, scalars for fixed)
        params = self.spec.default_params()
        params.update(param_grid)

        split_results = []
        n_splits = len(splitter.splits)

        for split_idx in range(n_splits):
            for set_idx in range(splitter.n_sets):
                raw_split = splitter.take(raw, split=split_idx, set_=set_idx)
                if len(raw_split) < 2:
                    continue
                ns_split = vbt.dt.to_ns(raw_split.index)

                # Run ALL combos at once — VBT broadcasts params to columns
                prepared_split = self._run_prepare(raw_split, None)
                ind = self._run_indicator(
                    raw_split, ns_split, params, parallel=True, prepared=prepared_split
                )
                pf = self._run_portfolio(
                    raw_split, ns_split, ind, params, parallel=True, prepared=prepared_split
                )

                metric_val = getattr(pf, metric)

                # Build index: (split, set, param1, param2, ...)
                set_label = (
                    splitter.set_labels[set_idx]
                    if hasattr(splitter, "set_labels")
                    and splitter.set_labels is not None
                    else set_idx
                )
                if isinstance(metric_val, pd.Series) and len(metric_val) > 1:
                    orig_idx = metric_val.index
                    tuples = []
                    for t in (
                        orig_idx
                        if isinstance(orig_idx, pd.MultiIndex)
                        else [(v,) for v in orig_idx]
                    ):
                        tuples.append((split_idx, set_label, *t))
                    names = ["split", "set"] + list(
                        orig_idx.names
                        if isinstance(orig_idx, pd.MultiIndex)
                        else [orig_idx.name]
                    )
                    new_idx = pd.MultiIndex.from_tuples(tuples, names=names)
                    split_results.append(pd.Series(metric_val.values, index=new_idx))
                else:
                    val = (
                        metric_val
                        if isinstance(metric_val, (int, float, np.floating))
                        else metric_val.iloc[0]
                    )
                    idx = pd.MultiIndex.from_tuples(
                        [(split_idx, set_label)], names=["split", "set"]
                    )
                    split_results.append(pd.Series([val], index=idx))

            print(
                f"\r  Split {split_idx + 1}/{n_splits}", end="", flush=True
            )

        print()

        grid_perf = pd.concat(split_results)

        # Best: mean metric per swept-param combo across train splits.
        # VBT prefixes param names (e.g. "lookback" → "mr_v1_lookback"),
        # so match by suffix.
        train_mask = grid_perf.index.get_level_values("set").isin(
            ["train", "set_0", 0]
        )
        train_perf = grid_perf[train_mask]

        idx_names = list(train_perf.index.names)
        sweep_levels = []
        for pname in param_grid:
            for iname in idx_names:
                if iname and iname.endswith(pname):
                    sweep_levels.append(iname)
                    break

        if sweep_levels:
            best_perf = train_perf.groupby(sweep_levels).mean()
        else:
            best_perf = train_perf

        return grid_perf, best_perf

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
        """Complete pipeline: holdout -> CV -> best params -> rerun -> test -> save.

        Returns a dict with keys: ``opt_params``, ``pf_train``, ``pf_test``,
        ``grid_perf``, ``best_perf``, ``comparison``.
        """
        if param_grid is None:
            param_grid = self.spec.sweep_grid()
        if results_dir is None:
            results_dir = str(
                _PROJECT_ROOT / "results" / self.spec.indicator.short_name
            )
        os.makedirs(results_dir, exist_ok=True)

        # -- Holdout split (VBT native) ------------------------------------
        n = len(self.raw)
        split_idx = int(n * (1 - holdout_ratio))
        holdout_splitter = vbt.Splitter.from_splits(
            self.raw.index,
            [[slice(0, split_idx), slice(split_idx, n)]],
            set_labels=["train", "test"],
        )
        raw_train = holdout_splitter.take(self.raw, split=0, set_="train")
        raw_test = holdout_splitter.take(self.raw, split=0, set_="test")
        ns_train = vbt.dt.to_ns(raw_train.index)

        print(
            f"  Train: {len(raw_train)} bars ({raw_train.index[0]} -> {raw_train.index[-1]})"
        )
        print(
            f"  Test:  {len(raw_test)} bars ({raw_test.index[0]} -> {raw_test.index[-1]})"
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
            f"\n  Grid: {n_combos} combos x {n_splits} splits = {n_combos * n_splits} backtests"
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
        pf_train, ind_train = train_runner.backtest(**opt_params)

        print(f"\n{'=' * 60}")
        print(f"OPTIMIZED — TRAIN ({self.spec.name})")
        print(f"{'=' * 60}")
        print(self._safe_stats(pf_train).to_string())
        self._print_trade_stats(pf_train)

        # -- Holdout test ---------------------------------------------------
        test_runner = StrategyRunner(self.spec, raw_test, self.data)
        pf_test, ind_test = test_runner.backtest(**opt_params)

        print(f"\n{'=' * 60}")
        print(f"HOLD-OUT TEST ({self.spec.name})")
        print(f"{'=' * 60}")
        print(self._safe_stats(pf_test).to_string())
        self._print_trade_stats(pf_test)

        # -- Comparison -----------------------------------------------------
        comparison = pd.DataFrame(
            {
                "Optimized (train)": self._safe_stats(pf_train),
                "Hold-out (test)": self._safe_stats(pf_test),
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
            raw_train,
            raw_test,
            ind_train,
            ind_test,
        )

        return {
            "opt_params": opt_params,
            "pf_train": pf_train,
            "pf_test": pf_test,
            "grid_perf": grid_perf,
            "best_perf": best_perf,
            "comparison": comparison,
        }

    # -- Backtest visualization (opt-in) ------------------------------------

    def save_backtest_plots(
        self,
        pf: vbt.Portfolio,
        ind: Any,
        results_dir: str,
        label: str = "backtest",
        prepared: dict[str, Any] | None = None,
    ) -> None:
        """Save plots for a single backtest run."""
        from framework.plotting import (
            build_trade_report,
            plot_monthly_heatmap,
            plot_portfolio_summary,
            plot_trade_analysis,
            plot_trade_signals,
            resolve_overlays,
            show_browser,
        )

        if prepared is None:
            prepared = self._run_prepare(self.raw, self.data)

        os.makedirs(results_dir, exist_ok=True)
        name = self.spec.name

        fig = plot_portfolio_summary(pf, f"{name} — {label.title()}")
        fig.write_html(f"{results_dir}/portfolio_{label}.html")

        fig_m = plot_monthly_heatmap(
            pf, f"{name} — {label.title()} Monthly Returns (%)"
        )
        fig_m.write_html(f"{results_dir}/monthly_{label}.html")

        overlays = resolve_overlays(self.spec, self.raw, ind, prepared)
        fig_ts = plot_trade_signals(pf, f"{name} — {label.title()} Signals", overlays)
        fig_ts.write_html(f"{results_dir}/trade_signals_{label}.html")

        fig_ta = plot_trade_analysis(pf, f"{name} — {label.title()} Trade Analysis")
        fig_ta.write_html(f"{results_dir}/trade_analysis_{label}.html")

        with open(f"{results_dir}/summary_{label}.txt", "w") as f:
            f.write(f"{name} — {label.title()}\n{'=' * 60}\n\n")
            f.write(build_trade_report(pf) + "\n")

        print(f"\n  Results saved to {results_dir}/")

        # Show key plots in browser
        show_browser(fig)

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
        *,
        parallel: bool = True,
        prepared: dict[str, Any] | None = None,
    ) -> Any:
        """Build ``vbt.IF``, run it, return the indicator result."""
        if prepared is None:
            prepared = {}
        ispec = self.spec.indicator

        # Collect indicator-level params (those declared in param_names)
        ind_params = {k: params[k] for k in ispec.param_names if k in params}

        # with_apply_func needs scalar defaults (for compilation);
        # lists/arrays are sweep values passed only to run().
        scalar_defaults = {
            k: (v[0] if isinstance(v, list) else v) for k, v in ind_params.items()
        }
        factory = vbt.IF(
            class_name=ispec.class_name,
            short_name=ispec.short_name,
            input_names=list(ispec.input_names),
            param_names=list(ispec.param_names),
            output_names=list(ispec.output_names),
        ).with_apply_func(
            ispec.kernel_func,
            takes_1d=True,
            **scalar_defaults,
        )

        # Build input kwargs from input_names -> raw columns
        input_kwargs = self._build_input_kwargs(raw, index_ns, ispec.input_names, prepared)

        run_kwargs: dict[str, Any] = {
            **input_kwargs,
            **ind_params,
            "jitted_loop": True,
            "jitted_warmup": True,
            "param_product": True,
        }
        if parallel:
            # Multi-column sweep: parallelize across parameter combos
            run_kwargs["execute_kwargs"] = {"engine": "threadpool", "n_chunks": "auto"}
        return factory.run(**run_kwargs)

    def _run_portfolio(
        self,
        raw: pd.DataFrame,
        index_ns: np.ndarray,
        ind_result: Any,
        params: dict[str, Any],
        *,
        parallel: bool = True,
        prepared: dict[str, Any] | None = None,
    ) -> vbt.Portfolio:
        """Resolve signal mapping and call ``Portfolio.from_signals``."""
        if prepared is None:
            prepared = {}
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
                    prepared,
                )

        # Detect multi-column mode from indicator output width
        n_cols = ind_result.wrapper.shape_2d[1]
        ind_columns = ind_result.wrapper.columns if n_cols > 1 else None

        # Broadcast 1D data arrays to match indicator column count.
        # Only broadcast "data.*" sources — "extra.*" (index_ns) and "param.*"
        # are accessed directly in signal functions, not via select_nb.
        if n_cols > 1:
            data_sources = {
                rep for rep, src in self.spec.signal_args_map if src.startswith("data.")
            }
            for key in data_sources:
                if key in broadcast_named_args:
                    arr = np.asarray(broadcast_named_args[key])
                    if arr.ndim == 1:
                        broadcast_named_args[key] = np.tile(arr[:, None], n_cols)

        # Resolve portfolio-level params
        leverage = self._resolve_portfolio_ref(pcfg.leverage, ind_result, params, prepared)
        sl_stop = self._resolve_portfolio_ref(pcfg.sl_stop, ind_result, params, prepared)
        tp_stop = self._resolve_portfolio_ref(pcfg.tp_stop, ind_result, params, prepared)

        # Broadcast close (and OHLC) to match indicator column count
        close_val = raw["close"]
        if ind_columns is not None:
            close_val = pd.DataFrame(
                np.tile(raw["close"].values[:, None], n_cols),
                index=raw.index,
                columns=ind_columns,
            )

        # Build from_signals kwargs
        pf_kwargs: dict[str, Any] = {
            "close": close_val,
            "signal_func_nb": self.spec.signal_func,
            "signal_args": tuple(signal_args),
            "broadcast_named_args": broadcast_named_args,
            "leverage": leverage,
            "slippage": pcfg.slippage,
            "fixed_fees": pcfg.fixed_fees,
            "init_cash": pcfg.init_cash,
            "freq": pcfg.freq,
        }

        if parallel and n_cols > 1:
            pf_kwargs["chunked"] = "threadpool"

        # Pass OHLC if available (needed for stop-loss simulation)
        for col in ("open", "high", "low"):
            if col in raw.columns:
                if ind_columns is not None:
                    pf_kwargs[col] = pd.DataFrame(
                        np.tile(raw[col].values[:, None], n_cols),
                        index=raw.index,
                        columns=ind_columns,
                    )
                else:
                    pf_kwargs[col] = raw[col]

        optional = {
            "sl_stop": sl_stop,
            "tp_stop": tp_stop,
            "size_type": pcfg.size_type,
            "upon_opposite_entry": pcfg.upon_opposite_entry,
        }
        pf_kwargs.update({k: v for k, v in optional.items() if v is not None})
        if pcfg.accumulate:
            pf_kwargs["accumulate"] = True

        # RepEval top-level args (e.g., size=RepEval("np.full(...)"))
        pf_kwargs.update(extra_top_level)

        # Strategy-specific extra kwargs (e.g., leverage_mode, fees)
        if pcfg.extra_kwargs:
            pf_kwargs.update(pcfg.extra_kwargs)

        return vbt.Portfolio.from_signals(**pf_kwargs)

    # (_build_cv_func removed — cv() now runs all combos at once per split)

    # -- Source resolution --------------------------------------------------

    @staticmethod
    def _resolve_source(
        source: str,
        raw: pd.DataFrame,
        index_ns: np.ndarray,
        ind_result: Any,
        params: dict[str, Any],
        prepared: dict[str, Any] | None = None,
    ) -> Any:
        """Resolve a ``signal_args_map`` source reference to a concrete value."""
        if prepared is None:
            prepared = {}
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
        if prefix == "pre":
            return prepared[name]
        raise ValueError(f"Unknown source prefix: {prefix!r} in {source!r}")

    @staticmethod
    def _resolve_portfolio_ref(
        value: Any,
        ind_result: Any,
        params: dict[str, Any],
        prepared: dict[str, Any] | None = None,
    ) -> Any:
        """Resolve a ``PortfolioConfig`` field that may be a string reference."""
        if not isinstance(value, str):
            return value
        prefix, _, name = value.partition(".")
        if prefix == "ind":
            return getattr(ind_result, name).values
        if prefix == "param":
            return params[name]
        if prefix == "pre":
            if prepared is None:
                raise ValueError(f"Portfolio ref {value!r} requires prepared dict")
            return prepared[name]
        raise ValueError(f"Unknown portfolio ref: {value!r}")

    # -- Input wiring -------------------------------------------------------

    def _build_input_kwargs(
        self,
        raw: pd.DataFrame,
        index_ns: np.ndarray,
        input_names: tuple[str, ...],
        prepared: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Map ``IndicatorSpec.input_names`` to concrete data arrays."""
        if prepared is None:
            prepared = {}
        kwargs: dict[str, Any] = {}
        for name in input_names:
            if name in prepared:
                kwargs[name] = prepared[name]
                continue
            mapped = DEFAULT_INPUT_MAP.get(name)
            if mapped == "__index_ns__":
                kwargs[name] = index_ns
            elif mapped == "__returns__":
                # Use cached returns if raw matches, else compute for slice
                if len(raw) == len(self._returns):
                    kwargs[name] = self._returns
                else:
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

    # -- Result extraction --------------------------------------------------

    @staticmethod
    def _extract_best_params(
        best_perf: Any,
        param_grid: dict[str, Any],
    ) -> dict[str, Any]:
        """Extract best parameter combination from CV results.

        Handles VBT-prefixed index names (e.g. ``mr_v1_lookback`` for ``lookback``).
        """
        idx = best_perf.index
        if not isinstance(idx, pd.MultiIndex) and idx.name is None:
            return {}

        best_row = best_perf[best_perf == best_perf.max()]
        level_names = list(
            idx.names if isinstance(idx, pd.MultiIndex) else [idx.name]
        )

        opt_params: dict[str, Any] = {}
        for name in param_grid:
            # Match exact name or VBT-prefixed name (e.g. "mr_v1_lookback")
            matched = name if name in level_names else None
            if matched is None:
                for ln in level_names:
                    if ln and ln.endswith(name):
                        matched = ln
                        break
            if matched is not None:
                val = best_row.index.get_level_values(matched)[0]
                opt_params[name] = val.item() if hasattr(val, "item") else val
        return opt_params

    # -- Printing helpers ---------------------------------------------------

    @staticmethod
    def _safe_stats(pf: vbt.Portfolio) -> pd.Series:
        """Get portfolio stats, skipping duration metrics on overflow."""
        try:
            return pf.stats()
        except OverflowError:
            return pf.stats(tags="!duration")

    @staticmethod
    def _print_trade_stats(pf: vbt.Portfolio) -> None:
        """Print returns and trade stats if trades exist."""
        print(f"\nRETURNS STATS\n{'-' * 40}")
        print(pf.returns_stats().to_string())
        if pf.trades.count() > 0:
            print(f"\nTRADE STATS\n{'-' * 40}")
            print(pf.trades.stats().to_string())

    # -- Saving results -----------------------------------------------------

    def _save_results(
        self,
        results_dir: str,
        opt_params: dict[str, Any],
        pf_train: vbt.Portfolio,
        pf_test: vbt.Portfolio,
        comparison: pd.DataFrame,
        grid_perf: Any,
        raw_train: pd.DataFrame,
        raw_test: pd.DataFrame,
        ind_train: Any,
        ind_test: Any,
    ) -> None:
        """Save plots and summary to *results_dir*."""
        from framework.plotting import (
            build_trade_report,
            plot_cv_stability,
            plot_monthly_heatmap,
            plot_partial_dependence,
            plot_portfolio_summary,
            plot_rolling_sharpe,
            plot_trade_analysis,
            plot_trade_signals,
            plot_train_vs_test,
            resolve_overlays,
            show_browser,
        )

        name = self.spec.name

        # Per-split plots — keep references for browser display
        fig_portfolio: dict[str, Any] = {}
        for label, pf, raw, ind in [
            ("train", pf_train, raw_train, ind_train),
            ("test", pf_test, raw_test, ind_test),
        ]:
            # Portfolio summary (enhanced with trade_pnl)
            fig = plot_portfolio_summary(pf, f"{name} — {label.title()}")
            fig.write_html(f"{results_dir}/portfolio_{label}.html")
            fig_portfolio[label] = fig

            # Monthly heatmap
            fig_m = plot_monthly_heatmap(
                pf, f"{name} — {label.title()} Monthly Returns (%)"
            )
            fig_m.write_html(f"{results_dir}/monthly_{label}.html")

            # Trade signals with indicator overlays
            prepared_set = self._run_prepare(raw, None)
            overlays = resolve_overlays(self.spec, raw, ind, prepared_set)
            fig_ts = plot_trade_signals(
                pf, f"{name} — {label.title()} Signals", overlays
            )
            fig_ts.write_html(f"{results_dir}/trade_signals_{label}.html")

            # Trade analysis grid
            fig_ta = plot_trade_analysis(pf, f"{name} — {label.title()} Trade Analysis")
            fig_ta.write_html(f"{results_dir}/trade_analysis_{label}.html")

        # CV heatmap (pick first two sweep params for axes)
        sweep_keys = list(self.spec.sweep_grid().keys())
        fig_cv = None
        if len(sweep_keys) >= 2:
            try:
                fig_cv = grid_perf.vbt.heatmap(
                    x_level=sweep_keys[0],
                    y_level=sweep_keys[1],
                    slider_level="split",
                )
                fig_cv.write_html(f"{results_dir}/cv_heatmap.html")
            except Exception:
                fig_cv = None

        # CV stability
        fig_stab = plot_cv_stability(grid_perf, f"{name} — CV Stability")
        fig_stab.write_html(f"{results_dir}/cv_stability.html")

        # Parameter sensitivity
        train_mask = grid_perf.index.get_level_values("set").isin(
            ["train", "set_0", 0]
        )
        fig_pd = plot_partial_dependence(
            grid_perf[train_mask],
            self.spec.sweep_grid(),
            f"{name} — Parameter Sensitivity",
        )
        fig_pd.write_html(f"{results_dir}/partial_dependence.html")

        # Train vs Test
        fig_tvt = plot_train_vs_test(grid_perf, f"{name} — Overfitting Check")
        fig_tvt.write_html(f"{results_dir}/train_vs_test.html")

        # Rolling Sharpe for train and test
        for lbl, pf_set in [("train", pf_train), ("test", pf_test)]:
            fig_rs = plot_rolling_sharpe(
                pf_set, title=f"{name} — {lbl.title()} Rolling Sharpe"
            )
            fig_rs.write_html(f"{results_dir}/rolling_sharpe_{lbl}.html")

        # Text summary (enhanced with trade stats)
        summary_path = f"{results_dir}/summary.txt"
        with open(summary_path, "w") as f:
            f.write(f"{name}\n{'=' * 60}\n\n")
            f.write("BEST PARAMETERS (walk-forward CV)\n")
            f.write("-" * 40 + "\n")
            for k, v in opt_params.items():
                f.write(f"  {k}: {v}\n")

            for label, pf in [("TRAIN", pf_train), ("HOLD-OUT", pf_test)]:
                f.write(f"\n\n{'=' * 40}\n{label} SET\n{'=' * 40}\n\n")
                f.write(build_trade_report(pf) + "\n")

            f.write(f"\n\nCOMPARISON\n{'-' * 40}\n")
            f.write(comparison.to_string() + "\n")

        print(f"\n  Results saved to {results_dir}/")

        # Show key analysis plots in browser
        show_browser(fig_portfolio["train"])
        show_browser(fig_portfolio["test"])
        if fig_cv is not None:
            show_browser(fig_cv)
        show_browser(fig_stab)
        show_browser(fig_tvt)


# ---------------------------------------------------------------------------
# Module-level convenience
# ---------------------------------------------------------------------------


def run_strategy(
    spec: StrategySpec,
    data_path: str = "data/EUR-USD-minute.parquet",
    shift_hours: int = 0,
    mode: str = "full",
    **kwargs: Any,
) -> dict[str, Any] | tuple[vbt.Portfolio, Any]:
    """Convenience function: load data and run a strategy in the given mode.

    Modes: ``"backtest"`` (single run), ``"full"`` (full pipeline with CV).
    """
    from utils import apply_vbt_settings, compute_ann_factor, load_fx_data

    apply_vbt_settings()
    raw, data = load_fx_data(data_path, shift_hours)
    runner = StrategyRunner(spec, raw, data)

    ann_factor = compute_ann_factor(raw.index)
    print(f"  {len(raw)} bars: {raw.index[0]} -> {raw.index[-1]}")
    print(f"  Annualization factor: {ann_factor:.0f}")

    if mode == "backtest":
        pf, ind = runner.backtest(**kwargs)
        results_dir = str(_PROJECT_ROOT / "results" / spec.indicator.short_name)
        prepared = runner._run_prepare(raw, data)
        runner.save_backtest_plots(pf, ind, results_dir, prepared=prepared)
        return pf, ind
    if mode == "full":
        return runner.full_pipeline(**kwargs)

    raise ValueError(f"Unknown mode: {mode!r}")
