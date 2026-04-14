"""Plotly visualization helpers for strategy results.

The original monolithic ``plotting.py`` (2654 lines) was split into
dedicated submodules so that every file stays under the 800-line rule:

- ``_core``: layout helpers, monthly heatmap, portfolio summary,
  overlay resolution, HTML tearsheet, ``generate_full_report``.
- ``_helpers``: ``_slice_pf_last`` / ``_pick_first_column`` shared
  between ``_core``, ``_trades``, ``_equity`` (broken out to avoid a
  circular import).
- ``_equity``: rolling Sharpe, drawdown, returns distribution,
  multi-strategy comparison, top-N equity, CV stability,
  partial dependence, train-vs-test.
- ``_trades``: trade signals, trade analysis, trade duration, orders
  on price, exposure, value & cash, orders heatmap.
- ``_params``: 2D heatmap (+ slider), 3D surface, 3D volume.
- ``_reports``: text-based report builders and terminal pretty printers.
- ``_pipelines``: standalone runners (single run, grid, full report).

Every public name is re-exported here so existing callers
(``from framework.plotting import plot_monthly_heatmap``) keep working.
"""

from __future__ import annotations

from ._core import (
    _apply_title_layout,
    make_fullscreen,
    save_fullscreen_html,
    show_browser,
    plot_monthly_heatmap,
    plot_portfolio_summary,
    generate_html_tearsheet,
    generate_full_report,
    resolve_overlays,
)
from ._equity import (
    plot_cv_stability,
    plot_drawdown_analysis,
    plot_equity_top_n,
    plot_multi_strategy_equity,
    plot_partial_dependence,
    plot_returns_distribution,
    plot_rolling_sharpe,
    plot_train_vs_test,
)
from ._params import (
    plot_param_heatmap,
    plot_param_heatmap_slider,
    plot_param_surface,
    plot_param_volume,
)
from ._pipelines import (
    compute_static_param_grid,
    generate_param_grid_plots,
    generate_single_run_plots,
    generate_standalone_report,
    run_standalone_grid,
)
from ._reports import (
    build_trade_report,
    print_cv_results,
    print_extended_stats,
    print_grid_results,
)
from ._robustness import (
    plot_bootstrap_distribution,
    plot_cpcv_distribution,
    plot_equity_fan_chart,
    plot_mdd_distribution,
    plot_metric_ci_forest,
    plot_pbo_logits,
    plot_rolling_metric_stability,
    plot_spa_pvalues,
)
from ._portfolio_mix import (
    generate_portfolio_mix_plots,
    plot_dd_cap_activity,
    plot_leverage_and_vol,
    plot_regime_overlay,
    plot_rolling_correlation_heatmap,
    plot_rolling_correlation_pairs,
    plot_strategy_contribution,
    plot_turnover,
    plot_weights_distribution,
    plot_weights_rolling_mean,
    plot_weights_stacked_area,
)
from ._trades import (
    plot_exposure,
    plot_orders_heatmap,
    plot_orders_on_price,
    plot_trade_analysis,
    plot_trade_duration,
    plot_trade_signals,
    plot_trades_on_price,
    plot_value_and_cash,
)

__all__ = [
    "_apply_title_layout",
    "make_fullscreen",
    "save_fullscreen_html",
    "show_browser",
    "plot_monthly_heatmap",
    "plot_portfolio_summary",
    "plot_rolling_sharpe",
    "plot_drawdown_analysis",
    "plot_multi_strategy_equity",
    "plot_returns_distribution",
    "plot_equity_top_n",
    "plot_trade_signals",
    "plot_trade_analysis",
    "plot_trade_duration",
    "plot_orders_on_price",
    "plot_trades_on_price",
    "plot_exposure",
    "plot_value_and_cash",
    "plot_orders_heatmap",
    "plot_param_heatmap",
    "plot_param_heatmap_slider",
    "plot_param_volume",
    "plot_param_surface",
    "plot_partial_dependence",
    "plot_cv_stability",
    "plot_train_vs_test",
    "build_trade_report",
    "print_extended_stats",
    "print_grid_results",
    "print_cv_results",
    "generate_html_tearsheet",
    "generate_full_report",
    "resolve_overlays",
    "compute_static_param_grid",
    "run_standalone_grid",
    "generate_single_run_plots",
    "generate_param_grid_plots",
    "generate_standalone_report",
    "plot_weights_stacked_area",
    "plot_weights_rolling_mean",
    "plot_weights_distribution",
    "plot_strategy_contribution",
    "plot_rolling_correlation_heatmap",
    "plot_rolling_correlation_pairs",
    "plot_leverage_and_vol",
    "plot_dd_cap_activity",
    "plot_regime_overlay",
    "plot_turnover",
    "generate_portfolio_mix_plots",
    "plot_bootstrap_distribution",
    "plot_metric_ci_forest",
    "plot_equity_fan_chart",
    "plot_mdd_distribution",
    "plot_pbo_logits",
    "plot_spa_pvalues",
    "plot_rolling_metric_stability",
    "plot_cpcv_distribution",
]
