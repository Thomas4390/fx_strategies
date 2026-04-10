"""VBT Pro strategy framework — ims_pipeline helpers.

The StrategyRunner / StrategySpec infrastructure has been removed after
the Phases 1-6 refactor migrated every active strategy to the
self-contained ``pipeline() / pipeline_nb() / create_cv_pipeline()``
pattern (see ``plans/fluttering-imagining-umbrella.md``).

Use:
    from framework.pipeline_utils import (
        compute_metric_nb,
        SHARPE_RATIO,
        analyze_portfolio,
        plot_cv_heatmap,
    )
"""
