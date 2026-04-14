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

Importing this package also applies ``vbt.yml`` (project root) to
``vbt.settings`` via ``framework.project_config`` so that every
``Portfolio.from_signals`` call picks up the centralized defaults.
"""

from framework import project_config as project_config  # noqa: F401
