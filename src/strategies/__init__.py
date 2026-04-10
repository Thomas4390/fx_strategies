"""Strategy registry — import specs by short name.

Active strategies (intraday):
  mr_turbo           — Validated intraday VWAP MR (baseline, Sharpe 0.23 w/o macro)
  mr_macro           — Macro-filtered MR (best intraday, Sharpe 0.94)
  ou_mean_reversion  — Vol-targeted VWAP MR with dynamic leverage

Active strategies (daily):
  daily_momentum     — XS momentum + TS momentum + RSI (standalone module)
  composite_fx_alpha — Daily multi-factor (carry + momentum + value)

Deprecated (moved to old_strategies/):
  mr_v1-v4, donchian_breakout, kalman_trend — superseded or negative on FX

Note: daily_momentum is a standalone module (not StrategySpec-based).
Use directly: from strategies.daily_momentum import backtest_xs_momentum, etc.
"""

from strategies.composite_fx_alpha import spec as composite_fx_alpha
from strategies.ou_mean_reversion import spec as ou_mean_reversion

# NOTE: mr_turbo has been migrated to the ims_pipeline format (Phase 1 of the
# refactor — see plans/fluttering-imagining-umbrella.md). It no longer exposes
# a StrategySpec. Use ``from strategies.mr_turbo import pipeline`` instead.
REGISTRY: dict[str, "StrategySpec"] = {  # noqa: F821
    "ou_mean_reversion": ou_mean_reversion,
    "composite_fx_alpha": composite_fx_alpha,
}
