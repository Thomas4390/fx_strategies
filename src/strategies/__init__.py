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

# NOTE: mr_turbo, mr_macro, rsi_daily, daily_momentum and ou_mean_reversion
# have been migrated to the ims_pipeline format (Phases 1-5 of the refactor).
# They no longer expose a StrategySpec — use ``from strategies.<name> import
# pipeline`` (or pipeline_xs/pipeline_ts for daily_momentum) instead.
REGISTRY: dict[str, "StrategySpec"] = {  # noqa: F821
    "composite_fx_alpha": composite_fx_alpha,
}
