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

# All active strategies expose the ims_pipeline triplet (``pipeline``,
# ``pipeline_nb``, ``create_cv_pipeline``) directly, so none of them
# register a StrategySpec anymore.
#
# Use ``from strategies.<name> import pipeline`` (or ``pipeline_xs`` /
# ``pipeline_ts`` for daily_momentum) to run a strategy.
REGISTRY: dict = {}
