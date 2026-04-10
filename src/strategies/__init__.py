"""Strategy registry — import specs by short name.

Active strategies:
  mr_turbo           — Validated intraday VWAP MR (baseline, Sharpe 0.23 w/o macro)
  mr_macro           — Macro-filtered MR (best validated, Sharpe 0.94)
  ou_mean_reversion  — Vol-targeted VWAP MR with dynamic leverage
  composite_fx_alpha — Daily multi-factor (carry + momentum + value)

Deprecated (moved to old_strategies/):
  mr_v1-v4, donchian_breakout, kalman_trend — superseded or negative on FX
"""

from strategies.composite_fx_alpha import spec as composite_fx_alpha
from strategies.mr_turbo import spec as mr_turbo
from strategies.ou_mean_reversion import spec as ou_mean_reversion

REGISTRY: dict[str, "StrategySpec"] = {  # noqa: F821
    "ou_mean_reversion": ou_mean_reversion,
    "mr_turbo": mr_turbo,
    "composite_fx_alpha": composite_fx_alpha,
}
