"""Strategy registry — import specs by short name."""

from strategies.composite_fx_alpha import spec as composite_fx_alpha
from strategies.donchian_breakout import spec as donchian_breakout
from strategies.kalman_trend import spec as kalman_trend
from strategies.mr_v1 import spec as mr_v1
from strategies.mr_v2 import spec as mr_v2
from strategies.mr_v3 import spec as mr_v3
from strategies.mr_v4 import spec as mr_v4
from strategies.mr_turbo import spec as mr_turbo
from strategies.ou_mean_reversion import spec as ou_mean_reversion

REGISTRY: dict[str, "StrategySpec"] = {  # noqa: F821
    "ou_mean_reversion": ou_mean_reversion,
    "mr_v1": mr_v1,
    "mr_v2": mr_v2,
    "mr_v3": mr_v3,
    "mr_v4": mr_v4,
    "mr_turbo": mr_turbo,
    "kalman_trend": kalman_trend,
    "donchian_breakout": donchian_breakout,
    "composite_fx_alpha": composite_fx_alpha,
}
