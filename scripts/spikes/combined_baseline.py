"""Capture baseline metrics for combined_portfolio{,_v2} before the native refonte.

Run on main (or pre-refonte branch) to capture the "ground truth" metrics,
then re-run after the refonte to confirm that Sharpe/Total Return/Max DD
stay within tolerance (<= 1e-6 absolute, given the spike showed 1e-14
equivalence on synthetic data).

Writes JSON to /tmp/combined_baseline.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))


from strategies.combined_portfolio import (
    build_combined_portfolio,
    get_strategy_daily_returns,
)
from strategies.combined_portfolio_v2 import build_combined_portfolio_v2
from utils import apply_vbt_settings


def _metrics(res: dict) -> dict[str, float]:
    return {
        "sharpe": float(res["sharpe"]),
        "annual_return": float(res["annual_return"]),
        "annual_vol": float(res["annual_vol"]),
        "max_drawdown": float(res["max_drawdown"]),
        "wf_avg_sharpe": float(res["wf_avg_sharpe"]),
        "wf_pos_years": int(res["wf_pos_years"]),
        "portfolio_returns_sum": float(
            res["portfolio_returns"].sum() if "portfolio_returns" in res else 0.0
        ),
    }


def main() -> None:
    apply_vbt_settings()
    import pandas as pd
    import vectorbtpro as vbt

    vbt.settings.returns.year_freq = pd.Timedelta(days=252)

    print("Loading strategy daily returns…")
    strat_rets = get_strategy_daily_returns()

    out: dict[str, dict[str, float]] = {}

    print("\n-- v1 --")
    for alloc in ("risk_parity", "equal", "mr_heavy"):
        res = build_combined_portfolio(strat_rets, allocation=alloc)
        out[f"v1/{alloc}"] = _metrics(res)
        print(
            f"  {alloc:<12} sharpe={res['sharpe']:+.6f}  "
            f"ret={res['annual_return']:+.6f}  "
            f"mdd={res['max_drawdown']:+.6f}"
        )

    print("\n-- v2 --")
    v2_configs = [
        (
            "risk_parity_nolev",
            dict(allocation="risk_parity", target_vol=None, dd_cap_enabled=False),
        ),
        (
            "regime_ml3_tv012",
            dict(
                allocation="regime_adaptive",
                target_vol=0.12,
                max_leverage=3.0,
                dd_cap_enabled=True,
            ),
        ),
        (
            "phase18_mr80",
            dict(
                allocation="custom",
                custom_weights={
                    "MR_Macro": 0.80,
                    "TS_Momentum_3p": 0.10,
                    "RSI_Daily_4p": 0.10,
                },
                target_vol=0.28,
                max_leverage=12.0,
                dd_cap_enabled=False,
            ),
        ),
    ]
    for name, kwargs in v2_configs:
        strat_subset = strat_rets
        if "custom_weights" in kwargs:
            strat_subset = {k: strat_rets[k] for k in kwargs["custom_weights"]}
        res = build_combined_portfolio_v2(strat_subset, **kwargs)
        out[f"v2/{name}"] = _metrics(res)
        print(
            f"  {name:<18} sharpe={res['sharpe']:+.6f}  "
            f"ret={res['annual_return']:+.6f}  "
            f"mdd={res['max_drawdown']:+.6f}"
        )

    out_path = Path("/tmp/combined_baseline.json")
    out_path.write_text(json.dumps(out, indent=2, sort_keys=True))
    print(f"\nBaseline written to {out_path}")


if __name__ == "__main__":
    main()
