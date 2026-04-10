#!/usr/bin/env python
"""Run any registered strategy via the unified framework.

Usage:
    python run_strategy.py mr_v1                         # full pipeline (CV + holdout)
    python run_strategy.py mr_v1 --mode=backtest         # single run with defaults
    python run_strategy.py mr_v1 --mode=backtest --lookback=120 --band_width=2.5
    python run_strategy.py all                           # full pipeline for all strategies
"""

from __future__ import annotations

import argparse

from framework.runner import run_strategy
from strategies import REGISTRY
from utils import apply_vbt_settings


def main() -> None:
    parser = argparse.ArgumentParser(description="Run FX strategies")
    parser.add_argument(
        "strategy",
        choices=[*REGISTRY.keys(), "all"],
        help="Strategy name or 'all'",
    )
    parser.add_argument(
        "--mode",
        choices=["backtest", "full"],
        default="full",
        help="Execution mode (default: full)",
    )
    parser.add_argument(
        "--metric",
        default="sharpe_ratio",
        help="Optimization metric (default: sharpe_ratio)",
    )
    parser.add_argument(
        "--data",
        default="data/EUR-USD_minute.parquet",
        help="Path to data file",
    )
    parser.add_argument(
        "--holdout",
        type=float,
        default=0.2,
        help="Hold-out ratio (default: 0.2)",
    )

    args, extra = parser.parse_known_args()

    # Parse extra --key=value pairs as parameter overrides
    overrides: dict[str, float | int] = {}
    for item in extra:
        if item.startswith("--") and "=" in item:
            key, val = item[2:].split("=", 1)
            try:
                overrides[key] = int(val)
            except ValueError:
                overrides[key] = float(val)

    apply_vbt_settings()

    specs = (
        REGISTRY if args.strategy == "all" else {args.strategy: REGISTRY[args.strategy]}
    )

    for name, spec in specs.items():
        print(f"\n{'=' * 60}")
        print(f"  {spec.name}")
        print(f"{'=' * 60}")

        kwargs = {**overrides}
        if args.mode == "full":
            kwargs["metric"] = args.metric
            kwargs["holdout_ratio"] = args.holdout

        run_strategy(spec, data_path=args.data, mode=args.mode, **kwargs)


if __name__ == "__main__":
    main()
