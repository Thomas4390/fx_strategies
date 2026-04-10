"""Generate frozen stats snapshots from the legacy ``backtest_*`` API.

This script is run ONCE per strategy, BEFORE the legacy function is deleted.
Each phase of the refactor produces a handful of ``tests/snapshots/<strat>_<label>.json``
files that become the immutable baseline for ``test_pipeline_equivalence.py``.

Usage:
    python tests/_generate_snapshots.py --strat mr_turbo
    python tests/_generate_snapshots.py --strat daily_momentum
    python tests/_generate_snapshots.py --strat all

Each case is a dict with fields:
    strat, label, module, callable, params, data_loader (optional)

``data_loader`` is a key in ``_DATA_LOADERS`` below. Defaults to
``"eurusd_minute"``, which matches the single-pair FX case used by most
strategies. Phase 4 (daily_momentum) introduces custom loaders for the
multi-pair DataFrame and single-pair daily series.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable

_TESTS_DIR = Path(__file__).resolve().parent
_SRC_DIR = _TESTS_DIR.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

_SNAPSHOT_DIR = _TESTS_DIR / "snapshots"
_SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════
# DATA LOADERS
# ═══════════════════════════════════════════════════════════════════════


def _load_eurusd_minute() -> Any:
    from utils import load_fx_data

    _, data = load_fx_data("data/EUR-USD_minute.parquet")
    return data


def _load_multi_pair_daily_closes() -> Any:
    """Load 4-pair daily closes DataFrame (used by XS momentum)."""
    import pandas as pd

    from utils import load_fx_data

    pairs = ["EUR-USD", "GBP-USD", "USD-JPY", "USD-CAD"]
    closes = {}
    for pair in pairs:
        _, data = load_fx_data(f"data/{pair}_minute.parquet")
        closes[pair] = data.close.resample("1D").last().dropna()
    return pd.DataFrame(closes).dropna()


def _load_gbpusd_daily_close() -> Any:
    """Load GBP-USD single-pair daily close Series (used by TS momentum)."""
    from utils import load_fx_data

    _, data = load_fx_data("data/GBP-USD_minute.parquet")
    return data.close.resample("1D").last().dropna()


_DATA_LOADERS: dict[str, Callable[[], Any]] = {
    "eurusd_minute": _load_eurusd_minute,
    "multi_pair_daily_closes": _load_multi_pair_daily_closes,
    "gbpusd_daily_close": _load_gbpusd_daily_close,
}


# ═══════════════════════════════════════════════════════════════════════
# SNAPSHOT CASES
# ═══════════════════════════════════════════════════════════════════════


SNAPSHOT_CASES: list[dict[str, Any]] = [
    # ── Phase 1: mr_turbo ────────────────────────────────────────────
    dict(
        strat="mr_turbo", label="default",
        module="strategies.mr_turbo", callable="backtest_mr_turbo",
        params=dict(bb_window=80, bb_alpha=5.0, sl_stop=0.005, tp_stop=0.006),
    ),
    dict(
        strat="mr_turbo", label="tight_bands",
        module="strategies.mr_turbo", callable="backtest_mr_turbo",
        params=dict(bb_window=60, bb_alpha=4.0, sl_stop=0.004, tp_stop=0.006),
    ),
    dict(
        strat="mr_turbo", label="loose_bands",
        module="strategies.mr_turbo", callable="backtest_mr_turbo",
        params=dict(bb_window=120, bb_alpha=6.0, sl_stop=0.005, tp_stop=0.008),
    ),
    # ── Phase 2: mr_macro ───────────────────────────────────────────
    dict(
        strat="mr_macro", label="default",
        module="strategies.mr_macro", callable="backtest_mr_macro",
        params=dict(bb_window=80, bb_alpha=5.0, sl_stop=0.005, tp_stop=0.006, spread_threshold=0.5),
    ),
    dict(
        strat="mr_macro", label="strict_spread",
        module="strategies.mr_macro", callable="backtest_mr_macro",
        params=dict(bb_window=60, bb_alpha=5.0, sl_stop=0.005, tp_stop=0.006, spread_threshold=0.3),
    ),
    dict(
        strat="mr_macro", label="loose_bands",
        module="strategies.mr_macro", callable="backtest_mr_macro",
        params=dict(bb_window=120, bb_alpha=6.0, sl_stop=0.005, tp_stop=0.008, spread_threshold=0.5),
    ),
    # ── Phase 3: rsi_daily ───────────────────────────────────────────
    dict(
        strat="rsi_daily", label="default",
        module="strategies.rsi_daily", callable="backtest_rsi_daily",
        params=dict(rsi_period=14, oversold=25.0, overbought=75.0),
    ),
    dict(
        strat="rsi_daily", label="short_period",
        module="strategies.rsi_daily", callable="backtest_rsi_daily",
        params=dict(rsi_period=7, oversold=20.0, overbought=80.0),
    ),
    dict(
        strat="rsi_daily", label="wide_band",
        module="strategies.rsi_daily", callable="backtest_rsi_daily",
        params=dict(rsi_period=21, oversold=30.0, overbought=70.0),
    ),
    # ── Phase 4: daily_momentum ──────────────────────────────────────
    dict(
        strat="daily_xs", label="default",
        module="strategies.daily_momentum", callable="backtest_xs_momentum_pf",
        params=dict(w_short=21, w_long=63, target_vol=0.10),
        data_loader="multi_pair_daily_closes",
    ),
    dict(
        strat="daily_xs", label="short_long",
        module="strategies.daily_momentum", callable="backtest_xs_momentum_pf",
        params=dict(w_short=10, w_long=42, target_vol=0.08),
        data_loader="multi_pair_daily_closes",
    ),
    dict(
        strat="daily_ts", label="default",
        module="strategies.daily_momentum", callable="backtest_ts_momentum_pf",
        params=dict(fast_ema=20, slow_ema=50, rsi_period=7, rsi_low=40, rsi_high=60),
        data_loader="gbpusd_daily_close",
    ),
    dict(
        strat="daily_ts", label="slow_trend",
        module="strategies.daily_momentum", callable="backtest_ts_momentum_pf",
        params=dict(fast_ema=30, slow_ema=100, rsi_period=14, rsi_low=40, rsi_high=60),
        data_loader="gbpusd_daily_close",
    ),
    # ── Phase 5: ou_mean_reversion ───────────────────────────────────
    dict(
        strat="ou_mr", label="default",
        module="strategies.ou_mean_reversion", callable="backtest_ou_mr",
        params=dict(bb_window=80, bb_alpha=5.0, sigma_target=0.10, max_leverage=3.0),
    ),
    dict(
        strat="ou_mr", label="low_vol",
        module="strategies.ou_mean_reversion", callable="backtest_ou_mr",
        params=dict(bb_window=60, bb_alpha=5.0, sigma_target=0.05, max_leverage=5.0),
    ),
    dict(
        strat="ou_mr", label="high_vol",
        module="strategies.ou_mean_reversion", callable="backtest_ou_mr",
        params=dict(bb_window=120, bb_alpha=6.0, sigma_target=0.20, max_leverage=2.0),
    ),
    # ── Phase 6: composite_fx_alpha ──────────────────────────────────
]


# ═══════════════════════════════════════════════════════════════════════
# GENERATION LOGIC
# ═══════════════════════════════════════════════════════════════════════


def _stats_to_dict(stats: Any) -> dict[str, Any]:
    import numpy as np

    out: dict[str, Any] = {}
    for key, val in stats.items():
        if isinstance(val, (int, float, np.integer, np.floating)):
            out[str(key)] = None if np.isnan(val) else float(val)
        else:
            out[str(key)] = str(val)
    return out


def generate_case(case: dict[str, Any]) -> Path:
    import importlib

    from utils import apply_vbt_settings

    strat = case["strat"]
    label = case["label"]
    module = case["module"]
    fn_name = case["callable"]
    params = case["params"]
    loader_key = case.get("data_loader", "eurusd_minute")

    snapshot_path = _SNAPSHOT_DIR / f"{strat}_{label}.json"

    print(f"▶ {strat}/{label} via {module}.{fn_name}({params}) [loader={loader_key}]")

    apply_vbt_settings()
    loader = _DATA_LOADERS[loader_key]
    data = loader()

    mod = importlib.import_module(module)
    fn = getattr(mod, fn_name)
    pf = fn(data, **params)
    stats = pf.stats()

    payload = {
        "strat": strat,
        "label": label,
        "module": module,
        "callable": fn_name,
        "params": params,
        "data_loader": loader_key,
        "stats": _stats_to_dict(stats),
    }
    with snapshot_path.open("w") as fh:
        json.dump(payload, fh, indent=2, default=str)
    print(f"  ✔ wrote {snapshot_path.name}")
    return snapshot_path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--strat", default="all", help="Strategy key, or 'all'")
    args = ap.parse_args()

    if args.strat == "all":
        cases = SNAPSHOT_CASES
    else:
        cases = [c for c in SNAPSHOT_CASES if c["strat"] == args.strat]

    if not cases:
        print(f"No snapshot cases for strat={args.strat!r}")
        return 1

    for case in cases:
        try:
            generate_case(case)
        except Exception as e:
            print(f"  ✗ FAILED: {type(e).__name__}: {e}")
            return 2

    print(f"\n✔ Generated {len(cases)} snapshot(s) in {_SNAPSHOT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
