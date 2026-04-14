"""Project-wide configuration for the FX strategies framework.

Two layers:

1. ``vbt.yml`` at the project root — auto-applied to ``vbt.settings`` on
   import of this module. Holds portfolio defaults (``init_cash``, ``fees``,
   ``slippage``, ``leverage``) so every ``vbt.Portfolio.from_signals`` call
   that receives ``None`` falls back to the same source of truth.

2. ``PROJECT_CONFIG`` below — a ``vbt.ReadonlyConfig`` for everything that
   ``vbt.settings`` does not cover (paths, pair registry, CV folds, regime
   slippage overrides). Import it from strategies and tests:

       from framework.project_config import PROJECT_CONFIG, data_path, results_dir
"""

from __future__ import annotations

from pathlib import Path

import vectorbtpro as vbt

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

_VBT_SETTINGS_FILE: Path = PROJECT_ROOT / "vbt.yml"
if _VBT_SETTINGS_FILE.exists():
    vbt.settings.load_update(str(_VBT_SETTINGS_FILE))


PROJECT_CONFIG = vbt.ReadonlyConfig(
    data_dir=PROJECT_ROOT / "data",
    results_dir=PROJECT_ROOT / "results",
    default_pair="EUR-USD",
    fx_pairs=("EUR-USD", "GBP-USD", "USD-JPY", "USD-CAD"),
    cv_folds=10,
    show_charts=True,
    session_start=6,
    session_end=14,
    # Intraday regime slippage (higher friction than vbt.yml default) — used as
    # pipeline() signature default for mr_turbo, mr_macro, ou_mean_reversion.
    slippage_intraday=0.00015,
)


def data_path(pair: str) -> Path:
    return PROJECT_CONFIG["data_dir"] / f"{pair}_minute.parquet"


def results_dir(strategy_name: str, pair: str | None = None) -> Path:
    suffix = f"_{pair.lower()}" if pair else ""
    return PROJECT_CONFIG["results_dir"] / f"{strategy_name}{suffix}"
