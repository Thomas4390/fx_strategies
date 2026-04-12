"""Benchmark harness for RAM / speed comparison between main and perf/chunking-ram.

Runs each hotspot (mr_macro, composite_fx_alpha, daily_momentum XS, and
the mr_macro walk-forward CV) twice, once with the ``small`` production
grid size and once with a deliberately larger ``large`` sweep so we can
verify how the chunking fix scales with grid cardinality.

Measures peak RAM via ``vbt.MemTracer`` and wall time via ``time.perf_counter``.
Writes a JSON file under /tmp named after the current branch + SHA so
the two runs (main vs chunked branch) can be compared by bench_diff.py.

Run from the repo root:

    PYTHONPATH=src python scripts/bench_ram.py

Approximate runtimes (on a 32-core box):
    small+large pass on main            : ~6 min
    small+large pass on perf/chunking-ram: ~7 min
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

import vectorbtpro as vbt  # noqa: E402


# ═══════════════════════════════════════════════════════════════════════
# Grid configurations — small = production sweep, large = stress test
# ═══════════════════════════════════════════════════════════════════════


MR_MACRO_GRID_SMALL = dict(
    bb_window=[40, 60, 80, 120],
    bb_alpha=[3.5, 4.0, 4.5, 5.0, 5.5, 6.0],
    sl_stop=0.005,
    tp_stop=0.006,
    spread_threshold=[0.3, 0.5, 0.7],
)  # 4 × 6 × 3 = 72 combos

MR_MACRO_GRID_LARGE = dict(
    bb_window=[30, 40, 60, 80, 100, 120],
    bb_alpha=[3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5],
    sl_stop=0.005,
    tp_stop=0.006,
    spread_threshold=[0.2, 0.3, 0.5, 0.7],
)  # 6 × 8 × 4 = 192 combos (2.67× the small grid)

COMPOSITE_GRID_SMALL = dict(
    w_short=[10, 21, 42],
    w_long=[42, 63, 126],
    target_vol=[0.05, 0.08, 0.10, 0.15],
)  # 3 × 3 × 4 = 36 combos

COMPOSITE_GRID_LARGE = dict(
    w_short=[5, 10, 15, 21, 30],
    w_long=[42, 63, 84, 105, 126],
    target_vol=[0.05, 0.08, 0.10, 0.12, 0.15],
)  # 5 × 5 × 5 = 125 combos

DAILY_MOM_GRID_SMALL = dict(
    w_short=[10, 21, 42],
    w_long=[42, 63, 126],
    target_vol=[0.08, 0.10, 0.12],
)  # 3 × 3 × 3 = 27 combos

DAILY_MOM_GRID_LARGE = dict(
    w_short=[5, 10, 15, 21, 42],
    w_long=[42, 63, 84, 105, 126],
    target_vol=[0.06, 0.08, 0.10, 0.12, 0.15],
)  # 5 × 5 × 5 = 125 combos

MR_MACRO_CV_SMALL = dict(
    bb_window=[60, 80, 120],
    bb_alpha=[4.0, 5.0, 6.0],
    spread_threshold=[0.3, 0.5],
)  # 3 × 3 × 2 = 18 combos × ~12 splits ≈ 216 runs

MR_MACRO_CV_LARGE = dict(
    bb_window=[40, 60, 80, 120],
    bb_alpha=[3.5, 4.5, 5.5, 6.0],
    spread_threshold=[0.3, 0.5],
)  # 4 × 4 × 2 = 32 combos × ~12 splits ≈ 384 runs


# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════


def _git(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=REPO_ROOT, text=True
    ).strip()


def _measure(name: str, fn) -> dict:
    """Run ``fn`` once, capturing peak RAM and wall time."""
    print(f"\n▶ {name}")
    vbt.flush()
    with vbt.MemTracer() as tracer:
        t0 = time.perf_counter()
        result = fn()
        elapsed_s = time.perf_counter() - t0
    peak_bytes = tracer.peak_usage(readable=False)
    n = None
    if hasattr(result, "__len__"):
        try:
            n = int(len(result))
        except Exception:
            n = None
    peak_gb = peak_bytes / 1e9
    print(f"  peak = {peak_gb:7.3f} GB   elapsed = {elapsed_s:7.2f} s   n = {n}")
    # Free the result before returning so it does not contaminate the
    # next measurement's baseline.
    del result
    vbt.flush()
    return {
        "name": name,
        "peak_ram_gb": round(peak_gb, 4),
        "elapsed_s": round(elapsed_s, 3),
        "n_combos": n,
    }


# ═══════════════════════════════════════════════════════════════════════
# Bench suites
# ═══════════════════════════════════════════════════════════════════════


def _bench_mr_macro_grid() -> list[dict]:
    from strategies.mr_macro import run_grid
    from utils import load_fx_data

    _, data = load_fx_data("data/EUR-USD_minute.parquet")
    return [
        _measure(
            "mr_macro grid SMALL (72 combos)",
            lambda: run_grid(data, **MR_MACRO_GRID_SMALL),
        ),
        _measure(
            "mr_macro grid LARGE (192 combos)",
            lambda: run_grid(data, **MR_MACRO_GRID_LARGE),
        ),
    ]


def _bench_composite_fx_alpha_grid() -> list[dict]:
    from strategies.composite_fx_alpha import run_grid
    from utils import load_fx_data

    _, data = load_fx_data("data/EUR-USD_minute.parquet")
    return [
        _measure(
            "composite_fx_alpha grid SMALL (36 combos)",
            lambda: run_grid(data, **COMPOSITE_GRID_SMALL),
        ),
        _measure(
            "composite_fx_alpha grid LARGE (125 combos)",
            lambda: run_grid(data, **COMPOSITE_GRID_LARGE),
        ),
    ]


def _bench_daily_momentum_xs_grid() -> list[dict]:
    from strategies.daily_momentum import load_daily_closes, run_grid_xs

    closes = load_daily_closes()
    return [
        _measure(
            "daily_momentum xs grid SMALL (27 combos)",
            lambda: run_grid_xs(closes, **DAILY_MOM_GRID_SMALL),
        ),
        _measure(
            "daily_momentum xs grid LARGE (125 combos)",
            lambda: run_grid_xs(closes, **DAILY_MOM_GRID_LARGE),
        ),
    ]


def _bench_mr_macro_cv() -> list[dict]:
    """Walk-forward CV sweeps mirroring the strategy __main__ config."""
    from framework.pipeline_utils import SHARPE_RATIO
    from strategies.mr_macro import create_cv_pipeline
    from utils import load_fx_data

    _, data = load_fx_data("data/EUR-USD_minute.parquet")
    daily_index = data.close.vbt.resample_apply("1D", "last").dropna().index
    splitter = vbt.Splitter.from_purged_walkforward(
        daily_index,
        n_folds=15,
        n_test_folds=1,
        purge_td="1 day",
        min_train_folds=3,
    )
    cv_pipeline = create_cv_pipeline(splitter, metric_type=SHARPE_RATIO)

    def _run_with(cv_params: dict) -> callable:
        return lambda: cv_pipeline(
            data,
            bb_window=vbt.Param(cv_params["bb_window"]),
            bb_alpha=vbt.Param(cv_params["bb_alpha"]),
            sl_stop=0.005,
            tp_stop=0.006,
            spread_threshold=vbt.Param(cv_params["spread_threshold"]),
        )

    return [
        _measure(
            "mr_macro CV SMALL (18 combos × ~12 splits)",
            _run_with(MR_MACRO_CV_SMALL),
        ),
        _measure(
            "mr_macro CV LARGE (32 combos × ~12 splits)",
            _run_with(MR_MACRO_CV_LARGE),
        ),
    ]


def main() -> None:
    branch = _git("rev-parse", "--abbrev-ref", "HEAD")
    sha = _git("rev-parse", "--short", "HEAD")
    print(f"\nBench harness — branch={branch} sha={sha}")
    print("=" * 78)

    results: list[dict] = []
    for bench_fn in (
        _bench_mr_macro_grid,
        _bench_composite_fx_alpha_grid,
        _bench_daily_momentum_xs_grid,
        _bench_mr_macro_cv,
    ):
        try:
            results.extend(bench_fn())
        except Exception as e:
            print(f"  ✗ {bench_fn.__name__} failed: {type(e).__name__}: {e}")
            results.append(
                {"name": bench_fn.__name__, "error": f"{type(e).__name__}: {e}"}
            )

    out = {
        "branch": branch,
        "sha": sha,
        "results": results,
    }
    safe_branch = branch.replace("/", "-")
    out_path = Path(f"/tmp/bench_{sha}_{safe_branch}.json")
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\n✔ wrote {out_path}")


if __name__ == "__main__":
    main()
