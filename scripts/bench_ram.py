"""Benchmark harness for RAM / speed comparison between main and perf/chunking-ram.

Runs the three biggest grid hotspots (mr_macro, composite_fx_alpha,
daily_momentum XS) and measures peak RAM via vbt.MemTracer + elapsed
time via vbt.Timer. Writes a JSON file under /tmp named after the
current branch + SHA so the two runs (main vs chunked branch) can be
compared by bench_diff.py.

Run from the repo root:

    PYTHONPATH=src python scripts/bench_ram.py

CV sweeps are deliberately skipped — they explode to minutes even on
the reduced grids, and the grid-only comparison already shows whether
chunking does its job.
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


def _git(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=REPO_ROOT, text=True
    ).strip()


def _measure(name: str, fn) -> dict:
    """Run ``fn`` once, capturing peak RAM and wall time."""
    print(f"\n▶ {name}")
    # Warmup GC so tracer baseline is clean.
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
    return {
        "name": name,
        "peak_ram_gb": round(peak_gb, 4),
        "elapsed_s": round(elapsed_s, 3),
        "n_combos": n,
    }


def _bench_mr_macro() -> list[dict]:
    from strategies.mr_macro import run_grid
    from utils import load_fx_data

    _, data = load_fx_data("data/EUR-USD_minute.parquet")
    grid_params = dict(
        bb_window=[40, 60, 80, 120],
        bb_alpha=[3.5, 4.0, 4.5, 5.0, 5.5, 6.0],
        sl_stop=0.005,
        tp_stop=0.006,
        spread_threshold=[0.3, 0.5, 0.7],
    )
    return [_measure("mr_macro grid (72 combos)", lambda: run_grid(data, **grid_params))]


def _bench_composite_fx_alpha() -> list[dict]:
    from strategies.composite_fx_alpha import run_grid
    from utils import load_fx_data

    _, data = load_fx_data("data/EUR-USD_minute.parquet")
    grid_params = dict(
        w_short=[10, 21, 42],
        w_long=[42, 63, 126],
        target_vol=[0.05, 0.08, 0.10, 0.15],
    )
    return [
        _measure(
            "composite_fx_alpha grid (36 combos)",
            lambda: run_grid(data, **grid_params),
        )
    ]


def _bench_daily_momentum_xs() -> list[dict]:
    from strategies.daily_momentum import load_daily_closes, run_grid_xs

    closes = load_daily_closes()
    grid_params = dict(
        w_short=[10, 21, 42],
        w_long=[42, 63, 126],
        target_vol=[0.08, 0.10, 0.12],
    )
    return [
        _measure(
            "daily_momentum xs grid (27 combos)",
            lambda: run_grid_xs(closes, **grid_params),
        )
    ]


def _bench_mr_macro_cv() -> list[dict]:
    """Full walk-forward CV sweep mirroring the strategy __main__.

    3 × 3 × 2 = 18 combos × 15 folds × 2 sets (train+test) ≈ 540
    portfolio simulations. This is the real RAM stress-test: the grid
    sweep runs inside each CV split, so the peak is the per-split batch
    held in memory at once.
    """
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

    def _run():
        return cv_pipeline(
            data,
            bb_window=vbt.Param([60, 80, 120]),
            bb_alpha=vbt.Param([4.0, 5.0, 6.0]),
            sl_stop=0.005,
            tp_stop=0.006,
            spread_threshold=vbt.Param([0.3, 0.5]),
        )

    return [_measure("mr_macro CV (18 combos × ~12 splits)", _run)]


def main() -> None:
    branch = _git("rev-parse", "--abbrev-ref", "HEAD")
    sha = _git("rev-parse", "--short", "HEAD")
    print(f"\nBench harness — branch={branch} sha={sha}")
    print("=" * 78)

    results: list[dict] = []
    for bench_fn in (
        _bench_mr_macro,
        _bench_composite_fx_alpha,
        _bench_daily_momentum_xs,
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
