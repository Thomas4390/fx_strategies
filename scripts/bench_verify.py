"""Numerical correctness canary — diff grid outputs across branches.

Runs three reduced grids (mr_macro, composite_fx_alpha, daily_momentum
xs) and writes every (param_combo, metric_value) pair to a JSON keyed
by branch + SHA. The companion script ``bench_verify_diff.py`` (or an
inline ``jq`` pipe) compares two such files and flags any difference
above ``1e-12``.

WHEN TO RUN THIS SCRIPT:

- Before merging any change to ``make_execute_kwargs`` /
  ``DEFAULT_EXECUTE_KWARGS`` in ``framework/pipeline_utils.py``.
- Before tuning ``chunk_len``, ``engine``, ``flush_every``,
  ``merge_func``, or any kwarg passed to ``@vbt.parameterized`` /
  ``@vbt.cv_split``.
- Before upgrading ``vectorbtpro`` to a new version.
- When debugging a "sharpe changed but I didn't touch anything" bug.

Usage:

    # On branch A
    PYTHONPATH=src python scripts/bench_verify.py
    # → writes /tmp/verify_<sha>_<branchA>.json

    # On branch B
    PYTHONPATH=src python scripts/bench_verify.py
    # → writes /tmp/verify_<sha>_<branchB>.json

    # Diff them (one-liner, no extra script needed):
    python -c "
    import json, sys
    a = json.load(open(sys.argv[1]))
    b = json.load(open(sys.argv[2]))
    for s in a['suites']:
        A = {tuple(r['key']): r['value'] for r in a['suites'][s]}
        B = {tuple(r['key']): r['value'] for r in b['suites'][s]}
        mismatches = [(k, A[k], B[k]) for k in A if abs((A[k] or 0) - (B[k] or 0)) > 1e-12]
        print(f'{s}: {len(mismatches)} mismatches' if mismatches else f'{s}: OK ({len(A)} combos)')
    " /tmp/verify_<sha_a>_<a>.json /tmp/verify_<sha_b>_<b>.json

The reduced grids (3 × 3 × 2 = 18 combos for mr_macro, 3 × 3 × 1 = 9
for the other two) are chosen so the whole run completes in under a
minute — small enough to run manually during code review, large enough
to catch real numerical regressions across all three grid shapes.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True).strip()


def _series_to_records(s) -> list[dict]:
    """Convert a grid ``pd.Series`` to a JSON-friendly list of records.

    ``value`` is coerced to float, with NaN encoded as ``null`` so the
    JSON diff can distinguish "not computed" from "computed as NaN".
    """
    records: list[dict] = []
    for idx, val in s.items():
        if not isinstance(idx, tuple):
            idx = (idx,)
        try:
            fval = float(val)
            if fval != fval:  # NaN
                fval = None
        except (TypeError, ValueError):
            fval = None
        records.append(
            {"key": [str(x) for x in idx], "value": fval}
        )
    return records


def _run_mr_macro_grid() -> list[dict]:
    from strategies.mr_macro import run_grid
    from utils import load_fx_data

    _, data = load_fx_data("data/EUR-USD_minute.parquet")
    result = run_grid(
        data,
        bb_window=[40, 60, 80],
        bb_alpha=[4.0, 5.0, 6.0],
        sl_stop=0.005,
        tp_stop=0.006,
        spread_threshold=[0.3, 0.5],
    )
    return _series_to_records(result)


def _run_composite_grid() -> list[dict]:
    from strategies.composite_fx_alpha import run_grid
    from utils import load_fx_data

    _, data = load_fx_data("data/EUR-USD_minute.parquet")
    result = run_grid(
        data,
        w_short=[10, 21, 42],
        w_long=[42, 63, 126],
        target_vol=[0.10],
    )
    return _series_to_records(result)


def _run_daily_momentum_xs() -> list[dict]:
    from strategies.daily_momentum import load_daily_closes, run_grid_xs

    closes = load_daily_closes()
    result = run_grid_xs(
        closes,
        w_short=[10, 21, 42],
        w_long=[42, 63, 126],
        target_vol=[0.10],
    )
    return _series_to_records(result)


def main() -> None:
    branch = _git("rev-parse", "--abbrev-ref", "HEAD")
    sha = _git("rev-parse", "--short", "HEAD")
    print(f"\nResult-verification — branch={branch} sha={sha}")

    suites = {
        "mr_macro_grid_18": _run_mr_macro_grid,
        "composite_fx_alpha_grid_9": _run_composite_grid,
        "daily_momentum_xs_grid_9": _run_daily_momentum_xs,
    }

    out: dict = {"branch": branch, "sha": sha, "suites": {}}
    for name, fn in suites.items():
        print(f"  running {name}...")
        out["suites"][name] = fn()

    safe_branch = branch.replace("/", "-")
    out_path = Path(f"/tmp/verify_{sha}_{safe_branch}.json")
    out_path.write_text(json.dumps(out, indent=2))
    print(f"✔ wrote {out_path}")


if __name__ == "__main__":
    main()
