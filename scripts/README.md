# scripts/

Out-of-tree investigation tools — not imported by any production code
under `src/`, not run in CI. They live here so that engineers hacking on
the VBT Pro chunking config or adding a new strategy can quickly answer
"did I break RAM?" or "did I break numerical output?" without rebuilding
the harness from scratch.

## Tools

### `bench_ram.py` — RAM + speed profiler

Runs four hotspots (mr_macro grid, composite_fx_alpha grid, daily_momentum
xs grid, mr_macro walk-forward CV) twice each — once on a production-sized
grid, once on a deliberately larger sweep — and captures peak RAM via
`vbt.MemTracer` and wall time via `time.perf_counter`. Writes a JSON under
`/tmp/bench_<sha>_<branch>.json`.

```bash
PYTHONPATH=src python scripts/bench_ram.py
```

When to run it:
- Before merging a change to `make_execute_kwargs` / `DEFAULT_EXECUTE_KWARGS`.
- When tuning `chunk_len`, `flush_every`, `engine`, `chunked=` kwargs.
- When onboarding a new minute-frequency strategy that might saturate RAM.
- After upgrading `vectorbtpro` to a new version.

### `bench_diff.py` — side-by-side comparison table

Reads two `bench_ram.py` JSON outputs and prints a table with ΔRAM and
ΔTime per workflow. Used for branch-vs-branch comparisons.

```bash
python scripts/bench_diff.py /tmp/bench_<sha_a>_main.json \
                             /tmp/bench_<sha_b>_perf-chunking-ram.json
```

### `bench_verify.py` — numerical correctness canary

Runs three reduced grids (18 + 9 + 9 = 36 combos total) and writes every
`(param_combo, metric_value)` pair to JSON. Comparing two such files
catches any silent numerical regression introduced by a chunking /
execution-engine change — the guarantee is that the grid sweep values
should be **bit-identical** regardless of `chunk_len`, `engine`, or
`flush_every`.

```bash
# On branch A
PYTHONPATH=src python scripts/bench_verify.py
# On branch B
PYTHONPATH=src python scripts/bench_verify.py

# Diff (no extra script — inline jq-style)
python -c "
import json, sys
a = json.load(open(sys.argv[1]))
b = json.load(open(sys.argv[2]))
for s in a['suites']:
    A = {tuple(r['key']): r['value'] for r in a['suites'][s]}
    B = {tuple(r['key']): r['value'] for r in b['suites'][s]}
    diff = [(k, A[k], B[k]) for k in A if abs((A[k] or 0) - (B[k] or 0)) > 1e-12]
    print(f'{s}: {len(diff)} mismatches' if diff else f'{s}: OK ({len(A)} combos)')
" /tmp/verify_<sha_a>_<a>.json /tmp/verify_<sha_b>_<b>.json
```

When to run it:
- Before merging any change to `make_execute_kwargs` / chunking config.
- Before upgrading `vectorbtpro`.
- When debugging a "sharpe silently changed" bug.

## Not committed intentionally

- `/tmp/bench_*.json` — transient measurement artifacts. Keep them in
  `/tmp` so they are wiped at reboot and never polluted git diffs.
- `/tmp/verify_*.json` — same rationale.

## Relationship to `src/framework/pipeline_utils.py`

The chunking settings these scripts validate live in
`src/framework/pipeline_utils.py`:

- `DEFAULT_EXECUTE_KWARGS["chunk_len"] = 8` — bounded default parallel
  batch; caps peak RAM to ≈ `8 × single_portfolio_size` regardless of
  grid cardinality.
- `make_execute_kwargs(..., flush_every=1)` — opt-in hook that runs
  `vbt.flush()` (gc.collect + VBT cache clear) after every chunk. Used
  only on strategies whose single-portfolio footprint is large enough
  to need explicit GC (currently: `mr_macro`).

If you change either of those, run `bench_ram.py` and `bench_verify.py`
on your branch + main, then `bench_diff.py` to confirm the trade-off.
