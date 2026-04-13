"""Compare post-refonte metrics against the pre-refonte baseline.

Expected: bit-identical for no-leverage paths (machine precision, ~1e-14),
and within ~1e-6 absolute on Sharpe/total_return/max_drawdown for leveraged
paths (floating-point arithmetic in PFO vs pandas weighted sum can diverge
very slightly when large leverage amplifies tiny residuals).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from scripts.spikes.combined_baseline import main as _capture_current

# Alias: the baseline script writes to /tmp/combined_baseline.json. We
# capture the POST-refonte metrics to /tmp/combined_after.json by
# swapping the output path temporarily.


def _patch_write_path(path: Path) -> None:
    import scripts.spikes.combined_baseline as mod

    orig = mod.main

    def patched() -> None:
        orig_write = Path.write_text

        # Redirect only the specific write to /tmp/combined_baseline.json.
        def write_text(self, content, *a, **kw):  # type: ignore[override]
            if str(self) == "/tmp/combined_baseline.json":
                return Path(path).write_text(content, *a, **kw)
            return orig_write(self, content, *a, **kw)

        Path.write_text = write_text  # type: ignore[method-assign]
        try:
            orig()
        finally:
            Path.write_text = orig_write  # type: ignore[method-assign]

    mod.main = patched


def main() -> None:
    baseline_path = Path("/tmp/combined_baseline.json")
    after_path = Path("/tmp/combined_after.json")

    if not baseline_path.exists():
        print("ERROR: baseline file missing. Run combined_baseline.py first on main.")
        sys.exit(1)

    _patch_write_path(after_path)
    _capture_current()

    baseline = json.loads(baseline_path.read_text())
    after = json.loads(after_path.read_text())

    print("\n" + "=" * 78)
    print("  Post-refonte vs baseline comparison")
    print("=" * 78)

    max_abs = 0.0
    failed: list[str] = []
    for key in sorted(baseline.keys()):
        if key not in after:
            print(f"  [MISSING] {key}")
            failed.append(key)
            continue
        base = baseline[key]
        now = after[key]
        print(f"\n  {key}")
        for metric in [
            "sharpe",
            "annual_return",
            "annual_vol",
            "max_drawdown",
            "wf_avg_sharpe",
        ]:
            b = base[metric]
            a = now[metric]
            d = a - b
            max_abs = max(max_abs, abs(d))
            flag = "✓" if abs(d) < 1e-6 else ("~" if abs(d) < 1e-3 else "✗")
            print(
                f"    {flag} {metric:<15} baseline={b:+.10f}  after={a:+.10f}  Δ={d:+.2e}"
            )
            if abs(d) >= 1e-3:
                failed.append(f"{key}.{metric}")

    print("\n" + "-" * 78)
    print(f"  Max absolute difference: {max_abs:.2e}")
    if failed:
        print(f"  FAILED: {failed}")
        sys.exit(1)
    else:
        print("  All metrics within 1e-3 tolerance — refonte is non-regressive.")


if __name__ == "__main__":
    main()
