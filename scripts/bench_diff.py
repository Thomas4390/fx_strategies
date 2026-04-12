"""Compare two bench_ram.py JSON outputs side-by-side.

Usage:

    python scripts/bench_diff.py /tmp/bench_<sha>_main.json \
                                 /tmp/bench_<sha>_perf-chunking-ram.json

Prints a table with ΔRAM and ΔTime per workflow.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def _load(path: str) -> dict:
    return json.loads(Path(path).read_text())


def _pct(base: float, new: float) -> str:
    if base == 0:
        return "—"
    return f"{(new - base) / base * 100:+.0f}%"


def main() -> None:
    if len(sys.argv) != 3:
        print("usage: bench_diff.py <baseline.json> <candidate.json>", file=sys.stderr)
        sys.exit(1)

    base = _load(sys.argv[1])
    cand = _load(sys.argv[2])
    base_by_name = {r["name"]: r for r in base["results"]}
    cand_by_name = {r["name"]: r for r in cand["results"]}

    rows: list[tuple[str, ...]] = []
    for name in base_by_name:
        b = base_by_name.get(name)
        c = cand_by_name.get(name)
        if b is None or c is None or "error" in b or "error" in c:
            rows.append((name, "—", "—", "—", "—", "—", "—"))
            continue
        rows.append(
            (
                name,
                f"{b['peak_ram_gb']:.2f} GB",
                f"{c['peak_ram_gb']:.2f} GB",
                _pct(b["peak_ram_gb"], c["peak_ram_gb"]),
                f"{b['elapsed_s']:.1f}s",
                f"{c['elapsed_s']:.1f}s",
                _pct(b["elapsed_s"], c["elapsed_s"]),
            )
        )

    headers = ("Workflow", "Base RAM", "Cand RAM", "ΔRAM", "Base Time", "Cand Time", "ΔTime")
    widths = [
        max(len(h), max((len(r[i]) for r in rows), default=0)) for i, h in enumerate(headers)
    ]

    def fmt(row: tuple[str, ...]) -> str:
        return "  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row))

    sep = "  ".join("-" * w for w in widths)

    print()
    print(f"Baseline  : branch={base['branch']} sha={base['sha']}")
    print(f"Candidate : branch={cand['branch']} sha={cand['sha']}")
    print()
    print(fmt(headers))
    print(sep)
    for row in rows:
        print(fmt(row))
    print()


if __name__ == "__main__":
    main()
