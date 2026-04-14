"""AST-based audit of VBT Pro native adoption in src/strategies/.

Scans each .py file under a target directory and flags anti-patterns
where raw pandas / numpy is used in place of a native VBT Pro
primitive. Writes a CSV under ``reports/audits/vbt_native_audit.csv``
and prints a per-file summary to stdout.

The scan is AST-based (no regex) so it ignores strings, comments,
and conditional imports. Each finding carries:

- file, line, col
- pattern id
- severity (H / M / B)
- the source snippet
- the VBT-native replacement
- function context (enclosing def name) and decorator context

Usage
-----
    python scripts/audit_vbt_native.py [TARGET_DIR]

Default ``TARGET_DIR`` is ``src/strategies``. Exit code 0 always —
this is a reporting tool, not a linter that should fail CI.
"""

from __future__ import annotations

import ast
import csv
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TARGET = REPO_ROOT / "src" / "strategies"
REPORT_DIR = REPO_ROOT / "reports" / "audits"
REPORT_CSV = REPORT_DIR / "vbt_native_audit.csv"


# ─────────────────────────────────────────────────────────────────────────
# Pattern catalogue
# ─────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Pattern:
    pid: str
    severity: str          # "H", "M", "B"
    description: str
    replacement: str
    skip_in_njit: bool     # True if the pattern is legitimate inside @njit kernels


PATTERNS: dict[str, Pattern] = {
    "P01_manual_ternary_signal": Pattern(
        pid="P01_manual_ternary_signal",
        severity="M",
        description="pd.Series(0.0, index=...) followed by boolean-indexed assignments (manual ternary signal)",
        replacement="Pass entries/short_entries masks to Portfolio.from_signals directly",
        skip_in_njit=True,
    ),
    "P02_pct_change_fillna": Pattern(
        pid="P02_pct_change_fillna",
        severity="B",
        description=".pct_change().fillna(...) chain",
        replacement="Consider .vbt.pct_change() accessor or vbt.ReturnsAccessor",
        skip_in_njit=True,
    ),
    "P03_raw_rolling_agg": Pattern(
        pid="P03_raw_rolling_agg",
        severity="M",
        description="Raw .rolling(n).mean()/.std()/.sum() (not .vbt.rolling_*)",
        replacement=".vbt.rolling_mean/std/sum(window, minp=window)",
        skip_in_njit=True,
    ),
    "P04_np_log_ratio_shift": Pattern(
        pid="P04_np_log_ratio_shift",
        severity="B",
        description="np.log(x / x.shift(n)) outside a @njit kernel",
        replacement="np.log1p(x.vbt.pct_change(n)) or keep as-is if perf-critical",
        skip_in_njit=True,
    ),
    "P05_pd_read_direct": Pattern(
        pid="P05_pd_read_direct",
        severity="B",
        description="pd.read_parquet/read_csv instead of vbt.*Data loaders",
        replacement="vbt.ParquetData.pull(path) / vbt.CSVData.pull(path)",
        skip_in_njit=False,
    ),
    "P06_reindex_fillna": Pattern(
        pid="P06_reindex_fillna",
        severity="M",
        description=".reindex(...).fillna(...) chain for cross-frequency alignment",
        replacement="vbt.Resampler(src_idx, tgt_idx).resample(series)",
        skip_in_njit=True,
    ),
    "P07_shift_fillna_causal": Pattern(
        pid="P07_shift_fillna_causal",
        severity="B",
        description=".shift(n).fillna(v) causal-shift pattern",
        replacement=".vbt.fshift(n, fill_value=v)",
        skip_in_njit=True,
    ),
    "P08_np_broadcast_to": Pattern(
        pid="P08_np_broadcast_to",
        severity="M",
        description="np.broadcast_to(...) — VBT handles broadcasting implicitly",
        replacement="Pass 1D Series to VBT; let from_signals/from_orders broadcast",
        skip_in_njit=True,
    ),
    "P09_manual_resample_apply_last": Pattern(
        pid="P09_manual_resample_apply_last",
        severity="B",
        description=".resample('1D').last() raw pandas (not vbt.resample_apply)",
        replacement=".vbt.resample_apply('1D', 'last') or vbt.Resampler",
        skip_in_njit=True,
    ),
}


# ─────────────────────────────────────────────────────────────────────────
# Finding
# ─────────────────────────────────────────────────────────────────────────


@dataclass
class Finding:
    file: str
    line: int
    col: int
    pattern: Pattern
    snippet: str
    func_context: str
    decorator_context: str

    def as_row(self) -> dict[str, str]:
        return {
            "file": self.file,
            "line": str(self.line),
            "col": str(self.col),
            "pattern": self.pattern.pid,
            "severity": self.pattern.severity,
            "description": self.pattern.description,
            "suggested_replacement": self.pattern.replacement,
            "snippet": self.snippet,
            "func_context": self.func_context,
            "decorator_context": self.decorator_context,
        }


# ─────────────────────────────────────────────────────────────────────────
# AST helpers
# ─────────────────────────────────────────────────────────────────────────


def _attr_chain(node: ast.AST) -> list[str]:
    """Return the dotted attribute chain for an AST node.

    ``close.vbt.rolling_std`` -> ``['close', 'vbt', 'rolling_std']``.
    ``np.log``                -> ``['np', 'log']``.
    Returns ``[]`` if the chain cannot be resolved to plain names.
    """
    parts: list[str] = []
    cur = node
    while isinstance(cur, ast.Attribute):
        parts.append(cur.attr)
        cur = cur.value
    if isinstance(cur, ast.Name):
        parts.append(cur.id)
    else:
        return []
    return list(reversed(parts))


def _is_vbt_accessor_chain(chain: list[str]) -> bool:
    """True if the chain contains '.vbt.*' segment."""
    return "vbt" in chain[1:]  # skip first element (could be a name)


def _call_method_name(call: ast.Call) -> str:
    """Return the final method name of a Call node, or ''."""
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    if isinstance(call.func, ast.Name):
        return call.func.id
    return ""


def _snippet_from(source_lines: list[str], node: ast.AST) -> str:
    """Extract up to 2 source lines starting at node.lineno."""
    start = getattr(node, "lineno", 1) - 1
    end = getattr(node, "end_lineno", start + 1)
    window = source_lines[start : min(end, start + 2)]
    return " ".join(s.strip() for s in window)[:200]


# ─────────────────────────────────────────────────────────────────────────
# Visitor
# ─────────────────────────────────────────────────────────────────────────


class VBTAuditVisitor(ast.NodeVisitor):
    """Walk an AST and emit findings.

    Tracks the enclosing function (and its decorators) so patterns
    inside ``@njit`` kernels or ``@vbt.parameterized`` functions can
    be filtered when appropriate.
    """

    def __init__(self, file: str, source_lines: list[str]) -> None:
        self.file = file
        self.source_lines = source_lines
        self.findings: list[Finding] = []
        self._func_stack: list[tuple[str, str]] = []  # (name, decorators)
        # Per-file call counts for the summary table
        self.stats: dict[str, int] = {
            "vbt_calls": 0,
            "vbt_accessor_calls": 0,
            "pd_calls": 0,
            "np_calls": 0,
            "njit_funcs": 0,
            "parameterized_funcs": 0,
            "total_functions": 0,
        }

    # --- function tracking ------------------------------------------------

    def _enter_func(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        decos: list[str] = []
        for d in node.decorator_list:
            if isinstance(d, ast.Call):
                d = d.func
            chain = _attr_chain(d)
            if chain:
                decos.append(".".join(chain))
        deco_str = ",".join(decos)
        self._func_stack.append((node.name, deco_str))
        self.stats["total_functions"] += 1
        if any("njit" in d for d in decos):
            self.stats["njit_funcs"] += 1
        if any("parameterized" in d for d in decos):
            self.stats["parameterized_funcs"] += 1

    def _leave_func(self) -> None:
        self._func_stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._enter_func(node)
        self.generic_visit(node)
        self._leave_func()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._enter_func(node)
        self.generic_visit(node)
        self._leave_func()

    @property
    def in_njit(self) -> bool:
        return bool(self._func_stack) and "njit" in self._func_stack[-1][1]

    @property
    def func_name(self) -> str:
        return self._func_stack[-1][0] if self._func_stack else "<module>"

    @property
    def func_decorators(self) -> str:
        return self._func_stack[-1][1] if self._func_stack else ""

    # --- call-level stats -------------------------------------------------

    def visit_Call(self, node: ast.Call) -> None:
        chain = _attr_chain(node.func)
        if chain:
            if chain[0] == "vbt":
                self.stats["vbt_calls"] += 1
            elif chain[0] == "pd":
                self.stats["pd_calls"] += 1
            elif chain[0] == "np":
                self.stats["np_calls"] += 1
            if _is_vbt_accessor_chain(chain):
                self.stats["vbt_accessor_calls"] += 1

        # Pattern detectors on Call nodes
        self._detect_pct_change_fillna(node)
        self._detect_raw_rolling_agg(node)
        self._detect_np_log_ratio_shift(node)
        self._detect_pd_read_direct(node)
        self._detect_reindex_fillna(node)
        self._detect_shift_fillna(node)
        self._detect_np_broadcast_to(node)
        self._detect_pd_resample_last(node)

        self.generic_visit(node)

    # --- assignment-level detector: P01 -----------------------------------

    def visit_Assign(self, node: ast.Assign) -> None:
        self._detect_manual_ternary_signal(node)
        self.generic_visit(node)

    # ─── individual detectors ─────────────────────────────────────────────

    def _emit(self, pid: str, node: ast.AST) -> None:
        pat = PATTERNS[pid]
        if pat.skip_in_njit and self.in_njit:
            return
        self.findings.append(
            Finding(
                file=self.file,
                line=getattr(node, "lineno", 0),
                col=getattr(node, "col_offset", 0),
                pattern=pat,
                snippet=_snippet_from(self.source_lines, node),
                func_context=self.func_name,
                decorator_context=self.func_decorators,
            )
        )

    def _detect_manual_ternary_signal(self, node: ast.Assign) -> None:
        # Match: name = pd.Series(0.0 | 0 | np.nan, index=...)
        if not isinstance(node.value, ast.Call):
            return
        chain = _attr_chain(node.value.func)
        if chain != ["pd", "Series"]:
            return
        if not node.value.args:
            return
        first = node.value.args[0]
        if isinstance(first, ast.Constant) and first.value in (0, 0.0):
            self._emit("P01_manual_ternary_signal", node)

    def _detect_pct_change_fillna(self, node: ast.Call) -> None:
        # Match: <...>.pct_change(...).fillna(...)
        if _call_method_name(node) != "fillna":
            return
        if not isinstance(node.func, ast.Attribute):
            return
        inner = node.func.value
        if isinstance(inner, ast.Call) and _call_method_name(inner) == "pct_change":
            # Skip if the pct_change is .vbt.pct_change()
            inner_chain = _attr_chain(inner.func)
            if "vbt" in inner_chain:
                return
            self._emit("P02_pct_change_fillna", node)

    def _detect_raw_rolling_agg(self, node: ast.Call) -> None:
        # Match: <...>.rolling(...).mean()/.std()/.sum()
        method = _call_method_name(node)
        if method not in {"mean", "std", "sum", "var"}:
            return
        if not isinstance(node.func, ast.Attribute):
            return
        inner = node.func.value
        if not (isinstance(inner, ast.Call) and _call_method_name(inner) == "rolling"):
            return
        # Skip if rolling is on a .vbt accessor chain (unlikely but guard)
        inner_chain = _attr_chain(inner.func)
        if "vbt" in inner_chain:
            return
        self._emit("P03_raw_rolling_agg", node)

    def _detect_np_log_ratio_shift(self, node: ast.Call) -> None:
        # Match: np.log(x / x.shift(n))
        chain = _attr_chain(node.func)
        if chain != ["np", "log"] and chain != ["numpy", "log"]:
            return
        if len(node.args) != 1:
            return
        arg = node.args[0]
        if not (isinstance(arg, ast.BinOp) and isinstance(arg.op, ast.Div)):
            return
        right = arg.right
        if isinstance(right, ast.Call) and _call_method_name(right) == "shift":
            self._emit("P04_np_log_ratio_shift", node)

    def _detect_pd_read_direct(self, node: ast.Call) -> None:
        chain = _attr_chain(node.func)
        if len(chain) >= 2 and chain[0] == "pd" and chain[1].startswith("read_"):
            self._emit("P05_pd_read_direct", node)

    def _detect_reindex_fillna(self, node: ast.Call) -> None:
        # Match: <...>.reindex(index=..., ...).fillna(...)
        # Only fire on index-reindex (cross-frequency alignment), not
        # column-reindex (pure column alignment — handled by pandas or
        # vbt.broadcast, not by vbt.Resampler).
        if _call_method_name(node) != "fillna":
            return
        if not isinstance(node.func, ast.Attribute):
            return
        inner = node.func.value
        if not (isinstance(inner, ast.Call) and _call_method_name(inner) == "reindex"):
            return
        # Require an ``index=`` kwarg OR a positional first arg (which
        # is the index in pd.Series.reindex). A call with only
        # ``columns=`` is a false positive.
        has_index = bool(inner.args) or any(
            kw.arg == "index" for kw in inner.keywords
        )
        if has_index:
            self._emit("P06_reindex_fillna", node)

    def _detect_shift_fillna(self, node: ast.Call) -> None:
        # Match: <...>.shift(n).fillna(v) with n > 0 (causal shift).
        # A negative n is a forward lookup (PFO semantics), not a
        # look-ahead fix, so we do NOT flag it.
        if _call_method_name(node) != "fillna":
            return
        if not isinstance(node.func, ast.Attribute):
            return
        inner = node.func.value
        if not (isinstance(inner, ast.Call) and _call_method_name(inner) == "shift"):
            return
        inner_chain = _attr_chain(inner.func)
        if "vbt" in inner_chain:
            return
        # Extract the shift argument. Default shift(1) if no arg.
        # Handle UnaryOp for negative literals: shift(-1) parses as
        # Call(args=[UnaryOp(USub, Constant(1))]), not Constant(-1).
        def _literal_int(expr: ast.AST) -> int | None:
            if isinstance(expr, ast.Constant) and isinstance(expr.value, int):
                return expr.value
            if (
                isinstance(expr, ast.UnaryOp)
                and isinstance(expr.op, ast.USub)
                and isinstance(expr.operand, ast.Constant)
                and isinstance(expr.operand.value, int)
            ):
                return -expr.operand.value
            return None

        shift_n: int | None = None
        if inner.args:
            shift_n = _literal_int(inner.args[0])
        elif not inner.keywords:
            shift_n = 1  # default
        else:
            for kw in inner.keywords:
                if kw.arg == "periods":
                    shift_n = _literal_int(kw.value)
        if shift_n is None or shift_n > 0:
            self._emit("P07_shift_fillna_causal", node)

    def _detect_np_broadcast_to(self, node: ast.Call) -> None:
        chain = _attr_chain(node.func)
        if chain == ["np", "broadcast_to"] or chain == ["numpy", "broadcast_to"]:
            self._emit("P08_np_broadcast_to", node)

    def _detect_pd_resample_last(self, node: ast.Call) -> None:
        # Match: <...>.resample(...).last()
        if _call_method_name(node) != "last":
            return
        if not isinstance(node.func, ast.Attribute):
            return
        inner = node.func.value
        if isinstance(inner, ast.Call) and _call_method_name(inner) == "resample":
            inner_chain = _attr_chain(inner.func)
            if "vbt" in inner_chain:
                return
            self._emit("P09_manual_resample_apply_last", node)


# ─────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────


def audit_file(path: Path) -> tuple[list[Finding], dict[str, int]]:
    source = path.read_text()
    tree = ast.parse(source, filename=str(path))
    visitor = VBTAuditVisitor(
        file=str(path.relative_to(REPO_ROOT)),
        source_lines=source.splitlines(),
    )
    visitor.visit(tree)
    return visitor.findings, visitor.stats


def iter_py_files(target: Path) -> Iterable[Path]:
    for p in sorted(target.rglob("*.py")):
        if "__pycache__" in p.parts:
            continue
        if p.name == "__init__.py":
            continue
        yield p


def write_csv(findings: list[Finding]) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "file", "line", "col", "pattern", "severity",
        "description", "suggested_replacement", "snippet",
        "func_context", "decorator_context",
    ]
    with REPORT_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for finding in findings:
            writer.writerow(finding.as_row())


def print_summary(
    per_file: dict[str, tuple[list[Finding], dict[str, int]]],
) -> None:
    print()
    print("═" * 88)
    print("VBT NATIVE ADOPTION AUDIT — per-file summary")
    print("═" * 88)
    header = (
        f"{'file':<42} {'vbt':>5} {'vbt.*':>6} {'pd':>4} "
        f"{'np':>4} {'njit':>5} {'H':>3} {'M':>3} {'B':>3}"
    )
    print(header)
    print("─" * 88)
    total_h = total_m = total_b = 0
    for file, (findings, stats) in sorted(per_file.items()):
        h = sum(1 for f in findings if f.pattern.severity == "H")
        m = sum(1 for f in findings if f.pattern.severity == "M")
        b = sum(1 for f in findings if f.pattern.severity == "B")
        total_h += h
        total_m += m
        total_b += b
        print(
            f"{file:<42} "
            f"{stats['vbt_calls']:>5} {stats['vbt_accessor_calls']:>6} "
            f"{stats['pd_calls']:>4} {stats['np_calls']:>4} "
            f"{stats['njit_funcs']:>5} "
            f"{h:>3} {m:>3} {b:>3}"
        )
    print("─" * 88)
    print(
        f"{'TOTAL':<42} {'':>5} {'':>6} {'':>4} {'':>4} {'':>5} "
        f"{total_h:>3} {total_m:>3} {total_b:>3}"
    )
    print()
    print(f"CSV written to: {REPORT_CSV.relative_to(REPO_ROOT)}")
    print(f"Total findings: {total_h + total_m + total_b}")
    print(f"  HIGH:   {total_h}")
    print(f"  MEDIUM: {total_m}")
    print(f"  LOW:    {total_b}")


def main() -> int:
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_TARGET
    if not target.is_absolute():
        target = REPO_ROOT / target
    if not target.exists():
        print(f"ERROR: target does not exist: {target}", file=sys.stderr)
        return 1

    per_file: dict[str, tuple[list[Finding], dict[str, int]]] = {}
    all_findings: list[Finding] = []

    for py in iter_py_files(target):
        findings, stats = audit_file(py)
        rel = str(py.relative_to(REPO_ROOT))
        per_file[rel] = (findings, stats)
        all_findings.extend(findings)

    write_csv(all_findings)
    print_summary(per_file)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
