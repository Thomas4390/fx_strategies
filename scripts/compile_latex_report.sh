#!/usr/bin/env bash
# Compile the Phase 18 LaTeX report with xelatex (two passes for ToC/refs).
set -euo pipefail

REPORT_DIR="$(cd "$(dirname "$0")/.." && pwd)/reports/latex_report"
cd "$REPORT_DIR"

echo "═══════════════════════════════════════════════════════════════"
echo "  LaTeX report compilation"
echo "═══════════════════════════════════════════════════════════════"
echo "Working dir: $REPORT_DIR"
echo ""

echo "[1/2] First XeLaTeX pass..."
xelatex -interaction=nonstopmode -halt-on-error main.tex > compile.log 2>&1 || {
    echo "✗ First pass failed. Tail of compile.log:"
    tail -40 compile.log
    exit 1
}

echo "[2/2] Second XeLaTeX pass (ToC/refs resolution)..."
xelatex -interaction=nonstopmode -halt-on-error main.tex >> compile.log 2>&1 || {
    echo "✗ Second pass failed. Tail of compile.log:"
    tail -40 compile.log
    exit 1
}

echo ""
echo "✓ Compilation successful"
echo ""
ls -lh main.pdf 2>/dev/null && pdfinfo main.pdf 2>/dev/null | grep -E "Pages|File size|Title|Author" || true
