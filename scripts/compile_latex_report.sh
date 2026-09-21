#!/usr/bin/env bash
# Compile un des documents LaTeX livrés au client (xelatex, deux passes).
#
# Ce script était codé en dur sur le seul rapport technique. Les trois
# autres livrables — la synthèse exécutive et les deux guides client — n'avaient
# aucun script de compilation et ont été produits à la main, ce qui explique
# qu'ils aient dérivé chacun de leur côté.
#
# Usage:
#   scripts/compile_latex_report.sh              # rapport technique (défaut)
#   scripts/compile_latex_report.sh executive    # synthèse exécutive
#   scripts/compile_latex_report.sh pedagogical  # guide pédagogique
#   scripts/compile_latex_report.sh setup        # guide d'installation
#   scripts/compile_latex_report.sh goldtrades   # analyse des trades du moteur or
#   scripts/compile_latex_report.sh usdjpytrades # analyse des trades du candidat USD/JPY
#   scripts/compile_latex_report.sh x10          # rapport technique stratégie 2 (XAUUSD x10)
#   scripts/compile_latex_report.sh all          # les sept
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"

compile_one() {
    local dir="$1" stem="$2"
    echo "═══════════════════════════════════════════════════════════════"
    echo "  $stem.tex  ($dir)"
    echo "═══════════════════════════════════════════════════════════════"
    cd "$ROOT/$dir"

    # Deux passes : la seconde résout table des matières et références croisées.
    echo "[1/2] First XeLaTeX pass..."
    xelatex -interaction=nonstopmode -halt-on-error "$stem.tex" > "compile_$stem.log" 2>&1 || {
        echo "✗ First pass failed. Tail of compile_$stem.log:"
        tail -40 "compile_$stem.log"
        return 1
    }

    echo "[2/2] Second XeLaTeX pass (ToC/refs resolution)..."
    xelatex -interaction=nonstopmode -halt-on-error "$stem.tex" >> "compile_$stem.log" 2>&1 || {
        echo "✗ Second pass failed. Tail of compile_$stem.log:"
        tail -40 "compile_$stem.log"
        return 1
    }

    echo "✓ $stem.pdf"
    pdfinfo "$stem.pdf" 2>/dev/null | grep -E "Pages|File size|Title" || true
    echo ""
}

target="${1:-report}"
case "$target" in
    report)      compile_one "reports/client/rapport_technique" "ApogeeInvest_Strategie1_FXMultiMoteurs_RapportTechnique" ;;
    executive)   compile_one "reports/client/rapport_technique" "ApogeeInvest_Strategie1_FXMultiMoteurs_SyntheseExecutive" ;;
    pedagogical) compile_one "reports/client/guide_pedagogique" "ApogeeInvest_Strategie1_GuidePedagogique" ;;
    setup)       compile_one "reports/client/guide_installation" "ApogeeInvest_Strategie1_GuideInstallation" ;;
    goldtrades)  compile_one "reports/client/rapport_technique" "ApogeeInvest_Strategie1_AnalyseTrades_Or" ;;
    usdjpytrades) compile_one "reports/client/rapport_technique" "ApogeeInvest_Strategie1_AnalyseTrades_USDJPY" ;;
    x10)         compile_one "reports/client/strategie2_xauusd_x10" "ApogeeInvest_Strategie2_XAUUSD_NiveauxX10_RapportTechnique" ;;
    all)
        compile_one "reports/client/rapport_technique" "ApogeeInvest_Strategie1_FXMultiMoteurs_RapportTechnique"
        compile_one "reports/client/rapport_technique" "ApogeeInvest_Strategie1_FXMultiMoteurs_SyntheseExecutive"
        compile_one "reports/client/guide_pedagogique" "ApogeeInvest_Strategie1_GuidePedagogique"
        compile_one "reports/client/guide_installation" "ApogeeInvest_Strategie1_GuideInstallation"
        compile_one "reports/client/rapport_technique" "ApogeeInvest_Strategie1_AnalyseTrades_Or"
        compile_one "reports/client/rapport_technique" "ApogeeInvest_Strategie1_AnalyseTrades_USDJPY"
        compile_one "reports/client/strategie2_xauusd_x10" "ApogeeInvest_Strategie2_XAUUSD_NiveauxX10_RapportTechnique"
        ;;
    *)
        echo "Cible inconnue : $target" >&2
        echo "Attendu : report | executive | pedagogical | setup | goldtrades | usdjpytrades | x10 | all" >&2
        exit 2
        ;;
esac
