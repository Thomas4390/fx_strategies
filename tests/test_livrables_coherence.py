"""Les livrables client doivent citer la configuration qui exécute réellement.

Le 26 juillet 2026, un audit adverse a trouvé un rapport client publiant les
tables d'une autre configuration. Le défaut a été fermé à la main. Il s'est
reproduit le 28 juillet au passage de la sleeve momentum de 20 % à 15 % : la
synthèse exécutive et le guide d'installation ont suivi — leurs tables sont
générées — mais le rapport technique a gardé un tableau d'allocation à
``62 / 9 / 9 / 20`` sous un titre annonçant ``67 / 9 / 9 / 15``, et le guide
pédagogique est resté intégralement sur l'ancienne configuration.

La cause n'est pas l'erreur de recopie, c'est qu'aucun contrôle ne confrontait
la prose des ``.tex`` à la source des poids. Ce fichier ferme ce trou. Trois
mesures, dans l'ordre où elles auraient attrapé le défaut :

- **le périmètre livré** est résolu depuis les six documents racines en suivant
  les ``\\input`` : un ``.tex`` de section hors de cet ensemble n'est pas livré,
  quoi qu'il contienne (c'est ainsi que l'annexe méthodologique orpheline s'est
  maintenue dix jours sans être lue par personne) ;
- **les allocations** citées dans les fichiers livrés doivent appartenir à
  l'ensemble des poids déclarés dans ``write_default_preset.PRESET_LINES``, qui
  est ce que l'EA charge ;
- **les métriques de tête** doivent être celles de ``mt5_reference.json``, la
  sortie du moteur qui exécute. Les valeurs des configurations antérieures sont
  listées explicitement : elles ne doivent plus apparaître nulle part.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from mt5.bridge.write_default_preset import PRESET_LINES  # noqa: E402

# Les six documents livrés au client, tous sous reports/client/. Tout le reste
# de reports/ est du matériau de travail.
_CLIENT_ROOT = "reports/client"
MAIN_DOCUMENTS: tuple[str, ...] = (
    f"{_CLIENT_ROOT}/rapport_technique/main.tex",
    f"{_CLIENT_ROOT}/rapport_technique/main_executive.tex",
    f"{_CLIENT_ROOT}/rapport_technique/main_gold_trades.tex",
    f"{_CLIENT_ROOT}/rapport_technique/main_usdjpy_trades.tex",
    f"{_CLIENT_ROOT}/guide_installation/main.tex",
    f"{_CLIENT_ROOT}/guide_pedagogique/main.tex",
)

_MT5_REFERENCE = _ROOT / "results" / "production_report" / "mt5_reference.json"

_INPUT_RE = re.compile(r"\\input\{([^}]+)\}")
_COMMENT_RE = re.compile(r"(?<!\\)%.*$")


def _uncommented(text: str) -> str:
    """LaTeX ignore ce qui suit un ``%`` non échappé — le contrôle aussi."""
    return "\n".join(_COMMENT_RE.sub("", line) for line in text.splitlines())


def _resolve_inputs(entry: Path, seen: set[Path]) -> None:
    """Ajoute ``entry`` et, récursivement, tout ce qu'il ``\\input``."""
    entry = entry.resolve()
    if entry in seen or not entry.is_file():
        return
    seen.add(entry)
    body = _uncommented(entry.read_text(encoding="utf-8"))
    for target in _INPUT_RE.findall(body):
        child = entry.parent / target
        if child.suffix != ".tex":
            child = child.with_suffix(".tex")
        _resolve_inputs(child, seen)


def delivered_tex_files() -> set[Path]:
    """L'ensemble des ``.tex`` réellement compilés dans un livrable client."""
    seen: set[Path] = set()
    for main in MAIN_DOCUMENTS:
        _resolve_inputs(_ROOT / main, seen)
    return seen


_DELIVERED = sorted(delivered_tex_files())


def _preset_allocations() -> dict[str, float]:
    """Les poids par sleeve, lus dans le preset que l'EA charge."""
    out: dict[str, float] = {}
    for line in PRESET_LINES:
        if not line.startswith("Inp_Alloc"):
            continue
        key, _, value = line.partition("=")
        out[key] = float(value.split("||")[0])
    return out


def _headline() -> dict[str, float]:
    return json.loads(_MT5_REFERENCE.read_text(encoding="utf-8"))["headline"]


# ── Périmètre livré ───────────────────────────────────────────────────────────


def test_every_section_tex_is_reachable_from_a_delivered_document():
    """Un .tex de section qu'aucun document n'inclut n'est pas livré."""
    on_disk = {
        p.resolve()
        for p in (_ROOT / _CLIENT_ROOT / "rapport_technique" / "sections").rglob("*.tex")
    }
    orphans = sorted(p.relative_to(_ROOT).as_posix() for p in on_disk - set(_DELIVERED))
    assert not orphans, (
        f"sections écrites mais jamais compilées : {orphans}. Soit les inclure "
        f"dans un des six documents, soit les sortir de sections/ — les laisser "
        f"là entretient un contenu que personne ne lit et qui diverge en silence."
    )


def test_the_six_documents_exist():
    missing = [m for m in MAIN_DOCUMENTS if not (_ROOT / m).is_file()]
    assert not missing, f"documents livrables introuvables : {missing}"


# ── Allocations ───────────────────────────────────────────────────────────────

# « Moteur 1 --- MR Macro & $62\,\%$ » (cellule de tableau) ou
# « \section{Moteur 1 --- Mean Reversion Intraday Filtré (62\,\%)} » (titre).
# Le pourcentage doit suivre immédiatement le libellé du moteur : sans cette
# contrainte, la règle capturait « le Moteur~2 ... un gain relatif de 30\,\% »,
# qui ne parle pas d'allocation.
_MOTEUR_PCT_RE = re.compile(
    r"Moteur[~ ]?(\d)\b[^\n&()]{0,55}?[&(]\s*\$?(?<![\d.,{])(\d{1,3})\\,\\%"
)
# « \code{Inp\_AllocMRMacro} & $0{,}72$ » ou « ... & \metric{0.67} ». La valeur
# doit occuper la cellule qui suit la clé : sans cette contrainte, la règle
# lisait le premier nombre venu d'une phrase citant l'input en prose.
_ALLOC_INPUT_RE = re.compile(
    r"Inp\\?_?Alloc(\w+)\}[^\n&]{0,12}&[^\n\d]{0,12}(\d)[.,{}]{1,3}(\d{1,2})",
)

_MOTEUR_TO_INPUT = {
    "1": "Inp_AllocMRMacro",
    "2": "Inp_AllocTSMomentum",
    "3": "Inp_AllocRSIDaily",
    "4": "Inp_AllocGoldMomentum",
    "5": "Inp_AllocH1Momentum",
}


@pytest.mark.parametrize("tex", _DELIVERED, ids=lambda p: p.name)
def test_allocation_percentages_match_the_preset(tex: Path):
    """Un poids cité en pourcentage doit être celui que l'EA charge."""
    allocations = _preset_allocations()
    body = _uncommented(tex.read_text(encoding="utf-8"))
    for line_no, line in enumerate(body.splitlines(), start=1):
        for moteur, pct in _MOTEUR_PCT_RE.findall(line):
            expected = allocations[_MOTEUR_TO_INPUT[moteur]]
            got = int(pct) / 100
            assert got == pytest.approx(expected), (
                f"{tex.relative_to(_ROOT)}:{line_no} — le Moteur {moteur} y est "
                f"donné à {pct}\\,\\% alors que le preset l'alloue à "
                f"{expected:.0%}. Source : write_default_preset.PRESET_LINES."
            )


@pytest.mark.parametrize("tex", _DELIVERED, ids=lambda p: p.name)
def test_allocation_inputs_match_the_preset(tex: Path):
    """Une table de paramètres doit citer la valeur compilée de l'input."""
    allocations = _preset_allocations()
    body = _uncommented(tex.read_text(encoding="utf-8"))
    for line_no, line in enumerate(body.splitlines(), start=1):
        for suffix, whole, frac in _ALLOC_INPUT_RE.findall(line):
            key = f"Inp_Alloc{suffix}"
            if key not in allocations:
                continue
            got = float(f"{whole}.{frac}")
            assert got == pytest.approx(allocations[key]), (
                f"{tex.relative_to(_ROOT)}:{line_no} — {key} y vaut {got} "
                f"alors que le preset compile {allocations[key]}."
            )


# ── Métriques de tête ─────────────────────────────────────────────────────────

# Valeurs publiées par des éditions antérieures, retirées depuis. Chaque entrée
# porte la raison de son retrait et la valeur qui la remplace ; toutes sont
# vérifiées contre mt5_reference.json par le test qui suit.
#
# La règle vise les valeurs qui se donnent pour courantes, pas l'histoire du
# portefeuille : le repli de 44,3 % de la configuration à 10 % est cité comme
# point de comparaison daté dans les livrables, et c'est légitime. Ce qui ne
# l'est pas, c'est une valeur d'une configuration jamais livrée (50,9 %) ou un
# poids abandonné présenté au présent.
SUPERSEDED: dict[str, str] = {
    r"50\{,\}9\\,\\%": "repli d'équité de la configuration à 20 % → 47,04 %",
    r"91\\,\\% du (résultat|profit)": "part du momentum sous 20 % → 87,0 %",
    r"(?<![\d.,{])62\\,\\%": "allocation MR Macro de la configuration à 20 % → 67 %",
    # Table de référence du guide d'installation : ce sont les valeurs que le
    # client doit reproduire chez lui. Périmées, elles feraient déclarer non
    # conforme une installation correcte.
    r"\b909[~ ]?(trades|transactions)|\\metric\{909\}": "décompte du run antérieur → 905 transactions",
    r"50\.88": "repli d'équité du run antérieur → 47.04",
    r"20\.46": "repli de balance du run antérieur → 14.27",
}


@pytest.mark.parametrize("tex", _DELIVERED, ids=lambda p: p.name)
def test_no_superseded_figure_survives(tex: Path):
    body = _uncommented(tex.read_text(encoding="utf-8"))
    for line_no, line in enumerate(body.splitlines(), start=1):
        for pattern, reason in SUPERSEDED.items():
            assert not re.search(pattern, line), (
                f"{tex.relative_to(_ROOT)}:{line_no} — valeur périmée "
                f"({pattern}) : {reason}."
            )


_VERDICT_TABLE = (
    _ROOT / _CLIENT_ROOT / "rapport_technique" / "tables" / "robustness_verdict_summary.tex"
)
_VERDICT_COUNT_RE = re.compile(r"(\d+) test\(s\) sur (\d+) franchissent")
# Formulations qui affirment que *tous* les tests passent. La prose de la
# section 10 a annoncé « les six tests franchissent leur seuil » pendant que sa
# propre table de verdict en comptait quatre sur six.
_ALL_PASS_RE = re.compile(
    r"(les|tous les)\s+(six|6|cinq|5|quatre|4)\s+tests?[^\n.]{0,40}franchissent"
)


@pytest.mark.parametrize("tex", _DELIVERED, ids=lambda p: p.name)
def test_no_document_claims_more_passing_tests_than_the_verdict_table(tex: Path):
    """La prose ne peut pas annoncer un verdict que la table contredit."""
    table = _uncommented(_VERDICT_TABLE.read_text(encoding="utf-8"))
    passed, total = (int(x) for x in _VERDICT_COUNT_RE.search(table).groups())
    if passed == total:
        pytest.skip("tous les tests passent : l'affirmation serait exacte")
    body = _uncommented(tex.read_text(encoding="utf-8"))
    for line_no, line in enumerate(body.splitlines(), start=1):
        match = _ALL_PASS_RE.search(line)
        assert match is None, (
            f"{tex.relative_to(_ROOT)}:{line_no} — «~{match.group(0)}~» affirme "
            f"que tous les tests passent, alors que la table de verdict en "
            f"compte {passed} sur {total}."
        )


def test_the_superseded_list_stays_anchored_to_the_reference():
    """Si la production change, la liste des valeurs retirées doit suivre.

    Ce test échoue quand une valeur encore publiée figure dans ``SUPERSEDED``,
    ou quand les remplacements annoncés ne sont plus ceux du moteur.
    """
    head = _headline()
    assert head["equity_dd_pct_mt5"] == pytest.approx(47.04)
    assert head["total_trades"] == pytest.approx(905)
    assert head["sharpe_ratio_mt5"] == pytest.approx(1.01)
    assert head["cagr"] == pytest.approx(0.3965, abs=5e-5)

    by_sleeve = json.loads(_MT5_REFERENCE.read_text(encoding="utf-8"))["by_sleeve"]
    momentum = next(s for s in by_sleeve if s["sleeve"] == "Gold Momentum")
    assert momentum["share_of_net_pct"] == pytest.approx(0.870, abs=5e-4), (
        "la part du momentum a bougé : mettre à jour SUPERSEDED et la prose des "
        "livrables avant de republier."
    )
