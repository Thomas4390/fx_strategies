"""Le rapport de la stratégie 2 ne doit publier aucun chiffre qu'il n'a mesuré.

Le squelette du rapport XAUUSD x10 est écrit avant la campagne. C'est
exactement la situation où un chiffre « provisoire » se glisse dans la prose
pour illustrer une phrase, puis y reste. Trois garde-fous :

- **la prose passe par les macros** : un pourcentage décimal ou un montant en
  dollars écrit en dur dans un ``.tex`` de la stratégie 2, hors ``tables/``,
  est refusé ; les valeurs viennent de ``tables/x10_headline.tex``, que
  ``scripts/build_x10_report_assets.py`` produira au lot L2 ;
- **tant qu'aucun résultat n'existe**, toutes les macros de tête valent
  ``n.d.`` et le document se déclare provisoire : c'est ce qui empêche de
  publier un PDF d'apparence définitive sur un dossier vide ;
- **le verdict est l'une des trois valeurs prévues**, parce que la section~14
  livrée et l'encadré du résumé en dépendent : une quatrième valeur ferait
  silencieusement basculer le rapport sur la branche « non déploiement ».
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
_S2 = _ROOT / "reports" / "client" / "strategie2_xauusd_x10"
_MAIN = _S2 / "ApogeeInvest_Strategie2_XAUUSD_NiveauxX10_RapportTechnique.tex"
_HEADLINE = _S2 / "tables" / "x10_headline.tex"
_VERDICT = _S2 / "tables" / "x10_verdict.tex"
_IS_SUMMARY = _ROOT / "results" / "xau_x10" / "is_summary.json"

_VERDICTS = {"GO", "NOGO", "INCONCLUSIF"}

# Les macros que la prose a le droit d'appeler. La liste est celle que le
# générateur devra produire : un ajout ici sans producteur casse la compilation.
HEADLINE_MACROS: tuple[str, ...] = (
    "XTenTradesIS",
    "XTenExpectancyR",
    "XTenProfitFactor",
    "XTenSharpeIS",
    "XTenDSR",
    "XTenPBO",
    "XTenMaxDDPct",
    "XTenTrialsCount",
    "XTenTradesOOS",
    "XTenMTFiveExpectancyR",
    "XTenQCExpectancyR",
)

_NEWCOMMAND_RE = re.compile(r"\\newcommand\{\\(\w+)\}\{([^}]*)\}")
_COMMENT_RE = re.compile(r"(?<!\\)%.*$")

# Un nombre décimal, dans les trois écritures du dépôt : « 1.5 », « 1,5 » et
# « 1{,}5 » (la forme mathématique française).
_DECIMAL = r"\d+(?:[.,]|\{,\})\d+"
# « 49\,143 », « 49~143 », « 49 143 » : un montant à séparateur de milliers.
_THOUSANDS = r"\d{1,3}(?:(?:\\,|~|\s)\d{3})+"

# Un pourcentage *décimal* écrit en dur. La règle vise les mesures, pas les
# repères du mandat : « M5 », « H1 », « EMA50 », « x10 » et « R \geq 1 » ne
# portent pas de décimale et ne sont pas suivis d'un signe de pourcentage.
_HARDCODED_PCT_RE = re.compile(rf"{_DECIMAL}\s*\\,\\%")
# Un montant en dollars écrit en dur. Le pas de la grille — « 10~\$ » — est un
# entier sans séparateur : il reste autorisé, c'est la définition de l'objet
# d'étude, pas un résultat.
_HARDCODED_USD_RE = re.compile(rf"(?:{_DECIMAL}|{_THOUSANDS})\s*(?:~|\\,)?\s*(?:USD|\\\$)")


def _uncommented(text: str) -> str:
    """LaTeX ignore ce qui suit un ``%`` non échappé — le contrôle aussi."""
    return "\n".join(_COMMENT_RE.sub("", line) for line in text.splitlines())


def _macros(path: Path) -> dict[str, str]:
    return dict(_NEWCOMMAND_RE.findall(path.read_text(encoding="utf-8")))


def _prose_tex_files() -> list[Path]:
    """Les ``.tex`` de la stratégie 2 qui portent de la prose, donc hors tables."""
    return sorted(p for p in _S2.rglob("*.tex") if p.parent.name != "tables")


# ── Le document existe et s'adresse au client ─────────────────────────────────


def test_the_strategy_two_report_exists_and_names_the_client():
    assert _MAIN.is_file(), f"document racine introuvable : {_MAIN}"
    body = _MAIN.read_text(encoding="utf-8")
    assert "Apogée Invest" in body, (
        "le rapport ne nomme pas le destinataire : la page de titre et "
        "l'encadré de confidentialité doivent citer Apogée Invest."
    )


# ── Aucun chiffre en dur dans la prose ────────────────────────────────────────


@pytest.mark.parametrize("tex", _prose_tex_files(), ids=lambda p: p.name)
def test_no_hardcoded_figure_in_the_prose(tex: Path):
    body = _uncommented(tex.read_text(encoding="utf-8"))
    for line_no, line in enumerate(body.splitlines(), start=1):
        for pattern, kind in (
            (_HARDCODED_PCT_RE, "pourcentage"),
            (_HARDCODED_USD_RE, "montant en dollars"),
        ):
            match = pattern.search(line)
            assert match is None, (
                f"{tex.relative_to(_ROOT)}:{line_no} — {kind} «~{match.group(0)}~» "
                f"écrit en dur. Les chiffres du rapport passent par les macros de "
                f"tables/x10_headline.tex, produites par "
                f"scripts/build_x10_report_assets.py."
            )


# ── Tant qu'il n'y a pas de résultat, il n'y a pas de chiffre ─────────────────


def test_every_headline_macro_is_undetermined_while_no_result_exists():
    if _IS_SUMMARY.is_file():
        pytest.skip("couvert au lot L2")
    macros = _macros(_HEADLINE)
    missing = [name for name in HEADLINE_MACROS if name not in macros]
    assert not missing, f"macros de tête absentes de x10_headline.tex : {missing}"
    published = {name: value for name, value in macros.items() if value != "n.d."}
    assert not published, (
        f"{_HEADLINE.relative_to(_ROOT)} publie {published} alors que "
        f"{_IS_SUMMARY.relative_to(_ROOT)} n'existe pas : aucune campagne n'a "
        f"produit ces valeurs."
    )


def test_the_report_declares_itself_provisional_while_no_result_exists():
    if _IS_SUMMARY.is_file():
        pytest.skip("couvert au lot L2")
    assert _macros(_VERDICT).get("XTenVerdictProvisoire") == "1", (
        "sans résultat lu, le rapport doit porter son encadré « Document "
        "provisoire » : \\XTenVerdictProvisoire vaut 1."
    )


# ── Le verdict est l'une des trois valeurs prévues ────────────────────────────


def test_the_verdict_is_one_of_the_three_expected_values():
    verdict = _macros(_VERDICT).get("XTenVerdict")
    assert verdict in _VERDICTS, (
        f"\\XTenVerdict vaut « {verdict} » : attendu l'un de "
        f"{sorted(_VERDICTS)}. Le document choisit la section~14 livrée sur "
        f"cette valeur, et toute autre le bascule en « non déploiement »."
    )
