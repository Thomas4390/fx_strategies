"""Le rapport de la stratégie 2 ne doit publier aucun chiffre qu'il n'a mesuré.

Le squelette du rapport XAUUSD x10 a été écrit avant la campagne. C'est
exactement la situation où un chiffre « provisoire » se glisse dans la prose
pour illustrer une phrase, puis y reste. Quatre garde-fous :

- **la prose passe par les macros** : un pourcentage ou un montant en dollars
  écrit en dur dans un ``.tex`` de la stratégie 2, hors ``tables/``, est
  refusé ; les valeurs viennent de ``tables/x10_headline.tex``, produit par
  ``scripts/build_x10_report_assets.py`` ;
- **chaque macro publiée redescend sur sa source** : le test refait le chemin
  JSON de ``MACRO_SOURCES`` et compare au fichier publié. Une macro éditée à
  la main, ou un fichier de résultats régénéré sans rejouer le producteur,
  fait rougir la suite ;
- **le verdict est l'une des trois valeurs prévues** et vaut celui du fichier
  de résultats, parce que la section~14 livrée et l'encadré du résumé en
  dépendent ;
- **le vocabulaire suit le verdict** : sous « ne pas déployer », une poignée
  de formulations de succès sont interdites hors négation.

Deux mécanismes d'exception, tous deux explicites et commentés :

``LITERAL_WHITELIST``
    une poignée de littéraux qui sont des **définitions** — constantes de la
    spécification, niveau de confiance conventionnel — et non des mesures.
``% <<< LITERAL-OK`` / ``% >>> LITERAL-OK``
    un bloc de prose exempté, dont le commentaire d'ouverture doit dire
    pourquoi. Deux blocs seulement : l'exemple pédagogique entièrement fictif
    de la section 4 et la table des tolérances de réconciliation, gelées dans
    la spécification et absentes de tout fichier de résultats.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
_SRC = _ROOT / "src"
_SCRIPTS = _ROOT / "scripts"
for _path in (_SRC, _SCRIPTS):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

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

_COMMENT_RE = re.compile(r"(?<!\\)%.*$")

# Un nombre décimal, dans les trois écritures du dépôt : « 1.5 », « 1,5 » et
# « 1{,}5 » (la forme mathématique française).
_DECIMAL = r"\d+(?:[.,]|\{,\})\d+"
# « 49\,143 », « 49~143 », « 49 143 » : un montant à séparateur de milliers.
_THOUSANDS = r"\d{1,3}(?:(?:\\,|~|\s)\d{3})+"

# Un pourcentage écrit en dur, décimal **ou entier**. La règle vise les
# mesures : « M5 », « H1 », « EMA50 », « x10 » et « R \geq 1 » ne sont pas
# suivis d'un signe de pourcentage et ne sont donc jamais capturés.
_HARDCODED_PCT_RE = re.compile(rf"(?:{_DECIMAL}|\d+)\s*(?:\\,|~)?\\%")
# Un montant en dollars écrit en dur. Le pas de la grille — « 10~\$ » — est un
# entier sans séparateur : il reste autorisé, c'est la définition de l'objet
# d'étude, pas un résultat.
_HARDCODED_USD_RE = re.compile(
    rf"(?:{_DECIMAL}|{_THOUSANDS})\s*(?:~|\\,)?\s*(?:USD|\\\$)"
)

# ── Liste blanche, littéral par littéral ──────────────────────────────────────
# Chaque entrée est une CONSTANTE, pas une mesure : elle ne vit dans aucun
# fichier de results/xau_x10/ et aucun générateur ne peut donc la produire.
LITERAL_WHITELIST: dict[str, str] = {
    "0,5~\\%": "risque par position, gelé à l'annexe A.1 de la spécification",
    "0,25~\\%": "risque réduit sous contexte dollar adverse, même annexe",
    "60~\\%": "seuil de plateau et de walk-forward, gelé dans la table de décision",
    "95~\\%": "niveau de confiance conventionnel des intervalles bootstrap",
}

# ── Blocs exemptés, avec justification obligatoire à l'ouverture ──────────────
_LITERAL_OPEN = "% <<< LITERAL-OK"
_LITERAL_CLOSE = "% >>> LITERAL-OK"

# ── Vocabulaire interdit sous verdict négatif ─────────────────────────────────
# Liste courte et ciblée : des adjectifs de succès appliqués à la stratégie.
# « robustesse » (le nom du chapitre de tests) n'est pas concerné, la frontière
# de mot l'exclut ; une formulation niée non plus, d'où la garde sur « ne ».
_SUCCESS_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"\brobuste\b", "la stratégie n'est pas robuste : le verdict est négatif"),
    (r"\bvalid(?:é|ée|és|ées)\b", "rien n'a été validé dans ce dossier"),
    (r"\bprometteu(?:r|se|rs|ses)\b", "aucune piste n'est promise, elles sont hypothétiques"),
    (r"\bsurperform", "il n'y a aucune surperformance à annoncer"),
    (r"prêt(?:e|s|es)?\s+(?:pour|au|à)\s+(?:le\s+)?déploiement", "rien n'est prêt à déployer"),
)
_NEGATION_RE = re.compile(r"\b(?:ne|n'|non|jamais|aucun|aucune|pas)\b")


def _uncommented(text: str) -> str:
    """LaTeX ignore ce qui suit un ``%`` non échappé — le contrôle aussi."""
    return "\n".join(_COMMENT_RE.sub("", line) for line in text.splitlines())


def _checked_lines(path: Path) -> list[tuple[int, str]]:
    """Les lignes de prose soumises au contrôle, blocs exemptés retirés.

    Le marqueur d'ouverture doit porter une justification sur sa propre ligne
    ou sur les suivantes : un bloc exempté sans motif écrit est refusé.
    """
    raw = path.read_text(encoding="utf-8").splitlines()
    out: list[tuple[int, str]] = []
    skipping = False
    for line_no, line in enumerate(raw, start=1):
        stripped = line.strip()
        if stripped.startswith(_LITERAL_OPEN):
            assert stripped != _LITERAL_OPEN, (
                f"{path.relative_to(_ROOT)}:{line_no} — bloc exempté sans "
                f"justification. Écrire « {_LITERAL_OPEN} : <pourquoi> »."
            )
            skipping = True
            continue
        if stripped.startswith(_LITERAL_CLOSE):
            skipping = False
            continue
        if not skipping:
            out.append((line_no, _COMMENT_RE.sub("", line)))
    assert not skipping, f"{path.relative_to(_ROOT)} — bloc exempté jamais refermé."
    return out


def _macros(path: Path) -> dict[str, str]:
    """``\\newcommand{\\Nom}{valeur}`` → ``{"Nom": "valeur"}``, accolades comprises.

    Une expression régulière naïve s'arrête à la première accolade fermante et
    coupe « $-0{,}1602$ » en plein milieu : la valeur est donc lue par un
    balayage à accolades équilibrées.
    """
    text = path.read_text(encoding="utf-8")
    out: dict[str, str] = {}
    for match in re.finditer(r"\\newcommand\{\\(\w+)\}\{", text):
        depth, i = 1, match.end()
        while i < len(text) and depth:
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
            i += 1
        out[match.group(1)] = text[match.end() : i - 1]
    return out


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
    for line_no, line in _checked_lines(tex):
        for pattern, kind in (
            (_HARDCODED_PCT_RE, "pourcentage"),
            (_HARDCODED_USD_RE, "montant en dollars"),
        ):
            for match in pattern.finditer(line):
                assert match.group(0) in LITERAL_WHITELIST, (
                    f"{tex.relative_to(_ROOT)}:{line_no} — {kind} "
                    f"«~{match.group(0)}~» écrit en dur. Les chiffres du rapport "
                    f"passent par les macros de tables/x10_headline.tex, produites "
                    f"par scripts/build_x10_report_assets.py. Si c'est une "
                    f"constante de la spécification, l'inscrire dans "
                    f"LITERAL_WHITELIST avec son motif ; si c'est un exemple "
                    f"fictif, l'encadrer d'un bloc « {_LITERAL_OPEN} : <pourquoi> »."
                )


# ── Tant qu'il n'y a pas de résultat, il n'y a pas de chiffre ─────────────────


def test_every_headline_macro_is_undetermined_while_no_result_exists():
    if _IS_SUMMARY.is_file():
        pytest.skip("la campagne a produit ses résultats : couvert plus bas")
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
        pytest.skip("la campagne a produit ses résultats : couvert plus bas")
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


# ══════════════════════════════════════════════════════════════════════════════
#  À partir d'ici : la campagne a tourné. Les macros publiées sont confrontées
#  à leurs sources, une par une.
# ══════════════════════════════════════════════════════════════════════════════

_HAS_RESULTS = _IS_SUMMARY.is_file()
_needs_results = pytest.mark.skipif(
    not _HAS_RESULTS, reason="aucun résultat de campagne : couvert par les tests ci-dessus"
)


@pytest.fixture(scope="module")
def assets():
    """Le générateur et le paquet de résultats qu'il lit."""
    import build_x10_report_assets as builder

    return builder, builder.load_bundle()


@_needs_results
def test_every_published_macro_matches_its_json_source(assets):
    """Chaque macro publiée est refaite depuis sa source, au format près."""
    builder, bundle = assets
    published = _macros(_HEADLINE)
    mismatched: list[str] = []
    for name, spec in builder.MACRO_SOURCES.items():
        assert name in published, f"macro {name} absente de x10_headline.tex"
        if name in builder.NON_RESULT_MACROS:
            continue
        # Même repli que le générateur : source absente — ou présente mais dont
        # le schéma ne porte pas le chemin attendu — donne « n.d. ».
        if spec.source not in bundle or bundle.get(spec.source) is None:
            expected = builder.UNDETERMINED
        else:
            try:
                expected = spec.fmt(spec.value(bundle))
            except (KeyError, IndexError, TypeError):
                expected = builder.UNDETERMINED
        if published[name] != expected:
            mismatched.append(
                f"\\{name} publie « {published[name]} », sa source "
                f"({spec.source}.{spec.path or 'dérivée'}) donne « {expected} »"
            )
    assert not mismatched, (
        "des macros publiées ne redescendent plus sur leur source :\n  "
        + "\n  ".join(mismatched)
        + "\nRelancer scripts/build_x10_report_assets.py."
    )


@_needs_results
def test_every_headline_macro_of_the_contract_is_published(assets):
    """Les macros du contrat initial existent toujours, sous peine de compilation cassée."""
    _builder, _bundle = assets
    published = _macros(_HEADLINE)
    missing = [name for name in HEADLINE_MACROS if name not in published]
    assert not missing, f"macros de tête absentes de x10_headline.tex : {missing}"


@_needs_results
def test_the_macro_names_are_alphabetic_only(assets):
    """``\\XTenSampleM5Bars`` se lirait ``\\XTenSampleM`` suivi de « 5Bars »."""
    builder, _bundle = assets
    illegal = sorted(name for name in builder.MACRO_SOURCES if not name.isalpha())
    assert not illegal, (
        f"noms de macro non alphabétiques : {illegal}. TeX n'accepte que des "
        f"lettres dans un nom de séquence de contrôle, et la compilation "
        f"échouerait très loin du vrai coupable."
    )


@_needs_results
def test_an_undetermined_execution_macro_keeps_the_pending_banner(assets):
    """Le bandeau « sections d'exécution en attente » suit les macros, pas l'humeur."""
    builder, bundle = assets
    published = _macros(_HEADLINE)
    undetermined = [
        name for name in builder.EXECUTION_MACROS if published[name] == builder.UNDETERMINED
    ]
    flag = _macros(_VERDICT).get("XTenVerdictProvisoire")
    assert flag == ("1" if undetermined else "0"), (
        f"macros d'exécution encore indéterminées : {undetermined}, mais "
        f"\\XTenVerdictProvisoire vaut {flag}. Le bandeau du résumé et la "
        f"section « recherche contre exécution » en dépendent."
    )


@_needs_results
def test_the_verdict_macro_matches_the_measured_verdict(assets):
    builder, bundle = assets
    expected = builder.verdict_of(bundle)
    assert _macros(_VERDICT).get("XTenVerdict") == expected, (
        f"\\XTenVerdict ne reflète pas is_summary.json, qui dit « "
        f"{bundle['is_summary']['decision_table']['provisional_in_sample_verdict']} »"
        f" soit {expected}."
    )


@_needs_results
def test_the_trial_count_matches_the_repository_registry(assets):
    """Le nombre d'essais cité déflate le Sharpe : il doit être celui du registre."""
    from framework.trials import distinct_trials

    _builder, bundle = assets
    registry = distinct_trials("xau_x10")
    assert int(bundle["is_summary"]["n_trials_logged"]) == registry, (
        f"is_summary.json logue {bundle['is_summary']['n_trials_logged']} essais, "
        f"le registre du dépôt en compte {registry}. Le DSR publié est déflaté "
        f"par ce nombre : un écart invalide le chiffre."
    )
    assert int(bundle["robustness"]["dsr"]["n_trials"]) == registry, (
        "le DSR n'a pas été déflaté par le nombre d'essais du registre."
    )


@_needs_results
def test_the_published_tables_match_what_the_generator_would_produce():
    """``--check`` : les tables publiées n'ont pas dérivé de leurs sources."""
    completed = subprocess.run(
        [sys.executable, str(_SCRIPTS / "build_x10_report_assets.py"), "--check"],
        cwd=_ROOT,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, (
        "scripts/build_x10_report_assets.py --check échoue :\n"
        f"{completed.stdout}\n{completed.stderr}"
    )


# ── Le vocabulaire suit le verdict ───────────────────────────────────────────


@_needs_results
@pytest.mark.parametrize("tex", _prose_tex_files(), ids=lambda p: p.name)
def test_no_success_wording_under_a_negative_verdict(tex: Path):
    verdict = _macros(_VERDICT).get("XTenVerdict")
    if verdict == "GO":
        pytest.skip("verdict favorable : ces formulations redeviennent recevables")
    for line_no, line in _checked_lines(tex):
        lowered = line.lower()
        for pattern, why in _SUCCESS_PATTERNS:
            match = re.search(pattern, lowered)
            if match is None:
                continue
            context = lowered[max(0, match.start() - 90) : match.end() + 20]
            assert _NEGATION_RE.search(context), (
                f"{tex.relative_to(_ROOT)}:{line_no} — «~{match.group(0)}~» "
                f"hors négation sous verdict {verdict} : {why}."
            )


# ── La table de décision publiée est bien celle des résultats ────────────────


@_needs_results
def test_the_decision_table_counts_match_the_results(assets):
    """Le résumé annonce « un critère sur neuf » : les deux nombres sont lus, pas écrits."""
    _builder, bundle = assets
    table = bundle["is_summary"]["decision_table"]
    criteria = table["criteria"]
    assert len(criteria) == int(table["n_criteria"])
    passing = sum(1 for crit in criteria.values() if crit["pass"])
    assert passing == int(table["n_passing"]), (
        "is_summary.json est incohérent avec lui-même sur le nombre de "
        "critères franchis ; le rapport publierait un décompte faux."
    )


@_needs_results
def test_the_holdout_was_not_read_by_the_published_campaign(assets):
    """Le rapport affirme que la tranche 2026 n'a pas été lue : le JSON doit le dire."""
    _builder, bundle = assets
    holdout = bundle["is_summary"]["holdout"]
    assert holdout["state"] == "LOCKED"
    assert holdout["touched_by_this_phase"] is False
    assert holdout["m1_max"] < holdout["holdout_start"], (
        "une barre postérieure au gel est entrée dans la campagne publiée."
    )


@_needs_results
def test_the_feasibility_table_stops_before_the_holdout():
    """Le tableau du pas de grille publié ne contient aucune ligne ≥ 2026."""
    path = _ROOT / "results" / "xau_x10" / "feasibility_table.json"
    assert path.is_file(), "results/xau_x10/feasibility_table.json non produit"
    payload = json.loads(path.read_text(encoding="utf-8"))
    years = [row["year"] for row in payload["years"]]
    assert years and max(years) <= 2025, (
        f"le tableau de faisabilité publie {max(years)} : le rapport ne doit "
        f"contenir aucune mesure postérieure au gel."
    )
