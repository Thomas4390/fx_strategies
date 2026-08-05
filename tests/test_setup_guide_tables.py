"""Le guide d'installation ne doit plus pouvoir dériver de l'EA compilé.

Sa table de paramètres était écrite à la main. Elle a dérivé deux fois : en avril
sur les allocations et la couche de risque, et le 2026-07-26 sur le plafond de
marge (`false`/`0.70` publié contre `true`/`0.50` compilé) et sur la fenêtre de
session du sleeve 1 (6-14 UTC publié contre 8-16 compilé). Aucun test ne pouvait
le voir : rien ne reliait le `.tex` à une source de vérité.

Ces tests ferment la boucle. `test_mt5_preset_sync.py` asservit déjà PRESET_LINES
aux défauts compilés du `.mq5` ; ceux-ci asservissent le `.tex` publié à
PRESET_LINES.
"""
from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
_SCRIPT = _ROOT / "scripts" / "build_setup_guide_tables.py"
_GUIDE = _ROOT / "reports" / "client" / "guide_installation" / "main.tex"


def _load_generator():
    if not _SCRIPT.exists():  # pragma: no cover
        pytest.skip(f"{_SCRIPT} absent")
    src = _ROOT / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    spec = importlib.util.spec_from_file_location("build_setup_guide_tables", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def gen():
    return _load_generator()


def test_every_preset_input_is_described_or_explicitly_excluded(gen):
    """Un input ajouté au preset ne peut pas être publié sans description.

    C'est le test qui aurait attrapé Inp_RiskScale et Inp_AllocGoldMomentum,
    absents du guide client pendant trois mois.
    """
    values = gen.preset_values()
    described = set(gen.ROLES) | set(gen.EXCLUDED)
    missing = sorted(set(values) - described)
    assert not missing, (
        f"{len(missing)} input(s) du preset sans description ni exclusion : "
        f"{missing}. Les ajouter dans build_setup_guide_tables.ROLES, ou dans "
        f"EXCLUDED avec la raison."
    )


def test_no_description_targets_a_vanished_input(gen):
    """Une description qui survit à son input signale une suppression incomplète."""
    values = gen.preset_values()
    orphans = sorted((set(gen.ROLES) | set(gen.EXCLUDED)) - set(values))
    assert not orphans, f"décrits mais absents du preset : {orphans}"


def test_every_described_input_actually_lands_in_a_table(gen):
    """Décrire un input ne suffit pas : il doit figurer dans une table publiée.

    Angle mort du test précédent, constaté sur ``Inp_Gold_TraceSymbol`` : décrit
    dans ROLES, absent de TABLES, donc jamais imprimé dans le guide client — et
    le contrôle de couverture le voyait « décrit », donc conforme.
    """
    published = {name for _, _, names, _ in gen.TABLES for name in names}
    described = set(gen.ROLES) - set(gen.EXCLUDED)
    unpublished = sorted(described - published)
    assert not unpublished, (
        f"{len(unpublished)} input(s) décrits mais absents de toute table : "
        f"{unpublished}. Les ajouter à la table adéquate de TABLES, ou les "
        f"déplacer dans EXCLUDED avec la raison."
    )


def test_published_tables_are_in_sync_with_the_preset(gen):
    """Les .tex sur disque doivent être ceux que le générateur produit aujourd'hui.

    Échoue — et ne saute pas — si un fichier manque : c'est précisément le mode
    de défaillance qui a laissé passer la dérive.
    """
    values = gen.preset_values()
    stale: list[str] = []
    for stem, col_spec, names, spacers in gen.TABLES:
        path = gen.TBL_DIR / f"{stem}.tex"
        assert path.exists(), (
            f"{path.relative_to(_ROOT)} absent : "
            f"lancer python scripts/build_setup_guide_tables.py"
        )
        expected = gen.render_table(col_spec, names, spacers, values)
        if path.read_text(encoding="utf-8") != expected:
            stale.append(stem)
    assert not stale, (
        f"table(s) périmée(s) : {stale}. "
        f"Lancer python scripts/build_setup_guide_tables.py"
    )


def test_the_guide_includes_every_generated_table(gen):
    """Une table générée mais non incluse ne protège personne."""
    text = _GUIDE.read_text(encoding="utf-8")
    for stem, *_ in gen.TABLES:
        assert f"\\input{{tables/{stem}.tex}}" in text, (
            f"{stem}.tex généré mais absent de main.tex"
        )


def test_the_guide_no_longer_hardcodes_parameter_values():
    """Aucune valeur d'input ne doit subsister en dur dans le corps du guide.

    Le motif recherché est celui de l'ancienne table : `\\code{Inp\\_X} & \\metric{v}`.
    """
    text = _GUIDE.read_text(encoding="utf-8")
    hardcoded = re.findall(r"\\code\{(Inp(?:\\_\w+)+)\}\s*&\s*\\metric\{", text)
    assert not hardcoded, (
        f"{len(hardcoded)} paramètre(s) encore codé(s) en dur dans main.tex : "
        f"{sorted(set(hardcoded))}. Ils doivent venir de tables/."
    )


def test_generated_values_match_the_compiled_defaults(gen):
    """Ceinture et bretelles : les valeurs publiées == les défauts du .mq5.

    PRESET_LINES est déjà asservi au .mq5 par test_mt5_preset_sync, mais ce test
    court-circuite l'intermédiaire pour que l'échec désigne directement le guide.
    """
    mq5 = _ROOT / "src" / "mt5" / "Experts" / "FxMultiSleeve.mq5"
    if not mq5.exists():  # pragma: no cover
        pytest.skip("source .mq5 absente")
    compiled: dict[str, str] = {}
    pat = re.compile(r"\s*input\s+\S+\s+(Inp_\w+)\s*=\s*([^;/]+)")
    for line in mq5.read_text(encoding="utf-8-sig").splitlines():
        m = pat.match(line)
        if m:
            compiled[m.group(1)] = m.group(2).strip().strip('"')

    def canon(v: str) -> str:
        v = v.strip().strip('"')
        try:
            return f"{float(v):g}"
        except ValueError:
            return v

    mismatches = []
    for name, value in gen.preset_values().items():
        if name not in compiled:
            continue
        want = compiled[name]
        # L'enum macro est écrite en littéral dans le preset (4 == MACRO_SOURCE_AUTO).
        if want.startswith("MACRO_SOURCE_"):
            continue
        if canon(value) != canon(want):
            mismatches.append(f"{name}: preset={value!r} .mq5={want!r}")
    assert not mismatches, "\n".join(mismatches)
