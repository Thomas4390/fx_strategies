"""Le preset Tester doit décrire les mêmes défauts que l'EA compilé.

``write_default_preset.py`` recopie à la main les ``input`` de
``FxMultiSleeve.mq5``. Cette duplication a dérivé sans que rien ne le signale,
et elle coûte cher parce que MT5 sert le ``.set`` **par-dessus** les défauts
compilés : un preset périmé fait tourner un backtest avec d'autres paramètres
que ceux du source, silencieusement.

Deux symptômes constatés le 2026-07-26, tous deux expliqués par la même dérive :

* ``Inp_Gold_*`` n'existait dans aucun preset — d'où la trace journalière qui
  ne s'écrivait jamais malgré ``Inp_Gold_Trace=true``, cherchée pendant deux
  sessions du côté de l'EA ;
* les allocations restaient figées à 0.80/0.10/0.10 alors que le source disait
  0.72/0.09/0.09/0.10, d'où un ``RiskManager init failed`` (somme = 1.10) et un
  backtest à zéro trade.

Ce test compare les deux listes plutôt que de faire confiance à la discipline.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_MQ5 = (
    Path(__file__).resolve().parent.parent
    / "src/mt5/Experts/FxMultiSleeve.mq5"
)

# input <type> <name> = <value>;   (le commentaire de fin de ligne est ignoré)
_INPUT_RE = re.compile(
    r"^\s*input\s+(?P<type>\w+)\s+(?P<name>\w+)\s*=\s*(?P<value>[^;]+);",
    re.MULTILINE,
)

# Valeurs non littérales (constantes d'enum) : la présence de la clé est
# vérifiée, pas sa valeur — le preset les encode en entier.
_ENUM_TYPES = {"EMacroSourceMode"}


def _parse_mq5_inputs() -> dict[str, tuple[str, str]]:
    if not _MQ5.exists():  # pragma: no cover - dépend du checkout
        pytest.skip(f"{_MQ5.name} absent")
    text = _MQ5.read_text(encoding="utf-8", errors="ignore")
    out: dict[str, tuple[str, str]] = {}
    for m in _INPUT_RE.finditer(text):
        out[m.group("name")] = (m.group("type"), m.group("value").strip())
    return out


def _parse_preset_values() -> dict[str, str]:
    from importlib import import_module
    import sys

    sys.path.insert(0, str(_MQ5.parent.parent / "bridge"))
    mod = import_module("write_default_preset")

    values: dict[str, str] = {}
    for line in mod.PRESET_LINES:
        if line.startswith(";") or "=" not in line:
            continue
        key, _, rest = line.partition("=")
        values[key.strip()] = rest.split("||")[0].strip()
    return values


def _as_float(raw: str) -> float | None:
    try:
        return float(raw)
    except ValueError:
        return None


def test_preset_covers_every_ea_input() -> None:
    """Un input absent du preset garde sa valeur compilée — ou pas, selon ce
    que MT5 a mémorisé. C'est exactement le piège Inp_Gold_Trace."""
    mq5 = _parse_mq5_inputs()
    preset = _parse_preset_values()
    missing = sorted(set(mq5) - set(preset))
    assert not missing, (
        f"{len(missing)} input(s) de FxMultiSleeve.mq5 absent(s) du preset : "
        f"{missing}. Les ajouter dans write_default_preset.PRESET_LINES."
    )


def test_preset_has_no_input_the_ea_dropped() -> None:
    """L'inverse : un input retiré du .mq5 mais laissé dans le preset fait
    rejeter le chargement par MT5."""
    mq5 = _parse_mq5_inputs()
    preset = _parse_preset_values()
    extra = sorted(set(preset) - set(mq5))
    assert not extra, (
        f"{len(extra)} clé(s) du preset n'existe(nt) plus dans le .mq5 : {extra}"
    )


def test_preset_values_match_compiled_defaults() -> None:
    """Les valeurs, pas seulement les clés — c'est la dérive qui a fait tourner
    des backtests avec les mauvaises allocations pendant trois mois."""
    mq5 = _parse_mq5_inputs()
    preset = _parse_preset_values()

    mismatches: list[str] = []
    for name, (typ, raw) in sorted(mq5.items()):
        if typ in _ENUM_TYPES or name not in preset:
            continue
        got = preset[name]
        if typ == "bool":
            if raw.strip().lower() != got.strip().lower():
                mismatches.append(f"{name}: .mq5={raw} preset={got}")
            continue
        if typ == "string":
            if raw.strip().strip('"') != got.strip().strip('"'):
                mismatches.append(f"{name}: .mq5={raw} preset={got}")
            continue
        want_f, got_f = _as_float(raw), _as_float(got)
        if want_f is None or got_f is None:
            continue  # expression non littérale : hors périmètre
        if abs(want_f - got_f) > 1e-9:
            mismatches.append(f"{name}: .mq5={raw} preset={got}")

    assert not mismatches, (
        "Le preset a dérivé des défauts compilés — MT5 le sert PAR-DESSUS le "
        "binaire, donc ces écarts changent silencieusement tout backtest :\n  "
        + "\n  ".join(mismatches)
    )


def test_allocations_sum_to_one() -> None:
    """CRiskManager::Init rejette toute somme != 1.0 à 1e-6, et l'EA refuse
    alors de démarrer — un backtest à zéro trade, sans autre signal."""
    mq5 = _parse_mq5_inputs()
    allocs = {
        n: _as_float(v) for n, (t, v) in mq5.items() if n.startswith("Inp_Alloc")
    }
    assert allocs, "aucune allocation trouvée dans le .mq5"
    assert all(v is not None for v in allocs.values()), f"non littéral : {allocs}"
    total = sum(allocs.values())  # type: ignore[arg-type]
    assert abs(total - 1.0) <= 1e-6, (
        f"somme des allocations = {total} != 1.0 — CRiskManager::Init "
        f"échouera : {allocs}"
    )
