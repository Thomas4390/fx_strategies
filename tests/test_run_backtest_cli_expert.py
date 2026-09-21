"""Le lanceur de backtests MT5 doit pouvoir piloter un second EA sans bouger d'un
octet son comportement par défaut.

`run_backtest_cli.py` codait en dur l'EA de la stratégie 1 : nom du `.ex5`,
inputs par défaut, marqueur de début de run dans le log, dossier de dump JSON.
La stratégie 2 (XAUUSD niveaux x10) a son propre EA, `XauX10`, et doit être
lancée et parsée par le même outil.

Le premier test est le garde-fou du refactor : l'INI produit par les défauts
historiques est figé dans `tests/snapshots/mt5_ini_fxmultisleeve.txt`, capturé
AVANT toute édition du module. Toute dérive du profil FxMultiSleeve — un input
renommé, une ligne `[Tester]` déplacée — rougit ici, pas trois semaines plus
tard dans un backtest de référence.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_BRIDGE = (
    Path(__file__).resolve().parent.parent
    / "src/mt5/bridge/run_backtest_cli.py"
)
_SNAPSHOT = Path(__file__).resolve().parent / "snapshots/mt5_ini_fxmultisleeve.txt"


def _load_bridge():
    """Le bridge n'est pas un package importable : il est lancé en sous-processus."""
    if not _BRIDGE.exists():  # pragma: no cover - dépend du checkout
        pytest.skip(f"{_BRIDGE} absent")
    spec = importlib.util.spec_from_file_location("run_backtest_cli", _BRIDGE)
    module = importlib.util.module_from_spec(spec)
    # @dataclass résout ses annotations via sys.modules[cls.__module__].
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


bridge = _load_bridge()


def _default_ini() -> str:
    """L'INI du run standard : que des défauts, aucun override."""
    return bridge.build_tester_ini(
        symbol=bridge.DEFAULT_SYMBOL,
        period=bridge.DEFAULT_PERIOD,
        model=bridge.DEFAULT_MODEL,
        from_date=bridge.DEFAULT_FROM,
        to_date=bridge.DEFAULT_TO,
        deposit=bridge.DEFAULT_DEPOSIT,
        leverage=bridge.DEFAULT_LEVERAGE,
        currency=bridge.DEFAULT_CURRENCY,
        report_name=bridge.DEFAULT_REPORT_NAME,
        inputs=dict(bridge.DEFAULT_TESTER_INPUTS),
    )


# ---------------------------------------------------------------------------
# Non-régression à l'octet du profil par défaut
# ---------------------------------------------------------------------------


def test_default_ini_is_byte_identical_to_the_snapshot():
    assert _default_ini() == _SNAPSHOT.read_text(encoding="utf-8")


def test_module_constants_still_describe_the_default_expert():
    """Constantes historiques conservées : des scripts externes s'y réfèrent."""
    profile = bridge.EXPERTS[bridge.DEFAULT_EXPERT]

    assert bridge.EX5_RELATIVE == profile.ex5_relative == (
        "fx_strategies\\FxMultiSleeve.ex5"
    )
    assert bridge.EX5_ABS.name == "FxMultiSleeve.ex5"
    assert bridge._RUN_START_MARKER == profile.start_marker
    assert bridge.REPORTS_OUT_DIR == profile.out_dir
    assert bridge.DEFAULT_TESTER_INPUTS == profile.default_inputs


# ---------------------------------------------------------------------------
# Profil XauX10
# ---------------------------------------------------------------------------


def _x10_ini(**overrides) -> str:
    profile = bridge.EXPERTS["XauX10"]
    params = dict(
        symbol=profile.default_symbol,
        period=profile.default_period,
        model=bridge.DEFAULT_MODEL,
        from_date=bridge.DEFAULT_FROM,
        to_date=bridge.DEFAULT_TO,
        deposit=bridge.DEFAULT_DEPOSIT,
        leverage=bridge.DEFAULT_LEVERAGE,
        currency=bridge.DEFAULT_CURRENCY,
        report_name=bridge.DEFAULT_REPORT_NAME,
        inputs=dict(profile.default_inputs),
        profile=profile,
    )
    params.update(overrides)
    return bridge.build_tester_ini(**params)


def test_xaux10_ini_points_at_its_own_ex5_and_market():
    ini = _x10_ini()

    assert "Expert=fx_strategies\\XauX10.ex5" in ini
    assert "Symbol=XAUUSD.c" in ini
    assert "Period=M5" in ini


def test_xaux10_ini_carries_no_fxmultisleeve_input():
    """Un input inconnu de l'EA fait échouer le chargement du preset côté MT5."""
    ini = _x10_ini()

    assert "Inp_MacroSourceMode" not in ini
    assert "Inp_LogVerbose" not in ini
    assert "FxMultiSleeve" not in ini


# ---------------------------------------------------------------------------
# Découpage du log par EA
# ---------------------------------------------------------------------------


def _run(marker_line: str, tag: str) -> str:
    return (
        f"PO\t0\t17:41:02.100\tCore 01\t2021.01.01 00:00:00   {marker_line}\n"
        f"DL\t0\t17:41:02.100\tCore 01\t2021.01.01 00:00:00   [INIT][INFO] EA ready\n"
        f"DJ\t0\t17:41:12.055\tCore 01\t2025.12.28 21:00:17   marqueur={tag}\n"
    )


_X10_RUN_1 = _run("[INIT][INFO] XauX10 start build 1000", "x10_premier")
_X10_RUN_2 = _run("[INIT][INFO] XauX10 start build 1000", "x10_dernier")
_SLEEVE_RUN = _run("[INIT][INFO] FxMultiSleeve start build 900", "sleeve")

_MIXED_LOG = _X10_RUN_1 + _X10_RUN_2 + _SLEEVE_RUN


def test_slice_last_run_follows_the_requested_expert():
    sliced = bridge._slice_last_run(_MIXED_LOG, bridge.EXPERTS["XauX10"])

    assert "marqueur=x10_dernier" in sliced
    assert "marqueur=x10_premier" not in sliced
    # Le run FxMultiSleeve qui suit reste dans la tranche : on ne coupe qu'en
    # amont, comme le fait déjà le profil par défaut.
    assert "marqueur=sleeve" in sliced


def test_slice_last_run_defaults_to_fxmultisleeve():
    sliced = bridge._slice_last_run(_MIXED_LOG)

    assert "marqueur=sleeve" in sliced
    assert "marqueur=x10_dernier" not in sliced


def test_parse_tester_log_reads_the_x10_run(tmp_path):
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    # MT5 écrit ses logs en UTF-16 LE avec BOM.
    (log_dir / "20260921.log").write_bytes(
        (_X10_RUN_1 + _X10_RUN_2).encode("utf-16")
    )

    summary = bridge.parse_tester_log(
        log_dir, since_epoch=0.0, profile=bridge.EXPERTS["XauX10"]
    )

    assert summary.init_ok is True
    assert summary.init_errors == []
    assert "marqueur=x10_dernier" in summary.raw_tail


# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------


def _fake_install(tmp_path, monkeypatch, ex5_relative: str) -> None:
    """Installe un faux portable MT5 : ni Wine ni MT5 requis."""
    portable = tmp_path / "portable"
    terminal = portable / "terminal64.exe"
    ex5 = portable / "MQL5/Experts" / ex5_relative.replace("\\", "/")
    ex5.parent.mkdir(parents=True, exist_ok=True)
    ex5.write_bytes(b"")
    terminal.write_bytes(b"")
    monkeypatch.setattr(bridge, "PORTABLE", portable)
    monkeypatch.setattr(bridge, "TERMINAL", terminal)
    monkeypatch.setattr(
        bridge, "MACRO_HISTORY_CSV_ACTIVE", tmp_path / "absent/macro_history.csv"
    )


def test_sanity_checks_does_not_require_macro_history_for_x10(tmp_path, monkeypatch):
    profile = bridge.EXPERTS["XauX10"]
    _fake_install(tmp_path, monkeypatch, profile.ex5_relative)

    issues = bridge.sanity_checks("2020.11.23", "2026.04.30", profile)

    assert not [msg for msg in issues if "macro_history" in msg]
    assert not [msg for msg in issues if "XauX10.ex5 absent" in msg]


def test_sanity_checks_still_requires_macro_history_by_default(tmp_path, monkeypatch):
    _fake_install(tmp_path, monkeypatch, bridge.EX5_RELATIVE)

    issues = bridge.sanity_checks("2020.11.23", "2026.04.30")

    assert [msg for msg in issues if "macro_history.csv absent" in msg]


# ---------------------------------------------------------------------------
# Dump JSON
# ---------------------------------------------------------------------------


def test_dump_json_records_the_expert(tmp_path):
    profile = bridge.EXPERTS["XauX10"]
    out_dir = tmp_path / "mt5_x10"

    out_path = bridge.dump_json(
        out_dir,
        bridge.HtmlReportMetrics(),
        bridge.TesterLogSummary(),
        exit_code=0,
        duration_sec=1.0,
        ini_path=tmp_path / "x10.ini",
        profile=profile,
    )

    assert out_path.parent == out_dir
    assert json.loads(out_path.read_text())["expert"] == "XauX10"


def test_dump_json_keeps_the_default_expert_without_the_argument(tmp_path):
    out_path = bridge.dump_json(
        tmp_path, bridge.HtmlReportMetrics(), bridge.TesterLogSummary(),
        0, 1.0, tmp_path / "fx.ini",
    )

    assert json.loads(out_path.read_text())["expert"] == "FxMultiSleeve"


def test_x10_profile_dumps_outside_the_strategy_1_reports():
    assert bridge.EXPERTS["XauX10"].out_dir == bridge.REPO_ROOT / "reports/mt5_x10"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse(monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["run_backtest_cli.py", *argv])
    return bridge.parse_args()


def test_expert_flag_pulls_the_profile_symbol_and_period(monkeypatch):
    args = _parse(monkeypatch, "--expert", "XauX10")

    assert args.expert == "XauX10"
    assert args.symbol == "XAUUSD.c"
    assert args.period == "M5"


def test_no_expert_flag_keeps_the_historical_defaults(monkeypatch):
    args = _parse(monkeypatch)

    assert args.expert == "FxMultiSleeve"
    assert args.symbol == bridge.DEFAULT_SYMBOL == "EURUSD.c"
    assert args.period == bridge.DEFAULT_PERIOD == "M1"


def test_explicit_symbol_wins_over_the_profile(monkeypatch):
    args = _parse(monkeypatch, "--expert", "XauX10", "--symbol", "XAUUSD", "--period",
                  "M15")

    assert (args.symbol, args.period) == ("XAUUSD", "M15")


def test_unknown_expert_is_refused(monkeypatch):
    with pytest.raises(SystemExit):
        _parse(monkeypatch, "--expert", "Inconnu")


def _dry_run_ini(tmp_path, monkeypatch, *argv) -> str:
    """Lance `main()` en `--dry-run` : l'INI est écrit, aucun terminal ne démarre."""
    portable = tmp_path / "portable"
    (portable / "Config").mkdir(parents=True)
    drive_c = tmp_path / "drive_c"
    drive_c.mkdir()
    monkeypatch.setattr(bridge, "PORTABLE", portable)
    monkeypatch.setattr(bridge, "DRIVE_C", drive_c)
    monkeypatch.setattr(
        sys, "argv", ["run_backtest_cli.py", "--skip-checks", "--dry-run", *argv]
    )

    assert bridge.main() == 0
    return (portable / "Config/fx_full_backtest.ini").read_bytes().decode("utf-16")


def test_dry_run_default_ini_matches_the_snapshot(tmp_path, monkeypatch):
    written = _dry_run_ini(tmp_path, monkeypatch)

    # L'écriture MT5 ajoute le CRLF et le BOM ; le contenu logique, lui, est figé.
    assert written.replace("\r\n", "\n").rstrip("\n") == _SNAPSHOT.read_text(
        encoding="utf-8"
    )


def test_dry_run_x10_ini_honours_an_input_override(tmp_path, monkeypatch):
    written = _dry_run_ini(
        tmp_path, monkeypatch,
        "--expert", "XauX10", "--input", "Inp_SymbolSuffix=",
        "--input", "Inp_Levels=10",
    )

    assert "Expert=fx_strategies\\XauX10.ex5" in written
    assert "Symbol=XAUUSD.c" in written
    assert "Period=M5" in written
    assert "Inp_SymbolSuffix=\r\n" in written  # le défaut ".c" a bien été écrasé
    assert "Inp_Levels=10" in written
