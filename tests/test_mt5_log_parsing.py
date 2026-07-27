"""Le résumé de log d'un run MT5 ne doit décrire que ce run, et un run vide
doit être rejeté.

``Tester/logs/YYYYMMDD.log`` accumule tous les backtests de la journée. Le
filtrage par mtime porte sur le *fichier*, pas sur son contenu : un
``[INIT][ERROR]`` d'un essai abandonné plus tôt remontait donc dans le JSON du
run courant. Le 2026-07-26, le run de référence du portefeuille de production
(812 trades, Sharpe 1.06) a ainsi paru douteux en embarquant deux
``RiskManager init failed`` datant de 16h57 et 16h58 — l'époque où le preset
sommait encore ses allocations à 1.10.

Ces tests bornent l'analyse au dernier démarrage d'EA.

La seconde moitié du fichier porte sur le critère de succès du CLI. Il testait
la présence des *chaînes* extraites du rapport HTML ; or un run dégénéré n'écrit
pas des chaînes vides mais des zéros (`"0"`, `"0.00"`), qui sont non vides. Le
2026-07-26, un run en ticks réels sans historique téléchargé a produit
`Period: M0 (1970.01.01 - 1970.01.01)`, 0 trade — et `EXIT=0` (audit §8). Ces
tests fixent le critère numérique qui le rejette.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_BRIDGE = (
    Path(__file__).resolve().parent.parent
    / "src/mt5/bridge/run_backtest_cli.py"
)


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


# Deux runs concaténés comme MT5 les écrit : le premier échoue à l'init, le
# second est le run utile.
_FAILED_RUN = """\
IH\t0\t16:57:08.782\tCore 01\t  Inp_MacroHistoryUseCommon=true
PO\t0\t16:57:08.782\tCore 01\t2021.01.01 00:00:00   [INIT][INFO] FxMultiSleeve start build
FR\t0\t16:57:08.782\tCore 01\t2021.01.01 00:00:00   CRiskManager::Init: allocations sum=1.1000 != 1.0
DL\t0\t16:57:08.782\tCore 01\t2021.01.01 00:00:00   [INIT][ERROR] RiskManager init failed
DJ\t0\t16:57:08.782\tCore 01\t2021.01.01 00:00:00   [DEINIT][INFO] EA stopped reason=8
"""

_GOOD_RUN = """\
PO\t0\t17:41:02.100\tCore 01\t2021.01.01 00:00:00   [INIT][INFO] FxMultiSleeve start build
FR\t0\t17:41:02.100\tCore 01\t2021.01.01 00:00:00   CMacroSourceHistory: loaded 1400 rows from macro_history.csv
DL\t0\t17:41:02.100\tCore 01\t2021.01.01 00:00:00   [INIT][INFO] EA ready
DJ\t0\t17:41:12.055\tCore 01\t2025.12.28 21:00:17   [DAILY][INFO] Daily recompute done at hour=21 UTC
"""


def test_slice_last_run_drops_the_previous_run():
    sliced = bridge._slice_last_run(_FAILED_RUN + _GOOD_RUN)

    assert "RiskManager init failed" not in sliced
    assert "EA ready" in sliced


def test_slice_last_run_keeps_everything_without_marker():
    """Un échec antérieur à la première ligne d'OnInit doit rester visible."""
    text = "XX\t0\t10:00:00.000\tCore 01\t[INIT][ERROR] terminal introuvable\n"

    assert bridge._slice_last_run(text) == text


def test_parse_tester_log_reports_only_the_current_run(tmp_path):
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    # MT5 écrit ses logs en UTF-16 LE avec BOM.
    (log_dir / "20260726.log").write_bytes(
        (_FAILED_RUN + _GOOD_RUN).encode("utf-16")
    )

    summary = bridge.parse_tester_log(log_dir, since_epoch=0.0)

    assert summary.init_ok is True
    assert summary.init_errors == []
    assert "loaded 1400 rows" in summary.macro_source_resolved


def test_parse_tester_log_still_surfaces_a_real_init_failure(tmp_path):
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    (log_dir / "20260726.log").write_bytes(
        (_GOOD_RUN + _FAILED_RUN).encode("utf-16")
    )

    summary = bridge.parse_tester_log(log_dir, since_epoch=0.0)

    assert summary.init_ok is False
    assert summary.init_errors == ["[INIT][ERROR] RiskManager init failed"]


# ---------------------------------------------------------------------------
# Critère de succès du run (audit 2026-07-26, §8)
# ---------------------------------------------------------------------------


def _metrics(**overrides):
    """Métriques du run de référence du 2026-07-26, surchargeables par cas.

    Les valeurs sont celles que MT5 a réellement écrites (cf.
    `reports/mt5/run_20260726T232429Z.json`), séparateurs de milliers compris.
    """
    base = dict(
        report_path="/tmp/fx_full_backtest_report.htm",
        symbol="EURUSD.c",
        period="M1 (2021.01.01 - 2026.04.30)",
        initial_deposit="10 000.00",
        total_net_profit="40 267.40",
        profit_factor="1.24",
        recovery_factor="2.03",
        sharpe_ratio="0.89",
        total_trades="851",
    )
    base.update(overrides)
    return bridge.HtmlReportMetrics(**base)


# Le rapport écrit par le run en ticks réels sans historique téléchargé.
_DEGENERATE = dict(
    symbol=None,
    period="M0 (1970.01.01 - 1970.01.01)",
    initial_deposit="0",
    total_net_profit="0",
    profit_factor="0.00",
    recovery_factor="0.00",
    sharpe_ratio="0.00",
    total_trades="0",
)


def test_degenerate_run_is_rejected():
    """0 trade + période à l'epoch : le cas que le garde-fou doit attraper."""
    problems = bridge.validate_run(_metrics(**_DEGENERATE))

    assert problems, "un run sans trade ni historique doit être rejeté"
    joined = " | ".join(problems)
    assert "dégénérée" in joined  # la période est nommée
    assert "aucun trade" in joined  # le zéro trade aussi
    assert bridge.resolve_exit_code(0, problems) == 1


def test_normal_run_is_accepted():
    assert bridge.validate_run(_metrics()) == []
    assert bridge.resolve_exit_code(0, []) == 0


def test_zero_sharpe_with_trades_is_accepted():
    """Un Sharpe nul sur un run qui a tradé est un résultat, pas une panne."""
    assert bridge.validate_run(_metrics(sharpe_ratio="0.00")) == []


def test_missing_sharpe_is_rejected():
    problems = bridge.validate_run(_metrics(sharpe_ratio=None))

    assert len(problems) == 1
    assert "Sharpe Ratio" in problems[0]


def test_missing_total_trades_is_rejected():
    problems = bridge.validate_run(_metrics(total_trades=None))

    assert len(problems) == 1
    assert "Total Trades" in problems[0]


def test_missing_period_is_rejected():
    problems = bridge.validate_run(_metrics(period=None))

    assert len(problems) == 1
    assert "période absente" in problems[0]


def test_thousands_separator_does_not_break_the_conversion():
    """MT5 sépare les milliers par une espace, ordinaire ou insécable."""
    assert bridge._to_number("40 267.40") == 40267.40
    assert bridge._to_number("10 000.00") == 10000.0
    assert bridge._to_number("11 211.90 (20.11%)") == 11211.90
    assert bridge._to_number("-1.25") == -1.25
    assert bridge._to_number(None) is None
    assert bridge._to_number("n/a") is None
    # Un compte de trades à 4 chiffres reste positif après nettoyage.
    assert bridge._to_number("1 200") == 1200.0


def test_period_bounds_reads_both_dates():
    assert bridge._period_bounds("M1 (2021.01.01 - 2026.04.30)") == (
        "2021.01.01", "2026.04.30",
    )
    assert bridge._period_bounds("M0 (1970.01.01 - 1970.01.01)") == (
        "1970.01.01", "1970.01.01",
    )
    assert bridge._period_bounds(None) is None
    assert bridge._period_bounds("M1") is None


def test_timeout_and_terminal_failures_keep_their_exit_codes():
    """124 = timeout : le diagnostic du rapport ne doit pas l'écraser."""
    assert bridge.resolve_exit_code(124, ["peu importe"]) == 124
    assert bridge.resolve_exit_code(124, []) == 124
    assert bridge.resolve_exit_code(3, []) == 3


# ---------------------------------------------------------------------------
# Le code de retour de `main()` (le défaut d'origine vivait là)
# ---------------------------------------------------------------------------


_HTML_REPORT = """\
<html><body><table>
<tr><td>Symbol:</td><td><b>{symbol}</b></td></tr>
<tr><td>Period:</td><td><b>{period}</b></td></tr>
<tr><td>Total Trades:</td><td><b>{total_trades}</b></td></tr>
<tr><td>Sharpe Ratio:</td><td><b>{sharpe_ratio}</b></td></tr>
</table></body></html>
"""


def _run_main_on_report(tmp_path, monkeypatch, **fields) -> int:
    """Exécute `main()` sur un rapport HTML de test, sans lancer MT5.

    Tous les chemins du module sont redirigés vers `tmp_path` et `run_terminal`
    est remplacé : aucun terminal ne démarre et aucun artefact partagé de
    `~/.mt5` n'est touché.
    """
    portable = tmp_path / "portable"
    (portable / "Config").mkdir(parents=True)
    drive_c = tmp_path / "drive_c"
    drive_c.mkdir()

    monkeypatch.setattr(bridge, "PORTABLE", portable)
    monkeypatch.setattr(bridge, "DRIVE_C", drive_c)
    monkeypatch.setattr(bridge, "TESTER_LOGS_DIR", tmp_path / "logs")
    monkeypatch.setattr(bridge, "REPORTS_OUT_DIR", tmp_path / "reports")
    # (exit_code, durée) — le terminal a rendu 0, comme sur le run dégénéré réel.
    monkeypatch.setattr(bridge, "run_terminal", lambda ini, timeout: (0, 1.0))

    # MT5 écrit son rapport en UTF-16 LE avec BOM.
    (portable / "rep.htm").write_bytes(_HTML_REPORT.format(**fields).encode("utf-16"))
    monkeypatch.setattr(
        sys, "argv", ["run_backtest_cli.py", "--skip-checks", "--report-name", "rep"]
    )

    return bridge.main()


def test_main_returns_nonzero_on_a_degenerate_report(tmp_path, monkeypatch, capsys):
    """Le run du 2026-07-26 : le terminal rend 0, le rapport ne vaut rien."""
    code = _run_main_on_report(
        tmp_path, monkeypatch,
        symbol="", period="M0 (1970.01.01 - 1970.01.01)",
        total_trades="0", sharpe_ratio="0.00",
    )

    assert code != 0
    assert "aucun trade" in capsys.readouterr().err


def test_main_returns_zero_on_a_real_report(tmp_path, monkeypatch):
    code = _run_main_on_report(
        tmp_path, monkeypatch,
        symbol="EURUSD.c", period="M1 (2021.01.01 - 2026.04.30)",
        total_trades="851", sharpe_ratio="0.89",
    )

    assert code == 0
