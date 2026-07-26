"""Le résumé de log d'un run MT5 ne doit décrire que ce run.

``Tester/logs/YYYYMMDD.log`` accumule tous les backtests de la journée. Le
filtrage par mtime porte sur le *fichier*, pas sur son contenu : un
``[INIT][ERROR]`` d'un essai abandonné plus tôt remontait donc dans le JSON du
run courant. Le 2026-07-26, le run de référence du portefeuille de production
(812 trades, Sharpe 1.06) a ainsi paru douteux en embarquant deux
``RiskManager init failed`` datant de 16h57 et 16h58 — l'époque où le preset
sommait encore ses allocations à 1.10.

Ces tests bornent l'analyse au dernier démarrage d'EA.
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
