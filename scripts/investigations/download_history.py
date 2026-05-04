#!/usr/bin/env python3
"""download_history — orchestre le pré-téléchargement de l'historique broker MT5.

Pipeline :
  1. Génère un INI MT5 avec [StartUp] qui auto-lance FxDownloadHistory
     sur un chart EURUSD.c D1 dès que MT5 démarre.
  2. Lance Wine MT5 en background.
  3. Monitore le log MT5 live jusqu'à voir « FxDownloadHistory: Done »
     ou timeout (par défaut 15 min).
  4. Tue MT5 proprement.
  5. Re-lance MT5 pour exécuter FxExportRates avec la même fenêtre.
  6. Lance import_mt5_rates.py pour générer les Parquets.

Pourquoi ? — MT5 ne pré-cache pas les bars M1/D1 hors tester ; il faut
forcer le download via CopyRates répété (cf. FxDownloadHistory.mq5).
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

WINEPREFIX = Path("/home/thomas/.mt5")
PORTABLE = WINEPREFIX / "drive_c/Program Files/MetaTrader 5"
TERMINAL = PORTABLE / "terminal64.exe"
DRIVE_C = WINEPREFIX / "drive_c"
LOG_DIR = PORTABLE / "MQL5/logs"


def write_startup_ini(ini_path: Path, script_name: str,
                      symbol: str = "EURUSD.c", period: str = "D1") -> None:
    """Écrit un INI MT5 minimal qui auto-lance un script au démarrage."""
    content = (
        "[Charts]\n"
        "PreloadCharts=1\n"
        "\n"
        "[StartUp]\n"
        "Profile=Default\n"
        f"Script=fx_strategies\\{script_name}\n"
        f"Symbol={symbol}\n"
        f"Period={period}\n"
    )
    crlf = content.replace("\n", "\r\n")
    ini_path.write_bytes(crlf.encode("utf-16"))


def read_log_utf16(log_path: Path) -> str:
    """Lit un log MT5 (UTF-16 LE avec BOM)."""
    if not log_path.exists():
        return ""
    raw = log_path.read_bytes()
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return raw.decode("utf-16")
    return raw.decode("utf-16-le", errors="replace")


def find_today_log() -> Path:
    """Renvoie le chemin du log MT5 du jour."""
    today = datetime.utcnow().strftime("%Y%m%d")
    return LOG_DIR / f"{today}.log"


def run_mt5_with_script(script_name: str, expected_finish_marker: str,
                        timeout: int, since_epoch: float) -> bool:
    """Lance MT5 avec un script auto-startup et attend qu'il finisse.

    Renvoie True si le marker `Done` a été détecté avant le timeout.
    """
    runtime_ini = DRIVE_C / "fxdl.ini"
    write_startup_ini(runtime_ini, script_name)
    print(f"[ini] {runtime_ini} written (script={script_name})", flush=True)

    cmd = [
        "env", f"WINEPREFIX={WINEPREFIX}", "wine",
        str(TERMINAL), "/portable", "/config:C:\\fxdl.ini",
    ]
    print(f"[run] launching MT5 (timeout={timeout}s)...", flush=True)
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL)

    started = time.monotonic()
    log_path = find_today_log()
    found = False
    last_size = 0

    try:
        while time.monotonic() - started < timeout:
            time.sleep(5)
            if not log_path.exists():
                continue
            size = log_path.stat().st_size
            if size == last_size:
                continue
            last_size = size
            text = read_log_utf16(log_path)
            # Filtrer le log à partir du timestamp de lancement
            recent_lines = [ln for ln in text.splitlines()
                            if expected_finish_marker in ln]
            if recent_lines:
                print(f"[ok] finish marker detected:", flush=True)
                for ln in recent_lines[-3:]:
                    print(f"     {ln.strip()[:200]}", flush=True)
                found = True
                break
            # Pulse de progression
            sym_lines = [ln for ln in text.splitlines()[-30:]
                         if "Script" in ln or script_name in ln
                         or "downloading" in ln or "OK" in ln]
            if sym_lines:
                last = sym_lines[-1].strip()
                if last:
                    elapsed = int(time.monotonic() - started)
                    print(f"[t={elapsed:>4}s] {last[:160]}", flush=True)
    finally:
        print("[stop] killing MT5...", flush=True)
        subprocess.run(["pkill", "-f", "terminal64.exe"], check=False)
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            subprocess.run(["pkill", "-9", "-f", "terminal64.exe"], check=False)

    return found


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--download-timeout", type=int, default=900,
                        help="Timeout download (s, défaut 900 = 15 min)")
    parser.add_argument("--export-timeout", type=int, default=300,
                        help="Timeout export (s, défaut 300 = 5 min)")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip download, juste re-export + import")
    parser.add_argument("--skip-export", action="store_true",
                        help="Skip export, juste import")
    parser.add_argument("--skip-import", action="store_true",
                        help="Skip import (juste export)")
    args = parser.parse_args()

    started_epoch = time.time()

    # Pre-flight
    for proc in subprocess.run(["pgrep", "-x", "terminal64.exe"],
                               capture_output=True).stdout.decode().split():
        print(f"[abort] terminal64.exe déjà actif (pid {proc})",
              file=sys.stderr)
        return 2

    # 1) Download history
    if not args.skip_download:
        print("\n=== STEP 1/3 — Force download history ===\n")
        ok = run_mt5_with_script(
            script_name="FxDownloadHistory",
            expected_finish_marker="FxDownloadHistory",
            timeout=args.download_timeout,
            since_epoch=started_epoch,
        )
        # Le script logue "=== Done: N OK, M FAILED ==="
        text = read_log_utf16(find_today_log())
        m = re.search(r"FxDownloadHistory[^\n]*Done: (\d+) OK, (\d+) FAILED", text)
        if m:
            print(f"[summary] download: {m.group(1)} OK, {m.group(2)} FAILED",
                  flush=True)
        elif not ok:
            print("[warn] download timeout — proceeding anyway", flush=True)

    # 2) Export rates
    if not args.skip_export:
        print("\n=== STEP 2/3 — Export to CSV ===\n")
        time.sleep(3)
        ok = run_mt5_with_script(
            script_name="FxExportRates",
            expected_finish_marker="FxExportRates",
            timeout=args.export_timeout,
            since_epoch=time.time(),
        )
        text = read_log_utf16(find_today_log())
        m = re.search(r"FxExportRates[^\n]*Done: (\d+) OK, (\d+) FAILED", text)
        if m:
            print(f"[summary] export: {m.group(1)} OK, {m.group(2)} FAILED",
                  flush=True)

    # 3) Import to Parquet
    if not args.skip_import:
        print("\n=== STEP 3/3 — Import CSV to Parquet ===\n")
        result = subprocess.run(
            ["python3", "src/mt5/bridge/import_mt5_rates.py"],
            cwd=Path(__file__).resolve().parents[2],
            check=False,
        )
        if result.returncode != 0:
            print(f"[fail] import returned {result.returncode}", file=sys.stderr)
            return result.returncode

    print("\n=== Pipeline complete ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
