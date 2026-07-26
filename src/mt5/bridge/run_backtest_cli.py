#!/usr/bin/env python3
"""run_backtest_cli — orchestre un backtest MT5 FxMultiSleeve sur Linux/Wine.

Pipeline :
  1. Sanity checks (.ex5 présent, macro_history.csv couvre la période, pas de
     terminal64.exe actif).
  2. Génère un fichier `.ini` UTF-16 LE BOM + CRLF dans `Config/` (committable
     comme référence) ET le copie dans `drive_c/` à un nom sans espaces — le
     flag `/config:` casse autrement (cf. docs/mt5/14_cli_backtest_linux.md).
  3. Lance `wine terminal64.exe /portable /config:C:\\<file>.ini` avec timeout.
  4. Parse le log Tester (`Tester/logs/YYYYMMDD.log`, UTF-16 LE) pour confirmer
     l'init OK des 3 sleeves et détecter les erreurs critiques.
  5. Parse le rapport HTML (`<install>/<report>.htm`, UTF-16 LE) pour extraire
     Sharpe / MaxDD / Profit Factor / Trades / Net Profit / Recovery Factor.
  6. Dump JSON dans `reports/mt5/run_<timestamp>.json` + résumé markdown stdout.

Usage standard (5.4 ans broker, M1, model 1) :
    python src/mt5/bridge/run_backtest_cli.py

Walk-forward / fenêtre custom :
    python src/mt5/bridge/run_backtest_cli.py --from 2024.01.01 --to 2024.12.31

Voir docs/mt5/14_cli_backtest_linux.md pour le détail Wine/MT5.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Constantes (chemins absolus — voir docs/mt5/14_cli_backtest_linux.md)
# ---------------------------------------------------------------------------

WINEPREFIX = Path("/home/thomas/.mt5")
PORTABLE = WINEPREFIX / "drive_c/Program Files/MetaTrader 5"
TERMINAL = PORTABLE / "terminal64.exe"
DRIVE_C = WINEPREFIX / "drive_c"
EX5_RELATIVE = "fx_strategies\\FxMultiSleeve.ex5"
EX5_ABS = PORTABLE / "MQL5/Experts/fx_strategies/FxMultiSleeve.ex5"
# Note: en mode /portable, MT5 lit FILE_COMMON dans `users/<user>/AppData/...`
# (pas le `Common/Files` à la racine portable, qui sert de cache GUI). On
# vérifie les DEUX et on alerte si l'effectif est stale.
MACRO_HISTORY_CSV_PORTABLE = PORTABLE / "Common/Files/macro_history.csv"
MACRO_HISTORY_CSV_ACTIVE = (
    WINEPREFIX / "drive_c/users/thomas/AppData/Roaming"
    / "MetaQuotes/Terminal/Common/Files/macro_history.csv"
)
TESTER_LOGS_DIR = PORTABLE / "Tester/logs"

REPO_ROOT = Path(__file__).resolve().parents[3]
REPORTS_OUT_DIR = REPO_ROOT / "reports/mt5"

DEFAULT_FROM = "2020.11.23"
DEFAULT_TO = "2026.04.30"
DEFAULT_SYMBOL = "EURUSD.c"
DEFAULT_PERIOD = "M1"
DEFAULT_MODEL = 4   # "every tick based on real ticks" — most rigorous mode.
                    # Other modes (per MT5 documentation):
                    #   0 = every tick (simulated, monotonic-bias risk)
                    #   1 = 1-minute OHLC (interpolated fills, optimistic Sharpe)
                    #   2 = open prices only (rough estimation)
                    #   3 = math calculations (no market data)
                    #   4 = every tick based on real ticks (recommended)
                    # Mode 4 is ~10x slower than mode 1 but reproduces broker
                    # tick order and floating spread at sub-minute resolution.
DEFAULT_REPORT_NAME = "fx_full_backtest_report"
DEFAULT_DEPOSIT = 10000
DEFAULT_LEVERAGE = "1:100"
DEFAULT_CURRENCY = "USD"
DEFAULT_TIMEOUT = 1800  # 30 min — large pour un run M1 sur 5.4 ans

# Default tester inputs written into the [TesterInputs] section. Aligns
# with the EA's compiled defaults; AUTO mode (=4) makes the macro
# filter switch to HISTORY automatically inside the strategy tester.
DEFAULT_TESTER_INPUTS: dict[str, str] = {
    "Inp_SymbolSuffix": ".c",
    "Inp_MacroSourceMode": "4",
    "Inp_LogVerbose": "false",
    "Inp_LogToFile": "true",
}

# ---------------------------------------------------------------------------
# Helpers I/O — encoding UTF-16 LE BOM + CRLF (cf. reset_tester_preset.py:104)
# ---------------------------------------------------------------------------


def write_utf16_le_bom_crlf(path: Path, content: str) -> None:
    """Écrit un fichier MT5-compatible : UTF-16 LE avec BOM, terminaisons CRLF.

    `write_bytes` est obligatoire — `write_text` activerait la translation
    universal-newline qui produit `\\r\\r\\n` (bug 11c5d83 du 2026-04-30).
    """
    text_with_crlf = content.replace("\r\n", "\n").replace("\n", "\r\n")
    if not text_with_crlf.endswith("\r\n"):
        text_with_crlf += "\r\n"
    # `utf-16` (sans LE) émet le BOM \xff\xfe automatiquement.
    path.write_bytes(text_with_crlf.encode("utf-16"))


def read_utf16_safe(path: Path) -> str:
    """Lit un fichier UTF-16 (BOM ou pas), tolère les variantes."""
    raw = path.read_bytes()
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return raw.decode("utf-16")
    # Fallback LE sans BOM (logs MT5 récents)
    try:
        return raw.decode("utf-16-le")
    except UnicodeDecodeError:
        return raw.decode("utf-8", errors="replace")


# ---------------------------------------------------------------------------
# 1) Sanity checks
# ---------------------------------------------------------------------------


def sanity_checks(from_date: str, to_date: str) -> list[str]:
    """Renvoie la liste des problèmes détectés (vide = OK)."""
    issues: list[str] = []

    if not TERMINAL.exists():
        issues.append(f"terminal64.exe absent: {TERMINAL}")
    if not EX5_ABS.exists():
        issues.append(f"FxMultiSleeve.ex5 absent: {EX5_ABS}")

    # MT5 portable lit `FILE_COMMON` depuis le Common Roaming (pas le portable).
    # On vérifie cet emplacement effectif et on resync depuis le portable si
    # nécessaire (le portable sert de copie source, ie. ce que `fx_macro_history.py`
    # écrit).
    if not MACRO_HISTORY_CSV_ACTIVE.exists():
        issues.append(f"macro_history.csv absent: {MACRO_HISTORY_CSV_ACTIVE}")
    else:
        first, last = _csv_date_range(MACRO_HISTORY_CSV_ACTIVE)
        if first and last:
            if first > from_date.replace(".", "-"):
                issues.append(
                    f"macro_history.csv (active) commence à {first}, "
                    f"avant {from_date} requis"
                )
            if last < to_date.replace(".", "-"):
                issues.append(
                    f"macro_history.csv (active) s'arrête à {last}, "
                    f"après {to_date} requis — resync depuis "
                    f"{MACRO_HISTORY_CSV_PORTABLE}"
                )

    # Pas de terminal64.exe actif (sinon le run plantera ou se collera à
    # l'instance existante).
    try:
        out = subprocess.check_output(
            ["pgrep", "-x", "terminal64.exe"], stderr=subprocess.DEVNULL
        )
        if out.strip():
            issues.append(f"terminal64.exe déjà actif (PIDs: {out.decode().strip()})")
    except subprocess.CalledProcessError:
        pass  # rien ne tourne — bon

    return issues


def _csv_date_range(csv: Path) -> tuple[str | None, str | None]:
    """Renvoie (premiere_date, derniere_date) au format ISO YYYY-MM-DD."""
    try:
        with csv.open() as f:
            header = f.readline()  # noqa: F841
            first_line = f.readline().strip()
            for line in f:
                last_line = line.strip()
        first_date = first_line.split(",", 1)[0][:10] if first_line else None
        last_date = last_line.split(",", 1)[0][:10] if last_line else None
        return first_date, last_date
    except (OSError, IndexError):
        return None, None


# ---------------------------------------------------------------------------
# 2) Génération de l'INI
# ---------------------------------------------------------------------------


def build_tester_ini(
    *,
    symbol: str,
    period: str,
    model: int,
    from_date: str,
    to_date: str,
    deposit: int,
    leverage: str,
    currency: str,
    report_name: str,
    inputs: dict[str, str],
) -> str:
    """Construit le contenu textuel d'un tester.ini."""
    tester_lines = [
        "[Tester]",
        f"Expert={EX5_RELATIVE}",
        f"Symbol={symbol}",
        f"Period={period}",
        f"Model={model}",
        # Spread=0 instructs the strategy tester to use the historical
        # floating spread embedded in the ticks (Model=4) or in each
        # bar (Model=1). Without this line MT5 defaults to "current
        # spread", freezing the live terminal spread across the full
        # backtest window which is unrealistic for multi-year runs.
        "Spread=0",
        f"FromDate={from_date}",
        f"ToDate={to_date}",
        f"Deposit={deposit}",
        f"Currency={currency}",
        f"Leverage={leverage}",
        "Optimization=0",
        "Visual=0",
        "ShutdownTerminal=1",
        f"Report={report_name}",
        "ReplaceReport=1",
        "",
        "[TesterInputs]",
    ]
    for key, value in inputs.items():
        tester_lines.append(f"{key}={value}")
    return "\n".join(tester_lines)


def write_tester_ini(
    persistent_ini: Path,
    runtime_ini: Path,
    **params,
) -> None:
    """Écrit l'INI à 2 emplacements : `Config/` (committable) + `drive_c/`
    (chemin sans espace, requis par `/config:`)."""
    content = build_tester_ini(**params)
    persistent_ini.parent.mkdir(parents=True, exist_ok=True)
    write_utf16_le_bom_crlf(persistent_ini, content)
    write_utf16_le_bom_crlf(runtime_ini, content)


# ---------------------------------------------------------------------------
# 3) Lancement Wine
# ---------------------------------------------------------------------------


def run_terminal(runtime_ini: Path, timeout: int) -> tuple[int, float]:
    """Lance MT5 via Wine. Renvoie (exit_code, durée_sec)."""
    wine_config_path = "C:\\" + runtime_ini.name
    cmd = [
        "env",
        f"WINEPREFIX={WINEPREFIX}",
        "wine",
        str(TERMINAL),
        "/portable",
        f"/config:{wine_config_path}",
    ]
    print(f"[run] {' '.join(cmd[:3])} ... /config:{wine_config_path}", flush=True)
    started = time.monotonic()
    try:
        result = subprocess.run(
            cmd, timeout=timeout, capture_output=True, check=False
        )
        return result.returncode, time.monotonic() - started
    except subprocess.TimeoutExpired:
        elapsed = time.monotonic() - started
        print(f"[run] TIMEOUT après {elapsed:.1f}s — pkill terminal64.exe", flush=True)
        subprocess.run(["pkill", "-f", "terminal64.exe"], check=False)
        return 124, elapsed


# ---------------------------------------------------------------------------
# 4) Parsing log Tester
# ---------------------------------------------------------------------------


@dataclass
class TesterLogSummary:
    log_path: str | None = None
    init_ok: bool = False
    init_errors: list[str] = field(default_factory=list)
    macro_source_resolved: str | None = None
    raw_tail: str = ""


# Première ligne écrite par OnInit() : elle borne le début du run courant dans un
# `YYYYMMDD.log` qui en accumule des dizaines.
_RUN_START_MARKER = r"\[INIT\]\[INFO\] FxMultiSleeve start build"


def _slice_last_run(text: str) -> str:
    """Ne garder que la portion du log postérieure au dernier démarrage d'EA.

    Filtrer les *fichiers* par mtime ne suffit pas : le log du jour accumule tous
    les runs, si bien qu'un `[INIT][ERROR]` d'un essai abandonné trois quarts
    d'heure plus tôt était rapporté comme un défaut du run courant. C'est ce qui
    a fait passer le run de référence du 2026-07-26 pour douteux alors qu'il
    était propre.
    """
    starts = list(re.finditer(_RUN_START_MARKER, text))
    return text[starts[-1].start():] if starts else text


def parse_tester_log(log_dir: Path, since_epoch: float) -> TesterLogSummary:
    """Parse le log Tester du jour. `since_epoch` = on ignore les logs antérieurs
    au lancement courant pour éviter de remonter un run précédent."""
    summary = TesterLogSummary()
    if not log_dir.exists():
        return summary

    candidates = sorted(
        (p for p in log_dir.glob("*.log") if p.stat().st_mtime >= since_epoch - 2),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        return summary

    log_path = candidates[0]
    summary.log_path = str(log_path)
    text = _slice_last_run(read_utf16_safe(log_path))

    if "[INIT][INFO] EA ready" in text:
        summary.init_ok = True
    for match in re.finditer(r"\[INIT\]\[ERROR\][^\n]+", text):
        summary.init_errors.append(match.group(0).strip())
    macro_matches = (
        list(re.finditer(r"Macro source resolved=\w+", text))
        or list(re.finditer(
            r"CMacroSourceHistory: loaded \d+ rows[^\n]*", text
        ))
    )
    if macro_matches:
        summary.macro_source_resolved = macro_matches[-1].group(0).strip()
    summary.raw_tail = "\n".join(text.splitlines()[-40:])
    return summary


# ---------------------------------------------------------------------------
# 5) Parsing du rapport HTML
# ---------------------------------------------------------------------------


@dataclass
class HtmlReportMetrics:
    report_path: str | None = None
    symbol: str | None = None
    period: str | None = None
    initial_deposit: str | None = None
    total_net_profit: str | None = None
    profit_factor: str | None = None
    recovery_factor: str | None = None
    sharpe_ratio: str | None = None
    total_trades: str | None = None
    balance_dd_max: str | None = None
    equity_dd_max: str | None = None
    short_trades_won_pct: str | None = None
    long_trades_won_pct: str | None = None


# Regex unique : `<td …>Label:</td> <td …><b>Value</b></td>`. Robuste aux
# colspan, attributs supplémentaires, et bold optionnel.
_HTML_FIELD_RE_TEMPLATE = (
    r"{label}:?</td>\s*<td[^>]*>(?:<b>)?([^<]+?)(?:</b>)?</td>"
)


def _extract(text: str, label: str) -> str | None:
    pattern = _HTML_FIELD_RE_TEMPLATE.format(label=re.escape(label))
    m = re.search(pattern, text)
    return m.group(1).strip() if m else None


def parse_html_report(report_path: Path) -> HtmlReportMetrics:
    metrics = HtmlReportMetrics(report_path=str(report_path))
    if not report_path.exists():
        return metrics
    text = read_utf16_safe(report_path)

    metrics.symbol = _extract(text, "Symbol")
    metrics.period = _extract(text, "Period")
    metrics.initial_deposit = _extract(text, "Initial Deposit")
    metrics.total_net_profit = _extract(text, "Total Net Profit")
    metrics.profit_factor = _extract(text, "Profit Factor")
    metrics.recovery_factor = _extract(text, "Recovery Factor")
    metrics.sharpe_ratio = _extract(text, "Sharpe Ratio")
    metrics.total_trades = _extract(text, "Total Trades")
    metrics.balance_dd_max = _extract(text, "Balance Drawdown Maximal")
    metrics.equity_dd_max = _extract(text, "Equity Drawdown Maximal")
    metrics.short_trades_won_pct = _extract(text, r"Short Trades \(won %\)")
    metrics.long_trades_won_pct = _extract(text, r"Long Trades \(won %\)")
    return metrics


# ---------------------------------------------------------------------------
# 6) Reporting
# ---------------------------------------------------------------------------


def render_summary(metrics: HtmlReportMetrics, log_summary: TesterLogSummary,
                   exit_code: int, duration_sec: float) -> str:
    lines = [
        "",
        "## Backtest MT5 FxMultiSleeve — résumé",
        "",
        f"- Symbol     : {metrics.symbol or 'n/a'}",
        f"- Period     : {metrics.period or 'n/a'}",
        f"- Deposit    : {metrics.initial_deposit or 'n/a'}",
        f"- Durée run  : {duration_sec:.1f}s (exit={exit_code})",
        f"- Init EA    : {'OK' if log_summary.init_ok else 'FAIL'}"
        + (f" ({len(log_summary.init_errors)} errors)"
           if log_summary.init_errors else ""),
        f"- Macro src  : {log_summary.macro_source_resolved or 'n/a'}",
        "",
        "| Métrique | Valeur |",
        "|---|---|",
        f"| Sharpe Ratio       | {metrics.sharpe_ratio or 'n/a'} |",
        f"| Total Net Profit   | {metrics.total_net_profit or 'n/a'} |",
        f"| Profit Factor      | {metrics.profit_factor or 'n/a'} |",
        f"| Recovery Factor    | {metrics.recovery_factor or 'n/a'} |",
        f"| Total Trades       | {metrics.total_trades or 'n/a'} |",
        f"| Equity DD Max      | {metrics.equity_dd_max or 'n/a'} |",
        f"| Balance DD Max     | {metrics.balance_dd_max or 'n/a'} |",
        f"| Long won %         | {metrics.long_trades_won_pct or 'n/a'} |",
        f"| Short won %        | {metrics.short_trades_won_pct or 'n/a'} |",
        "",
        f"Rapport HTML : {metrics.report_path}",
        f"Log Tester   : {log_summary.log_path}",
    ]
    return "\n".join(lines)


def dump_json(out_dir: Path, metrics: HtmlReportMetrics,
              log_summary: TesterLogSummary, exit_code: int,
              duration_sec: float, ini_path: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = out_dir / f"run_{timestamp}.json"
    payload = {
        "run_id": timestamp,
        "exit_code": exit_code,
        "duration_sec": duration_sec,
        "ini_path": str(ini_path),
        "metrics": asdict(metrics),
        "log_summary": asdict(log_summary),
    }
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    return out_path


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--from", dest="from_date", default=DEFAULT_FROM,
                   help=f"Date début (YYYY.MM.DD, défaut {DEFAULT_FROM})")
    p.add_argument("--to", dest="to_date", default=DEFAULT_TO,
                   help=f"Date fin (YYYY.MM.DD, défaut {DEFAULT_TO})")
    p.add_argument("--symbol", default=DEFAULT_SYMBOL)
    p.add_argument("--period", default=DEFAULT_PERIOD)
    p.add_argument("--model", type=int, default=DEFAULT_MODEL,
                   help="0=Every tick (simulated), 1=1-min OHLC (interpolation, "
                        "surestime), 2=Open prices only, 3=Math calc, "
                        "4=Real ticks (recommandé, rigoureux)")
    p.add_argument("--deposit", type=int, default=DEFAULT_DEPOSIT)
    p.add_argument("--leverage", default=DEFAULT_LEVERAGE)
    p.add_argument("--currency", default=DEFAULT_CURRENCY)
    p.add_argument("--report-name", default=DEFAULT_REPORT_NAME)
    p.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT,
                   help=f"Timeout subprocess en secondes (défaut {DEFAULT_TIMEOUT})")
    p.add_argument("--ini-name", default="fx_full_backtest.ini",
                   help="Nom de l'INI persistant écrit dans Config/")
    p.add_argument("--runtime-ini", default="fxbk.ini",
                   help="Nom de l'INI de runtime copié dans drive_c/ (sans espace)")
    p.add_argument("--input", action="append", default=[], metavar="KEY=VAL",
                   dest="input_overrides",
                   help="Override d'un input EA (répétable). "
                        "Ex: --input Inp_AllocMRMacro=0 --input Inp_AllocRSIDaily=1.0")
    p.add_argument("--skip-checks", action="store_true",
                   help="Ignorer les sanity checks pré-run")
    p.add_argument("--dry-run", action="store_true",
                   help="Génère l'INI sans lancer le tester")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    print(f"[start] FxMultiSleeve backtest {args.from_date} → {args.to_date} "
          f"on {args.symbol} {args.period}", flush=True)

    if not args.skip_checks:
        issues = sanity_checks(args.from_date, args.to_date)
        if issues:
            print("[abort] Sanity checks failed:", file=sys.stderr)
            for msg in issues:
                print(f"  - {msg}", file=sys.stderr)
            return 2
        print("[ok] sanity checks passed", flush=True)

    persistent_ini = PORTABLE / "Config" / args.ini_name
    runtime_ini = DRIVE_C / args.runtime_ini

    # Merge inputs default + overrides
    inputs = dict(DEFAULT_TESTER_INPUTS)
    for override in args.input_overrides:
        if "=" not in override:
            print(f"[abort] --input invalide (format KEY=VAL): {override!r}",
                  file=sys.stderr)
            return 2
        key, value = override.split("=", 1)
        inputs[key.strip()] = value.strip()
    if args.input_overrides:
        print(f"[ok] {len(args.input_overrides)} input override(s) appliqué(s)",
              flush=True)

    write_tester_ini(
        persistent_ini=persistent_ini,
        runtime_ini=runtime_ini,
        symbol=args.symbol,
        period=args.period,
        model=args.model,
        from_date=args.from_date,
        to_date=args.to_date,
        deposit=args.deposit,
        leverage=args.leverage,
        currency=args.currency,
        report_name=args.report_name,
        inputs=inputs,
    )
    print(f"[ok] tester.ini écrit (persistent={persistent_ini.name}, "
          f"runtime={runtime_ini})", flush=True)

    if args.dry_run:
        print("[dry-run] terminé sans lancer le tester")
        return 0

    started_epoch = time.time()
    exit_code, duration_sec = run_terminal(runtime_ini, args.timeout)

    log_summary = parse_tester_log(TESTER_LOGS_DIR, started_epoch)

    report_htm = PORTABLE / f"{args.report_name}.htm"
    metrics = parse_html_report(report_htm)

    summary_md = render_summary(metrics, log_summary, exit_code, duration_sec)
    print(summary_md, flush=True)

    json_path = dump_json(REPORTS_OUT_DIR, metrics, log_summary,
                          exit_code, duration_sec, persistent_ini)
    print(f"[ok] JSON dump → {json_path}", flush=True)

    has_metrics = bool(metrics.sharpe_ratio and metrics.total_trades)
    return 0 if exit_code == 0 and has_metrics else max(1, exit_code)


if __name__ == "__main__":
    sys.exit(main())
