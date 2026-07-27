#!/usr/bin/env python3
"""Passe la sleeve TSMOM (moteur or) sur chaque instrument, dans le tester MT5.

C'est le CLASSEMENT du screening multi-instruments — le pré-filtre vbt
(`scripts/screen_tsmom_universe.py`) ne sert qu'à écarter les instruments sans
edge structurel. La sanity du 2026-07-27 a montré que même en fill
``next_open`` + coûts + swap-drag, le vbt reste à ~1.08 de Sharpe sur l'or
contre 0.73 au tester : le résidu (sizing en lots, levier non décalé, borne de
décision de l'EA, interpolation M1) n'est pas modélisable proprement côté
recherche. Doctrine du repo : les chiffres qui comptent sont ceux du moteur
qui exécute.

Un run par instrument, configuration or de PRODUCTION inchangée
(lookbacks 40/60/120/250, target_vol 0.55, cap 6.6) : ce balayage ne
sélectionne pas de paramètres, il classe des instruments — le compteur de
trials reste à 1 config × N instruments.

``Inp_Gold_Symbols`` accepte le nom de base : la sleeve tente
``<base>+Inp_SymbolSuffix`` puis retombe sur le nom nu (métaux, énergies et
indices n'ont pas de suffixe chez ce broker).

⚠️ Fenêtre close au 2025-12-31 (``docs/research/HOLDOUT_POLICY.md``). Les
symboles exportés le 2026-07-27 n'ont d'historique broker que depuis
2022-11-04 : leur fenêtre démarre là, et le warmup de 250 séances D1 repousse
le premier trade vers ~2023-11 — ~2,1 ans de trading effectif, à garder en
tête en comparant les CAGR.

Les runs sont séquentiels : un seul terminal64.exe à la fois.

Usage :
    python scripts/sweep_tsmom_mt5.py                # univers complet
    python scripts/sweep_tsmom_mt5.py --only XAGUSD
    python scripts/sweep_tsmom_mt5.py --dry-run
"""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
_CLI = _REPO / "src/mt5/bridge/run_backtest_cli.py"
_TESTER_LOGS = Path(
    "/home/thomas/.mt5/drive_c/Program Files/MetaTrader 5/Tester/logs"
)

CHART_SYMBOL = "EURUSD.c"   # chart de base du tester ; l'EA lit le reste par CopyRates
TO_DATE = "2025.12.31"      # holdout : ne pas étendre sans lire HOLDOUT_POLICY.md
MODEL = 1                   # OHLC M1 — les ticks réels ne sont pas téléchargés

FROM_LEGACY = "2021.01.01"   # 11 symboles avec historique broker profond
FROM_NEW = "2022.11.04"      # symboles exportés le 2026-07-27 (limite demo)

# (valeur de Inp_Gold_Symbols, from_date). Un seul instrument par run : ce
# balayage classe des instruments joués isolément, il ne teste pas de panier.
# Le nom de base suffit — fallback nom nu.
INSTRUMENTS: list[tuple[str, str]] = [
    ("XAUUSD", FROM_LEGACY),
    ("EURUSD", FROM_LEGACY),
    ("GBPUSD", FROM_LEGACY),
    ("USDJPY", FROM_LEGACY),
    ("USDCAD", FROM_LEGACY),
    ("USDCHF", FROM_LEGACY),
    ("AUDUSD", FROM_LEGACY),
    ("NZDUSD", FROM_LEGACY),
    ("EURGBP", FROM_NEW),    # broker : 2022-11-04 aussi pour ces crosses
    ("EURJPY", FROM_NEW),
    ("GBPJPY", FROM_LEGACY),
    ("XAGUSD", FROM_NEW),
    ("XTIUSD", FROM_NEW),
    ("XBRUSD", FROM_NEW),
    ("XNGUSD", FROM_NEW),
    ("US500Cash", FROM_NEW),
    ("US100Cash", FROM_NEW),
    ("US30Cash", FROM_NEW),
    ("GER40Cash", "2022.11.10"),   # l'historique broker GER40 commence le 2022-11-06
    ("JPN225Cash", FROM_NEW),
    ("UK100Cash", FROM_NEW),
]

_OPTIM_RE = re.compile(
    r"\[OPTIM\].*?cagr=(?P<cagr>-?[\d.]+)\s+dd=(?P<dd>-?[\d.]+)\s+"
    r"sharpe=(?P<sharpe>-?[\d.]+)\s+pf=(?P<pf>-?[\d.]+)\s+rf=(?P<rf>-?[\d.]+)\s+"
    r"trades=(?P<trades>\d+)\s+net=(?P<net>-?[\d.]+)"
)


@dataclass
class Result:
    symbol: str
    from_date: str
    cagr: float | None = None
    max_dd: float | None = None
    sharpe: float | None = None
    profit_factor: float | None = None
    trades: int | None = None
    net: float | None = None
    error: str = ""


def _latest_tester_log() -> Path | None:
    logs = sorted(_TESTER_LOGS.glob("*.log"), key=lambda p: p.stat().st_mtime)
    return logs[-1] if logs else None


def _read_optim_line(log: Path) -> dict[str, str] | None:
    """Dernière ligne [OPTIM] du log — le CAGR calculé par l'EA lui-même."""
    try:
        text = log.read_bytes().decode("utf-16-le", errors="ignore")
    except OSError:
        return None
    matches = list(_OPTIM_RE.finditer(text))
    return matches[-1].groupdict() if matches else None


def run_one(symbol: str, from_date: str, deposit: int) -> Result:
    res = Result(symbol=symbol, from_date=from_date)
    tag = f"tsmom_{symbol.lower()}"
    cmd = [
        sys.executable, str(_CLI),
        "--symbol", CHART_SYMBOL, "--from", from_date, "--to", TO_DATE,
        "--model", str(MODEL), "--timeout", "900",
        # 100k par défaut : sur 10k, la granularité de lot (step 0.01, planchers
        # par symbole) écrase le sizing des indices — US500 tournait à 0.3-0.7x
        # de levier effectif pour ~4.5x voulu, JPN225 à lots figés 1.00.
        "--deposit", str(deposit),
        "--report-name", tag, "--ini-name", f"{tag}.ini",
        "--runtime-ini", f"{tag}.ini",
        "--input", "Inp_AllocMRMacro=0",
        "--input", "Inp_AllocTSMomentum=0",
        "--input", "Inp_AllocRSIDaily=0",
        "--input", "Inp_AllocGoldMomentum=1.0",
        "--input", "Inp_EnableDDCap=false",
        # ⚠️ Épingler TOUT le sizing. Inp_RiskScale=4.5 est le défaut compilé
        # depuis le 2026-07-26, calibré pour le portefeuille (or à 10 %) ; en
        # sleeve isolée à Alloc=1.0 il porte le levier effectif à ~27x et ruine
        # le compte (constaté le 2026-07-27 : net -9 981 contre +45 596 à 1.0).
        # Les chiffres de ce balayage sont donc en sizing vol-target pur.
        "--input", "Inp_RiskScale=1.0",
        "--input", f"Inp_Gold_Symbols={symbol}",
        "--input", "Inp_Gold_LookbackA=40",
        "--input", "Inp_Gold_LookbackB=60",
        "--input", "Inp_Gold_LookbackC=120",
        "--input", "Inp_Gold_LookbackD=250",
        "--input", "Inp_Gold_TargetVol=0.55",
        "--input", "Inp_Gold_MaxLeverage=6.6",
        "--input", "Inp_Gold_AllowShort=false",
        "--input", "Inp_Gold_SafetySL=0.04",
        "--input", "Inp_Gold_SlippageBps=2",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1200)
    if proc.returncode != 0:
        res.error = f"exit={proc.returncode}"
        return res

    log = _latest_tester_log()
    if log is None:
        res.error = "no tester log"
        return res
    optim = _read_optim_line(log)
    if optim is None:
        res.error = "no [OPTIM] line"
        return res

    res.cagr = float(optim["cagr"])
    res.max_dd = float(optim["dd"])
    res.sharpe = float(optim["sharpe"])
    res.profit_factor = float(optim["pf"])
    res.trades = int(optim["trades"])
    res.net = float(optim["net"])
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true", help="lister les runs")
    ap.add_argument("--only", help="un seul symbole (nom de base, ex. XAGUSD)")
    ap.add_argument("--out", default="reports/mt5/tsmom_universe_sweep.csv")
    ap.add_argument("--deposit", type=int, default=100_000)
    args = ap.parse_args()

    instruments = INSTRUMENTS
    if args.only:
        instruments = [(s, f) for s, f in INSTRUMENTS if s == args.only]
        if not instruments:
            print(f"symbole inconnu {args.only!r}")
            sys.exit(2)

    print(f"{len(instruments)} instruments — config or de production, "
          f"→ {TO_DATE} (model={MODEL})")
    if args.dry_run:
        for s, f in instruments:
            print(f"  {s:<12} from {f}")
        return

    results: list[Result] = []
    for i, (sym, frm) in enumerate(instruments, 1):
        print(f"[{i}/{len(instruments)}] {sym} (from {frm}) ...", flush=True)
        try:
            r = run_one(sym, frm, args.deposit)
        except subprocess.TimeoutExpired:
            r = Result(symbol=sym, from_date=frm, error="timeout")
        results.append(r)
        if r.error:
            print(f"    ÉCHEC: {r.error}", flush=True)
        else:
            print(f"    CAGR {r.cagr * 100:6.2f}%  dd {r.max_dd:5.2f}%  "
                  f"sharpe {r.sharpe:5.3f}  trades {r.trades}", flush=True)

    out = _REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    write_header = not out.exists() or not args.only
    mode = "a" if args.only and out.exists() else "w"
    with out.open(mode, newline="") as fh:
        w = csv.writer(fh)
        if write_header and mode == "w":
            w.writerow(["symbol", "from_date", "cagr", "max_dd", "sharpe",
                        "profit_factor", "trades", "net", "error"])
        for r in results:
            w.writerow([r.symbol, r.from_date, r.cagr, r.max_dd, r.sharpe,
                        r.profit_factor, r.trades, r.net, r.error])

    ok = [r for r in results if not r.error]
    print(f"\n{len(ok)}/{len(results)} runs exploitables → {out}")
    if not ok:
        return
    print("\nClassement par Sharpe :")
    print(f"{'symbol':<12} {'from':>10} {'CAGR':>8} {'maxDD':>8} "
          f"{'Sharpe':>7} {'trades':>7}")
    for r in sorted(ok, key=lambda x: x.sharpe or -9, reverse=True):
        print(f"{r.symbol:<12} {r.from_date:>10} {r.cagr * 100:>7.2f}% "
              f"{r.max_dd:>7.2f}% {r.sharpe:>7.3f} {r.trades:>7d}")


if __name__ == "__main__":
    main()
