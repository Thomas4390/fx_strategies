#!/usr/bin/env python3
"""Balaye les paramètres de la sleeve or **directement dans le tester MT5**.

Pourquoi sur MT5 plutôt que sur vbt : le retune du 2026-07-26, calibré en
Python, rend 42,98 % de CAGR en vbt et ~23,4 % en MT5 sur la même fenêtre.
L'écart vient de la convention d'exécution (`docs/specs/gold_momentum_spec.md`
§6 — vbt décide sur ``close[t]`` et remplit à ce même ``close[t]``, MT5 remplit
à l'ouverture suivante). Calibrer sur vbt optimise donc en partie un artefact.
Le moteur qui décide de la performance réelle est MT5, c'est là qu'il faut
mesurer.

Le balayage est possible sans recompiler depuis que les slots de lookback sont
variables (un slot à 0 est désactivé) — voir ``FxSleeveGoldMomentum.mqh``.

⚠️ Fenêtre de calibrage close au 2025-12-31 : la tranche gelée reste hors du
balayage (``docs/research/HOLDOUT_POLICY.md``). Ne pas la déplacer sans lire
la politique.

Les runs sont **séquentiels** : un seul terminal64.exe peut tourner à la fois.

Usage :
    python scripts/sweep_gold_mt5.py                 # grille par défaut
    python scripts/sweep_gold_mt5.py --dry-run       # liste les combos
    python scripts/sweep_gold_mt5.py --out sweep.csv
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

SYMBOL = "XAUUSD.c"
FROM_DATE = "2021.01.01"
TO_DATE = "2025.12.31"   # holdout: ne pas étendre sans lire HOLDOUT_POLICY.md
MODEL = 1                # 1-min OHLC: ~15 s/run. Le real-ticks (4) bloque sur
                         # la resynchronisation d'historique.

# Grilles de lookbacks. Un 0 final = slot désactivé (3 horizons au lieu de 4).
LOOKBACK_GRIDS: list[tuple[int, int, int, int]] = [
    (40, 60, 120, 250),   # référence historique, avant le retune
    (15, 30, 60, 0),      # retune actuel, calibré sur vbt
    (20, 40, 80, 0),
    (30, 60, 120, 0),
    (10, 20, 40, 0),
    (20, 60, 120, 0),
]
TARGET_VOLS: list[float] = [0.25, 0.45, 0.65]

# Le cap suit la cible au même ratio que la config d'origine (12x), pour que le
# balayage fasse varier le niveau de risque et non la fréquence de saturation.
CAP_RATIO = 12.0

_OPTIM_RE = re.compile(
    r"\[OPTIM\].*?cagr=(?P<cagr>-?[\d.]+)\s+dd=(?P<dd>-?[\d.]+)\s+"
    r"sharpe=(?P<sharpe>-?[\d.]+)\s+pf=(?P<pf>-?[\d.]+)\s+rf=(?P<rf>-?[\d.]+)\s+"
    r"trades=(?P<trades>\d+)\s+net=(?P<net>-?[\d.]+)"
)


@dataclass
class Result:
    lookbacks: str
    target_vol: float
    max_leverage: float
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
    """Le tester écrit une ligne [OPTIM] par passe; on prend la dernière.

    Lue depuis le log plutôt que du rapport HTML parce qu'elle porte le CAGR
    calculé par l'EA lui-même, donc sur la même définition des deux côtés.
    """
    try:
        text = log.read_bytes().decode("utf-16-le", errors="ignore")
    except OSError:
        return None
    matches = list(_OPTIM_RE.finditer(text))
    return matches[-1].groupdict() if matches else None


def run_one(lookbacks: tuple[int, int, int, int], target_vol: float) -> Result:
    cap = round(target_vol * CAP_RATIO, 2)
    label = "/".join(str(x) for x in lookbacks if x > 0)
    res = Result(lookbacks=label, target_vol=target_vol, max_leverage=cap)

    tag = f"gold_sweep_{label.replace('/', '-')}_tv{target_vol:.2f}"
    cmd = [
        sys.executable, str(_CLI),
        "--symbol", SYMBOL, "--from", FROM_DATE, "--to", TO_DATE,
        "--model", str(MODEL), "--timeout", "900",
        "--report-name", tag, "--ini-name", f"{tag}.ini",
        "--runtime-ini", f"{tag}.ini",
        "--input", "Inp_AllocMRMacro=0",
        "--input", "Inp_AllocTSMomentum=0",
        "--input", "Inp_AllocRSIDaily=0",
        "--input", "Inp_AllocGoldMomentum=1.0",
        "--input", "Inp_EnableDDCap=false",
        "--input", f"Inp_Gold_LookbackA={lookbacks[0]}",
        "--input", f"Inp_Gold_LookbackB={lookbacks[1]}",
        "--input", f"Inp_Gold_LookbackC={lookbacks[2]}",
        "--input", f"Inp_Gold_LookbackD={lookbacks[3]}",
        "--input", f"Inp_Gold_TargetVol={target_vol}",
        "--input", f"Inp_Gold_MaxLeverage={cap}",
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
    ap.add_argument("--dry-run", action="store_true", help="lister les combos")
    ap.add_argument("--out", default="reports/mt5/gold_sweep.csv")
    args = ap.parse_args()

    combos = [(lb, tv) for lb in LOOKBACK_GRIDS for tv in TARGET_VOLS]
    print(f"{len(combos)} combinaisons — {SYMBOL} {FROM_DATE} → {TO_DATE} "
          f"(model={MODEL})")
    if args.dry_run:
        for lb, tv in combos:
            label = "/".join(str(x) for x in lb if x > 0)
            print(f"  {label:<18} tv={tv:.2f} cap={tv * CAP_RATIO:.1f}")
        return

    results: list[Result] = []
    for i, (lb, tv) in enumerate(combos, 1):
        label = "/".join(str(x) for x in lb if x > 0)
        print(f"[{i}/{len(combos)}] {label} tv={tv:.2f} ...", flush=True)
        try:
            r = run_one(lb, tv)
        except subprocess.TimeoutExpired:
            r = Result(lookbacks=label, target_vol=tv,
                       max_leverage=tv * CAP_RATIO, error="timeout")
        results.append(r)
        if r.error:
            print(f"    ÉCHEC: {r.error}", flush=True)
        else:
            print(f"    CAGR {r.cagr * 100:6.2f}%  dd {r.max_dd:5.2f}%  "
                  f"sharpe {r.sharpe:5.3f}  trades {r.trades}", flush=True)

    out = _REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["lookbacks", "target_vol", "max_leverage", "cagr", "max_dd",
                    "sharpe", "profit_factor", "trades", "net", "error"])
        for r in results:
            w.writerow([r.lookbacks, r.target_vol, r.max_leverage, r.cagr,
                        r.max_dd, r.sharpe, r.profit_factor, r.trades, r.net,
                        r.error])

    ok = [r for r in results if not r.error]
    print(f"\n{len(ok)}/{len(results)} runs exploitables → {out}")
    if not ok:
        return
    print("\nClassement par CAGR :")
    print(f"{'lookbacks':<18} {'tv':>5} {'CAGR':>8} {'maxDD':>8} "
          f"{'Sharpe':>7} {'trades':>7}")
    for r in sorted(ok, key=lambda x: x.cagr or -9, reverse=True):
        print(f"{r.lookbacks:<18} {r.target_vol:>5.2f} {r.cagr * 100:>7.2f}% "
              f"{r.max_dd:>7.2f}% {r.sharpe:>7.3f} {r.trades:>7d}")


if __name__ == "__main__":
    main()
