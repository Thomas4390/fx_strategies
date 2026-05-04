#!/usr/bin/env python3
"""run_mt5_optimization — pilote l'optimiseur natif MT5 (parallèle multi-cores).

Génère un INI avec `Optimization=1` (slow complete) ou `Optimization=2` (fast
genetic), définit les bornes sur Inp_GlobalTargetVol et Inp_GlobalMaxLeverage,
lance MT5 en mode optimisation. MT5 distribue les combos sur ses agents
locaux (32 cores théoriques) en parallèle. Une fois fini, parse les logs des
agents pour extraire chaque ligne `[OPTIM] vt=… maxlev=… cagr=… dd=…` émise
par `OnTester()` (cf. FxMultiSleeve.mq5).

Usage :
    # Slow complete sur grille fine (toutes les combos)
    python scripts/optimization/run_mt5_optimization.py

    # Fast genetic (converge vers optimum)
    python scripts/optimization/run_mt5_optimization.py --mode genetic

Output : reports/optimization/optim_<ts>.{csv,png,ini}
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
WINEPREFIX = Path("/home/thomas/.mt5")
PORTABLE = WINEPREFIX / "drive_c/Program Files/MetaTrader 5"
TERMINAL = PORTABLE / "terminal64.exe"
DRIVE_C = WINEPREFIX / "drive_c"
EX5_RELATIVE = "fx_strategies\\FxMultiSleeve.ex5"
TESTER_DIR = PORTABLE / "Tester"
COMMON_FILES_ACTIVE = (
    WINEPREFIX / "drive_c/users/thomas/AppData/Roaming/MetaQuotes/Terminal/Common/Files"
)
OPTIM_CSV = COMMON_FILES_ACTIVE / "optim_results.csv"

OUT_DIR = ROOT / "reports/optimization"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# INI generation (UTF-16 LE BOM CRLF)
# ---------------------------------------------------------------------------


def write_utf16_ini(path: Path, content: str) -> None:
    crlf = content.replace("\n", "\r\n")
    if not crlf.endswith("\r\n"):
        crlf += "\r\n"
    path.write_bytes(crlf.encode("utf-16"))


def build_optim_ini(*, mode: int, criterion: int,
                    vt_start: float, vt_step: float, vt_stop: float,
                    lev_start: int, lev_step: int, lev_stop: int,
                    from_date: str, to_date: str,
                    symbol: str, period: str, model: int,
                    deposit: int, leverage: str, currency: str,
                    report_name: str) -> str:
    """Build optimization INI.

    Format input range MT5: `key=value||from||step||to||N` — N=Y inclus dans
    optim, N=N fixe.
    """
    return "\n".join([
        "[Tester]",
        f"Expert={EX5_RELATIVE}",
        f"Symbol={symbol}",
        f"Period={period}",
        f"Model={model}",
        f"Optimization={mode}",
        f"OptimizationCriterion={criterion}",
        f"FromDate={from_date}",
        f"ToDate={to_date}",
        f"Deposit={deposit}",
        f"Currency={currency}",
        f"Leverage={leverage}",
        "Visual=0",
        "ShutdownTerminal=1",
        f"Report={report_name}",
        "ReplaceReport=1",
        "ForwardMode=0",
        "",
        "[TesterInputs]",
        # Inputs cibles : optimisés
        f"Inp_GlobalTargetVol={(vt_start+vt_stop)/2:.4f}||"
        f"{vt_start:.4f}||{vt_step:.4f}||{vt_stop:.4f}||Y",
        f"Inp_GlobalMaxLeverage={(lev_start+lev_stop)/2:.4f}||"
        f"{lev_start:.4f}||{lev_step:.4f}||{lev_stop:.4f}||Y",
        # Inputs fixes (mêmes que défauts compilés)
        "Inp_SymbolSuffix=.c",
        "Inp_MacroSourceMode=4",
        "Inp_LogVerbose=false",
        "Inp_LogToFile=false",
    ]) + "\n"


# ---------------------------------------------------------------------------
# Run MT5 optimization
# ---------------------------------------------------------------------------


def kill_mt5() -> None:
    subprocess.run(["pkill", "-f", "terminal64.exe"], check=False)
    subprocess.run(["pkill", "-f", "metatester64.exe"], check=False)


def list_recent_files(dirs: list[Path], cutoff_epoch: float) -> list[Path]:
    out = []
    for d in dirs:
        if not d.exists():
            continue
        for p in d.rglob("*.log"):
            try:
                if p.stat().st_mtime >= cutoff_epoch - 5:
                    out.append(p)
            except OSError:
                pass
    return out


def read_utf16(path: Path) -> str:
    raw = path.read_bytes()
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return raw.decode("utf-16", errors="replace")
    return raw.decode("utf-16-le", errors="replace")


OPTIM_RE = re.compile(
    r"\[OPTIM\]\s+"
    r"vt=([+-]?\d+(?:\.\d+)?)\s+"
    r"maxlev=([+-]?\d+(?:\.\d+)?)\s+"
    r"volfloor=([+-]?\d+(?:\.\d+)?)\s+"
    r"cagr=([+-]?\d+(?:\.\d+)?)\s+"
    r"dd=([+-]?\d+(?:\.\d+)?)\s+"
    r"sharpe=([+-]?\d+(?:\.\d+)?)\s+"
    r"pf=([+-]?\d+(?:\.\d+)?)\s+"
    r"rf=([+-]?\d+(?:\.\d+)?)\s+"
    r"trades=([+-]?\d+(?:\.\d+)?)\s+"
    r"net=([+-]?\d+(?:\.\d+)?)\s+"
    r"years=([+-]?\d+(?:\.\d+)?)"
)


CSV_COLUMNS = [
    "ts_utc", "target_vol", "max_lev", "vol_floor",
    "cagr", "equity_dd_pct", "sharpe", "profit_factor",
    "recovery_factor", "trades", "net_profit", "years",
]


def parse_optim_csv() -> list[dict]:
    """Lit Common/Files/optim_results.csv écrit par OnTester() de l'EA.

    MT5 FileWrite produit du UTF-16 LE avec BOM. Pas de header (MT5 ne ré-ouvre
    pas en append au début de chaque agent → on hardcode les colonnes).
    """
    if not OPTIM_CSV.exists():
        return []
    try:
        text = OPTIM_CSV.read_text(encoding="utf-16")
    except (OSError, UnicodeDecodeError):
        return []
    if not text.strip():
        return []
    from io import StringIO
    try:
        df = pd.read_csv(StringIO(text), header=None, names=CSV_COLUMNS,
                         skip_blank_lines=True)
    except (pd.errors.ParserError, pd.errors.EmptyDataError):
        return []
    # Si la première ligne est en fait un header (texte au lieu de num)
    try:
        float(df["target_vol"].iloc[0])
    except (ValueError, TypeError):
        df = df.iloc[1:]
    rows = []
    seen = set()
    for _, r in df.iterrows():
        try:
            vt = float(r["target_vol"])
            lev = float(r["max_lev"])
        except (ValueError, TypeError):
            continue
        sig = (round(vt, 5), round(lev, 5))
        if sig in seen:
            continue
        seen.add(sig)
        rows.append({
            "target_vol": vt,
            "max_lev": lev,
            "vol_floor": float(r["vol_floor"]),
            "cagr_pct": float(r["cagr"]) * 100.0,
            "equity_dd_pct": float(r["equity_dd_pct"]),
            "sharpe": float(r["sharpe"]),
            "profit_factor": float(r["profit_factor"]),
            "recovery_factor": float(r["recovery_factor"]),
            "trades": int(float(r["trades"])),
            "net_profit": float(r["net_profit"]),
            "years": float(r["years"]),
        })
    return rows


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_heatmaps(df: pd.DataFrame, png_path: Path, title_suffix: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[warn] matplotlib unavailable", flush=True)
        return

    df["calmar"] = df.apply(
        lambda r: r["cagr_pct"] / abs(r["equity_dd_pct"])
        if r["equity_dd_pct"] not in (None, 0)
        else None, axis=1)

    pivots = {
        "CAGR (%)": df.pivot(index="target_vol", columns="max_lev",
                             values="cagr_pct"),
        "Equity DD (%)": df.pivot(index="target_vol", columns="max_lev",
                                  values="equity_dd_pct"),
        "Sharpe Ratio": df.pivot(index="target_vol", columns="max_lev",
                                 values="sharpe"),
        "Calmar (CAGR/|DD|)": df.pivot(index="target_vol", columns="max_lev",
                                       values="calmar"),
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax, (title, pv) in zip(axes.flat, pivots.items()):
        if pv is None or pv.empty:
            ax.text(0.5, 0.5, "no data", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_title(title)
            continue
        cmap = "RdYlGn"
        im = ax.imshow(pv.values, aspect="auto", cmap=cmap, origin="lower")
        ax.set_xticks(range(len(pv.columns)))
        ax.set_xticklabels([f"{c:g}" for c in pv.columns], rotation=0)
        ax.set_yticks(range(len(pv.index)))
        ax.set_yticklabels([f"{i:.2f}" for i in pv.index])
        ax.set_xlabel("Inp_GlobalMaxLeverage")
        ax.set_ylabel("Inp_GlobalTargetVol")
        ax.set_title(title)
        for ii in range(pv.shape[0]):
            for jj in range(pv.shape[1]):
                v = pv.values[ii, jj]
                if pd.notna(v):
                    ax.text(jj, ii, f"{v:.1f}", ha="center", va="center",
                            fontsize=7, color="black")
        plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)

    plt.suptitle(
        f"FxMultiSleeve — Sensitivity (TargetVol × MaxLev) {title_suffix}",
        fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(png_path, dpi=110, bbox_inches="tight")
    print(f"[ok] heatmap → {png_path}", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=("complete", "genetic"),
                    default="complete",
                    help="complete=1 (slow grille complète), genetic=2 (rapide)")
    ap.add_argument("--vt-start", type=float, default=0.10)
    ap.add_argument("--vt-stop",  type=float, default=0.50)
    ap.add_argument("--vt-step",  type=float, default=0.05)
    ap.add_argument("--lev-start", type=int, default=2)
    ap.add_argument("--lev-stop",  type=int, default=20)
    ap.add_argument("--lev-step",  type=int, default=2)
    ap.add_argument("--from-date", default="2020.11.23")
    ap.add_argument("--to-date",   default="2026.04.30")
    ap.add_argument("--symbol", default="EURUSD.c")
    ap.add_argument("--period", default="M1")
    ap.add_argument("--model",  type=int, default=1,
                    help="0=tick 1=1-min OHLC 2=open 4=real ticks")
    ap.add_argument("--timeout", type=int, default=3600,
                    help="Timeout MT5 en sec (defaut 1h)")
    ap.add_argument("--out-prefix", default=None)
    args = ap.parse_args()

    mode_map = {"complete": 1, "genetic": 2}
    optim_mode = mode_map[args.mode]

    n_vt = int(round((args.vt_stop - args.vt_start) / args.vt_step)) + 1
    n_lev = int(round((args.lev_stop - args.lev_start) / args.lev_step)) + 1
    n_combos_estimated = n_vt * n_lev
    print(f"=== Optim MT5 ({args.mode}) ===")
    print(f"  Window  : {args.from_date} → {args.to_date}")
    print(f"  Symbol  : {args.symbol} {args.period} (model={args.model})")
    print(f"  TargetVol: {args.vt_start} → {args.vt_stop} step {args.vt_step} ({n_vt})")
    print(f"  MaxLev   : {args.lev_start} → {args.lev_stop} step {args.lev_step} ({n_lev})")
    if args.mode == "complete":
        print(f"  Total    : {n_combos_estimated} combos (slow complete)")
    else:
        print(f"  Total    : génétique sur ~{n_combos_estimated} combos")

    # Pre-flight
    out = subprocess.run(["pgrep", "-x", "terminal64.exe"], capture_output=True)
    if out.stdout.strip():
        print(f"[abort] terminal64.exe déjà actif", file=sys.stderr)
        return 2

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    prefix = args.out_prefix or f"optim_{args.mode}_{ts}"
    report_name = f"fx_{prefix}"

    ini_persistent = PORTABLE / "Config" / f"{prefix}.ini"
    ini_runtime = DRIVE_C / "fxopt.ini"

    content = build_optim_ini(
        mode=optim_mode,
        criterion=6,  # Custom max — utilise OnTester() qui retourne CAGR
        vt_start=args.vt_start, vt_step=args.vt_step, vt_stop=args.vt_stop,
        lev_start=args.lev_start, lev_step=args.lev_step, lev_stop=args.lev_stop,
        from_date=args.from_date, to_date=args.to_date,
        symbol=args.symbol, period=args.period, model=args.model,
        deposit=10000, leverage="1:100", currency="USD",
        report_name=report_name,
    )
    write_utf16_ini(ini_persistent, content)
    write_utf16_ini(ini_runtime, content)
    out_ini = OUT_DIR / f"{prefix}.ini"
    out_ini.write_bytes(ini_runtime.read_bytes())
    print(f"[ok] INI written ({ini_persistent.name}, {ini_runtime}, {out_ini.name})")

    # Clear previous CSV so we read only this run's results
    if OPTIM_CSV.exists():
        OPTIM_CSV.unlink()
        print(f"[ok] cleared previous {OPTIM_CSV.name}", flush=True)

    started_epoch = time.time()
    cmd = [
        "env", f"WINEPREFIX={WINEPREFIX}", "wine",
        str(TERMINAL), "/portable", "/config:C:\\fxopt.ini",
    ]
    print(f"\n[run] launching MT5 optimization (timeout={args.timeout}s)")
    print(f"      MT5 will distribute combos across local agents (cores)")
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL)

    last_count = 0
    try:
        deadline = time.monotonic() + args.timeout
        while time.monotonic() < deadline:
            time.sleep(10)
            results = parse_optim_csv()
            n = len(results)
            elapsed = int(time.time() - started_epoch)
            if n > last_count:
                last_one = results[-1]
                print(f"  [t={elapsed:>4}s] {n:>4} combos written "
                      f"(last: vt={last_one['target_vol']:.2f} "
                      f"lev={last_one['max_lev']:.0f} "
                      f"CAGR={last_one['cagr_pct']:+.2f}%)", flush=True)
                last_count = n
            # MT5 termine et le proc Popen exit (ShutdownTerminal=1)
            if proc.poll() is not None:
                print(f"  [t={elapsed:>4}s] MT5 terminated "
                      f"(exit={proc.returncode}, {n} combos collected)",
                      flush=True)
                # Petit délai pour laisser les agents flusher leur CSV
                time.sleep(3)
                break
    finally:
        kill_mt5()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            pass

    # Final parse + dump
    results = parse_optim_csv()
    print(f"\n[done] {len(results)} combos collected from {OPTIM_CSV.name}")
    if not results:
        print(f"[error] no rows in {OPTIM_CSV} — check that EA was recompiled "
              f"with OnTester() FileWrite enabled", file=sys.stderr)
        return 3

    df = pd.DataFrame(results).sort_values(["target_vol", "max_lev"])
    csv_path = OUT_DIR / f"{prefix}.csv"
    df.to_csv(csv_path, index=False)
    print(f"[ok] CSV → {csv_path}")

    png_path = OUT_DIR / f"{prefix}.png"
    plot_heatmaps(df.copy(), png_path,
                  title_suffix=f"({args.from_date}→{args.to_date}, {args.mode})")

    # Top 5 par CAGR / Calmar
    df["calmar"] = df.apply(
        lambda r: r["cagr_pct"] / abs(r["equity_dd_pct"])
        if r["equity_dd_pct"] not in (None, 0) else None, axis=1)

    print(f"\n=== TOP 5 par CAGR ===")
    for _, r in df.sort_values("cagr_pct", ascending=False).head(5).iterrows():
        print(f"  vt={r['target_vol']:.2f} lev={r['max_lev']:.0f}: "
              f"CAGR={r['cagr_pct']:+.2f}% "
              f"DD={r['equity_dd_pct']:+.2f}% "
              f"Sharpe={r['sharpe']:+.2f} "
              f"Calmar={r['calmar']:+.2f}")

    print(f"\n=== TOP 5 par Calmar ===")
    for _, r in df.dropna(subset=["calmar"]).sort_values(
            "calmar", ascending=False).head(5).iterrows():
        print(f"  vt={r['target_vol']:.2f} lev={r['max_lev']:.0f}: "
              f"CAGR={r['cagr_pct']:+.2f}% "
              f"DD={r['equity_dd_pct']:+.2f}% "
              f"Sharpe={r['sharpe']:+.2f} "
              f"Calmar={r['calmar']:+.2f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
