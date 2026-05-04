#!/usr/bin/env python3
"""rsi_vbt_variants — fait varier slippage / fees / leverage côté VBT.

Objectif : voir si en simulant les coûts MT5 dans VBT on retrouve le Sharpe
MT5 isolated (-0.46). Si oui → l'écart est purement coûts. Sinon → reste
à investiguer (sizing, exécution, fenêtre).
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from strategies.rsi_daily import pipeline
from utils import load_fx_data

PAIRS = ["EUR-USD", "GBP-USD", "USD-JPY", "USD-CAD"]
WINDOW_START = "2020-11-23"
WINDOW_END = "2026-04-30"

OUTPUT_DIR = ROOT / "reports/investigations/rsi_daily"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def measure_portfolio(*, slippage=None, fees=None, leverage=None, label=""):
    """Run pipeline sur 4 paires et agrège equal-weight la mean(daily_returns)."""
    daily_per_pair = {}
    total_trades = 0
    for pair in PAIRS:
        _, data = load_fx_data(f"data/{pair}_minute.parquet")
        close_window = data.close.loc[WINDOW_START:WINDOW_END]
        pf, _ = pipeline(
            close_window,
            rsi_period=14,
            oversold=25.0,
            overbought=75.0,
            exit_mid=50.0,
            slippage=slippage,
            fees=fees,
            leverage=leverage,
        )
        daily_per_pair[pair] = pf.daily_returns
        total_trades += int(pf.trades.count())

    rets = pd.concat(daily_per_pair, axis=1).mean(axis=1, skipna=True)
    if rets.std() > 0:
        sharpe = float(rets.mean() / rets.std() * (252.0 ** 0.5))
    else:
        sharpe = float("nan")
    cum = (1.0 + rets.fillna(0.0)).cumprod()
    max_dd = float(((cum / cum.cummax()) - 1.0).min())
    total_ret = float(cum.iloc[-1] - 1.0)
    return {
        "label": label,
        "slippage": slippage,
        "fees": fees,
        "leverage": leverage,
        "trades": total_trades,
        "sharpe": sharpe,
        "max_dd": max_dd,
        "total_return": total_ret,
    }


def main() -> int:
    variants = [
        ("default", dict()),
        ("slippage_10bps", dict(slippage=0.001)),
        ("slippage_30bps", dict(slippage=0.003)),
        ("slippage_50bps", dict(slippage=0.005)),
        ("fees_5bps", dict(fees=0.0005)),
        ("slip10_fees5", dict(slippage=0.001, fees=0.0005)),
        ("leverage_0.25", dict(leverage=0.25)),
        ("leverage_0.05", dict(leverage=0.05)),
    ]
    rows = []
    for name, kwargs in variants:
        print(f"\n  === {name} === {kwargs}", flush=True)
        r = measure_portfolio(label=name, **kwargs)
        rows.append(r)
        print(f"     Sharpe={r['sharpe']:+.3f} Trades={r['trades']:3d} "
              f"DD={r['max_dd']*100:+.2f}% Ret={r['total_return']*100:+.2f}%",
              flush=True)

    out_csv = OUTPUT_DIR / "vbt_variants.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"\n=== {len(rows)} variants → {out_csv} ===")

    print(f"\n{'Variant':<22} {'Sharpe':>8} {'Trades':>7} {'DD%':>8} {'Ret%':>8}")
    print("-" * 55)
    for r in rows:
        print(f"{r['label']:<22} {r['sharpe']:>+8.3f} {r['trades']:>7d} "
              f"{r['max_dd']*100:>+8.2f} {r['total_return']*100:>+8.2f}")
    print("-" * 55)
    print(f"{'(MT5 isolated B)':<22} {-0.46:>+8.3f} {45:>7d} "
          f"{-4.30:>+8.2f} {-2.00:>+8.2f}  ← cible MT5")
    return 0


if __name__ == "__main__":
    sys.exit(main())
