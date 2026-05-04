#!/usr/bin/env python3
"""rsi_baseline — reproduit le backtest VBT RSI Daily 4-pair par paire.

Phase 0 de l'investigation `docs/investigations/rsi_daily_vbt_vs_mt5.md`.
Pour chaque paire (EUR-USD, GBP-USD, USD-JPY, USD-CAD) :
  - charge `data/<PAIR>_minute.parquet` (Dukascopy)
  - run `strategies.rsi_daily.pipeline()` avec les defaults compilés
  - extrait Sharpe / Total trades / Max DD / Net Return / Period

Sortie : `reports/investigations/rsi_daily/baseline_vbt.csv`.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ajout du src/ au PYTHONPATH
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd
import vectorbtpro as vbt

from strategies.rsi_daily import pipeline
from utils import load_fx_data

PAIRS = ["EUR-USD", "GBP-USD", "USD-JPY", "USD-CAD"]
WINDOW_START = "2020-11-23"
WINDOW_END = "2026-04-30"

OUTPUT_DIR = ROOT / "reports/investigations/rsi_daily"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def measure_pair(pair: str) -> dict:
    """Run pipeline on one pair, return a dict of metrics."""
    print(f"  [{pair}] loading data...", flush=True)
    _, data = load_fx_data(f"data/{pair}_minute.parquet")

    # Restreindre à la fenêtre commune avec MT5 broker (5.4 ans)
    close_full = data.close
    close_window = close_full.loc[WINDOW_START:WINDOW_END]

    print(f"  [{pair}] running pipeline (rsi_period=14, OS=25, OB=75)...", flush=True)
    pf, ind = pipeline(
        close_window,
        rsi_period=14,
        oversold=25.0,
        overbought=75.0,
        exit_mid=50.0,
    )

    daily_rets = pf.daily_returns
    metrics = {
        "pair": pair,
        "data_source": "dukascopy_minute",
        "window_start": str(close_window.index.min().date()),
        "window_end": str(close_window.index.max().date()),
        "n_days_used": len(daily_rets),
        "total_trades": int(pf.trades.count()),
        "sharpe_ratio": float(pf.sharpe_ratio),
        "annualized_return": float(pf.annualized_return),
        "max_drawdown": float(pf.max_drawdown),
        "total_return": float(pf.total_return),
        "win_rate": float(pf.trades.win_rate)
            if pf.trades.count() > 0 else float("nan"),
        "profit_factor": float(pf.trades.profit_factor)
            if pf.trades.count() > 0 else float("nan"),
    }
    return metrics, daily_rets


def main() -> int:
    vbt.settings.returns.year_freq = pd.Timedelta(days=252)
    print(f"=== RSI Daily VBT baseline — fenêtre {WINDOW_START} → {WINDOW_END} ===")
    print(f"=== Pairs: {', '.join(PAIRS)} ===\n")

    rows = []
    daily_rets_per_pair: dict[str, pd.Series] = {}
    for pair in PAIRS:
        try:
            m, rets = measure_pair(pair)
            rows.append(m)
            daily_rets_per_pair[pair] = rets
            print(f"  → Sharpe={m['sharpe_ratio']:+.3f} "
                  f"Trades={m['total_trades']:3d} "
                  f"MaxDD={m['max_drawdown']*100:+.2f}% "
                  f"Return={m['total_return']*100:+.2f}%\n", flush=True)
        except Exception as e:
            print(f"  [{pair}] FAILED: {e}\n", flush=True)
            rows.append({"pair": pair, "error": str(e)})

    df = pd.DataFrame(rows)
    out_csv = OUTPUT_DIR / "baseline_vbt.csv"
    df.to_csv(out_csv, index=False)
    print(f"=== {len(rows)} rows → {out_csv} ===")

    # Calcul du portfolio agrégé : equal-weight mean des daily_returns
    if daily_rets_per_pair:
        all_rets = pd.concat(daily_rets_per_pair, axis=1)
        portfolio_rets = all_rets.mean(axis=1, skipna=True)
        # Sharpe annualisé
        ann_factor = 252.0
        if portfolio_rets.std() > 0:
            agg_sharpe = (portfolio_rets.mean() / portfolio_rets.std()
                          * (ann_factor ** 0.5))
        else:
            agg_sharpe = float("nan")
        cum_ret = (1.0 + portfolio_rets.fillna(0.0)).cumprod()
        agg_max_dd = float(((cum_ret / cum_ret.cummax()) - 1.0).min())
        agg_total_ret = float(cum_ret.iloc[-1] - 1.0) if len(cum_ret) > 0 else float("nan")

        print(f"\n=== Portfolio agrégé equal-weight (mean axis=1) ===")
        print(f"  Sharpe annualisé : {agg_sharpe:+.3f}")
        print(f"  Total return     : {agg_total_ret*100:+.2f}%")
        print(f"  Max drawdown     : {agg_max_dd*100:+.2f}%")
        print(f"  Days used        : {len(portfolio_rets)}")

        # Sauve les daily returns agrégés pour analyses ultérieures
        out_rets_csv = OUTPUT_DIR / "baseline_vbt_daily_returns.csv"
        all_rets.columns = list(daily_rets_per_pair.keys())
        all_rets["portfolio_mean"] = portfolio_rets
        all_rets.to_csv(out_rets_csv)
        print(f"  → daily returns: {out_rets_csv}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
