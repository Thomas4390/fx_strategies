#!/usr/bin/env python
"""Compare position-sizing regimes on the gold momentum sleeve.

Runs identical signals through every regime in ``framework.sizing_nb`` and
ranks them on tail-risk metrics rather than Sharpe — see ``framework.ruin`` for
why Sharpe is the wrong instrument here.

Two comparisons are produced, and the second is the one that matters:

- **raw**: each regime at its natural exposure. Martingale looks good here
  purely because it is more levered on average.
- **risk-matched**: every regime rescaled to the same realized volatility, so
  the only remaining difference is the *shape* of the return distribution.

Holdout discipline: selection metrics use data before ``HOLDOUT_START`` only.
Pass ``--holdout`` to score the frozen winner on the blind period — once.

    python scripts/sweep_gold_sizing.py --smoke
    python scripts/sweep_gold_sizing.py
    python scripts/sweep_gold_sizing.py --holdout
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import vectorbtpro as vbt

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from framework.ruin import compare_regimes, format_comparison, ruin_report  # noqa: E402
from framework.sizing_nb import (  # noqa: E402
    MODE_ANTI_MART,
    MODE_COMBO,
    MODE_FLAT,
    MODE_GRID,
    MODE_MARTINGALE,
    build_overlay_kwargs,
    make_params,
)
from strategies.gold_momentum import pipeline  # noqa: E402
from utils import load_gold_data  # noqa: E402

# Adapted from docs/research/HOLDOUT_POLICY.md. The repo locks 2026-01-01; this
# sleeve locks six months earlier so the blind period spans both the parabolic
# 2025 advance and the 2026 drawdown — the regime that decides whether a
# path-dependent sizing rule is survivable.
HOLDOUT_START = pd.Timestamp("2025-07-01")

INIT_CASH = 100_000.0
TARGET_VOL = 0.25
SLIPPAGE = 0.0001  # 1 bp per side; XAUUSD CFD spread + commission
ATR_WINDOW = 14
RISK_MATCH_VOL = 0.25

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "results" / "gold_sizing"


def build_grid(smoke: bool) -> list[tuple[str, int, dict]]:
    """Regimes to compare. Kept small on purpose — every extra combination is
    another draw in a multiple-testing lottery."""
    if smoke:
        return [
            ("flat", MODE_FLAT, {}),
            ("martingale m=2.0 n=3", MODE_MARTINGALE, dict(mult=2.0, n_max=3)),
            ("grid k=0.5 lv=3", MODE_GRID, dict(grid_k=0.5, n_levels=3, grid_mult=1.0)),
        ]

    grid: list[tuple[str, int, dict]] = [("flat", MODE_FLAT, {})]
    for mult in (1.5, 2.0):
        for n_max in (2, 3, 4):
            grid.append(
                (f"martingale m={mult} n={n_max}", MODE_MARTINGALE, dict(mult=mult, n_max=n_max))
            )
    for k in (0.5, 1.0, 2.0):
        for lv in (2, 3, 5):
            grid.append(
                (f"grid k={k} lv={lv}", MODE_GRID, dict(grid_k=k, n_levels=lv, grid_mult=1.0))
            )
    # Deep single-level grid: the one grid shape with a prior from the data
    # (post-capitulation rebound beyond ~1.5% adverse).
    for k in (3.0, 4.0):
        grid.append(
            (f"grid deep k={k} lv=1", MODE_GRID, dict(grid_k=k, n_levels=1, grid_mult=1.0))
        )
    # Martingale-scaled grid levels.
    for gm in (1.5, 2.0):
        grid.append(
            (f"grid k=1.0 lv=3 gm={gm}", MODE_GRID, dict(grid_k=1.0, n_levels=3, grid_mult=gm))
        )
    for mult in (1.5, 2.0):
        grid.append(
            (
                f"combo m={mult} k=1.0",
                MODE_COMBO,
                dict(mult=mult, n_max=3, grid_k=1.0, n_levels=3),
            )
        )
    for mult in (1.5, 2.0):
        grid.append(
            (f"anti-martingale m={mult}", MODE_ANTI_MART, dict(mult=mult, n_max=3))
        )
    return grid


def load_daily() -> tuple[pd.DataFrame, np.ndarray]:
    """Daily OHLC and ATR over the **full** history.

    The simulation always runs on the whole series and the *returns* are sliced
    afterwards. Slicing the prices first would strand the longest momentum
    lookback (250 sessions) with no history, so most of a 333-session holdout
    would be scored on a signal that does not exist yet.
    """
    raw, _ = load_gold_data()
    daily = pd.DataFrame(
        {
            "high": raw.high.resample("D").max(),
            "low": raw.low.resample("D").min(),
            "close": raw.close.resample("D").last(),
        }
    ).dropna()
    atr = (
        vbt.ATR.run(daily.high, daily.low, daily.close, window=ATR_WINDOW)
        .atr.bfill()
        .to_numpy()
        .reshape(-1, 1)
    )
    return daily, atr


def run_regime(daily: pd.DataFrame, atr: np.ndarray, mode: int, params: dict, holdout: bool):
    """Simulate one regime over full history; return (returns, exposure, worst).

    ``returns`` are restricted to the requested period; ``exposure`` is the mean
    gross exposure over that same period, which is what makes a raw return
    comparable across regimes that lever differently.
    """
    memory: dict = {}
    p = make_params(mode, base_size=1.0, max_total=4.0, **params)
    pf, _ = pipeline(
        daily.close,
        target_vol=TARGET_VOL,
        init_cash=INIT_CASH,
        fees=0.0,
        slippage=SLIPPAGE,
        **build_overlay_kwargs(p, atr, memory=memory),
    )
    mask = (
        pf.returns.index >= HOLDOUT_START if holdout else pf.returns.index < HOLDOUT_START
    )
    returns = pf.returns[mask]
    exposure = float((pf.asset_value / pf.value)[mask].abs().mean())

    trades = pf.trades.records_readable
    if len(trades):
        col = "Exit Index" if "Exit Index" in trades.columns else "Entry Index"
        in_period = (
            trades[col] >= HOLDOUT_START if holdout else trades[col] < HOLDOUT_START
        )
        sub = trades[in_period]
        worst = float(sub["PnL"].min() / INIT_CASH) if len(sub) else float("nan")
    else:
        worst = float("nan")
    return returns, memory["state"][0], worst, exposure


def evaluate(daily, atr, grid, n_boot: int, risk_matched: bool, holdout: bool):
    reports = []
    for label, mode, params in grid:
        returns, state, worst, exposure = run_regime(daily, atr, mode, params, holdout)
        series = returns
        if risk_matched:
            vol = series.std() * np.sqrt(252)
            if vol > 0:
                series = series * (RISK_MATCH_VOL / vol)
        rep = ruin_report(
            series,
            label=label,
            worst_trade=worst,
            peak_exposure=float(state["max_total_seen"]),
            n_addons=int(state["n_addons"]),
            n_kills=int(state["n_killed"]),
            n_boot=n_boot,
        )
        rep.extras["mean_exposure"] = exposure
        reports.append(rep)
    return reports


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true", help="3 regimes, few bootstraps")
    ap.add_argument(
        "--holdout",
        action="store_true",
        help="score on the blind period instead of the selection period",
    )
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()

    vbt.settings.returns.year_freq = pd.Timedelta(days=252)

    n_boot = 200 if args.smoke else args.n_boot
    grid = build_grid(args.smoke)
    daily, atr = load_daily()

    period = "HOLDOUT (blind)" if args.holdout else "SELECTION"
    print(f"\n{'=' * 92}")
    print(f"  Gold sizing sweep — {period}")
    scored = daily[daily.index >= HOLDOUT_START] if args.holdout else daily[daily.index < HOLDOUT_START]
    print(f"  {len(scored)} sessions scored, {scored.index.min().date()} -> {scored.index.max().date()}")
    print(f"  (simulated on the full {len(daily)} sessions so the 250-day lookback is warm)")
    print(f"  {len(grid)} regimes, {n_boot} bootstrap paths, slippage {SLIPPAGE * 1e4:.1f} bp/side")
    print(f"{'=' * 92}\n")

    if args.holdout:
        print("!! Blind period. Read once, do not tune against these numbers.\n")

    raw_reports = evaluate(daily, atr, grid, n_boot, risk_matched=False, holdout=args.holdout)
    raw_df = compare_regimes(raw_reports)
    raw_df["mean_exposure"] = [
        next(r.extras.get("mean_exposure", float("nan")) for r in raw_reports if r.label == g)
        for g in raw_df["regime"]
    ]
    print("RAW — each regime at its natural exposure (mean_exposure shows how much of the")
    print("      return is simply more leverage rather than a better distribution shape)")
    print(format_comparison(raw_df))
    print("\n  mean gross exposure: " + ", ".join(
        f"{r['regime']}={r['mean_exposure'] * 100:.0f}%" for _, r in raw_df.head(6).iterrows()))

    matched = evaluate(daily, atr, grid, n_boot, risk_matched=True, holdout=args.holdout)
    print(f"\n\nRISK-MATCHED — all rescaled to {RISK_MATCH_VOL:.0%} volatility")
    print(format_comparison(compare_regimes(matched)))

    df = compare_regimes(matched)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    tag = "holdout" if args.holdout else "selection"
    csv_path = OUTPUT_DIR / f"sizing_{tag}_{stamp}.csv"
    df.to_csv(csv_path, index=False)
    meta = {
        "period": period,
        "sessions": len(daily),
        "start": str(daily.index.min().date()),
        "end": str(daily.index.max().date()),
        "holdout_start": str(HOLDOUT_START.date()),
        "slippage_bps_per_side": SLIPPAGE * 1e4,
        "target_vol": TARGET_VOL,
        "n_boot": n_boot,
        "best_by_mar": df.iloc[0]["regime"],
    }
    (OUTPUT_DIR / f"sizing_{tag}_{stamp}.json").write_text(json.dumps(meta, indent=2))
    print(f"\nWritten: {csv_path}")


if __name__ == "__main__":
    main()
