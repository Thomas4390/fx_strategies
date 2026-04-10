"""Phase 2 Research: Broad multi-strategy exploration.

Explores 8 strategy variants across 4 FX pairs with strict look-ahead controls.
All multi-TF strategies use .shift(1) on resampled data before ffill.

Strategies:
  A1. Multi-TF BB (5min/15min/1h with proper lag)
  A2. Keltner Channel on VWAP deviation
  A3. Hurst exponent regime filter
  B2. Cross-sectional momentum
  C1. EMA crossover daily
  C2. SuperTrend daily
  C3. ADX breakout
"""

from __future__ import annotations

import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import vectorbtpro as vbt

from strategies.mr_macro import load_macro_filters
from utils import apply_vbt_settings, load_fx_data

warnings.filterwarnings("ignore", category=FutureWarning)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

WF_PERIODS: list[tuple[str, str]] = [
    (f"{y}-01-01", f"{y}-12-31") for y in range(2019, 2025)
] + [("2025-01-01", "2026-04-01")]
FX_PAIRS = ["EUR-USD", "GBP-USD", "USD-JPY", "USD-CAD"]


# ===================================================================
# HELPERS
# ===================================================================

def _safe_sharpe(pf: vbt.Portfolio) -> float:
    try:
        if pf.trades.count() == 0:
            return 0.0
        sr = pf.sharpe_ratio
        return 0.0 if (pd.isna(sr) or np.isinf(sr)) else float(sr)
    except Exception:
        return 0.0

def _safe_trades(pf: vbt.Portfolio) -> int:
    try:
        return int(pf.trades.count())
    except Exception:
        return 0

def _h(title: str) -> None:
    print(f"\n{'=' * 70}\n  {title}\n{'=' * 70}")

def _sh(title: str) -> None:
    print(f"\n{'-' * 60}\n  {title}\n{'-' * 60}")


def walkforward(data, backtest_fn, label="", **kw) -> dict[str, Any]:
    """Walk-forward assessment across 7 annual periods."""
    sharpes, trades = [], []
    for start, end in WF_PERIODS:
        try:
            d = data.loc[start:end]
            if d.shape[0] < 1000:
                sharpes.append(0.0); trades.append(0)
                continue
            pf = backtest_fn(d, **kw)
            sharpes.append(_safe_sharpe(pf))
            trades.append(_safe_trades(pf))
        except Exception:
            sharpes.append(0.0); trades.append(0)
    return {
        "label": label,
        "sharpes": sharpes,
        "avg_sharpe": float(np.mean(sharpes)),
        "pos_years": sum(1 for s in sharpes if s > 0),
        "oos_sharpe": sharpes[-1],
        "total_trades": sum(trades),
    }

def pr(r: dict) -> None:
    """Print walk-forward result."""
    detail = " ".join(f"{s:>6.2f}" for s in r["sharpes"])
    print(
        f"  {r['label']:<50} avg={r['avg_sharpe']:>5.2f}"
        f" pos={r['pos_years']}/7 oos={r['oos_sharpe']:>5.2f}"
        f" tc={r['total_trades']:>5} | {detail}"
    )


# ===================================================================
# A1: MULTI-TF BB (safe: .shift(1) before ffill)
# ===================================================================

def backtest_multitf_safe(
    data: vbt.Data,
    tf: str = "1h",
    bb_window: int = 6,
    bb_alpha: float = 4.0,
    macro_filter: pd.Series | None = None,
    sl_stop: float = 0.005,
    tp_stop: float = 0.006,
    session_start: int = 6,
    session_end: int = 14,
    slippage: float = 0.00015,
) -> vbt.Portfolio:
    """Multi-TF BB with .shift(1) to prevent look-ahead."""
    close = data.close
    idx = close.index

    c_tf = close.resample(tf).last().dropna()
    h_tf = data.high.resample(tf).max().dropna()
    l_tf = data.low.resample(tf).min().dropna()
    v_tf = data.volume.resample(tf).sum().dropna()
    ci = c_tf.index.intersection(h_tf.index).intersection(
        l_tf.index
    ).intersection(v_tf.index)
    c_tf, h_tf, l_tf, v_tf = (
        c_tf.loc[ci], h_tf.loc[ci], l_tf.loc[ci], v_tf.loc[ci],
    )

    vwap_tf = vbt.VWAP.run(h_tf, l_tf, c_tf, v_tf, anchor="D").vwap
    dev_tf = c_tf - vwap_tf
    bb = vbt.BBANDS.run(dev_tf, window=bb_window, alpha=bb_alpha)

    # CRITICAL: .shift(1) prevents look-ahead
    upper = (vwap_tf + bb.upper).shift(1).reindex(idx, method="ffill")
    lower = (vwap_tf + bb.lower).shift(1).reindex(idx, method="ffill")

    hours = idx.hour
    session = (hours >= session_start) & (hours < session_end)
    filt = pd.Series(True, index=idx)
    if macro_filter is not None:
        filt = macro_filter.reindex(idx, method="ffill").fillna(False)

    entries = (close < lower) & session & filt
    short_entries = (close > upper) & session & filt

    return vbt.Portfolio.from_signals(
        data, entries=entries, exits=False,
        short_entries=short_entries, short_exits=False,
        sl_stop=sl_stop, tp_stop=tp_stop,
        dt_stop="21:00", td_stop="6h",
        slippage=slippage, init_cash=1_000_000, freq="1min",
    )


def phase_a1(data: vbt.Data) -> list[dict]:
    """A1: Multi-TF BB with safe lag."""
    _h("A1: Multi-TF BB (safe, .shift(1))")
    results = []

    from research.macro_score_research import backtest_mr_with_filter
    macro = load_macro_filters(data.close.index, spread_threshold=0.5)

    r = walkforward(data, backtest_mr_with_filter,
                    "BASELINE 1min BB(80,5)+macro", macro_filter=macro)
    pr(r); results.append(r)

    configs = [
        ("5min", 16, 4.0), ("5min", 24, 4.0), ("5min", 32, 5.0),
        ("15min", 8, 3.0), ("15min", 8, 4.0), ("15min", 12, 4.0),
        ("15min", 16, 5.0),
        ("1h", 4, 3.0), ("1h", 6, 3.0), ("1h", 6, 4.0),
        ("1h", 8, 4.0), ("1h", 12, 5.0),
    ]
    for tf, w, a in configs:
        for filt_name, filt in [("no_macro", None), ("macro", macro)]:
            label = f"{tf} BB({w},{a}) {filt_name}"
            r = walkforward(data, backtest_multitf_safe, label,
                            tf=tf, bb_window=w, bb_alpha=a,
                            macro_filter=filt)
            results.append(r)

    results.sort(key=lambda x: x["avg_sharpe"], reverse=True)
    _sh("Top 10 Multi-TF (safe)")
    for r in results[:10]:
        pr(r)
    return results


# ===================================================================
# A2: KELTNER CHANNEL on VWAP deviation
# ===================================================================

def backtest_keltner(
    data: vbt.Data,
    atr_period: int = 14,
    kc_mult: float = 2.0,
    ema_span: int = 40,
    macro_filter: pd.Series | None = None,
    sl_stop: float = 0.005,
    tp_stop: float = 0.006,
    session_start: int = 6,
    session_end: int = 14,
    slippage: float = 0.00015,
) -> vbt.Portfolio:
    """Keltner Channel on VWAP deviation (ATR-based bands)."""
    close = data.close
    idx = close.index

    vwap = vbt.VWAP.run(
        data.high, data.low, close, data.volume, anchor="D"
    ).vwap
    deviation = close - vwap
    ema_dev = deviation.ewm(span=ema_span, min_periods=ema_span).mean()

    atr = vbt.talib("ATR").run(
        data.high, data.low, close, timeperiod=atr_period
    )
    atr_vals = atr.real

    upper = vwap + ema_dev + kc_mult * atr_vals
    lower = vwap + ema_dev - kc_mult * atr_vals

    hours = idx.hour
    session = (hours >= session_start) & (hours < session_end)
    filt = pd.Series(True, index=idx)
    if macro_filter is not None:
        filt = macro_filter.reindex(idx, method="ffill").fillna(False)

    entries = (close < lower) & session & filt
    short_entries = (close > upper) & session & filt

    return vbt.Portfolio.from_signals(
        data, entries=entries, exits=False,
        short_entries=short_entries, short_exits=False,
        sl_stop=sl_stop, tp_stop=tp_stop,
        dt_stop="21:00", td_stop="6h",
        slippage=slippage, init_cash=1_000_000, freq="1min",
    )


def phase_a2(data: vbt.Data) -> list[dict]:
    """A2: Keltner Channel sweep."""
    _h("A2: Keltner Channel on VWAP Deviation")
    results = []
    macro = load_macro_filters(data.close.index, spread_threshold=0.5)

    for atr_p in [10, 14, 20]:
        for mult in [1.5, 2.0, 2.5, 3.0]:
            for ema in [20, 40, 60]:
                for filt_name, filt in [("no_macro", None), ("macro", macro)]:
                    label = f"KC(atr={atr_p},m={mult},ema={ema}) {filt_name}"
                    r = walkforward(
                        data, backtest_keltner, label,
                        atr_period=atr_p, kc_mult=mult, ema_span=ema,
                        macro_filter=filt,
                    )
                    results.append(r)

    results.sort(key=lambda x: x["avg_sharpe"], reverse=True)
    _sh("Top 10 Keltner")
    for r in results[:10]:
        pr(r)
    return results


# ===================================================================
# A3: HURST EXPONENT REGIME FILTER
# ===================================================================

def backtest_hurst_filtered(
    data: vbt.Data,
    hurst_threshold: float = 0.5,
    hurst_window: int = 252,
    bb_window: int = 80,
    bb_alpha: float = 5.0,
    macro_filter: pd.Series | None = None,
    sl_stop: float = 0.005,
    tp_stop: float = 0.006,
    session_start: int = 6,
    session_end: int = 14,
    slippage: float = 0.00015,
) -> vbt.Portfolio:
    """BB MR filtered by Hurst exponent (trade only when H < threshold)."""
    close = data.close
    idx = close.index

    # Hurst on daily close (shift(1) for safety)
    close_daily = close.resample("1D").last().dropna()
    hurst = vbt.HURST.run(close_daily, window=hurst_window, method="RS")
    hurst_daily = hurst.hurst.shift(1)  # use yesterday's Hurst
    hurst_min = hurst_daily.reindex(idx, method="ffill")
    is_mr_regime = (hurst_min < hurst_threshold).fillna(False)

    vwap = vbt.VWAP.run(
        data.high, data.low, close, data.volume, anchor="D"
    ).vwap
    dev = close - vwap
    bb = vbt.BBANDS.run(dev, window=bb_window, alpha=bb_alpha)
    upper = vwap + bb.upper
    lower = vwap + bb.lower

    hours = idx.hour
    session = (hours >= session_start) & (hours < session_end)

    filt = is_mr_regime
    if macro_filter is not None:
        macro_aligned = macro_filter.reindex(idx, method="ffill").fillna(False)
        filt = filt & macro_aligned

    entries = (close < lower) & session & filt
    short_entries = (close > upper) & session & filt

    return vbt.Portfolio.from_signals(
        data, entries=entries, exits=False,
        short_entries=short_entries, short_exits=False,
        sl_stop=sl_stop, tp_stop=tp_stop,
        dt_stop="21:00", td_stop="6h",
        slippage=slippage, init_cash=1_000_000, freq="1min",
    )


def phase_a3(data: vbt.Data) -> list[dict]:
    """A3: Hurst exponent regime filter."""
    _h("A3: Hurst Exponent Regime Filter")
    results = []
    macro = load_macro_filters(data.close.index, spread_threshold=0.5)

    for h_thresh in [0.40, 0.45, 0.50, 0.55]:
        for h_win in [126, 252]:
            for filt_name, filt in [("no_macro", None), ("macro", macro)]:
                label = f"Hurst<{h_thresh} w={h_win} {filt_name}"
                r = walkforward(
                    data, backtest_hurst_filtered, label,
                    hurst_threshold=h_thresh, hurst_window=h_win,
                    macro_filter=filt,
                )
                results.append(r)

    results.sort(key=lambda x: x["avg_sharpe"], reverse=True)
    _sh("Top 10 Hurst Filtered")
    for r in results[:10]:
        pr(r)
    return results


# ===================================================================
# C1: EMA CROSSOVER DAILY
# ===================================================================

def backtest_ema_cross(
    data: vbt.Data,
    fast_period: int = 10,
    slow_period: int = 30,
    sl_stop: float = 0.008,
    tp_stop: float = 0.012,
    session_start: int = 6,
    session_end: int = 14,
    slippage: float = 0.00015,
) -> vbt.Portfolio:
    """EMA crossover on daily data, execute on 1min VWAP dips."""
    close = data.close
    idx = close.index

    close_d = close.resample("1D").last().dropna()
    ema_f = close_d.ewm(span=fast_period, min_periods=fast_period).mean()
    ema_s = close_d.ewm(span=slow_period, min_periods=slow_period).mean()
    trend_long_d = (ema_f > ema_s).shift(1)
    trend_short_d = (ema_f < ema_s).shift(1)
    trend_long = trend_long_d.reindex(idx, method="ffill").fillna(False)
    trend_short = trend_short_d.reindex(idx, method="ffill").fillna(False)

    vwap = vbt.VWAP.run(
        data.high, data.low, close, data.volume, anchor="D"
    ).vwap
    at_vwap_dip = close < vwap * 0.9998
    at_vwap_pop = close > vwap * 1.0002

    hours = idx.hour
    session = (hours >= session_start) & (hours < session_end)

    entries = trend_long & at_vwap_dip & session
    short_entries = trend_short & at_vwap_pop & session

    return vbt.Portfolio.from_signals(
        data, entries=entries, exits=False,
        short_entries=short_entries, short_exits=False,
        sl_stop=sl_stop, tp_stop=tp_stop,
        dt_stop="21:00", td_stop="8h",
        slippage=slippage, init_cash=1_000_000, freq="1min",
    )


def phase_c1(data: vbt.Data) -> list[dict]:
    """C1: EMA crossover daily."""
    _h("C1: EMA Crossover Daily + Intraday VWAP")
    results = []

    for fast in [5, 10, 20]:
        for slow in [20, 30, 50]:
            if fast >= slow:
                continue
            label = f"EMA({fast}/{slow})"
            r = walkforward(data, backtest_ema_cross, label,
                            fast_period=fast, slow_period=slow)
            results.append(r)

    results.sort(key=lambda x: x["avg_sharpe"], reverse=True)
    _sh("Top 10 EMA Cross")
    for r in results[:10]:
        pr(r)
    return results


# ===================================================================
# C2: SUPERTREND DAILY
# ===================================================================

def backtest_supertrend(
    data: vbt.Data,
    st_period: int = 10,
    st_factor: float = 3.0,
    sl_stop: float = 0.008,
    tp_stop: float = 0.015,
    session_start: int = 6,
    session_end: int = 18,
    slippage: float = 0.00015,
) -> vbt.Portfolio:
    """SuperTrend on daily data, execute on 1min."""
    close = data.close
    idx = close.index

    c_d = close.resample("1D").last().dropna()
    h_d = data.high.resample("1D").max().dropna()
    l_d = data.low.resample("1D").min().dropna()
    ci = c_d.index.intersection(h_d.index).intersection(l_d.index)
    c_d, h_d, l_d = c_d.loc[ci], h_d.loc[ci], l_d.loc[ci]

    st = vbt.SUPERTREND.run(h_d, l_d, c_d, period=st_period, factor=st_factor)

    dir_d = st.direction.shift(1)
    trend_long = (dir_d == 1).reindex(idx, method="ffill").fillna(False)
    trend_short = (dir_d == -1).reindex(idx, method="ffill").fillna(False)

    hours = idx.hour
    session = (hours >= session_start) & (hours < session_end)

    long_change = trend_long & (~trend_long.shift(1).fillna(False))
    short_change = trend_short & (~trend_short.shift(1).fillna(False))

    entries = long_change & session
    short_entries = short_change & session

    return vbt.Portfolio.from_signals(
        data, entries=entries, exits=False,
        short_entries=short_entries, short_exits=False,
        sl_stop=sl_stop, tp_stop=tp_stop,
        dt_stop="21:00", td_stop="24h",
        slippage=slippage, init_cash=1_000_000, freq="1min",
    )


def phase_c2(data: vbt.Data) -> list[dict]:
    """C2: SuperTrend daily sweep."""
    _h("C2: SuperTrend Daily")
    results = []

    for period in [7, 10, 14, 20]:
        for factor in [2.0, 3.0, 4.0]:
            label = f"ST(p={period},f={factor})"
            r = walkforward(data, backtest_supertrend, label,
                            st_period=period, st_factor=factor)
            results.append(r)

    results.sort(key=lambda x: x["avg_sharpe"], reverse=True)
    _sh("Top 10 SuperTrend")
    for r in results[:10]:
        pr(r)
    return results


# ===================================================================
# C3: ADX BREAKOUT
# ===================================================================

def backtest_adx_breakout(
    data: vbt.Data,
    adx_period: int = 14,
    adx_threshold: int = 25,
    sl_stop: float = 0.006,
    tp_stop: float = 0.012,
    session_start: int = 6,
    session_end: int = 18,
    slippage: float = 0.00015,
) -> vbt.Portfolio:
    """ADX breakout: enter when trend strong, in DI direction."""
    close = data.close
    idx = close.index

    c_d = close.resample("1D").last().dropna()
    h_d = data.high.resample("1D").max().dropna()
    l_d = data.low.resample("1D").min().dropna()
    ci = c_d.index.intersection(h_d.index).intersection(l_d.index)
    c_d, h_d, l_d = c_d.loc[ci], h_d.loc[ci], l_d.loc[ci]

    adx = vbt.talib("ADX").run(h_d, l_d, c_d, timeperiod=adx_period)
    plus_di = vbt.talib("PLUS_DI").run(
        h_d, l_d, c_d, timeperiod=adx_period
    )
    minus_di = vbt.talib("MINUS_DI").run(
        h_d, l_d, c_d, timeperiod=adx_period
    )

    strong_trend = (adx.real > adx_threshold).shift(1)
    bull_di = (plus_di.real > minus_di.real).shift(1)
    bear_di = (minus_di.real > plus_di.real).shift(1)

    long_d = (strong_trend & bull_di).reindex(idx, method="ffill").fillna(False)
    short_d = (strong_trend & bear_di).reindex(idx, method="ffill").fillna(False)

    hours = idx.hour
    session = (hours >= session_start) & (hours < session_end)

    long_change = long_d & (~long_d.shift(1).fillna(False))
    short_change = short_d & (~short_d.shift(1).fillna(False))

    entries = long_change & session
    short_entries = short_change & session

    return vbt.Portfolio.from_signals(
        data, entries=entries, exits=False,
        short_entries=short_entries, short_exits=False,
        sl_stop=sl_stop, tp_stop=tp_stop,
        dt_stop="21:00", td_stop="24h",
        slippage=slippage, init_cash=1_000_000, freq="1min",
    )


def phase_c3(data: vbt.Data) -> list[dict]:
    """C3: ADX Breakout sweep."""
    _h("C3: ADX Breakout")
    results = []

    for period in [10, 14, 20]:
        for thresh in [20, 25, 30, 35]:
            label = f"ADX(p={period},th={thresh})"
            r = walkforward(data, backtest_adx_breakout, label,
                            adx_period=period, adx_threshold=thresh)
            results.append(r)

    results.sort(key=lambda x: x["avg_sharpe"], reverse=True)
    _sh("Top 10 ADX Breakout")
    for r in results[:10]:
        pr(r)
    return results


# ===================================================================
# B2: CROSS-SECTIONAL MOMENTUM
# ===================================================================

def phase_b2_cross_momentum() -> list[dict]:
    """B2: Cross-sectional momentum across 4 pairs."""
    _h("B2: Cross-Sectional Momentum (4 pairs)")
    results = []

    pair_data = {}
    for pair in FX_PAIRS:
        _, d = load_fx_data(f"data/{pair}_minute.parquet")
        pair_data[pair] = d

    closes = pd.DataFrame({
        p: d.close for p, d in pair_data.items()
    }).dropna()

    for lookback in [60, 240, 1440]:
        for hold in ["4h", "8h", "24h"]:
            label = f"XSMom lb={lookback}min hold={hold}"

            sharpes, trades = [], []
            for start, end in WF_PERIODS:
                try:
                    cl = closes.loc[start:end]
                    if len(cl) < lookback + 100:
                        sharpes.append(0.0); trades.append(0)
                        continue

                    ret = cl.pct_change(lookback)
                    ranks = ret.rank(axis=1)
                    n_pairs = len(FX_PAIRS)

                    long_sig = (ranks.shift(1) == n_pairs)
                    short_sig = (ranks.shift(1) == 1)

                    hours = cl.index.hour
                    session = (hours >= 6) & (hours < 18)

                    p = FX_PAIRS[0]
                    d_yr = pair_data[p].loc[start:end]
                    le = long_sig[p].reindex(
                        d_yr.close.index
                    ).fillna(False) & session.reindex(
                        d_yr.close.index
                    ).fillna(False)
                    se = short_sig[p].reindex(
                        d_yr.close.index
                    ).fillna(False) & session.reindex(
                        d_yr.close.index
                    ).fillna(False)

                    pf = vbt.Portfolio.from_signals(
                        d_yr, entries=le, exits=False,
                        short_entries=se, short_exits=False,
                        sl_stop=0.005, tp_stop=0.008,
                        dt_stop="21:00", td_stop=hold,
                        slippage=0.00015, init_cash=1_000_000, freq="1min",
                    )
                    sharpes.append(_safe_sharpe(pf))
                    trades.append(_safe_trades(pf))
                except Exception:
                    sharpes.append(0.0); trades.append(0)

            r = {
                "label": label,
                "sharpes": sharpes,
                "avg_sharpe": float(np.mean(sharpes)),
                "pos_years": sum(1 for s in sharpes if s > 0),
                "oos_sharpe": sharpes[-1],
                "total_trades": sum(trades),
            }
            results.append(r)

    results.sort(key=lambda x: x["avg_sharpe"], reverse=True)
    _sh("Cross-Sectional Momentum")
    for r in results:
        pr(r)
    return results


# ===================================================================
# MAIN
# ===================================================================

def main() -> None:
    apply_vbt_settings()
    vbt.settings.returns.year_freq = pd.Timedelta(hours=24) * 252

    print("Loading EUR-USD...")
    _, data = load_fx_data()
    print(f"  {data.shape[0]:,} bars")

    t_start = time.time()
    all_results: dict[str, list[dict]] = {}

    # Phase A: Mean Reversion Variants
    all_results["A1_multitf"] = phase_a1(data)
    all_results["A2_keltner"] = phase_a2(data)
    all_results["A3_hurst"] = phase_a3(data)

    # Phase C: Trend Following
    all_results["C1_ema_cross"] = phase_c1(data)
    all_results["C2_supertrend"] = phase_c2(data)
    all_results["C3_adx"] = phase_c3(data)

    # Phase B: Cross-Sectional
    all_results["B2_xsmom"] = phase_b2_cross_momentum()

    # Multi-pair test of baseline
    _h("MULTI-PAIR: Baseline MR + macro (4 pairs)")
    from research.macro_score_research import backtest_mr_with_filter
    for pair in FX_PAIRS:
        _, d = load_fx_data(f"data/{pair}_minute.parquet")
        macro = load_macro_filters(d.close.index, spread_threshold=0.5)
        r = walkforward(d, backtest_mr_with_filter, f"BB MR+macro {pair}",
                        macro_filter=macro)
        pr(r)

    # Summary
    _h("OVERALL SUMMARY - Best per Phase")
    for phase, res in all_results.items():
        if res:
            best = res[0]
            print(f"\n  {phase}:")
            pr(best)

    elapsed = time.time() - t_start
    print(f"\nTotal research time: {elapsed / 60:.1f} minutes")


if __name__ == "__main__":
    main()
