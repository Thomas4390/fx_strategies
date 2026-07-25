"""Gold Momentum (XAUUSD) — ims_pipeline format.

Time-series momentum on gold, following Moskowitz, Ooi & Pedersen, "Time series
momentum", Journal of Financial Economics 104(2), 2012: a security's own past
return predicts its future return over 1-12 month horizons.

Rather than selecting a single lookback — the classic overfitting trap — the
signal **averages the sign of four fixed lookbacks** (40/60/120/250 sessions).
Averaging replaces a choice with an aggregate, which is both closer to how the
TSMOM literature is applied in practice and far harder to overfit.

Trades are discrete: the position opens when the ensemble turns positive and
closes when it turns negative, so a position is held for weeks. That matters
downstream — it is the structure a martingale (sized on the previous trade's
outcome) and a grid (adding on adverse excursions) actually need.

Why not intraday? A Market Intraday Momentum variant (Gao, Han, Li & Zhou 2018)
was implemented and rejected on evidence: the predictor replicates on gold
(r1 -> last half-hour, beta=+0.0200, t=+6.10, R2=1.86%) but its gross edge is
only 1.5-2.9 bps per trade against ~2 bps of round-trip cost, at one trade per
session. See ``mim_signal`` and ``docs/research/`` for the record.

Three entry points, matching the other sleeves:
- ``pipeline(data, **params) -> (pf, ind)`` — investigation path
- ``pipeline_nb(data, **params)`` — ``@vbt.parameterized`` scalar metric
- ``create_cv_pipeline(splitter, metric_type)`` — ``@vbt.cv_split`` factory
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import vectorbtpro as vbt

from framework.leverage import vol_target_leverage
from framework.pipeline_utils import (
    SHARPE_RATIO,
    compute_metric_nb,
    make_execute_kwargs,
)

# Signals live on daily bars, so the sleeve annualizes like the other daily
# sleeves and stays directly comparable in the combiner.
GOLD_DAILY_ANN_FACTOR: float = 252.0

# Fixed lookbacks, averaged rather than selected. Spanning ~2 to ~12 months
# covers the horizon range the TSMOM literature documents.
DEFAULT_LOOKBACKS: tuple[int, ...] = (40, 60, 120, 250)

# Realized-volatility window for the vol-target layer, in sessions.
VOL_WINDOW: int = 21

# US session close, New York. Used only by the rejected intraday variant.
SESSION_CLOSE = "16:00"
FIRST_HALF_HOUR_END = "10:00"

# Hour that closes a daily session, New York. 17:00 is the CFD convention and
# the boundary QuantConnect uses for its daily gold bars. See ``_daily_close``.
SESSION_CLOSE_HOUR: int = 17


def _daily_close(data: Any) -> pd.Series:
    """Daily close series from a vbt.Data, DataFrame or Series of any frequency.

    Sessions close at 17:00 New York, not at midnight. Gold trades from Sunday
    18:00 to Friday 17:00, so aggregating on the calendar day carves every
    Sunday evening out as a session of its own — 392 of them over 2019-2026,
    each about 356 minutes long against 1375 for a real one. That inflates the
    session count by 20% and shortens every lookback by the same proportion: a
    250-session lookback was really spanning ~208 market sessions.

    The 17:00 boundary is also the convention of the QuantConnect daily CFD bar,
    which matters because the local parquet was exported from QuantConnect —
    aligning here is aligning on the data's producer.
    """
    close = data.close if hasattr(data, "close") else data
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    if not isinstance(close.index, pd.DatetimeIndex):
        return close

    steps = close.index.to_series().diff().dropna()
    if len(steps) and steps.median() >= pd.Timedelta(days=1):
        return close  # already daily or coarser — nothing to aggregate

    return close.groupby(session_dates(close.index)).last().dropna()


def session_dates(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Session date each timestamp belongs to, under a 17:00 New York close.

    Shifting by the distance to midnight turns "which session does this bar
    close in?" into a plain calendar-day question: 16:59 stays on day J,
    17:01 moves to J+1. Expressing the boundary once, here, is what keeps the
    sleeve and the sizing sweep on the same sessions — they disagreed before,
    and nothing in either output said so.

    The index must be tz-naive New York (``utils.load_gold_data``): the boundary
    is a wall-clock hour, so a UTC index would drift by one hour across DST.
    """
    return (index + pd.Timedelta(hours=24 - SESSION_CLOSE_HOUR)).normalize()


def momentum_ensemble(
    close_daily: pd.Series,
    lookbacks: tuple[int, ...] = DEFAULT_LOOKBACKS,
) -> pd.Series:
    """Mean of ``sign(return over N)`` across ``lookbacks``, in [-1, 1].

    Each lookback contributes an equally weighted vote. The result is a
    conviction score, not a binary state: 1.0 means every horizon agrees the
    trend is up.
    """
    votes = [np.sign(close_daily.vbt.pct_change(n)) for n in lookbacks]
    return sum(votes) / float(len(lookbacks))


def mim_signal(close_minute: pd.Series) -> pd.Series:
    """Market Intraday Momentum predictor — kept for the rejection record.

    ``r1`` is the log return from the previous session close (16:00 NY) to the
    end of the first half hour (10:00 NY). Its sign is the direction the
    Gao-Han-Li-Zhou rule would take into the session close.

    Requires a **tz-naive New York** minute index: the boundaries are clock
    times, so a UTC index silently drifts by an hour across DST.
    """
    if close_minute.index.tz is not None:
        raise ValueError(
            f"mim_signal expects a tz-naive New York index, got tz={close_minute.index.tz}. "
            "Use utils.load_gold_data()."
        )
    mod = (close_minute.index.hour * 60 + close_minute.index.minute).to_numpy()

    def _at(clock: str) -> pd.Series:
        ts = pd.Timestamp(clock)
        picked = close_minute[mod == int(ts.hour) * 60 + int(ts.minute)]
        return picked.groupby(picked.index.normalize()).last()

    ref = _at(SESSION_CLOSE)
    first = _at(FIRST_HALF_HOUR_END)
    return np.log(first / ref.vbt.fshift(1)).dropna().rename("r1")


# ═══════════════════════════════════════════════════════════════════════
# 1. INVESTIGATION PATH — pipeline() returns (pf, indicator)
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class GoldMomentumIndicator:
    """Signal diagnostics: conviction score and the leverage actually applied."""

    close: pd.Series
    score: pd.Series
    leverage: pd.Series

    def plot(self, fig: go.Figure | None = None, **layout_kwargs) -> go.Figure:
        fig = fig or go.Figure()
        self.score.vbt.plot(
            fig=fig,
            trace_kwargs=dict(name="momentum score", line=dict(width=2, color="teal")),
        )
        fig.add_hline(
            y=0.0,
            line=dict(color="grey", dash="dash", width=1),
            annotation_text="long above",
            annotation_position="right",
        )
        fig.update_layout(**layout_kwargs)
        return fig


def backtest_gold_momentum(data: vbt.Data, **kwargs: Any) -> vbt.Portfolio:
    """Thin shim — use ``pipeline(data, ...)`` in new code."""
    pf, _ = pipeline(data, **kwargs)
    return pf


def pipeline(
    data: vbt.Data,
    lookbacks: tuple[int, ...] = DEFAULT_LOOKBACKS,
    allow_short: bool = False,
    target_vol: float | None = 0.25,
    max_leverage: float = 3.0,
    base_size: float = 1.0,
    sl_stop: float | None = None,
    leverage: float | None = None,
    init_cash: float | None = None,
    slippage: float | None = None,
    fees: float | None = None,
    signal_func_nb: Any = None,
    signal_args: tuple = (),
    **pf_kwargs: Any,
) -> tuple[vbt.Portfolio, GoldMomentumIndicator]:
    """Investigation path — constant sizing unless a ``signal_func_nb`` is given.

    ``allow_short`` defaults to False. Gold carries a structural positive drift,
    so a sustained short fights the drift rather than harvesting a premium; the
    short side is available for testing but is not the default posture.

    ``target_vol`` scales the position by inverse realized volatility through
    ``framework.leverage.vol_target_leverage``. Set to None to size flat.

    ``signal_func_nb`` / ``signal_args`` are forwarded untouched to
    ``from_signals``: this is the seam through which the path-dependent sizing
    overlays of ``framework.sizing_nb`` plug in.
    """
    if not lookbacks:
        raise ValueError("lookbacks must be a non-empty tuple")
    if base_size <= 0:
        raise ValueError(f"base_size must be > 0, got {base_size}")

    close_daily = _daily_close(data)
    score = momentum_ensemble(close_daily, lookbacks)

    long_ok = score > 0.0
    short_ok = (score < 0.0) if allow_short else pd.Series(False, index=score.index)

    # Edge transitions: one entry when the state turns on, one exit when it
    # turns off. This is what makes a trade a trade, and it mirrors the
    # existing TS Momentum sleeve (daily_momentum.py).
    prev_long = long_ok.vbt.fshift(1, fill_value=False)
    prev_short = short_ok.vbt.fshift(1, fill_value=False)
    entries = long_ok & ~prev_long
    exits = ~long_ok & prev_long
    short_entries = short_ok & ~prev_short
    short_exits = ~short_ok & prev_short

    if target_vol is not None:
        daily_ret = close_daily.vbt.pct_change()
        realized = daily_ret.vbt.rolling_std(VOL_WINDOW, minp=VOL_WINDOW, ddof=1) * np.sqrt(252)
        lev_ts = vol_target_leverage(realized, target_vol, max_leverage=max_leverage)
    else:
        lev_ts = pd.Series(1.0, index=close_daily.index)

    # Per-bar leverage array rather than a percent-of-cash size: with
    # `size_type="percent"` VBT caps the order at the available cash, silently
    # truncating any target above 1x. This is the same plumbing the TS Momentum
    # sleeve uses (daily_momentum.py).
    lev_arr = (lev_ts * float(leverage if leverage is not None else 1.0)).to_numpy()

    pf_args: dict[str, Any] = dict(
        close=close_daily,
        entries=entries,
        exits=exits,
        short_entries=short_entries,
        short_exits=short_exits,
        size=base_size,
        size_type="percent",
        leverage=lev_arr,
        init_cash=init_cash,
        slippage=slippage,
        fees=fees,
        freq="1D",
    )
    if sl_stop is not None:
        pf_args["sl_stop"] = sl_stop
    if signal_func_nb is not None:
        pf_args["signal_func_nb"] = signal_func_nb
        pf_args["signal_args"] = signal_args
    pf_args.update(pf_kwargs)

    pf = vbt.Portfolio.from_signals(**pf_args)

    indicator = GoldMomentumIndicator(
        close=close_daily,
        score=score.rename("score"),
        leverage=lev_ts.rename("leverage"),
    )
    return pf, indicator


# ═══════════════════════════════════════════════════════════════════════
# 1b. RECONCILIATION TRACE — the cross-engine contract
# ═══════════════════════════════════════════════════════════════════════

# Column order and rounding are a contract shared with the QuantConnect and
# MQL5 ports. Changing either breaks the diff, so they live here as constants
# rather than inline. Spec: docs/specs/gold_momentum_spec.md §9.
TRACE_COLUMNS: tuple[str, ...] = (
    "date",
    "close",
    "score",
    "target_weight",
    "position_units",
    "equity",
)
_TRACE_DECIMALS: dict[str, int] = {
    "close": 6,
    "score": 6,
    "target_weight": 6,
    "position_units": 6,
    "equity": 2,
}


def emit_daily_trace(
    pf: vbt.Portfolio,
    indicator: GoldMomentumIndicator,
    path: str | Path | None = None,
    *,
    allow_short: bool = False,
) -> pd.DataFrame:
    """Emit the daily reconciliation trace, one row per usable session.

    The trace is the unit of comparison between vbt, QuantConnect and MT5:
    each engine emits these six columns and they are diffed rung by rung, so a
    divergence lands on a named quantity rather than on an aggregate. Comparing
    Sharpe ratios is the weakest available test — two implementations can agree
    on Sharpe through offsetting errors.

    Rows before the momentum score is defined (the 250-session warmup) are
    dropped rather than zero-filled: a score of 0.0 means "the horizons
    disagree", which is not the same statement as "no score yet".

    ``allow_short`` must match the value passed to ``pipeline``. It cannot be
    recovered from ``pf`` — an ``allow_short=True`` run that never happened to
    go short is indistinguishable from a long-only one — and it is needed to
    know whether a negative score targets a short or targets flat.

    Parameters
    ----------
    pf
        Portfolio returned by ``pipeline``.
    indicator
        The ``GoldMomentumIndicator`` returned alongside it, which carries the
        score and the vol-target leverage.
    path
        Destination CSV. When None the frame is returned without being written.
    allow_short
        Whether the short side was enabled in the run being traced.

    Returns
    -------
    The trace as a DataFrame, already rounded to the contract's precision so
    that the file and the returned frame agree value for value.
    """
    score = indicator.score
    close = indicator.close
    leverage = indicator.leverage

    if not pf.wrapper.index.equals(close.index):
        raise ValueError(
            "portfolio and indicator are not aligned on the same index; "
            "pass the (pf, indicator) pair returned by a single pipeline() call"
        )

    # Target weight is what the sleeve aims to hold, not what it holds: flat
    # whenever the score does not call for a position.
    direction = pd.Series(0.0, index=score.index)
    direction[score > 0.0] = 1.0
    if allow_short:
        direction[score < 0.0] = -1.0
    target_weight = leverage * direction

    trace = pd.DataFrame(
        {
            "date": close.index,
            "close": close.to_numpy(dtype=float),
            "score": score.to_numpy(dtype=float),
            "target_weight": target_weight.to_numpy(dtype=float),
            "position_units": np.asarray(pf.assets, dtype=float),
            "equity": np.asarray(pf.value, dtype=float),
        }
    )
    trace = trace[score.notna().to_numpy()].reset_index(drop=True)
    trace["date"] = pd.to_datetime(trace["date"]).dt.strftime("%Y-%m-%d")
    for column, decimals in _TRACE_DECIMALS.items():
        trace[column] = trace[column].round(decimals)

    if path is not None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        trace.to_csv(path, index=False, columns=list(TRACE_COLUMNS))
    return trace


# ═══════════════════════════════════════════════════════════════════════
# 2. GRID-SEARCH PATH — pipeline_nb (@vbt.parameterized)
# ═══════════════════════════════════════════════════════════════════════


@vbt.parameterized(
    merge_func="concat",
    execute_kwargs=make_execute_kwargs("Gold Momentum grid", chunk_len=8),
)
def pipeline_nb(
    data: vbt.Data,
    lookbacks: tuple[int, ...] = DEFAULT_LOOKBACKS,
    allow_short: bool = False,
    target_vol: float | None = 0.25,
    max_leverage: float = 3.0,
    sl_stop: float | None = None,
    leverage: float | None = None,
    init_cash: float | None = None,
    slippage: float | None = None,
    fees: float | None = None,
    ann_factor: float = GOLD_DAILY_ANN_FACTOR,
    cutoff: float = 0.05,
    metric_type: int = SHARPE_RATIO,
    **kwargs: Any,
) -> float:
    """Grid-search path — scalar metric per param combo."""
    pf, _ = pipeline(
        data,
        lookbacks=lookbacks,
        allow_short=allow_short,
        target_vol=target_vol,
        max_leverage=max_leverage,
        sl_stop=sl_stop,
        leverage=leverage,
        init_cash=init_cash,
        slippage=slippage,
        fees=fees,
        **kwargs,
    )
    returns = pf.returns.values
    if returns.ndim > 1:
        returns = returns[:, 0]
    return float(compute_metric_nb(returns, metric_type, ann_factor, cutoff))


def run_grid(
    data: vbt.Data,
    *,
    metric_type: int = SHARPE_RATIO,
    **params: Any,
) -> pd.Series:
    """Wrap list inputs as ``vbt.Param`` and call ``pipeline_nb``."""

    def _param(v):
        if isinstance(v, list):
            return vbt.Param(v)
        return v

    return pipeline_nb(
        data,
        metric_type=metric_type,
        **{k: _param(v) for k, v in params.items()},
    )


# ═══════════════════════════════════════════════════════════════════════
# 3. CV FACTORY — create_cv_pipeline (@vbt.cv_split)
# ═══════════════════════════════════════════════════════════════════════


def create_cv_pipeline(
    splitter: Any,
    metric_type: int = SHARPE_RATIO,
    **pipeline_defaults: Any,
):
    """Build a ``@vbt.cv_split`` pipeline for walk-forward cross-validation."""
    splitter_kwargs = pipeline_defaults.pop("splitter_kwargs", {})

    defaults = dict(
        lookbacks=DEFAULT_LOOKBACKS,
        allow_short=False,
        target_vol=0.25,
        max_leverage=3.0,
        sl_stop=None,
        leverage=None,
        init_cash=None,
        slippage=None,
        fees=None,
        ann_factor=GOLD_DAILY_ANN_FACTOR,
        cutoff=0.05,
        metric_type=metric_type,
    )
    defaults.update(pipeline_defaults)

    @vbt.cv_split(
        splitter=splitter,
        splitter_kwargs=splitter_kwargs,
        takeable_args=["data"],
        parameterized_kwargs=dict(
            execute_kwargs=make_execute_kwargs(
                "Gold Momentum combos", pbar_kwargs=dict(leave=False)
            ),
            merge_func="concat",
        ),
        execute_kwargs=make_execute_kwargs("Gold Momentum CV splits"),
        merge_func="concat",
        return_grid="all",
        attach_bounds="index",
    )
    def cv_pipeline(
        data: vbt.Data,
        lookbacks: tuple[int, ...] = defaults["lookbacks"],
        allow_short: bool = defaults["allow_short"],
        target_vol: float | None = defaults["target_vol"],
        max_leverage: float = defaults["max_leverage"],
        sl_stop: float | None = defaults["sl_stop"],
        leverage: float | None = defaults["leverage"],
        init_cash: float | None = defaults["init_cash"],
        slippage: float | None = defaults["slippage"],
        fees: float | None = defaults["fees"],
        ann_factor: float = defaults["ann_factor"],
        cutoff: float = defaults["cutoff"],
        metric_type: int = defaults["metric_type"],
    ) -> float:
        pf, _ = pipeline(
            data,
            lookbacks=lookbacks,
            allow_short=allow_short,
            target_vol=target_vol,
            max_leverage=max_leverage,
            sl_stop=sl_stop,
            leverage=leverage,
            init_cash=init_cash,
            slippage=slippage,
            fees=fees,
        )
        returns = pf.returns.values
        if returns.ndim > 1:
            returns = returns[:, 0]
        return float(compute_metric_nb(returns, metric_type, ann_factor, cutoff))

    return cv_pipeline
