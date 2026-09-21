"""XAUUSD x10 levels — reference engine (``docs/specs/xau_x10_spec.md``).

This module is the **reference implementation** of the spec: an MQL5 expert and
a QuantConnect algorithm are written against it and reconciled event by event
(§13). Everything here is therefore arranged to be readable and portable rather
than clever — the decisions live in one Numba kernel
(``framework.x10_engine``), the calendars in one pandas module
(``framework.x10_context``), and this file only assembles them.

Four entry points, in the ``ims_pipeline`` shape the repo uses everywhere:

- ``prepare_inputs(data)`` — M1 gold in, every array the kernel reads out. The
  expensive half of a run (two resamplings and six indicators) lives here so a
  27-point grid pays for it once.
- ``pipeline(data, **params) -> (pf, ind)`` — runs the engine and turns its
  order registry into a ``vbt.Portfolio`` built with ``from_orders`` on the
  **M1** index. ``from_signals`` is not usable: the pessimistic double-contact
  rule of §9 (stop wins) has to be decided inside the minute, and only the
  kernel sees it. ``ind`` carries the two registries.
- ``pipeline_nb`` / ``run_grid`` / ``create_cv_pipeline`` — the sweep paths,
  both guarded by ``assert_not_optimizing``: the 27 configurations of annexe
  A.2 are ranked on 2019-2025 and never on the frozen slice.
- ``emit_event_trace(ind, path)`` — the §13 CSV, one line per event, which is
  the only artefact the three engines compare.

Costs are already inside the prices the kernel works with (entry at
``mid ± s/2``, exits on the trigger side), so the portfolio is built with
``fees=0`` and ``slippage=0``. Charging them again at the portfolio level would
double-count the spread the way the first draft of the ``R`` formula did (§8).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import vectorbtpro as vbt

from framework.holdout import assert_not_optimizing
from framework.pipeline_utils import (
    FX_MINUTE_ANN_FACTOR,
    SHARPE_RATIO,
    compute_metric_nb,
    make_execute_kwargs,
)
from framework.project_config import PROJECT_CONFIG
from framework.x10_context import (
    align_h1_to_m5,
    atr_wilder,
    ema_h1,
    resample_ohlc,
    session_ids,
    session_mean_typical,
    to_session_clock,
)
from framework.x10_engine import (
    CANCEL_REASON_NAMES,
    CONTRACT_SIZE,
    CR_CTX,
    CR_R,
    CR_SIZE,
    CR_WINDOW,
    EVENT_CANCEL,
    EVENT_COLUMNS,
    EVENT_ENTRY,
    EVENT_NAMES,
    EXIT_REASON_NAMES,
    RISK_FRAC,
    SCENARIO_NAMES,
    T_EXIT_PX,
    T_FILL_PX,
    T_I_EXIT_M1,
    T_I_FILL_M1,
    T_LOTS,
    T_Q,
    TRADE_COLUMNS,
    run_engine,
)
from framework.x10_kernels import kinematics_nb, x10_levels_nb

# ═══════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════

# §12 — catalogue snapshot 2026-07-28, 290 points x 0.001. The hourly medians
# of `costs_xau_intraday.yml` replace it once lot D lands; until then this is
# the documented fallback, and the sensitivity at x2 is mandatory.
X10_SPREAD_DEFAULT = 0.29

DXY4_H1_PATH = Path(PROJECT_CONFIG["data_dir"]) / "DXY4_h1.parquet"

# §14 — the strategy is quoted and sized in account currency on a 100 oz
# contract; the initial equity only sets the scale of §11.
X10_INIT_CASH = 10_000.0

# Annexe A.2 — the grid, and nothing else.
GRID_Z = (0.5, 1.0, 1.5)
GRID_A_MIN = (0.1, 0.2, 0.3)
GRID_K_S = (0.75, 1.0, 1.5)

# §13 — the trace contract, in this order, with these precisions.
TRACE_COLUMNS: tuple[str, ...] = (
    "ts_decision",
    "event",
    "scenario",
    "level",
    "d",
    "atr",
    "v",
    "m",
    "a",
    "ctx_ema",
    "ctx_vwap",
    "ctx_dxy",
    "vwap",
    "ema50_h1",
    "dxy",
    "stop",
    "target",
    "r_est",
    "fill_px",
    "exit_px",
    "exit_reason",
)
_TRACE_DECIMALS: dict[str, int] = {
    "level": 3,
    "atr": 6,
    "v": 6,
    "m": 6,
    "a": 6,
    "vwap": 6,
    "ema50_h1": 6,
    "dxy": 6,
    "stop": 6,
    "target": 6,
    "r_est": 6,
    "fill_px": 3,
    "exit_px": 3,
}

__all__ = [
    "DXY4_H1_PATH",
    "X10_INIT_CASH",
    "X10_SPREAD_DEFAULT",
    "X10Indicator",
    "X10Inputs",
    "count_summary",
    "create_cv_pipeline",
    "emit_event_trace",
    "pipeline",
    "pipeline_nb",
    "prepare_inputs",
    "run_grid",
]


# ═══════════════════════════════════════════════════════════════════════
# 1. INPUTS — M1 gold to the arrays the kernel reads
# ═══════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class X10Inputs:
    """Everything ``x10_engine_nb`` reads, plus the two indexes to read it back.

    Frozen and parameter-free on purpose: nothing in here depends on ``z``,
    ``a_min`` or ``k_s``, so one instance serves the whole grid of annexe A.2.
    """

    m5: pd.DataFrame
    m1: pd.DataFrame
    atr: np.ndarray
    v: np.ndarray
    m: np.ndarray
    a: np.ndarray
    l_inf: np.ndarray
    l_sup: np.ndarray
    vwap: np.ndarray
    ema50_h1: np.ndarray
    atr_h1: np.ndarray
    dxy: np.ndarray
    dxy_ema50_h1: np.ndarray
    minute_of_day: np.ndarray
    session_id: np.ndarray
    m1_start: np.ndarray
    m1_end: np.ndarray

    def kernel_kwargs(self, spread: float) -> dict[str, Any]:
        """The keyword arguments of ``run_engine``, spread broadcast per bar."""
        return dict(
            m5_open=self.m5["open"].to_numpy(dtype=np.float64),
            m5_high=self.m5["high"].to_numpy(dtype=np.float64),
            m5_low=self.m5["low"].to_numpy(dtype=np.float64),
            m5_close=self.m5["close"].to_numpy(dtype=np.float64),
            atr=self.atr,
            v=self.v,
            m=self.m,
            a=self.a,
            l_inf=self.l_inf,
            l_sup=self.l_sup,
            vwap=self.vwap,
            ema50_h1=self.ema50_h1,
            atr_h1=self.atr_h1,
            dxy=self.dxy,
            dxy_ema50_h1=self.dxy_ema50_h1,
            minute_of_day=self.minute_of_day,
            session_id=self.session_id,
            spread=np.full(len(self.m5), float(spread)),
            m1_open=self.m1["open"].to_numpy(dtype=np.float64),
            m1_high=self.m1["high"].to_numpy(dtype=np.float64),
            m1_low=self.m1["low"].to_numpy(dtype=np.float64),
            m1_close=self.m1["close"].to_numpy(dtype=np.float64),
            m1_start=self.m1_start,
            m1_end=self.m1_end,
        )


def _m1_frame(data: Any) -> pd.DataFrame:
    """Lower-cased M1 OHLC on the naive New York clock of §2.

    Accepts the two shapes the repo passes around: the ``vbt.Data`` wrapper
    (capitalised columns) returned by ``load_gold_data`` and the raw frame next
    to it. A tz-aware index is converted, never assumed.
    """
    frame = data.get() if isinstance(data, vbt.Data) else data
    if isinstance(frame.columns, pd.MultiIndex):
        frame = frame.droplevel(-1, axis=1)
    frame = frame.rename(columns=str.lower)
    frame = frame[["open", "high", "low", "close"]].copy()
    frame.index = to_session_clock(pd.DatetimeIndex(frame.index))
    return frame.sort_index()


def _load_dxy_h1(path: str | Path = DXY4_H1_PATH) -> pd.Series | None:
    """The synthetic four-leg dollar of §6.3, H1, naive New York.

    Missing file returns ``None`` and the run goes on with ``ctx_dxy = 0``:
    §6.3 makes an undefined basket neutral, not fatal, and the MT5 tester will
    routinely be in that case.
    """
    path = Path(path)
    if not path.exists():
        return None
    frame = pd.read_parquet(path)
    if "date" in frame.columns:
        frame = frame.set_index("date")
    series = frame["close"] if "close" in frame.columns else frame.iloc[:, 0]
    series.index = to_session_clock(pd.DatetimeIndex(series.index))
    return series.sort_index().rename("dxy4")


def prepare_inputs(
    data: Any,
    *,
    dxy: pd.Series | None = None,
    dxy_path: str | Path = DXY4_H1_PATH,
) -> X10Inputs:
    """Aggregate M1 gold into the M5 decision grid and its context (§3-§6).

    The M5 bars carry the decisions, the M1 bars the resolution, and
    ``m1_start``/``m1_end`` index the second inside the first. The H1 context
    goes through ``align_h1_to_m5`` — ``shift(1)`` then ``ffill`` — while the
    session VWAP does not: they are the two families of §6 and the whole point
    is that they obey different causality rules.
    """
    m1 = _m1_frame(data)
    m5 = resample_ohlc(m1, "5min")
    h1 = resample_ohlc(m1, "1h")

    close5 = m5["close"].to_numpy(dtype=np.float64)
    atr5 = atr_wilder(m5).to_numpy(dtype=np.float64)
    v, m, a = kinematics_nb(close5, atr5)
    l_inf, l_sup = x10_levels_nb(close5)

    ema50 = align_h1_to_m5(ema_h1(h1["close"]), m5.index).to_numpy(dtype=np.float64)
    atr_h1 = align_h1_to_m5(atr_wilder(h1), m5.index).to_numpy(dtype=np.float64)

    dxy_h1 = dxy if dxy is not None else _load_dxy_h1(dxy_path)
    if dxy_h1 is None:
        dxy_arr = np.full(len(m5), np.nan)
        dxy_ema_arr = np.full(len(m5), np.nan)
    else:
        dxy_arr = align_h1_to_m5(dxy_h1, m5.index).to_numpy(dtype=np.float64)
        dxy_ema_arr = align_h1_to_m5(ema_h1(dxy_h1), m5.index).to_numpy(dtype=np.float64)

    # Which M5 bar owns each M1 bar. ``resample_ohlc`` drops empty bins, so the
    # mapping goes through the M5 index itself rather than through arithmetic
    # on the timestamps — a market halt must not shift a whole session by one.
    owner = m5.index.get_indexer(m1.index.floor("5min"))
    if (owner < 0).any():
        raise ValueError("some M1 bars fall outside the M5 grid they were built from")
    bars = np.arange(len(m5))
    m1_start = np.searchsorted(owner, bars, side="left").astype(np.int64)
    m1_end = np.searchsorted(owner, bars, side="right").astype(np.int64)

    return X10Inputs(
        m5=m5,
        m1=m1,
        atr=atr5,
        v=v,
        m=m,
        a=a,
        l_inf=l_inf,
        l_sup=l_sup,
        vwap=session_mean_typical(m5).to_numpy(dtype=np.float64),
        ema50_h1=ema50,
        atr_h1=atr_h1,
        dxy=dxy_arr,
        dxy_ema50_h1=dxy_ema_arr,
        minute_of_day=(m5.index.hour * 60 + m5.index.minute).to_numpy(dtype=np.float64),
        session_id=session_ids(m5.index),
        m1_start=m1_start,
        m1_end=m1_end,
    )


# ═══════════════════════════════════════════════════════════════════════
# 2. INVESTIGATION PATH — pipeline() returns (pf, indicator)
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class X10Indicator:
    """The two registries of the run, indexed back onto wall-clock time."""

    events: pd.DataFrame
    trades: pd.DataFrame
    m5: pd.DataFrame
    m1: pd.DataFrame
    params: dict[str, Any]


def _events_frame(events: np.ndarray, m5_index: pd.DatetimeIndex) -> pd.DataFrame:
    frame = pd.DataFrame(events, columns=list(EVENT_COLUMNS))
    idx = frame["i_decision"].to_numpy(dtype=np.int64)
    frame.insert(0, "ts_decision", m5_index[idx] if len(frame) else pd.DatetimeIndex([]))
    for column in ("event", "scenario", "exit_reason", "cancel_reason"):
        frame[column] = frame[column].astype(np.int64)
    frame["event_name"] = [EVENT_NAMES[c] for c in frame["event"]]
    frame["scenario_name"] = [SCENARIO_NAMES[c] for c in frame["scenario"]]
    frame["exit_reason_name"] = [EXIT_REASON_NAMES[c] for c in frame["exit_reason"]]
    frame["cancel_reason_name"] = [CANCEL_REASON_NAMES[c] for c in frame["cancel_reason"]]
    return frame


def _trades_frame(
    trades: np.ndarray, m5_index: pd.DatetimeIndex, m1_index: pd.DatetimeIndex
) -> pd.DataFrame:
    frame = pd.DataFrame(trades, columns=list(TRADE_COLUMNS))
    if len(frame):
        frame.insert(0, "ts_decision", m5_index[frame["i_decision"].to_numpy(dtype=np.int64)])
        frame.insert(1, "ts_fill", m1_index[frame["i_fill_m1"].to_numpy(dtype=np.int64)])
        frame.insert(2, "ts_exit", m1_index[frame["i_exit_m1"].to_numpy(dtype=np.int64)])
    else:
        for i, name in enumerate(("ts_decision", "ts_fill", "ts_exit")):
            frame.insert(i, name, pd.DatetimeIndex([]))
    frame["scenario_name"] = [SCENARIO_NAMES[int(c)] for c in frame["scenario"]]
    frame["exit_reason_name"] = [EXIT_REASON_NAMES[int(c)] for c in frame["exit_reason"]]
    return frame


def _orders_from_trades(
    trades: np.ndarray, n_m1: int
) -> tuple[np.ndarray, np.ndarray]:
    """Turn the trade registry into the ``(size, price)`` arrays of ``from_orders``.

    One row of the registry becomes two orders in ounces at the prices the
    kernel resolved. When entry and exit land on the *same* minute — a stop
    taken in the very minute of the fill — the exit is moved one bar forward
    (or, at the very end of the history, the entry one bar back): a bar holds a
    single order, and shifting the stamp keeps both prices, hence the P&L,
    exactly as the kernel computed them.
    """
    size = np.full(n_m1, np.nan)
    price = np.full(n_m1, np.nan)
    for row in range(trades.shape[0]):
        i_fill = int(trades[row, T_I_FILL_M1])
        i_exit = int(trades[row, T_I_EXIT_M1])
        if i_exit == i_fill:
            if i_exit + 1 < n_m1:
                i_exit += 1
            else:
                i_fill -= 1
        units = trades[row, T_Q] * trades[row, T_LOTS] * CONTRACT_SIZE
        for idx, signed, px in (
            (i_fill, units, trades[row, T_FILL_PX]),
            (i_exit, -units, trades[row, T_EXIT_PX]),
        ):
            if not np.isnan(size[idx]):
                raise ValueError(f"two orders collide on M1 bar {idx}")
            size[idx] = signed
            price[idx] = px
    return size, price


def pipeline(
    data: Any,
    z: float = 1.0,
    a_min: float = 0.2,
    k_s: float = 1.0,
    spread: float = X10_SPREAD_DEFAULT,
    init_cash: float = X10_INIT_CASH,
    risk_frac: float = RISK_FRAC,
    inputs: X10Inputs | None = None,
) -> tuple[vbt.Portfolio, X10Indicator]:
    """Run the x10 engine and wrap its orders in a portfolio.

    ``z``, ``a_min`` and ``k_s`` are the three axes of annexe A.2; everything
    else the spec freezes lives in ``framework.x10_engine`` and is not
    reachable from here.

    The portfolio is built on the **M1** index with ``from_orders`` at the
    kernel's own fill and exit prices, zero fees and zero slippage: the spread
    is already inside those prices (§9, §12), and leverage is left unbounded
    because the 1:100 cap of §11 has already been applied lot by lot — a second
    cap here would silently reject orders the engine believes it placed.
    """
    inputs = inputs if inputs is not None else prepare_inputs(data)
    events, trades = run_engine(
        **inputs.kernel_kwargs(spread),
        z=z,
        a_min=a_min,
        k_s=k_s,
        init_cash=init_cash,
        risk_frac=risk_frac,
    )

    close_m1 = inputs.m1["close"]
    size, price = _orders_from_trades(trades, len(close_m1))
    pf = vbt.Portfolio.from_orders(
        close=close_m1,
        size=size,
        price=price,
        size_type="amount",
        direction="both",
        fees=0.0,
        slippage=0.0,
        init_cash=init_cash,
        leverage=np.inf,
        freq="1min",
    )

    indicator = X10Indicator(
        events=_events_frame(events, inputs.m5.index),
        trades=_trades_frame(trades, inputs.m5.index, inputs.m1.index),
        m5=inputs.m5,
        m1=inputs.m1,
        params=dict(
            z=z,
            a_min=a_min,
            k_s=k_s,
            spread=spread,
            init_cash=init_cash,
            risk_frac=risk_frac,
        ),
    )
    return pf, indicator


# ═══════════════════════════════════════════════════════════════════════
# 3. TRACE (§13) AND THE COUNTS THE SPEC REQUIRES
# ═══════════════════════════════════════════════════════════════════════


def emit_event_trace(
    indicator: X10Indicator,
    path: str | Path | None = None,
    *,
    extended: bool = False,
) -> pd.DataFrame:
    """Emit the event trace of §13, one row per event, timestamps in UTC.

    Twenty-one columns in the order the spec fixes, so that
    ``scripts/reconcile_x10_events.py`` can diff it rung by rung against the
    MT5 and QuantConnect traces. ``extended=True`` appends ``cancel_reason``,
    which §13 does not define but which turns "the EA armed and the Python did
    not" into a named disagreement; leave it off for the contract file.

    The decision stamps are converted back to UTC from the naive New York clock
    the engine runs on. ``ambiguous``/``nonexistent`` are pinned so a DST fold
    can never raise here — gold has no bar in the 17:00-18:00 break where the
    switch happens, but a synthetic index might.
    """
    events = indicator.events
    stamps = pd.DatetimeIndex(events["ts_decision"])
    utc = stamps.tz_localize(
        "America/New_York", ambiguous=True, nonexistent="shift_forward"
    ).tz_convert("UTC")

    trace = pd.DataFrame(
        {
            "ts_decision": utc.strftime("%Y-%m-%d %H:%M:%S"),
            "event": events["event_name"].to_numpy(),
            "scenario": events["scenario_name"].to_numpy(),
            "level": events["level"].to_numpy(),
            "d": events["d"].to_numpy(dtype=np.int64),
            "atr": events["atr"].to_numpy(),
            "v": events["v"].to_numpy(),
            "m": events["m"].to_numpy(),
            "a": events["a"].to_numpy(),
            "ctx_ema": events["ctx_ema"].to_numpy(dtype=np.int64),
            "ctx_vwap": events["ctx_vwap"].to_numpy(dtype=np.int64),
            "ctx_dxy": events["ctx_dxy"].to_numpy(dtype=np.int64),
            "vwap": events["vwap"].to_numpy(),
            "ema50_h1": events["ema50_h1"].to_numpy(),
            "dxy": events["dxy"].to_numpy(),
            "stop": events["stop"].to_numpy(),
            "target": events["target"].to_numpy(),
            "r_est": events["r_est"].to_numpy(),
            "fill_px": events["fill_px"].to_numpy(),
            "exit_px": events["exit_px"].to_numpy(),
            "exit_reason": events["exit_reason_name"].to_numpy(),
        },
        columns=list(TRACE_COLUMNS),
    )
    for column, decimals in _TRACE_DECIMALS.items():
        trace[column] = trace[column].round(decimals)
    if extended:
        trace["cancel_reason"] = events["cancel_reason_name"].to_numpy()

    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        trace.to_csv(path, index=False)
    return trace


def count_summary(indicator: X10Indicator) -> dict[str, Any]:
    """Counts, and only counts — the figures §8 obliges the campaign to publish.

    Nothing monetary is computed here on purpose: reading a P&L outside a
    declared campaign consumes the trial budget of annexe A.3 without
    declaring it (``docs/research/HOLDOUT_POLICY.md``).
    """
    events = indicator.events
    trades = indicator.trades
    cancels = events[events["event"] == EVENT_CANCEL]

    # §8: the R rejection rate is "candidates below 1 over candidates". A
    # candidate is anything that reached the R test, i.e. survived the context
    # gate: an entry, or a refusal decided at R or after it (window, size).
    reached_r = (events["event"] == EVENT_ENTRY) | (
        (events["event"] == EVENT_CANCEL)
        & events["cancel_reason"].isin([CR_R, CR_WINDOW, CR_SIZE])
    )
    r_frame = events[reached_r]
    year = r_frame["ts_decision"].dt.year if len(r_frame) else pd.Series(dtype=int)
    r_rejected = (r_frame["cancel_reason"] == CR_R) if len(r_frame) else pd.Series(dtype=bool)

    return {
        "n_m5": len(indicator.m5),
        "n_m1": len(indicator.m1),
        "events_by_type": events["event_name"].value_counts(),
        "cancels_by_reason": cancels["cancel_reason_name"].value_counts(),
        "trades_by_scenario": trades["scenario_name"].value_counts(),
        "trades_by_year": (
            trades["ts_decision"].dt.year.value_counts().sort_index()
            if len(trades)
            else pd.Series(dtype=int)
        ),
        "exits_by_reason": trades["exit_reason_name"].value_counts(),
        "r_candidates_by_year": year.value_counts().sort_index(),
        "r_rejected_by_year": (
            r_rejected.groupby(year).sum().sort_index() if len(r_frame) else pd.Series(dtype=int)
        ),
        "ctx_refused": int((cancels["cancel_reason"] == CR_CTX).sum()),
    }


# ═══════════════════════════════════════════════════════════════════════
# 4. GRID-SEARCH PATH — pipeline_nb (@vbt.parameterized)
# ═══════════════════════════════════════════════════════════════════════


@vbt.parameterized(
    merge_func="concat",
    execute_kwargs=make_execute_kwargs("XAU x10 grid"),
)
def pipeline_nb(
    data: Any,
    z: float,
    a_min: float,
    k_s: float,
    spread: float = X10_SPREAD_DEFAULT,
    init_cash: float = X10_INIT_CASH,
    risk_frac: float = RISK_FRAC,
    inputs: X10Inputs | None = None,
    ann_factor: float = FX_MINUTE_ANN_FACTOR,
    cutoff: float = 0.05,
    metric_type: int = SHARPE_RATIO,
) -> float:
    """Grid path — one scalar metric per ``(z, a_min, k_s)`` of annexe A.2.

    Numerically identical to ``pipeline()``: the same kernel, the same
    portfolio. The parallelism is the decorator's, not a second code path.
    """
    pf, _ = pipeline(
        data,
        z=z,
        a_min=a_min,
        k_s=k_s,
        spread=spread,
        init_cash=init_cash,
        risk_frac=risk_frac,
        inputs=inputs,
    )
    returns = pf.returns.values
    if returns.ndim > 1:
        returns = returns[:, 0]
    return float(compute_metric_nb(returns, metric_type, ann_factor, cutoff))


def _selection_index(data: Any) -> pd.DatetimeIndex:
    if isinstance(data, vbt.Data):
        return pd.DatetimeIndex(data.wrapper.index)
    return pd.DatetimeIndex(data.index)


def run_grid(
    data: Any,
    *,
    z: list[float] | float = list(GRID_Z),
    a_min: list[float] | float = list(GRID_A_MIN),
    k_s: list[float] | float = list(GRID_K_S),
    metric_type: int = SHARPE_RATIO,
    **kwargs: Any,
) -> pd.Series:
    """Sweep the 27 configurations of annexe A.2 — never on the frozen slice.

    The context is resampled once and handed to every combination: the grid
    varies three thresholds, not the price history.
    """
    assert_not_optimizing(_selection_index(data))

    def _param(value):
        if isinstance(value, (list, tuple, np.ndarray)):
            return vbt.Param(list(value))
        return value

    kwargs.setdefault("inputs", prepare_inputs(data))
    return pipeline_nb(
        data,
        z=_param(z),
        a_min=_param(a_min),
        k_s=_param(k_s),
        metric_type=metric_type,
        **kwargs,
    )


# ═══════════════════════════════════════════════════════════════════════
# 5. CV FACTORY — create_cv_pipeline (@vbt.cv_split)
# ═══════════════════════════════════════════════════════════════════════


def create_cv_pipeline(
    splitter: Any,
    metric_type: int = SHARPE_RATIO,
    **pipeline_defaults: Any,
):
    """Build a ``@vbt.cv_split`` walk-forward pipeline over the A.2 grid.

    The holdout guard fires twice: on the splitter's own index when it carries
    one, and on the data handed to the returned callable. A walk-forward that
    selects on 2026 bars is the exact failure ``docs/research/HOLDOUT_POLICY.md``
    exists to prevent, and by the time the folds are cut it is too late to see.
    """
    splitter_kwargs = pipeline_defaults.pop("splitter_kwargs", {})
    splitter_index = getattr(splitter, "index", None)
    if isinstance(splitter_index, pd.Index):
        assert_not_optimizing(pd.DatetimeIndex(splitter_index))

    defaults = dict(
        spread=X10_SPREAD_DEFAULT,
        init_cash=X10_INIT_CASH,
        risk_frac=RISK_FRAC,
        ann_factor=FX_MINUTE_ANN_FACTOR,
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
                "XAU x10 combos", pbar_kwargs=dict(leave=False)
            ),
            merge_func="concat",
        ),
        execute_kwargs=make_execute_kwargs("XAU x10 CV splits"),
        merge_func="concat",
        return_grid="all",
        attach_bounds="index",
    )
    def _cv_pipeline(
        data: Any,
        z: float,
        a_min: float,
        k_s: float,
        spread: float = defaults["spread"],
        init_cash: float = defaults["init_cash"],
        risk_frac: float = defaults["risk_frac"],
        ann_factor: float = defaults["ann_factor"],
        cutoff: float = defaults["cutoff"],
        metric_type: int = defaults["metric_type"],
    ) -> float:
        pf, _ = pipeline(
            data,
            z=z,
            a_min=a_min,
            k_s=k_s,
            spread=spread,
            init_cash=init_cash,
            risk_frac=risk_frac,
        )
        returns = pf.returns.values
        if returns.ndim > 1:
            returns = returns[:, 0]
        return float(compute_metric_nb(returns, metric_type, ann_factor, cutoff))

    def cv_pipeline(data: Any, **kwargs: Any):
        assert_not_optimizing(_selection_index(data))
        return _cv_pipeline(data, **kwargs)

    return cv_pipeline


# ═══════════════════════════════════════════════════════════════════════
# 6. CLI — single / grid / cv
# ═══════════════════════════════════════════════════════════════════════


def _print_counts(summary: dict[str, Any]) -> None:
    print(f"\nM5 bars: {summary['n_m5']}   M1 bars: {summary['n_m1']}")
    for key in (
        "events_by_type",
        "cancels_by_reason",
        "trades_by_scenario",
        "trades_by_year",
        "exits_by_reason",
    ):
        print(f"\n[{key}]")
        print(summary[key].to_string() if len(summary[key]) else "  (none)")
    print("\n[R rule, §8 — rejected / candidates, by year]")
    cand = summary["r_candidates_by_year"]
    rej = summary["r_rejected_by_year"]
    for year in cand.index:
        print(f"  {year}: {int(rej.get(year, 0))} / {int(cand[year])}")
    print(f"\nCandidates refused on context (§6.4): {summary['ctx_refused']}")


if __name__ == "__main__":
    import argparse
    import sys
    import time
    from pathlib import Path as _Path

    _SRC = _Path(__file__).resolve().parent.parent
    if str(_SRC) not in sys.path:
        sys.path.insert(0, str(_SRC))

    from utils import load_gold_data

    parser = argparse.ArgumentParser(description="XAUUSD x10 levels reference engine")
    parser.add_argument("--mode", choices=("single", "grid", "cv"), default="single")
    parser.add_argument("--start", default="2019-01-01")
    parser.add_argument("--end", default="2025-12-31")
    parser.add_argument("--z", type=float, default=1.0)
    parser.add_argument("--a-min", type=float, default=0.2)
    parser.add_argument("--k-s", type=float, default=1.0)
    parser.add_argument("--spread", type=float, default=X10_SPREAD_DEFAULT)
    parser.add_argument("--init-cash", type=float, default=X10_INIT_CASH)
    parser.add_argument("--trace-out", default=None)
    parser.add_argument("--trades-out", default=None)
    parser.add_argument("--n-folds", type=int, default=8)
    # Counts are printed always; the performance report is opt-in. A trial read
    # outside a declared campaign is a trial spent (annexe A.3).
    parser.add_argument("--stats", dest="stats", action="store_true")
    parser.add_argument("--no-stats", dest="stats", action="store_false")
    parser.set_defaults(stats=False)
    args = parser.parse_args()

    raw, _ = load_gold_data()
    raw = raw.loc[args.start : args.end]
    print(f"Loaded {len(raw)} M1 bars  {raw.index[0]} -> {raw.index[-1]}")

    t0 = time.perf_counter()
    inputs = prepare_inputs(raw)
    t1 = time.perf_counter()
    print(f"prepare_inputs: {t1 - t0:.2f}s  ({len(inputs.m5)} M5 bars)")

    if args.mode == "single":
        pf, ind = pipeline(
            raw,
            z=args.z,
            a_min=args.a_min,
            k_s=args.k_s,
            spread=args.spread,
            init_cash=args.init_cash,
            inputs=inputs,
        )
        print(f"pipeline: {time.perf_counter() - t1:.2f}s")
        _print_counts(count_summary(ind))
        if args.trace_out:
            trace = emit_event_trace(ind, args.trace_out)
            print(f"\nTrace written: {args.trace_out}  ({len(trace)} events)")
        if args.trades_out:
            ind.trades.to_csv(args.trades_out, index=False)
            print(f"Trades written: {args.trades_out}  ({len(ind.trades)} trades)")
        if args.stats:
            from framework.pipeline_utils import analyze_portfolio

            analyze_portfolio(pf, name="XAU x10", show_charts=False)

    elif args.mode == "grid":
        grid = run_grid(raw, inputs=inputs)
        print(grid.to_string())

    else:
        daily = inputs.m5["close"].resample("1D").last().dropna().index
        splitter = vbt.Splitter.from_purged_walkforward(
            daily, n_folds=args.n_folds, n_test_folds=1, purge_td="1 day", min_train_folds=3
        )
        cv_pipeline = create_cv_pipeline(splitter)
        grid_perf, best_perf = cv_pipeline(
            raw,
            z=vbt.Param(list(GRID_Z)),
            a_min=vbt.Param(list(GRID_A_MIN)),
            k_s=vbt.Param(list(GRID_K_S)),
        )
        print(best_perf.to_string())
