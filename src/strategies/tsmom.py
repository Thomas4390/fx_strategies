"""TSMOM, one sleeve across instruments — gold plus the ten broker FX pairs.

The signal, the sizing and the execution conventions all live in
``strategies.gold_momentum``: forking it per instrument would fork the spec the
three engines (vbt, QuantConnect, MQL5) are diffed against. What differs from
one tradable to the next is the *data* — where the bars come from, at what hour
the broker cuts the session, how many sessions a year it trades — not the rule.

So this module is a registry, not a strategy. ``Instrument`` holds exactly those
facts, ``INSTRUMENTS`` enumerates them, and ``pipeline(symbol, ...)`` resolves
them before handing over to the gold sleeve untouched. Adding a pair is a dict
entry plus a parquet, never a new signal. ``mt5_symbol`` carries the broker's
own spelling ("EURUSD", no suffix): the exports are dashed, the terminal is not,
and that mapping is written down once rather than guessed at each call site.

Two entry points, matching the gold sleeve:
- ``pipeline(symbol, **params) -> (pf, ind)`` — investigation path
- ``create_cv_pipeline(splitter, metric_type)`` — ``@vbt.cv_split`` factory
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
import vectorbtpro as vbt

from framework.pipeline_utils import (
    SHARPE_RATIO,
    compute_metric_nb,
    make_execute_kwargs,
)
from strategies import gold_momentum
from strategies.gold_momentum import (
    DEFAULT_LOOKBACKS,
    GOLD_DAILY_ANN_FACTOR,
    SESSION_CLOSE_HOUR,
    GoldMomentumIndicator,
)

@dataclass(frozen=True)
class Instrument:
    """What the sleeve needs to know about a tradable, and nothing more.

    ``loader`` picks the source: ``"qc"`` for the QuantConnect gold export,
    ``"mt5"`` for a broker daily parquet. ``session_close_hour`` and
    ``ann_factor`` are the conventions ``gold_momentum.pipeline`` exposes so
    another instrument runs through it without a fork.
    """

    symbol: str
    loader: str
    session_close_hour: int = SESSION_CLOSE_HOUR
    ann_factor: float = GOLD_DAILY_ANN_FACTOR
    mt5_symbol: str = ""
    note: str = ""


# The ten majors/crosses exported from the broker terminal.
_MT5_PAIRS: tuple[str, ...] = (
    "EUR-USD", "GBP-USD", "USD-JPY", "USD-CAD", "USD-CHF",
    "AUD-USD", "NZD-USD", "EUR-GBP", "EUR-JPY", "GBP-JPY",
)

INSTRUMENTS: dict[str, Instrument] = {
    "XAU-USD": Instrument(
        "XAU-USD", "qc", mt5_symbol="XAUUSD", note="QuantConnect minute export"
    ),
    **{
        pair: Instrument(
            pair, "mt5", mt5_symbol=pair.replace("-", ""), note="broker daily export"
        )
        for pair in _MT5_PAIRS
    },
}


def load_instrument(symbol: str) -> tuple[pd.DataFrame, vbt.Data]:
    """Load ``symbol`` through its registered loader, gold-export conventions.

    Returns what the loaders return — raw lowercase frame for Numba kernels,
    capitalized ``vbt.Data`` for native VBT functions — so the caller cannot
    tell which source it came from. ``utils`` is imported here rather than at
    module level: it pulls the data layer in, and the strategy modules must
    stay importable without it.
    """
    try:
        inst = INSTRUMENTS[symbol]
    except KeyError:
        raise KeyError(
            f"unknown instrument {symbol!r}; available: {', '.join(sorted(INSTRUMENTS))}"
        ) from None

    from utils import load_gold_data, load_mt5_daily

    if inst.loader == "qc":
        if symbol != "XAU-USD":
            raise ValueError(
                f"the 'qc' loader only carries XAU-USD, not {symbol!r}: "
                "load_gold_data reads one hardcoded export."
            )
        return load_gold_data()
    if inst.loader == "mt5":
        return load_mt5_daily(symbol)
    raise ValueError(f"{symbol}: unknown loader {inst.loader!r}, expected 'qc' or 'mt5'")


# ═══════════════════════════════════════════════════════════════════════
# 1. INVESTIGATION PATH — pipeline() returns (pf, indicator)
# ═══════════════════════════════════════════════════════════════════════


def pipeline(symbol: str, **kwargs: Any) -> tuple[vbt.Portfolio, GoldMomentumIndicator]:
    """Run the gold sleeve on ``symbol``, with that instrument's conventions.

    Every ``gold_momentum.pipeline`` keyword is forwarded untouched. The two the
    registry knows about — ``session_close_hour`` and ``ann_factor`` — are
    injected only when the caller did not pass them: explicit wins.
    """
    _, data = load_instrument(symbol)
    inst = INSTRUMENTS[symbol]

    kwargs.setdefault("session_close_hour", inst.session_close_hour)
    kwargs.setdefault("ann_factor", inst.ann_factor)
    return gold_momentum.pipeline(data, **kwargs)


# ═══════════════════════════════════════════════════════════════════════
# 2. CV FACTORY — create_cv_pipeline (@vbt.cv_split)
# ═══════════════════════════════════════════════════════════════════════


def create_cv_pipeline(
    splitter: Any,
    metric_type: int = SHARPE_RATIO,
    **pipeline_defaults: Any,
):
    """Build a ``@vbt.cv_split`` pipeline for walk-forward cross-validation.

    Mirrors ``gold_momentum.create_cv_pipeline`` but carries the **production**
    sizing: that factory still defaults to target_vol=0.25 / max_leverage=3.0,
    which is the pre-2026-07-26 tuning and no longer what the sleeve trades.
    """
    splitter_kwargs = pipeline_defaults.pop("splitter_kwargs", {})

    defaults = dict(
        lookbacks=DEFAULT_LOOKBACKS, allow_short=False, target_vol=0.55,
        max_leverage=6.6, sl_stop=None, leverage=None, init_cash=None,
        slippage=None, fees=None, ann_factor=GOLD_DAILY_ANN_FACTOR,
        cutoff=0.05, metric_type=metric_type,
    )
    defaults.update(pipeline_defaults)

    @vbt.cv_split(
        splitter=splitter,
        splitter_kwargs=splitter_kwargs,
        takeable_args=["data"],
        parameterized_kwargs=dict(
            execute_kwargs=make_execute_kwargs(
                "TSMOM combos", pbar_kwargs=dict(leave=False)
            ),
            merge_func="concat",
        ),
        execute_kwargs=make_execute_kwargs("TSMOM CV splits"),
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
        pf, _ = gold_momentum.pipeline(
            data, lookbacks=lookbacks, allow_short=allow_short,
            target_vol=target_vol, max_leverage=max_leverage, sl_stop=sl_stop,
            leverage=leverage, init_cash=init_cash, slippage=slippage,
            fees=fees, ann_factor=ann_factor,
        )
        returns = pf.returns.values
        if returns.ndim > 1:
            returns = returns[:, 0]
        return float(compute_metric_nb(returns, metric_type, ann_factor, cutoff))

    return cv_pipeline
