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

from framework import costs
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
    ``"mt5"`` for a broker daily parquet, ``"yahoo"`` for the long daily
    screening parquet, ``"fx_minute"`` for the long FX minute parquet.
    ``session_close_hour`` and ``ann_factor`` are the conventions
    ``gold_momentum.pipeline`` exposes so another instrument runs through it
    without a fork.
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

# Metals, energies and cash indices screened alongside the FX book. Their
# broker exports only start in 2022-11, too short to judge an edge, so the
# registry points them at the long daily parquets; the screening script
# overrides back to "mt5" for the ones whose long series does not match the
# broker's (see reports/research/screening_source_check.json).
_SCREENING_NAMES: dict[str, str] = {
    "XAG-USD": "XAGUSD", "XTI-USD": "XTIUSD", "XBR-USD": "XBRUSD",
    "XNG-USD": "XNGUSD", "US500": "US500Cash", "US100": "US100Cash",
    "US30": "US30Cash", "GER40": "GER40Cash", "JPN225": "JPN225Cash",
    "UK100": "UK100Cash",
}

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
    **{
        name: Instrument(
            name, "yahoo", mt5_symbol=mt5_name, note="long daily screening export"
        )
        for name, mt5_name in _SCREENING_NAMES.items()
    },
}


def load_instrument(
    symbol: str,
    loader_override: str | None = None,
) -> tuple[pd.DataFrame, vbt.Data]:
    """Load ``symbol`` through its registered loader, gold-export conventions.

    Returns what the loaders return — raw lowercase frame for Numba kernels,
    capitalized ``vbt.Data`` for native VBT functions — so the caller cannot
    tell which source it came from. ``utils`` is imported here rather than at
    module level: it pulls the data layer in, and the strategy modules must
    stay importable without it.

    ``loader_override`` picks another source for the same instrument without
    editing the registry: which of two histories is the more trustworthy is a
    per-study call (the screening script reads it off a verdict file), not a
    property of the tradable.
    """
    try:
        inst = INSTRUMENTS[symbol]
    except KeyError:
        raise KeyError(
            f"unknown instrument {symbol!r}; available: {', '.join(sorted(INSTRUMENTS))}"
        ) from None

    from utils import load_fx_data, load_gold_data, load_mt5_daily, load_screening_daily

    loader = loader_override or inst.loader
    if loader == "qc":
        if symbol != "XAU-USD":
            raise ValueError(
                f"the 'qc' loader only carries XAU-USD, not {symbol!r}: "
                "load_gold_data reads one hardcoded export."
            )
        return load_gold_data()
    if loader == "mt5":
        return load_mt5_daily(symbol)
    if loader == "yahoo":
        return load_screening_daily(symbol)
    if loader == "fx_minute":
        # Long minute export, naive broker-time index (not New York): only the
        # wall-clock label of the session boundary shifts, not the ordering.
        return load_fx_data(f"data/{symbol}_minute.parquet")
    raise ValueError(
        f"{symbol}: unknown loader {loader!r}, expected 'qc', 'mt5', 'yahoo' or 'fx_minute'"
    )


# ═══════════════════════════════════════════════════════════════════════
# 1. INVESTIGATION PATH — pipeline() returns (pf, indicator)
# ═══════════════════════════════════════════════════════════════════════


def pipeline(
    symbol: str,
    loader_override: str | None = None,
    **kwargs: Any,
) -> tuple[vbt.Portfolio, GoldMomentumIndicator]:
    """Run the gold sleeve on ``symbol``, with that instrument's conventions.

    Every ``gold_momentum.pipeline`` keyword is forwarded untouched. The two the
    registry knows about — ``session_close_hour`` and ``ann_factor`` — are
    injected only when the caller did not pass them: explicit wins.
    ``loader_override`` goes to ``load_instrument`` and picks the source.
    """
    _, data = load_instrument(symbol, loader_override=loader_override)
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

    Mirrors ``gold_momentum.create_cv_pipeline``, both carrying the
    **production** sizing (target_vol=0.55 / max_leverage=6.6). The gold factory
    was left on the pre-2026-07-26 tuning (0.25 / 3.0) until 2026-07-28 ; it is
    now aligned, so the two agree.
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


def carry_sign(symbol: str) -> float:
    """``+1`` si une position longue sur ``symbol`` encaisse le portage, ``-1`` sinon.

    Le registre est le seul endroit qui connaît la correspondance entre le nom
    projet (``USD-JPY``) et le nom courtier (``USDJPY``), donc c'est ici que la
    traduction se fait ; le signe lui-même vient du catalogue broker archivé.

    Un symbole inconnu paie : le défaut pessimiste évite qu'un instrument non
    catalogué se voie créditer un portage qu'on n'a pas vérifié.
    """
    instrument = INSTRUMENTS.get(symbol)
    if instrument is None:
        return -1.0
    return costs.swap_sign(instrument.mt5_symbol)
