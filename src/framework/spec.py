"""Strategy specification dataclasses.

A strategy is pure data: kernel function, signal function, parameter config,
and the mapping that connects them. The StrategyRunner consumes these specs
to execute backtests, parameter sweeps, and cross-validation pipelines.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ParamDef:
    """A parameter with its default value and optional sweep range.

    Parameters with ``sweep=None`` are fixed during grid search / CV.
    Parameters with a sweep list become ``vbt.Param([...])`` in sweep/CV mode.
    """

    default: Any
    sweep: list[Any] | None = None


@dataclass(frozen=True)
class IndicatorSpec:
    """Declares the ``vbt.IF`` setup for an indicator kernel.

    Attributes:
        class_name: VBT indicator class name (e.g. ``"MR_V1"``).
        short_name: Short name used in VBT internals (e.g. ``"mr_v1"``).
        input_names: Ordered input names matching the kernel's first positional args
            (e.g. ``("index_ns", "high_minute", "low_minute", "close_minute", "open_minute")``).
        param_names: Parameter names that the kernel accepts after the inputs.
        output_names: Names of the arrays returned by the kernel (in order).
        kernel_func: The ``@njit`` function that computes indicators.
    """

    class_name: str
    short_name: str
    input_names: tuple[str, ...]
    param_names: tuple[str, ...]
    output_names: tuple[str, ...]
    kernel_func: Callable


# Default mapping from IndicatorSpec.input_names to raw DataFrame columns.
# The runner uses this to auto-wire inputs unless overridden.
DEFAULT_INPUT_MAP: dict[str, str] = {
    "index_ns": "__index_ns__",  # special: resolved from vbt.dt.to_ns(raw.index)
    "high_minute": "high",
    "low_minute": "low",
    "close_minute": "close",
    "open_minute": "open",
    "volume_minute": "volume",
    # Daily strategies
    "close": "close",
    "returns": "__returns__",  # special: computed from close
}


@dataclass(frozen=True)
class MarketConfig:
    """Market-level conventions shared across strategies for the same asset class.

    Avoids duplicating FX-specific constants (EOD hour, evaluation frequency)
    in every strategy spec.
    """

    eod_hour: int = 21
    eod_minute: int = 0
    eval_freq: int = 5
    trading_days_per_year: int = 252


FX_MARKET = MarketConfig()
CRYPTO_MARKET = MarketConfig(eod_hour=0, eval_freq=1, trading_days_per_year=365)


@dataclass(frozen=True)
class PortfolioConfig:
    """Strategy-specific portfolio settings beyond global defaults.

    Attributes:
        leverage: Fixed float or ``"ind.<output_name>"`` for indicator-derived leverage.
        sl_stop: Fixed float, ``"param.<name>"`` reference, or ``None``.
        tp_stop: Fixed float, ``"param.<name>"`` reference, or ``None``.
        extra_kwargs: Additional kwargs passed directly to ``Portfolio.from_signals``.
            Use for strategy-specific settings like ``leverage_mode``, ``fees``, etc.
    """

    slippage: float = 0.00015
    fixed_fees: float = 0.0
    init_cash: float = 1_000_000.0
    freq: str = "1min"
    leverage: float | str = 1.0
    sl_stop: float | str | None = None
    tp_stop: float | str | None = None
    size_type: str | None = None
    accumulate: bool = False
    upon_opposite_entry: str | None = None
    extra_kwargs: dict[str, Any] | None = None


@dataclass(frozen=True)
class OverlayLine:
    """A line to overlay on the price/trade-signals chart.

    Attributes:
        source: ``"ind.<output>"`` or ``"data.<col>"`` following signal_args_map convention.
        label: Legend label for the trace.
        color: CSS color string (e.g. ``"#FF9800"``). None = auto.
        dash: Line dash style: ``"solid"``, ``"dash"``, or ``"dot"``.
    """

    source: str
    label: str
    color: str | None = None
    dash: str | None = None


@dataclass(frozen=True)
class PlotConfig:
    """Per-strategy plotting configuration.

    Attributes:
        overlays: Lines to draw on the price/trade-signals chart.
        subplot_indicators: Indicator outputs to plot as separate subplots.
            Each tuple is ``(source, label, fill_to_zero)``.
    """

    overlays: tuple[OverlayLine, ...] = ()
    subplot_indicators: tuple[tuple[str, str, bool], ...] = ()


@dataclass(frozen=True)
class StrategySpec:
    """Complete, self-contained strategy specification.

    Attributes:
        name: Human-readable strategy name.
        indicator: Indicator factory specification.
        signal_func: The ``@njit`` signal callback for ``Portfolio.from_signals``.
        signal_args_map: Ordered sequence of ``(rep_name, source)`` tuples.
            The order MUST match the positional args of ``signal_func`` after ``c``.
            Source conventions:
            - ``"data.<col>"`` → ``raw[col]``
            - ``"ind.<output>"`` → ``indicator_result.<output>.values``
            - ``"extra.index_ns"`` → the ``index_ns`` array
            - ``"param.<name>"`` → concrete parameter value (scalar or array)
        params: All parameters (indicator + signal + portfolio) with defaults and
            optional sweep ranges.
        portfolio_config: Portfolio simulation settings.
        takeable_args: Argument names that ``vbt.cv_split`` should slice across folds.
    """

    name: str
    indicator: IndicatorSpec
    signal_func: Callable
    signal_args_map: tuple[tuple[str, str], ...]
    params: dict[str, ParamDef]
    portfolio_config: PortfolioConfig = field(default_factory=PortfolioConfig)
    market: MarketConfig = field(default_factory=lambda: FX_MARKET)
    plot_config: PlotConfig = field(default_factory=PlotConfig)
    takeable_args: tuple[str, ...] = (
        "high_arr",
        "low_arr",
        "close_arr",
        "open_arr",
        "idx_ns",
    )

    def __post_init__(self) -> None:
        """Validate spec consistency at construction time."""
        # Indicator param_names must have corresponding ParamDef entries
        missing = set(self.indicator.param_names) - set(self.params.keys())
        if missing:
            raise ValueError(
                f"[{self.name}] Indicator params missing from spec.params: {missing}"
            )

        # Indicator output_names referenced in signal_args_map must exist
        for _, source in self.signal_args_map:
            if source.startswith("ind."):
                output_name = source.partition(".")[2]
                if output_name not in self.indicator.output_names:
                    raise ValueError(
                        f"[{self.name}] signal_args_map references unknown "
                        f"indicator output: {source!r}"
                    )

        # Source prefixes must be valid
        valid_prefixes = {"data", "ind", "extra", "param", "eval"}
        for _, source in self.signal_args_map:
            prefix = source.partition(".")[0]
            if prefix.startswith("eval:"):
                continue
            if prefix not in valid_prefixes:
                raise ValueError(
                    f"[{self.name}] Unknown source prefix in "
                    f"signal_args_map: {source!r}"
                )

    def default_params(self) -> dict[str, Any]:
        """Return a dict of parameter_name → default_value."""
        return {k: v.default for k, v in self.params.items()}

    def sweep_grid(self) -> dict[str, list[Any]]:
        """Return only the parameters that have a sweep range defined."""
        return {k: v.sweep for k, v in self.params.items() if v.sweep is not None}
