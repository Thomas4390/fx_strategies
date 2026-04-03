import vectorbtpro as vbt
from numba import njit
import numpy as np
import pandas as pd
from typing import Tuple
import plotly.graph_objects as go



def configure_figure_for_fullscreen(fig):
    """Configure automatiquement chaque figure pour s'adapter à la taille du browser"""
    fig.update_layout(
        # Supprime les dimensions fixes pour être responsive
        width=None,
        height=None,
        autosize=True,

        # Marges en pourcentage de la taille de l'écran
        margin=dict(l=30, r=30, t=60, b=30),

        title=dict(
            font=dict(size=20),
            x=0.5,
            xanchor='center'
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            font=dict(size=12)
        ),
        xaxis=dict(
            title_font=dict(size=14),
            tickfont=dict(size=12)
        ),
        yaxis=dict(
            title_font=dict(size=14),
            tickfont=dict(size=12)
        )
    )
    return fig

vbt.settings.set("plotting.pre_show_func", configure_figure_for_fullscreen)
vbt.settings.returns.year_freq = pd.Timedelta(hours=6.5) * 252


@njit
def find_day_boundaries_nb(index_ns: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int]:
    """Return the start / end indices of each trading day."""
    n = len(index_ns)
    start_idx = np.empty(n, dtype=np.int64)
    end_idx = np.empty(n, dtype=np.int64)

    if n == 0:
        return start_idx, end_idx, 0

    day_number = vbt.dt_nb.days_nb(ts=index_ns)
    current_day = day_number[0]
    day_counter = 0
    current_start = 0

    for i in range(1, n):
        if day_number[i] != current_day:
            start_idx[day_counter] = current_start
            end_idx[day_counter] = i
            day_counter += 1
            current_day = day_number[i]
            current_start = i

    start_idx[day_counter] = current_start
    end_idx[day_counter] = n
    day_counter += 1

    return start_idx, end_idx, day_counter


@njit
def compute_abs_move_from_open_nb(
        index_ns: np.ndarray,
        close_minute: np.ndarray,
        open_minute: np.ndarray,
) -> np.ndarray:
    """Compute |Close / FirstOpen − 1| for every intraday bar."""
    n = len(index_ns)
    move_open = np.full(n, np.nan)
    if n == 0:
        return move_open

    day_number = vbt.dt_nb.days_nb(ts=index_ns)
    current_day = day_number[0]
    current_start = 0

    for i in range(1, n):
        if day_number[i] != current_day:
            slice_ = slice(current_start, i)
            first_open = open_minute[current_start]
            if not np.isnan(first_open) and first_open != 0:
                move_open[slice_] = np.abs(close_minute[slice_] / first_open - 1.0)
            current_start = i
            current_day = day_number[i]

    if current_start < n:
        slice_ = slice(current_start, n)
        first_open = open_minute[current_start]
        if not np.isnan(first_open) and first_open != 0:
            move_open[slice_] = np.abs(close_minute[slice_] / first_open - 1.0)

    return move_open


@njit
def compute_sigma_open_nb(
        index_ns: np.ndarray,
        close_minute: np.ndarray,
        open_minute: np.ndarray,
        window_size: int,
) -> np.ndarray:
    """Return *sigma_open*: mean(|Close − Open|) by minute‑of‑day lagged 1 bar."""
    move_open = compute_abs_move_from_open_nb(index_ns, close_minute, open_minute)
    n = len(move_open)
    rolled = np.full(n, np.nan)

    minute_of_day = vbt.dt_nb.minute_nb(ts=index_ns)
    hour_of_day = vbt.dt_nb.hour_nb(ts=index_ns)
    group_key = hour_of_day * 60 + minute_of_day

    minp = max(1, window_size - 1)
    for minute_id in np.unique(group_key):
        idx = np.where(group_key == minute_id)[0]
        if idx.size >= minp:
            sub = move_open[idx]
            rolled[idx] = vbt.generic.nb.rolling_mean_1d_nb(
                sub,
                window=window_size,
                minp=minp,
            )

    sigma_open = np.full(n, np.nan)
    if n > 1:
        sigma_open[1:] = rolled[:-1]
    return sigma_open


@njit
def compute_daily_rolling_volatility_nb(
        index_ns: np.ndarray,
        close_minute: np.ndarray,
        window_size: int,
) -> np.ndarray:
    """Compute close‑to‑close rolling volatility and broadcast to minutes."""
    n = len(close_minute)
    if n == 0 or window_size <= 0:
        return np.full(n, np.nan)

    start_arr, end_arr, n_days = find_day_boundaries_nb(index_ns)
    if n_days < 2:
        return np.full(n, np.nan)

    last_close = np.full(n_days, np.nan)
    for d in range(n_days):
        if end_arr[d] > 0:
            last_close[d] = close_minute[end_arr[d] - 1]

    returns = np.full(n_days - 1, np.nan)
    for i in range(1, n_days):
        prev = last_close[i - 1]
        if not np.isnan(prev) and np.abs(prev) > 1e-9:
            returns[i - 1] = last_close[i] / prev - 1.0

    if len(returns) < window_size:
        return np.full(n, np.nan)

    rolling_std = vbt.generic.nb.rolling_std_1d_nb(
        returns,
        window=window_size,
        minp=window_size,
        ddof=1,
    )

    vol_per_minute = np.full(n, np.nan)
    for d in range(1, n_days):
        if d - 1 < rolling_std.size:
            std_val = rolling_std[d - 1]
            if start_arr[d] < end_arr[d]:
                vol_per_minute[start_arr[d]: end_arr[d]] = std_val

    return vol_per_minute


@njit
def compute_leverage_nb(
        rolling_vol_per_minute: np.ndarray,
        sigma_target: float,
        max_leverage: float,
        use_leverage: bool,
) -> np.ndarray:
    """Compute volatility‑targeted leverage capped at ``max_leverage``."""
    n = len(rolling_vol_per_minute)
    leverage = np.full(n, 1.0)

    if not use_leverage:
        return leverage

    for i in range(n):
        vol = rolling_vol_per_minute[i]
        if not np.isnan(vol) and vol > 1e-9:
            val = sigma_target / vol
            leverage[i] = min(val, max_leverage)

    return leverage


@njit
def compute_intraday_bands_nb(
        start_arr: np.ndarray,
        end_arr: np.ndarray,
        n_days: int,
        close_minute: np.ndarray,
        open_minute: np.ndarray,
        sigma_open: np.ndarray,
        band_mult: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (upper_band, lower_band) arrays for every intraday bar."""
    n = len(close_minute)
    upper_band = np.full(n, np.nan)
    lower_band = np.full(n, np.nan)

    for d in range(1, n_days):
        if d-1 >= 0 and end_arr[d-1] > 0:
            start = start_arr[d]
            end = end_arr[d]

            previous_close = close_minute[end_arr[d - 1] - 1]
            first_open = open_minute[start]

            if np.isnan(previous_close) or np.isnan(first_open):
                continue

            intraday_high_anchor = max(first_open, previous_close)
            intraday_low_anchor = min(first_open, previous_close)

            for i in range(start, min(end, n)):
                sigma_val = sigma_open[i]
                if not np.isnan(sigma_val):
                    upper_band[i] = intraday_high_anchor * (1.0 + band_mult * sigma_val)
                    lower_band[i] = intraday_low_anchor * (1.0 - band_mult * sigma_val)

    return upper_band, lower_band


@njit
def compute_bands_nb(
        index_ns: np.ndarray,
        high_minute: np.ndarray,
        low_minute: np.ndarray,
        volume_minute: np.ndarray,
        close_minute: np.ndarray,
        open_minute: np.ndarray,
        window_size: int,
        band_mult: float,
        max_leverage: float,
        sigma_target: float,
        use_leverage: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute all intraday indicators required by the strategy."""
    n = len(close_minute)
    if n == 0:
        nan_arr = np.full(n, np.nan)
        return nan_arr, nan_arr, nan_arr, nan_arr, nan_arr, nan_arr, nan_arr

    abs_move_open = compute_abs_move_from_open_nb(index_ns, close_minute, open_minute)
    sigma_open = compute_sigma_open_nb(index_ns, close_minute, open_minute, window_size)
    rolling_vol = compute_daily_rolling_volatility_nb(index_ns, close_minute, window_size)
    leverage = compute_leverage_nb(rolling_vol, sigma_target, max_leverage, use_leverage)

    start_arr, end_arr, n_days = find_day_boundaries_nb(index_ns)
    upper_band, lower_band = compute_intraday_bands_nb(
        start_arr,
        end_arr,
        n_days,
        close_minute,
        open_minute,
        sigma_open,
        band_mult,
    )

    group_lens = end_arr[:n_days] - start_arr[:n_days]

    vwap = vbt.indicators.nb.vwap_1d_nb(
        high_minute,
        low_minute,
        close_minute,
        volume_minute,
        group_lens,
    )

    return (
        upper_band,
        lower_band,
        sigma_open,
        abs_move_open,
        rolling_vol,
        leverage,
        vwap,
    )


@njit
def intraday_signal_nb(
        c,
        close_price_arr: np.ndarray,
        upper_band_arr: np.ndarray,
        lower_band_arr: np.ndarray,
        vwap_arr: np.ndarray,
        index_ns_arr: np.ndarray,
        eod_exit_trigger_hour_param: np.ndarray,
        eod_exit_trigger_minute_param: np.ndarray,
):
    """Return a 4‑tuple (entry_long, exit_long, entry_short, exit_short)."""
    ts_ns = index_ns_arr[c.i]
    cur_hour = vbt.dt_nb.hour_nb(ts_ns)
    cur_minute = vbt.dt_nb.minute_nb(ts_ns)

    selected_eod_hour = vbt.pf_nb.select_nb(c, eod_exit_trigger_hour_param)
    selected_eod_minute = vbt.pf_nb.select_nb(c, eod_exit_trigger_minute_param)

    entry_long = False
    exit_long = False
    entry_short = False
    exit_short = False

    is_eod_period = (
            (cur_hour > selected_eod_hour) or
            (cur_hour == selected_eod_hour and cur_minute >= selected_eod_minute)
    )

    if is_eod_period:
        if vbt.pf_nb.ctx_helpers.in_long_position_nb(c):
            exit_long = True
        if vbt.pf_nb.ctx_helpers.in_short_position_nb(c):
            exit_short = True
        return False, exit_long, False, exit_short

    if cur_minute % 30 == 0:
        close_px = vbt.pf_nb.select_nb(c, close_price_arr)
        up_band = vbt.pf_nb.select_nb(c, upper_band_arr)
        low_band = vbt.pf_nb.select_nb(c, lower_band_arr)
        vwap_val = vbt.pf_nb.select_nb(c, vwap_arr)

        if (
                np.isnan(close_px) or
                np.isnan(up_band) or
                np.isnan(low_band) or
                np.isnan(vwap_val)
        ):
            return False, False, False, False

        in_long_pos = vbt.pf_nb.ctx_helpers.in_long_position_nb(c)
        in_short_pos = vbt.pf_nb.ctx_helpers.in_short_position_nb(c)

        if not in_long_pos and not in_short_pos:
            if close_px > up_band:
                entry_long = True
                return entry_long, exit_long, entry_short, exit_short
            elif close_px < low_band:
                entry_short = True
                return entry_long, exit_long, entry_short, exit_short

        elif in_long_pos:
            long_exit_threshold = max(up_band, vwap_val)
            if close_px < long_exit_threshold:
                exit_long = True
                return entry_long, exit_long, entry_short, exit_short

        elif in_short_pos:
            short_exit_threshold = min(low_band, vwap_val)
            if close_px > short_exit_threshold:
                exit_short = True
                return entry_long, exit_long, entry_short, exit_short

    return False, False, False, False

@vbt.parameterized(execute_kwargs=dict(chunk_len="auto", engine="threadpool"), merge_func='column_stack')
def pipeline(
    high,
    low,
    close,
    open_,
    volume,
    idx_ns,
    window_size,
    band_mult,
    max_leverage,
    sigma_target,
    use_leverage,
    eod_exit_trigger_hour,
    eod_exit_trigger_minute,
    fixed_fees=0.0035,
):
    # ------------------------------------------------------------------
    # 1. Build custom indicator class (vectorbt interface factory)
    # ------------------------------------------------------------------
    IMIBase = (
        vbt.IF(
            class_name="IntradayMomentumIndicator",
            short_name="imi",
            input_names=[
                "index_ns",
                "high_minute",
                "low_minute",
                "volume_minute",
                "close_minute",
                "open_minute",
            ],
            param_names=[
                "window_size",
                "band_mult",
                "max_leverage",
                "sigma_target",
                "use_leverage",
            ],
            output_names=[
                "upper_band",
                "lower_band",
                "sigma_open",
                "abs_move_open",
                "rolling_vol",
                "leverage",
                "vwap",
            ],
        )
        .with_apply_func(
            compute_bands_nb,
            takes_1d=True,
            window_size=window_size,
            band_mult=band_mult,
            max_leverage=max_leverage,
            sigma_target=sigma_target,
            use_leverage=use_leverage,
        )
    )

    class IntradayMomentumIndicator(IMIBase):
        """Indicator subclass adding a convenience Plotly chart."""

        def plot(
            self,
            column: tuple | str | int | None = None,
            fig: go.Figure | None = None,
            **layout_kwargs,
        ) -> go.Figure:
            close = (
                self.select_col_from_obj(self.close_minute, column).rename("Close")
            )
            upper = (
                self.select_col_from_obj(self.upper_band, column).rename("Upper Band")
            )
            lower = (
                self.select_col_from_obj(self.lower_band, column).rename("Lower Band")
            )
            vwap_line = self.select_col_from_obj(self.vwap, column).rename("VWAP")

            fig = fig or go.Figure()

            close.vbt.plot(
                fig=fig,
                trace_kwargs=dict(
                    name="Close",
                    line=dict(width=2, color="blue"),
                )
            )

            # Étape 1: Tracer la bande inférieure
            lower.vbt.plot(
                fig=fig,
                trace_kwargs=dict(
                    name="Lower Band",
                    line=dict(width=1.2, color="grey"),
                ),
            )

            # Étape 2: Tracer la bande supérieure et la remplir vers la bande inférieure
            upper.vbt.plot(
                fig=fig,
                trace_kwargs=dict(
                    name="Upper Band",
                    line=dict(width=1.2, color="grey"),
                    fill="tonexty",
                    fillcolor="rgba(255, 255, 0, 0.2)",
                ),
            )

            vwap_line.vbt.plot(
                fig=fig,
                trace_kwargs=dict(
                    name="VWAP",
                    line=dict(
                        color="red",
                        width=1,
                        dash="dot"
                    )
                )
            )

            fig.update_layout(**layout_kwargs)
            return fig

    imi = IntradayMomentumIndicator.run(
        index_ns=idx_ns,
        high_minute=high,
        low_minute=low,
        volume_minute=volume,
        close_minute=close,
        open_minute=open_,
        window_size=window_size,
        band_mult=band_mult,
        max_leverage=max_leverage,
        sigma_target=sigma_target,
        use_leverage=use_leverage,
        jitted_loop=True,
        jitted_warmup=True,
        execute_kwargs=dict(engine='threadpool', n_chunks='auto'),
    )

    pf = vbt.Portfolio.from_signals(
        close,
        signal_func_nb=intraday_signal_nb,
        signal_args=(
            vbt.Rep("close_arr"),
            vbt.Rep("upper_band_arr"),
            vbt.Rep("lower_band_arr"),
            vbt.Rep("vwap_arr"),
            vbt.Rep("index_arr"),
            vbt.Rep("eod_exit_trigger_hour"),
            vbt.Rep("eod_exit_trigger_minute"),
        ),
        broadcast_named_args=dict(
            close_arr=close,
            upper_band_arr=imi.upper_band.values,
            lower_band_arr=imi.lower_band.values,
            vwap_arr=imi.vwap.values,
            index_arr=idx_ns,
            eod_exit_trigger_hour=eod_exit_trigger_hour,
            eod_exit_trigger_minute=eod_exit_trigger_minute,
        ),
        leverage=imi.leverage.values,
        fixed_fees=fixed_fees,
    )

    return pf, imi



@vbt.parameterized(execute_kwargs=dict(chunk_len="auto", engine="threadpool"), merge_func='column_stack')
def pipeline_parameterized(
    high,
    low,
    close,
    open_,
    volume,
    idx_ns,
    window_size,
    band_mult,
    max_leverage,
    sigma_target,
    use_leverage,
    eod_exit_trigger_hour,
    eod_exit_trigger_minute,
    fixed_fees=0.0035,
):
    # ------------------------------------------------------------------
    # 1. Build custom indicator class (vectorbt interface factory)
    # ------------------------------------------------------------------
    IMIBase = (
        vbt.IF(
            class_name="IntradayMomentumIndicator",
            short_name="imi",
            input_names=[
                "index_ns",
                "high_minute",
                "low_minute",
                "volume_minute",
                "close_minute",
                "open_minute",
            ],
            param_names=[
                "window_size",
                "band_mult",
                "max_leverage",
                "sigma_target",
                "use_leverage",
            ],
            output_names=[
                "upper_band",
                "lower_band",
                "sigma_open",
                "abs_move_open",
                "rolling_vol",
                "leverage",
                "vwap",
            ],
        )
        .with_apply_func(
            compute_bands_nb,
            takes_1d=True,
            window_size=window_size,
            band_mult=band_mult,
            max_leverage=max_leverage,
            sigma_target=sigma_target,
            use_leverage=use_leverage,
        )
    )

    class IntradayMomentumIndicator(IMIBase):
        """Indicator subclass adding a convenience Plotly chart."""

        def plot(
            self,
            column: tuple | str | int | None = None,
            fig: go.Figure | None = None,
            **layout_kwargs,
        ) -> go.Figure:
            close = (
                self.select_col_from_obj(self.close_minute, column).rename("Close")
            )
            upper = (
                self.select_col_from_obj(self.upper_band, column).rename("Upper Band")
            )
            lower = (
                self.select_col_from_obj(self.lower_band, column).rename("Lower Band")
            )
            vwap_line = self.select_col_from_obj(self.vwap, column).rename("VWAP")

            fig = fig or go.Figure()

            close.vbt.plot(
                fig=fig,
                trace_kwargs=dict(
                    name="Close",
                    line=dict(width=2, color="blue"),
                )
            )

            # Étape 1: Tracer la bande inférieure
            lower.vbt.plot(
                fig=fig,
                trace_kwargs=dict(
                    name="Lower Band",
                    line=dict(width=1.2, color="grey"),
                ),
            )

            # Étape 2: Tracer la bande supérieure et la remplir vers la bande inférieure
            upper.vbt.plot(
                fig=fig,
                trace_kwargs=dict(
                    name="Upper Band",
                    line=dict(width=1.2, color="grey"),
                    fill="tonexty",
                    fillcolor="rgba(255, 255, 0, 0.2)",
                ),
            )

            vwap_line.vbt.plot(
                fig=fig,
                trace_kwargs=dict(
                    name="VWAP",
                    line=dict(
                        color="red",
                        width=1,
                        dash="dot"
                    )
                )
            )

            fig.update_layout(**layout_kwargs)
            return fig

    imi = IntradayMomentumIndicator.run(
        index_ns=idx_ns,
        high_minute=high,
        low_minute=low,
        volume_minute=volume,
        close_minute=close,
        open_minute=open_,
        window_size=window_size,
        band_mult=band_mult,
        max_leverage=max_leverage,
        sigma_target=sigma_target,
        use_leverage=use_leverage,
        jitted_loop=True,
        jitted_warmup=True,
        execute_kwargs=dict(engine='threadpool', n_chunks='auto'),
    )

    pf = vbt.Portfolio.from_signals(
        close,
        signal_func_nb=intraday_signal_nb,
        signal_args=(
            vbt.Rep("close_arr"),
            vbt.Rep("upper_band_arr"),
            vbt.Rep("lower_band_arr"),
            vbt.Rep("vwap_arr"),
            vbt.Rep("index_arr"),
            vbt.Rep("eod_exit_trigger_hour"),
            vbt.Rep("eod_exit_trigger_minute"),
        ),
        broadcast_named_args=dict(
            close_arr=close,
            upper_band_arr=imi.upper_band.values,
            lower_band_arr=imi.lower_band.values,
            vwap_arr=imi.vwap.values,
            index_arr=idx_ns,
            eod_exit_trigger_hour=eod_exit_trigger_hour,
            eod_exit_trigger_minute=eod_exit_trigger_minute,
        ),
        leverage=imi.leverage.values,
        fixed_fees=fixed_fees,
    )

    return pf


# =============================================================================
# 1. DÉFINITION DES CONSTANTES MÉTRIQUES (COMPATIBLES NUMBA)
# =============================================================================

# Constantes pour les métriques (utilisables dans Numba)
TOTAL_RETURN = 0
SHARPE_RATIO = 1
CALMAR_RATIO = 2
SORTINO_RATIO = 3
OMEGA_RATIO = 4
ANNUALIZED_RETURN = 5
MAX_DRAWDOWN = 6
PROFIT_FACTOR = 7
VALUE_AT_RISK = 8
TAIL_RATIO = 9
ANNUALIZED_VOLATILITY = 10
INFORMATION_RATIO = 11
DOWNSIDE_RISK = 12
COND_VALUE_AT_RISK = 13


# =============================================================================
# 2. FONCTION DE DISPATCH POUR LES MÉTRIQUES (NUMBA COMPATIBLE)
# =============================================================================

@njit(nogil=True)
def compute_metric_nb(
        returns,
        metric_type,
        ann_factor=252.0 * 390.0,
        cutoff=0.05,
):
    """
    Fonction de dispatch pour calculer différentes métriques de performance.

    Args:
        returns: Array des rendements
        metric_type: Type de métrique (constante entière)
        ann_factor: Facteur d'annualisation
        cutoff: Seuil pour VaR

    Returns:
        float: Valeur de la métrique calculée
    """
    if metric_type == TOTAL_RETURN:
        return vbt.ret_nb.total_return_nb(returns=returns)

    elif metric_type == SHARPE_RATIO:
        return vbt.ret_nb.sharpe_ratio_nb(returns=returns, ann_factor=ann_factor)

    elif metric_type == CALMAR_RATIO:
        return vbt.ret_nb.calmar_ratio_nb(returns=returns, ann_factor=ann_factor)

    elif metric_type == SORTINO_RATIO:
        return vbt.ret_nb.sortino_ratio_nb(returns=returns, ann_factor=ann_factor)

    elif metric_type == OMEGA_RATIO:
        return vbt.ret_nb.omega_ratio_nb(returns=returns)  # Pas d'ann_factor

    elif metric_type == ANNUALIZED_RETURN:
        return vbt.ret_nb.annualized_return_nb(returns=returns, ann_factor=ann_factor)

    elif metric_type == MAX_DRAWDOWN:
        return -vbt.ret_nb.max_drawdown_nb(returns=returns)  # Négatif pour maximiser

    elif metric_type == PROFIT_FACTOR:
        return vbt.ret_nb.profit_factor_nb(returns=returns)

    elif metric_type == VALUE_AT_RISK:
        return -vbt.ret_nb.value_at_risk_nb(returns=returns, cutoff=cutoff)  # Négatif pour minimiser risque

    elif metric_type == TAIL_RATIO:
        return vbt.ret_nb.tail_ratio_nb(returns=returns)

    elif metric_type == ANNUALIZED_VOLATILITY:
        return -vbt.ret_nb.annualized_volatility_nb(returns=returns, ann_factor=ann_factor)  # Négatif pour minimiser

    elif metric_type == INFORMATION_RATIO:
        return vbt.ret_nb.information_ratio_nb(returns=returns)  # Pas d'ann_factor

    elif metric_type == DOWNSIDE_RISK:
        return -vbt.ret_nb.downside_risk_nb(returns=returns, ann_factor=ann_factor)  # Négatif pour minimiser

    elif metric_type == COND_VALUE_AT_RISK:
        return -vbt.ret_nb.cond_value_at_risk_nb(returns=returns, cutoff=cutoff)  # Négatif pour minimiser

    else:
        # Fallback vers total return
        return vbt.ret_nb.total_return_nb(returns=returns)


@vbt.parameterized(execute_kwargs=dict(chunk_len="auto", engine="threadpool"), merge_func='concat')
@njit(nogil=True)
def pipeline_nb(
        high_arr,
        low_arr,
        close_arr,
        open_arr,
        volume_arr,
        idx_ns,
        window_size,
        band_mult,
        max_leverage,
        sigma_target,
        use_leverage,
        eod_exit_trigger_hour,
        eod_exit_trigger_minute,
        init_cash: float = 1_000_000,
        fixed_fees: float = 0.0035,
        ann_factor: float = 252. * 390.,
        cutoff: float = 0.05,
        metric_type: int = 0
):
    """
    Pipeline de cross-validation avec splitter dynamique.
    """
    target_shape = close_arr.shape

    # Calcul des indicateurs techniques
    upper_band, lower_band, sigma_open, abs_move_open, rolling_vol, leverage, vwap = compute_bands_nb(
        index_ns=idx_ns,
        close_minute=close_arr[:, 0],
        high_minute=high_arr[:, 0],
        low_minute=low_arr[:, 0],
        open_minute=open_arr[:, 0],
        volume_minute=volume_arr[:, 0],
        window_size=window_size,
        band_mult=band_mult,
        max_leverage=max_leverage,
        sigma_target=sigma_target,
        use_leverage=use_leverage,
    )

    # Configuration des arrays de sortie
    eod_hour_arr = np.full(target_shape, eod_exit_trigger_hour, dtype=np.int32)
    eod_minute_arr = np.full(target_shape, eod_exit_trigger_minute, dtype=np.int32)
    group_lens = np.full(close_arr.shape[1], 1)

    # Simulation du portfolio
    sim_out = vbt.pf_nb.from_signal_func_nb(
        target_shape=target_shape,
        group_lens=group_lens,
        init_cash=init_cash,
        cash_sharing=False,
        open=open_arr,
        high=high_arr,
        low=low_arr,
        close=close_arr,
        signal_func_nb=intraday_signal_nb,
        signal_args=(
            close_arr,
            upper_band.reshape(-1, 1),
            lower_band.reshape(-1, 1),
            vwap.reshape(-1, 1),
            idx_ns,
            eod_hour_arr,
            eod_minute_arr,
        ),
        fixed_fees=fixed_fees,
        leverage=leverage,
        post_segment_func_nb=vbt.pf_nb.save_post_segment_func_nb,
        in_outputs=vbt.pf_nb.init_FSInOutputs_nb(target_shape, group_lens, cash_sharing=False),
    )

    # Extraction des rendements
    returns = sim_out.in_outputs.returns

    # Calcul de la métrique sélectionnée
    metric_value = compute_metric_nb(
        returns=returns,
        metric_type=metric_type,
        ann_factor=ann_factor,
        cutoff=cutoff
    )

    return metric_value


# =============================================================================
# 3. PIPELINE GÉNÉRIQUE AVEC SÉLECTION DE MÉTRIQUE
# =============================================================================

def create_cv_pipeline(splitter, metric_type=TOTAL_RETURN, **pipeline_defaults):
    """
    Crée une pipeline de cross-validation avec un splitter et une métrique spécifiés.

    Args:
        splitter: Instance de Splitter VectorBT PRO ou nom de méthode
        metric_type: Type de métrique à optimiser (utiliser les constantes)
        **pipeline_defaults: Paramètres par défaut pour la pipeline

    Returns:
        Fonction décorée pour la cross-validation
    """

    # Paramètres par défaut pour la pipeline
    default_params = {
        'init_cash': 1_000_000.,
        'fixed_fees': 0.0035,
        'ann_factor': 252.0 * 390.0,  # Votre valeur
        'cutoff': 0.05,
        'metric_type': metric_type
    }
    default_params.update(pipeline_defaults)

    # Créer la fonction décorée dynamiquement
    @vbt.cv_split(
        splitter=splitter,
        takeable_args=["high_arr", "low_arr", "close_arr", "open_arr", "volume_arr", "idx_ns"],
        parameterized_kwargs=dict(
            execute_kwargs=dict(chunk_len="auto", engine="threadpool"),
            merge_func="concat"
        ),
        return_grid='all',
        merge_func="concat",
        attach_bounds="index",
    )
    @njit(nogil=True)
    def cv_pipeline(
            high_arr,
            low_arr,
            close_arr,
            open_arr,
            volume_arr,
            idx_ns,
            window_size,
            band_mult,
            max_leverage,
            sigma_target,
            use_leverage,
            eod_exit_trigger_hour,
            eod_exit_trigger_minute,
            init_cash: float = default_params['init_cash'],
            fixed_fees: float = default_params['fixed_fees'],
            ann_factor: float = default_params['ann_factor'],
            cutoff: float = default_params['cutoff'],
            metric_type: int = default_params['metric_type']
    ):
        """
        Pipeline de cross-validation avec splitter dynamique.
        """
        target_shape = close_arr.shape

        # Calcul des indicateurs techniques
        upper_band, lower_band, sigma_open, abs_move_open, rolling_vol, leverage, vwap = compute_bands_nb(
            index_ns=idx_ns,
            close_minute=close_arr[:, 0],
            high_minute=high_arr[:, 0],
            low_minute=low_arr[:, 0],
            open_minute=open_arr[:, 0],
            volume_minute=volume_arr[:, 0],
            window_size=window_size,
            band_mult=band_mult,
            max_leverage=max_leverage,
            sigma_target=sigma_target,
            use_leverage=use_leverage,
        )

        # Configuration des arrays de sortie
        eod_hour_arr = np.full(target_shape, eod_exit_trigger_hour, dtype=np.int32)
        eod_minute_arr = np.full(target_shape, eod_exit_trigger_minute, dtype=np.int32)
        group_lens = np.full(close_arr.shape[1], 1)

        # Simulation du portfolio
        sim_out = vbt.pf_nb.from_signal_func_nb(
            target_shape=target_shape,
            group_lens=group_lens,
            init_cash=init_cash,
            cash_sharing=False,
            open=open_arr,
            high=high_arr,
            low=low_arr,
            close=close_arr,
            signal_func_nb=intraday_signal_nb,
            signal_args=(
                close_arr,
                upper_band.reshape(-1, 1),
                lower_band.reshape(-1, 1),
                vwap.reshape(-1, 1),
                idx_ns,
                eod_hour_arr,
                eod_minute_arr,
            ),
            fixed_fees=fixed_fees,
            leverage=leverage,
            post_segment_func_nb=vbt.pf_nb.save_post_segment_func_nb,
            in_outputs=vbt.pf_nb.init_FSInOutputs_nb(target_shape, group_lens, cash_sharing=False),
        )

        # Extraction des rendements
        returns = sim_out.in_outputs.returns

        # Calcul de la métrique sélectionnée
        metric_value = compute_metric_nb(
            returns=returns,
            metric_type=metric_type,
            ann_factor=ann_factor,
            cutoff=cutoff
        )

        return metric_value

    return cv_pipeline


def run_cv_pipeline(splitter, metric_type=TOTAL_RETURN, **kwargs):
    """
    Crée et exécute une pipeline de cross-validation, retournant les résultats de la grille et la sélection.

    Args:
        splitter: Instance de Splitter VectorBT PRO
        metric_type: Type de métrique à optimiser
        **kwargs: Paramètres pour la pipeline (données + paramètres d'optimisation)

    Returns:
        tuple: (grid_results, selection_results)
    """
    # Séparer les paramètres de pipeline des paramètres d'exécution
    pipeline_params = {}
    execution_params = {}

    # Paramètres de pipeline par défaut
    pipeline_param_names = {'init_cash', 'fixed_fees', 'ann_factor', 'cutoff'}

    for key, value in kwargs.items():
        if key in pipeline_param_names:
            pipeline_params[key] = value
        else:
            execution_params[key] = value

    # Créer la pipeline
    pipeline = create_cv_pipeline(
        splitter=splitter,
        metric_type=metric_type,
        **pipeline_params
    )

    # Exécuter la pipeline
    results = pipeline(**execution_params)

    # Avec return_grid='all', results est un tuple (grid_results, selection_results)
    if isinstance(results, tuple) and len(results) == 2:
        grid_results, selection_results = results
        return grid_results, selection_results
    else:
        # Si pour une raison quelconque, on n'a qu'un seul résultat
        return results



if __name__ == '__main__':
    data = vbt.HDFData.pull("../data/cleaned/us_tickers_verified_cleaned.h5")
    data = data.xloc['2017-01-01':]
    index_ns = vbt.dt.to_ns(data.index)

    print(data.data)
    print(index_ns)

    window_size = 10
    band_mult = 0.7
    max_leverage = 1.0
    sigma_target = 0.02
    use_leverage = False
    eod_exit_trigger_hour = 15
    eod_exit_trigger_minute = 55

    ticker = 'NFLX'

    high = data.data[ticker].High
    low = data.data[ticker].Low
    close = data.data[ticker].Close
    volume = data.data[ticker].Volume
    open_ = data.data[ticker].Open

    high_arr = vbt.to_2d_array(high)
    low_arr = vbt.to_2d_array(low)
    close_arr = vbt.to_2d_array(close)
    open_arr = vbt.to_2d_array(open_)
    volume_arr = vbt.to_2d_array(volume)

    # metric = pipeline_nb(
    #     open_arr=open_arr,
    #     high_arr=high_arr,
    #     low_arr=low_arr,
    #     close_arr=close_arr,
    #     volume_arr=volume_arr,
    #     idx_ns=index_ns,
    #     window_size=vbt.Param(list(np.arange(start=2, stop=100, step=1))),
    #     band_mult=vbt.Param(list(np.linspace(start=0.1, stop=2.0, num=20))),
    #     max_leverage=vbt.Param([max_leverage]),
    #     sigma_target=vbt.Param([sigma_target]),
    #     use_leverage=vbt.Param([use_leverage]),
    #     eod_exit_trigger_hour=vbt.Param([eod_exit_trigger_hour]),
    #     eod_exit_trigger_minute=vbt.Param([eod_exit_trigger_minute]),
    #     metric_type=SHARPE_RATIO,
    # )
    #
    # print(metric)

    pf, imi = pipeline(
        high=high,
        low=low,
        close=close,
        open_=open_,
        volume=volume,
        idx_ns=index_ns,
        window_size=window_size,
        band_mult=band_mult,
        max_leverage=max_leverage,
        sigma_target=sigma_target,
        use_leverage=use_leverage,
        eod_exit_trigger_hour=eod_exit_trigger_hour,
        eod_exit_trigger_minute=eod_exit_trigger_minute,
    )

    print(pf.wrapper.columns)
    print(pf.stats())

    # date_slice = slice('2025-01-15', '2025-01-16')
    #
    # fig = pf.xloc[date_slice].plot_trade_signals()
    # imi.xloc[date_slice].plot(fig=fig)
    # fig.show(renderer="browser")

    from portfolio_analyzer import analyze_portfolio_v5

    results = analyze_portfolio_v5(
        portfolio=pf,
        all_data=data,
        ticker=ticker,
        output_dir="../reports",
        show_charts=True,
        save_excel=True
    )

    # index_for_splitter_daily = data.resample('D').index
    # index_for_splitter = data.index
    #
    # # Créer le splitter avec des ranges annuels
    # annual_splitter_daily = vbt.Splitter.from_ranges(
    #     index_for_splitter_daily,
    #     every="YS",  # Year Start - crée un range pour chaque année
    #     split=0.5,  # 80% train, 20% test
    #     set_labels=["train", "test"]
    # )
    #
    # annual_splitter = vbt.Splitter.from_ranges(
    #     index_for_splitter,
    #     every="YS",  # Year Start - crée un range pour chaque année
    #     split=0.5,  # 80% train, 20% test
    #     set_labels=["train", "test"]
    # )
    #
    # # Visualiser le splitter
    # print("📊 Visualisation du splitter annuel :")
    # annual_splitter_daily.plot().show(renderer="browser")
    # print(f"📈 Nombre de splits : {len(annual_splitter_daily.splits)}")
    # print(f"📋 Informations sur les splits :")
    # for i, split in enumerate(annual_splitter_daily.splits):
    #     print(f"  Split {i}: {split}")
    #
    #
    # grid_perf, best_perf = run_cv_pipeline(
    #     splitter=annual_splitter,
    #     metric_type=SHARPE_RATIO,
    #     ann_factor=252.0 * 390.0,
    #     high_arr=high_arr,
    #     low_arr=low_arr,
    #     close_arr=close_arr,
    #     open_arr=open_arr,
    #     volume_arr=volume_arr,
    #     idx_ns=index_ns,
    #     window_size=vbt.Param(list(np.arange(start=2, stop=100, step=3))),
    #     band_mult=vbt.Param(list(np.linspace(start=0.01, stop=2.0, num=20))),
    #     max_leverage=vbt.Param([1.0]),
    #     sigma_target=vbt.Param(list(np.linspace(start=0.01, stop=0.5, num=1))),
    #     use_leverage=vbt.Param([False]),
    #     eod_exit_trigger_hour=vbt.Param([15]),
    #     eod_exit_trigger_minute=vbt.Param([55])
    # )
    #
    # print(grid_perf)
    # print(best_perf)
    #
    # fig1 = grid_perf.vbt.heatmap(
    #     x_level='window_size',
    #     y_level='band_mult',
    #     slider_level='split'
    # ).show(renderer="browser")
    #
    # fig2 = grid_perf.vbt.volume(
    #     x_level='window_size',
    #     y_level='band_mult',
    #     z_level='sigma_target',
    #     slider_level='split'
    # ).show(renderer="browser")
    #
    # metric_name = 'sharpe_ratio'
    #
    # # Sauvegarder les résultats dans Excel
    # excel_path = save_cv_results_to_excel(
    #     ticker=ticker,
    #     grid_perf=grid_perf,
    #     best_perf=best_perf,
    #     metric_name=f'{metric_name}',
    #     file_path=f'../data/results/{ticker}_cv_results_{metric_name}.xlsx'
    # )
    #
    # print(f"Résultats sauvegardés dans: {excel_path}")




