//+------------------------------------------------------------------+
//| FxRiskManager.mqh                                                |
//|                                                                  |
//| Portfolio-level risk controller. Provides:                       |
//|   * Virtual sub-equity per sleeve (allocation-weighted equity)   |
//|   * Risk-based position sizing                                   |
//|   * Daily volatility-targeted leverage (sigma21 / sigma63)       |
//|   * Peak-equity drawdown circuit breaker                         |
//|   * Margin-usage breaker for tail-risk protection                |
//|                                                                  |
//| All multi-call state (peak equity, current global leverage,      |
//| breaker flag) is persisted via Global Variables so it survives   |
//| EA reloads and recompiles.                                       |
//+------------------------------------------------------------------+
#ifndef __FX_RISK_MANAGER_MQH__
#define __FX_RISK_MANAGER_MQH__

#include "FxCommon.mqh"
#include "FxLogger.mqh"
#include "FxTradeHelpers.mqh"

class CRiskManager
{
private:
    double m_alloc_mr;
    double m_alloc_ts;
    double m_alloc_rsi;
    double m_alloc_h1;
    double m_alloc_gold;
    double m_target_vol;
    double m_max_leverage;
    double m_vol_floor;
    double m_dd_cap;
    bool   m_dd_cap_enabled;
    bool   m_margin_cap_enabled;
    double m_margin_cap_pct;

    // Throttle for the per-tick drawdown breaker check (1 Hz cap).
    datetime m_last_dd_check;

public:
    CRiskManager() : m_alloc_mr(0.80), m_alloc_ts(0.10), m_alloc_rsi(0.10),
                     m_alloc_h1(0.0), m_alloc_gold(0.0),
                     m_target_vol(FX_VOL_TARGET_GLOBAL),
                     m_max_leverage(FX_MAX_LEVERAGE_GLOBAL),
                     m_vol_floor(FX_MIN_VOL_FLOOR),
                     m_dd_cap(FX_DD_CAP_DEFAULT),
                     m_dd_cap_enabled(false),
                     m_margin_cap_enabled(true),
                     m_margin_cap_pct(FX_MARGIN_CAP_DEFAULT),
                     m_last_dd_check(0) {}

    bool Init(double alloc_mr, double alloc_ts, double alloc_rsi,
              double target_vol, double max_leverage, double vol_floor,
              bool dd_cap_enabled, double dd_cap, bool reset_dd_state,
              bool margin_cap_enabled = true,
              double margin_cap_pct = FX_MARGIN_CAP_DEFAULT,
              double alloc_h1 = 0.0,
              double alloc_gold = 0.0)
    {
        // Allocations must sum exactly to 1.0 across all sleeves.
        double sum = alloc_mr + alloc_ts + alloc_rsi + alloc_h1 + alloc_gold;
        if(MathAbs(sum - 1.0) > 1e-6)
        {
            PrintFormat("CRiskManager::Init: allocations sum=%.4f != 1.0 "
                        "(mr=%.2f ts=%.2f rsi=%.2f h1=%.2f gold=%.2f)",
                        sum, alloc_mr, alloc_ts, alloc_rsi, alloc_h1,
                        alloc_gold);
            return false;
        }
        m_alloc_mr  = alloc_mr;
        m_alloc_ts  = alloc_ts;
        m_alloc_rsi = alloc_rsi;
        m_alloc_h1  = alloc_h1;
        m_alloc_gold = alloc_gold;
        m_target_vol = target_vol;
        m_max_leverage = max_leverage;
        m_vol_floor = vol_floor;
        m_dd_cap_enabled = dd_cap_enabled;
        m_dd_cap = dd_cap;
        m_margin_cap_enabled = margin_cap_enabled;
        m_margin_cap_pct = margin_cap_pct;

        if(reset_dd_state)
        {
            GlobalVariableSet(GV_PEAK_EQUITY, AccountInfoDouble(ACCOUNT_EQUITY));
            if(GlobalVariableCheck(GV_DD_TRIGGERED))
                GlobalVariableDel(GV_DD_TRIGGERED);
        }
        if(!GlobalVariableCheck(GV_PEAK_EQUITY))
            GlobalVariableSet(GV_PEAK_EQUITY, AccountInfoDouble(ACCOUNT_EQUITY));
        if(!GlobalVariableCheck(GV_GLOBAL_LEVERAGE))
            GlobalVariableSet(GV_GLOBAL_LEVERAGE, 1.0);
        return true;
    }

    //--- Virtual sub-equity for a sleeve = total equity * allocation %.
    double SubEquity(ESleeveID id) const
    {
        double equity = AccountInfoDouble(ACCOUNT_EQUITY);
        switch(id)
        {
            case SLEEVE_MR_MACRO:    return equity * m_alloc_mr;
            case SLEEVE_TS_MOMENTUM: return equity * m_alloc_ts;
            case SLEEVE_RSI_DAILY:   return equity * m_alloc_rsi;
            case SLEEVE_H1_MOMENTUM: return equity * m_alloc_h1;
            case SLEEVE_GOLD_MOMENTUM: return equity * m_alloc_gold;
        }
        return 0.0;
    }

    //--- Latest global leverage (vol-target based, persisted GV).
    double GlobalLeverage() const
    {
        if(!GlobalVariableCheck(GV_GLOBAL_LEVERAGE)) return 1.0;
        double lev = GlobalVariableGet(GV_GLOBAL_LEVERAGE);
        if(lev <= 0.0 || !MathIsValidNumber(lev)) return 1.0;
        return lev;
    }

    //--- Translate a sleeve-level risk budget into a normalized lot size.
    double LotsFor(ESleeveID sleeve, string symbol, double risk_pct,
                   double sl_distance_price, double extra_lev = 1.0)
    {
        double sub_eq = SubEquity(sleeve);
        if(sub_eq <= 0.0) return 0.0;
        double risk_money = sub_eq * risk_pct * GlobalLeverage() * extra_lev;
        return LotsForRisk(symbol, risk_money, sl_distance_price);
    }

    //+--------------------------------------------------------------+
    //| Reconstruct a daily return series for the lookback window.   |
    //|                                                              |
    //| Algorithm:                                                   |
    //|   1. Aggregate realised P&L (profit + commission + swap) per |
    //|      day from the account deal history.                      |
    //|   2. Add the current floating P&L of open positions to the   |
    //|      most recent bucket so trending periods are not biased   |
    //|      down by ignored open exposure.                          |
    //|   3. Roll an equity curve forward from initial deposit +     |
    //|      pre-window cumulative P&L, then convert each daily P&L  |
    //|      change into a percentage return relative to the         |
    //|      preceding day's equity (no within-day compounding).     |
    //|                                                              |
    //| Returns lookback_days entries; older days first.             |
    //+--------------------------------------------------------------+
    int BuildDailyEquityReturns(int lookback_days, double &rets[])
    {
        ArrayResize(rets, lookback_days);
        ArrayInitialize(rets, 0.0);

        datetime now = TimeGMT();
        datetime today_start = FloorToDayUTC(now);
        datetime from = (datetime)((long)today_start
                                   - (long)lookback_days * 86400L);
        // Pull deals from one extra year prior so cumulative P&L before
        // the window is captured in equity[0]'s baseline.
        datetime cum_from = (datetime)((long)from - 365L * 86400L);
        if(!HistorySelect(cum_from, now)) return 0;

        // Per-day realised P&L bucket (one slot per lookback day).
        double daily_pnl[];
        ArrayResize(daily_pnl, lookback_days);
        ArrayInitialize(daily_pnl, 0.0);

        double pnl_before_window = 0.0;
        int total = HistoryDealsTotal();
        for(int i = 0; i < total; i++)
        {
            ulong tk = HistoryDealGetTicket(i);
            if(tk == 0) continue;
            datetime t = (datetime)HistoryDealGetInteger(tk, DEAL_TIME);
            double profit = HistoryDealGetDouble(tk, DEAL_PROFIT);
            double comm   = HistoryDealGetDouble(tk, DEAL_COMMISSION);
            double swap   = HistoryDealGetDouble(tk, DEAL_SWAP);
            double pnl    = profit + comm + swap;
            if(t < from)
            {
                pnl_before_window += pnl;
            }
            else
            {
                int day_idx = (int)((t - from) / 86400);
                if(day_idx >= 0 && day_idx < lookback_days)
                    daily_pnl[day_idx] += pnl;
            }
        }

        // Add floating P&L (open positions) to today's bucket.
        double floating_pnl = 0.0;
        int n_pos = PositionsTotal();
        for(int i = 0; i < n_pos; i++)
        {
            ulong tk = PositionGetTicket(i);
            if(tk == 0) continue;
            floating_pnl += PositionGetDouble(POSITION_PROFIT);
            floating_pnl += PositionGetDouble(POSITION_SWAP);
        }
        if(lookback_days > 0)
            daily_pnl[lookback_days - 1] += floating_pnl;

        // Equity baseline = initial deposit + cumulative pre-window P&L.
        double initial_deposit = TesterStatistics(STAT_INITIAL_DEPOSIT);
        if(initial_deposit <= 0.0)
            initial_deposit = AccountInfoDouble(ACCOUNT_BALANCE);
        if(initial_deposit <= 0.0) return 0;

        double equity_prev = initial_deposit + pnl_before_window;
        if(equity_prev <= 0.0) equity_prev = initial_deposit;

        for(int i = 0; i < lookback_days; i++)
        {
            double equity_cur = equity_prev + daily_pnl[i];
            rets[i] = (equity_prev > 0.0)
                      ? (equity_cur - equity_prev) / equity_prev
                      : 0.0;
            equity_prev = (equity_cur > 0.0) ? equity_cur : equity_prev;
        }
        return lookback_days;
    }

    //--- Recompute the global leverage from realised volatility using
    //--- a max(sigma21, sigma63) blend (more conservative than either
    //--- horizon alone). Persisted to GV_GLOBAL_LEVERAGE.
    void RecomputeGlobalLeverage(CFxLogger &logger)
    {
        double rets[];
        int n = BuildDailyEquityReturns(80, rets);
        if(n < 21)
        {
            GlobalVariableSet(GV_GLOBAL_LEVERAGE, 1.0);
            logger.Info("RISK", "Insufficient history; leverage=1.0");
            return;
        }
        double sigma21 = ArrayStdDDof1(rets, n - 21, 21) * MathSqrt(252.0);
        double sigma63 = (n >= 63)
                         ? ArrayStdDDof1(rets, n - 63, 63) * MathSqrt(252.0)
                         : sigma21;
        double realized = MathMax(MathMax(sigma21, sigma63), m_vol_floor);
        double leverage = MathMin(m_target_vol / realized, m_max_leverage);
        if(!MathIsValidNumber(leverage) || leverage <= 0.0) leverage = 1.0;
        GlobalVariableSet(GV_GLOBAL_LEVERAGE, leverage);
        GlobalVariableSet(GV_LAST_DAILY_RECOMP, (double)TimeGMT());
        logger.Info("RISK",
                    StringFormat("Daily recompute: sigma21=%.4f sigma63=%.4f "
                                 "realized=%.4f -> lev=%.3f",
                                 sigma21, sigma63, realized, leverage));
    }

    //--- Peak-equity drawdown breaker. Closes every sleeve when the
    //--- running drawdown exceeds the configured cap. Throttled to one
    //--- check per second to keep tick handlers cheap on real-tick mode.
    bool CheckDDCircuitBreaker(CFxLogger &logger)
    {
        if(!m_dd_cap_enabled) return false;
        datetime now = TimeGMT();
        if(now == m_last_dd_check) return IsDDLocked();
        m_last_dd_check = now;

        double peak = GlobalVariableGet(GV_PEAK_EQUITY);
        double cur  = AccountInfoDouble(ACCOUNT_EQUITY);
        if(cur > peak)
        {
            GlobalVariableSet(GV_PEAK_EQUITY, cur);
            return false;
        }
        if(peak <= 0.0) return false;
        double dd = 1.0 - cur / peak;
        if(dd >= m_dd_cap)
        {
            if(!GlobalVariableCheck(GV_DD_TRIGGERED) ||
               GlobalVariableGet(GV_DD_TRIGGERED) == 0.0)
            {
                logger.Error("RISK",
                    StringFormat("DD circuit-breaker FIRED: dd=%.2f%% "
                                 "(cap=%.2f%%) — closing all",
                                 dd * 100, m_dd_cap * 100));
                // These close by #define, not by the Inp_Magic* inputs.
                // Any new sleeve MUST be added here or it escapes the
                // breaker entirely.
                CloseAllByMagic(MAGIC_MR_MACRO,      "DD breaker");
                CloseAllByMagic(MAGIC_TS_MOMENTUM,   "DD breaker");
                CloseAllByMagic(MAGIC_RSI_DAILY,     "DD breaker");
                CloseAllByMagic(MAGIC_H1_MOMENTUM,   "DD breaker");
                CloseAllByMagic(MAGIC_GOLD_MOMENTUM, "DD breaker");
                GlobalVariableSet(GV_DD_TRIGGERED, 1.0);
                Alert("FX DD circuit-breaker triggered");
            }
            return true;
        }
        return false;
    }

    //--- True once the breaker has fired and has not been reset.
    bool IsDDLocked() const
    {
        if(!m_dd_cap_enabled) return false;
        if(!GlobalVariableCheck(GV_DD_TRIGGERED)) return false;
        return GlobalVariableGet(GV_DD_TRIGGERED) == 1.0;
    }

    //--- Margin-usage cap. Idempotent: deleverages or force-closes the
    //--- account once usage exceeds the threshold, then defers to the
    //--- daily recompute to restore the desired leverage as exposure
    //--- normalises.
    bool CheckMarginCap(CFxLogger &logger)
    {
        if(!m_margin_cap_enabled) return false;
        double equity = AccountInfoDouble(ACCOUNT_EQUITY);
        double margin = AccountInfoDouble(ACCOUNT_MARGIN);
        if(equity <= 0.0) return false;
        double usage = margin / equity;
        if(usage >= 0.85)
        {
            logger.Error("RISK",
                StringFormat("Margin usage critical %.1f%% — force-closing all "
                             "(cap=%.1f%%)",
                             usage * 100, m_margin_cap_pct * 100));
            CloseAllByMagic(MAGIC_MR_MACRO,      "margin critical");
            CloseAllByMagic(MAGIC_TS_MOMENTUM,   "margin critical");
            CloseAllByMagic(MAGIC_RSI_DAILY,     "margin critical");
            CloseAllByMagic(MAGIC_H1_MOMENTUM,   "margin critical");
            CloseAllByMagic(MAGIC_GOLD_MOMENTUM, "margin critical");
            return true;
        }
        if(usage >= m_margin_cap_pct)
        {
            double cur_lev = GlobalLeverage();
            double new_lev = cur_lev * 0.5;
            if(new_lev < 1.0) new_lev = 1.0;
            GlobalVariableSet(GV_GLOBAL_LEVERAGE, new_lev);
            logger.Warn("RISK",
                StringFormat("Margin usage %.1f%% > cap %.1f%% — leverage "
                             "%.2f -> %.2f",
                             usage * 100, m_margin_cap_pct * 100,
                             cur_lev, new_lev));
            return true;
        }
        return false;
    }
};

#endif // __FX_RISK_MANAGER_MQH__
