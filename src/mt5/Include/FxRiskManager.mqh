//+------------------------------------------------------------------+
//| FxRiskManager.mqh                                                |
//| Sub-equity virtuel par sleeve, sizing par risque, vol-targeting  |
//| global (σ21 / σ63), circuit-breaker DD.                          |
//|                                                                  |
//| Reproduit la logique de combined_portfolio_v2.py (config         |
//| PRODUCTION : target_vol=0.28, max_lev=12, dd_cap désactivé).     |
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
    double m_alloc_h1;       // Phase D — alloc sleeve H1 momentum
    double m_target_vol;
    double m_max_leverage;
    double m_vol_floor;
    double m_dd_cap;
    bool   m_dd_cap_enabled;
    bool   m_margin_cap_enabled;
    double m_margin_cap_pct;       // Seuil margin/equity au-dessus duquel on déleverage (LaTeX § 13.2)

public:
    CRiskManager() : m_alloc_mr(0.80), m_alloc_ts(0.10), m_alloc_rsi(0.10),
                     m_alloc_h1(0.0),
                     m_target_vol(FX_VOL_TARGET_GLOBAL),
                     m_max_leverage(FX_MAX_LEVERAGE_GLOBAL),
                     m_vol_floor(FX_MIN_VOL_FLOOR),
                     m_dd_cap(FX_DD_CAP_DEFAULT),
                     m_dd_cap_enabled(false),
                     m_margin_cap_enabled(true),
                     m_margin_cap_pct(0.70) {}

    bool Init(double alloc_mr, double alloc_ts, double alloc_rsi,
              double target_vol, double max_leverage, double vol_floor,
              bool dd_cap_enabled, double dd_cap, bool reset_dd_state,
              bool margin_cap_enabled = true, double margin_cap_pct = 0.70,
              double alloc_h1 = 0.0)
    {
        // Validation des allocations (4 sleeves, sum = 1.0)
        double sum = alloc_mr + alloc_ts + alloc_rsi + alloc_h1;
        if(MathAbs(sum - 1.0) > 1e-6)
        {
            PrintFormat("CRiskManager::Init: allocations sum=%.4f != 1.0 "
                        "(mr=%.2f ts=%.2f rsi=%.2f h1=%.2f)",
                        sum, alloc_mr, alloc_ts, alloc_rsi, alloc_h1);
            return false;
        }
        m_alloc_mr  = alloc_mr;
        m_alloc_ts  = alloc_ts;
        m_alloc_rsi = alloc_rsi;
        m_alloc_h1  = alloc_h1;
        m_target_vol = target_vol;
        m_max_leverage = max_leverage;
        m_vol_floor = vol_floor;
        m_dd_cap_enabled = dd_cap_enabled;
        m_dd_cap = dd_cap;
        m_margin_cap_enabled = margin_cap_enabled;
        m_margin_cap_pct = margin_cap_pct;

        // Init state persistant
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

    //--- Sub-equity virtuel par sleeve = equity × allocation
    double SubEquity(ESleeveID id) const
    {
        double equity = AccountInfoDouble(ACCOUNT_EQUITY);
        switch(id)
        {
            case SLEEVE_MR_MACRO:    return equity * m_alloc_mr;
            case SLEEVE_TS_MOMENTUM: return equity * m_alloc_ts;
            case SLEEVE_RSI_DAILY:   return equity * m_alloc_rsi;
            case SLEEVE_H1_MOMENTUM: return equity * m_alloc_h1;
        }
        return 0.0;
    }

    //--- Levier global appliqué uniformément (lit la GlobalVariable)
    double GlobalLeverage() const
    {
        if(!GlobalVariableCheck(GV_GLOBAL_LEVERAGE)) return 1.0;
        double lev = GlobalVariableGet(GV_GLOBAL_LEVERAGE);
        if(lev <= 0.0 || !MathIsValidNumber(lev)) return 1.0;
        return lev;
    }

    //--- Sizing par risque pour un trade
    double LotsFor(ESleeveID sleeve, string symbol, double risk_pct,
                   double sl_distance_price, double extra_lev = 1.0)
    {
        double sub_eq = SubEquity(sleeve);
        if(sub_eq <= 0.0) return 0.0;
        double risk_money = sub_eq * risk_pct * GlobalLeverage() * extra_lev;
        return LotsForRisk(symbol, risk_money, sl_distance_price);
    }

    //--- Reconstruit returns daily de l'equity sur les `lookback` derniers jours.
    //--- Utilise HistorySelect + iteration des deals.
    int BuildDailyEquityReturns(int lookback_days, double &rets[])
    {
        ArrayResize(rets, lookback_days);
        ArrayInitialize(rets, 0.0);

        datetime now = TimeGMT();
        datetime from = now - lookback_days * 86400;
        if(!HistorySelect(from, now)) return 0;

        // Cumule profit par jour UTC
        int total = HistoryDealsTotal();
        // Map jour → P&L (utilise un buffer indexé par "jour relatif")
        double daily_pnl[];
        ArrayResize(daily_pnl, lookback_days);
        ArrayInitialize(daily_pnl, 0.0);

        for(int i = 0; i < total; i++)
        {
            ulong tk = HistoryDealGetTicket(i);
            if(tk == 0) continue;
            datetime t = (datetime)HistoryDealGetInteger(tk, DEAL_TIME);
            int day_idx = (int)((t - from) / 86400);
            if(day_idx < 0 || day_idx >= lookback_days) continue;
            double profit = HistoryDealGetDouble(tk, DEAL_PROFIT);
            double comm   = HistoryDealGetDouble(tk, DEAL_COMMISSION);
            double swap   = HistoryDealGetDouble(tk, DEAL_SWAP);
            daily_pnl[day_idx] += profit + comm + swap;
        }

        // Convertit P&L en "return" approx en divisant par equity courant
        // (approximation : pas d'equity par-jour disponible nativement)
        double equity = AccountInfoDouble(ACCOUNT_EQUITY);
        if(equity <= 0.0) return 0;
        for(int i = 0; i < lookback_days; i++)
            rets[i] = daily_pnl[i] / equity;
        return lookback_days;
    }

    //--- Recompute le levier global à partir des returns daily.
    void RecomputeGlobalLeverage(CFxLogger &logger)
    {
        double rets[];
        int n = BuildDailyEquityReturns(80, rets);
        if(n < 21)
        {
            // Pas assez d'historique → levier conservateur 1.0
            GlobalVariableSet(GV_GLOBAL_LEVERAGE, 1.0);
            logger.Info("RISK", "Insufficient history; leverage=1.0");
            return;
        }
        // σ21 et σ63 ddof=1 annualisés
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
                    StringFormat("Daily recompute: σ21=%.4f σ63=%.4f realized=%.4f → lev=%.3f",
                                 sigma21, sigma63, realized, leverage));
    }

    //--- Circuit-breaker DD : ferme tout si DD ≥ seuil.
    bool CheckDDCircuitBreaker(CFxLogger &logger)
    {
        if(!m_dd_cap_enabled) return false;
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
                    StringFormat("DD circuit-breaker FIRED: dd=%.2f%% (cap=%.2f%%) — closing all",
                                 dd*100, m_dd_cap*100));
                CloseAllByMagic(MAGIC_MR_MACRO,    "DD breaker");
                CloseAllByMagic(MAGIC_TS_MOMENTUM, "DD breaker");
                CloseAllByMagic(MAGIC_RSI_DAILY,   "DD breaker");
                CloseAllByMagic(MAGIC_H1_MOMENTUM, "DD breaker");
                GlobalVariableSet(GV_DD_TRIGGERED, 1.0);
                Alert("FX DD circuit-breaker triggered");
            }
            return true;
        }
        return false;
    }

    //--- True si le breaker a déjà été déclenché et pas reset.
    bool IsDDLocked() const
    {
        if(!m_dd_cap_enabled) return false;
        if(!GlobalVariableCheck(GV_DD_TRIGGERED)) return false;
        return GlobalVariableGet(GV_DD_TRIGGERED) == 1.0;
    }

    //--- Cap d'utilisation marge (LaTeX § 13.2 — production checklist).
    //--- usage = margin / equity.
    //---   > m_margin_cap_pct (70 %)  → réduit le levier global de moitié et logue.
    //---   > 0.85                     → ferme tout par sécurité.
    //--- Idempotent : si la marge revient sous le seuil, le RecomputeGlobalLeverage
    //--- daily reprendra la main et le levier remontera selon la vol-target.
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
                StringFormat("Margin usage critical %.1f%% — force-closing all (cap=%.1f%%)",
                             usage * 100, m_margin_cap_pct * 100));
            CloseAllByMagic(MAGIC_MR_MACRO,    "margin critical");
            CloseAllByMagic(MAGIC_TS_MOMENTUM, "margin critical");
            CloseAllByMagic(MAGIC_RSI_DAILY,   "margin critical");
            CloseAllByMagic(MAGIC_H1_MOMENTUM, "margin critical");
            return true;
        }
        if(usage >= m_margin_cap_pct)
        {
            double cur_lev = GlobalLeverage();
            double new_lev = cur_lev * 0.5;
            if(new_lev < 1.0) new_lev = 1.0;
            GlobalVariableSet(GV_GLOBAL_LEVERAGE, new_lev);
            logger.Warn("RISK",
                StringFormat("Margin usage %.1f%% > cap %.1f%% — leverage %.2f → %.2f",
                             usage * 100, m_margin_cap_pct * 100, cur_lev, new_lev));
            return true;
        }
        return false;
    }
};

#endif // __FX_RISK_MANAGER_MQH__
