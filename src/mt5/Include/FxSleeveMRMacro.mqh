//+------------------------------------------------------------------+
//| FxSleeveMRMacro.mqh                                              |
//|                                                                  |
//| Sleeve 1: intraday mean-reversion on the deviation between price |
//| and a daily-anchored VWAP, gated by a macro-regime filter.       |
//|                                                                  |
//| Specification:                                                   |
//|   * Universe        : 4 majors equal-weighted (EUR/GBP/USD/JPY)  |
//|   * Indicators      : VWAP daily anchor + Bollinger Bands on the |
//|                       (close - VWAP) deviation series            |
//|   * Macro filter    : 10Y-2Y spread < threshold AND unemployment |
//|                       not in a 3-month uptrend                   |
//|   * Session         : configurable UTC window (default 8-16)     |
//|   * Time stop       : 6 hours per trade                          |
//|   * Daily forced    : flat at 21:00 UTC                          |
//+------------------------------------------------------------------+
#ifndef __FX_SLEEVE_MR_MACRO_MQH__
#define __FX_SLEEVE_MR_MACRO_MQH__

#include <Trade/Trade.mqh>
#include "FxSleeveBase.mqh"
#include "FxCommon.mqh"
#include "FxIndicatorVWAP.mqh"
#include "FxIndicatorBBDeviation.mqh"
#include "FxMacroFilter.mqh"
#include "FxRiskManager.mqh"
#include "FxTradeHelpers.mqh"

// Inputs declared in the main EA file are visible here via textual
// inclusion (MQL5 inputs are global symbols).

#define FX_MR_MAX_PAIRS 8

class CSleeveMRMacro : public CSleeveBase
{
private:
    int           m_n_pairs;
    string        m_symbols[FX_MR_MAX_PAIRS];
    CVWAPDaily    m_vwap[FX_MR_MAX_PAIRS];
    CBBDeviation  m_bb[FX_MR_MAX_PAIRS];
    datetime      m_last_m1_bar[FX_MR_MAX_PAIRS];
    CTrade        m_trade;

public:
    bool Init() override
    {
        m_magic  = Inp_MagicMR;
        m_name   = "MR_Macro";

        string raw[];
        int n = SplitCsv(Inp_MR_Pairs, raw);
        if(n <= 0 || n > FX_MR_MAX_PAIRS)
        {
            g_logger.Error(m_name, StringFormat("invalid Inp_MR_Pairs=%s",
                                                Inp_MR_Pairs));
            return false;
        }
        m_n_pairs = n;
        for(int i = 0; i < n; i++)
        {
            m_symbols[i] = MakeSymbolWithSuffix(raw[i], Inp_SymbolSuffix);
            if(!EnsureSymbolSelected(m_symbols[i])) return false;
            // Aim for 1500 M1 bars (~one trading day plus buffer). Accept
            // graceful degradation down to BBWindow+20 so backtests can
            // start at the broker's M1 history boundary; below that the
            // sleeve cannot warm the BB and disables itself.
            if(!EnsureHistory(m_symbols[i], PERIOD_M1, 1500))
            {
                int floor_bars = Inp_MR_BBWindow + 20;
                if(!EnsureHistory(m_symbols[i], PERIOD_M1, floor_bars))
                {
                    g_logger.Error(m_name, StringFormat(
                        "%s: cannot load even %d M1 bars; sleeve disabled",
                        m_symbols[i], floor_bars));
                    return false;
                }
                g_logger.Warn(m_name, StringFormat(
                    "%s: %d/1500 M1 bars; BB warmup degraded, signals "
                    "improve as bars accumulate",
                    m_symbols[i], (int)Bars(m_symbols[i], PERIOD_M1)));
            }

            m_bb[i].Init(Inp_MR_BBWindow, Inp_MR_BBAlpha);
            if(!m_vwap[i].Warmup(m_symbols[i]))
                g_logger.Warn(m_name,
                    StringFormat("VWAP warmup empty for %s; rebuilding "
                                 "as bars arrive", m_symbols[i]));
            WarmupBBFromHistory(i);
            m_last_m1_bar[i] = iTime(m_symbols[i], PERIOD_M1, 0);
        }
        m_trade.SetExpertMagicNumber(m_magic);
        m_trade.SetDeviationInPoints(FX_DEFAULT_DEVIATION);
        g_logger.Info(m_name, StringFormat("Init OK %d pairs (%s)",
                                           n, Inp_MR_Pairs));
        return true;
    }

    //--- New-bar M1 hook. Called from OnTick by the orchestrator; we
    //--- detect the per-pair bar transition internally so the EA can be
    //--- attached to any chart symbol.
    void OnNewBarM1(CMacroFilter &macro, CRiskManager &risk) override
    {
        if(risk.IsDDLocked()) return;
        for(int i = 0; i < m_n_pairs; i++)
        {
            datetime current_m1 = iTime(m_symbols[i], PERIOD_M1, 0);
            if(current_m1 == 0 || current_m1 == m_last_m1_bar[i]) continue;
            m_last_m1_bar[i] = current_m1;
            ProcessPair(i, macro, risk);
        }
    }

    //--- Enforce the per-trade time stop and the daily forced close.
    void CheckIntradayExits() override
    {
        datetime now = TimeGMT();
        MqlDateTime tm; TimeToStruct(now, tm);
        bool past_forced_close = (tm.hour >= Inp_MR_ForcedCloseHr);

        for(int i = PositionsTotal() - 1; i >= 0; i--)
        {
            ulong ticket = PositionGetTicket(i);
            if(ticket == 0) continue;
            if(PositionGetInteger(POSITION_MAGIC) != m_magic) continue;

            string sym = PositionGetString(POSITION_SYMBOL);
            if(!IsManagedSymbol(sym)) continue;

            datetime opened = (datetime)PositionGetInteger(POSITION_TIME);
            int age_hours = (int)((now - opened) / 3600);

            if(age_hours >= Inp_MR_TimeStopHours)
            {
                m_trade.PositionClose(ticket);
                g_logger.Info(m_name,
                    StringFormat("td_stop close ticket=%I64u sym=%s age=%dh",
                                 ticket, sym, age_hours));
            }
            else if(past_forced_close)
            {
                m_trade.PositionClose(ticket);
                g_logger.Info(m_name,
                    StringFormat("dt_stop close ticket=%I64u sym=%s (21h UTC)",
                                 ticket, sym));
            }
        }
    }

    int CloseAll(string reason) override
    {
        return CloseAllByMagic(m_magic, reason);
    }

private:
    //--- Process a single pair: feed the indicators with the latest
    //--- closed bar, evaluate filters, and submit an entry order if the
    //--- signal is active.
    void ProcessPair(int idx, CMacroFilter &macro, CRiskManager &risk)
    {
        MqlRates last[];
        if(CopyRates(m_symbols[idx], PERIOD_M1, 1, 1, last) != 1) return;

        m_vwap[idx].OnNewBarM1(last[0]);
        double dev = last[0].close - m_vwap[idx].Get();
        m_bb[idx].Push(dev);

        double mean, upper_dev, lower_dev;
        if(!m_bb[idx].Compute(mean, upper_dev, lower_dev)) return;  // warmup

        // Filters: trading window, macro regime, news blackout.
        if(!IsInUTCSession(last[0].time, Inp_MR_SessionStart, Inp_MR_SessionEnd))
            return;
        if(!macro.MacroOk()) return;
        if(!macro.IsValid()) return;
        if(macro.NewsFilterEnabled() && macro.IsInNewsWindow(last[0].time))
            return;

        // No pyramiding: at most one position per pair at a time.
        if(CountSleevePositions(m_magic, m_symbols[idx]) > 0) return;

        double abs_upper = m_vwap[idx].Get() + upper_dev;
        double abs_lower = m_vwap[idx].Get() + lower_dev;

        if(last[0].close < abs_lower)
            OpenPosition(idx, ORDER_TYPE_BUY, risk);
        else if(last[0].close > abs_upper)
            OpenPosition(idx, ORDER_TYPE_SELL, risk);
    }

    bool IsManagedSymbol(string sym)
    {
        for(int i = 0; i < m_n_pairs; i++)
            if(m_symbols[i] == sym) return true;
        return false;
    }

    //--- Pre-fill the BB buffer from history so the sleeve can trade
    //--- shortly after attach. A local VWAP accumulator is used during
    //--- the replay (it resets at each UTC midnight); the live VWAP for
    //--- the current session is owned by m_vwap[idx].
    void WarmupBBFromHistory(int idx)
    {
        int warmup_bars = Inp_MR_BBWindow + 20;
        MqlRates rates[];
        int copied = CopyRates(m_symbols[idx], PERIOD_M1, 1, warmup_bars, rates);
        if(copied <= 0) return;

        CVWAPDaily warmup_vwap;
        for(int i = 0; i < copied; i++)
        {
            warmup_vwap.OnNewBarM1(rates[i]);
            double dev = rates[i].close - warmup_vwap.Get();
            m_bb[idx].Push(dev);
        }
    }

    //--- Submit a market entry for pair `idx`.
    //---
    //--- Sizing: equal-weight across pairs (1 / n_pairs of the sleeve
    //--- sub-equity), risk_pct = FX_RISK_PCT_MR_MACRO, multiplied by
    //--- the global leverage and a slippage drag factor that pre-pays
    //--- the round-trip transaction cost (slip + commission). This is
    //--- needed because non-SL/TP exits (time stop, forced close) do
    //--- not deduct slippage on their own in the Strategy Tester.
    //---
    //--- Stop placement: SL and TP are shifted by slip_pct so each
    //--- triggered exit absorbs one leg of slippage cost on top of the
    //--- pre-paid drag, mirroring the convention used by from_signals
    //--- backtests (slippage charged on every signal-driven exit).
    void OpenPosition(int idx, ENUM_ORDER_TYPE type, CRiskManager &risk)
    {
        string sym = m_symbols[idx];
        double price = (type == ORDER_TYPE_BUY)
                       ? SymbolInfoDouble(sym, SYMBOL_ASK)
                       : SymbolInfoDouble(sym, SYMBOL_BID);
        if(price <= 0.0) return;

        double slip_pct = SlippageFraction(Inp_MR_SlippageBps,
                                           Inp_CommissionBpsPerSide);

        double sl = (type == ORDER_TYPE_BUY)
                    ? price * (1.0 - Inp_MR_SLStop - slip_pct)
                    : price * (1.0 + Inp_MR_SLStop + slip_pct);
        double tp = (type == ORDER_TYPE_BUY)
                    ? price * (1.0 + Inp_MR_TPStop - slip_pct)
                    : price * (1.0 - Inp_MR_TPStop + slip_pct);

        sl = EnforceStopLevel(sym, price, sl, type, true);
        tp = EnforceStopLevel(sym, price, tp, type, false);

        m_trade.SetDeviationInPoints(FX_DEVIATION_POINTS);

        // MR Macro is intraday (no overnight holdings) so swap drag is 0.
        double slip_drag = SizingDrag(slip_pct, 0.0, 0.0);
        double sl_distance = MathAbs(price - sl);
        double per_pair_alloc = 1.0 / (double)m_n_pairs;
        double lots = risk.LotsFor(SLEEVE_MR_MACRO, sym,
                                   FX_RISK_PCT_MR_MACRO,
                                   sl_distance,
                                   per_pair_alloc * slip_drag * Inp_RiskScale);
        if(lots <= 0.0)
        {
            g_logger.Warn(m_name, StringFormat("lots=0 on %s, skipping entry",
                                               sym));
            return;
        }

        bool ok = (type == ORDER_TYPE_BUY)
                  ? m_trade.Buy(lots, sym, price, sl, tp, "MR Macro long")
                  : m_trade.Sell(lots, sym, price, sl, tp, "MR Macro short");

        if(!ok || m_trade.ResultRetcode() != TRADE_RETCODE_DONE)
        {
            g_logger.Error(m_name,
                StringFormat("Entry %s failed: retcode=%d desc=%s lots=%.2f "
                             "price=%.5f sl=%.5f tp=%.5f",
                             sym, m_trade.ResultRetcode(),
                             m_trade.ResultRetcodeDescription(),
                             lots, price, sl, tp));
            return;
        }
        g_logger.Info(m_name,
            StringFormat("Entry %s %s lots=%.2f price=%.5f sl=%.5f tp=%.5f",
                         (type == ORDER_TYPE_BUY ? "LONG" : "SHORT"),
                         sym, lots, price, sl, tp));
    }
};

#endif // __FX_SLEEVE_MR_MACRO_MQH__
