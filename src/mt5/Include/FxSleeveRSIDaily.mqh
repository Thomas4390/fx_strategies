//+------------------------------------------------------------------+
//| FxSleeveRSIDaily.mqh                                             |
//|                                                                  |
//| Sleeve 3: daily RSI mean reversion across multiple FX majors.    |
//|                                                                  |
//| Specification:                                                   |
//|   * Universe        : 3-4 majors equal-weighted                  |
//|   * Long entry      : RSI(N) crosses below the oversold level    |
//|   * Short entry     : RSI(N) crosses above the overbought level  |
//|   * Long exit       : RSI(N) crosses above the mid level         |
//|   * Short exit      : RSI(N) crosses below the mid level         |
//|   * Time stop       : configurable (caps overnight swap drag)    |
//+------------------------------------------------------------------+
#ifndef __FX_SLEEVE_RSI_DAILY_MQH__
#define __FX_SLEEVE_RSI_DAILY_MQH__

#include <Trade/Trade.mqh>
#include "FxSleeveBase.mqh"
#include "FxCommon.mqh"
#include "FxRiskManager.mqh"
#include "FxTradeHelpers.mqh"

#define FX_RSI_MAX_PAIRS 8

class CSleeveRSIDaily : public CSleeveBase
{
private:
    int    m_n_pairs;
    string m_pairs[FX_RSI_MAX_PAIRS];
    int    m_h_rsi[FX_RSI_MAX_PAIRS];
    CTrade m_trade;

public:
    bool Init() override
    {
        m_magic = Inp_MagicRSI;
        m_name  = "RSI_Daily";

        if(Inp_RSI_Oversold >= Inp_RSI_ExitMid ||
           Inp_RSI_ExitMid >= Inp_RSI_Overbought)
        {
            g_logger.Error(m_name,
                "RSI thresholds order invalid (need OS < mid < OB)");
            return false;
        }

        string raw[];
        int n = SplitCsv(Inp_RSI_Pairs, raw);
        if(n <= 0 || n > FX_RSI_MAX_PAIRS)
        {
            g_logger.Error(m_name, "invalid Inp_RSI_Pairs");
            return false;
        }
        m_n_pairs = n;
        for(int i = 0; i < n; i++)
        {
            m_pairs[i] = MakeSymbolWithSuffix(raw[i], Inp_SymbolSuffix);
            if(!EnsureSymbolSelected(m_pairs[i])) return false;
            // Aim for 100 D1 bars; gracefully accept any history >= 1.
            // The iRSI handle returns no value until Inp_RSI_Period bars
            // are available so the sleeve simply waits during warmup.
            if(!EnsureHistory(m_pairs[i], PERIOD_D1, 100))
            {
                if(!EnsureHistory(m_pairs[i], PERIOD_D1, 1))
                {
                    g_logger.Error(m_name, StringFormat(
                        "%s: no D1 history at all; sleeve disabled",
                        m_pairs[i]));
                    return false;
                }
                g_logger.Warn(m_name, StringFormat(
                    "%s: %d/100 D1 bars; iRSI(%d) warms up as bars accumulate",
                    m_pairs[i], (int)Bars(m_pairs[i], PERIOD_D1),
                    Inp_RSI_Period));
            }
            m_h_rsi[i] = iRSI(m_pairs[i], PERIOD_D1, Inp_RSI_Period, PRICE_CLOSE);
            if(m_h_rsi[i] == INVALID_HANDLE)
            {
                g_logger.Error(m_name, StringFormat(
                    "iRSI handle FAIL for %s", m_pairs[i]));
                return false;
            }
        }
        m_trade.SetExpertMagicNumber(m_magic);
        m_trade.SetDeviationInPoints(FX_DEFAULT_DEVIATION);
        g_logger.Info(m_name, StringFormat("Init OK %d pairs", n));
        return true;
    }

    void Shutdown() override
    {
        for(int i = 0; i < m_n_pairs; i++)
            if(m_h_rsi[i] != INVALID_HANDLE) IndicatorRelease(m_h_rsi[i]);
    }

    void OnNewBarD1(CRiskManager &risk) override
    {
        if(risk.IsDDLocked()) return;
        // Time-stop check before signal evaluation so any forced close
        // happens in the same daily slot as the new entry decision.
        CheckTimeStops();
        for(int i = 0; i < m_n_pairs; i++)
            ProcessPair(i, risk);
    }

    int CloseAll(string reason) override
    {
        return CloseAllByMagic(m_magic, reason);
    }

private:
    //--- Close any sleeve position older than Inp_RSI_TimeStopDays. The
    //--- limit caps cumulative overnight swap drag, which can dominate
    //--- the trade P&L when RSI oscillates around the mid level.
    void CheckTimeStops()
    {
        if(Inp_RSI_TimeStopDays <= 0) return;
        datetime now = TimeGMT();
        long max_secs = (long)Inp_RSI_TimeStopDays * 86400L;
        for(int i = PositionsTotal() - 1; i >= 0; i--)
        {
            ulong tk = PositionGetTicket(i);
            if(tk == 0) continue;
            if(PositionGetInteger(POSITION_MAGIC) != m_magic) continue;
            datetime opened = (datetime)PositionGetInteger(POSITION_TIME);
            if((long)(now - opened) >= max_secs)
            {
                m_trade.PositionClose(tk);
                g_logger.Info(m_name, StringFormat(
                    "Time-stop close ticket=%I64u sym=%s age_days=%d",
                    tk, PositionGetString(POSITION_SYMBOL),
                    (int)((now - opened) / 86400)));
            }
        }
    }

    void ProcessPair(int i, CRiskManager &risk)
    {
        // Read RSI on shift=1 (yesterday) and shift=2 (the day before).
        // Both must be available; the cross detection compares the two.
        double rsi_now = 0.0, rsi_prev = 0.0;
        if(!ReadRSI(i, 1, rsi_now)) return;
        if(!ReadRSI(i, 2, rsi_prev)) return;

        bool entry_long  = (rsi_prev >= Inp_RSI_Oversold)   && (rsi_now < Inp_RSI_Oversold);
        bool exit_long   = (rsi_prev <= Inp_RSI_ExitMid)    && (rsi_now > Inp_RSI_ExitMid);
        bool entry_short = (rsi_prev <= Inp_RSI_Overbought) && (rsi_now > Inp_RSI_Overbought);
        bool exit_short  = (rsi_prev >= Inp_RSI_ExitMid)    && (rsi_now < Inp_RSI_ExitMid);

        ulong existing = FindPositionByMagicSymbol(m_magic, m_pairs[i]);
        long pos_type = -1;
        if(existing != 0) pos_type = PositionGetInteger(POSITION_TYPE);

        // Exits take priority over entries on the same bar.
        if(pos_type == (long)POSITION_TYPE_BUY && exit_long)
        {
            m_trade.PositionClose(existing);
            g_logger.Info(m_name, StringFormat("Exit LONG %s (RSI>%.0f)",
                                                m_pairs[i], Inp_RSI_ExitMid));
            return;
        }
        if(pos_type == (long)POSITION_TYPE_SELL && exit_short)
        {
            m_trade.PositionClose(existing);
            g_logger.Info(m_name, StringFormat("Exit SHORT %s (RSI<%.0f)",
                                                m_pairs[i], Inp_RSI_ExitMid));
            return;
        }

        if(existing != 0) return;
        if(entry_long)       OpenPosition(m_pairs[i], ORDER_TYPE_BUY,  risk);
        else if(entry_short) OpenPosition(m_pairs[i], ORDER_TYPE_SELL, risk);
    }

    bool ReadRSI(int pair_idx, int shift, double &out)
    {
        double buf[];
        if(CopyBuffer(m_h_rsi[pair_idx], 0, shift, 1, buf) != 1) return false;
        out = buf[0];
        return true;
    }

    //--- Submit a market entry. Sleeve native leverage is 1.0 (the per-
    //--- pair vol-target is implicit in the global leverage). Slippage,
    //--- commission, and overnight swap drag are pre-paid via
    //--- SizingDrag() to keep cost accounting consistent across exits.
    void OpenPosition(string symbol, ENUM_ORDER_TYPE type, CRiskManager &risk)
    {
        double price = (type == ORDER_TYPE_BUY)
                       ? SymbolInfoDouble(symbol, SYMBOL_ASK)
                       : SymbolInfoDouble(symbol, SYMBOL_BID);
        if(price <= 0.0) return;

        double slip_pct = SlippageFraction(Inp_RSI_SlippageBps,
                                           Inp_CommissionBpsPerSide);

        double sl_dist = price * (0.02 + slip_pct);
        double sl = (type == ORDER_TYPE_BUY) ? price - sl_dist
                                             : price + sl_dist;
        sl = EnforceStopLevel(symbol, price, sl, type, true);

        m_trade.SetDeviationInPoints(FX_DEVIATION_POINTS);

        double sub_eq = risk.SubEquity(SLEEVE_RSI_DAILY) / m_n_pairs;
        double slip_drag = SizingDrag(slip_pct, Inp_SwapBpsPerNight,
                                      FX_RSI_AVG_NIGHTS_HELD);
        double risk_money = sub_eq * FX_RISK_PCT_RSI_DAILY
                            * risk.GlobalLeverage() * slip_drag;
        double lots = LotsForRisk(symbol, risk_money, sl_dist);
        if(lots <= 0.0) return;

        bool ok = (type == ORDER_TYPE_BUY)
                  ? m_trade.Buy(lots, symbol, price, sl, 0.0, "RSI Daily long")
                  : m_trade.Sell(lots, symbol, price, sl, 0.0, "RSI Daily short");

        if(!ok || m_trade.ResultRetcode() != TRADE_RETCODE_DONE)
        {
            g_logger.Error(m_name,
                StringFormat("Entry %s failed: retcode=%d",
                             symbol, m_trade.ResultRetcode()));
            return;
        }
        g_logger.Info(m_name,
            StringFormat("Entry %s %s lots=%.2f price=%.5f",
                         (type == ORDER_TYPE_BUY ? "LONG" : "SHORT"),
                         symbol, lots, price));
    }
};

#endif // __FX_SLEEVE_RSI_DAILY_MQH__
