//+------------------------------------------------------------------+
//| FxSleeveRSIDaily.mqh                                             |
//| Sleeve 3 — RSI Daily mean reversion multi-paires.                |
//|                                                                  |
//| Source de vérité : src/strategies/rsi_daily.py                   |
//|   rsi_period=14, oversold=25, overbought=75, exit_mid=50         |
//|   LONG : RSI crossed_below(25)                                   |
//|   SHORT : RSI crossed_above(75)                                  |
//|   Exit LONG : RSI crossed_above(50)                              |
//|   Exit SHORT : RSI crossed_below(50)                             |
//|   Levier natif 1.0 (multiplié par global_leverage)                |
//|   Paires : EURUSD, GBPUSD, USDJPY, USDCAD (4 paires equal-weight) |
//+------------------------------------------------------------------+
#ifndef __FX_SLEEVE_RSI_DAILY_MQH__
#define __FX_SLEEVE_RSI_DAILY_MQH__

#include <Trade/Trade.mqh>
#include "FxSleeveBase.mqh"
#include "FxCommon.mqh"
#include "FxRiskManager.mqh"
#include "FxTradeHelpers.mqh"

//--- Inputs accessibles via #include textuel depuis l'EA principal.

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
            g_logger.Error(m_name, "RSI thresholds order invalid (need OS < mid < OB)");
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
            if(!EnsureHistory(m_pairs[i], PERIOD_D1, 100)) return false;
            m_h_rsi[i] = iRSI(m_pairs[i], PERIOD_D1, Inp_RSI_Period, PRICE_CLOSE);
            if(m_h_rsi[i] == INVALID_HANDLE)
            {
                g_logger.Error(m_name, StringFormat("iRSI handle FAIL for %s", m_pairs[i]));
                return false;
            }
        }
        m_trade.SetExpertMagicNumber(m_magic);
        m_trade.SetDeviationInPoints(20);
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
        for(int i = 0; i < m_n_pairs; i++)
            ProcessPair(i, risk);
    }

    int CloseAll(string reason) override
    {
        return CloseAllByMagic(m_magic, reason);
    }

private:
    void ProcessPair(int i, CRiskManager &risk)
    {
        // Détection des crossings : on lit RSI à shift=1 (hier) et shift=2 (avant-hier)
        double rsi_now = 0.0, rsi_prev = 0.0;
        if(!ReadRSI(i, 1, rsi_now)) return;
        if(!ReadRSI(i, 2, rsi_prev)) return;

        bool entry_long  = (rsi_prev >= Inp_RSI_Oversold)   && (rsi_now < Inp_RSI_Oversold);
        bool exit_long   = (rsi_prev <= Inp_RSI_ExitMid)    && (rsi_now > Inp_RSI_ExitMid);
        bool entry_short = (rsi_prev <= Inp_RSI_Overbought) && (rsi_now > Inp_RSI_Overbought);
        bool exit_short  = (rsi_prev >= Inp_RSI_ExitMid)    && (rsi_now < Inp_RSI_ExitMid);

        ulong existing = FindOpenPosition(m_pairs[i]);
        long pos_type = -1;
        if(existing != 0) pos_type = PositionGetInteger(POSITION_TYPE);

        // Exits (priorité aux sorties)
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

        // Entries (uniquement si pas de position)
        if(existing != 0) return;
        if(entry_long)  OpenPosition(m_pairs[i], ORDER_TYPE_BUY,  risk);
        else if(entry_short) OpenPosition(m_pairs[i], ORDER_TYPE_SELL, risk);
    }

    bool ReadRSI(int pair_idx, int shift, double &out)
    {
        double buf[];
        if(CopyBuffer(m_h_rsi[pair_idx], 0, shift, 1, buf) != 1) return false;
        out = buf[0];
        return true;
    }

    ulong FindOpenPosition(string symbol)
    {
        for(int i = 0; i < PositionsTotal(); i++)
        {
            ulong tk = PositionGetTicket(i);
            if(tk == 0) continue;
            if(PositionGetInteger(POSITION_MAGIC) != m_magic) continue;
            if(PositionGetString(POSITION_SYMBOL) != symbol) continue;
            return tk;
        }
        return 0;
    }

    //--- Slippage Inp_RSI_SlippageBps converti en SYMBOL_POINT puis appliqué
    //--- via SetDeviationInPoints. SL distance majorée de slip_price pour
    //--- intégrer le coût slippage dans le sizing.
    //--- Le risk_per_trade=0.05 (5% sub-equity par paire) reflète la
    //--- sémantique notional Python (lev=1.0 × sub_equity_per_pair).
    void OpenPosition(string symbol, ENUM_ORDER_TYPE type, CRiskManager &risk)
    {
        double price = (type == ORDER_TYPE_BUY)
                       ? SymbolInfoDouble(symbol, SYMBOL_ASK)
                       : SymbolInfoDouble(symbol, SYMBOL_BID);
        if(price <= 0.0) return;

        // Pas de SL/TP dur — sortie par RSI mid. SL safety très large 5%.
        double sl_dist = price * 0.05;
        double sl = (type == ORDER_TYPE_BUY) ? price - sl_dist : price + sl_dist;
        sl = EnforceStopLevel(symbol, price, sl, type, true);

        // Slippage paramétré
        double point = SymbolInfoDouble(symbol, SYMBOL_POINT);
        double slip_price = (Inp_RSI_SlippageBps / 10000.0) * price;
        int slip_pts = (point > 0.0) ? (int)MathCeil(slip_price / point) : 20;
        m_trade.SetDeviationInPoints(MathMax(slip_pts, 5));

        // Sizing : sub_equity_RSI / n_pairs * 1.0 (levier natif) * global_leverage
        double sub_eq = risk.SubEquity(SLEEVE_RSI_DAILY) / m_n_pairs;
        double risk_money = sub_eq * 0.05 * risk.GlobalLeverage();
        double lots = LotsForRisk(symbol, risk_money, sl_dist + slip_price);
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
            StringFormat("Entry %s %s lots=%.2f price=%.5f slip_pts=%d",
                         (type == ORDER_TYPE_BUY ? "LONG" : "SHORT"),
                         symbol, lots, price, slip_pts));
    }
};

#endif // __FX_SLEEVE_RSI_DAILY_MQH__
