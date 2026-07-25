//+------------------------------------------------------------------+
//| FxSleeveGoldMomentum.mqh                                         |
//|                                                                  |
//| Sleeve 5: daily time-series momentum on gold (XAUUSD).           |
//|                                                                  |
//| Specification (mirrors src/strategies/gold_momentum.py):         |
//|   * Universe        : one metal symbol (XAUUSD)                  |
//|   * Score           : mean of sign(return over N) for N in       |
//|                       {40, 60, 120, 250} D1 bars, in [-1, +1]    |
//|   * Long signal     : score > 0                                  |
//|   * Short signal    : score < 0, disabled by default             |
//|   * Exit            : signal flip                                |
//|   * Sizing          : FLAT. Martingale and grid overlays were    |
//|                       tested in Python and rejected on tail      |
//|                       risk; read the research note first.        |
//|   * Vol target      : min(target / sigma21, max leverage)        |
//|                                                                  |
//| The lookbacks are averaged rather than selected: replacing a     |
//| fitted choice with an aggregate is what keeps this signal from   |
//| being an overfit. Do NOT grid-search them.                       |
//|                                                                  |
//| Long-only is the default because gold carries a structural       |
//| positive drift; enabling shorts cost 3.8 pp of return and        |
//| doubled the drawdown in the Python study.                        |
//+------------------------------------------------------------------+
#ifndef __FX_SLEEVE_GOLD_MOMENTUM_MQH__
#define __FX_SLEEVE_GOLD_MOMENTUM_MQH__

#include <Trade/Trade.mqh>
#include "FxSleeveBase.mqh"
#include "FxCommon.mqh"
#include "FxRiskManager.mqh"
#include "FxTradeHelpers.mqh"

//--- Longest lookback plus the shift-1 offset and one spare bar.
#define FX_GOLD_N_LOOKBACKS  4
#define FX_GOLD_MAX_LOOKBACK 250
#define FX_GOLD_HISTORY_BARS (FX_GOLD_MAX_LOOKBACK + 2)

class CSleeveGoldMomentum : public CSleeveBase
{
private:
    string m_symbol;
    int    m_lookbacks[FX_GOLD_N_LOOKBACKS];
    CTrade m_trade;
    bool   m_trace_warned;   // warn once, not once per session

public:
    bool Init() override
    {
        m_magic = Inp_MagicGold;
        m_name  = "Gold_Momentum";
        m_trace_warned = false;

        m_lookbacks[0] = Inp_Gold_LookbackA;
        m_lookbacks[1] = Inp_Gold_LookbackB;
        m_lookbacks[2] = Inp_Gold_LookbackC;
        m_lookbacks[3] = Inp_Gold_LookbackD;
        for(int i = 0; i < FX_GOLD_N_LOOKBACKS; i++)
        {
            if(m_lookbacks[i] <= 0 || m_lookbacks[i] > FX_GOLD_MAX_LOOKBACK)
            {
                g_logger.Error(m_name, StringFormat(
                    "invalid lookback[%d]=%d (expected 1..%d)",
                    i, m_lookbacks[i], FX_GOLD_MAX_LOOKBACK));
                return false;
            }
        }

        m_symbol = MakeSymbolWithSuffix(Inp_Gold_Symbol, Inp_SymbolSuffix);
        if(!EnsureSymbolSelected(m_symbol))
        {
            // Try the bare name: metals often carry no broker suffix even
            // when FX pairs do.
            m_symbol = Inp_Gold_Symbol;
            if(!EnsureSymbolSelected(m_symbol))
            {
                g_logger.Error(m_name, StringFormat(
                    "symbol %s not selectable (tried with and without "
                    "suffix '%s'); sleeve disabled",
                    Inp_Gold_Symbol, Inp_SymbolSuffix));
                return false;
            }
        }

        if(!EnsureHistory(m_symbol, PERIOD_D1, FX_GOLD_HISTORY_BARS))
        {
            if(!EnsureHistory(m_symbol, PERIOD_D1, 1))
            {
                g_logger.Error(m_name, StringFormat(
                    "%s: no D1 history available; sleeve disabled", m_symbol));
                return false;
            }
            g_logger.Warn(m_name, StringFormat(
                "%s: %d/%d D1 bars; the 250-bar lookback stays flat until "
                "history accumulates", m_symbol,
                (int)Bars(m_symbol, PERIOD_D1), FX_GOLD_HISTORY_BARS));
        }

        m_trade.SetExpertMagicNumber(m_magic);
        m_trade.SetDeviationInPoints(FX_DEFAULT_DEVIATION);
        g_logger.Info(m_name, StringFormat(
            "Init OK symbol=%s lookbacks=%d/%d/%d/%d short=%s",
            m_symbol, m_lookbacks[0], m_lookbacks[1], m_lookbacks[2],
            m_lookbacks[3], (Inp_Gold_AllowShort ? "on" : "off")));
        return true;
    }

    //--- Daily processing hook (triggered after the UTC close).
    void OnNewBarD1(CRiskManager &risk) override
    {
        if(risk.IsDDLocked()) return;
        ProcessSymbol(risk);
    }

    int CloseAll(string reason) override
    {
        return CloseAllByMagic(m_magic, reason);
    }

private:
    void ProcessSymbol(CRiskManager &risk)
    {
        double score;
        if(!ComputeScore(score)) return;

        bool long_signal  = (score > 0.0);
        bool short_signal = Inp_Gold_AllowShort && (score < 0.0);

        double sigma21 = ComputeSigma21();
        double lev = MathMin(Inp_Gold_TargetVol / MathMax(sigma21, 0.05),
                             Inp_Gold_MaxLeverage);

        ulong existing = FindPositionByMagicSymbol(m_magic, m_symbol);
        long pos_type = -1;
        if(existing != 0) pos_type = PositionGetInteger(POSITION_TYPE);

        // Exit on signal flip.
        if(pos_type == (long)POSITION_TYPE_BUY && !long_signal)
        {
            m_trade.PositionClose(existing);
            g_logger.Info(m_name, StringFormat(
                "Exit LONG %s (score=%.2f)", m_symbol, score));
            existing = 0;
        }
        else if(pos_type == (long)POSITION_TYPE_SELL && !short_signal)
        {
            m_trade.PositionClose(existing);
            g_logger.Info(m_name, StringFormat(
                "Exit SHORT %s (score=%.2f)", m_symbol, score));
            existing = 0;
        }

        if(existing == 0)
        {
            if(long_signal)       OpenPosition(ORDER_TYPE_BUY,  lev, score, risk);
            else if(short_signal) OpenPosition(ORDER_TYPE_SELL, lev, score, risk);
        }

        if(Inp_Gold_Trace)
            WriteTraceRow(score, lev, long_signal, short_signal, risk);
    }

    //--- Append one row of the cross-engine reconciliation trace.
    //--- Contract and column order: docs/specs/gold_momentum_spec.md §9.
    //--- Off by default: this writes to disk on every session and is a
    //--- diagnostic, not a production behaviour.
    void WriteTraceRow(double score, double lev, bool long_signal,
                       bool short_signal, CRiskManager &risk)
    {
        double direction     = long_signal ? 1.0 : (short_signal ? -1.0 : 0.0);
        double target_weight = lev * direction;

        //--- Position in units (ounces), not lots, so the column means the
        //--- same thing as the vbt one.
        double units  = 0.0;
        ulong  ticket = FindPositionByMagicSymbol(m_magic, m_symbol);
        if(ticket != 0 && PositionSelectByTicket(ticket))
        {
            double vol = PositionGetDouble(POSITION_VOLUME)
                         * SymbolInfoDouble(m_symbol, SYMBOL_TRADE_CONTRACT_SIZE);
            units = (PositionGetInteger(POSITION_TYPE) == (long)POSITION_TYPE_SELL)
                    ? -vol : vol;
        }

        //--- The row is stamped with the session whose close produced the
        //--- score (shift 1), NOT the session the order is sent in. Stamping
        //--- it with the execution day would shift the whole trace one bar
        //--- against vbt and break rung 2 for a reason unrelated to the signal.
        datetime bar_time = iTime(m_symbol, PERIOD_D1, 1);
        double   close_px = iClose(m_symbol, PERIOD_D1, 1);
        if(bar_time == 0 || close_px <= 0.0) return;

        MqlDateTime dt;
        TimeToStruct(bar_time, dt);

        //--- Sleeve-attributable equity, not account equity: the EA runs
        //--- several sleeves and the account figure would not be comparable.
        double equity = risk.SubEquity(SLEEVE_GOLD_MOMENTUM);

        int handle = FileOpen(Inp_Gold_TraceFile,
                              FILE_READ | FILE_WRITE | FILE_TXT | FILE_ANSI | FILE_COMMON);
        if(handle == INVALID_HANDLE)
        {
            if(!m_trace_warned)
            {
                m_trace_warned = true;
                g_logger.Warn(m_name, StringFormat(
                    "cannot open trace file %s (err=%d); tracing disabled for this run",
                    Inp_Gold_TraceFile, GetLastError()));
            }
            return;
        }
        if(FileSize(handle) == 0)
            FileWriteString(handle,
                "date,close,score,target_weight,position_units,equity\n");
        FileSeek(handle, 0, SEEK_END);
        FileWriteString(handle, StringFormat(
            "%04d-%02d-%02d,%.6f,%.6f,%.6f,%.6f,%.2f\n",
            dt.year, dt.mon, dt.day,
            close_px, score, target_weight, units, equity));
        FileClose(handle);
    }

    //--- Mean of sign(return over N) across the configured lookbacks.
    //--- Reads shift 1 onwards, so the current forming bar is never used.
    bool ComputeScore(double &score)
    {
        double closes[];
        int copied = CopyClose(m_symbol, PERIOD_D1, 1,
                               FX_GOLD_HISTORY_BARS, closes);
        if(copied < FX_GOLD_HISTORY_BARS)
        {
            g_logger.Warn(m_name, StringFormat(
                "%s: only %d/%d D1 bars copied; skipping this session",
                m_symbol, copied, FX_GOLD_HISTORY_BARS));
            return false;
        }
        // closes[] is oldest-first, so the most recent completed bar is last.
        int last = copied - 1;
        double newest = closes[last];
        if(newest <= 0.0) return false;

        double sum = 0.0;
        for(int i = 0; i < FX_GOLD_N_LOOKBACKS; i++)
        {
            int idx = last - m_lookbacks[i];
            if(idx < 0) return false;
            double past = closes[idx];
            if(past <= 0.0) return false;
            double ret = newest / past - 1.0;
            sum += (ret > 0.0) ? 1.0 : ((ret < 0.0) ? -1.0 : 0.0);
        }
        score = sum / (double)FX_GOLD_N_LOOKBACKS;
        return true;
    }

    //--- Annualised standard deviation of log returns over the last 21 D1
    //--- bars. Mirrors CSleeveTSMomentum::ComputePairSigma21; the fallback
    //--- is gold's long-run volatility rather than the FX default.
    double ComputeSigma21()
    {
        double closes[22];
        int copied = CopyClose(m_symbol, PERIOD_D1, 1, 22, closes);
        if(copied < 22) return 0.16;
        double rets[21];
        for(int i = 0; i < 21; i++)
        {
            if(closes[i] <= 0.0) return 0.16;
            rets[i] = MathLog(closes[i + 1] / closes[i]);
        }
        double s = 0.0, s2 = 0.0;
        for(int i = 0; i < 21; i++) { s += rets[i]; s2 += rets[i] * rets[i]; }
        double mean = s / 21.0;
        double var = (s2 - 21.0 * mean * mean) / 20.0;
        return MathSqrt(MathMax(var, 0.0)) * MathSqrt(252.0);
    }

    //--- Submit a market entry with FLAT sizing. Slippage, commission and an
    //--- empirical swap drag are pre-paid via SizingDrag() so signal-flip
    //--- exits do not undercount costs.
    void OpenPosition(ENUM_ORDER_TYPE type, double lev, double score,
                      CRiskManager &risk)
    {
        double price = (type == ORDER_TYPE_BUY)
                       ? SymbolInfoDouble(m_symbol, SYMBOL_ASK)
                       : SymbolInfoDouble(m_symbol, SYMBOL_BID);
        if(price <= 0.0) return;

        double slip_pct = SlippageFraction(Inp_Gold_SlippageBps,
                                           Inp_CommissionBpsPerSide);

        // Wide safety stop. The sleeve normally exits on a signal flip, so
        // this guards an overnight gap rather than acting as a regular stop.
        // Gold is roughly twice as volatile as a major FX pair, hence the
        // wider distance than the 2% used by the FX sleeves.
        double sl_dist = price * (Inp_Gold_SafetySL + slip_pct);
        double sl = (type == ORDER_TYPE_BUY) ? price - sl_dist
                                             : price + sl_dist;
        sl = EnforceStopLevel(m_symbol, price, sl, type, true);

        m_trade.SetDeviationInPoints(FX_DEVIATION_POINTS);

        double sub_eq = risk.SubEquity(SLEEVE_GOLD_MOMENTUM);
        double slip_drag = SizingDrag(slip_pct, Inp_SwapBpsPerNight,
                                      FX_GOLD_AVG_NIGHTS_HELD);
        double risk_money = sub_eq * FX_RISK_PCT_GOLD_MOMENTUM
                            * lev * slip_drag;
        double lots = LotsForRisk(m_symbol, risk_money, sl_dist);
        if(lots <= 0.0)
        {
            g_logger.Warn(m_name, StringFormat(
                "%s: computed lots=0 (risk_money=%.2f sl_dist=%.2f); "
                "check SYMBOL_VOLUME_MIN", m_symbol, risk_money, sl_dist));
            return;
        }

        bool ok = (type == ORDER_TYPE_BUY)
                  ? m_trade.Buy(lots, m_symbol, price, sl, 0.0, "Gold momentum long")
                  : m_trade.Sell(lots, m_symbol, price, sl, 0.0, "Gold momentum short");

        if(!ok || m_trade.ResultRetcode() != TRADE_RETCODE_DONE)
        {
            g_logger.Error(m_name, StringFormat(
                "Entry %s failed: retcode=%d", m_symbol,
                m_trade.ResultRetcode()));
            return;
        }
        g_logger.Info(m_name, StringFormat(
            "Entry %s %s lots=%.2f price=%.2f score=%.2f lev=%.2f",
            (type == ORDER_TYPE_BUY ? "LONG" : "SHORT"),
            m_symbol, lots, price, score, lev));
    }
};

#endif // __FX_SLEEVE_GOLD_MOMENTUM_MQH__
