//+------------------------------------------------------------------+
//| FxSleeveTSMomentum.mqh                                           |
//|                                                                  |
//| Sleeve 2: daily time-series momentum across multiple FX majors.  |
//|                                                                  |
//| Specification:                                                   |
//|   * Universe        : 3 majors equal-weighted                    |
//|   * Long signal     : EMA(fast) > EMA(slow) AND RSI < RSIHigh    |
//|   * Short signal    : EMA(fast) < EMA(slow) AND RSI > RSILow     |
//|   * Exit            : signal flip                                |
//|   * Per-pair vol-target leverage : min(target / sigma21, max)    |
//+------------------------------------------------------------------+
#ifndef __FX_SLEEVE_TS_MOMENTUM_MQH__
#define __FX_SLEEVE_TS_MOMENTUM_MQH__

#include <Trade/Trade.mqh>
#include "FxSleeveBase.mqh"
#include "FxCommon.mqh"
#include "FxRiskManager.mqh"
#include "FxTradeHelpers.mqh"

#define FX_TS_MAX_PAIRS 8

class CSleeveTSMomentum : public CSleeveBase
{
private:
    int    m_n_pairs;
    string m_pairs[FX_TS_MAX_PAIRS];
    int    m_h_ema_fast[FX_TS_MAX_PAIRS];
    int    m_h_ema_slow[FX_TS_MAX_PAIRS];
    int    m_h_rsi[FX_TS_MAX_PAIRS];
    CTrade m_trade;

public:
    bool Init() override
    {
        m_magic = Inp_MagicTS;
        m_name  = "TS_Momentum";

        string raw[];
        int n = SplitCsv(Inp_TS_Pairs, raw);
        if(n <= 0 || n > FX_TS_MAX_PAIRS)
        {
            g_logger.Error(m_name, StringFormat("invalid Inp_TS_Pairs=%s",
                                                Inp_TS_Pairs));
            return false;
        }
        // Pack valid pairs into [0..valid_n-1]; skip pairs without any
        // D1 history so adding a pair that started later than the
        // backtest window does not disable the entire sleeve.
        int valid_n = 0;
        for(int i = 0; i < n; i++)
        {
            string pair = MakeSymbolWithSuffix(raw[i], Inp_SymbolSuffix);
            if(!EnsureSymbolSelected(pair))
            {
                g_logger.Warn(m_name, StringFormat(
                    "%s: skipped (symbol not selectable on broker)", pair));
                continue;
            }
            // Aim for 250 D1 bars; gracefully accept any history >= 1.
            if(!EnsureHistory(pair, PERIOD_D1, 250))
            {
                if(!EnsureHistory(pair, PERIOD_D1, 1))
                {
                    g_logger.Warn(m_name, StringFormat(
                        "%s: skipped (no D1 history available on broker)",
                        pair));
                    continue;
                }
                g_logger.Warn(m_name, StringFormat(
                    "%s: %d/250 D1 bars; iMA(%d)/iRSI(%d) warm up as "
                    "bars accumulate", pair, (int)Bars(pair, PERIOD_D1),
                    Inp_TS_SlowEMA, Inp_TS_RSIPeriod));
            }
            int h_fast = iMA(pair, PERIOD_D1, Inp_TS_FastEMA,
                             0, MODE_EMA, PRICE_CLOSE);
            int h_slow = iMA(pair, PERIOD_D1, Inp_TS_SlowEMA,
                             0, MODE_EMA, PRICE_CLOSE);
            int h_rsi  = iRSI(pair, PERIOD_D1, Inp_TS_RSIPeriod, PRICE_CLOSE);
            if(h_fast == INVALID_HANDLE || h_slow == INVALID_HANDLE ||
               h_rsi == INVALID_HANDLE)
            {
                g_logger.Warn(m_name, StringFormat(
                    "%s: skipped (indicator handle FAIL)", pair));
                continue;
            }
            m_pairs[valid_n]      = pair;
            m_h_ema_fast[valid_n] = h_fast;
            m_h_ema_slow[valid_n] = h_slow;
            m_h_rsi[valid_n]      = h_rsi;
            valid_n++;
        }
        if(valid_n == 0)
        {
            g_logger.Error(m_name,
                "no valid pairs after filtering; sleeve disabled");
            return false;
        }
        m_n_pairs = valid_n;
        m_trade.SetExpertMagicNumber(m_magic);
        m_trade.SetDeviationInPoints(FX_DEFAULT_DEVIATION);
        g_logger.Info(m_name, StringFormat(
            "Init OK %d/%d pairs (skipped %d for missing D1/symbol)",
            valid_n, n, n - valid_n));
        return true;
    }

    void Shutdown() override
    {
        for(int i = 0; i < m_n_pairs; i++)
        {
            if(m_h_ema_fast[i] != INVALID_HANDLE)
                IndicatorRelease(m_h_ema_fast[i]);
            if(m_h_ema_slow[i] != INVALID_HANDLE)
                IndicatorRelease(m_h_ema_slow[i]);
            if(m_h_rsi[i]      != INVALID_HANDLE)
                IndicatorRelease(m_h_rsi[i]);
        }
    }

    //--- Daily processing hook (triggered after the UTC close).
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
        double ema_fast, ema_slow, rsi;
        if(!ReadShift1(m_h_ema_fast[i], ema_fast)) return;
        if(!ReadShift1(m_h_ema_slow[i], ema_slow)) return;
        if(!ReadShift1(m_h_rsi[i], rsi)) return;

        bool long_signal  = (ema_fast > ema_slow) && (rsi < Inp_TS_RSIHigh);
        bool short_signal = (ema_fast < ema_slow) && (rsi > Inp_TS_RSILow);

        // Per-pair vol target leverage based on rolling 21-day sigma.
        double sigma21 = ComputePairSigma21(m_pairs[i]);
        double lev_pair = MathMin(Inp_TS_TargetVol / MathMax(sigma21, 0.01),
                                   Inp_TS_MaxLeverage);

        ulong existing = FindPositionByMagicSymbol(m_magic, m_pairs[i]);
        long pos_type = -1;
        if(existing != 0) pos_type = PositionGetInteger(POSITION_TYPE);

        // Exit on signal flip.
        if(pos_type == (long)POSITION_TYPE_BUY && !long_signal)
        {
            m_trade.PositionClose(existing);
            g_logger.Info(m_name, StringFormat("Exit LONG %s (signal flip)",
                                                m_pairs[i]));
        }
        else if(pos_type == (long)POSITION_TYPE_SELL && !short_signal)
        {
            m_trade.PositionClose(existing);
            g_logger.Info(m_name, StringFormat("Exit SHORT %s (signal flip)",
                                                m_pairs[i]));
        }

        // Entry only if no position currently held on this pair.
        if(existing == 0)
        {
            if(long_signal)       OpenPosition(m_pairs[i], ORDER_TYPE_BUY,  lev_pair, risk);
            else if(short_signal) OpenPosition(m_pairs[i], ORDER_TYPE_SELL, lev_pair, risk);
        }
    }

    bool ReadShift1(int handle, double &out)
    {
        double buf[];
        if(CopyBuffer(handle, 0, 1, 1, buf) != 1) return false;
        out = buf[0];
        return true;
    }

    //--- Annualised standard deviation of log returns over the last 21 D1
    //--- bars. Falls back to the target volatility when history is short.
    double ComputePairSigma21(string symbol)
    {
        double closes[22];
        int copied = CopyClose(symbol, PERIOD_D1, 1, 22, closes);
        if(copied < 22) return 0.10;
        double rets[21];
        for(int i = 0; i < 21; i++)
        {
            if(closes[i] <= 0.0) return 0.10;
            rets[i] = MathLog(closes[i + 1] / closes[i]);
        }
        double s = 0.0, s2 = 0.0;
        for(int i = 0; i < 21; i++) { s += rets[i]; s2 += rets[i] * rets[i]; }
        double mean = s / 21.0;
        double var = (s2 - 21.0 * mean * mean) / 20.0;
        return MathSqrt(MathMax(var, 0.0)) * MathSqrt(252.0);
    }

    //--- Submit a market entry. Sizing combines per-pair vol-target
    //--- leverage with the sleeve sub-equity; the global leverage is
    //--- already applied via SubEquity() inside CRiskManager. Slippage
    //--- + commission + an empirical overnight swap drag are pre-paid
    //--- via SizingDrag() so non-SL/TP exits do not undercount costs.
    void OpenPosition(string symbol, ENUM_ORDER_TYPE type, double lev_pair,
                      CRiskManager &risk)
    {
        double price = (type == ORDER_TYPE_BUY)
                       ? SymbolInfoDouble(symbol, SYMBOL_ASK)
                       : SymbolInfoDouble(symbol, SYMBOL_BID);
        if(price <= 0.0) return;

        double slip_pct = SlippageFraction(Inp_TS_SlippageBps,
                                           Inp_CommissionBpsPerSide);

        // Wide safety SL (configured fraction of price). The sleeve
        // ordinarily exits on signal flip, so this is a tail-risk hedge
        // against an overnight gap rather than a regular stop.
        double sl_dist_safety = price * (0.02 + slip_pct);
        double sl = (type == ORDER_TYPE_BUY) ? price - sl_dist_safety
                                             : price + sl_dist_safety;
        sl = EnforceStopLevel(symbol, price, sl, type, true);

        m_trade.SetDeviationInPoints(FX_DEVIATION_POINTS);

        double sub_eq = risk.SubEquity(SLEEVE_TS_MOMENTUM) / m_n_pairs;
        double slip_drag = SizingDrag(slip_pct, Inp_SwapBpsPerNight,
                                      FX_TS_AVG_NIGHTS_HELD);
        double risk_money = sub_eq * FX_RISK_PCT_TS_MOMENTUM
                            * lev_pair * slip_drag * Inp_RiskScale;
        double lots = LotsForRisk(symbol, risk_money, sl_dist_safety);
        if(lots <= 0.0) return;

        bool ok = (type == ORDER_TYPE_BUY)
                  ? m_trade.Buy(lots, symbol, price, sl, 0.0, "TS Momentum long")
                  : m_trade.Sell(lots, symbol, price, sl, 0.0, "TS Momentum short");

        if(!ok || m_trade.ResultRetcode() != TRADE_RETCODE_DONE)
        {
            g_logger.Error(m_name,
                StringFormat("Entry %s failed: retcode=%d",
                             symbol, m_trade.ResultRetcode()));
            return;
        }
        g_logger.Info(m_name,
            StringFormat("Entry %s %s lots=%.2f price=%.5f lev_pair=%.2f",
                         (type == ORDER_TYPE_BUY ? "LONG" : "SHORT"),
                         symbol, lots, price, lev_pair));
    }
};

#endif // __FX_SLEEVE_TS_MOMENTUM_MQH__
