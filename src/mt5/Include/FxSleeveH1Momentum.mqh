//+------------------------------------------------------------------+
//| FxSleeveH1Momentum.mqh                                           |
//| Sleeve 4 — Momentum H1 (Phase D du plan CAGR improvement).       |
//|                                                                  |
//| Variante de TS Momentum mais sur timeframe H1 au lieu de D1.     |
//| Idée : capter momentum intraday entre M1 (MR Macro) et D1 (TS).  |
//|                                                                  |
//|   fast_ema=20, slow_ema=50, rsi_period=7 (sur barres H1)          |
//|   LONG  : EMA20 > EMA50 AND RSI(7) < 60                            |
//|   SHORT : EMA20 < EMA50 AND RSI(7) > 40                            |
//|   Sortie : inversion du critère                                    |
//|   SL dynamique : 2× ATR(14) H1                                     |
//|   Vol target par paire : lev = min(0.10 / max(σ21, 0.01), 3.0)     |
//|   Paires : EURUSD, GBPUSD, USDJPY (3 paires equal-weight)          |
//+------------------------------------------------------------------+
#ifndef __FX_SLEEVE_H1_MOMENTUM_MQH__
#define __FX_SLEEVE_H1_MOMENTUM_MQH__

#include <Trade/Trade.mqh>
#include "FxSleeveBase.mqh"
#include "FxCommon.mqh"
#include "FxRiskManager.mqh"
#include "FxTradeHelpers.mqh"

#define FX_H1_MAX_PAIRS 8

class CSleeveH1Momentum : public CSleeveBase
{
private:
    int    m_n_pairs;
    string m_pairs[FX_H1_MAX_PAIRS];
    int    m_h_ema_fast[FX_H1_MAX_PAIRS];
    int    m_h_ema_slow[FX_H1_MAX_PAIRS];
    int    m_h_rsi[FX_H1_MAX_PAIRS];
    int    m_h_atr[FX_H1_MAX_PAIRS];
    datetime m_last_h1_bar[FX_H1_MAX_PAIRS];
    CTrade m_trade;

public:
    bool Init() override
    {
        m_magic = Inp_MagicH1;
        m_name  = "H1_Momentum";

        string raw[];
        int n = SplitCsv(Inp_H1_Pairs, raw);
        if(n <= 0 || n > FX_H1_MAX_PAIRS)
        {
            g_logger.Error(m_name, StringFormat("invalid Inp_H1_Pairs=%s", Inp_H1_Pairs));
            return false;
        }

        // Pack valid pairs into [0..valid_n-1]. Skip pairs without H1 history.
        int valid_n = 0;
        for(int i = 0; i < n; i++)
        {
            string pair = MakeSymbolWithSuffix(raw[i], Inp_SymbolSuffix);
            if(!EnsureSymbolSelected(pair))
            {
                g_logger.Warn(m_name, StringFormat(
                    "%s: skipped (symbol not selectable)", pair));
                continue;
            }
            // Ideal 250 H1 bars (~10 jours). Soft floor à 1 bar : indicateurs
            // warm up gracieusement comme dans TS Momentum.
            if(!EnsureHistory(pair, PERIOD_H1, 250))
            {
                if(!EnsureHistory(pair, PERIOD_H1, 1))
                {
                    g_logger.Warn(m_name, StringFormat(
                        "%s: skipped (no H1 history)", pair));
                    continue;
                }
                g_logger.Warn(m_name, StringFormat(
                    "%s: %d/250 H1 bars; iMA(%d)/iRSI(%d)/iATR(%d) warm up",
                    pair, (int)Bars(pair, PERIOD_H1),
                    Inp_H1_SlowEMA, Inp_H1_RSIPeriod, Inp_H1_ATRPeriod));
            }
            int h_fast = iMA(pair, PERIOD_H1, Inp_H1_FastEMA,
                             0, MODE_EMA, PRICE_CLOSE);
            int h_slow = iMA(pair, PERIOD_H1, Inp_H1_SlowEMA,
                             0, MODE_EMA, PRICE_CLOSE);
            int h_rsi  = iRSI(pair, PERIOD_H1, Inp_H1_RSIPeriod, PRICE_CLOSE);
            int h_atr  = iATR(pair, PERIOD_H1, Inp_H1_ATRPeriod);
            if(h_fast == INVALID_HANDLE || h_slow == INVALID_HANDLE ||
               h_rsi == INVALID_HANDLE  || h_atr == INVALID_HANDLE)
            {
                g_logger.Warn(m_name, StringFormat(
                    "%s: skipped (indicator handle FAIL)", pair));
                continue;
            }
            m_pairs[valid_n]      = pair;
            m_h_ema_fast[valid_n] = h_fast;
            m_h_ema_slow[valid_n] = h_slow;
            m_h_rsi[valid_n]      = h_rsi;
            m_h_atr[valid_n]      = h_atr;
            m_last_h1_bar[valid_n] = 0;
            valid_n++;
        }
        if(valid_n == 0)
        {
            g_logger.Error(m_name, "no valid pairs; sleeve disabled");
            return false;
        }
        m_n_pairs = valid_n;
        m_trade.SetExpertMagicNumber(m_magic);
        m_trade.SetDeviationInPoints(20);
        g_logger.Info(m_name, StringFormat(
            "Init OK %d/%d pairs", valid_n, n));
        return true;
    }

    void Shutdown() override
    {
        for(int i = 0; i < m_n_pairs; i++)
        {
            if(m_h_ema_fast[i] != INVALID_HANDLE) IndicatorRelease(m_h_ema_fast[i]);
            if(m_h_ema_slow[i] != INVALID_HANDLE) IndicatorRelease(m_h_ema_slow[i]);
            if(m_h_rsi[i]      != INVALID_HANDLE) IndicatorRelease(m_h_rsi[i]);
            if(m_h_atr[i]      != INVALID_HANDLE) IndicatorRelease(m_h_atr[i]);
        }
    }

    //--- Hook NewBar H1 — appelé chaque OnTick(), détecte interne new H1 bar.
    void OnNewBarH1Multi(CRiskManager &risk)
    {
        if(risk.IsDDLocked()) return;
        for(int i = 0; i < m_n_pairs; i++)
        {
            datetime cur = iTime(m_pairs[i], PERIOD_H1, 0);
            if(cur == 0 || cur == m_last_h1_bar[i]) continue;
            m_last_h1_bar[i] = cur;
            ProcessPair(i, risk);
        }
    }

    // CSleeveBase requires OnNewBarD1 — no-op here (we use H1 detection)
    void OnNewBarD1(CRiskManager &risk) override
    {
        // H1 sleeve doesn't trigger on daily recompute; uses OnNewBarH1Multi
    }

    int CloseAll(string reason) override
    {
        return CloseAllByMagic(m_magic, reason);
    }

private:
    void ProcessPair(int i, CRiskManager &risk)
    {
        double ema_fast, ema_slow, rsi, atr;
        if(!ReadShift1(m_h_ema_fast[i], ema_fast)) return;
        if(!ReadShift1(m_h_ema_slow[i], ema_slow)) return;
        if(!ReadShift1(m_h_rsi[i], rsi)) return;
        if(!ReadShift1(m_h_atr[i], atr)) return;

        bool long_signal  = (ema_fast > ema_slow) && (rsi < Inp_H1_RSIHigh);
        bool short_signal = (ema_fast < ema_slow) && (rsi > Inp_H1_RSILow);

        // Vol-target par paire : σ21 sur returns daily (consistent avec TS)
        double sigma21 = ComputePairSigma21(m_pairs[i]);
        double lev_pair = MathMin(Inp_H1_TargetVol / MathMax(sigma21, 0.01),
                                   Inp_H1_MaxLeverage);

        // Trouver position existante
        ulong existing = FindOpenPosition(m_pairs[i]);
        long pos_type = -1;
        if(existing != 0) pos_type = PositionGetInteger(POSITION_TYPE);

        if(pos_type == (long)POSITION_TYPE_BUY && !long_signal)
        {
            m_trade.PositionClose(existing);
            g_logger.Info(m_name, StringFormat("Exit LONG %s (signal flip)", m_pairs[i]));
        }
        else if(pos_type == (long)POSITION_TYPE_SELL && !short_signal)
        {
            m_trade.PositionClose(existing);
            g_logger.Info(m_name, StringFormat("Exit SHORT %s (signal flip)", m_pairs[i]));
        }

        if(existing == 0)
        {
            if(long_signal)  OpenPosition(m_pairs[i], ORDER_TYPE_BUY, lev_pair, atr, risk);
            else if(short_signal) OpenPosition(m_pairs[i], ORDER_TYPE_SELL, lev_pair, atr, risk);
        }
    }

    bool ReadShift1(int handle, double &out)
    {
        double buf[];
        if(CopyBuffer(handle, 0, 1, 1, buf) != 1) return false;
        out = buf[0];
        return true;
    }

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

    void OpenPosition(string symbol, ENUM_ORDER_TYPE type, double lev_pair,
                      double atr, CRiskManager &risk)
    {
        double price = (type == ORDER_TYPE_BUY)
                       ? SymbolInfoDouble(symbol, SYMBOL_ASK)
                       : SymbolInfoDouble(symbol, SYMBOL_BID);
        if(price <= 0.0) return;

        // SL dynamique = 2× ATR(14) H1
        double sl_dist = MathMax(atr * Inp_H1_ATRMultSL, price * 0.005);
        double sl = (type == ORDER_TYPE_BUY) ? price - sl_dist : price + sl_dist;
        sl = EnforceStopLevel(symbol, price, sl, type, true);

        // Slippage paramétré
        double point = SymbolInfoDouble(symbol, SYMBOL_POINT);
        double slip_price = (Inp_H1_SlippageBps / 10000.0) * price;
        int slip_pts = (point > 0.0) ? (int)MathCeil(slip_price / point) : 20;
        m_trade.SetDeviationInPoints(MathMax(slip_pts, 5));

        // Sizing : sub_equity_H1 / n_pairs * lev_pair * global_leverage
        double sub_eq = risk.SubEquity(SLEEVE_H1_MOMENTUM) / m_n_pairs;
        double risk_money = sub_eq * 0.05 * lev_pair * risk.GlobalLeverage();
        double lots = LotsForRisk(symbol, risk_money, sl_dist + slip_price);
        if(lots <= 0.0) return;

        bool ok = false;
        if(type == ORDER_TYPE_BUY)
            ok = m_trade.Buy(lots, symbol, price, sl, 0.0, "H1 Momentum long");
        else
            ok = m_trade.Sell(lots, symbol, price, sl, 0.0, "H1 Momentum short");

        if(!ok || m_trade.ResultRetcode() != TRADE_RETCODE_DONE)
        {
            g_logger.Error(m_name,
                StringFormat("Entry %s failed: retcode=%d", symbol, m_trade.ResultRetcode()));
            return;
        }
        g_logger.Info(m_name,
            StringFormat("Entry %s %s lots=%.2f price=%.5f atr=%.5f lev_pair=%.2f",
                         (type == ORDER_TYPE_BUY ? "LONG" : "SHORT"),
                         symbol, lots, price, atr, lev_pair));
    }
};

#endif // __FX_SLEEVE_H1_MOMENTUM_MQH__
