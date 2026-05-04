//+------------------------------------------------------------------+
//| FxSleeveTSMomentum.mqh                                           |
//| Sleeve 2 — Time-Series Momentum daily multi-paires.              |
//|                                                                  |
//| Source de vérité : src/strategies/daily_momentum.py              |
//|   fast_ema=20, slow_ema=50, rsi_period=7                          |
//|   LONG : EMA20 > EMA50 AND RSI(7) < 60                            |
//|   SHORT : EMA20 < EMA50 AND RSI(7) > 40                           |
//|   Sortie : inversion du critère                                   |
//|   Vol target par paire : lev = min(0.10 / max(σ21, 0.01), 3.0)    |
//|   Paires : EURUSD, GBPUSD, USDJPY (3 paires equal-weight)         |
//+------------------------------------------------------------------+
#ifndef __FX_SLEEVE_TS_MOMENTUM_MQH__
#define __FX_SLEEVE_TS_MOMENTUM_MQH__

#include <Trade/Trade.mqh>
#include "FxSleeveBase.mqh"
#include "FxCommon.mqh"
#include "FxRiskManager.mqh"
#include "FxTradeHelpers.mqh"

//--- Inputs accessibles via #include textuel depuis l'EA principal.

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
            g_logger.Error(m_name, StringFormat("invalid Inp_TS_Pairs=%s", Inp_TS_Pairs));
            return false;
        }
        // Pack valid pairs into [0..valid_n-1]. Skip pairs without any D1
        // history so adding e.g. EURJPY (broker D1 starts 2022-11) on a
        // 2020-2026 window doesn't disable the entire sleeve.
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
            // Try ideal 250 D1 bars, gracefully accept any history >= 1.
            // Below 1 bar there is no D1 data at all -> skip this pair only
            // (was hard-fail before 2026-05-04).
            if(!EnsureHistory(pair, PERIOD_D1, 250))
            {
                if(!EnsureHistory(pair, PERIOD_D1, 1))
                {
                    g_logger.Warn(m_name, StringFormat(
                        "%s: skipped (no D1 history available on broker)", pair));
                    continue;
                }
                g_logger.Warn(m_name, StringFormat(
                    "%s: %d/250 D1 bars; iMA(%d)/iRSI(%d) warm up as bars accumulate",
                    pair, (int)Bars(pair, PERIOD_D1),
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
            g_logger.Error(m_name, "no valid pairs after filtering; sleeve disabled");
            return false;
        }
        m_n_pairs = valid_n;
        m_trade.SetExpertMagicNumber(m_magic);
        m_trade.SetDeviationInPoints(20);
        g_logger.Info(m_name, StringFormat(
            "Init OK %d/%d pairs (skipped %d for missing D1/symbol)",
            valid_n, n, n - valid_n));
        return true;
    }

    void Shutdown() override
    {
        for(int i = 0; i < m_n_pairs; i++)
        {
            if(m_h_ema_fast[i] != INVALID_HANDLE) IndicatorRelease(m_h_ema_fast[i]);
            if(m_h_ema_slow[i] != INVALID_HANDLE) IndicatorRelease(m_h_ema_slow[i]);
            if(m_h_rsi[i]      != INVALID_HANDLE) IndicatorRelease(m_h_rsi[i]);
        }
    }

    //--- Hook NewBar D1 (déclenché par OnTimer après le close UTC).
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

        // Vol-target par paire : σ21 sur returns daily de la paire
        double sigma21 = ComputePairSigma21(m_pairs[i]);
        double lev_pair = MathMin(Inp_TS_TargetVol / MathMax(sigma21, 0.01),
                                   Inp_TS_MaxLeverage);

        // Trouver la position existante sur cette paire
        ulong existing = FindOpenPosition(m_pairs[i]);
        long pos_type = -1;
        if(existing != 0) pos_type = PositionGetInteger(POSITION_TYPE);

        // Sortie : inversion du critère (cf. daily_momentum.py)
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

        // Entrée si pas de position
        if(existing == 0)
        {
            if(long_signal)  OpenPosition(m_pairs[i], ORDER_TYPE_BUY, lev_pair, risk);
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

    double ComputePairSigma21(string symbol)
    {
        double closes[22];
        int copied = CopyClose(symbol, PERIOD_D1, 1, 22, closes);
        if(copied < 22) return 0.10;  // fallback : suppose 10% (cible)
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

    //--- Slippage Inp_TS_SlippageBps converti en SYMBOL_POINT puis appliqué
    //--- via SetDeviationInPoints. Le sl_distance est aussi majoré pour
    //--- intégrer le coût slippage dans le sizing (cf. plan d'alignement).
    //--- Le risk_per_trade=0.05 (5% du sub-equity par paire) reste cohérent
    //--- avec la sémantique notional Python : sub_equity_per_pair × lev_pair
    //--- avec SL safety à 5% donne un risk_money équivalent.
    void OpenPosition(string symbol, ENUM_ORDER_TYPE type, double lev_pair,
                      CRiskManager &risk)
    {
        double price = (type == ORDER_TYPE_BUY)
                       ? SymbolInfoDouble(symbol, SYMBOL_ASK)
                       : SymbolInfoDouble(symbol, SYMBOL_BID);
        if(price <= 0.0) return;

        // Pas de SL/TP dur en TS Momentum (sortie par signal flip)
        // Mais on pose un SL "garde-fou" très large à 5% pour limiter les surprises
        double sl_dist_safety = price * 0.05;
        double sl = (type == ORDER_TYPE_BUY) ? price - sl_dist_safety
                                             : price + sl_dist_safety;
        sl = EnforceStopLevel(symbol, price, sl, type, true);

        // Slippage paramétré
        double point = SymbolInfoDouble(symbol, SYMBOL_POINT);
        double slip_price = (Inp_TS_SlippageBps / 10000.0) * price;
        int slip_pts = (point > 0.0) ? (int)MathCeil(slip_price / point) : 20;
        m_trade.SetDeviationInPoints(MathMax(slip_pts, 5));

        // Sizing : sub_equity_TS / n_pairs * lev_pair * global_leverage
        // SL distance majoré du slippage pour ne pas sur-sizer face au coût attendu
        double sub_eq = risk.SubEquity(SLEEVE_TS_MOMENTUM) / m_n_pairs;
        double risk_money = sub_eq * 0.05 * lev_pair * risk.GlobalLeverage();
        double lots = LotsForRisk(symbol, risk_money, sl_dist_safety + slip_price);
        if(lots <= 0.0) return;

        bool ok = false;
        if(type == ORDER_TYPE_BUY)
            ok = m_trade.Buy(lots, symbol, price, sl, 0.0, "TS Momentum long");
        else
            ok = m_trade.Sell(lots, symbol, price, sl, 0.0, "TS Momentum short");

        if(!ok || m_trade.ResultRetcode() != TRADE_RETCODE_DONE)
        {
            g_logger.Error(m_name,
                StringFormat("Entry %s failed: retcode=%d", symbol, m_trade.ResultRetcode()));
            return;
        }
        g_logger.Info(m_name,
            StringFormat("Entry %s %s lots=%.2f price=%.5f lev_pair=%.2f slip_pts=%d",
                         (type == ORDER_TYPE_BUY ? "LONG" : "SHORT"),
                         symbol, lots, price, lev_pair, slip_pts));
    }
};

#endif // __FX_SLEEVE_TS_MOMENTUM_MQH__
