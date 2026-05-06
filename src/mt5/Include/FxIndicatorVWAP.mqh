//+------------------------------------------------------------------+
//| FxIndicatorVWAP.mqh                                              |
//|                                                                  |
//| Daily-anchored Volume-Weighted Average Price.                    |
//|                                                                  |
//|   typical_price = (high + low + close) / 3                       |
//|   VWAP          = sum(typical_price * tick_volume) / sum(volume) |
//|                                                                  |
//| Cumulative buffers reset at 00:00 UTC. Tick volume is used as a  |
//| proxy for true volume since spot Forex is OTC and no centralised |
//| volume feed exists.                                              |
//+------------------------------------------------------------------+
#ifndef __FX_INDICATOR_VWAP_MQH__
#define __FX_INDICATOR_VWAP_MQH__

#include "FxCommon.mqh"

//+------------------------------------------------------------------+
//| CVWAPDaily: streaming volume-weighted accumulator that resets    |
//| each calendar day in UTC.                                        |
//+------------------------------------------------------------------+
class CVWAPDaily
{
private:
    double   m_cum_pv;       // sum(typical_price * tick_volume) for the day
    double   m_cum_v;        // sum(tick_volume) for the day
    double   m_last_vwap;    // last computed VWAP (or close on first bar)
    datetime m_anchor_day;   // UTC midnight of the active session
    int      m_bars_today;   // number of bars accumulated today

public:
    CVWAPDaily() { Reset(); }

    void Reset()
    {
        m_cum_pv = 0.0;
        m_cum_v  = 0.0;
        m_last_vwap = 0.0;
        m_anchor_day = 0;
        m_bars_today = 0;
    }

    //--- Ingest a closed M1 bar (call exactly once per bar).
    void OnNewBarM1(const MqlRates &bar)
    {
        datetime day = FloorToDayUTC(bar.time);
        if(day != m_anchor_day)
        {
            m_cum_pv = 0.0;
            m_cum_v  = 0.0;
            m_anchor_day = day;
            m_bars_today = 0;
        }
        double tp = (bar.high + bar.low + bar.close) / 3.0;
        double v  = (double)bar.tick_volume;
        m_cum_pv += tp * v;
        m_cum_v  += v;
        m_last_vwap = (m_cum_v > 0.0) ? m_cum_pv / m_cum_v : bar.close;
        m_bars_today++;
    }

    double Get() const       { return m_last_vwap; }
    int    BarsToday() const { return m_bars_today; }
    bool   IsReady() const   { return m_bars_today > 0; }

    //--- Replay every M1 bar of today's session so the accumulator is in
    //--- sync when the EA attaches mid-session.
    bool Warmup(string symbol)
    {
        Reset();
        datetime now = TimeGMT();
        datetime midnight = FloorToDayUTC(now);
        MqlRates rates[];
        int copied = CopyRates(symbol, PERIOD_M1, midnight, now, rates);
        if(copied <= 0)
        {
            PrintFormat("CVWAPDaily::Warmup: no bars for %s since midnight UTC",
                        symbol);
            return false;
        }
        // CopyRates returns chronological ascending order by default.
        for(int i = 0; i < copied; i++) OnNewBarM1(rates[i]);
        return true;
    }
};

#endif // __FX_INDICATOR_VWAP_MQH__
