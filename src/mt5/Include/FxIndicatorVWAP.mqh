//+------------------------------------------------------------------+
//| FxIndicatorVWAP.mqh                                              |
//| VWAP daily-anchored : reset à 00:00 UTC, cumul tick_volume       |
//| × typical_price = (high+low+close)/3 (convention vbt.VWAP).      |
//+------------------------------------------------------------------+
#ifndef __FX_INDICATOR_VWAP_MQH__
#define __FX_INDICATOR_VWAP_MQH__

#include "FxCommon.mqh"

//+------------------------------------------------------------------+
//| CVWAPDaily — accumulateur volume-weighted price reset à chaque   |
//| nouveau jour UTC. Maintient cum_pv et cum_v en interne.          |
//+------------------------------------------------------------------+
class CVWAPDaily
{
private:
    double   m_cum_pv;       // Sum(typical_price * volume) du jour
    double   m_cum_v;        // Sum(volume) du jour
    double   m_last_vwap;    // Dernière valeur calculée
    datetime m_anchor_day;   // Minuit UTC du jour courant
    int      m_bars_today;   // Nombre de bars ingérées sur le jour

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

    //--- Ingère une bar M1. À appeler une fois par bar fermée.
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

    double Get() const { return m_last_vwap; }

    int    BarsToday() const { return m_bars_today; }

    bool   IsReady() const { return m_bars_today > 0; }

    //--- Reconstruit le cumul depuis minuit UTC du jour courant.
    //--- À appeler à OnInit pour ne pas démarrer "vide" en milieu de session.
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
        // CopyRates retourne en ordre chronologique croissant par défaut
        for(int i = 0; i < copied; i++) OnNewBarM1(rates[i]);
        return true;
    }
};

#endif // __FX_INDICATOR_VWAP_MQH__
