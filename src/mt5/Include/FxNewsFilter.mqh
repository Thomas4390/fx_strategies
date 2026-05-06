//+------------------------------------------------------------------+
//| FxNewsFilter.mqh                                                 |
//|                                                                  |
//| Time-based filter that flags windows around scheduled high-      |
//| impact macroeconomic releases for a given country (default US). |
//| Used by intraday sleeves to avoid trading the seconds before and |
//| after events such as Non-Farm Payrolls, CPI, or FOMC where the   |
//| typical EUR/USD spread can spike from ~1 pip to 8-10 pips.       |
//|                                                                  |
//| The cache is refreshed at most once per FX_NEWS_CACHE_HOURS      |
//| period to keep tick-handler overhead negligible. The MT5         |
//| economic calendar is queried both in live mode and in the        |
//| Strategy Tester (where it returns the historical event set).    |
//+------------------------------------------------------------------+
#ifndef __FX_NEWS_FILTER_MQH__
#define __FX_NEWS_FILTER_MQH__

#include "FxCommon.mqh"

#define FX_NEWS_BUFFER_SECONDS  900   // ±15 minutes around each release
#define FX_NEWS_CACHE_HOURS     1     // refresh interval for the cache
#define FX_NEWS_LOOKAHEAD_HOURS 48    // calendar query window
#define FX_NEWS_MAX_EVENTS      200   // upper bound on cached events

//+------------------------------------------------------------------+
//| CFxNewsFilter                                                    |
//+------------------------------------------------------------------+
class CFxNewsFilter
{
private:
    string   m_country;
    datetime m_event_times[];
    int      m_event_count;
    datetime m_last_refresh;
    bool     m_enabled;

public:
    CFxNewsFilter() : m_country("US"),
                      m_event_count(0),
                      m_last_refresh(0),
                      m_enabled(true) {}

    void Init(string country = "US", bool enabled = true)
    {
        m_country = country;
        m_enabled = enabled;
        m_event_count = 0;
        m_last_refresh = 0;
        ArrayResize(m_event_times, FX_NEWS_MAX_EVENTS);
    }

    bool Enabled() const { return m_enabled; }
    int  CachedEventsCount() const { return m_event_count; }

    //--- Refresh the cached release timestamps. The function is idempotent
    //--- and exits early when the previous refresh is still recent enough.
    void Refresh(datetime now)
    {
        if(!m_enabled) return;
        if(m_last_refresh != 0 &&
           (now - m_last_refresh) < FX_NEWS_CACHE_HOURS * 3600)
            return;

        datetime from = (datetime)((long)now - FX_NEWS_LOOKAHEAD_HOURS * 3600L);
        datetime to   = (datetime)((long)now + FX_NEWS_LOOKAHEAD_HOURS * 3600L);

        MqlCalendarValue values[];
        int total = CalendarValueHistory(values, from, to, m_country);
        if(total <= 0)
        {
            m_event_count = 0;
            m_last_refresh = now;
            return;
        }

        int kept = 0;
        for(int i = 0; i < total && kept < FX_NEWS_MAX_EVENTS; i++)
        {
            // Resolve the event so we can inspect its importance flag.
            MqlCalendarEvent ev;
            if(!CalendarEventById(values[i].event_id, ev)) continue;
            if(ev.importance != CALENDAR_IMPORTANCE_HIGH)  continue;
            m_event_times[kept] = values[i].time;
            kept++;
        }
        m_event_count = kept;
        m_last_refresh = now;
    }

    //--- Return true when 't' falls within ±FX_NEWS_BUFFER_SECONDS of any
    //--- cached high-impact release for the configured country.
    bool IsInNewsWindow(datetime t)
    {
        if(!m_enabled || m_event_count == 0) return false;
        for(int i = 0; i < m_event_count; i++)
        {
            long delta = (long)(t - m_event_times[i]);
            if(delta < 0) delta = -delta;
            if(delta <= FX_NEWS_BUFFER_SECONDS) return true;
        }
        return false;
    }
};

#endif // __FX_NEWS_FILTER_MQH__
