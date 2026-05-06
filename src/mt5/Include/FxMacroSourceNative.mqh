//+------------------------------------------------------------------+
//| FxMacroSourceNative.mqh                                          |
//|                                                                  |
//| Native MQL5 macro sources used in live mode (the Strategy        |
//| Tester blocks both endpoints, so use CMacroSourceHistory there). |
//|                                                                  |
//|   - CMacroSourceCalendar : US unemployment rate via the MT5      |
//|     economic calendar (CalendarValueHistoryByEvent).             |
//|                                                                  |
//|   - CMacroSourceFRED     : Treasury 10Y-2Y spread via WebRequest |
//|     against the FRED API (St. Louis Fed) using the ALFRED        |
//|     endpoint to retrieve release-date-indexed observations.      |
//|                                                                  |
//| FRED prerequisites:                                              |
//|   1. Free API key: https://fred.stlouisfed.org/docs/api/api_key  |
//|   2. Whitelist https://api.stlouisfed.org in MetaTrader options  |
//|      ("Allow WebRequest for listed URL").                        |
//|                                                                  |
//| Calendar prerequisites: none (native MT5).                       |
//+------------------------------------------------------------------+
#ifndef __FX_MACRO_SOURCE_NATIVE_MQH__
#define __FX_MACRO_SOURCE_NATIVE_MQH__

#include "FxCommon.mqh"

#define FX_FRED_TIMEOUT_MS    5000
#define FX_FRED_MAX_RETRIES   2
#define FX_FRED_RETRY_BACKOFF 2000

//+------------------------------------------------------------------+
//| CMacroSourceCalendar: fetches the latest US unemployment rate    |
//| releases from the MT5 calendar and computes a 3-month change.   |
//+------------------------------------------------------------------+
class CMacroSourceCalendar
{
private:
    string m_country;
    string m_event_keyword;
    ulong  m_cached_event_id;
    bool   m_event_resolved;

public:
    CMacroSourceCalendar() : m_country("US"),
                             m_event_keyword("Unemployment Rate"),
                             m_cached_event_id(0),
                             m_event_resolved(false) {}

    void Init(string country, string event_keyword)
    {
        m_country = country;
        m_event_keyword = event_keyword;
        m_event_resolved = false;
        m_cached_event_id = 0;
    }

    //--- Resolve the calendar event ID for the unemployment release.
    bool ResolveEventId()
    {
        if(m_event_resolved) return true;
        MqlCalendarEvent events[];
        int n = CalendarEventByCountry(m_country, events);
        if(n <= 0)
        {
            PrintFormat("CalendarEventByCountry(%s) returned %d events (err=%d)",
                        m_country, n, GetLastError());
            return false;
        }
        for(int i = 0; i < n; i++)
        {
            if(StringFind(events[i].name, m_event_keyword) >= 0)
            {
                m_cached_event_id = events[i].id;
                m_event_resolved = true;
                PrintFormat("CMacroSourceCalendar: resolved '%s' event_id=%I64u",
                            events[i].name, m_cached_event_id);
                return true;
            }
        }
        PrintFormat("CMacroSourceCalendar: no event matching '%s' for %s",
                    m_event_keyword, m_country);
        return false;
    }

    //--- Copy the most recent 'n_releases' published values into the
    //--- output arrays, in chronological ascending order. Returns the
    //--- number of values written.
    //---
    //--- The raw 'actual_value' returned by the calendar is scaled by
    //--- 'event.multiplier' (THOUSANDS, MILLIONS, ...). The function
    //--- divides by the resolved scale so callers get plain percentages.
    int GetRecentReleases(int n_releases, double &out_values[], datetime &out_times[])
    {
        if(!ResolveEventId()) return 0;

        MqlCalendarEvent ev;
        double divisor = 1e6;  // safe default for unemployment rate
        if(CalendarEventById(m_cached_event_id, ev) && ev.multiplier > 0)
        {
            switch((int)ev.multiplier)
            {
                case 0: divisor = 1.0;       break;  // NONE
                case 1: divisor = 1e3;       break;  // THOUSANDS
                case 2: divisor = 1e6;       break;  // MILLIONS
                case 3: divisor = 1e9;       break;  // BILLIONS
                case 4: divisor = 1e12;      break;  // TRILLIONS
                default: divisor = 1e6;      break;
            }
        }

        // 18-month lookback window covers monthly publications with margin.
        datetime now = TimeGMT();
        datetime from = (datetime)((long)now - 86400L * 30L * 18L);

        MqlCalendarValue values[];
        int total = CalendarValueHistoryByEvent(m_cached_event_id, values, from, now);
        if(total <= 0) return 0;

        int valid = 0;
        for(int i = 0; i < total; i++)
        {
            if(values[i].HasActualValue()) valid++;
        }
        if(valid == 0) return 0;

        int take = (n_releases < valid) ? n_releases : valid;
        ArrayResize(out_values, take);
        ArrayResize(out_times, take);

        // CalendarValueHistoryByEvent returns ascending order; copy the
        // last 'take' entries while skipping invalid actuals.
        int written = 0;
        for(int i = total - 1; i >= 0 && written < take; i--)
        {
            if(!values[i].HasActualValue()) continue;
            out_values[take - 1 - written] = (double)values[i].actual_value / divisor;
            out_times[take - 1 - written]  = values[i].time;
            written++;
        }
        return written;
    }

    //--- Compute a "rising unemployment" flag from the 3-month diff.
    bool ComputeUnempRising(bool &out_rising)
    {
        double vals[];
        datetime times[];
        int n = GetRecentReleases(4, vals, times);
        if(n < 4) return false;
        double diff_3m = vals[n - 1] - vals[n - 4];
        out_rising = (diff_3m > 0.0);
        return true;
    }
};

//+------------------------------------------------------------------+
//| CMacroSourceFRED: fetch the latest observation for a FRED        |
//| series via WebRequest. Default series: T10Y2Y (Treasury 10Y-2Y). |
//|                                                                  |
//| Uses the ALFRED endpoint (realtime_start parameter) so revised   |
//| observations are anchored to their actual publication date and   |
//| not the period date. This avoids a subtle look-ahead bias when   |
//| the series is later revised retroactively.                       |
//+------------------------------------------------------------------+
class CMacroSourceFRED
{
private:
    string m_series_id;
    string m_api_key;
    string m_base_url;

public:
    CMacroSourceFRED() : m_series_id("T10Y2Y"),
                         m_api_key(""),
                         m_base_url("https://api.stlouisfed.org") {}

    void Init(string series_id, string api_key)
    {
        m_series_id = series_id;
        m_api_key = api_key;
    }

    bool HasApiKey() const { return StringLen(m_api_key) > 0; }

    //--- Retrieve the latest observation. Performs up to
    //--- FX_FRED_MAX_RETRIES attempts with FX_FRED_RETRY_BACKOFF ms
    //--- between attempts to absorb transient timeouts.
    bool FetchLatest(double &out_value, datetime &out_obs_date)
    {
        if(!HasApiKey())
        {
            Print("CMacroSourceFRED: no API key configured");
            return false;
        }
        string url = m_base_url + "/fred/series/observations"
                     + "?series_id=" + m_series_id
                     + "&api_key=" + m_api_key
                     + "&file_type=json"
                     + "&realtime_start=2000-01-01"
                     + "&realtime_end=9999-12-31"
                     + "&limit=1"
                     + "&sort_order=desc";

        for(int attempt = 1; attempt <= FX_FRED_MAX_RETRIES; attempt++)
        {
            char post[], result[];
            string response_headers;
            ResetLastError();
            int code = WebRequest("GET", url, NULL, NULL, FX_FRED_TIMEOUT_MS,
                                  post, 0, result, response_headers);
            if(code == 200)
            {
                string body = CharArrayToString(result, 0, WHOLE_ARRAY, CP_UTF8);
                return ParseLatestObservation(body, out_value, out_obs_date);
            }

            if(code == -1)
            {
                PrintFormat("CMacroSourceFRED::WebRequest err=%d "
                            "(check URL whitelist for %s) attempt=%d/%d",
                            GetLastError(), m_base_url,
                            attempt, FX_FRED_MAX_RETRIES);
            }
            else
            {
                PrintFormat("CMacroSourceFRED::WebRequest HTTP %d attempt=%d/%d",
                            code, attempt, FX_FRED_MAX_RETRIES);
            }
            if(attempt < FX_FRED_MAX_RETRIES)
                Sleep(FX_FRED_RETRY_BACKOFF);
        }
        return false;
    }

private:
    //--- Minimal JSON extractor: locates the first "value":"..." and
    //--- "date":"..." substrings in the FRED payload. Sufficient for
    //--- the deterministic single-observation response we request.
    bool ParseLatestObservation(string body, double &out_value, datetime &out_date)
    {
        int idx_value = StringFind(body, "\"value\":\"");
        int idx_date  = StringFind(body, "\"date\":\"");
        if(idx_value < 0 || idx_date < 0)
        {
            Print("CMacroSourceFRED: cannot find value/date in JSON body");
            return false;
        }
        // Value
        int v_start = idx_value + StringLen("\"value\":\"");
        int v_end   = StringFind(body, "\"", v_start);
        if(v_end <= v_start) return false;
        string v_str = StringSubstr(body, v_start, v_end - v_start);
        if(v_str == "." || v_str == "")
        {
            Print("CMacroSourceFRED: empty value");
            return false;
        }
        out_value = StringToDouble(v_str);

        // Date
        int d_start = idx_date + StringLen("\"date\":\"");
        int d_end   = StringFind(body, "\"", d_start);
        if(d_end <= d_start) return false;
        string d_str = StringSubstr(body, d_start, d_end - d_start);
        StringReplace(d_str, "-", ".");
        out_date = StringToTime(d_str);
        return true;
    }
};

#endif // __FX_MACRO_SOURCE_NATIVE_MQH__
