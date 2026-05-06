//+------------------------------------------------------------------+
//| FxMacroSourceHistory.mqh                                         |
//|                                                                  |
//| Historical macro source for the Strategy Tester. Loads a CSV     |
//| of pre-computed macro filter values and answers point-in-time    |
//| lookups via binary search.                                       |
//|                                                                  |
//| In tester mode TimeCurrent() returns simulated time; the lookup  |
//| therefore gives the latest observation visible at that instant,  |
//| reproducing what a live system would have seen at that date.     |
//|                                                                  |
//| Counterpart of CMacroSourceCalendar / CMacroSourceFRED, which    |
//| query MT5 calendar and the FRED API directly in live mode (both  |
//| unavailable inside the Strategy Tester).                         |
//|                                                                  |
//| Expected CSV schema (header + N rows sorted ascending):          |
//|     timestamp_utc,spread_10y2y,unemp_rising,spread_threshold,    |
//|     macro_ok                                                     |
//|     2019-01-02T00:00:00Z,0.160000,1,0.5000,0                     |
//|     ...                                                          |
//+------------------------------------------------------------------+
#ifndef __FX_MACRO_SOURCE_HISTORY_MQH__
#define __FX_MACRO_SOURCE_HISTORY_MQH__

#include "FxCommon.mqh"

//+------------------------------------------------------------------+
//| CMacroSourceHistory: time-indexed CSV reader with binary search.|
//+------------------------------------------------------------------+
class CMacroSourceHistory
{
private:
    string   m_filename;
    bool     m_use_common;

    datetime m_ts[];
    double   m_spread[];
    bool     m_unemp_rising[];
    bool     m_macro_ok[];
    int      m_count;
    bool     m_loaded;

public:
    CMacroSourceHistory() : m_filename("macro_history.csv"),
                            m_use_common(true),
                            m_count(0), m_loaded(false) {}

    void Init(string filename, bool use_common)
    {
        m_filename = filename;
        m_use_common = use_common;
        m_loaded = false;
        m_count = 0;
        ArrayResize(m_ts, 0);
        ArrayResize(m_spread, 0);
        ArrayResize(m_unemp_rising, 0);
        ArrayResize(m_macro_ok, 0);
    }

    bool IsLoaded() const { return m_loaded; }
    int  Count()    const { return m_count; }

    //--- Load every row of the CSV into memory. Idempotent: subsequent
    //--- calls are a no-op once the data has been cached.
    bool LoadAll()
    {
        if(m_loaded) return true;

        int flags = FILE_READ | FILE_CSV | FILE_ANSI;
        if(m_use_common) flags |= FILE_COMMON;
        int h = FileOpen(m_filename, flags, ',');
        if(h == INVALID_HANDLE)
        {
            PrintFormat("CMacroSourceHistory: cannot open %s (err=%d, common=%d)",
                        m_filename, GetLastError(), (int)m_use_common);
            return false;
        }

        // Skip the header (5 columns).
        for(int i = 0; i < 5; i++) FileReadString(h);

        int cap = 4096;
        ArrayResize(m_ts, cap);
        ArrayResize(m_spread, cap);
        ArrayResize(m_unemp_rising, cap);
        ArrayResize(m_macro_ok, cap);
        m_count = 0;

        while(!FileIsEnding(h))
        {
            string ts_str = FileReadString(h);
            if(StringLen(ts_str) == 0) break;
            string sp_str = FileReadString(h);
            string ur_str = FileReadString(h);
            string th_str = FileReadString(h);  // threshold (informational)
            string mo_str = FileReadString(h);

            datetime ts = ParseIsoUTC(ts_str);
            if(ts == 0) continue;

            if(m_count >= ArraySize(m_ts))
            {
                int new_size = ArraySize(m_ts) * 2;
                ArrayResize(m_ts, new_size);
                ArrayResize(m_spread, new_size);
                ArrayResize(m_unemp_rising, new_size);
                ArrayResize(m_macro_ok, new_size);
            }

            m_ts[m_count]            = ts;
            m_spread[m_count]        = StringToDouble(sp_str);
            m_unemp_rising[m_count]  = (StringToInteger(ur_str) == 1);
            m_macro_ok[m_count]      = (StringToInteger(mo_str) == 1);
            m_count++;
        }
        FileClose(h);

        if(m_count == 0)
        {
            PrintFormat("CMacroSourceHistory: %s has no data rows", m_filename);
            return false;
        }

        ArrayResize(m_ts, m_count);
        ArrayResize(m_spread, m_count);
        ArrayResize(m_unemp_rising, m_count);
        ArrayResize(m_macro_ok, m_count);

        m_loaded = true;
        PrintFormat("CMacroSourceHistory: loaded %d rows from %s [%s ... %s]",
                    m_count, m_filename,
                    TimeToString(m_ts[0], TIME_DATE),
                    TimeToString(m_ts[m_count - 1], TIME_DATE));
        return true;
    }

    //--- Locate the row with the largest timestamp <= 't'. Returns false
    //--- when 't' precedes the first observation.
    bool LookupAt(datetime t,
                  double   &out_spread,
                  bool     &out_unemp_rising,
                  bool     &out_macro_ok,
                  datetime &out_row_ts)
    {
        if(!m_loaded || m_count == 0) return false;
        if(t < m_ts[0]) return false;

        if(t >= m_ts[m_count - 1])
        {
            int last = m_count - 1;
            out_row_ts       = m_ts[last];
            out_spread       = m_spread[last];
            out_unemp_rising = m_unemp_rising[last];
            out_macro_ok     = m_macro_ok[last];
            return true;
        }

        // Binary search invariant: m_ts[lo] <= t < m_ts[hi].
        int lo = 0, hi = m_count - 1;
        while(lo < hi - 1)
        {
            int mid = (lo + hi) / 2;
            if(m_ts[mid] <= t) lo = mid;
            else               hi = mid;
        }
        out_row_ts       = m_ts[lo];
        out_spread       = m_spread[lo];
        out_unemp_rising = m_unemp_rising[lo];
        out_macro_ok     = m_macro_ok[lo];
        return true;
    }

private:
    //--- Parse "YYYY-MM-DDTHH:MM:SSZ" into MQL5 datetime (UTC).
    datetime ParseIsoUTC(string iso)
    {
        if(StringLen(iso) < 19) return 0;
        string ymd = StringSubstr(iso, 0, 10);
        string hms = StringSubstr(iso, 11, 8);
        StringReplace(ymd, "-", ".");
        return StringToTime(ymd + " " + hms);
    }
};

#endif // __FX_MACRO_SOURCE_HISTORY_MQH__
