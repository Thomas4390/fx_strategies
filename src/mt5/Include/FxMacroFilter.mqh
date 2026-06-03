//+------------------------------------------------------------------+
//| FxMacroFilter.mqh                                                |
//|                                                                  |
//| Two-stage macro regime filter consumed by the MR Macro sleeve.   |
//|                                                                  |
//|   Stage 1: 10Y-2Y Treasury spread below configurable threshold.  |
//|   Stage 2: US unemployment rate not in a 3-month uptrend.        |
//|                                                                  |
//| Macro state can be sourced from five backends through            |
//| EMacroSourceMode (FILE / NATIVE / HYBRID / HISTORY / AUTO).      |
//| AUTO is the recommended setting: HISTORY is selected when the    |
//| EA runs inside the Strategy Tester (WebRequest is unavailable    |
//| there) and NATIVE when running live.                             |
//|                                                                  |
//| The filter additionally hosts a news-window guard that blocks    |
//| entries during ±15 minutes around scheduled high-impact USD      |
//| releases (NFP, CPI, FOMC, etc.) where typical Forex spreads can  |
//| widen by an order of magnitude.                                  |
//|                                                                  |
//| Common CSV schema (FILE & HISTORY):                              |
//|     timestamp_utc,spread_10y2y,unemp_rising,spread_threshold,    |
//|     macro_ok                                                     |
//|     2026-04-24T18:00:00Z,0.3520,0,0.50,1                         |
//+------------------------------------------------------------------+
#ifndef __FX_MACRO_FILTER_MQH__
#define __FX_MACRO_FILTER_MQH__

#include "FxCommon.mqh"
#include "FxMacroSourceNative.mqh"
#include "FxMacroSourceHistory.mqh"
#include "FxNewsFilter.mqh"

//+------------------------------------------------------------------+
//| CMacroFilter: source dispatcher, validity check, accessors.      |
//+------------------------------------------------------------------+
class CMacroFilter
{
private:
    EMacroSourceMode m_mode;
    string   m_filename;
    int      m_max_age_seconds;
    bool     m_use_common;
    double   m_user_threshold;
    string   m_fred_api_key_file;
    bool     m_fred_key_use_common;
    string   m_history_filename;
    bool     m_history_use_common;
    bool     m_disable_filter;       // diagnostic bypass that forces MacroOk()=true

    datetime m_last_refresh;
    datetime m_last_read_at;
    double   m_spread;
    bool     m_unemp_rising;
    double   m_spread_threshold;
    bool     m_macro_ok;
    bool     m_loaded;
    string   m_last_source;          // "file" / "native" / "history"
    string   m_last_native_error;    // reason the last NATIVE refresh failed

    CMacroSourceCalendar m_cal;
    CMacroSourceFRED     m_fred;
    CMacroSourceHistory  m_history;
    CFxNewsFilter        m_news;

public:
    CMacroFilter() : m_mode(MACRO_SOURCE_AUTO),
                     m_filename("macro_cache.csv"),
                     m_max_age_seconds(86400),
                     m_use_common(true),
                     m_user_threshold(0.5),
                     m_fred_api_key_file("fred_api_key.txt"),
                     m_fred_key_use_common(true),
                     m_history_filename("macro_history.csv"),
                     m_history_use_common(true),
                     m_disable_filter(false),
                     m_last_refresh(0), m_last_read_at(0),
                     m_spread(0.0), m_unemp_rising(false),
                     m_spread_threshold(0.5), m_macro_ok(false),
                     m_loaded(false), m_last_source("none"),
                     m_last_native_error("") {}

    void Init(EMacroSourceMode mode,
              string filename, int max_age_hours, bool use_common,
              double spread_threshold,
              string fred_api_key_file, bool fred_key_use_common,
              string history_filename = "macro_history.csv",
              bool   history_use_common = true,
              string fred_series_id = "T10Y2Y",
              bool   disable_filter = false,
              bool   news_filter_enabled = true)
    {
        m_mode = mode;
        m_filename = filename;
        m_max_age_seconds = max_age_hours * 3600;
        m_use_common = use_common;
        m_user_threshold = spread_threshold;
        m_fred_api_key_file = fred_api_key_file;
        m_fred_key_use_common = fred_key_use_common;
        m_history_filename = history_filename;
        m_history_use_common = history_use_common;
        m_disable_filter = disable_filter;

        m_cal.Init("US", "Unemployment Rate");
        m_fred.Init(fred_series_id, ReadFREDKey());
        m_history.Init(history_filename, history_use_common);
        m_news.Init("US", news_filter_enabled);
    }

    //--- Refresh the macro state. Dispatches to the active source.
    //--- AUTO mode picks HISTORY in the tester or NATIVE in live.
    bool Refresh()
    {
        EMacroSourceMode effective = ResolveEffectiveMode();
        switch(effective)
        {
            case MACRO_SOURCE_FILE:    return RefreshFromFile();
            case MACRO_SOURCE_NATIVE:  return RefreshFromNative();
            case MACRO_SOURCE_HISTORY: return RefreshFromHistory();
            case MACRO_SOURCE_HYBRID:
            {
                if(RefreshFromNative()) return true;
                Print("CMacroFilter: native sources failed, fallback to file");
                return RefreshFromFile();
            }
            default: break;
        }
        return false;
    }

    //--- Resolve AUTO to a concrete mode based on runtime context. The
    //--- result is idempotent and exposed for logging / preflight.
    EMacroSourceMode ResolveEffectiveMode() const
    {
        if(m_mode != MACRO_SOURCE_AUTO) return m_mode;
        if(MQLInfoInteger(MQL_TESTER)) return MACRO_SOURCE_HISTORY;
        return MACRO_SOURCE_NATIVE;
    }

    bool IsValid() const
    {
        if(!m_loaded) return false;
        if(m_last_refresh == 0) return false;
        return (TimeGMT() - m_last_refresh) <= m_max_age_seconds;
    }

    bool   MacroOk() const
    {
        if(m_disable_filter) return true;   // bypass for filter-impact tests
        return m_loaded && m_macro_ok;
    }
    double Spread() const             { return m_spread; }
    bool   UnempRising() const        { return m_unemp_rising; }
    double SpreadThreshold() const    { return m_spread_threshold; }
    datetime LastRefresh() const      { return m_last_refresh; }
    string LastSource() const         { return m_last_source; }

    //--- Reason the last NATIVE (live) refresh failed, for a user-facing
    //--- alert. Empty after a successful refresh. Only meaningful when the
    //--- effective mode is NATIVE.
    string LastError() const          { return m_last_native_error; }
    int    AgeSeconds() const
    {
        if(!m_loaded || m_last_refresh == 0) return 999999;
        return (int)(TimeGMT() - m_last_refresh);
    }

    //--- News-window helper: true if 't' is within ±15 minutes of a
    //--- scheduled high-impact USD event. Refresh of the underlying
    //--- cache is idempotent and rate-limited internally.
    bool IsInNewsWindow(datetime t)
    {
        m_news.Refresh(t);
        return m_news.IsInNewsWindow(t);
    }

    bool NewsFilterEnabled() const { return m_news.Enabled(); }

private:
    //--- Read a single-row CSV (legacy bridge format).
    bool RefreshFromFile()
    {
        int flags = FILE_READ | FILE_CSV | FILE_ANSI;
        if(m_use_common) flags |= FILE_COMMON;
        int h = FileOpen(m_filename, flags, ',');
        if(h == INVALID_HANDLE)
        {
            PrintFormat("CMacroFilter::FILE: cannot open %s (err=%d, common=%d)",
                        m_filename, GetLastError(), (int)m_use_common);
            return false;
        }
        for(int i = 0; i < 5; i++) FileReadString(h);  // header
        if(FileIsEnding(h))
        {
            PrintFormat("CMacroFilter::FILE: %s has no data row", m_filename);
            FileClose(h);
            return false;
        }
        string ts = FileReadString(h);
        m_spread            = StringToDouble(FileReadString(h));
        m_unemp_rising      = (StringToInteger(FileReadString(h)) == 1);
        m_spread_threshold  = StringToDouble(FileReadString(h));
        m_macro_ok          = (StringToInteger(FileReadString(h)) == 1);
        FileClose(h);

        m_last_refresh = ParseIsoUTC(ts);
        m_last_read_at = TimeGMT();
        m_loaded = true;
        m_last_source = "file";
        return true;
    }

    //--- Pull macro state from MT5 calendar + FRED API.
    bool RefreshFromNative()
    {
        bool unemp_rising = false;
        if(!m_cal.ComputeUnempRising(unemp_rising))
        {
            m_last_native_error = "Calendar MT5 : lecture du chomage US echouee";
            Print("CMacroFilter::NATIVE: calendar unemployment failed");
            return false;
        }
        double spread = 0.0;
        datetime obs_date = 0;
        if(!m_fred.FetchLatest(spread, obs_date))
        {
            m_last_native_error = m_fred.LastError();
            Print("CMacroFilter::NATIVE: FRED fetch failed");
            return false;
        }
        m_spread = spread;
        m_unemp_rising = unemp_rising;
        m_spread_threshold = m_user_threshold;
        m_macro_ok = (m_spread < m_spread_threshold) && (!m_unemp_rising);
        m_last_refresh = TimeGMT();
        m_last_read_at = m_last_refresh;
        m_loaded = true;
        m_last_source = "native";
        m_last_native_error = "";
        PrintFormat("CMacroFilter::NATIVE OK: spread=%.4f unemp_rising=%d "
                    "macro_ok=%d", m_spread, (int)m_unemp_rising,
                    (int)m_macro_ok);
        return true;
    }

    //--- Time-indexed lookup in the historical CSV. Lazy-loaded on the
    //--- first call. TimeCurrent() is the simulated time in tester and
    //--- the real wall-clock in live mode.
    bool RefreshFromHistory()
    {
        if(!m_history.IsLoaded())
        {
            if(!m_history.LoadAll())
            {
                Print("CMacroFilter::HISTORY: failed to load history CSV");
                return false;
            }
        }
        datetime t = TimeCurrent();
        double sp = 0.0;
        bool ur = false, mo = false;
        datetime row_ts = 0;
        if(!m_history.LookupAt(t, sp, ur, mo, row_ts))
        {
            PrintFormat("CMacroFilter::HISTORY: no row at or before %s",
                        TimeToString(t, TIME_DATE | TIME_SECONDS));
            return false;
        }
        m_spread           = sp;
        m_unemp_rising     = ur;
        m_macro_ok         = mo;
        m_spread_threshold = m_user_threshold;
        m_last_refresh     = row_ts;
        m_last_read_at     = TimeGMT();
        m_loaded           = true;
        m_last_source      = "history";
        return true;
    }

    //--- Read the FRED API key from a sandbox-local file. The key is
    //--- kept out of inputs and out of the source tree.
    string ReadFREDKey()
    {
        int flags = FILE_READ | FILE_TXT | FILE_ANSI;
        if(m_fred_key_use_common) flags |= FILE_COMMON;
        int h = FileOpen(m_fred_api_key_file, flags);
        if(h == INVALID_HANDLE) return "";
        string key = FileReadString(h);
        FileClose(h);
        StringTrimLeft(key);
        StringTrimRight(key);
        return key;
    }

    datetime ParseIsoUTC(string iso)
    {
        if(StringLen(iso) < 19) return 0;
        string ymd = StringSubstr(iso, 0, 10);
        string hms = StringSubstr(iso, 11, 8);
        StringReplace(ymd, "-", ".");
        return StringToTime(ymd + " " + hms);
    }
};

#endif // __FX_MACRO_FILTER_MQH__
