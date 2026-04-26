//+------------------------------------------------------------------+
//| FxMacroSourceNative.mqh                                          |
//| Sources macro natives MQL5 (sans bridge Python externe).         |
//|                                                                  |
//|  - CMacroSourceCalendar : taux de chômage US via le calendrier   |
//|    économique intégré MT5 (CalendarValueHistoryByEvent).          |
//|  - CMacroSourceFRED     : spread Treasury 10Y-2Y via WebRequest  |
//|    sur l'API FRED (St. Louis Fed).                                |
//|                                                                  |
//| Pré-requis FRED :                                                 |
//|  1. Clé API gratuite : https://fred.stlouisfed.org/docs/api/      |
//|     api_key.html                                                  |
//|  2. URL whitelistée dans Terminal → Outils → Options → Expert    |
//|     Advisors → "Allow WebRequest for listed URL" :                |
//|         https://api.stlouisfed.org                                |
//|                                                                  |
//| Pré-requis Calendar : aucun (natif MT5).                          |
//+------------------------------------------------------------------+
#ifndef __FX_MACRO_SOURCE_NATIVE_MQH__
#define __FX_MACRO_SOURCE_NATIVE_MQH__

#include "FxCommon.mqh"

//+------------------------------------------------------------------+
//| CMacroSourceCalendar : accès au calendar économique MT5 pour     |
//| récupérer les 4 dernières publications du US Unemployment Rate   |
//| et calculer la variation 3 mois.                                 |
//+------------------------------------------------------------------+
class CMacroSourceCalendar
{
private:
    string m_country;       // "US"
    string m_event_keyword; // "Unemployment Rate"
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

    //--- Résout l'ID de l'événement "Unemployment Rate" pour `country`.
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

    //--- Récupère les `n_releases` dernières publications dans `out` (chrono asc).
    //--- Retourne le nombre de publications copiées.
    int GetRecentReleases(int n_releases, double &out_values[], datetime &out_times[])
    {
        if(!ResolveEventId()) return 0;
        // Fenêtre : 18 mois en arrière (chômage publié mensuellement)
        datetime now = TimeGMT();
        datetime from = now - (long)(86400 * 30 * 18);

        MqlCalendarValue values[];
        int total = CalendarValueHistoryByEvent(m_cached_event_id, values, from, now);
        if(total <= 0) return 0;

        // Filtrer : ne garder que les publications avec actual_value valide
        // (HasValue() équivaut à `actual_value != LONG_MAX` dans la doc MQL5).
        int valid = 0;
        for(int i = 0; i < total; i++)
        {
            if(values[i].HasActualValue())
                valid++;
        }
        if(valid == 0) return 0;

        // Allocation et copie chrono ascendante des `n_releases` dernières
        int take = (n_releases < valid) ? n_releases : valid;
        ArrayResize(out_values, take);
        ArrayResize(out_times, take);

        // Les MqlCalendarValue retournés sont déjà chrono ascendant (doc MQL5).
        // On prend les `take` dernières.
        int written = 0;
        for(int i = total - 1; i >= 0 && written < take; i--)
        {
            if(!values[i].HasActualValue()) continue;
            // actual_value est en 1e6 ; pour un % comme 3.7, on lit 3700000
            out_values[take - 1 - written] = (double)values[i].actual_value / 1e6;
            out_times[take - 1 - written]  = values[i].time;
            written++;
        }
        return written;
    }

    //--- Calcule unemp_rising : variation 3 mois > 0.
    //--- Retourne true si OK, false si pas assez de data.
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
//| CMacroSourceFRED : récupère la dernière valeur d'une série FRED  |
//| via WebRequest. Par défaut : T10Y2Y (10-Year Treasury Constant   |
//| Maturity Minus 2-Year Treasury Constant Maturity).               |
//+------------------------------------------------------------------+
class CMacroSourceFRED
{
private:
    string m_series_id;     // T10Y2Y
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

    //--- Fetch la dernière observation. Retourne true et remplit `out_value`.
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
                     + "&limit=1"
                     + "&sort_order=desc";

        char post[], result[];
        string response_headers;
        ResetLastError();
        int code = WebRequest("GET", url, NULL, NULL, 5000,
                              post, 0, result, response_headers);
        if(code == -1)
        {
            PrintFormat("CMacroSourceFRED::WebRequest failed err=%d "
                        "(check URL whitelist for %s)",
                        GetLastError(), m_base_url);
            return false;
        }
        if(code != 200)
        {
            PrintFormat("CMacroSourceFRED::WebRequest HTTP %d", code);
            return false;
        }
        string body = CharArrayToString(result, 0, WHOLE_ARRAY, CP_UTF8);
        return ParseLatestObservation(body, out_value, out_obs_date);
    }

private:
    //--- Parse le JSON FRED minimaliste : on cherche la première occurrence
    //--- de "value":"X.XX" et "date":"YYYY-MM-DD" dans `body`.
    //--- Format typique :
    //---   {"observations":[{"date":"2026-04-23","value":"0.51",...}]}
    bool ParseLatestObservation(string body, double &out_value, datetime &out_date)
    {
        int idx_value = StringFind(body, "\"value\":\"");
        int idx_date  = StringFind(body, "\"date\":\"");
        if(idx_value < 0 || idx_date < 0)
        {
            Print("CMacroSourceFRED: cannot find value/date in JSON body");
            return false;
        }
        // value
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

        // date
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
