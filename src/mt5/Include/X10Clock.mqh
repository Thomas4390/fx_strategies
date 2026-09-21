//+------------------------------------------------------------------+
//| X10Clock.mqh                                                     |
//|                                                                  |
//| Every session rule of the XAUUSD x10 strategy is expressed in    |
//| New York wall-clock time (docs/specs/xau_x10_spec.md §2): the    |
//| 18:00 -> 17:00 session, the 16:55 forced flat, the 16:30-18:15   |
//| entry blackout. MT5 hands out broker SERVER time, so exactly one |
//| conversion stands between the tester and the spec, and it lives  |
//| here - alone, pure, and unit-tested by Scripts/X10ClockTest.mq5. |
//|                                                                  |
//| SERVER -> UTC, the assumption. The repo does NOT document a      |
//| server->New York convention for SquaredFinancial:                |
//| docs/specs/gold_momentum_spec.md §2 says the D1 bound is taken   |
//| in server time and that the gap is to be MEASURED, and           |
//| docs/mt5/07_history_timeseries.md:155 only carries the generic   |
//| "brokers are often GMT+2/GMT+3". So the offset was measured on   |
//| the broker's own exported bars (data/XAG-USD_minute_mt5.parquet, |
//| metals, same session as gold, raw server stamps):                |
//|                                                                  |
//|   2026-03-02..06 (US on EST) : last bar 21:58, reopen 23:05      |
//|   2026-03-09..27 (US on EDT) : last bar 20:58, reopen 23:05      |
//|   2026-03-30..    (EU on DST): last bar 20:58 -> unchanged       |
//|                                                                  |
//| The daily close tracks 16:58 New York across the US switch and   |
//| does NOT move on the European switch: the server clock carries   |
//| no DST of its own and sits on UTC. Hence the default mode,       |
//| X10_SERVER_UTC. The two other modes exist so the assumption can  |
//| be overridden from the tester inputs without recompiling, on a   |
//| broker whose clock is European (EET/EEST) or a plain fixed       |
//| offset - the conversion is a measurement, not a law.             |
//|                                                                  |
//| UTC -> New York uses the US rules in force since 2007: DST from  |
//| the 2nd Sunday of March 02:00 local standard (07:00 UTC) to the  |
//| 1st Sunday of November 02:00 local daylight (06:00 UTC).         |
//+------------------------------------------------------------------+
#ifndef __X10_CLOCK_MQH__
#define __X10_CLOCK_MQH__

//--- How the broker server clock relates to UTC.
enum EX10ServerClock
{
    X10_SERVER_UTC    = 0,  // server time IS UTC (measured on this broker)
    X10_SERVER_EUROPE = 1,  // server is EET/EEST (UTC+2 winter, UTC+3 summer)
    X10_SERVER_FIXED  = 2   // server = UTC + Inp_ServerToNYOffsetMin minutes
};

//--- Spec §2 / §10, in minutes since New York midnight.
#define X10_SESSION_CLOSE_HOUR   17
#define X10_ENTRY_BLOCK_START    (16 * 60 + 30)
#define X10_ENTRY_BLOCK_END      (18 * 60 + 15)
#define X10_SESSION_FORCE_MINUTE (16 * 60 + 55)

#define X10_NY_STD_OFFSET (-5 * 3600)
#define X10_NY_DST_OFFSET (-4 * 3600)

//--- Module configuration, set once from OnInit.
EX10ServerClock g_x10_clock_mode       = X10_SERVER_UTC;
int             g_x10_clock_offset_min = 0;

void X10ClockConfigure(EX10ServerClock mode, int offset_minutes)
{
    g_x10_clock_mode       = mode;
    g_x10_clock_offset_min = offset_minutes;
}

//+------------------------------------------------------------------+
//| Calendar primitives - pure integer arithmetic, no TimeCurrent(), |
//| no terminal timezone, so the script can test them offline.       |
//| days_from_civil of Howard Hinnant; day 0 = 1970-01-01 (Thursday).|
//+------------------------------------------------------------------+
long X10DaysFromCivil(int y, int m, int d)
{
    y -= (m <= 2) ? 1 : 0;
    long era = (long)((y >= 0 ? y : y - 399) / 400);
    long yoe = (long)y - era * 400;
    long doy = (long)((153 * (m + (m > 2 ? -3 : 9)) + 2) / 5 + d - 1);
    long doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    return era * 146097 + doe - 719468;
}

datetime X10MakeUTC(int y, int mo, int d, int h, int mi, int s = 0)
{
    return (datetime)(X10DaysFromCivil(y, mo, d) * 86400 + h * 3600 + mi * 60 + s);
}

//--- 0 = Sunday .. 6 = Saturday, for an epoch day count.
int X10DayOfWeek(long epoch_days)
{
    return (int)(((epoch_days + 4) % 7 + 7) % 7);
}

int X10DaysInMonth(int y, int mo)
{
    int len[] = {31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
    if(mo == 2 && ((y % 4 == 0 && y % 100 != 0) || y % 400 == 0)) return 29;
    return len[mo - 1];
}

datetime X10NthSundayUTC(int y, int mo, int nth, int hour_utc)
{
    long first = X10DaysFromCivil(y, mo, 1);
    long sunday = first + ((7 - X10DayOfWeek(first)) % 7) + 7 * (nth - 1);
    return (datetime)(sunday * 86400 + hour_utc * 3600);
}

datetime X10LastSundayUTC(int y, int mo, int hour_utc)
{
    long last = X10DaysFromCivil(y, mo, X10DaysInMonth(y, mo));
    long sunday = last - X10DayOfWeek(last);
    return (datetime)(sunday * 86400 + hour_utc * 3600);
}

//--- US daylight saving, 2007 rules. Boundaries expressed in UTC so the
//--- test is a plain comparison and never re-enters the conversion.
bool X10IsUSDstUTC(datetime utc)
{
    MqlDateTime dt;
    TimeToStruct(utc, dt);
    datetime start = X10NthSundayUTC(dt.year, 3, 2, 7);
    datetime end   = X10NthSundayUTC(dt.year, 11, 1, 6);
    return (utc >= start && utc < end);
}

//--- European summer time: last Sunday of March 01:00 UTC to last
//--- Sunday of October 01:00 UTC. Only used by X10_SERVER_EUROPE.
bool X10IsEUDstUTC(datetime utc)
{
    MqlDateTime dt;
    TimeToStruct(utc, dt);
    datetime start = X10LastSundayUTC(dt.year, 3, 1);
    datetime end   = X10LastSundayUTC(dt.year, 10, 1);
    return (utc >= start && utc < end);
}

//+------------------------------------------------------------------+
//| SERVER -> UTC -> New York. The two steps are kept separate       |
//| because the trace is stamped in UTC (§13) and the session logic  |
//| in New York (§2): one conversion, two consumers.                 |
//+------------------------------------------------------------------+
datetime X10ServerToUTC(datetime server)
{
    if(g_x10_clock_mode == X10_SERVER_FIXED)
        return (datetime)((long)server - (long)g_x10_clock_offset_min * 60);
    if(g_x10_clock_mode == X10_SERVER_EUROPE)
    {
        // The DST test needs UTC, which is what we are computing: probe with
        // the winter offset. The one ambiguous hour per year sits inside the
        // 01:00-02:00 UTC window, where gold does not trade.
        datetime probe = (datetime)((long)server - 2 * 3600);
        return (datetime)((long)server - (X10IsEUDstUTC(probe) ? 3 : 2) * 3600);
    }
    return server;
}

datetime X10UTCToNY(datetime utc)
{
    return (datetime)((long)utc + (X10IsUSDstUTC(utc) ? X10_NY_DST_OFFSET
                                                      : X10_NY_STD_OFFSET));
}

datetime X10ServerToNY(datetime server)
{
    return X10UTCToNY(X10ServerToUTC(server));
}

//--- Minutes since New York midnight, the unit §10 blocks and forces on.
int X10NYMinuteOfDay(datetime server)
{
    MqlDateTime dt;
    TimeToStruct(X10ServerToNY(server), dt);
    return dt.hour * 60 + dt.min;
}

//+------------------------------------------------------------------+
//| Session label of a stamp, under the 17:00 New York close (§2).   |
//| Same convention as x10_context.session_dates: the interval is    |
//| closed on the right, (17:00 of J-1, 17:00 of J], so 17:00 closes |
//| session J and the 18:00 reopen starts J+1. Returned as days      |
//| since epoch, which is what the Python kernels compare too.       |
//+------------------------------------------------------------------+
long X10SessionId(datetime server)
{
    long ny = (long)X10ServerToNY(server);
    return (ny + (24 - X10_SESSION_CLOSE_HOUR) * 3600 - 1) / 86400;
}

//--- §10: no entry in [16:30, 18:15) New York. Takes the OPEN of the
//--- bar the fill would happen on, never the decision bar.
bool X10IsEntryBlocked(datetime fill_bar_open_server)
{
    int minute = X10NYMinuteOfDay(fill_bar_open_server);
    return (minute >= X10_ENTRY_BLOCK_START && minute < X10_ENTRY_BLOCK_END);
}

//--- §13: "YYYY-MM-DD HH:MM:SS", UTC, the trace stamp format.
string X10FormatUTC(datetime utc)
{
    MqlDateTime dt;
    TimeToStruct(utc, dt);
    return StringFormat("%04d-%02d-%02d %02d:%02d:%02d",
                        dt.year, dt.mon, dt.day, dt.hour, dt.min, dt.sec);
}

#endif // __X10_CLOCK_MQH__
