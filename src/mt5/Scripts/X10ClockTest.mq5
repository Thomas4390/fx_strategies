//+------------------------------------------------------------------+
//| X10ClockTest.mq5                                                 |
//|                                                                  |
//| Unit tests for X10Clock.mqh - the single conversion between the  |
//| broker server clock and the New York wall clock every session    |
//| rule of the x10 spec is written in (§2).                         |
//|                                                                  |
//| Covered: the four US daylight-saving switches the spec names     |
//| (2023-03-12, 2023-11-05, 2024-03-10, 2024-11-03) probed one      |
//| minute either side, the two European switches that matter when   |
//| the server clock is EET/EEST, the 17:00 session boundary, and    |
//| the 16:30 / 18:15 edges of the entry blackout.                   |
//|                                                                  |
//| Lancer manuellement (Navigator -> Scripts -> X10ClockTest).      |
//| Sortie : une ligne PASS/FAIL par cas, puis le verdict global.    |
//+------------------------------------------------------------------+
#property copyright "fx_strategies - Apogee Invest strategy 2"
#property version   "1.00"

#include "..\Include\X10Clock.mqh"

int g_pass = 0;
int g_fail = 0;

string Stamp(datetime t)
{
    MqlDateTime dt;
    TimeToStruct(t, dt);
    return StringFormat("%04d-%02d-%02d %02d:%02d",
                        dt.year, dt.mon, dt.day, dt.hour, dt.min);
}

//--- One conversion case: a server stamp and the New York wall clock
//--- it must map to.
void CheckNY(string label, int sy, int smo, int sd, int sh, int smi,
             int ey, int emo, int ed, int eh, int emi)
{
    datetime server   = X10MakeUTC(sy, smo, sd, sh, smi);
    datetime expected = X10MakeUTC(ey, emo, ed, eh, emi);
    datetime got      = X10ServerToNY(server);
    if(got == expected)
    {
        g_pass++;
        PrintFormat("PASS %s: server %s -> NY %s", label, Stamp(server),
                    Stamp(got));
    }
    else
    {
        g_fail++;
        PrintFormat("FAIL %s: server %s -> NY %s, expected %s", label,
                    Stamp(server), Stamp(got), Stamp(expected));
    }
}

void CheckBool(string label, bool got, bool expected)
{
    if(got == expected)
    {
        g_pass++;
        PrintFormat("PASS %s (%s)", label, (got ? "true" : "false"));
    }
    else
    {
        g_fail++;
        PrintFormat("FAIL %s: got %s expected %s", label,
                    (got ? "true" : "false"), (expected ? "true" : "false"));
    }
}

void CheckSession(string label, int sy, int smo, int sd, int sh, int smi,
                  int ey, int emo, int ed)
{
    datetime server = X10MakeUTC(sy, smo, sd, sh, smi);
    long got = X10SessionId(server);
    long expected = X10DaysFromCivil(ey, emo, ed);
    if(got == expected)
    {
        g_pass++;
        PrintFormat("PASS %s: server %s -> session %04d-%02d-%02d", label,
                    Stamp(server), ey, emo, ed);
    }
    else
    {
        g_fail++;
        PrintFormat("FAIL %s: server %s -> session day %I64d, expected %I64d "
                    "(%04d-%02d-%02d)", label, Stamp(server), got, expected,
                    ey, emo, ed);
    }
}

void OnStart()
{
    Print("=== X10ClockTest: server -> New York conversion (spec 2) ===");

    //================================================== X10_SERVER_UTC
    // The measured default for SquaredFinancial: the server clock has no
    // DST of its own and sits on UTC (see the header of X10Clock.mqh).
    X10ClockConfigure(X10_SERVER_UTC, 0);
    Print("--- mode X10_SERVER_UTC (server time is UTC) ---");

    // US spring forward 2023: 02:00 EST -> 03:00 EDT at 07:00 UTC.
    CheckNY("2023-03-12 before", 2023, 3, 12, 6, 59, 2023, 3, 12, 1, 59);
    CheckNY("2023-03-12 after",  2023, 3, 12, 7,  0, 2023, 3, 12, 3,  0);
    // US fall back 2023: 02:00 EDT -> 01:00 EST at 06:00 UTC.
    CheckNY("2023-11-05 before", 2023, 11, 5, 5, 59, 2023, 11, 5, 1, 59);
    CheckNY("2023-11-05 after",  2023, 11, 5, 6,  0, 2023, 11, 5, 1,  0);
    // US spring forward 2024.
    CheckNY("2024-03-10 before", 2024, 3, 10, 6, 59, 2024, 3, 10, 1, 59);
    CheckNY("2024-03-10 after",  2024, 3, 10, 7,  0, 2024, 3, 10, 3,  0);
    // US fall back 2024.
    CheckNY("2024-11-03 before", 2024, 11, 3, 5, 59, 2024, 11, 3, 1, 59);
    CheckNY("2024-11-03 after",  2024, 11, 3, 6,  0, 2024, 11, 3, 1,  0);

    // The European switches must NOT move a UTC server clock: the same
    // server stamp keeps mapping to the same New York hour across them.
    CheckNY("2024-03-30 (EU eve)",  2024, 3, 30, 12, 0, 2024, 3, 30, 8, 0);
    CheckNY("2024-04-01 (EU done)", 2024, 4,  1, 12, 0, 2024, 4,  1, 8, 0);

    //--- Session boundary of §2: (17:00 of J-1, 17:00 of J].
    CheckSession("session close (17:00 NY)", 2024, 1, 15, 22,  0, 2024, 1, 15);
    CheckSession("session reopen (17:01 NY)", 2024, 1, 15, 22, 1, 2024, 1, 16);
    CheckSession("session 18:00 NY", 2024, 1, 15, 23, 0, 2024, 1, 16);
    // Same two probes in summer, one hour earlier in UTC.
    CheckSession("session close, DST", 2024, 7, 15, 21,  0, 2024, 7, 15);
    CheckSession("session reopen, DST", 2024, 7, 15, 21, 1, 2024, 7, 16);

    //--- Entry blackout of §10: [16:30, 18:15) New York.
    CheckBool("16:29 NY not blocked",
              X10IsEntryBlocked(X10MakeUTC(2024, 1, 15, 21, 29)), false);
    CheckBool("16:30 NY blocked",
              X10IsEntryBlocked(X10MakeUTC(2024, 1, 15, 21, 30)), true);
    CheckBool("18:14 NY blocked",
              X10IsEntryBlocked(X10MakeUTC(2024, 1, 15, 23, 14)), true);
    CheckBool("18:15 NY not blocked",
              X10IsEntryBlocked(X10MakeUTC(2024, 1, 15, 23, 15)), false);

    //================================================ X10_SERVER_EUROPE
    // Only used if the broker clock turns out to follow European DST.
    X10ClockConfigure(X10_SERVER_EUROPE, 0);
    Print("--- mode X10_SERVER_EUROPE (EET winter, EEST summer) ---");

    // Winter both sides: server UTC+2, New York UTC-5 -> NY = server - 7h.
    CheckNY("EU winter", 2024, 1, 15, 12, 0, 2024, 1, 15, 5, 0);
    // US on DST, EU not yet (10-31 March): server UTC+2, NY UTC-4 -> -6h.
    CheckNY("US DST only", 2024, 3, 25, 12, 0, 2024, 3, 25, 6, 0);
    // Both on DST: server UTC+3, NY UTC-4 -> -7h.
    CheckNY("EU DST", 2024, 4, 15, 12, 0, 2024, 4, 15, 5, 0);
    // EU spring forward 2024-03-31 01:00 UTC = 03:00 server. The New York
    // side is already on DST (since 10 March), so it reads UTC-4 on both
    // sides and the switch shows up as a one-hour hole in UTC.
    CheckNY("2024-03-31 before", 2024, 3, 31, 2, 59, 2024, 3, 30, 20, 59);
    CheckNY("2024-03-31 after",  2024, 3, 31, 4,  0, 2024, 3, 30, 21,  0);
    // EU fall back 2024-10-27 01:00 UTC: 04:00 EEST becomes 03:00 EET, so
    // server stamps 03:00-03:59 happen twice. X10ServerToUTC resolves the
    // fold to standard time (the second pass), which is what this case
    // pins; the hour sits on a Saturday evening in New York, market shut.
    CheckNY("2024-10-27 fold (EET)", 2024, 10, 27, 3, 59, 2024, 10, 26, 21, 59);
    CheckNY("2024-10-27 after",  2024, 10, 27, 4,  0, 2024, 10, 26, 22,  0);
    // US fall back while EU already back: both standard -> -7h.
    CheckNY("2024-11-03 EU mode", 2024, 11, 3, 12, 0, 2024, 11, 3, 5, 0);

    //================================================= X10_SERVER_FIXED
    X10ClockConfigure(X10_SERVER_FIXED, 120);
    Print("--- mode X10_SERVER_FIXED (offset 120 min) ---");
    CheckNY("fixed +120 winter", 2024, 1, 15, 12, 0, 2024, 1, 15, 5, 0);
    CheckNY("fixed +120 summer", 2024, 7, 15, 12, 0, 2024, 7, 15, 6, 0);

    //--- Restore the default so a following script sees the real config.
    X10ClockConfigure(X10_SERVER_UTC, 0);

    PrintFormat("=== X10ClockTest: %d passed, %d failed -> %s ===",
                g_pass, g_fail, (g_fail == 0 ? "PASS" : "FAIL"));
}
