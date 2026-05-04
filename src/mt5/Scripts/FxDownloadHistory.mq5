//+------------------------------------------------------------------+
//| FxDownloadHistory.mq5                                            |
//| Force le téléchargement de l'historique broker pour les bars M1, |
//| D1 (et autres TF) sur la fenêtre demandée. À lancer AVANT        |
//| FxExportRates si CopyRates renvoie moins de bars qu'attendu.     |
//|                                                                  |
//| MT5 ne pré-cache pas les bars synthétisés depuis les ticks       |
//| (utilisés par le tester) — il faut forcer le download via        |
//| CopyRates en boucle avec Sleep pour laisser le broker pousser.   |
//+------------------------------------------------------------------+
#property copyright "fx_strategies — Apogee Invest"
#property version   "1.00"
#property strict
#property script_show_inputs

#include "..\Include\FxCommon.mqh"

input string  Inp_SymbolSuffix = ".c";
input string  Inp_SymbolsCSV   = "EURUSD,GBPUSD,USDJPY,USDCAD";
input string  Inp_PeriodsCSV   = "M1,D1";
input datetime Inp_FromDate    = D'2019.01.01 00:00';
input datetime Inp_ToDate      = D'2026.05.01 00:00';
input int     Inp_MaxRetries   = 60;     // 60 × 2s = 2 min max par (sym, tf)
input int     Inp_SleepMs      = 2000;   // sleep entre retries
input int     Inp_StableRetries = 3;     // arrêter dès que N retries consécutifs n'ajoutent rien

void OnStart()
{
    Print("=== FxDownloadHistory start ===");

    string syms[];
    int n_syms = StringSplit(Inp_SymbolsCSV, ',', syms);
    string tf_labels[];
    int n_tfs = StringSplit(Inp_PeriodsCSV, ',', tf_labels);

    PrintFormat("Symbols=%d TF=%d, Window: %s → %s, MaxRetries=%d Sleep=%dms",
                n_syms, n_tfs,
                TimeToString(Inp_FromDate, TIME_DATE),
                TimeToString(Inp_ToDate, TIME_DATE),
                Inp_MaxRetries, Inp_SleepMs);

    int n_ok = 0, n_fail = 0;
    for(int i = 0; i < n_syms; i++)
    {
        string sym = MakeSymbolWithSuffix(syms[i], Inp_SymbolSuffix);
        if(!EnsureSymbolSelected(sym))
        {
            PrintFormat("FAIL %s: cannot add to MarketWatch", sym);
            n_fail++;
            continue;
        }

        for(int j = 0; j < n_tfs; j++)
        {
            ENUM_TIMEFRAMES tf = TimeframeFromLabel(tf_labels[j]);
            if(tf == PERIOD_CURRENT && tf_labels[j] != "CURRENT")
            {
                PrintFormat("FAIL %s/%s: unknown TF label", sym, tf_labels[j]);
                n_fail++;
                continue;
            }
            if(WaitForHistory(sym, tf, tf_labels[j]))
                n_ok++;
            else
                n_fail++;
        }
    }

    PrintFormat("=== Done: %d OK, %d FAILED ===", n_ok, n_fail);
}

ENUM_TIMEFRAMES TimeframeFromLabel(string label)
{
    StringTrimLeft(label); StringTrimRight(label);
    if(label == "M1")  return PERIOD_M1;
    if(label == "M5")  return PERIOD_M5;
    if(label == "M15") return PERIOD_M15;
    if(label == "M30") return PERIOD_M30;
    if(label == "H1")  return PERIOD_H1;
    if(label == "H4")  return PERIOD_H4;
    if(label == "D1")  return PERIOD_D1;
    if(label == "W1")  return PERIOD_W1;
    if(label == "MN1") return PERIOD_MN1;
    return PERIOD_CURRENT;
}

//+------------------------------------------------------------------+
//| Boucle CopyRates jusqu'à ce que le compte se stabilise.           |
//+------------------------------------------------------------------+
bool WaitForHistory(string sym, ENUM_TIMEFRAMES tf, string tf_label)
{
    MqlRates rates[];
    int last_count = -1;
    int stable_streak = 0;
    int total_attempts = 0;

    PrintFormat("  [%s/%s] downloading...", sym, tf_label);

    while(total_attempts < Inp_MaxRetries)
    {
        int copied = CopyRates(sym, tf, Inp_FromDate, Inp_ToDate, rates);

        if(copied < 0)
        {
            // Force a tick to wake up the loader
            MqlTick tick;
            SymbolInfoTick(sym, tick);
            Sleep(Inp_SleepMs);
            total_attempts++;
            continue;
        }

        if(copied > last_count)
        {
            last_count = copied;
            stable_streak = 0;
            if(copied > 0)
            {
                datetime first_t = rates[0].time;
                datetime last_t  = rates[copied-1].time;
                PrintFormat("    %s/%s: %d bars [%s ... %s]",
                            sym, tf_label, copied,
                            TimeToString(first_t, TIME_DATE),
                            TimeToString(last_t, TIME_DATE));
            }
        }
        else
        {
            stable_streak++;
            if(stable_streak >= Inp_StableRetries)
            {
                PrintFormat("OK   %s/%s: stable at %d bars after %d attempts",
                            sym, tf_label, last_count, total_attempts);
                return last_count > 0;
            }
        }

        Sleep(Inp_SleepMs);
        total_attempts++;
    }

    PrintFormat("WARN %s/%s: timeout after %d attempts, final=%d bars",
                sym, tf_label, total_attempts, last_count);
    return last_count > 0;
}
