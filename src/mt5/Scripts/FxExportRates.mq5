//+------------------------------------------------------------------+
//| FxExportRates.mq5                                                |
//| Exporte l'historique OHLCV de N symboles × N timeframes en CSV   |
//| dans `MQL5/Files/exports/`, ensuite consommé par                 |
//| `bridge/import_mt5_rates.py` qui convertit en Parquet            |
//| dans `data/<PAIR-DASH>_<period>_mt5_<broker>.parquet`.           |
//|                                                                  |
//| Usage : drag-and-drop sur n'importe quel chart. Avant export,    |
//| force le téléchargement via `CopyRates` sur la fenêtre           |
//| demandée. Si le broker n'a pas l'historique demandé, exporte ce  |
//| qu'il a et logge le manque dans le journal.                      |
//|                                                                  |
//| Format CSV (ASCII, comma-separated, 1 row header) :              |
//|   time,open,high,low,close,tick_volume,spread,real_volume        |
//|   2024-06-03T00:00:00Z,1.08510,1.08620,1.08490,1.08580,1234,1,0  |
//+------------------------------------------------------------------+
#property copyright "fx_strategies port"
#property version   "1.00"
#property script_show_inputs
#property strict

#include "..\Include\FxCommon.mqh"

input string  Inp_SymbolSuffix = ".c";
input string  Inp_SymbolsCSV   = "EURUSD,GBPUSD,USDJPY,USDCAD,USDCHF,AUDUSD,NZDUSD,EURGBP,EURJPY,GBPJPY";
input string  Inp_PeriodsCSV   = "M1,D1";
input datetime Inp_FromDate    = D'2019.01.01 00:00';                  // Inclusive start
input datetime Inp_ToDate      = D'2026.05.01 00:00';                  // Exclusive end (use future date to grab all)
input string  Inp_OutputDir    = "exports";                            // Subdir of MQL5\Files\
input bool    Inp_OverwriteCSV = true;                                 // Overwrite existing CSVs

void OnStart()
{
    Print("=== FxExportRates start ===");

    string syms[];
    int n_syms = StringSplit(Inp_SymbolsCSV, ',', syms);
    string tf_labels[];
    int n_tfs = StringSplit(Inp_PeriodsCSV, ',', tf_labels);

    PrintFormat("Symbols=%d, Timeframes=%d, From=%s To=%s, OutDir=Files\\%s",
                n_syms, n_tfs, TimeToString(Inp_FromDate, TIME_DATE),
                TimeToString(Inp_ToDate, TIME_DATE), Inp_OutputDir);

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
                PrintFormat("FAIL %s/%s: unknown timeframe label", sym, tf_labels[j]);
                n_fail++;
                continue;
            }
            if(ExportSymbolTF(sym, tf, tf_labels[j]))
                n_ok++;
            else
                n_fail++;
        }
    }

    PrintFormat("=== Done: %d OK, %d FAILED ===", n_ok, n_fail);
}

//+------------------------------------------------------------------+
//| Map TF label like "M1", "H1", "D1" → ENUM_TIMEFRAMES.            |
//+------------------------------------------------------------------+
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
//| Pull all bars in [from,to] for (sym, tf) and write to a CSV.     |
//+------------------------------------------------------------------+
bool ExportSymbolTF(string sym, ENUM_TIMEFRAMES tf, string tf_label)
{
    MqlRates rates[];
    int copied = CopyRates(sym, tf, Inp_FromDate, Inp_ToDate, rates);
    if(copied <= 0)
    {
        PrintFormat("FAIL %s/%s: CopyRates returned %d (err=%d). "
                    "History likely not downloaded — open the chart manually first.",
                    sym, tf_label, copied, GetLastError());
        return false;
    }

    string fname = Inp_OutputDir + "\\" + sym + "_" + tf_label + ".csv";
    int flags = FILE_WRITE | FILE_CSV | FILE_ANSI;
    if(!Inp_OverwriteCSV)
        flags |= FILE_READ;

    int h = FileOpen(fname, flags, ',');
    if(h == INVALID_HANDLE)
    {
        PrintFormat("FAIL %s/%s: FileOpen %s err=%d",
                    sym, tf_label, fname, GetLastError());
        return false;
    }

    FileWrite(h, "time", "open", "high", "low", "close",
              "tick_volume", "spread", "real_volume");

    for(int k = 0; k < copied; k++)
    {
        string ts = TimeToString(rates[k].time, TIME_DATE | TIME_SECONDS);
        StringReplace(ts, ".", "-");
        StringReplace(ts, " ", "T");
        ts = ts + "Z";
        FileWrite(h, ts,
                  DoubleToString(rates[k].open, _Digits),
                  DoubleToString(rates[k].high, _Digits),
                  DoubleToString(rates[k].low, _Digits),
                  DoubleToString(rates[k].close, _Digits),
                  rates[k].tick_volume,
                  rates[k].spread,
                  rates[k].real_volume);
    }
    FileClose(h);

    PrintFormat("OK   %s/%s: %d bars → Files\\%s [%s ... %s]",
                sym, tf_label, copied, fname,
                TimeToString(rates[0].time, TIME_DATE | TIME_SECONDS),
                TimeToString(rates[copied-1].time, TIME_DATE | TIME_SECONDS));
    return true;
}
