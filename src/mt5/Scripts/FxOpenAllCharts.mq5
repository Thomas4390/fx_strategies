//+------------------------------------------------------------------+
//| FxOpenAllCharts.mq5                                              |
//| Ouvre les 4 charts FX traités par FxMultiSleeve en 1 clic        |
//| (EURUSD, GBPUSD, USDJPY, USDCAD avec suffixe broker).            |
//|                                                                  |
//| Usage : drag-and-drop sur n'importe quel chart. Le script ouvre  |
//| les 4 charts manquants au timeframe choisi, puis suggère         |
//| `Window -> Tile Vertically` pour les arranger en grille 2x2.     |
//|                                                                  |
//| Pour sauvegarder la disposition en 1 clic réutilisable :         |
//| File -> Profiles -> Save Profile As -> "FxMultiSleeve_4pairs"    |
//+------------------------------------------------------------------+
#property copyright "fx_strategies port"
#property version   "1.00"
#property script_show_inputs

#include "..\Include\FxCommon.mqh"

input string           Inp_SymbolSuffix = ".c";       // Broker suffix (matches FxMultiSleeve default)
input ENUM_TIMEFRAMES  Inp_Timeframe    = PERIOD_M1;  // M1 (Sleeve 1) or PERIOD_D1 (Sleeves 2/3)
input bool             Inp_SkipIfOpen   = true;       // Skip pairs that already have a chart

void OnStart()
{
    Print("=== FxOpenAllCharts start ===");

    string pairs[] = {"EURUSD", "GBPUSD", "USDJPY", "USDCAD"};
    int n_opened = 0, n_skipped = 0, n_failed = 0;

    for(int i = 0; i < ArraySize(pairs); i++)
    {
        string sym = MakeSymbolWithSuffix(pairs[i], Inp_SymbolSuffix);

        // Add to MarketWatch first (chart open requires it)
        if(!EnsureSymbolSelected(sym))
        {
            PrintFormat("FAIL: cannot add %s to MarketWatch (check broker suffix)", sym);
            n_failed++;
            continue;
        }

        // Skip if a chart for this symbol+timeframe is already open
        if(Inp_SkipIfOpen && IsChartOpenForSymbol(sym, Inp_Timeframe))
        {
            PrintFormat("SKIP: chart already open for %s %s",
                        sym, EnumToString(Inp_Timeframe));
            n_skipped++;
            continue;
        }

        long chart_id = ChartOpen(sym, Inp_Timeframe);
        if(chart_id == 0)
        {
            PrintFormat("FAIL: ChartOpen returned 0 for %s (err=%d)",
                        sym, GetLastError());
            n_failed++;
            continue;
        }
        PrintFormat("OK: opened %s on %s (chart_id=%I64d)",
                    sym, EnumToString(Inp_Timeframe), chart_id);
        n_opened++;
    }

    PrintFormat("=== Done: %d opened, %d skipped, %d failed ===",
                n_opened, n_skipped, n_failed);
    if(n_opened > 0)
        Print("Tip: now use Window -> Tile Vertically (or Cascade) to arrange them.");
}

//+------------------------------------------------------------------+
//| Helper : vérifie si un chart est déjà ouvert pour ce             |
//| (symbole, timeframe). Itère sur tous les chart IDs ouverts.      |
//+------------------------------------------------------------------+
bool IsChartOpenForSymbol(string symbol, ENUM_TIMEFRAMES tf)
{
    long id = ChartFirst();
    while(id >= 0)
    {
        if(ChartSymbol(id) == symbol && ChartPeriod(id) == tf)
            return true;
        id = ChartNext(id);
    }
    return false;
}
