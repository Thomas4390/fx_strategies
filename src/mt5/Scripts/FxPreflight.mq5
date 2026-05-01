//+------------------------------------------------------------------+
//| FxPreflight.mq5                                                  |
//| Vérifie l'environnement avant de lancer FxMultiSleeve :          |
//|   - les 4 paires sont dans MarketWatch                           |
//|   - history M1 EURUSD ≥ 1500 bars                                |
//|   - history D1 ≥ 250 bars sur chaque paire                       |
//|   - macro_cache.csv accessible et frais (< 24h)                  |
//+------------------------------------------------------------------+
#property copyright "fx_strategies port"
#property version   "1.00"
#property script_show_inputs

#include "..\Include\FxCommon.mqh"
#include "..\Include\FxMacroFilter.mqh"

input string           Inp_SymbolSuffix          = ".c";              // Broker-specific (ECN/Raw uses ".c")
input EMacroSourceMode Inp_MacroSourceMode       = MACRO_SOURCE_AUTO; // tester->HISTORY, live->NATIVE
input string           Inp_MacroCacheFile        = "macro_cache.csv";
input bool             Inp_MacroUseCommon        = true;
input int              Inp_MacroMaxAgeHours      = 24;
input double           Inp_MR_SpreadThresh       = 0.5;
input string           Inp_FREDApiKeyFile        = "fred_api_key.txt";
input bool             Inp_FREDKeyUseCommon      = true;
input string           Inp_FREDSeriesId          = "T10Y2Y";
input string           Inp_MacroHistoryFile      = "macro_history.csv";
input bool             Inp_MacroHistoryUseCommon = true;

void OnStart()
{
    Print("=== FxPreflight start ===");
    bool all_ok = true;

    // 1. Symboles
    string pairs[] = {"EURUSD", "GBPUSD", "USDJPY", "USDCAD"};
    for(int i = 0; i < ArraySize(pairs); i++)
    {
        string sym = MakeSymbolWithSuffix(pairs[i], Inp_SymbolSuffix);
        if(!EnsureSymbolSelected(sym))
        {
            PrintFormat("FAIL symbol %s not selectable", sym);
            all_ok = false;
            continue;
        }
        PrintFormat("PASS %s in MarketWatch", sym);
    }

    // 2. History M1 EURUSD
    {
        string sym = MakeSymbolWithSuffix("EURUSD", Inp_SymbolSuffix);
        if(EnsureHistory(sym, PERIOD_M1, 1500))
            PrintFormat("PASS %s M1 history (≥1500 bars)", sym);
        else { PrintFormat("FAIL %s M1 history", sym); all_ok = false; }
    }

    // 3. History D1 sur 4 paires
    for(int i = 0; i < ArraySize(pairs); i++)
    {
        string sym = MakeSymbolWithSuffix(pairs[i], Inp_SymbolSuffix);
        if(EnsureHistory(sym, PERIOD_D1, 250))
            PrintFormat("PASS %s D1 history (≥250 bars)", sym);
        else { PrintFormat("FAIL %s D1 history", sym); all_ok = false; }
    }

    // 4. Macro cache
    {
        CMacroFilter macro;
        macro.Init(Inp_MacroSourceMode,
                   Inp_MacroCacheFile, Inp_MacroMaxAgeHours, Inp_MacroUseCommon,
                   Inp_MR_SpreadThresh,
                   Inp_FREDApiKeyFile, Inp_FREDKeyUseCommon,
                   Inp_MacroHistoryFile, Inp_MacroHistoryUseCommon,
                   Inp_FREDSeriesId);
        if(!macro.Refresh())
        {
            PrintFormat("FAIL macro refresh (mode=%s)",
                        EnumToString(Inp_MacroSourceMode));
            all_ok = false;
        }
        else if(!macro.IsValid())
        {
            PrintFormat("WARN macro stale: age=%ds source=%s",
                        macro.AgeSeconds(), macro.LastSource());
            all_ok = false;
        }
        else
        {
            PrintFormat("PASS macro source=%s spread=%.4f unemp_rising=%d macro_ok=%d age=%ds",
                        macro.LastSource(), macro.Spread(),
                        (int)macro.UnempRising(), (int)macro.MacroOk(),
                        macro.AgeSeconds());
        }
    }

    if(all_ok) Print("=== ALL PREFLIGHT CHECKS PASSED ===");
    else       Print("=== SOME CHECKS FAILED — DO NOT DEPLOY ===");
}
