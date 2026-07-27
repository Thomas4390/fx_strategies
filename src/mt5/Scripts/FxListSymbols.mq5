//+------------------------------------------------------------------+
//| FxListSymbols.mq5                                                |
//| Catalogue tous les symboles offerts par le broker dans un CSV    |
//| `MQL5/Files/symbols_catalog.csv`, consommé par la phase de       |
//| découverte d'instruments (docs/research — expansion momentum).   |
//|                                                                  |
//| Usage : drag-and-drop sur n'importe quel chart, ou auto-start    |
//| via un INI [StartUp] (pattern scripts/investigations/            |
//| download_history.py). Ne modifie rien : lecture seule du         |
//| catalogue broker, MarketWatch restauré tel quel.                 |
//|                                                                  |
//| Format CSV (ASCII, comma-separated, 1 row header) :              |
//|   name,path,digits,point,trade_mode,calc_mode,contract_size,     |
//|   currency_base,currency_profit,spread_current,description       |
//+------------------------------------------------------------------+
#property copyright "fx_strategies port"
#property version   "1.00"
#property strict

void OnStart()
{
    Print("=== FxListSymbols start ===");

    const string out_name = "symbols_catalog.csv";
    int fh = FileOpen(out_name, FILE_WRITE | FILE_ANSI | FILE_TXT);
    if(fh == INVALID_HANDLE)
    {
        PrintFormat("FAIL: cannot open Files\\%s (err=%d)", out_name, GetLastError());
        return;
    }

    FileWriteString(fh, "name,path,digits,point,trade_mode,calc_mode,"
                        "contract_size,currency_base,currency_profit,"
                        "spread_current,description\r\n");

    int total = SymbolsTotal(false);   // false = tout le catalogue broker
    int n_written = 0;
    for(int i = 0; i < total; i++)
    {
        string name = SymbolName(i, false);
        // SymbolInfo* sans SymbolSelect : les propriétés statiques du
        // catalogue sont lisibles sans ajouter le symbole au MarketWatch.
        string path  = SymbolInfoString(name, SYMBOL_PATH);
        string descr = SymbolInfoString(name, SYMBOL_DESCRIPTION);
        StringReplace(path, ",", ";");
        StringReplace(descr, ",", ";");

        long digits     = SymbolInfoInteger(name, SYMBOL_DIGITS);
        double point    = SymbolInfoDouble(name, SYMBOL_POINT);
        long trade_mode = SymbolInfoInteger(name, SYMBOL_TRADE_MODE);
        long calc_mode  = SymbolInfoInteger(name, SYMBOL_TRADE_CALC_MODE);
        double csize    = SymbolInfoDouble(name, SYMBOL_TRADE_CONTRACT_SIZE);
        string cur_base = SymbolInfoString(name, SYMBOL_CURRENCY_BASE);
        string cur_prof = SymbolInfoString(name, SYMBOL_CURRENCY_PROFIT);
        long spread     = SymbolInfoInteger(name, SYMBOL_SPREAD);

        FileWriteString(fh, StringFormat("%s,%s,%d,%.8f,%d,%d,%.2f,%s,%s,%d,%s\r\n",
                        name, path, (int)digits, point, (int)trade_mode,
                        (int)calc_mode, csize, cur_base, cur_prof,
                        (int)spread, descr));
        n_written++;
    }

    FileClose(fh);
    PrintFormat("Wrote %d symbols to Files\\%s", n_written, out_name);
    Print("=== FxListSymbols done ===");
}
