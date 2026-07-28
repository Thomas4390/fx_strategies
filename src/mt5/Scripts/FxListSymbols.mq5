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
//| 2026-07-28 — trois familles de champs ajoutées. Elles décident    |
//| d'un candidat AVANT tout backtest, donc pour zéro essai :        |
//|   * SWAPS : le portage réellement payé par le compte. La thèse    |
//|     du dossier est qu'un moteur long-only lent doit préférer les  |
//|     instruments payés pour attendre — sans ces champs, c'est une  |
//|     histoire racontée après coup, avec eux c'est un gate ex ante. |
//|   * VOLUME : le plafond de lots. C'est lui qui a rendu les runs   |
//|     indices ininterprétables (le cap neutralise le vol-targeting) |
//|     et JPN225 non exécutable. Le lire évite de le redécouvrir     |
//|     par un backtest à notionnel nul.                             |
//|   * MARGE / STOPS : contraintes d'exécution.                     |
//|                                                                  |
//| Le CSV doit être archivé dans data/broker/ : la sortie de ce     |
//| script ne vivait que dans MQL5/Files, donc le catalogue broker    |
//| n'existait nulle part dans le dépôt.                             |
//+------------------------------------------------------------------+
#property copyright "fx_strategies port"
#property version   "1.10"
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
                        "spread_current,swap_long,swap_short,swap_mode,"
                        "swap_rollover3days,volume_min,volume_max,volume_step,"
                        "tick_value,tick_size,stops_level,margin_initial,"
                        "description\r\n");

    int total = SymbolsTotal(false);   // false = tout le catalogue broker
    int n_written = 0;
    for(int i = 0; i < total; i++)
    {
        string name = SymbolName(i, false);

        // Certaines propriétés — spread courant, tick value — sont DYNAMIQUES
        // et valent 0 tant que le symbole n'est pas dans le MarketWatch. Le
        // premier export les lisait sans sélectionner : 254 spreads sur 272 en
        // sont sortis à zéro, et un gate bâti dessus ne voyait rien. On
        // sélectionne donc temporairement, puis on restaure l'état initial
        // pour tenir la promesse de lecture seule.
        bool was_selected = SymbolInfoInteger(name, SYMBOL_SELECT);
        if(!was_selected) SymbolSelect(name, true);

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

        //--- Portage. swap_mode dit dans quelle unité swap_long/short sont
        //--- exprimés (points, devise, pourcentage annuel...) : le signe est
        //--- interprétable sans lui, l'amplitude non.
        double swap_long  = SymbolInfoDouble(name, SYMBOL_SWAP_LONG);
        double swap_short = SymbolInfoDouble(name, SYMBOL_SWAP_SHORT);
        long swap_mode    = SymbolInfoInteger(name, SYMBOL_SWAP_MODE);
        long swap_r3d     = SymbolInfoInteger(name, SYMBOL_SWAP_ROLLOVER3DAYS);

        //--- Plafond de volume : notionnel max = volume_max * contract_size * prix.
        double vol_min  = SymbolInfoDouble(name, SYMBOL_VOLUME_MIN);
        double vol_max  = SymbolInfoDouble(name, SYMBOL_VOLUME_MAX);
        double vol_step = SymbolInfoDouble(name, SYMBOL_VOLUME_STEP);

        double tick_value = SymbolInfoDouble(name, SYMBOL_TRADE_TICK_VALUE);
        double tick_size  = SymbolInfoDouble(name, SYMBOL_TRADE_TICK_SIZE);
        long stops_level  = SymbolInfoInteger(name, SYMBOL_TRADE_STOPS_LEVEL);
        double margin_ini = SymbolInfoDouble(name, SYMBOL_MARGIN_INITIAL);

        FileWriteString(fh, StringFormat(
            "%s,%s,%d,%.8f,%d,%d,%.2f,%s,%s,%d,"
            "%.4f,%.4f,%d,%d,%.4f,%.2f,%.4f,%.6f,%.8f,%d,%.2f,%s\r\n",
            name, path, (int)digits, point, (int)trade_mode,
            (int)calc_mode, csize, cur_base, cur_prof, (int)spread,
            swap_long, swap_short, (int)swap_mode, (int)swap_r3d,
            vol_min, vol_max, vol_step, tick_value, tick_size,
            (int)stops_level, margin_ini, descr));
        n_written++;

        //--- MarketWatch restauré tel quel : on n'ajoute pas 250 symboles au
        //--- terminal de l'utilisateur pour lire un catalogue.
        if(!was_selected) SymbolSelect(name, false);
    }

    FileClose(fh);
    PrintFormat("Wrote %d symbols to Files\\%s", n_written, out_name);
    Print("=== FxListSymbols done ===");
}
