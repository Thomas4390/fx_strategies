//+------------------------------------------------------------------+
//| FxCommon.mqh                                                     |
//| Constantes, enums, helpers communs partagés par les sleeves.     |
//+------------------------------------------------------------------+
#ifndef __FX_COMMON_MQH__
#define __FX_COMMON_MQH__

//--- Magic numbers (1 par sleeve)
#define MAGIC_MR_MACRO     831
#define MAGIC_TS_MOMENTUM  832
#define MAGIC_RSI_DAILY    833

//--- Identifiants de sleeve
enum ESleeveID
{
    SLEEVE_MR_MACRO    = 0,
    SLEEVE_TS_MOMENTUM = 1,
    SLEEVE_RSI_DAILY   = 2
};

//--- Source des données macro pour le filtre MR Macro
//---   FILE   : lit macro_cache.csv (bridge Python)
//---   NATIVE : Calendar MT5 + WebRequest FRED
//---   HYBRID : NATIVE puis fallback FILE en cas d'échec
enum EMacroSourceMode
{
    MACRO_SOURCE_FILE   = 0,
    MACRO_SOURCE_NATIVE = 1,
    MACRO_SOURCE_HYBRID = 2
};

//--- Constantes globales
#define FX_MIN_VOL_FLOOR        0.02   // plancher vol annualisée (vol-targeting)
#define FX_VOL_TARGET_GLOBAL    0.28   // vol cible 28% annualisé
#define FX_MAX_LEVERAGE_GLOBAL  12.0   // plafond levier global
#define FX_DD_CAP_DEFAULT       0.15   // seuil circuit-breaker DD

//--- Noms des Global Variables persistées entre redémarrages
#define GV_PEAK_EQUITY        "FX_PEAK_EQUITY"
#define GV_DD_TRIGGERED       "FX_DD_TRIGGERED"
#define GV_GLOBAL_LEVERAGE    "FX_GLOBAL_LEVERAGE"
#define GV_LAST_DAILY_RECOMP  "FX_LAST_DAILY_RECOMP"

//+------------------------------------------------------------------+
//| Construit le nom de symbole en ajoutant le suffixe broker.       |
//+------------------------------------------------------------------+
string MakeSymbolWithSuffix(string base, string suffix)
{
    if(StringLen(suffix) == 0) return base;
    return base + suffix;
}

//+------------------------------------------------------------------+
//| Vérifie qu'un symbole est sélectionné dans MarketWatch ;         |
//| sinon tente de l'ajouter.                                        |
//+------------------------------------------------------------------+
bool EnsureSymbolSelected(string symbol)
{
    if(SymbolInfoInteger(symbol, SYMBOL_SELECT)) return true;
    if(!SymbolSelect(symbol, true))
    {
        PrintFormat("EnsureSymbolSelected: cannot add %s (err=%d)",
                    symbol, GetLastError());
        return false;
    }
    return true;
}

//+------------------------------------------------------------------+
//| Force le chargement de l'historique d'un symbole/timeframe.      |
//| Tente jusqu'à 25 fois (cohérent avec snippets MQL5 officiels).   |
//+------------------------------------------------------------------+
bool EnsureHistory(string symbol, ENUM_TIMEFRAMES tf, int min_bars)
{
    MqlRates rates[];
    int copied = 0;
    for(int attempt = 0; attempt < 25; attempt++)
    {
        copied = CopyRates(symbol, tf, 0, min_bars, rates);
        if(copied >= min_bars) return true;
        Sleep(100);
    }
    PrintFormat("EnsureHistory: %s %s only %d/%d bars",
                symbol, EnumToString(tf), copied, min_bars);
    return false;
}

//+------------------------------------------------------------------+
//| Retourne minuit UTC du jour de `t`.                              |
//+------------------------------------------------------------------+
datetime FloorToDayUTC(datetime t)
{
    return t - (t % 86400);
}

//+------------------------------------------------------------------+
//| Vérifie si l'instant `t` est dans la session [start_h, end_h[ UTC|
//+------------------------------------------------------------------+
bool IsInUTCSession(datetime t, int start_h, int end_h)
{
    MqlDateTime st;
    TimeToStruct(t, st);
    return (st.hour >= start_h && st.hour < end_h);
}

//+------------------------------------------------------------------+
//| Sépare une chaîne CSV en tableau de strings.                      |
//| ex: "EURUSD,GBPUSD,USDJPY" -> ["EURUSD","GBPUSD","USDJPY"]        |
//+------------------------------------------------------------------+
int SplitCsv(string csv, string &out[])
{
    return StringSplit(csv, ',', out);
}

//+------------------------------------------------------------------+
//| Calcule la moyenne d'un tableau (utility).                        |
//+------------------------------------------------------------------+
double ArrayMean(const double &arr[], int from, int count)
{
    if(count <= 0) return 0.0;
    double s = 0.0;
    for(int i = from; i < from + count; i++) s += arr[i];
    return s / count;
}

//+------------------------------------------------------------------+
//| Calcule l'écart-type ddof=1 (équivalent pandas .std()).          |
//+------------------------------------------------------------------+
double ArrayStdDDof1(const double &arr[], int from, int count)
{
    if(count < 2) return 0.0;
    double mean = ArrayMean(arr, from, count);
    double s2 = 0.0;
    for(int i = from; i < from + count; i++)
    {
        double d = arr[i] - mean;
        s2 += d * d;
    }
    return MathSqrt(s2 / (count - 1));
}

//+------------------------------------------------------------------+
//| Compte les positions actives portant le `magic` donné            |
//| (et optionnellement filtrées par symbole).                       |
//+------------------------------------------------------------------+
int CountSleevePositions(int magic, string symbol = "")
{
    int total = PositionsTotal();
    int count = 0;
    for(int i = 0; i < total; i++)
    {
        ulong ticket = PositionGetTicket(i);
        if(ticket == 0) continue;
        if(PositionGetInteger(POSITION_MAGIC) != magic) continue;
        if(symbol != "" && PositionGetString(POSITION_SYMBOL) != symbol) continue;
        count++;
    }
    return count;
}

#endif // __FX_COMMON_MQH__
