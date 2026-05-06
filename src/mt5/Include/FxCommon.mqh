//+------------------------------------------------------------------+
//| FxCommon.mqh                                                     |
//|                                                                  |
//| Shared constants, enumerations, and utility helpers used by all  |
//| sleeves and the risk manager. Single source of truth for magic   |
//| numbers, default thresholds, and reusable primitives.            |
//+------------------------------------------------------------------+
#ifndef __FX_COMMON_MQH__
#define __FX_COMMON_MQH__

//+------------------------------------------------------------------+
//| Magic numbers (one per sleeve)                                   |
//+------------------------------------------------------------------+
#define MAGIC_MR_MACRO     831
#define MAGIC_TS_MOMENTUM  832
#define MAGIC_RSI_DAILY    833
#define MAGIC_H1_MOMENTUM  834   // reserved for hourly momentum sleeve

//+------------------------------------------------------------------+
//| Sleeve identifiers                                               |
//+------------------------------------------------------------------+
enum ESleeveID
{
    SLEEVE_MR_MACRO    = 0,
    SLEEVE_TS_MOMENTUM = 1,
    SLEEVE_RSI_DAILY   = 2,
    SLEEVE_H1_MOMENTUM = 3
};

//+------------------------------------------------------------------+
//| Macro data source modes for the MR Macro filter.                 |
//|                                                                  |
//|   FILE    : single-row CSV produced by an external bridge        |
//|             (legacy, polled by an external scheduler).           |
//|   NATIVE  : MT5 economic calendar + WebRequest to FRED API       |
//|             (live only; WebRequest is blocked in Strategy Tester)|
//|   HYBRID  : NATIVE first, fallback to FILE on failure.           |
//|   HISTORY : multi-row CSV pre-indexed by release date for the    |
//|             Strategy Tester.                                     |
//|   AUTO    : recommended. Detects MQLInfoInteger(MQL_TESTER) at   |
//|             runtime and selects HISTORY in tester or NATIVE in   |
//|             live, removing the need for manual configuration.    |
//+------------------------------------------------------------------+
enum EMacroSourceMode
{
    MACRO_SOURCE_FILE    = 0,
    MACRO_SOURCE_NATIVE  = 1,
    MACRO_SOURCE_HYBRID  = 2,
    MACRO_SOURCE_HISTORY = 3,
    MACRO_SOURCE_AUTO    = 4
};

//+------------------------------------------------------------------+
//| Risk-manager defaults (overridable via inputs).                  |
//+------------------------------------------------------------------+
#define FX_MIN_VOL_FLOOR        0.02   // lower bound for realised vol (annualised)
#define FX_VOL_TARGET_GLOBAL    0.28   // legacy default target volatility
#define FX_MAX_LEVERAGE_GLOBAL  12.0   // legacy default max leverage
#define FX_DD_CAP_DEFAULT       0.20   // peak-equity drawdown circuit breaker
#define FX_MARGIN_CAP_DEFAULT   0.50   // margin/equity cap before deleveraging

//+------------------------------------------------------------------+
//| Sizing constants (per-trade risk percent of sleeve sub-equity).  |
//+------------------------------------------------------------------+
#define FX_RISK_PCT_MR_MACRO     0.01    // intraday MR sleeve (4 pairs)
#define FX_RISK_PCT_TS_MOMENTUM  0.05    // daily TS sleeve (3 pairs)
#define FX_RISK_PCT_RSI_DAILY    0.05    // daily RSI sleeve (3-4 pairs)

//+------------------------------------------------------------------+
//| Execution constants.                                             |
//+------------------------------------------------------------------+
#define FX_DEVIATION_POINTS      50      // 5 pips cap on Instant Execution
                                         // (no-op on ECN/STP/NDD market exec)
#define FX_DEFAULT_DEVIATION     20      // CTrade default deviation
#define FX_HEALTH_FLOOR_DRAG     0.5     // floor for sizing drag multiplier

//+------------------------------------------------------------------+
//| Empirical holding periods used for swap drag pre-payment.        |
//+------------------------------------------------------------------+
#define FX_TS_AVG_NIGHTS_HELD    10.0    // TS Momentum median holding
#define FX_RSI_AVG_NIGHTS_HELD   7.0     // RSI Daily median holding

//+------------------------------------------------------------------+
//| Stops-level safety extra cushion (in points).                    |
//+------------------------------------------------------------------+
#define FX_STOPS_SAFETY_POINTS   5

//+------------------------------------------------------------------+
//| Persistent state keys (Global Variables).                        |
//+------------------------------------------------------------------+
#define GV_PEAK_EQUITY        "FX_PEAK_EQUITY"
#define GV_DD_TRIGGERED       "FX_DD_TRIGGERED"
#define GV_GLOBAL_LEVERAGE    "FX_GLOBAL_LEVERAGE"
#define GV_LAST_DAILY_RECOMP  "FX_LAST_DAILY_RECOMP"

//+------------------------------------------------------------------+
//| Build a broker-suffixed symbol name (e.g. "EURUSD" + ".c").      |
//+------------------------------------------------------------------+
string MakeSymbolWithSuffix(string base, string suffix)
{
    if(StringLen(suffix) == 0) return base;
    return base + suffix;
}

//+------------------------------------------------------------------+
//| Ensure the symbol is selected in MarketWatch; add it if missing. |
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
//| Force the loading of historical bars for a symbol/timeframe.     |
//| Retries up to 25 times with 100 ms back-off, mirroring the       |
//| official MQL5 sample.                                            |
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
//| Return midnight UTC of the day containing 't'.                   |
//+------------------------------------------------------------------+
datetime FloorToDayUTC(datetime t)
{
    return t - (t % 86400);
}

//+------------------------------------------------------------------+
//| Test whether 't' falls within the [start_h, end_h) UTC window.   |
//+------------------------------------------------------------------+
bool IsInUTCSession(datetime t, int start_h, int end_h)
{
    MqlDateTime st;
    TimeToStruct(t, st);
    return (st.hour >= start_h && st.hour < end_h);
}

//+------------------------------------------------------------------+
//| Split a comma-separated string into an array.                    |
//|     "EURUSD,GBPUSD,USDJPY" -> ["EURUSD","GBPUSD","USDJPY"]       |
//+------------------------------------------------------------------+
int SplitCsv(string csv, string &out[])
{
    return StringSplit(csv, ',', out);
}

//+------------------------------------------------------------------+
//| Arithmetic mean of arr[from .. from+count-1].                    |
//+------------------------------------------------------------------+
double ArrayMean(const double &arr[], int from, int count)
{
    if(count <= 0) return 0.0;
    double s = 0.0;
    for(int i = from; i < from + count; i++) s += arr[i];
    return s / count;
}

//+------------------------------------------------------------------+
//| Sample standard deviation with ddof=1 (matches pandas default).  |
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
//| Count active positions for a given magic, optionally filtered    |
//| by symbol.                                                       |
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
