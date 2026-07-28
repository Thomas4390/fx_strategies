//+------------------------------------------------------------------+
//| FxMultiSleeve.mq5                                                |
//|                                                                  |
//| Multi-sleeve FX expert advisor combining three independent       |
//| trading systems behind a single shared risk manager:             |
//|                                                                  |
//|   1. MR Macro     (80%) - intraday VWAP / Bollinger band mean    |
//|                           reversion, gated by a macro filter     |
//|                           (4 majors equal-weighted on M1).       |
//|   2. TS Momentum  (10%) - daily EMA cross + RSI confirmation     |
//|                           (3 majors equal-weighted on D1).       |
//|   3. RSI Daily    (10%) - daily RSI mean reversion               |
//|                           (3-4 majors equal-weighted on D1).     |
//|                                                                  |
//| The risk manager applies a portfolio-wide vol-target leverage    |
//| recomputed every UTC trading day, a peak-equity drawdown         |
//| circuit breaker, and a margin-usage breaker for tail-risk        |
//| protection.                                                      |
//|                                                                  |
//| One EA instance is attached to any chart; each sleeve manages    |
//| its own pair universe and bar-detection logic so the chart       |
//| symbol/timeframe is irrelevant to the active sleeves.            |
//+------------------------------------------------------------------+
#property copyright "FxMultiSleeve"
#property version   "1.00"
#property strict

#include "..\Include\FxCommon.mqh"

//============================================================ INPUTS

// === Allocation & Risk =============================================
input double Inp_AllocMRMacro       = 0.67;     // sum across all sleeves must equal 1.0
input double Inp_AllocTSMomentum    = 0.09;
input double Inp_AllocRSIDaily      = 0.09;
input double Inp_AllocH1Momentum    = 0.0;      // optional sleeve, off by default
input double Inp_AllocGoldMomentum  = 0.15;     // Sleeve 5 — trio; 0.20 rejected by the PBO gate 2026-07-28

// Tail-risk protections.
//
// ⚠️ 2026-07-26 — the drawdown breaker is DISABLED by owner decision, under a
// CAGR mandate that declared risk non-binding. It is not a dead knob: at the
// current sizing the sleeves draw down well past the old 20% threshold (gold
// alone reaches -76% in the tester), so the breaker would have fired early and
// stayed latched, since it persists in a GlobalVariable and needs
// Inp_ResetDDState to clear. Re-enabling it without re-tuning the vol targets
// would flatten the account instead of protecting it.
input bool   Inp_EnableDDCap        = false;
input double Inp_DDCap              = 0.20;     // peak-equity drawdown threshold
input bool   Inp_ResetDDState       = false;    // wipe persisted DD state on init
input bool   Inp_EnableMarginCap    = true;
input double Inp_MarginCapPct       = 0.50;     // margin / equity threshold

// === Global vol-targeting overlay ==================================
// Retunés 2026-07-26 avec l'entrée de la sleeve or : la cible BAISSE (0.75 ->
// 0.37) alors que le CAGR MONTE (31,66% -> 40,49% en vbt), parce que l'or
// quasi orthogonal porte le Sharpe du combiné de 0,661 à 1,084. Le trio FX
// seul plafonnait à 31,66% quel que soit le levier.
input double Inp_GlobalTargetVol    = 0.37;
input double Inp_GlobalMaxLeverage  = 31.0;

// Échelle commune des budgets de risque par trade (FX_RISK_PCT_* dans
// FxCommon.mqh), 1.0 = valeurs historiques.
//
// Pourquoi ce réglage existe — mesuré le 2026-07-26. vbt atteint sa cible de
// vol (36,60% pour 37% visés), MT5 non (~10%). La cause n'est pas la cible mais
// le sizing de BASE : MT5 ouvre un notionnel de risk_pct/distance_au_stop fois
// la sous-équité, et son levier global est plafonné à target_vol/vol_floor =
// 18,5. Même au plafond, ce notionnel ne porte pas la vol du compte à 37% —
// relever Inp_GlobalTargetVol ne change donc rien, seul le sizing de base agit.
//
// Calibré à 4.5 pour délivrer le mandat de ~40% de CAGR SUR CE MOTEUR, mesuré
// (EURUSD.c, 2021-01 → 2025-12, model=1, 4 sleeves) :
//
//   scale   CAGR     maxDD    Sharpe   trades
//   1.0    12,19%    9,39%    1,219      812
//   2.0    21,03%   15,83%    1,134      812
//   3.5    33,41%   24,55%    1,089      812
//   4.5    40,45%   30,82%    1,062      812     <- retenu
//   5.0    44,35%   33,86%    1,057      812
//
// Le CAGR répond quasi linéairement et le nombre de trades ne bouge pas : ce
// paramètre dimensionne, il ne change aucun signal. Le Sharpe s'érode
// lentement (1,219 -> 1,062), les coûts de transaction pesant plus lourd à
// mesure que la taille monte.
//
// ⚠️ Ce paramètre multiplie le risque réellement pris, dans la proportion exacte
// où il multiplie le rendement attendu. Le monter est une décision de risque,
// pas un réglage de calibration.
input double Inp_RiskScale          = 4.5;
input double Inp_GlobalVolFloor     = 0.02;

// === Sleeve 1 - MR Macro ===========================================
input string Inp_MR_Pairs           = "EURUSD,GBPUSD,USDJPY,USDCAD";
input int    Inp_MR_BBWindow        = 80;
input double Inp_MR_BBAlpha         = 5.0;
input double Inp_MR_TPStop          = 0.006;
input double Inp_MR_SLStop          = 0.005;
input int    Inp_MR_SessionStart    = 8;        // UTC trading window start
input int    Inp_MR_SessionEnd      = 16;       // UTC trading window end (exclusive)
input int    Inp_MR_TimeStopHours   = 6;        // per-trade time stop
input int    Inp_MR_ForcedCloseHr   = 21;       // daily forced flat (UTC)
input double Inp_MR_SpreadThresh    = 0.5;      // 10Y-2Y spread threshold
input int    Inp_MR_SlippageBps     = 15;       // intraday slippage (bps per side)
input bool   Inp_MR_DisableMacroFilter = false; // diagnostic bypass for macro_ok
input bool   Inp_MR_NewsFilterEnabled  = true;  // skip entries near high-impact USD events

// === Sleeve 2 - TS Momentum ========================================
input string Inp_TS_Pairs           = "EURUSD,GBPUSD,USDJPY";
input int    Inp_TS_FastEMA         = 20;
input int    Inp_TS_SlowEMA         = 50;
input int    Inp_TS_RSIPeriod       = 7;
input int    Inp_TS_RSILow          = 40;
input int    Inp_TS_RSIHigh         = 60;
input double Inp_TS_TargetVol       = 0.10;
input double Inp_TS_MaxLeverage     = 3.0;
input int    Inp_TS_SlippageBps     = 10;       // daily slippage (bps per side)

// === Sleeve 3 - RSI Daily ==========================================
input string Inp_RSI_Pairs          = "EURUSD,GBPUSD,USDCAD";
input int    Inp_RSI_Period         = 14;
input double Inp_RSI_Oversold       = 25.0;
input double Inp_RSI_Overbought     = 75.0;
input double Inp_RSI_ExitMid        = 50.0;
input int    Inp_RSI_SlippageBps    = 10;       // daily slippage (bps per side)
input int    Inp_RSI_TimeStopDays   = 21;       // 0 = disabled

// === Sleeve 4 - H1 Momentum (optional) =============================
input string Inp_H1_Pairs           = "EURUSD,GBPUSD,USDJPY";
input int    Inp_H1_FastEMA         = 20;
input int    Inp_H1_SlowEMA         = 50;
input int    Inp_H1_RSIPeriod       = 7;
input int    Inp_H1_RSILow           = 40;
input int    Inp_H1_RSIHigh          = 60;
input int    Inp_H1_ATRPeriod       = 14;
input double Inp_H1_ATRMultSL       = 2.0;
input double Inp_H1_TargetVol       = 0.10;
input double Inp_H1_MaxLeverage     = 3.0;
input int    Inp_H1_SlippageBps     = 12;

//--- Sleeve 5: Gold Momentum (daily TS momentum on XAUUSD) --------
// The four lookbacks are averaged, not selected. Do NOT grid-search them:
// the averaging is what keeps this signal from being an overfit.
// Inp_Gold_Symbols is a CSV: the sub-equity is split equally across the
// instruments listed, so the default single symbol keeps the sizing of the
// mono-instrument sleeve untouched.
input string Inp_Gold_Symbols      = "XAUUSD,USDJPY,XAGUSD";
input int    Inp_Gold_LookbackA     = 40;      // 0 disables a slot
input int    Inp_Gold_LookbackB     = 60;
input int    Inp_Gold_LookbackC     = 120;
input int    Inp_Gold_LookbackD     = 250;
input bool   Inp_Gold_AllowShort    = false;   // gold has a structural long drift
input double Inp_Gold_TargetVol     = 0.55;    // retuned 2026-07-26 (was 0.25)
input double Inp_Gold_MaxLeverage   = 6.6;     // retuned 2026-07-26 (was 3.0)
input double Inp_Gold_SafetySL      = 0.04;    // gold is ~2x FX volatility
input int    Inp_Gold_SlippageBps   = 2;       // XAUUSD CFD spread + commission
input bool   Inp_Gold_Trace         = false;   // daily reconciliation trace — diagnostic only
input string Inp_Gold_TraceFile     = "gold_trace.csv";  // in Common\Files
input string Inp_Gold_TraceSymbol   = "";      // base name to trace; "" = first configured

// === Execution costs ===============================================
// Per-side commission in basis points. Calibrate to the live broker:
//   * Spread-only (e.g. OANDA Standard): 0.0
//   * Raw-spread + commission (e.g. OANDA Core, IC Markets Raw,
//     Pepperstone Razor): typically 3.0 - 5.0 on EUR/USD.
input double Inp_CommissionBpsPerSide = 5.0;

// Per-night swap drag in basis points, applied to overnight holdings
// (TS Momentum and RSI Daily). Models the cumulative funding cost the
// strategy tester does not always reproduce when historical swap data
// is missing or outdated.
input double Inp_SwapBpsPerNight     = 0.5;

// === Operational ===================================================
input string Inp_SymbolSuffix       = ".c";    // broker-specific symbol suffix
input int    Inp_MagicMR            = 831;
input int    Inp_MagicTS            = 832;
input int    Inp_MagicRSI           = 833;
input int    Inp_MagicH1            = 834;
input int    Inp_MagicGold          = 835;
input bool   Inp_LogVerbose         = false;
input bool   Inp_LogToFile          = true;
input bool   Inp_ExportDeals        = false;   // dump per-deal CSV in OnTester
input string Inp_MacroCacheFile     = "macro_cache.csv";
input bool   Inp_MacroUseCommon     = true;
input int    Inp_MacroMaxAgeHours   = 168;     // freshness window for cached macro state
input int    Inp_DailyRecomputeHr   = 21;      // UTC hour for daily portfolio recompute

// === Macro source mode =============================================
//   FILE    : single-row CSV produced by an external bridge (legacy).
//   NATIVE  : MT5 calendar + FRED API via WebRequest (live only).
//   HYBRID  : NATIVE first, fallback to FILE on failure.
//   HISTORY : multi-row CSV pre-indexed by release date (tester only).
//   AUTO    : recommended; HISTORY in the tester, NATIVE when live.
//
// Prerequisites for NATIVE mode:
//   * Whitelist https://api.stlouisfed.org under
//     Tools -> Options -> Expert Advisors -> "Allow WebRequest".
//   * Place an API key in the file referenced by Inp_FREDApiKeyFile
//     (default: <Common>/Files/fred_api_key.txt).
//
// Prerequisites for HISTORY (and AUTO inside the tester):
//   * Generate macro_history.csv covering the backtest window via the
//     companion bridge script.
input EMacroSourceMode Inp_MacroSourceMode      = MACRO_SOURCE_AUTO;
input string           Inp_FREDApiKeyFile       = "fred_api_key.txt";
input bool             Inp_FREDKeyUseCommon     = true;
input string           Inp_FREDSeriesId         = "T10Y2Y";
input string           Inp_MacroHistoryFile     = "macro_history.csv";
input bool             Inp_MacroHistoryUseCommon = true;

//============================================================ INCLUDES

#include "..\Include\FxLogger.mqh"
#include "..\Include\FxRiskManager.mqh"
#include "..\Include\FxMacroFilter.mqh"
#include "..\Include\FxSleeveBase.mqh"
#include "..\Include\FxSleeveMRMacro.mqh"
#include "..\Include\FxSleeveTSMomentum.mqh"
#include "..\Include\FxSleeveRSIDaily.mqh"
#include "..\Include\FxSleeveH1Momentum.mqh"
#include "..\Include\FxSleeveGoldMomentum.mqh"

//============================================================ STATE

CRiskManager       g_risk;
CMacroFilter       g_macro;
CSleeveMRMacro     g_sleeve_mr;
CSleeveTSMomentum  g_sleeve_ts;
CSleeveRSIDaily    g_sleeve_rsi;
CSleeveH1Momentum  g_sleeve_h1;
CSleeveGoldMomentum g_sleeve_gold;

datetime           g_last_d1_bar         = 0;
datetime           g_last_macro_age_warn = 0;
datetime           g_last_margin_check   = 0;
datetime           g_session_start       = 0;  // captured at OnInit, used for OnTester window length

//============================================================ ON INIT

int OnInit()
{
    g_logger.Init(Inp_LogVerbose, Inp_LogToFile);
    g_logger.Info("INIT", StringFormat("FxMultiSleeve start build %s", __DATE__));
    g_session_start = TimeCurrent();

    if(!g_risk.Init(Inp_AllocMRMacro, Inp_AllocTSMomentum, Inp_AllocRSIDaily,
                    Inp_GlobalTargetVol, Inp_GlobalMaxLeverage, Inp_GlobalVolFloor,
                    Inp_EnableDDCap, Inp_DDCap, Inp_ResetDDState,
                    Inp_EnableMarginCap, Inp_MarginCapPct,
                    Inp_AllocH1Momentum, Inp_AllocGoldMomentum))
    {
        g_logger.Error("INIT", "RiskManager init failed");
        return INIT_PARAMETERS_INCORRECT;
    }

    g_macro.Init(Inp_MacroSourceMode,
                 Inp_MacroCacheFile, Inp_MacroMaxAgeHours, Inp_MacroUseCommon,
                 Inp_MR_SpreadThresh,
                 Inp_FREDApiKeyFile, Inp_FREDKeyUseCommon,
                 Inp_MacroHistoryFile, Inp_MacroHistoryUseCommon,
                 Inp_FREDSeriesId, Inp_MR_DisableMacroFilter,
                 Inp_MR_NewsFilterEnabled);

    // Diagnostic: surface the actual macro mode resolved at runtime.
    g_logger.Info("INIT",
        StringFormat("MacroSource: input=%s resolved=%s tester=%d",
                     EnumToString(Inp_MacroSourceMode),
                     EnumToString(g_macro.ResolveEffectiveMode()),
                     (int)MQLInfoInteger(MQL_TESTER)));
    g_logger.Info("INIT",
        StringFormat("Inputs: SymbolSuffix='%s' Alloc=%.2f/%.2f/%.2f "
                     "TargetVol=%.2f MaxLev=%.1f",
                     Inp_SymbolSuffix,
                     Inp_AllocMRMacro, Inp_AllocTSMomentum, Inp_AllocRSIDaily,
                     Inp_GlobalTargetVol, Inp_GlobalMaxLeverage));

    if(!g_macro.Refresh())
    {
        g_logger.Warn("INIT",
            StringFormat("Macro initial load failed (mode=%s); MR sleeve "
                         "disabled until next refresh",
                         EnumToString(Inp_MacroSourceMode)));
        // Live (NATIVE) only: raise a visible popup so the operator notices
        // the macro outage without scanning the Experts log. Alert() is a
        // no-op in the Strategy Tester. The message carries the precise
        // cause (e.g. the WebRequest 4014 whitelist hint from CMacroSourceFRED).
        if(g_macro.ResolveEffectiveMode() == MACRO_SOURCE_NATIVE)
            Alert(StringFormat("FxMultiSleeve : macro FRED indisponible - %s. "
                               "Sleeve MR Macro desactive jusqu'au prochain refresh.",
                               g_macro.LastError()));
    }
    else
        g_logger.Info("INIT",
            StringFormat("Macro source=%s spread=%.4f unemp_rising=%d macro_ok=%d",
                         g_macro.LastSource(), g_macro.Spread(),
                         (int)g_macro.UnempRising(), (int)g_macro.MacroOk()));

    if(!g_sleeve_mr.Init())
    {
        g_logger.Error("INIT", "Sleeve MR Macro init failed");
        return INIT_FAILED;
    }
    if(!g_sleeve_ts.Init())
    {
        g_logger.Error("INIT", "Sleeve TS Momentum init failed");
        return INIT_FAILED;
    }
    if(!g_sleeve_rsi.Init())
    {
        g_logger.Error("INIT", "Sleeve RSI Daily init failed");
        return INIT_FAILED;
    }
    if(Inp_AllocGoldMomentum > 0.0 && !g_sleeve_gold.Init())
        return INIT_FAILED;
    if(Inp_AllocH1Momentum > 0.0 && !g_sleeve_h1.Init())
    {
        g_logger.Error("INIT", "Sleeve H1 Momentum init failed");
        return INIT_FAILED;
    }

    // 60-second timer drives macro refresh, drawdown monitoring, and
    // the daily portfolio recompute trigger.
    EventSetTimer(60);
    g_logger.Info("INIT", "EA ready");
    return INIT_SUCCEEDED;
}

//============================================================ ON DEINIT

void OnDeinit(const int reason)
{
    EventKillTimer();
    g_sleeve_mr.Shutdown();
    g_sleeve_ts.Shutdown();
    g_sleeve_rsi.Shutdown();
    if(Inp_AllocH1Momentum > 0.0) g_sleeve_h1.Shutdown();
    if(Inp_AllocGoldMomentum > 0.0) g_sleeve_gold.Shutdown();
    g_logger.Info("DEINIT", StringFormat("EA stopped reason=%d", reason));
    g_logger.Shutdown();
}

//============================================================ ON TICK

void OnTick()
{
    // Drawdown breaker is rate-limited internally (1 Hz) so calling it
    // here on every tick stays cheap.
    g_risk.CheckDDCircuitBreaker(g_logger);
    if(g_risk.IsDDLocked()) return;

    // Margin usage check is throttled to once per 30 seconds.
    datetime now = TimeGMT();
    if(now - g_last_margin_check >= 30)
    {
        g_risk.CheckMarginCap(g_logger);
        g_last_margin_check = now;
    }

    // Multi-pair new-bar detection happens inside the sleeve so the EA
    // can be attached to any chart symbol/timeframe.
    g_sleeve_mr.OnNewBarM1(g_macro, g_risk);

    // Time stops and the daily forced close run on every tick to stay
    // responsive to the deadline crossing.
    g_sleeve_mr.CheckIntradayExits();

    if(Inp_AllocH1Momentum > 0.0)
        g_sleeve_h1.OnNewBarH1Multi(g_risk);
}

//============================================================ ON TIMER

void OnTimer()
{
    g_macro.Refresh();
    if(!g_macro.IsValid())
    {
        // Cache stale: close MR Macro positions until the source returns
        // a fresh value. Throttle the warning to one per hour to keep
        // the log readable during long outages.
        datetime now = TimeGMT();
        if(now - g_last_macro_age_warn > 3600)
        {
            g_logger.Warn("MACRO",
                StringFormat("Cache stale (age=%ds) - closing MR Macro positions",
                             g_macro.AgeSeconds()));
            g_last_macro_age_warn = now;
        }
        g_sleeve_mr.CloseAll("macro cache stale");
    }

    // Detect the UTC day rollover past the recompute hour and trigger
    // the daily portfolio update + daily sleeves.
    datetime now = TimeGMT();
    datetime today = FloorToDayUTC(now);
    int hour_utc = (int)((now / 3600) % 24);

    if(today != g_last_d1_bar && hour_utc >= Inp_DailyRecomputeHr)
    {
        g_risk.RecomputeGlobalLeverage(g_logger);
        g_sleeve_ts.OnNewBarD1(g_risk);
        g_sleeve_rsi.OnNewBarD1(g_risk);
        if(Inp_AllocGoldMomentum > 0.0)
            g_sleeve_gold.OnNewBarD1(g_risk);
        g_last_d1_bar = today;
        g_logger.Info("DAILY",
            StringFormat("Daily recompute done at hour=%d UTC", hour_utc));
    }
}

//============================================================ ON TRADE TRANSACTION

void OnTradeTransaction(const MqlTradeTransaction &trans,
                        const MqlTradeRequest &req,
                        const MqlTradeResult &res)
{
    if(trans.type == TRADE_TRANSACTION_DEAL_ADD)
    {
        if(Inp_LogVerbose)
        {
            string sym = trans.symbol;
            ulong deal = trans.deal;
            g_logger.Debug("DEAL",
                StringFormat("symbol=%s deal=%I64u type=%d",
                             sym, deal, (int)trans.deal_type));
        }
    }
}

//============================================================ ON TESTER
//
// Custom optimization metric. Computes a CAGR from initial deposit and
// final equity, plus a structured "[OPTIM]" log line summarising the
// run. The strategy tester aggregates this output across optimisation
// agents; downstream tooling parses the log lines and the optional
// CSV dump to rebuild the parameter surface.
//
// The returned value drives OptimizationCriterion=6 (custom maximum).
double OnTester()
{
    double initial = TesterStatistics(STAT_INITIAL_DEPOSIT);
    double net     = TesterStatistics(STAT_PROFIT);
    double dd_pct  = TesterStatistics(STAT_EQUITY_DDREL_PERCENT);

    // STAT_SHARPE_RATIO is clamped at -5.0 by MT5 when the equity-curve
    // Sharpe is undefined or below the floor (catastrophic configs).
    // Treat -5.0 as a sentinel rather than a real metric downstream.
    double sharpe  = TesterStatistics(STAT_SHARPE_RATIO);
    double pf      = TesterStatistics(STAT_PROFIT_FACTOR);
    double rf      = TesterStatistics(STAT_RECOVERY_FACTOR);
    double trades  = TesterStatistics(STAT_TRADES);

    double final_eq = initial + net;

    // Window length in years from the OnInit timestamp to OnTester. In
    // backtest these correspond to the configured FromDate / ToDate.
    double years = 1.0;
    if(g_session_start > 0)
    {
        datetime now = TimeCurrent();
        if(now > g_session_start)
            years = (double)(now - g_session_start) / 31557600.0;
    }
    if(years <= 0.01) years = 1.0;

    double cagr = (final_eq > 0.0)
                  ? MathPow(final_eq / initial, 1.0 / years) - 1.0
                  : -1.0;

    PrintFormat("[OPTIM] vt=%.4f maxlev=%.4f volfloor=%.4f"
                " cagr=%.6f dd=%.4f sharpe=%.4f pf=%.4f rf=%.4f"
                " trades=%.0f net=%.2f years=%.3f",
                Inp_GlobalTargetVol, Inp_GlobalMaxLeverage, Inp_GlobalVolFloor,
                cagr, dd_pct, sharpe, pf, rf, trades, net, years);

    // Optimisation runs filter Print() output across agents, so mirror
    // each result to a shared CSV for offline aggregation. Each agent
    // appends one row; FILE_SHARE_WRITE serialises concurrent writes.
    int h = FileOpen("optim_results.csv",
        FILE_READ | FILE_WRITE | FILE_COMMON | FILE_TXT | FILE_SHARE_READ |
        FILE_SHARE_WRITE, ',', CP_UTF8);
    if(h != INVALID_HANDLE)
    {
        FileSeek(h, 0, SEEK_END);
        if(FileTell(h) == 0)
            FileWrite(h, "ts_utc", "target_vol", "max_lev", "vol_floor",
                      "cagr", "equity_dd_pct", "sharpe", "profit_factor",
                      "recovery_factor", "trades", "net_profit", "years");
        string ts = TimeToString(TimeGMT(), TIME_DATE | TIME_SECONDS);
        FileWrite(h, ts,
                  DoubleToString(Inp_GlobalTargetVol, 4),
                  DoubleToString(Inp_GlobalMaxLeverage, 4),
                  DoubleToString(Inp_GlobalVolFloor, 4),
                  DoubleToString(cagr, 6),
                  DoubleToString(dd_pct, 4),
                  DoubleToString(sharpe, 4),
                  DoubleToString(pf, 4),
                  DoubleToString(rf, 4),
                  IntegerToString((int)trades),
                  DoubleToString(net, 2),
                  DoubleToString(years, 3));
        FileClose(h);
    }

    // Optional per-deal CSV dump for trade-level inspection. Disabled
    // by default to avoid overhead during sweeps where only aggregate
    // metrics matter; enable on a single run via the input override.
    if(Inp_ExportDeals)
    {
        string ts_run = TimeToString(TimeGMT(), TIME_DATE | TIME_MINUTES);
        StringReplace(ts_run, ".", "");
        StringReplace(ts_run, ":", "");
        StringReplace(ts_run, " ", "T");
        string deals_file = StringFormat("deals_%s.csv", ts_run);
        int hd = FileOpen(deals_file,
            FILE_WRITE | FILE_COMMON | FILE_TXT | FILE_SHARE_READ |
            FILE_SHARE_WRITE, ',', CP_UTF8);
        if(hd != INVALID_HANDLE)
        {
            FileWrite(hd, "deal_id", "position_id", "time_utc", "symbol",
                      "magic", "sleeve", "type", "entry", "volume", "price",
                      "profit", "commission", "swap");
            if(HistorySelect(0, TimeCurrent()))
            {
                int nd = HistoryDealsTotal();
                for(int k = 0; k < nd; k++)
                {
                    ulong tk = HistoryDealGetTicket(k);
                    if(tk == 0) continue;
                    long magic = HistoryDealGetInteger(tk, DEAL_MAGIC);
                    string sleeve;
                    if(magic == Inp_MagicMR)        sleeve = "MR_MACRO";
                    else if(magic == Inp_MagicTS)   sleeve = "TS_MOMENTUM";
                    else if(magic == Inp_MagicRSI)  sleeve = "RSI_DAILY";
                    else if(magic == Inp_MagicGold) sleeve = "GOLD_MOMENTUM";
                    else if(magic == Inp_MagicH1)   sleeve = "H1_MOMENTUM";
                    else                            sleeve = "OTHER";
                    long deal_type  = HistoryDealGetInteger(tk, DEAL_TYPE);
                    long deal_entry = HistoryDealGetInteger(tk, DEAL_ENTRY);
                    datetime t      = (datetime)HistoryDealGetInteger(tk, DEAL_TIME);
                    string sym      = HistoryDealGetString(tk, DEAL_SYMBOL);
                    double vol      = HistoryDealGetDouble(tk, DEAL_VOLUME);
                    double price    = HistoryDealGetDouble(tk, DEAL_PRICE);
                    double profit   = HistoryDealGetDouble(tk, DEAL_PROFIT);
                    double comm     = HistoryDealGetDouble(tk, DEAL_COMMISSION);
                    double swap     = HistoryDealGetDouble(tk, DEAL_SWAP);
                    long pos_id = HistoryDealGetInteger(tk, DEAL_POSITION_ID);
                    FileWrite(hd,
                        IntegerToString((int)tk),
                        IntegerToString((int)pos_id),
                        TimeToString(t, TIME_DATE | TIME_SECONDS),
                        sym,
                        IntegerToString((int)magic),
                        sleeve,
                        IntegerToString((int)deal_type),
                        IntegerToString((int)deal_entry),
                        DoubleToString(vol, 4),
                        DoubleToString(price, 5),
                        DoubleToString(profit, 2),
                        DoubleToString(comm, 4),
                        DoubleToString(swap, 4));
                }
            }
            FileClose(hd);
            PrintFormat("[OPTIM] deals exported -> %s", deals_file);
        }
    }

    return cagr;
}
