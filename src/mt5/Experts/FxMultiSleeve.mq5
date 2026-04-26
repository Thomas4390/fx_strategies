//+------------------------------------------------------------------+
//| FxMultiSleeve.mq5                                                |
//| EA orchestrateur de la stratégie FX Tri-Signaux (LaTeX § 2-13). |
//| Mono-EA attaché à n'importe quel chart (typiquement EUR/USD M1).|
//| Chaque sleeve gère son propre univers de paires.                 |
//|                                                                  |
//| Sleeves :                                                        |
//|   1. MR Macro     (80%) — intraday 4 paires, VWAP+BB + macro    |
//|   2. TS Momentum  (10%) — daily 3 paires, EMA20/50 + RSI(7)      |
//|   3. RSI Daily    (10%) — daily 4 paires, RSI(14)                |
//|                                                                  |
//| Overlay : vol-targeting global (target=0.28, max_lev=12),       |
//| circuit-breaker DD activé à 15%, cap marge 70%, freshness        |
//| macro 7 jours (cf. plan d'alignement MQL5 ↔ Python ↔ LaTeX).     |
//+------------------------------------------------------------------+
#property copyright "fx_strategies port — Apogee Invest"
#property version   "1.00"
#property strict

//--- Include FxCommon en premier pour que les enums (ESleeveID,
//--- EMacroSourceMode) soient connus au moment de la déclaration des inputs.
#include "..\Include\FxCommon.mqh"

//============================================================ INPUTS

// === Allocation & Risk ===
input double Inp_AllocMRMacro      = 0.80;
input double Inp_AllocTSMomentum   = 0.10;
input double Inp_AllocRSIDaily     = 0.10;
input bool   Inp_EnableDDCap       = true;    // LaTeX § 13.3 — alerte DD 15 %
input double Inp_DDCap             = 0.15;
input bool   Inp_ResetDDState      = false;
input bool   Inp_EnableMarginCap   = true;    // LaTeX § 13.2 — cap marge 70 %
input double Inp_MarginCapPct      = 0.70;

// === Vol-targeting global ===
input double Inp_GlobalTargetVol   = 0.28;
input double Inp_GlobalMaxLeverage = 12.0;
input double Inp_GlobalVolFloor    = 0.02;

// === Sleeve 1 — MR Macro (4 paires equal-weight, LaTeX § 03) ===
input string Inp_MR_Pairs          = "EURUSD,GBPUSD,USDJPY,USDCAD";
input int    Inp_MR_BBWindow       = 80;
input double Inp_MR_BBAlpha        = 5.0;
input double Inp_MR_TPStop         = 0.006;
input double Inp_MR_SLStop         = 0.005;
input int    Inp_MR_SessionStart   = 6;
input int    Inp_MR_SessionEnd     = 14;
input int    Inp_MR_TimeStopHours  = 6;
input int    Inp_MR_ForcedCloseHr  = 21;
input double Inp_MR_SpreadThresh   = 0.5;
input int    Inp_MR_SlippageBps    = 15;       // LaTeX § 03 — 15 bps intraday

// === Sleeve 2 — TS Momentum ===
input string Inp_TS_Pairs          = "EURUSD,GBPUSD,USDJPY";
input int    Inp_TS_FastEMA        = 20;
input int    Inp_TS_SlowEMA        = 50;
input int    Inp_TS_RSIPeriod      = 7;
input int    Inp_TS_RSILow         = 40;
input int    Inp_TS_RSIHigh        = 60;
input double Inp_TS_TargetVol      = 0.10;
input double Inp_TS_MaxLeverage    = 3.0;
input int    Inp_TS_SlippageBps    = 10;       // LaTeX § 04 — 10 bps daily

// === Sleeve 3 — RSI Daily ===
input string Inp_RSI_Pairs         = "EURUSD,GBPUSD,USDJPY,USDCAD";
input int    Inp_RSI_Period        = 14;
input double Inp_RSI_Oversold      = 25.0;
input double Inp_RSI_Overbought    = 75.0;
input double Inp_RSI_ExitMid       = 50.0;
input int    Inp_RSI_SlippageBps   = 10;       // LaTeX § 05 — 10 bps daily

// === Operational ===
input string Inp_SymbolSuffix      = "";
input int    Inp_MagicMR           = 831;
input int    Inp_MagicTS           = 832;
input int    Inp_MagicRSI          = 833;
input bool   Inp_LogVerbose        = false;
input bool   Inp_LogToFile         = true;
input string Inp_MacroCacheFile    = "macro_cache.csv";
input bool   Inp_MacroUseCommon    = true;     // FILE_COMMON ou MQL5/Files
input int    Inp_MacroMaxAgeHours  = 168;      // LaTeX § 13.1 — freshness 7 jours
input int    Inp_DailyRecomputeHr  = 21;       // heure UTC de recompute daily

// === Macro source mode ===
//   FILE   : lit macro_cache.csv produit par bridge/fx_macro_bridge.py
//   NATIVE : Calendar MT5 (chômage US) + WebRequest FRED (spread 10Y-2Y)
//   HYBRID : essaie NATIVE, fallback FILE
// Pré-requis NATIVE : URL FRED whitelistée dans Outils → Options → EA →
//   "Allow WebRequest for listed URL" → https://api.stlouisfed.org
//   Et un fichier `fred_api_key.txt` dans Common/Files contenant la clé.
input EMacroSourceMode Inp_MacroSourceMode      = MACRO_SOURCE_FILE;
input string           Inp_FREDApiKeyFile       = "fred_api_key.txt";
input bool             Inp_FREDKeyUseCommon     = true;
input string           Inp_FREDSeriesId         = "T10Y2Y";

//============================================================ INCLUDES

// FxCommon est déjà inclus en haut du fichier (avant les inputs).
#include "..\Include\FxLogger.mqh"
#include "..\Include\FxRiskManager.mqh"
#include "..\Include\FxMacroFilter.mqh"
#include "..\Include\FxSleeveBase.mqh"
#include "..\Include\FxSleeveMRMacro.mqh"
#include "..\Include\FxSleeveTSMomentum.mqh"
#include "..\Include\FxSleeveRSIDaily.mqh"

//============================================================ STATE

CRiskManager       g_risk;
CMacroFilter       g_macro;
CSleeveMRMacro     g_sleeve_mr;
CSleeveTSMomentum  g_sleeve_ts;
CSleeveRSIDaily    g_sleeve_rsi;

datetime           g_last_d1_bar     = 0;
datetime           g_last_macro_age_warn = 0;
datetime           g_last_margin_check   = 0;

//============================================================ ON INIT

int OnInit()
{
    g_logger.Init(Inp_LogVerbose, Inp_LogToFile);
    g_logger.Info("INIT", StringFormat("FxMultiSleeve start build %s", __DATE__));

    // Validation des allocations
    if(!g_risk.Init(Inp_AllocMRMacro, Inp_AllocTSMomentum, Inp_AllocRSIDaily,
                    Inp_GlobalTargetVol, Inp_GlobalMaxLeverage, Inp_GlobalVolFloor,
                    Inp_EnableDDCap, Inp_DDCap, Inp_ResetDDState,
                    Inp_EnableMarginCap, Inp_MarginCapPct))
    {
        g_logger.Error("INIT", "RiskManager init failed");
        return INIT_PARAMETERS_INCORRECT;
    }

    g_macro.Init(Inp_MacroSourceMode,
                 Inp_MacroCacheFile, Inp_MacroMaxAgeHours, Inp_MacroUseCommon,
                 Inp_MR_SpreadThresh,
                 Inp_FREDApiKeyFile, Inp_FREDKeyUseCommon,
                 Inp_FREDSeriesId);
    if(!g_macro.Refresh())
        g_logger.Warn("INIT",
            StringFormat("Macro initial load failed (mode=%s); sleeve 1 disabled until refresh",
                         EnumToString(Inp_MacroSourceMode)));
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

    // Timer 1 minute (refresh macro, monitoring DD, déclenche daily)
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
    g_logger.Info("DEINIT", StringFormat("EA stopped reason=%d", reason));
    g_logger.Shutdown();
}

//============================================================ ON TICK

void OnTick()
{
    // Circuit-breaker DD à chaque tick
    g_risk.CheckDDCircuitBreaker(g_logger);
    if(g_risk.IsDDLocked()) return;

    // Cap marge : check max toutes les 30s pour éviter le spam de log
    datetime now = TimeGMT();
    if(now - g_last_margin_check >= 30)
    {
        g_risk.CheckMarginCap(g_logger);
        g_last_margin_check = now;
    }

    // NewBar M1 multi-pair : le sleeve MR Macro itère sur ses 4 paires
    // et détecte ses propres nouvelles bars (EA peut être attaché à
    // n'importe quel chart).
    g_sleeve_mr.OnNewBarM1(g_macro, g_risk);

    // Time-stops intraday + close forcé 21h UTC (vérif fréquente)
    g_sleeve_mr.CheckIntradayExits();
}

//============================================================ ON TIMER

void OnTimer()
{
    // Refresh macro cache (toutes les minutes)
    g_macro.Refresh();
    if(!g_macro.IsValid())
    {
        // Politique : fermer sleeve 1 si cache > 24h
        datetime now = TimeGMT();
        if(now - g_last_macro_age_warn > 3600)
        {
            g_logger.Warn("MACRO",
                StringFormat("Cache stale (age=%ds) — closing MR Macro positions",
                             g_macro.AgeSeconds()));
            g_last_macro_age_warn = now;
        }
        g_sleeve_mr.CloseAll("macro cache stale");
    }

    // Détecte le passage à un nouveau jour UTC après l'heure de recompute
    datetime now = TimeGMT();
    datetime today = FloorToDayUTC(now);
    int hour_utc = (int)((now / 3600) % 24);

    if(today != g_last_d1_bar && hour_utc >= Inp_DailyRecomputeHr)
    {
        // Recompute le levier global (vol-targeting)
        g_risk.RecomputeGlobalLeverage(g_logger);
        // Déclenche les sleeves daily
        g_sleeve_ts.OnNewBarD1(g_risk);
        g_sleeve_rsi.OnNewBarD1(g_risk);
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
        // Optionnel : logger les deals pour audit
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
