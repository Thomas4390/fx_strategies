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
input double Inp_AllocH1Momentum   = 0.0;     // Phase D — sleeve H1 (default off)
                                              // sum(MR+TS+RSI+H1) doit être 1.0
input bool   Inp_EnableDDCap       = false;   // Désactivé par défaut (2026-05-04)
                                              // — circuit-breaker tail-risk LaTeX § 13.3.
                                              // Findings cumulés : DDCap=0.15 freinait
                                              // 24% configs IS pour aucun bénéfice OOS ;
                                              // DDCap=0.30 jamais touché en backtest 5.4 ans.
                                              // Code conservé pour ré-activation manuelle
                                              // (tail-risk insurance optionnelle).
input double Inp_DDCap             = 0.30;    // Seuil (utilisé uniquement si Inp_EnableDDCap=true)
input bool   Inp_ResetDDState      = false;
input bool   Inp_EnableMarginCap   = false;   // Désactivé par défaut (2026-05-04)
                                              // — cap marge 70 % LaTeX § 13.2.
                                              // Test on/off identique sur baseline 5.4 ans
                                              // (Sharpe 1.15, +4615 USD, DD 7.21%) :
                                              // marge utilisée jamais > 70% en config
                                              // normale (vol-targeting suffit).
                                              // Code conservé pour ré-activation manuelle.
input double Inp_MarginCapPct      = 0.70;    // Seuil (utilisé uniquement si Inp_EnableMarginCap=true)

// === Vol-targeting global ===
// Phase I (2026-05-05) : leverage uplift validé walk-forward N=5.
// Pré-Phase I : vt=0.28, lev=12, Sharpe 1.44, CAGR 9.18%, DD 7.77%.
// Phase I C1  : vt=0.75, lev=64, Sharpe 1.38, CAGR 21.82% OOS_med, DD 13.0%.
// Anti-overfit confirmé : PSR 100%, DSR 82.7% (235 trials), Bootstrap P5 +0.70.
input double Inp_GlobalTargetVol   = 0.75;
input double Inp_GlobalMaxLeverage = 64.0;
input double Inp_GlobalVolFloor    = 0.02;

// === Sleeve 1 — MR Macro (4 paires equal-weight, LaTeX § 03) ===
input string Inp_MR_Pairs          = "EURUSD,GBPUSD,USDJPY,USDCAD";
input int    Inp_MR_BBWindow       = 80;
input double Inp_MR_BBAlpha        = 5.0;
input double Inp_MR_TPStop         = 0.006;
input double Inp_MR_SLStop         = 0.005;
input int    Inp_MR_SessionStart   = 8;       // London full + early NY (Phase E.1)
                                              // ex 6 ; 8-16 UTC validé N=5 OOS :
                                              // ΔSharpe_med +0.27 ΔDD -0.91 pp.
input int    Inp_MR_SessionEnd     = 16;
input int    Inp_MR_TimeStopHours  = 6;
input int    Inp_MR_ForcedCloseHr  = 21;
input double Inp_MR_SpreadThresh   = 0.5;
input int    Inp_MR_SlippageBps    = 15;       // LaTeX § 03 — 15 bps intraday
input bool   Inp_MR_DisableMacroFilter = false; // Phase B.4 bypass macro_ok
                                              // (force MacroOk()=true). Off
                                              // par défaut, à utiliser pour
                                              // mesurer l'impact du filtre.

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
input string Inp_RSI_Pairs         = "EURUSD,GBPUSD,USDCAD";  // Phase E.3:
                                              // USDJPY retiré (drag -295 USD
                                              // sur 5.4y, validé N=5 OOS
                                              // ΔSharpe_med +0.06).
input int    Inp_RSI_Period        = 14;
input double Inp_RSI_Oversold      = 25.0;
input double Inp_RSI_Overbought    = 75.0;
input double Inp_RSI_ExitMid       = 50.0;
input int    Inp_RSI_SlippageBps   = 10;       // LaTeX § 05 — 10 bps daily

// === Sleeve 4 — H1 Momentum (Phase D) ===
input string Inp_H1_Pairs          = "EURUSD,GBPUSD,USDJPY";
input int    Inp_H1_FastEMA        = 20;
input int    Inp_H1_SlowEMA        = 50;
input int    Inp_H1_RSIPeriod      = 7;
input int    Inp_H1_RSILow         = 40;
input int    Inp_H1_RSIHigh        = 60;
input int    Inp_H1_ATRPeriod      = 14;
input double Inp_H1_ATRMultSL      = 2.0;
input double Inp_H1_TargetVol      = 0.10;
input double Inp_H1_MaxLeverage    = 3.0;
input int    Inp_H1_SlippageBps    = 12;       // 12 bps H1 (entre TS daily 10
                                              // et MR M1 15 bps)

// === Operational ===
input string Inp_SymbolSuffix      = ".c";    // Broker-specific (ECN/Raw uses ".c"; change for other brokers)
input int    Inp_MagicMR           = 831;
input int    Inp_MagicTS           = 832;
input int    Inp_MagicRSI          = 833;
input int    Inp_MagicH1           = 834;     // Phase D
input bool   Inp_LogVerbose        = false;
input bool   Inp_LogToFile         = true;
input bool   Inp_ExportDeals       = false;   // Dump per-deal CSV in OnTester
                                              // → deals_<ts>.csv en FILE_COMMON.
                                              // Phase B trade inspection (Plan CAGR).
input string Inp_MacroCacheFile    = "macro_cache.csv";
input bool   Inp_MacroUseCommon    = true;     // FILE_COMMON ou MQL5/Files
input int    Inp_MacroMaxAgeHours  = 168;      // LaTeX § 13.1 — freshness 7 jours
input int    Inp_DailyRecomputeHr  = 21;       // heure UTC de recompute daily

// === Macro source mode ===
//   FILE    : lit macro_cache.csv (bridge/fx_macro_bridge.py, 1 ligne live)
//   NATIVE  : Calendar MT5 (chômage US) + WebRequest FRED (spread 10Y-2Y) — live only
//   HYBRID  : essaie NATIVE, fallback FILE
//   HISTORY : lit macro_history.csv (bridge/fx_macro_history.py, time-indexed) — backtest
//   AUTO    : recommandé. MQLInfoInteger(MQL_TESTER) → HISTORY en tester, NATIVE en live
// Pré-requis NATIVE : URL FRED whitelistée dans Outils → Options → EA →
//   "Allow WebRequest for listed URL" → https://api.stlouisfed.org
//   Et un fichier `fred_api_key.txt` dans Common/Files contenant la clé.
// Pré-requis HISTORY/AUTO-en-tester : `macro_history.csv` généré au préalable
//   via `python src/mt5/bridge/fx_macro_history.py` (couvre la période de backtest).
input EMacroSourceMode Inp_MacroSourceMode      = MACRO_SOURCE_AUTO;  // dispatch tester vs live
input string           Inp_FREDApiKeyFile       = "fred_api_key.txt";
input bool             Inp_FREDKeyUseCommon     = true;
input string           Inp_FREDSeriesId         = "T10Y2Y";
input string           Inp_MacroHistoryFile     = "macro_history.csv";
input bool             Inp_MacroHistoryUseCommon = true;

//============================================================ INCLUDES

// FxCommon est déjà inclus en haut du fichier (avant les inputs).
#include "..\Include\FxLogger.mqh"
#include "..\Include\FxRiskManager.mqh"
#include "..\Include\FxMacroFilter.mqh"
#include "..\Include\FxSleeveBase.mqh"
#include "..\Include\FxSleeveMRMacro.mqh"
#include "..\Include\FxSleeveTSMomentum.mqh"
#include "..\Include\FxSleeveRSIDaily.mqh"
#include "..\Include\FxSleeveH1Momentum.mqh"

//============================================================ STATE

CRiskManager       g_risk;
CMacroFilter       g_macro;
CSleeveMRMacro     g_sleeve_mr;
CSleeveTSMomentum  g_sleeve_ts;
CSleeveRSIDaily    g_sleeve_rsi;
CSleeveH1Momentum  g_sleeve_h1;

datetime           g_last_d1_bar     = 0;
datetime           g_last_macro_age_warn = 0;
datetime           g_last_margin_check   = 0;
datetime           g_session_start   = 0;     // Captured at OnInit() to compute years
                                              // window in OnTester() (robust to 0-deal runs)

//============================================================ ON INIT

int OnInit()
{
    g_logger.Init(Inp_LogVerbose, Inp_LogToFile);
    g_logger.Info("INIT", StringFormat("FxMultiSleeve start build %s", __DATE__));
    g_session_start = TimeCurrent();   // tester start_date in backtest, live time otherwise

    // Validation des allocations
    if(!g_risk.Init(Inp_AllocMRMacro, Inp_AllocTSMomentum, Inp_AllocRSIDaily,
                    Inp_GlobalTargetVol, Inp_GlobalMaxLeverage, Inp_GlobalVolFloor,
                    Inp_EnableDDCap, Inp_DDCap, Inp_ResetDDState,
                    Inp_EnableMarginCap, Inp_MarginCapPct,
                    Inp_AllocH1Momentum))
    {
        g_logger.Error("INIT", "RiskManager init failed");
        return INIT_PARAMETERS_INCORRECT;
    }

    g_macro.Init(Inp_MacroSourceMode,
                 Inp_MacroCacheFile, Inp_MacroMaxAgeHours, Inp_MacroUseCommon,
                 Inp_MR_SpreadThresh,
                 Inp_FREDApiKeyFile, Inp_FREDKeyUseCommon,
                 Inp_MacroHistoryFile, Inp_MacroHistoryUseCommon,
                 Inp_FREDSeriesId, Inp_MR_DisableMacroFilter);

    // Explicit diagnostic so the user can verify the actual input mode (vs the
    // mode that AUTO resolves to). MQL_TESTER=1 inside Strategy Tester, 0 live.
    g_logger.Info("INIT",
        StringFormat("MacroSource: input=%s resolved=%s tester=%d",
                     EnumToString(Inp_MacroSourceMode),
                     EnumToString(g_macro.ResolveEffectiveMode()),
                     (int)MQLInfoInteger(MQL_TESTER)));
    g_logger.Info("INIT",
        StringFormat("Inputs: SymbolSuffix='%s' Alloc=%.2f/%.2f/%.2f TargetVol=%.2f MaxLev=%.1f",
                     Inp_SymbolSuffix,
                     Inp_AllocMRMacro, Inp_AllocTSMomentum, Inp_AllocRSIDaily,
                     Inp_GlobalTargetVol, Inp_GlobalMaxLeverage));

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
    if(Inp_AllocH1Momentum > 0.0 && !g_sleeve_h1.Init())
    {
        g_logger.Error("INIT", "Sleeve H1 Momentum init failed");
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
    if(Inp_AllocH1Momentum > 0.0) g_sleeve_h1.Shutdown();
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

    // H1 momentum (Phase D) : détection new H1 bar en multi-pair.
    if(Inp_AllocH1Momentum > 0.0)
        g_sleeve_h1.OnNewBarH1Multi(g_risk);
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

//============================================================ ON TESTER
//
// Custom optimization metric : computes CAGR over the test window and
// emits a structured `[OPTIM]` log line with all key inputs + metrics.
// MT5 distributes optimization runs across local agents (parallel cores)
// and each agent's log accumulates these lines — `scripts/optimization/
// run_optimization_cli.py` parses them post-run to rebuild the surface.
//
// Returned value drives `OptimizationCriterion=6` (Custom max).
double OnTester()
{
    double initial = TesterStatistics(STAT_INITIAL_DEPOSIT);
    double net     = TesterStatistics(STAT_PROFIT);
    double dd_pct  = TesterStatistics(STAT_EQUITY_DDREL_PERCENT);
    // Note: STAT_SHARPE_RATIO is capped at -5.00 by MT5 when the underlying
    // equity-curve Sharpe is -inf or below floor (configs that wipe equity
    // very fast). Treat sharpe == -5.0 as a sentinel "config plante" rather
    // than a real metric. Sortino/Calmar can complement if needed.
    double sharpe  = TesterStatistics(STAT_SHARPE_RATIO);
    double pf      = TesterStatistics(STAT_PROFIT_FACTOR);
    double rf      = TesterStatistics(STAT_RECOVERY_FACTOR);
    double trades  = TesterStatistics(STAT_TRADES);

    double final_eq = initial + net;

    // Window in years computed from session timestamps captured at OnInit()
    // (= tester start_date in backtest) and TimeCurrent() at OnTester
    // (= tester end_date). Robust to 0-deal runs and avoids drift when only
    // a few deals fire near edges of the window.
    double years = 1.0;
    if(g_session_start > 0)
    {
        datetime now = TimeCurrent();
        if(now > g_session_start)
            years = (double)(now - g_session_start) / 31557600.0;  // s/an
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

    // En mode optimization MT5 filtre les Print() côté agents → fallback sur
    // une écriture CSV en FILE_COMMON. Chaque agent ajoute une ligne ; le
    // FILE_SHARE_WRITE + le lock interne MT5 sérialisent les writes.
    int h = FileOpen("optim_results.csv",
        FILE_READ | FILE_WRITE | FILE_COMMON | FILE_TXT | FILE_SHARE_READ |
        FILE_SHARE_WRITE, ',', CP_UTF8);
    if(h != INVALID_HANDLE)
    {
        FileSeek(h, 0, SEEK_END);
        // Header automatique si fichier vide
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

    // Per-deal CSV export (Phase B trade inspection). Disabled by default to
    // avoid overhead during optimization sweeps where only aggregate metrics
    // matter. Enable via --input Inp_ExportDeals=true on a single backtest run.
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
                    if(magic == Inp_MagicMR)       sleeve = "MR_MACRO";
                    else if(magic == Inp_MagicTS)  sleeve = "TS_MOMENTUM";
                    else if(magic == Inp_MagicRSI) sleeve = "RSI_DAILY";
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
            PrintFormat("[OPTIM] deals exported → %s", deals_file);
        }
    }

    return cagr;
}
