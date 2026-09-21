//+------------------------------------------------------------------+
//| XauX10.mq5                                                       |
//|                                                                  |
//| Strategy 2 for Apogee Invest: XAUUSD on the psychological $10    |
//| levels. MQL5 port of docs/specs/xau_x10_spec.md, mirroring the   |
//| reference Python engine (src/framework/x10_engine.py) state by   |
//| state so that an MT5 backtest can be reconciled event by event   |
//| against the vbt trace (§13).                                     |
//|                                                                  |
//| Chain of decision (§0), in this order and no other:              |
//|   x10 level -> context -> velocity -> acceleration -> breakout   |
//|   or sweep -> hold/retest -> R >= 1 -> entry.                    |
//|                                                                  |
//| What this EA does on every NEW M5 bar of the chart symbol, and   |
//| nothing on any other tick:                                       |
//|   1. rebuild the causal indicators on the bar that just closed   |
//|      (index 1) - ATR by hand (§4: iATR is forbidden), v/m/a,     |
//|      session VWAP, H1 context lagged by one H1 bar, DXY4;        |
//|   2. resolve an open position: the broker owns STOP and TARGET,  |
//|      the EA owns TIME (48 bars) and SESSION (16:55 New York);    |
//|   3. run the automaton of §7 and, if it fires, size (§11) and    |
//|      send a market order at the open of the bar now forming -    |
//|      which is exactly the "next open" of §2.                     |
//|                                                                  |
//| Everything annexe A.1 freezes is a #define in the includes, not  |
//| an input. The inputs carry the 27-point grid of A.2, the risk    |
//| fraction, the operational switches, and the one assumption that  |
//| could not be read anywhere in the repo: the server-to-New-York   |
//| clock (see X10Clock.mqh for how it was measured).                |
//+------------------------------------------------------------------+
#property copyright "fx_strategies - Apogee Invest strategy 2"
#property version   "1.00"
#property strict

#include <Trade/Trade.mqh>
#include "..\Include\FxCommon.mqh"
#include "..\Include\FxLogger.mqh"
#include "..\Include\X10Clock.mqh"
#include "..\Include\X10Levels.mqh"
#include "..\Include\X10Kinematics.mqh"
#include "..\Include\X10Context.mqh"
#include "..\Include\X10Trace.mqh"
#include "..\Include\X10StateMachine.mqh"

//============================================================ INPUTS

// === Grid of annexe A.2 - the only three tunable numbers ===========
input double Inp_Z              = 1.0;    // zone width, in ATR (0.5 / 1 / 1.5)
input double Inp_AMin           = 0.2;    // acceleration threshold (0.1 / 0.2 / 0.3)
input double Inp_KS             = 1.0;    // breakout stop, in ATR (0.75 / 1 / 1.5)

// === Sizing (§11) ==================================================
input double Inp_RiskFrac       = 0.005;  // 0.5% of equity, halved if DXY adverse

// === Costs (§12) ===================================================
input double Inp_SlippageUSD    = 0.0;    // per side, added to the half spread in R

// === Clock (§2) ====================================================
// X10_SERVER_UTC is the measured default for this broker; the other
// two modes exist to override it without recompiling. See X10Clock.mqh.
input EX10ServerClock Inp_ServerToNYMode      = X10_SERVER_UTC;
input int             Inp_ServerToNYOffsetMin = 0;  // used by X10_SERVER_FIXED

// === Operational ===================================================
input string Inp_SymbolSuffix   = ".c";   // broker-specific symbol suffix
input bool   Inp_Trace          = true;   // §13 event trace in Common\Files
input bool   Inp_DumpBars       = false;  // M1 OHLC + spread dump (diagnostic)
input bool   Inp_LogToFile      = true;   // -> MQL5\Files\x10_log.csv
input bool   Inp_LogVerbose     = false;
input bool   Inp_ExportDeals    = true;   // per-deal CSV in OnTester
input int    Inp_MagicBreakLong  = 841;
input int    Inp_MagicBreakShort = 842;
input int    Inp_MagicRevLong    = 843;
input int    Inp_MagicRevShort   = 844;

//============================================================ CONSTANTS

// M5 lookback rebuilt on every decision. Must cover one full session
// (276 M5 bars from 18:00 to 17:00 New York) so the session VWAP of the
// decision bar is complete, plus the ATR warmup. The Wilder seed of §4
// is reproduced exactly; starting the recursion 700 bars back rather
// than at the first bar of the dataset leaves a residue of (13/14)^686,
// i.e. below 1e-22 of one ATR.
#define X10_M5_WINDOW 700
// H1 lookback for EMA50 / ATR14 / DXY4. 500 bars is ten EMA spans: the
// seeding residue is under 1e-8 and the spec only asks for 60.
#define X10_H1_WINDOW 500
#define X10_H1_MIN_BARS 60
#define X10_SPREAD_HIST 4096

//============================================================ STATE

CX10Trace        g_trace;
CX10StateMachine g_sm;
CTrade           g_trade;

string   g_symbol = "";
double   g_point = 0.001;
int      g_digits = 3;
double   g_contract = 100.0;
double   g_vol_step = 0.01;
double   g_vol_min = 0.01;
double   g_vol_max = 100.0;

string   g_leg_eur = "", g_leg_jpy = "", g_leg_gbp = "", g_leg_cad = "";
bool     g_legs_ok = false;

datetime g_last_m5 = 0;
datetime g_last_m1 = 0;
long     g_bar_index = -1;
datetime g_session_start = 0;

//--- H1 context cache, rebuilt only when a new H1 bar closes.
datetime g_h1_cache_key = 0;
int      g_h1_n = 0;
datetime g_h1_time[];
double   g_h1_ema[];
double   g_h1_atr[];
double   g_h1_dxy[];
double   g_h1_dxy_ema[];

//--- open position, mirrored from the trade the EA sent.
ulong    g_pos_id = 0;
int      g_pos_q = 0;
int      g_pos_d = 0;
int      g_pos_scn = X10_SC_NONE;
double   g_pos_level = 0.0;
double   g_pos_stop = 0.0;
double   g_pos_target = 0.0;
double   g_pos_r = 0.0;
double   g_pos_fill = 0.0;
double   g_pos_lots = 0.0;
long     g_pos_fill_index = 0;
long     g_pos_session = 0;

//--- run counters, all of them countable and none of them monetary.
long     g_bars_seen = 0;
long     g_decisions = 0;
long     g_frozen_bars = 0;
long     g_dxy_undef_bars = 0;
long     g_entries = 0;
long     g_order_failures = 0;
int      g_spread_hist[X10_SPREAD_HIST];
long     g_spread_samples = 0;
long     g_spread_zero = 0;
int      g_dump_handle = INVALID_HANDLE;
string   g_trace_file = "";

//============================================================ HELPERS

//--- §11: rounding is DOWNWARDS to the volume step, never to nearest.
//--- Same logic as LotsForRisk / NormalizeLots (FxTradeHelpers.mqh:17-27),
//--- minus their clamp UP to volume_min: a lot below the minimum
//--- cancels the trade here (§11), it does not become the minimum.
double X10FloorToStep(double raw, double step)
{
    if(raw <= 0.0 || step <= 0.0) return 0.0;
    return MathFloor(raw / step) * step;
}

string X10ScenarioMagicName(long magic)
{
    if(magic == Inp_MagicBreakLong)  return "BREAK_LONG";
    if(magic == Inp_MagicBreakShort) return "BREAK_SHORT";
    if(magic == Inp_MagicRevLong)    return "REV_LONG";
    if(magic == Inp_MagicRevShort)   return "REV_SHORT";
    return "OTHER";
}

int X10MagicFor(int scenario)
{
    switch(scenario)
    {
        case X10_SC_BREAK_LONG:  return Inp_MagicBreakLong;
        case X10_SC_BREAK_SHORT: return Inp_MagicBreakShort;
        case X10_SC_REV_LONG:    return Inp_MagicRevLong;
        case X10_SC_REV_SHORT:   return Inp_MagicRevShort;
    }
    return Inp_MagicBreakLong;
}

//--- Median of the spread samples, in points. The §14 checklist asks
//--- for proof that the tester spread is not zero; this is it.
int X10SpreadMedianPoints()
{
    if(g_spread_samples <= 0) return -1;
    long half = g_spread_samples / 2;
    long cum = 0;
    for(int i = 0; i < X10_SPREAD_HIST; i++)
    {
        cum += g_spread_hist[i];
        if(cum > half) return i;
    }
    return -1;
}

//+------------------------------------------------------------------+
//| Align a leg's H1 closes onto the gold H1 labels - §6.3.          |
//| Last leg bar at or before the label, and undefined beyond         |
//| X10_DXY_FFILL_LIMIT bars: a frozen leg would keep scoring the    |
//| dollar on a quote that no longer exists.                         |
//+------------------------------------------------------------------+
void X10AlignLeg(const MqlRates &leg[], int n_leg,
                 const datetime &labels[], int n, double &out[])
{
    ArrayResize(out, n);
    int j = 0;
    for(int i = 0; i < n; i++)
    {
        out[i] = X10_UNDEF;
        while(j < n_leg && leg[j].time <= labels[i]) j++;
        int k = j - 1;
        if(k < 0) continue;
        if((long)labels[i] - (long)leg[k].time > X10_DXY_FFILL_LIMIT * 3600)
            continue;
        out[i] = leg[k].close;
    }
}

//+------------------------------------------------------------------+
//| Rebuild the H1 context (EMA50, ATR14, DXY4 and its EMA50) when a |
//| new H1 bar has closed. Everything goes through the §6.1 rule at  |
//| read time, so this cache holds H1-indexed series only.           |
//+------------------------------------------------------------------+
bool RefreshH1Context()
{
    MqlRates rh[];
    ArraySetAsSeries(rh, false);
    int nh = CopyRates(g_symbol, PERIOD_H1, 1, X10_H1_WINDOW, rh);
    if(nh < X10_H1_MIN_BARS) return false;
    if(rh[nh - 1].time == g_h1_cache_key && g_h1_n == nh) return true;

    g_h1_cache_key = rh[nh - 1].time;
    g_h1_n = nh;

    double high[], low[], close[];
    ArrayResize(high, nh);
    ArrayResize(low, nh);
    ArrayResize(close, nh);
    ArrayResize(g_h1_time, nh);
    for(int i = 0; i < nh; i++)
    {
        g_h1_time[i] = rh[i].time;
        high[i]  = rh[i].high;
        low[i]   = rh[i].low;
        close[i] = rh[i].close;
    }

    X10EmaSpan(close, nh, X10_EMA_SPAN_H1, X10_EMA_SPAN_H1, g_h1_ema);
    X10WilderATR(high, low, close, nh, X10_ATR_PERIOD, g_h1_atr);

    //--- DXY4 (§6.3). Any leg the tester cannot serve leaves the basket
    //--- undefined = neutral, which is the normal case in MT5 where
    //--- multi-symbol history is not guaranteed.
    ArrayResize(g_h1_dxy, nh);
    for(int i = 0; i < nh; i++) g_h1_dxy[i] = X10_UNDEF;

    if(g_legs_ok)
    {
        MqlRates re[], rj[], rg[], rc[];
        ArraySetAsSeries(re, false);
        ArraySetAsSeries(rj, false);
        ArraySetAsSeries(rg, false);
        ArraySetAsSeries(rc, false);
        int ne = CopyRates(g_leg_eur, PERIOD_H1, 1, X10_H1_WINDOW, re);
        int nj = CopyRates(g_leg_jpy, PERIOD_H1, 1, X10_H1_WINDOW, rj);
        int ng = CopyRates(g_leg_gbp, PERIOD_H1, 1, X10_H1_WINDOW, rg);
        int nc = CopyRates(g_leg_cad, PERIOD_H1, 1, X10_H1_WINDOW, rc);
        if(ne > 0 && nj > 0 && ng > 0 && nc > 0)
        {
            double ae[], aj[], ag[], ac[];
            X10AlignLeg(re, ne, g_h1_time, nh, ae);
            X10AlignLeg(rj, nj, g_h1_time, nh, aj);
            X10AlignLeg(rg, ng, g_h1_time, nh, ag);
            X10AlignLeg(rc, nc, g_h1_time, nh, ac);
            for(int i = 0; i < nh; i++)
                g_h1_dxy[i] = X10Dxy4(ae[i], aj[i], ag[i], ac[i]);
        }
    }
    X10EmaSpan(g_h1_dxy, nh, X10_EMA_SPAN_H1, X10_EMA_SPAN_H1, g_h1_dxy_ema);
    return true;
}

//+------------------------------------------------------------------+
//| Build the snapshot of the M5 bar that just closed (index 1).     |
//| Every series is recomputed left to right over a fixed window, so |
//| the value written for bar i reads only [i-window, i] - the same  |
//| causality property the Python kernels are tested on.             |
//+------------------------------------------------------------------+
bool BuildSnapshot(X10Snapshot &s)
{
    MqlRates r5[];
    ArraySetAsSeries(r5, false);
    int n5 = CopyRates(g_symbol, PERIOD_M5, 1, X10_M5_WINDOW, r5);
    if(n5 < X10_H_MOMENTUM + X10_ATR_PERIOD + 2) return false;

    double high[], low[], close[];
    ArrayResize(high, n5);
    ArrayResize(low, n5);
    ArrayResize(close, n5);
    for(int i = 0; i < n5; i++)
    {
        high[i]  = r5[i].high;
        low[i]   = r5[i].low;
        close[i] = r5[i].close;
    }

    double atr[], v[], m[], a[];
    X10WilderATR(high, low, close, n5, X10_ATR_PERIOD, atr);
    X10Kinematics(close, atr, n5, v, m, a);

    //--- §6.2: unweighted cumulative session mean, reset at 18:00 New
    //--- York, valid after 12 bars, NOT shifted.
    CSessionMeanTypical vwap;
    vwap.Init(X10_VWAP_MIN_BARS);
    for(int i = 0; i < n5; i++)
        vwap.Push(high[i], low[i], close[i], X10SessionId(r5[i].time));

    int last = n5 - 1;
    s.index         = g_bar_index;
    s.bar_open      = r5[last].time;
    s.fill_bar_open = iTime(g_symbol, PERIOD_M5, 0);
    //--- §2: the fill only exists on the bar that opens exactly 300 s
    //--- after the decision bar. A hole or a session reopen is not one.
    s.fill_bar_contiguous = ((long)s.fill_bar_open - (long)s.bar_open == 300);
    s.open   = r5[last].open;
    s.high   = high[last];
    s.low    = low[last];
    s.close  = close[last];
    s.atr    = atr[last];
    s.v      = v[last];
    s.v_prev = (last >= 1) ? v[last - 1] : X10_UNDEF;
    s.m      = m[last];
    s.a      = a[last];
    s.l_inf  = X10LevelInf(s.close);
    s.l_sup  = s.l_inf + X10_LEVEL_SIZE;
    s.vwap   = vwap.Value();
    s.minute_ny  = X10NYMinuteOfDay(s.bar_open);
    s.session_id = X10SessionId(s.bar_open);

    long spread_points = SymbolInfoInteger(g_symbol, SYMBOL_SPREAD);
    if(spread_points < 0) spread_points = 0;
    s.spread = (double)spread_points * g_point;
    if(spread_points < X10_SPREAD_HIST) g_spread_hist[(int)spread_points]++;
    if(spread_points == 0) g_spread_zero++;
    g_spread_samples++;

    //--- §6.1: the H1 context of the last H1 bar CLOSED at or before
    //--- the open of this M5 bar - the shift(1) of the reference.
    s.ema50_h1     = X10_UNDEF;
    s.atr_h1       = X10_UNDEF;
    s.dxy          = X10_UNDEF;
    s.dxy_ema50_h1 = X10_UNDEF;
    if(RefreshH1Context())
    {
        int j = X10AlignH1(g_h1_time, g_h1_n, s.bar_open);
        if(j >= 0)
        {
            s.ema50_h1     = g_h1_ema[j];
            s.atr_h1       = g_h1_atr[j];
            s.dxy          = g_h1_dxy[j];
            s.dxy_ema50_h1 = g_h1_dxy_ema[j];
        }
    }
    if(!X10Def(s.dxy)) g_dxy_undef_bars++;

    s.ok = X10Def(s.atr) && s.atr > 0.0 && X10Def(s.v) && X10Def(s.m)
           && X10Def(s.a);
    return true;
}

//============================================================ TRADING

//--- Emit the EXIT row of §13 and free the automaton (§9 cooldown).
void CloseOut(const X10Snapshot &s, double exit_px, int reason)
{
    X10Event e;
    X10EventInit(e, s.bar_open, X10_EV_EXIT);
    e.scenario    = g_pos_scn;
    e.level       = g_pos_level;
    e.d           = g_pos_d;
    e.atr         = s.atr;
    e.v           = s.v;
    e.m           = s.m;
    e.a           = s.a;
    e.ctx_ema     = X10CtxScore(g_pos_q, s.close, s.ema50_h1);
    e.ctx_vwap    = X10CtxScore(g_pos_q, s.close, s.vwap);
    e.ctx_dxy     = X10CtxDxy(g_pos_q, s.dxy, s.dxy_ema50_h1);
    e.vwap        = s.vwap;
    e.ema50_h1    = s.ema50_h1;
    e.dxy         = s.dxy;
    e.stop        = g_pos_stop;
    e.target      = g_pos_target;
    e.r_est       = g_pos_r;
    e.fill_px     = g_pos_fill;
    e.exit_px     = exit_px;
    e.exit_reason = reason;
    g_trace.Write(e);

    g_sm.RegisterCooldown(g_pos_level, s.index);
    g_sm.ToIdle();
    g_pos_id = 0;

    g_logger.Debug("EXIT", StringFormat("%s level=%.3f px=%.3f reason=%s",
        X10ScenarioName(g_pos_scn), g_pos_level, exit_px,
        X10ExitReasonName(reason)));
}

//--- The broker closed the position (SL or TP fired intrabar, §9).
//--- Read the closing deal to recover its price and its reason.
bool HarvestBrokerExit(const X10Snapshot &s)
{
    double exit_px = 0.0;
    int    reason  = X10_XR_NONE;
    if(HistorySelectByPosition(g_pos_id))
    {
        int nd = HistoryDealsTotal();
        for(int k = nd - 1; k >= 0; k--)
        {
            ulong tk = HistoryDealGetTicket(k);
            if(tk == 0) continue;
            if(HistoryDealGetInteger(tk, DEAL_ENTRY) != DEAL_ENTRY_OUT) continue;
            exit_px = HistoryDealGetDouble(tk, DEAL_PRICE);
            long dr = HistoryDealGetInteger(tk, DEAL_REASON);
            if(dr == DEAL_REASON_SL)      reason = X10_XR_STOP;
            else if(dr == DEAL_REASON_TP) reason = X10_XR_TARGET;
            break;
        }
    }
    if(reason == X10_XR_NONE)
    {
        //--- Closed by something other than our SL/TP (stop out, manual
        //--- intervention). Attribute it on the price, which is what the
        //--- reconciliation will read anyway.
        if(exit_px <= 0.0) exit_px = s.close;
        double d_stop   = MathAbs(exit_px - g_pos_stop);
        double d_target = MathAbs(exit_px - g_pos_target);
        reason = (d_stop <= d_target) ? X10_XR_STOP : X10_XR_TARGET;
        g_logger.Warn("EXIT", StringFormat(
            "position %I64u closed outside SL/TP at %.3f - attributed %s",
            g_pos_id, exit_px, X10ExitReasonName(reason)));
    }
    CloseOut(s, exit_px, reason);
    return true;
}

//--- Time exits of §10, both owned by the EA. SESSION fires only
//--- inside [16:55, 18:00) New York: outside that window a position
//--- opened in the evening lives on, crosses New York midnight, and is
//--- flattened at the next 16:55 - or earlier by STOP/TARGET/TIME.
bool CheckTimeExits(const X10Snapshot &s)
{
    long bars_held = s.index - g_pos_fill_index + 1;
    bool session_due = (s.minute_ny >= X10_SESSION_FORCE_MINUTE
                        && s.minute_ny < X10_SESSION_CLOSE_HOUR * 60 + 60)
                       || (s.session_id != g_pos_session);
    bool time_due = (bars_held >= X10_MAX_HOLD_BARS);
    if(!session_due && !time_due) return false;

    if(!g_trade.PositionClose(g_pos_id))
    {
        g_logger.Warn("EXIT", StringFormat(
            "PositionClose %I64u failed retcode=%d", g_pos_id,
            g_trade.ResultRetcode()));
        return false;
    }
    double exit_px = g_trade.ResultPrice();
    if(exit_px <= 0.0) exit_px = s.close;
    CloseOut(s, exit_px, session_due ? X10_XR_SESSION : X10_XR_TIME);
    return true;
}

//+------------------------------------------------------------------+
//| Size (§11) and send the market order at the open of the forming  |
//| bar. Emits the ENTRY row, or the CANCEL(size) row when the lot   |
//| computed falls under the broker minimum.                         |
//+------------------------------------------------------------------+
void ExecuteIntent(const X10Snapshot &s, const X10Intent &intent)
{
    double price = (intent.q > 0) ? SymbolInfoDouble(g_symbol, SYMBOL_ASK)
                                  : SymbolInfoDouble(g_symbol, SYMBOL_BID);
    if(price <= 0.0) price = s.close;

    //--- §6.4 usage 3 / §11: an adverse dollar halves the risk, it never
    //--- blocks the trade, and it does so for the four scenarios.
    double frac   = (intent.ctx_dxy < 0) ? Inp_RiskFrac * 0.5 : Inp_RiskFrac;
    double equity = AccountInfoDouble(ACCOUNT_EQUITY);
    double stop_dist = MathAbs(price - intent.stop);

    double lots = 0.0;
    if(stop_dist > 0.0 && equity > 0.0 && g_contract > 0.0)
    {
        lots = X10FloorToStep(frac * equity / (stop_dist * g_contract),
                              g_vol_step);
        double cap = X10FloorToStep(equity * X10_LEVERAGE_CAP
                                    / (g_contract * price), g_vol_step);
        if(lots > cap) lots = cap;
        if(lots > g_vol_max) lots = X10FloorToStep(g_vol_max, g_vol_step);
    }

    X10Event e;
    X10EventInit(e, s.bar_open, X10_EV_ENTRY);
    e.scenario = intent.scenario;
    e.level    = intent.level;
    e.d        = intent.d;
    e.atr      = s.atr;
    e.v        = s.v;
    e.m        = s.m;
    e.a        = s.a;
    e.ctx_ema  = intent.ctx_ema;
    e.ctx_vwap = intent.ctx_vwap;
    e.ctx_dxy  = intent.ctx_dxy;
    e.vwap     = s.vwap;
    e.ema50_h1 = s.ema50_h1;
    e.dxy      = s.dxy;
    e.stop     = intent.stop;
    e.target   = intent.target;
    e.r_est    = intent.r_est;

    if(lots < g_vol_min)
    {
        e.event         = X10_EV_CANCEL;
        e.cancel_reason = X10_CR_SIZE;
        g_trace.Write(e);
        g_sm.RegisterCooldown(intent.level, s.index);
        return;
    }

    double sl = NormalizeDouble(intent.stop, g_digits);
    double tp = NormalizeDouble(intent.target, g_digits);
    g_trade.SetExpertMagicNumber(X10MagicFor(intent.scenario));
    ENUM_ORDER_TYPE type = (intent.q > 0) ? ORDER_TYPE_BUY : ORDER_TYPE_SELL;
    if(!g_trade.PositionOpen(g_symbol, type, lots, price, sl, tp,
                             X10ScenarioName(intent.scenario)))
    {
        g_order_failures++;
        g_logger.Warn("ENTRY", StringFormat(
            "PositionOpen %s lots=%.2f sl=%.3f tp=%.3f failed retcode=%d",
            X10ScenarioName(intent.scenario), lots, sl, tp,
            g_trade.ResultRetcode()));
        g_sm.RegisterCooldown(intent.level, s.index);
        return;
    }

    double fill = g_trade.ResultPrice();
    if(fill <= 0.0) fill = price;
    ulong deal = g_trade.ResultDeal();
    g_pos_id = 0;
    if(deal != 0 && HistoryDealSelect(deal))
        g_pos_id = (ulong)HistoryDealGetInteger(deal, DEAL_POSITION_ID);
    if(g_pos_id == 0) g_pos_id = g_trade.ResultOrder();

    e.fill_px = fill;
    g_trace.Write(e);
    g_entries++;

    g_pos_q          = intent.q;
    g_pos_d          = intent.d;
    g_pos_scn        = intent.scenario;
    g_pos_level      = intent.level;
    g_pos_stop       = intent.stop;
    g_pos_target     = intent.target;
    g_pos_r          = intent.r_est;
    g_pos_fill       = fill;
    g_pos_lots       = lots;
    g_pos_fill_index = s.index + 1;   // the bar now forming carries the fill
    g_pos_session    = X10SessionId(s.fill_bar_open);

    g_logger.Debug("ENTRY", StringFormat(
        "%s level=%.3f lots=%.2f fill=%.3f sl=%.3f tp=%.3f r=%.3f ext=%d ret=%d",
        X10ScenarioName(intent.scenario), intent.level, lots, fill, sl, tp,
        intent.r_est, (int)intent.extension_vwap, (int)intent.retest));
}

//============================================================ BAR HANDLER

void OnNewM5Bar()
{
    g_bar_index++;
    g_bars_seen++;

    X10Snapshot s;
    if(!BuildSnapshot(s)) return;
    if(s.ok) g_decisions++; else g_frozen_bars++;

    //--- 1. an open position, resolved before anything is decided (§9).
    if(g_pos_id != 0)
    {
        if(!PositionSelectByTicket(g_pos_id)) HarvestBrokerExit(s);
        else                                  CheckTimeExits(s);
    }
    if(g_pos_id != 0) return;   // one position at a time (§9)

    //--- 2. the automaton, at the close of this bar (§2, §7).
    X10Intent intent;
    if(g_sm.Decide(s, intent) && intent.enter) ExecuteIntent(s, intent);
}

//--- Inp_DumpBars: one row per closed M1 bar, used to measure the
//--- historical spread outside the ~100 000 line cap of the MT5 bar
//--- export (§14). Diagnostic only, off by default.
void DumpM1Bar()
{
    if(g_dump_handle == INVALID_HANDLE) return;
    MqlRates r[];
    ArraySetAsSeries(r, false);
    if(CopyRates(g_symbol, PERIOD_M1, 1, 1, r) != 1) return;
    FileWriteString(g_dump_handle, StringFormat(
        "%s,%.3f,%.3f,%.3f,%.3f,%d,%I64d\n",
        X10FormatUTC(X10ServerToUTC(r[0].time)),
        r[0].open, r[0].high, r[0].low, r[0].close,
        (int)r[0].spread, r[0].tick_volume));
}

//============================================================ ON INIT

int OnInit()
{
    g_logger.Init(Inp_LogVerbose, Inp_LogToFile, "x10_log.csv");
    g_logger.Info("INIT", StringFormat("XauX10 start build %d",
                                       (int)TerminalInfoInteger(TERMINAL_BUILD)));
    g_session_start = TimeCurrent();
    ArrayInitialize(g_spread_hist, 0);

    g_symbol = _Symbol;
    X10ClockConfigure(Inp_ServerToNYMode, Inp_ServerToNYOffsetMin);

    if(Inp_Z <= 0.0 || Inp_AMin < 0.0 || Inp_KS <= 0.0)
    {
        g_logger.Error("INIT", StringFormat(
            "invalid grid point z=%.3f a_min=%.3f k_s=%.3f", Inp_Z, Inp_AMin,
            Inp_KS));
        return INIT_PARAMETERS_INCORRECT;
    }
    if(Inp_RiskFrac <= 0.0 || Inp_RiskFrac > 0.1)
    {
        g_logger.Error("INIT", StringFormat(
            "invalid Inp_RiskFrac=%.5f (expected in (0, 0.1])", Inp_RiskFrac));
        return INIT_PARAMETERS_INCORRECT;
    }
    if(Inp_SlippageUSD < 0.0)
    {
        g_logger.Error("INIT", "invalid Inp_SlippageUSD (expected >= 0)");
        return INIT_PARAMETERS_INCORRECT;
    }

    g_point    = SymbolInfoDouble(g_symbol, SYMBOL_POINT);
    g_digits   = (int)SymbolInfoInteger(g_symbol, SYMBOL_DIGITS);
    g_contract = SymbolInfoDouble(g_symbol, SYMBOL_TRADE_CONTRACT_SIZE);
    g_vol_step = SymbolInfoDouble(g_symbol, SYMBOL_VOLUME_STEP);
    g_vol_min  = SymbolInfoDouble(g_symbol, SYMBOL_VOLUME_MIN);
    g_vol_max  = SymbolInfoDouble(g_symbol, SYMBOL_VOLUME_MAX);
    if(g_point <= 0.0 || g_contract <= 0.0 || g_vol_step <= 0.0)
    {
        g_logger.Error("INIT", StringFormat(
            "symbol %s exposes no point/contract/step (point=%.6f contract=%.2f "
            "step=%.4f)", g_symbol, g_point, g_contract, g_vol_step));
        return INIT_FAILED;
    }

    //--- §6.3 legs. A missing leg is NOT fatal: the basket becomes
    //--- undefined = neutral and the occurrence is counted (§6.3).
    g_leg_eur = MakeSymbolWithSuffix("EURUSD", Inp_SymbolSuffix);
    g_leg_jpy = MakeSymbolWithSuffix("USDJPY", Inp_SymbolSuffix);
    g_leg_gbp = MakeSymbolWithSuffix("GBPUSD", Inp_SymbolSuffix);
    g_leg_cad = MakeSymbolWithSuffix("USDCAD", Inp_SymbolSuffix);
    g_legs_ok = EnsureSymbolSelected(g_leg_eur)
             && EnsureSymbolSelected(g_leg_jpy)
             && EnsureSymbolSelected(g_leg_gbp)
             && EnsureSymbolSelected(g_leg_cad);
    if(!g_legs_ok)
        g_logger.Warn("INIT", StringFormat(
            "DXY4 legs unavailable with suffix '%s' - ctx_dxy stays 0 (spec 6.3)",
            Inp_SymbolSuffix));

    if(Inp_Trace)
    {
        string stamp = TimeToString(TimeGMT(), TIME_DATE | TIME_MINUTES);
        StringReplace(stamp, ".", "");
        StringReplace(stamp, ":", "");
        StringReplace(stamp, " ", "T");
        g_trace_file = StringFormat("x10_trace_%s.csv", stamp);
        if(!g_trace.Open(g_trace_file))
        {
            g_logger.Error("INIT", StringFormat(
                "cannot open trace file %s (err=%d)", g_trace_file,
                GetLastError()));
            return INIT_FAILED;
        }
    }

    if(Inp_DumpBars)
    {
        string name = StringFormat("x10_m1_%s.csv", g_symbol);
        g_dump_handle = FileOpen(name, FILE_WRITE | FILE_TXT | FILE_ANSI
                                       | FILE_COMMON | FILE_SHARE_READ);
        if(g_dump_handle == INVALID_HANDLE)
        {
            g_logger.Error("INIT", StringFormat(
                "cannot open bar dump %s (err=%d)", name, GetLastError()));
            return INIT_FAILED;
        }
        FileWriteString(g_dump_handle,
            "time_utc,open,high,low,close,spread_points,tick_volume\n");
    }

    g_sm.Init(Inp_Z, Inp_AMin, Inp_KS, Inp_SlippageUSD, GetPointer(g_trace));
    g_trade.SetExpertMagicNumber(Inp_MagicBreakLong);
    g_trade.SetDeviationInPoints(FX_DEVIATION_POINTS);
    g_trade.SetTypeFillingBySymbol(g_symbol);
    g_trade.LogLevel(0);

    g_logger.Info("INIT", StringFormat(
        "Inputs: symbol=%s z=%.3f a_min=%.3f k_s=%.3f risk=%.4f slip=%.3f "
        "suffix='%s' clock=%s offset=%dmin trace=%s",
        g_symbol, Inp_Z, Inp_AMin, Inp_KS, Inp_RiskFrac, Inp_SlippageUSD,
        Inp_SymbolSuffix, EnumToString(Inp_ServerToNYMode),
        Inp_ServerToNYOffsetMin, (Inp_Trace ? g_trace_file : "off")));
    g_logger.Info("INIT", StringFormat(
        "Symbol: point=%.5f digits=%d contract=%.2f vol=[%.2f..%.2f] step=%.2f",
        g_point, g_digits, g_contract, g_vol_min, g_vol_max, g_vol_step));

    g_logger.Info("INIT", "EA ready");
    return INIT_SUCCEEDED;
}

//============================================================ ON DEINIT

void LogSummary(string tag)
{
    g_logger.Info(tag, StringFormat(
        "bars=%I64d decisions=%I64d frozen=%I64d arm=%d break=%d sweep=%d "
        "entry=%d exit=%d cancel=%d order_failures=%I64d",
        g_bars_seen, g_decisions, g_frozen_bars,
        g_trace.Count(X10_EV_ARM), g_trace.Count(X10_EV_BREAK),
        g_trace.Count(X10_EV_SWEEP), g_trace.Count(X10_EV_ENTRY),
        g_trace.Count(X10_EV_EXIT), g_trace.Count(X10_EV_CANCEL),
        g_order_failures));
    g_logger.Info(tag, StringFormat(
        "dxy_undef_bars=%I64d spread_samples=%I64d spread_zero=%I64d "
        "spread_median_points=%d trace=%s",
        g_dxy_undef_bars, g_spread_samples, g_spread_zero,
        X10SpreadMedianPoints(), g_trace_file));
}

void OnDeinit(const int reason)
{
    LogSummary("SUMMARY");
    g_trace.Close();
    if(g_dump_handle != INVALID_HANDLE)
    {
        FileClose(g_dump_handle);
        g_dump_handle = INVALID_HANDLE;
    }
    g_logger.Info("DEINIT", StringFormat("EA stopped reason=%d", reason));
    g_logger.Shutdown();
}

//============================================================ ON TICK

void OnTick()
{
    //--- Nothing is decided on a tick (§2): only the arrival of a new
    //--- M5 bar of the chart symbol opens a decision.
    if(Inp_DumpBars)
    {
        datetime t1 = iTime(g_symbol, PERIOD_M1, 0);
        if(t1 != g_last_m1)
        {
            if(g_last_m1 != 0) DumpM1Bar();
            g_last_m1 = t1;
        }
    }

    datetime t5 = iTime(g_symbol, PERIOD_M5, 0);
    if(t5 == 0 || t5 == g_last_m5) return;
    g_last_m5 = t5;
    OnNewM5Bar();
}

//============================================================ ON TESTER

double OnTester()
{
    double initial = TesterStatistics(STAT_INITIAL_DEPOSIT);
    double net     = TesterStatistics(STAT_PROFIT);
    double dd_pct  = TesterStatistics(STAT_EQUITY_DDREL_PERCENT);
    double sharpe  = TesterStatistics(STAT_SHARPE_RATIO);
    double pf      = TesterStatistics(STAT_PROFIT_FACTOR);
    double rf      = TesterStatistics(STAT_RECOVERY_FACTOR);
    double trades  = TesterStatistics(STAT_TRADES);

    double final_eq = initial + net;
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

    PrintFormat("[OPTIM] z=%.4f amin=%.4f ks=%.4f riskfrac=%.4f"
                " cagr=%.6f dd=%.4f sharpe=%.4f pf=%.4f rf=%.4f"
                " trades=%.0f net=%.2f years=%.3f",
                Inp_Z, Inp_AMin, Inp_KS, Inp_RiskFrac,
                cagr, dd_pct, sharpe, pf, rf, trades, net, years);
    LogSummary("OPTIM");

    //--- Same per-deal schema as FxMultiSleeve so
    //--- scripts/parse_mt5_report.py reads both without a branch; the
    //--- 'sleeve' column carries the x10 scenario.
    if(Inp_ExportDeals)
    {
        string ts_run = TimeToString(TimeGMT(), TIME_DATE | TIME_MINUTES);
        StringReplace(ts_run, ".", "");
        StringReplace(ts_run, ":", "");
        StringReplace(ts_run, " ", "T");
        string deals_file = StringFormat("deals_x10_%s.csv", ts_run);
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
                    long magic      = HistoryDealGetInteger(tk, DEAL_MAGIC);
                    long deal_type  = HistoryDealGetInteger(tk, DEAL_TYPE);
                    long deal_entry = HistoryDealGetInteger(tk, DEAL_ENTRY);
                    datetime t      = (datetime)HistoryDealGetInteger(tk, DEAL_TIME);
                    string sym      = HistoryDealGetString(tk, DEAL_SYMBOL);
                    double vol      = HistoryDealGetDouble(tk, DEAL_VOLUME);
                    double price    = HistoryDealGetDouble(tk, DEAL_PRICE);
                    double profit   = HistoryDealGetDouble(tk, DEAL_PROFIT);
                    double comm     = HistoryDealGetDouble(tk, DEAL_COMMISSION);
                    double swap     = HistoryDealGetDouble(tk, DEAL_SWAP);
                    long pos_id     = HistoryDealGetInteger(tk, DEAL_POSITION_ID);
                    FileWrite(hd,
                        IntegerToString((int)tk),
                        IntegerToString((int)pos_id),
                        TimeToString(t, TIME_DATE | TIME_SECONDS),
                        sym,
                        IntegerToString((int)magic),
                        X10ScenarioMagicName(magic),
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
