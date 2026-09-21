//+------------------------------------------------------------------+
//| X10Trace.mqh                                                     |
//|                                                                  |
//| The event trace of §13 - one line per ARM / BREAK / SWEEP /      |
//| ENTRY / EXIT / CANCEL, 21 columns in the order the spec fixes,   |
//| plus the optional 22nd 'cancel_reason' that §13 does not define  |
//| but without which an unexplained CANCEL cannot be attributed on  |
//| rung 3 of the reconciliation ladder (same choice as the Python   |
//| reference, emit_event_trace(extended=True)).                     |
//|                                                                  |
//| Written in Common\Files so scripts/reconcile_x10_events.py picks |
//| it up next to the vbt and QuantConnect traces. Stamps are UTC,   |
//| "YYYY-MM-DD HH:MM:SS"; precisions are the ones of                |
//| src/strategies/xau_x10.py::_TRACE_DECIMALS; an undefined value   |
//| is an EMPTY field, never a 0.                                    |
//+------------------------------------------------------------------+
#ifndef __X10_TRACE_MQH__
#define __X10_TRACE_MQH__

#include "X10Levels.mqh"
#include "X10Clock.mqh"

//--- §7.4 events, §13 scenarios, §10 exit reasons, and the cancel
//--- reasons of the Python registry (x10_engine.py CR_*).
#define X10_EV_ARM    0
#define X10_EV_BREAK  1
#define X10_EV_SWEEP  2
#define X10_EV_ENTRY  3
#define X10_EV_EXIT   4
#define X10_EV_CANCEL 5

#define X10_SC_NONE        0
#define X10_SC_BREAK_LONG  1
#define X10_SC_BREAK_SHORT 2
#define X10_SC_REV_LONG    3
#define X10_SC_REV_SHORT   4

#define X10_XR_NONE    0
#define X10_XR_STOP    1
#define X10_XR_TARGET  2
#define X10_XR_TIME    3
#define X10_XR_SESSION 4

#define X10_CR_NONE   0
#define X10_CR_ZONE   1  // §7.1 price left the zone backwards
#define X10_CR_NARM   2  // §7.1 N_arm bars without an event
#define X10_CR_HOLD   3  // §7.2 a close came back through the level
#define X10_CR_NSWEEP 4  // §7.3 no reintegration within N_sweep bars
#define X10_CR_SWEEP  5  // §7.3 reintegration without depth/decel/flip
#define X10_CR_CTX    6  // §6.4 breakout refused by EMA50 H1 or VWAP
#define X10_CR_R      7  // §8   R < 1
#define X10_CR_WINDOW 8  // §10  the fill bar falls in the blackout window
#define X10_CR_SIZE   9  // §11  lots below volume_min

string X10EventName(int code)
{
    switch(code)
    {
        case X10_EV_ARM:    return "ARM";
        case X10_EV_BREAK:  return "BREAK";
        case X10_EV_SWEEP:  return "SWEEP";
        case X10_EV_ENTRY:  return "ENTRY";
        case X10_EV_EXIT:   return "EXIT";
        case X10_EV_CANCEL: return "CANCEL";
    }
    return "";
}

string X10ScenarioName(int code)
{
    switch(code)
    {
        case X10_SC_BREAK_LONG:  return "BREAK_LONG";
        case X10_SC_BREAK_SHORT: return "BREAK_SHORT";
        case X10_SC_REV_LONG:    return "REV_LONG";
        case X10_SC_REV_SHORT:   return "REV_SHORT";
    }
    return "";
}

string X10ExitReasonName(int code)
{
    switch(code)
    {
        case X10_XR_STOP:    return "STOP";
        case X10_XR_TARGET:  return "TARGET";
        case X10_XR_TIME:    return "TIME";
        case X10_XR_SESSION: return "SESSION";
    }
    return "";
}

string X10CancelReasonName(int code)
{
    switch(code)
    {
        case X10_CR_ZONE:   return "zone";
        case X10_CR_NARM:   return "narm";
        case X10_CR_HOLD:   return "hold";
        case X10_CR_NSWEEP: return "nsweep";
        case X10_CR_SWEEP:  return "sweep";
        case X10_CR_CTX:    return "ctx";
        case X10_CR_R:      return "r";
        case X10_CR_WINDOW: return "window";
        case X10_CR_SIZE:   return "size";
    }
    return "";
}

//+------------------------------------------------------------------+
//| One trace row. Built field by field by the caller so an event    |
//| never inherits a value from the previous one.                    |
//+------------------------------------------------------------------+
struct X10Event
{
    datetime ts_decision;   // server time of the decision bar; converted on write
    int      event;
    int      scenario;
    double   level;
    int      d;
    double   atr, v, m, a;
    int      ctx_ema, ctx_vwap, ctx_dxy;
    double   vwap, ema50_h1, dxy;
    double   stop, target, r_est;
    double   fill_px, exit_px;
    int      exit_reason;
    int      cancel_reason;
};

//--- Reset every field: the struct is reused bar after bar.
void X10EventInit(X10Event &e, datetime ts, int event_code)
{
    e.ts_decision   = ts;
    e.event         = event_code;
    e.scenario      = X10_SC_NONE;
    e.level         = X10_UNDEF;
    e.d             = 0;
    e.atr           = X10_UNDEF;
    e.v             = X10_UNDEF;
    e.m             = X10_UNDEF;
    e.a             = X10_UNDEF;
    e.ctx_ema       = 0;
    e.ctx_vwap      = 0;
    e.ctx_dxy       = 0;
    e.vwap          = X10_UNDEF;
    e.ema50_h1      = X10_UNDEF;
    e.dxy           = X10_UNDEF;
    e.stop          = X10_UNDEF;
    e.target        = X10_UNDEF;
    e.r_est         = X10_UNDEF;
    e.fill_px       = X10_UNDEF;
    e.exit_px       = X10_UNDEF;
    e.exit_reason   = X10_XR_NONE;
    e.cancel_reason = X10_CR_NONE;
}

//+------------------------------------------------------------------+
//| CX10Trace - the CSV sink. One handle kept open for the run and   |
//| flushed on every row: a tester that dies mid-run must still      |
//| leave a readable trace.                                          |
//+------------------------------------------------------------------+
class CX10Trace
{
private:
    int    m_handle;
    string m_path;
    long   m_rows;
    int    m_counts[6];   // one per event code, for the run summary

    //--- Undefined prints as an empty field, exactly like a NaN does
    //--- through pandas to_csv.
    string Num(double value, int digits) const
    {
        if(!X10Def(value)) return "";
        return DoubleToString(value, digits);
    }

public:
    CX10Trace() : m_handle(INVALID_HANDLE), m_path(""), m_rows(0)
    {
        for(int i = 0; i < 6; i++) m_counts[i] = 0;
    }

    bool IsOpen() const { return m_handle != INVALID_HANDLE; }
    string Path() const { return m_path; }
    long Rows() const { return m_rows; }
    int Count(int event_code) const
    {
        if(event_code < 0 || event_code > 5) return 0;
        return m_counts[event_code];
    }

    bool Open(string filename)
    {
        m_path   = filename;
        m_handle = FileOpen(filename,
                            FILE_WRITE | FILE_TXT | FILE_ANSI | FILE_COMMON |
                            FILE_SHARE_READ);
        if(m_handle == INVALID_HANDLE) return false;
        FileWriteString(m_handle,
            "ts_decision,event,scenario,level,d,atr,v,m,a,"
            "ctx_ema,ctx_vwap,ctx_dxy,vwap,ema50_h1,dxy,"
            "stop,target,r_est,fill_px,exit_px,exit_reason,cancel_reason\n");
        return true;
    }

    void Close()
    {
        if(m_handle == INVALID_HANDLE) return;
        FileClose(m_handle);
        m_handle = INVALID_HANDLE;
    }

    void Write(const X10Event &e)
    {
        if(e.event >= 0 && e.event <= 5) m_counts[e.event]++;
        m_rows++;
        if(m_handle == INVALID_HANDLE) return;

        string line = StringFormat(
            "%s,%s,%s,%s,%d,%s,%s,%s,%s,%d,%d,%d,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n",
            X10FormatUTC(X10ServerToUTC(e.ts_decision)),
            X10EventName(e.event),
            X10ScenarioName(e.scenario),
            Num(e.level, 3),
            e.d,
            Num(e.atr, 6), Num(e.v, 6), Num(e.m, 6), Num(e.a, 6),
            e.ctx_ema, e.ctx_vwap, e.ctx_dxy,
            Num(e.vwap, 6), Num(e.ema50_h1, 6), Num(e.dxy, 6),
            Num(e.stop, 6), Num(e.target, 6), Num(e.r_est, 6),
            Num(e.fill_px, 3), Num(e.exit_px, 3),
            X10ExitReasonName(e.exit_reason),
            X10CancelReasonName(e.cancel_reason));
        FileWriteString(m_handle, line);
        FileFlush(m_handle);
    }
};

#endif // __X10_TRACE_MQH__
