//+------------------------------------------------------------------+
//| X10Context.mqh                                                   |
//|                                                                  |
//| The context of §6, whose whole difficulty is that its two        |
//| families do NOT obey the same causality rule:                    |
//|                                                                  |
//|   * H1 family (EMA50_H1, ATR_H1, DXY) - §6.1: an H1 bar is only  |
//|     visible once closed. Python does shift(1) then reindex+ffill;|
//|     here the same rule reads "the last H1 bar whose CLOSE time is |
//|     at or before the OPEN of the decision M5 bar", which is the  |
//|     literal translation of that shift on a left-labelled grid.   |
//|   * session family (VWAP) - §6.2: a cumulative mean INCLUDING    |
//|     bar t, known at the close of t, shifted by nothing at all.   |
//|                                                                  |
//| Port of src/framework/x10_context.py and of                      |
//| x10_kernels.py::session_mean_typical_nb.                         |
//+------------------------------------------------------------------+
#ifndef __X10_CONTEXT_MQH__
#define __X10_CONTEXT_MQH__

#include "X10Levels.mqh"
#include "X10Clock.mqh"

#define X10_EMA_SPAN_H1  50   // alpha = 2/51, §6.1
#define X10_VWAP_MIN_BARS 12  // §6.2, annexe A.1

//--- §6.3: ICE weights renormalised to sum 1, and the ICE constant.
//--- Copied from x10_context.py; the sign is +1 when the pair is quoted
//--- USD-base (USDJPY, USDCAD) and -1 otherwise.
#define X10_DXY_TOTAL    (0.576 + 0.136 + 0.119 + 0.091)
#define X10_DXY_W_EUR    (0.576 / X10_DXY_TOTAL)
#define X10_DXY_W_JPY    (0.136 / X10_DXY_TOTAL)
#define X10_DXY_W_GBP    (0.119 / X10_DXY_TOTAL)
#define X10_DXY_W_CAD    (0.091 / X10_DXY_TOTAL)
#define X10_DXY_CONSTANT 50.14348112
#define X10_DXY_FFILL_LIMIT 5   // H1 bars; beyond that the basket is undefined

//+------------------------------------------------------------------+
//| Everything one M5 decision bar carries. Built by the EA, read by |
//| the state machine; one struct so no consumer can silently read a |
//| quantity taken on another bar.                                   |
//+------------------------------------------------------------------+
struct X10Snapshot
{
    long     index;            // monotonic M5 bar counter, the N_* unit
    datetime bar_open;         // server time, open of the decision bar
    datetime fill_bar_open;    // server time, open of the bar a fill lands on
    bool     fill_bar_contiguous; // §2: fill bar opens exactly 300 s later
    double   open, high, low, close;
    double   atr, v, v_prev, m, a;
    double   l_inf, l_sup;
    double   vwap, ema50_h1, atr_h1, dxy, dxy_ema50_h1;
    int      minute_ny;        // of the decision bar open
    long     session_id;
    double   spread;           // current broker spread, in dollars
    bool     ok;               // atr, v, m and a all defined (§5)
};

//+------------------------------------------------------------------+
//| signe() of §6.4: signe(0) = 0, and an undefined value is neutral |
//| rather than poisoning the trace with a NaN.                      |
//+------------------------------------------------------------------+
int X10Sign(double x)
{
    if(!X10Def(x)) return 0;
    if(x > 0.0) return 1;
    if(x < 0.0) return -1;
    return 0;
}

//+------------------------------------------------------------------+
//| Context scores of §6.4, ALWAYS evaluated in the direction of the |
//| TRADE q, never in the approach direction d. For a breakout the   |
//| two coincide, for a reversal they are opposite, and a            |
//| "favourable" context must mean favourable to the position that   |
//| is actually taken.                                               |
//|                                                                  |
//| An undefined reference scores 0 - neutral - as the spec requires |
//| for the VWAP before its 12-bar warmup (§6.2) and for the basket  |
//| with a stale leg (§6.3). Doing the subtraction on the sentinel    |
//| would silently produce a confident -1 instead.                   |
//+------------------------------------------------------------------+
int X10CtxScore(int q, double price, double reference)
{
    if(!X10Def(price) || !X10Def(reference)) return 0;
    return X10Sign(q * (price - reference));
}

//--- ctx_dxy carries the extra minus of §6.4: a dollar above its own
//--- EMA50 is adverse to a long gold trade.
int X10CtxDxy(int q, double dxy, double dxy_ema50_h1)
{
    if(!X10Def(dxy) || !X10Def(dxy_ema50_h1)) return 0;
    return X10Sign(-q * (dxy - dxy_ema50_h1));
}

//+------------------------------------------------------------------+
//| CSessionMeanTypical - the "VWAP" of §6.2.                        |
//|                                                                  |
//| Unweighted cumulative session mean of (H+L+C)/3. Deliberately    |
//| NOT CVWAPDaily (FxIndicatorVWAP.mqh): that one is weighted by    |
//| tick volume and anchored at 00:00 UTC, and gold carries no       |
//| usable volume (§1) - weighting it in one engine only would make  |
//| the three engines incomparable.                                  |
//|                                                                  |
//| Resets on every session change (18:00 New York, DST included)    |
//| and stays undefined for the first 12 bars of session.            |
//+------------------------------------------------------------------+
class CSessionMeanTypical
{
private:
    double m_cum;
    int    m_count;
    long   m_session;
    int    m_min_bars;
    bool   m_started;

public:
    CSessionMeanTypical() : m_cum(0.0), m_count(0), m_session(0),
                            m_min_bars(X10_VWAP_MIN_BARS), m_started(false) {}

    void Init(int min_bars = X10_VWAP_MIN_BARS)
    {
        m_min_bars = min_bars;
        Reset();
    }

    void Reset()
    {
        m_cum     = 0.0;
        m_count   = 0;
        m_session = 0;
        m_started = false;
    }

    //--- Feed one M5 bar, in chronological order.
    void Push(double high, double low, double close, long session_id)
    {
        if(!m_started || session_id != m_session)
        {
            m_session = session_id;
            m_started = true;
            m_cum     = 0.0;
            m_count   = 0;
        }
        m_cum += (high + low + close) / 3.0;
        m_count++;
    }

    //--- Value at the last pushed bar; undefined before the warmup.
    double Value() const
    {
        if(m_count < m_min_bars) return X10_UNDEF;
        return m_cum / m_count;
    }
};

//+------------------------------------------------------------------+
//| Exponential mean, alpha = 2/(span+1), adjust=False, undefined    |
//| before min_periods observations - the pandas ewm() of §6.1.      |
//|                                                                  |
//| An undefined input leaves the recursion untouched and produces   |
//| an undefined output at that index: a basket with a stale leg is  |
//| explicitly neutral (§6.3), and carrying its EMA forward there    |
//| would be scoring a dollar nobody quoted.                         |
//+------------------------------------------------------------------+
void X10EmaSpan(const double &x[], int n, int span, int min_periods,
                double &out[])
{
    ArrayResize(out, n);
    for(int i = 0; i < n; i++) out[i] = X10_UNDEF;
    if(n <= 0 || span < 1) return;

    double alpha = 2.0 / (span + 1.0);
    double run   = 0.0;
    int    seen  = 0;
    for(int i = 0; i < n; i++)
    {
        if(!X10Def(x[i])) continue;
        seen++;
        run = (seen == 1) ? x[i] : run + alpha * (x[i] - run);
        if(seen >= min_periods) out[i] = run;
    }
}

//+------------------------------------------------------------------+
//| §6.1 alignment: index of the last H1 bar whose close time is at  |
//| or before 'm5_open'. Returns -1 when no bar qualifies.           |
//|                                                                  |
//| Equality is deliberate: the H1 bar labelled 09:00 closes exactly |
//| when the M5 bar labelled 10:00 opens, and pandas' shift(1) +     |
//| ffill serves precisely that bar to that M5 bar. The decision     |
//| itself is taken five minutes later, so nothing is read early.    |
//+------------------------------------------------------------------+
int X10AlignH1(const datetime &h1_time[], int n, datetime m5_open)
{
    for(int i = n - 1; i >= 0; i--)
        if((datetime)((long)h1_time[i] + 3600) <= m5_open) return i;
    return -1;
}

//+------------------------------------------------------------------+
//| Four-leg synthetic dollar - §6.3.                                |
//|                                                                  |
//| 50.14348112 * EURUSD^-w_e * USDJPY^w_j * GBPUSD^-w_g * USDCAD^w_c|
//| computed in logs so the four exponents are a single dot product. |
//| Any missing leg makes the basket undefined = neutral, never a    |
//| stale value.                                                     |
//+------------------------------------------------------------------+
double X10Dxy4(double eurusd, double usdjpy, double gbpusd, double usdcad)
{
    if(eurusd <= 0.0 || usdjpy <= 0.0 || gbpusd <= 0.0 || usdcad <= 0.0)
        return X10_UNDEF;
    if(!X10Def(eurusd) || !X10Def(usdjpy) || !X10Def(gbpusd) || !X10Def(usdcad))
        return X10_UNDEF;

    double log_value = MathLog(X10_DXY_CONSTANT)
                     - X10_DXY_W_EUR * MathLog(eurusd)
                     + X10_DXY_W_JPY * MathLog(usdjpy)
                     - X10_DXY_W_GBP * MathLog(gbpusd)
                     + X10_DXY_W_CAD * MathLog(usdcad);
    return MathExp(log_value);
}

#endif // __X10_CONTEXT_MQH__
