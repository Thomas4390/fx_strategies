//+------------------------------------------------------------------+
//| X10StateMachine.mqh                                              |
//|                                                                  |
//| The automaton of §7 and the entry filters of §6.4 / §8 / §10,    |
//| ported state by state, counter by counter and in the same order  |
//| as src/framework/x10_engine.py::x10_engine_nb - the reference    |
//| the MT5 trace is reconciled against event by event (§13).        |
//|                                                                  |
//| Three states (IDLE / ARMED / BROKEN), one armed level at a time, |
//| one position at a time. The counters N_arm, N_hold and N_sweep   |
//| are DIFFERENCES OF BAR INDICES, not event counters: they keep    |
//| running through bars frozen by an undefined indicator (§5), so   |
//| a sequence too broken up expires instead of surviving forever.   |
//|                                                                  |
//| What this class does NOT do: size the trade and send the order.  |
//| Sizing needs the broker (volume step, contract size, equity) and |
//| lives in the EA, which emits the ENTRY / CANCEL(size) rows.      |
//|                                                                  |
//| Ordering of the entry filters, normative (§0, §8):               |
//|     context (§6.4) -> R (§8) -> time window (§10) -> size (§11)  |
//+------------------------------------------------------------------+
#ifndef __X10_STATE_MACHINE_MQH__
#define __X10_STATE_MACHINE_MQH__

#include "X10Levels.mqh"
#include "X10Kinematics.mqh"
#include "X10Context.mqh"
#include "X10Trace.mqh"

//--- Annexe A.1 - frozen, and therefore #define and not inputs. A
//--- frozen parameter a caller can override is a frozen parameter
//--- waiting to be optimised.
#define X10_V_MIN         0.2   // §7.1
#define X10_K_B           0.25  // §7.2 breakout strength
#define X10_S_MIN         0.3   // §7.3 sweep depth
#define X10_N_ARM         12    // §7.1 arming lifetime, in M5 bars
#define X10_N_HOLD        3     // §7.2 hold window
#define X10_N_SWEEP       3     // §7.3 reintegration window
#define X10_REV_STOP_ATR  0.5   // §7.3 stop = extreme + 0.5 ATR
#define X10_R_MIN         1.0   // §8
#define X10_MAX_HOLD_BARS 48    // §10, four hours of M5
#define X10_COOLDOWN_BARS 6     // §9, per level
#define X10_LEVERAGE_CAP  100.0 // §11, notional capped at 1:100 of equity
#define X10_COOLDOWN_SLOTS 16

#define X10_ST_IDLE   0
#define X10_ST_ARMED  1
#define X10_ST_BROKEN 2

//+------------------------------------------------------------------+
//| What the automaton hands to the EA when a trade is decided.      |
//+------------------------------------------------------------------+
struct X10Intent
{
    bool   enter;
    int    q;          // +1 long, -1 short - the TRADE direction
    int    d;          // approach direction of the armed level
    int    scenario;
    double level;
    double stop;
    double target;
    double r_est;
    int    ctx_ema, ctx_vwap, ctx_dxy;
    bool   retest;
    bool   extension_vwap;
};

class CX10StateMachine
{
private:
    //--- grid of annexe A.2, the only three tunable numbers
    double m_z;
    double m_a_min;
    double m_k_s;
    double m_slippage;      // §12 sensitivity, per side, in dollars

    CX10Trace *m_trace;

    int    m_state;
    double m_level;
    int    m_d;
    long   m_i_arm;
    long   m_i_break;
    bool   m_has_exc;
    long   m_i_exc;
    double m_extreme;
    bool   m_retest;

    //--- §9 cooldown ring: 6 bars per level, at most one opened per
    //--- bar, so a handful of slots always suffices; scanned in full.
    double m_cd_level[X10_COOLDOWN_SLOTS];
    long   m_cd_until[X10_COOLDOWN_SLOTS];
    int    m_cd_next;

    //--- Fill the columns every event shares, scored in the direction
    //--- of the trade q (§6.4).
    void FillCommon(X10Event &e, const X10Snapshot &s, int q, int d, double level)
    {
        e.scenario = X10_SC_NONE;
        e.level    = level;
        e.d        = d;
        e.atr      = s.atr;
        e.v        = s.v;
        e.m        = s.m;
        e.a        = s.a;
        e.ctx_ema  = X10CtxScore(q, s.close, s.ema50_h1);
        e.ctx_vwap = X10CtxScore(q, s.close, s.vwap);
        e.ctx_dxy  = X10CtxDxy(q, s.dxy, s.dxy_ema50_h1);
        e.vwap     = s.vwap;
        e.ema50_h1 = s.ema50_h1;
        e.dxy      = s.dxy;
    }

    bool CooldownBlocked(double level, long index) const
    {
        for(int c = 0; c < X10_COOLDOWN_SLOTS; c++)
            if(m_cd_until[c] >= index && m_cd_level[c] == level) return true;
        return false;
    }

public:
    CX10StateMachine() : m_z(1.0), m_a_min(0.2), m_k_s(1.0), m_slippage(0.0),
                         m_trace(NULL) { Clear(); }

    void Init(double z, double a_min, double k_s, double slippage_usd,
              CX10Trace *trace)
    {
        m_z        = z;
        m_a_min    = a_min;
        m_k_s      = k_s;
        m_slippage = slippage_usd;
        m_trace    = trace;
        Clear();
    }

    void Clear()
    {
        m_state    = X10_ST_IDLE;
        m_level    = X10_UNDEF;
        m_d        = 0;
        m_i_arm    = -1;
        m_i_break  = -1;
        m_has_exc  = false;
        m_i_exc    = -1;
        m_extreme  = X10_UNDEF;
        m_retest   = false;
        m_cd_next  = 0;
        for(int c = 0; c < X10_COOLDOWN_SLOTS; c++)
        {
            m_cd_level[c] = X10_UNDEF;
            m_cd_until[c] = -1;
        }
    }

    //--- Back to IDLE without touching the cooldown ring: used after an
    //--- entry, which owns no cooldown until it exits (§9).
    void ToIdle() { m_state = X10_ST_IDLE; }

    int    State() const { return m_state; }
    double Level() const { return m_level; }
    int    Direction() const { return m_d; }

    void RegisterCooldown(double level, long index)
    {
        m_cd_level[m_cd_next] = level;
        m_cd_until[m_cd_next] = index + X10_COOLDOWN_BARS;
        m_cd_next = (m_cd_next + 1) % X10_COOLDOWN_SLOTS;
    }

    //+--------------------------------------------------------------+
    //| One M5 decision bar. Emits every ARM / BREAK / SWEEP / CANCEL |
    //| row itself and returns true when the EA must open a position  |
    //| at the open of the next bar.                                  |
    //+--------------------------------------------------------------+
    bool Decide(const X10Snapshot &s, X10Intent &intent)
    {
        intent.enter          = false;
        intent.q              = 0;
        intent.d              = m_d;
        intent.scenario       = X10_SC_NONE;
        intent.level          = X10_UNDEF;
        intent.stop           = X10_UNDEF;
        intent.target         = X10_UNDEF;
        intent.r_est          = X10_UNDEF;
        intent.ctx_ema        = 0;
        intent.ctx_vwap       = 0;
        intent.ctx_dxy        = 0;
        intent.retest         = false;
        intent.extension_vwap = false;

        //--- §5: a bar with an undefined ATR, v, m or a freezes every
        //--- transition. The counters below keep running anyway.
        if(!s.ok) return false;

        double close = s.close;
        double atr_i = s.atr;
        X10Event e;

        //================================================ IDLE -> ARMED
        if(m_state == X10_ST_IDLE)
        {
            int    cand_d     = 0;
            double cand_level = X10_UNDEF;
            for(int side = 0; side < 2; side++)
            {
                int    dd = (side == 0) ? 1 : -1;
                double lv = (dd == 1) ? s.l_sup : s.l_inf;
                double gap = dd * (lv - close);
                if(gap <= 0.0 || gap > m_z * atr_i) continue;
                if(dd * s.v < X10_V_MIN) continue;
                if(dd * s.a < m_a_min) continue;
                if(dd * s.m <= 0.0) continue;
                if(CooldownBlocked(lv, s.index)) continue;
                cand_d     = dd;
                cand_level = lv;
                break;
            }
            if(cand_d == 0) return false;

            m_d       = cand_d;
            m_level   = cand_level;
            m_i_arm   = s.index;
            m_state   = X10_ST_ARMED;
            m_retest  = false;
            m_has_exc = false;
            m_i_exc   = -1;
            m_extreme = X10_UNDEF;

            //--- The trace carries the scores evaluated at q = d, the
            //--- only trade conceivable while nothing is identified.
            X10EventInit(e, s.bar_open, X10_EV_ARM);
            FillCommon(e, s, m_d, m_d, m_level);
            if(m_trace != NULL) m_trace.Write(e);

            //--- §7.3: the reintegration window counts from the first
            //--- bar whose extreme goes past the level, possibly this one.
            double ext_i = (m_d == 1) ? s.high : s.low;
            if(m_d * (ext_i - m_level) > 0.0)
            {
                m_has_exc = true;
                m_i_exc   = s.index;
                m_extreme = ext_i;
            }
            return false;
        }

        //=============================== busy on m_level / m_d
        int scn_break = (m_d == 1) ? X10_SC_BREAK_LONG : X10_SC_BREAK_SHORT;
        int scn_rev   = (m_d == 1) ? X10_SC_REV_SHORT  : X10_SC_REV_LONG;
        double bar_ext = (m_d == 1) ? s.high : s.low;

        bool   emitted       = false;
        int    cancel_reason = X10_CR_NONE;
        int    cancel_scn    = X10_SC_NONE;
        int    entry_q       = 0;
        int    entry_scn     = X10_SC_NONE;
        double entry_stop    = X10_UNDEF;
        double entry_target  = X10_UNDEF;
        bool   entry_ext     = false;
        bool   sweep_taken   = false;

        if(m_state == X10_ST_ARMED)
        {
            //--- §7.2: strength AND real close, i.e. C in the
            //--- directional half of the bar.
            bool broke = (m_d * (close - m_level) >= X10_K_B * atr_i)
                      && (m_d * (close - 0.5 * (s.high + s.low)) > 0.0);
            if(broke)
            {
                m_state   = X10_ST_BROKEN;
                m_i_break = s.index;
                m_extreme = bar_ext;
                m_has_exc = false;
                m_i_exc   = -1;
                m_retest  = false;

                X10EventInit(e, s.bar_open, X10_EV_BREAK);
                FillCommon(e, s, m_d, m_d, m_level);
                e.scenario = scn_break;
                if(m_trace != NULL) m_trace.Write(e);
                return false;
            }

            if(!m_has_exc)
            {
                if(m_d * (bar_ext - m_level) > 0.0)
                {
                    m_has_exc = true;
                    m_i_exc   = s.index;
                    m_extreme = bar_ext;
                }
            }
            else if(m_d * (bar_ext - m_extreme) > 0.0)
                m_extreme = bar_ext;

            if(m_has_exc && s.index - m_i_exc >= X10_N_SWEEP)
            {
                emitted       = true;
                cancel_reason = X10_CR_NSWEEP;
            }
            else if(m_has_exc && m_d * (close - m_level) < 0.0)
            {
                //--- Direct sweep: depth, deceleration and flip, all
                //--- three, at the reintegration bar (§7.3).
                bool deep  = m_d * (m_extreme - m_level) >= X10_S_MIN * atr_i;
                bool decel = X10Def(s.v_prev) && (m_d * s.v < m_d * s.v_prev);
                bool flip  = m_d * s.a <= -m_a_min;
                if(deep && decel && flip)
                {
                    sweep_taken  = true;
                    entry_q      = -m_d;
                    entry_scn    = scn_rev;
                    entry_target = m_level - X10_LEVEL_SIZE * m_d;
                    entry_stop   = m_extreme + X10_REV_STOP_ATR * atr_i * m_d;
                }
                else
                {
                    emitted       = true;
                    cancel_reason = X10_CR_SWEEP;
                }
            }
            else if(m_d * (m_level - close) > m_z * atr_i)
            {
                emitted       = true;
                cancel_reason = X10_CR_ZONE;
            }
            else if(s.index - m_i_arm >= X10_N_ARM)
            {
                emitted       = true;
                cancel_reason = X10_CR_NARM;
            }
        }
        else // X10_ST_BROKEN - the hold window of §7.2
        {
            if(m_d * (bar_ext - m_extreme) > 0.0) m_extreme = bar_ext;
            if((m_d == 1 && s.low <= m_level) || (m_d == -1 && s.high >= m_level))
                m_retest = true;   // traced, never required (§7.2)

            if(m_d * (close - m_level) < 0.0)
            {
                //--- Failed breakout (§7.2, points 1-5): cancel it, and
                //--- THIS bar is the candidate reintegration of a
                //--- reversal - no re-arming, no N_sweep countdown.
                emitted       = true;
                cancel_reason = X10_CR_HOLD;
                cancel_scn    = scn_break;
                bool deep  = m_d * (m_extreme - m_level) >= X10_S_MIN * atr_i;
                bool decel = X10Def(s.v_prev) && (m_d * s.v < m_d * s.v_prev);
                bool flip  = m_d * s.a <= -m_a_min;
                if(deep && decel && flip)
                {
                    sweep_taken  = true;
                    entry_q      = -m_d;
                    entry_scn    = scn_rev;
                    entry_target = m_level - X10_LEVEL_SIZE * m_d;
                    entry_stop   = m_extreme + X10_REV_STOP_ATR * atr_i * m_d;
                }
            }
            else if(s.index - m_i_break >= X10_N_HOLD)
            {
                entry_q      = m_d;
                entry_scn    = scn_break;
                entry_target = m_level + X10_LEVEL_SIZE * m_d;
                entry_stop   = m_level - m_d * m_k_s * atr_i;
            }
        }

        if(emitted)
        {
            X10EventInit(e, s.bar_open, X10_EV_CANCEL);
            FillCommon(e, s, m_d, m_d, m_level);
            e.scenario      = cancel_scn;
            e.cancel_reason = cancel_reason;
            if(m_trace != NULL) m_trace.Write(e);
            if(!sweep_taken)
            {
                m_state = X10_ST_IDLE;
                RegisterCooldown(m_level, s.index);
                return false;
            }
        }

        if(entry_q == 0) return false;

        //=============================== entry candidate
        int q        = entry_q;
        int ctx_ema  = X10CtxScore(q, close, s.ema50_h1);
        int ctx_vwap = X10CtxScore(q, close, s.vwap);
        int ctx_dxy  = X10CtxDxy(q, s.dxy, s.dxy_ema50_h1);

        if(sweep_taken)
        {
            X10EventInit(e, s.bar_open, X10_EV_SWEEP);
            FillCommon(e, s, q, m_d, m_level);
            e.scenario = entry_scn;
            e.ctx_ema  = ctx_ema;
            e.ctx_vwap = ctx_vwap;
            e.ctx_dxy  = ctx_dxy;
            e.stop     = entry_stop;
            e.target   = entry_target;
            if(m_trace != NULL) m_trace.Write(e);

            //--- §7.3 VWAP extension: traced and measured, never blocking.
            if(X10Def(s.vwap) && X10Def(s.atr_h1)
               && MathAbs(close - s.vwap) >= s.atr_h1)
                entry_ext = true;
        }

        int    refuse = X10_CR_NONE;
        double r_est  = X10_UNDEF;

        //--- 1. context (§6.4): breakouts require it, reversals never do.
        if(entry_scn == X10_SC_BREAK_LONG || entry_scn == X10_SC_BREAK_SHORT)
            if(ctx_ema <= 0 || ctx_vwap <= 0) refuse = X10_CR_CTX;

        //--- 2. R (§8): one half-spread, once, on each side of the ratio.
        //--- Inp_SlippageUSD is the §12 sensitivity knob and is 0 in the
        //--- reference configuration, where this reduces to the Python
        //--- formula exactly.
        if(refuse == X10_CR_NONE)
        {
            double half  = 0.5 * s.spread + m_slippage;
            double denom = MathAbs(close - entry_stop) + half;
            double num   = MathAbs(entry_target - close) - half;
            if(denom > 0.0) r_est = num / denom;
            if(!X10Def(r_est) || r_est < X10_R_MIN) refuse = X10_CR_R;
        }

        //--- 3. time window (§10), read on the bar the fill lands on.
        //--- A fill bar that does not open exactly 300 s after the
        //--- decision bar is a data hole or a session reopen: the fill
        //--- of §2 does not exist, so the candidate is cancelled.
        if(refuse == X10_CR_NONE
           && (!s.fill_bar_contiguous || X10IsEntryBlocked(s.fill_bar_open)))
            refuse = X10_CR_WINDOW;

        if(refuse != X10_CR_NONE)
        {
            X10EventInit(e, s.bar_open, X10_EV_CANCEL);
            FillCommon(e, s, q, m_d, m_level);
            e.scenario      = entry_scn;
            e.ctx_ema       = ctx_ema;
            e.ctx_vwap      = ctx_vwap;
            e.ctx_dxy       = ctx_dxy;
            e.stop          = entry_stop;
            e.target        = entry_target;
            e.r_est         = r_est;
            e.cancel_reason = refuse;
            if(m_trace != NULL) m_trace.Write(e);

            m_state = X10_ST_IDLE;
            RegisterCooldown(m_level, s.index);
            return false;
        }

        //--- The candidate survived context, R and the window. Sizing
        //--- (§11) is the EA's, and so is the ENTRY row.
        intent.enter          = true;
        intent.q              = q;
        intent.d              = m_d;
        intent.scenario       = entry_scn;
        intent.level          = m_level;
        intent.stop           = entry_stop;
        intent.target         = entry_target;
        intent.r_est          = r_est;
        intent.ctx_ema        = ctx_ema;
        intent.ctx_vwap       = ctx_vwap;
        intent.ctx_dxy        = ctx_dxy;
        intent.retest         = m_retest;
        intent.extension_vwap = entry_ext;
        m_state = X10_ST_IDLE;
        return true;
    }
};

#endif // __X10_STATE_MACHINE_MQH__
