//+------------------------------------------------------------------+
//| X10Kinematics.mqh                                                |
//|                                                                  |
//| Velocity, momentum and acceleration in ATR units - spec §5.      |
//| Port of src/framework/x10_kernels.py::kinematics_nb.             |
//|                                                                  |
//|   v[t] = (C[t] - C[t-3])  / (3 * A[t])                           |
//|   m[t] = (C[t] - C[t-12]) / (A[t] * sqrt(12))                    |
//|   a[t] = v[t] - v[t-3]                                           |
//|                                                                  |
//| The three are undefined while A[t] is, or while the lookback is  |
//| not covered; 'a' additionally needs v[t-3], so it warms up three |
//| bars after v. A NaN propagates and freezes the bar (§5), it is   |
//| never substituted by 0.                                          |
//|                                                                  |
//| The ATR is passed in rather than recomputed: the decision bar's  |
//| ATR is the single unit of the whole spec, and a second           |
//| definition here would be one waiting to drift from the first.    |
//+------------------------------------------------------------------+
#ifndef __X10_KINEMATICS_MQH__
#define __X10_KINEMATICS_MQH__

#include "X10Levels.mqh"

#define X10_H_VELOCITY 3
#define X10_H_MOMENTUM 12

void X10Kinematics(const double &close[], const double &atr[], int n,
                   double &v[], double &m[], double &a[],
                   int h_v = X10_H_VELOCITY, int h_m = X10_H_MOMENTUM)
{
    ArrayResize(v, n);
    ArrayResize(m, n);
    ArrayResize(a, n);
    for(int i = 0; i < n; i++)
    {
        v[i] = X10_UNDEF;
        m[i] = X10_UNDEF;
        a[i] = X10_UNDEF;
    }
    if(n <= 0) return;

    double sqrt_h_m = MathSqrt((double)h_m);
    for(int i = 0; i < n; i++)
    {
        double atr_i = atr[i];
        if(!X10Def(atr_i) || atr_i <= 0.0) continue;
        if(i >= h_v) v[i] = (close[i] - close[i - h_v]) / (h_v * atr_i);
        if(i >= h_m) m[i] = (close[i] - close[i - h_m]) / (atr_i * sqrt_h_m);
    }

    for(int i = h_v; i < n; i++)
        if(X10Def(v[i]) && X10Def(v[i - h_v])) a[i] = v[i] - v[i - h_v];
}

//--- Ring size: the deepest lookback any of v/m/a needs is h_m = 12,
//--- plus the current bar and v[t-1] for the deceleration test of §7.3.
#define X10_KIN_RING 16

//+------------------------------------------------------------------+
//| CX10Kinematics - the same three formulas, fed one bar at a time. |
//|                                                                  |
//| X10Kinematics above rebuilds three full arrays to read their last |
//| element; this object keeps only what those formulas reach back    |
//| to: thirteen closes and four velocities. Same divisions, same     |
//| operand order, same undefined-propagation rules, so the values    |
//| are identical to the swept ones, not merely close.                |
//+------------------------------------------------------------------+
class CX10Kinematics
{
private:
    double m_close[X10_KIN_RING];
    double m_v[X10_KIN_RING];
    int    m_head;     // ring slot of the last bar pushed
    long   m_count;    // bars pushed since the seed = absolute index + 1
    int    m_h_v;
    int    m_h_m;
    double m_sqrt_h_m;

    double CloseBack(int back) const
    {
        return m_close[((m_head - back) % X10_KIN_RING + X10_KIN_RING) % X10_KIN_RING];
    }
    double VBack(int back) const
    {
        return m_v[((m_head - back) % X10_KIN_RING + X10_KIN_RING) % X10_KIN_RING];
    }

public:
    CX10Kinematics() { Init(X10_H_VELOCITY, X10_H_MOMENTUM); }

    void Init(int h_v, int h_m)
    {
        m_h_v      = h_v;
        m_h_m      = h_m;
        m_sqrt_h_m = MathSqrt((double)h_m);
        Reset();
    }

    void Reset()
    {
        for(int i = 0; i < X10_KIN_RING; i++)
        {
            m_close[i] = X10_UNDEF;
            m_v[i]     = X10_UNDEF;
        }
        m_head  = 0;
        m_count = 0;
    }

    //--- v[t-1], the only past velocity §7.3 reads outside of 'a'.
    double VPrev() const { return (m_count >= 2) ? VBack(1) : X10_UNDEF; }

    //--- One bar, chronological, with the ATR of that same bar.
    void Push(double close, double atr, double &v, double &m, double &a)
    {
        m_head = (m_head + 1) % X10_KIN_RING;
        m_close[m_head] = close;
        m_count++;
        long i = m_count - 1;   // absolute index, as in the swept version

        v = X10_UNDEF;
        m = X10_UNDEF;
        a = X10_UNDEF;
        if(X10Def(atr) && atr > 0.0)
        {
            if(i >= m_h_v)
                v = (close - CloseBack(m_h_v)) / (m_h_v * atr);
            if(i >= m_h_m)
                m = (close - CloseBack(m_h_m)) / (atr * m_sqrt_h_m);
        }
        m_v[m_head] = v;

        if(i >= m_h_v && X10Def(v) && X10Def(VBack(m_h_v)))
            a = v - VBack(m_h_v);
    }
};

#endif // __X10_KINEMATICS_MQH__
