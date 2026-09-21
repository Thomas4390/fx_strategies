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

#endif // __X10_KINEMATICS_MQH__
