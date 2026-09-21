//+------------------------------------------------------------------+
//| X10Levels.mqh                                                    |
//|                                                                  |
//| The x10 grid (§3) and the Wilder ATR that normalises every       |
//| distance of the spec (§4). Line-by-line port of                  |
//| src/framework/x10_kernels.py::x10_levels_nb / atr_wilder_nb -    |
//| the reference implementation the three engines reconcile on.     |
//|                                                                  |
//| iATR is FORBIDDEN here (§4): it seeds on the simple mean of the  |
//| first 14 true ranges, this recursion seeds on the first TR, and  |
//| the difference propagates to every threshold of the spec.        |
//+------------------------------------------------------------------+
#ifndef __X10_LEVELS_MQH__
#define __X10_LEVELS_MQH__

//--- Undefined is NOT zero anywhere in this strategy (§4, §5): a NaN
//--- freezes the bar instead of scoring it. MQL5 has no NaN literal we
//--- can compare safely, so EMPTY_VALUE is the sentinel and X10Def the
//--- only admitted test.
#define X10_UNDEF EMPTY_VALUE

bool X10Def(double v)
{
    return (v != X10_UNDEF && MathIsValidNumber(v));
}

//--- Broker grid: XAUUSD quotes to the mil (digits=3, point=0.001) and
//--- the x10 levels sit 10 000 points apart. Integer arithmetic on
//--- points is what keeps 4409.999999996 on the correct side of 4410.
#define X10_POINTS_PER_UNIT  1000
#define X10_POINTS_PER_LEVEL 10000
#define X10_LEVEL_SIZE       10.0
#define X10_ATR_PERIOD       14

//+------------------------------------------------------------------+
//| L_inf = floor(round(C*1000) / 10000) * 10, L_sup = L_inf + 10.   |
//| A price sitting exactly on a level opens the box above it, so    |
//| L_inf <= C < L_sup holds without a special case.                 |
//+------------------------------------------------------------------+
double X10LevelInf(double close)
{
    long points  = (long)MathRound(close * X10_POINTS_PER_UNIT);
    long floored = (points / X10_POINTS_PER_LEVEL) * X10_POINTS_PER_LEVEL;
    return (double)floored / X10_POINTS_PER_UNIT;
}

double X10LevelSup(double close)
{
    return X10LevelInf(close) + X10_LEVEL_SIZE;
}

//+------------------------------------------------------------------+
//| Wilder ATR (alpha = 1/period, recursive) - spec §4.              |
//|                                                                  |
//| TR[0] is undefined (it needs C[-1]), so the recursion is seeded  |
//| on TR[1] and the output stays undefined until 'period' true      |
//| ranges have been seen: the first defined value lands on bar      |
//| 'period'. Undefined, never 0 - a zero ATR would make every       |
//| distance of the spec infinite in ATR units.                      |
//|                                                                  |
//| Arrays are chronological (index 0 = oldest), NOT series-indexed. |
//+------------------------------------------------------------------+
void X10WilderATR(const double &high[], const double &low[],
                  const double &close[], int n, int period, double &atr[])
{
    ArrayResize(atr, n);
    for(int i = 0; i < n; i++) atr[i] = X10_UNDEF;
    if(n <= 0 || period < 1) return;

    double alpha = 1.0 / period;
    double run   = 0.0;
    int    seen  = 0;
    for(int i = 1; i < n; i++)
    {
        double prev = close[i - 1];
        double tr   = high[i] - low[i];
        double up   = MathAbs(high[i] - prev);
        double dn   = MathAbs(low[i] - prev);
        if(up > tr) tr = up;
        if(dn > tr) tr = dn;

        seen++;
        run = (seen == 1) ? tr : run + alpha * (tr - run);
        if(seen >= period) atr[i] = run;
    }
}

#endif // __X10_LEVELS_MQH__
