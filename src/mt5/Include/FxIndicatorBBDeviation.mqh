//+------------------------------------------------------------------+
//| FxIndicatorBBDeviation.mqh                                       |
//|                                                                  |
//| Bollinger Bands computed on a custom value series via a circular |
//| buffer. Variance uses ddof=1 (the pandas default), which is what |
//| typical Python references rely on.                               |
//+------------------------------------------------------------------+
#ifndef __FX_INDICATOR_BBDEV_MQH__
#define __FX_INDICATOR_BBDEV_MQH__

//+------------------------------------------------------------------+
//| CBBDeviation: Bollinger Bands with a streaming circular buffer.  |
//| Push() appends one value; Compute() returns mean ± alpha * std.  |
//+------------------------------------------------------------------+
class CBBDeviation
{
private:
    int    m_window;        // rolling window size
    double m_alpha;         // standard-deviation multiplier (band width)
    double m_buf[];         // circular buffer of recent values
    int    m_pos;           // next insertion index
    int    m_count;         // values ingested (saturates at m_window)

public:
    void Init(int window, double alpha)
    {
        m_window = window;
        m_alpha  = alpha;
        ArrayResize(m_buf, window);
        ArrayInitialize(m_buf, 0.0);
        m_pos = 0;
        m_count = 0;
    }

    //--- Append a value (typically `close - vwap`).
    void Push(double value)
    {
        m_buf[m_pos] = value;
        m_pos = (m_pos + 1) % m_window;
        if(m_count < m_window) m_count++;
    }

    bool IsReady() const { return m_count >= m_window; }

    //--- Compute the current mean and band edges. Returns false during
    //--- the warm-up phase, before the buffer has been filled.
    bool Compute(double &mean, double &upper, double &lower)
    {
        if(m_count < m_window) return false;
        double s = 0.0, s2 = 0.0;
        for(int i = 0; i < m_window; i++)
        {
            s  += m_buf[i];
            s2 += m_buf[i] * m_buf[i];
        }
        mean = s / m_window;
        double var = (s2 - s * s / m_window) / (m_window - 1);
        double std = MathSqrt(MathMax(var, 0.0));
        upper = mean + m_alpha * std;
        lower = mean - m_alpha * std;
        return true;
    }

    int    Window() const { return m_window; }
    double Alpha() const  { return m_alpha; }
};

#endif // __FX_INDICATOR_BBDEV_MQH__
