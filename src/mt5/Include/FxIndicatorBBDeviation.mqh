//+------------------------------------------------------------------+
//| FxIndicatorBBDeviation.mqh                                       |
//| Bollinger Bands sur série custom (close - VWAP) avec buffer      |
//| circulaire. ddof=1 (équivalent pandas .std() par défaut, qui est |
//| ce que vbt.BBANDS utilise).                                      |
//+------------------------------------------------------------------+
#ifndef __FX_INDICATOR_BBDEV_MQH__
#define __FX_INDICATOR_BBDEV_MQH__

//+------------------------------------------------------------------+
//| CBBDeviation — Bollinger Bands sur valeurs poussées une à une.    |
//| Stocke un buffer circulaire de taille `window`.                   |
//+------------------------------------------------------------------+
class CBBDeviation
{
private:
    int    m_window;        // taille de la fenêtre rolling (80)
    double m_alpha;         // multiplicateur d'écart-type (5.0)
    double m_buf[];         // buffer circulaire des valeurs
    int    m_pos;           // position d'insertion suivante
    int    m_count;         // nombre de valeurs ingérées (saturé à window)

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

    //--- Ingère une nouvelle valeur (ex. close - vwap).
    void Push(double value)
    {
        m_buf[m_pos] = value;
        m_pos = (m_pos + 1) % m_window;
        if(m_count < m_window) m_count++;
    }

    bool IsReady() const { return m_count >= m_window; }

    //--- Calcule mean, upper, lower courants. Retourne false en warmup.
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
        // ddof=1 : variance = (Σx² − (Σx)²/n) / (n-1)
        double var = (s2 - s * s / m_window) / (m_window - 1);
        double std = MathSqrt(MathMax(var, 0.0));
        upper = mean + m_alpha * std;
        lower = mean - m_alpha * std;
        return true;
    }

    int Window() const { return m_window; }
    double Alpha() const { return m_alpha; }
};

#endif // __FX_INDICATOR_BBDEV_MQH__
