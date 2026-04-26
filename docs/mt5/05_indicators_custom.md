# 05 — Indicateurs custom

Quand MQL5 n'a pas d'indicateur natif (VWAP, Bollinger sur série custom, RSI exotique,
realized volatility), il faut l'implémenter à la main. Cette page documente les
patterns utilisés dans ce projet.

## VWAP daily-anchored

MQL5 n'a pas de VWAP natif. Le code Python utilise `vbt.VWAP.run()` ancré quotidiennement.

### Algorithme

À chaque nouvelle barre M1 :
1. Si la barre franchit minuit UTC → reset cumulatif
2. Sinon : `cum_pv += typical_price × volume` ; `cum_v += volume`
3. `VWAP = cum_pv / cum_v`

`typical_price = (high + low + close) / 3` (convention vbt).

### Implémentation

```mql5
class CVWAPDaily
{
private:
    double m_cum_pv;
    double m_cum_v;
    double m_last_vwap;
    datetime m_anchor_day;
public:
    void Reset() { m_cum_pv = 0.0; m_cum_v = 0.0; m_anchor_day = 0; m_last_vwap = 0.0; }

    void OnNewBarM1(const MqlRates &bar)
    {
        datetime day = bar.time - (bar.time % 86400);
        if(day != m_anchor_day)
        {
            m_cum_pv = 0.0;
            m_cum_v  = 0.0;
            m_anchor_day = day;
        }
        double tp = (bar.high + bar.low + bar.close) / 3.0;
        double v  = (double)bar.tick_volume;
        m_cum_pv += tp * v;
        m_cum_v  += v;
        m_last_vwap = (m_cum_v > 0.0) ? m_cum_pv / m_cum_v : bar.close;
    }

    double Get() const { return m_last_vwap; }
};
```

### Warmup à `OnInit`

Au démarrage de l'EA en plein milieu de la journée, le cumul est vide. Reconstruire
depuis minuit UTC du jour courant :

```mql5
void WarmupVWAPFromHistory(CVWAPDaily &vwap)
{
    datetime now = TimeGMT();
    datetime midnight = now - (now % 86400);
    MqlRates rates[];
    int copied = CopyRates(_Symbol, PERIOD_M1, midnight, now, rates);
    if(copied <= 0) return;
    for(int i = 0; i < copied; i++)
        vwap.OnNewBarM1(rates[i]);
}
```

### Tick volume vs real volume

Sur FX retail, `MqlRates.real_volume` est presque toujours 0 (broker n'expose pas le
volume vrai). `tick_volume` est un proxy basé sur le nombre de ticks dans la barre.
Le code Python utilise déjà du tick volume — donc cohérent.

## Bollinger Bands sur série custom

MQL5 a `iBands` mais il s'applique sur prix d'un symbole+TF — pas sur une série custom
(comme `close - VWAP`). Il faut implémenter rolling mean/std soi-même.

### Algorithme avec buffer circulaire

```mql5
class CBBDeviation
{
private:
    int    m_window;
    double m_alpha;
    double m_buf[];
    int    m_pos;
    int    m_count;
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

    void Push(double value)
    {
        m_buf[m_pos] = value;
        m_pos = (m_pos + 1) % m_window;
        if(m_count < m_window) m_count++;
    }

    bool Compute(double &mean, double &upper, double &lower)
    {
        if(m_count < m_window) return false;     // warmup
        double s = 0.0, s2 = 0.0;
        for(int i = 0; i < m_window; i++)
        {
            s  += m_buf[i];
            s2 += m_buf[i] * m_buf[i];
        }
        mean = s / m_window;
        // ddof=1 (équivalent pandas std() par défaut)
        double var = (s2 - s * s / m_window) / (m_window - 1);
        double std = MathSqrt(MathMax(var, 0.0));
        upper = mean + m_alpha * std;
        lower = mean - m_alpha * std;
        return true;
    }

    bool IsReady() const { return m_count >= m_window; }
};
```

### ddof=0 vs ddof=1

`pandas.Series.std()` utilise `ddof=1` par défaut (n-1).
`numpy.std()` utilise `ddof=0` par défaut (n).
`vectorbtpro.BBANDS` : à vérifier (souvent ddof=1 mais peut dépendre de la version).

**Si le test numérique diverge** : essayer la formule `var = (s2 - s*s/n) / n` (ddof=0).

## Rolling realized volatility

Pour le vol-targeting global :

```mql5
class CRollingVol
{
private:
    double m_buf[];
    int    m_pos, m_count, m_window;
public:
    void Init(int window) { m_window = window; ArrayResize(m_buf, window); m_pos = 0; m_count = 0; }
    void Push(double daily_return)
    {
        m_buf[m_pos] = daily_return;
        m_pos = (m_pos + 1) % m_window;
        if(m_count < m_window) m_count++;
    }
    // Annualized via sqrt(252)
    double AnnualizedStd()
    {
        if(m_count < m_window) return 0.0;
        double s = 0.0, s2 = 0.0;
        for(int i = 0; i < m_window; i++) { s += m_buf[i]; s2 += m_buf[i]*m_buf[i]; }
        double mean = s / m_window;
        double var  = (s2 - s*s/m_window) / (m_window - 1);
        return MathSqrt(MathMax(var, 0.0)) * MathSqrt(252.0);
    }
};
```

## EMA équivalent pandas `ewm(span, adjust=False)`

L'`iMA(MODE_EMA)` de MQL5 utilise la formule :

```
EMA[t] = (close[t] * 2 + EMA[t-1] * (period - 1)) / (period + 1)
```

C'est strictement équivalent à `pandas.Series.ewm(span=period, adjust=False).mean()`.

À vérifier numériquement sur 250 bars avec un dump CSV — tolérance attendue ≈ 1e-6.

## Pattern : appel à chaque nouvelle barre

```mql5
static datetime g_last_bar = 0;
datetime current = iTime(_Symbol, PERIOD_M1, 0);

if(current != g_last_bar)
{
    MqlRates last[];
    if(CopyRates(_Symbol, PERIOD_M1, 1, 1, last) == 1)
    {
        g_vwap.OnNewBarM1(last[0]);
        double dev = last[0].close - g_vwap.Get();
        g_bb.Push(dev);
        // ...signal...
    }
    g_last_bar = current;
}
```

## Validation numérique vs Python

Pour valider un indicateur custom contre la version Python :
1. Côté Python : `df[["timestamp", "vwap", "bb_upper", "bb_lower"]].to_csv("py_dump.csv")`
2. Côté MQL5 : un script lit `py_dump.csv` et compare bar par bar
3. Tolérance : 1e-4 sur VWAP/BB (différences d'arrondi accumulées)

## Voir aussi

- [04_indicators_native.md](./04_indicators_native.md) — iMA, iRSI, iBands
- [11_porting_from_python.md](./11_porting_from_python.md) — table de mapping pandas → MQL5
