# 11 — Porting from Python (vbtpro/pandas → MQL5)

Mapping spécifique au repo `fx_strategies` : comment traduire les idiomes Python utilisés
dans `src/strategies/` en MQL5 idiomatique.

## Table de mapping général

| Python / pandas / vbtpro | MQL5 équivalent |
|--------------------------|-----------------|
| `pd.Series.shift(1)` | Lire `shift = 1` dans `CopyBuffer` ou `CopyClose` |
| `pd.Series.rolling(n).mean()` | Buffer circulaire + somme glissante (cf. `CBBDeviation`) |
| `pd.Series.rolling(n).std(ddof=1)` | Idem avec `var = (Σx² − (Σx)²/n) / (n-1)` |
| `pd.Series.ewm(span=N, adjust=False).mean()` | `iMA(symbol, tf, N, 0, MODE_EMA, PRICE_CLOSE)` |
| `pd.Series.diff()` | `arr[i] - arr[i-1]` manuel sur le buffer |
| `pd.Series.pct_change()` | `(arr[i] - arr[i-1]) / arr[i-1]` |
| `np.log(close[t]/close[t-N])` | `MathLog(close_t / close_t_minus_N)` |
| `np.maximum(a, b)` | `MathMax(a, b)` |
| `np.minimum(a, b)` | `MathMin(a, b)` |
| `np.clip(x, lo, hi)` | `MathMax(lo, MathMin(hi, x))` |
| `pd.Timestamp(...).floor('D')` | `t - (t % 86400)` pour minuit UTC |
| `pd.Timestamp(...).hour` | `MqlDateTime st; TimeToStruct(t, st); st.hour` |
| `vbt.RSI.run(close, window=14)` | `iRSI(symbol, tf, 14, PRICE_CLOSE)` |
| `vbt.BBANDS.run(price, window, alpha)` | `iBands(symbol, tf, window, 0, alpha, PRICE_CLOSE)` |
| `vbt.VWAP.run(high, low, close, volume, anchor='D')` | `CVWAPDaily` custom (cf. 05_indicators_custom.md) |
| `Portfolio.from_signals(...)` | `CTrade::Buy/Sell` + SL/TP serveur |
| `Portfolio.from_optimizer(allocations)` | Recompute lots daily, scaler par symbole |
| `min(target_vol/realized_vol, max_lev)` | `MathMin(target/realized, max_lev)` (identique) |

## Cas concrets du projet

### MR Macro — entry signal

**Python** (`src/strategies/mr_macro.py`) :
```python
upper = vwap + bb_upper(close - vwap, window=80, alpha=5.0)
lower = vwap + bb_lower(close - vwap, window=80, alpha=5.0)
entry_long  = (close < lower) & session_active & macro_ok
entry_short = (close > upper) & session_active & macro_ok
```

**MQL5** :
```mql5
// Dans OnTick après NewBar M1, sur la barre fermée (shift=1):
MqlRates last[];
CopyRates(_Symbol, PERIOD_M1, 1, 1, last);

g_vwap.OnNewBarM1(last[0]);
double dev = last[0].close - g_vwap.Get();
g_bb.Push(dev);

double mean, upper_dev, lower_dev;
if(!g_bb.Compute(mean, upper_dev, lower_dev)) return;
double abs_upper = g_vwap.Get() + upper_dev;
double abs_lower = g_vwap.Get() + lower_dev;

MqlDateTime tm; TimeToStruct(last[0].time, tm);
bool session = (tm.hour >= 6 && tm.hour < 14);

if(session && macro_ok)
{
    if(last[0].close < abs_lower) OpenLong();
    if(last[0].close > abs_upper) OpenShort();
}
```

### TS Momentum — vol-target leverage

**Python** (`src/strategies/daily_momentum.py`) :
```python
def vol_target_leverage(realized_vol, target_vol=0.10, max_leverage=3.0):
    return np.minimum(target_vol / np.maximum(realized_vol, 0.01), max_leverage)
```

**MQL5** :
```mql5
double VolTargetLeverage(double realized_vol, double target_vol, double max_lev)
{
    return MathMin(target_vol / MathMax(realized_vol, 0.01), max_lev);
}
```

`shift(1)` causal : on calcule `realized_vol` sur les returns daily jusqu'à hier, puis
on applique pour les trades d'aujourd'hui. En MQL5 c'est implicite — on calcule au
close d'hier (`OnTimer` daily à 21:05 UTC) et on stocke dans `GlobalVariable` pour
être consommé par les `OnTick` du jour suivant.

### RSI Daily — cross detection

**Python** :
```python
entry_long = (rsi.shift(1) >= 25) & (rsi < 25)
exit_long  = (rsi.shift(1) <= 50) & (rsi > 50)
```

**MQL5** :
```mql5
double rsi_now  = ReadRSI(symbol, /*shift=*/1);  // dernière barre fermée
double rsi_prev = ReadRSI(symbol, /*shift=*/2);  // avant-dernière

bool entry_long = (rsi_prev >= 25.0) && (rsi_now < 25.0);
bool exit_long  = (rsi_prev <= 50.0) && (rsi_now > 50.0);
```

### Vol-targeting global

**Python** (`combined_portfolio_v2.py`) :
```python
sigma21 = port_rets.rolling(21).std() * np.sqrt(252)
sigma63 = port_rets.rolling(63).std() * np.sqrt(252)
realized = np.maximum(np.maximum(sigma21, sigma63), 0.02)
leverage = np.minimum(0.28 / realized, 12.0).shift(1).fillna(1.0)
```

**MQL5** :
```mql5
// Dans OnTimer daily, après reconstruction des returns daily de l'equity:
double sigma21 = AnnualizedStd(rets, 21);
double sigma63 = AnnualizedStd(rets, 63);
double realized = MathMax(MathMax(sigma21, sigma63), 0.02);
double leverage = MathMin(0.28 / realized, 12.0);
GlobalVariableSet("FX_GLOBAL_LEVERAGE", leverage);
// Le shift(1) est implicite : ce levier sera consommé par les trades du jour suivant
```

## Différences fondamentales à connaître

### 1. Pas de DataFrame en MQL5

Python manipule des `pd.DataFrame` indexés par timestamp. MQL5 manipule des `double[]`
indexés par position. Pour reconstruire une "série" :
```mql5
double values[100];
datetime times[100];
ArraySetAsSeries(values, false);
ArraySetAsSeries(times, false);
CopyClose(symbol, PERIOD_D1, 0, 100, values);
CopyTime(symbol, PERIOD_D1, 0, 100, times);
// values[i] correspond à times[i]
```

### 2. Pas de vectorisation

Python : `result = (close - vwap) / vol`. MQL5 : boucle `for(int i; ...)`.

Mais en pratique en MQL5 on calcule un seul point à la fois (la barre courante) →
pas besoin de vectoriser.

### 3. Pas de NaN

`pandas` propage les NaN automatiquement. En MQL5 il faut tester explicitement :
```mql5
if(!MathIsValidNumber(value) || value == 0.0) return;
```

### 4. Coûts de transaction

vbtpro : configurés une fois (`fees=0.0001, slippage=0.00015`) puis appliqués
automatiquement à chaque trade.

MQL5 : pas de "coûts globaux" — le slippage est appliqué par le broker au moment de
l'exécution. La commission est appliquée par le broker en `commission` du deal.

→ Les chiffres de backtest ne matchent pas exactement, c'est attendu.

### 5. Anti look-ahead

Python `shift(1)` est explicite. MQL5 utilise `shift = 1` dans `CopyBuffer` ou lit la
barre fermée (pas la barre en formation). Convention claire : **toujours lire la barre
shift=1**, jamais shift=0 pour les signaux.

## Validation numérique

Pour valider qu'une fonction MQL5 reproduit bien la version Python :

1. **Côté Python** : dump CSV avec timestamps et valeurs intermédiaires
   ```python
   df[["timestamp", "vwap", "bb_upper", "bb_lower", "ema20", "rsi7"]].to_csv("py_dump.csv")
   ```
2. **Côté MQL5** : un script qui lit le CSV et compare bar par bar
3. **Tolérances** :
   - VWAP, BB : 1e-4 (différences d'arrondi accumulées sur 80 barres)
   - EMA : 1e-6
   - RSI : 1e-4 (Wilder smoothing accumule)
   - Signaux booléens : match exact, ou taux de divergence < 1%

## Voir aussi

- [05_indicators_custom.md](./05_indicators_custom.md) — VWAP, Bollinger custom
- [04_indicators_native.md](./04_indicators_native.md) — iMA, iRSI
- [09_strategy_tester.md](./09_strategy_tester.md) — backtest MQL5 vs Python
