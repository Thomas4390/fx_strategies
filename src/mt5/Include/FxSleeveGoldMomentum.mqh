//+------------------------------------------------------------------+
//| FxSleeveGoldMomentum.mqh                                         |
//|                                                                  |
//| Sleeve 5: daily time-series momentum, on gold and on whatever    |
//| other instruments the same engine is pointed at.                 |
//|                                                                  |
//| Specification (mirrors src/strategies/gold_momentum.py):         |
//|   * Universe        : Inp_Gold_Symbols, a CSV of instruments.    |
//|                       Default "XAUUSD", i.e. one symbol and the  |
//|                       historical behaviour bit for bit.          |
//|   * Score           : mean of sign(return over N) for N in       |
//|                       {40, 60, 120, 250} D1 bars, in [-1, +1]    |
//|   * Long signal     : score > 0                                  |
//|   * Short signal    : score < 0, disabled by default             |
//|   * Exit            : signal flip                                |
//|   * Sizing          : FLAT, sub-equity split equally across the  |
//|                       instruments. Martingale and grid overlays  |
//|                       were tested in Python and rejected on tail |
//|                       risk; read the research note first.        |
//|   * Vol target      : min(target / sigma21, max leverage)        |
//|                                                                  |
//| The lookbacks are averaged rather than selected: replacing a     |
//| fitted choice with an aggregate is what keeps this signal from   |
//| being an overfit. Do NOT grid-search them.                       |
//|                                                                  |
//| 2026-07-26 — that rule was tested and it held. A CAGR mandate    |
//| moved the grid to {15,30,60}, which looked better in vbt (CAGR   |
//| 40.7% vs 33.5%) and was far worse HERE (17.3% vs 36.4%), because |
//| the short grid flips twice as often (66 trades vs 31) and vbt    |
//| fills at the same close it decided on while this tester fills at |
//| the next open. Reverted. Only the sizing was kept.               |
//|                                                                  |
//| A lookback slot set to 0 is disabled, so the count is variable   |
//| (1..4) and combinations can be tested from the Strategy Tester   |
//| without recompiling — which is how the above was measured.       |
//|                                                                  |
//| Long-only is the default because gold carries a structural       |
//| positive drift; enabling shorts cost 3.8 pp of return and        |
//| doubled the drawdown in the Python study.                        |
//|                                                                  |
//| 2026-07-27 — the sleeve went multi-instrument. Two rules keep    |
//| that from changing anything when a single symbol is configured:  |
//| the per-instrument budget divides by the number of CONFIGURED    |
//| symbols and never by the number of ACTIVE ones, so leverage does |
//| not creep up when an instrument sits out its warmup or is        |
//| skipped by the broker; and the gold-calibrated constants         |
//| (SafetySL, the sigma fallback, the vol floor, the average nights |
//| held) apply unchanged to every instrument — they were measured   |
//| symbol by symbol in the 2026-07-27 sweep and held there.         |
//+------------------------------------------------------------------+
#ifndef __FX_SLEEVE_GOLD_MOMENTUM_MQH__
#define __FX_SLEEVE_GOLD_MOMENTUM_MQH__

#include <Trade/Trade.mqh>
#include "FxSleeveBase.mqh"
#include "FxCommon.mqh"
#include "FxRiskManager.mqh"
#include "FxTradeHelpers.mqh"

//--- Input slots available; a slot set to 0 is skipped, so the effective
//--- count is m_n_lookbacks and the history need is derived from the
//--- longest ACTIVE lookback rather than from the hard cap.
#define FX_GOLD_N_SLOTS      4
#define FX_GOLD_MAX_LOOKBACK 250
#define FX_GOLD_HISTORY_CAP  (FX_GOLD_MAX_LOOKBACK + 2)
#define FX_GOLD_MAX_SYMBOLS  8

class CSleeveGoldMomentum : public CSleeveBase
{
private:
    string m_symbols[FX_GOLD_MAX_SYMBOLS];
    int    m_n_symbols;      // instruments that survived the init filtering
    int    m_n_configured;   // instruments listed in Inp_Gold_Symbols
    int    m_lookbacks[FX_GOLD_N_SLOTS];
    int    m_n_lookbacks;    // active slots (those left > 0)
    int    m_history_bars;   // longest active lookback + 2
    CTrade m_trade;
    bool   m_trace_warned;   // warn once, not once per session

public:
    bool Init() override
    {
        m_magic = Inp_MagicGold;
        m_name  = "Gold_Momentum";
        m_trace_warned = false;

        // A slot at 0 is "unused", not "invalid": it lets the tester sweep
        // 3-lookback and 4-lookback grids off the same compiled EA. Anything
        // else out of range is still a hard error.
        int slots[FX_GOLD_N_SLOTS];
        slots[0] = Inp_Gold_LookbackA;
        slots[1] = Inp_Gold_LookbackB;
        slots[2] = Inp_Gold_LookbackC;
        slots[3] = Inp_Gold_LookbackD;

        m_n_lookbacks = 0;
        int longest = 0;
        for(int i = 0; i < FX_GOLD_N_SLOTS; i++)
        {
            if(slots[i] == 0) continue;
            if(slots[i] < 0 || slots[i] > FX_GOLD_MAX_LOOKBACK)
            {
                g_logger.Error(m_name, StringFormat(
                    "invalid lookback[%d]=%d (expected 0 to disable, or 1..%d)",
                    i, slots[i], FX_GOLD_MAX_LOOKBACK));
                return false;
            }
            m_lookbacks[m_n_lookbacks++] = slots[i];
            if(slots[i] > longest) longest = slots[i];
        }
        if(m_n_lookbacks == 0)
        {
            g_logger.Error(m_name,
                "every lookback slot is 0; the sleeve has no signal");
            return false;
        }
        m_history_bars = longest + 2;

        string raw[];
        int n = SplitCsv(Inp_Gold_Symbols, raw);
        if(n <= 0 || n > FX_GOLD_MAX_SYMBOLS)
        {
            g_logger.Error(m_name, StringFormat("invalid Inp_Gold_Symbols=%s",
                                                Inp_Gold_Symbols));
            return false;
        }
        // The risk budget is divided by the CONFIGURED count, fixed here once
        // and for all. Dividing by the ACTIVE count instead would silently
        // re-lever the survivors whenever an instrument drops out (warmup,
        // broker outage), which is exactly the kind of drift a backtest never
        // shows and live trading pays for.
        m_n_configured = n;

        // Pack valid instruments into [0..valid_n-1]. An unusable instrument
        // is dropped with a warning, never a hard failure: the sleeve keeps
        // trading the others.
        int valid_n = 0;
        for(int i = 0; i < n; i++)
        {
            string sym = MakeSymbolWithSuffix(raw[i], Inp_SymbolSuffix);
            if(!EnsureSymbolSelected(sym))
            {
                // Try the bare name: metals and indices often carry no broker
                // suffix even when FX pairs do.
                sym = raw[i];
                if(!EnsureSymbolSelected(sym))
                {
                    g_logger.Warn(m_name, StringFormat(
                        "%s: skipped (not selectable, tried with and without "
                        "suffix '%s')", raw[i], Inp_SymbolSuffix));
                    continue;
                }
            }

            // No history yet is NOT a reason to drop the instrument: in the
            // tester, a symbol whose broker data starts inside the window has
            // zero D1 bars at Init time (simulated clock), and ComputeScore
            // already skips bar by bar until the warmup fills. Dropping here
            // silently killed XAGUSD on any window opening before 2022-11.
            if(!EnsureHistory(sym, PERIOD_D1, m_history_bars))
                g_logger.Warn(m_name, StringFormat(
                    "%s: %d/%d D1 bars; the instrument stays flat until "
                    "history accumulates", sym,
                    (int)Bars(sym, PERIOD_D1), m_history_bars));
            m_symbols[valid_n++] = sym;
        }
        if(valid_n == 0)
        {
            g_logger.Error(m_name,
                "no usable instrument after filtering; sleeve disabled");
            return false;
        }
        m_n_symbols = valid_n;

        string lb_desc = "";
        for(int i = 0; i < m_n_lookbacks; i++)
            lb_desc += StringFormat("%s%d", (i > 0 ? "/" : ""), m_lookbacks[i]);

        string sym_desc = "";
        for(int i = 0; i < m_n_symbols; i++)
            sym_desc += StringFormat("%s%s", (i > 0 ? "/" : ""), m_symbols[i]);

        m_trade.SetExpertMagicNumber(m_magic);
        m_trade.SetDeviationInPoints(FX_DEFAULT_DEVIATION);
        g_logger.Info(m_name, StringFormat(
            "Init OK symbols=%s (%d/%d configured) lookbacks=%s "
            "(n=%d, warmup=%d) short=%s",
            sym_desc, m_n_symbols, m_n_configured, lb_desc,
            m_n_lookbacks, m_history_bars,
            (Inp_Gold_AllowShort ? "on" : "off")));
        return true;
    }

    //--- Daily processing hook (triggered after the UTC close).
    void OnNewBarD1(CRiskManager &risk) override
    {
        if(risk.IsDDLocked()) return;
        for(int i = 0; i < m_n_symbols; i++)
            ProcessSymbol(m_symbols[i], risk);
    }

    int CloseAll(string reason) override
    {
        return CloseAllByMagic(m_magic, reason);
    }

private:
    void ProcessSymbol(string symbol, CRiskManager &risk)
    {
        double score;
        if(!ComputeScore(symbol, score)) return;

        bool long_signal  = (score > 0.0);
        bool short_signal = Inp_Gold_AllowShort && (score < 0.0);

        double sigma21 = ComputeSigma21(symbol);
        double lev = MathMin(Inp_Gold_TargetVol / MathMax(sigma21, 0.05),
                             Inp_Gold_MaxLeverage);

        ulong existing = FindPositionByMagicSymbol(m_magic, symbol);
        long pos_type = -1;
        if(existing != 0) pos_type = PositionGetInteger(POSITION_TYPE);

        // Exit on signal flip.
        if(pos_type == (long)POSITION_TYPE_BUY && !long_signal)
        {
            m_trade.PositionClose(existing);
            g_logger.Info(m_name, StringFormat(
                "Exit LONG %s (score=%.2f)", symbol, score));
            existing = 0;
        }
        else if(pos_type == (long)POSITION_TYPE_SELL && !short_signal)
        {
            m_trade.PositionClose(existing);
            g_logger.Info(m_name, StringFormat(
                "Exit SHORT %s (score=%.2f)", symbol, score));
            existing = 0;
        }

        if(existing == 0)
        {
            if(long_signal)       OpenPosition(symbol, ORDER_TYPE_BUY,  lev, score, risk);
            else if(short_signal) OpenPosition(symbol, ORDER_TYPE_SELL, lev, score, risk);
        }

        // The trace stays wired to the first instrument only. Its file format
        // is single-series (one row per date, no symbol column) and it is
        // already known broken against vbt; widening it to the whole universe
        // would only produce interleaved rows nothing can read.
        if(Inp_Gold_Trace && symbol == m_symbols[0])
            WriteTraceRow(symbol, score, lev, long_signal, short_signal, risk);
    }

    //--- Append one row of the cross-engine reconciliation trace.
    //--- Contract and column order: docs/specs/gold_momentum_spec.md §9.
    //--- Off by default: this writes to disk on every session and is a
    //--- diagnostic, not a production behaviour.
    void WriteTraceRow(string symbol, double score, double lev,
                       bool long_signal, bool short_signal, CRiskManager &risk)
    {
        double direction     = long_signal ? 1.0 : (short_signal ? -1.0 : 0.0);
        double target_weight = lev * direction;

        //--- Position in units (ounces), not lots, so the column means the
        //--- same thing as the vbt one.
        double units  = 0.0;
        ulong  ticket = FindPositionByMagicSymbol(m_magic, symbol);
        if(ticket != 0 && PositionSelectByTicket(ticket))
        {
            double vol = PositionGetDouble(POSITION_VOLUME)
                         * SymbolInfoDouble(symbol, SYMBOL_TRADE_CONTRACT_SIZE);
            units = (PositionGetInteger(POSITION_TYPE) == (long)POSITION_TYPE_SELL)
                    ? -vol : vol;
        }

        //--- The row is stamped with the session whose close produced the
        //--- score (shift 1), NOT the session the order is sent in. Stamping
        //--- it with the execution day would shift the whole trace one bar
        //--- against vbt and break rung 2 for a reason unrelated to the signal.
        datetime bar_time = iTime(symbol, PERIOD_D1, 1);
        double   close_px = iClose(symbol, PERIOD_D1, 1);
        if(bar_time == 0 || close_px <= 0.0) return;

        MqlDateTime dt;
        TimeToStruct(bar_time, dt);

        //--- Sleeve-attributable equity, not account equity: the EA runs
        //--- several sleeves and the account figure would not be comparable.
        double equity = risk.SubEquity(SLEEVE_GOLD_MOMENTUM);

        int handle = FileOpen(Inp_Gold_TraceFile,
                              FILE_READ | FILE_WRITE | FILE_TXT | FILE_ANSI | FILE_COMMON);
        if(handle == INVALID_HANDLE)
        {
            if(!m_trace_warned)
            {
                m_trace_warned = true;
                g_logger.Warn(m_name, StringFormat(
                    "cannot open trace file %s (err=%d); tracing disabled for this run",
                    Inp_Gold_TraceFile, GetLastError()));
            }
            return;
        }
        if(FileSize(handle) == 0)
            FileWriteString(handle,
                "date,close,score,target_weight,position_units,equity\n");
        FileSeek(handle, 0, SEEK_END);
        FileWriteString(handle, StringFormat(
            "%04d-%02d-%02d,%.6f,%.6f,%.6f,%.6f,%.2f\n",
            dt.year, dt.mon, dt.day,
            close_px, score, target_weight, units, equity));
        FileClose(handle);
    }

    //--- Mean of sign(return over N) across the configured lookbacks.
    //--- Reads shift 1 onwards, so the current forming bar is never used.
    bool ComputeScore(string symbol, double &score)
    {
        double closes[];
        int copied = CopyClose(symbol, PERIOD_D1, 1,
                               m_history_bars, closes);
        if(copied < m_history_bars)
        {
            g_logger.Warn(m_name, StringFormat(
                "%s: only %d/%d D1 bars copied; skipping this session",
                symbol, copied, m_history_bars));
            return false;
        }
        // closes[] is oldest-first, so the most recent completed bar is last.
        int last = copied - 1;
        double newest = closes[last];
        if(newest <= 0.0) return false;

        double sum = 0.0;
        for(int i = 0; i < m_n_lookbacks; i++)
        {
            int idx = last - m_lookbacks[i];
            if(idx < 0) return false;
            double past = closes[idx];
            if(past <= 0.0) return false;
            double ret = newest / past - 1.0;
            sum += (ret > 0.0) ? 1.0 : ((ret < 0.0) ? -1.0 : 0.0);
        }
        score = sum / (double)m_n_lookbacks;
        return true;
    }

    //--- Annualised standard deviation of log returns over the last 21 D1
    //--- bars. Mirrors CSleeveTSMomentum::ComputePairSigma21; the fallback
    //--- is gold's long-run volatility rather than the FX default; it is
    //--- applied as-is to every instrument, see the header note.
    double ComputeSigma21(string symbol)
    {
        double closes[22];
        int copied = CopyClose(symbol, PERIOD_D1, 1, 22, closes);
        if(copied < 22) return 0.16;
        double rets[21];
        for(int i = 0; i < 21; i++)
        {
            if(closes[i] <= 0.0) return 0.16;
            rets[i] = MathLog(closes[i + 1] / closes[i]);
        }
        double s = 0.0, s2 = 0.0;
        for(int i = 0; i < 21; i++) { s += rets[i]; s2 += rets[i] * rets[i]; }
        double mean = s / 21.0;
        double var = (s2 - 21.0 * mean * mean) / 20.0;
        return MathSqrt(MathMax(var, 0.0)) * MathSqrt(252.0);
    }

    //--- Submit a market entry with FLAT sizing. Slippage, commission and an
    //--- empirical swap drag are pre-paid via SizingDrag() so signal-flip
    //--- exits do not undercount costs.
    void OpenPosition(string symbol, ENUM_ORDER_TYPE type, double lev,
                      double score, CRiskManager &risk)
    {
        double price = (type == ORDER_TYPE_BUY)
                       ? SymbolInfoDouble(symbol, SYMBOL_ASK)
                       : SymbolInfoDouble(symbol, SYMBOL_BID);
        if(price <= 0.0) return;

        double slip_pct = SlippageFraction(Inp_Gold_SlippageBps,
                                           Inp_CommissionBpsPerSide);

        // Wide safety stop. The sleeve normally exits on a signal flip, so
        // this guards an overnight gap rather than acting as a regular stop.
        // Gold is roughly twice as volatile as a major FX pair, hence the
        // wider distance than the 2% used by the FX sleeves.
        double sl_dist = price * (Inp_Gold_SafetySL + slip_pct);
        double sl = (type == ORDER_TYPE_BUY) ? price - sl_dist
                                             : price + sl_dist;
        sl = EnforceStopLevel(symbol, price, sl, type, true);

        m_trade.SetDeviationInPoints(FX_DEVIATION_POINTS);

        // Equal-weight split across the CONFIGURED instruments (see header):
        // with a single symbol the divisor is 1 and the sizing is unchanged.
        double sub_eq = risk.SubEquity(SLEEVE_GOLD_MOMENTUM) / m_n_configured;
        double slip_drag = SizingDrag(slip_pct, Inp_SwapBpsPerNight,
                                      FX_GOLD_AVG_NIGHTS_HELD);
        double risk_money = sub_eq * FX_RISK_PCT_GOLD_MOMENTUM
                            * lev * slip_drag * Inp_RiskScale;
        double lots = LotsForRisk(symbol, risk_money, sl_dist);
        if(lots <= 0.0)
        {
            g_logger.Warn(m_name, StringFormat(
                "%s: computed lots=0 (risk_money=%.2f sl_dist=%.2f); "
                "check SYMBOL_VOLUME_MIN", symbol, risk_money, sl_dist));
            return;
        }

        bool ok = (type == ORDER_TYPE_BUY)
                  ? m_trade.Buy(lots, symbol, price, sl, 0.0, "Gold momentum long")
                  : m_trade.Sell(lots, symbol, price, sl, 0.0, "Gold momentum short");

        if(!ok || m_trade.ResultRetcode() != TRADE_RETCODE_DONE)
        {
            g_logger.Error(m_name, StringFormat(
                "Entry %s failed: retcode=%d", symbol,
                m_trade.ResultRetcode()));
            return;
        }
        g_logger.Info(m_name, StringFormat(
            "Entry %s %s lots=%.2f price=%.2f score=%.2f lev=%.2f",
            (type == ORDER_TYPE_BUY ? "LONG" : "SHORT"),
            symbol, lots, price, score, lev));
    }
};

#endif // __FX_SLEEVE_GOLD_MOMENTUM_MQH__
