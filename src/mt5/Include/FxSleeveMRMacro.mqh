//+------------------------------------------------------------------+
//| FxSleeveMRMacro.mqh                                              |
//| Sleeve 1 — Mean Reversion intraday VWAP+BB filtré macro,         |
//| equal-weight 4 paires (LaTeX § 03_sleeve_mr_macro).              |
//|                                                                  |
//| Source de vérité : src/strategies/mr_macro.py                    |
//|   bb_window=80, bb_alpha=5.0                                     |
//|   sl_stop=0.005, tp_stop=0.006                                   |
//|   session 6-14h UTC, td_stop=6h, dt_stop=21:00 UTC               |
//|   spread_threshold=0.5, unemp 3m non-rising                      |
//|   Univers : EURUSD,GBPUSD,USDJPY,USDCAD (sub-equity / n_pairs)   |
//+------------------------------------------------------------------+
#ifndef __FX_SLEEVE_MR_MACRO_MQH__
#define __FX_SLEEVE_MR_MACRO_MQH__

#include <Trade/Trade.mqh>
#include "FxSleeveBase.mqh"
#include "FxCommon.mqh"
#include "FxIndicatorVWAP.mqh"
#include "FxIndicatorBBDeviation.mqh"
#include "FxMacroFilter.mqh"
#include "FxRiskManager.mqh"
#include "FxTradeHelpers.mqh"

//--- Inputs : déclarés dans Experts/FxMultiSleeve.mq5 et accessibles
//--- automatiquement ici car #include est textuel.
//---
//--- Inputs lus :
//---   Inp_MR_Pairs (CSV des paires à trader)
//---   Inp_MR_BBWindow, Inp_MR_BBAlpha, Inp_MR_SLStop, Inp_MR_TPStop
//---   Inp_MR_SessionStart, Inp_MR_SessionEnd
//---   Inp_MR_TimeStopHours, Inp_MR_ForcedCloseHr
//---   Inp_MR_SlippageBps (cf. plan d'alignement § slippage)
//---   Inp_SymbolSuffix, Inp_MagicMR

#define FX_MR_MAX_PAIRS 8

class CSleeveMRMacro : public CSleeveBase
{
private:
    int           m_n_pairs;
    string        m_symbols[FX_MR_MAX_PAIRS];
    CVWAPDaily    m_vwap[FX_MR_MAX_PAIRS];
    CBBDeviation  m_bb[FX_MR_MAX_PAIRS];
    datetime      m_last_m1_bar[FX_MR_MAX_PAIRS];
    CTrade        m_trade;

public:
    bool Init() override
    {
        m_magic  = Inp_MagicMR;
        m_name   = "MR_Macro";

        string raw[];
        int n = SplitCsv(Inp_MR_Pairs, raw);
        if(n <= 0 || n > FX_MR_MAX_PAIRS)
        {
            g_logger.Error(m_name, StringFormat("invalid Inp_MR_Pairs=%s", Inp_MR_Pairs));
            return false;
        }
        m_n_pairs = n;
        for(int i = 0; i < n; i++)
        {
            m_symbols[i] = MakeSymbolWithSuffix(raw[i], Inp_SymbolSuffix);
            if(!EnsureSymbolSelected(m_symbols[i])) return false;
            // 1500 M1 bars = ~1 jour de trading + warmup BB(80)
            if(!EnsureHistory(m_symbols[i], PERIOD_M1, 1500)) return false;

            m_bb[i].Init(Inp_MR_BBWindow, Inp_MR_BBAlpha);
            if(!m_vwap[i].Warmup(m_symbols[i]))
                g_logger.Warn(m_name,
                    StringFormat("VWAP warmup empty for %s; will rebuild as bars arrive",
                                 m_symbols[i]));
            WarmupBBFromHistory(i);
            m_last_m1_bar[i] = iTime(m_symbols[i], PERIOD_M1, 0);
        }
        m_trade.SetExpertMagicNumber(m_magic);
        m_trade.SetDeviationInPoints(20);  // override par paire dans OpenPosition
        g_logger.Info(m_name, StringFormat("Init OK %d pairs (%s)", n, Inp_MR_Pairs));
        return true;
    }

    //--- Hook NewBar M1 : boucle sur toutes les paires, détecte une nouvelle
    //--- bar fermée par paire et déclenche signal/sortie.
    //--- L'EA principal appelle cette fonction sur chaque tick ; c'est ici
    //--- qu'on filtre la cadence par paire pour éviter les recalculs intra-bar.
    void OnNewBarM1(CMacroFilter &macro, CRiskManager &risk) override
    {
        if(risk.IsDDLocked()) return;
        for(int i = 0; i < m_n_pairs; i++)
        {
            datetime current_m1 = iTime(m_symbols[i], PERIOD_M1, 0);
            if(current_m1 == 0 || current_m1 == m_last_m1_bar[i]) continue;
            m_last_m1_bar[i] = current_m1;
            ProcessPair(i, macro, risk);
        }
    }

    //--- Vérifie time-stop (6h max) et close forcé 21h UTC sur toutes les paires.
    void CheckIntradayExits() override
    {
        datetime now = TimeGMT();
        MqlDateTime tm; TimeToStruct(now, tm);
        bool past_forced_close = (tm.hour >= Inp_MR_ForcedCloseHr);

        for(int i = PositionsTotal() - 1; i >= 0; i--)
        {
            ulong ticket = PositionGetTicket(i);
            if(ticket == 0) continue;
            if(PositionGetInteger(POSITION_MAGIC) != m_magic) continue;

            // On laisse passer toutes les paires gérées par ce sleeve
            string sym = PositionGetString(POSITION_SYMBOL);
            if(!IsManagedSymbol(sym)) continue;

            datetime opened = (datetime)PositionGetInteger(POSITION_TIME);
            int age_hours = (int)((now - opened) / 3600);

            if(age_hours >= Inp_MR_TimeStopHours)
            {
                m_trade.PositionClose(ticket);
                g_logger.Info(m_name,
                    StringFormat("td_stop close ticket=%I64u sym=%s age=%dh",
                                 ticket, sym, age_hours));
            }
            else if(past_forced_close)
            {
                m_trade.PositionClose(ticket);
                g_logger.Info(m_name,
                    StringFormat("dt_stop close ticket=%I64u sym=%s (21h UTC)",
                                 ticket, sym));
            }
        }
    }

    int CloseAll(string reason) override
    {
        return CloseAllByMagic(m_magic, reason);
    }

private:
    //--- Traite une paire : push VWAP/BB sur la dernière bar fermée, applique
    //--- les filtres (session, macro, no-pyramiding), envoie l'ordre.
    void ProcessPair(int idx, CMacroFilter &macro, CRiskManager &risk)
    {
        MqlRates last[];
        if(CopyRates(m_symbols[idx], PERIOD_M1, 1, 1, last) != 1) return;

        m_vwap[idx].OnNewBarM1(last[0]);
        double dev = last[0].close - m_vwap[idx].Get();
        m_bb[idx].Push(dev);

        double mean, upper_dev, lower_dev;
        if(!m_bb[idx].Compute(mean, upper_dev, lower_dev)) return;  // warmup

        // Filtres
        if(!IsInUTCSession(last[0].time, Inp_MR_SessionStart, Inp_MR_SessionEnd))
            return;
        if(!macro.MacroOk()) return;
        if(!macro.IsValid()) return;

        // Pas de pyramiding : 1 position max sur cette paire
        if(CountSleevePositions(m_magic, m_symbols[idx]) > 0) return;

        double abs_upper = m_vwap[idx].Get() + upper_dev;
        double abs_lower = m_vwap[idx].Get() + lower_dev;

        if(last[0].close < abs_lower)
            OpenPosition(idx, ORDER_TYPE_BUY, risk);
        else if(last[0].close > abs_upper)
            OpenPosition(idx, ORDER_TYPE_SELL, risk);
    }

    bool IsManagedSymbol(string sym)
    {
        for(int i = 0; i < m_n_pairs; i++)
            if(m_symbols[i] == sym) return true;
        return false;
    }

    //--- Reconstruit ~bb_window + 20 minutes pour pré-remplir le buffer BB.
    //--- Utilise un VWAP local UNIQUE qui reset à chaque changement de jour
    //--- UTC (cf. fix du bug "buffer rempli de zéros" mentionné dans le plan
    //--- d'alignement). Le VWAP local est jeté à la fin — m_vwap[idx] reste
    //--- celui calé sur le jour courant via Warmup().
    void WarmupBBFromHistory(int idx)
    {
        int warmup_bars = Inp_MR_BBWindow + 20;
        MqlRates rates[];
        int copied = CopyRates(m_symbols[idx], PERIOD_M1, 1, warmup_bars, rates);
        if(copied <= 0) return;

        CVWAPDaily warmup_vwap;     // accumule cum_pv/cum_v en streaming
        for(int i = 0; i < copied; i++)
        {
            warmup_vwap.OnNewBarM1(rates[i]);  // reset auto à minuit UTC interne
            double dev = rates[i].close - warmup_vwap.Get();
            m_bb[idx].Push(dev);
        }
    }

    //--- Ouvre une position sur la paire idx.
    //--- Sizing : sub_equity_MR / n_pairs (allocation egal entre paires)
    //--- × global_leverage. Le SL à 0.5% × leverage donne le risque effectif.
    //--- Slippage : Inp_MR_SlippageBps converti en SYMBOL_POINT pour
    //---             SetDeviationInPoints + majoration SL_distance pour
    //---             ne pas sur-sizer face au coût attendu.
    void OpenPosition(int idx, ENUM_ORDER_TYPE type, CRiskManager &risk)
    {
        string sym = m_symbols[idx];
        double price = (type == ORDER_TYPE_BUY)
                       ? SymbolInfoDouble(sym, SYMBOL_ASK)
                       : SymbolInfoDouble(sym, SYMBOL_BID);
        if(price <= 0.0) return;

        double sl = (type == ORDER_TYPE_BUY)
                    ? price * (1.0 - Inp_MR_SLStop)
                    : price * (1.0 + Inp_MR_SLStop);
        double tp = (type == ORDER_TYPE_BUY)
                    ? price * (1.0 + Inp_MR_TPStop)
                    : price * (1.0 - Inp_MR_TPStop);

        sl = EnforceStopLevel(sym, price, sl, type, true);
        tp = EnforceStopLevel(sym, price, tp, type, false);

        double point = SymbolInfoDouble(sym, SYMBOL_POINT);
        double slip_price = (Inp_MR_SlippageBps / 10000.0) * price;
        int slip_pts = (point > 0.0) ? (int)MathCeil(slip_price / point) : 20;
        m_trade.SetDeviationInPoints(MathMax(slip_pts, 5));

        // Sizing : ¼ du sub-equity MR Macro par paire (equal-weight 4 paires)
        // SL distance majoré du slippage pour que LotsForRisk ne sur-size pas.
        double sl_distance = MathAbs(price - sl) + slip_price;
        double per_pair_alloc = 1.0 / (double)m_n_pairs;
        double risk_pct = 0.01;   // 1 % du sub-equity par trade (cf. v1)
        double lots = risk.LotsFor(SLEEVE_MR_MACRO, sym, risk_pct, sl_distance,
                                   per_pair_alloc);
        if(lots <= 0.0)
        {
            g_logger.Warn(m_name, StringFormat("lots=0 on %s, skipping entry", sym));
            return;
        }

        bool ok = false;
        if(type == ORDER_TYPE_BUY)
            ok = m_trade.Buy(lots, sym, price, sl, tp, "MR Macro long");
        else
            ok = m_trade.Sell(lots, sym, price, sl, tp, "MR Macro short");

        if(!ok || m_trade.ResultRetcode() != TRADE_RETCODE_DONE)
        {
            g_logger.Error(m_name,
                StringFormat("Entry %s failed: retcode=%d desc=%s lots=%.2f price=%.5f sl=%.5f tp=%.5f",
                             sym, m_trade.ResultRetcode(),
                             m_trade.ResultRetcodeDescription(),
                             lots, price, sl, tp));
            return;
        }
        g_logger.Info(m_name,
            StringFormat("Entry %s %s lots=%.2f price=%.5f sl=%.5f tp=%.5f slip_pts=%d",
                         (type == ORDER_TYPE_BUY ? "LONG" : "SHORT"),
                         sym, lots, price, sl, tp, slip_pts));
    }
};

#endif // __FX_SLEEVE_MR_MACRO_MQH__
