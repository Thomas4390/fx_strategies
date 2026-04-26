//+------------------------------------------------------------------+
//| FxTradeHelpers.mqh                                               |
//| Wrappers CTrade + sanity checks (lot normalization, stops level).|
//+------------------------------------------------------------------+
#ifndef __FX_TRADE_HELPERS_MQH__
#define __FX_TRADE_HELPERS_MQH__

#include <Trade/Trade.mqh>
#include "FxCommon.mqh"

//+------------------------------------------------------------------+
//| Normalise un volume au step broker, bornée à [min, max].          |
//+------------------------------------------------------------------+
double NormalizeLots(string symbol, double raw_lots)
{
    double step = SymbolInfoDouble(symbol, SYMBOL_VOLUME_STEP);
    double minv = SymbolInfoDouble(symbol, SYMBOL_VOLUME_MIN);
    double maxv = SymbolInfoDouble(symbol, SYMBOL_VOLUME_MAX);
    if(step <= 0.0) return raw_lots;
    double lots = MathFloor(raw_lots / step) * step;
    if(lots < minv) lots = minv;
    if(lots > maxv) lots = maxv;
    return lots;
}

//+------------------------------------------------------------------+
//| S'assure que SL/TP respectent SYMBOL_TRADE_STOPS_LEVEL.          |
//| Retourne le SL ajusté.                                           |
//+------------------------------------------------------------------+
double EnforceStopLevel(string symbol, double price, double stop,
                        ENUM_ORDER_TYPE type, bool is_sl)
{
    double point = SymbolInfoDouble(symbol, SYMBOL_POINT);
    long stops_level = SymbolInfoInteger(symbol, SYMBOL_TRADE_STOPS_LEVEL);
    double min_dist = stops_level * point;
    if(min_dist <= 0.0) return stop;

    if(type == ORDER_TYPE_BUY)
    {
        if(is_sl)
        {
            // SL doit être < price - min_dist
            double max_sl = price - min_dist - point;
            if(stop > max_sl) stop = max_sl;
        }
        else
        {
            // TP doit être > price + min_dist
            double min_tp = price + min_dist + point;
            if(stop < min_tp) stop = min_tp;
        }
    }
    else // SELL
    {
        if(is_sl)
        {
            double min_sl = price + min_dist + point;
            if(stop < min_sl) stop = min_sl;
        }
        else
        {
            double max_tp = price - min_dist - point;
            if(stop > max_tp) stop = max_tp;
        }
    }
    int digits = (int)SymbolInfoInteger(symbol, SYMBOL_DIGITS);
    return NormalizeDouble(stop, digits);
}

//+------------------------------------------------------------------+
//| Calcule la taille en lots pour un risque monétaire donné.        |
//|                                                                  |
//| risk_money       : montant à risquer (devise compte)             |
//| sl_distance_price: distance prix-SL en unités de prix             |
//+------------------------------------------------------------------+
double LotsForRisk(string symbol, double risk_money, double sl_distance_price)
{
    if(risk_money <= 0.0 || sl_distance_price <= 0.0) return 0.0;
    double tick_value = SymbolInfoDouble(symbol, SYMBOL_TRADE_TICK_VALUE);
    double tick_size  = SymbolInfoDouble(symbol, SYMBOL_TRADE_TICK_SIZE);
    if(tick_value <= 0.0 || tick_size <= 0.0) return 0.0;
    double points = sl_distance_price / tick_size;
    double raw_lots = risk_money / (points * tick_value);
    return NormalizeLots(symbol, raw_lots);
}

//+------------------------------------------------------------------+
//| Ferme toutes les positions portant un magic donné.               |
//+------------------------------------------------------------------+
int CloseAllByMagic(int magic, string reason = "")
{
    CTrade trade;
    trade.SetExpertMagicNumber(magic);
    int closed = 0;
    for(int i = PositionsTotal() - 1; i >= 0; i--)
    {
        ulong ticket = PositionGetTicket(i);
        if(ticket == 0) continue;
        if(PositionGetInteger(POSITION_MAGIC) != magic) continue;
        if(trade.PositionClose(ticket)) closed++;
        else
            PrintFormat("CloseAllByMagic(%d): close %I64u failed retcode=%d",
                        magic, ticket, trade.ResultRetcode());
    }
    if(closed > 0 && StringLen(reason) > 0)
        PrintFormat("CloseAllByMagic(%d): closed %d positions (%s)",
                    magic, closed, reason);
    return closed;
}

#endif // __FX_TRADE_HELPERS_MQH__
