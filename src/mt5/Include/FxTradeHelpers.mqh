//+------------------------------------------------------------------+
//| FxTradeHelpers.mqh                                               |
//|                                                                  |
//| Lot normalization, stops-level enforcement, risk-based position  |
//| sizing, and bulk position closing utilities used by every sleeve.|
//+------------------------------------------------------------------+
#ifndef __FX_TRADE_HELPERS_MQH__
#define __FX_TRADE_HELPERS_MQH__

#include <Trade/Trade.mqh>
#include "FxCommon.mqh"

//+------------------------------------------------------------------+
//| Normalize a raw lot size to the broker volume step, clamped to   |
//| the [SYMBOL_VOLUME_MIN, SYMBOL_VOLUME_MAX] interval.             |
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
//| Clamp a stop level so it respects SYMBOL_TRADE_STOPS_LEVEL plus  |
//| a safety cushion, then normalize to the symbol's price digits.   |
//|                                                                  |
//| Some brokers raise SYMBOL_TRADE_STOPS_LEVEL during news events.  |
//| The cushion absorbs that change without rejecting the order.     |
//+------------------------------------------------------------------+
double EnforceStopLevel(string symbol, double price, double stop,
                        ENUM_ORDER_TYPE type, bool is_sl)
{
    double point = SymbolInfoDouble(symbol, SYMBOL_POINT);
    long stops_level = SymbolInfoInteger(symbol, SYMBOL_TRADE_STOPS_LEVEL);
    double min_dist = stops_level * point;
    if(min_dist <= 0.0) min_dist = FX_STOPS_SAFETY_POINTS * point;
    else                min_dist += FX_STOPS_SAFETY_POINTS * point;

    if(type == ORDER_TYPE_BUY)
    {
        if(is_sl)
        {
            double max_sl = price - min_dist - point;
            if(stop > max_sl) stop = max_sl;
        }
        else
        {
            double min_tp = price + min_dist + point;
            if(stop < min_tp) stop = min_tp;
        }
    }
    else
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
//| Translate a monetary risk budget into a normalized lot size.     |
//|                                                                  |
//|   risk_money        : risk budget in account currency            |
//|   sl_distance_price : stop distance in price units (e.g. 0.0050) |
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
//| Close every open position carrying the given magic number.      |
//| Iterates in reverse order so the index remains valid as          |
//| positions disappear from the list.                               |
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

//+------------------------------------------------------------------+
//| Find the open position for (magic, symbol). Returns 0 if none.  |
//+------------------------------------------------------------------+
ulong FindPositionByMagicSymbol(int magic, string symbol)
{
    int total = PositionsTotal();
    for(int i = 0; i < total; i++)
    {
        ulong ticket = PositionGetTicket(i);
        if(ticket == 0) continue;
        if(PositionGetInteger(POSITION_MAGIC) != magic) continue;
        if(PositionGetString(POSITION_SYMBOL) != symbol) continue;
        return ticket;
    }
    return 0;
}

//+------------------------------------------------------------------+
//| Convert per-side basis points to a fractional price shift.       |
//|     (slip_bps + commission_bps) / 10000.                         |
//+------------------------------------------------------------------+
double SlippageFraction(int slip_bps, double commission_bps)
{
    return ((double)slip_bps + commission_bps) / 10000.0;
}

//+------------------------------------------------------------------+
//| Apply round-trip slippage and (optional) overnight swap drag to  |
//| a sizing multiplier. Used by sleeves to pre-pay execution costs  |
//| that the strategy tester does not deduct on non-SL/TP exits.     |
//+------------------------------------------------------------------+
double SizingDrag(double slip_pct, double swap_bps_per_night,
                  double avg_nights_held)
{
    double swap_drag_pct = (swap_bps_per_night * avg_nights_held) / 10000.0;
    double drag = 1.0 - 2.0 * slip_pct - swap_drag_pct;
    if(drag < FX_HEALTH_FLOOR_DRAG) drag = FX_HEALTH_FLOOR_DRAG;
    return drag;
}

#endif // __FX_TRADE_HELPERS_MQH__
