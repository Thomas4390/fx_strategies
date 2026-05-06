//+------------------------------------------------------------------+
//| FxSleeveBase.mqh                                                 |
//|                                                                  |
//| Abstract base class shared by all trading sleeves. Provides a    |
//| uniform lifecycle (Init/Shutdown), event hooks (new bar M1/D1),  |
//| and a forced-close entry point. Concrete sleeves override the    |
//| relevant methods.                                                |
//+------------------------------------------------------------------+
#ifndef __FX_SLEEVE_BASE_MQH__
#define __FX_SLEEVE_BASE_MQH__

#include "FxCommon.mqh"
#include "FxLogger.mqh"

// Forward declarations to break circular dependencies.
class CMacroFilter;
class CRiskManager;

//+------------------------------------------------------------------+
//| CSleeveBase: shared interface for MR, TS, and RSI sleeves.       |
//|                                                                  |
//| MQL5 does not support pure virtual ("= 0") declarations so the   |
//| default implementations are empty bodies that derived classes    |
//| override as needed.                                              |
//+------------------------------------------------------------------+
class CSleeveBase
{
protected:
    int    m_magic;
    string m_name;

public:
    CSleeveBase() : m_magic(0), m_name("Base") {}

    int    Magic() const { return m_magic; }
    string Name()  const { return m_name; }

    // Lifecycle
    virtual bool Init() { return true; }
    virtual void Shutdown() {}

    // Event hooks (one per timeframe / responsibility)
    virtual void OnNewBarM1(CMacroFilter &macro, CRiskManager &risk) {}
    virtual void OnNewBarD1(CRiskManager &risk) {}
    virtual void CheckIntradayExits() {}

    // Forced shutdown (drawdown breaker, margin breaker, manual override).
    virtual int  CloseAll(string reason) { return 0; }
};

#endif // __FX_SLEEVE_BASE_MQH__
