//+------------------------------------------------------------------+
//| FxSleeveBase.mqh                                                 |
//| Classe abstraite commune aux trois sleeves.                      |
//+------------------------------------------------------------------+
#ifndef __FX_SLEEVE_BASE_MQH__
#define __FX_SLEEVE_BASE_MQH__

#include "FxCommon.mqh"
#include "FxLogger.mqh"

//--- Forward decls (pour briser les dépendances circulaires)
class CMacroFilter;
class CRiskManager;

//+------------------------------------------------------------------+
//| CSleeveBase : interface commune.                                 |
//|                                                                  |
//| Les méthodes virtuelles doivent être surchargées par les classes |
//| concrètes (CSleeveMRMacro, CSleeveTSMomentum, CSleeveRSIDaily).  |
//| Note MQL5 : pas de "= 0" pour méthode pure ; on utilise un       |
//| corps vide par défaut.                                            |
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

    //--- Cycle de vie
    virtual bool Init() { return true; }
    virtual void Shutdown() {}

    //--- Hooks par fréquence
    virtual void OnNewBarM1(CMacroFilter &macro, CRiskManager &risk) {}
    virtual void OnNewBarD1(CRiskManager &risk) {}
    virtual void CheckIntradayExits() {}

    //--- Fermeture forcée
    virtual int  CloseAll(string reason) { return 0; }
};

#endif // __FX_SLEEVE_BASE_MQH__
