//+------------------------------------------------------------------+
//| FxIndicatorTest.mq5                                              |
//| Script de tests unitaires pour CVWAPDaily et CBBDeviation.       |
//| Lancer manuellement (Navigator → Scripts → FxIndicatorTest).     |
//+------------------------------------------------------------------+
#property copyright "fx_strategies port"
#property version   "1.00"
#property script_show_inputs

#include "..\Include\FxIndicatorVWAP.mqh"
#include "..\Include\FxIndicatorBBDeviation.mqh"
#include "..\Include\FxCommon.mqh"

input int    Inp_BBWindow = 80;
input double Inp_BBAlpha  = 5.0;

//+------------------------------------------------------------------+
//| TestBBDeviationConstants : suite de cas connus.                  |
//+------------------------------------------------------------------+
bool TestBBDeviationConstants()
{
    bool ok = true;

    // Cas 1 : valeurs constantes 0.0 → mean=0, std=0
    {
        CBBDeviation bb;
        bb.Init(80, 5.0);
        for(int i = 0; i < 80; i++) bb.Push(0.0);
        double m, u, l;
        if(!bb.Compute(m, u, l) || m != 0.0 || u != 0.0 || l != 0.0)
        {
            Print("FAIL Test 1: const 0.0 — mean/upper/lower != 0");
            ok = false;
        }
        else Print("PASS Test 1: const 0.0");
    }

    // Cas 2 : suite [0..79] → mean=39.5, std (ddof=1) = 23.2379...
    {
        CBBDeviation bb;
        bb.Init(80, 1.0);  // alpha=1 pour vérifier directement std
        for(int i = 0; i < 80; i++) bb.Push((double)i);
        double m, u, l;
        if(!bb.Compute(m, u, l))
        {
            Print("FAIL Test 2: not ready");
            ok = false;
        }
        else
        {
            double expected_mean = 39.5;
            double expected_std  = 23.2379000772;  // ddof=1
            if(MathAbs(m - expected_mean) > 1e-6)
            {
                PrintFormat("FAIL Test 2: mean got %.10f expected %.10f",
                            m, expected_mean);
                ok = false;
            }
            else if(MathAbs((u - m) - expected_std) > 1e-4)
            {
                PrintFormat("FAIL Test 2: std got %.10f expected %.10f",
                            (u - m), expected_std);
                ok = false;
            }
            else
                PrintFormat("PASS Test 2: mean=%.4f std=%.4f", m, (u - m));
        }
    }

    // Cas 3 : warmup → not ready avec count < window
    {
        CBBDeviation bb;
        bb.Init(80, 5.0);
        for(int i = 0; i < 79; i++) bb.Push(1.0);
        double m, u, l;
        if(bb.Compute(m, u, l) || bb.IsReady())
        {
            Print("FAIL Test 3: should not be ready at count=79");
            ok = false;
        }
        else Print("PASS Test 3: warmup gating");
    }

    return ok;
}

//+------------------------------------------------------------------+
//| TestVWAPDailyAnchored : ingère les bars du jour courant et       |
//| affiche la valeur courante. Ne fait pas de comparaison numérique |
//| (à valider contre dump Python séparément).                       |
//+------------------------------------------------------------------+
bool TestVWAPDailyAnchored()
{
    CVWAPDaily vwap;
    if(!vwap.Warmup(_Symbol))
    {
        Print("FAIL VWAP Warmup: pas de bars depuis minuit UTC");
        return false;
    }
    PrintFormat("PASS VWAP: bars_today=%d vwap=%.5f",
                vwap.BarsToday(), vwap.Get());
    return true;
}

//+------------------------------------------------------------------+
//| OnStart                                                          |
//+------------------------------------------------------------------+
void OnStart()
{
    Print("=== FxIndicatorTest start ===");
    bool all_ok = true;
    if(!TestBBDeviationConstants()) all_ok = false;
    if(!TestVWAPDailyAnchored())    all_ok = false;
    if(all_ok) Print("=== ALL TESTS PASSED ===");
    else       Print("=== SOME TESTS FAILED — see log above ===");
}
