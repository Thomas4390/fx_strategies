#!/usr/bin/env python3
"""write_default_preset — génère `FxMultiSleeve_Default.set` (preset Tester GUI).

Le preset contient les inputs par défaut compilés du `.mq5` (lignes 27-100),
avec mode macro AUTO=4 et logging verbose désactivé. Une fois écrit, l'utilisateur
peut le charger depuis MT5 GUI Strategy Tester via `Inputs → Load → ...set`.

Format MT5 :
- UTF-16 LE avec BOM, terminaisons CRLF.
- Numerics / bools : `key=value||default||min||max||N` (forme étendue MT5).
- Strings : `key=value` simple.

Usage :
    python src/mt5/bridge/write_default_preset.py
    python src/mt5/bridge/write_default_preset.py --out /tmp/test.set --check
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PORTABLE = Path("/home/thomas/.mt5/drive_c/Program Files/MetaTrader 5")
DEFAULT_OUTPUT = PORTABLE / "MQL5/Profiles/Tester/FxMultiSleeve_Default.set"


def _ext(value, lo, hi, *, is_bool=False) -> str:
    """Forme étendue MT5 : value||default||min||max||N (N = no optimisation)."""
    if is_bool:
        v = "true" if value else "false"
        return f"{v}||false||0||true||N"
    return f"{value}||{value}||{lo}||{hi}||N"


# Defaults alignés sur src/mt5/Experts/FxMultiSleeve.mq5:27-101
PRESET_LINES: list[str] = [
    "; FxMultiSleeve_Default.set — defaults compilés du .mq5 (lignes 27-101)",
    "; Mode macro = AUTO (4) : tester→HISTORY, live→NATIVE (zero-config).",
    "; Charger depuis MT5 GUI : Strategy Tester → Inputs → Load",
    ";",
    "; --- Allocation & Risk (somme des allocations = 1.0, validée à 1e-6) ---",
    f"Inp_AllocMRMacro={_ext(0.72, 0.072000, 7.200000)}",
    f"Inp_AllocTSMomentum={_ext(0.09, 0.009000, 0.900000)}",
    f"Inp_AllocRSIDaily={_ext(0.09, 0.009000, 0.900000)}",
    f"Inp_AllocH1Momentum={_ext(0.0, 0.0, 1.0)}",
    f"Inp_AllocGoldMomentum={_ext(0.10, 0.010000, 1.000000)}",
    f"Inp_EnableDDCap={_ext(False, 0, 0, is_bool=True)}",
    f"Inp_DDCap={_ext(0.20, 0.020000, 2.000000)}",
    f"Inp_ResetDDState={_ext(False, 0, 0, is_bool=True)}",
    f"Inp_EnableMarginCap={_ext(True, 0, 0, is_bool=True)}",
    f"Inp_MarginCapPct={_ext(0.5, 0.050000, 5.000000)}",
    "; --- Vol-targeting global (retuné 2026-07-26 avec l'entrée de l'or) ---",
    f"Inp_GlobalTargetVol={_ext(0.37, 0.037000, 3.700000)}",
    f"Inp_GlobalMaxLeverage={_ext(31.0, 3.100000, 310.000000)}",
    f"Inp_GlobalVolFloor={_ext(0.02, 0.002000, 0.200000)}",
    "; Échelle des budgets de risque — 4.5 délivre ~40% de CAGR sur ce moteur.",
    f"Inp_RiskScale={_ext(4.5, 0.450000, 45.000000)}",
    "; --- Sleeve 1 — MR Macro ---",
    "Inp_MR_Pairs=EURUSD,GBPUSD,USDJPY,USDCAD",
    f"Inp_MR_BBWindow={_ext(80, 1, 800)}",
    f"Inp_MR_BBAlpha={_ext(5.0, 0.500000, 50.000000)}",
    f"Inp_MR_TPStop={_ext(0.006, 0.000600, 0.060000)}",
    f"Inp_MR_SLStop={_ext(0.005, 0.000500, 0.050000)}",
    f"Inp_MR_SessionStart={_ext(8, 1, 60)}",
    f"Inp_MR_SessionEnd={_ext(16, 1, 140)}",
    f"Inp_MR_TimeStopHours={_ext(6, 1, 60)}",
    f"Inp_MR_ForcedCloseHr={_ext(21, 1, 210)}",
    f"Inp_MR_SpreadThresh={_ext(0.5, 0.050000, 5.000000)}",
    f"Inp_MR_SlippageBps={_ext(15, 1, 150)}",
    f"Inp_MR_DisableMacroFilter={_ext(False, 0, 0, is_bool=True)}",
    f"Inp_MR_NewsFilterEnabled={_ext(True, 0, 0, is_bool=True)}",
    "; --- Sleeve 2 — TS Momentum ---",
    "Inp_TS_Pairs=EURUSD,GBPUSD,USDJPY",
    f"Inp_TS_FastEMA={_ext(20, 1, 200)}",
    f"Inp_TS_SlowEMA={_ext(50, 1, 500)}",
    f"Inp_TS_RSIPeriod={_ext(7, 1, 70)}",
    f"Inp_TS_RSILow={_ext(40, 1, 400)}",
    f"Inp_TS_RSIHigh={_ext(60, 1, 600)}",
    f"Inp_TS_TargetVol={_ext(0.1, 0.010000, 1.000000)}",
    f"Inp_TS_MaxLeverage={_ext(3.0, 0.300000, 30.000000)}",
    f"Inp_TS_SlippageBps={_ext(10, 1, 100)}",
    "; --- Sleeve 4 — H1 Momentum (Phase D, off by default) ---",
    "Inp_H1_Pairs=EURUSD,GBPUSD,USDJPY",
    f"Inp_H1_FastEMA={_ext(20, 1, 200)}",
    f"Inp_H1_SlowEMA={_ext(50, 1, 500)}",
    f"Inp_H1_RSIPeriod={_ext(7, 1, 70)}",
    f"Inp_H1_RSILow={_ext(40, 1, 400)}",
    f"Inp_H1_RSIHigh={_ext(60, 1, 600)}",
    f"Inp_H1_ATRPeriod={_ext(14, 1, 140)}",
    f"Inp_H1_ATRMultSL={_ext(2.0, 0.5, 10.0)}",
    f"Inp_H1_TargetVol={_ext(0.10, 0.010000, 1.000000)}",
    f"Inp_H1_MaxLeverage={_ext(3.0, 0.300000, 30.000000)}",
    f"Inp_H1_SlippageBps={_ext(12, 1, 120)}",
    "; --- Sleeve 3 — RSI Daily ---",
    "Inp_RSI_Pairs=EURUSD,GBPUSD,USDCAD",
    f"Inp_RSI_Period={_ext(14, 1, 140)}",
    f"Inp_RSI_Oversold={_ext(25.0, 2.500000, 250.000000)}",
    f"Inp_RSI_Overbought={_ext(75.0, 7.500000, 750.000000)}",
    f"Inp_RSI_ExitMid={_ext(50.0, 5.000000, 500.000000)}",
    f"Inp_RSI_SlippageBps={_ext(10, 1, 100)}",
    f"Inp_RSI_TimeStopDays={_ext(21, 0, 210)}",
    "; --- Coûts de transaction communs ---",
    f"Inp_CommissionBpsPerSide={_ext(5.0, 0.500000, 50.000000)}",
    f"Inp_SwapBpsPerNight={_ext(0.5, 0.050000, 5.000000)}",
    "; --- Sleeve 5 — Gold Momentum (en production depuis 2026-07-26) ---",
    "Inp_Gold_Symbols=XAUUSD",
    f"Inp_Gold_LookbackA={_ext(40, 0, 400)}",
    f"Inp_Gold_LookbackB={_ext(60, 0, 600)}",
    f"Inp_Gold_LookbackC={_ext(120, 0, 1200)}",
    f"Inp_Gold_LookbackD={_ext(250, 0, 2500)}",
    f"Inp_Gold_AllowShort={_ext(False, 0, 0, is_bool=True)}",
    f"Inp_Gold_TargetVol={_ext(0.55, 0.055000, 5.500000)}",
    f"Inp_Gold_MaxLeverage={_ext(6.6, 0.660000, 66.000000)}",
    f"Inp_Gold_SafetySL={_ext(0.04, 0.004000, 0.400000)}",
    f"Inp_Gold_SlippageBps={_ext(2, 1, 20)}",
    f"Inp_Gold_Trace={_ext(False, 0, 0, is_bool=True)}",
    "Inp_Gold_TraceFile=gold_trace.csv",
    "; --- Operational ---",
    "Inp_SymbolSuffix=.c",
    f"Inp_MagicMR={_ext(831, 1, 8310)}",
    f"Inp_MagicTS={_ext(832, 1, 8320)}",
    f"Inp_MagicRSI={_ext(833, 1, 8330)}",
    f"Inp_MagicH1={_ext(834, 1, 8340)}",
    f"Inp_MagicGold={_ext(835, 1, 8350)}",
    f"Inp_LogVerbose={_ext(False, 0, 0, is_bool=True)}",
    f"Inp_LogToFile={_ext(True, 0, 0, is_bool=True)}",
    f"Inp_ExportDeals={_ext(False, 0, 0, is_bool=True)}",
    "Inp_MacroCacheFile=macro_cache.csv",
    f"Inp_MacroUseCommon={_ext(True, 0, 0, is_bool=True)}",
    f"Inp_MacroMaxAgeHours={_ext(168, 1, 1680)}",
    f"Inp_DailyRecomputeHr={_ext(21, 1, 210)}",
    "; --- Macro source mode (4 = AUTO : tester→HISTORY, live→NATIVE) ---",
    "Inp_MacroSourceMode=4||0||0||4||N",
    "Inp_FREDApiKeyFile=fred_api_key.txt",
    f"Inp_FREDKeyUseCommon={_ext(True, 0, 0, is_bool=True)}",
    "Inp_FREDSeriesId=T10Y2Y",
    "Inp_MacroHistoryFile=macro_history.csv",
    f"Inp_MacroHistoryUseCommon={_ext(True, 0, 0, is_bool=True)}",
]


def write_preset(path: Path) -> None:
    text = "\n".join(PRESET_LINES) + "\n"
    crlf = text.replace("\n", "\r\n")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(crlf.encode("utf-16"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true",
                        help="Affiche le contenu sans écrire")
    args = parser.parse_args()

    if args.check:
        print("\n".join(PRESET_LINES))
        return 0

    write_preset(args.out)
    print(f"[ok] wrote preset → {args.out}")
    print(f"[ok] {len(PRESET_LINES)} lines, "
          f"{args.out.stat().st_size} bytes (UTF-16 LE BOM CRLF)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
