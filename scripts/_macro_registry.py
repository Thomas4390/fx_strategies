"""Curated FX-relevant FRED series — written to both data/ and external drive.

These are the ~50 series that drive FX trading research in this repo. They are
the highest-priority work in Phase B, written in legacy schema
``[date, <snake_metric>]`` matching the 9 pre-existing parquets.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


Freq = Literal["daily", "weekly", "monthly", "quarterly", "annual"]


@dataclass(frozen=True)
class MacroSeries:
    stem: str          # filename stem (UPPERCASE)
    series_id: str     # FRED series ID
    freq: Freq
    column: str        # snake_case column name in the legacy parquet
    category: str
    rationale: str


REGISTRY: tuple[MacroSeries, ...] = (
    # ── Existing 9 (preserved) ────────────────────────────────────────────
    MacroSeries("FED_FUNDS",     "FEDFUNDS",  "monthly",   "fed_funds",     "rates", "Fed Funds Rate (monthly)"),
    MacroSeries("DGS10",         "DGS10",     "daily",     "dgs10",         "rates", "10Y Treasury benchmark"),
    MacroSeries("DGS2",          "DGS2",      "daily",     "dgs2",          "rates", "2Y Treasury benchmark"),
    MacroSeries("CPI",           "CPIAUCSL",  "monthly",   "cpi",           "inflation", "CPI Headline (monthly)"),
    MacroSeries("CPI_CORE",      "CPILFESL",  "monthly",   "cpi_core",      "inflation", "CPI Core (monthly)"),
    MacroSeries("PCE",           "PCEPI",     "monthly",   "pce",           "inflation", "PCE Headline (monthly)"),
    MacroSeries("UNEMPLOYMENT",  "UNRATE",    "monthly",   "unemployment",  "labor", "Unemployment rate"),
    MacroSeries("NFP",           "PAYEMS",    "monthly",   "nfp",           "labor", "Non-farm payrolls"),
    MacroSeries("SPREAD_10Y2Y",  "T10Y2Y",    "daily",     "spread_10y2y",  "rates", "10Y-2Y curve spread"),

    # ── Rates ────────────────────────────────────────────────────────────
    MacroSeries("DGS3MO",        "DGS3MO",    "daily",     "dgs3mo",        "rates", "3M Treasury, front end"),
    MacroSeries("DGS5",          "DGS5",      "daily",     "dgs5",          "rates", "5Y Treasury, belly"),
    MacroSeries("DGS30",         "DGS30",     "daily",     "dgs30",         "rates", "30Y Treasury, term premium"),
    MacroSeries("SPREAD_10Y3M",  "T10Y3M",    "daily",     "spread_10y3m",  "rates", "10Y-3M (NY Fed recession curve)"),
    MacroSeries("EFFR",          "EFFR",      "daily",     "effr",          "rates", "Effective Fed Funds (daily)"),
    MacroSeries("SOFR",          "SOFR",      "daily",     "sofr",          "rates", "SOFR (post-LIBOR risk-free)"),

    # ── Inflation ────────────────────────────────────────────────────────
    MacroSeries("PCE_CORE",      "PCEPILFE",  "monthly",   "pce_core",      "inflation", "Core PCE (Fed's preferred)"),
    MacroSeries("BREAKEVEN_5Y",  "T5YIE",     "daily",     "breakeven_5y",  "inflation", "5Y inflation breakeven"),
    MacroSeries("BREAKEVEN_10Y", "T10YIE",    "daily",     "breakeven_10y", "inflation", "10Y inflation breakeven"),
    MacroSeries("BREAKEVEN_5Y5Y","T5YIFR",    "daily",     "breakeven_5y5y","inflation", "5y5y forward inflation"),

    # ── Labor ────────────────────────────────────────────────────────────
    MacroSeries("ICSA",          "ICSA",      "weekly",    "icsa",          "labor", "Initial jobless claims (weekly)"),
    MacroSeries("CIVPART",       "CIVPART",   "monthly",   "civpart",       "labor", "Labor force participation"),
    MacroSeries("AHETPI",        "AHETPI",    "monthly",   "ahetpi",        "labor", "Average hourly earnings"),

    # ── Activity ─────────────────────────────────────────────────────────
    MacroSeries("GDPC1",         "GDPC1",     "quarterly", "gdpc1",         "activity", "Real GDP (quarterly)"),
    MacroSeries("INDPRO",        "INDPRO",    "monthly",   "indpro",        "activity", "Industrial production"),
    MacroSeries("RSAFS",         "RSAFS",     "monthly",   "rsafs",         "activity", "Retail sales"),
    MacroSeries("HOUST",         "HOUST",     "monthly",   "houst",         "activity", "Housing starts"),
    MacroSeries("UMCSENT",       "UMCSENT",   "monthly",   "umcsent",       "activity", "Michigan consumer sentiment"),

    # ── FX references ────────────────────────────────────────────────────
    MacroSeries("DTWEXBGS",      "DTWEXBGS",  "daily",     "dtwexbgs",      "fx", "Broad USD index (DXY proxy)"),
    MacroSeries("DEXUSEU",       "DEXUSEU",   "daily",     "dexuseu",       "fx", "EUR/USD reference"),
    MacroSeries("DEXJPUS",       "DEXJPUS",   "daily",     "dexjpus",       "fx", "JPY/USD reference"),
    MacroSeries("DEXUSUK",       "DEXUSUK",   "daily",     "dexusuk",       "fx", "GBP/USD reference"),
    MacroSeries("DEXCAUS",       "DEXCAUS",   "daily",     "dexcaus",       "fx", "CAD/USD reference"),
    MacroSeries("DEXCHUS",       "DEXCHUS",   "daily",     "dexchus",       "fx", "CHF/USD reference"),

    # ── Money / credit ───────────────────────────────────────────────────
    MacroSeries("M2",            "M2SL",      "monthly",   "m2",            "money", "Money supply M2"),
    MacroSeries("BAA10Y",        "BAA10Y",    "daily",     "baa10y",        "money", "BAA-10Y credit spread"),
    MacroSeries("AAA10Y",        "AAA10Y",    "daily",     "aaa10y",        "money", "AAA-10Y credit spread"),
    MacroSeries("MORTGAGE30",    "MORTGAGE30US","weekly",  "mortgage30",    "money", "30Y mortgage rate"),

    # ── Commodities ──────────────────────────────────────────────────────
    MacroSeries("WTI",           "DCOILWTICO","daily",     "wti",           "commodities", "Oil WTI"),
    MacroSeries("BRENT",         "DCOILBRENTEU","daily",   "brent",         "commodities", "Oil Brent"),
    MacroSeries("GOLD",          "GOLDAMGBD228NLBM","daily","gold",         "commodities", "Gold price (LBMA)"),

    # ── Risk ────────────────────────────────────────────────────────────
    MacroSeries("VIX",           "VIXCLS",    "daily",     "vix",           "risk", "VIX equity vol"),
    MacroSeries("STLFSI",        "STLFSI4",   "weekly",    "stlfsi",        "risk", "St. Louis Financial Stress"),
    MacroSeries("NFCI",          "NFCI",      "weekly",    "nfci",          "risk", "Chicago Fed financial conditions"),

    # ── Foreign rates ────────────────────────────────────────────────────
    MacroSeries("ECB_DFR",       "ECBDFR",    "daily",     "ecb_dfr",       "foreign_rates", "ECB Deposit Facility Rate"),
    MacroSeries("GBP_LT",        "IRLTLT01GBM156N","monthly","gbp_lt",      "foreign_rates", "UK 10Y benchmark"),
    MacroSeries("JPY_LT",        "IRLTLT01JPM156N","monthly","jpy_lt",      "foreign_rates", "Japan 10Y benchmark"),
)


def by_series_id(series_id: str) -> MacroSeries | None:
    for s in REGISTRY:
        if s.series_id == series_id:
            return s
    return None


def by_stem(stem: str) -> MacroSeries | None:
    for s in REGISTRY:
        if s.stem == stem:
            return s
    return None


def all_series_ids() -> list[str]:
    return [s.series_id for s in REGISTRY]
