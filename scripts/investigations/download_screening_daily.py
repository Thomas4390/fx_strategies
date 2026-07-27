#!/usr/bin/env python3
"""download_screening_daily — daily OHLC longs pour le screening momentum.

L'historique broker (SquaredFinancial demo) des symboles non-FX ne remonte
qu'à 2022-11-04, trop court pour un screening TSMOM à lookback 250 séances.
L'export depuis l'object store QuantConnect est réservé aux comptes
institutionnels. Ce script télécharge donc des séries daily OHLC longues via
l'API chart de Yahoo Finance, en miroir des symboles broker retenus.

Ces séries servent AU SCREENING UNIQUEMENT : la validation d'exécution se
fait sur les données broker (`*_daily_mt5.parquet`) et dans le tester MT5.
`scripts/investigations/check_screening_vs_broker.py` vérifie, sur la fenêtre
commune, que chaque série longue est représentative du CFD broker (corrélation
des rendements quotidiens) avant qu'elle n'entre dans un screening.

Caveats par construction, à garder en tête dans toute lecture :
- SI=F / CL=F / BZ=F / NG=F sont des futures front-month continus NON ajustés
  du roll — chaque roll introduit un gap de prix qui n'existe pas sur le CFD
  spot du broker.
- Les indices (^GSPC, ...) sont des indices cash, sans coûts de financement,
  chacun dans le fuseau de sa bourse.

Usage :
    python scripts/investigations/download_screening_daily.py [--only US500]
    # puis, obligatoire après tout ajout dans data/ :
    python scripts/update_data_manifest.py
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"

# Nom projet -> (ticker Yahoo, symbole broker miroir, note)
YAHOO_SOURCES: dict[str, tuple[str, str, str]] = {
    "XAG-USD": ("SI=F", "XAGUSD.c", "futures COMEX front-month, rolls"),
    "XTI-USD": ("CL=F", "XTIUSD", "futures NYMEX front-month, rolls"),
    "XBR-USD": ("BZ=F", "XBRUSD", "futures ICE front-month, rolls"),
    "XNG-USD": ("NG=F", "XNGUSD", "futures NYMEX front-month, rolls"),
    "US500": ("^GSPC", "US500Cash", "indice cash"),
    "US100": ("^NDX", "US100Cash", "indice cash"),
    "US30": ("^DJI", "US30Cash", "indice cash"),
    "GER40": ("^GDAXI", "GER40Cash", "indice cash, tz Berlin"),
    "JPN225": ("^N225", "JPN225Cash", "indice cash, tz Tokyo"),
    "UK100": ("^FTSE", "UK100Cash", "indice cash, tz Londres"),
}

_UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"
# range=max fait rétrograder Yahoo en barres mensuelles ; des bornes epoch
# explicites (1990 -> horizon lointain) préservent l'interval quotidien.
_CHART_URL = (
    "https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
    "?period1=631152000&period2=4102444800&interval=1d"
)


def fetch_yahoo_daily(ticker: str) -> pd.DataFrame:
    """OHLC daily complet pour un ticker Yahoo, index tz-aware UTC."""
    url = _CHART_URL.format(ticker=urllib.parse.quote(ticker))
    req = urllib.request.Request(url, headers={"User-Agent": _UA})
    with urllib.request.urlopen(req, timeout=30) as resp:
        payload = json.load(resp)

    result = payload["chart"]["result"]
    if not result:
        raise ValueError(f"{ticker}: réponse Yahoo vide ({payload['chart'].get('error')})")
    node = result[0]
    quote = node["indicators"]["quote"][0]
    idx = pd.to_datetime(node["timestamp"], unit="s", utc=True)
    df = pd.DataFrame(
        {
            "open": quote["open"],
            "high": quote["high"],
            "low": quote["low"],
            "close": quote["close"],
        },
        index=pd.DatetimeIndex(idx, name="time"),
    )
    # Yahoo intercale des barres nulles (jours fériés, données manquantes).
    df = df.dropna(how="any")
    # Doublons possibles sur la dernière barre live : garder la dernière.
    df = df[~df.index.duplicated(keep="last")].sort_index()

    # Réparations de screening, documentées :
    # 1. Les vieux futures Yahoo portent des barres où high/low sont incohérents
    #    avec open/close (243 barres sur SI=F, 69 sur BZ=F) — on resserre
    #    high/low sur l'enveloppe des quatre prix, le signal n'utilise que
    #    open/close.
    # 2. Le WTI est allé négatif en avril 2020 (CL=F à -40) : un prix <= 0 casse
    #    les rendements en pourcentage — ces jours sont retirés. Le CFD broker,
    #    lui, n'a jamais coté négatif.
    ohlc = df[["open", "high", "low", "close"]]
    df["high"] = ohlc.max(axis=1)
    df["low"] = ohlc.min(axis=1)
    df = df[(df[["open", "high", "low", "close"]] > 0).all(axis=1)]
    return df


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--only", help="ne télécharger qu'un nom projet (ex. US500)")
    parser.add_argument("--sleep", type=float, default=2.0, help="pause entre requêtes (s)")
    args = parser.parse_args()

    sources = YAHOO_SOURCES
    if args.only:
        if args.only not in sources:
            print(f"nom inconnu {args.only!r} — choix : {', '.join(sources)}")
            return 2
        sources = {args.only: sources[args.only]}

    n_ok = 0
    for name, (ticker, broker_symbol, note) in sources.items():
        try:
            df = fetch_yahoo_daily(ticker)
        except Exception as exc:  # réseau/format : continuer, bilan en sortie
            print(f"FAIL {name} ({ticker}): {exc}")
            continue
        out = DATA_DIR / f"{name}_daily_yahoo.parquet"
        df.to_parquet(out, engine="pyarrow", compression="snappy")
        print(
            f"OK   {name:8s} ({ticker:7s} -> {broker_symbol:11s}) "
            f"{len(df):6d} barres {df.index[0].date()} -> {df.index[-1].date()}  [{note}]"
        )
        n_ok += 1
        time.sleep(args.sleep)

    print(f"=== {n_ok}/{len(sources)} séries écrites dans {DATA_DIR} ===")
    if n_ok:
        print("Ne pas oublier : python scripts/update_data_manifest.py")
    return 0 if n_ok == len(sources) else 1


if __name__ == "__main__":
    sys.exit(main())
