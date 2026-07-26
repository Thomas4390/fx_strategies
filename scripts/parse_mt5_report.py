#!/usr/bin/env python3
"""Métriques du portefeuille telles que le moteur d'exécution les produit.

Le rapport client publie désormais les chiffres MT5, pas ceux de vbt : les deux
moteurs ne mesurent pas la même quantité (vbt applique un levier au poids de
position, MT5 un budget de risque borné par la distance au stop), et c'est MT5
qui exécute. Voir la note de décision du 2026-07-26.

``run_backtest_cli.py`` ne remonte que les six chiffres d'en-tête du rapport
HTML. Ce script produit le reste — courbe de balance, métriques annuelles,
ventilation par sleeve et par symbole — à partir des deux artefacts d'un run :

* le CSV par deal écrit par l'EA quand ``Inp_ExportDeals=true``, qui porte le
  ``magic`` et donc la sleeve d'origine de chaque deal ;
* le rapport HTML, dont on garde les métriques calculées par MT5 lui-même
  (Sharpe, drawdown d'équité) plutôt que de les recalculer moins bien.

⚠️ Portée de ce qu'on peut mesurer. La balance ne bouge qu'à la clôture d'une
position : une volatilité calculée dessus ignore tout le chemin intra-position
et sous-estime le risque réellement porté. Le drawdown d'**équité**, lui, est
mesuré tick par tick par MT5 — c'est celui qu'il faut publier. Les métriques
dérivées ici de la balance (CAGR, séries annuelles) sont exactes ; les métriques
de dispersion ne le sont pas et ne sont pas produites.

Usage:
    python scripts/parse_mt5_report.py
    python scripts/parse_mt5_report.py --run reports/mt5/run_2026....json
    python scripts/parse_mt5_report.py --deals <chemin.csv> --html <chemin.htm>
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
MT5_REPORTS = _PROJECT_ROOT / "reports/mt5"
DEFAULT_OUT = _PROJECT_ROOT / "results/production_report/mt5_reference.json"

# Piège documenté dans src/mt5/CLAUDE.md : sous Wine en mode portable,
# FILE_COMMON ne résout PAS vers la racine portable.
FILE_COMMON = Path.home() / (
    ".mt5/drive_c/users/thomas/AppData/Roaming/MetaQuotes/Terminal/Common/Files"
)

# L'EA nomme les sleeves d'après le magic ; on garde ses libellés et on les
# mappe vers les clés du portefeuille Python pour que les deux rapports se
# lisent l'un contre l'autre.
SLEEVE_LABELS = {
    "MR_MACRO": "MR Macro",
    "TS_MOMENTUM": "TS Momentum",
    "RSI_DAILY": "RSI Daily",
    "GOLD_MOMENTUM": "Gold Momentum",
    "H1_MOMENTUM": "H1 Momentum",
    "OTHER": "Hors sleeve",
}


# ---------------------------------------------------------------------------
# Lecture des artefacts
# ---------------------------------------------------------------------------


def _read_utf16_safe(path: Path) -> str:
    """MT5 écrit en UTF-16 LE avec BOM, sauf quand il écrit en UTF-8."""
    raw = path.read_bytes()
    for encoding in ("utf-16", "utf-8-sig", "utf-8", "cp1252"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


DEAL_TYPE_BALANCE = 2   # dépôt / retrait, pas un trade
DEAL_ENTRY_OUT = 1      # le deal qui porte le résultat de la position


def load_deals(csv_path: Path) -> pd.DataFrame:
    """Charger le CSV par deal produit par ``Inp_ExportDeals``.

    Deux corrections y sont appliquées, sans quoi les agrégats sont faux :

    * le deal de type ``balance`` (le dépôt initial) n'est pas un trade ;
    * les positions encore ouvertes à la fin du backtest sont liquidées par le
      tester lui-même, avec ``magic = 0`` : leur résultat tombait donc « hors
      sleeve ». On leur rend la sleeve de leur deal d'ouverture via
      ``position_id``.
    """
    text = _read_utf16_safe(csv_path)
    from io import StringIO

    deals = pd.read_csv(StringIO(text))
    deals["time_utc"] = pd.to_datetime(deals["time_utc"], format="%Y.%m.%d %H:%M:%S")
    deals["sleeve"] = deals["sleeve"].astype(str).str.strip()
    deals["symbol"] = deals["symbol"].fillna("").astype(str).str.strip()
    # profit net : MT5 sépare le résultat, la commission et le swap.
    deals["net"] = deals["profit"] + deals["commission"] + deals["swap"]
    deals["is_balance_op"] = deals["type"] == DEAL_TYPE_BALANCE

    entries = deals[(deals["entry"] == 0) & (~deals["is_balance_op"])]
    sleeve_of_position = entries.set_index("position_id")["sleeve"].to_dict()
    orphans = (deals["magic"] == 0) & (~deals["is_balance_op"])
    deals.loc[orphans, "sleeve"] = (
        deals.loc[orphans, "position_id"]
        .map(sleeve_of_position)
        .fillna(deals.loc[orphans, "sleeve"])
    )
    # Une liquidation de fin de test n'est pas une sortie décidée par la
    # stratégie : le rapport doit pouvoir l'isoler.
    deals["forced_close"] = orphans & (deals["entry"] == DEAL_ENTRY_OUT)

    return deals.sort_values("time_utc").reset_index(drop=True)


def _extract_html_field(text: str, label: str) -> str | None:
    pattern = rf"{re.escape(label)}:?</td>\s*<td[^>]*>(?:<b>)?([^<]+?)(?:</b>)?</td>"
    match = re.search(pattern, text)
    return match.group(1).strip() if match else None


def _to_float(value: str | None) -> float | None:
    """MT5 écrit '44 579.29' et '11 211.90 (20.11%)'."""
    if not value:
        return None
    cleaned = str(value).replace("\xa0", "").replace(" ", "").replace(" ", "")
    match = re.search(r"-?\d+\.?\d*", cleaned)
    return float(match.group()) if match else None


def _to_pct(value: str | None) -> float | None:
    """Extraire le pourcentage de 'montant (xx.xx%)'."""
    if not value:
        return None
    match = re.search(r"\(([\d.]+)%\)", str(value))
    return float(match.group(1)) if match else None


def _period_start(period: str | None) -> pd.Timestamp | None:
    """MT5 écrit la période 'M1 (2021.01.01 - 2025.12.31)'."""
    if not period:
        return None
    match = re.search(r"(\d{4}\.\d{2}\.\d{2})", period)
    return pd.Timestamp(match.group(1).replace(".", "-")) if match else None


def load_html_header(html_path: Path) -> dict[str, Any]:
    """Les métriques que MT5 calcule lui-même, gardées telles quelles."""
    text = _read_utf16_safe(html_path)
    equity_dd = _extract_html_field(text, "Equity Drawdown Maximal")
    balance_dd = _extract_html_field(text, "Balance Drawdown Maximal")
    return {
        "symbol": _extract_html_field(text, "Symbol"),
        "period": _extract_html_field(text, "Period"),
        "initial_deposit": _to_float(_extract_html_field(text, "Initial Deposit")),
        "total_net_profit": _to_float(_extract_html_field(text, "Total Net Profit")),
        "profit_factor": _to_float(_extract_html_field(text, "Profit Factor")),
        "recovery_factor": _to_float(_extract_html_field(text, "Recovery Factor")),
        "sharpe_ratio": _to_float(_extract_html_field(text, "Sharpe Ratio")),
        "total_trades": _to_float(_extract_html_field(text, "Total Trades")),
        "equity_dd_pct": _to_pct(equity_dd),
        "equity_dd_amount": _to_float(equity_dd),
        "balance_dd_pct": _to_pct(balance_dd),
        "balance_dd_amount": _to_float(balance_dd),
    }


# ---------------------------------------------------------------------------
# Dérivations
# ---------------------------------------------------------------------------


def balance_curve(
    deals: pd.DataFrame,
    initial_deposit: float,
    start: pd.Timestamp | None = None,
) -> pd.Series:
    """Balance de fin de journée, calendrier plein, sans trou.

    Les jours sans clôture reprennent la valeur de la veille : c'est bien ce
    qu'était la balance ces jours-là. Le dépôt initial est le point de départ de
    la courbe, pas un gain du premier jour.

    ``start`` est la date d'ouverture du backtest. Sans elle, la courbe
    démarrerait au premier trade et le CAGR serait calculé sur une fenêtre plus
    courte que celle réellement exposée — trois semaines de trop ici.
    """
    trades = deals[~deals["is_balance_op"]]
    daily = trades.set_index("time_utc")["net"].resample("D").sum()
    curve = initial_deposit + daily.cumsum()
    if start is not None and start < curve.index[0]:
        head = pd.Series(
            initial_deposit,
            index=pd.date_range(start, curve.index[0] - pd.Timedelta(days=1), freq="D"),
        )
        curve = pd.concat([head, curve])
    return curve.ffill()


def _max_drawdown(curve: pd.Series) -> float:
    running_max = curve.cummax()
    return float((curve / running_max - 1.0).min())


def _cagr(curve: pd.Series) -> float:
    years = (curve.index[-1] - curve.index[0]).days / 365.25
    if years <= 0 or curve.iloc[0] <= 0:
        return float("nan")
    return float((curve.iloc[-1] / curve.iloc[0]) ** (1.0 / years) - 1.0)


def yearly_metrics(deals: pd.DataFrame, curve: pd.Series) -> list[dict[str, Any]]:
    """Rendement, drawdown et activité par année civile."""
    rows: list[dict[str, Any]] = []
    for year, chunk in curve.groupby(curve.index.year):
        start = float(chunk.iloc[0])
        end = float(chunk.iloc[-1])
        trades = deals[
            (deals["time_utc"].dt.year == year)
            & (deals["entry"] == DEAL_ENTRY_OUT)
            & (~deals["is_balance_op"])
        ]
        rows.append({
            "year": int(year),
            "balance_start": start,
            "balance_end": end,
            "return_pct": (end / start - 1.0) if start else float("nan"),
            "max_dd_pct": _max_drawdown(chunk),
            "closed_trades": int(len(trades)),
            "net_profit": float(chunk.iloc[-1] - chunk.iloc[0]),
        })
    return rows


def _group_metrics(deals: pd.DataFrame, key: str) -> list[dict[str, Any]]:
    """Agréger le résultat par sleeve ou par symbole.

    On compte les deals de sortie (``entry == 1``) : ce sont eux qui portent le
    résultat d'une position. Le dépôt initial, qui n'appartient à aucune sleeve,
    est exclu.
    """
    closes = deals[
        (deals["entry"] == DEAL_ENTRY_OUT) & (~deals["is_balance_op"])
    ]
    total_net = closes["net"].sum()
    rows: list[dict[str, Any]] = []
    for name, chunk in closes.groupby(key):
        net = float(chunk["net"].sum())
        wins = chunk[chunk["net"] > 0]
        forced = chunk[chunk["forced_close"]]
        rows.append({
            key: SLEEVE_LABELS.get(str(name), str(name)) if key == "sleeve" else str(name),
            "trades": int(len(chunk)),
            "net_profit": net,
            "share_of_net_pct": float(net / total_net) if total_net else float("nan"),
            "win_rate": float(len(wins) / len(chunk)) if len(chunk) else float("nan"),
            "avg_net_per_trade": float(chunk["net"].mean()),
            "gross_profit": float(chunk.loc[chunk["net"] > 0, "net"].sum()),
            "gross_loss": float(chunk.loc[chunk["net"] <= 0, "net"].sum()),
            # Part du résultat qui vient d'une position liquidée d'office à la
            # fin du backtest plutôt que d'une sortie décidée par la stratégie.
            "forced_close_net": float(forced["net"].sum()),
        })
    return sorted(rows, key=lambda r: -r["net_profit"])


def build_reference(
    deals: pd.DataFrame, header: dict[str, Any]
) -> dict[str, Any]:
    deposit = header.get("initial_deposit") or float(
        deals.loc[deals["is_balance_op"], "profit"].iloc[0]
    )
    curve = balance_curve(deals, deposit, start=_period_start(header.get("period")))
    forced = deals[deals["forced_close"]]

    return {
        "source": "MetaTrader 5 Strategy Tester",
        "run": {
            "symbol": header.get("symbol"),
            "period": header.get("period"),
            "initial_deposit": deposit,
            "start": curve.index[0].strftime("%Y-%m-%d"),
            "end": curve.index[-1].strftime("%Y-%m-%d"),
        },
        # Sharpe et drawdown d'équité viennent de MT5 : ils sont mesurés sur
        # l'équité tick par tick, que le CSV des deals ne contient pas.
        "headline": {
            "total_net_profit": header.get("total_net_profit"),
            "final_balance": float(curve.iloc[-1]),
            "cagr": _cagr(curve),
            "sharpe_ratio_mt5": header.get("sharpe_ratio"),
            "equity_dd_pct_mt5": header.get("equity_dd_pct"),
            "balance_dd_pct_mt5": header.get("balance_dd_pct"),
            "balance_dd_pct_daily": _max_drawdown(curve),
            "profit_factor": header.get("profit_factor"),
            "recovery_factor": header.get("recovery_factor"),
            "total_trades": header.get("total_trades"),
        },
        "balance_curve": {
            "dates": [d.strftime("%Y-%m-%d") for d in curve.index],
            "balance": [round(float(v), 2) for v in curve.to_numpy()],
        },
        # Les positions encore ouvertes au dernier tick sont liquidées par le
        # tester. Ce n'est pas un résultat de stratégie : il faut pouvoir dire
        # quelle part du profit en dépend.
        "forced_closes": {
            "count": int(len(forced)),
            "net_profit": float(forced["net"].sum()),
            "share_of_net_pct": (
                float(forced["net"].sum() / header["total_net_profit"])
                if header.get("total_net_profit") else float("nan")
            ),
            "positions": [
                {
                    "symbol": row.symbol,
                    "sleeve": SLEEVE_LABELS.get(row.sleeve, row.sleeve),
                    "volume": float(row.volume),
                    "net": float(row.net),
                }
                for row in forced.itertuples()
            ],
        },
        "yearly": yearly_metrics(deals, curve),
        "by_sleeve": _group_metrics(deals, "sleeve"),
        "by_symbol": _group_metrics(deals, "symbol"),
        "caveats": [
            "Le drawdown d'équité et le Sharpe sont ceux calculés par MT5 sur "
            "l'équité tick par tick ; le CSV des deals ne permet pas de les "
            "recalculer.",
            "Les séries dérivées de la balance ignorent le chemin "
            "intra-position : elles mesurent le résultat réalisé, pas le "
            "risque porté.",
            "Les positions ouvertes au dernier tick sont liquidées par le "
            "tester au prix du moment : voir `forced_closes`.",
        ],
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _latest(pattern: str, directory: Path) -> Path | None:
    matches = sorted(directory.glob(pattern), key=lambda p: p.stat().st_mtime)
    return matches[-1] if matches else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--deals", type=Path, default=None,
                        help="CSV par deal (défaut : le plus récent de FILE_COMMON)")
    parser.add_argument("--html", type=Path, default=None,
                        help="Rapport HTML du tester")
    parser.add_argument("--run", type=Path, default=None,
                        help="JSON reports/mt5/run_*.json d'où lire le chemin du HTML")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    deals_path = args.deals or _latest("deals_*.csv", FILE_COMMON)
    if deals_path is None or not deals_path.exists():
        raise SystemExit(
            f"CSV des deals introuvable dans {FILE_COMMON}. "
            "Relancer un backtest avec --input Inp_ExportDeals=true."
        )

    html_path = args.html
    if html_path is None:
        run_json = args.run or _latest("run_*.json", MT5_REPORTS)
        if run_json is not None:
            payload = json.loads(run_json.read_text())
            html_path = Path(payload["metrics"]["report_path"])
    if html_path is None or not html_path.exists():
        raise SystemExit(f"Rapport HTML introuvable : {html_path}")

    print(f"[deals] {deals_path}")
    print(f"[html ] {html_path}")

    deals = load_deals(deals_path)
    header = load_html_header(html_path)
    reference = build_reference(deals, header)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(reference, indent=2, ensure_ascii=False))

    head = reference["headline"]
    print(
        f"\n{reference['run']['symbol']} {reference['run']['start']} → "
        f"{reference['run']['end']}"
    )
    print(
        f"  net {head['total_net_profit']:,.2f} | CAGR {head['cagr']:.2%} | "
        f"Sharpe {head['sharpe_ratio_mt5']} | equity DD "
        f"{head['equity_dd_pct_mt5']:.2f}% | {int(head['total_trades'])} trades"
    )
    print("\n  par sleeve :")
    for row in reference["by_sleeve"]:
        print(
            f"    {row['sleeve']:<16} {row['trades']:>4} trades  "
            f"net {row['net_profit']:>10,.2f}  "
            f"{row['share_of_net_pct']:>7.1%} du résultat  "
            f"win {row['win_rate']:.1%}"
        )
    print(f"\n[ok] → {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
