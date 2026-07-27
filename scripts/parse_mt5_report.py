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

⚠️ Replis : MT5 publie **deux** statistiques de repli de balance qui ne mesurent
pas la même chose — ``Balance Drawdown Maximal`` est le maximum en *monnaie*
(son pourcentage est celui de cet instant-là), ``Balance Drawdown Relative`` est
le maximum en *pourcentage*, atteint à un autre instant. Seule la seconde se
compare à une reconstruction faite depuis les deals. Le bloc
``headline["drawdowns"]`` range les cinq grandeurs sous une convention unique ;
les clés plates ``*_dd_pct_*`` qui l'entourent sont conservées telles quelles
parce que ``scripts/build_latex_report_assets.py`` les lit.

Le bloc ``provenance`` rattache le JSON à un run précis : empreintes des deux
artefacts lus, mode de simulation, inputs de l'EA, horodatage. Sans lui, la
sélection par ``mtime`` la plus récente ne laissait aucune trace de *quel* run
avait produit les chiffres publiés.

Usage:
    python scripts/parse_mt5_report.py
    python scripts/parse_mt5_report.py --run reports/mt5/run_2026....json
    python scripts/parse_mt5_report.py --deals <chemin.csv> --html <chemin.htm>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
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


def _to_leading_pct(value: str | None) -> float | None:
    """Extraire le pourcentage de '23.37% (3 263.01)'.

    ``Balance Drawdown Relative`` inverse l'ordre des deux autres champs de
    repli : le pourcentage vient d'abord, le montant est entre parenthèses.
    ``_to_pct`` y renverrait ``None`` et ``_to_float`` prendrait le pourcentage
    pour un montant.
    """
    if not value:
        return None
    match = re.match(r"\s*([\d.]+)%", str(value))
    return float(match.group(1)) if match else None


def _to_paren_amount(value: str | None) -> float | None:
    """Extraire le montant de '23.37% (3 263.01)'."""
    if not value:
        return None
    match = re.search(r"\(([-\d.\s  ]+)\)", str(value))
    return _to_float(match.group(1)) if match else None


def _extract_inputs(text: str) -> dict[str, str]:
    """Les inputs de l'EA, tels que l'en-tête HTML les récapitule.

    C'est la seule trace de la configuration qui a produit les chiffres : ni le
    JSON de run ni le CSV des deals ne la portent.
    """
    return dict(re.findall(r"<b>(Inp_[A-Za-z0-9_]+)=([^<]*)</b>", text))


def _period_start(period: str | None) -> pd.Timestamp | None:
    """MT5 écrit la période 'M1 (2021.01.01 - 2025.12.31)'."""
    if not period:
        return None
    match = re.search(r"(\d{4}\.\d{2}\.\d{2})", period)
    return pd.Timestamp(match.group(1).replace(".", "-")) if match else None


def _period_end(period: str | None) -> pd.Timestamp | None:
    """La borne droite de 'M1 (2021.01.01 - 2026.04.30)'."""
    dates = re.findall(r"(\d{4}\.\d{2}\.\d{2})", period or "")
    return pd.Timestamp(dates[-1].replace(".", "-")) if len(dates) >= 2 else None


def load_html_header(html_path: Path) -> dict[str, Any]:
    """Les métriques que MT5 calcule lui-même, gardées telles quelles."""
    text = _read_utf16_safe(html_path)
    equity_dd = _extract_html_field(text, "Equity Drawdown Maximal")
    equity_dd_rel = _extract_html_field(text, "Equity Drawdown Relative")
    balance_dd = _extract_html_field(text, "Balance Drawdown Maximal")
    balance_dd_rel = _extract_html_field(text, "Balance Drawdown Relative")
    return {
        "expert": _extract_html_field(text, "Expert"),
        "inputs": _extract_inputs(text),
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
        "equity_dd_relative_pct": _to_leading_pct(equity_dd_rel),
        "balance_dd_pct": _to_pct(balance_dd),
        "balance_dd_amount": _to_float(balance_dd),
        # Le maximum de repli de balance exprimé en pourcentage — atteint à un
        # autre instant que le maximum en monnaie ci-dessus. C'est celui-ci, et
        # lui seul, qui se compare à une reconstruction depuis les deals.
        "balance_dd_relative_pct": _to_leading_pct(balance_dd_rel),
        "balance_dd_relative_amount": _to_paren_amount(balance_dd_rel),
    }


# Les libellés du Strategy Tester pour ``Model``. Le run publié tourne en
# ``Model=1`` (barres M1) et non en ticks réels : ce fait n'apparaît ni dans le
# rapport HTML ni dans le JSON de run, seul le .ini le porte.
MODEL_LABELS = {
    "0": "Every tick",
    "1": "1 minute OHLC",
    "2": "Open prices only",
    "3": "Math calculations",
    "4": "Every tick based on real ticks",
}

_TESTER_INI_FIELDS = (
    "Expert", "Symbol", "Period", "Model", "Spread",
    "FromDate", "ToDate", "Deposit", "Leverage",
)


def load_tester_ini(ini_path: Path) -> dict[str, Any]:
    """Les réglages de simulation, lus dans le .ini UTF-16 écrit par le CLI."""
    text = _read_utf16_safe(ini_path)
    values: dict[str, Any] = {}
    for field in _TESTER_INI_FIELDS:
        match = re.search(rf"^{field}=(.*?)\s*$", text, re.MULTILINE)
        if match:
            values[field.lower()] = match.group(1)
    if "model" in values:
        values["model_label"] = MODEL_LABELS.get(values["model"], "inconnu")
    return values


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


def balance_path_by_deal(deals: pd.DataFrame, initial_deposit: float) -> pd.Series:
    """Balance après chaque deal, à la granularité native du CSV.

    ``balance_curve`` agrège par jour : un creux qui se creuse et se referme
    dans la même journée y est invisible (celui du 21 août 2023 coûte 0,62 point
    de repli). Cette série-ci ne rate rien de ce que le CSV contient — elle sert
    à reconstruire le repli, pas à porter une chronologie.
    """
    trades = deals[~deals["is_balance_op"]]
    return pd.concat([
        pd.Series([initial_deposit]),
        initial_deposit + trades["net"].cumsum(),
    ], ignore_index=True)


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


def _as_positive_pct(fraction: float | None) -> float | None:
    """Fraction signée → pourcentage positif, la convention du bloc `drawdowns`."""
    return None if fraction is None else abs(float(fraction)) * 100.0


def _drawdown_block(
    deals: pd.DataFrame,
    header: dict[str, Any],
    curve: pd.Series,
    deposit: float,
) -> dict[str, Any]:
    """Les replis sous une convention unique : pourcentage positif.

    23.37 s'y lit « −23,37 % ». MT5 en publie quatre, la balance en reconstruit
    deux, et une seule paire est comparable : ``balance_relative_mt5_pct`` avec
    ``balance_relative_per_deal_pct``. Les rapprocher de ``balance_max_money_*``
    revient à comparer un maximum en monnaie à un maximum en pourcentage.
    """
    return {
        "unit": "pourcentage positif (ampleur du repli)",
        # Mesuré tick par tick par MT5, positions ouvertes comprises : le CSV
        # des deals ne permet pas de le recalculer. Ici les deux conventions
        # MT5 coïncident, le repli d'équité publié n'est donc pas ambigu.
        "equity_max_money_mt5_pct": header.get("equity_dd_pct"),
        "equity_relative_mt5_pct": header.get("equity_dd_relative_pct"),
        # Maximum en MONNAIE : le pourcentage est celui de cet instant-là.
        "balance_max_money_mt5_pct": header.get("balance_dd_pct"),
        "balance_max_money_mt5_amount": header.get("balance_dd_amount"),
        # Maximum en POURCENTAGE, atteint à un autre instant.
        "balance_relative_mt5_pct": header.get("balance_dd_relative_pct"),
        "balance_relative_mt5_amount": header.get("balance_dd_relative_amount"),
        # Reconstructions depuis le CSV. La journalière agrège par
        # ``resample("D")`` et rate les creux qui se referment dans la journée ;
        # celle par deal ne rate rien de ce que le CSV contient.
        "balance_relative_daily_pct": _as_positive_pct(_max_drawdown(curve)),
        "balance_relative_per_deal_pct": _as_positive_pct(
            _max_drawdown(balance_path_by_deal(deals, deposit))
        ),
    }


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------

# Le tester liquide d'office les positions encore ouvertes au dernier tick : le
# dernier deal tombe donc sur la fin de la fenêtre simulée. Un CSV dont le
# dernier deal précède de plus d'une semaine la fin du rapport HTML vient d'un
# autre run — le cas se produit sans bruit, puisque le CSV est sélectionné par
# ``mtime`` et le HTML par le JSON de run.
DEALS_WINDOW_TOLERANCE_DAYS = 7


def _artifact_fingerprint(path: Path | None) -> dict[str, Any] | None:
    """Chemin, date de modification, taille et empreinte d'un artefact lu.

    Le nom du CSV des deals vient de l'heure *simulée* du tester, pas de l'heure
    réelle : deux runs sur la même fenêtre écrasent le même fichier. Le ``mtime``
    date le contenu, le sha256 l'identifie.
    """
    if path is None or not path.exists():
        return None
    stat = path.stat()
    return {
        "path": str(path),
        "mtime_utc": datetime.fromtimestamp(
            stat.st_mtime, tz=timezone.utc
        ).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "size_bytes": stat.st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def build_provenance(
    deals: pd.DataFrame,
    header: dict[str, Any],
    deals_path: Path | None = None,
    html_path: Path | None = None,
    run_json_path: Path | None = None,
) -> dict[str, Any]:
    """Ce qui rattache ce JSON à un run précis plutôt qu'à « le plus récent »."""
    trades = deals[~deals["is_balance_op"]]
    last_deal = trades["time_utc"].max() if len(trades) else None
    period_end = _period_end(header.get("period"))

    provenance: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        ),
        "generator": "scripts/parse_mt5_report.py",
        "expert": header.get("expert"),
        "deals_csv": _artifact_fingerprint(deals_path),
        "html_report": _artifact_fingerprint(html_path),
        "deals_window": {
            "first_deal_utc": (
                trades["time_utc"].min().strftime("%Y-%m-%d %H:%M:%S")
                if len(trades) else None
            ),
            "last_deal_utc": (
                last_deal.strftime("%Y-%m-%d %H:%M:%S") if last_deal is not None
                else None
            ),
            "deal_rows": int(len(trades)),
        },
        "html_period": header.get("period"),
        # Faux = les deux artefacts ne viennent pas du même run.
        "deals_match_html_period": (
            bool(
                last_deal
                >= period_end - pd.Timedelta(days=DEALS_WINDOW_TOLERANCE_DAYS)
            )
            if last_deal is not None and period_end is not None else None
        ),
        "ea_inputs": header.get("inputs") or {},
    }

    if run_json_path is not None and run_json_path.exists():
        payload = json.loads(run_json_path.read_text())
        # Le JSON de run n'atteste de rien s'il décrit un autre rapport que
        # celui qu'on vient de lire : `_latest()` rend le plus récent, et
        # `--html` peut pointer ailleurs. Mieux vaut pas de provenance qu'une
        # provenance fausse.
        reported = (payload.get("metrics") or {}).get("report_path")
        if html_path is not None and reported is not None and (
            Path(reported).resolve() != html_path.resolve()
        ):
            return provenance
        run_json = _artifact_fingerprint(run_json_path) or {}
        run_json["run_id"] = payload.get("run_id")
        run_json["ini_path"] = payload.get("ini_path")
        provenance["run_json"] = run_json
        ini_path = Path(payload["ini_path"]) if payload.get("ini_path") else None
        if ini_path is not None and ini_path.exists():
            provenance["tester"] = load_tester_ini(ini_path)

    return provenance


def build_reference(
    deals: pd.DataFrame,
    header: dict[str, Any],
    deals_path: Path | None = None,
    html_path: Path | None = None,
    run_json_path: Path | None = None,
) -> dict[str, Any]:
    deposit = header.get("initial_deposit") or float(
        deals.loc[deals["is_balance_op"], "profit"].iloc[0]
    )
    curve = balance_curve(deals, deposit, start=_period_start(header.get("period")))
    forced = deals[deals["forced_close"]]

    return {
        "source": "MetaTrader 5 Strategy Tester",
        "provenance": build_provenance(
            deals, header, deals_path, html_path, run_json_path
        ),
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
            # Les quatre clés `*_dd_pct_*` ci-dessus mélangent deux conventions
            # (fraction signée pour la reconstruction, pourcentage positif pour
            # MT5) et deux grandeurs (maximum en monnaie contre maximum en
            # pourcentage). Elles restent en place parce que le générateur LaTeX
            # les lit ; ce bloc-ci est la version cohérente et complète.
            "drawdowns": _drawdown_block(deals, header, curve, deposit),
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
            "Les replis se lisent dans `headline.drawdowns`, tous en "
            "pourcentage positif ; les clés plates `*_dd_pct_*` gardent deux "
            "conventions et deux grandeurs pour ne pas casser le générateur "
            "LaTeX qui les consomme.",
            "La reconstruction journalière du repli de balance est plus "
            "grossière que celle par deal : elle rate les creux qui se "
            "referment dans la journée.",
        ],
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _latest(pattern: str, directory: Path) -> Path | None:
    matches = sorted(directory.glob(pattern), key=lambda p: p.stat().st_mtime)
    return matches[-1] if matches else None


def _pick_deals_for_period(directory: Path, period_end: pd.Timestamp | None) -> Path | None:
    """Choisir le CSV dont la fenêtre correspond au rapport HTML lu.

    Sélectionner par `mtime` seul est un piège actif : le nom du CSV vient de
    l'heure *simulée* du tester, si bien qu'un backtest sur une fenêtre plus
    courte lancé plus tard laisse son fichier en tête du classement. Le 2026-07-26,
    `deals_20251230T2359.csv` (fenêtre courte) était plus récent que
    `deals_20260429T2359.csv` (fenêtre publiée) : relancer ce script sans argument
    aurait produit un CAGR de 40,47 % au lieu de 35,44 %, avec les 851 trades et
    le profit net du HTML inchangés — l'incohérence était indétectable à l'œil.

    On retient donc le CSV dont le dernier deal tombe dans la fenêtre du HTML, le
    plus récent en cas d'égalité. Sans candidat, on rend `None` et l'appelant
    échoue plutôt que de publier un mélange.
    """
    candidates = sorted(directory.glob("deals_*.csv"), key=lambda p: p.stat().st_mtime)
    if not candidates or period_end is None:
        return candidates[-1] if candidates else None

    tol = pd.Timedelta(days=DEALS_WINDOW_TOLERANCE_DAYS)
    matching = []
    for path in candidates:
        try:
            last = load_deals(path)["time_utc"].max()
        except Exception:  # noqa: BLE001 - un CSV illisible n'est pas un candidat
            continue
        if last is not None and period_end - tol <= last <= period_end + tol:
            matching.append(path)
    return matching[-1] if matching else None


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

    # Le HTML d'abord : c'est lui qui définit la fenêtre à laquelle le CSV doit
    # correspondre.
    run_json = args.run or _latest("run_*.json", MT5_REPORTS)
    html_path = args.html
    if html_path is None and run_json is not None:
        payload = json.loads(run_json.read_text())
        html_path = Path(payload["metrics"]["report_path"])
    if html_path is None or not html_path.exists():
        raise SystemExit(f"Rapport HTML introuvable : {html_path}")

    period_end = _period_end(load_html_header(html_path).get("period"))
    deals_path = args.deals or _pick_deals_for_period(FILE_COMMON, period_end)
    if deals_path is None or not deals_path.exists():
        raise SystemExit(
            f"Aucun CSV de deals ne couvre la fenêtre du rapport "
            f"({load_html_header(html_path).get('period')}) dans {FILE_COMMON}. "
            "Relancer le backtest avec --input Inp_ExportDeals=true, ou passer "
            "--deals explicitement."
        )

    print(f"[deals] {deals_path}")
    print(f"[html ] {html_path}")
    print(f"[run  ] {run_json}")

    deals = load_deals(deals_path)
    header = load_html_header(html_path)
    reference = build_reference(deals, header, deals_path, html_path, run_json)

    # Le CSV est choisi par `mtime`, le HTML par le JSON de run : rien ne
    # garantit qu'ils viennent du même backtest, et l'incohérence est muette.
    if reference["provenance"]["deals_match_html_period"] is False:
        print(
            f"\n[abort] Le CSV des deals s'arrête le "
            f"{reference['provenance']['deals_window']['last_deal_utc']}, hors de "
            f"la fenêtre du rapport HTML ({header.get('period')}). Les deux "
            f"artefacts ne viennent pas du même backtest.\n"
            f"        CSV  : {deals_path}\n"
            f"        HTML : {html_path}\n"
            f"        Rien n'a été écrit. Relancer le backtest sur la fenêtre "
            f"voulue, ou passer --deals et --html du même run.",
            file=sys.stderr,
        )
        return 2

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
