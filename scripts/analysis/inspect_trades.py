"""inspect_trades — Phase B trade quality inspection (Plan CAGR docs/investigations).

Lit le CSV deals_<ts>.csv exporté par FxMultiSleeve.OnTester() (UTF-16 LE BOM,
écrit par MT5 via FILE_TXT en CP_UTF8). Apparie les deals in/out par
`position_id` pour reconstruire trades complets, puis calcule par sleeve :

- Win rate, avg win/loss, profit factor
- Top/bottom 5 trades (alerte si top-5 > 30 % PnL total)
- Holding time distribution
- Distribution PnL (mean, std, skew, kurtosis)
- Per-pair, per-hour, per-weekday, per-month breakdowns
- Outliers : trades > 5σ, holding > 3× médian

Usage :
    python scripts/analysis/inspect_trades.py \
        --deals reports/mt5/deals_20260504.csv \
        --output reports/analysis/trade_inspection_20260504.html

Pour copier le CSV depuis MT5 Common/Files :
    cp ~/.mt5/drive_c/users/thomas/AppData/Roaming/MetaQuotes/Terminal/Common/Files/deals_*.csv \
       reports/mt5/
"""
from __future__ import annotations

import argparse
import codecs
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Lecture CSV MT5 (UTF-16 LE BOM)
# ---------------------------------------------------------------------------


def read_deals(path: Path) -> pd.DataFrame:
    """Lit deals_*.csv UTF-16 LE BOM. Strip OTHER (initial deposit)."""
    raw = path.read_bytes()
    # Strip BOM if present
    if raw[:2] == b"\xff\xfe":
        text = raw[2:].decode("utf-16-le")
    else:
        text = raw.decode("utf-16-le", errors="replace")
    from io import StringIO

    df = pd.read_csv(StringIO(text))
    df["time_utc"] = pd.to_datetime(df["time_utc"], format="%Y.%m.%d %H:%M:%S")
    df = df[df["sleeve"] != "OTHER"].copy()  # exclude initial deposit balance
    return df


# ---------------------------------------------------------------------------
# Pairing in/out → trades
# ---------------------------------------------------------------------------


@dataclass
class TradeRecord:
    """Trade complet (in + out apparié par position_id)."""

    position_id: int
    sleeve: str
    symbol: str
    direction: str  # LONG / SHORT
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_price: float
    exit_price: float
    volume: float
    profit: float
    commission: float
    swap: float
    net_pnl: float  # profit + commission + swap
    holding_hours: float


def pair_deals(df: pd.DataFrame) -> pd.DataFrame:
    """Pair entry=0 (IN) avec entry=1 (OUT) sur le même position_id."""
    rows: list[TradeRecord] = []
    for pos_id, group in df.groupby("position_id"):
        if pos_id == 0:
            continue
        group_in = group[group["entry"] == 0]
        group_out = group[group["entry"] == 1]
        if group_in.empty or group_out.empty:
            continue
        d_in = group_in.iloc[0]
        d_out = group_out.iloc[-1]
        # MT5 deal type: 0=BUY, 1=SELL
        direction = "LONG" if d_in["type"] == 0 else "SHORT"
        holding = (d_out["time_utc"] - d_in["time_utc"]).total_seconds() / 3600.0
        net = float(d_out["profit"]) + float(d_out["commission"]) + float(d_out["swap"])
        rows.append(
            TradeRecord(
                position_id=int(pos_id),
                sleeve=str(d_in["sleeve"]),
                symbol=str(d_in["symbol"]),
                direction=direction,
                entry_time=d_in["time_utc"],
                exit_time=d_out["time_utc"],
                entry_price=float(d_in["price"]),
                exit_price=float(d_out["price"]),
                volume=float(d_in["volume"]),
                profit=float(d_out["profit"]),
                commission=float(d_out["commission"]),
                swap=float(d_out["swap"]),
                net_pnl=net,
                holding_hours=float(holding),
            )
        )
    return pd.DataFrame([r.__dict__ for r in rows])


# ---------------------------------------------------------------------------
# Métriques par sleeve
# ---------------------------------------------------------------------------


def sleeve_metrics(trades: pd.DataFrame) -> dict:
    """Métriques agrégées sur un sous-set de trades (1 sleeve ou all)."""
    if trades.empty:
        return {}
    pnl = trades["net_pnl"]
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    total = len(trades)
    sum_total = float(pnl.sum())
    sum_wins = float(wins.sum()) if not wins.empty else 0.0
    pf = (
        float(wins.sum() / abs(losses.sum())) if not losses.empty and losses.sum() < 0 else float("inf")
    )
    top5 = pnl.nlargest(5).sum()
    bot5 = pnl.nsmallest(5).sum()
    # Robust concentration metric: top-5 wins / sum(wins). Stable when net is ~0.
    top5_wins = wins.nlargest(5).sum() if len(wins) >= 1 else 0.0
    return {
        "trades": total,
        "win_rate_pct": 100.0 * len(wins) / total,
        "avg_win": float(wins.mean()) if not wins.empty else 0.0,
        "avg_loss": float(losses.mean()) if not losses.empty else 0.0,
        "profit_factor": pf,
        "net_pnl": sum_total,
        "mean_pnl": float(pnl.mean()),
        "std_pnl": float(pnl.std()),
        "skew_pnl": float(pnl.skew()),
        "kurt_pnl": float(pnl.kurt()),
        "top5_pnl": float(top5),
        "top5_pct_of_total": 100.0 * top5 / sum_total if sum_total > 0 else float("nan"),
        "top5_pct_of_wins": 100.0 * top5_wins / sum_wins if sum_wins > 0 else float("nan"),
        "bot5_pnl": float(bot5),
        "bot5_pct_of_total": 100.0 * bot5 / sum_total if sum_total < 0 else float("nan"),
        "median_hold_h": float(trades["holding_hours"].median()),
        "max_hold_h": float(trades["holding_hours"].max()),
    }


def per_breakdown(
    trades: pd.DataFrame, group_col: str, label: str
) -> pd.DataFrame:
    """Breakdown per group_col : count, sum_pnl, mean_pnl, win_rate."""
    if trades.empty:
        return pd.DataFrame()
    g = trades.groupby(group_col, observed=False)["net_pnl"]
    out = pd.DataFrame(
        {
            "count": g.count(),
            "sum_pnl": g.sum(),
            "mean_pnl": g.mean(),
            "win_rate_pct": g.apply(lambda s: 100.0 * (s > 0).sum() / len(s)),
        }
    )
    out.index.name = label
    return out.sort_values("sum_pnl", ascending=False)


# ---------------------------------------------------------------------------
# Outliers
# ---------------------------------------------------------------------------


def detect_outliers(trades: pd.DataFrame) -> pd.DataFrame:
    """Trades > 5σ PnL OU holding > 3× médian. Tag colonne `flag`."""
    if trades.empty:
        return pd.DataFrame()
    out = trades.copy()
    mu = out["net_pnl"].mean()
    sd = out["net_pnl"].std()
    out["pnl_zscore"] = (out["net_pnl"] - mu) / sd if sd > 0 else 0.0
    median_h = out["holding_hours"].median()
    out["hold_ratio_to_median"] = out["holding_hours"] / max(median_h, 1e-6)
    flags = []
    for _, row in out.iterrows():
        f = []
        if abs(row["pnl_zscore"]) > 5:
            f.append("PNL_5SIGMA")
        if row["hold_ratio_to_median"] > 3:
            f.append("HOLD_3X")
        flags.append(",".join(f))
    out["flag"] = flags
    return out[out["flag"] != ""].sort_values("pnl_zscore", key=abs, ascending=False)


# ---------------------------------------------------------------------------
# Rapport HTML
# ---------------------------------------------------------------------------


def _df_to_html(df: pd.DataFrame, title: str) -> str:
    if df.empty:
        return f"<h3>{title}</h3><p><em>(empty)</em></p>"
    return f"<h3>{title}</h3>\n" + df.to_html(
        float_format="%.4f", border=0, classes="data"
    )


def render_html(
    deals_path: Path,
    overall: dict,
    per_sleeve: dict[str, dict],
    sleeves_breakdowns: dict[str, dict[str, pd.DataFrame]],
    outliers_by_sleeve: dict[str, pd.DataFrame],
    n_deals: int,
    n_trades: int,
    alerts: list[str],
) -> str:
    """Génère HTML auto-contenu avec tables et alertes."""
    css = """
    body { font-family: -apple-system, monospace; max-width: 1200px;
           margin: 1rem auto; padding: 0 1rem; color: #222; }
    h1 { border-bottom: 2px solid #333; padding-bottom: 0.3rem; }
    h2 { margin-top: 2rem; border-bottom: 1px solid #aaa; }
    table.data { border-collapse: collapse; margin: 0.5rem 0; }
    table.data th, table.data td { padding: 4px 12px;
        border-bottom: 1px solid #ddd; text-align: right; }
    table.data th { background: #f5f5f5; text-align: left; }
    .alert { background: #ffe5e5; border-left: 4px solid #c00;
             padding: 0.7rem 1rem; margin: 1rem 0; }
    .ok { background: #e5f5e5; border-left: 4px solid #0a0;
          padding: 0.7rem 1rem; margin: 1rem 0; }
    pre { background: #f8f8f8; padding: 0.6rem; border-left: 3px solid #888; }
    """

    alerts_html = ""
    if alerts:
        alerts_html = '<div class="alert"><b>⚠️ ALERTS</b><ul>' + "".join(
            f"<li>{a}</li>" for a in alerts
        ) + "</ul></div>"
    else:
        alerts_html = '<div class="ok"><b>✅ Aucune alerte critique</b></div>'

    overall_df = pd.DataFrame([overall]).T.rename(columns={0: "value"})
    sleeve_df = pd.DataFrame(per_sleeve).T

    parts = [
        f"<!DOCTYPE html><html><head><meta charset='utf-8'>",
        f"<title>Trade inspection — {deals_path.name}</title>",
        f"<style>{css}</style></head><body>",
        f"<h1>Trade inspection — Phase B</h1>",
        f"<p>Source : <code>{deals_path}</code> · "
        f"Deals exportés : <b>{n_deals}</b> · "
        f"Trades reconstruits (in/out paired) : <b>{n_trades}</b></p>",
        alerts_html,
        "<h2>Métriques globales</h2>",
        overall_df.to_html(float_format="%.4f", border=0, classes="data"),
        "<h2>Métriques par sleeve</h2>",
        sleeve_df.to_html(float_format="%.4f", border=0, classes="data"),
    ]

    for sleeve, breaks in sleeves_breakdowns.items():
        parts.append(f"<h2>{sleeve} — breakdowns</h2>")
        for label, dfb in breaks.items():
            parts.append(_df_to_html(dfb, f"By {label}"))

    for sleeve, outliers in outliers_by_sleeve.items():
        if outliers.empty:
            continue
        parts.append(f"<h2>{sleeve} — outliers (>5σ ou hold > 3× médian)</h2>")
        cols = ["entry_time", "symbol", "direction", "net_pnl",
                "pnl_zscore", "holding_hours", "hold_ratio_to_median", "flag"]
        parts.append(_df_to_html(outliers[cols].head(20), "Top 20 outliers"))

    parts.append("</body></html>")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--deals", type=Path, required=True, help="deals_*.csv path")
    p.add_argument(
        "--output",
        type=Path,
        default=Path("reports/analysis/trade_inspection.html"),
    )
    args = p.parse_args(argv)

    if not args.deals.exists():
        print(f"[abort] deals file not found: {args.deals}", file=sys.stderr)
        return 2

    print(f"[read] {args.deals}")
    df = read_deals(args.deals)
    n_deals = len(df)
    print(f"[parse] {n_deals} deals after stripping OTHER (initial deposit)")

    trades = pair_deals(df)
    n_trades = len(trades)
    print(f"[pair]  {n_trades} trades reconstruits (entry → exit)")

    if trades.empty:
        print("[abort] no trades to inspect", file=sys.stderr)
        return 2

    # Enrichissements pour breakdowns
    trades["entry_hour"] = trades["entry_time"].dt.hour
    trades["entry_weekday"] = trades["entry_time"].dt.day_name()
    trades["entry_month"] = trades["entry_time"].dt.to_period("M").astype(str)

    overall = sleeve_metrics(trades)
    per_sleeve = {
        sleeve: sleeve_metrics(g) for sleeve, g in trades.groupby("sleeve")
    }

    sleeves_breakdowns = {}
    outliers_by_sleeve = {}
    for sleeve, g in trades.groupby("sleeve"):
        sleeves_breakdowns[sleeve] = {
            "symbol": per_breakdown(g, "symbol", "symbol"),
            "direction": per_breakdown(g, "direction", "direction"),
            "entry_hour": per_breakdown(g, "entry_hour", "hour_utc"),
            "entry_weekday": per_breakdown(g, "entry_weekday", "weekday"),
            "entry_month": per_breakdown(g, "entry_month", "month").head(15),
        }
        outliers_by_sleeve[sleeve] = detect_outliers(g)

    # Alerts
    alerts: list[str] = []
    for sleeve, m in per_sleeve.items():
        # Skip lucky-flag if sleeve is flat (net < 100 USD ≈ noise on 5.4 ans)
        if m.get("net_pnl", 0) < 100:
            alerts.append(
                f"<b>{sleeve}</b>: net PnL = {m['net_pnl']:.2f} USD sur 5.4 ans — "
                f"sleeve quasi-flat (PF={m['profit_factor']:.2f}), pas d'edge net"
            )
            continue
        # Robust metric : top-5 wins / total wins. >40 % = signal lucky.
        top5w = m.get("top5_pct_of_wins", 0)
        if top5w > 40:
            alerts.append(
                f"<b>{sleeve}</b>: top-5 wins = "
                f"{top5w:.1f} % des gains — risque lucky run"
            )
    # 1 paire dominante (basé sur abs PnL pour gérer les sleeves flats)
    for sleeve, g in trades.groupby("sleeve"):
        per_pair_abs = g.groupby("symbol")["net_pnl"].apply(lambda s: s.abs().sum())
        if per_pair_abs.empty or per_pair_abs.sum() == 0:
            continue
        max_share = 100 * per_pair_abs.max() / per_pair_abs.sum()
        if max_share > 60:
            top_pair = per_pair_abs.idxmax()
            alerts.append(
                f"<b>{sleeve}</b>: paire <code>{top_pair}</code> = "
                f"{max_share:.1f} % de l'activité (abs PnL) — concentration excessive"
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    html = render_html(
        args.deals,
        overall,
        per_sleeve,
        sleeves_breakdowns,
        outliers_by_sleeve,
        n_deals,
        n_trades,
        alerts,
    )
    args.output.write_text(html, encoding="utf-8")
    print(f"[write] {args.output}")

    # Console summary
    print("\n=== Per-sleeve summary ===")
    for sleeve, m in per_sleeve.items():
        print(
            f"  {sleeve:14s} trades={m['trades']:4d} "
            f"win={m['win_rate_pct']:5.1f}% PF={m['profit_factor']:5.2f} "
            f"net={m['net_pnl']:8.2f} top5={m['top5_pct_of_total']:5.1f}%"
        )
    if alerts:
        print(f"\n⚠️  {len(alerts)} alert(s) — see HTML")
        for a in alerts:
            # strip HTML
            import re

            print(f"  - {re.sub(r'<[^>]+>', '', a)}")
    else:
        print("\n✅ Aucune alerte critique")
    return 0


if __name__ == "__main__":
    sys.exit(main())
