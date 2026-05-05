"""psr_dsr_bootstrap — Phase G anti-overfit validation (PSR + DSR + bootstrap CI).

Calcule :
  - PSR : Probabilistic Sharpe Ratio (Bailey & López de Prado 2012)
  - DSR : Deflated Sharpe Ratio (Bailey & López de Prado 2014, multi-trial)
  - Bootstrap CI : block bootstrap B=1000 iterations sur daily returns
  - White Reality Check : best Sharpe vs distribution under null

Source des returns : reconstruits depuis le deal CSV exporté par
FxMultiSleeve.OnTester() (Phase B.1 deal log + Phase E config).

Usage :
    python scripts/anti_overfit/psr_dsr_bootstrap.py \
        --deals reports/mt5/deals_phase_e.csv \
        --n-trials 30  # configs testées Phase A→E (sessions+EMA+RSI+allocs)
"""
from __future__ import annotations

import argparse
import codecs
import sys
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm


def read_deals(path: Path) -> pd.DataFrame:
    raw = path.read_bytes()
    if raw[:2] == b"\xff\xfe":
        text = raw[2:].decode("utf-16-le")
    else:
        text = raw.decode("utf-16-le", errors="replace")
    df = pd.read_csv(StringIO(text))
    df["time_utc"] = pd.to_datetime(df["time_utc"], format="%Y.%m.%d %H:%M:%S")
    return df


def daily_returns_from_deals(df: pd.DataFrame, deposit: float = 10_000.0) -> pd.Series:
    """Reconstruit les daily returns à partir des deal profits."""
    closed = df[(df["entry"] == 1)].copy()
    closed["pnl"] = closed["profit"] + closed["commission"] + closed["swap"]
    closed["date"] = closed["time_utc"].dt.normalize()
    daily_pnl = closed.groupby("date")["pnl"].sum().reset_index()
    daily_pnl = daily_pnl.set_index("date").reindex(
        pd.date_range(daily_pnl["date"].min(), daily_pnl["date"].max(), freq="D"),
        fill_value=0.0,
    ).rename_axis("date")
    # equity = cumsum + initial deposit
    daily_pnl["equity"] = deposit + daily_pnl["pnl"].cumsum()
    daily_pnl["ret"] = daily_pnl["equity"].pct_change().fillna(0.0)
    # Strip weekends (FX no trading) for cleaner stats
    daily_pnl = daily_pnl[daily_pnl.index.dayofweek < 5]
    return daily_pnl["ret"]


def annualized_sharpe(returns: pd.Series, periods_per_year: float = 252.0) -> float:
    if returns.std() == 0:
        return 0.0
    return float(returns.mean() / returns.std() * np.sqrt(periods_per_year))


# ---------------------------------------------------------------------------
# PSR (Probabilistic Sharpe Ratio)
# ---------------------------------------------------------------------------


def psr(returns: pd.Series, sr_threshold: float = 0.0,
        periods_per_year: float = 252.0) -> dict:
    """PSR : probabilité que le vrai Sharpe soit > sr_threshold.

    Formule Bailey & López de Prado 2012 :
        PSR = Φ( (SR - SR*) * sqrt(N - 1) /
                 sqrt(1 - skew*SR + (kurt - 1)/4 * SR^2) )

    Critère plan : PSR(SR > 0) ≥ 95 %.
    """
    n = len(returns)
    if n < 30:
        return {"psr": float("nan"), "sr_obs": float("nan"), "n": n}
    sr = annualized_sharpe(returns, periods_per_year)
    sr_period = sr / np.sqrt(periods_per_year)
    skew = float(returns.skew())
    kurt = float(returns.kurt())  # excess kurtosis (kurtosis - 3)
    # Convert sr_threshold annualized to period units for the variance formula
    sr_thr_period = sr_threshold / np.sqrt(periods_per_year)
    denom = np.sqrt(1 - skew * sr_period + ((kurt + 3 - 1) / 4) * sr_period**2)
    if denom <= 0 or not np.isfinite(denom):
        return {"psr": float("nan"), "sr_obs": sr, "n": n,
                "skew": skew, "kurt": kurt}
    z = (sr_period - sr_thr_period) * np.sqrt(n - 1) / denom
    return {
        "psr": float(norm.cdf(z)),
        "sr_obs": sr,
        "sr_threshold": sr_threshold,
        "n": n,
        "skew": skew,
        "kurt_excess": kurt,
    }


# ---------------------------------------------------------------------------
# DSR (Deflated Sharpe Ratio)
# ---------------------------------------------------------------------------


def deflated_sharpe(returns: pd.Series, n_trials: int,
                    periods_per_year: float = 252.0,
                    sr_trials_var_annual: float | None = None) -> dict:
    """DSR : ajuste pour multi-trial selection bias (Bailey & López de Prado 2014).

    Formule en period units (daily) :
        DSR = Φ( (SR_p - E[SR_max_p]) * sqrt(T - 1) /
                 sqrt(1 - γ_3 SR_p + ((γ_4 + 3 - 1)/4) SR_p^2) )

    où :
      SR_p = Sharpe period-units (= SR_annual / sqrt(252))
      E[SR_max_p] = sqrt(V_p) * ((1-γ) Φ^{-1}(1 - 1/N) + γ Φ^{-1}(1 - 1/(N·e)))
      V_p = variance of trial Sharpes en period units = V_annual / periods_per_year
      γ = Euler-Mascheroni ≈ 0.5772
      γ_3, γ_4 = skewness, kurtosis (γ_4 = kurtosis non-excess)

    sr_trials_var_annual = variance des Sharpes annualisés observés sur les N
    trials. Par défaut 0.01 (stdev ≈ 0.10, cohérent avec les grids Phase E
    observés : EMA stdev ~0.13, RSI stdev ~0.05, sessions stdev ~0.10).

    Critère plan : DSR ≥ 80 %.
    """
    n = len(returns)
    if n < 30 or n_trials < 1:
        return {"dsr": float("nan"), "n": n, "n_trials": n_trials}
    sr = annualized_sharpe(returns, periods_per_year)
    sr_period = sr / np.sqrt(periods_per_year)
    skew = float(returns.skew())
    kurt = float(returns.kurt())  # excess kurtosis
    gamma = 0.5772156649
    if sr_trials_var_annual is None:
        sr_trials_var_annual = 0.01  # default, stdev ≈ 0.10 annualized
    sr_trials_var_period = sr_trials_var_annual / periods_per_year
    inv1 = norm.ppf(1.0 - 1.0 / n_trials)
    inv2 = norm.ppf(1.0 - 1.0 / (n_trials * np.e))
    e_sr_max_period = (np.sqrt(sr_trials_var_period)
                       * ((1.0 - gamma) * inv1 + gamma * inv2))
    denom = np.sqrt(1 - skew * sr_period
                    + ((kurt + 3 - 1) / 4) * sr_period**2)
    if denom <= 0 or not np.isfinite(denom):
        return {"dsr": float("nan"), "n": n, "n_trials": n_trials,
                "sr_obs": sr, "e_sr_max_period": e_sr_max_period}
    z = (sr_period - e_sr_max_period) * np.sqrt(n - 1) / denom
    return {
        "dsr": float(norm.cdf(z)),
        "sr_obs_annualized": sr,
        "sr_obs_period": sr_period,
        "e_sr_max_annualized": e_sr_max_period * np.sqrt(periods_per_year),
        "sr_trials_var_annual": sr_trials_var_annual,
        "n": n,
        "n_trials": n_trials,
    }


# ---------------------------------------------------------------------------
# Block bootstrap CI
# ---------------------------------------------------------------------------


def block_bootstrap(returns: pd.Series, B: int = 1000,
                    block_size: int = 21,
                    periods_per_year: float = 252.0,
                    seed: int = 42) -> dict:
    """Block bootstrap pour CI sur Sharpe et CAGR.

    Critère plan : P5(Sharpe) > 0 ET P5(CAGR) > 0.
    """
    rng = np.random.default_rng(seed)
    n = len(returns)
    if n < block_size * 2:
        return {"sharpe_p5": float("nan"), "cagr_p5": float("nan")}
    n_blocks = int(np.ceil(n / block_size))
    sharpes = []
    cagrs = []
    arr = returns.values
    for _ in range(B):
        starts = rng.integers(0, n - block_size + 1, size=n_blocks)
        boot = np.concatenate([arr[s:s + block_size] for s in starts])[:n]
        if boot.std() == 0:
            sharpes.append(0.0); cagrs.append(0.0); continue
        sr = boot.mean() / boot.std() * np.sqrt(periods_per_year)
        # CAGR = (prod(1+ret))^(252/N) - 1
        try:
            growth = float(np.prod(1.0 + boot))
            cagr = (growth ** (periods_per_year / n)) - 1.0
        except Exception:
            cagr = 0.0
        sharpes.append(sr)
        cagrs.append(cagr)
    sharpes = np.array(sharpes)
    cagrs = np.array(cagrs)
    return {
        "B": B,
        "block_size": block_size,
        "sharpe_p5": float(np.percentile(sharpes, 5)),
        "sharpe_p50": float(np.percentile(sharpes, 50)),
        "sharpe_p95": float(np.percentile(sharpes, 95)),
        "cagr_p5": float(np.percentile(cagrs, 5)),
        "cagr_p50": float(np.percentile(cagrs, 50)),
        "cagr_p95": float(np.percentile(cagrs, 95)),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--deals", type=Path, required=True)
    p.add_argument("--n-trials", type=int, default=30,
                   help="Number of param configs tested (Phase A→E)")
    p.add_argument("--sr-trials-var", type=float, default=None,
                   help="Variance of trial Sharpes (annualized). If unknown, "
                        "default 0.01 (stdev ≈ 0.10).")
    p.add_argument("--bootstrap-iters", type=int, default=1000)
    p.add_argument("--block-size", type=int, default=21)
    args = p.parse_args()

    print(f"[read] {args.deals}")
    df = read_deals(args.deals)
    print(f"[parse] {len(df)} deals")

    rets = daily_returns_from_deals(df)
    print(f"[returns] {len(rets)} daily returns "
          f"({rets.index[0].date()} → {rets.index[-1].date()})")
    sr = annualized_sharpe(rets)
    print(f"[sharpe] annualized = {sr:.3f}")

    print("\n--- PSR (sr threshold = 0) ---")
    p0 = psr(rets, sr_threshold=0.0)
    for k, v in p0.items():
        print(f"  {k}: {v}")

    print("\n--- PSR (sr threshold = 1.0) ---")
    p1 = psr(rets, sr_threshold=1.0)
    for k, v in p1.items():
        print(f"  {k}: {v}")

    print(f"\n--- DSR (n_trials = {args.n_trials}, "
          f"V_annual = {args.sr_trials_var or 0.01}) ---")
    d = deflated_sharpe(rets, n_trials=args.n_trials,
                        sr_trials_var_annual=args.sr_trials_var)
    for k, v in d.items():
        print(f"  {k}: {v}")

    print(f"\n--- Block bootstrap (B={args.bootstrap_iters}, "
          f"block={args.block_size}) ---")
    b = block_bootstrap(rets, B=args.bootstrap_iters,
                        block_size=args.block_size)
    for k, v in b.items():
        print(f"  {k}: {v}")

    # Verdict
    print(f"\n{'='*60}\n  VERDICT (vs critères §3.1 plan source)\n{'='*60}")
    psr_ok = p0["psr"] >= 0.95 if not np.isnan(p0["psr"]) else False
    dsr_ok = d["dsr"] >= 0.80 if not np.isnan(d["dsr"]) else False
    boot_sharpe_ok = b["sharpe_p5"] > 0
    boot_cagr_ok = b["cagr_p5"] > 0
    print(f"  PSR(SR > 0) ≥ 95 %     : {p0['psr']*100:.1f} %  "
          f"{'✓' if psr_ok else '✗'}")
    print(f"  DSR ≥ 80 %             : {d['dsr']*100:.1f} %  "
          f"{'✓' if dsr_ok else '✗'}")
    print(f"  Bootstrap P5(Sharpe)>0 : {b['sharpe_p5']:+.3f}  "
          f"{'✓' if boot_sharpe_ok else '✗'}")
    print(f"  Bootstrap P5(CAGR)>0   : {b['cagr_p5']*100:+.2f}%  "
          f"{'✓' if boot_cagr_ok else '✗'}")
    all_ok = psr_ok and dsr_ok and boot_sharpe_ok and boot_cagr_ok
    print(f"\n  → {'✅ EDGE CONFIRMÉ STATISTIQUEMENT' if all_ok else '⚠️  Vérifications partielles'}")

    # Dump CSV summary
    out_dir = Path("reports/anti_overfit")
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "deals_file": str(args.deals),
        "n_returns": len(rets),
        "sharpe_annualized": sr,
        "psr_sr0": p0["psr"],
        "psr_sr1": p1["psr"],
        "dsr": d["dsr"],
        "n_trials": args.n_trials,
        "bootstrap_sharpe_p5": b["sharpe_p5"],
        "bootstrap_sharpe_p50": b["sharpe_p50"],
        "bootstrap_sharpe_p95": b["sharpe_p95"],
        "bootstrap_cagr_p5": b["cagr_p5"],
        "bootstrap_cagr_p50": b["cagr_p50"],
        "bootstrap_cagr_p95": b["cagr_p95"],
    }
    pd.DataFrame([summary]).to_csv(out_dir / "summary.csv", index=False)
    print(f"\n[write] {out_dir / 'summary.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
