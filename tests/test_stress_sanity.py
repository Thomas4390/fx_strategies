"""Smoke tests for scripts/stress_test_combined.py (Phase 5).

These tests do NOT run the full 1000-sample bootstrap — that's a
one-shot diagnostic script, not something to exercise in CI. Instead
they verify the helper functions on a minimal synthetic return set
with a tiny number of iterations, to catch refactoring regressions.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SCRIPTS = _PROJECT_ROOT / "scripts"
_SRC = _PROJECT_ROOT / "src"
for p in (_SCRIPTS, _SRC):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


@pytest.fixture(scope="module")
def synthetic_strat_rets() -> dict[str, pd.Series]:
    """3 strategies × 1200 bdays — enough for 20-day blocks and WF cells."""
    rng = np.random.default_rng(20260413)
    n = 1200
    idx = pd.bdate_range("2020-01-01", periods=n, freq="B")
    return {
        "MR_Macro": pd.Series(rng.normal(0.0004, 0.006, n), index=idx),
        "XS_Momentum": pd.Series(rng.normal(0.0003, 0.009, n), index=idx),
        "TS_Momentum_RSI": pd.Series(rng.normal(0.0002, 0.008, n), index=idx),
    }


def test_block_bootstrap_indices_shape():
    from stress_test_combined import block_bootstrap_indices

    rng = np.random.default_rng(42)
    indices = block_bootstrap_indices(n=500, block_size=20, rng=rng)
    assert len(indices) == 500
    assert indices.min() >= 0
    assert indices.max() < 500
    # Block structure: 25 blocks of 20 indices, each contiguous
    # starting at a random position.
    for i in range(0, 500, 20):
        block = indices[i : i + 20]
        diffs = np.diff(block)
        assert (diffs == 1).all(), f"block starting at {i} is not contiguous"


def test_block_bootstrap_indices_rejects_bad_args():
    from stress_test_combined import block_bootstrap_indices

    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        block_bootstrap_indices(n=100, block_size=0, rng=rng)
    with pytest.raises(ValueError):
        block_bootstrap_indices(n=100, block_size=200, rng=rng)


def test_run_block_bootstrap_smoke(synthetic_strat_rets, monkeypatch):
    """Small-n bootstrap smoke — just verify it produces a valid stats object."""
    from stress_test_combined import run_block_bootstrap

    stats = run_block_bootstrap(
        synthetic_strat_rets, n_runs=10, block_size=20, seed=1
    )
    assert stats.n_runs >= 1  # a few runs may be skipped on degenerate draws
    assert 0.0 <= stats.pos_fraction <= 1.0
    assert 0.0 <= stats.target_hit_fraction <= 1.0
    # Percentiles must be ordered.
    assert stats.cagr_p05 <= stats.cagr_p50 <= stats.cagr_p95


def test_run_scenario_replay_smoke(synthetic_strat_rets):
    """Scenarios that fall outside the synthetic period must report skipped."""
    from stress_test_combined import run_scenario_replay

    scenarios = run_scenario_replay(synthetic_strat_rets)
    # Every row must have a scenario name and either skipped=True or
    # numeric metrics. Mixing string and float types is a common bug.
    assert len(scenarios) > 0
    for s in scenarios:
        assert "scenario" in s
        if not s.get("skipped"):
            for key in ("cagr", "vol", "max_dd", "sharpe"):
                assert isinstance(s[key], float)


def test_run_is_oos_split_smoke(synthetic_strat_rets):
    """IS/OOS split must produce a dict with both halves present."""
    from stress_test_combined import run_is_oos_split

    # Force the split in the middle of the synthetic window.
    split = run_is_oos_split(synthetic_strat_rets, split_date="2022-06-01")
    assert "in_sample" in split
    assert "out_of_sample" in split
    assert split["split_date"] == "2022-06-01"
    for half in ("in_sample", "out_of_sample"):
        s = split[half]
        if not s.get("skipped"):
            assert "cagr" in s
            assert "max_dd" in s
            assert "sharpe" in s


def _contains(grid: tuple[float, ...], value: float) -> bool:
    """Appartenance tolérante aux flottants, pour ne pas dépendre de l'écriture."""
    return any(g == pytest.approx(value) for g in grid)


def test_sensitivity_grid_brackets_the_production_config():
    """La grille doit contenir le point de production, sinon la table ne dit rien.

    La table de sensibilité publiée au client sert à situer la configuration de
    production dans son voisinage. Si ``PRODUCTION_TARGET_VOL`` ou
    ``PRODUCTION_MAX_LEVERAGE`` n'est pas balayé, la table décrit des réglages
    que personne ne trade. C'est arrivé : la production est passée à 0.37 / 31
    pendant que la grille balayait 0.15-0.28 × 10/15/20, et rien ne l'a signalé.
    """
    from stress_test_combined import (
        SENSITIVITY_MAX_LEVERAGES,
        SENSITIVITY_TARGET_VOLS,
    )
    from strategies.combined_portfolio_v2 import (
        PRODUCTION_MAX_LEVERAGE,
        PRODUCTION_TARGET_VOL,
    )

    assert _contains(SENSITIVITY_TARGET_VOLS, PRODUCTION_TARGET_VOL), (
        f"target_vol de production ({PRODUCTION_TARGET_VOL}) absent de la grille "
        f"{SENSITIVITY_TARGET_VOLS} : la table de sensibilité du rapport ne "
        f"contient pas la ligne de la configuration réellement tradée."
    )
    assert _contains(SENSITIVITY_MAX_LEVERAGES, PRODUCTION_MAX_LEVERAGE), (
        f"max_leverage de production ({PRODUCTION_MAX_LEVERAGE}) absent de la "
        f"grille {SENSITIVITY_MAX_LEVERAGES} : la table de sensibilité du rapport "
        f"ne contient pas la colonne du levier réellement utilisé."
    )


def test_run_parameter_sensitivity_smoke(synthetic_strat_rets):
    """Le balayage doit couvrir exactement la grille déclarée, point de prod inclus.

    Ce test comptait 18 lignes. La grille est passée de 0.15-0.28 × 10/15/20 à
    0.20-0.50 × 20/31/45 sans qu'il bronche, parce que 6 × 3 = 18 des deux côtés :
    il vérifiait une forme, pas un contenu. Les assertions portent maintenant sur
    les couples balayés et sur la présence du point de production.
    """
    from stress_test_combined import (
        SENSITIVITY_MAX_LEVERAGES,
        SENSITIVITY_TARGET_VOLS,
        run_parameter_sensitivity,
    )
    from strategies.combined_portfolio_v2 import (
        PRODUCTION_MAX_LEVERAGE,
        PRODUCTION_TARGET_VOL,
    )

    sweeps = run_parameter_sensitivity(synthetic_strat_rets)

    # Le nombre de points dérive de la grille, il n'est pas recopié.
    assert len(sweeps) == len(SENSITIVITY_TARGET_VOLS) * len(SENSITIVITY_MAX_LEVERAGES)

    swept = {(s["target_vol"], s["max_leverage"]) for s in sweeps}
    expected = {
        (tv, ml)
        for tv in SENSITIVITY_TARGET_VOLS
        for ml in SENSITIVITY_MAX_LEVERAGES
    }
    assert swept == expected, (
        "le balayage ne correspond pas au produit cartésien déclaré : "
        f"manquants={sorted(expected - swept)}, en trop={sorted(swept - expected)}"
    )

    # L'assertion qui manquait : la configuration de production est une ligne du
    # balayage, pas un point situé hors de la table publiée.
    assert any(
        tv == pytest.approx(PRODUCTION_TARGET_VOL)
        and ml == pytest.approx(PRODUCTION_MAX_LEVERAGE)
        for tv, ml in swept
    ), (
        f"le point de production ({PRODUCTION_TARGET_VOL}, "
        f"{PRODUCTION_MAX_LEVERAGE}) n'est pas balayé : la table de sensibilité "
        f"ne situe pas la configuration tradée dans sa grille."
    )

    for s in sweeps:
        assert set(s) >= {"target_vol", "max_leverage", "cagr", "vol", "max_dd", "sharpe"}
