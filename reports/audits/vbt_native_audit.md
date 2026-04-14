# VBT Pro native adoption audit — `src/strategies/`

**Date** : 2026-04-14
**Scope** : 9 fichiers de stratégies FX
**Outils** : `scripts/audit_vbt_native.py` (AST-based) + revue manuelle
**Résultats bruts** : `reports/audits/vbt_native_audit.csv`

## Résumé exécutif

| Sévérité | Findings bruts | R-REFACTOR | R-REWRITE | R-KEEP-LEGACY |
|----------|----------------|------------|-----------|----------------|
| H (haut gain) | 0 | 0 | 0 | 0 |
| M (moyen)     | 6 | 5 | 1 | 0 |
| B (bas)       | 11 | 4 | 0 | 7 |
| **Total**     | **17** | **9** | **1** | **7** |

**Conclusion** : le codebase est déjà majoritairement idiomatique. Sur
17 patterns détectés, **7 sont des faux positifs légitimes** (sémantique
différente de ce que le détecteur cible), **9 sont des refactors
directs et sans risque** à faible gain (cosmétique/cohérence), et
**1 seul nécessite un refactor sémantique avec test de régression**.

Les 3 fichiers **déjà exemplaires** (0 findings, 0 `pd`/`np` calls) :
- `mr_turbo.py`
- `ou_mean_reversion.py`
- `rsi_daily.py`

Les 2 fichiers à plus forte densité raw/np :
- `composite_fx_alpha.py` — **R-KEEP** : 6 kernels `@njit` custom,
  31 appels `np.*` dont la majorité légitime dans les kernels. Aucun
  refactor recommandé hors du P02 L349 trivial.
- `combined_portfolio_v2.py` — densité pd/np due au pipeline
  multi-étages de régime + dd-cap. Les 2 findings sont des
  `.shift(1).fillna(v)` cosmétiques.

## Classification par finding

Légende :
- **R-REFACTOR** : refactor direct, bit-equivalence attendue,
  test numérique optionnel mais recommandé.
- **R-REWRITE** : refactor non-trivial, **test de non-régression
  obligatoire** (bit-equivalence ou tolérance documentée).
- **R-KEEP-LEGACY** : le pattern est intentionnel ou le refactor
  n'apporte pas de gain (faux positif sémantique, intention claire).

---

### `daily_momentum.py`

#### L72-73 — P04 `np.log(closes / closes.shift(...))` ×2 → **R-KEEP-LEGACY**

```python
ret_s = np.log(closes / closes.shift(w_short))
ret_l = np.log(closes / closes.shift(w_long))
```

**Analyse** : `np.log(a/b)` et `np.log1p((a-b)/b)` sont mathématiquement
équivalents à la précision machine, mais le code actuel est déjà
parfaitement vectorisé, lisible et standard pour du log-momentum
cross-sectional. Le remplacement `np.log1p(closes.vbt.pct_change(w))`
n'apporte aucun gain de perf (tous deux sont vectorisés) et dégrade la
lisibilité. **Garder en l'état.**

#### L126-134 — P01 Signal ternaire manuel → **R-REWRITE**

```python
signal = pd.Series(0.0, index=close_daily.index)
signal[trend_long & rsi_ok_long] = 1.0
signal[trend_short & rsi_ok_short] = -1.0
signal = signal.shift(1)
# ...
return (signal * daily_ret * lev.fillna(1.0)).dropna()
```

**Analyse** : la fonction produit des **returns**, pas un Portfolio.
Le pattern "Series 0/1/-1 + shift + multiply" est l'ancien modèle
avant l'adoption complète de `Portfolio.from_signals`. Le refactor
idiomatique consiste à construire un vrai Portfolio via `from_signals`
avec entries/short_entries et à en extraire `.returns`. **Attention** :
la fonction est consommée par `combined_portfolio.build_combined_portfolio`
qui combine les returns de plusieurs sleeves — toute divergence
numérique impacte les tests de régression combined.

**Recommandation** : refactor R-REWRITE avec test
`assert np.allclose(pf_new.returns, pf_old_returns, atol=1e-9)`.
Impact combined à valider avec `bench_verify.py`.

**Gain** : cohérence avec `combined_core.py` (même sémantique partout),
élimination d'une branche raw-returns dans le pipeline.

#### L219 — P07 vol-target `.shift(1).fillna(1.0)` → **R-REFACTOR**

```python
lev_mult = (
    (target_vol / vol_21.clip(lower=0.01)).clip(upper=5.0).shift(1).fillna(1.0)
)
```

**Analyse** : pattern vol-target leverage dupliqué (L219 `pipeline_xs`,
L450 `pipeline_ts`, et 2 occurrences analogues dans
`combined_portfolio_v2.py`). Candidat idéal pour extraction en helper
partagé.

**Recommandation** : créer `src/framework/leverage.py::vol_target_leverage`
avec signature explicite :
```python
def vol_target_leverage(
    returns: pd.Series,
    target_vol: float,
    window: int = 21,
    vol_floor: float = 0.01,
    max_leverage: float = 5.0,
    ann_factor: float = 252.0,
    default: float = 1.0,
) -> pd.Series:
    vol = returns.vbt.rolling_std(window, minp=window, ddof=1) * np.sqrt(ann_factor)
    return (target_vol / vol.clip(lower=vol_floor)).clip(upper=max_leverage).vbt.fshift(1, fill_value=default)
```
Remplace ≥4 sites dupliqués. Bit-equivalent si `vbt.fshift` est
strictement équivalent à `.shift(1).fillna(default)` — à vérifier en
unit test.

#### L450 — P07 vol-target `.shift(1).fillna(1.0)` → **R-REFACTOR**

Mêmes remarques que L219. Même helper.

---

### `mr_macro.py`

#### L119, L123 — P05 `pd.read_parquet` → **R-KEEP-LEGACY**

```python
spread_df = pd.read_parquet(data_dir / "SPREAD_10Y2Y_daily.parquet")
unemp_df = pd.read_parquet(data_dir / "UNEMPLOYMENT_monthly.parquet")
```

**Analyse** : `_load_macro_series` charge 2 Series macro simples (pas
d'OHLCV) et les réalligne ensuite via `vbt.Resampler` + `.vbt.realign_opening`
(L151-168) — l'alignement cross-fréquence est **déjà idiomatique VBT**.
Passer par `vbt.ParquetData.pull(path)` reconstruirait un objet Data
seulement pour en extraire une Series, au prix d'un overhead inutile.

**Recommandation** : garder. Ajouter un commentaire
`# Raw load — alignment via vbt.Resampler happens in _get_aligned_macro` ?
Optionnel.

#### L358, L363 — P08 `np.broadcast_to` (session/macro masks) → **R-REFACTOR**

```python
session = pd.DataFrame(
    np.broadcast_to(session_1d[:, None], (len(close), n_cols)),
    index=close.index,
    columns=close.columns,
)
macro_ok = pd.DataFrame(
    np.broadcast_to(macro_ok_1d.values[:, None], (len(close), n_cols)),
    index=close.index,
    columns=close.columns,
)
```

**Analyse** : le DataFrame 2D n'est construit que pour pouvoir faire
`entries = (close < lower) & session & macro_ok` en raw pandas (pandas
ne broadcast pas row-wise une Series 1D contre un DataFrame 2D). Trois
alternatives :

1. **Remplacement direct** : `np.broadcast_to` est strictement équivalent
   à une vue numpy ; on peut simplifier en utilisant `session_1d[:, None].repeat(n_cols, axis=1)` ou directement
   passer par un broadcasting numpy sans wrapper DataFrame :
   ```python
   mask_2d = session_1d[:, None] & macro_ok_1d.values[:, None]
   entries = (close.values < lower.values) & mask_2d
   ```
   Plus compact, mais on perd les labels pandas en cours de route.

2. **Natif VBT** : `vbt.pd_acc.combine([sig_a, sig_b, sig_c], op=lambda *xs: functools.reduce(operator.and_, xs))`.
   Lourd et pas plus lisible.

3. **Laisser VBT broadcaster dans `from_signals`** : passer `entries`
   comme Series 1D et `short_entries` comme Series 1D, et laisser VBT
   gérer le broadcasting aux colonnes. **Mais** `(close < lower)` est
   déjà 2D par paire, donc on ne peut pas garder `entries` en 1D sans
   perdre la composante par-paire.

**Recommandation** : le refactor le plus propre est de **supprimer
`np.broadcast_to`** et utiliser `.values[:, None]` suivi d'une
multiplication numpy implicite :
```python
session_mask = session_1d[:, None]  # shape (T, 1) — broadcasts row-wise
macro_mask = macro_ok_1d.values[:, None]
entries = (close.values < lower.values) & session_mask & macro_mask
short_entries = (close.values > upper.values) & session_mask & macro_mask
entries = pd.DataFrame(entries, index=close.index, columns=close.columns)
short_entries = pd.DataFrame(short_entries, index=close.index, columns=close.columns)
```
Bit-equivalent, supprime 2 appels `np.broadcast_to`, ~4 lignes de code
économisées.

**Gain** : faible (cosmétique). **Priorité** : basse.

---

### `composite_fx_alpha.py`

#### L349 — P02 `close_daily.pct_change().fillna(0.0)` → **R-REFACTOR**

```python
returns_daily = close_daily.pct_change().fillna(0.0)
```

**Analyse** : remplacement trivial par `close_daily.vbt.pct_change().fillna(0.0)`.
Bit-equivalent. Gain : cohérence avec le reste du codebase qui utilise
l'accessor `.vbt.pct_change()`.

#### L384 — P07 `pd.Series(...).shift(1).fillna(0.0)` → **R-REFACTOR**

```python
target_w_series = (
    pd.Series(target_weights, index=close_daily.index, name="target_weight")
    .shift(1)
    .fillna(0.0)
)
```

**Analyse** : pattern causal shift avant construction de Portfolio.
Remplaçable par `.vbt.fshift(1, fill_value=0.0)`. Bit-equivalent.

#### Kernels `@njit` (5 fonctions, L48-149) → **R-KEEP-LEGACY**

Kernels `momentum_signal_nb`, `regime_weight_nb`, `vol_scaling_nb`,
`drawdown_control_nb`, `sub_portfolio_weights_nb` — non flaggés par
le scan (patterns skipés dans `@njit`). Les 31 appels `np.*` sont en
grande majorité localisés dans ces kernels, ce qui est **idiomatique
Numba** et plus rapide que toute alternative VBT-native. **Aucun
refactor.**

---

### `mr_turbo.py`, `ou_mean_reversion.py`, `rsi_daily.py`

**0 findings. Rien à faire.** Ces 3 fichiers représentent le standard
cible du codebase : `vbt.RSI`, `vbt.BBANDS`, `vbt.VWAP`,
`Portfolio.from_signals` avec stops natifs, `.vbt.rolling_std`,
`.vbt.ewm_mean`, `.vbt.resample_apply`, `@vbt.parameterized` pour les
sweeps. À utiliser comme référence dans le cheatsheet Phase 3.

---

### `combined_core.py`

#### L121 — P06 `reindex(...).fillna(0.0)` → **R-KEEP-LEGACY**

```python
aligned = allocations.reindex(
    index=prices.index, columns=prices.columns
).fillna(0.0)
```

**Analyse** : **faux positif sémantique**. Le détecteur cible les
reindex **cross-fréquence** (daily → minute, etc.). Ici c'est un
simple alignement sur `prices.index` (même fréquence) + un alignement
colonnes. `vbt.Resampler` n'est pas applicable (pas de changement de
fréquence). Garder.

#### L129 — P07 `aligned.shift(-1).fillna(0.0)` → **R-KEEP-LEGACY**

```python
allocations_shifted = aligned.shift(-1).fillna(0.0)
```

**Analyse** : **faux positif** — c'est un `shift(-1)` (forward shift,
pas causal) pour aligner la sémantique `PortfolioOptimizer` (weights à
`t` appliqués au position durant `t+1`) avec l'héritage
`weights[t] * rets[t]`. Le remplacement par `.vbt.fshift(-1, ...)` est
techniquement possible mais apporte zéro gain et le commentaire
existant (L125-128) est plus important que la forme de l'expression.
Garder.

---

### `combined_portfolio.py`

#### L243 — P06 `static.reindex(common.columns).fillna(0.0)` → **R-KEEP-LEGACY**

```python
static = static.reindex(common.columns).fillna(0.0)
```

**Analyse** : **faux positif** — c'est un column-reindex de weights
statiques vers les colonnes attendues, pas un cross-frequency alignment.
`vbt.Resampler` n'est pas applicable. Garder.

#### L247 — P08 `np.broadcast_to(static.values, common.shape)` → **R-REFACTOR**

```python
return pd.DataFrame(
    np.broadcast_to(static.values, common.shape),
    index=common.index,
    columns=common.columns,
)
```

**Analyse** : construit un DataFrame de poids constants répétés sur
toutes les lignes. Alternatives :

1. `pd.DataFrame(np.tile(static.values, (len(common), 1)), ...)` —
   plus explicite, matérialise la copie.
2. `common * 0 + static` — broadcast naturel pandas, très concis mais
   plus obscur.
3. Laisser tel quel — `np.broadcast_to` est optimal (zéro-copie).

**Recommandation** : **garder** `np.broadcast_to` (zero-copy) mais
supprimer de la liste anti-pattern — c'est le bon outil ici. Le scanner
doit peut-être apprendre à distinguer broadcast-to-DataFrame-literal
de broadcast-to-mask. **Re-classifier comme R-KEEP-LEGACY.**

#### L214-230 — Risk-parity manuel (non flaggé par scan)

**Ajout manuel** : bloc mentionné dans le plan initial, contient un fix
de bug documenté (weights > 1.40 bug 2018-03-16). **NE PAS
REFACTORISER** vers `vbt.PortfolioOptimizer` sans bit-equivalence test.
Classer R-KEEP-LEGACY explicite (avec commentaire de renvoi au fix).

---

### `combined_portfolio_v2.py`

#### L388 — P07 vol-target `.shift(1).fillna(1.0)` → **R-REFACTOR**

```python
leverage = (
    (target_vol / realized_vol.clip(lower=min_vol_floor))
    .clip(upper=max_leverage)
    .shift(1)
    .fillna(1.0)
)
```

**Analyse** : 3e occurrence du pattern vol-target leverage. Candidate
directe pour l'helper `vol_target_leverage` proposée dans
`daily_momentum.py`. **Même refactor, même test.**

#### L438 — P07 `.shift(1).fillna(0.0)` (drawdown state) → **R-REFACTOR**

```python
dd = (equity / running_max - 1.0).shift(1).fillna(0.0)
```

**Analyse** : causal shift pour éviter look-ahead dans le DD cap.
Remplacement direct par `.vbt.fshift(1, fill_value=0.0)`. Bit-equivalent
si `vbt.fshift` respecte la sémantique pandas. À vérifier en unit test
dans le helper.

---

## Plan d'action consolidé

### Backlog R-REFACTOR (direct, faible risque)

Priorité par gain :

1. **[HAUTE COHÉSION]** Créer `src/framework/leverage.py::vol_target_leverage`
   (helper factorisée) et remplacer 4 sites :
   - `daily_momentum.py:219` (pipeline_xs)
   - `daily_momentum.py:450` (pipeline_ts)
   - `combined_portfolio_v2.py:388` (compute_global_leverage)
   - potentiellement `combined_portfolio_v2.py:438` (compute_dd_cap_scale — pattern similaire mais pas vol-target)
   - **Test** : unit test bit-equivalent `fshift vs shift+fillna` + test de non-régression Sharpe sur daily_momentum pipelines.
   - **Gain** : -20 lignes dupliquées, 1 source de vérité, aligné avec VBT accessors.

2. **[COSMÉTIQUE]** Remplacer `close.pct_change().fillna(0.0)` par
   `close.vbt.pct_change().fillna(0.0)` dans `composite_fx_alpha.py:349`.
   Patch 1 ligne. Zéro risque.

3. **[COSMÉTIQUE]** Remplacer `np.broadcast_to` dans `mr_macro.py:358,363`
   par une construction pandas équivalente (ou garder et annoter — le
   gain est marginal).

4. **[COSMÉTIQUE]** Remplacer `pd.Series(...).shift(1).fillna(0.0)` par
   `.vbt.fshift(1, fill_value=0.0)` dans `composite_fx_alpha.py:384`.

### Backlog R-REWRITE (test de régression obligatoire)

1. **[SÉMANTIQUE]** `daily_momentum.py:126-134 backtest_ts_momentum_rsi` —
   migrer du pipeline `signal * ret * lev` vers un vrai
   `Portfolio.from_signals` et en extraire `.returns`.
   - **Test bloquant** : tolérance ≤ 1e-9 sur Sharpe et returns vs
     baseline, car cette fonction feed `combined_portfolio`.
   - **Gain** : cohérence, élimination d'une branche raw-returns.
   - **Risque** : moyen — sémantique des fees/slippage/leverage peut
     diverger légèrement.
   - **Recommandation** : à faire **après** avoir créé le helper
     `tests/helpers/portfolio_equivalence.py` et les baselines de
     `tests/test_vbt_refactor_invariants.py`.

### Améliorations du scanner

Deux faux positifs sémantiques à traiter pour une future itération :
- **P06 reindex_fillna** : distinguer index-reindex (cross-freq) de
  column-reindex (pur column align). Détectable via la présence du
  kwarg `index=` vs `columns=`.
- **P07 shift_fillna_causal** : exclure `shift(n)` avec `n < 0`
  (forward shift sémantique, pas causal).

Améliorations low-effort : patcher `scripts/audit_vbt_native.py` avec
2 conditions supplémentaires dans `_detect_reindex_fillna` et
`_detect_shift_fillna`.

### Ce qui ne sera PAS touché

- **Tous les kernels `@njit`** de `composite_fx_alpha.py` — optimalité
  Numba confirmée.
- **Risk-parity manuel** `combined_portfolio.py:214-230` — fix de bug
  documenté, test de régression existant à respecter.
- **`pd.read_parquet`** dans `mr_macro.py` — chargement brut simple,
  alignement déjà idiomatique VBT en aval.
- **`combined_core.py:121,129`** — faux positifs sémantiques (column
  reindex + forward shift).
- **`combined_portfolio.py:247`** — `np.broadcast_to` zero-copy est
  optimal pour du broadcast-to-constant-DataFrame.

## Prochaines phases

- **Phase 3** — cheatsheet `docs/vbt_native_cheatsheet.md` basé sur la
  doc officielle VBT (MCP `vectorbtpro__search`) + patterns observés
  dans les 3 fichiers exemplaires.
- **Phase 4** — `tests/helpers/portfolio_equivalence.py` +
  `tests/test_vbt_refactor_invariants.py` (baselines figées sur main).
- **Exécution** — backlog R-REFACTOR (priorité 1 → 4) puis R-REWRITE,
  chaque refactor atomique avec commit et re-run de `bench_verify.py`.
