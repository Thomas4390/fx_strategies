# Réconciliation vbt ↔ MT5 ↔ QuantConnect — plan d'implémentation

> **Pour workers agentiques :** ce plan s'exécute à froid, sans contexte de la session
> d'origine. Chaque phase a des critères d'acceptation chiffrés. Les étapes utilisent
> la syntaxe checkbox (`- [ ]`). Lire d'abord `docs/research/gold_2026-07-25_momentum_sizing.md`.

**Objectif** — faire converger les trois moteurs sur la sleeve Gold Momentum, puis
généraliser la méthode aux trois sleeves FX. Deux cibles distinctes, et c'est le point
central de ce plan :

| Paire | Données | Cible | Justification |
|---|---|---|---|
| **vbt ↔ QC** | **identiques** | écart **quasi nul** (< 2% sur chaque métrique) | `data/XAU-USD_minute_qc.parquet` a été exporté depuis QuantConnect. Toute divergence est donc 100% imputable à la sémantique des moteurs, pas aux données. C'est exactement soluble. |
| **vbt ↔ MT5** | différentes | écart **borné et expliqué** | Flux broker SquaredFinancial, bornes de barres D1 en heure serveur, spread bid-ask réel, swap. L'écart résiduel ne doit pas être annulé mais **attribué ligne par ligne**. |

Ne jamais viser un écart nul sur MT5 : ce serait le signe qu'on a idéalisé le backtest
broker, pas qu'on a réconcilié quoi que ce soit.

**État de départ** — commits `3928ad6` et `6035e15` sur `main`.

> **Révision du 2026-07-25 (session d'exécution).** La colonne vbt d'origine avait
> été mesurée avec le vol-targeting inactif. Re-mesurée aux défauts documentés
> (`target_vol=0.25`, `max_leverage=3.0`), elle change tout le diagnostic : l'écart
> de volatilité vbt ↔ QC, qui motivait la phase 1, n'existe pas.

| | vbt (publié) | vbt (re-mesuré) | QC | MT5 |
|---|---|---|---|---|
| Sharpe | 0.726 | 0.700 | 0.575 – 0.797 | non mesuré |
| Vol ann. | 12.19% | **23.74%** | **23.3%** | non mesuré |
| CAGR | 8.44% | **18.65%** | 20.17% | non mesuré |
| maxDD | −23.31% | **−46.51%** | −51.9% | non mesuré |
| trades | 50 | 50 | 128 ordres | — |

---

## Principe directeur : réconcilier les artefacts intermédiaires, pas les métriques finales

Comparer des Sharpe est le test le plus faible possible — deux implémentations différentes
peuvent produire le même Sharpe par compensation d'erreurs, et un écart de Sharpe ne dit
pas *où* est le problème.

La méthode est une échelle. On descend jusqu'au premier barreau qui casse, on le répare,
on remonte. Chaque moteur émet le **même CSV journalier** et on les diffe directement :

```
date, close, score, target_weight, position_units, equity
```

| Barreau | Ce qu'on compare | Si ça casse, la cause est |
|---|---|---|
| 1. `close` | la série de clôtures quotidiennes | bornes de barre / fuseau / calendrier |
| 2. `score` | le score momentum jour par jour | indexation du lookback, warmup, look-ahead |
| 3. `target_weight` | la taille visée | sémantique de sizing (le défaut connu) |
| 4. `position_units` | la position effective | timing de fill, arrondi de lots, marge |
| 5. `equity` | la P&L | coûts, spread, swap |

**Un écart au barreau N rend tous les barreaux > N ininterprétables.** Ne jamais analyser
un écart de Sharpe avant que les barreaux 1 à 4 soient verts.

---

## Phase 0 — Geler la spécification et bâtir le harnais

Sans spécification unique écrite, les trois implémentations dérivent à chaque correctif.

- [x] `docs/specs/gold_momentum_spec.md` — écrite. Une correction par rapport à l'énoncé :
      le lookback est bien indexé sur `close[t]`, **pas** `close[t-1]` comme ce plan le
      supposait. C'est l'implémentation de référence qui fait foi, et l'idéalisation
      qui en découle (décider et exécuter au même close) est documentée au §6 comme poste
      d'attribution plutôt que corrigée en douce.
- [x] `gold_momentum.emit_daily_trace()` — signature `(pf, indicator, path)` et non
      `(pf, path)` : le score n'est pas récupérable depuis le portefeuille. 2113 lignes
      émises, soit 2363 séances moins 250 de warmup.
- [x] Trace QC (projet `34489845`) — compile (`BuildSuccess`), backtest relancé, métriques
      **inchangées** (128 ordres, Sharpe 0.575, DD 51.9%), donc la trace n'a pas altéré le
      trading. ⚠️ `read_backtest_logs` que ce plan citait **n'existe pas** dans le serveur
      MCP, et l'export ObjectStore est réservé aux comptes Institutional : la récupération
      est **manuelle** par l'interface web. D'où la double émission (ObjectStore + log
      préfixé `TRACE,`). Voir spec §9.
- [x] Trace MQL5 — `Inp_Gold_Trace` à `false` par défaut, compile sous Wine/MetaEditor
      (0 erreur, 0 warning). Non exécutée : le tester est bloqué, c'est la phase 3.
- [x] `scripts/reconcile_three_way.py` — descend l'échelle, s'arrête au premier barreau
      cassé et marque les suivants ininterprétables. Deux profils de tolérance selon la
      paire. Sortie non nulle en cas de dépassement.

**Acceptation — atteinte.** Vérifié sur trois cas fabriqués : traces identiques (exit 0) ;
défaut injecté au barreau `score` (nommé correctement, barreaux supérieurs marqués
ininterprétables, exit 1) ; calendrier partiel avec défaut au seul barreau `equity`
(2013 séances communes sur 2113, barreaux 1-4 verts, exit 1).

**Écarts relevés à la lecture des trois portages, à traiter en phases 2 et 4** — la phase 0
outille, elle ne répare pas :

| | vbt | QC | MT5 |
|---|---|---|---|
| rendements de σ21 | arithmétiques | **log** | **log** |
| plancher de σ | 0.01 | **0.05** | **0.05** |
| décalage causal du levier | **oui, 1 séance** | non | non |
| repli si historique court | levier 1.0 | pas de position | **σ = 0.16** |
| capital initial | 1 000 000 | 100 000 | sub-equity de la sleeve |

---

## Phase 1 — ~~Corriger le défaut de sizing vbt~~ — SANS OBJET (vérifié le 2026-07-25)

**Le défaut n'existe pas, et le correctif prescrit est rejeté par le moteur.** Cette phase
est close sans action ; ce qui suit est le constat, à conserver pour ne pas la rouvrir.

Le diagnostic d'origine — `size_type="percent"` plafonnerait l'ordre au cash disponible,
empêchant le tableau `leverage` de relever l'exposition — a été démenti par la mesure :

| mesure aux défauts (`target_vol=0.25`) | valeur | ce que le plan annonçait |
|---|---|---|
| volatilité réalisée | **23.74%** (cible 25%) | 12.19% |
| exposition brute, en position | **189.67%** (levier médian 2.007) | — |
| exposition brute, toutes barres | 102.66% | 52.7% |

Volatilité recoupée par trois méthodes indépendantes : 23.74 / 23.74 / 23.86%. La vol
réalisée est donc **déjà dans la tolérance 25% ± 3 pp** que cette phase visait comme
critère d'acceptation, et `percent` ne plafonne rien.

**D'où venaient les chiffres publiés** : `target_vol=None` (sizing plat à 1×) les
reproduit — 12.46% de vol, −24.47% de DD, 54.13% d'exposition. La colonne vbt de la §7 du
rapport de recherche était un run à levier 1× comparé à une cible de 25%. L'arithmétique
le confirme : 52.7% ≈ 100% × 54.1% de temps passé en position.

**Le correctif prescrit est impossible tel quel** : `from_signals` lève
`ValueError: Target size types are not supported`. Les deux patrons cités en référence
(`daily_momentum.py:224-233`, `composite_fx_alpha.py:389-398`) utilisent `from_orders`,
qui supporte le sizing en cible. La sleeve or a besoin de `from_signals` pour ses
transitions edge-triggered, son `sl_stop` et le seam `signal_func_nb` où se branchent les
overlays de sizing — les deux API ne sont pas interchangeables. Y migrer serait une refonte,
pas un changement d'une ligne, et rien ne la justifie.

**Conséquence pour la phase 2** : elle part d'une situation bien meilleure qu'annoncé.
vbt et QC concordent déjà sur la volatilité (23.74% vs 23.3%) et le Sharpe. Restent le
maxDD (−46.51% vs −51.9%) et le CAGR (18.65% vs 20.17%), tous deux cohérents avec la
différence de timing de fill déjà documentée (barreau 4).

---

## Phase 2 — vbt ↔ QC : viser l'écart quasi nul

Mêmes données, donc chaque écart restant a une cause identifiable. Traiter dans cet ordre,
un seul changement à la fois, en re-mesurant à chaque étape.

- [ ] **Barreau 1 — bornes de barre.** vbt fait `resample('D')` sur des minutes converties
      en heure de New York naïve ; QC livre des barres Daily CFD dont la borne suit la
      convention 17:00 New York. Diffe les deux séries de clôtures. Si elles divergent,
      aligner vbt sur la convention QC (`origin` du resample), **pas l'inverse** — QC est
      le producteur des données.
- [ ] **Barreau 2 — score.** Comparer jour par jour. Pièges attendus : `RollingWindow[0]`
      en QC est la barre courante alors que vbt décale déjà d'un cran ; le warmup QC de 252
      barres ne consomme pas exactement les mêmes jours que le `dropna` de vbt.
- [ ] **Barreau 3 — poids cible.** Après la phase 1, la formule doit être identique.
      Vérifier `ddof=1` des deux côtés et le plancher de σ (vbt `vol_floor`, QC
      `max(sigma, 0.05)`).
- [ ] **Barreau 4 — timing de fill.** Différence structurelle connue et documentée
      (`docs/quantconnect_validation_report.md` §2.3) : vbt remplit au **close de la barre
      de signal**, QC au **T+1 open**. Deux options — trancher et écrire la décision dans
      la spec :
      - aligner vbt sur QC via `price="nextopen"` dans `from_signals` (plus réaliste)
      - ou accepter l'écart et le quantifier comme poste attribué
      **Recommandation : aligner vbt sur QC.** Remplir au close de la barre qui produit le
      signal est une idéalisation, et le MT5 ne le fait pas non plus.
- [ ] **Barreau 5 — coûts.** vbt applique `slippage` aux entrées de signal ; QC applique
      son `ConstantSlippageModel` partout, y compris aux sorties. Le bug #3 du portage
      précédent (§5.3 du rapport QC) est exactement ce piège. Uniformiser et documenter.

**Acceptation** : sur la période complète, écart ≤ **2%** en relatif sur CAGR, volatilité
et maxDD, et ≤ **0.05** en absolu sur le Sharpe. Tout écart supérieur doit être attribué
nommément à une différence de moteur, jamais laissé « résiduel ».

---

## Phase 3 — Débloquer le tester MT5

Symptôme constaté : le terminal démarre, s'authentifie chez le broker
(`SquaredFinancialSC-MT5 Demo`, ping 100 ms), charge l'EA sur un graphique
`EURUSD.c,Daily` **en mode GUI** et le log s'arrête là. Le tester ne démarre jamais, alors
que le `.ini` UTF-16 contient bien `[Tester] Symbol=XAUUSD.c`. Aucun log n'apparaît dans
`Tester/logs/`.

Hypothèses à tester dans cet ordre (de la moins à la plus invasive) :

- [x] ~~**Le symbole n'existe pas chez le broker.**~~ **Infirmé.** `XAUUSD.c` existe et
      porte des données : `XAUUSD.c,M1: 229815 ticks, 57565 bars generated`. L'absence
      d'historique dans `Bases/` signalait un cache local vide, pas un symbole manquant —
      le dossier `History/` du broker est vide **pour tous les symboles**, FX compris.
- [x] **Le run FX échoue aussi** — donc la panne n'avait rien à voir avec l'or, comme ce
      plan le prévoyait. C'est ce test qui a payé.
- [x] **Cause racine : le modèle de simulation.** `DEFAULT_MODEL = 4` (« every tick based
      on real ticks ») exige des ticks que le terminal en `/config:` ne parvient pas à
      télécharger — `preliminary downloading of history ticks canceled`, puis
      `no history data, stop testing`. Ni l'état de session, ni le chemin `/config:`, ni
      l'encodage de l'INI n'étaient en cause.
- [x] **Correctif : `--model 1`** (OHLC M1). ⚠️ Contrepartie assumée : les fills sont
      interpolés et le Sharpe flatté ; tout chiffre MT5 obtenu ainsi est un **majorant**.
- [x] **Obstacle secondaire** : `Inp_AllocGoldMomentum = 0.0` par défaut — la sleeve or ne
      trade jamais en configuration de production. Les runs de réconciliation doivent
      l'isoler (allocation 1.0, les autres à 0).

**Acceptation — atteinte.** `reports/mt5/run_20260725T200546Z.json`, `exit_code=0`,
5.3 ans simulés, sleeve or seule : 35 trades, Sharpe 0.73, profit net +173.8 %.

- [ ] **Reste ouvert : la trace journalière MT5 n'est pas produite.** `Inp_Gold_Trace=true`
      est bien écrit dans `[TesterInputs]` et l'allocation passée par le même mécanisme est
      honorée, mais `WriteTraceRow` n'est jamais atteint — pas même son avertissement
      d'échec d'ouverture. Preset `.set` caché et régénération du preset par défaut :
      écartés par test. Le journal du tester porte en attendant score, levier, lots et prix
      par trade, donc l'attribution reste faisable par parsing.

---

## Phase 4 — vbt ↔ MT5 : borner et attribuer l'écart

Ne commencer qu'une fois la phase 3 verte. Même échelle, mais la cible change : on ne
cherche pas l'égalité, on cherche à **expliquer** chaque poste.

- [ ] **Barreau 1 — données.** Exporter le D1 or du broker via
      `src/mt5/Scripts/FxExportRates.mq5`, importer avec
      `src/mt5/bridge/import_mt5_rates.py`, puis comparer à la série QC. Mesurer et
      publier : nombre de séances communes, écart médian et P95 des clôtures, décalage de
      borne de journée. **Ce tableau est un livrable en soi** : il fixe le plancher
      d'écart irréductible entre les deux moteurs.
- [ ] **Barreau 2 — score.** Rejouer le score vbt sur les données broker. S'il diverge
      encore, le bug est dans le portage MQL5, pas dans les données. Piège à vérifier en
      premier : `CopyClose(symbol, PERIOD_D1, 1, N, arr)` remplit `arr` du plus ancien au
      plus récent — `FxSleeveGoldMomentum.mqh::ComputeScore` en dépend.
- [ ] **Barreau 3 — sizing.** MT5 convertit un montant de risque en lots via
      `LotsForRisk(symbol, risk_money, sl_distance)`, ce qui n'est **pas** un poids de
      portefeuille. Écrire la correspondance explicite entre `target_weight` (vbt/QC) et
      `risk_money / sl_distance` (MT5), et vérifier que `SYMBOL_VOLUME_STEP` de l'or
      n'écrase pas la granularité — l'arrondi de lots sur un seul symbole est bien plus
      grossier que sur un panier de 4 paires.
- [ ] **Barreau 4 — exécution.** Quantifier séparément : spread bid-ask, commission
      (`Inp_CommissionBpsPerSide`), swap (`Inp_SwapBpsPerNight` × ~35 nuits de détention
      médiane, cf. `FX_GOLD_AVG_NIGHTS_HELD`), et le stop de sécurité à 4%.
- [ ] Produire `reports/investigations/vbt_vs_mt5_gold_parity.md` avec un **tableau
      d'attribution** : chaque poste de coût, sa contribution en points de CAGR, et le
      résidu inexpliqué. Modèle : §6 de `docs/quantconnect_validation_report.md`, qui
      attribue proprement les 50% de Sharpe perdus sur MR Macro au spread Oanda.

**Acceptation** : tolérances du repo (`compare_vbt_vs_mt5_c1.py`) — Sharpe ±0.10,
CAGR ±2 pp, maxDD ±2 pp, trades ±10% — **ou bien** un résidu hors tolérance nommément
attribué et chiffré. Un résidu « inexpliqué » supérieur à 20% de l'écart total est un échec
de la phase, pas un résultat.

---

## Phase 5 — Verrouiller

- [ ] `scripts/compare_vbt_vs_mt5_gold.py` sur le patron de `compare_vbt_vs_mt5_c1.py`.
- [ ] `tests/test_gold_momentum.py` : snapshot de `pipeline().stats()` à `rtol=1e-10`,
      contrat de `tests/test_pipeline_equivalence.py`. Générer via
      `tests/_generate_snapshots.py`.
- [ ] Figer les tolérances trois-voies dans la spec de la phase 0 et les faire vérifier
      par le script de réconciliation, en sortie non nulle si dépassement.
- [ ] Ajouter `Gold_Momentum` à `_compute_strategy_daily_returns()`
      (`combined_portfolio.py:88-156`) **et bumper `_SLEEVES_VERSION`**
      (`src/framework/data_cache.py:66`) — sans ce bump le cache sert silencieusement du
      périmé. Relancer `scripts/sweep_fourth_sleeve.py`.
      Poids conseillé **10-15%** : l'optimum d'échantillon est à la borne (30%), ce qui
      signale un biais de période et non un optimum.

---

## Dette préexistante à traiter au passage

- [x] **`utils.apply_vbt_settings()` casse les tests de `test_pipeline_equivalence.py`.**
      Fait (`c3f1809`). La clé est choisie selon la version installée. Deux effets de bord
      que le plan n'anticipait pas : le `KeyError` se déclenchait *avant* la ligne
      `year_freq = 252 jours`, qui n'était donc jamais appliquée (métriques annualisées
      lues sur le défaut vbt de 365 jours) ; et il masquait 9 snapshots périmés,
      réétalonnés dans `23c02e3`. 17/17 verts.

- [ ] **Épingler l'environnement — nouveau prérequis, découvert en exécutant ce plan.**
      Le venv a dérivé de `uv.lock` sur **18 des 35 paquets** : pandas 2.3.3 → 3.0.5
      (montée majeure), pyarrow 23.0.1 → 25.0.0, numpy, numba. `vectorbtpro` n'est pas
      couvert par le lock du tout. C'est ce qui a périmé les snapshots, et c'est
      rédhibitoire ici : **on ne peut pas réconcilier trois moteurs si l'un d'eux ne
      redonne pas le même résultat d'un mois sur l'autre.** À traiter avant la phase 2,
      et avant toute republication de chiffres.
      Attention : `uv sync` ramènerait pandas 2.3.3 et invaliderait les snapshots qu'on
      vient de réétalonner — trancher explicitement le sens de la convergence (regénérer
      le lock sur l'environnement courant, ou revenir au lock) plutôt que de lancer la
      commande.
- [ ] `assert_manifest_fresh()` (`scripts/update_data_manifest.py:175`) n'a aucun appelant
      alors que sa docstring affirme le contraire. Le brancher au démarrage des scripts de
      sweep.
- [ ] `output/` n'est ni suivi ni ignoré par git.
- [ ] `load_fx_data` est dupliqué dans `notebooks/nb_utils.py:71`.

---

## Ordre d'exécution et parallélisme

```
Dette : pre_show_func [FAIT] ──> épinglage de l'environnement [À FAIRE]
   │
Phase 0 (spec + harnais)
   ├─> Phase 1 (fix sizing) [SANS OBJET] ──> Phase 2 (vbt ↔ QC)  ─┐
   └─> Phase 3 (débloquer MT5)            ──> Phase 4 (vbt ↔ MT5) ─┴─> Phase 5 (verrouiller)
```

Les phases 2 et 3 sont **indépendantes** et peuvent avancer en parallèle : la première est
du Python, la seconde du diagnostic d'environnement. La phase 4 dépend des deux.

L'épinglage de l'environnement conditionne toute revendication de reproductibilité, donc
les phases 2, 4 et 5.

## Ce qui ferait échouer ce plan

- **Le broker n'offre pas l'or.** Toute la branche MT5 tombe. À vérifier en premier de la
  phase 3, avant d'investir dans le débogage du tester.
- **Chercher l'égalité avec MT5.** Le flux broker, les bornes de barres et le spread réel
  diffèrent par construction. Un écart nul signifierait qu'on a idéalisé le backtest
  broker — le contraire du but recherché.
- **Réconcilier sur les métriques finales.** Sans les traces journalières de la phase 0,
  chaque écart devient une conjecture. La phase 0 n'est pas optionnelle.
- **Réconcilier sur un environnement non épinglé.** Ajouté après coup, et c'est le piège
  qui s'est effectivement refermé : un moteur qui ne redonne pas le même résultat d'un mois
  sur l'autre n'est pas réconciliable. Il l'a d'abord fait silencieusement, les tests qui
  auraient dû le signaler étant hors service.
- **Bâtir sur une mesure qu'on n'a pas refaite soi-même.** La phase 1 de ce plan visait un
  défaut qui n'existait pas, déduit d'une colonne de tableau mesurée sans vol-targeting.
  Trente secondes de re-mesure l'auraient évité. Avant de corriger un écart, reproduire
  l'écart.
