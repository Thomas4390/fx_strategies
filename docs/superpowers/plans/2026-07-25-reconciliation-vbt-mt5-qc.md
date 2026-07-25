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

| | vbt | QC | MT5 |
|---|---|---|---|
| Sharpe | 0.726 | 0.575 – 0.797 | non mesuré |
| Vol ann. | **12.19%** | **23.3%** | non mesuré |
| maxDD | −23.31% | −51.9% | non mesuré |
| trades | 50 | 128 ordres | — |

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

- [ ] Écrire `docs/specs/gold_momentum_spec.md` : **la** source de vérité, en pseudo-code
      neutre, indépendante de tout moteur. Doit fixer sans ambiguïté :
      - la borne de journée (quel instant clôt une barre D1) et le fuseau de référence
      - l'indexation exacte du lookback (`close[t-1] / close[t-1-N]`, jamais `close[t]`)
      - la fenêtre et le ddof de σ21, le facteur d'annualisation (252)
      - la formule du poids cible et son plafond
      - l'instant de décision et l'instant de fill (et leur écart)
      - le modèle de coût, en bps de notionnel par côté
- [ ] Ajouter à `src/strategies/gold_momentum.py` un `emit_daily_trace(pf, path)` qui
      écrit le CSV à 6 colonnes ci-dessus.
- [ ] Ajouter la même trace au projet QC `34489845` via `self.object_store` ou un
      `self.log()` structuré récupérable par `read_backtest_logs`.
- [ ] Ajouter la même trace au MQL5 : `CSleeveGoldMomentum` écrit une ligne par jour dans
      `Common/Files/gold_trace.csv`, gardée par un `Inp_Gold_Trace` à `false` par défaut.
- [ ] Écrire `scripts/reconcile_three_way.py` — charge les trois traces, les aligne sur
      la date, et sort un tableau par barreau avec le **premier jour de divergence** et
      l'amplitude. Reprendre le style de sortie de `scripts/compare_vbt_vs_mt5_c1.py:167`.

**Acceptation** : le script tourne sur des traces partielles et nomme correctement le
premier barreau cassé.

---

## Phase 1 — Corriger le défaut de sizing vbt (bloquant)

Défaut identifié par la validation QC : `gold_momentum.pipeline()` passe
`size_type="percent"` avec un tableau `leverage`. Or `percent` désigne un pourcentage du
**cash disponible** — VBT plafonne l'ordre et le levier ne peut pas le relever.
Exposition brute moyenne mesurée **52.7%**, volatilité réalisée **12.19%** pour une cible
de 25% : la couche de vol-targeting ne délivre que la moitié de ce qu'elle demande.

- [ ] Remplacer par `size_type="targetpercent"` avec le poids vol-cible comme `size`.
      Patron exact : `daily_momentum.py:224-233` et `composite_fx_alpha.py:389-398`.
- [ ] Vérifier que le seam `signal_func_nb` de `framework/sizing_nb.py` reste fonctionnel :
      le kernel écrit `SizeType.ValuePercent`, cohérent avec `targetpercent`. Relancer
      `pytest tests/test_sizing_nb.py -v` — les 13 tests doivent rester verts.
- [ ] Contrôle de bon sens : la volatilité réalisée doit atterrir à **25% ± 3 pp** et
      l'exposition brute moyenne doit être cohérente avec le levier médian (~2.0×).
- [ ] Relancer `scripts/sweep_gold_sizing.py` (sélection **et** holdout) et **réécrire les
      sections 4 et 6** de `docs/research/gold_2026-07-25_momentum_sizing.md`.

**Attention méthodologique** : le classement des régimes de sizing devrait survivre (le
biais était commun à tous, et la comparaison à risque égal normalise la volatilité). S'il
change, c'est une information de premier ordre — le documenter explicitement plutôt que de
le lisser.

- [ ] Recalculer les poids portefeuille de la section 6 : ils dérivaient de rendements
      sous-estimés.

**Acceptation** : vol réalisée dans 25% ± 3 pp ; 13 tests verts ; sections 4 et 6 à jour.

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

- [ ] **Le symbole n'existe pas chez le broker.** Aucun historique or dans
      `Bases/SquaredFinancialSC-MT5 Demo/history/` (que du FX). Lancer
      `src/mt5/Scripts/FxPreflight.mq5` ou un script listant `SymbolsTotal(false)` pour
      obtenir le nom exact. Candidats : `XAUUSD`, `XAUUSD.c`, `GOLD`, `GOLD.c`.
      **Si le broker n'offre pas l'or, tout le reste de cette phase est sans objet** — le
      dire immédiatement plutôt que de continuer à déboguer.
- [ ] **État de session sauvegardé qui prime sur `/config:`.** MT5 restaure le dernier
      profil. Tester avec `Config/` vidé de ses `.ini` persistants, ou avec un
      `[StartUp]` explicite.
- [ ] **Build 5836 et `/config:`.** Vérifier qu'un backtest FX connu (celui de mai, qui
      fonctionnait : `reports/mt5/run_20260506T174130Z.json`) repasse aujourd'hui à
      l'identique. **Si le run FX échoue aussi, la régression n'a rien à voir avec l'or**
      et c'est l'environnement Wine/MT5 qu'il faut traiter.
- [ ] Vérifier que le chemin `/config:` reste sans espace (`C:\fxgold.ini`) et que l'INI
      est bien **UTF-16 LE avec BOM et CRLF** — `run_backtest_cli.py:95-105`.

**Acceptation** : un `run_*.json` frais dans `reports/mt5/` avec `exit_code=0` et une
ligne `[OPTIM]` exploitable.

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

- [ ] **`utils.apply_vbt_settings()` casse les 16 tests de `test_pipeline_equivalence.py`.**
      Il écrit `plotting.pre_show_func`, clé renommée `pre_render_func` dans vbt 2026.6.27 ;
      la config étant gelée, cela lève un `KeyError` au setup de la fixture. Vérifié
      préexistant. **À corriger en premier** : sans ces tests, aucune non-régression n'est
      possible sur les trois sleeves FX.
- [ ] `assert_manifest_fresh()` (`scripts/update_data_manifest.py:175`) n'a aucun appelant
      alors que sa docstring affirme le contraire. Le brancher au démarrage des scripts de
      sweep.
- [ ] `output/` n'est ni suivi ni ignoré par git.
- [ ] `load_fx_data` est dupliqué dans `notebooks/nb_utils.py:71`.

---

## Ordre d'exécution et parallélisme

```
Phase 0 (spec + harnais)
   ├─> Phase 1 (fix sizing vbt)  ──> Phase 2 (vbt ↔ QC)  ─┐
   └─> Phase 3 (débloquer MT5)   ──> Phase 4 (vbt ↔ MT5) ─┴─> Phase 5 (verrouiller)
```

Les phases 1-2 et 3 sont **indépendantes** et peuvent avancer en parallèle : la première
est du Python, la seconde du diagnostic d'environnement. La phase 4 dépend des deux.

La dette préexistante (`pre_show_func`) est à traiter avant toute revendication de
non-régression.

## Ce qui ferait échouer ce plan

- **Le broker n'offre pas l'or.** Toute la branche MT5 tombe. À vérifier en premier de la
  phase 3, avant d'investir dans le débogage du tester.
- **Chercher l'égalité avec MT5.** Le flux broker, les bornes de barres et le spread réel
  diffèrent par construction. Un écart nul signifierait qu'on a idéalisé le backtest
  broker — le contraire du but recherché.
- **Réconcilier sur les métriques finales.** Sans les traces journalières de la phase 0,
  chaque écart devient une conjecture. La phase 0 n'est pas optionnelle.
