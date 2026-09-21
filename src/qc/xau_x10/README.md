# QuantConnect — XAUUSD niveaux x10 (stratégie 2)

Miroir local versionné du projet cloud `ApogeeInvest_S2_XAUUSD_X10`. **Ce dossier
fait foi** : le projet cloud en est une copie, jamais l'inverse. Toute divergence
avec `docs/specs/xau_x10_spec.md` est un défaut de ce port, pas de la spec.

Troisième moteur, après la référence vectorbt (`src/strategies/xau_x10.py`) et
l'EA MQL5 (`src/mt5/Experts/XauX10.mq5`). QC et la référence partagent les
**mêmes barres** — le parquet de l'or a été exporté de ce compte — donc §13 fixe
la cible d'appariement des entrées à **≥ 98 %**.

## Fichiers

| fichier | contenu | importe QC ? |
|---|---|---|
| `main.py` | abonnements, boucle `on_data`, ordres, trace | **oui**, seul |
| `x10_bars.py` | agrégation M1 → M5/H1, horloge New York | non |
| `x10_indicators.py` | ATR Wilder, EMA, cinématique, VWAP, DXY4 | non |
| `x10_state.py` | automate §7, dimensionnement §11, trace §13 | non |

La séparation n'est pas cosmétique : les trois modules purs (Python standard,
ni pandas ni numpy) sont rejoués hors QC par `tests/test_qc_x10_parity.py`, qui
les compare au noyau de référence `framework.x10_engine` sur des minutes
synthétiques. C'est le seul filet disponible localement — le cloud n'est pas
accessible depuis le dépôt.

```bash
uv run pytest tests/test_qc_x10_parity.py -q
uv run ruff check src/qc/xau_x10 tests/test_qc_x10_parity.py
```

## Pousser et compiler (MCP QuantConnect)

Le serveur MCP `quantconnect` fait tout ; il n'y a **pas** de CLI `lean` dans ce
dépôt et aucun appel réseau n'est fait depuis les tests.

1. `read_project` / `create_project` pour obtenir le `projectId`
   (`ApogeeInvest_S2_XAUUSD_X10` ; le `cloud-id` est dans le fichier de config
   du projet local une fois créé).
2. Pousser les **quatre** `.py` avec `update_file_contents`, un appel par
   fichier, en conservant les noms : les imports sont plats (`from x10_bars
   import …`) parce que QC met tous les fichiers d'un projet dans le même
   dossier. Renommer un fichier casse les imports.
3. `create_compile` puis `read_compile` **avant** tout backtest, et corriger les
   erreurs avec `patch_file` plutôt que de réécrire un fichier entier.
4. `create_backtest` / `read_backtest`. Les ordres sont récupérables par API
   (`read_backtest_orders`) : leur `tag` est le **canal principal** de
   réconciliation. Format ci-dessous.

### Format des tags d'ordre

Les **six premiers champs sont gelés**, dans cet ordre, et restent lisibles
positionnellement par `scripts/reconcile_x10_events.py` (non modifié) :

```
scenario|level|ts_decision|stop|target|r_est
```

Tout ce qui suit est en `k=v`, ajouté **après** le préfixe gelé, tronqué à 200
caractères en conservant l'ordre de priorité ci-dessous.

| ordre | champ | entrée | sortie | pourquoi |
|---|---|---|---|---|
| 1 | `bm=` | ✓ | | début de la barre de décision, minutes epoch UTC |
| 2 | `c=` | ✓ | | clôture mid de la barre de décision, 4 décimales |
| 3 | `atr=` | ✓ | | ATR M5 de la décision — le barreau 2 par l'API |
| 4 | `sp=` | ✓ | | spread plein utilisé dans `R` |
| 5 | `v=` `a=` `vw=` `ema=` | ✓ | | cinématique et contexte |
| — | `xr=` `xp=` | | ✓ | raison et prix théorique de sortie |

`atr`, `c` et `sp` existent pour une raison précise : le barreau 2 de §13 n'est
pas observable de l'extérieur (la trace vit dans l'ObjectStore, non exportable),
et l'écart D8 n'a pas pu être départagé faute de ces trois nombres. `xr=` ferme
la lacune de §8.2, où la raison de sortie QC devait être **inférée**.
5. La trace §13 n'est **pas** récupérable par API (ObjectStore réservé aux
   comptes Institutional, pas d'endpoint MCP pour les logs de backtest) : la
   télécharger depuis l'Object Store par l'interface web, ou copier les lignes
   `TRACE,` du journal en retirant le préfixe.

## Paramètres

| paramètre | défaut | §|
|---|---|---|
| `z` | `1.0` | A.2, largeur de zone en ATR |
| `a_min` | `0.2` | A.2, seuil d'accélération (armement **et** flip) |
| `k_s` | `1.0` | A.2, stop du breakout en ATR |
| `start` | `2019-01-01` | §1 |
| `end` | `2025-12-31` | §1 |
| `allow_frozen_oos` | absent | garde holdout |
| `spread_override` | absent | §8, voir ci-dessous |
| `diag` | `0` | diagnostic barreau 2 |

`spread_override` n'est **pas** un axe de grille. Sans lui, `R` (§8) est calculé
sur la vraie fourchette bid/ask de la barre de décision ; avec `spread_override=0.29`
il utilise le spread constant de la référence. Les deux runs répondent à deux
questions différentes : « exécution réelle » et « signaux comparables ». La
mesure de `docs/research/xau_x10_reconciliation.md` §2.2 impose les deux — le
demi-spread OANDA vaut **1,67 fois** celui de la référence.

`diag=1` publie deux séries de graphique (`X10Diag/m5_close`, `X10Diag/atr`) à
chaque barre M5 des trois premiers jours, plafonnées à 900 points. Les séries de
graphiques **sont** récupérables par API (`read_backtest_chart`), contrairement à
l'ObjectStore et aux logs : c'est le seul canal qui permette de comparer les
barres M5 elles-mêmes sans la trace. Aucun ordre de diagnostic n'est émis.

Capital initial 10 000 USD, comme MT5 et la référence. Un `end ≥ 2026-01-01`
lève une exception tant que `allow_frozen_oos=1` n'est pas posé : la tranche
gelée se lit une fois, après la campagne in-sample
(`docs/research/HOLDOUT_POLICY.md`).

Ce qu'annexe A.1 gèle (`v_min`, `k_b`, `N_arm`, `N_hold`, `N_sweep`, `s_min`,
`R_min`, 48 barres, cooldown 6, risque 0,5 %) est une constante de module dans
`x10_state.py`, pas un paramètre : un paramètre gelé qu'un appelant peut
surcharger est un paramètre gelé qui attend d'être optimisé.

## Écarts structurels assumés QC ↔ référence Python

Chacun est connu, mesurable, et à citer dans le rapport de réconciliation — pas
une surprise à découvrir au barreau qui casse.

1. **Warm-up.** `set_warm_up(12 jours)` consomme de l'historique **avant** la
   date de départ, alors que la référence démarre à froid au 2019-01-01 : sur
   les ~60 premières barres H1, QC a une EMA50 et un ATR_H1 définis là où la
   référence a des NaN, donc QC peut prendre des breakouts que la référence
   refuse (§6.1 : `ctx_ema = 0` interdit le breakout pendant l'amorce).
2. **Prix de fill.** La trace porte le prix **théorique** de la spec
   (`mid ± s/2` à l'open de la barre M5 de fill) ; LEAN, lui, remplit l'ordre
   market au bid/ask de la minute suivant son envoi, l'ordre partant à la
   clôture de la minute qui ferme la barre de décision. Les deux nombres sont
   conservés : le théorique dans la trace, le réalisé dans le journal d'ordres.
3. **Sorties stop/cible.** Même écart, amplifié : la spec sort *au niveau* du
   stop ou de la cible, LEAN remplit un ordre market au prix courant. Le niveau
   théorique reste dans la trace (`exit_px`) et dans le tag de l'ordre.
4. **Fin de barre contre début de barre.** L'horodatage QC d'une barre minute
   est sa **fin** ; la référence indexe par **début**. `main.py` retranche une
   minute avant tout binnage ; un oubli décalerait toutes les barres d'un cran.
5. **Spread réel contre spread constant — terme de premier ordre, mesuré.**
   La référence facture 0,29 $ constant (demi-spread 0,145 $) ; le flux OANDA
   donne un demi-spread implicite de **0,242 $ (P1) à 0,260 $ (P2)**, soit
   **×1,67 à ×1,79** (`docs/research/xau_x10_reconciliation.md` §2.2). Avec
   `R_MIN = 1`, un demi-spread qui double ne déplace pas les candidats
   marginaux : il déplace **tout le flux de candidats**. Ce n'est pas un effet
   de frontière et ce n'est pas un défaut du portage — c'est la référence qui
   idéalise le coût, et la sensibilité ×2 déjà exigée par §12 est le cas
   central, pas une précaution. Utiliser `spread_override=0.29` pour obtenir un
   run à signaux comparables.
6. **Équité de dimensionnement.** §11 dimensionne sur
   `portfolio.total_portfolio_value` côté QC ; la référence compose sa propre
   équité à partir des P&L qu'elle a calculés. Les deux divergent dès que les
   prix de fill divergent (points 2 et 3).
7. **Fenêtre interdite sur la barre suivante.** La référence teste la
   `minute_of_day` de la **barre M5 suivante existante** ; le port teste celle
   de `début + 5 min`. La règle §2 rend les deux équivalents — si la barre
   suivante n'est pas à +5 min, le candidat est annulé en `window` de toute
   façon — mais le port prend cette décision **au fill** et la référence **à la
   décision**. Même ligne de trace, même raison, même ordre ; c'est vérifié par
   `test_event_sequence_is_identical`.
8. **Pas de barre « dernière ».** La référence solde toute position ouverte sur
   `i == n5-1` et n'y prend aucune décision. Un algorithme en flux n'a pas cette
   notion : les événements de la toute dernière barre sont hors comparaison des
   deux côtés plutôt que simulés d'un seul.
9. **DXY4 sur l'heure d'union.** Le panier §6.3 est construit sur l'union des
   heures où **au moins une** jambe a produit une barre H1. Le port publie
   toujours la valeur de la dernière heure close ; si une heure comporte des
   barres d'or mais **aucune** minute sur les quatre majeures, la référence
   servirait l'heure d'avant. Cas non observé sur des majeures OANDA, non
   traité, signalé.
10. **Pas de modèle de slippage, frais nuls.** `ConstantFeeModel(0)` et aucun
    `SlippageModel` : le coût est porté une seule fois par la fourchette réelle.
    Un `ConstantSlippageModel` la facturerait une seconde fois sur chaque
    sortie — le bug #3 du premier portage
    (`docs/quantconnect_validation_report.md` §5.3).
11. **Sharpe natif QC inutilisable.** Il n'est pas calculé à `rf = 0` ; les
    métriques publiées viennent de la chaîne locale, jamais de l'en-tête du
    backtest.
12. **Clôture de séance émise une minute plus tôt.** §10 ferme à la barre M5 de
    [16:55, 18:00[ au prix de clôture de sa dernière M1. Le CFD XAUUSD cesse de
    coter **après la minute commençant à 16:58** (mesuré : sur 261 séances 2024
    du parquet de référence, 16:59 et 17:00 n'existent jamais, et le flux
    reprend à 18:04). L'ordre part donc à la fin de cette minute, soit **16:59
    heure de New York**, marché encore ouvert. La trace conserve
    `ts_decision` = barre 16:55 et le prix théorique de la spec ; seule la
    soumission est avancée. La barre M5 elle-même est inchangée — celle de la
    référence ne contient pas non plus de minute 16:59.

> **Correction post-mortem 2024.** Les écarts 2, 3 et 6 ci-dessus décrivaient
> des coûts ; le backtest `x10_calage_2024_centre` a révélé un **défaut**, pas
> un écart : sans `on_order_event`, huit sorties refusées avaient laissé un
> inventaire fantôme valant **81,5 % de la perte du compte**
> (`docs/research/xau_x10_reconciliation.md` §4). Corrigé ici par
> `x10_state.BrokerSync` (intention contre réalité, une seule place où un ordre
> est émis), la garde `exchange_open`, le flush de séance de l'écart 12, la
> liquidation finale et un invariant quotidien logué en `self.error`.

## Lecture de la réconciliation

L'échelle de §13 se descend jusqu'au premier barreau qui casse ; au-delà, rien
n'est interprétable.

| barreau | objet | où le port peut le casser |
|---|---|---|
| 1 | barres M5 | fin/début de barre (écart 4), bins vides |
| 2 | `atr,v,m,a,vwap,ema50_h1,dxy` | warm-up (écart 1), amorçage ATR, `ignore_na` de l'EMA DXY, `fill_forward` |
| 3 | `ARM/BREAK/SWEEP/CANCEL` | seuils, cooldown, ordre M5/H1 à l'heure pile |
| 4 | `ENTRY` | `R` et fenêtre (écarts 5 et 7) |
| 5 | `EXIT`, PnL | fills LEAN (écarts 2, 3), lots (écarts 5, 6) |

Les barreaux 1 à 5 sont verts localement, sur données synthétiques, contre le
noyau de référence : `tests/test_qc_x10_parity.py`. Ce que ce test **ne** couvre
pas, c'est tout ce qui appartient à LEAN — les écarts 1, 2, 3, 5 et 6 ci-dessus.

## Ce que le portage sait du flux, et d'où ça vient

Trois constantes du portage sont mesurées sur `data/XAU-USD_minute_qc.parquet`,
l'export que lit la référence, et non devinées :

| fait | mesure (2024, 359 232 barres) |
|---|---|
| dernière minute cotée avant la coupure | **16:58** New York, 261/261 séances |
| première minute cotée après | **18:04** New York, 262/262 séances |
| minutes manquantes **dans** une séance | **0** — la grille est contiguë |
| barres à amplitude nulle | 311 (0,09 %) — l'export n'est pas forward-fillé |
| jambes FX (`EUR-USD_minute.parquet`) | dernière 16:58, **reprise 17:04** |

La dernière ligne est la cause exacte des huit ordres refusés de 2024 : les
jambes FX reprennent une heure avant l'or, réveillent le feed à 17:04 et
déclenchent la clôture d'une barre M5 restée ouverte, sur un marché XAUUSD
fermé depuis 17:00. Deux gardes indépendantes le couvrent désormais — le flush
de séance (écart 12) et le test `exchange_open` dans `_sync`.
