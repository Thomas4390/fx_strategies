# XAUUSD niveaux x10 : faisabilité avant construction

> **Date** : 2026-09-21 · **Statut** : ouvert — **construire, verdict négatif publiable**
> **Holdout state** : LOCKED.
> **Holdout touched by this phase** : descriptif uniquement (ATR M5, fréquence de croisement
> des x10) ; **0 lecture de performance**.
> **Essais consommés** : 0 — aucun signal, aucun P&L, aucun backtest n'est calculé ici.
> Un run de calage au centre de la grille a été exécuté le 2026-09-21 pour **compter des
> événements** (§2) ; aucune performance lue, aucun essai logué.

## 0. Pourquoi cette note existe avant le code

Le client Apogée Invest commande une deuxième stratégie : XAUUSD en M5 sur les niveaux
psychologiques de 10 $ (`docs/specs/xau_x10_spec.md` §0). Le chantier est long — trois moteurs,
un EA neuf, une réconciliation événementielle, un rapport.

Avant de l'engager, une question descriptive se pose et se répond **sans toucher à la
performance** : *la grille de 10 $ représente-t-elle la même chose en 2019 et en 2026 ?*
La réponse est non, et l'écart n'est pas marginal. Cette note mesure l'écart, en tire
l'arithmétique des coûts, rappelle trois précédents du dépôt qui pointent dans la même
direction, et conclut malgré tout à la construction — parce que le mandat l'impose et qu'un
verdict négatif est un livrable.

## 1. La grille de 10 $ n'est pas stationnaire en ATR M5

Recalculé le 2026-09-21 sur `data/XAU-USD_minute_qc.parquet` (M1 OHLC UTC, 2 703 484 barres,
2019-01-01 → 2026-07-24). Agrégation M5 `label=left, closed=left` ; ATR de Wilder 14 sur M5 ;
un **croisement** = changement de `floor(close/10)` entre deux clôtures M5 consécutives ; un
**jour** = une date UTC portant au moins 100 barres M5. Médianes annuelles.

| année | jours | prix médian | ATR M5 médian | **10 $ / ATR** | spread 0,29 $ / ATR | croisements / jour (méd.) | (moy.) |
|---|---|---|---|---|---|---|---|
| 2019 | 260 | 1 406,35 | 0,579 | **17,3** | 0,501 | 7,0 | 8,7 |
| 2020 | 261 | 1 775,01 | 1,439 | **7,0** | 0,202 | 20,0 | 21,1 |
| 2021 | 260 | 1 793,97 | 1,037 | **9,6** | 0,280 | 13,0 | 14,4 |
| 2022 | 259 | 1 805,37 | 1,148 | **8,7** | 0,253 | 15,0 | 16,5 |
| 2023 | 259 | 1 945,65 | 1,001 | **10,0** | 0,290 | 13,0 | 14,3 |
| 2024 | 261 | 2 380,75 | 1,477 | **6,8** | 0,196 | 21,0 | 20,9 |
| 2025 | 260 | 3 345,08 | 2,637 | **3,8** | 0,110 | 36,0 | 39,8 |
| 2026 (au 23/07) | 146 | 4 618,46 | 5,500 | **1,8** | 0,053 | 74,0 | 80,7 |

Ces chiffres reproduisent à l'identique les ordres de grandeur annoncés au plan de phase
(17,3 en 2019 ; ≈ 10 en 2021-2023 ; 3,8 en 2025 ; 1,8 en 2026). **Aucun écart supérieur à
15 %** sur les années citées. Seul 2020 (7,0) s'écarte de la lecture rapide « ≈ 10 avant
2024 » — l'année COVID a une volatilité d'or à part, et le plan ne la citait pas.

Ce que le tableau dit, en clair :

- **Le même niveau ne désigne pas le même objet selon l'année.** En 2019, franchir un x10
  demandait 17 écarts-types de barre ; c'était un événement. En 2026 il en faut moins de 2 :
  le prix traverse la grille **74 fois par jour**, soit dix fois plus qu'en 2019. Un
  « niveau psychologique » traversé toutes les six minutes n'est plus un niveau.
- **Le signal et la cible changent de nature ensemble.** La cible est le prochain x10 (spec
  §7). Elle valait 17,3 ATR en 2019 et vaut 1,8 ATR en 2026 : la même règle vise une sortie
  hors de portée en début d'échantillon et une sortie dans le bruit en fin.
- **Le holdout 2026 vit dans un régime absent de l'in-sample.** Le minimum in-sample est 3,8
  (2025) ; 2026 est à 1,8, soit deux fois plus fin que tout ce que la sélection aura vu. C'est
  la raison pour laquelle le verdict **NON CONCLUANT** est pré-écrit dans la table de décision.

## 2. Arithmétique des coûts

Spread de référence `XAUUSD.c` : **0,29 $** (`spread_current = 290` points, `point = 0,001`,
`data/broker/symbols_catalog_2026-07-28.csv`). Il se paie une fois par aller-retour.

Deux lectures, opposées, et il faut tenir les deux.

**(a) Rapporté à la cible, le coût est constant** : 0,29 $ sur un objectif de 10 $, soit
**2,9 % du gain brut**, toutes années confondues. Ce chiffre ne dit rien du régime.

**(b) Rapporté au risque, le coût s'effondre avec le temps.** Le stop du breakout vaut
`k_s · ATR` (spec §7.2). Le coût exprimé en unités de risque vaut donc `0,29 / (k_s · ATR)` :

| année | ATR M5 | coût en R (`k_s = 0,75`) | (`k_s = 1`) | (`k_s = 1,5`) | R disponible (`k_s = 1`) | R disponible (`k_s = 1,5`) |
|---|---|---|---|---|---|---|
| 2019 | 0,579 | 0,668 | **0,501** | 0,334 | 13,6 | 9,7 |
| 2021 | 1,037 | 0,373 | **0,280** | 0,186 | 8,3 | 5,8 |
| 2023 | 1,001 | 0,386 | **0,290** | 0,193 | 8,6 | 6,0 |
| 2025 | 2,637 | 0,147 | **0,110** | 0,073 | 3,5 | 2,4 |
| 2026 | 5,500 | 0,070 | **0,053** | 0,035 | **1,75** | **1,17** |

`R disponible = (10 − s/2) / (k_s·ATR + s/2)` avec `s = 0,29 $`, la formule de la spec §8 avec
un niveau franchi exactement. Le coût en R, lui, est l'aller-retour complet rapporté au risque,
`0,29 / (k_s·ATR)`, indépendant de la formule de `R`.

Trois conséquences opérationnelles :

1. **En 2019, chaque aller-retour part avec un demi-R de retard** (0,501 R à `k_s = 1`, deux
   tiers de R à `k_s = 0,75`). Pour une stratégie à 1:1 nominal, c'est un handicap que peu de
   taux de réussite compensent.
2. ~~**Le garde-fou `R ≥ 1` ne mord presque jamais.**~~ — **hypothèse d'entrée au niveau,
   RÉFUTÉE par le comptage du 2026-09-21** (voir juste après). Raisonnement conservé pour
   mémoire : le R disponible ne descend sous 2 qu'en 2026 et ne passe sous 1 dans aucune année,
   même au stop le plus large (`k_s = 1,5`, 2026 : **1,17**, minimum de l'échantillon) ; on en
   concluait que le filtre du mandat était inopérant. **Cette conclusion est fausse.**
3. **Le régime favorable aux coûts est précisément le régime où la cible est du bruit.** 2026
   offre le coût relatif le plus faible (0,053 R) et la cible la plus dérisoire (1,8 ATR). Les
   deux effets tirent en sens contraire et ne se compensent pas *a priori* : c'est exactement
   ce que la campagne in-sample doit décomposer (terciles de `10 $/ATR`, spec §13 barreau 3).

### 2.1 Correction du 2026-09-21 — la règle R mord, et de plus en plus

Le moteur Python étant écrit, la question se compte au lieu de se raisonner. Comptage
**non financier** (nombre d'événements, aucune performance lue) d'un run de calage au **centre
de la grille**, candidats rejetés par la règle R sur candidats ayant atteint le test R :

| année | rejetés / candidats | taux de rejet |
|---|---|---|
| 2019 | 11 / 168 | 6,5 % |
| 2020 | 90 / 340 | 26,5 % |
| 2021 | 39 / 276 | 14,1 % |
| 2022 | 49 / 299 | 16,4 % |
| 2023 | 55 / 276 | 19,9 % |
| 2024 | 104 / 417 | 24,9 % |
| 2025 | **413 / 744** | **55,5 %** |

**Pourquoi le calcul du §2 se trompait** : il supposait une entrée **au niveau**. L'entrée
réelle n'a jamais lieu au niveau. Pour un breakout, elle suit le maintien — le prix a déjà
parcouru une part de la cible, qui se raccourcit d'autant. Pour un reversal, elle suit une
excursion, dont la profondeur éloigne le stop. Dans les deux cas le `R` effectif est
structurellement **inférieur** au `R disponible` tabulé plus haut, et l'écart grandit avec
l'ATR — d'où le décrochage de 2025.

Conséquence pour le rapport client : `R ≥ 1` **est** un filtre actif, il rejette plus d'un
candidat sur deux dans le régime récent, et le taux de rejet annuel doit être publié (spec §8).
L'ancien raisonnement reste ci-dessus, marqué réfuté, parce qu'il documente une erreur de
méthode utile : raisonner sur une entrée idéalisée avant d'avoir un moteur.

Nombre d'opportunités brutes : de 7 croisements par jour (2019) à 74 (2026), soit de l'ordre
de 1 800 à 19 000 franchissements par an. Même un taux de conversion de 2 % place la stratégie
à plusieurs centaines de trades par an. Ce chiffre est le pivot du §3.

## 3. Trois précédents du dépôt, tous défavorables

### 3.1 Le Sharpe MT5 décroît avec la fréquence — le résultat le plus robuste du dossier

`reports/mt5/gold_sweep.csv`, cité par `docs/research/xs_longshort_feasibility_2026H2.md` §3 :
même instrument, même moteur, seule la grille de lookbacks change.

| lookbacks | trades/an | Sharpe MT5 |
|---|---|---|
| 40/60/120/250 | 5,8 | **0,873** |
| 20/40/80 | 7,7 | 0,800 |
| 30/60/120 | 7,5 | 0,658 |
| 15/30/60 | 12,4 | 0,565 |
| 10/20/40 | 14,3 | **0,433** |

**Multiplier la fréquence par 2,5 coûte 0,44 de Sharpe MT5.** La loi est monotone sur cinq
points. Le candidat x10 tourne en M5 avec un objectif de 300 trades in-sample minimum
(table de décision), soit **plusieurs dizaines de fois** la fréquence de la configuration de
production. L'extrapolation n'est pas linéaire et ne doit pas être présentée comme une
prévision — mais aucun point du dépôt ne suggère que le haircut d'exécution s'amenuise quand
on accélère.

### 3.2 La sleeve H1 Momentum : le précédent direct d'un signal intraday sans filtre

`reports/optimization/h1_momentum/findings.md` : sleeve construite, intégrée, testée en
standalone et abandonnée. **Sharpe −3,98** (paramètres du plan, 301 trades) et **−3,12** en
variante plus lente (173 trades). Diagnostic consigné : durées de trade de 1 à 7 heures,
whipsaw constant, « spread + slippage = ~24 bps round-trip, mange l'edge », et surtout
« pas de filtre macro / session restriction → trade 24/24 ».

Le mandat x10 demande explicitement **M5 toute la journée**. La stratégie 2 reprend donc le
facteur aggravant nommément identifié au précédent négatif ; elle le compense par un filtre de
contexte H1 (EMA50, VWAP, DXY) et une fenêtre de sortie forcée, mais pas par une restriction
de session. C'est un risque assumé du mandat, à rappeler au §13 « Limites » du rapport client.

### 3.3 Le look-ahead H1, déjà payé une fois

`docs/research/eur-usd-bb-mr-research-plan.md:148-197` : `close.resample('1h').last()` produit
une barre `08:00` contenant le close de `08:59` ; le forward-fill la rend visible **59 minutes
trop tôt**. Effet mesuré, après correction par `.shift(1)` :

| config | Sharpe avant | Sharpe après |
|---|---|---|
| 1H BB(4 ; 3,0) | 3,14 | **−2,22** |
| 1H BB(6 ; 4,0) | 3,10 | **−1,48** |
| 1H BB(8 ; 5,0) | 1,32 | **−1,15** |

« **TOUTES** les configurations sont négatives après correction. L'alpha multi-TF était
entièrement un artefact du look-ahead. » La leçon méthodologique y est écrite noir sur blanc :
le fix est `.shift(1)` **avant** le forward-fill, et un Sharpe intraday supérieur à 2-3 doit
déclencher une vérification.

La stratégie x10 est multi-timeframe par construction (contexte H1, décision M5). La spec §6
impose la chaîne `H1 → shift(1) → reindex ffill` et le lot A1 doit livrer un test qui **rougit**
sur une version `.last() + ffill`. Sans ce test, aucun chiffre de cette stratégie n'est
recevable.

## 4. Fenêtre MT5

L'historique or du broker commence au **2022-11-04** : la réconciliation MT5 ne couvre donc
que 2022-11-04 → 2025-12-31, soit **3,2 ans sur les 7 ans d'in-sample**. Les régimes 2019-2022
(10 $/ATR entre 7 et 17) ne sont **pas vérifiables sur le moteur d'exécution**, et c'est
précisément la moitié de l'échantillon où le coût en R est le plus lourd (§2).

Le tester tourne en `--model 1` (OHLC M1), qui interpole les fills : les chiffres MT5 sont un
**majorant**. L'export de barres est plafonné à ~100 000 lignes, d'où le repli par
`Inp_DumpBars` (CSV écrit pendant une passe du tester) pour récupérer le spread réel. Si cet
export échoue, la spec §12 impose le spread constant de 0,29 $ marqué
`source: catalog_snapshot` **plus** une sensibilité ×2 obligatoire.

## 5. Limite de la lecture hors échantillon

Le holdout commence au **2026-01-01** (`docs/research/HOLDOUT_POLICY.md`). Mais le contexte
DXY dépend de quatre parquets FX locaux dont l'index s'arrête tous les quatre au
**2026-04-01 00:00** (vérifié le 2026-09-21 sur `data/{EUR-USD,USD-JPY,GBP-USD,USD-CAD}_minute.parquet`),
alors que l'or local va jusqu'au 2026-07-24. Le panier DXY4 construit sur ces jambes corrèle
**0,87** en rendements journaliers à `DTWEXBGS` au fixing de midi New York (0,67 en fin de jour
calendaire, écart purement horloger, 2 047 jours 2018-2026) : `DTWEXBGS` étant un indice large
à 26 devises, la cible de 0,98 du plan était mal calibrée et est abandonnée — rien n'a été
ajusté pour la remonter (spec §6.3).

Conséquence : la fenêtre OOS **commune aux trois moteurs** est **2026-01-01 → 2026-03-31**,
soit un trimestre. Une extension jusqu'au 2026-07-24 n'est possible que sous la règle
« DXY indéfini = neutre » (spec §6), qui doit être **pré-écrite dans la table de décision avant
la lecture** — sans quoi elle devient un degré de liberté choisi après coup.

Un trimestre de M5 fournit beaucoup de barres mais peu de trades indépendants. Le seuil
`n ≥ 30` de la table de décision n'est pas garanti ; s'il n'est pas atteint, le verdict est
**NON CONCLUANT**, et ce n'est pas un échec de procédure.

## 6. Go / no-go de construction

**Décision : construire.**

Trois raisons, dans cet ordre :

1. **Le client le demande.** Le mandat Apogée Invest fixe l'instrument, la famille de signaux
   et les quatre scénarios. Ce n'est pas un candidat sélectionné par le dépôt sur ses propres
   critères, comme l'était le XS long-short refusé le 2026-07-28 ; l'arbitrage coût/bénéfice
   de construction ne nous appartient pas.
2. **Un verdict négatif est publiable et utile.** La structure du rapport prévoit une section
   14b « Motifs de non-déploiement » pilotée par la macro `\XTenVerdict`. Répondre « cette
   famille de signaux ne survit pas aux coûts sur cet instrument, voici la mesure » est un
   livrable complet.
3. **Le risque réel n'est pas de mesurer un échec, c'est de re-sélectionner après l'avoir
   mesuré.** D'où le dispositif de cette phase : spec gelée avant le code, grille de 27
   configurations et budget de 40 essais gelés avant la première mesure, table de décision et
   verdicts gelés avant la lecture, plateau plutôt que pic, registre d'essais contrôlé par un
   test.

**Ce que cette note n'autorise pas** : introduire une grille adaptative (des niveaux en ATR
plutôt qu'en dollars) après avoir constaté la non-stationnarité du §1. Ce serait une stratégie
différente du mandat, donc une nouvelle spec et un nouveau budget d'essais — pas une variante.

## 7. Ce qui reste à mesurer (et n'est pas mesuré ici)

- Le spread réel par heure New York et par année (lot D) : le 0,29 $ du catalogue est un
  instantané de 2026-07-28, pas une série.
- La fréquence d'un **franchissement exploitable** (armé, confirmé, R ≥ 1), qui n'a rien à
  voir avec les 7 à 74 croisements bruts par jour du §1.
- La corrélation de la stratégie 2 au portefeuille de la stratégie 1. Aucun chiffre ici.
