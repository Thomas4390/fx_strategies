# Validation rétrospective de la promotion du trio momentum — cycle 2026-H2 bis

> **Date** : 2026-07-28 · **Statut** : table de décision pré-gelée, mesures à venir
> **Holdout state** : LOCKED (frozen from 2026-01-01 until Phase 25 / 2026-12-31).
> **Holdout touched by this phase** : NO pour la sélection. Ce document **relit** une
> tranche gelée déjà publiée (le run de référence 2021→2026-04-30) pour en documenter
> la composition. Aucune sélection, aucun classement, aucune configuration nouvelle.

## 0. Pourquoi cette note

Le 2026-07-27, la sleeve momentum trio {XAUUSD, USDJPY, XAGUSD} a été promue à **20 %**
du portefeuille. Quatre livrables client ont été republiés sur cette configuration.

En rouvrant le dossier, quatre constats ont été établis sur les artefacts publiés
eux-mêmes. Ils ne remettent en cause aucun calcul : ils montrent que **ce qui est publié
ne dit pas ce que le résultat contient**, et que le gate de promotion que le projet
s'était donné n'a jamais été appliqué.

Cette note pré-gèle les conséquences avant de mesurer, pour qu'aucun chiffre ne soit lu
après coup afin de justifier ce qui est déjà en place. C'est le protocole de
`momentum_expansion_2026H2.md` §4.7, appliqué cette fois à une décision de production.

## 1. Table de décision — ENGAGÉE AVANT MESURE

Cette section est écrite et commitée **avant** l'exécution de
`scripts/audit_momentum_promotion.py`. Elle est exécutoire : la mesure ne sera pas
rediscutée, la conséquence sera appliquée telle qu'écrite ici.

| résultat mesuré | lecture | conséquence appliquée |
|---|---|---|
| **PBO-I < 0,5** et **PBO-W < 0,5** | la sélection d'instruments n'est pas contredite | la composition trio est conservée. Le **poids** reste arbitré par §1.2 |
| **PBO-I ≥ 0,5** | le classement des 21 instruments ne généralise pas : les survivants sont des tirages, pas des choix | **retrait de XAGUSD** de `Inp_Gold_Symbols` → sleeve = {XAUUSD, USDJPY}. XAGUSD est le seul survivant sans argument hors-modèle (cf. §1.3). Poids ramené à **0,125** |
| **PBO-W ≥ 0,5** | w = 0,20 est un artefact de plateau — le cas exact de la Phase 20A, rétractée | retour à **0,15** (Sharpe 1,134, maxDD −45,00 % au CSV du sweep) |

### 1.2 Le poids, indépendamment du PBO

Trois faits déjà établis, qui ne dépendent d'aucune mesure à venir :

1. Le CSV du sweep (`reports/research/momentum_weights_sweep_2026H2.csv`) montre que le
   maxDD atteint son **minimum à w = 0,175** (−41,91 %) et **remonte à 0,20** (−42,11 %).
   Le bénéfice de diversification se retourne : au-delà de 0,175, le Sharpe s'achète par
   concentration pure.
2. La contribution au risque est de **48,2 %** (vbt) / **52,5 %** (cache production) pour
   **20 %** du capital. Le garde-fou `tests/test_sleeve_sizing_conventions.py` ne protège
   pas contre cela : son ratio poids↔risque est concave (1,54 → 2,41 sur la grille) et ne
   franchira jamais le seuil de 3×, qui attrape un levier empilé, pas une surpondération.
3. `momentum_expansion_2026H2.md` §4.4 établit qu'**aucun survivant n'atteint DSR > 0**.

**Décision engagée** : quelle que soit l'issue du PBO, le poids ne sera pas maintenu à
0,20. Le cran retenu est **0,175 au maximum** si PBO-I et PBO-W passent tous deux, et
0,15 ou 0,125 selon les branches du tableau §1. La justification est le retournement du
drawdown, mesuré, pas un jugement sur le PBO.

### 1.3 Ce qui soutient chaque instrument hors du classement

Pour arbitrer la branche « PBO-I ≥ 0,5 », les arguments hors-modèle sont fixés ici,
avant de connaître le résultat :

| instrument | argument hors classement | tient si le classement tombe ? |
|---|---|---|
| XAUUSD | contrôle du screen, 7,2 ans vbt, parité MQL5 vérifiée bit-à-bit, sleeve d'origine | **oui** |
| USDJPY | accord des deux moteurs (vbt 0,51 sur 8 ans / MT5 0,81 sur 5 ans), orthogonalité mesurée (corr or −0,13), mécanisme de portage documenté | **oui** |
| XAGUSD | aucun. 2 ans de CFD propre, corr or 0,469 (le seul cluster du screen), **perd 4 511 $ sur la fenêtre de sélection** | **non** |

## 2. Ce que chaque PBO mesure — et ce qu'il ne mesure pas

Trois matrices sont candidates. Une seule porte de l'information ; le dire fait partie
de l'honnêteté du gate.

**PBO-I — la sélection d'instruments (celle qui compte).** 3 survivants tirés de 21
instruments classés : c'est le seul endroit du cycle où un choix réel a eu lieu. Matrice
= rendements quotidiens nets par instrument, aux conventions du screen (fill next_open,
coûts `costs.yml`, swap-drag). Deux réserves à publier avec le chiffre : on ne
`fillna(0.0)` jamais (un instrument qui ne cote pas n'est pas une position plate, et le
zéro écraserait sa vol), et le `dropna(how="any")` sur 21 colonnes irrégulières coupe la
fenêtre à 2022-11 — **PBO-I juge la sélection sur ~3,1 ans, pas 5**.

**PBO-W — la grille de poids (application littérale du gate, valeur faible).** 10 des 11
colonnes ne diffèrent que par un scalaire appliqué à une sleeve commune ; leurs
corrélations dépassent 0,99 et le Sharpe est monotone en w. Un CSCV sur une grille
monotone unidimensionnelle rendra PBO ≈ 0 quelle que soit la réalité sous-jacente : il
dira « le Sharpe est monotone en w dans chaque sous-échantillon », pas « la sélection
généralise ». **Un pass de PBO-W ne vaut rien comme preuve.** Il est calculé et publié
parce que c'est le gate promis en Phase 21, avec cette mise en garde à côté.

**PBO sur les 3 compositions — non calculable.** 3 colonnes ⇒ rang OOS ∈ {1,2,3} ⇒ le
logit ne peut prendre que 3 valeurs. Dégénéré. Consigné comme tel, pas calculé.

### 2.2 Le DSR doit nommer son univers de trials

Le rapport client publie **un** DSR sans dire de quoi. La sortie sera un tableau à
univers explicite, parce que le résultat n'est pas uniforme : le Sharpe **du
portefeuille** survit à un N correct (la dispersion des Sharpe portefeuille est faible),
tandis que le Sharpe **de chaque instrument** reste ≤ 0 (dispersion des Sharpe MT5 ≈ 0,8).
Ce n'est pas une contradiction, c'est le résultat central : **ce qui n'est pas prouvé,
c'est le choix des instruments, pas l'assemblage.**

Univers de trials publiés ensemble : **382 distincts** (chiffre de tête), 544 bruts
(borne conservatrice, re-runs inclus), 92 distincts hors `fx_legacy` (budget du seul
cycle momentum). Voir `framework.trials.distinct_trials`.

## 3. Les faits d'attribution (établis, à régénérer par le script)

Mesurés sur `reports/mt5/prod_ref_trio20_deals.csv` — sha256 identique à celui tracé
dans `results/production_report/mt5_reference.json`, désormais archivé dans le dépôt
(il ne vivait que dans `Common/Files/`, où le nom dérive de l'heure simulée et s'écrase
au run suivant : le run de référence publié n'était pas reproductible).

**Concentration par position** — 3 positions sur 909 trades portent 89 % du net :

| position | fenêtre | net | part du net publié |
|---|---|---|---|
| XAGUSD.c | 2025-05-21 → 2026-03-19 | +35 216 $ | **51,4 %** |
| XAUUSD.c | 2025-08-21 → 2026-03-23 | +15 062 $ | 22,0 % |
| XAUUSD.c | 2025-01-13 → 2025-07-31 | +10 692 $ | 15,6 % |

**Concentration temporelle** — scission par date de sortie :

| instrument | 2021 → 2025-12 (sélection) | 2026-01 → 04 (gelé) |
|---|---|---|
| XAGUSD.c | **−4 511 $** | +35 167 $ |
| XAUUSD.c | +12 592 $ | +12 190 $ |
| USDJPY.c | +3 428 $ | −1 051 $ |

Au niveau portefeuille (`mt5_reference.json`, bloc `yearly`) : 2021-2025 cumulent
+20 244 $, l'année 2026 sur 4 mois apporte **+48 291 $**, soit **70,5 % du net publié**.

Ce n'est **pas** une violation du holdout : la sélection s'est bien arrêtée au
2025-12-31 et le run de référence est un backtest de production, pas une lecture de
sélection. Mais le titre « 909 trades, Sharpe 0,97, +68 534 $ » décrit majoritairement
quatre mois que la recherche n'avait pas le droit d'utiliser, et cette scission n'est
publiée nulle part. `12_limitations.tex` mentionne que « l'argent apporte près de la
moitié du résultat de la sleeve » sans dire que c'est **une position**, ni que l'argent
**perd** sur la fenêtre de décision.

## 4. Le gate promis, et les deux défauts du livrable

`docs/research/phase21_2026-04-13_dsr_retrofit.md` avait mesuré un PBO de **0,853** sur
la Phase 20A, rétracté son « top », et conclu : *« DSR seul est insuffisant à cette
taille d'échantillon — il faut PBO (ou CPCV) comme test de gating : promouvoir seulement
si PBO < 0,5. »* Ce gate n'apparaît nulle part dans le cycle momentum.

Deux défauts distincts dans `reports/latex_report/tables/robustness_overfitting.tex` :

- ligne 10 : `DSR (N = 6 trials) = 1.0000`. La source
  (`scripts/build_latex_report_assets.py:1432`) passe `grid_sharpes=sleeve_sharpes` — le
  vecteur des Sharpe des **6 sleeves**, pas des configurations testées. Le registre en
  compte 382 distinctes. La légende annonce « tous les tests convergent vers un verdict
  favorable ».
- ligne 13 : `PBO CSCV = 0.335`, calculé sur la matrice des 4-6 séries de **sleeves**. Il
  répond à « choisir la meilleure sleeve tient-il hors échantillon » — une décision que
  le projet n'a jamais prise — et non au gate de la Phase 21, qui porte sur la grille de
  configurations ayant produit la promotion. Le rapport note que ce test « échouait à
  0,532 sur l'allocation précédente » : le chiffre a bougé parce que les **poids** ont
  changé, sur une matrice à 4 colonnes.

S'y ajoute une incohérence d'annualisation (`src/framework/robustness.py:181-198`) : le
bloc bootstrap annualise à 252 j, `_section_dsr_haircut` appelle `deflated_sharpe_ratio`
et `sharpe_ratio()` **sans `year_freq`**, donc sous le réglage ambiant vbt de 365 j.
Contrôle : 1,20 / 0,999 = 1,201 = √(365/252).

## 5. Protocole

1. Cette note (§0-§4) est commitée **avant** toute exécution.
2. `python scripts/audit_momentum_promotion.py --selfcheck` — la reconstruction doit
   redonner les 11 Sharpe de `momentum_weights_sweep_2026H2.csv` à 1e-9. Si elle échoue,
   la matrice n'est pas celle du sweep et rien n'est publié.
3. `python scripts/audit_momentum_promotion.py` — écrit `momentum_pbo_2026H2.csv`,
   `momentum_dsr_2026H2.csv`, `momentum_attribution_2026H2.csv`.
4. La conséquence du §1 est appliquée sans rediscussion, republication comprise.
5. Le script **n'appelle pas** `log_trials` : ré-évaluer des configurations déjà loguées
   n'est pas un test nouveau. Aucun essai n'est consommé par cette phase.

## 6. Résultats

_À remplir après exécution. Cette section est vide au moment du commit de la table de
décision — c'est la garantie que les seuils n'ont pas été choisis en connaissance du
résultat._
