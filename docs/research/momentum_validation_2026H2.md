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

Univers de trials publiés ensemble : **382 distincts** (chiffre de tête), 92 distincts
hors `fx_legacy` (budget du seul cycle momentum), plus le total brut re-runs inclus —
544 au moment de cette mesure, et **il croît à chaque relance d'un script** sans qu'aucun
espace de configurations nouveau soit exploré. C'est précisément pourquoi le chiffre de
tête est le distinct. Voir `framework.trials.distinct_trials`.

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

Exécution du 2026-07-28, `scripts/audit_momentum_promotion.py`. `--selfcheck` OK :
reconstruction des 11 Sharpe du sweep à **2,2e-16** (epsilon machine).

### 6.1 PBO — le verdict

| matrice | n_configs | fenêtre | PBO (8/10/12/16 bins) | verdict |
|---|---|---|---|---|
| `weights_sweep` | 11 | 1822 séances, 2019-01→2025-12 | **0,786 / 0,841 / 0,824 / 0,754** | **OVERFIT** |
| `instruments_common` | 21 | 608 séances, 2022-11→2025-12 | 0,200 / 0,302 / 0,176 / — | SAIN |
| `instruments_deep` | 5 | 8298 séances, 1992→2025 | 0,014 / 0,024 / 0,038 / — | SAIN |

**Branche déclenchée : PBO-W ≥ 0,5.** La table §1 commande le retour à **w = 0,15**.
PBO-I étant sain, la composition trio est **conservée** — XAGUSD n'est pas retiré.

### 6.2 Ce que j'avais prédit et qui était faux

Le §2 annonçait qu'un CSCV sur une grille monotone unidimensionnelle rendrait
« PBO ≈ 0 quoi qu'il arrive » et que le pass serait vide de sens. **C'est démenti** : le
test avait du pouvoir discriminant, et il a rejeté.

Le mécanisme est cohérent avec le §3 : la monotonie du Sharpe en w sur la fenêtre
*complète* n'entraîne pas la monotonie dans *chaque sous-échantillon*. Quand trois
positions portent 89 % du net, le poids fort domine les splits qui contiennent ces trades
et se retrouve sous la médiane dans les autres. Le PBO mesure exactement cela : le poids
fort ne gagne hors échantillon que dans ~20 % des découpes.

### 6.3 Décomposition post-hoc — publiée, non utilisée pour décider

Découpes décidées **après** avoir vu le résultat global, marquées `post_hoc=True` dans
`momentum_pbo_2026H2.csv`. Elles informent sur l'origine du rejet ; les utiliser pour
renverser la décision serait exactement le biais que le pré-gel interdit.

| sous-ensemble | PBO (10 bins) | PBO (12 bins) | lecture |
|---|---|---|---|
| toutes (11) | 0,841 | 0,824 | la matrice qui décide |
| sans baseline (10) | 0,849 | 0,843 | la baseline n'explique pas le rejet |
| **trio seul (5 poids)** | **0,282** | **0,445** | à composition fixée, la grille de poids passe |
| duo seul (5 poids) | 0,837 | 0,840 | la composition duo ne généralise pas |

Lecture : la matrice complète mélange deux décisions — quelle composition, quel poids. Le
rejet vient surtout de la **composition**, pas du poids à composition donnée. Le trio
passe seul, mais son PBO monte avec le nombre de bins (0,25 → 0,44), ce qui n'est pas le
profil d'un résultat franc.

Cela ne change pas la décision appliquée, et c'est voulu : la table a été écrite sur la
matrice complète, elle s'applique sur la matrice complète.

### 6.4 DSR — l'assemblage tient, le choix des instruments non

| objet déflaté | univers | N | Sharpe | E[max SR] | DSR |
|---|---|---|---|---|---|
| portefeuille w=0,20 | 11 Sharpe du sweep | 11 | 1,185 | 0,067 | 1,00 PASS |
| portefeuille w=0,20 | registre, distinctes | 382 | 1,185 | 0,122 | 1,00 PASS |
| portefeuille w=0,20 | registre brut à la mesure | 544 | 1,185 | 0,126 | 1,00 PASS |
| XAU-USD | classement MT5, 21 instr. | 382 | 0,738 | 2,891 | 0,00 FAIL |
| USD-JPY | idem | 382 | 0,510 | 2,891 | 0,00 FAIL |
| XAG-USD | idem | 382 | 0,201 | 2,891 | 0,00 FAIL |

**Le PASS du portefeuille doit être lu avec sa réserve** : l'écart-type des 11 Sharpe vaut
0,024, parce que les 11 configurations sont quasi identiques. Déflater avec la dispersion
d'une grille dégénérée donne un `E[max SR]` de 0,12 — donc un DSR de 1,00 quasi
automatique. Ce chiffre est moins faux que le `N = 6` du livrable, il n'est pas beaucoup
plus informatif. **C'est le PBO, pas le DSR, qui a tranché.**

Le FAIL des instruments est, lui, franc : la dispersion des Sharpe MT5 du classement
(≈ 0,8) porte `E[max SR]` à 2,89, très au-dessus du meilleur survivant.

### 6.5 Attribution — recoupée avec le JSON publié

`momentum_attribution_2026H2.csv` retrouve exactement les parts de `mt5_reference.json` :
Gold Momentum **91,08 %** du net, XAGUSD 44,73 %, XAUUSD 36,16 %, USDJPY 15,05 %. La
position dominante XAGUSD #1575 (2025-05-21 → 2026-03-19) fait **51,38 %** du net publié.
Le dépôt initial (`DEAL_TYPE_BALANCE`) est exclu du dénominateur : l'y laisser aurait
sous-estimé toutes les parts de 15 %.

### 6.6 Deux découvertes de reproductibilité

**Le sweep de poids publié n'était plus reproductible.** La promotion a changé les deux
entrées dont il dépend : `PRODUCTION_WEIGHTS` (la baseline « or seul à 0,10 » est devenue
« trio à 0,20 ») et le contenu du cache `Gold_Momentum`, qui porte désormais le trio et
remonte à 2000 par la série Yahoo de l'argent. L'intersection des quatre sleeves n'est donc
plus bornée à gauche par l'or : **la fenêtre passe de 1822 séances (2019-01-02) à 2081
(2018-01-02)**, et tous les Sharpe bougent de 0,04 à 0,33. `sweep_context()` refait la
sleeve or seul et reprend les poids d'alors ; le `--selfcheck` prouve que cela redonne le
CSV publié au bit près.

Corollaire non documenté à ce jour : **la fenêtre du portefeuille de production a glissé de
2019 à 2018 lors de la promotion** (2146 séances, 2018-01-02 → 2026-04-01). Sur
2018→2019, la sleeve momentum ne contient ni l'or (export QC à partir de 2019) : elle vaut
`(usdjpy + xag)/3`, conformément à la règle « absent = 0, pas de redistribution » qui
mirroir le `sub_equity/n` MQL5. Ce n'est pas un défaut de logique, mais c'est un
changement d'échantillon qui a accompagné le changement de poids sans être noté.

**Les « deux conventions de Sharpe » n'en sont qu'une.** Le portefeuille de production
mesure 0,9983 à 252 jours ; 0,9983 × √(365/252) = **1,2018**, soit le « Sharpe vbt 1,20 »
publié. Ce n'est pas deux mesures divergentes, c'est le même nombre sous deux
annualisations, publiées côte à côte comme si elles étaient cohérentes. Cause dans
`src/framework/robustness.py:181-198`.

### 6.7 Conséquence appliquée

Conformément au §1, sans rediscussion :

- **poids de la sleeve momentum : 0,20 → 0,15** (MR Macro 0,62 → 0,67) ;
- **composition inchangée** : {XAUUSD, USDJPY, XAGUSD}, PBO-I sain ;
- nouveau run de référence MT5, allocations épinglées, et republication des livrables sur
  cette configuration ;
- correction des deux défauts du tableau de robustesse (N = 6 → 382, PBO du bon objet) et
  de l'annualisation, plus divulgation de la concentration.

---

## 7. Re-mesure après correction du modèle de coût (2026-07-28, phase 3)

Le modèle de coût de la recherche facturait le portage en valeur absolue à tous les
instruments — `ret - swap × |exposition|` — donc il **prélevait sur USD/JPY un portage que
le compte encaisse**, alors que la thèse du dossier attribue à ce portage 63 % du résultat
de cette jambe. Le modèle contredisait ce qu'il servait à établir. Corrigé
(`strategies.tsmom.carry_sign`, signe issu du catalogue broker archivé).

**Re-mesurer après correction d'un bug est légitime ; choisir entre l'ancienne et la
nouvelle mesure selon celle qui arrange ne le serait pas.** La table de décision du §1
s'applique donc telle qu'écrite, aux nouveaux chiffres.

### 7.1 Ce que la correction déplace

| instrument | Sharpe vbt avant | après | Δ |
|---|---|---|---|
| USD-JPY | 0,538 | **0,770** | +0,232 |
| GBP-JPY | 0,079 | **0,314** | +0,235 |
| XAU-USD, XAG-USD, indices, EUR-USD | inchangés | | 0 |

Le criblage **sous-estimait systématiquement de ~0,23 de Sharpe la classe d'instruments que
la thèse désigne**. Conséquences sur le classement du pré-filtre : USD-JPY passe 5ᵉ → 3ᵉ
(0,73, à un cheveu de l'or à 0,74), EUR-JPY 7ᵉ → 4ᵉ (0,71), GBP-JPY 12ᵉ → 9ᵉ, et
**USD-CAD cesse d'être tué** (−0,24 → 0,00). Les kills passent de 7 à 6.

### 7.2 La décision ne change pas

| matrice | PBO avant | PBO après | verdict |
|---|---|---|---|
| `weights_sweep` (11 configs) | 0,754 – 0,841 | **0,649 – 0,732** | **OVERFIT** dans les deux cas |
| `instruments_common` (21) | 0,176 – 0,302 | 0,242 – 0,393 | SAIN dans les deux cas |
| `weights_trio_only` (post-hoc) | 0,249 – 0,445 | 0,167 – 0,336 | sain |
| `weights_duo_only` (post-hoc) | 0,812 – 0,840 | 0,598 – 0,661 | overfit |

Le PBO baisse — la correction améliore le trio — mais **reste largement au-dessus du seuil
de 0,5**. La branche « PBO-W ≥ 0,5 → w = 0,15 » est déclenchée une seconde fois, sur une
mesure indépendante du bug. **Le poids de 0,15 est confirmé, pas reconduit par inertie.**

Le sweep corrigé montre par ailleurs un trio nettement meilleur qu'estimé : Sharpe 1,261 à
w = 0,20 (contre 1,185) et surtout **maxDD −40,4 % contre −58,5 % pour la baseline or seul**,
là où l'ancien modèle affichait −42,1 %. Le minimum de drawdown reste à w = 0,175
(−40,2 %), donc l'argument du §1.2 tient aussi.

### 7.3 Un second défaut de reproductibilité, corrigé

La baseline du sweep dérivait de `PRODUCTION_WEIGHTS`. Un sweep dont la référence suit la
production **se réécrit à chaque changement de production**, donc ne peut plus servir à
juger ce changement : relancé après la promotion, il comparait le trio à lui-même sur une
fenêtre de 2081 séances au lieu de 1822. La baseline est désormais figée en clair
(`BASELINE_WEIGHTS`, or seul à 0,10), et `sweep_context()` de l'audit délègue à cette
fonction unique plutôt que d'en tenir une copie.
