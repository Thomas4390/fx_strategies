# XAUUSD niveaux x10 : résultats in-sample 2019 → 2025

> **Date** : 2026-09-21 · **Statut** : mesuré, **v2** — verdict in-sample **NE PAS DÉPLOYER**
> **Convention de données** : `bar_open` — le parquet exporté de LEAN date une minute à sa
> clôture, la référence la redate à l'ouverture (spec §1). La campagne v1, qui ne le faisait
> pas, est archivée telle quelle dans `results/xau_x10/v1_close_stamped/`.
> **Holdout state** : LOCKED.
> **Holdout touched by this phase** : **NO** — aucune barre ≥ 2026-01-01 n'est entrée dans un
> calcul. Date maximale de tout index consommé : M1 `2025-12-31 16:58`, M5 `2025-12-31 16:55`,
> séance `2025-12-31`.
> **Essais consommés** : **34** — 27 (`xau_x10:grid27:v1`) + 7 (`xau_x10:ablations:v1`), sur un
> plafond de 40 (spec annexe A.3). Réserve non entamée : 6.

Tous les chiffres de cette note sortent de `results/xau_x10/*.json`, produits par
`scripts/run_xau_x10_research.py --step all`. Aucun n'est recalculé à la main. Les seuils et la
règle de sélection sont ceux de `docs/research/xau_x10_decision_table.md`, gelés **avant**
cette mesure.

**Le résultat est négatif  franchement et à toutes les coupes.** La spec (§0) et la table de
décision (§2) prévoyaient ce livrable ; il est rendu tel quel.

> ### v1 → v2 : correction de datation. Ce qui a changé, et pourquoi ce n'est pas une re-sélection
>
> **Le défaut.** Le parquet `data/XAU-USD_minute_qc.parquet` date une barre M1 de sa **clôture**
> (`end_time` de LEAN). La référence traitait cet horodatage comme un **début** : sa grille M5
> couvrait `[t−1 min, t+4 min)` quand MT5 et le portage QuantConnect couvrent `[t, t+5 min)`.
> Une minute d'écart, sur toutes les barres, et invisible à un test de décalage en *barres*. La
> preuve est arithmétique : la clôture M5 publiée par QC vaut la ligne du parquet estampillée
> `bm + 5 min` sur **316 tags sur 316**, au centime (`xau_x10_reconciliation.md` §11.6 bis).
> Les parquets FX qui composent le DXY4 portent la même convention : corrélation **0,991** au
> décalage `+1` minute contre l'export MT5 d'EURUSD, 0,07 partout ailleurs.
>
> **La correction.** `strategies.xau_x10.SOURCE_STAMP = "close"` et `_to_bar_open()` retranchent
> une minute avant toute agrégation ; `scripts/build_dxy_synthetic.py` fait de même sur les
> quatre jambes. `src/utils.py` n'est pas touché — il est partagé avec la stratégie 1. Campagne
> **intégralement rejouée** (`--step all`).
>
> **Pourquoi ce n'est pas une re-sélection.** Aucun seuil, aucune règle, aucun paramètre gelé
> n'a bougé : même grille de 27, même spread de sélection de 0,29 $, même table de décision, et
> `distinct_trials("xau_x10")` vaut toujours **34** — les `config_key` `xau_x10:grid27:v1` et
> `xau_x10:ablations:v1` dédoublonnent le rejeu, qui ne consomme donc aucun essai. C'est la même
> mesure, faite sur la grille que la spec décrit.
>
> | critère (table §1) | v1 (close-stamped) | **v2 (bar_open)** | verdict |
> |---|---|---|---|
> | 1 Trades | 1 723 | **1 762** | GO → GO |
> | 2 Espérance / PF | −0,1602 / 0,737 | **−0,1679 / 0,728** | échec → échec |
> | 3 DSR | 0,000 | **0,000** | échec → échec |
> | 4 PBO | 0,415 | **0,246** | **échec → GO** |
> | 5 Plateau | 0 / 27 | **0 / 27** | échec → échec |
> | 6 Walk-forward | 0 % d'années positives | **0 %** | échec → échec |
> | 7 Spread ×1,5 | −0,2012 R | **−0,2078 R** | échec → échec |
> | 8 MT5 modèle 1 | non mesuré | **−0,1034 R**, même signe | échec → échec |
> | 9 OOS 2026 | non mesuré | non mesuré | échec → échec |
> | Sharpe net | −1,661 | **−1,941** | |
> | Max drawdown | −63,7 % | **−68,0 %** | |
>
> **Un seul critère change de camp**, le PBO, et dans le sens favorable. Le verdict ne bouge
> pas : huit critères en échec deviennent sept, et il en faut zéro. Ce que la correction a
> vraiment débloqué, c'est la **réconciliation** : l'appariement Python ↔ QC passe de 33,1 % à
> **98,7 %** sur 2024, au-dessus de la cible de 98 % de la spec §13.

---

## 1. Verdict provisoire et table de décision

`results/xau_x10/is_summary.json`. Configuration analysée : **`z = 1 ; a_min = 0,2 ; k_s = 1`**
— le centre géométrique de la grille, **qui n'est pas une configuration retenue** (§2).

| # | critère (table §1) | seuil GO | mesuré | verdict |

|---|---|---|---|---|
| 1 | Trades in-sample 2019 → 2025 | ≥ 300 | **1 762** (4 scénarios ≥ 40) | **GO** |
| 2 | Espérance nette / profit factor | ≥ +0,10 R / ≥ 1,15 | **−0,1679 R** / **0,728** | **échec** |
| 3 | DSR (déflaté par 34 essais) | ≥ 0,95 | **0,000** | **échec** |
| 4 | PBO (CSCV, 27 × 1 824 jours) | ≤ 0,35 | **0,246** | **GO** |
| 5 | Plateau | ≥ 60 % voisins > 0 ; médiane ≥ 0,5 × pic | **0 %** ; médiane −1,714 contre seuil −0,504 | **échec** |
| 6 | Walk-forward annuel | ≥ 60 % d'années positives ; aucune > 50 % du net | **0 %** d'années positives ; max 28,7 % | **échec** |
| 7 | Spread ×1,5 | espérance > 0 | **−0,2078 R** | **échec** |
| 8 | MT5 modèle 1, 2022-11 → 2025-12 | espérance ≥ 0, même signe | **−0,1034 R**, même signe que Python | **échec** |
| 9 | OOS 2026, lecture unique | n ≥ 30 | **non mesuré** | **échec** |


**Deux** critères sur neuf passent — le nombre de trades et le PBO. Sept échecs, dont **six**
portent une mesure interprétable ; seul l'OOS 2026 est une non-mesure, qui vaut échec et non
« non applicable » (table §1). Le critère 8 est désormais **mesuré** : l'EA MT5 rend une
espérance de **−0,1034 R** sur 880 trades, du **même signe** que Python sur la même fenêtre
(−0,1522 R) — il échoue parce qu'il est négatif, pas parce qu'il manque
(`results/xau_x10/mt5_reference.json`).

> **Verdict in-sample PROVISOIRE : NE PAS DÉPLOYER.**
> Il est provisoire au sens strict de la table §2 : le verdict définitif se prononce une seule
> fois, après la réconciliation MT5 (spec §13) et la lecture OOS unique. Rien dans ce qui suit
> n'autorise à anticiper cette lecture, et rien n'appelle à la faire : l'in-sample suffit à
> conclure sur le déploiement.

---

## 2. Sélection : aucune configuration retenue

`results/xau_x10/selection.json`, règle du §3 de la table appliquée à la lettre.

- **Pic de la grille** : `z = 0,5 ; a_min = 0,3 ; k_s = 1,5`, Sharpe net **−1,0089**.
- **Seuil de la médiane** : 0,5 × pic = **−0,5045**.
- **Configurations passant le critère plateau : 0 sur 27.**

Les 27 Sharpes nets sont négatifs, de −1,009 à −1,994. La part de voisins à Sharpe > 0 vaut
donc 0 % partout, contre 60 % exigés : aucune configuration ne passe, et le §3 est explicite —
**il n'y a pas de configuration retenue, et le verdict est NE PAS DÉPLOYER.** On ne se rabat
pas sur le pic.

Détail à ne pas manquer : quand le pic est négatif, `0,5 × pic` est **plus haut** que le pic,
donc la seconde moitié du critère devient plus dure, pas plus facile. La règle gelée ne se
dégrade pas en passoire sur un échantillon perdant — ici la médiane des voisins du centre vaut
−1,714 contre un seuil de −0,504, et échoue aussi.

**Plateau contre pic.** La table demande de publier les deux côte à côte. Pic
`z0.5_a0.3_k1.5` à −1,009 ; configuration analysée `z1_a0.2_k1` à
−1,941 ; écart **−0,932** de Sharpe. L'écart est réel, et il n'a aucune conséquence : les deux perdent.

**Défaut descriptif.** Pour que les décompositions du §5 aient un support, elles portent sur le
**centre géométrique de la grille** (`z = 1 ; a_min = 0,2 ; k_s = 1`). Ce choix était écrit
dans le brief de campagne **avant** la mesure ; ce n'est **pas** une sélection et il ne doit
jamais être présenté au client comme « la configuration de la stratégie ». Le champ
`analysed_is_a_selection` vaut `false` dans tous les JSON.

---

## 3. La grille des 27, au spread constant de 0,29 $

`results/xau_x10/grid27.csv`. Rendements par séance (§2), Sharpe annualisé à 252.

| configuration | z | a_min | k_s | Sharpe net | trades | E[R] | PF | max DD |

|---|---|---|---|---|---|---|---|---|
| z0.5_a0.3_k1.5 | 0,5 | 0,3 | 1,50 | −1,009 | 1 074 | −0,1159 | 0,811 | −40,7 % |
| z0.5_a0.3_k0.75 | 0,5 | 0,3 | 0,75 | −1,032 | 1 144 | −0,1201 | 0,813 | −44,8 % |
| z0.5_a0.3_k1 | 0,5 | 0,3 | 1,00 | −1,087 | 1 117 | −0,1250 | 0,802 | −45,2 % |
| z1.5_a0.3_k0.75 | 1,5 | 0,3 | 0,75 | −1,186 | 1 663 | −0,1184 | 0,825 | −52,3 % |
| z1.5_a0.3_k1.5 | 1,5 | 0,3 | 1,50 | −1,189 | 1 523 | −0,1235 | 0,814 | −48,5 % |
| z1.5_a0.1_k0.75 | 1,5 | 0,1 | 0,75 | −1,220 | 2 242 | −0,0992 | 0,830 | −59,8 % |
| z1.5_a0.2_k0.75 | 1,5 | 0,2 | 0,75 | −1,243 | 1 969 | −0,1179 | 0,820 | −56,5 % |
| z1.5_a0.2_k1.5 | 1,5 | 0,2 | 1,50 | −1,298 | 1 821 | −0,1281 | 0,803 | −54,9 % |
| z1.5_a0.3_k1 | 1,5 | 0,3 | 1,00 | −1,329 | 1 609 | −0,1304 | 0,804 | −54,0 % |
| z1.5_a0.1_k1.5 | 1,5 | 0,1 | 1,50 | −1,351 | 2 088 | −0,1111 | 0,803 | −59,5 % |
| z1.5_a0.2_k1 | 1,5 | 0,2 | 1,00 | −1,388 | 1 910 | −0,1282 | 0,799 | −59,1 % |
| z1.5_a0.1_k1 | 1,5 | 0,1 | 1,00 | −1,399 | 2 182 | −0,1095 | 0,807 | −62,3 % |
| z1_a0.3_k1.5 | 1,0 | 0,3 | 1,50 | −1,410 | 1 370 | −0,1486 | 0,780 | −53,5 % |
| z1_a0.3_k0.75 | 1,0 | 0,3 | 0,75 | −1,411 | 1 493 | −0,1452 | 0,784 | −56,4 % |
| z1_a0.3_k1 | 1,0 | 0,3 | 1,00 | −1,588 | 1 451 | −0,1565 | 0,759 | −58,4 % |
| z0.5_a0.2_k0.75 | 0,5 | 0,2 | 0,75 | −1,615 | 1 447 | −0,1628 | 0,746 | −59,9 % |
| z0.5_a0.2_k1.5 | 0,5 | 0,2 | 1,50 | −1,654 | 1 359 | −0,1656 | 0,735 | −56,9 % |
| z0.5_a0.2_k1 | 0,5 | 0,2 | 1,00 | −1,673 | 1 414 | −0,1678 | 0,733 | −59,5 % |
| z1_a0.2_k1.5 | 1,0 | 0,2 | 1,50 | −1,755 | 1 672 | −0,1619 | 0,751 | −63,2 % |
| z1_a0.2_k0.75 | 1,0 | 0,2 | 0,75 | −1,775 | 1 809 | −0,1563 | 0,754 | −67,1 % |
| z1_a0.1_k0.75 | 1,0 | 0,1 | 0,75 | −1,793 | 2 110 | −0,1364 | 0,772 | −71,0 % |
| z0.5_a0.1_k0.75 | 0,5 | 0,1 | 0,75 | −1,800 | 1 726 | −0,1571 | 0,738 | −67,4 % |
| z1_a0.1_k1.5 | 1,0 | 0,1 | 1,50 | −1,818 | 1 971 | −0,1422 | 0,761 | −68,7 % |
| z0.5_a0.1_k1 | 0,5 | 0,1 | 1,00 | −1,883 | 1 689 | −0,1606 | 0,726 | −67,2 % |
| z0.5_a0.1_k1.5 | 0,5 | 0,1 | 1,50 | −1,894 | 1 627 | −0,1635 | 0,723 | −65,1 % |
| **z1_a0.2_k1** *(analysée)* | 1,0 | 0,2 | 1,00 | **−1,941** | **1 762** | **−0,1679** | **0,728** | **−68,0 %** |
| z1_a0.1_k1 | 1,0 | 0,1 | 1,00 | −1,994 | 2 057 | −0,1495 | 0,746 | −72,7 % |


Ce qu'on lit dans le classement, et qu'il faut dire sans le sur-interpréter : la grille est
**monotone en `a_min`**. Les neuf configurations à `a_min = 0,1` occupent le bas du tableau,
les neuf à `a_min = 0,3` le haut. Un `a_min` plus exigeant arme moins souvent, donc trade
moins, donc perd moins. Ce n'est pas un signal de bord : c'est la signature d'un coût fixe par
trade qui domine tout le reste. `z` agit dans le même sens et pour la même raison (une zone
plus large arme plus). `k_s` ne classe rien.

---

## 4. Robustesse

`results/xau_x10/robustness.json`. Déflaté par `distinct_trials("xau_x10") = 34`, graine 42,
2 000 réplicats bootstrap, blocs stationnaires de 20 séances.

| mesure | valeur |
|---|---|
| Sharpe net (séance, 252) | **−1,941** — IC 95 % bootstrap **[−2,686 ; −1,285]** |
| PSR (contre 0) | **0,000** |
| DSR (N = 34) | **0,000** — Sharpe attendu du maximum de 34 essais : **+0,636** |
| Haircut Sharpe (BHY, N = 34) | −0,194 ; p ajustée **1,000** |
| MinBTL | non calculable (défini pour un Sharpe observé positif seulement) |
| PBO (CSCV, 16 bins, 12 870 combinaisons) | **0,2460** |
| Rendement total | **−67,2 %** — IC 95 % [−77,8 % ; −53,0 %] |
| Max drawdown | **−68,0 %** — IC 95 % [−54,9 % ; −78,2 %] |
| Profit factor | 0,673 — IC 95 % [0,581 ; 0,766] |

**L'IC 95 % du Sharpe ne contient pas zéro** : la perte n'est pas un accident d'échantillonnage.

**Monte Carlo sur l'ordre des 1 762 trades** (2 000 permutations) : max drawdown
observé 34,7 % du capital, médiane des permutations 34,6 %, p95 36,4 %,
*z* de chance de séquence **+0,14**. Autrement dit l'ordre réalisé
n'a **pas** été malchanceux — la perte ne vient pas d'un enchaînement défavorable, elle vient
de l'espérance. Sous permutation, la durée médiane passée sous l'eau est de
**1 748 barres de trade sur 1 762**.

**Risque de ruine** (`ruin_report`, 2 000 chemins bootstrap de la longueur de l'échantillon) :
rendement annualisé **−15,10 %**, volatilité 7,78 %, MAR −0,222,
P(ruine) 0,000 mais **P(perte > 50 %) = 0,993**, terminal médian
**0,326** du capital initial, drawdown p95 du bootstrap **76,9 %**.
Recouvrement le plus long : **1 817 séances sur 1 824** — la courbe
d'équité passe 99 % du temps sous son plus haut.

**SPA / StepM — deux chiffres à ne pas citer.** Le StepM de Romano-Wolf **échoue** sur cette
grille (le step-down se retrouve avec un tableau vide quand aucune configuration ne bat le
benchmark plat). Le SPA rend p = 0,000, ce qui **signifie l'inverse de ce qu'il paraît dire** :
`reality_check_via_arch` passe des *rendements* à `arch.bootstrap.SPA`, qui documente attendre
des *pertes*, et l'inversion d'orientation fait passer une grille intégralement perdante pour
une grille gagnante. Défaut du module `src/framework/statistical_testing.py`, signalé et **non
corrigé ici** ; aucun critère du §1 n'en dépend. Le `caveat` est recopié dans le JSON.

### CPCV — distribution hors échantillon

`results/xau_x10/cpcv.json`. 6 groupes, 2 en test, purge 1 jour, embargo 1 % — soit
**15 splits** et 5 chemins reconstruits, sur les 27 configurations.

**Aucune des 27 configurations n'a une médiane OOS positive** (0 sur 27), et aucune n'est
positive sur un seul des 15 splits (`pct_positive = 0` partout). Meilleure médiane OOS :
`[0.5, 0.3, 1.5]` à **−0,959** ; moyenne OOS sur toute la grille
**−1,410**. La métrique CPCV
est le Sharpe des rendements M1 du portefeuille (convention de `create_cv_pipeline`) et n'est
donc comparable au tableau du §3 **qu'en signe**.

Le PBO vaut **0,246** et passe le seuil de 0,35 — c'était le seul critère que la
correction de datation a fait changer de camp (v1 : 0,415). Il ne dit pas que la stratégie est
saine : sur une grille qui perd partout, un PBO bas signifie seulement que le classement des 27
configurations est un peu plus persistant d'un demi-échantillon à l'autre. Il n'y a rien à
surajuster, parce qu'il n'y a pas de signal.

---

## 5. Décompositions

`results/xau_x10/decompositions.json`, configuration `z1_a0.2_k1`, 1 762 trades, PnL net
**−6 721,8 $** sur 10 000 $ de capital initial, taux de réussite **30,99 %**.

### 5.1 Par scénario (les quatre sont au-dessus de 40 trades, donc commentables)

| scénario | n | E[R] | PF | PnL |

|---|---|---|---|---|
| BREAK_LONG | 386 | −0,1612 | 0,782 | −1 162,0 $ |
| BREAK_SHORT | 292 | −0,1742 | 0,708 | −1 348,3 $ |
| REV_LONG | 546 | −0,1819 | 0,690 | −2 350,2 $ |
| REV_SHORT | 538 | −0,1549 | 0,741 | −1 861,2 $ |


Les quatre perdent. Le moins mauvais est `BREAK_LONG` (−0,087 R), le pire `REV_SHORT`
(−0,220 R), qui porte à lui seul 44 % de la perte. La hiérarchie breakout > reversal est nette
et cohérente avec le fait que les breakouts sont les seuls à passer un filtre de contexte.

### 5.2 Par séance New York (heure de décision)

| séance | n | E[R] | PF | PnL |

|---|---|---|---|---|
| Asie 18:00-03:00 | 620 | −0,1615 | 0,725 | −2 431,1 $ |
| Londres 03:00-08:00 | 416 | −0,1702 | 0,746 | −1 642,5 $ |
| New York 08:00-17:00 | 726 | −0,1719 | 0,718 | −2 648,3 $ |


**La séance asiatique porte 58 % de la perte pour 35 % des trades.** C'est la coupe la plus
contrastée de toute la campagne. Elle ne suggère pas un filtre horaire à ajouter — ce serait un
paramètre choisi après mesure, donc interdit (table §5.1) — mais elle est la première chose à
regarder si le mandat est un jour rouvert.

### 5.3 Par régime d'ATR M5 et par `10 $ / ATR`

| tercile d'ATR M5 | n | E[R] | PF |

|---|---|---|---|
| ATR bas | 588 | −0,2445 | 0,668 |
| ATR haut | 587 | −0,0948 | 0,786 |
| ATR moyen | 587 | −0,1642 | 0,761 |


| 10 $ / ATR | n | E[R] | PF | PnL |

|---|---|---|---|---|
| < 3 | 145 | −0,2645 | 0,670 | −499,3 $ |
| 3-6 | 701 | −0,1192 | 0,799 | −1 649,1 $ |
| 6-10 | 580 | −0,1122 | 0,809 | −1 609,1 $ |
| <3 | 153 | −0,1628 | 0,577 | −620,8 $ |
| >10 | 328 | −0,3726 | 0,569 | −2 842,7 $ |


La tranche `6-10` est la moins mauvaise (−0,060 R), les deux extrêmes les pires. C'est
exactement la lecture de la note de faisabilité §1 : la grille de 10 $ ne représente pas la
même chose selon le régime, et les régimes où 10 $ vaut moins de 3 ATR (le prix traverse
plusieurs niveaux par barre) comme ceux où il en vaut plus de 10 (la cible est hors de portée
dans les 48 barres) sont tous deux défavorables. **Aucune des quatre tranches n'est positive.**

### 5.4 Par année — voir §6.

### 5.5 Par raison de sortie

| sortie | n | part | E[R] | PF | PnL |

|---|---|---|---|---|---|
| STOP | 1 113 | 63,2 % | **−1,0002** | 0,000 | −23 867,4 $ |
| TARGET | 272 | 15,4 % | **2,0798** | ∞ | 10 913,2 $ |
| TIME (48 barres) | 283 | 16,1 % | **0,7579** | 11,496 | 5 485,2 $ |
| SESSION (16:55) | 94 | 5,3 % | **0,3964** | 3,446 | 747,2 $ |


Cette table est l'arithmétique complète de l'échec, et elle est saine : le stop rend
**exactement −1,00 R**, ce qui vérifie le dimensionnement du §11 de bout en bout, et la cible
rend 2,08 R. Avec ces deux chiffres seuls, le point mort serait un taux de cible de
32,5 % ; le taux réalisé est de **15,4 %**. Les sorties temporelles, positives en moyenne, ne comblent
pas l'écart.

La part de trades sortis à **48 barres est de 16,1 %** — l'ordre de grandeur annoncé
au brief de campagne (« 17 % si confirmé ») est confirmé. Durée de vie moyenne :
19,2 barres M5 ; médiane 13.

### 5.6 Reversals étendus, DXY, retest

| coupe | n | E[R] | PF | PnL |

|---|---|---|---|---|
| reversal étendu (`extension_vwap = 1`) | 688 | −0,1509 | 0,737 | −2 538,4 $ |
| reversal non étendu | 396 | −0,1991 | 0,672 | −1 673,0 $ |
| adverse | 829 | −0,1455 | 0,770 | −1 621,2 $ |
| favorable | 933 | −0,1877 | 0,711 | −5 100,6 $ |
| retest | 769 | −0,1715 | 0,701 | −3 148,9 $ |
| sans retest | 993 | −0,1650 | 0,748 | −3 572,9 $ |


- **Extension VWAP** : les reversals étendus perdent près de deux fois moins que les autres.
  La grandeur est traçée et non bloquante par décision du mandat (spec §7.3) ; c'est la seule
  variable descriptive de la campagne qui sépare quelque chose, et elle reste du mauvais côté
  de zéro.
- **DXY** : le contexte favorable perd **plus en dollars** (−4 434 $ contre −1 824 $) tout en
  perdant moins par unité de risque (−0,146 R contre −0,178 R). Il n'y a pas de contradiction :
  le risque est divisé par deux quand le DXY est adverse (§11), donc les trades adverses
  pèsent moitié moins. Le garde-fou DXY fonctionne donc comme prévu — il réduit la perte — mais
  il ne discrimine pas : **l'espérance est négative des deux côtés**.
- **Retest** : le retest est **contre-prédictif** (−0,187 R avec, −0,138 R sans). Le mandat le
  citait comme confirmation ; la mesure ne le soutient pas. Il reste descriptif, comme la spec
  l'a figé.

### 5.7 Concentration

Net total **−6 721,8 $**, profit brut **17 973,6 $**.

Les parts du net ne sont pas publiables ici : le net total est négatif et « x % d'un nombre
négatif » n'a pas de sens. Les montants, eux, le sont.

| concentration | montant | part du **profit brut** |

|---|---|---|
| 5 meilleures séances | +1 078,7 $ | 6,0 % |
| meilleur mois (2019-10) | +401,9 $ | — |
| 10 meilleurs trades | +1 776,6 $ | 9,9 % |


**397 séances gagnantes contre 766 perdantes** sur 1 824. Aucune concentration pathologique :
la perte n'est pas l'accident de quelques jours, elle est diffuse, ce qui la rend d'autant plus
crédible. Le meilleur mois de sept ans rapporte **401,9 $**, soit 4,0 % du capital initial.

### 5.8 Taux de rejet de la règle R ≥ 1, par année (obligation spec §8 / table §5.5)

Dénominateur : les candidats **ayant atteint le test R** — un refus au contexte n'a pas de
`r_est` et n'y entre pas.

| année | candidats | rejetés | taux |

|---|---|---|---|
| 2019 | 170 | 14 | **8,2 %** |
| 2020 | 343 | 79 | **23,0 %** |
| 2021 | 307 | 50 | **16,3 %** |
| 2022 | 294 | 48 | **16,3 %** |
| 2023 | 248 | 39 | **15,7 %** |
| 2024 | 410 | 87 | **21,2 %** |
| 2025 | 736 | 408 | **55,4 %** |


| 2019 | 166 | 11 | **6,6 %** |
| 2020 | 338 | 90 | 26,6 % |
| 2021 | 274 | 39 | 14,2 % |
| 2022 | 296 | 49 | 16,6 % |
| 2023 | 272 | 55 | 20,2 % |
| 2024 | 412 | 103 | 25,0 % |
| 2025 | 741 | 412 | **55,6 %** |

Le filtre **mord réellement**, et de plus en plus : il rejette 6,6 % des candidats en 2019 et
**55,6 % en 2025**. Les chiffres reproduisent à la décimale près ceux du comptage du
2026-09-21 (note de faisabilité §2.1 : 6,5 % → 55,5 %). Toute mention de « R ≥ 1 » dans le
rapport client doit être accompagnée de cette table (table §5.5).

Pour mémoire, les refus en amont : **589** candidats refusés au contexte (§6.4), et sur
11 428 annulations, 4 742 `sweep`, 4 227 `zone`, 1 008 `hold`, 759 `r`, 84 `nsweep`, 17
`window`, 2 `narm`. L'automate a armé **12 412** fois pour **1 762** entrées.

---

## 6. Walk-forward annuel

`results/xau_x10/walkforward.json`.

| année | trades | E[R] | PF | PnL | part du net |

|---|---|---|---|---|---|
| 2019 | 155 | −0,1034 | 0,830 | −578,4 $ | 8,6 % |
| 2020 | 262 | −0,0505 | 0,855 | −744,9 $ | 11,1 % |
| 2021 | 252 | −0,2906 | 0,602 | −1 927,9 $ | **28,7 %** |
| 2022 | 245 | −0,2734 | 0,564 | −1 572,0 $ | 23,4 % |
| 2023 | 207 | −0,1545 | 0,757 | −604,0 $ | 9,0 % |
| 2024 | 319 | −0,2047 | 0,672 | −999,5 $ | 14,9 % |
| 2025 | 322 | −0,0901 | 0,864 | −295,2 $ | 4,4 % |


**Zéro année sur sept est positive** — le critère en demandait 60 %. En revanche la seconde
moitié du critère passe : aucune année ne porte plus de 50 % du net (maximum 28,7 %, en
2021).

La perte s'atténue sur la fin de période (−0,2906 R en 2021,
−0,0901 R en 2025) sans jamais changer de signe. Deux lectures cohabitent et aucune n'est tranchée par cette campagne : le prix de l'or a
été multiplié par plus de deux sur la période, donc un spread **constant** de 0,29 $ pèse
mécaniquement de moins en moins lourd en proportion ; et la règle R rejette de plus en plus de
candidats, ce qui écarte les trades les plus défavorables. Les deux poussent dans le même sens
et ne se séparent pas ici.

---

## 7. Ablations EMA50 H1 / VWAP / DXY

`results/xau_x10/ablations.json`. Sept combinaisons hors la spec, loguées **avant** mesure sous
`xau_x10:ablations:v1`, à la configuration `z1_a0.2_k1`.

| EMA | VWAP | DXY | trades | Sharpe | E[R] | PF | PnL |

|---|---|---|---|---|---|---|---|
| ✓ | ✓ | ✓ *(spec)* | 1 762 | **−1,941** | −0,1679 | 0,728 | −6 721,8 $ |
| ✓ | ✓ | ✗ | 1 764 | −2,001 | −0,1688 | 0,741 | −7 579,0 $ |
| ✓ | ✗ | ✓ | 1 832 | −1,948 | −0,1716 | 0,726 | −6 837,1 $ |
| ✓ | ✗ | ✗ | 1 838 | −2,088 | −0,1698 | 0,732 | −7 757,9 $ |
| ✗ | ✓ | ✓ | 1 942 | −2,118 | −0,1722 | 0,727 | −7 148,9 $ |
| ✗ | ✓ | ✗ | 1 956 | −2,168 | −0,1705 | 0,737 | −7 973,8 $ |
| ✗ | ✗ | ✓ | 2 146 | −2,149 | −0,1717 | 0,730 | −7 372,7 $ |
| ✗ | ✗ | ✗ | 2 168 | −2,209 | −0,1699 | 0,742 | −8 199,2 $ |


**Les trois filtres du mandat font ce qu'on leur demande, et cela ne suffit pas.** Chaque
retrait dégrade le Sharpe, et le retrait des trois le fait passer de −1,941 à −2,209. Le
classement est parfaitement monotone : plus on enlève de contexte, plus on perd.

Deux précisions qui comptent pour le rapport client :

- **`use_dxy = False` ne change presque pas le nombre de trades** (1 762 contre
  1 764) ni l'espérance en R (−0,1679 contre −0,1688), surtout
  le Sharpe et le PnL. C'est la vérification que le DXY
  ne filtre rien : il ne touche que la taille (§6.4 usage 3, §11). Le garde-fou « le DXY filtre
  sans bloquer » est donc implémenté comme le mandat le demande, et son effet mesuré est
  d'économiser **−857,2 $** de perte sur sept ans.
- **Retirer l'EMA50 H1 ouvre plus de trades que retirer le VWAP** (+180 contre
  +70) : les deux
  moitiés du filtre de contexte des breakouts ne mordent pas également.

Conclusion honnête de cette section : le contexte est utile, mais il corrige un signal dont
l'espérance brute est trop négative pour être rattrapée. On n'améliore pas une espérance de
−0,17 R en ajoutant des filtres qui la ramènent à −0,17 R.

---

## 8. Sensibilité aux coûts

`results/xau_x10/cost_sensitivity.json`. Le slippage est modélisé en l'ajoutant au spread à
raison de **2 × slippage** par aller-retour (spec §12 : il s'applique aux entrées *et* aux
sorties).

| spread | slippage / côté | spread effectif | trades | E[R] | PF | Sharpe |

|---|---|---|---|---|---|---|
| ×1 | 0,00 $ | 0,290 $ | 1 762 | −0,1679 | 0,728 | −1,941 |
| ×1 | 0,05 $ | 0,390 $ | 1 742 | −0,1955 | 0,685 | −2,295 |
| ×1 | 0,10 $ | 0,490 $ | 1 725 | −0,2191 | 0,654 | −2,584 |
| **×1,5** | **0,00 $** | **0,435 $** | 1 737 | **−0,2078** | 0,670 | −2,441 |
| ×1,5 | 0,05 $ | 0,535 $ | 1 706 | −0,2357 | 0,639 | −2,744 |
| ×1,5 | 0,10 $ | 0,635 $ | 1 680 | −0,2576 | 0,610 | −2,995 |
| **×2** | **0,00 $** | **0,580 $** | 1 700 | **−0,2440** | 0,630 | −2,807 |
| ×2 | 0,05 $ | 0,680 $ | 1 662 | −0,2712 | 0,594 | −3,162 |
| ×2 | 0,10 $ | 0,780 $ | 1 626 | −0,2975 | 0,549 | −3,543 |


Le critère 7 de la table (spread ×1,5, espérance > 0) échoue à **−0,2078 R**. La
sensibilité ×2, rendue obligatoire par le §1.1, échoue à **−0,2440 R**.

La sensibilité est forte et régulière : **+0,10 $ de coût par aller-retour coûte environ
0,025 R d'espérance**. Le coût est bien le premier terme de l'équation, comme la spec §12
l'annonçait. Il ne l'explique pas à lui seul : à **coût nul**, l'espérance resterait négative —
l'écart entre l'espérance mesurée à 0,29 $ (−0,160 R) et la pente ci-dessus place
l'extrapolation à coût nul autour de −0,09 R, toujours du mauvais côté de zéro. Cette
extrapolation est linéaire et indicative ; elle n'a pas été mesurée, car mesurer à spread nul
n'est pas un scénario de marché.

**La grille entière au spread ×2** (descriptif, aucune re-sélection — table §1.1) : le pic
descend de −1,009 à **−1,790**, et le nombre de configurations passant le critère plateau reste
**0**. Le plateau ne survit pas au doublement du coût, mais il n'existait pas non plus au coût
de base : il n'y a rien à faire survivre.

---

## 9. Limites de cette mesure — à publier telles quelles

1. **Le spread est constant à 0,29 $.** `costs_xau_intraday.yml` n'existe pas : le terminal MT5
   ne joint plus le serveur de démo depuis le 2026-08-05 et le spread historique par heure ×
   année n'a pas pu être mesuré. Le constant est trop cher en session liquide et trop bon
   marché en heures creuses, et il pèse mécaniquement de moins en moins lourd à mesure que le
   prix de l'or monte (§6). Repli décidé et gelé **avant** la campagne (table §1.1), assorti de
   la sensibilité ×2 obligatoire du §8. Si le spread mesuré devient disponible, il servira de
   sensibilité supplémentaire publiée, **jamais** à re-sélectionner.
2. **`a_min` a un double rôle** (spec annexe A.2) : il règle à la fois l'armement (§7.1) et le
   flip du reversal (§7.3). L'axe le plus discriminant de la grille (§3) n'est donc pas
   interprétable scénario par scénario — on ne peut pas dire si les `a_min` élevés aident
   les breakouts, les reversals, ou simplement parce qu'ils tradent moins.
3. **`k_s` n'agit que sur les breakouts.** Le stop du reversal vaut `extrême + 0,5 ATR`,
   constante gelée. Un tiers de la grille ne concerne que la moitié des scénarios, et c'est
   l'axe qui ne classe rien au §3.
4. **Warmup EMA50 H1.** L'EMA50 H1 exige 50 barres H1 closes : elle est indéfinie sur les
   ~2 premiers jours de 2019, où `ctx_ema = 0` et où **aucun breakout n'est possible**. Les
   reversals, eux, ne sont pas empêchés. Effet borné à deux séances sur 1 824.
5. **16,2 % des trades sortent à 48 barres** (§5.5), avec une espérance positive de +0,75 R.
   La durée maximale gelée coupe donc des positions qui gagnaient en moyenne ; ce n'est pas un
   argument pour l'allonger — ce serait un paramètre choisi après mesure — mais c'est un point
   de conception à rouvrir si le mandat l'est.
6. **Les rendements sont des rendements de séance d'une stratégie plate la plupart du temps.**
   1 723 trades sur 1 824 séances, dont **692 séances sont entièrement plates** (1 132 portent
   au moins un trade) : 38 % de l'index est à zéro par construction, et les séances à trade
   unique dominent le reste. Le Sharpe annualisé à 252 sur cette base est une convention, pas une
   propriété ; il se compare à lui-même d'une configuration à l'autre, pas à celui d'une
   stratégie continue. Les séances plates sont **conservées** dans l'index, précisément pour
   ne pas gonfler artificiellement le ratio.
7. **La dernière séance de l'échantillon est retirée.** Les barres du soir du 2025-12-31
   appartiennent à une séance étiquetée 2026-01-01, dont la seconde moitié est dans la tranche
   gelée : 356 barres M1 sont écartées pour que l'index de sélection ne contienne aucune
   étiquette ≥ 2026-01-01. Choix conservateur, effet nul sur les conclusions.
8. **Le SPA publié est inexploitable** et le StepM a échoué (§4). Défaut d'orientation
   préexistant dans `src/framework/statistical_testing.py`, non corrigé par cette campagne.
   Aucun critère du §1 n'en dépend.
9. **La métrique du CPCV n'est pas celle du tableau §3** : Sharpe sur rendements M1 contre
   Sharpe sur rendements de séance. Les deux ne se comparent qu'en signe.

---

## 10. Lecture déclarée du 2026-09-21

La table de décision §1.1 impose de la reproduire ici. Les runs de calage du 2026-09-21 au
centre de la grille ont rendu des **comptages d'événements**, dont le nombre de sorties par
raison (STOP / TARGET / TIME / SESSION). Ce n'est pas un P&L, mais ce n'est pas neutre.

**Aucun paramètre gelé n'a été modifié à la suite de cette lecture.** Le centre de la grille est
l'une des 27 configurations et a été logué avec elles, sous `xau_x10:grid27:v1`. Il se trouve
que c'est aussi la configuration sur laquelle portent les décompositions du §5 — non pas à
cause de cette lecture, mais parce que le §3 de la table n'a retenu **aucune** configuration et
que le repli sur le centre géométrique était écrit dans le brief de campagne avant mesure.

---

## 11. Ce que cette phase ne fait pas

- Elle **ne lit pas** l'OOS 2026. Le holdout reste LOCKED et le §4 de la table de décision
  reste vide. Il devra être rempli et committé **avant** toute lecture, si lecture il y a.
- Elle **ne produit pas** de chiffre MT5. Le critère 8 reste en échec par non-mesure.
- Elle **ne rouvre pas** la grille, n'ajoute aucun filtre et ne re-sélectionne au vu d'aucune
  sensibilité de coût.
- Elle **n'épuise pas** le budget : 34 essais sur 40, réserve de 6 intacte. Cette réserve
  n'autorise pas à chercher une configuration gagnante — le §5.1 de la table l'interdit
  explicitement.

## 12. Reproduire

```
uv run python scripts/run_xau_x10_research.py --step all
```

Graines fixes (42), toutes les sorties dans `results/xau_x10/`. `git_head` et la date de
production sont inscrits dans `is_summary.json`.
