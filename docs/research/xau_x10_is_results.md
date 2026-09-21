# XAUUSD niveaux x10 : résultats in-sample 2019 → 2025

> **Date** : 2026-09-21 · **Statut** : mesuré — verdict in-sample **NE PAS DÉPLOYER**
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

**Le résultat est négatif, franchement et à toutes les coupes.** La spec (§0) et la table de
décision (§2) prévoyaient ce livrable ; il est rendu tel quel.

---

## 1. Verdict provisoire et table de décision

`results/xau_x10/is_summary.json`. Configuration analysée : **`z = 1 ; a_min = 0,2 ; k_s = 1`**
— le centre géométrique de la grille, **qui n'est pas une configuration retenue** (§2).

| # | critère (table §1) | seuil GO | mesuré | verdict |
|---|---|---|---|---|
| 1 | Trades in-sample 2019 → 2025 | ≥ 300 | **1 723** (4 scénarios ≥ 40) | **GO** |
| 2 | Espérance nette / profit factor | ≥ +0,10 R / ≥ 1,15 | **−0,1602 R** / **0,737** | **échec** |
| 3 | DSR (déflaté par 34 essais) | ≥ 0,95 | **0,000** | **échec** |
| 4 | PBO (CSCV, 27 × 1 824 jours) | ≤ 0,35 | **0,415** | **échec** |
| 5 | Plateau | ≥ 60 % voisins > 0 ; médiane ≥ 0,5 × pic | **0 %** ; médiane −1,603 contre seuil −0,589 | **échec** |
| 6 | Walk-forward annuel | ≥ 60 % d'années positives ; aucune > 50 % du net | **0 %** d'années positives ; max 34,8 % | **échec** |
| 7 | Spread ×1,5 | espérance > 0 | **−0,2012 R** | **échec** |
| 8 | MT5 modèle 1, 2022-11 → 2025-12 | espérance ≥ 0, même signe | **non mesuré** | **échec** |
| 9 | OOS 2026, lecture unique | n ≥ 30 | **non mesuré** | **échec** |

Un seul critère sur neuf passe, et c'est celui du nombre de trades. Six échecs portent une
mesure interprétable ; deux sont des non-mesures, qui valent échec et non « non applicable »
(table §1).

> **Verdict in-sample PROVISOIRE : NE PAS DÉPLOYER.**
> Il est provisoire au sens strict de la table §2 : le verdict définitif se prononce une seule
> fois, après la réconciliation MT5 (spec §13) et la lecture OOS unique. Rien dans ce qui suit
> n'autorise à anticiper cette lecture, et rien n'appelle à la faire : l'in-sample suffit à
> conclure sur le déploiement.

---

## 2. Sélection : aucune configuration retenue

`results/xau_x10/selection.json`, règle du §3 de la table appliquée à la lettre.

- **Pic de la grille** : `z = 0,5 ; a_min = 0,3 ; k_s = 1,5`, Sharpe net **−1,1776**.
- **Seuil de la médiane** : 0,5 × pic = **−0,5888**.
- **Configurations passant le critère plateau : 0 sur 27.**

Les 27 Sharpes nets sont négatifs, de −1,178 à −2,220. La part de voisins à Sharpe > 0 vaut
donc 0 % partout, contre 60 % exigés : aucune configuration ne passe, et le §3 est explicite —
**il n'y a pas de configuration retenue, et le verdict est NE PAS DÉPLOYER.** On ne se rabat
pas sur le pic.

Détail à ne pas manquer : quand le pic est négatif, `0,5 × pic` est **plus haut** que le pic,
donc la seconde moitié du critère devient plus dure, pas plus facile. La règle gelée ne se
dégrade pas en passoire sur un échantillon perdant — ici la médiane des voisins du centre vaut
−1,603 contre un seuil de −0,589, et échoue aussi.

**Plateau contre pic.** La table demande de publier les deux côte à côte. Pic
`z0.5_a0.3_k1.5` à −1,178 ; configuration analysée `z1_a0.2_k1` à −1,661 ; écart **−0,483**
de Sharpe. L'écart est réel, et il n'a aucune conséquence : les deux perdent.

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
| z0.5_a0.3_k1.5 | 0,5 | 0,3 | 1,5 | −1,178 | 1 020 | −0,1596 | 0,760 | −44,0 % |
| z0.5_a0.3_k1 | 0,5 | 0,3 | 1 | −1,226 | 1 070 | −0,1740 | 0,746 | −47,7 % |
| z0.5_a0.3_k0.75 | 0,5 | 0,3 | 0,75 | −1,251 | 1 093 | −0,1797 | 0,743 | −49,3 % |
| z0.5_a0.2_k1.5 | 0,5 | 0,2 | 1,5 | −1,331 | 1 299 | −0,1533 | 0,759 | −52,0 % |
| z0.5_a0.2_k0.75 | 0,5 | 0,2 | 0,75 | −1,344 | 1 372 | −0,1669 | 0,751 | −56,6 % |
| z0.5_a0.2_k1 | 0,5 | 0,2 | 1 | −1,381 | 1 351 | −0,1657 | 0,745 | −55,2 % |
| z1_a0.3_k1.5 | 1 | 0,3 | 1,5 | −1,394 | 1 333 | −0,1645 | 0,757 | −51,3 % |
| z1_a0.3_k0.75 | 1 | 0,3 | 0,75 | −1,431 | 1 432 | −0,1733 | 0,751 | −56,8 % |
| z1_a0.3_k1 | 1 | 0,3 | 1 | −1,469 | 1 397 | −0,1695 | 0,744 | −55,3 % |
| z1.5_a0.3_k0.75 | 1,5 | 0,3 | 0,75 | −1,538 | 1 618 | −0,1592 | 0,755 | −60,3 % |
| z1.5_a0.2_k0.75 | 1,5 | 0,2 | 0,75 | −1,555 | 1 930 | −0,1412 | 0,772 | −65,2 % |
| z1_a0.2_k1.5 | 1 | 0,2 | 1,5 | −1,598 | 1 653 | −0,1555 | 0,752 | −61,4 % |
| z1_a0.2_k0.75 | 1 | 0,2 | 0,75 | −1,607 | 1 760 | −0,1603 | 0,748 | −65,0 % |
| **z1_a0.2_k1** *(analysée)* | 1 | 0,2 | 1 | **−1,661** | **1 723** | **−0,1602** | **0,737** | **−63,7 %** |
| z1.5_a0.3_k1 | 1,5 | 0,3 | 1 | −1,666 | 1 579 | −0,1668 | 0,737 | −60,7 % |
| z1.5_a0.3_k1.5 | 1,5 | 0,3 | 1,5 | −1,668 | 1 498 | −0,1621 | 0,736 | −58,0 % |
| z1.5_a0.2_k1.5 | 1,5 | 0,2 | 1,5 | −1,669 | 1 809 | −0,1468 | 0,757 | −64,2 % |
| z1.5_a0.2_k1 | 1,5 | 0,2 | 1 | −1,707 | 1 891 | −0,1508 | 0,749 | −65,2 % |
| z1_a0.1_k0.75 | 1 | 0,1 | 0,75 | −1,735 | 2 058 | −0,1601 | 0,740 | −70,3 % |
| z0.5_a0.1_k1.5 | 0,5 | 0,1 | 1,5 | −1,849 | 1 607 | −0,1730 | 0,699 | −65,1 % |
| z1_a0.1_k1 | 1 | 0,1 | 1 | −1,855 | 2 017 | −0,1634 | 0,723 | −70,2 % |
| z1_a0.1_k1.5 | 1 | 0,1 | 1,5 | −1,861 | 1 948 | −0,1638 | 0,733 | −68,0 % |
| z0.5_a0.1_k0.75 | 0,5 | 0,1 | 0,75 | −1,884 | 1 678 | −0,1904 | 0,694 | −69,9 % |
| z0.5_a0.1_k1 | 0,5 | 0,1 | 1 | −1,905 | 1 655 | −0,1889 | 0,689 | −68,3 % |
| z1.5_a0.1_k0.75 | 1,5 | 0,1 | 0,75 | −2,009 | 2 222 | −0,1607 | 0,735 | −75,0 % |
| z1.5_a0.1_k1.5 | 1,5 | 0,1 | 1,5 | −2,196 | 2 092 | −0,1635 | 0,720 | −74,0 % |
| z1.5_a0.1_k1 | 1,5 | 0,1 | 1 | −2,220 | 2 172 | −0,1687 | 0,709 | −75,8 % |

Ce qu'on lit dans le classement, et qu'il faut dire sans le sur-interpréter : la grille est
**monotone en `a_min`**. Les neuf configurations à `a_min = 0,1` occupent le bas du tableau,
les neuf à `a_min = 0,3` le haut. Un `a_min` plus exigeant arme moins souvent, donc trade
moins, donc perd moins. Ce n'est pas un signal de bord : c'est la signature d'un coût fixe par
trade qui domine tout le reste. `z` agit dans le même sens et pour la même raison (une zone
plus large arme plus). `k_s` ne classe rien.

---

## 4. Robustesse

`results/xau_x10/robustness.json`. Déflaté par `distinct_trials("xau_x10") = 34`, graine 42,
2 000 réplicats bootstrap, blocs stationnaires de 20 séances.

| mesure | valeur |
|---|---|
| Sharpe net (séance, 252) | **−1,661** — IC 95 % bootstrap **[−2,439 ; −0,918]** |
| PSR (contre 0) | **0,000** |
| DSR (N = 34) | **0,000** — Sharpe attendu du maximum de 34 essais : **+0,590** |
| Haircut Sharpe (BHY, N = 34) | −0,194 ; p ajustée **1,000** |
| MinBTL | non calculable (défini pour un Sharpe observé positif seulement) |
| PBO (CSCV, 16 bins, 12 870 combinaisons) | **0,4155** |
| Rendement total | **−62,6 %** — IC 95 % [−74,8 % ; −43,5 %] |
| Max drawdown | **−63,7 %** — IC 95 % [−75,4 % ; −47,1 %] |
| Profit factor | 0,707 — IC 95 % [0,604 ; 0,824] |

**L'IC 95 % du Sharpe ne contient pas zéro** : la perte n'est pas un accident d'échantillonnage.

**Monte Carlo sur l'ordre des 1 723 trades** (2 000 permutations) : max drawdown observé
33,6 % du capital, médiane des permutations 33,1 %, p95 35,0 %, *z* de chance de séquence
**+0,59**. Autrement dit l'ordre réalisé n'a **pas** été malchanceux — la perte ne vient pas
d'un enchaînement défavorable, elle vient de l'espérance. Sous permutation, la durée médiane
passée sous l'eau est de **1 711 barres de trade sur 1 723**.

**Risque de ruine** (`ruin_report`, 2 000 chemins bootstrap de la longueur de l'échantillon) :
rendement annualisé **−13,26 %**, volatilité 7,98 %, MAR −0,208, P(ruine) 0,000 mais
**P(perte > 50 %) = 0,921**, terminal médian **0,370** du capital initial, drawdown p95 du
bootstrap **74,0 %**. Recouvrement le plus long : **1 793 séances sur 1 824** — la courbe
d'équité passe 98 % du temps sous son plus haut.

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
`z0.5_a0.3_k1.5` à **−1,129** ; moyenne OOS sur toute la grille **−1,556**. La métrique CPCV
est le Sharpe des rendements M1 du portefeuille (convention de `create_cv_pipeline`) et n'est
donc comparable au tableau du §3 **qu'en signe**.

Un PBO de 0,42 sur une grille qui perd partout ne dit pas « le surajustement explique la
perte » — il dit que le classement des 27 configurations entre le premier et le second demi-
échantillon n'a presque aucune persistance. Il n'y a rien à surajuster : il n'y a pas de signal.

---

## 5. Décompositions

`results/xau_x10/decompositions.json`, configuration `z1_a0.2_k1`, 1 723 trades, PnL net
**−6 258,14 $** sur 10 000 $ de capital initial, taux de réussite **31,17 %**.

### 5.1 Par scénario (les quatre sont au-dessus de 40 trades, donc commentables)

| scénario | n | E[R] | PF | PnL |
|---|---|---|---|---|
| BREAK_LONG | 377 | −0,0870 | 0,796 | −1 013,1 $ |
| BREAK_SHORT | 280 | −0,1488 | 0,814 | −772,1 $ |
| REV_LONG | 513 | −0,1554 | 0,755 | −1 711,3 $ |
| REV_SHORT | 553 | −0,2203 | 0,640 | −2 761,7 $ |

Les quatre perdent. Le moins mauvais est `BREAK_LONG` (−0,087 R), le pire `REV_SHORT`
(−0,220 R), qui porte à lui seul 44 % de la perte. La hiérarchie breakout > reversal est nette
et cohérente avec le fait que les breakouts sont les seuls à passer un filtre de contexte.

### 5.2 Par séance New York (heure de décision)

| séance | n | E[R] | PF | PnL |
|---|---|---|---|---|
| Asie 18:00-03:00 | 595 | −0,2567 | 0,602 | −3 605,4 $ |
| Londres 03:00-08:00 | 379 | −0,1367 | 0,841 | −910,6 $ |
| New York 08:00-17:00 | 749 | −0,0954 | 0,807 | −1 742,1 $ |

**La séance asiatique porte 58 % de la perte pour 35 % des trades.** C'est la coupe la plus
contrastée de toute la campagne. Elle ne suggère pas un filtre horaire à ajouter — ce serait un
paramètre choisi après mesure, donc interdit (table §5.1) — mais elle est la première chose à
regarder si le mandat est un jour rouvert.

### 5.3 Par régime d'ATR M5 et par `10 $ / ATR`

| tercile d'ATR M5 | n | E[R] | PF |
|---|---|---|---|
| ATR bas | 575 | −0,1898 | 0,725 |
| ATR moyen | 574 | −0,1545 | 0,730 |
| ATR haut | 574 | −0,1362 | 0,763 |

| 10 $ / ATR | n | E[R] | PF | PnL |
|---|---|---|---|---|
| < 3 | 145 | −0,2645 | 0,670 | −499,3 $ |
| 3 – 6 | 700 | −0,1644 | 0,693 | −2 642,9 $ |
| 6 – 10 | 560 | −0,0603 | 0,866 | −1 052,9 $ |
| > 10 | 318 | −0,2793 | 0,645 | −2 063,0 $ |

La tranche `6-10` est la moins mauvaise (−0,060 R), les deux extrêmes les pires. C'est
exactement la lecture de la note de faisabilité §1 : la grille de 10 $ ne représente pas la
même chose selon le régime, et les régimes où 10 $ vaut moins de 3 ATR (le prix traverse
plusieurs niveaux par barre) comme ceux où il en vaut plus de 10 (la cible est hors de portée
dans les 48 barres) sont tous deux défavorables. **Aucune des quatre tranches n'est positive.**

### 5.4 Par année — voir §6.

### 5.5 Par raison de sortie

| sortie | n | part | E[R] | PF | PnL |
|---|---|---|---|---|---|
| STOP | 1 072 | 62,2 % | **−1,0002** | 0,000 | −23 072,6 $ |
| TARGET | 265 | 15,4 % | **+2,0870** | ∞ | +10 980,1 $ |
| TIME (48 barres) | 279 | **16,2 %** | +0,7515 | 14,943 | +5 184,3 $ |
| SESSION (16:55) | 107 | 6,2 % | +0,3127 | 2,898 | +650,0 $ |

Cette table est l'arithmétique complète de l'échec, et elle est saine : le stop rend
**exactement −1,00 R**, ce qui vérifie le dimensionnement du §11 de bout en bout, et la cible
rend +2,09 R. Avec ces deux chiffres seuls, le point mort serait un taux de cible de 32,4 % ;
le taux réalisé est de **15,4 %**. Les sorties temporelles, positives en moyenne, ne comblent
pas l'écart.

La part de trades sortis à **48 barres est de 16,2 %** — l'ordre de grandeur annoncé au brief
de campagne (« 17 % si confirmé ») est confirmé. Durée de vie moyenne : 19,6 barres M5 ;
médiane 13.

### 5.6 Reversals étendus, DXY, retest

| coupe | n | E[R] | PF | PnL |
|---|---|---|---|---|
| reversal étendu (`extension_vwap = 1`) | 436 | −0,1282 | 0,779 | −1 204,6 $ |
| reversal non étendu | 630 | −0,2312 | 0,645 | −3 268,3 $ |
| DXY favorable | 957 | −0,1460 | 0,745 | −4 434,3 $ |
| DXY adverse | 766 | −0,1779 | 0,716 | −1 823,8 $ |
| avec retest | 770 | −0,1871 | 0,690 | −3 227,6 $ |
| sans retest | 953 | −0,1384 | 0,773 | −3 030,5 $ |

- **Extension VWAP** : les reversals étendus perdent près de deux fois moins que les autres.
  La grandeur est traçée et non bloquante par décision du mandat (spec §7.3) ; c'est la seule
  variable descriptive de la campagne qui sépare quelque chose, et elle reste du mauvais côté
  de zéro.
- **DXY** : le contexte favorable perd **plus en dollars** (−4 434 $ contre −1 824 $) tout en
  perdant moins par unité de risque (−0,146 R contre −0,178 R). Il n'y a pas de contradiction :
  le risque est divisé par deux quand le DXY est adverse (§11), donc les trades adverses
  pèsent moitié moins. Le garde-fou DXY fonctionne donc comme prévu — il réduit la perte — mais
  il ne discrimine pas : **l'espérance est négative des deux côtés**.
- **Retest** : le retest est **contre-prédictif** (−0,187 R avec, −0,138 R sans). Le mandat le
  citait comme confirmation ; la mesure ne le soutient pas. Il reste descriptif, comme la spec
  l'a figé.

### 5.7 Concentration

Net total **−6 258,14 $**, profit brut **17 528,63 $**.

Les parts du net ne sont pas publiables ici : le net total est négatif et « x % d'un nombre
négatif » n'a pas de sens. Les montants, eux, le sont.

| concentration | montant | part du **profit brut** |
|---|---|---|
| 5 meilleures séances | +1 261,48 $ | 7,2 % |
| meilleur mois (2019-08) | +325,97 $ | — |
| 10 meilleurs trades | +1 997,79 $ | 11,4 % |

**376 séances gagnantes contre 756 perdantes** sur 1 824. Aucune concentration pathologique :
la perte n'est pas l'accident de quelques jours, elle est diffuse, ce qui la rend d'autant plus
crédible. Le meilleur mois de sept ans rapporte **326 $**, soit 3,3 % du capital initial.

### 5.8 Taux de rejet de la règle R ≥ 1, par année (obligation spec §8 / table §5.5)

Dénominateur : les candidats **ayant atteint le test R** — un refus au contexte n'a pas de
`r_est` et n'y entre pas.

| année | candidats | rejetés | taux |
|---|---|---|---|
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
11 428 annulations, 4 742 `sweep`, 4 227 `zone`, 1 008 `hold`, 759 `r`, 84 `nsweep`, 17
`window`, 2 `narm`. L'automate a armé **12 433** fois pour **1 723** entrées.

---

## 6. Walk-forward annuel

`results/xau_x10/walkforward.json`.

| année | trades | E[R] | PF | PnL | part du net |
|---|---|---|---|---|---|
| 2019 | 149 | −0,0444 | 0,965 | −114,2 $ | 1,8 % |
| 2020 | 246 | −0,2740 | 0,580 | −2 178,9 $ | **34,8 %** |
| 2021 | 234 | −0,2196 | 0,634 | −1 465,2 $ | 23,4 % |
| 2022 | 247 | −0,2235 | 0,660 | −1 146,4 $ | 18,3 % |
| 2023 | 216 | −0,1479 | 0,764 | −548,2 $ | 8,8 % |
| 2024 | 307 | −0,1096 | 0,788 | −646,4 $ | 10,3 % |
| 2025 | 324 | −0,0920 | 0,939 | −158,9 $ | 2,5 % |

**Zéro année sur sept est positive** — le critère en demandait 60 %. En revanche la seconde
moitié du critère passe : aucune année ne porte plus de 50 % du net (maximum 34,8 %, en 2020).

La perte s'atténue avec le temps (−0,274 R en 2020, −0,092 R en 2025) sans jamais changer de
signe. Deux lectures cohabitent et aucune n'est tranchée par cette campagne : le prix de l'or a
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
| ✓ | ✓ | ✓ *(spec)* | 1 723 | **−1,661** | −0,1602 | 0,737 | −6 258,1 $ |
| ✓ | ✓ | ✗ | 1 723 | −1,786 | −0,1602 | 0,727 | −7 256,6 $ |
| ✓ | ✗ | ✓ | 1 803 | −1,781 | −0,1653 | 0,725 | −6 574,5 $ |
| ✓ | ✗ | ✗ | 1 804 | −1,949 | −0,1646 | 0,714 | −7 613,4 $ |
| ✗ | ✓ | ✓ | 1 909 | −1,859 | −0,1665 | 0,723 | −6 744,6 $ |
| ✗ | ✓ | ✗ | 1 911 | −2,028 | −0,1663 | 0,709 | −7 780,2 $ |
| ✗ | ✗ | ✓ | 2 102 | −2,029 | −0,1722 | 0,709 | −7 122,3 $ |
| ✗ | ✗ | ✗ | 2 117 | −2,285 | −0,1722 | 0,685 | −8 212,1 $ |

**Les trois filtres du mandat font ce qu'on leur demande, et cela ne suffit pas.** Chaque
retrait dégrade le Sharpe, et le retrait des trois le fait passer de −1,661 à −2,285. Le
classement est parfaitement monotone : plus on enlève de contexte, plus on perd.

Deux précisions qui comptent pour le rapport client :

- **`use_dxy = False` ne change pas le nombre de trades** (1 723 dans les deux cas) ni
  l'espérance en R (−0,1602), seulement le Sharpe et le PnL. C'est la vérification que le DXY
  ne filtre rien : il ne touche que la taille (§6.4 usage 3, §11). Le garde-fou « le DXY filtre
  sans bloquer » est donc implémenté comme le mandat le demande, et son effet mesuré est
  d'économiser **998,50 $** de perte sur sept ans.
- **Retirer l'EMA50 H1 ouvre plus de trades que retirer le VWAP** (+186 contre +80) : les deux
  moitiés du filtre de contexte des breakouts ne mordent pas également.

Conclusion honnête de cette section : le contexte est utile, mais il corrige un signal dont
l'espérance brute est trop négative pour être rattrapée. On n'améliore pas une espérance de
−0,17 R en ajoutant des filtres qui la ramènent à −0,16 R.

---

## 8. Sensibilité aux coûts

`results/xau_x10/cost_sensitivity.json`. Le slippage est modélisé en l'ajoutant au spread à
raison de **2 × slippage** par aller-retour (spec §12 : il s'applique aux entrées *et* aux
sorties).

| spread | slippage / côté | spread effectif | trades | E[R] | PF | Sharpe |
|---|---|---|---|---|---|---|
| ×1 | 0,00 $ | 0,290 $ | 1 723 | −0,1602 | 0,737 | −1,661 |
| ×1 | 0,05 $ | 0,390 $ | 1 710 | −0,1885 | 0,686 | −2,052 |
| ×1 | 0,10 $ | 0,490 $ | 1 689 | −0,2123 | 0,650 | −2,340 |
| **×1,5** | **0,00 $** | **0,435 $** | 1 704 | **−0,2012** | 0,673 | −2,170 |
| ×1,5 | 0,05 $ | 0,535 $ | 1 683 | −0,2237 | 0,632 | −2,557 |
| ×1,5 | 0,10 $ | 0,635 $ | 1 654 | −0,2585 | 0,581 | −2,966 |
| **×2** | **0,00 $** | **0,580 $** | 1 679 | **−0,2362** | 0,619 | −2,645 |
| ×2 | 0,05 $ | 0,680 $ | 1 650 | −0,2662 | 0,572 | −3,007 |
| ×2 | 0,10 $ | 0,780 $ | 1 622 | −0,2857 | 0,546 | −3,241 |

Le critère 7 de la table (spread ×1,5, espérance > 0) échoue à **−0,2012 R**. La sensibilité
×2, rendue obligatoire par le §1.1, échoue à **−0,2362 R**.

La sensibilité est forte et régulière : **+0,10 $ de coût par aller-retour coûte environ
0,025 R d'espérance**. Le coût est bien le premier terme de l'équation, comme la spec §12
l'annonçait. Il ne l'explique pas à lui seul : à **coût nul**, l'espérance resterait négative —
l'écart entre l'espérance mesurée à 0,29 $ (−0,160 R) et la pente ci-dessus place
l'extrapolation à coût nul autour de −0,09 R, toujours du mauvais côté de zéro. Cette
extrapolation est linéaire et indicative ; elle n'a pas été mesurée, car mesurer à spread nul
n'est pas un scénario de marché.

**La grille entière au spread ×2** (descriptif, aucune re-sélection — table §1.1) : le pic
descend de −1,178 à **−1,975**, et le nombre de configurations passant le critère plateau reste
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
   reversals, eux, ne sont pas empêchés. Effet borné à deux séances sur 1 824.
5. **16,2 % des trades sortent à 48 barres** (§5.5), avec une espérance positive de +0,75 R.
   La durée maximale gelée coupe donc des positions qui gagnaient en moyenne ; ce n'est pas un
   argument pour l'allonger — ce serait un paramètre choisi après mesure — mais c'est un point
   de conception à rouvrir si le mandat l'est.
6. **Les rendements sont des rendements de séance d'une stratégie plate la plupart du temps.**
   1 723 trades sur 1 824 séances, dont **692 séances sont entièrement plates** (1 132 portent
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
