# XAUUSD niveaux x10 : table de décision, gelée avant mesure

> **Date** : 2026-09-21 · **Statut** : gelé — aucune mesure n'existe encore
> **Holdout state** : LOCKED.
> **Holdout touched by this phase** : descriptif uniquement (ATR M5, fréquence de croisement
> des x10) ; **0 lecture de performance**.
> **Essais consommés** : 0.

Ce document fixe les seuils **avant** de mesurer quoi que ce soit. Son seul pouvoir est d'être
antérieur : `git log --oneline -- docs/research/xau_x10_decision_table.md` doit montrer un
commit de gel **antérieur** à la lecture hors échantillon. Un seuil modifié après une mesure
n'est plus un seuil, c'est une sélection.

Référence des définitions : `docs/specs/xau_x10_spec.md`. Budget d'essais : **40** pour la
famille `xau_x10` (27 grille + 7 ablations + 6 réserve), annexe A.3 de la spec.

## 1. Table des seuils GO

Tous les chiffres sont **nets de coûts**, au spread ×1 de `costs_xau_intraday.yml` sauf
mention contraire.

| critère | seuil GO |
|---|---|
| Trades in-sample 2019 → 2025-12-31 | **≥ 300** ; **≥ 40 par scénario** pour qu'un scénario soit commenté |
| Espérance nette / profit factor | **≥ +0,10 R** / **≥ 1,15** |
| DSR (déflaté par `distinct_trials("xau_x10")`) | **≥ 0,95** |
| PBO (CSCV, matrice 27 × jours) | **≤ 0,35** |
| Plateau | **≥ 60 %** des voisins à Sharpe net > 0 ; **médiane des voisins ≥ 0,5 × pic** |
| Walk-forward annuel | **≥ 60 %** d'années positives ; **aucune année > 50 %** du net total |
| Spread ×1,5 | espérance **> 0** |
| MT5 modèle 1, 2022-11-04 → 2025-12-31 | espérance **≥ 0**, **même signe** que Python |
| OOS 2026, lecture unique | **n ≥ 30**, sinon NON CONCLUANT |

Un critère non mesuré vaut **échec**, jamais « non applicable ».

### 1.1 Modèle de coût de la sélection — gelé le 2026-09-21, avant la campagne

`costs_xau_intraday.yml` n'existe pas : le spread historique se mesure par un run du tester
MT5 (`Inp_DumpBars`), et le terminal ne joint plus le serveur de démo du broker depuis au moins
le 2026-08-05. La campagne ne l'attend pas. Règle écrite **avant** toute mesure :

- **La sélection (grille, plateau, config retenue) se fait au spread constant de 0,29 $**,
  relevé au catalogue broker du 2026-07-28 (`source: catalog_snapshot`). C'est le « spread ×1 »
  de la table ci-dessus.
- La sensibilité **×2** devient **obligatoire** et s'ajoute au ×1,5 (repli prévu au plan).
- Si le spread mesuré devient disponible plus tard, il sert de **sensibilité supplémentaire
  publiée**, jamais à re-sélectionner : la config retenue ne change pas.
- Un spread constant est trop cher pour 2024-2026 (0,05 à 0,20 ATR) et trop bon marché en
  heures creuses ; ce biais est une limite à publier, pas un motif de rejouer la grille.

**Lecture déclarée.** Les runs de calage du 2026-09-21 au centre de la grille ont rendu des
comptages d'événements, dont le nombre de sorties par raison (STOP / TARGET / TIME / SESSION).
Ce n'est pas un P&L, mais ce n'est pas neutre. Aucun paramètre gelé n'a été modifié à la suite
de cette lecture ; le centre de la grille est l'une des 27 configurations et sera logué avec
elles.

### 1.2 Correction de datation du 2026-09-21

**Nature du défaut.** Le parquet exporté de LEAN date une barre M1 de sa **clôture** ; la
référence Python traitait cet horodatage comme un **début**. Sa grille M5 couvrait donc
`[t−1 min, t+4 min)` là où MT5 et le portage QC couvrent `[t, t+5 min)` — une minute d'écart sur
toutes les barres, prouvée par 316 tags QC sur 316 (`xau_x10_reconciliation.md` §11.6 bis). La
référence redate désormais à l'ouverture (`strategies.xau_x10.SOURCE_STAMP`), le panier DXY4
aussi, et la campagne a été **intégralement rejouée**.

**Aucun seuil, aucune règle, aucun paramètre gelé de ce document n'est modifié.** La grille reste
celle des 27 configurations, le budget celui de 34 essais (`distinct_trials("xau_x10") = 34`
après rejeu, les `config_key` dédoublonnant les deux sweeps), le spread de sélection celui de
0,29 $ du §1.1. Ce n'est pas une re-sélection : c'est la même mesure, faite sur la grille que la
spec décrit.

**Verdict inchangé : NE PAS DÉPLOYER.** Espérance −0,1602 → **−0,1679 R**, profit factor
0,737 → **0,728**, 1 723 → 1 762 trades. Un seul critère change de camp, et dans le sens
favorable : le PBO passe de 0,415 à **0,246** (GO). Huit critères en échec deviennent sept ;
il en faut zéro.

## 2. Les trois verdicts

Un seul verdict est prononcé, et il est prononcé une seule fois.

### DÉPLOIEMENT À BLANC (démo)

Les **neuf** critères du §1 sont GO, **et** la réconciliation du §13 de la spec tient
(≥ 95 % / ≥ 98 % / ≥ 70 % d'appariement, divergence non attribuée ≤ 5 % des trades), **et** la
lecture OOS 2026 ne contredit pas l'intervalle de non-contradiction gelé au §4.

Ce verdict n'autorise **pas** le capital réel. Il autorise un compte démo, une durée et une
revue datées, que le rapport doit nommer.

### NE PAS DÉPLOYER

**Au moins un** critère du §1 est en échec avec une mesure interprétable. Cas typiques
attendus au vu de la note de faisabilité : espérance nette négative après coûts, signe MT5
opposé au signe Python, PBO > 0,35, année unique portant plus de la moitié du net.

Ce verdict est un résultat complet, pas un abandon. Le rapport le documente en section 14b.

### NON CONCLUANT

La mesure existe mais n'est **pas interprétable**. Déclenché par l'un de :

1. **n < 30** trades sur la fenêtre OOS (critère 9) ;
2. régime OOS hors de la distribution in-sample : `10 $/ATR` médian de la fenêtre OOS
   **inférieur au minimum annuel in-sample** (3,8 en 2025, `xau_x10_feasibility_2026H2.md` §1) ;
3. réconciliation bloquée : divergence non attribuée sur > 5 % des trades (spec §13) ;
4. impossibilité de produire un chiffre MT5 (spread nul dans le tester, export échoué) sur un
   critère qui en dépend.

**Ce verdict est pré-écrit comme plausible** : au 2026-07-23, `10 $/ATR` vaut 1,8 contre 3,8
au minimum in-sample. Le déclencheur n°2 est donc attendu, et le constater ne sera pas une
surprise à commenter mais une prédiction à confirmer.

## 3. Config retenue : le centre du plateau, jamais le pic

La grille est un cube 3 × 3 × 3 (spec, annexe A.2). On indexe chaque configuration par
`(i_z, i_a, i_k) ∈ {0,1,2}³`, les indices suivant l'ordre croissant des valeurs :
`z ∈ {0,5 ; 1 ; 1,5}`, `a_min ∈ {0,1 ; 0,2 ; 0,3}`, `k_s ∈ {0,75 ; 1 ; 1,5}`.

**Voisin** — définition opératoire : `c'` est voisin de `c` si les deux triplets d'indices
diffèrent sur **exactement un axe**, et de **exactement 1**. Une configuration a donc 3 voisins
si elle est à un coin, 4 ou 5 sur une arête ou une face, et 6 au centre géométrique. La
configuration elle-même n'est jamais son propre voisin.

**Critère plateau** — une configuration `c` passe le plateau si :

```
part_positifs(c) := |{ c' voisin de c : sharpe_net(c') > 0 }| / |voisins(c)|  >=  0,60
mediane( sharpe_net(c') pour c' voisin de c )                                >=  0,50 * pic
avec pic := max sur les 27 configurations du sharpe net
```

**Centre du plateau** — parmi les seules configurations qui passent le critère plateau, la
config retenue est celle qui **maximise la médiane du Sharpe net de ses voisins**. Le Sharpe
propre de la configuration n'entre pas dans ce classement : on choisit un point dont
l'*entourage* est bon, pas un point qui est bon.

Départage, dans cet ordre, pour que la règle soit déterministe :

1. la plus petite distance de Manhattan au centre géométrique `(1,1,1)` ;
2. le plus grand `k_s` (stop le plus large = le moins dépendant de la microstructure) ;
3. l'ordre lexicographique croissant de `(i_z, i_a, i_k)`.

**Si aucune configuration ne passe le critère plateau, il n'y a pas de configuration retenue**,
et le verdict est NE PAS DÉPLOYER. On ne se rabat pas sur le pic : un pic isolé dans une grille
de 27 points est la signature d'un surajustement, pas d'un réglage.

Le rapport publie côte à côte le pic et le centre retenu (annexe « plateau contre pic »), avec
l'écart de Sharpe entre les deux. Un centre très en dessous du pic est une information à
donner au client, pas une gêne à masquer.

## 4. Gel de la lecture OOS (à remplir et committer AVANT la lecture)

> Cette section est **vide par construction**. Elle doit être remplie, relue et **committée**
> avant que la moindre métrique de performance postérieure au 2026-01-01 soit calculée, quel
> que soit le moteur. Le hash de ce commit est reproduit dans l'annexe « table de décision »
> du rapport client. Remplir cette section après la lecture invalide la lecture.

**Configuration gelée**

- `z` = _à remplir_
- `a_min` = _à remplir_
- `k_s` = _à remplir_
- justification « centre du plateau » (indices, voisins, médiane) = _à remplir_
- pic de la grille, pour mémoire = _à remplir_

**Fenêtre de lecture**

- fenêtre commune aux trois moteurs = _à remplir_ (par défaut 2026-01-01 → 2026-03-31, limite
  des parquets FX du DXY)
- extension éventuelle jusqu'au 2026-07-24 = _à remplir_ (uniquement sous la règle
  « DXY indéfini = neutre » de la spec §6, décidée **ici**, avant lecture)
- nombre de trades attendu = _à remplir_

**Intervalles de non-contradiction** (l'OOS ne *valide* rien ; il peut seulement contredire)

- espérance nette en R : intervalle = _à remplir_
- profit factor : intervalle = _à remplir_
- taux de réussite par scénario : intervalle = _à remplir_
- part des sorties par `TIME` et `SESSION` : intervalle = _à remplir_

**Protocole**

- une lecture par moteur, **même session**, aucune reprise
- Python : `frozen_oos_slice` (`src/framework/holdout.py:93`), résultat marqué `FROZEN_OOS_RESULT`
- ligne ajoutée au journal de `docs/research/HOLDOUT_POLICY.md` **au moment de la lecture**
- aucune re-sélection, aucune nouvelle configuration, aucun nouvel essai après lecture

### 4.1 Décision du 2026-09-21 : la lecture OOS n'est pas effectuée

La campagne in-sample (`docs/research/xau_x10_is_results.md`, commit `74cfaeb`) rend un seul
critère sur neuf. **Aucune configuration ne passe le plateau : il n'y a rien à geler ci-dessus.**
La section 4 reste donc vide, et le holdout ≥ 2026-01-01 **n'est pas lu** pour cette stratégie.

- Le verdict est acquis in-sample : six critères mesurés échouent franchement. Un critère non
  mesuré vaut échec (§1) ; une lecture OOS ne peut donc rien sauver.
- Le holdout est un budget (`HOLDOUT_POLICY.md`). Le dépenser sur une stratégie déjà rejetée
  n'apprend rien et entame la tranche 2026 de l'or pour tout travail futur.
- Le régime 2026 (1,8 ATR M5 par niveau) est hors de la distribution in-sample. Un chiffre
  positif y serait un argument trompeur face à sept années négatives ; un chiffre négatif
  n'ajouterait rien.

Le critère « OOS 2026 » reste `non mesuré`, donc en échec, et le verdict est **NE PAS DÉPLOYER**.
Aucune ligne n'est ajoutée au journal de consommation du holdout : il n'y a pas eu de lecture.
Les mesures descriptives du 2026-09-21 (ATR, fréquence de croisement) restent les seules à
avoir touché des barres de 2026.

## 5. Ce que cette table interdit explicitement

1. Ajouter un axe à la grille après avoir vu les 27 résultats.
2. Remplacer la grille en dollars par une grille en ATR au vu du §1 de la note de faisabilité :
   ce serait une autre stratégie, donc une autre spec et un autre budget.
3. Relancer la grille sans `config_key` — le re-run compterait double au registre et
   dégraderait le DSR publié (`tests/test_trials_matches_notes.py`).
4. Lire l'OOS deux fois, sous quelque prétexte que ce soit, y compris « correction d'un bug ».
   Un bug découvert après lecture rend le résultat NON CONCLUANT ; il ne rend pas la lecture
   rejouable.
5. Mentionner `R ≥ 1` **sans** publier son taux de rejet annuel. Le comptage du 2026-09-21
   (faisabilité §2.1) montre que ce filtre rejette de 6,5 % (2019) à 55,5 % (2025) des
   candidats : il est actif, et le chiffre doit accompagner toute mention du garde-fou
   (spec §8).
