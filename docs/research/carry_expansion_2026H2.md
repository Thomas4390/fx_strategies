# Élargir l'univers par le portage — un gate ex ante, et sa réfutation partielle

> **Date** : 2026-07-28 · **Statut** : clos
> **Holdout state** : LOCKED (frozen from 2026-01-01 until Phase 25 / 2026-12-31).
> **Holdout touched by this phase** : NO. Tout ferme au 2025-12-31.

## 0. Ce que cet axe testait

Le cycle précédent a établi que **63 % du résultat de la jambe USD/JPY vient du portage
positif** — être payé pour attendre — contre un portage négatif qui coûte 28 % sur l'or.
Le corollaire testable : un moteur long-only lent, qui tient ses positions des mois, devrait
préférer les instruments qui le rémunèrent pendant l'attente.

Jusqu'ici cette thèse était une **attribution après coup**. Le catalogue broker permet d'en
faire une **prédiction avant mesure** : sélectionner les candidats sur le signe de leur
swap, puis regarder s'ils performent. C'est le seul moyen de la réfuter.

## 1. Le catalogue tranche deux dossiers pour zéro essai

`data/broker/symbols_catalog_2026-07-28.csv` — 272 symboles, archivés pour la première fois.

**Les EM sont éliminés, et pas pour la raison anticipée.** Le plan d'expansion FX de
2026-05-04 les gardait en réserve avec une réserve sur leur spread. Le vrai obstacle est
le **sens** du portage :

| instrument | swap long | swap short |
|---|---|---|
| EUR/TRY | **−13 656** | −2 399 |
| USD/TRY | **−7 190** | +828 |
| USD/ZAR | **−446** | −92 |
| USD/MXN | **−310** | +85 |

Sur ces paires, le carry se gagne en étant **short** le dollar. La sleeve est long-only :
elle paierait le différentiel au lieu de l'encaisser, ce qui contredit le mécanisme même
que le dossier revendique. Dossier clos, aucun backtest lancé.

**Le critère est rare, ce qui le rend informatif** : seuls **21 des 207 symboles éligibles**
ont un portage long positif. Trois crosses yen jamais screenés y figurent — AUD/JPY +6,03,
NZD/JPY +3,20, CAD/JPY +2,91 — à côté d'USD/JPY (+9,99) déjà en production.

## 2. Le modèle de coût contredisait la thèse — corrigé d'abord

Avant de tester quoi que ce soit, un défaut du pré-filtre a dû être réparé :
`ret_net = ret − swap × |exposition|` facturait le portage **en valeur absolue à tous les
instruments**. Le criblage prélevait donc sur USD/JPY un portage que le compte encaisse.

| instrument | Sharpe vbt avant | après | Δ |
|---|---|---|---|
| USD-JPY | 0,538 | **0,770** | +0,232 |
| GBP-JPY | 0,079 | **0,314** | +0,235 |
| instruments à portage négatif | inchangés | | 0 |

Le criblage **sous-estimait de ~0,23 de Sharpe exactement la classe d'instruments que la
thèse désigne** — un biais dirigé contre l'hypothèse qu'il servait à tester.

## 3. Le résultat : la thèse ne prédit pas

Pré-filtre vbt, configuration de production, historique Yahoo de ~22 ans, coûts au proxy
GBP-JPY (le plus large des crosses yen mesurés, choix pessimiste assumé) :

| candidat | portage long | Sharpe net | CAGR | maxDD | trades/an | corr or | verdict |
|---|---|---|---|---|---|---|---|
| CAD-JPY | +2,91 | **0,19** | +0,9 % | −76,7 % | 7,0 | −0,01 | PASS faible |
| AUD-JPY | +6,03 | **0,17** | −0,1 % | −82,6 % | 7,4 | +0,02 | PASS faible |
| NZD-JPY | +3,20 | **−0,01** | −6,4 % | −91,2 % | 6,7 | −0,01 | **KILL_NEG_EDGE** |

Pour situer : le gate de classement MT5 est **0,436** — le Sharpe de XAG-USD, l'instrument
le plus faible réellement promu — et USD/JPY est à 0,73 sur le même pré-filtre.

**Aucun des trois candidats désignés par le portage n'en approche.** Et le rang interne
contredit le critère : le plus fort portage des trois (AUD/JPY, +6,03) donne le Sharpe le
plus faible des deux qui passent.

### Ce que cela réfute, et ce que cela ne réfute pas

**Réfuté** : « portage long positif » n'est **pas** un critère de sélection d'instrument.
La rareté du critère laissait espérer qu'il isolait quelque chose ; il n'isole rien
d'exploitable ici.

**Non réfuté** : l'attribution du cycle précédent reste valide. Le portage explique bien
63 % du P&L réalisé sur USD/JPY. Mais expliquer un P&L *ex post* et prédire une performance
*ex ante* sont deux choses différentes — et c'est précisément l'écart que ce test met en
évidence. USD/JPY ne performe pas *parce qu'il* porte : il porte **et** il tend, et c'est
la tendance que le moteur capture.

**Décision** : aucun run MT5 dépensé sur ces trois candidats. Un pré-filtre à 0,17-0,19 ne
peut pas produire un classement MT5 à 0,436 — sur toutes les jambes mesurées, le tester est
**inférieur** au vbt (or 0,74 → 0,73 ; argent 1,05 → 0,44). Le pré-filtre a fait ce pour
quoi il existe : tuer avant de dépenser.

## 4. XBRUSD — le dossier énergie se ferme sur une cause d'exécution

La cellule `XBRUSD` du classement MT5 était en `error=exit=1` depuis le cycle précédent,
jamais diagnostiquée. Rejouée : **elle échoue toujours, et la cause n'est pas celle qu'on
supposait**. Ce n'est pas un trou d'historique — le run tourne, 3,6 millions de ticks sont
générés — c'est que la sleeve **ne peut pas exécuter** :

```
[Gold_Momentum][ERROR] Entry XBRUSD failed: retcode=10018   × 162
```

`10018` = `TRADE_RETCODE_MARKET_CLOSED`. La sleeve décide à la borne de séance de l'EA,
21:00 UTC, heure à laquelle le CFD Brent ne cote plus. **162 tentatives d'entrée, zéro
exécutée** sur trois ans.

C'est un rejet de même nature que JPN225 (notionnel plafonné) : structurel au couple
instrument × moment de décision, pas une question d'edge. Le classement porte désormais
`market_closed_at_decision (retcode=10018 ×162)` au lieu d'une cellule vide.

Condition de réouverture, à écrire pour ne pas re-plaider : un instrument dont la séance
ferme avant 21:00 UTC exigerait de **décaler la borne de décision de la sleeve**, ce qui
changerait le signal de tous les autres instruments. Le rapport coût/bénéfice n'est pas
favorable pour un candidat dont le pré-filtre vbt vaut 0,12.

## 5. Bilan de l'axe

| dossier | issue | coût |
|---|---|---|
| EM (MXN, ZAR, TRY, CNH) | **clos** — portage du mauvais sens pour un moteur long-only | 0 essai |
| Crosses yen à portage positif | **clos** — le critère ne prédit pas (0,17-0,19 contre un gate à 0,436) | 3 essais |
| XBRUSD / énergie | **clos** — non exécutable à la borne de décision | 0 essai (rejeu) |
| Modèle de coût | **corrigé** — le portage est désormais signé | 0 essai |

Registre : `tsmom_universe` reste à **21 configurations distinctes** — l'univers passe de 21
à 24 instruments, mais c'est le même espace de configurations (une seule config, appliquée
à chaque instrument), et le `config_key` le reflète.

Le plafond structurel reste inchangé : `FX_GOLD_MAX_SYMBOLS = 8`, trois occupés, **cinq
promotions possibles au maximum** — et cet axe n'en a produit aucune.
