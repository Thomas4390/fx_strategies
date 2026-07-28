# XS momentum long-short : faisabilité avant construction

> **Date** : 2026-07-28 · **Statut** : clos — **ne pas construire**, condition de réouverture écrite
> **Holdout state** : LOCKED. **Holdout touched by this phase** : NO.
> **Essais consommés** : 0 — cette note ne calcule rien de neuf, elle confronte des mesures existantes.

## 0. Pourquoi cette note existe avant le code

Le XS momentum long-short est le **seul signal réellement orthogonal** que le dossier ait
produit : corrélation **0,148** au panier TSMOM, contre 0,66 pour sa variante long-only,
0,79 pour Donchian et 0,97 pour le dual momentum. À ce titre il revient à chaque cycle.

Le construire coûterait des semaines : le MQL5 n'a **aucun chemin de rebalancement à poids
cibles**. `FxSleeveGoldMomentum` dimensionne instrument par instrument sur `sub_equity/n`
en ouvertures et fermetures discrètes, tandis que `xs_momentum.pipeline` tourne sur
`from_orders` + `size_type="targetpercent"` + `cash_sharing=True` + `leverage_mode="eager"`.
Ce n'est pas un portage, c'est un moteur d'exécution neuf.

**La question à trancher avant d'engager ce chantier est donc : le signal survivrait-il à la
chaîne d'exécution ?** Elle se répond avec ce qui est déjà mesuré.

## 1. Le candidat

`reports/research/xs_screen_2026H2.csv`, ligne `lb252_L3S3` — la seule variante orthogonale :

| grandeur | valeur |
|---|---|
| Sharpe net (vbt, 25,3 ans) | **0,281** |
| CAGR net | +3,99 % |
| maxDD | −57,1 % |
| trades | 1 004 sur 6 373 séances, soit **39,7 par an** |
| turnover annuel | 14,45 |
| corrélation au panier TSMOM | **0,148** |

Les cinq autres configurations de la grille sont soit corrélées (0,59-0,66), soit pires
(0,12 et −0,05). C'est donc ce point précis qu'il faudrait construire.

## 2. Le haircut vbt → MT5, mesuré sur trois jambes

Le dépôt n'a pas à extrapoler : la parité est mesurée sur les trois instruments en
production, sleeve isolée, configuration épinglée.

| jambe | Sharpe vbt | Sharpe MT5 | **haircut** | trades/an |
|---|---|---|---|---|
| USD-JPY | +0,02 | −0,18 | **0,20** | 9,7 |
| XAU-USD (référence du cycle) | 1,08 | 0,73 | **0,35** | 5,8 |
| XAG-USD | 1,05 | 0,39 | **0,66** | 5,2 |

Le haircut n'est jamais inférieur à **0,20**, sur des moteurs **quatre à huit fois plus
lents** que le candidat. Son origine est documentée et non modélisable côté recherche :
dimensionnement en lots, levier non décalé, borne de décision 21:00 UTC contre 17:00 New
York, interpolation OHLC M1.

## 3. Et la fréquence l'aggrave — c'est le résultat le plus robuste du dépôt

`reports/mt5/gold_sweep.csv`, même instrument, même moteur, seule la grille de lookbacks
change :

| lookbacks | trades | trades/an | Sharpe MT5 |
|---|---|---|---|
| 40/60/120/250 | 31 | 5,8 | **0,873** |
| 20/40/80 | 41 | 7,7 | 0,800 |
| 30/60/120 | 40 | 7,5 | 0,658 |
| 15/30/60 | 66 | 12,4 | 0,565 |
| 10/20/40 | 76 | 14,3 | **0,433** |

**Multiplier la fréquence par 2,5 coûte 0,44 de Sharpe MT5.** Le candidat tourne à
39,7 trades/an, soit **×6,8** la fréquence de la configuration de production.

## 4. Le calcul qui décide

Le critère fixé au plan : ne construire que si le **Sharpe net projeté dépasse 0,15**.

| hypothèse de haircut | justification | Sharpe projeté |
|---|---|---|
| 0,20 | le **minimum** jamais mesuré, sur un moteur 4× plus lent | **+0,08** |
| 0,35 | la médiane des trois jambes | **−0,07** |
| 0,66 | le maximum mesuré | **−0,38** |

**Aucune hypothèse ne franchit 0,15.** Même l'hypothèse la plus favorable — un haircut de
0,20, obtenu sur un moteur quatre fois plus lent, en ignorant la pénalité de fréquence —
laisse le candidat à 0,08, c'est-à-dire **sous le plancher de résolution de la chaîne de
mesure**. On ne saurait pas le distinguer de zéro même après l'avoir construit.

Pour franchir le critère, il faudrait un Sharpe vbt d'au moins **0,35** (avec le haircut le
plus favorable) ou **0,50** (avec le médian). Le candidat est à 0,281.

**Décision : ne pas construire.**

## 5. Deux réserves qui jouent contre cette conclusion — et pourquoi elles ne la renversent pas

**Le modèle de coût du screen XS a le même défaut que celui du TSMOM** : il facture
`swap × gross`, donc en valeur absolue, sur un livre **long-short**. Le corriger demanderait
de modéliser les deux jambes séparément — une position courte sur un instrument à portage
long positif paie généralement le swap short, souvent plus cher que ce que la longue
encaisse. L'effet net sur un livre 3/3 n'est pas signé a priori.

Ce chantier n'a pas été fait, et c'est délibéré : il faudrait qu'il rapporte **+0,07 de
Sharpe au minimum** (pour atteindre 0,35) rien que pour rendre la question ouverte, alors
que la correction équivalente sur le TSMOM long-only valait +0,23 sur un livre où *toutes*
les positions bénéficiaient du signe. Sur un livre équilibré, l'ordre de grandeur attendu
est bien plus faible.

**Le haircut pourrait être plus favorable sur un moteur à rebalancement.** Une partie du
haircut mesuré vient de la granularité des lots sur des positions discrètes ; un livre à
poids cibles pourrait en souffrir moins. Mais il souffrirait davantage de la fréquence, et
le §3 montre que c'est le terme dominant.

## 6. Condition de réouverture — datée, pour ne pas re-plaider à chaque cycle

Ce dossier ne sera rouvert que si **l'une** de ces deux conditions est remplie :

1. **Un chemin d'exécution à poids cibles apparaît dans le MQL5 pour une autre raison.** Le
   coût de construction disparaît alors, et il ne reste qu'à mesurer — un run, pas un
   chantier.
2. **L'univers dépasse nettement 14 instruments classés.** Une coupe transversale sur 6
   positions parmi 14 est étroite ; le pouvoir du signal croît avec la largeur de la coupe.
   L'axe portage n'a rien apporté (aucune promotion), donc cette condition n'est pas
   remplie aujourd'hui.

En dehors de ces deux cas, la réponse est celle-ci, et elle tient tant que le Sharpe vbt du
candidat reste sous 0,35.
