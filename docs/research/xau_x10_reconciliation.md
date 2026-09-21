# XAUUSD niveaux x10 : réconciliation événementielle Python ↔ QuantConnect ↔ MT5

> **Date** : 2026-09-21 · **Statut** : mesuré — **le portage QC est réparé (v4) et son backtest
> est sain**, mais l'appariement des entrées reste à **33 %** contre une cible de 98 % : le
> barreau 2 est cassé et **les deux moteurs ne lisent pas les mêmes minutes**. Aucun chiffre de
> performance QC ne doit être publié tant que ce point n'est pas fermé.
> **La branche MT5 est mesurée depuis le 2026-09-21** (§9) : appariement **98,1 %** dès que les
> deux moteurs lisent le même prix, divergence non attribuée **1,1 %**, et **critère 8 en
> ÉCHEC** — espérance MT5 **−0,1034 R**, du même signe que Python.
> **Holdout state** : LOCKED.
> **Holdout touched by this phase** : **NO** — aucune barre ≥ 2026-01-01 n'est entrée dans un
> calcul. Date maximale de tout index consommé : M1 `2025-12-31 16:58` (variante P1), et
> `2024-12-31` pour le backtest QC comme pour la variante P2.
> **Essais consommés** : **0** — aucune grille, aucun appel à `log_trials`. La configuration
> lue (`z = 1 ; a_min = 0,2 ; k_s = 1`) est le centre géométrique de la grille, déjà logé au
> registre par `xau_x10:grid27:v1` ; la relire pour réconcilier deux moteurs ne consomme pas
> d'essai.

Tous les chiffres de cette note sortent de `results/xau_x10/reconciliation_qc_2024.json`,
produit par `scripts/reconcile_x10_events.py`. Aucun n'est recalculé à la main.

```bash
uv run python scripts/reconcile_x10_events.py \
    --py-trades reports/qc_x10/py_trades_2024_P1.csv reports/qc_x10/py_trades_2024_P2.csv \
    --qc-orders reports/qc_x10/x10_calage_2024_centre_orders.json \
    --qc-stats  reports/qc_x10/x10_calage_2024_centre.json \
    --out       results/xau_x10/reconciliation_qc_2024.json
```

---

## 0 bis. Historique v1 → v4 (ajouté après correction du portage)

`docs/research/xau_x10_reconciliation.md` a d'abord été écrite sur le backtest **v1**. Le
portage a ensuite été corrigé deux fois. Les sections 1 à 10 décrivent le **v1** et restent
valables telles quelles — elles documentent des défauts réels et la méthode qui les a trouvés.
Les sections **11 et suivantes** portent le résultat courant. Le JSON courant est
`results/xau_x10/reconciliation_qc_2024.json` (v4) ; celui du v1 est archivé sous
`results/xau_x10/reconciliation_qc_2024_v1.json`.

| run | backtestId | ordres | rejets | position finale | rendement | appariement |
|---|---|---|---|---|---|---|
| **v1** | `ac7bcf55a96ed8e6d67ec506b45a2d5e` | 550 | 9 + 14 appels de marge | **−5 oz** | **−91,82 %** | 28,7 % |
| **v3** | `8f81be6c2582b44e0d2e58839cc4c490` | **20** | 1 | **−7 oz** | −21,71 %* | **0 %** |
| **v4** | `90d2b200b364fa7e871d3f07e668bda2` | **632** | **0** | **0 oz** | **−21,71 %** | **33,1 %** |
| v4 spread 0,29 | `947756097ee174d4a82c0e6ad25eccde` | 634 | 0 | 0 oz | −21,54 % | 33,4 % |

\* le v3 n'a tradé que du 2 au 12 janvier ; son rendement ne mesure rien.

Référence Python 2024, même capital : **−15,34 %** (P2, 305 trades).

---

## 0. Ce qu'il faut retenir

Le backtest QuantConnect perd **−91,8 %** sur 2024 là où la référence Python perd **−15,3 %**
sur le même millésime à capital égal. L'hypothèse de départ — « LEAN sort au marché *après*
avoir détecté le stop, donc remplit au-delà » — est **confirmée comme coût réel mais réfutée
comme explication** : elle vaut **−0,129 R par trade**, pas quatre-vingt-onze pour cent.

**81,5 % de la perte du compte QC (−7 476 USD sur −9 178 USD) n'appartient à aucun trade de la
stratégie.** Elle vient d'un inventaire que personne n'a ouvert : huit ordres de sortie ont été
**refusés par le broker** et le portage ne lit jamais le sort de ses ordres, si bien qu'il croit
plat un compte qui porte jusqu'à 63 onces d'or. Sur 541 remplissages de l'année, le compte n'est
réellement à plat que **deux fois**.

Et même cela ne se lit qu'avec des pincettes, parce que **l'échelle de §13 casse bien avant le
barreau 5** : seules **28,7 %** des entrées s'apparient (cible ≥ 98 %), et le `stop` théorique
diffère sur **88 appariements sur 88**. Les deux moteurs ne prennent pas les mêmes trades.

---

## 1. Méthode

Le seul canal de réconciliation exportable est le **journal d'ordres** : la trace §13 vit dans
l'ObjectStore, que l'API ne rend pas sur ce compte (`src/qc/xau_x10/README.md` §5). Chaque ordre
porte `scenario|level|ts_decision|stop|target|r_est` (`src/qc/xau_x10/main.py:211-224`).

**Reconstruction des trades QC.** Le portage n'a qu'une position à la fois (§9) et `_route`
n'émet qu'un ordre par événement : la parité entrée/sortie est une propriété du code. On la suit
littéralement — premier ordre stratégique = entrée, suivant = sa sortie — et **jamais le signe
des quantités**, qui ment dès qu'une sortie est refusée. Le quintuplet
`(scénario, niveau, stop, cible, r_est)` est recopié tel quel entre l'entrée et la sortie d'une
même position, ce qui donne un contrôle d'intégrité gratuit : **0 désaccord sur 268 positions**,
et **0 tag illisible sur 550 ordres**.

**Appariement.** Sur `(scénario, niveau, ts_decision ± 1 barre M5)`, un à un, au plus proche.
Les horodatages tombent **exactement** sur la même minute pour 72 des 88 appariements, donc la
tolérance ne fabrique rien.

**Décomposition du barreau 5.** Un télescopage : partant du PnL Python on substitue une à une la
taille, le prix d'entrée, le gap de sortie Python, la raison de sortie, le prix de sortie. La
somme des cinq postes vaut l'écart total **à 3,55 × 10⁻¹⁵ USD près** (champ
`decomposition_closure_max_abs`), donc il n'y a aucun résidu où cacher quoi que ce soit. Le
dénominateur commun est le risque Python : normaliser chaque côté par son propre risque casserait
le télescopage.

**Deux variantes Python**, toutes deux au centre de la grille, spread gelé 0,29 $ :

| | échantillon | capital | trades 2024 | équité finale |
|---|---|---|---|---|
| **P1** | 2019-01-01 → 2025-12-31, filtré `ts_decision ∈ 2024` | 10 000 en 2019 | **307** | 3 741,86 (fin 2025) |
| **P2** | 2024-01-01 → 2024-12-31 | 10 000 au 2024-01-01 | **305** | **8 466,29** (−15,3 %) |
| **QC** | 2024-01-01 → 2024-12-31 | 10 000 | **268** positions | **817,72** (−91,8 %) |

P1 est la campagne officielle ; P2 est la seule comparable à QC pour l'équité et donc pour les
lots. Les deux donnent le même verdict d'appariement (28,7 % contre 28,5 %).

---

## 2. Échelle de lecture (§13)

| barreau | objet | mesuré | verdict |
|---|---|---|---|
| 1 | barres M5 | horodatages de décision identiques à la minute sur 72/88 appariements | **présumé OK** |
| 2 | indicateurs | ATR M5 implicite QC/Python : médiane **0,9946**, étendue **[0,924 ; 1,073]** | **CASSÉ** |
| 3 | `ARM/BREAK/SWEEP/CANCEL` | non observable — la trace n'est pas exportable | **non vérifiable** |
| 4 | entrées `ENTRY` | appariement **28,7 %** (cible ≥ 98 %) ; `stop` hors tolérance **88/88** | **CASSÉ** |
| 5 | `EXIT`, PnL | mesuré et décomposé, **mais ininterprétable** au sens de §13 | lu à titre indicatif |

§13 est explicite : « un écart au barreau *N* rend les barreaux au-delà ininterprétables ».
Le barreau 5 est donc publié **sous réserve** — il décrit 86 trades que les deux moteurs ont
effectivement pris ensemble, pas la stratégie.

### 2.1 Barreau 4 — appariement des entrées

| | P1 | P2 |
|---|---|---|
| entrées Python | 307 | 305 |
| positions QC | 268 | 268 |
| appariées | **88** | **87** |
| taux Python → QC | **28,7 %** | **28,5 %** |
| taux QC → Python | **32,8 %** | **32,5 %** |

Causes candidates des orphelins (P1) :

| cause | orphelins Python | orphelins QC |
|---|---|---|
| `rung2_3_automaton_divergence` | 201 | 158 |
| `sample_edge_warmup` | 7 | 4 |
| `counterpart_position_open` (§9, position unique) | 5 | 9 |
| `other_candidate_same_window` | 4 | 2 |
| `near_miss_time` (> 1 barre, ≤ 4 barres) | 2 | 6 |
| `size_below_min` (§11) | — | 1 |

**359 entrées sur 575 (62,4 %) n'ont pas de cause d'exécution nommable** : ce ne sont ni des
décalages de fenêtre, ni des refus de taille, ni la règle de position unique. Les deux automates
divergent en amont de l'exécution, et le distance médiane à la décision la plus proche de l'autre
moteur est de **515 minutes** (342 minutes dans l'autre sens) — ce ne sont pas des
quasi-appariements, ce sont des décisions différentes. §13 bloque la lecture hors échantillon au-delà de 5 % de divergence non attribuée ;
on est à douze fois le seuil.

Le désaccord est **attribuable au barreau 2/3**, et la preuve tient en deux colonnes du tag.

### 2.2 Barreau 4 — les trois quantités théoriques

Sur les 88 appariements de P1, tolérance `TOL_SAME_DATA` (`rtol = atol = 1e-6`) :

| quantité | max \|Δ\| | q95 \|Δ\| | hors tolérance | verdict |
|---|---|---|---|---|
| `target` | **0,000000** | 0,000000 | **0 / 88** | **OK** |
| `stop` | **0,2286** | 0,1294 | **88 / 88** | **CASSÉ** |
| `r_est` | **2,2277** | 1,2405 | **88 / 88** | **CASSÉ** |

`target` est exact parce qu'il ne dépend que du niveau : `target = level ± 10`
(`x10_engine.py:859`, `x10_state.py:751`). Les deux autres dépendent de l'ATR et du spread, et
c'est exactement là que ça casse. Deux mesures séparent les deux causes :

**(a) L'ATR M5 diffère de quelques pour cent.** Le stop d'un breakout vaut
`level − d · k_s · atr` des deux côtés, à la ligne près (`x10_engine.py:860`,
`x10_state.py:752`), donc `|level − stop|` *est* l'ATR à `k_s = 1`. Sur les 36 breakouts
appariés, le ratio QC/Python a pour **médiane 0,9946** et pour étendue **[0,924 ; 1,073]**.
Ce n'est pas un écart de formule — la formule, la période (14), le *seeding* (récursif sur le
premier *true range*, jamais la moyenne simple) et l'index de barre sont **identiques** dans
`x10_kernels.py:56-100` et `x10_indicators.py:76-112`. Un écart dispersé et non constant sur un
ATR identiquement calculé ne peut venir que des **barres M5 elles-mêmes**.

**(b) Le spread réellement facturé par QC vaut 1,67 fois celui de la référence.** §8 pose
`R = (|cible − close| − h) / (|close − stop| + h)`. La référence y met `h = 0,145` constant
(`xau_x10.py:91` et `:213` → `x10_engine.py:442`) ; le portage y met la vraie demi-fourchette de
la dernière minute de la barre de décision (`x10_state.py:1065`). En inversant §8 —
`close + q·h = (cible + R·stop)/(R+1)` — à `close` commun, on retrouve la demi-fourchette de QC :

| | médiane | référence | rapport |
|---|---|---|---|
| demi-spread implicite QC, P1 | **0,242 $** | 0,145 $ | **×1,67** |
| demi-spread implicite QC, P2 | **0,260 $** | 0,145 $ | **×1,79** |

Soit un spread plein de l'ordre de **0,48 $ à 0,52 $** contre les **0,29 $** du catalogue broker.
`README.md` du portage (écart 5) annonçait que cela « peut faire basculer un candidat à la
frontière `R = 1` ». La mesure dit autre chose : avec `R_MIN = 1`, un demi-spread qui double
déplace la frontière sur **tout** le flux de candidats, pas sur les marginaux. C'est la première
cause documentée du désaccord d'entrées, et elle n'est **pas** un défaut du portage : c'est la
référence qui idéalise le coût, comme §12 l'assume (« repli si l'export MT5 échoue : 0,29 $
constant, + sensibilité ×2 obligatoire »). La sensibilité ×2 de §12 était la bonne intuition ;
elle est ici la réalité.

> ⚠️ Cette mesure suppose que les deux moteurs voient le **même `close` M5**. Le point (a) montre
> que ce n'est pas exactement vrai, donc le nombre absorbe aussi une part de l'écart de barres.
> Il est publié comme un ordre de grandeur, pas comme une égalité.

---

## 3. Barreau 5 — attribution, en unités de R

86 trades (P1) appariés **et** clos des deux côtés. R = PnL / risque engagé, dénominateur commun
au risque Python pour que les postes se somment.

| poste | somme (R) | **moyenne (R/trade)** |
|---|---|---|
| R réalisé **Python** | −19,79 | **−0,2301** |
| R réalisé **QC** | −20,70 | **−0,2407** |
| **écart total** | **+0,91** | **+0,0106** |
| (i) glissement à l'**entrée** | +3,005 | **+0,0349** |
| (ii) glissement à la **sortie** | **−11,124** | **−0,1293** |
| (iii) différence de **raison de sortie** | +4,340 | **+0,0505** |
| (iv) différence de **taille** (lots) | +5,484 | **+0,0638** |
| (v) gap de sortie Python (open déjà au-delà du niveau) | −0,794 | −0,0092 |
| **non attribué** | **−0,000** | **−0,0000** |

Fermeture de la décomposition : **max \|résidu\| = 3,553 × 10⁻¹⁵ USD** sur 86 trades. La part
non attribuée est **nulle par construction**, pas par arrondi.

Variante P2 (lots comparables, puisque l'équité l'est) :

| poste | moyenne (R/trade) |
|---|---|
| R Python / R QC / écart | −0,2501 / −0,1313 / **−0,1188** |
| (i) entrée | +0,0132 |
| (ii) **sortie** | **−0,0444** |
| (iii) raison | +0,0125 |
| (iv) taille | −0,0960 |
| (v) gap Python | −0,0041 |

Le poste (iv) change de signe entre P1 et P2 : c'est attendu et c'est la raison d'être des deux
variantes. P1 arrive en 2024 avec une équité de campagne (~10 000 → 3 742 sur 2019-2025) qui n'a
aucune raison de coïncider avec celle de QC ; P2 part du même capital le même jour. **Seul P2 est
lisible sur la taille**, et il dit que QC prend des positions plus petites que la référence — par
le plancher de 0,01 lot, voir §5.

### 3.1 Glissement, en dollars et par côté

| | moyenne | médiane | p05 | p95 | min | max |
|---|---|---|---|---|---|---|
| entrée (sens du trade) | **+0,036 $** | +0,105 | −1,202 | +1,247 | −2,930 | +3,090 |
| **sortie** (vs niveau théorique) | **+0,241 $** | +0,044 | −1,046 | **+1,981** | −3,690 | **+4,905** |

Signe positif = **défavorable** au trade. À l'entrée, le glissement est quasi centré : le retard
d'une minute entre la clôture de la barre de décision et le fill LEAN coûte autant qu'il rapporte.
À la sortie, il est **franchement biaisé, et c'est le cœur de l'hypothèse testée** : somme
+20,76 $ sur 86 trades, queue à +4,91 $ sur un seul.

### 3.2 Par raison de sortie Python (P1)

| raison | n | R Python | R QC | écart | dont glissement de sortie |
|---|---|---|---|---|---|
| `STOP` | 60 | **−1,0000** | −1,9446 | +0,9446 | −0,0037 |
| `TARGET` | 16 | +2,3120 | +5,7722 | −3,4601 | −0,1422 |
| `TIME` | 10 | +0,3218 | +0,3619 | −0,0402 | −0,8626 |

Deux lectures obligatoires ici.

**Le `R = −1,0000` exact de la référence sur 60 stops est une tautologie, pas un résultat.** Le
moteur de référence sort *au* niveau du stop, et le risque est *défini* comme la distance à ce
stop : le quotient vaut −1 par construction. Le **−1,94 R** de QC sur les mêmes 60 trades est, lui,
une mesure : LEAN sort au marché et le risque réellement engagé n'est pas celui qui a servi à
dimensionner. C'est la différence entre un stop modélisé et un stop exécuté, et elle est presque
du simple au double.

**L'accord sur la raison de sortie est de 87,2 %**, et cette raison est **inférée** côté QC (§6).

---

## 4. D'où viennent vraiment les −91,8 %

Comptabilité de trésorerie sur les 541 ordres remplis, indépendante de tout appariement :

| poste | USD | part |
|---|---|---|
| équité initiale | 10 000,00 | |
| équité finale publiée par QC | **817,72** | **−91,8 %** |
| PnL du compte reconstruit depuis les ordres | **−9 177,98** | |
| *bouclage contre le backtest* | *+4,30* | *0,05 %* |
| dont **allers-retours voulus** par la stratégie | **−1 702,27** | **18,5 %** |
| dont **inventaire résiduel** | **−7 475,71** | **81,5 %** |

L'espérance par aller-retour voulu est de **−6,57 USD**, soit un compte qui finirait autour de
**8 300 USD** — l'ordre de grandeur des **8 466 USD** de la référence P2. **Les quatre cinquièmes
de la perte n'ont rien à voir avec la stratégie.**

### 4.1 L'inventaire fantôme

| fait | valeur |
|---|---|
| ordres de sortie **refusés** | **8** |
| dont convertis en `MarketOnOpen` puis rejetés par le modèle OANDA | **8 / 8** |
| heure de New York de ces 8 refus | **17:04, les huit fois** |
| remplissages laissant le compte réellement à plat | **2 / 541** |
| position nette maximale | **63 oz** |
| position nette finale | **−5 oz** (13 122 USD de nominal sur 818 USD d'équité, ×16) |
| appels de marge déclenchés par LEAN | **14** |

Les huit refus, et leur message :

| ordre | horodatage UTC | New York | quantité non exécutée | position laissée ouverte |
|---|---|---|---|---|
| 2 | 2024-01-02 22:04 | **17:04** | +25 | **−25 oz** |
| 26 | 2024-01-22 22:04 | **17:04** | −16 | −9 oz |
| 58 | 2024-02-26 22:04 | **17:04** | −20 | +11 oz |
| 160 | 2024-04-28 21:04 | **17:04** | +23 | −12 oz |
| 316 | 2024-07-24 21:04 | **17:04** | +19 | −31 oz |
| 384 | 2024-08-26 21:04 | **17:04** | +9 | −40 oz |
| 492 | 2024-10-15 21:04 | **17:04** | −1 | −28 oz |
| 522 | 2024-11-12 22:04 | **17:04** | +1 | −5 oz |

> `BrokerageModel declared unable to submit order: Warning - Code: NotSupported - The
> OandaBrokerageModel does not support MarketOnOpen order type. Only supports
> [Limit,Market,StopMarket,StopLimit]`

**Les huit, sans exception, tombent à 17:04 New York** — dans la coupure quotidienne du CFD XAUUSD
(17:00-18:00, §2 et §10, « aucune position n'est tenue pendant la coupure quotidienne »). Le
mécanisme est déterministe : la sortie `SESSION` se résout sur la barre M5 de 16:55, les quatre
jambes FX continuent de livrer des minutes après 17:00 et font avancer l'horloge du feed, le
portage envoie alors un `market_order` sur un marché **fermé**, LEAN le convertit
automatiquement en `MarketOnOpen`, et OANDA refuse ce type. Le portage n'en sait rien.

Le premier refus suffit à raconter l'année : **le tout premier trade du backtest**, un short de
25 onces à 2 058,76 le 2 janvier, n'a jamais été fermé. L'or a fini 2024 au-dessus de 2 600. Ce
seul short non soldé vaut, à lui seul, davantage que le capital de départ.

La trajectoire d'équité le montre sans commentaire (fin de mois, et position nette portée) :

| | jan | fév | mar | avr | mai | jui | jul | aoû | sep | oct | nov | déc |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| équité | 11 020 | 10 728 | 13 066 | 14 256 | 13 438 | 13 301 | 11 772 | 8 289 | **2 677** | **251** | 654 | 783 |
| position (oz) | −9 | +11 | +11 | −12 | −12 | −12 | −31 | −40 | −29 | −4 | −5 | −5 |

Le compte **gagne** jusqu'en avril, puis se désintègre entre juillet et octobre — exactement
quand l'inventaire fantôme atteint −31 puis −40 onces pendant que l'or monte. Les 14 appels de
marge, de septembre à octobre, sont la conséquence, pas la cause.

### 4.2 Le même défaut, côté entrée

L'ordre **510** (2024-10-22 13:41) est une **entrée** refusée pour
`Insufficient buying power ... Initial Margin: -54.761, Free Margin: 6.87`. Le portage ne le lit
pas davantage : à la sortie de cette position inexistante, il envoie l'ordre **511**, qui **se
remplit** pour +1 once à 2 742,16. Le compte achète une once pour solder une position qu'il n'a
jamais eue.

C'est le même défaut que §4.1, dans l'autre sens, et c'est aussi le piège de la réconciliation :
un appariement qui « saute » une entrée refusée décale tous les ordres suivants d'un cran. Le
contrôle d'intégrité sur le quintuplet du tag l'a signalé (3 désaccords) avant correction, et
en signale **0** après — `tests/test_reconcile_x10_events.py::test_a_rejected_entry_still_consumes_its_exit_order`
verrouille le comportement.

---

## 5. Le dimensionnement n'est pas en cause

La question posée était : le risque engagé côté QC vaut-il bien 0,5 % de l'équité, et sinon
s'agit-il d'un bug de taille ou d'**unité de quantité** ? Risque engagé = `|fill − stop| × quantité`.

| | valeur |
|---|---|
| médiane | **0,3146 %** |
| moyenne | 0,3434 % |
| p95 | 0,5816 % |
| **maximum** | **0,7122 %** |
| trades au-dessus de 1,5 × la cible | **0 / 267** |
| trades sous 0,5 × la cible | 103 / 267 |

*(267 entrées réellement remplies : 268 positions moins l'entrée refusée de §4.2.)*

**Il n'y a pas de bug de taille ni d'unité.** Le risque est *inférieur* à la cible, jamais
« bien plus » : le maximum de l'année est à 0,71 %, soit 1,42 × la cible, et aucun trade ne
dépasse 1,5 ×. L'unité de quantité est correcte — LEAN reçoit des **onces**, et 25 onces valent
bien 0,25 lot de 100 (`main.py:202`, `quantity = event.q * event.lots * CONTRACT_SIZE`).

Les cinq trades au risque relatif le plus élevé de l'année :

| date | quantité | fill | stop | distance | risque | équité | % |
|---|---|---|---|---|---|---|---|
| 2024-01-08 07:46 | +20 oz | 2 032,11 | 2 028,336 | 3,774 | 75,47 $ | 10 598 | **0,712 %** |
| 2024-03-12 22:31 | −21 oz | 2 157,98 | 2 162,135 | 4,155 | 87,26 $ | 12 336 | **0,707 %** |
| 2024-04-15 15:01 | −11 oz | 2 346,46 | 2 355,416 | 8,956 | 98,51 $ | 14 204 | **0,694 %** |
| 2024-11-12 18:26 | −1 oz | 2 599,53 | 2 603,358 | 3,828 | 3,83 $ | 559 | **0,685 %** |
| 2024-02-12 14:16 | −20 oz | 2 017,93 | 2 021,508 | 3,578 | 71,56 $ | 10 932 | **0,655 %** |

Le dépassement résiduel (0,5 % → 0,71 %) s'explique entièrement par le **glissement d'entrée** :
§11 dimensionne sur `e_fill` **théorique** (`x10_state.py:443`, `quote.ask_open`/`bid_open` de la
barre de fill), et LEAN remplit une minute plus tard, à un prix qui peut être jusqu'à 3,09 $ plus
loin. La distance au stop réelle est alors plus grande que celle qui a servi à calculer les lots.

Le biais *vers le bas* — médiane à 0,31 %, 106 trades sous la moitié de la cible — est le
**plancher de 0,01 lot**, et il n'est pas un défaut : §11 impose un arrondi **inférieur** sans
epsilon (`x10_engine.py:293-302`, `x10_state.py:323-332`, tous deux alignés sur `NormalizeLots`
de `FxTradeHelpers.mqh:17-27`). À 800 USD d'équité, le budget de risque est de 4 USD ; sur une
distance au stop de 3,8 $, le calcul donne 0,0105 lot, arrondi à **0,01**. La quantification
mange 5 % du risque voulu, et bien davantage quand l'équité fond. C'est une conséquence
mécanique de l'effondrement décrit au §4, pas une cause.

**Effet de composition.** Le risque par trade restant un pourcentage de l'équité, la perte est
multiplicative et non additive : les −6,57 USD d'espérance par aller-retour voulu ne produisent
pas −91,8 % sur 259 trades, ils produisent environ −17 %. Les −91,8 % sont la composition d'une
espérance légèrement négative **et** d'un inventaire non couvert qui, lui, n'est pas dimensionné
du tout — les 25 onces du 2 janvier ne sont jamais redimensionnées quand l'équité tombe à 800 USD.

---

## 6. Verdict sur l'hypothèse

> « L'écart vient de l'exécution LEAN : sortie par ordre au marché **après** détection du stop
> sur barre minute, donc remplissage au-delà du stop. »

**PARTIELLE — confirmée comme coût, réfutée comme explication.**

**Confirmée** : le glissement de sortie est réel, mesuré, et systématiquement défavorable. Il
vaut **+0,241 $ par trade** en moyenne (contre +0,036 $ à l'entrée), avec une queue à +4,91 $.
En unités de R c'est **−0,129 R par trade** sur P1 (**−0,044 R** sur P2), le **poste le plus
lourd des cinq** de la décomposition. Sur les 60 stops appariés, il transforme un `R = −1,00`
modélisé en **−1,94 R** exécuté. C'est un coût de portage authentique, et il doit figurer au
rapport client comme tel.

**Réfutée** : ce coût explique **−0,13 R par trade**, soit de l'ordre de **−11 R sur l'année**.
L'écart à expliquer était de **7 476 USD** d'inventaire non voulu, sur un compte de 10 000 USD.
Trois ordres de grandeur séparent l'hypothèse du phénomène. Qui plus est, **l'écart net entre les
deux moteurs sur les 86 trades communs est de +0,0106 R par trade en faveur de Python** (P1) :
les postes se compensent presque, parce que le glissement de sortie est en grande partie annulé
par la différence de taille et par la différence de raison de sortie.

L'explication est ailleurs, et elle est structurelle : **le portage n'a aucun retour sur le sort
de ses ordres.** Voir §7.

---

## 7. Défauts du portage QC

Aucun fichier de `src/qc/xau_x10/` n'a été modifié. Ce qui suit est un **constat** et une
**proposition**.

### D1 — `main.py:207-209` : la sortie est réputée exécutée, jamais vérifiée

```python
elif event.event == EVENT_EXIT and self._open_qty != 0.0:
    self.market_order(self._xau, -self._open_qty, tag=self._tag(event))
    self._open_qty = 0.0
```

`self._open_qty = 0.0` est posé sans lire le statut de l'ordre. **Gravité : critique** — c'est
l'origine directe des **81,5 %** de perte hors stratégie. 8 occurrences en 2024.

### D2 — `main.py:206` et `:208` : `market_order` pendant la coupure quotidienne

`QCAlgorithm.market_order` convertit silencieusement l'ordre en `MarketOnOpen` lorsque la place
est fermée, et `OandaBrokerageModel` refuse ce type. Le CFD XAUUSD ferme de 17:00 à 18:00 New
York, mais les quatre jambes FX (`main.py:100-104`) continuent d'alimenter `on_data` et font
avancer l'horloge du feed : la sortie `SESSION` de 16:55 part donc systématiquement dans la
coupure. **Gravité : critique**, et **déterministe** — 8 refus sur 8 à 17:04 New York.

### D3 — `main.py:205-206` : même défaut à l'entrée

```python
self._open_qty = quantity
self.market_order(self._xau, quantity, tag=self._tag(event))
```

`_open_qty` est posé **avant** l'envoi. Une entrée refusée (ordre 510) laisse le portage croire
qu'il porte une position, et l'ordre de sortie correspondant (511) se remplit dans le vide.
**Gravité : haute.** 1 occurrence en 2024.

### D4 — aucun `on_order_event`

`main.py` ne définit que `initialize`, `on_data`, `_quote`, `_route`, `_tag` et
`on_end_of_algorithm`. Il n'existe **aucun** point où le portage apprend qu'un ordre a été
refusé, partiellement rempli ou annulé. D1, D2 et D3 sont trois symptômes de cette absence
unique. **Gravité : critique (cause racine).**

### D5 — aucune liquidation en fin d'algorithme

`on_end_of_algorithm` (`main.py:227-247`) écrit la trace et ne solde rien. Le backtest se termine
avec **−5 oz** ouvertes, soit 13 122 USD de nominal sur 817 USD d'équité. La spec est pourtant
normative (§10, « fin de fichier : la position est fermée d'office au dernier M1 disponible,
raison `SESSION` »). **Gravité : moyenne** (45,89 USD de non-réalisé ici, mais le principe est
faux).

### D6 — `main.py:173` : l'équité de dimensionnement inclut l'inventaire fantôme

```python
self._feed.machine.equity = float(self.portfolio.total_portfolio_value)
```

Correct en soi (§11), mais tant que D1-D3 vivent, `total_portfolio_value` porte le non-réalisé
d'une position que la stratégie ignore. Le dimensionnement de tous les trades postérieurs au
2 janvier est contaminé. **Gravité : haute tant que D1 n'est pas corrigé, nulle après.**

### D7 — `README.md` écart 5 sous-estimé

L'écart « spread réel contre spread constant » est annoncé comme pouvant « faire basculer un
candidat à la frontière `R = 1` ». Mesure : le demi-spread QC vaut **1,67 fois** celui de la
référence (0,242 $ contre 0,145 $). Ce n'est pas un effet de frontière, c'est un décalage de tout
le flux de candidats. **Gravité : documentaire**, mais c'est un terme de premier ordre du
désaccord d'entrées et il doit remonter au rapport client.

### D8 — écart de barres M5, non résolu

L'ATR M5 diffère de −7,6 % à +7,3 % (médiane −0,5 %) alors que le calcul est identique ligne à
ligne. Les barres consommées par les deux moteurs ne sont donc pas les mêmes, bien que le parquet
local (`data/XAU-USD_minute_qc.parquet`, export mid de ce compte) soit censé l'être. Candidats non
départagés : minutes où `_quote` renvoie `None` (cotation unilatérale, `main.py:180-189`), minutes
où l'or n'a pas de barre mais une jambe FX en a (`main.py:164-167`), différence entre le binnage
en flux de `x10_bars` et le `resample_ohlc` de la référence. **Gravité : haute** — c'est ce qui
casse le barreau 2, donc tout le reste. **Non résolu dans cette note** : le tag ne porte ni l'ATR
ni le `close`, et la trace §13 n'est pas exportable.

### Correctif proposé (non appliqué)

Trois points, dans cet ordre de priorité.

**1. Lire le sort des ordres.** Ajouter un `on_order_event` qui resynchronise l'état logique sur
l'état réel du portefeuille, et n'écrire `_open_qty` que sur un remplissage confirmé :

```python
def on_order_event(self, order_event) -> None:
    if order_event.status in (OrderStatus.INVALID, OrderStatus.CANCELED):
        self.log(f"x10 ORDER REJECTED {order_event.order_id}: {order_event.message}")
    # L'état logique ne peut pas diverger de l'état réel : on le recale, point.
    self._open_qty = float(self.portfolio[self._xau].quantity)
```

C'est le correctif de D1, D3, D4 et D6 en une fois, et il est robuste à toutes les causes de
refus, pas seulement à celle observée.

**2. Ne pas envoyer d'ordre sur une place fermée.** Dans `_route`, refuser l'envoi et le
**consigner comme un écart de trace** plutôt que le perdre :

```python
if not self.securities[self._xau].exchange.exchange_open:
    self.log(f"x10 EXIT DEFERRED (exchange closed) {self._tag(event)}")
    self._pending_exit = event          # soldé à la première minute cotée
    return
```

La spec (§10) dit qu'aucune position n'est tenue pendant la coupure ; si le broker ne le permet
pas, c'est un **écart à divulguer**, pas un ordre à laisser tomber. À décider avant correction :
solder à la dernière minute cotée *avant* 17:00, ou à la première minute *après* 18:00. La
première option respecte §10 et doit être préférée ; elle impose d'anticiper la clôture de
séance d'une barre.

**3. Solder en fin d'algorithme.** `self.liquidate(self._xau, tag="SESSION|end_of_algorithm")`
en tête de `on_end_of_algorithm`, correctif de D5.

**Aucun de ces correctifs ne touche au désaccord d'entrées (D8).** Même corrigés, les deux
moteurs ne prendront pas les mêmes trades tant que le barreau 2 n'est pas vert.

---

## 8. Ce qui n'a pas pu être vérifié

1. **Barreau 3** (`ARM` / `BREAK` / `SWEEP` / `CANCEL`) — **non observable**. Seuls les ordres
   sortent par l'API ; la trace §13 vit dans l'ObjectStore, réservé aux comptes Institutional. Les
   201 entrées Python et 158 entrées QC sans contrepartie ne peuvent pas être poussées jusqu'à
   l'événement d'automate qui les sépare.
2. **La raison de sortie QC est inférée**, pas lue : le tag porte `stop` et `target` mais jamais la
   raison retenue. L'inférence compare le prix rempli aux deux bornes (tolérance 35 % de
   `|cible − stop|`) et **s'accorde avec Python sur 87,2 %** des trades appariés. Les 12,8 %
   restants alimentent le poste (iii), qui est donc partiellement contaminé par l'incertitude de
   l'inférence.
3. **Le glissement de sortie des sorties `TIME` / `SESSION` n'est pas mesurable** : §9 les fait
   sortir à la clôture de la dernière M1, et aucun niveau théorique ne vit dans le tag. Le poste
   (ii) vaut zéro **par construction** sur les 5 trades appariés dont la sortie QC est inférée
   `TIME_OR_SESSION` (13 sur les 259 positions closes de l'année) — il n'est pas nul, il est
   inconnu. Le chiffre de **−0,129 R/trade** est donc un **minorant**.
4. **Le demi-spread implicite de QC suppose un `close` M5 commun aux deux moteurs.** Le point D8
   montre que cette hypothèse est approximative ; le ×1,67 est un ordre de grandeur.
5. **La cause racine de l'écart de barres M5 (D8) n'est pas identifiée.** Il faudrait la trace §13
   — ou, à défaut, un backtest QC qui journalise `atr`, `close` et le compte de minutes par barre
   M5. Aucun backtest n'a été lancé pour cette note.
6. **Le bouclage comptable laisse 4,30 USD** (0,05 % de la perte) entre le PnL reconstruit depuis
   les ordres et celui publié par QC, dû au prix de marque de la position finale, déduit de
   `Holdings / |position|`.

---

## 11. Le portage réparé (v3, puis v4)

### 11.1 v3 — un ulp gèle l'automate pendant onze mois et demi

Le premier correctif (commit `9d2da58`) a réglé D1-D5 et produit un backtest de **20 ordres**.
Neuf allers-retours propres du 2 au 12 janvier, puis **plus rien jusqu'au 31 décembre**.

**Cause, à la ligne.** §11 dimensionne en lots de 0,01 et multiplie par `CONTRACT_SIZE = 100`.
Ce produit n'est pas exact en binaire :

```
0.07 * 100.0 == 7.000000000000001     # et non 7.0
```

C'est **la seule** des dix tailles de lot du run qui rate — 0,38, 0,21, 0,25, 0,12, 0,09, 0,03,
0,13, 0,10 et 0,04 tombent toutes juste — et c'est celle de l'**ordre 19**, le dernier ordre du
backtest. Deux verrous se referment sur cette différence de 8,9 × 10⁻¹⁶ :

1. `main.py:238` (v3) `broker_blocked = held != self._broker.target_qty` → `-7.0` contre
   `-7.000000000000001` → vrai pour toujours → `x10_state.py:641` gèle l'automate, qui n'émet
   donc **jamais** la sortie ;
2. `x10_state.py:1231-1238` (v3) `delta = target - held` = −8,9e-16, jugé non nul → un
   `market_order(-8.9e-16)` que LEAN refuse **avant de créer l'ordre**, donc sans
   `on_order_event`, donc `inflight` reste vrai et plus aucun ordre ne part.

La liquidation finale (ordre 20, `MarketOnOpen`) est refusée par OANDA, exactement comme les
huit sorties du v1 : un ordre émis depuis `on_end_of_algorithm` arrive marché fermé.

### 11.2 v3 — l'horloge décalée de cinq heures

Défaut indépendant, trouvé dans les mêmes tags. `main.py:207/212` (v3) lisait
`minute_index(bar.time)` ; un `QuoteBar` LEAN porte un horodatage **naïf sur l'horloge de la
place** — New York pour un CFD OANDA — et `minute_index` lit un stamp naïf comme de l'UTC.
`ny_fields` reconvertissait ensuite une seconde fois.

Mesure, sur les quatre entrées v3 dont la barre tombe dans les trois premiers jours : en lisant
`bm` tel quel, le close du tag s'écarte du close de référence de **4 à 17 $** ; en ajoutant
**300 minutes**, l'écart tombe à **0,005 – 0,77 $**. L'hypothèse « `bm` est de l'UTC » est donc
réfutée et le décalage est exactement l'offset New York/UTC d'hiver. Conséquences : la clôture
de séance de 16:55 se déclenchait à **21:55 New York**, sur un marché fermé depuis 17:00 — c'est
ce que dit le tag `xr=SESSION` de l'ordre 16 — et la fenêtre interdite de §10 glissait d'autant.
Corollaire : `last_of_session` étant testé par `minute_of_day >= 16:58` **sans borne haute**,
il était vrai de 16:58 à 23:59 et **chaque minute du soir scellait son propre bin M5**.

### 11.3 Correctifs appliqués

| défaut | fichier:ligne (v4) | correctif |
|---|---|---|
| gel sur un ulp | `x10_state.py` `QTY_EPSILON`, `quantize_qty`, `BrokerSync.matches` | la cible est quantifiée, la comparaison passe par une tolérance, plus jamais `!=` |
| ordre sans réponse | `x10_state.py` `INFLIGHT_TIMEOUT`, `BrokerSync.order` | un ordre resté sans événement 5 minutes cotées est abandonné et réémis |
| horloge | `main.py:230` | `minute_index(self.utc_time) - 1` ; `quote_bar.time` n'est plus qu'une **sonde** qui rapporte l'offset |
| scellage du soir | `main.py:94`, `main.py:264-267` | fenêtre bornée `[16:58, 18:00)` |
| gel silencieux | `main.py:283-305` `_watchdog` | `self.error` explicite dès que le gel franchit une séance |
| solde final | `main.py:276-279` | la position est soldée **dans `on_data`**, dernière minute cotée de la dernière séance, marché ouvert ; `on_end_of_algorithm` ne garde qu'un filet qui crie s'il sert |

Vérification locale : `tests/test_qc_x10_parity.py` **58 tests verts** (53 + 5 nouveaux qui
rejouent la cause sans LEAN), parité des événements avec le noyau de référence intacte ;
`ruff` propre.

### 11.4 v4 — santé du backtest

Les quatre critères, avant toute lecture de performance :

| critère | mesure | verdict |
|---|---|---|
| nombre d'ordres ≈ 2 × 307 | **632** | **OK** |
| aucun ordre rejeté non retenté | **0** rejet, **0** appel de marge | **OK** |
| position nulle en fin de run | **0 oz**, `Holdings = $0.00` | **OK** |
| compte à plat entre deux trades | **316 / 632** remplissages, soit toutes les sorties | **OK** |

`health.healthy = true`. Le PnL reconstruit depuis les ordres vaut **−2 171,38 USD** et le
backtest publie **−2 171,38 USD** : le bouclage comptable est **exact à 0,00 USD** (il était à
4,30 USD sur le v1, où la position résiduelle devait être marquée).

L'inventaire fantôme a disparu : **0,0 %** de la perte lui revient, contre **81,5 %** au v1.

> ⚠️ Piège rencontré : lu trop tôt, `read_backtest` renvoie un instantané de mi-parcours —
> `statistics` entièrement à `null` et un `Net Profit` de −1 525,60 USD qui n'est pas le
> résultat final. Les chiffres ci-dessus sont ceux de la lecture complète (`Total Orders` = 632).
> Un backtest « completed » n'est pas forcément un backtest matérialisé.

### 11.5 Ce que le v4 mesure

| | Python P2 | QC v4 |
|---|---|---|
| trades | 305 | **316** |
| rendement 2024, capital 10 000 | **−15,34 %** | **−21,71 %** |
| espérance par trade apparié (R) | **−0,2379** | **−0,2071** |

Sur les **101 trades appariés et clos des deux côtés**, l'écart n'est que de **−0,0309 R par
trade**, décomposé exactement (résidu max **5,33 × 10⁻¹⁵ USD**, part non attribuée **−8,9 ×
10⁻¹⁶ R**) :

| poste | somme (R) | moyenne (R/trade) |
|---|---|---|
| (iv) taille | +0,071 | +0,0007 |
| (i) glissement d'entrée | +0,565 | +0,0056 |
| (v) gap de sortie Python | −0,298 | −0,0030 |
| (iii) raison de sortie | −0,376 | −0,0037 |
| **(ii) glissement de sortie** | **−3,079** | **−0,0305** |

Le glissement de sortie reste le poste dominant, mais il vaut désormais **−0,0305 R/trade**
contre −0,1293 au v1 : la barre 16:55 scellée à l'heure et l'horloge corrigée en retirent
l'essentiel. **Et il n'est plus un minorant** : le tag porte `xr=` et `xp=`, donc la raison et le
prix théorique de sortie sont **lus** et non plus devinés, y compris pour les sorties `TIME` et
`SESSION` que le v1 ne savait pas chiffrer. L'accord de raison de sortie passe de 87,2 %
(inférée) à **99,0 %** (lue). Glissement de sortie moyen : **+0,148 $** contre +0,241 $ au v1.

### 11.6 Barreau 2 — enfin observable, et cassé

Les champs `k=v` du tag v4 (`bm`, `c`, `atr`, `sp`) rendent le barreau 2 mesurable depuis
l'API. Sur les 101 entrées appariées, à `bm` identique :

| quantité | écart médian | écart max | identiques à 1e-6 |
|---|---|---|---|
| `bm` (barre de décision) | **0 minute** (83/101 exactement la même barre) | ±5 min | — |
| **close M5** | **0,265 $** | 2,575 $ | **1 / 101** |
| **ATR M5** | 0,041 $ (ratio médian **0,9943**) | 0,457 $ | **0 / 101** |
| spread | 0,110 $ (QC vaut **1,38 ×** la référence) | 0,400 $ | — |

**L'horloge est réparée** — plus aucun décalage systématique. **Les barres, non.**

Deux tests séparent les hypothèses, sur les 316 entrées du v4 :

1. **Décalage ?** Pour chaque entrée on cherche le décalage `k` (en barres M5) qui minimise
   `|close_référence(bm + 5k) − c|`. `k = 0` arrive en tête (119 / 316) et aucun autre `k` ne
   domine ; le meilleur `k` ne ramène l'écart médian que de 0,325 $ à 0,145 $. **Il n'y a pas de
   décalage de barres.**
2. **Minutes manquantes ?** Les **316 bins sur 316** contiennent leurs **cinq** minutes de
   référence. Et pourtant le close M5 de QC n'est celui d'**aucune** minute du bin : au mieux
   celle de rang +4 (144 / 316, la dernière), avec un résidu médian de **0,172 $** et seulement
   **3 / 316** exacts.

**Le parquet de référence est-il du mid, ou du bid/ask ?** L'écart médian de close (0,265 $) est
du même ordre que le demi-spread QC (0,225 $), ce qui ferait un coupable commode : un parquet en
**bid** donnerait `close_qc − close_py = +sp/2` partout, un parquet en **ask** `−sp/2`. Trois
tests, et ils disent tous non.

| test | attendu si bid/ask | mesuré |
|---|---|---|
| **signe** de `close_qc − close_py` | ~100 % d'un seul côté | **52,5 % positif / 46,5 % négatif** |
| médiane de `(close_qc − close_py) / (sp/2)` | **±1** | **+0,204** |
| **régression** sur `sp/2` : pente / **R²** | ±1 / élevé | **+2,35** / **R² = 0,016** |
| pente forcée par l'origine | ±1 | +0,258 |
| recaler de ∓`sp/2` puis retrouver une minute du bin (à 0,005 $) | ~316/316 | **9/316** (`−sp/2`), **5/316** (`+sp/2`), contre **6/316** brut |

Le signe est une pièce de monnaie, le R² est nul, et recaler d'un demi-spread **n'améliore
rien**. Le champ `price_basis.verdict` du JSON rend `mid_des_deux_cotes_donnees_differentes`.

Une quatrième preuve, indépendante des tags : dans `data/raw_qc/xauusd_minute.txt`, **35,8 %**
des prix portent une **quatrième décimale, toujours un 5** (`1281.9585`), le reste en ayant
trois. C'est exactement la signature de `(bid + ask) / 2` sur des cotations à trois décimales —
une série de bid *ou* d'ask pur n'aurait jamais de quatrième décimale. **Le parquet est bien du
mid.** Le modèle de coût de la référence (entrée à `close ± s/2` en supposant un mid, §9 et §12)
est donc **intact** : rien à corriger de ce côté.

**Conclusion : les bornes des bins sont bonnes, les minutes sont toutes là, les deux séries sont
du mid — et les prix de ces minutes diffèrent quand même.** Les deux moteurs ne lisent pas la même donnée minute. Ce n'est plus un
défaut de portage — c'est un écart entre `data/XAU-USD_minute_qc.parquet` et ce que le cloud
sert aujourd'hui pour `XAUUSD` OANDA. La prémisse de §13 (« QC et la référence partagent les
mêmes barres, donc ≥ 98 % ») est **fausse en l'état**, et la cible de 98 % est hors d'atteinte
tant que le parquet n'est pas ré-exporté.

### 11.7 Le spread n'explique pas la divergence

Backtest de contrôle `947756097ee174d4a82c0e6ad25eccde`, identique au v4 avec
`spread_override=0.29` — le spread constant de la référence :

| | v4, spread réel | v4, spread forcé à 0,29 |
|---|---|---|
| positions QC | 316 | 317 |
| **appariement Python → QC** | **33,1 %** | **33,4 %** |
| rendement | −21,71 % | −21,54 % |

Forcer le spread déplace l'appariement de **+0,3 point**. **L'écart 5 / D7 n'est donc pas la
cause** des deux tiers de divergence d'entrées : il coûte des dixièmes de point, les barres
coûtent le reste. C'était la question que ce backtest devait trancher, et elle est tranchée.

### 11.8 Dimensionnement, toujours conforme

Risque engagé médian **0,3038 %** de l'équité (cible 0,5 %), maximum **0,7797 %**, **aucun**
trade au-dessus de 1,5 × la cible. Même lecture qu'au §5 : pas de bug de taille ni d'unité, le
biais vers le bas est le plancher de 0,01 lot.

### 11.9 Contrat de sortie pour le rapport client

`results/xau_x10/reconciliation_qc_2024.json`, bloc racine `summary`, **noms gelés** :

```json
{
  "qc_backtest_id": "90d2b200b364fa7e871d3f07e668bda2",
  "qc_trades": 316,
  "py_trades": 305,
  "match_rate_py_to_qc": 0.33114754098360655,
  "match_rate_qc_to_py": 0.31962025316455694,
  "py_expectancy_r": -0.23791381534131012,
  "qc_expectancy_r": -0.207053646718123,
  "exit_slippage_r_per_trade": -0.030482829918751075,
  "qc_stop_realised_r": -0.9017737933883637,
  "qc_half_spread_usd": 0.2249992732913779,
  "unattributed_share": 0.5958132045088567,
  "qc_net_return_pct": -21.713799999999683,
  "healthy": true
}
```

`qc_stop_realised_r` mérite une lecture : le stop exécuté rend **−0,902 R** au v4, contre
**−1,944 R** au v1 et **−1,000 R** par construction côté référence. Le portage réparé sort
désormais *au-dessus* du stop en moyenne — la résolution minute par minute attrape le niveau
avant que le marché ne s'en éloigne — là où le v1 le franchissait de loin.

### 11.10 Ce qui reste ouvert

1. **Le barreau 2 (§11.6) : les minutes diffèrent, et ce n'est ni un décalage, ni une minute
   manquante, ni une base de prix bid/ask.** Prochaine action : ré-exporter
   `data/XAU-USD_minute_qc.parquet` depuis ce compte et rejouer la campagne. Tant que ce n'est
   pas fait, **l'appariement de 33 % n'est pas imputable au portage** et la lecture hors
   échantillon reste bloquée au titre de §13.

   ⚠️ **La provenance du parquet n'est pas reproductible.** `docs/specs/gold_momentum_spec.md`
   §1 et `docs/superpowers/plans/2026-07-25-reconciliation-vbt-mt5-qc.md` affirment tous deux
   qu'il « a été exporté depuis QuantConnect », mais **aucun script du dépôt ne le produit** et
   ni la méthode (`history()` en `QuoteBar` ? en `TradeBar` ?) ni la date ni la révision de
   données ne sont consignées. `data/MANIFEST.json` ne garde que le sha256, la taille et les
   bornes d'index (`2019-01-01 23:04` → `2026-07-24 03:59`). Le ré-export doit donc **d'abord**
   écrire le script qui le fabrique, sans quoi le prochain écart sera aussi peu diagnosticable
   que celui-ci.
2. **Le canal « chart » n'a rien rendu.** `read_backtest_chart` sur « X10Diag » renvoie une série
   **vide** alors que le graphique est bien déclaré dans le backtest — même limitation d'API que
   l'ObjectStore et les logs sur ce compte. La comparaison barre à barre a donc été faite par les
   tags `k=v`, sur les 316 vraies barres de décision plutôt que sur trois jours : meilleur canal,
   conclusion inchangée.
3. **Le barreau 3** (`ARM`/`BREAK`/`SWEEP`/`CANCEL`) reste non observable.
4. **Les 59,6 % d'entrées sans cause nommée** sont désormais attribuées collectivement au
   barreau 2 (§11.6), mais pas une par une : le tag ne voyage que sur les entrées **retenues**,
   jamais sur les candidats refusés.

---

## 12. Période complète 2019-2025

Backtest `1982926f999d70fced811f64ac0297b7` (« x10_reference_2019_2025_centre »), centre de
grille, capital 10 000, 2019-01-01 → 2025-12-31, **3 297 ordres**. Référence : la campagne
in-sample officielle au même centre et au spread gelé de 0,29 $ (**1 723 trades**, équité finale
**3 741,86 $**, dernière barre M1 lue `2025-12-31 16:58`, aucune donnée ≥ 2026). JSON :
`results/xau_x10/reconciliation_qc_full.json`.

### 12.1 Santé — **NON SAIN**, et c'est un défaut que j'ai introduit

| critère | mesure | verdict |
|---|---|---|
| ordres ≈ 2 × 1 723 | **3 297** (1 642 entrées + 1 655 sorties) | OK |
| aucun ordre rejeté non retenté | **0 rejet**, mais **13 sorties orphelines** | **NON** |
| position nulle en fin de run | **0 oz**, `Holdings $0.00` | OK |
| compte à plat entre deux trades | **1 643 / 3 297** au lieu de 1 648 | **NON** |

Bouclage comptable **exact à 0,00 USD** (−7 782,90 reconstruit = −7 782,90 publié).

**La cause est le garde-fou `INFLIGHT_TIMEOUT` que j'ai ajouté au §11.3.** Les ordres 428 à 441
sont **le même ordre de sortie réémis toutes les cinq minutes** — exactement la valeur du délai —
portant tous le tag identique `REV_SHORT|1630.000|2020-03-27 20:55:00|…|xr=SESSION`, et **tous
remplis**, pour +3 oz chacun. Le garde-fou ne sait pas distinguer « l'ordre n'a jamais existé »
de « l'ordre est rempli mais son événement ne m'est pas parvenu », et dans le second cas il
duplique la transaction.

Portée réelle, mesurée sur les trois backtests disponibles :

| run | entrées | sorties | ordres en double |
|---|---|---|---|
| 2024 v4 | 316 | 316 | **0** |
| 2024 v4, spread 0,29 | 317 | 317 | **0** |
| **2019-2025** | 1 642 | 1 655 | **13, tous le 2020-03-29** |

Treize ordres sur 3 297 (**0,4 %**), tous concentrés sur **une seule séance de reprise
dominicale** en sept ans, pour **−193,10 USD** — soit **2,5 %** de la perte. Le reste du journal
est sain. Ce n'est donc pas un défaut qui invalide la lecture, mais c'est un défaut, il est de
moi, et il est signalé plutôt que lissé.

**Correctif proposé** (non appliqué, aucun backtest disponible pour le valider) : ne jamais
réémettre à l'aveugle. Conserver le ticket rendu par `market_order`, et n'abandonner un ordre que
si `ticket.status` le dit mort ; à défaut, n'autoriser la réémission que si la position détenue
**n'a pas bougé** depuis l'envoi — ce qui suffit ici, puisque les treize doublons suivent chacun
un remplissage qui avait déjà changé la position.

> **Conséquence sur l'outil.** `reconstruct_qc_trades` appariait les ordres par simple
> alternance ; un seul ordre de trop décalait tout le reste du journal, et la première lecture de
> ce backtest annonçait un « inventaire résiduel » de 52,9 % qui n'existait pas. La
> reconstruction s'appuie désormais sur le tag lui-même (`bm=` = entrée, `xr=` = sortie) et met
> les sorties orphelines de côté au lieu de les apparier de force ; l'alternance reste le
> repli pour les tags v1. C'est la deuxième fois qu'un contrôle d'intégrité de cet outil rattrape
> un décalage d'un ordre — la première était l'entrée refusée du §4.2.

### 12.2 Année par année, en R

Le R est obligatoire ici : l'équité QC se compose de 10 000 à 2 217 sur sept ans, donc les lots
fondent et un PnL en dollars mélangerait le signal et la taille. Les espérances portent sur
**tous** les trades de l'année, pas seulement les appariés.

| année | trades Py | trades QC | **E[R] Python** | **E[R] QC** | appariement | entrées QC au plancher 0,01 lot |
|---|---|---|---|---|---|---|
| 2019 | 149 | 155 | **−0,0444** | **−0,1001** | 34,9 % | 0 |
| 2020 | 246 | 239 | **−0,2740** | **−0,3003** | 35,4 % | 0 |
| 2021 | 234 | 254 | **−0,2196** | **−0,3842** | 37,6 % | 1 |
| 2022 | 247 | 244 | **−0,2235** | **−0,2883** | 32,0 % | 9 |
| 2023 | 216 | 206 | **−0,1479** | **−0,1984** | 31,0 % | 13 |
| 2024 | 307 | 315 | **−0,1096** | **−0,2080** | 33,2 % | 63 |
| 2025 | 324 | 229 | **−0,0920** | **−0,3484** | 24,7 % | 111 |

**Les refus de taille de §11 ne sont pas observables** : le tag ne voyage que sur les entrées
**retenues**, jamais sur un `CANCEL` de raison `size`. La colonne publiée est donc le seul proxy
honnête — le nombre d'entrées déjà collées au plancher de 0,01 lot. Il passe de **0 en 2019-2020
à 111 en 2025**, soit près de la moitié des entrées de l'année : c'est la trace directe de
l'équité qui fond, et c'est aussi pourquoi 2025 ne compte que 229 trades QC contre 324 côté
Python, et pourquoi son appariement (24,7 %) décroche du plateau des autres années.

### 12.3 Attribution sur les 555 trades appariés

| poste | moyenne (R/trade) |
|---|---|
| R Python / R QC / **écart** | −0,1374 / −0,1709 / **+0,0335** |
| (iv) taille | −0,0160 |
| (i) glissement d'entrée | +0,0162 |
| (v) gap de sortie Python | −0,0014 |
| (iii) raison de sortie | +0,0252 |
| (ii) glissement de sortie | +0,0094 |

Part non attribuée **+3,55 × 10⁻¹⁵ R**. Sur sept ans, le glissement de sortie n'est plus le poste
dominant et change même de signe : le portage réparé sort en moyenne **au-dessus** du niveau
théorique. Les raisons de sortie sont toutes **lues** (`xr=`) — plus aucune inférence
`TIME_OR_SESSION` : STOP 1 076, TIME 268, TARGET 212, SESSION 86.

Le barreau 2 reste cassé dans les mêmes proportions qu'en 2024 : ATR implicite QC/Python de
médiane **0,9998** sur 249 breakouts, demi-spread QC **×1,56**, `stop` hors tolérance sur
536/555. L'appariement de **32,2 %** sur sept ans confirme que le 33 % de 2024 n'était pas un
accident d'échantillon.

### 12.4 Conclusion pour le rapport client

> **Les deux moteurs concordent sur le verdict, et ils y concordent les sept années.**
> L'espérance par trade est **négative en 2019, 2020, 2021, 2022, 2023, 2024 et 2025**, sur la
> référence Python **comme** sur QuantConnect, sans une seule exception. Elle vaut **−0,04 à
> −0,27 R** côté Python et **−0,10 à −0,38 R** côté QC : QuantConnect est **systématiquement
> plus sévère**, jamais plus favorable. Aucune année, aucun moteur ne produit une espérance
> positive.
>
> Cette concordance de signe est solide **bien que** les deux moteurs ne prennent que 32 % des
> mêmes trades : ils divergent sur *quels* trades prendre — un écart de données minute encore
> ouvert, §11.6 — mais pas sur ce que la stratégie rapporte. Deux tirages largement différents
> du même univers de décisions donnent le même signe sept fois sur sept, ce qui rend le verdict
> **plus** robuste qu'un accord obtenu sur des trades identiques.
>
> Les chiffres de **niveau** restent, eux, inexploitables : le −77,8 % du backtest QC compose une
> espérance négative avec un plancher de lot qui mord de plus en plus (111 entrées sur 229 en
> 2025), et ne mesure pas la stratégie. Le verdict publiable est celui de l'espérance en R, pas
> celui de la courbe d'équité.

---

## 9. MT5 — mesuré, 2022-11-04 → 2025-12-31

> **Statut** : mesuré. **Critère 8 de la table de décision : ÉCHEC.** L'espérance MT5 vaut
> **−0,1034 R** ; le critère exige `≥ 0`. Le second membre — même signe que Python — est GO
> (Python sur la même fenêtre : **−0,1189 R**), ce qui rend l'échec *interprétable* et non
> « non concluant » : les deux moteurs disent la même chose, et ce qu'ils disent est négatif.
> **Essais consommés** : **0**. La configuration lue est le centre de la grille
> (`z = 1 ; a_min = 0,2 ; k_s = 1 ; risk = 0,5 %`), déjà logé par `xau_x10:grid27:v1`.
> **Holdout** : aucune barre ≥ 2026-01-01 n'entre dans un calcul ; le dump s'arrête au
> **2025-12-30 23:58 UTC** et le parquet QC est coupé au 2025-12-31.

Tous les chiffres de cette section viennent de trois JSON, aucun n'est recalculé à la main :
`results/xau_x10/mt5_reference.json`, `results/xau_x10/reconciliation_mt5.json`,
`results/xau_x10/cost_sensitivity_measured.json`.

```bash
uv run python scripts/parse_mt5_report.py \
    --deals reports/mt5_x10/deals_x10_20251230T2359.csv \
    --html  "…/MetaTrader 5/x10_reference_report.htm" \
    --run   reports/mt5_x10/run_20260921T205244Z.json \
    --x10-trace reports/mt5_x10/x10_trace_20221104T0000.csv \
    --x10-dump  data/XAU-USD_minute_mt5_x10dump.parquet \
    --x10-log   "…/Tester/Agent-127.0.0.1-3001/logs/20260921.log" \
    --x10-py-trades reports/qc_x10/py_trades_full_2019_2025.csv \
    --out results/xau_x10/mt5_reference.json

uv run python scripts/reconcile_x10_events.py \
    --mt5-trace reports/mt5_x10/x10_trace_20221104T0000.csv \
    --mt5-deals reports/mt5_x10/deals_x10_20251230T2359.csv \
    --mt5-dump  data/XAU-USD_minute_mt5_x10dump.parquet \
    --mt5-log   "…/Tester/Agent-127.0.0.1-3001/logs/20260921.log" \
    --py-trades reports/qc_x10/py_trades_full_2019_2025.csv \
    --out results/xau_x10/reconciliation_mt5.json

uv run python scripts/build_xau_intraday_costs.py \
    --sensitivity-out results/xau_x10/cost_sensitivity_measured.json
```

### 9.0 Le run

Tester MT5, **modèle 1 (barres M1 OHLC)**, `XAUUSD.c` M5, 2022.11.04 → 2025.12.31, dépôt
10 000 USD, levier 1:100, spread **flottant historique** du broker SquaredFinancial (démo),
un seul agent, une seule graine. Compteurs de l'EA : `bars = 222708`, `decisions = 222708`,
`frozen = 0`, `arm = 7045`, `break = 1638`, `sweep = 764`, `entry = 880`, `exit = 880`,
`cancel = 6560`, `order_failures = 13`, `dxy_undef_bars = 84`, `spread_zero = 0`,
`spread_median_points = 250`.

**Bouclage comptable, vérifié avant toute lecture.** Somme des `profit + commission + swap`
des 1 760 deals de trading = **−3 019,26 USD**, exactement le `Total Net Profit` du rapport
HTML. 880 positions reconstruites, autant que d'`ENTRY` et d'`EXIT` dans la trace. `fill_px`
de la trace = prix du deal d'entrée, écart maximal **0,000 $** sur 880 positions. Le scénario
déduit du magic (841-844) coïncide avec celui de la trace sur **880/880**.

### 9.1 Contrôle d'horloge — l'heure serveur est bien UTC

L'EA suppose « heure serveur = heure UTC » (`Inp_ServerToNYMode = 0`,
`Inp_ServerToNYOffsetMin = 0`). L'hypothèse est vérifiée en corrélant les rendements minute du
dump à ceux de `data/XAU-USD_minute_qc.parquet`, qui est en UTC :

| décalage (min) | −180 | −120 | −60 | −5 | −1 | **0** | **+1** | +5 | +60 | +120 | +180 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| corrélation | 0,000 | 0,001 | −0,001 | −0,000 | −0,005 | **0,003** | **0,966** | −0,001 | −0,003 | −0,001 | −0,002 |

**Aucun décalage horaire** : `−180`, `−120`, `−60`, `+60`, `+120`, `+180` sont tous à la
corrélation d'un bruit. `server_clock_offset_min = 0`, et séance, fenêtres horaires et VWAP de
l'EA ne sont donc pas décalés.

Le maximum tombe à **+1 minute** parce que les deux fichiers ne datent pas une barre de la même
façon : **MT5 estampille une barre M1 de son ouverture, LEAN de sa clôture**. La preuve est dans
les niveaux de prix, pas seulement dans la corrélation — à décalage 0, l'écart QC − dump a un
p05/p95 de −0,90/+1,24 $ ; à +1 minute il tombe à 0,00/+0,27 $, et en comparant au **mid**
(`bid + spread/2`) à −0,095/+0,130 $ pour une médiane de **+0,015 $**. C'est une convention de
datation, pas une horloge, et le JSON les nomme séparément
(`bar_stamp_convention_lag_min = 1`, `server_clock_offset_min = 0`).

La même convention se lit dans la trace de l'EA : le deal d'entrée tombe **exactement cinq
minutes** après le `ts_decision` de son `ENTRY`, sur les 880 positions. `ts_decision` est donc
l'**ouverture** de la barre M5 de décision, et le fill a lieu à l'ouverture de la suivante,
ce qui est le §9 à la lettre.

### 9.2 Ce que MT5 mesure

| grandeur | valeur |
|---|---|
| trades | **880** |
| espérance | **−0,1034 R** |
| profit factor (HTML) | **0,83** |
| rendement net | **−30,19 %** (10 000 → 6 980,74 USD) |
| repli d'équité maximal | **34,90 %** (tick par tick, mesuré par MT5) |
| Sharpe (MT5) | −4,72 |
| spread médian mesuré | **0,25 $** |
| ordres refusés | **13** |

**Par année** — quatre années sur quatre négatives :

| année | trades | espérance R | profit factor | net USD | Python même fenêtre, espérance R |
|---|---|---|---|---|---|
| 2022 (2 mois) | 31 | −0,0776 | 0,884 | −123,98 | −0,2824 |
| 2023 | 231 | −0,0736 | 0,882 | −618,90 | −0,1479 |
| 2024 | 289 | −0,1674 | 0,751 | −1 699,75 | −0,1096 |
| 2025 | 329 | −0,0706 | 0,888 | −576,63 | −0,0905 |

**Par scénario** — quatre scénarios sur quatre négatifs, aucune poche à sauver :

| scénario | trades | espérance R | profit factor | taux de gain | net USD |
|---|---|---|---|---|---|
| BREAK_LONG | 206 | −0,1405 | 0,779 | 33,5 % | −778,24 |
| BREAK_SHORT | 130 | −0,0636 | 0,895 | 36,9 % | −462,55 |
| REV_LONG | 281 | −0,1160 | 0,817 | 34,5 % | −1 051,27 |
| REV_SHORT | 263 | −0,0806 | 0,881 | 30,0 % | −727,20 |

**Par raison de sortie** — la structure est celle qu'un stop serré produit :

| raison | trades | espérance R | net USD |
|---|---|---|---|
| STOP | 554 (63,0 %) | −1,0010 | −17 329,26 |
| TARGET | 181 (20,6 %) | +1,9596 | +10 875,24 |
| TIME | 108 (12,3 %) | +0,7383 | +2 411,51 |
| SESSION | 37 (4,2 %) | +0,7868 | +1 023,25 |

Le stop est touché sur **63 %** des trades et coûte exactement **1 R** — l'exécution est propre,
c'est la fréquence qui tue : il faudrait 34 % de cibles à 2 R pour équilibrer, il y en a 20,6 %.

### 9.3 C1 « mêmes données » — la référence Python rejouée sur le dump du broker

La référence tourne sur les **222 708 barres M5** reconstruites depuis le dump M1 du broker
(prix mid = `bid + spread/2` par barre M1 ; spread par barre M5 = médiane des M1 du bin ; DXY
lu dans `data/DXY4_h1.parquet`, écart assumé — le broker reconstruit le sien). Le **barreau 1
est exact** : 222 708 barres M5 des deux côtés.

**Barreau 2 — indicateurs à `ts_decision` identique** (14 373 lignes appariées à l'instant
exact, soit 81,6 % de la trace Python) :

| indicateur | écart absolu médian | p95 |
|---|---|---|
| `atr` | 0,000445 | 0,00536 |
| `v` | 0,000084 | 0,00669 |
| `m` | 0,000133 | 0,00721 |
| `a` | 0,000162 | 0,01201 |
| **`vwap`** | **0,1183** | 0,1758 |
| **`ema50_h1`** | **0,1022** | 0,1541 |
| `dxy` | 0,00540 | 0,02825 |

Ce tableau **nomme le poste d'écart principal à lui seul**. ATR et cinématique sont des
*différences* de prix : la demi-fourchette s'y annule, et les deux moteurs s'accordent à 1e-4.
VWAP et EMA50 H1 sont des *niveaux* : ils encaissent la demi-fourchette en entier, et l'écart
médian (0,118 $ et 0,102 $) vaut précisément la demi-fourchette médiane mesurée (**0,125 $**).
**L'EA lit les barres BID du terminal ; la référence lit le mid que §9 prescrit.**

Scores de contexte : `ctx_ema` 99,75 % d'accord, `ctx_vwap` 99,88 %, `ctx_dxy` 97,77 %.

**Barreau 3 — automate** (appariement `(level, d, ts ± 1 barre)`) :

| événement | Python | MT5 | appariés | taux Python | taux MT5 |
|---|---|---|---|---|---|
| ARM | 7 005 | 7 045 | 6 404 | 91,4 % | 90,9 % |
| BREAK | 1 633 | 1 638 | 1 341 | 82,1 % | 81,9 % |
| SWEEP | 762 | 764 | 578 | 75,9 % | 75,7 % |
| CANCEL | 6 494 | 6 560 | 5 555 | 85,5 % | 84,7 % |

Sur les 15 372 couples appariés tous types confondus, **97,1 %** portent le même événement des
deux côtés. Les raisons d'annulation se répartissent presque identiquement — `sweep` 2 568 / 2 598,
`zone` 2 390 / 2 396, `hold` 560 / 574, `r` 520 / 534, `ctx` 397 / 395, `nsweep` 51 / 55,
`window` 7 / 6, `narm` 1 / 2 : l'automate est le même, il ne bascule pas aux mêmes instants.

**Barreau 4 — entrées.** 911 entrées Python, 880 MT5, **645 appariées** :
**Python → MT5 = 70,8 %**, **MT5 → Python = 73,3 %**, contre la cible §13 de **95 %**.

**La cible est manquée, et la cause est mesurée.** En rejouant *exactement le même moteur, les
mêmes réglages et les mêmes barres* mais sur le **bid brut** — le prix que l'EA lit vraiment —
le même appariement passe à :

| variante de prix | entrées Python | appariées | Python → MT5 | MT5 → Python | `vwap` écart médian | `ema50_h1` écart médian |
|---|---|---|---|---|---|---|
| **mid** (§9, le brief) | 911 | 645 | **70,8 %** | 73,3 % | 0,1183 | 0,1022 |
| **bid** (ce que l'EA lit) | 894 | 877 | **98,1 %** | **99,7 %** | **0,0000** | **0,0000** |

Sur le bid, VWAP et EMA50 H1 coïncident **à zéro exactement**, et l'appariement dépasse la cible
de 95 %. L'écart de 27,3 points entre les deux lignes n'est donc pas un désaccord d'automate :
c'est **un seul poste, entièrement attribué**, et le rejeu le chiffre au lieu de le supposer.

**Barreau 4 — quantités théoriques** (645 paires, Python − MT5) :

| quantité | médiane | p05 | p95 |
|---|---|---|---|
| `stop` | +0,0755 $ | −0,0017 | +0,1611 |
| `target` | **0,000000 $** | 0,000 | 0,000 |
| `r_est` | −0,0130 | −0,2008 | +0,1839 |
| `fill_px` | 0,000 $ | −0,020 | +0,010 |
| `lots` | 0,00 | −0,02 | +0,01 |

La **cible est identique au bit près sur les 645 paires** : les niveaux x10 sont les mêmes des
deux côtés, et le barreau 3 n'a donc rien à voir avec un décalage de grille. Le stop diffère de
0,076 $ en médiane, soit à nouveau la moitié d'une fourchette pour les scénarios dont le stop
dérive d'un extrême de prix.

**Barreau 5 — exécution.** Sur les 645 paires : R réalisé Python **−0,0517**, MT5 **−0,0586**,
écart médian **+0,0000097 R** (p05 −0,116, p95 +0,079). Les raisons de sortie s'accordent sur
**97,8 %** des paires ; les 14 désaccords sont 5 `SESSION → STOP`, 6 `SESSION → TARGET`,
1 `TIME → STOP`, 1 `STOP → TARGET`, 1 `TARGET → STOP`.

### 9.4 C2 « données différentes » — la campagne officielle contre MT5

La campagne Python officielle (parquet QC, spread constant 0,29 $) restreinte à la fenêtre du
run : **879 entrées**, contre 880 pour MT5. Appariées : **248**, soit **28,2 %**, contre la
cible §13 de **70 %**. Par année : 11,4 % (2022), 31,5 % (2023), 27,4 % (2024), 28,7 % (2025).

Espérance en R sur la fenêtre commune : Python **−0,1189**, MT5 **−0,1034**. **Même signe,
écart de 0,016 R** — les deux moteurs ne prennent pas les mêmes trades mais disent la même
chose sur ce que la stratégie rapporte, exactement comme QC au §8.

**Pourquoi C2 s'effondre alors que C1 tient à 98 %** — à moteur strictement identique, trois
postes séparent les deux, et deux seulement pèsent :

1. **La convention de binning M5.** Le parquet QC date une barre M1 de sa **clôture**, le dump
   de son **ouverture**. À grille M5 `label="left"` identique, les deux moteurs n'agrègent donc
   pas les mêmes minutes. Mesuré sur le **même dump**, en ne changeant que la datation :
   **98,7 % des 222 705 barres M5 ont une clôture différente**, d'un écart absolu médian de
   **0,24 $** et de p95 **1,37 $** — soit, aux heures calmes, un demi-ATR. C'est le poste
   structurel, et il touche **chaque barre**.
2. **Le flux.** Mid du broker contre clôture QC, minute par minute et datation recalée :
   médiane **+0,015 $**, p05/p95 **−0,095/+0,130 $**, stable d'une année sur l'autre (médiane
   0,010 à 0,020 $). Négligeable en médiane, mais 0,1 $ vaut 2 à 20 % d'un ATR M5 : de quoi
   faire basculer un test de seuil.
3. **La règle de position unique**, en revanche, **n'explique presque rien** : seuls **4,8 %**
   des orphelins Python et **3,5 %** des orphelins MT5 sont émis pendant que le moteur d'en
   face tenait une position. L'hypothèse « les deux chemins se bloquent mutuellement » est
   **mesurée et écartée**.

### 9.5 Attribution, poste par poste

| poste | mesure | statut |
|---|---|---|
| **signaux sur bid (EA) contre mid (référence)** | appariement 70,8 % → **98,1 %** en rejouant sur le bid ; `vwap` et `ema50_h1` passent de 0,118/0,102 $ d'écart médian à **0,000** | **mesuré** |
| **séance du broker qui rouvre plus tard** | cotations reprises à **18:01 New York** (618 jours) ou **18:05** (193 jours) ; les 13 ordres refusés tombent **20 à 59 min** après cette reprise (médiane 34), les fills acceptés à 704 min en médiane. Preuve directe : `retcode 10018 = TRADE_RETCODE_MARKET_CLOSED` | **mesuré** |
| **sorties TIME/SESSION à l'open de la barre suivante** | 106 trades concernés ; écart de prix de sortie médian **0,000 $** (p05 −4,49, p95 +0,58), écart en R médian **+0,0036** (p05 −0,68, p95 +0,63) | **mesuré** |
| **double contact tranché par le tester** | **2** inversions `STOP ↔ TARGET` sur 645 paires, soit **0,31 %** | **mesuré** |
| **spread courant contre constant** | médiane mesurée **0,250 $**, p95 **0,380 $**, contre 0,29 $ constant. Sans effet en C1, qui rejoue le spread mesuré par barre | **mesuré** (C2) |
| **DXY du broker contre parquet local** | `ctx_dxy` d'accord sur **97,77 %** des lignes ; écart absolu médian sur la valeur brute **0,0054**, p95 0,0283, max 1,238 | **mesuré** |
| **ordres refusés** | **13** ouvertures refusées (`retcode 10018`), toutes entre 22:01 et 23:00 UTC. Les `[EXIT][WARN]` du log sont des **réessais** : 283 lignes pour **37 positions**, pas 283 incidents | **mesuré** |
| **convention de binning M5** | **98,7 %** des barres M5 changent de clôture selon la datation M1 retenue ; écart médian 0,24 $ | **mesuré** (C2) |
| **flux QC contre flux broker** | médiane +0,015 $, p05/p95 −0,095/+0,130 $ | **mesuré** (C2) |
| **règle de position unique** | 4,8 % / 3,5 % des orphelins | **mesuré, et écarté** |

**Part non attribuée, C1 : 1,1 %** — 13 orphelins sur 1 146 entrées de l'union, contre un seuil
de blocage de **5 %** (§13). Sur les 501 orphelins de la variante mid, **488** sont expliqués
par le seul poste bid/mid. **La réconciliation MT5 ne bloque pas la lecture hors échantillon.**

### 9.6 Les 13 ordres refusés

Tous `retcode 10018` (*marché fermé*), tous des **ouvertures**, tous entre 22:01 et 23:00 UTC :

| date UTC | sens | lots | stop | cible |
|---|---|---|---|---|
| 2023.04.16 22:55 | buy | 0,13 | 1 998,983 | 2 010,000 |
| 2023.04.26 22:35 | sell | 0,21 | 1 990,822 | 1 980,000 |
| 2023.06.11 22:25 | buy | 0,40 | 1 959,468 | 1 970,000 |
| 2024.05.08 22:35 | sell | 0,29 | 2 311,178 | 2 300,000 |
| 2024.05.19 22:35 | sell | 0,04 | 2 423,578 | 2 410,000 |
| 2024.07.28 22:30 | sell | 0,07 | 2 402,537 | 2 390,000 |
| 2024.09.22 22:35 | buy | 0,20 | 2 619,086 | 2 630,000 |
| 2025.03.24 22:55 | buy | 0,05 | 3 008,391 | 3 020,000 |
| 2025.04.01 22:30 | buy | 0,03 | 3 106,979 | 3 120,000 |
| 2025.07.09 22:40 | sell | 0,15 | 3 321,807 | 3 310,000 |
| 2025.09.10 22:35 | buy | 0,04 | 3 637,025 | 3 650,000 |
| 2025.09.28 22:25 | sell | 0,06 | 3 772,375 | 3 760,000 |
| 2025.10.14 23:00 | sell | 0,03 | 4 163,851 | 4 150,000 |

Toutes les dates tombent en heure d'été de New York, c'est-à-dire entre **18:01 et 19:00 heure
locale** : la stratégie autorise l'entrée dès 18:15 (§10) alors que la session de négociation du
symbole n'est pas encore ouverte. **13 trades sur 893 tentatives, soit 1,5 %** : le poste est
réel, nommé, et trop petit pour renverser quoi que ce soit.

### 9.7 Le spread mesuré, et ce qu'il change

`costs_xau_intraday.yml`, généré par `scripts/build_xau_intraday_costs.py` depuis le dump M1
(1 111 881 barres) : **médiane globale 0,250 $**, p75 **0,300 $**, p95 **0,380 $**.

| année | source | médiane des heures | heure la plus chère (New York) | heures les moins chères |
|---|---|---|---|---|
| 2019 | extrapolée (×0,723) | 0,188 $ | 18 h → 0,268 $ | — |
| 2020 | extrapolée | 0,237 $ | 18 h → 0,338 $ | — |
| 2021 | extrapolée (×0,922) | 0,240 $ | 18 h → 0,341 $ | — |
| 2022 | mesurée (**partielle**, nov-déc) | 0,260 $ | 18 h → 0,370 $ | — |
| 2023 | mesurée | 0,260 $ | 18 h → 0,370 $ ; 19 h et 23 h → 0,34 $ | 3-5 h → 0,26 $ |
| 2024 | mesurée | 0,250 $ | 18 h → 0,350 $ ; 19 h → 0,34 $ | 2-4 h → 0,25 $ |
| 2025 | mesurée | 0,160 $ | 18 h → 0,190 $ | 19-20 h → 0,15 $ |

**L'heure la plus chère est 18 h New York, toutes années confondues** — la reprise de séance,
celle-là même où la stratégie a le droit d'entrer à partir de 18:15. Le spread s'est **effondré
en 2025** (0,16 $ contre 0,26 $ en 2023) alors que l'once a doublé : en fraction du prix, la
fourchette a été divisée par trois.

**Sensibilité, centre de la grille, 2019 → 2025-12-31** (`cost_sensitivity_measured.json`) :

| spread appliqué | moyenne | trades | espérance R | profit factor | Δ espérance |
|---|---|---|---|---|---|
| constant 0,29 $ (sélection) | 0,290 $ | 1 723 | **−0,1602** | 0,751 | — |
| mesuré, médiane heure × année | 0,251 $ | 1 728 | **−0,1475** | 0,769 | **+0,0127** |
| mesuré, p75 heure × année | 0,276 $ | 1 726 | **−0,1564** | 0,757 | +0,0038 |

La ligne « constant 0,29 $ » **reproduit la campagne officielle au trade près** (1 723 trades,
−0,1602 R), ce qui valide la chaîne de rejeu. Le spread réel est donc **meilleur marché que
celui de la sélection**, de 0,039 $ en moyenne — et il rapporte **+0,013 R**, soit **8 %** du
déficit. Il en manque **0,147 R**. Conformément à §1.1 de la table de décision, cette mesure est
publiée comme sensibilité et **ne re-sélectionne rien** : la configuration retenue ne bouge pas.

### 9.8 Verdict du critère 8

> | membre du critère | seuil | mesure | verdict |
> |---|---|---|---|
> | espérance MT5 | ≥ 0 | **−0,1034 R** | **ÉCHEC** |
> | même signe que Python | oui | Python −0,1189 R, MT5 −0,1034 R | GO |
>
> **Critère 8 : ÉCHEC**, avec une mesure interprétable. Le verdict de la table de décision reste
> **NE PAS DÉPLOYER** ; MT5 ne le contredit pas, il le confirme sur le moteur d'exécution du
> client, quatre années sur quatre et quatre scénarios sur quatre.

Le verdict n'est **pas** « NON CONCLUANT » : aucun des quatre déclencheurs du §2 de la table de
décision n'est armé. La divergence non attribuée vaut 1,1 %, loin du seuil de 5 % ; le chiffre
MT5 existe et boucle au centime ; la fenêtre est in-sample et non OOS.

### 9.9 Limites, et ce qui n'a pas pu être vérifié

1. **Modèle 1 — barres M1 OHLC, pas de vrais ticks.** Le tester interpole quatre prix par
   minute. Les 554 sorties au stop sont donc résolues sur une trajectoire reconstruite ; leur
   espérance mesurée de −1,0010 R est trop propre pour être crédible tick par tick.
2. **Compte de démo expiré.** Le terminal ne joint plus le serveur du broker depuis au moins le
   2026-08-05 : le run tourne hors connexion, sur l'historique en cache. Cet historique ne peut
   pas être revalidé contre le serveur, et un futur re-téléchargement pourrait ne pas le
   reproduire.
3. **DXY indéfini sur 84 barres.** §6.3 rend le panier neutre dans ce cas ; l'effet est borné
   mais non isolé. Par ailleurs le DXY de l'EA est reconstruit depuis les quatre paires du
   broker, celui de la référence lu dans `data/DXY4_h1.parquet` : accord `ctx_dxy` 97,77 %,
   écart assumé et non corrigé.
4. **13 ordres refusés** (§9.6), dont l'effet sur le chemin d'équité n'est pas simulable : on
   sait ce que l'EA a voulu faire, pas ce que ces trades auraient rapporté.
5. **2022 n'est pas une année.** Le run démarre le 4 novembre ; les 31 trades et la ligne 2022
   de la table de coût portent deux mois. Le YAML le déclare (`partial_year: true`).
6. **La cible §13 de 95 % n'est atteinte qu'en variante bid.** Le brief fixe le mid pour C1, et
   sur le mid le taux vaut 70,8 %. La lecture honnête est : *la cible est atteinte dès que les
   deux moteurs lisent le même prix*, et l'écart restant est un poste nommé, pas un résidu.
7. **L'extrapolation du spread d'avant 2022-11 est une hypothèse**, pas une mesure : profil
   horaire de 2023 remis à l'échelle du prix médian annuel. Chaque ligne concernée porte
   `source: extrapolated`.
8. **Non mesurable en l'état** : l'effet du slippage réel du broker (le run tourne à
   `Inp_SlippageUSD = 0`), la sensibilité à la graine du générateur de ticks (une seule graine),
   et le comportement sous modèle 0 ou 4 (ticks réels indisponibles hors connexion).

---

## 10. Conséquences

1. **Le backtest `x10_calage_2024_centre` est inexploitable** et ne doit apparaître dans aucun
   document client, même comme illustration négative. Les −91,8 % ne mesurent pas la stratégie.
2. **La lecture hors échantillon reste bloquée** au titre de §13 : 62,4 % des entrées divergent
   sans cause d'exécution nommable, contre un seuil de blocage de 5 %.
3. **Le verdict in-sample de `docs/research/xau_x10_is_results.md` n'est pas affecté** : il ne
   dépend que de la chaîne Python, et il était déjà « NE PAS DÉPLOYER ». Cette note ne le sauve ni
   ne l'aggrave.
4. **Deux chantiers, dans cet ordre** : (a) corriger D1-D5 et relancer, pour obtenir un backtest
   QC qui mesure au moins quelque chose ; (b) fermer D8, sans quoi la cible de 98 % de §13 est
   hors d'atteinte et la réconciliation à trois moteurs n'a pas de sens.
5. **Le spread de 0,29 $ de §12 est optimiste d'un facteur ~1,7** sur le flux OANDA. La
   sensibilité ×2 déjà exigée par §12 n'est pas une précaution : c'est le cas central.
