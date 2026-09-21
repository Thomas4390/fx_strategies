# XAUUSD niveaux x10 : réconciliation événementielle Python ↔ QuantConnect, 2024

> **Date** : 2026-09-21 · **Statut** : mesuré — **le backtest QC `x10_calage_2024_centre` n'est
> pas une mesure de la stratégie** et ne doit être cité nulle part comme telle.
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

## 9. MT5 : en attente

*Rien à publier. L'EA `src/mt5/Experts/XauX10.mq5` n'a pas encore tourné dans le tester.*
`scripts/reconcile_x10_events.py` expose déjà l'option `--mt5-trace` ; elle s'arrête aujourd'hui
sur un message explicite. Tolérance d'appariement visée par §13 pour cette paire : **≥ 70 %**,
écart attribué (mid contre bid, niveaux frôlés, flux broker).

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
