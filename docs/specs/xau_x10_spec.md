# Spécification — stratégie XAUUSD niveaux x10 (Apogée Invest, stratégie 2)

> **Statut : source de vérité.** Ce document prime sur les trois implémentations à venir
> (vbt `src/strategies/xau_x10.py`, QuantConnect projet `ApogeeInvest_S2_XAUUSD_X10`, MQL5
> `src/mt5/Experts/XauX10.mq5`). Toute divergence entre un moteur et cette spec est un défaut
> du moteur, pas de la spec. Si la spec est jugée fausse, on la corrige ici **d'abord**, puis
> on propage — jamais l'inverse.
>
> Établie le 2026-09-21, **avant toute ligne de code et tout backtest**. Les paramètres gelés,
> la grille et le budget d'essais de l'annexe A sont opposables : ils interdisent la
> re-sélection après mesure.

Le pseudo-code est volontairement neutre : pas d'API de moteur, indices explicites. `t` désigne
une **barre M5** close, `t−1` la barre M5 précédente. Les majuscules `O, H, L, C` désignent les
prix de la barre M5 courante ; `L` seul, quand il est suivi d'un argument, désigne un niveau
x10 — le contexte lève l'ambiguïté, et la trace du §13 nomme la colonne `level`.

Douze sous-spécifications du plan de phase (momentum inopérant, sens des scores de contexte,
statut causal du VWAP, double facturation du spread dans `R`, chemin « breakout raté », fin de
vie de l'armement…) ont été tranchées le **2026-09-21** et sont intégrées au fil du texte. Il
ne reste **aucun point ouvert** : tout ce qu'un moteur doit savoir est écrit ici, et ce qui est
volontairement imparfait est signalé comme une obligation de publication, pas comme une marge
d'interprétation.

---

## 0. Brief client et périmètre

Mandat, tel que formulé par Apogée Invest et repris sans reformulation :

> Stratégie XAUUSD fondée **uniquement** sur les niveaux psychologiques de 10 $. Exécution en
> M5, toute la journée. Contexte H1 = EMA50 H1 + VWAP + DXY, le DXY filtrant sans bloquer
> automatiquement. Le cœur est la **vitesse**, le **momentum** et l'**accélération** à
> l'approche du x10. L'ATR normalise tout : largeur de zone, force de cassure, accélération,
> profondeur du sweep, stop. La cible est le **prochain x10**. Le rapport gain/risque doit
> valoir au moins 1:1. Quatre scénarios : Breakout Long, Breakout Short, Reversal Short,
> Reversal Long. Chaîne de décision : x10 → contexte → vitesse → accélération → breakout ou
> sweep → confirmation/retest → R ≥ 1 → entrée.

Périmètre de ce document : définitions, horloge, indicateurs, machine à états, exécution,
dimensionnement, coûts, contrat de trace, environnement. **Hors périmètre** : tout chiffre de
performance, tout choix de configuration, toute conclusion. Ils appartiennent à
`docs/research/xau_x10_is_results.md` et à `docs/research/xau_x10_decision_table.md`.

Le mandat impose un instrument et une famille de signaux ; il ne garantit aucun résultat. Un
verdict négatif est un livrable acceptable et pré-prévu (voir la table de décision).

## 1. Univers et données

| | |
|---|---|
| Instrument | XAUUSD (or spot / CFD selon le moteur) ; symbole broker `XAUUSD.c` |
| Fréquence de décision | M5 |
| Fréquence de résolution des ordres | M1 |
| Période de référence | 2019-01-01 → 2025-12-31 (sélection) ; ≥ 2026-01-01 verrouillé |
| Source vbt/QC | `data/XAU-USD_minute_qc.parquet`, exporté de QuantConnect, M1 OHLC UTC |
| Source MT5 | flux broker SquaredFinancial, **différent par construction**, à partir du 2022-11-04 |
| Contexte devises | `data/DXY4_{h1,minute}.parquet`, panier synthétique construit au lot D |

vbt et QC partagent les mêmes barres ; MT5 non. Cette asymétrie fixe deux cibles de
réconciliation distinctes — voir §13.

XAUUSD **n'a aucun volume exploitable** sur QC : `load_gold_data` (`src/utils.py`) pose
`volume = 1,0`. Toute grandeur de cette spec est donc calculée **sans volume**, ce qui décide
la définition du VWAP au §6.

## 2. Horloge, séance et bornes

```
REFERENCE_TZ  := America/New_York, horloge locale
SEANCE        := 18:00 (J-1)  ->  17:00 (J)
```

La séance or court du dimanche 18:00 au vendredi 17:00 New York. `gold_momentum.session_dates()`
(`src/strategies/gold_momentum.py:162`) est le **seul** producteur de cette borne ; les trois
moteurs doivent en reproduire la convention, MT5 par conversion explicite heure serveur → New
York (test DST dédié, quatre bascules).

```
decision[t] := a la cloture de la barre M5 t
fill[t]     := a l'open de la barre M5 t+1
```

Aucune décision ne consomme un prix postérieur à `C[t]`. C'est le point où la stratégie 1
s'autorisait une idéalisation (`fill = close[t]`, `gold_momentum_spec.md` §6) ; **ici non**.

## 3. Niveaux x10

```
L_inf[t] := floor( round(C[t] * 1000) / 10000 ) * 10      # arithmetique entiere
L_sup[t] := L_inf[t] + 10
```

Le passage par `round(C·1000)` (le millième de dollar, soit le `point` du symbole broker :
`digits=3`, `point=0,001`) évite qu'un prix stocké `4409,999999996` ne tombe du mauvais côté
de la grille. Propriété opposable : `L_inf ≤ C < L_sup`, vérifiée sur 10⁵ prix tirés au lot A1.

Direction d'approche `d ∈ {+1, −1}` : `d = +1` vise `L = L_sup`, `d = −1` vise `L = L_inf`.
Par construction `d·(L − C) > 0` : le niveau visé est toujours devant le prix.

Un **croisement** est un changement de `floor(C/10)` entre deux clôtures M5 consécutives. Sa
fréquence est mesurée dans `docs/research/xau_x10_feasibility_2026H2.md` §1.

## 4. ATR — l'unité de toutes les distances

```
ATR_PERIOD := 14                               # gele
TR[t]      := max( H[t]-L[t], |H[t]-C[t-1]|, |L[t]-C[t-1]| )
A[t]       := A[t-1] + ( TR[t] - A[t-1] ) / 14        # recursion de Wilder
A[1]       := TR[1]                                   # amorcage sur le premier TR disponible
```

**Amorçage — normatif.** La récursion démarre sur le **premier TR disponible** (`TR[1]`, la
deuxième barre M5, la première n'ayant pas de `C[t−1]`), et la valeur n'est **publiée qu'à
partir de 14 TR** accumulés : `A[t]` est **indéfini** avant, et les barres antérieures sont
exclues, elles ne valent pas 0. Équivalent pandas exact :
`tr.ewm(alpha=1/14, adjust=False, min_periods=14).mean()`. Référence d'implémentation :
`src/framework/x10_kernels.py::atr_wilder_nb`.

Lissage de Wilder (`alpha = 1/14`, récursif), **pas** de moyenne simple ni d'EMA à `2/(n+1)`.

⚠️ **L'`iATR` natif de MT5 n'amorce pas ainsi** : il part de la moyenne simple des 14 premiers
TR, ce qui décale durablement la série. **L'EA DOIT recalculer l'ATR à la main** avec la
récursion ci-dessus. Même consigne côté QuantConnect : aucun indicateur ATR natif ne doit être
utilisé sans avoir vérifié son amorçage. Un écart d'amorçage se propage à *toutes* les
distances de la spec, donc à tous les seuils.

**Toute distance de cette spec s'exprime en multiples de `A[t]`**, `A` étant celui de la barre
de décision, jamais recalculé en cours de trade. La normalisation ATR est l'exigence explicite
du mandat (§0) et la raison pour laquelle la stratégie peut prétendre traverser des régimes
dont `10 $ / A` va de 17,3 à 1,8 — prétention que la note de faisabilité met en doute.

## 5. Cinématique — vitesse, momentum, accélération

Horizons **gelés** (3 et 12 barres M5, soit 15 et 60 minutes) :

```
v[t] := ( C[t] - C[t-3]  ) / ( 3 * A[t] )          # vitesse, en ATR par barre
m[t] := ( C[t] - C[t-12] ) / ( A[t] * racine(12) ) # momentum, normalise en racine du temps
a[t] := v[t] - v[t-3]                              # acceleration
```

- **vitesse** `v` : pente moyenne des trois dernières barres, en ATR par barre. `d·v` est la
  vitesse *vers* le niveau visé.
- **momentum** `m` : déplacement sur une heure, normalisé en `racine(12)` pour être comparable
  à un pas aléatoire. `m` **conditionne l'armement** : `d·m > 0` exige que le prix arrive au
  niveau avec un déplacement horaire orienté vers lui (§7.1). Le seuil est **gelé à 0** —
  c'est un test de signe, pas un paramètre.
- **accélération** `a` : variation de la vitesse sur trois barres, en ATR par barre et par
  fenêtre de trois barres. `d·a > 0` = le prix accélère vers le niveau ; `d·a < 0` = il
  décélère.

**Domaine de définition — règle exacte.** `v[t]` et `m[t]` sont définis dès que `A[t]` l'est
**et** que le close retardé existe (`C[t−3]`, respectivement `C[t−12]`). `a[t] = v[t] − v[t−3]`
n'est donc défini que **trois barres après** `v`. Tout NaN se propage : **aucun armement n'est
possible sur une barre où `v`, `m` ou `a` est NaN** (§7.1), sans exception ni substitution
par 0.

> **Unités — à ne pas confondre.** `v` et `m` sont sans dimension une fois divisés par `A`,
> mais pas homogènes entre eux (`v` est une pente par barre, `m` un déplacement normalisé en
> racine du temps), et `a` est une différence de `v` sur trois barres, donc une troisième
> grandeur. Les seuils `v_min = 0,2` et `a_min ∈ {0,1 ; 0,2 ; 0,3}` ont des ordres de grandeur
> voisins **par coïncidence** : ils ne mesurent pas la même chose et ne se comparent pas.

## 6. Contexte — H1 décalé, séance M5, scores

Deux familles de grandeurs de contexte cohabitent et **n'obéissent pas à la même règle de
causalité**. Les confondre est la première source d'erreur possible dans cette stratégie.

### 6.1 Contexte H1 — construction anti look-ahead

Concerne **EMA50_H1**, **ATR_H1** et **DXY** (et rien d'autre).

**Règle de causalité, non négociable.** La construction est, dans cet ordre exact :

```
serie_H1  := agregation des barres M1 en H1 (label = debut d'heure)
serie_H1  := serie_H1.shift(1)                   # <- obligatoire
contexte  := serie_H1.reindex(index_M5).ffill()
```

Une barre H1 n'est visible qu'après sa clôture, donc au plus tôt sur la barre M5 qui ouvre
après elle. `resample().last() + ffill` sans `.shift(1)` rend le close de 08:59 disponible à
08:00 : ce défaut exact a produit un Sharpe de 3,10 qui est tombé à **−1,48** après correction
(`docs/research/eur-usd-bb-mr-research-plan.md:148-197`). Le test 2 du lot A1 doit rougir sur
une version fautive.

**EMA50 H1** : moyenne exponentielle à 50 périodes (`alpha = 2/51`) des clôtures H1 de l'or,
puis décalage et reindex ci-dessus. **ATR_H1** : ATR de Wilder 14 sur barres H1, même
traitement ; il ne sert qu'à la mesure d'extension VWAP (§7.3).

### 6.2 Contexte de séance (M5) — VWAP

Le VWAP **n'est pas une série H1** et la règle du §6.1 ne le concerne pas. C'est une moyenne
**cumulée de séance du prix typique, non pondérée**, calculée sur les barres M5 :

```
typ[t]  := ( H[t] + L[t] + C[t] ) / 3
VWAP[t] := moyenne de typ sur toutes les barres de la seance en cours, jusqu'a t inclus
```

**Causalité** : la moyenne s'arrête à la barre M5 `t` **incluse**. `VWAP[t]` est donc
entièrement connu à la clôture de `t`, qui est l'instant de décision (§2) — aucun décalage
n'est requis, et en introduire un fausserait la grandeur.

Remise à zéro à 18:00 New York (§2), bascules DST comprises. **Valide après 12 barres** de
séance ; indéfini avant, donc `ctx_vwap = 0`. Motif de la non-pondération : l'or n'a pas de
volume sur QC (§1) ; une pondération rendrait les trois moteurs incomparables. MT5 ne doit
**pas** réutiliser `CVWAPDaily` (pondérée et ancrée 00:00 UTC) mais une classe neuve
`CSessionMeanTypical`.

### 6.3 DXY4

Panier synthétique à quatre jambes, poids ICE renormalisés, construit sur des barres H1 et
donc soumis à la règle du §6.1.

```
w_e = 0,576 ; w_j = 0,136 ; w_g = 0,119 ; w_c = 0,091   renormalises a somme 1
DXY4 := 50,14348112 * EURUSD^(-w_e) * USDJPY^(w_j) * GBPUSD^(-w_g) * USDCAD^(w_c)
```

Jambe absente depuis plus de **5 barres** → **DXY indéfini = neutre** (`ctx_dxy = 0`), et
l'occurrence est comptée. Ce cas est la règle normale dans le tester MT5, où `CopyRates`
multi-symboles n'est pas garanti.

**Contrôle de vraisemblance.** Corrélation des rendements journaliers DXY4 / FRED
`DTWEXBGS` : **0,87** au fixing de midi New York, **0,67** en fin de jour calendaire — l'écart
est purement horloger, pas un défaut de construction. Mesure sur **2 047 jours**,
2018 → 2026. `DTWEXBGS` est un **indice large à 26 devises**, pas le DXY ICE : la cible de 0,98
inscrite au plan de phase était mal calibrée et est **abandonnée**. Rien n'a été ajusté pour la
remonter. Niveau médian du panier : **94,35** (effet de la renormalisation après retrait du CHF
et de la SEK) ; sans conséquence, puisque **seul le signe de `DXY − EMA50_H1(DXY)` est
consommé** (§6.4). Couverture des quatre jambes : **2018-01-01 → 2026-04-01** ; au-delà,
`ctx_dxy = 0` par la règle ci-dessus. Vérification :
`uv run python scripts/build_dxy_synthetic.py --check`.

### 6.4 Scores de contexte — toujours dans le sens du trade

```
q := sens du TRADE envisage        # q = +d pour un breakout, q = -d pour un reversal

ctx_ema  := signe( q * ( C[t] - EMA50_H1[t] ) )
ctx_vwap := signe( q * ( C[t] - VWAP[t]     ) )
ctx_dxy  := signe( - q * ( DXY[t] - EMA50_H1(DXY)[t] ) )
```

`signe(0) = 0`. Les trois scores sont évalués **dans le sens du trade `q`**, jamais dans la
direction d'approche `d`. Pour un breakout les deux coïncident ; pour un reversal ils sont
opposés, et c'est le sens du trade qui compte — un contexte « favorable » doit vouloir dire
favorable à la position qu'on prend. Tant qu'aucun scénario n'est identifié (état `ARMED`
seul), la trace porte les scores évalués en `q = d`, sens du seul trade alors envisageable.

Le **score de contexte** est le triplet `(ctx_ema, ctx_vwap, ctx_dxy)`, tracé tel quel ; aucune
agrégation en un scalaire.

Usages, et eux seuls :

1. **Breakout** : exige `ctx_ema > 0` **et** `ctx_vwap > 0`. Sinon pas d'entrée.
2. **Reversal** : **aucune exigence** de contexte, ni sur `ctx_ema` ni sur `ctx_vwap`. Il est
   contre-tendance par construction, donc autorisé contre l'EMA50 H1 — décision explicite du
   mandat. Les deux scores sont tracés et servent aux décompositions, pas au filtrage.
3. **DXY adverse** (`ctx_dxy < 0`, donc adverse **au sens du trade**) : le trade **reste
   autorisé** ; le risque passe de 0,5 % à 0,25 % de l'équité (§11). Cette règle s'applique aux
   **quatre scénarios**, breakouts comme reversals. C'est la lecture retenue de « le DXY filtre
   sans bloquer ».

## 7. Machine à états

Invariant : **un seul niveau armé à la fois**, et **une seule position à la fois**.

### 7.1 ARMED

```
ARMED(L, d)  si  0 < d*(L - C[t]) <= z * A[t]      # zone
             et  d * v[t] >= v_min                 # vitesse vers le niveau
             et  d * a[t] >= a_min                 # acceleration vers le niveau
             et  d * m[t] >  0                     # momentum horaire oriente (seuil gele a 0)
```

- `z·A` est la **zone** : la largeur, en ATR, de la bande devant le niveau à l'intérieur de
  laquelle on commence à surveiller. `z` est un axe de grille.
- `d·m > 0` traduit l'exigence du mandat que le prix **arrive** au niveau avec du momentum.
  C'est un test de signe : aucun paramètre, aucun axe de grille, rien à optimiser.
- Les deux directions ne peuvent pas être armées ensemble : `d·v ≥ v_min > 0` est exclusif en
  `d`.

**Cycle de vie de l'armement.** Tant qu'un niveau est `ARMED`, **tout autre niveau armable est
ignoré** — aucun remplacement, aucune file d'attente. L'état se termine par exactement une de
ces sorties :

| sortie | condition |
|---|---|
| `BREAK` | cassure validée au sens du §7.2 |
| `SWEEP` | excursion validée au sens du §7.3 |
| `CANCEL` | `d·(L − C[t]) > z·A[t]` — le prix ressort de la zone **vers l'arrière** |
| `CANCEL` | `N_arm = 12` barres M5 écoulées depuis l'`ARM` sans aucun événement (constante gelée) |

`N_hold` et `N_sweep` ne décomptent pas l'armement : ils décomptent ce qui suit un `BREAK` ou
une excursion.

### 7.2 Breakout (scénarios Breakout Long `d=+1`, Breakout Short `d=−1`)

```
CASSURE si  d * ( C[t] - L ) >= k_b * A[t]
        et  cloture reelle : C[t] situe dans la moitie directionnelle de la barre,
            c.-a-d.  d * ( C[t] - (Hi[t]+Lo[t])/2 ) > 0     # Hi/Lo = haut/bas de la barre ; L = niveau
MAINTIEN : pendant N_hold barres apres la cassure, aucune cloture M5 ne repasse
           du mauvais cote du niveau ( d * (C - L) < 0 ).
RETEST   : une barre de la fenetre de maintien dont la meche revient toucher le niveau
           ( Lo[t] <= L si d=+1 ; Hi[t] >= L si d=-1 ) sans cloture repassee ; trace, non exige.
ENTREE   : a l'open de la barre suivant la fin du maintien, sens d.
CIBLE    := L + 10*d
STOP     := L - d * k_s * A
```

- **clôture réelle** : la condition `C` dans la moitié directionnelle de la barre. Elle écarte
  les barres qui dépassent le niveau en mèche et referment contre.
- **maintien** : `N_hold = 3` barres sans clôture repassée, gelé.
- **retest** : retour de la mèche sur le niveau pendant le maintien. Il est **mesuré et
  tracé**, il ne conditionne pas l'entrée. Le mandat cite « confirmation/retest » ; la
  confirmation est le maintien, le retest est descriptif.
- **Breakout raté → candidat sweep**, règle normative. Soit `b` la barre du `BREAK`. Si une
  clôture vérifie `d·(C − L) < 0` pendant la fenêtre de maintien :
  1. le breakout est annulé (`CANCEL`) ;
  2. **cette barre est la barre de réintégration candidate** ;
  3. l'extrême `E` est le plus haut (`d = +1`) ou le plus bas (`d = −1`) atteint **depuis `b`** ;
  4. le reversal est pris **à cette barre** si, à cette barre, l'excursion vaut
     `d·(E − L) ≥ s_min·A`, `d·v` décroît et le flip `d·a ≤ −a_min` sont vrais ;
  5. sinon retour à `IDLE`.

  **Aucun ré-armement n'est requis** : le reversal se greffe directement sur l'armement qui a
  produit le breakout. `N_sweep` ne s'applique pas à ce chemin (voir §7.3).

### 7.3 Reversal (scénarios Reversal Short après approche haussière, Reversal Long après approche baissière)

```
EXCURSION      : E := extreme atteint au-dela du niveau depuis le franchissement
                 ( plus haut si d=+1, plus bas si d=-1 )
                 profondeur du sweep :  d * ( E - L ) >= s_min * A
DECELERATION   : d * v[t] < d * v[t-1]            # barre precedente, sans seuil
FLIP           : d * a[t] <= - a_min
REINTEGRATION  : une cloture M5 avec  d * ( C[t] - L ) < 0,  dans les N_sweep barres
                 comptees depuis la premiere barre dont l'extreme depasse L
ENTREE         : a l'open de la barre suivante, sens  -d
CIBLE          := L - 10*d
STOP           := E + 0,5 * A * d        # l'extreme de l'excursion, augmente de 0,5 ATR
```

- **sweep** : la séquence complète excursion ≥ `s_min·A` au-delà du niveau, puis
  réintégration sous `N_sweep` barres. `s_min = 0,3` et `N_sweep = 3` sont gelés.
  **`N_sweep` ne s'applique qu'au sweep direct** — mèche ou clôture au-delà de `L` **sans**
  `BREAK` validé — et se compte depuis la **première barre dont l'extrême dépasse `L`**. Le
  chemin « breakout raté » du §7.2 a son propre décompte, celui de `N_hold`.
- **réintégration** : la première clôture M5 revenue du côté d'origine du niveau.
- **décélération** : `d·v[t] < d·v[t−1]`, comparaison à la barre précédente, sans seuil.
- **flip** : le changement de signe de l'accélération, `d·a ≤ −a_min`. Le même `a_min` sert à
  armer (§7.1) et à retourner : c'est le paramètre le plus chargé de la spec (annexe A.2).
- **Extension VWAP** : `|C − VWAP| ≥ 1 · ATR_H1` au moment du sweep. **Tracée, mesurée,
  NON bloquante** — décision explicite du mandat. Elle sert à la décomposition « reversals
  étendus contre non étendus » du rapport, jamais à filtrer.

### 7.4 Événements

L'automate émet exactement : `ARM`, `BREAK`, `SWEEP`, `ENTRY`, `EXIT`, `CANCEL`. Un état n'est
jamais changé sans événement correspondant dans la trace (§13).

## 8. Règle R

```
s     := spread de la tranche horaire courante (§12), en dollars
e_mid := cloture mid de la barre de decision
R     := ( |cible - e_mid| - s/2 ) / ( |e_mid - stop| + s/2 )
ENTREE si R >= R_min,  R_min = 1   (gele)
```

**Pourquoi `s/2` et une seule fois.** L'entrée se fait à `mid ± s/2` (§9) : c'est la seule
demi-fourchette payée à l'ouverture. La sortie, elle, se fait **au prix du niveau** — cible ou
stop — dès que le côté déclencheur l'atteint ; le franchissement du côté opposé au trade
n'ajoute pas de coût supplémentaire à modéliser dans `R`. Facturer `s` entier des deux côtés,
comme le faisait la formule initiale du plan, comptait donc le spread deux fois.

`R` est calculé **à la décision**, à partir de `e_mid` ; il est tracé sous `r_est`. Le `R`
réalisé peut différer (gap, §9) — cela ne rouvre pas la décision.

**Obligation de publication.** La campagne in-sample doit publier le **taux de rejet par la
règle R, par année** : nombre de candidats à `R < 1` sur nombre de candidats. La note de
faisabilité (§2) montre que ce filtre ne mord quasiment jamais sur l'historique ; le rapport
client ne doit donc pas le présenter comme un filtre actif sans ce chiffre à l'appui.

## 9. Exécution

| poste | règle |
|---|---|
| instant | `fill` à l'**open de la barre M5 suivant la décision** |
| côté | achat à `mid + s/2`, vente à `mid − s/2` |
| résolution stop/cible | sur les barres **M1** à l'intérieur de la barre M5 |
| double contact dans une même M1 | **le stop l'emporte** (règle pessimiste, sans exception) |
| gap | si l'open dépasse déjà le stop ou la cible, fill à l'**open**, pas au niveau théorique |
| cooldown | **6 barres M5** par niveau après un `EXIT` ou un `CANCEL` sur ce niveau |
| position | **une seule à la fois**, tous scénarios confondus |

La règle pessimiste du double contact est la raison pour laquelle l'architecture Python passe
par un registre d'ordres et `Portfolio.from_orders` sur l'index M1 : `from_signals` ne la
garantit pas. QC la gère à la main dans `on_data`.

## 10. Sorties temporelles

```
DUREE_MAX     := 48 barres M5 (4 heures)      # gele
CLOTURE_JOUR  := 16:55 New York, inconditionnelle
FENETRE_INTERDITE := aucune entree entre 16:30 et 18:15 New York
```

Les **sorties** possibles, et leur `exit_reason` dans la trace : `STOP`, `TARGET`, `TIME`
(48 barres), `SESSION` (16:55), et rien d'autre. Aucune position n'est tenue pendant la coupure
quotidienne.

⚠️ **À divulguer dans le rapport client.** La fenêtre interdite mord des deux côtés de la
coupure : la dernière entrée d'une séance a lieu avant 16:30 et la première de la suivante à
18:15 New York. Le mandat dit « M5 toute la journée » ; la stratégie ne trade donc pas
l'ouverture de séance asiatique. C'est un écart au brief, assumé et mesuré (part des séances
et des croisements x10 tombant dans la fenêtre interdite, à publier).

## 11. Dimensionnement

```
f      := 0,005                      # 0,5 % de l'equite, gele
f      := 0,0025  si ctx_dxy < 0     # DXY adverse : risque * 0,5 (§6)
lots   := arrondi_inferieur_0,01( f * equite / ( |e - stop| * 100 ) )
```

`100` est la taille de contrat XAUUSD (`contract_size = 100`,
`data/broker/symbols_catalog_2026-07-28.csv`). L'arrondi au pas de 0,01 lot est **inférieur**,
jamais au plus proche : la logique est celle de `LotsForRisk`
(`src/mt5/Include/FxRiskManager.mqh:122-128`), que les trois moteurs doivent reproduire y
compris dans son arrondi. Un lot calculé sous `volume_min = 0,01` annule le trade.

## 12. Modèle de coûts

| poste | valeur | application |
|---|---|---|
| spread de référence | **0,29 $** (`spread_current = 290` points × `point = 0,001`) | catalogue broker 2026-07-28 |
| spread retenu | médiane par **heure New York × année**, `costs_xau_intraday.yml` (lot D) | à chaque entrée et chaque sortie |
| repli si l'export MT5 échoue | 0,29 $ constant, `source: catalog_snapshot` | + sensibilité ×2 obligatoire |
| avant 2022-11 | extrapolation au prorata du prix, `source: extrapolated` | marquée comme telle |
| slippage | 0 / 0,05 / 0,10 $ en sensibilité | entrées **et** sorties |
| swap | non modélisé (aucune position tenue la nuit, §10) | — |
| frais | 0 | — |

`costs.yml:151-153` reste intact : la stratégie 1 ne doit rien voir de ce chantier.

Le spread vaut de **5 % à 50 % d'un ATR M5** selon l'année (note de faisabilité §1). Ce n'est
pas un détail d'implémentation, c'est le premier terme de l'équation.

## 13. Contrat de trace événementiel et tolérances

Chaque moteur émet un CSV, **une ligne par événement**, colonnes dans cet ordre exact :

```
ts_decision,event,scenario,level,d,atr,v,m,a,ctx_ema,ctx_vwap,ctx_dxy,vwap,ema50_h1,dxy,
stop,target,r_est,fill_px,exit_px,exit_reason
```

| colonne | définition | format |
|---|---|---|
| `ts_decision` | clôture M5 de la décision, **UTC** | `YYYY-MM-DD HH:MM:SS` |
| `event` | `ARM` / `BREAK` / `SWEEP` / `ENTRY` / `EXIT` / `CANCEL` | chaîne |
| `scenario` | `BREAK_LONG` / `BREAK_SHORT` / `REV_LONG` / `REV_SHORT` / vide | chaîne |
| `level` | niveau x10 concerné (§3) | flottant, 3 décimales |
| `d` | direction d'approche | `-1` / `+1` |
| `atr`, `v`, `m`, `a` | §4, §5 | flottant, 6 décimales |
| `ctx_ema`, `ctx_vwap`, `ctx_dxy` | §6.4, signés dans le **sens du trade** `q` | `-1` / `0` / `+1` |
| `vwap`, `ema50_h1`, `dxy` | valeurs brutes vues à la décision | flottant, 6 décimales |
| `stop`, `target`, `r_est` | §7, §8 | flottant, 6 décimales |
| `fill_px`, `exit_px` | prix effectifs, vides hors `ENTRY`/`EXIT` | flottant, 3 décimales |
| `exit_reason` | `STOP` / `TARGET` / `TIME` / `SESSION`, vide sinon | chaîne |

Séparateur `,`, point décimal, en-tête obligatoire, une ligne par événement même sans entrée.
Émission : vbt `emit_event_trace`, QC ObjectStore **et** log préfixé `TRACE,` (l'ObjectStore
n'est pas exportable par API sur ce compte), MT5 `Common\Files\x10_trace.csv`.

**Échelle de lecture** — on descend jusqu'au premier barreau qui casse ; un écart au barreau
*N* rend les barreaux au-delà ininterprétables.

| barreau | objet comparé | si ça casse, la cause est |
|---|---|---|
| 1 | barres M5 | bornes, fuseau, calendrier (§2) |
| 2 | indicateurs `atr,v,m,a,vwap,ema50_h1,dxy` | lissage, warmup, causalité H1 (§4, §5, §6) |
| 3 | événements `ARM/BREAK/SWEEP/CANCEL` | seuils, zone, maintien, cooldown (§7) |
| 4 | entrées `ENTRY` | règle R, fenêtres horaires, position unique (§8, §10) |
| 5 | exécution `EXIT`, PnL | côté bid/ask, double contact, gap, lots, coûts (§9, §11, §12) |

**Tolérances par paire de moteurs**, figées ici, vérifiées par
`scripts/reconcile_x10_events.py` (qui réutilise `Tolerance`, `TOL_SAME_DATA`,
`TOL_BROKER_FEED`, `load_trace`, `compare_rung` de `scripts/reconcile_three_way.py:59-109`) :

| comparaison | données | appariement des entrées attendu |
|---|---|---|
| Python (données MT5) ↔ EA MT5 | identiques | **≥ 95 %** |
| Python (données QC) ↔ QC | identiques | **≥ 98 %** |
| Python / QC ↔ MT5 | différentes | **≥ 70 %**, écart attribué (mid contre bid, niveaux frôlés, flux) |

Une divergence **non attribuée** sur plus de **5 % des trades** bloque la lecture hors
échantillon. Viser l'égalité avec MT5 serait le signe qu'on a idéalisé le backtest broker, pas
qu'on a réconcilié quoi que ce soit ; un écart n'est un échec que s'il reste inexpliqué.

## 14. Environnement par moteur

| moteur | environnement | points à épingler |
|---|---|---|
| vbt | `uv` + lock du dépôt : `vectorbtpro` au commit `f0de7dcb`, pandas/numpy/numba/pyarrow aux versions des baselines | un environnement non épinglé a déjà périmé neuf baselines de tests |
| QuantConnect | OANDA margin, UTC, `ConstantFeeModel(0)`, slippage constant **entrées et sorties**, `Resolution.MINUTE`, lecture via `slice.quote_bars` (jamais `data.bars`), `QuoteBarConsolidator` 5 min et 1 h sur l'or et les quatre jambes | miroir local `src/qc/xau_x10/` faisant foi ; aucun attribut `self.symbol` |
| MT5 | terminal broker, `XAUUSD.c`, période M5, `--model 1` (OHLC M1), fenêtre 2022-11-04 → 2025-12-31 | `--model 1` interpole les fills : les chiffres sont un **majorant** ; vérifier dès le premier run que le spread du tester n'est pas nul ; export de barres plafonné à ~100 000 lignes |

Toute mesure publiée indique la version du moteur qui l'a produite.

---

## Annexe A — Paramètres

### A.1 Gelés (ne bougent pas, quel que soit le résultat)

| paramètre | valeur | §|
|---|---|---|
| période ATR (Wilder, M5) | 14 | §4 |
| période EMA (H1) | 50 | §6 |
| horizon de vitesse / d'accélération | 3 barres | §5 |
| horizon de momentum | 12 barres | §5 |
| seuil de momentum à l'armement | 0 (test de signe, `d·m > 0`) | §7.1 |
| `N_arm` (durée de vie de l'armement) | 12 barres | §7.1 |
| `k_b` (force de cassure) | 0,25 | §7.2 |
| `N_hold` (maintien) | 3 barres | §7.2 |
| `N_sweep` (réintégration) | 3 barres | §7.3 |
| `s_min` (profondeur du sweep) | 0,3 | §7.3 |
| `v_min` (vitesse minimale) | 0,2 | §7.1 |
| risque par trade | 0,5 % de l'équité (0,25 % si DXY adverse) | §11 |
| durée maximale | 48 barres M5 | §10 |
| `R_min` | 1 | §8 |
| cooldown | 6 barres par niveau | §9 |
| garde VWAP | 12 barres de séance | §6 |

### A.2 Grille — 27 configurations, et rien d'autre

| axe | valeurs |
|---|---|
| `z` (largeur de zone, en ATR) | 0,5 · 1 · 1,5 |
| `a_min` (seuil d'accélération) | 0,1 · 0,2 · 0,3 |
| `k_s` (stop du breakout, en ATR) | 0,75 · 1 · 1,5 |

3 × 3 × 3 = **27 configurations**.

Deux limites connues de cette grille, assumées pour tenir le budget de 27 configurations, et
**à publier telles quelles dans le rapport** :

- **`a_min` règle à la fois l'armement (§7.1) et le flip (§7.3).** Un seul curseur déplace
  simultanément la sensibilité des breakouts et celle des reversals ; l'axe n'est donc pas
  interprétable scénario par scénario.
- **`k_s` n'agit que sur les breakouts.** Le stop du reversal vaut `extrême + 0,5·A`, constante
  gelée. Un tiers de la grille ne concerne que la moitié des scénarios.

### A.3 Budget d'essais — famille `xau_x10`

| poste | essais |
|---|---|
| grille (`config_key = "xau_x10:grid27:v1"`) | 27 |
| ablations EMA50 / VWAP / DXY (`"xau_x10:ablations:v1"`) | 7 |
| réserve | 6 |
| **plafond** | **40** |

Ce plafond est déclaré dans `tests/test_trials_matches_notes.py` et déflate le Sharpe publié
(`distinct_trials("xau_x10")`). Le dépasser n'est pas une erreur de procédure : c'est une
invalidation du DSR publié.

---

## Annexe B — Checklist des termes du mandat

Chaque terme du brief client (§0) renvoie à sa définition opposable. Un terme sans § est un
terme non spécifié, donc interdit d'usage dans le rapport.

| # | terme | défini en |
|---|---|---|
| 1 | **vitesse** | §5 |
| 2 | **momentum** | §5 (définition), §7.1 (condition `d·m > 0`) |
| 3 | **accélération** | §5 |
| 4 | **zone** | §7.1 (`z·A`) |
| 5 | **clôture réelle** | §7.2 |
| 6 | **maintien** | §7.2 (`N_hold`) |
| 7 | **retest** | §7.2 (tracé, non exigé) |
| 8 | **sweep** | §7.3 (direct), §7.2 (par breakout raté) |
| 9 | **réintégration** | §7.3 (`N_sweep`) |
| 10 | **flip** | §7.3 (`d·a ≤ −a_min`) |
| 11 | **score de contexte** | §6.4 (`ctx_ema`, `ctx_vwap`, `ctx_dxy`, signés dans le sens du trade) |
| 12 | **stop** | §7.2 (breakout), §7.3 (reversal) |
| 13 | **R** | §8 |
| 14 | **séance** | §2 |
| 15 | **sorties** | §10 (`STOP`/`TARGET`/`TIME`/`SESSION`, §9 pour leur résolution) |
| 16 | **dimensionnement** | §11 |

**Aucun point n'est laissé ouvert.** Les douze questions soulevées à la rédaction ont été
tranchées le 2026-09-21 et intégrées ci-dessus : momentum rendu opérant (§5, §7.1), scores de
contexte évalués dans le sens du trade (§6.4), VWAP sorti du contexte H1 (§6.2), formule `R`
sans double facturation (§8), chemin « breakout raté » rendu normatif (§7.2), cycle de vie de
l'armement fermé (§7.1). Les limites conservées volontairement — double rôle de `a_min`, `k_s`
limité aux breakouts, `R ≥ 1` quasi inopérant, fenêtre 16:30-18:15 — sont des **obligations de
publication**, pas des zones grises : annexe A.2, §8 et §10.
