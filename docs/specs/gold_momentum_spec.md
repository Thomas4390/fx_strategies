# Spécification — sleeve Gold Momentum

> **Statut : source de vérité.** Ce document prime sur les trois implémentations
> (vbt `src/strategies/gold_momentum.py`, QuantConnect projet `34489845`, MQL5
> `FxSleeveGoldMomentum.mqh`). Toute divergence entre un moteur et cette spec est un
> défaut du moteur, pas de la spec. Si la spec est jugée fausse, on la corrige ici
> **d'abord**, puis on propage — jamais l'inverse.
>
> Établie le 2026-07-25 par lecture de l'implémentation vbt de référence
> (commit `3928ad6`), avec les valeurs vérifiées par exécution.

Le pseudo-code est volontairement neutre : pas d'API de moteur, indices explicites.
`t` désigne une **séance** (barre journalière), `t=0` la première séance disponible.

---

## 1. Univers et données

| | |
|---|---|
| Instrument | XAUUSD (or spot / CFD selon le moteur) |
| Fréquence de décision | journalière |
| Période de référence | 2019-01-01 → date du jour |
| Source vbt/QC | `data/XAU-USD_minute_qc.parquet`, exporté de QuantConnect |
| Source MT5 | flux broker SquaredFinancial (**différent par construction**) |

vbt et QC partagent donc les mêmes données ; MT5 non. Cette asymétrie fixe deux cibles
de réconciliation distinctes — voir §9.

## 2. Borne de journée et fuseau — **point de divergence n°1**

```
REFERENCE_TZ  := America/New_York, horloge locale, sans information de fuseau
DAY_BOUNDARY  := minuit dans REFERENCE_TZ
close[t]      := dernier prix observé dans [minuit_t, minuit_{t+1})
```

L'implémentation vbt convertit l'index minute en heure de New York **naïve**, puis agrège
par jour calendaire (`resample "1D"`, agrégat `last`). Les séances sans observation sont
supprimées (`dropna`) : il n'y a **pas** de barre week-end.

⚠️ **QuantConnect ne borne pas ainsi.** Les barres Daily d'un CFD OANDA suivent la
convention 17:00 New York. Une séance QC datée `d` ne couvre donc pas le même intervalle
qu'une séance vbt datée `d`. C'est le barreau 1 de la réconciliation, et il doit être
mesuré avant toute analyse des barreaux suivants.

⚠️ **MT5 borne en heure serveur du broker**, distincte des deux précédentes.

## 3. Score de momentum

```
LOOKBACKS := [40, 60, 120, 250]        # séances
score[t]  := moyenne sur N dans LOOKBACKS de  sign( close[t] / close[t-N] - 1 )
```

Propriétés à respecter :

- `sign(0) = 0`. Une variation exactement nulle ne vote ni pour ni contre.
- Chaque horizon pèse **1/4**, sans pondération ni sélection. Le score vit dans
  `[-1, +1]` et vaut `+1` quand les quatre horizons s'accordent à la hausse.
- `score[t]` est **indéfini** tant que `t < 250` : la moyenne n'est calculable que
  lorsque les quatre horizons le sont. Les séances antérieures sont exclues du backtest,
  elles ne valent pas 0.
- Le score utilise `close[t]`, **pas** `close[t-1]`. C'est un choix assumé, et il n'est
  pas neutre : combiné au §6 il produit une exécution idéalisée. Voir §6.

## 4. Volatilité réalisée et levier vol-cible

```
VOL_WINDOW   := 21                     # séances
ANN_FACTOR   := 252
TARGET_VOL   := 0.25
MAX_LEVERAGE := 3.0
VOL_FLOOR    := 0.01
ret[t]       := close[t] / close[t-1] - 1
sigma[t]     := ecart_type( ret[t-20 .. t], ddof=1 ) * racine(ANN_FACTOR)
lev_brut[t]  := min( TARGET_VOL / max(sigma[t], VOL_FLOOR), MAX_LEVERAGE )
leverage[t]  := lev_brut[t-1]          # décalage causal d'UNE séance
                si indéfini  ->  1.0
```

Trois points qu'aucune implémentation ne doit modifier en silence :

- **`ddof=1`** (écart-type d'échantillon). `ddof=0` change `sigma` de ~2.4 % sur 21 points.
- **`sigma[t]` exige 21 rendements complets** ; pas de fenêtre partielle en début
  d'échantillon. La valeur est indéfinie avant, et `leverage` retombe alors sur `1.0`.
- **Le décalage d'une séance sur le levier est obligatoire** et n'existe pas sur le score.
  Cette asymétrie est intentionnelle côté vbt : le dimensionnement n'utilise que de
  l'information close de la veille, le signal utilise le close du jour.

⚠️ **`VOL_FLOOR` vaut 0.01 ici, contre `max(sigma, 0.05)` côté QuantConnect.** Le plancher
ne mord que si la volatilité annualisée de l'or tombe sous 5 %, ce qui est rare, mais
l'écart doit être aligné ou assumé explicitement, pas découvert en aval.

## 5. État visé et signaux

```
allow_short := faux              # posture par défaut
long_ok[t]  := score[t] > 0
short_ok[t] := score[t] < 0  si allow_short, sinon faux

entry_long[t]  := long_ok[t]  ET NON long_ok[t-1]
exit_long[t]   := NON long_ok[t]  ET long_ok[t-1]
entry_short[t] := short_ok[t] ET NON short_ok[t-1]
exit_short[t]  := NON short_ok[t] ET short_ok[t-1]
```

Signaux **sur transition d'état**, pas sur état. La position est ouverte au passage de
`score` au-dessus de zéro et tenue jusqu'au passage en dessous : une séance à score
positif au milieu d'une tenue ne produit **aucun ordre**.

`long_ok[-1]` est considéré faux : un score positif à la première séance exploitable
déclenche une entrée.

Le côté court est désactivé par défaut. L'or porte une dérive positive structurelle ; un
short soutenu combat la dérive au lieu de récolter une prime.

⚠️ Une implémentation qui rééquilibre **chaque jour** au lieu de suivre les transitions
n'implémente pas cette spec. Sur QC, la variante « daily rebalance » produit 1215 ordres
contre 128 pour la variante à transitions — ce n'est pas un détail d'exécution mais une
stratégie différente.

## 6. Instant de décision et instant d'exécution — **point de divergence n°2**

```
decision[t] := au close de la séance t, à partir de close[t]
fill[t]     := au close de la séance t, au prix close[t]        # convention vbt actuelle
```

**C'est une idéalisation, et elle doit être traitée comme telle.** La décision consomme
`close[t]` et l'exécution a lieu à ce même `close[t]` : aucun délai entre l'observation du
prix et l'ordre. Aucun des deux autres moteurs ne fait cela — QC remplit au **T+1 open**,
MT5 au prix de marché à l'ouverture de la barre suivante.

Cette convention est **conservée en l'état pour l'instant** et constitue un poste
d'attribution connu de l'écart vbt ↔ QC (barreau 4). La trancher — aligner vbt sur le
T+1 open, ou conserver et quantifier — est une décision ouverte ; jusqu'à ce qu'elle soit
prise, aucune implémentation ne doit changer sa convention unilatéralement.

## 7. Dimensionnement de la position

```
poids_cible[t] := leverage[t]        # fraction de la valeur du portefeuille
position[t]    := poids_cible[t] * equity[t] / close[t]      # en unités
```

La position visée est la valeur du portefeuille multipliée par le levier vol-cible. Avec
`TARGET_VOL = 0.25` et `MAX_LEVERAGE = 3.0`, le levier médian mesuré vaut **2.007**, et
l'exposition brute moyenne **189.67 % lorsque la sleeve est en position** — elle l'est
54.1 % du temps.

**Contrôle de conformité** : la volatilité réalisée de la sleeve doit tomber dans
**25 % ± 3 pp**. Mesure de référence vbt au 2026-07-25 : **23.74 %**, recoupée par trois
méthodes indépendantes (23.74 / 23.74 / 23.86 %).

> Note historique, pour ne pas rouvrir un faux problème : un « défaut de sizing » a été
> diagnostiqué puis **réfuté** (`size_type="percent"` plafonnerait l'ordre au cash
> disponible). Les chiffres qui l'appuyaient provenaient d'un run à levier 1×. Détail dans
> `docs/research/gold_2026-07-25_momentum_sizing.md` §7.

## 8. Modèle de coût

| poste | valeur | application |
|---|---|---|
| capital initial | 1 000 000 | — |
| slippage | **1 bp de notionnel** | à **chaque** ordre, entrée **et** sortie |
| frais | 0 | — |
| swap / financement | non modélisé côté vbt et QC | poste d'écart MT5 |
| stop de sécurité | aucun par défaut (`sl_stop = None`) | — |

Valeurs vbt issues de `vbt.yml`, appliquées à l'import de `framework.project_config`.

⚠️ Le slippage porte sur **tous** les ordres. Une implémentation qui ne l'applique qu'aux
entrées sous-estime le coût d'un facteur deux — c'est exactement le bug n°3 du portage QC
précédent (`docs/quantconnect_validation_report.md` §5.3).

## 9. Trace journalière — contrat de réconciliation

Chaque moteur émet un CSV, une ligne par séance **exploitable** (score défini), colonnes
dans cet ordre exact :

```
date,close,score,target_weight,position_units,equity
```

| colonne | définition | format |
|---|---|---|
| `date` | séance, dans REFERENCE_TZ | `YYYY-MM-DD` |
| `close` | `close[t]` du §2 | flottant, 6 décimales |
| `score` | `score[t]` du §3 | flottant, 6 décimales |
| `target_weight` | `poids_cible[t]` du §7, 0 hors position | flottant, 6 décimales |
| `position_units` | unités détenues **après** exécution de la séance | flottant, 6 décimales |
| `equity` | valeur du portefeuille à la clôture de `t` | flottant, 2 décimales |

Séparateur `,`, point décimal, pas de séparateur de milliers, en-tête obligatoire.

⚠️ **`target_weight` est la cible instantanée, pas le poids détenu.** Elle est recalculée
chaque séance puisque `leverage[t]` évolue avec la volatilité, mais elle n'est **actionnée
qu'aux transitions du §5** : hors transition, `target_weight` bouge tandis que
`position_units` reste constant. C'est le comportement attendu, pas une dérive. La colonne
sert à vérifier que les trois moteurs calculent le même levier (barreau 3),
indépendamment de la question de savoir quand ils l'exécutent (barreau 4).

**Échelle de lecture.** On descend jusqu'au premier barreau qui casse ; un écart au
barreau *N* rend tous les barreaux au-delà ininterprétables.

| barreau | colonne | si ça casse, la cause est |
|---|---|---|
| 1 | `close` | bornes de barre, fuseau, calendrier (§2) |
| 2 | `score` | indexation du lookback, warmup (§3) |
| 3 | `target_weight` | fenêtre/ddof de sigma, plancher, décalage causal (§4, §7) |
| 4 | `position_units` | timing de fill, arrondi de lots, marge (§6) |
| 5 | `equity` | coûts, spread, swap (§8) |

**Tolérances.** Elles diffèrent selon la paire de moteurs, et c'est le point central :

| paire | données | barreaux 1-3 | barreaux 4-5 |
|---|---|---|---|
| vbt ↔ QC | identiques | écart relatif ≤ 1e-6 | ≤ 2 % sur CAGR/vol/maxDD, ≤ 0.05 sur le Sharpe |
| vbt ↔ MT5 | différentes | écart **borné et attribué**, jamais nul | idem, chaque poste chiffré |

Viser l'égalité avec MT5 serait le signe qu'on a idéalisé le backtest broker, pas qu'on a
réconcilié quoi que ce soit.

## 10. Environnement

Les résultats ne sont reproductibles qu'à environnement fixé. Le lock du dépôt épingle
`vectorbtpro` au commit `f0de7dcb` et pandas/numpy/numba/pyarrow aux versions ayant servi
aux baselines. Un environnement non épinglé a déjà périmé silencieusement neuf baselines
de tests — voir `docs/research/gold_2026-07-25_momentum_sizing.md` §9.

Toute mesure publiée doit indiquer la version du moteur qui l'a produite.
