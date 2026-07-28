# Découpler le stop de sécurité du dimensionnement — prédictions pré-gelées

> **Date** : 2026-07-28 · **Statut** : prédictions écrites AVANT exécution
> **Holdout touched** : NO — fenêtre de vérification 2021-01-01 → 2026-04-30, celle du run
> de référence publié, en comparaison de configuration et non en sélection.

## Le défaut

`Inp_Gold_SafetySL = 0,04` est présenté dans le code comme un « stop large qui pare le gap
overnight, la sleeve sortant normalement sur retournement de signal ». Rapporté à la
volatilité de chaque instrument, cette distance unique ne veut pas dire la même chose :

| instrument | σ21 médian (annualisé) | 4σ quotidien | séances où 4σ > 4 % | `SafetySL` en σ quotidiens |
|---|---|---|---|---|
| USD-JPY | 7,59 % | 1,91 % | **2,3 %** | 8,4 σ |
| XAU-USD | 12,42 % | 3,13 % | 26,6 % | 5,1 σ |
| XAG-USD | 25,01 % | 6,30 % | **95,0 %** | **2,5 σ** |

Sur l'argent, ce n'est pas un garde-fou : c'est un stop ordinaire à 2,5 σ. Les deals du run
de référence le confirment — **6 sorties intraday sur 23** sur XAGUSD (stop touché), contre
0 sur 34 pour USD/JPY et 1 sur 35 pour l'or. Et la parité vbt↔MT5 le confirme
indépendamment : XAG-USD est la jambe la moins réconciliée du dossier (Δ Sharpe 0,61) et la
seule où **MT5 ferme plus de positions que vbt** (18 contre 15), alors que le moteur vbt n'a
aucun stop (`sl_stop=None`).

Autrement dit : **le comportement de la sleeve sur l'argent n'est pas celui que la recherche
a mesuré**, sur l'instrument qui porte 41 % du résultat net.

## La contrainte cachée

Dans `FxSleeveGoldMomentum.mqh::OpenPosition`, `sl_dist` sert **deux fois** — comme distance
de stop et comme dénominateur du dimensionnement (`LotsForRisk(symbol, risk_money, sl_dist)`,
donc notionnel ∝ 1/`sl_dist`). Élargir le stop sans découpler diviserait le notionnel de
l'argent par 6,3/4,0 = 1,57 et mêlerait deux effets dans un seul chiffre.

## La correction

```cpp
#define FX_GOLD_SL_SIGMAS 4.0   // = 0.04 / (0.16/sqrt(252)) — quotient de deux constantes publiées

sl_dist_sizing  = price * (Inp_Gold_SafetySL + slip_pct);                       // INCHANGÉ
sl_frac_protect = MathMax(Inp_Gold_SafetySL, FX_GOLD_SL_SIGMAS * sigma21 / MathSqrt(252.0));
sl_dist_protect = price * (sl_frac_protect + slip_pct);                         // stop réellement posé
```

**Aucun essai consommé.** `k = 4` n'est pas un point choisi dans une grille : c'est le
quotient de deux constantes déjà publiées du dossier — le `SafetySL` de 0,04 et la σ long
terme de l'or de 0,16 qui vit dans `ComputeSigma21`. Aucune métrique n'est optimisée, aucun
degré de liberté n'est ajouté. Logué `n=0` au registre avec cette note.

Le `MathMax` est délibérément **unilatéral** : le stop ne peut qu'être élargi, jamais
resserré, donc la correction ne peut pas créer un stop-out là où il n'y en avait pas.

## Prédictions — à confronter, pas à ajuster

Fenêtre 2021-01-01 → 2026-04-30, configuration de production intégralement épinglée.

1. **USD-JPY : identique au bit.** 4σ ≈ 1,91 % < 4 % dans 97,7 % des séances → le plancher
   `SafetySL` reste actif. Attendu : mêmes 34 sorties, mêmes horodatages, mêmes prix, mêmes
   volumes. **Contradiction si un seul deal diffère.**
2. **XAU-USD : au plus 1 trade change.** 4σ dépasse 4 % dans 26,6 % des séances, mais l'or
   n'a qu'**une** sortie intraday sur 35 (2026-04-02). Attendu : cette sortie disparaît ou
   se décale, rien d'autre. **Contradiction si plus de 2 trades diffèrent.**
3. **XAG-USD : les 6 stop-outs sont candidats à disparaître.** 4σ ≈ 6,30 % > 4 % dans 95 %
   des séances. Attendu : les 6 sorties intraday deviennent des sorties sur retournement de
   signal, le compte de trades baisse, et l'écart de parité vbt↔MT5 se **réduit** depuis
   0,61. **Contradiction si le compte de trades monte, ou si l'écart de parité s'aggrave.**
4. **Notionnel inchangé partout.** `sl_dist_sizing` n'est pas touché, donc les volumes
   d'entrée doivent être identiques au lot près sur les trois instruments. **Contradiction
   si un volume d'entrée bouge.**

## Garde à poser d'avance

Retirer 6 stop-outs perdants **améliorera** le backtest. Cette amélioration ne sera jamais
présentée comme un gain de stratégie : c'est une correction de spécification pré-enregistrée
dont l'effet pouvait aller dans l'autre sens — le stop pouvait tout aussi bien couper des
positions avant qu'elles n'empirent. Le seul critère de succès est **la conformité aux
quatre prédictions ci-dessus**, pas le signe du P&L.

## Ce qu'il ne faut PAS toucher (vérifié)

- **Plancher de vol 0,05** (`:217`) : inerte. Il ne mord que si `tv/0,05 < cap`, or
  `0,55/0,05 = 11 > 6,6` — le cap domine dans toutes les branches, y compris sur les 19,5 %
  de séances où σ21 < 0,05 sur USD/JPY. Un `Warn()` à l'Init suffit à protéger d'un futur
  retune qui réveillerait la constante.
- **Sigma fallback 0,16** (`:356`, `:360`) : inatteignable. `ComputeSigma21` n'est appelé
  qu'après un `ComputeScore` qui exige 252 barres, et qui a 252 barres en a 22. Un test
  verrouillant l'ordre des deux appels suffit.
- **`FX_GOLD_AVG_NIGHTS_HELD`** : ±0,2 % sur le dimensionnement, sous le bruit d'arrondi.

---

## Résultats — confrontation aux prédictions

Run de vérification `prod_ref_slfix`, fenêtre 2021-01-01 → 2026-04-30, configuration de
production intégralement épinglée. Portefeuille : **905 trades** (contre 909), net
**+49 143 $** (contre +47 953), Sharpe **1,01** (contre 1,00), repli d'équité **47,04 %**
(contre 47,29 %).

### Sorties intraday (stop touché) contre sorties sur retournement à 21:00

| instrument | avant | après |
|---|---|---|
| USD-JPY | 34 flips, **0 stop** | 34 flips, **0 stop** |
| XAU-USD | 34 flips, **1 stop** | 34 flips, **0 stop** |
| XAG-USD | 17 flips, **6 stops** | 19 flips, **1 stop** |

### Prédiction par prédiction

**1. « USD-JPY identique au bit » — CONTREDITE sur la forme, confirmée sur le fond.**
Les 69 deals ont des **horodatages, sens et prix strictement identiques**. Seuls les
**volumes** diffèrent, de 0,01 lot au maximum, sur 21 deals — et le premier écart
(2024-12-05) survient *après* le premier stop-out d'argent supprimé (2023-12-11).

La prédiction était trop forte parce qu'elle raisonnait comme si les trois jambes étaient
indépendantes. Elles ne le sont pas : elles partagent `SubEquity(GOLD) / n`. Supprimer une
perte sur l'argent en 2023 relève l'équité du compte, donc le budget de chaque jambe, donc
les lots — d'un cran d'arrondi. La formulation correcte aurait été « mêmes dates, sens et
prix ; volumes à ±1 cran d'arrondi par cascade de budget commun ». **La décision USD/JPY,
elle, est strictement inchangée** — ce que la prédiction visait réellement.

**2. « XAU-USD : au plus 1 trade change » — CONFIRMÉE exactement.** L'unique sortie
intraday disparaît, 35 → 34 sorties. Rien d'autre ne bouge.

**3. « XAG-USD : les 6 stop-outs candidats à disparaître » — CONFIRMÉE.** Cinq disparaissent.
Le sixième (2025-04-04) subsiste mais **décalé de 9 minutes et 0,14 point plus bas**
(14:06:40 à 30,274 → 14:15:40 à 30,130) : c'est le stop élargi touché plus tard, exactement
le comportement attendu. Sorties 23 → 20, compte de trades en baisse comme prédit.

**4. « Notionnel inchangé » — CONTREDITE sur la forme, même cause.** Les volumes d'entrée
bougent de 0,01 lot au maximum, par la même cascade de sub-equity. `sl_dist_sizing` n'a bien
pas été touché : c'est l'équité qui a changé, pas la formule.

### Attribution de l'écart — critère de résidu

| poste | delta |
|---|---|
| sleeve momentum | **+1 252,87** |
| dont XAG-USD | +676,84 |
| dont XAU-USD | +586,54 |
| dont USD-JPY | −10,51 |
| sleeves FX (cascade d'équité sur le vol-targeting global) | −62,31 |
| **total portefeuille** | **+1 190,56** |

Résidu hors sleeve momentum : **5,2 %** de l'écart total, sous le seuil de 20 % fixé au
§9 du spec. Critère rempli.

### Le chiffre qui empêche de sur-vendre la correction

Les six stop-outs d'argent coûtaient **−3 873 $** ; il en reste un à **−1 262 $**, soit
**2 611 $ de pertes évitées**. Or le gain net de la jambe argent n'est que de **+677 $**.

L'écart n'est pas une erreur : les positions qu'on cesse de couper **continuent de perdre**
avant leur retournement de signal. Le stop, en coupant tôt, en sauvait une partie. Le bilan
reste positif, mais il vaut le quart de ce que la simple somme des pertes évitées laisserait
croire — et il aurait pu être négatif. C'est la raison pour laquelle le critère de succès
était la conformité aux prédictions, jamais le signe du P&L.

### Le critère de parité XAG : moitié tenu, moitié contredit

La prédiction 3 posait deux conditions : le compte de trades devait baisser, et l'écart de
parité vbt↔MT5 devait se réduire depuis 0,61. Re-mesuré en sleeve isolée sur
2022-11-04 → 2025-12-31 :

| | avant | après | prédit |
|---|---|---|---|
| trades MT5 / vbt | 18 / 15 (−16,7 %) | **16 / 15 (−6,3 %)** | baisse — **tenu**, et désormais dans la tolérance ±10 % |
| Δ Sharpe | 0,61 | **0,66** | réduction — **contredit**, il s'aggrave de 0,05 |
| Sharpe MT5 isolé | 0,44 | 0,39 | — |

**Le second critère est contredit et je ne le réécris pas après coup.** Sur cette fenêtre de
trois ans, retirer les stops dégrade le Sharpe isolé de l'argent : les positions qu'on cesse
de couper continuent de perdre plus souvent qu'elles ne se redressent. C'est le même
mécanisme que le §précédent, vu à l'échelle d'un ratio plutôt que d'un cumul de P&L — et
c'est le rappel que le stop, en coupant tôt, faisait parfois du bon travail.

**La correction est conservée quand même**, pour une raison qui ne dépend pas de cette
métrique : le moteur de recherche n'a **aucun stop** (`sl_stop=None`), et le moteur
d'exécution en avait un qui déclenchait sur **26 %** des trades d'argent. Après correction
il déclenche sur **5 %**. Ce qui est corrigé, c'est un écart de spécification entre les deux
moteurs — un backtest de recherche qui ne modélise pas ce que l'exécution fait réellement.
Ce motif tient indépendamment du signe de 0,05 de Sharpe sur une fenêtre de trois ans.

Reste ouvert, et honnêtement non résolu : **l'argent demeure la jambe la moins bien
réconciliée du dossier** (0,66 contre 0,35 pour l'or et 0,20 pour l'USD/JPY). Le compte de
trades est désormais aligné, donc le résidu ne vient plus des sorties : il vient du
dimensionnement en lots et du levier non décalé, comme sur l'or, mais amplifié par une
volatilité deux fois supérieure. À instruire si l'argent devait peser davantage.
