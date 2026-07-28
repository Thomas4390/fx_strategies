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
