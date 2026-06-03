# FxMultiSleeve — Correctif : « FRED fetch failed » / erreur 4014

> Note de dépannage prête à transmettre au client. Le diagnostic est issu de la
> capture du terminal MT5 (onglet *Experts*) montrant des lignes répétées
> `CMacroSourceFRED::WebRequest err=4014` puis `MR sleeve disabled`.

---

## En une phrase

**Ce n'est pas votre clé API.** MetaTrader 5 bloque l'appel vers FRED parce que
l'adresse `https://api.stlouisfed.org` n'est **pas autorisée** dans les options du
terminal. Tant que ce n'est pas corrigé, votre clé — même parfaitement valide —
n'est jamais utilisée. C'est exactement ce que signifie l'erreur **4014**
(*Function is not allowed for call*).

Conséquence : seul le sleeve **MR Macro** (le filtre macro) est désactivé. Les deux
autres sleeves (TS Momentum et RSI Daily) continuent de fonctionner normalement.

---

## Le correctif (≈ 2 minutes)

1. Dans MT5 : **Outils → Options → onglet « Expert Advisors »**.
2. Cocher **« Allow WebRequest for listed URL »** (Autoriser WebRequest pour les
   URL listées).
3. Dans la liste, ajouter **exactement** cette adresse — sans `/` ni rien après :

   ```
   https://api.stlouisfed.org
   ```

4. **Fermer complètement MT5, puis le relancer.** Ensuite, sur le graphique :
   clic-droit → **Expert Advisors → Remove**, puis re-glissez `FxMultiSleeve`
   dessus. (MT5 ne relit la liste blanche qu'au redémarrage + ré-attachement —
   c'est le piège classique.)
5. Vérifier que le bouton **Algo Trading** est bien vert.

---

## Vérifier que c'est réglé

Dans l'onglet **Experts**, au redémarrage de l'EA vous devez maintenant voir :

```
CMacroFilter::NATIVE OK: spread=… unemp_rising=… macro_ok=…
[INIT][INFO] Macro source=native spread=… macro_ok=…
```

➡️ Plus **aucune** ligne `err=4014` : le problème est résolu, le sleeve MR Macro
est réactivé (il se réactive seul, sans rien fermer/rouvrir d'autre).

### Si — et seulement si — une nouvelle ligne apparaît

| Ce que vous voyez | Ce que ça veut dire | Quoi faire |
|---|---|---|
| `WebRequest HTTP 400` ou `403` | *Là* c'est la clé API (invalide, mal copiée, ou quota) | Regénérer la clé sur https://fredaccount.stlouisfed.org/apikeys et la recopier dans `Common\Files\fred_api_key.txt` (une seule ligne, **sans espace ni saut de ligne**, encodage UTF-8) |
| `err=4014` encore | La liste blanche n'a pas été relue | Vérifier l'URL exacte (pas de faute, pas de `/` final), **redémarrer MT5**, ré-attacher l'EA |
| `Aucune cle API FRED…` | Le fichier clé est absent ou vide | Créer/remplir `Common\Files\fred_api_key.txt` |

---

## Bon à savoir

- **Aucune réinstallation nécessaire.** L'EA réessaie tout seul toutes les 60 s ;
  une fois l'URL autorisée, le sleeve MR Macro repart sans manipulation
  supplémentaire.
- **Seulement en live.** Cette autorisation ne sert qu'en compte réel/démo. En
  *Strategy Tester* (backtest), `WebRequest` est de toute façon désactivé par MT5 :
  l'EA bascule automatiquement sur le fichier `macro_history.csv` — donc la
  whitelist n'a aucun effet là-bas et n'est pas requise pour backtester.
- **Référence** : étape documentée au §4.2 du guide d'installation. La
  documentation officielle MQL5 confirme l'exigence de la liste blanche et la
  nécessité de redémarrer le terminal après ajout.
