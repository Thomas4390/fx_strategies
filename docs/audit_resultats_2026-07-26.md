# Résultats de l'audit adverse — republication du rapport client (2026-07-26)

> Audit exécuté le 2026-07-26 en réponse à `docs/audit_rapport_client_2026-07-26.md`.
> Périmètre : commits `43ed505` → `87756cf` sur `main`, et les quatre livrables client.
> Méthode : reproduction systématique. Chaque constat porte la commande exécutée.

**Verdict d'ensemble.** Le socle chiffré est solide : les quatre assertions de la
section A se reproduisent, dont une **bit-à-bit**. La chaîne de génération dérive
réellement de la configuration. En revanche, **le rapport technique livré au client
publie deux verdicts contradictoires sur son test le plus grave**, et le guide
d'installation documente un mécanisme de protection à l'envers. Ces deux défauts
étaient invisibles depuis le prompt d'audit, qui ne les mentionne pas.

> ## ✅ Suites données — 2026-07-26
>
> **Les quatre défauts graves (§1 à §4) ont été corrigés et les quatre PDF recompilés.**
> Les **trois correctifs de fond** identifiés par l'audit ont également été appliqués : la
> légende calculée (§2), la génération de la table de paramètres du guide d'installation
> (§3), et la fermeture du test qui sautait (§6). Le détail et sa vérification sont en fin
> de document, section « Correctifs appliqués ».
>
> **2026-07-27 — tous les défauts de l'audit sont désormais fermés**, §5 à §12 compris,
> chacun avec son test vérifié par mutation. Voir « Fermeture des défauts §5 à §12 » en
> fin de document. Trois défauts supplémentaires, non vus par l'audit, ont été trouvés
> pendant la correction et fermés aussi.

---

## Défauts, par gravité décroissante

### 1. Le rapport technique s'auto-contredit sur le sur-ajustement — GRAVE

**Ce qui est faux.** `sections/10_advanced_robustness.tex` affirme simultanément que
la probabilité de sur-ajustement vaut 0,31 (« confortable », « les cinq tests
convergent vers un verdict favorable ») et qu'elle vaut 0,532 (« c'est un échec »).

**La preuve.**
```bash
pdftotext reports/latex_report/main.pdf - | grep -c "0\.31"    # → 6
pdftotext reports/latex_report/main.pdf - | grep -c "0\.532"   # → 5
```
Passages périmés, tous dans le PDF livré (`main.tex:139` inclut bien cette section) :

| Ligne | Contenu | Valeur à jour |
|---|---|---|
| 64 | « Cinq tests… tous convergent vers le même verdict favorable » | 5 sur 6, un échoue |
| 87 | Haircut Sharpe $=0.84$, survie 87 % | 1.035 / 79.29 % |
| 88 | MinBTL $=3.5$ ans, SR cible $0.97$, « on dispose de 7 ans » | 2.10 ans / 1.31 / 7.5 ans |
| 89 | « PBO $=0.31$ … $0.31$ est confortable » | 0.532, échec |
| 131 | « 3 926 partitions sur 12 870 … confortablement sous le seuil » | 6 850 / 12 870 |
| 137 | Légende de figure : « Le PBO de $0.31$ signifie… » | 0.532 |
| 141 | « Les cinq tests convergent … le PBO à $0.31$ confirme la stabilité » | contredit l. 175 |

La thesisbox finale (l. 175) est, elle, à jour. Le diff de `4918fd7` sur ce fichier ne
compte que 9 insertions / 7 suppressions : la réécriture a été partielle.

**Conséquence client.** Sur la seule question qui décide de la crédibilité d'un
backtest — ce résultat tient-il hors échantillon ? — le document affirme une chose
page 33 et son contraire page 36. Un lecteur attentif conclut que le rapport n'est
pas relu ; un lecteur inattentif retient le message rassurant, qui est le faux.

**Correctif proposé.** Réécrire les sept passages ci-dessus depuis les valeurs de
`report["pbo"]`, `report["haircut"]` et `report["minbtl"]`, ou les supprimer et
renvoyer à la table générée. La prose narrative de la section 10 n'est asservie à
rien : c'est la cause racine, pas le symptôme.

---

### 2. Une légende générée affirme l'inverse de sa propre table — GRAVE

**Ce qui est faux.** `tables/robustness_overfitting.tex:3` imprime « Tous les tests
convergent vers un verdict favorable » directement au-dessus de la ligne
`PBO CSCV ($n_{bins}=16$) & 0.532 & Seuil de danger $= 0.5$ *(OVERFIT)*`.

**La preuve.** La légende est un littéral dans `scripts/build_latex_report_assets.py:1866-1871` :
elle ne dérive d'aucun calcul, contrairement à celle de `robustness_verdict_summary.tex`
qui compte `n_passed`/`n_tested` correctement.

**Conséquence client.** Le défaut est plus embarrassant que le précédent : il est
*généré*, donc il reviendra à chaque exécution de la chaîne. C'est exactement ce que
la note de vault `learning-2026-07-26-une-legende-qui-affirme-un-chiffre-doit-le-calculer`,
écrite le même jour, prétend avoir tiré comme leçon.

**Correctif proposé.** Calculer la légende comme celle du verdict consolidé :
`f"{n_passed} test(s) sur {n_tested} franchissent leur seuil"`.

---

### 3. Le guide d'installation documente le plafond de marge à l'envers — GRAVE

**Ce qui est faux.** Les deux guides client donnent des valeurs opposées à celles de
l'EA compilé et du preset généré.

| Paramètre | Guides client | `FxMultiSleeve.mq5` | `write_default_preset.py` |
|---|---|---|---|
| `Inp_EnableMarginCap` | `false` | **`true`** (l. 51) | **`True`** (l. 50) |
| `Inp_MarginCapPct` | `0.70` | **`0.50`** (l. 52) | **`0.5`** (l. 51) |

**La preuve.** `client_setup_guide/main.tex:515-516` et `:811` ; `client_pedagogical_guide/main.tex:783-784` ;
`src/mt5/Experts/FxMultiSleeve.mq5:51-52`. La checklist du guide (l. 811) va plus loin :
« Cap d'utilisation de marge *désactivé* par défaut ; ré-activer manuellement si désiré,
l'auto-deleverage s'active alors $> 70\,\%$ marge. »

**Conséquence client.** Double. D'abord un mécanisme de désendettement automatique
que le client croit inactif se déclenchera, et à 50 % d'utilisation de marge et non 70 % —
il verra ses positions réduites sans comprendre pourquoi. Ensuite, la reproductibilité :
les métriques publiées (851 trades, CAGR 35,44 %) ont été produites **avec** le plafond
actif à 0,50, puisque le CLI n'override que `SymbolSuffix`, `MacroSourceMode`,
`LogVerbose`, `LogToFile` et `ExportDeals`. Un client qui suit le guide configure une
machine différente de celle qui a produit les chiffres.

**Correctif proposé.** Corriger les quatre lignes. Et surtout : le guide d'installation
ne consomme **aucun** asset généré — il est intégralement écrit à la main, donc
structurellement incapable de suivre la config. Générer sa table de paramètres depuis
`write_default_preset.PRESET_LINES` fermerait la classe entière de défauts.

> **Fait.** Les sept tables de la section 6 sont générées et testées — voir « Génération de
> la table de paramètres du guide d'installation » plus bas. Le §3 n'est plus seulement
> corrigé, il ne peut plus se reproduire silencieusement.

---

### 4. Le guide pédagogique annonce quatre moteurs et n'en décrit que trois — GRAVE

**Ce qui est faux.** La section 2 s'intitule « Architecture des **quatre** moteurs ».
Les sections 3, 4 et 5 décrivent les Moteurs 1, 2 et 3. La section 6 s'intitule
« Gestion du risque ». **Il n'y a pas de section pour le Moteur 4.**

**La preuve.**
```bash
pdftotext reports/client_pedagogical_guide/main.pdf - | grep -E "^[0-9]+ [A-ZÉ]"
#   1 Vue d'ensemble / 2 Architecture des quatre moteurs
#   3 Moteur 1 — MR Macro / 4 Moteur 2 — TS Momentum / 5 Moteur 3 — RSI Daily
#   6 Gestion du risque / 7 Coûts / 8 Paramètres / 9 Résultats / 10 Intervenir / 11 Annexes
```
Dans tout le document de 30 pages, l'or n'apparaît que deux fois : une ligne de tableau
(`Inp_AllocGoldMomentum 0,10`) et une mention en passant. Aucune figure de la sleeve or
non plus (`sleeve_gold_momentum_equity.png` n'est pas inclus).

Deux phrases résiduelles disent encore « **trois** moteurs » (l. 291 « Elle combine trois
moteurs de rendement indépendants », l. 510 « les trois moteurs sont approximativement
équipondérés »), contredisant le titre de la section 2 du même document.

**Conséquence client.** Le document dont la fonction est d'expliquer la stratégie omet
le moteur qui produit **79,8 % du résultat net** et porte l'essentiel du risque. Le
client ne peut pas comprendre ce qu'il détient. C'est aussi le document où la
contradiction interne est la plus visible pour un non-spécialiste.

**Correctif proposé.** Ajouter une section « Moteur 4 — Gold Momentum » sur le modèle
des sections 3-5, corriger les deux « trois moteurs », inclure la figure d'équité or.

---

### 5. Un script peut écraser le JSON de production avec une configuration périmée — MOYEN

**Ce qui est faux.** `scripts/generate_report_artifacts.py:398-412` contient une branche
morte qui force l'ancienne configuration dans le dict écrit vers
`results/production_report/stress_test_report.json` :

```python
"weights": dict(stress_test_combined.RECOMMENDED_CONFIG["custom_weights"])
if False
else {"MR_Macro": 0.80, "TS_Momentum_3p": 0.10, "RSI_Daily_3p": 0.10},
"target_vol": 0.28,
"max_leverage": 12.0,
```

**La preuve.** Deux scripts écrivent le même chemin avec deux schémas différents
(`stress_test_combined.py` en `__main__`, et celui-ci). Le JSON actuel est le bon.
Aucun test ne couvre le second écrivain : `test_stress_test_writes_where_the_report_reads_it`
ne teste que la présence de la chaîne du chemin dans le source de `stress_test_combined.py`.

**Conséquence client.** Latente. Si ce script est exécuté après l'autre, la table de
sensibilité et les scénarios du rapport repartent sur 80/10/10 à `target_vol=0.28`,
et `test_report_config_sync.py:89` lève un `KeyError` au lieu d'un `AssertionError`.

**Correctif proposé.** Supprimer la branche morte et faire dériver le bloc `config` de
`PRODUCTION_WEIGHTS` / `PRODUCTION_TARGET_VOL` / `PRODUCTION_MAX_LEVERAGE`, ou retirer
l'écriture concurrente.

---

### 6. Le seul test qui détecte une dérive de configuration est inactif en CI — MOYEN

**Ce qui est faux.** `test_published_stress_json_matches_the_current_config` est le seul
des six tests de `test_report_config_sync.py` qui compare les artefacts publiés à la
config courante. Il `skip` silencieusement si le JSON est absent — et `results/` est
gitignoré (`.gitignore:13`), donc il est inactif sur tout clone frais.

**La preuve.**
```bash
mv results/production_report/stress_test_report.json{,.hidden}
pytest tests/test_report_config_sync.py -q -rs
# → 5 passed, 1 skipped
#   SKIPPED [1] …:85: …stress_test_report.json absent : relancer stress_test_combined.py
```
Vérifié aussi par mutation : passer `PRODUCTION_TARGET_VOL` à 0.99 ne fait rougir que
ce test-là. Les cinq autres comparent des valeurs qui bougent ensemble.

**Correctif proposé.** Faire échouer plutôt que sauter quand le JSON manque, ou versionner
les deux JSON de référence (ils pèsent 83 Ko).

> **Fait — les deux.** Faire échouer seul aurait rendu la CI rouge sur tout clone frais ;
> c'est le versionnement qui rend l'échec tenable. Voir « Fermeture du test qui sautait »
> plus bas.

---

### 7. `test_stress_sanity.py` reste vert quand la grille change — MOYEN

**Ce qui est faux.** `assert len(sweeps) == 18` (l. 117) est un littéral qui ne dérive ni
de la grille ni de la config. Le défaut, déjà cité par le prompt d'audit, **n'a pas été
corrigé**.

**La preuve — mutation testing.** J'ai remplacé la grille `(0.20, 0.28, 0.33, 0.37, 0.42, 0.50)`
par `(0.01, 0.02, 0.03, 0.04, 0.05, 0.06)` — une grille sans aucun rapport avec la
production :
```
resultat : GREEN (attendu RED)   6 passed in 5.11s
==> ECART : le test ne protege pas ce quil pretend
```
Et git confirme que la grille a bien changé dans les commits audités
(`0.15–0.28 × 10/15/20` → `0.20–0.50 × 20/31/45`) sans que le test bronche : 6×3 = 18
des deux côtés.

**Correctif proposé.** Asserter que `PRODUCTION_TARGET_VOL` et `PRODUCTION_MAX_LEVERAGE`
figurent parmi les points balayés, plutôt que de compter les lignes.

---

### 8. Le CLI de backtest retourne 0 sur un run vide — MOYEN

**Ce qui est faux.** `run_backtest_cli.py:561-562` :
```python
has_metrics = bool(metrics.sharpe_ratio and metrics.total_trades)
return 0 if exit_code == 0 and has_metrics else max(1, exit_code)
```
Ce sont des **chaînes** extraites du HTML. Sur un run dégénéré elles valent `"0.00"` et
`"0"` — non vides, donc vraies.

**La preuve.** Le run en ticks réels a produit `Period M0 (1970.01.01 - 1970.01.01)`,
dépôt 0, 0 trade, et `EXIT=0`. Le log du tester dit `preliminary downloading of history
ticks canceled` puis `no history data, stop testing`.

**Conséquence.** Une chaîne automatisée traiterait un backtest inexistant comme valide.

**Correctif proposé.** Convertir en nombre et exiger `total_trades > 0` et une période
non dégénérée.

---

### 9. Les marqueurs « production » des figures de sensibilité sont codés en dur — MOYEN

**Ce qui est faux.** Les trois figures de `generate_weight_sensitivity_figures.py`
placent leur étoile « production » à 80/10/10 en littéral : l. 390 `ax.axvline(80, …)`,
l. 396 `argmin(|mr - 0.80|)`, l. 406 `label="80/10/10 (prod)"`, l. 497
`_barycentric_to_cartesian([0.80], [0.10], [0.10])`, l. 510 annotation `"80 / 10 / 10"`,
l. 575-577 et 589 pour le Pareto.

**La preuve — test d'idempotence (B2).** En basculant `PRODUCTION_WEIGHTS` sur
60/15/15/10 et en relançant la chaîne :
```
table named : a change
  < 80 / 10 / 10 (production) & 1.119 & 38.94 % …
  > 67 / 17 / 17 (production) & 1.142 & 42.47 % …
```
Les tables suivent. Les étoiles, non : les six littéraux subsistent.

**Conséquence.** Nulle aujourd'hui — les parts internes de production *sont* 80/10/10,
donc les figures tombent juste par coïncidence. Au premier changement de poids, elles
étiquetteraient silencieusement le mauvais point.

**Correctif proposé.** Dériver les trois marqueurs de `PRODUCTION_WEIGHTS` renormalisés
par `fx_share`, comme le fait déjà `make_weights()`.

---

### 10. Compte de tests incohérent dans le rapport — FAIBLE

`sections/10` annonce « Cette section applique **sept** tests statistiques » alors que la
table de verdicts en compte 6, dont un sans verdict. Vérifié :
`pdftotext main.pdf - | grep "sept tests"` → 1 occurrence, page 33.

---

### 11. Le CAGR hors échantillon de 161 % sans avertissement propre en §7 — FAIBLE

Le chiffre n'apparaît que dans `main.pdf`, **3 fois** — absent de la synthèse exécutive
et des deux guides. Deux occurrences sont en section 08 sous un `warningbox` explicite.
La troisième est `tables/metrics_summary.tex:9`, insérée en section 07, dont l'encadré
prévient du décalage recherche/exécution mais pas de l'annualisation d'une fenêtre de
448 barres. Portée bien plus faible que ce que suppose l'assertion C3.

---

### 12. Défauts mineurs vérifiés

| Défaut | Emplacement | Note |
|---|---|---|
| Incohérence d'unité et de signe | `mt5_reference.json` : `balance_dd_pct_daily = -0.2275` (fraction signée) voisine de `balance_dd_pct_mt5 = 6.66` et `equity_dd_pct_mt5 = 44.33` (pourcentages positifs) | Prête à la confusion ; la grandeur MT5 réellement comparable (23,37 %) n'est pas stockée |
| Aucune provenance | `mt5_reference.json` ne trace ni le CSV, ni le HTML, ni `Model`, ni les inputs, ni l'horodatage | Rien ne rattache les chiffres publiés à un run précis |
| Phrase dupliquée | `appendix_c_weight_sensitivity.tex:58` | « Les sommets correspondent aux allocations pures 100 % MR… **Les sommets correspondent aux répartitions pures du trio.** » |
| Fichier orphelin | `sections/appendix_c_robustness_methodology.tex` (17 Ko) | Chiffres périmés (MinBTL 3.45, BHY 86.94 %) mais **aucune référence dans le dépôt** : n'entre dans aucun PDF. Dette de maintenance, pas un défaut client |
| « 3 sleeves » résiduel | `client_setup_guide/main.tex`, tableau des fichiers | Le reste du guide dit bien « quatre sleeves » |
| Documentation périmée | `src/mt5/CLAUDE.md` | 3 sleeves, 0.80/0.10/0.10, vol 0.75, levier 64 ; ignore `Inp_RiskScale` et la sleeve or |
| `WeightPoint` risk-parity | `generate_weight_sensitivity_figures.py:236-246` | Seul point dont `w_mr+w_ts+w_rsi ≠ 1` (pas de renormalisation par `fx_total`). Sans impact publié aujourd'hui |

---

## Corrigé pendant l'audit

**Le graphique MT5 sauvegardé lançait l'EA en live avec la configuration pré-production.**
Chaque démarrage du terminal (donc chaque backtest CLI) restaurait le profil « Default »,
où `FxMultiSleeve` était attaché à un graphique EURUSD.c D1 en mode live avec
`Alloc 0.80/0.10/0.10`, `Gold 0.0`, `TargetVol 0.75`, `MaxLev 64.0`, `RiskScale 1.0` —
d'où les alertes FRED `err 4014` répétées (39 dans le journal du jour). Le binaire déployé
était pourtant à jour (sha256 identique au dépôt) : c'étaient les inputs sauvegardés du
graphique qui écrasaient les défauts compilés. `expertmode=0`, donc rien n'a été tradé.

Le bloc `<expert>` a été retiré de `Charts/Default/chart05.chr` (sauvegarde dans le
scratchpad de session). Vérifié : aucune nouvelle alerte après un backtest de contrôle,
et le tester continue de servir la macro depuis `macro_history.csv`.

À noter : en Strategy Tester, `MACRO_SOURCE_AUTO` bascule sur `HISTORY` et FRED n'est
jamais appelé — **aucun chiffre publié n'était affecté**.

---

## Vérifié et trouvé correct

Ces points ont été reproduits et tiennent. C'est aussi informatif que les défauts.

**A1 — reproduction du run de référence : exacte, et bit-à-bit.**
```bash
python src/mt5/bridge/run_backtest_cli.py --from 2021.01.01 --to 2026.04.30 \
  --model 1 --report-name audit_a1 --ini-name audit_a1.ini --input Inp_ExportDeals=true
# → 851 trades, 40 267,40, Sharpe 0.89, Equity DD 44.33 %, en 18,5 s
```
Le CSV régénéré est **identique octet pour octet** à celui qui a produit
`mt5_reference.json`.

**A2 — tous les chiffres publiés recalculés indépendamment**, sans passer par
`parse_mt5_report.py` : net 40 267,40 (écart −0,0000), CAGR 35,4446 %, or à 79,82 % pour
35 transactions, liquidations 3 / 83,07 / 0,2063 %, taux de gain par sleeve. Le biais de
la convention d'annualisation (fin non corrigée sur la fenêtre) vaut **+0,02 point** —
immatériel. Aucun trade à `net == 0`, donc la question du dénominateur du taux de gain
est sans objet sur ce jeu.

**A3 — l'écart de drawdown est entièrement expliqué.** Le prompt opposait 22,75 % à
« `balance_dd_max` = 23,37 % ». Le nom est faux mais le chiffre existe : c'est le
`Balance Drawdown Relative` de MT5 (le `Maximal` vaut 6,66 %). Reconstruction par deal :
**−23,3691 %**, soit MT5 au centième. Reconstruction journalière : −22,7470 %, exactement
la valeur publiée. Les 0,62 point d'écart viennent du `resample("D").sum()`, qui rate le
creux intra-journalier du 21 août 2023. La reconstruction n'est pas défectueuse, elle est
plus grossière.

**A4 — l'affirmation la plus lourde du prompt est exacte.** Sur la fenêtre courte, 3
liquidations d'office totalisent 21 283,31, soit **47,74 %** du profit net, dont
21 258,30 pour la seule position or (47,68 %). Repli d'équité 20,11 % contre 44,33 % sur
la fenêtre longue. La décision de publier la fenêtre longue repose sur du solide.

**B1 — aucun littéral figé ne survit dans les tables adossées au JSON.** Rapprochement
automatique de tous les nombres des 20 `tables/*.tex` contre les deux JSON : les
apparents orphelins sont des fragments de dates ou des valeurs légitimement calculées en
direct (la ligne « Volatilité ann. » est bootstrapée à la volée parce que
`stress_test_report.json` n'expose pas de percentiles de volatilité).

**B2 — la chaîne dérive bien de la configuration** (hormis les étoiles du §9).

**B3 — les trois nouveaux tests détectent ce qu'ils prétendent protéger.** Mutation
testing, chacun restauré ensuite :

| Test | Mutation | Résultat |
|---|---|---|
| `test_report_config_sync` | `PRODUCTION_TARGET_VOL` → 0.99 | RED ✓ |
| `test_parse_mt5_report` | neutralisation de la réattribution des `magic=0` | RED ✓ (2 tests) |
| `test_mt5_log_parsing` | `_slice_last_run` renvoie tout le texte | RED ✓ (3 tests) |
| `test_mt5_preset_sync` | valeur d'un input du preset | RED ✓ |
| `test_stress_sanity` | grille vidée de sa substance | **GREEN** ✗ (§7) |

**B4 — la sémantique du simplexe est cohérente.** `make_weights()` balaie bien les parts
internes au trio, l'or reste fixe à 10 % et *présent* dans chaque portefeuille évalué,
`WeightPoint` stocke des parts internes sommant à 1, et le diagramme ternaire, la
frontière de Pareto et les deux tables sont d'accord entre eux. `FIXED_WEIGHT` dérive de
`PRODUCTION_WEIGHTS`.

**C1 — le PBO n'est pas un artefact : le prompt se trompait de soupçon.** La fonction
n'a **aucun RNG** — elle énumère exhaustivement $C(16,8) = 12\,870$ partitions ; « faire
varier le seed » est sans objet, et trois appels identiques rendent la même valeur.
Reproduction fidèle de la matrice du rapport (1887 × 6, 2019-01-02 → 2026-04-01) :
**PBO = 0,5322**, soit le 0,532 publié. Sensibilité au découpage :

| `n_bins` | 4 | 6 | 8 | 10 | 12 | 14 | 16 |
|---|---|---|---|---|---|---|---|
| PBO | 0.667 | 0.750 | 0.643 | 0.722 | 0.623 | 0.503 | **0.532** |

Le verdict échoue sur **toutes** les valeurs testées. Le 0,532 publié est la lecture la
plus clémente de la plage. Le rapport ne sur-interprète pas un bruit — il sous-estime
plutôt la sévérité.

*Réserve méthodologique* : la matrice porte sur 6 **sleeves** (dont `XS_Momentum` et
`TS_Momentum_RSI`, absents du portefeuille livré), pas sur les configurations d'un sweep
de paramètres. Le CSCV mesure donc la stabilité du classement entre sleeves, alors que la
prose l'interprète comme validant la sélection de paramètres. À reformuler, mais cela ne
change pas le verdict.

**C2 — la table de verdicts est entièrement calculée.** Les six conditions vérifiées
ligne à ligne (`ci_low > 0`, `psr > 0.95`, `dsr > 0.95`, `haircut_ratio > 0.50`,
`minbtl < backtest_years`, `pbo < 0.5`), sens des comparaisons correct, signe du drawdown
rétabli (`fmt_pct(-maxdd_row["ci_high"])` → −70,01 %), ligne sans plafond marquée
`\emph{mesure}`, légende « 5 test(s) sur 6 » dérivée de `n_passed`/`n_tested`. MinBTL :
`backtest_years = 1887/252 = 7,5` ans, dérivé ; $2\ln(6)/1.31^2 = 2.088 → 2.10$ ✓.

**D2 — aucune référence non résolue.** Les quatre documents compilent en deux passes
sans référence pendante en seconde passe. L'insertion des sections « Moteur 4 » et
« Recherche et exécution » n'a rien cassé.

**Suite de tests : 203 passés** en 73 s — exactement le chiffre annoncé dans le handoff.

**Preset déployé en phase avec l'EA compilé** : 86 des 87 inputs identiques, le 87e étant
`Inp_MacroSourceMode` écrit en littéral `4` contre l'enum `MACRO_SOURCE_AUTO` — même
valeur. Le binaire `.ex5` déployé sous Wine a le même sha256 que celui du dépôt.

**Motifs de l'ancienne configuration : proprement éliminés.** Aucune occurrence, dans les
quatre PDF, de `13.33`, `17.93`, `Tri-Signaux`, `64×`, `1,33`, `1.38`, `785`, `12×`,
`30,68`, `-6,27`, `0,956`, `39,0`, « cinq avertissements », `0.80 / 0.10 / 0.10`,
« vol-targeting global 28 », « levier 64 », « levier 12 », `3.45`, `86.94`.

---

## Non vérifié

- **A1 bis — le mode de modélisation recommandé.** Impossible sur ce poste : les ticks
  réels ne sont pas téléchargés (`preliminary downloading of history ticks canceled` →
  `no history data, stop testing`). C'est un angle mort réel : le CLI applique `Model=4`
  par **défaut** et son aide le qualifie de « recommandé, rigoureux », alors que les
  chiffres publiés utilisent `Model=1`, que la même aide qualifie de « interpolation,
  **surestime** ». L'écart entre les deux n'a jamais été mesuré.
- **Le whitelist FRED en conditions réelles.** La clé est présente dans le `common.ini`
  portable (`WebRequest=1`, `WebRequestUrl=https://api.stlouisfed.org`) et l'EA reçoit
  malgré tout 4014. L'hypothèse — le démarrage via `/config:C:\fxbk.ini`, un INI qui ne
  porte que `[Tester]` et `[TesterInputs]`, ne fait pas appliquer la liste blanche — n'a
  pas été prouvée. Sans impact sur les chiffres publiés (le tester n'appelle jamais FRED).
- **Les figures d'anatomie de trade** (`build_sleeve_signals_figures.py`, kaleido) :
  hors périmètre, non instruit.
- **La conformité à `HOLDOUT_POLICY.md`.** Signalé sans être tranché : `OOS_SPLIT` vaut
  `2025-04-01` alors que la politique gèle à partir du `2026-01-01` et que la calibration
  or va jusqu'au `2025-12-31` ; la fenêtre étiquetée « hors échantillon » recouvre donc
  des données utilisées en sélection. Le journal de la politique porte lui-même la tranche
  2026 en statut « at risk », et `src/framework/holdout.py`, garde-fou qu'elle annonce,
  n'existe pas. Trancher demande de rejouer l'historique de sélection — hors périmètre.

---

## Correctifs appliqués — 2026-07-26

Les quatre défauts graves ont été corrigés après remise de l'audit, sur demande explicite.
Les défauts §5 à §12 n'ont **pas** été traités et restent ouverts.

### §1 — contradiction sur le sur-ajustement

`sections/10_advanced_robustness.tex` : les sept passages périmés réécrits depuis les
valeurs générées (PBO $0.31 \to 0.532$, Haircut $0.84/87\,\% \to 1.035/79.29\,\%$,
MinBTL $3.5 \to 2.10$ ans pour SR cible $1.31$, « sept tests » $\to$ le verdict consolidé,
$N=5 \to N=6$, $T=1\,764 \to 1\,887$, $\widehat{SR}=0.97 \to 1.31$). L'insightbox de
synthèse explique désormais pourquoi PSR/DSR favorables et PBO défavorable **ne se
contredisent pas** : ils répondent à des questions différentes.

Deux légendes périmées supplémentaires, non repérées au premier passage, ont été trouvées
et corrigées au même endroit : la distribution bootstrap du Sharpe (valeur observée
$0.97 \to 1.085$, IC $[0.37,\ 1.55] \to [0.356,\ 1.841]$) et le *forest plot* (borne basse
$0.365 \to 0.356$, chiffres transposés).

Ajout d'un paragraphe sur la limite du DSR : le $N=6$ déclaré ne compte que les composantes
entrées en validation croisée, pas les configurations explorées en amont — ce qui explique
que son verdict favorable pèse moins lourd que le verdict défavorable du PBO.

```bash
pdftotext reports/latex_report/main.pdf - | grep -c "0\.532"   # 5 → 10
pdftotext reports/latex_report/main.pdf - | grep -c "0\.31"    # 6 → 1
```
L'occurrence restante est un Calmar légitime de l'annexe de sensibilité aux poids.

### §2 — légende de `robustness_overfitting.tex`

`scripts/build_latex_report_assets.py` : chaque test enregistre son verdict
(`psr > 0.95`, `dsr > 0.95`, `haircut_ratio > 0.50`, `minbtl < backtest_years`,
`pbo < 0.5`) et la légende les compte. Elle imprime « Tous les tests convergent vers un
verdict favorable » **seulement** si c'est vrai, sinon « N test(s) sur M franchissent leur
seuil ».

Régénération complète de la chaîne : **seule la légende a changé**. Les 19 autres tables et
les 26 figures sont identiques au bit près — confirmation que la chaîne est déterministe.

### §3 — plafond de marge

Comportement réel relu dans `src/mt5/Include/FxRiskManager.mqh:306-338` : actif par défaut,
levier divisé par deux à $50\,\%$ d'utilisation de marge, fermeture forcée à $85\,\%$.

Corrigés : la table de paramètres du guide d'installation (l. 515-516), sa checklist de
mise en production (l. 811), et la table du guide pédagogique (l. 783-784). La narration du
guide pédagogique décrivait déjà correctement les seuils 50/85 — c'était sa propre table de
paramètres qui la contredisait ; un `\label{sec:margin_cap}` relie désormais les deux.

> **Correction — cette vérification était défectueuse.** Elle annonçait « 0 écart réel sur
> 31 inputs cités ». Le motif d'extraction, `\\code\{Inp\\_(\w+)\}`, s'arrêtait au premier
> `\_` échappé : `Inp\_MR\_SessionStart` était lu comme `Inp_MR`. Sur les **64** paramètres
> réellement cités par le guide, seuls 31 noms — tronqués — étaient comparés. Trois écarts
> réels avaient donc échappé au contrôle, tous corrigés depuis (voir « Génération de la
> table de paramètres » ci-dessous) :
>
> | Paramètre | Guide | `.mq5` |
> |---|---|---|
> | `Inp_MR_SessionStart` | 6 | **8** |
> | `Inp_MR_SessionEnd` | 14 | **16** |
> | `Inp_RSI_Pairs` | 4 paires (EUR, GBP, JPY, CAD) | **3** (EURUSD, GBPUSD, USDCAD) |
>
> Le guide publiait une fenêtre de session décalée de deux heures et une paire de trop sur
> le Sleeve 3. La leçon est la même que celle du §2 : un contrôle qui ne peut pas voir ce
> qu'il prétend couvrir est plus dangereux qu'aucun contrôle, parce qu'il clôt la question.

### §4 — Moteur 4 dans le guide pédagogique

Nouvelle section « Moteur 4 — Gold Momentum », sur le modèle des sections 3 à 5 :
intuition économique (TSMOM, Moskowitz-Ooi-Pedersen 2012), mécanique du signal (ensemble de
quatre horizons 40/60/120/250, moyenne des **signes**, long-only), table des sept paramètres
avec justification, figure d'équité, et performance historique.

L'encadré d'avertissement est explicite sur la concentration : $79{,}8\,\%$ du profit net
pour $10\,\%$ d'allocation et $4\,\%$ des transactions, un taux de réussite de $31{,}4\,\%$,
un repli isolé de $-79{,}62\,\%$, et ce qu'il advient du portefeuille si l'or cesse de
tendre. Un second encadré explique pourquoi le poids reste à $10\,\%$ : l'optimum apparent
est à la borne du domaine testé, donc indécidable.

Corrigés également : « trois moteurs » → « quatre moteurs » en tête de document et dans le
titre de sous-section, ajout du Moteur 4 à la liste de présentation, nuance dans la
thesisbox d'allocation (les $10\,\%$ de l'or ne sont pas un poids vol-équipondéré mais un
plafond d'exposition délibéré), et « 3 sleeves » → « 4 sleeves » dans le tableau de fichiers
du guide d'installation.

Le guide passe de 30 à 34 pages ; les mentions de l'or y passent de 2 à 30.

### Génération de la table de paramètres du guide d'installation

Correctif de fond appliqué après les quatre précédents. Le guide listait ses 64 paramètres
à la main : c'est ce qui a produit l'erreur sur le plafond de marge, puis les trois écarts
que ma vérification défectueuse avait manqués. Rien ne reliait le `.tex` à une source de
vérité, donc aucun test ne pouvait voir la dérive.

`scripts/build_setup_guide_tables.py` génère désormais les sept tables de la section 6
depuis `write_default_preset.PRESET_LINES` — que `tests/test_mt5_preset_sync.py` asservit
déjà aux défauts compilés du `.mq5`. La chaîne est fermée :
`.mq5` → `PRESET_LINES` → `tables/*.tex` → PDF.

Choix de conception :

- **Seule la prose vit dans le script.** Le rôle de chaque paramètre est le seul contenu
  qu'aucune source de vérité ne porte. Les valeurs, elles, sont le littéral exact du preset
  — ce que MT5 reçoit, donc la seule chose défendable à publier.
- **Les décomptes sont dérivés.** « Univers de {n} paires » est calculé depuis la liste
  réelle. Écrire « 4 paires » en toutes lettres est précisément ce qui avait permis au
  guide d'en annoncer 4 pour un univers qui en compte 3.
- **Aucun paramètre ne peut être oublié.** Les 87 inputs du preset sont soit décrits (71),
  soit dans `EXCLUDED` avec une raison (16 : sleeve H1 non allouée, horizons or regroupés,
  coûts traités ailleurs). Un input ajouté sans description fait échouer la suite.

Six tests dans `tests/test_setup_guide_tables.py`, tous vérifiés par mutation testing —
chacun rougit quand on casse ce qu'il protège :

| Mutation | Test attendu rouge | Résultat |
|---|---|---|
| `Inp_MarginCapPct` 0.50 → 0.70 dans le `.mq5` | valeurs vs défauts compilés | ✅ RED |
| `Inp_MR_SessionStart` 8 → 6 dans `PRESET_LINES` | tables en phase + valeurs | ✅ RED (2) |
| suppression d'un `.tex` généré | tables en phase | ✅ RED (échoue, ne saute pas) |
| input ajouté au preset sans description | couverture des descriptions | ✅ RED |
| valeur recodée en dur dans `main.tex` | absence de littéraux + inclusions | ✅ RED (2) |

Le troisième point est délibéré : `test_published_stress_json_matches_the_current_config`
*saute* quand son fichier manque, et c'est ce qui l'a rendu inopérant en CI. Ces tests-ci
échouent.

### Fermeture du test qui sautait

Troisième et dernier correctif de fond. `test_published_stress_json_matches_the_current_config`
était le seul test comparant les artefacts publiés à la configuration courante, et il
`skip` quand son fichier manque — c'est-à-dire par défaut, puisque `results/` était
intégralement gitignoré.

**Le versionnement d'abord, l'échec ensuite.** Faire échouer sans versionner aurait rendu
la CI rouge sur tout clone frais : le test aurait été désactivé pour de bon, ou ignoré. Le
raisonnement qui tranche est ailleurs — `reports/latex_report/tables/*.tex` est versionné
et ces tables **dérivent** de ces JSON. Versionner la sortie sans l'entrée rend le document
publié irreproductible et inauditable. 83 Ko pour les deux.

```gitignore
results/*
!results/production_report/
results/production_report/*
!results/production_report/*.json
*/**/results/
```

La dernière ligne répare un effet de bord découvert au passage : l'ancien motif `results/`,
sans slash initial, matchait **tout** répertoire de ce nom à n'importe quelle profondeur, y
compris `src/strategies/results/` (sorties de recherche par stratégie). Le motif ancré ne
le faisait plus, et 200+ fichiers seraient entrés dans le dépôt. `*/**/` exige au moins un
composant de chemin, donc ne peut pas matcher la racine et casser la négation.

**Le générateur aussi avalait l'absence.** `build_mt5_assets()` se contentait d'un `⚠` puis
d'un `return` : les trois tables MT5 gardaient le contenu du run précédent et la chaîne
rendait un rapport d'apparence complète, bâti sur d'autres chiffres. Il lève désormais.

Un septième test a été ajouté, `test_published_mt5_reference_covers_every_allocated_sleeve` :
le nombre de sleeves décrites par le JSON MT5 doit égaler le nombre de sleeves à poids non
nul. L'assertion porte sur le compte et non sur les noms — les libellés MT5
(« Gold Momentum ») et les clés Python (`Gold_Momentum`) diffèrent, et recopier la
correspondance créerait une troisième copie à maintenir.

Mutation testing, quatre scénarios :

| Mutation | Attendu | Résultat |
|---|---|---|
| `stress_test_report.json` absent | RED (et non SKIP) | ✅ RED |
| `mt5_reference.json` absent | RED | ✅ RED |
| idem, côté générateur | `FileNotFoundError` | ✅ levée |
| sleeve or retirée de la production | RED | ✅ RED |
| `PRODUCTION_TARGET_VOL` 0.37 → 0.99 | RED | ✅ RED |

Le script de mutation détecte explicitement le statut `SKIP` : un test qui saute sort en 0
et se confondrait avec un succès.

### Fermeture des défauts §5 à §12 — 2026-07-27

Traités en parallèle, chacun sur un périmètre de fichiers disjoint, chacun avec ses tests
vérifiés par mutation avant d'être crus.

| § | Défaut | Correctif | Mutations rouges |
|---|---|---|---|
| 5 | Branche morte `if False` écrasant le JSON canonique | L'écriture concurrente est **supprimée** : un seul producteur de `stress_test_report.json`. `summary.txt` dérive des constantes. | 6/6 |
| 7 | `assert len(sweeps) == 18` littéral | Grille extraite en constantes ; le test vérifie le **contenu** et que le point de production est balayé | 2/2 |
| 8 | CLI rendant 0 sur un run vide | `validate_run()` : trades > 0, période non dégénérée, métriques numériques | 2/2 |
| 9 | Étoiles « production » figées à 80/10/10 | Les trois marqueurs et leurs libellés dérivent de `PRODUCTION_WEIGHTS` renormalisés | 2/2 |
| 11 | CAGR OOS 161 % sans avertissement en §7 | `warningbox` ajouté sous la table, renvoyant à la section 08 | — |
| 12 | Unités incohérentes, aucune provenance | Sous-bloc `drawdowns` autoporteur (dont le `Balance Drawdown Relative` manquant) + bloc `provenance` avec sha256 des artefacts | 4/4 |
| 12 | Phrase dupliquée, annexe orpheline, doc MT5 périmée | Corrigés ; l'annexe orpheline porte un en-tête disant qu'elle n'est incluse nulle part | — |

**Trois défauts que l'audit n'avait pas vus**, trouvés en corrigeant :

1. **`_latest()` désignait le mauvais CSV.** Le nom du fichier venant de l'heure *simulée*
   du tester, le backtest de fenêtre courte relancé pendant l'audit (21:47) était plus
   récent que celui de la fenêtre publiée (21:42). Relancer `parse_mt5_report.py` sans
   argument aurait publié un CAGR de **40,47 % au lieu de 35,44 %**, avec les 851 trades et
   le profit net du HTML inchangés — indétectable à l'œil. Le CSV est désormais choisi par
   correspondance avec la fenêtre du HTML, et l'incohérence est **fatale** : rien n'est écrit.
2. **La branche morte était fausse deux fois.** Même en retirant le `if False`, l'expression
   lisait `RECOMMENDED_CONFIG` **après** le `finally` qui la restaure — elle aurait décrit la
   configuration restaurée, pas celle du calcul.
3. **`figure_bootstrap_scatter()` annotait l'ancien mandat** (cible 10-15 %, cap −35 %). Sans
   impact client — la figure du PDF dérive déjà la bande du JSON — mais le même mensonge
   subsistait dans le script orphelin. Corrigé, ainsi que l'absence de la sleeve or dans sa
   palette.

### Vérification des correctifs

| Contrôle | Résultat |
|---|---|
| Compilation des 4 PDF (`-halt-on-error`) | ✅ 60 / 15 / 34 / 17 pages |
| Références non résolues en 2\ieme passe | ✅ 0 sur les 4 documents |
| « convergent vers un verdict favorable » | ✅ 0 occurrence |
| « sept tests » | ✅ 0 occurrence |
| Inputs du guide vs `.mq5` (extraction corrigée, 64 params) | ✅ 0 écart, et désormais dérivés |
| Générateur idempotent (`--check`) | ✅ 7 tables en phase |
| Suite de tests | ✅ **239 passés** (203 au départ de l'audit) |
| Tables/figures hors légende corrigée | ✅ identiques au bit près |

---

## État du dépôt

Toutes les mutations de B2 et B3 ont été restaurées (`git checkout`). Vérifié après coup :

```bash
sha256sum -c fingerprints_phase0.txt   # → les 50 fichiers (4 PDF, 20 tables, 26 figures)
                                       #   sont INCHANGES
git diff --stat                        # → vide, aucune modification suivie
git status --porcelain                 # → .claude/, output/, les deux docs d'audit
```

Les quatre `reports/mt5/run_*.json` produits par les backtests de cet audit ont été
déplacés vers le scratchpad de session : ce sont des sous-produits de la vérification,
pas des livrables. Les artefacts non versionnés (`results/production_report/*.json`,
`deals_*.csv`, l'INI du tester) avaient été sauvegardés avant le premier run ; le CSV
régénéré s'étant révélé identique, aucune restauration n'a été nécessaire.

Écrit hors du dépôt, et réversible : le bloc `<expert>` retiré de
`~/.mt5/…/MQL5/Profiles/Charts/Default/chart05.chr` (voir « Corrigé pendant l'audit »),
sauvegarde intégrale dans le scratchpad.
