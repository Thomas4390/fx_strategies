# Prompt d'audit — republication du rapport client (2026-07-26)

> À coller tel quel au début d'une session fraîche. Il est autonome : il ne suppose
> aucun contexte de la session qui a produit le travail à auditer.

---

## Rôle

Tu es auditeur adverse sur un travail que **quelqu'un d'autre** a produit et déclaré
terminé. Ta mission n'est pas de confirmer, c'est de **trouver ce qui est faux**. Le
travail auditera d'autant mieux qu'il t'aura résisté.

Trois règles qui priment sur tout le reste :

1. **Ne fais confiance à aucune affirmation de ce document.** Les chiffres cités
   ci-dessous sont ceux que l'auteur *prétend* avoir obtenus. Ils font partie de ce que
   tu dois vérifier, pas de ce sur quoi tu peux t'appuyer.
2. **Reproduis avant de conclure.** Une conclusion sans commande exécutée et sortie lue
   n'est pas une conclusion, c'est une intuition. Cite la commande et la sortie.
3. **Ne fabrique pas de problèmes.** Si un point est correct, dis-le en une ligne et
   passe. Un audit qui invente des défauts pour paraître rigoureux est aussi inutile
   qu'un audit complaisant. Le succès de cette mission n'est pas « trouver N bugs »,
   c'est « savoir avec certitude ce qui tient et ce qui ne tient pas ».

---

## Contexte factuel

Le repo produit un portefeuille FX multi-stratégies livré à un client, exécuté par un
Expert Advisor MetaTrader 5. La recherche se fait en VectorBT PRO ; MT5 exécute.

Le 2026-07-26, une session a republié les **quatre livrables client** (rapport
technique, synthèse exécutive, guide d'installation, guide pédagogique) qui décrivaient
depuis avril une configuration périmée. Commits `43ed505` → `87756cf` sur `main`.

Le changement de fond : **les chiffres publiés au client sont désormais ceux de MT5**
et non ceux de vbt, les deux moteurs ne mesurant pas la même quantité (vbt applique un
levier au poids de position, MT5 ouvre un notionnel borné par la distance au stop).

Lis `git log 43ed505^..HEAD` et le handoff du vault
(`projects/fx_strategies/handoff-2026-07-26-quatre-livrables-republies.md`) pour le
détail. **Lis-les comme des déclarations d'intention, pas comme des faits établis.**

---

## Les affirmations à casser

Chacune est vérifiable. Pour chacune : confirme avec preuve, ou réfute avec preuve.

### A. Le socle chiffré MT5

**A1.** Le run de référence est reproductible à l'identique :
```bash
python src/mt5/bridge/run_backtest_cli.py --from 2021.01.01 --to 2026.04.30 \
    --model 1 --report-name audit --input Inp_ExportDeals=true
```
Attendu : 851 trades, profit net 40 267,40, Sharpe 0,89, equity DD 44,33 %.
*Si les chiffres diffèrent, c'est le résultat le plus important de tout cet audit.*

**A2.** `scripts/parse_mt5_report.py` calcule juste. **Ne relis pas le script pour
juger : recalcule à la main depuis le CSV brut** (`deals_*.csv` dans le répertoire
`FILE_COMMON`, voir pièges plus bas) et compare. En particulier :
- le CAGR de 35,44 % — vérifie la fenêtre exacte utilisée pour l'annualisation ;
- la ventilation par sleeve, dont l'affirmation que **l'or produit 79,8 % du résultat
  net pour 35 transactions** ;
- l'affirmation que les liquidations de fin de test ne pèsent que 0,2 % du profit sur
  cette fenêtre ;
- le taux de gain par sleeve : sur quels deals est-il calculé, et est-ce le bon
  dénominateur ?

**A3.** Le script rapporte `balance_dd_pct_daily` = 22,75 % quand MT5 annonce
`balance_dd_max` = 23,37 %. **Cet écart n'a pas été expliqué.** Explique-le ou montre
qu'il révèle un défaut de la reconstruction de la courbe (granularité journalière,
`resample("D").sum()`, `ffill`, traitement des week-ends).

**A4.** L'affirmation la plus lourde de conséquence de la session : sur la fenêtre
courte (2021→2025-12), **47,7 % du profit venait d'une position or liquidée d'office au
dernier tick**, et le repli d'équité y paraissait deux fois moindre (20,11 % contre
44,33 %). Reproduis les deux runs et vérifie. Si c'est faux, la décision de publier la
fenêtre longue repose sur du vide.

### B. La chaîne de génération

**B1.** Chaque nombre imprimé dans `reports/latex_report/tables/*.tex` se retrouve dans
`results/production_report/stress_test_report.json` ou `mt5_reference.json`. Écris un
script de rapprochement plutôt que de comparer à l'œil. Signale tout littéral survivant.

**B2.** Les scripts sont réellement idempotents et dérivés de la config. Test décisif :
change temporairement `PRODUCTION_WEIGHTS` dans
`src/strategies/combined_portfolio_v2.py` (par exemple 60/15/15/10), relance la chaîne,
et vérifie que **tables, légendes et figures suivent**. Rien ne doit rester à 72/9/9/10.
Restaure ensuite.

**B3.** Les trois nouveaux tests (`test_report_config_sync.py`,
`test_parse_mt5_report.py`, `test_mt5_log_parsing.py`) détectent vraiment quelque chose.
Casse le code volontairement et vérifie qu'ils rougissent — un test qui reste vert quand
on casse ce qu'il prétend protéger est pire qu'aucun test. C'est exactement le défaut
qu'avait `test_stress_sanity.py`, dont l'assertion `len(sweeps) == 18` est restée verte
alors que toute la grille avait changé.

**B4.** L'annexe de sensibilité aux poids a changé de sémantique : le simplexe balaie
désormais la répartition **interne au trio FX**, l'or restant fixé à 10 %. Vérifie que
cette sémantique est cohérente entre `make_weights()`, les coordonnées stockées dans
`WeightPoint`, le diagramme ternaire, la frontière de Pareto, les deux tables et la
prose de `appendix_c_weight_sensitivity.tex`. **C'est le point le plus susceptible
d'être incohérent** : la conversion a été faite tard et à plusieurs endroits.

### C. Les résultats statistiques

**C1.** La probabilité de sur-ajustement (PBO) vaut 0,532 pour un seuil de 0,5 — le
rapport en fait un avertissement de gravité élevée. Est-ce robuste, ou un artefact ?
Fais varier le seed et le nombre de partitions (`n_bins`). Si le verdict bascule d'un
côté à l'autre du seuil selon le tirage, **le rapport sur-interprète un bruit** et doit
être corrigé.

**C2.** La table de verdicts calcule maintenant ses conclusions au lieu de les coder en
dur. Vérifie chaque condition ligne à ligne : le sens des comparaisons, le signe du
drawdown (il s'affichait à +70 % faute d'inversion), et la ligne MinBTL dont la
longueur de référence est désormais dérivée.

**C3.** Le CAGR hors échantillon de 161 % apparaît dans `metrics_summary`. Le rapport
l'encadre d'un avertissement. Vérifie que cet avertissement est présent **partout où le
chiffre apparaît**, y compris dans la synthèse exécutive et les figures.

### D. La cohérence du livrable

**D1.** Aucune phrase du corpus ne contredit une table voisine. C'est le défaut
historique de ce rapport : des légendes affirmaient « résistance validée pour Covid
2020 » au-dessus d'un Sharpe de −2,00. Une réécriture massive vient d'avoir lieu :
**cherche les contradictions résiduelles**, en particulier dans les sections dont seules
quelques phrases ont été retouchées (03, 04, 05, 10) et dans le guide pédagogique.

**D2.** La numérotation et les renvois tiennent : une section « Moteur 4 » et une
section « Recherche et exécution » ont été insérées. Vérifie qu'aucun renvoi ne pointe
vers la mauvaise section, et que la table des matières est cohérente.

**D3.** Les quatre PDF ne contiennent plus aucun chiffre de l'ancienne configuration.
Élargis la liste de motifs au-delà de celle qu'a utilisée l'auteur (13.33, 17.93,
Tri-Signaux, 64×, 1,33, 1.38, 785) — cherche aussi `0.28`, `12×`, `30,68`, `-6,27`,
`0,956`, `39,0 %`, `99,8 %`, « sept tests », « cinq avertissements ».

**D4.** Le guide d'installation permet réellement à un client de reproduire les
métriques annoncées. Lis-le **comme si tu déployais** : les valeurs d'inputs
correspondent-elles à `src/mt5/bridge/write_default_preset.py` et au `.mq5` compilé ?
Un input manquant ou faux dans ce document est le défaut le plus coûteux du lot.

---

## Pièges connus de ce repo

Ils feront perdre du temps ou produiront des faux positifs si tu les ignores.

- **`FILE_COMMON` sous Wine ne résout pas vers la racine portable** mais vers
  `~/.mt5/drive_c/users/thomas/AppData/Roaming/MetaQuotes/Terminal/Common/Files/`.
  C'est là que l'EA écrit `deals_*.csv`.
- **Le nom du CSV de deals vient de l'heure simulée du tester**, pas de l'heure réelle :
  deux runs sur la même période écrasent le même fichier. Vérifie le `mtime` avant de
  conclure que tu lis le bon.
- **`results/` est gitignoré.** Les JSON de référence ne sont pas versionnés : si tu ne
  les régénères pas, tu audites ceux de la session précédente.
- **Cache de sleeves** (`src/framework/data_cache.py`) : les rendements journaliers sont
  servis depuis un parquet clé par l'empreinte de `data/MANIFEST.json` **et**
  `_SLEEVES_VERSION`. Modifier un défaut de sleeve sans bumper cette constante sert
  silencieusement du périmé.
- **Politique de holdout** (`docs/research/HOLDOUT_POLICY.md`) : les données ≥ 2026-01-01
  sont gelées pour la *sélection de modèle*. La fenêtre publiée les inclut — c'est du
  reporting, pas de la sélection. Vérifie que rien dans les commits n'a utilisé cette
  tranche pour *choisir* un paramètre.
- **`compile_latex_report.sh` utilise `-halt-on-error`.** Les PDF antérieurs à cette
  session ne l'utilisaient pas : ils sortaient tronqués sans que rien ne le signale.
  Si un PDF a moins de pages qu'attendu, regarde le log avant toute autre hypothèse.
- **Les `.ini` du tester s'écrivent en UTF-16 LE avec BOM et CRLF**, via `write_bytes`.
- **`scripts/build_sleeve_signals_figures.py` ne converge pas** sur ce poste (export
  Plotly/kaleido, Chrome headless bloqué au rendu). Trois blocs `\begin{figure}` ont été
  retirés des sections 03/04/05 pour cette raison. Ce n'est pas une régression à
  corriger à l'aveugle — mais si tu trouves *pourquoi* kaleido bloque, c'est un gain.

---

## Ce qui reste ouvert et n'a pas à être « corrigé »

Ne les compte pas comme des défauts, sauf si tu établis qu'ils sont plus graves
qu'annoncé :

- `Inp_RiskScale=4.5` est conservé malgré un repli d'équité de 44 % — décision explicite
  du propriétaire, risque déclaré non contraignant.
- Le poids de l'or (10 % vs 15 %) reste indécidable sur les données disponibles : son
  optimum apparent est à la borne du domaine testé.
- La trace journalière MT5 ne s'écrit pas malgré `Inp_Gold_Trace=true`.
- Le whitelist FRED `WebRequest` n'a jamais été validé en conditions réelles (le tester
  ne le permet pas).
- `src/mt5/CLAUDE.md` décrit encore les allocations 0.80/0.10/0.10 et 0.75/64.

---

## Restitution attendue

Un rapport court, ordonné **par gravité décroissante**, où chaque entrée porte :

1. **Ce qui est faux**, en une phrase.
2. **La preuve** : commande exécutée, sortie obtenue, écart avec l'attendu.
3. **La conséquence** pour le client, si le document part en l'état.
4. **Le correctif** proposé.

Termine par la liste explicite de ce que tu as vérifié **et trouvé correct** — c'est
aussi informatif que les défauts, et cela délimite ce que ton audit ne couvre pas.

Si tu n'as pas pu vérifier un point (environnement, temps, dépendance manquante),
**dis-le** plutôt que de l'omettre. Un point non vérifié annoncé comme tel vaut mieux
qu'un audit qui paraît exhaustif et ne l'est pas.
