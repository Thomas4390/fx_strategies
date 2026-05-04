# Investigation — Écart RSI Daily : VBT Pro vs MT5

> **Statut** : ouvert · **Créé** : 2026-05-04 · **Cible** : agent fresh ou Thomas
>
> **TL;DR** — Le sleeve RSI Daily produit des résultats différents entre la
> référence VBT~Pro (`src/strategies/rsi_daily.py`) et le port MQL5
> (`src/mt5/Include/FxSleeveRSIDaily.mqh`). Ce document liste les hypothèses
> par ordre de probabilité décroissante avec, pour chacune, **un test isolé
> et reproductible** et **le fix attendu**. Objectif : identifier le ou les
> facteurs dominants et faire converger les deux implémentations à une
> tolérance définie en page~6.

---

## 1. Contexte et fichiers clés

### Implémentation Python (référence)

- `src/strategies/rsi_daily.py` — `pipeline(data, ...)` retourne
  `(vbt.Portfolio, RSIDailyIndicator)`. Defaults : `rsi_period=14,
  oversold=25, overbought=75, exit_mid=50`. Sans leverage, slippage ni
  fees par défaut (`None`).
- `src/strategies/combined_portfolio.py:49` — `backtest_rsi_daily_portfolio()`
  agrège **equal-weight la moyenne des `daily_returns` par paire**
  (4~paires : EUR-USD, GBP-USD, USD-JPY, USD-CAD).
- Données : `data/<PAIR-DASH>_minute.parquet` (probablement Dukascopy ou
  équivalent), couverture 2019-01 → 2026-04~environ.

### Implémentation MT5 (port en cours d'audit)

- `src/mt5/Include/FxSleeveRSIDaily.mqh` — sleeve actif sur `OnNewBarD1`,
  4~paires, magic 833, sizing : `sub_equity_RSI / n_pairs × 0.05 ×
  global_leverage` avec SL safety à 5\,\% et slippage `Inp_RSI_SlippageBps=10`
  (10~bps).
- Backtest baseline (5.4~ans, EUR/USD.c~M1, mode 1-min~OHLC, levier 1:100,
  dépôt 10\,000~USD) :
  **Sharpe~1.15 / DD~$-7.21\,\%$ / 835~trades** (toutes sleeves cumulés). Le
  ventilation par sleeve indique typiquement **~45 trades pour RSI Daily**
  (cf. `src/mt5/SESSION_NOTES.md` ligne~62).

### Référence Python observée

- `combined_portfolio.py` ligne~127~: \enquote{Phase~18~: RSI Daily 4-pair
  positive en 2019 et 2023}, \enquote{near-zero correlation with MR~Macro
  ($+0.056$) and slightly negative with TS~Momentum ($-0.251$)}.
- Sharpe standalone faible ($\sim 0.16$) mais positif anti-corrélé sur les
  années difficiles. C'est intentionnel.

### Pipeline d'exécution disponible

- CLI : `python src/mt5/bridge/run_backtest_cli.py` — backtest complet 5.4~ans
  en $\sim 22$~secondes via Wine. Voir `docs/mt5/14_cli_backtest_linux.md`.
- Preset GUI : `MQL5/Profiles/Tester/FxMultiSleeve_Default.set` (généré par
  `src/mt5/bridge/write_default_preset.py`).

---

## 2. Hypothèses ordonnées par magnitude attendue

Chaque hypothèse contient~: \textbf{description}, \textbf{test isolé},
\textbf{fix attendu si confirmée}.

### H1 — Sizing / notional dramatiquement différent (impact attendu : énorme)

\textbf{Description.} Le pipeline VBT par défaut tourne avec `init_cash=None`
et `leverage=None`, ce qui équivaut à \emph{full equity sur la paire},
soit~$\sim 100\,\%$ du capital sur 1~paire à la fois. Le port MT5 calcule
\code{risk\_money = sub\_equity\_RSI / n\_pairs × 0.05 × global\_leverage},
donc en pratique un \emph{notional cible} très inférieur~: pour un capital
de 10~k\$ avec allocation $10\,\%$ au sleeve RSI, sub\_equity~$=1\,000\$$,
divisée par 4~paires~$=250\$$, fois~$5\,\%$~$=12.50\$$ de risk\_money par
trade × levier global ($\sim 8\times$ observé en backtest)~$=100\$$ de
risk\_money. Translate~: pour la \emph{même} série de signaux, MT5 prend
des positions $\sim 50\times$ plus petites que VBT.

\textbf{Test isolé.}
\begin{enumerate}
\item Lancer VBT \code{pipeline(data\_eurusd, leverage=0.005, init\_cash=10000,
slippage=0.0)} et comparer le \code{pf.daily\_returns} agrégé sur 4~paires
versus le sub-equity du magic~833 dans le log MT5.
\item Sur MT5, instrumenter \code{OpenPosition} pour logger
\code{lots × price / equity} (notional fraction). Cible attendue~$0.005$
si le calcul ci-dessus est correct.
\end{enumerate}

\textbf{Fix.} Soit (a)~aligner VBT au notional MT5 via \code{leverage} et
\code{init\_cash} explicites, soit (b)~documenter clairement que les deux
mesures \emph{n'ont pas la même unité} et comparer plutôt la \emph{série
des returns normalisés} (Sharpe, signe, tracking error) plutôt que les
PnL bruts.

---

### H2 — Vol-targeting global qui module ou bloque les trades MT5 (impact attendu : moyen-fort)

\textbf{Description.} L'EA appelle \code{risk.GlobalLeverage()} pour
multiplier le sizing. Si la volatilité réalisée dépasse la cible
($\sigma^\star = 28\,\%$), le levier descend sous~1.0, et si elle est très
basse il monte jusqu'au plafond~$12\times$. VBT pipeline ne fait rien de
tel. Sur les régimes calmes, MT5 prend $\sim 10\times$ plus de notional
que dans les régimes volatils~--- VBT trade le même notional partout.

\textbf{Test isolé.}
\begin{enumerate}
\item Re-runner MT5 avec \code{Inp\_GlobalTargetVol=0} et
\code{Inp\_GlobalMaxLeverage=1.0} pour neutraliser le vol-targeting (ou
forcer \code{global\_leverage = 1.0} dans `FxRiskManager.mqh`).
\item Comparer le ratio \code{notional\_t / sub\_equity\_t} dans le log MT5
sur une fenêtre où VBT trade aussi.
\end{enumerate}

\textbf{Fix.} Soit (a)~ajouter un overlay vol-target dans VBT pour matcher
MT5, soit (b)~configurer un mode \enquote{plat} dans MT5 pour la
comparaison.

---

### H3 — DD circuit-breaker MT5 désactive le sleeve (impact attendu : moyen)

\textbf{Description.} `OnNewBarD1` commence par \code{if(risk.IsDDLocked()) return}.
Si le portfolio combiné touche $-15\,\%$ de drawdown global, \emph{toutes
les sleeves s'arrêtent} jusqu'à reset manuel. VBT pipeline n'a aucun
mécanisme similaire. Si l'EA s'est verrouillé une fois en backtest, le RSI
Daily perd des trades VBT prendrait.

\textbf{Test isolé.}
\begin{enumerate}
\item Grepper \code{[RISK][LOCKED]} ou \code{IsDDLocked} dans le log Tester
courant.
\item Re-runner avec \code{Inp\_EnableDDCap=false} et comparer le nombre
de trades RSI Daily.
\end{enumerate}

\textbf{Fix.} Désactiver le DD cap pour la comparaison, ou ajouter
l'équivalent dans VBT.

---

### H4 — Données différentes entre VBT et MT5 (impact attendu : moyen)

\textbf{Description.} VBT lit \code{data/<PAIR-DASH>\_minute.parquet} tandis
que MT5 utilise les ticks broker SquaredFinancial. Trois sources de
divergence~:
\begin{itemize}
\item \textbf{Cut-off du daily close.} VBT
\code{vbt.resample\_apply("1D", "last")} prend la dernière minute UTC du
jour. MT5 \code{PERIOD\_D1} respecte le calendrier broker (typiquement
22:00 ou 00:00~UTC selon le serveur). Sur EUR/USD, $\sim 5\,\%$ des
\enquote{daily~closes} peuvent différer de quelques pips.
\item \textbf{Couverture.} VBT~: 2019-01 → 2026-04 ($\sim 7$~ans). MT5~:
2020-11-23 → 2026-04-30 (5.4~ans, limite broker). \textbf{2~ans de
données en moins côté MT5.}
\item \textbf{Spread/slippage broker.} Les ticks SquaredFinancial intègrent
le spread du broker (typiquement 0.1--0.5~pip sur EUR/USD ECN), absent
du Parquet Dukascopy.
\end{itemize}

\textbf{Test isolé.}
\begin{enumerate}
\item Exporter le D1 broker via \code{src/mt5/Scripts/FxExportRates.mq5}
(déjà disponible) sur les 4~paires, charger via \code{import\_mt5\_rates.py},
et relancer VBT \emph{sur ces données} (\code{data/EUR-USD\_daily\_mt5.parquet}).
\item Comparer Sharpe RSI Daily VBT \emph{sur Dukascopy} vs VBT \emph{sur
broker MT5} sur la fenêtre commune 2020-11-23 → 2026-04-30.
\end{enumerate}

\textbf{Fix.} Aligner systématiquement les comparaisons sur la même source
(\code{*\_mt5.parquet}) et la même fenêtre.

---

### H5 — Slippage / fees / SL execution (impact attendu : modéré)

\textbf{Description.} VBT par défaut~: \code{slippage=None, fees=None}.
MT5~: \code{Inp\_RSI\_SlippageBps=10} (10~bps), spread broker simulé via
les ticks réels, et \code{SL safety = 5\,\%} qui peut déclencher en cas
de gap weekend. Sur 45~trades, 10~bps × 2~legs = $\sim 0.9\,\%$ de coût
total~--- pas dominant mais non négligeable sur un Sharpe modeste.

\textbf{Test isolé.}
\begin{enumerate}
\item Re-runner VBT avec \code{slippage=0.001, fees=0.0} pour matcher les
10~bps MT5.
\item Vérifier dans le log MT5 si des positions ont touché le SL à $-5\,\%$
(occurrence rare attendue).
\end{enumerate}

\textbf{Fix.} Aligner les paramètres slippage/fees ou les neutraliser
dans les deux sens.

---

### H6 — Filtre macro inactif sur RSI Daily mais pas symétrique (impact attendu : faible)

\textbf{Description.} Le filtre macro (`macro\_ok`) n'est actif que sur le
\textbf{sleeve~1 MR Macro}, pas sur le RSI Daily. À vérifier que rien ne
bloque RSI Daily à cause d'un effet collatéral (par exemple le DD cap
qui se déclenche à cause de pertes du sleeve~1 et fige RSI Daily).

\textbf{Test isolé.} Setter \code{Inp\_AllocMRMacro=0,
Inp\_AllocTSMomentum=0, Inp\_AllocRSIDaily=1.0} pour isoler le sleeve~3
seul, puis comparer aux \code{pf.daily\_returns} VBT par paire.

\textbf{Fix.} Aucun a priori~--- ce test sert à \emph{isoler} le sleeve~3
des effets de couplage avec les deux autres.

---

### H7 — Différence subtile dans le calcul du RSI (impact attendu : marginal)

\textbf{Description.} Les deux implémentations utilisent la définition
classique de Wilder (RMA), mais l'initialisation diffère~:
\begin{itemize}
\item VBT \code{vbt.RSI.run} commence le calcul au bar~14 (Wilder pur).
\item MT5 \code{iRSI(D1, 14, PRICE\_CLOSE)} commence aussi au bar~14, mais
peut utiliser un \emph{seed} légèrement différent (initialisation par
moyenne arithmétique des 14~premières barres avant lissage RMA).
\end{itemize}

L'écart converge en quelques dizaines de bars, mais peut affecter le
\emph{premier} signal après init.

\textbf{Test isolé.} Exporter \code{rsi.values} VBT pour EUR-USD et comparer
au log MT5 (instrumenter \code{ProcessPair} pour logger \code{rsi\_now}
ligne par ligne sur 250~bars). Tolérance attendue~: $< 0.1\,\%$ après
50~bars.

\textbf{Fix.} Si écart confirmé, ajuster l'init RSI dans VBT pour
matcher MT5 (rare~--- l'écart d'initialisation s'efface en deux semaines
de daily).

---

### H8 — Aggregation 4-pairs : equal-weight \emph{returns} vs equal-weight \emph{equity} (impact attendu : faible-modéré)

\textbf{Description.} VBT \code{backtest\_rsi\_daily\_portfolio} fait
\code{pd.concat([pf.daily\_returns par paire], axis=1).mean(axis=1)}.
C'est \emph{equal-weight les returns}, ce qui suppose un rebalancement
quotidien sans coût. MT5 trade les 4~paires sur \emph{une seule equity
partagée}~--- les pertes/gains se compensent immédiatement, et le sizing
de chaque trade dépend de l'equity courante (donc des trades précédents
des 3~autres paires).

\textbf{Test isolé.} Implémenter VBT \code{vbt.Portfolio.from\_signals}
\emph{multi-colonnes} (4~paires en parallèle, même portfolio) avec
\code{group\_by=True} et \code{cash\_sharing=True}. Comparer au
\code{mean(axis=1)} actuel.

\textbf{Fix.} Si écart significatif, basculer
\code{backtest\_rsi\_daily\_portfolio} sur le mode multi-symboles natif
de VBT.

---

## 3. Méthodologie d'investigation

Procéder hypothèse par hypothèse, en isolant une variable à la fois.
\textbf{Ne pas tester plusieurs hypothèses en parallèle}~: les
interactions noient le signal.

### 3.1 Préparer le terrain

\begin{enumerate}
\item Geler une fenêtre de référence~: \code{2020-11-23 → 2026-04-30}
(intersection des deux couvertures).
\item Geler une seule paire pour les premiers tests~: EUR/USD (la mieux
fournie côté broker MT5 \emph{après} le réimport \code{data/*\_mt5.parquet}).
\item Exporter les données broker MT5 vers Parquet via
\code{src/mt5/Scripts/FxExportRates.mq5} si pas déjà fait, et utiliser
\code{data/EUR-USD\_minute\_mt5.parquet} dans VBT (élimine H4 d'entrée).
\end{enumerate}

### 3.2 Tableau de mesures à recueillir

Pour chaque test, mesurer~:
\begin{itemize}
\item Nombre de trades RSI Daily.
\item Sharpe annualisé (252~jours).
\item Max Drawdown.
\item Liste des dates d'entrée/sortie pour la première année (2021).
\item Liste des valeurs RSI au moment des crossings (10~premiers signaux).
\end{itemize}

\textbf{Format de sortie suggéré}~: tableau Markdown ou CSV dans
\code{reports/investigations/rsi\_daily/<test\_id>\_<timestamp>.csv}.

### 3.3 Ordre d'attaque recommandé

\begin{enumerate}
\item \textbf{H4 d'abord} (aligner les données)~: c'est le seul test qui
touche \emph{toutes} les hypothèses suivantes. Sans même source de
données, les autres tests sont biaisés.
\item \textbf{H7 ensuite} (vérifier le RSI numériquement). Si le RSI lui-même
diffère, tout le reste est moot.
\item \textbf{H1 puis H2} (sizing et vol-target)~: probablement le facteur
dominant en magnitude.
\item \textbf{H3, H5, H6, H8} en suivant.
\end{enumerate}

### 3.4 Outils à instrumenter

\begin{itemize}
\item Côté VBT~: ajouter un mode \code{verbose=True} dans \code{pipeline}
qui retourne les signaux sous forme de DataFrame
(\code{entries, exits, short\_entries, short\_exits, rsi}).
\item Côté MT5~: dans \code{FxSleeveRSIDaily.mqh}, ajouter un log
\code{[RSI][TRACE]} de chaque ligne de RSI lue (sortir uniquement quand
\code{Inp\_LogVerbose=true}).
\item Script Python~: \code{scripts/investigations/diff\_rsi\_daily.py}
qui lit le log Tester MT5 (UTF-16 LE) et compare ligne à ligne avec la
trace VBT.
\end{itemize}

---

## 4. Procédure de référence — reproduire chaque mesure

### 4.1 Mesure VBT \enquote{vanilla}

\begin{verbatim}
cd /home/thomas/Documents_Thomas/11_CodingProjects/fx_strategies/fx_strategies
python -m strategies.rsi_daily        # SINGLE RUN sur PROJECT_CONFIG default_pair
\end{verbatim}

Pour modifier les paramètres, éditer \code{SINGLE\_PARAMS} en bas du fichier
ou créer un script \code{scripts/investigations/rsi\_vbt\_param\_sweep.py}.

### 4.2 Mesure MT5 isolée sur le sleeve~3

\begin{enumerate}
\item Régénérer le preset~: \code{python src/mt5/bridge/write\_default\_preset.py}.
\item Éditer le \code{.set} pour mettre \code{Inp\_AllocMRMacro=0},
\code{Inp\_AllocTSMomentum=0}, \code{Inp\_AllocRSIDaily=1.0}.
\item Lancer~: \code{python src/mt5/bridge/run\_backtest\_cli.py
--report-name fx\_rsi\_only}.
\item Récupérer \code{reports/mt5/run\_<timestamp>.json}~--- contient les
métriques HTML extraites.
\end{enumerate}

### 4.3 Diff ligne à ligne

À implémenter~: \code{scripts/investigations/diff\_rsi\_daily.py}~:
\begin{enumerate}
\item Charger les RSI VBT par paire en DataFrame.
\item Parser le log Tester MT5 (UTF-16~LE) pour extraire les lignes
\code{[RSI][TRACE] <date> <symbol> rsi=<val>}.
\item Joindre sur \code{(date, symbol)}, calculer \code{abs(diff)}.
\item Émettre une table~: \code{date | symbol | rsi\_vbt | rsi\_mt5 | diff}.
\item Cible~: $\max(\text{diff}) < 0.1$ après warmup ($> 30$~bars).
\end{enumerate}

---

## 5. Critères de succès et tolérances

L'investigation est terminée quand~:

\begin{enumerate}
\item Le RSI numérique converge à $< 0.1$ (absolu) entre VBT et MT5 sur
$\geq 95\,\%$ des dates après warmup.
\item Le \emph{nombre de signaux RSI} (entries~+~exits) est identique à
$\pm 5\,\%$ près sur la fenêtre commune 2020-11 → 2026-04.
\item Le \emph{signe} des returns daily mensuels est identique à $\geq 90\,\%$.
\item L'écart Sharpe est documenté et expliqué (par exemple~: \enquote{MT5
Sharpe = VBT Sharpe / 1.4 à cause du vol-targeting global}).
\item Les sources d'écart résiduelles sont \emph{listées et justifiées}
(par exemple~: \enquote{accepté que le slippage broker ajoute 0.05~Sharpe
en faveur de VBT~--- non bloquant pour la mise en prod}).
\end{enumerate}

---

## 6. Risques et scope creep

\begin{itemize}
\item \textbf{Tentation de fixer les écarts au lieu de les comprendre.}
Avant de patcher l'une ou l'autre implémentation, \emph{toujours}
comprendre la cause. Un fix qui colle les chiffres sans comprendre
masque souvent un bug structurel.
\item \textbf{Aller trop loin.} Le RSI Daily est un sleeve à 10\,\% du
portefeuille, contributeur diversificateur. Une convergence à $\pm 5\,\%$
sur le Sharpe est suffisante~--- ne pas chercher la précision $1$~ppm.
\item \textbf{Altérer la stratégie en croyant la corriger.} Toute
modification de \code{rsi\_daily.py} doit passer par~:
(a)~test bit-equivalence avec l'historique \code{combined\_portfolio.py}
qui a produit les chiffres du rapport client (Sharpe~OOS 1.44 ne doit
pas changer après l'investigation).
\item \textbf{Investigation qui dérive sur les autres sleeves.} Garder
focus sur RSI Daily~; les sleeves~1 et~2 sont hors scope. Si l'enquête
remonte un bug commun (ex.~: vol-targeting), traiter dans un ticket
séparé.
\end{itemize}

---

## 7. Outputs attendus

À la fin de l'investigation, livrer~:

\begin{enumerate}
\item \code{reports/investigations/rsi\_daily/findings.md}~--- résumé
exécutif, hypothèses validées, fix appliqué, tolérance résiduelle.
\item \code{reports/investigations/rsi\_daily/numerics.csv}~--- comparaison
ligne à ligne RSI VBT vs MT5 sur EUR/USD.
\item \code{reports/investigations/rsi\_daily/equity\_curves.png}~---
courbes d'équité superposées, fenêtre commune.
\item Mise à jour de \code{src/mt5/CLAUDE.md} si un fix est appliqué côté
MQL5, ou de \code{src/strategies/rsi\_daily.py} si fix côté VBT.
\item Mise à jour de \code{src/mt5/SESSION\_NOTES.md} pour fermer ce
ticket.
\end{enumerate}

---

## 8. Pour reprendre cette investigation en nouvelle session

Un agent fresh peut reprendre l'enquête en lisant, dans cet ordre~:

\begin{enumerate}
\item \code{src/mt5/SESSION\_NOTES.md}~--- état global du projet MT5.
\item \code{src/mt5/CLAUDE.md}~--- environnement opérationnel.
\item \code{docs/mt5/14\_cli\_backtest\_linux.md}~--- comment lancer un
backtest CLI sur Linux/Wine.
\item Ce document.
\item \code{src/mt5/Include/FxSleeveRSIDaily.mqh}~+
\code{src/strategies/rsi\_daily.py} en parallèle.
\end{enumerate}

\textbf{Commande de démarrage à froid}~:
\begin{verbatim}
# 1. Vérifier que le baseline MT5 reproduit Sharpe 1.15
python src/mt5/bridge/run_backtest_cli.py
# Inspecter reports/mt5/run_<timestamp>.json

# 2. Vérifier que VBT tourne
python -m strategies.rsi_daily

# 3. Commencer par H4 — exporter données broker, re-runner VBT dessus
\end{verbatim}
