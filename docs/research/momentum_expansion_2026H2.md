# Expansion momentum multi-instruments — cycle 2026-H2

> **Date** : 2026-07-27 · **Statut** : en cours (Phases 0-2 closes, Phase 3 à venir)
> **Holdout state** : LOCKED (frozen from 2026-01-01 until Phase 25 / 2026-12-31).
> Toute sélection de ce document ferme au **2025-12-31**. Aucune lecture FROZEN_OOS
> n'a été consommée par ce cycle à ce jour.

## 0. Objectif et périmètre

La sleeve Gold Momentum produit 79,8 % du résultat du portefeuille avec 35 trades
sur 2021-2026 — trop peu pour la puissance statistique et la régularité du P&L.
Objectif : **~100+ trades agrégés** via des sleeves momentum validées sur de
nouveaux instruments (même moteur TSMOM), puis d'autres familles momentum.
Production hors scope de ce cycle. Périmètre validé par le propriétaire le
2026-07-27 (univers FX + non-FX broker, quatre familles, livrable = recherche +
rapport trade par trade du meilleur candidat).

Leçon fondatrice (réfutation des lookbacks courts, `reports/mt5/gold_sweep.csv`) :
le Sharpe MT5 décroît avec la fréquence quand le gain vbt vient de l'exécution
idéalisée. **Plus de trades = plus d'instruments, pas plus de fréquence.**

## 1. Données et univers (Phase 1)

- **Catalogue broker** (`src/mt5/Scripts/FxListSymbols.mq5` → 272 symboles) :
  univers retenu = 10 paires FX existantes + XAGUSD.c, XTIUSD, XBRUSD, XNGUSD,
  US500Cash, US100Cash, US30Cash, GER40Cash, JPN225Cash, UK100Cash. Écartés :
  DXY (non tradable), futures datés (expirations), actions CFD et crypto (hors
  mandat).
- **Historique broker** des 10 non-FX : téléchargé et exporté
  (`data/*_daily_mt5.parquet`) — limite serveur demo : **2022-11-04**, soit
  ~2,1 ans de trading effectif après le warmup de 250 séances. Suffisant pour la
  validation d'exécution, pas pour un screening puissant.
- **Séries longues de screening** (`data/*_daily_yahoo.parquet`, 1990-2000 →) :
  l'export object-store QuantConnect est réservé aux comptes institutionnels ;
  repli sur l'API chart Yahoo (SI=F, CL=F, BZ=F, NG=F, indices cash).
  Réparations documentées dans `scripts/investigations/download_screening_daily.py`
  (barres OHLC incohérentes des vieux futures, WTI négatif d'avril 2020 retiré).
- **Représentativité** (`scripts/investigations/check_screening_vs_broker.py`,
  verdicts dans `reports/research/screening_source_check.json`) : corr quotidienne
  au meilleur lag + corr hebdomadaire + ratio de vol sur la fenêtre commune.
  US500/US100/US30 `LONG_OK` ; XAG/XTI/XBR/GER40/UK100 `FLAGGED` (corr hebdo
  ~0,94-0,96, bruit de conventions de clôture) ; XNG (`vol_ratio` 1,45 — contango
  du front-month vs CFD spot) et JPN225 (corr 0,58 même au meilleur lag)
  `BROKER_ONLY`.

## 2. Le verrou méthodologique : quelle simulation prédit MT5 ? (Phase 2, étape 0)

Le plan prévoyait de neutraliser le biais d'exécution vbt (décision et fill au
même close) par `fill="next_open"` et d'exiger un accord ±15 % avec le MT5
publié. **Cette sanity a échoué, et le résultat est instructif** :

| Convention vbt (or, fenêtre de parité 2021-01 → 2026-04) | Sharpe |
|---|---|
| fill au close décideur (historique) | 1.150 |
| fill `next_open` | 1.171 |
| fill `next_open` + slippage 2 bps | 1.154 |
| + swap-drag 0,5 bp/nuit × levier | 1.082 |
| **MT5, sleeve isolée, même fenêtre** | **0.73** |

Sur une grille lente (35 trades / 5 ans, or quasi 24h), l'open suivant est à une
heure du close décideur : le biais de fill est **négligeable à cette fréquence**
— il ne devient dominant que sur les grilles rapides (leçon 15/30/60). Le swap
ferme 0,07 de Sharpe. Le résidu (~0,35) vient des postes non modélisables côté
recherche : sizing en lots (`LotsForRisk`, granularité, planchers/plafonds de
volume), levier non décalé côté MQL5, borne de décision 21:00 UTC vs 17:00 NY
l'hiver, interpolation OHLC M1.

**Décision de protocole** (application de la doctrine « les chiffres publiés
sont ceux du moteur qui exécute ») :
- le screening vbt long (données 1990→) n'est qu'un **pré-filtre d'edge brut**
  (kill si Sharpe net < 0) en `next_open` + coûts empiriques par symbole
  (`costs.yml`) + swap-drag 0,5 bp/nuit × exposition ;
- le **classement se fait dans le tester MT5** (`scripts/sweep_tsmom_mt5.py`),
  sleeve isolée par symbole, config or de production épinglée, ~15 s/run.

## 3. Deux pièges d'environnement découverts (et fermés)

1. **`Inp_RiskScale=4.5` est le défaut compilé depuis le 2026-07-26**, calibré
   pour le portefeuille (or à 10 %). En sleeve isolée à `Alloc=1.0`, il porte le
   levier effectif à ~27× et **ruine le compte** (net −9 981 $ contre +45 596 $ à
   `RiskScale=1.0`, mêmes barres, mêmes lookbacks). Les chiffres du
   `gold_sweep.csv` du 26-07 tournaient sous l'ancien défaut. Depuis :
   `sweep_tsmom_mt5.py` épingle **tous** les inputs de sizing dans la commande.
   Règle générale : *un input non épinglé dans un script de sweep est une bombe à
   retardement — il prend la valeur d'un défaut compilé ou d'un cache `.set` qui
   changera.*
2. **Les indices Cash du broker ont des plafonds de volume bas** (~12-19 lots).
   En sleeve isolée à gros dépôt, le cap neutralise le vol-targeting (levier
   effectif 0,02-0,5× pour ~4,5× visé) : les Sharpe MT5 des indices mesurent le
   *signal sous cap*, pas la stratégie vol-ciblée. En production réelle
   (allocation ~10 % d'un compte 10 k), le cap ne mord pas. Les runs indices
   sont donc **indicatifs**, à re-valider en configuration taille production.

## 4. Résultats Phase 2

### 4.1 Pré-filtre vbt (`reports/research/tsmom_screen_2026H2.csv`)

Config unique = défauts or (40/60/120/250, tv 0,55, cap 6,6, long-only),
`next_open`, coûts par symbole, swap-drag, fenêtre ≤ 2025-12-31. 21 instruments,
1 config — trials logués (`tsmom_universe`, n=21).

Tête de classement (Sharpe net) : **US100 0,78 (36 ans, 209 trades)**,
XAU-USD 0,74 (contrôle), GER40 0,53, US500 0,51, **USD-JPY 0,51 (8 ans)**,
JPN225 0,45, EUR-JPY 0,43, US30 0,37. Kills (edge négatif long) : EUR-USD,
GBP-USD, EUR-GBP, USD-CAD, USD-CHF, AUD-USD, NZD-USD. Énergie/argent en séries
à rolls : 0,07-0,22 (les rolls polluent le signal — le CFD broker propre est
plus favorable, cf. MT5).

### 4.2 Classement MT5 (`reports/mt5/tsmom_universe_sweep.csv`)

Sleeve isolée, dépôt 100 k, `RiskScale=1.0`, fenêtres 2021-01 (11 historiques)
ou 2022-11 (nouveaux) → 2025-12-31 :

| symbol | from | CAGR | maxDD | Sharpe | trades |
|---|---|---|---|---|---|
| USDJPY | 2021 | 25,9 % | 49,4 % | **0,814** | 31 |
| JPN225Cash* | 2022-11 | 0,3 % | 0,9 % | 0,806* | 29 |
| XAUUSD | 2021 | 41,1 % | 76,7 % | 0,763 | 31 |
| XAGUSD | 2022-11 | 29,8 % | 66,3 % | 0,436 | 18 |
| GBPUSD | 2021 | 7,3 % | 39,5 % | 0,364 | 36 |
| GBPJPY | 2021 | 9,8 % | 59,4 % | 0,264 | 33 |
| GER40Cash* | 2022-11 | 6,5 % | 58,4 % | 0,167 | 22 |
| EURJPY | 2022-11 | 21,9 % | 48,5 % | 0,156 | 11 |
| US500Cash* | 2022-11 | 1,9 % | 5,6 % | 0,137 | 11 |
| US100Cash* | 2022-11 | 6,8 % | 53,0 % | 0,123 | 16 |
| US30Cash* | 2022-11 | 3,2 % | 52,1 % | 0,058 | 17 |
| (le reste) | | négatif | | −0,48 à −2,18 | |

\* indices sous cap de volume — voir §3.2.

### 4.3 Lecture croisée et survivants

- **USD-JPY — survivant n°1.** Positif sur les deux moteurs (0,51 vbt/8 ans,
  0,81 MT5/5 ans), 31 trades, corr quotidienne à l'or **−0,13**. Réserve : la
  fenêtre 2021-2024 contient le mégatrend yen ; le vbt 8 ans (dont 2018-2020
  plats) tient à 0,51, ce qui limite le soupçon de fenêtre.
- **XAG-USD — survivant n°2 (fragile).** MT5 0,44 sur 2 ans de CFD propre ;
  la série longue à rolls ne l'appuie qu'à 0,20. Corr à l'or 0,47 — sous le
  seuil 0,50, mais le cluster métaux est réel.
- **US100 — survivant n°3 (edge long, exécution à confirmer).** Le meilleur
  edge structurel du screening (0,78 / 36 ans / 209 trades, corr or ~0), mais le
  MT5 court sous cap ne le confirme qu'à 0,12. À re-valider en taille
  production avant toute promotion. US500/US30/GER40/JPN225 forment un cluster
  avec lui — un seul représentant.
- **Rejets cohérents** : toutes les majors dollar (les deux moteurs d'accord),
  UK100, l'énergie (XTI/XBR marginaux, XNG tué par ~55 bp de spread médian —
  `costs.yml`), EUR-GBP. GBP-USD (MT5 0,36 court vs vbt −0,07 long) : rejeté,
  l'historique long fait foi contre une fenêtre courte favorable.
- **Volumétrie** : or 35 + USDJPY ~31 + XAG ~35-40 (5 ans, à confirmer) +
  US100 ~30 → **la cible ~100+ trades agrégés est atteignable avec 3 sleeves
  additionnelles**, sans toucher à la fréquence du moteur.

## 4.4 Passe 2 des survivants (`scripts/stress_tsmom_survivors.py`)

- **Stabilité** (9 configs de voisinage par instrument, aucune resélection) :
  plateau pour l'or (0,70-1,16), USD-JPY (0,76-0,95) et XAG (0,30-0,43) ; US100
  plus sensible aux horizons (0,68 sur la grille 30/50/100/200 contre 1,03 en
  production) — l'edge indices est réel sur 36 ans mais sa forme dépend plus du
  choix d'horizons que celui des métaux/yen. Flag conservé au dossier.
- **DSR** : avec le n_trials honnête du registre (~500 essais cumulés, rivaux =
  la dispersion du classement MT5), **aucun survivant n'atteint un DSR > 0** —
  le screening désigne des candidats, il ne prouve aucun edge. Ce qui soutient
  réellement chaque candidat : US100 — 36 ans, 209 trades, t-stat ≈ 4,7
  standalone ; USD-JPY — cohérence des deux moteurs et orthogonalité ; XAG —
  le plus fragile des trois (2 ans propres + série longue polluée par les
  rolls). Le registre sur-compte les re-runs du même stress (biais
  conservateur, assumé).
- **Corrélations quotidiennes** (config production, in-sample, alignement par
  date calendaire) : USD-JPY orthogonal à tout (−0,005 vs or) ; US100
  orthogonal (0,01 vs or) ; seul cluster réel : or-argent à 0,469 (sous le
  seuil de 0,50, mais un poids commun « métaux » devra le refléter).
- **allow_short (famille d) — TUÉE en 3 trials** : la jambe short détruit du
  Sharpe sur les trois survivants (XAU −0,19, USD-JPY −0,32, XAG −0,23) tout en
  doublant les trades. La volumétrie par le short est de la mauvaise
  volumétrie. In-sample seulement (tranche holdout or épuisée), mais l'ampleur
  et l'unanimité suffisent au kill.

## 4.5 Familles nouvelles (Phase 3)

- **allow_short — TUÉE** (§4.4).
- **Dual/acceleration momentum — TUÉE** (`reports/research/dual_screen_2026H2.csv`,
  4 configs + référence, 14 instruments). Le meilleur agrégé (brake 21/126,
  Sharpe moyen 0,369) ne bat pas le TSMOM non filtré (0,375) avec une
  corrélation de panier de **0,973** ; le gate strict (long seulement si
  l'accélération est positive) coupe ~40 % des trades et effondre l'agrégé à
  0,08 — la décélération n'est pas un prédicteur de perte sur ce moteur lent.
  La baseline recalculée reproduit les Sharpe du screening au bit près (mêmes
  conventions garanties).
- **Donchian — REJETÉE** (`reports/research/donchian_screen_2026H2.csv`,
  6 configs, 14 instruments). Le gate Sharpe passe (0,437 agrégé pour
  entry=252/exit=126, 10/14 instruments ≥ 0,30) mais : (a) **redondance** —
  corr de panier 0,790 avec le TSMOM (0,73-0,75 par instrument : le même flux
  de rendements) ; (b) **contre-productivité volumétrique** — la config
  gagnante fait 0,2-0,6 trade/an/instrument (exposition médiane 0,85 : un
  quasi buy-and-hold vol-ciblé), à l'opposé de l'objectif du cycle, et les
  configs rapides (55/13 : ~49 trades/an agrégés) perdent le Sharpe (0,278)
  avant même le mur de la fréquence côté MT5. Une famille breakout n'ajoute
  ni un pari distinct ni des trades utiles sur cet univers.
- **XS momentum — PASS formel, NON PROMUE ce cycle**
  (`reports/research/xs_screen_2026H2.csv`, 6 configs, panel des 14 PASS).
  Meilleur panier : lb252 long-only top-3, Sharpe 0,609, corr 0,66 au panier
  TSMOM — additif en théorie de portefeuille (0,609 > 0,66 × 0,856 = 0,565)
  mais marginalement, et SOUS le panier TSMOM (0,856). La variante long-short
  est vraiment orthogonale (corr 0,15-0,18) et volumétrique (~40 trades/an)
  mais à Sharpe 0,28. Motif de non-promotion : l'XS exigerait une sleeve
  MQL5 multi-symboles à rebalancement — un chantier d'exécution entier —
  pour un apport marginal, quand les sleeves TSMOM atteignent l'objectif
  volumétrique avec l'EA existant (`Inp_Gold_Symbol` runtime). Piste notée
  pour un cycle futur si le mandat volumétrique devait aller au-delà.

**Conclusion Phase 3 : les quatre familles convergent — le portage TSMOM
multi-instruments (USD-JPY, XAG-USD, US100) est la voie du cycle.** Aucune
famille nouvelle n'ajoute un pari à la fois distinct, positif et exécutable.

## 4.6 Vérifications MT5 ciblées (Phase 4)

- **USD-JPY — CONFIRMÉ.** Stress de cible MT5 (lookbacks de production fixes) :
  Sharpe 0,903 / 0,814 / 0,730 pour tv 0,40 / 0,55 / 0,70 — décroissance douce
  par le drag de variance, aucun pic. 31 trades dans tous les cas.
- **US100 — DIFFÉRÉ.** À 10 k, 20 k et 100 k de dépôt, le Sharpe MT5 reste
  ~0,12 : ce n'était pas (que) le cap de volume, c'est la fenêtre broker
  2023-2025 qui ne montre pas l'edge de long terme (0,78 sur 36 ans vbt). Le
  promouvoir sur le vbt seul contredirait la doctrine du moteur exécutant —
  candidat à revoir quand une fenêtre d'exécution plus longue existera.
- **JPN225 — NON EXÉCUTABLE** chez ce broker : notionnel symbolique quel que
  soit le dépôt (CAGR 0,25 %, dd 0,67 % à 20 k) — plafond de volume structurel
  face à un tick value JPY minuscule. Hors liste.

## 4.7 Pré-gel de la lecture FROZEN_OOS (écrit AVANT toute lecture)

Candidats consommant leur tranche : **USD-JPY** et **XAG-USD**. Pas de lecture
pour US100 (différé — ne pas brûler la tranche d'un candidat non promu) ni
pour l'or (tranche 2025-07→2026-07 épuisée, cf. HOLDOUT_POLICY.md).

Configuration figée : TSMOM 40/60/120/250, long-only, tv 0,55, cap 6,6,
`RiskScale=1.0`, sleeve isolée. Fenêtre gelée : 2026-01-01 → fin des données
(broker : 2026-05-01 pour USDJPY, 2026-06-19 pour XAG ; vbt : idem sources).
~4-5 mois, 2-4 trades attendus par instrument : **une lecture de
non-contradiction, pas une confirmation** (puissance quasi nulle).

Prédictions (à confronter, pas à ajuster) :
- USD-JPY : in-sample MT5 Sharpe 0,81, ~6 trades/an. Attendu OOS : 1-3 trades,
  et un rendement dans [−15 %, +25 %] du sous-compte. Contradiction si perte
  > 20 % ou si la sleeve cesse structurellement de trader.
- XAG-USD : in-sample MT5 0,44. Attendu OOS : 2-4 trades, rendement dans
  [−20 %, +30 %]. Contradiction si perte > 25 %.

## 5. Prochaines étapes

1. Phase 3 — familles nouvelles (allow_short or in-sample seulement — tranche
   holdout or épuisée ; Donchian ; dual momentum ; XS momentum si univers
   suffisant), budgets de configs par famille, gates identiques + additivité.
2. Passe 2 des survivants : stress de stabilité (lookbacks ±, tv ±), DSR avec
   n_trials du registre, PBO, corrélations candidat-candidat.
3. Phase 5 — pré-gel de cette note complétée AVANT toute lecture FROZEN_OOS
   (un run par candidat survivant, vbt + MT5 2026), puis rapport LaTeX trade
   par trade du meilleur candidat.

## 6. Registre des décisions de ce cycle

- 2026-07-27 : pivot du protocole — le screening vbt est un pré-filtre, le
  classement se fait sur MT5 (§2). Conséquence de l'échec documenté de la
  sanity ±15 %.
- 2026-07-27 : `RiskScale=1.0` épinglé dans tout sweep sleeve-isolée (§3.1).
- 2026-07-27 : dépôt 100 k pour les sweeps (granularité de lots), indices
  flagués « sous cap » (§3.2).
