# Parité vbt ↔ QuantConnect — sleeve Gold Momentum

**Date** : 2026-07-25 · **Phase 2** du plan de réconciliation ·
**Référence QC** : projet `34489845`, LEAN 2.5.0.0.17941, backtest `92984793…`

---

## Méthode : attribution différentielle, faute de traces diffables

Les deux moteurs mangent les mêmes octets — `data/XAU-USD_minute_qc.parquet` a été exporté
de QuantConnect. Tout écart restant est donc de la sémantique de moteur, et c'est ce qui
rend l'**attribution** possible plutôt qu'une simple tolérance.

La méthode prévue — differ les deux traces journalières — s'est heurtée à la plateforme :
l'export de l'ObjectStore est réservé aux comptes Institutional et le serveur MCP n'expose
aucun point d'entrée pour les logs de backtest. La trace QC est produite, mais pas
récupérable par API.

D'où une approche plus forte que le diff : partir de la sleeve vbt et appliquer les
conventions QC **une par une**, en mesurant ce que chacune vaut. Si les métriques QC
publiées sont retrouvées au bout, l'attribution est complète et chaque poste est nommé.
Script : `scripts/attribute_vbt_qc_gold.py`.

## Le tableau d'attribution

Chaque ligne conserve les précédentes ; la dernière est « vbt avec toutes les conventions
QC ». Écarts en points de pourcentage sur la ligne précédente.

| étape | CAGR | vol | Sharpe | maxDD | trades | ΔCAGR | ΔmaxDD |
|---|---|---|---|---|---|---|---|
| vbt, bornes minuit *(état d'avant)* | 18.65 % | 23.74 % | 0.700 | 46.51 % | 50 | — | — |
| **+ bornes 17:00** | 16.41 % | 23.54 % | 0.742 | 51.74 % | 47 | **−2.24** | **+5.23** |
| + σ sur rendements log | 16.37 % | 23.54 % | 0.741 | 51.80 % | 47 | −0.04 | +0.06 |
| + plancher de σ à 0.05 | 16.37 % | 23.54 % | 0.741 | 51.80 % | 47 | **0.00** | **0.00** |
| + sans décalage causal | 17.72 % | 23.67 % | 0.785 | 49.85 % | 47 | **+1.34** | −1.95 |
| + fill au T+1 open | 17.24 % | 23.46 % | 0.773 | 50.30 % | 47 | −0.48 | +0.45 |
| **QC (publié)** | **20.17 %** | **23.30 %** | **0.575** | **51.90 %** | 128 ordres | | |

Trois enseignements, dont deux inattendus :

**1. La borne de journée domine tout le reste.** Elle vaut à elle seule −2.24 pp de CAGR et
+5.23 pp de drawdown, et elle réconcilie le maxDD presque exactement : 51.74 % contre
51.90 %. Ce n'était pas une divergence de convention entre moteurs mais **un défaut de la
sleeve vbt** : découper au jour calendaire fabriquait 392 séances dominicales de ~356
minutes, gonflant le compte de séances de 20 % et raccourcissant tous les lookbacks
d'autant. Corrigé (`gold_momentum.session_dates`), donc la ligne 2 est désormais l'état de
production.

**2. Deux écarts inventoriés ne valent rien.** σ sur rendements log plutôt qu'arithmétiques
pèse 0.04 pp ; le plancher de volatilité à 0.05 pèse **exactement zéro** — il ne mord
jamais, la volatilité de l'or ne descendant pas sous 5 %. Les aligner serait du travail
sans effet. Cela valait la peine de le mesurer plutôt que de le supposer.

**3. Le décalage causal du levier est le deuxième poste** (+1.34 pp de CAGR quand on le
retire). vbt dimensionne sur la volatilité de la veille, QC et MT5 sur celle du jour. C'est
un vrai choix de conception, pas un bug : vbt est le plus conservateur des trois.

## Les deux postes hors du tableau

**Le warmup, et il pèse ~2 pp.** QC appelle `set_warm_up(252, Resolution.DAILY)`, qui
consomme de l'historique **antérieur** à la date de départ : son premier ordre tombe le
**2019-01-02**. vbt n'a pas de données avant 2019-01-01 et perd ses 250 premières séances,
donc ne trade qu'à partir du **2019-10-25**. QC engrange 140 258 $ d'équité avant même que
vbt ne commence — 2019 fut une bonne année pour l'or.

Recalculé sur la période réellement commune, à partir de la courbe d'équité QC :

| | CAGR |
|---|---|
| QC, période complète | 20.41 % *(publié 20.17 %)* |
| QC, à partir du 2019-11-01 | **18.11 %** |
| vbt aligné sur QC | **17.24 %** |
| **résidu** | **+0.87 pp** *(contre −2.93 pp avant correction)* |

**Le Sharpe n'est pas comparable tel quel.** Le rapport de validation antérieur note
systématiquement « Sharpe (rf=0) » : la statistique Sharpe native de QC n'est donc pas à
taux sans risque nul, alors que vbt calcule à rf=0. Comparer 0.773 à 0.575 n'a pas de sens,
et l'écart de +0.198 ne mesure aucune différence de performance. Poste identifié, non
chiffré : la formule exacte de QC n'est pas documentée côté projet.

## Où en est la cible, et ce qui manque

Cible de la phase 2 : ≤ 2 % en relatif sur CAGR, volatilité et maxDD.

| métrique | vbt | QC | écart relatif | cible |
|---|---|---|---|---|
| volatilité | 23.46 % | 23.30 % | **0.7 %** | ✅ |
| maxDD | 50.30 % | 51.90 % | **3.1 %** | ⚠️ |
| CAGR *(période commune)* | 17.24 % | 18.11 % | **4.8 %** | ⚠️ |

La volatilité est réconciliée. Le maxDD et le CAGR restent au-dessus de la cible, et le
résidu a un suspect nommé : **le nombre de transitions**. Sur la période commune QC compte
57 allers-retours contre 47 pour vbt. Les barres journalières natives OANDA de QC ne sont
pas exactement le ré-échantillonnage des minutes que reconstruit ce rapport, et quelques
franchissements de zéro supplémentaires suffisent à expliquer l'écart restant.

**Fermer ce résidu demande la trace QC**, donc une récupération manuelle depuis l'interface
web (Object Store `gold_trace_qc.csv`, ou les lignes `TRACE,` du journal). Le barreau 1 se
diffe alors directement au lieu d'être reconstruit, et l'hypothèse se vérifie en une passe
de `scripts/reconcile_three_way.py`.

## Reproduire

```bash
python scripts/attribute_vbt_qc_gold.py     # le tableau d'attribution
python scripts/reconcile_three_way.py --vbt <trace_vbt.csv> --qc <trace_qc.csv>
```
