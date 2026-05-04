# Plan d'investigation — Expansion des paires FX

> **Date** : 2026-05-04 · **Statut** : ouvert · **Cible** : agent fresh ou Thomas
>
> Plan structuré pour évaluer l'ajout de nouvelles paires de devises au portfolio
> FxMultiSleeve, dans l'objectif de **dépasser le plafond observé de +9.54%
> CAGR OOS** sur la fenêtre 2024-11 → 2026-04 (cf. findings sweep agressif).

## 1. Contexte

### 1.1 Pourquoi élargir les paires

Investigations précédentes ont établi que sur le portfolio combiné actuel :
- Sweep 720 combos × 3 fenêtres : plafond OOS **+9.54%** insensible aux
  paramètres (target_vol, max_lev, vol_floor, DDCap)
- Walk-forward IS/OOS : ρ Spearman CAGR = +0.85 (stable mais plafonné)
- L'edge actuel des 4 paires est saturé

→ Hypothèse : **diversification = ajouter des edges décorrélés**, pas
augmenter le levier sur les mêmes paires.

### 1.2 Univers actuel

| Sleeve | Paires actuelles | Allocation |
|---|---|---|
| MR Macro (M1) | EUR/USD, GBP/USD, USD/JPY, USD/CAD | 80% |
| TS Momentum (D1) | EUR/USD, GBP/USD, USD/JPY (CAD exclu) | 10% |
| RSI Daily (D1) | EUR/USD, GBP/USD, USD/JPY, USD/CAD | 10% |

### 1.3 Univers cible (à tester)

**G10 majors** : ajout USD/CHF, AUD/USD, NZD/USD
**Crosses majeurs** : EUR/GBP, EUR/JPY, GBP/JPY, AUD/JPY
**EM (optionnel)** : USD/MXN, USD/ZAR, USD/TRY (très spread, à évaluer)

---

## 2. Hypothèses ordonnées par valeur attendue

### H-A : Ajout USD/CHF (G10 safe-haven, décorrélation EUR/USD)

**Rationale** : CHF traditionnellement contra-cyclique. Corrélation USD/CHF
↔ EUR/USD typiquement -0.85 → mais en termes de signaux MR/TS, les régimes
peuvent diverger. Spread broker SquaredFinancial typique ≈ 0.5 pip ECN.

**Coût estimé** : médian. Spread ≈ EUR/USD.

**Test** :
- Ajout dans `Inp_MR_Pairs`, `Inp_TS_Pairs`, `Inp_RSI_Pairs`
- Re-run 4-pair vs 5-pair sur 5.4 ans
- Métrique : ΔSharpe / ΔCAGR / impact corrélation inter-sleeves

### H-B : Ajout AUD/USD + NZD/USD (G10 risk-on)

**Rationale** : AUD et NZD = "commodity currencies", forte corrélation
avec risk-on / risk-off et matières premières. Edge potentiel pour TS
Momentum (régimes plus durables). MR Macro probablement moins
performant (volatilité moins prévisible que les majors USD).

**Coût estimé** : modéré. Spread NZD plus large (typique 1-2 pips ECN).

**Test** :
- Évaluer chaque paire isolément avant ajout combiné
- Mesurer Sharpe standalone du sleeve TS Momentum sur AUD seul, NZD seul

### H-C : Crosses EUR/GBP, EUR/JPY, GBP/JPY (G10 non-USD)

**Rationale** : crosses = différentiel taux entre 2 banques centrales.
EUR/GBP très lent (carry possible), EUR/JPY et GBP/JPY plus volatils
(corrélation forte avec risk-on). **Décorrélation USD = edge orthogonal**.

**Coût estimé** : modéré-élevé. EUR/JPY spread ≈ 1-1.5 pip ECN.

**Test** :
- Particulièrement pour TS Momentum daily : crosses ont des trends plus
  durables que paires USD majeurs
- RSI Daily également pertinent (mean reversion sur crosses)

### H-D : EM (USD/MXN, USD/ZAR) — exploratoire

**Rationale** : carry FX traditionnellement profitable mais coût de
transaction élevé (spread 5-15 pips ECN). Risque de gap (devises
fragiles). Probablement non rentable retail.

**Coût estimé** : ÉLEVÉ. À évaluer en dernier.

**Test** :
- Skip si coûts > 50% de l'edge brut
- Sinon, à mettre dans un sleeve dédié "EM Carry" avec allocation max 5%

---

## 3. Méthodologie

### 3.1 Pré-requis infra (avant tout test)

1. **Vérifier disponibilité broker** : pour chaque paire candidate,
   confirmer présence dans MarketWatch SquaredFinancialSC-MT5 Demo avec
   suffixe `.c`.
   ```bash
   # Script à créer : check_pair_availability.mq5
   for symbol in USDCHF.c AUDUSD.c NZDUSD.c EURGBP.c EURJPY.c GBPJPY.c:
       SymbolSelect(symbol, true)
       SymbolInfoTick(symbol, tick) → vérifier bid/ask non-zero
   ```

2. **Pré-télécharger l'historique** broker pour chaque paire (M1 + D1)
   sur 2020-11-23 → 2026-04-30 via `FxDownloadHistory.mq5`.

3. **Mesurer le spread effectif** (bid-ask moyen sur 1 jour) pour chaque
   paire — input critique pour estimer l'overhead vs EUR/USD.

### 3.2 Workflow d'évaluation par paire

Pour chaque paire candidate, **3 mesures** :

#### Mesure 1 — Sharpe standalone par sleeve

Lancer chaque sleeve seul sur la paire seule. Identifie l'edge brut.

```bash
# Sleeve MR Macro sur USD/CHF
python src/mt5/bridge/run_backtest_cli.py \
    --input Inp_AllocMRMacro=1.0 \
    --input Inp_AllocTSMomentum=0 \
    --input Inp_AllocRSIDaily=1.0e-6 \
    --input Inp_MR_Pairs=USDCHF \
    --report-name mrm_usdchf
```

Idem pour TS Momentum et RSI Daily.

#### Mesure 2 — Corrélation des returns vs sleeve actuel

Calculer corrélation des returns daily du sleeve {MR, TS, RSI} sur la
paire candidate vs sleeve actuel sur les 4 paires de base. **Si
corrélation > 0.7, peu d'apport diversificateur**.

#### Mesure 3 — Impact incrémental sur portfolio combiné

Run portfolio complet 4-pair vs 5-pair vs 6-pair etc. Mesurer ΔCAGR_OOS,
ΔSharpe_OOS, ΔMaxDD. **Si ΔSharpe ≤ +0.05, écarter la paire**.

### 3.3 Critères de retenue (filtre quality gate)

Une paire candidate est retenue ssi :

1. **Sharpe standalone ≥ 0.30** sur au moins un sleeve (edge minimum)
2. **Corrélation moyenne returns vs portfolio actuel < 0.50** (apport
   diversificateur)
3. **ΔSharpe portfolio combiné ≥ +0.03** (gain mesurable)
4. **Spread broker < 200% du spread EUR/USD** (coût acceptable)
5. **Pas de PBO > 50%** sur walk-forward IS/OOS (robustesse)

### 3.4 Workflow de décision

```
Pour chaque paire candidate (USD/CHF, AUD/USD, NZD/USD, EUR/GBP,
                              EUR/JPY, GBP/JPY) :

  1. Pre-flight (broker dispo, history téléchargé, spread mesuré)
     → si KO, skip
  2. Mesure 1 : Sharpe standalone par sleeve
     → si tous < 0.30, skip
  3. Mesure 2 : corrélation returns
     → si > 0.50, skip
  4. Mesure 3 : ΔSharpe portfolio
     → si ≥ +0.03, RETAIN
     → sinon, skip
  5. Walk-forward IS/OOS sur la nouvelle config
     → si Spearman ρ < 0.50, REJECT (overfit)
     → sinon, ACCEPT
```

---

## 4. Code à créer

### 4.1 Scripts MQL5

- `Scripts/FxCheckPairs.mq5` — vérifie disponibilité d'une liste de
  paires dans MarketWatch + spread + history.

### 4.2 Scripts Python

- `scripts/optimization/eval_new_pair.py` — orchestre les 3 mesures
  pour une paire candidate.
- `scripts/optimization/walkforward_pairs.py` — walk-forward IS/OOS
  multi-paires (étend `walkforward_aggressive.py`).

### 4.3 Modifications EA (potentielles)

Si l'ajout de paires nécessite de modifier `Inp_*_Pairs`, **pas de
modification du `.mq5`** — passer via `--input Inp_MR_Pairs=...` au
runtime. Aucune recompilation requise.

---

## 5. Ordre d'attaque recommandé

```
Phase 1 (1-2h) : Pre-flight infra
  - Compiler FxCheckPairs.mq5
  - Vérifier broker dispo pour USDCHF, AUDUSD, NZDUSD, EURGBP,
    EURJPY, GBPJPY
  - Pré-télécharger M1+D1 sur 5.4 ans pour les paires dispo
  - Mesurer spread effectif

Phase 2 (1h) : Évaluation par paire
  - Mesure 1 (Sharpe standalone) pour les 6 paires × 3 sleeves
    = 18 backtests, ~10 min sur 32 cores
  - Mesure 2 (corrélation) en post-process Python
  - Filtre selon critères 3.3

Phase 3 (30 min) : Mesure incrémentale
  - Pour chaque paire retenue : run portfolio combiné avec ajout
  - Identifier le subset de paires qui maximise ΔSharpe

Phase 4 (1h) : Walk-forward IS/OOS sur subset retenu
  - Confirme robustesse vs overfit

Phase 5 (30 min) : Synthèse + recommandation
  - findings.md avec optimum candidat
  - Décision : modif Inp_*_Pairs en runtime ou défauts compilés
```

**Budget temps total estimé** : 3-5 heures de calcul + analyse.

---

## 6. Pièges connus à anticiper

1. **Crosses (EUR/JPY etc.)** ont des points/pip différents — vérifier
   le calcul de slippage et SL en MQL5 (`SymbolInfoDouble(SYMBOL_POINT)`).
2. **CAD trades en NY session** (mauvais alignement avec session 6-14 UTC
   du sleeve MR Macro) — TS Momentum daily mieux adapté.
3. **AUD/NZD** ouvrent en Asia session, peuvent générer des entries M1
   pendant les heures où la macro filter est moins fiable.
4. **EM** (MXN, ZAR, TRY) : risque political-driven non modélisable, gap
   risk élevé (weekend coup d'État, dévaluation surprise).

---

## 7. Critères d'arrêt / pivot

- **Si aucune paire ne passe le filtre quality gate** → pivot vers
  d'autres axes : nouvelles timeframes (H1, H4), nouveaux signaux
  (carry, news momentum), changement de broker.
- **Si l'ajout de paires baisse Sharpe global** → garder config 4-pair,
  documenter les paires écartées et raisons.
- **Si ajout débloque CAGR_OOS au-delà de +12%** → procéder au
  walk-forward N=5 fenêtres pour validation finale.

---

## 8. Pour reprendre l'investigation en nouvelle session

Lire dans cet ordre :
1. `src/mt5/SESSION_NOTES.md` — état projet
2. `reports/optimization/walkforward_aggressive/findings.md` — pourquoi
   plafond +9.54% est physique
3. Ce document
4. Lancer Phase 1 puis Phase 2 séquentiellement
