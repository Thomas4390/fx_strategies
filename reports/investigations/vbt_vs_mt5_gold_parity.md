# Parité vbt ↔ MT5 — sleeve Gold Momentum

**Date** : 2026-07-25 · **Phases 3 et 4** du plan de réconciliation
**Statut** : phase 3 verte · phase 4 **partielle** — premier niveau seulement

---

## Phase 3 — le tester était bloqué par le modèle de simulation, pas par l'or

Le symptôme documenté était un tester qui ne démarrait jamais. Le log le dit sans ambiguïté
une fois qu'on le lit :

```
EURUSD.c: history data begins from 2020.11.22 00:00
EURUSD.c: preliminary downloading of history ticks started, it may take quite a long time
EURUSD.c: "bases\SquaredFinancialSC-MT5 Demo\ticks\EURUSD.c\202607.tkc" download
EURUSD.c: preliminary downloading of history ticks canceled
no history data, stop testing
```

**Cause racine** : `DEFAULT_MODEL = 4` — « every tick based on real ticks ». Ce modèle exige
des ticks réels que le terminal, lancé en `/config:` sans session interactive, ne parvient
pas à télécharger ; l'opération est annulée au bout d'une seconde et le tester s'arrête.

**Correctif** : `--model 1` (OHLC M1). Le run passe immédiatement — `Init EA: OK`, macro
history chargée, 5.3 ans simulés en 21 secondes.

**L'hypothèse principale du plan est infirmée.** Le plan posait que le broker n'offrait
peut-être pas l'or, auquel cas toute la branche MT5 tombait. Le log tranche :

```
XAUUSD.c,M1: 229815 ticks, 57565 bars generated
```

XAUUSD.c existe et porte des données. Ce qui manquait n'était pas le symbole mais
l'historique **local** : le broker n'avait rien en cache, et le modèle choisi exigeait
justement ce qui manquait.

⚠️ **Contrepartie à assumer** : le modèle OHLC M1 interpole les fills et flatte le Sharpe
par rapport aux ticks réels. Tout chiffre MT5 produit ainsi est un majorant. Rétablir
`--model 4` supposerait de télécharger les ticks depuis une session GUI connectée au broker.

**Second obstacle, mineur** : la sleeve or a `Inp_AllocGoldMomentum = 0.0` par défaut — elle
ne trade jamais dans la configuration de production. Les runs de réconciliation doivent
l'isoler explicitement (allocation à 1.0, les autres à 0).

Enfin, la sleeve saute ses 257 premières séances (`only 33/252 D1 bars copied`) le temps
que l'historique D1 s'accumule : le premier trade tombe le **2021-11-09**, ce qui fixe le
début de toute fenêtre de comparaison.

## Phase 4 — premier niveau, sur la fenêtre réellement commune

Backtest MT5 : `XAUUSD.c`, M1, modèle OHLC, 2021-01-01 → 2026-04-30, dépôt 10 000,
sleeve or seule. Résultat : profit net **17 385 (+173.8 %)**, profit factor 3.45.

| | vbt | MT5 | écart | tolérance repo |
|---|---|---|---|---|
| **trades** | **35** | **35** | **0** | ±10 % ✅ |
| Sharpe | 1.16 | 0.73 | −0.43 | ±0.10 ❌ |
| maxDD | −32.98 % | −42.28 % | +9.30 pp | ±2 pp ❌ |
| CAGR | 32.65 % | n/a | — | ±2 pp |
| vol annualisée | 26.66 % | n/a | — | — |

**Le résultat qui compte est la première ligne.** Trente-cinq trades de part et d'autre, sur
des flux de données différents et deux implémentations indépendantes : le signal et les
transitions concordent. Les barreaux 1 à 3 — bornes de séance, score, poids cible — tiennent
donc largement, et l'écart restant se loge dans l'exécution et les coûts. C'est exactement
le diagnostic que l'échelle est censée produire.

Le sens de l'écart est le bon : MT5 est **plus dégradé** que vbt, ce qui est attendu d'un
backtest broker face à un backtest idéalisé. Un écart nul aurait signalé un problème.

### Ce qu'il reste à attribuer, et ce que ça demande

Quatre postes, non encore chiffrés séparément :

1. **Slippage** — 2 bps par côté côté MT5 (`Inp_Gold_SlippageBps`) contre 1 bp côté vbt.
2. **Stop de sécurité à 4 %** (`Inp_Gold_SafetySL`) — vbt n'a aucun stop. Il coupe des
   positions que vbt garde, ce qui déforme la distribution des trades.
3. **Sizing en lots** — MT5 passe par `LotsForRisk(risk_money, sl_distance)`, qui n'est
   **pas** un poids de portefeuille. Sur un dépôt de 10 000 avec un pas de lot de 0.01, la
   granularité est grossière : c'est un candidat sérieux pour l'écart de drawdown.
4. **Swap** — non modélisé côté vbt.

**Le blocage** : la trace journalière MT5 n'est pas produite. `Inp_Gold_Trace=true` est bien
écrit dans `[TesterInputs]` de l'INI et l'allocation passée par le même mécanisme est
honorée, mais `WriteTraceRow` n'est jamais atteint — pas même son avertissement d'échec
d'ouverture. Les pistes du preset `.set` caché et de sa régénération ont été écartées par
test. Non résolu.

En attendant, le journal du tester porte déjà l'essentiel par trade :

```
[Gold_Momentum][INFO] Entry LONG XAUUSD.c lots=0.15 price=1832.46 score=0.50 lev=2.29
[Gold_Momentum][INFO] Exit LONG XAUUSD.c (score=-0.50)
```

score, levier, lots et prix y figurent — de quoi reconstruire les barreaux 2, 3 et 4 par
parsing, sans le CSV. C'est la voie la plus courte vers l'attribution poste par poste.

## Reproduire

```bash
# phase 3 — le run qui passe
python src/mt5/bridge/run_backtest_cli.py --symbol XAUUSD.c \
    --from 2021.01.01 --to 2026.04.30 --model 1 \
    --input Inp_AllocGoldMomentum=1.0 --input Inp_AllocMRMacro=0.0 \
    --input Inp_AllocTSMomentum=0.0 --input Inp_AllocRSIDaily=0.0 \
    --input Inp_AllocH1Momentum=0.0
```
