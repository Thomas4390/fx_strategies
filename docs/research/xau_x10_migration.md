# XAUUSD x10 — dossier de référence transféré dans fx_apogee

Le 22 septembre 2026, à la demande de Thomas, la stratégie XAUUSD x10 a été
transférée dans `/home/thomas/Documents_Thomas/12_Synerqo/codes/fx_apogee/strategies/xau_x10`.
Les évolutions de cette stratégie doivent désormais se faire dans ce dossier.
Le portefeuille FX historique de ce dépôt reste à son emplacement actuel.

- Rapport final et sources : commit `c0b3ce0af4665dd3b22c3fd648783e8035bbbaad`.
- Archives v1 rendues versionnables, sans modification : commit
  `f27bd7f455900cdef2e2470ad24923235307aceb`.
- Import destination : `Thomas4390/fx_apogee`, commit
  `e4cd4524594ec02a5e205719eacdfa70d07b97cd`.

Le manifeste `strategies/xau_x10/MIGRATION.json` du dépôt destination enregistre
les empreintes de 241 fichiers importés, les neuf parquets copiés localement et
les deux sorties volumineuses non versionnées. Le PDF de 64 pages est identique
à celui du commit source. Aucun essai supplémentaire ni lecture de performance
hors échantillon n'a été effectué pendant la migration.

Le sous-projet possède son propre environnement et fichier de verrouillage :
pandas 3 / NumPy 2.4 ne sont pas mélangés aux dépendances de l'OPR Nasdaq.
Les seules adaptations des fichiers importés déclarent matplotlib/PyYAML pour
le rapport et lisent le commit de gel scientifique dans le manifeste plutôt
que dans l'historique Git du dépôt destination. Le moteur et les mesures sont
inchangés. Les archives v1 étaient auparavant présentes localement mais ignorées
par Git ; leur versionnement ferme cette dépendance implicite du rapport.

Validation destination : 494 tests X10 réussis, deux ignorés, 25 tables conformes,
aucun écart d'intégrité ; 403 tests et 37 sous-tests OPR préservés, Ruff propre.
Le verdict reste **NE PAS DÉPLOYER**. Les copies de ce dépôt restent historiques
pour préserver les liens de provenance, les dépendances partagées et les anciens
commits ; elles ne sont plus le point d'entrée de maintenance de X10.
