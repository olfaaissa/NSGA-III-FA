# Code de l'article NSGA-III-FA, classé par objectif

Ce dossier contient **tout le code utilisé pour obtenir les résultats de la version finale** de l'article :

*NSGA-III-FA: Guided Firefly Mutation with a Self-Adaptive Footprint for Energy- and Carbon-Aware Many-Objective Cloud Brokerage*

Chaque sous-dossier correspond à **une question de l'article**. Pour lancer une étape, double-cliquez sur le fichier `.bat` du sous-dossier. Les résultats arrivent toujours dans `resultats\`.

## Installation (une seule fois)

```
pip install numpy pandas scipy matplotlib scikit-learn pymoo==0.6.* numba
```

Lancez ensuite `01_test_installation\test_rapide.bat`, qui dure quelques minutes. S'il affiche des lignes « ok », tout est prêt.

## Organisation

| Dossier | À quoi il sert | Lancer | Résultats (`resultats\…`) | Dans l'article | Durée |
|---|---|---|---|---|---|
| `noyau\` | **Bibliothèque partagée, à ne pas lancer directement.** Opérateur Firefly, modèle de courtage, lanceur de campagnes, analyse statistique | — | — | — | — |
| `01_test_installation\` | Vérifier que Python et les bibliothèques fonctionnent | `test_rapide.bat` | `test_rapide` | — | 2-5 min |
| `02_mise_au_point_operateur\` | (a) Ablation par composant A0-A7 et effet de l'empreinte ; (b) choix de la méthode finale (dispersion adaptative), testée aussi sur des instances jamais vues | `a_…bat`, puis `b_…bat` | `results_ablation`, `results_varprob`, `results_adaptive` | Tableaux 13, A1, A2 | 8-12 h + 10-14 h |
| `03_campagne_principale_courtage\` | **Campagne principale** : 4 instances CloudBroker (10×3 à 200×10), 9 méthodes, 30 runs, budget égal | `campagne_courtage_4_instances.bat` | `results_cloud` | Tableaux 9, 10, 11, 15 ; Fig. 4, 5, A1 | 5-8 h |
| `04_instances_donnees_reelles_azure\` | Instances dont la demande vient de la **trace réelle Azure 2019** et dont les sites sont 10 régions Google Cloud | `campagne_instances_azure.bat` | `results_azure` | Tableaux 12, A4 | 4-6 h |
| `05_generalisation_dtlz\` | Généralisation : DTLZ1-4 et DTLZ7 avec 3, 5, 8 et 10 objectifs | `campagne_dtlz_20_instances.bat` | `results_dtlz` | Tableaux 16, A5, A6 ; Fig. A2 | 1-2 jours |
| `06_ablation_finale\` | Ablation « **un mécanisme retiré à la fois** » (L1-L8) de la méthode finale | `ablation_un_mecanisme_retire.bat` | `results_ablation_final` | Tableaux 14, A3 | 6-7 h |
| `07_sensibilite_q_sigma\` | Sensibilité du seuil adaptatif (grille 5×5 de q et σ) | `sensibilite_q_sigma.bat` | `results_sens_qsigma` | Fig. A3 | 6 h |
| `08_comparaison_gkm\` | Comparaison avec **NSGA-III-GKM++** sur la formulation harmonisée (contraintes de Kumrai et al. 2017), avec la variante GKM++ à initialisation complétée à 100 solutions | `comparaison_gkm.bat` | `results_gkm` | Tableau 18 | 30-45 min |
| `10_sensibilite_modele_energie\` | Les conclusions changent-elles si l'on modifie les **PUE** et les **parts décarbonées** ? 4 scénarios, 2 instances, 3 méthodes | `sensibilite_energie.bat` | `results_sens_energie` | Tableau 20 (scénarios PUE et parts décarbonées) | 7-9 h |
| `11_sensibilite_q_sigma_autres_instances\` | Sensibilité de (q, σ) sur **3 instances jamais utilisées pour un réglage** (Azure 100×10, DTLZ2 m=8, DTLZ4 m=5) | `sensibilite_q_sigma_autres.bat`, puis `python figure_article.py` | `results_sens_qsigma_autres` | Fig. 6 | 6-8 h |
| `12_comparaison_gkm_grandes_instances\` | Comparaison avec **NSGA-III-GKM++** sur des instances plus grandes (50×5, 100×10, 200×10) | `comparaison_gkm_grandes.bat` | `results_gkm_grandes` | Tableau 19 | 2-3 h |
| `13_consolidation_serveurs_recents\` | Modèle énergétique avec **consolidation** (serveurs éteints) et **serveurs récents 2024** (SPECpower) | `consolidation_serveurs_recents.bat` | `results_consolidation_recents` | Tableau 20 (consolidation, serveurs 2024) | 3-5 h |
| `14_generalisation_wfg\` | Généralisation : **WFG1-2 et WFG4-9**, m = 3, 5, 8 objectifs | `generalisation_wfg.bat` | `results_wfg` | Tableaux 17, A7 | 10-15 h |
| `09_figures_et_tableaux_article\` | Figures de convergence et tableaux LaTeX de l'article, produits à partir des résultats | `figures_convergence.bat`, `tableaux_latex.bat`, `python diagrammes_cd.py` | `…\_figs_article`, `article_tableaux_figures\` | Fig. 2, 3, 4, 5, A1, A2, A4 ; tableaux | quelques minutes |

## Ordre conseillé

1. `01` (test)
2. `03`, puis `05`
3. `02`, `04` et `08`, dans n'importe quel ordre
4. `06` et `07`, qui **réutilisent** les runs de `03` et `05`
5. `10` et `11` (une nuit chacun), qui réutilisent les runs de `03`, `04` et `05`
6. `09`

Si un calcul s'arrête, relancez le même fichier `.bat` : les runs déjà faits sont sautés.

**Réutiliser les résultats déjà calculés.** Pour ne pas tout recalculer, téléchargez les archives de résultats bruts attachées à la release GitHub et décompressez-les dans `resultats\`. Le dossier `resultats\results_gkm_25k.zip` contient déjà les résultats bruts de la comparaison avec GKM++.

## Contenu du noyau

| Fichier | Rôle |
|---|---|
| `operators_v2.py` | La mutation Firefly guidée, avec repli dominant → même niche → premier front, empreinte auto-adaptative et décroissance du pas, ainsi que toutes les variantes d'ablation (A0-A13, L1-L8) |
| `cloud_v2.py` | Le modèle de courtage : 4 objectifs, énergie SPECpower + PUE + part décarbonée, contraintes Σ x ≤ A_j et Σ x ≥ R_i, réparation, instances `cb_…` (générées) et `az_…` (trace Azure) |
| `common_v2.py` | Problèmes, populations, directions de référence, budgets |
| `run_v2.py` | Lanceur de campagnes : préréglages `ablation`, `varprob`, `adaptive`, `adaptive2`, `cloud`, `azure`, `dtlz_gen`. Même population initiale par graine, même budget pour toutes les méthodes |
| `analyze_v2.py` | HV, IGD+, Wilcoxon-Holm, Friedman, diagrammes CD, courbes de convergence, activation |
| `data\azure2019_h215_N100.csv`, `_N200.csv` | Demandes des clients tirées de la trace Azure (refaites par `04\construire_instances_azure.py`) |

## Noms du code et noms de l'article

| Code | Article |
|---|---|
| `A8_adaptfoot` | **NSGA-III-FA** (méthode proposée) |
| `A1_FC` | NSGA-III-FC (Firefly classique) |
| `A5_FA2` | FA-fixed (empreinte fixe 1/n) |
| `NSGA3-PM` | NSGA-III-PM |
| `NSGA3-PMm` | NSGA-III-PM* |
| `RVEA`, `MOEAD`, `AGEMOEA2`, `MOPSO` | RVEA, MOEA/D, AGE-MOEA-II, MOPSO-CD |
| `L1` … `L8` | variantes de l'ablation finale (Tableaux 14 et A3) |
| `cb_N50_M5` | CloudBroker 50×5 |
| `az_N100_M10` | Azure-trace 100×10 |

## Reproductibilité

- **Graines** : 42 à 71, donc 30 runs. Pour une graine donnée, toutes les méthodes partent de la même population (tests appariés).
- **Budget** : 125 000 évaluations sur le courtage, pour toutes les méthodes. Le nombre réellement effectué est enregistré dans chaque `runXX_meta.json`.
- **Traçabilité** : chaque choix de conception est repéré dans le code par une étiquette (`[OP-x]`, `[RUN-x]`, `[AN-x]`, `[AZ-1]`).
- **Notation** : dans l'article et dans les figures, *m* désigne le nombre d'objectifs et *M* le nombre de fournisseurs (par exemple `dtlz2_M8` = DTLZ2 avec m = 8 objectifs ; `cb_N50_M5` = 50 clients et M = 5 fournisseurs).
- **Données des instances** : le dossier `instances\` contient, pour les six instances de courtage, les capacités $A_j$, le nombre de serveurs $S_j=A_j/\kappa_j$ (nombre réel), les demandes $R_i$, les prix, les latences et les paramètres d'énergie. Il est produit par `python noyau\exporter_instances.py`.
- **Bornes et réparation** : les variables codées sont dans [0, 1] (instances générées) ou [0, R_i] (instances Azure) ; chaque enfant est réparé (`repair_vectorized`, même réparation pour toutes les méthodes) et les objectifs sont calculés sur l'allocation réparée, qui respecte la demande, la capacité et x_ij ≤ R_i.
