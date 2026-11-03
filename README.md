# NSGA-III-FA

Code, data and experiment scripts for the article

> **NSGA-III-FA: Guided Firefly Mutation with a Self-Adaptive Footprint for Energy- and Carbon-Aware Many-Objective Cloud Brokerage**
> Olfa Aissa, Olfa Harrabi Khlif, Ines Ben Jaafar

NSGA-III-FA keeps the selection of NSGA-III and replaces its polynomial mutation by a guided Firefly mutation: each offspring is attracted towards a better parent (dominating → same niche → first front), uses no extra function evaluation, and perturbs many variables while the population is spread out and one when it contracts (self-adaptive footprint driven by the normalized population dispersion).

## Contents

| Folder | Purpose | Article |
|---|---|---|
| `noyau/` | Shared library: operator (`operators_v2.py`), brokerage model (`cloud_v2.py`), campaign runner (`run_v2.py`), statistics and figures (`analyze_v2.py`) | Sections 3–5 |
| `instances/` | Full data of the six brokerage instances (capacities, number of servers, demands, prices, latencies, energy parameters) | Section 3.3 |
| `01_test_installation/` | Quick installation test | – |
| `02_mise_au_point_operateur/` | Component-wise ablation and footprint study | Tables 13, A1, A2 |
| `03_campagne_principale_courtage/` | Main brokerage campaign (4 instances, 9 methods, 30 runs) | Tables 9–11, 15; Figs. 4, 5, A1 |
| `04_instances_donnees_reelles_azure/` | Instances built from the Azure public VM trace | Tables 12, A4 |
| `05_generalisation_dtlz/` | DTLZ1–4 and DTLZ7, 3 to 10 objectives | Tables 16, A5, A6; Fig. A2 |
| `06_ablation_finale/` | Leave-one-out ablation (L1–L8) | Tables 14, A3 |
| `07_sensibilite_q_sigma/` | Threshold sensitivity on CloudBroker 50×5 | Fig. A3 |
| `08_comparaison_gkm/`, `12_comparaison_gkm_grandes_instances/` | Comparison with NSGA-III-GKM++ (harmonized formulation) | Tables 18, 19 |
| `09_figures_et_tableaux_article/` | Convergence curves, critical-difference diagrams and LaTeX tables | Figs. 2–5, A1, A2, A4 |
| `10_sensibilite_modele_energie/`, `13_consolidation_serveurs_recents/` | Sensitivity to the energy model (PUE, carbon-free shares, consolidation, 2024 servers) | Table 20 |
| `11_sensibilite_q_sigma_autres_instances/` | Independent validation of the threshold parameters | Fig. 6 |
| `14_generalisation_wfg/` | WFG1–2 and WFG4–9, 3 to 8 objectives | Tables 17, A7 |
| `resultats/` | Analysis outputs; raw results of every run are attached to the GitHub release | – |

A detailed description in French is given in `LISEZMOI.md`.

## Installation

Python 3.10 or later:

```
pip install numpy pandas scipy matplotlib scikit-learn pymoo==0.6.* numba
```

Then run `01_test_installation/test_rapide.bat` (Windows) or the equivalent command it contains. Each experiment folder has a `.bat` file that launches the corresponding campaign; already completed runs are skipped when a campaign is restarted.

## Reproducibility

- Seeds 42 to 71 (30 runs); for a given seed, all methods start from the same initial population.
- All methods use the same number of function evaluations (125,000 on brokerage); the measured number is stored in each `runXX_meta.json`.
- In the code, `A8_adaptfoot` is NSGA-III-FA, `A1_FC` is NSGA-III-FC, `A5_FA2` is FA-fixed, `NSGA3-PM` and `NSGA3-PMm` are NSGA-III-PM and NSGA-III-PM*.

## License

Released under the MIT License (see `LICENSE`).
