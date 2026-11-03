"""
gkm_compare.py -- Comparaison NSGA-III-FA vs NSGA-III-GKM++ (Samti et al., 2025)
sur la formulation a 3 objectifs [RT, E, -Profit] de leur article.

Ce qui est repris EXACTEMENT du code public de NSGA-III-GKM++
(github.com/ahmedsam888/NSGA-III-GKM-Framework-for-Multi-Objective-Cloud-Resource-Brokerage-Optimization) :
  - donnees de l'instance N=10, M=3 (np.random.seed(42) puis L_ij, P_i, R_i dans cet ordre),
    T_j, C_j, E_j, A_j ;
  - objectifs RT = sum x (L+T), E = sum x E_j, P = sum x (P_i - C_j) ;
  - initialisation Genetic K-Means++ (fonction genetic_kmeans_plus_plus recopiee),
    pop 100, 91 directions Das-Dennis (12 partitions), SBX(0.9, 15), PM(prob 0.1, eta 15),
    budget 100 x 250 generations (~25 000 evaluations).

Ce qui differe du code public (et pourquoi) :
  - les contraintes sont celles du modele de l'article : sum_j x_ij = R_i et
    sum_i x_ij <= A_j. Dans le code public, la capacite est codee comme une borne
    INFERIEURE (A_j - sum_i x_ij <= 0) et la demande comme une borne superieure ;
    avec A_3 = 12 > N * x_max = 10, aucune solution n'est alors realisable
    (verifie : 0 individu realisable sur 2 executions). Les deux algorithmes
    utilisent donc la meme reparation (projection iterative) sur le modele de l'article.

Variante supplementaire :
  - "GKM++(completed)" : memes reglages que GKM++, mais les 4-5 centres GKM++
    sont COMPLETES par des solutions aleatoires uniformes jusqu'a 100 individus.
    pymoo 0.6 ne complete pas une population initiale trop petite (verifie :
    generation 1 = 5 individus). Cette variante teste l'hypothese selon
    laquelle l'ecart vient de la faible diversite initiale.

Usage : python comparaison_gkm.py --workers 6   (resultats dans results_gkm/)
        python comparaison_gkm.py --algos "GKM++(completed)"   (variante seule)
"""
# --- chemins du dossier organise : le noyau partage et le dossier resultats/ ---
import os as _os, sys as _sys
_ICI = _os.path.dirname(_os.path.abspath(__file__))
_RACINE = _os.path.dirname(_ICI)
_sys.path.insert(0, _os.path.join(_RACINE, "noyau"))
RESULTATS = _os.path.join(_RACINE, "resultats")
# ---------------------------------------------------------------------------
import argparse, json, os, random, time
from multiprocessing import Pool

import numpy as np
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.core.problem import Problem
from pymoo.core.repair import Repair
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

from cloud_v2 import repair_vectorized
from operators_v2 import ABLATIONS, FireflyMutationV2

HERE = RESULTATS

# ---------------- donnees exactes du code public GKM++ ----------------
N, M = 10, 3
T_j = np.array([0.2525, 0.1541, 0.19])
C_j = np.array([0.1, 0.125, 0.143])
E_j = np.array([190, 220, 240], dtype=float)
A_j = np.array([5, 8, 12], dtype=float)
_st = np.random.get_state()
np.random.seed(42)
L_ij = np.random.uniform(0.05, 0.5, size=(N, M))
P_i = np.random.uniform(0.2, 0.5, N)
R_i = np.random.randint(1, 3, size=N).astype(float)
np.random.set_state(_st)


class GKMBroker(Problem):
    """Formulation a 3 objectifs de Samti et al. (2025), contraintes du modele de l'article."""

    def __init__(self):
        super().__init__(n_var=N * M, n_obj=3, n_ieq_constr=N + M, xl=0.0, xu=1.0)

    def _evaluate(self, X, out, *args, **kwargs):
        F = np.zeros((len(X), 3)); G = np.zeros((len(X), N + M))
        for k, x in enumerate(X):
            a = repair_vectorized(x.reshape(N, M), R_i, A_j)
            RT = np.sum(a * (L_ij + T_j)); E = np.sum(a * E_j); P = np.sum(a * (P_i[:, None] - C_j[None, :]))
            F[k] = [RT, E, -P]
            G[k] = np.concatenate([R_i - a.sum(1), a.sum(0) - A_j])
        G[np.abs(G) < 1e-2] = 0.0
        out["F"], out["G"] = F, G


class GKMRepair(Repair):
    def _do(self, problem, X, **kwargs):
        for i in range(len(X)):
            X[i] = repair_vectorized(X[i].reshape(N, M), R_i, A_j).flatten()
        return np.clip(X, 0.0, 1.0)


def genetic_kmeans_plus_plus(pop_size_gkm=5, n_gen_gkm=10, a=2, b=1.2):
    """Recopie de la fonction du code public (memes appels aleatoires)."""
    from sklearn.cluster import KMeans
    population = np.random.rand(pop_size_gkm, N * M)
    best_centers, best_fit = None, float("-inf")
    for _ in range(n_gen_gkm):
        n_clusters = min(random.randint(2, pop_size_gkm), len(population))
        km = KMeans(n_clusters=n_clusters, init="k-means++", n_init=10, random_state=42).fit(population)
        c = km.cluster_centers_
        E = np.sum([np.linalg.norm(population[i] - c[km.labels_[i]]) ** 2 for i in range(len(population))])
        Gb = (np.sum([np.linalg.norm(c[i] - c[j]) ** 2 for i in range(len(c)) for j in range(i + 1, len(c))])
              / (2 * len(c) * (len(c) - 1))) if len(c) > 1 else 0
        fit = Gb / (b + a * E)
        if fit > best_fit:
            best_fit, best_centers = fit, c
    return best_centers


def run(job):
    algo, run_id, fe_max, out = job
    d = os.path.join(out, algo); os.makedirs(d, exist_ok=True)
    stem = os.path.join(d, f"run{run_id + 1:02d}")
    if os.path.exists(stem + "_meta.json"):
        return f"skip {algo} {run_id + 1}"
    seed = 42 + run_id
    np.random.seed(seed); random.seed(seed)
    ref = get_reference_directions("das-dennis", 3, n_partitions=12)
    prob = GKMBroker()
    if algo == "GKM++":
        init = genetic_kmeans_plus_plus()
        alg = NSGA3(pop_size=100, ref_dirs=ref, sampling=init, crossover=SBX(prob=0.9, eta=15),
                    mutation=PM(prob=0.1, eta=15), repair=GKMRepair())
    elif algo == "GKM++(completed)":
        init = genetic_kmeans_plus_plus()
        init = np.vstack([init, np.random.rand(100 - len(init), N * M)])
        alg = NSGA3(pop_size=100, ref_dirs=ref, sampling=init, crossover=SBX(prob=0.9, eta=15),
                    mutation=PM(prob=0.1, eta=15), repair=GKMRepair())
    elif algo == "NSGA-III-PM(GKM settings)":
        alg = NSGA3(pop_size=100, ref_dirs=ref, sampling=np.random.rand(100, N * M), crossover=SBX(prob=0.9, eta=15),
                    mutation=PM(prob=0.1, eta=15), repair=GKMRepair())
    elif algo == "NSGA-III-FA":
        alg = NSGA3(pop_size=100, ref_dirs=ref, sampling=np.random.rand(100, N * M), crossover=SBX(prob=1.0, eta=30),
                    mutation=FireflyMutationV2(seed=seed, **ABLATIONS["A8_adaptfoot"]), repair=GKMRepair())
    t = time.time()
    res = minimize(prob, alg, get_termination("n_eval", fe_max), seed=seed, copy_algorithm=False, verbose=False)
    F, CV = res.pop.get("F"), res.pop.get("CV").ravel()
    F = F[CV <= 1e-6] if (CV <= 1e-6).any() else F
    F = F[NonDominatedSorting().do(F, only_non_dominated_front=True)]
    np.savetxt(stem + "_F.csv", F, delimiter=",")
    json.dump(dict(algo=algo, run=run_id + 1, seed=seed, fe=int(res.algorithm.evaluator.n_eval),
                   cpu=round(time.time() - t, 2), feasible=int((CV <= 1e-6).sum())), open(stem + "_meta.json", "w"))
    return f"ok   {algo} run{run_id + 1} FE={res.algorithm.evaluator.n_eval}"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=30)
    ap.add_argument("--fe", type=int, default=25000)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--out", default=os.path.join(HERE, "results_gkm"))
    ap.add_argument("--algos", nargs="*",
                    default=["GKM++", "NSGA-III-PM(GKM settings)", "NSGA-III-FA", "GKM++(completed)"])
    a = ap.parse_args()
    jobs = [(al, r, a.fe, a.out) for al in a.algos for r in range(a.runs)]
    with Pool(a.workers) as p:
        for m in p.imap_unordered(run, jobs):
            print(m, flush=True)
