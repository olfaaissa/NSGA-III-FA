"""
common_v2.py -- Problemes, budgets et directions de reference partages par
toutes les experiences de NSGA-III-FA.

Choix experimentaux (etiquettes reprises dans le code) :

  [CHG-1] DTLZ en vrai many-objective : M = 3, 5, 8, 10.
  [CHG-2] Taille de population = nombre de directions Das-Dennis (+ arrondi
          au multiple de 4), comme Deb & Jain 2014.
  [CHG-3] Budget exprime en EVALUATIONS (FE_max), identique pour toutes les
          methodes.
  [CHG-4] Front de Pareto vrai pour DTLZ (IGD+/HV) ; l'union des fronts
          n'est utilisee que pour CloudBroker (front inconnu).
  [CHG-5] CloudBroker 10x3 : instance de reference (memes donnees et memes
          parametres que Samti et al. 2025). Le modele energetique P(u) et les
          autres tailles d'instances sont definis dans cloud_v2.py.
"""

import numpy as np
from pymoo.core.problem import Problem
from pymoo.core.repair import Repair
from pymoo.problems import get_problem
from pymoo.util.ref_dirs import get_reference_directions

SEED = 42

# ============================================================================
# [CHG-1] Benchmarks DTLZ many-objective
# ============================================================================
# Nombre de variables : n = M + k - 1, avec k = 5 (DTLZ1), k = 10 (DTLZ2-4),
# k = 20 (DTLZ7) -- valeurs standard de Deb et al. 2005 / Deb & Jain 2014.
DTLZ_K = {"dtlz1": 5, "dtlz2": 10, "dtlz3": 10, "dtlz4": 10, "dtlz7": 20}

# [CHG-3] Nombre de generations de reference (Deb & Jain 2014, Tableau III)
# -> FE_max = N * G_ref. DTLZ7 n'est pas dans leur tableau : 1000 (hypothese
# signalee comme telle dans l'article).
G_REF = {
    "dtlz1": {3: 400, 5: 600, 8: 750, 10: 1000},
    "dtlz2": {3: 250, 5: 350, 8: 500, 10: 750},
    "dtlz3": {3: 1000, 5: 1000, 8: 1000, 10: 1500},
    "dtlz4": {3: 600, 5: 1000, 8: 1250, 10: 2000},
    "dtlz7": {3: 1000, 5: 1000, 8: 1000, 10: 1000},
}

# [WFG-1] Benchmarks WFG (Huband et al. 2006) : k = 2(M-1)
# parametres de position, l = 20 parametres de distance (reglage usuel), et
# nombre de generations de Yuan et al. 2016 (theta-DEA) : 400 / 750 / 1500.
# WFG3 est exclu pour la meme raison que DTLZ5-6 (front non degenere en plus
# de la partie attendue, Ishibuchi et al. 2016).
WFG_L = 20
G_REF_WFG = {3: 400, 5: 750, 8: 1500}

# CloudBroker : budget IDENTIQUE pour toutes les methodes.
CLOUDBROKER_FE_MAX = 125_000


# ============================================================================
# [CHG-2] Directions de reference et taille de population
# ============================================================================
def get_ref_dirs(n_obj):
    """Das-Dennis a une couche pour M <= 5, a deux couches pour M >= 8
    (configuration de Deb & Jain 2014)."""
    if n_obj == 2:
        return get_reference_directions("das-dennis", 2, n_partitions=99)
    if n_obj == 3:
        return get_reference_directions("das-dennis", 3, n_partitions=12)   # 91
    if n_obj == 4:
        return get_reference_directions("das-dennis", 4, n_partitions=8)    # 165
    if n_obj == 5:
        return get_reference_directions("das-dennis", 5, n_partitions=6)    # 210
    if n_obj in (8, 10):
        outer = get_reference_directions("das-dennis", n_obj, n_partitions=3, scaling=1.0)
        inner = get_reference_directions("das-dennis", n_obj, n_partitions=2, scaling=0.5)
        return get_reference_directions("multi-layer", outer, inner)       # 156 / 275
    raise ValueError(f"M={n_obj} non prevu")


def pop_size_for(ref_dirs):
    n = len(ref_dirs)
    return int(np.ceil(n / 4.0) * 4)


# ============================================================================
# [CHG-5] CloudBroker 10x3 -- instance de reference (Samti et al. 2025)
# ============================================================================
CB_N, CB_M = 10, 3
T_j = np.array([0.2525, 0.1541, 0.19])[:CB_M]
C_j = np.array([0.1, 0.125, 0.143])[:CB_M]
E_j = np.array([190, 220, 240])[:CB_M]
A_j = np.array([5, 8, 12]).astype(float)
_rng_cb = np.random.default_rng(SEED)
P_i = _rng_cb.uniform(0.2, 0.5, size=CB_N)
R_i = _rng_cb.integers(1, 3, size=CB_N).astype(float)
L_ij = _rng_cb.uniform(0.05, 0.5, size=(CB_N, CB_M))
r_j = np.array([0.9, 0.5, 0.2])[:CB_M]


def repair_solution(X, demands, capacities, tol=1e-8, max_iter=200):
    """Projection iterative demande/capacite."""
    X = np.maximum(np.array(X, dtype=float), 0.0)
    n_clients, m_providers = X.shape
    for i in range(n_clients):
        if np.sum(X[i]) <= tol and demands[i] > 0:
            X[i] = np.ones(m_providers) * (demands[i] / float(m_providers))
    for _ in range(max_iter):
        row_sums = X.sum(axis=1)
        for i in range(n_clients):
            if row_sums[i] < demands[i] - tol:
                X[i] *= (demands[i] / (row_sums[i] + 1e-12)) if row_sums[i] > tol else 1.0
        col_sums = X.sum(axis=0)
        for j in range(m_providers):
            if col_sums[j] > capacities[j] + tol and col_sums[j] > 0:
                X[:, j] *= (capacities[j] / (col_sums[j] + 1e-12))
        row_sums = X.sum(axis=1)
        col_sums = X.sum(axis=0)
        if max(np.max(np.maximum(0.0, demands - row_sums)),
               np.max(np.maximum(0.0, col_sums - capacities))) < 1e-6:
            break
    return np.clip(X, 0.0, None)


def compute_metrics_for_solution(X):
    """Modele energetique lineaire de l'instance de reference : E = sum x_ij E_j."""
    X = np.array(X, dtype=float).reshape(CB_N, CB_M)
    RT = float(np.sum(X * (L_ij + T_j)))
    E = float(np.sum(X * E_j))
    Profit = float(np.sum(X * (P_i[:, None] - C_j[None, :])))
    E_non_green = float(np.sum(X * (1.0 - r_j)[None, :] * E_j))
    return RT, E, Profit, E_non_green


class CloudBrokerProblem(Problem):
    """Objectifs [RT, -Profit, E, E_ng]."""

    def __init__(self):
        self.n_clients, self.m_providers = CB_N, CB_M
        self.demands, self.capacities = R_i, A_j
        super().__init__(n_var=CB_N * CB_M, n_obj=4,
                         n_ieq_constr=CB_N + CB_M, xl=0.0, xu=1.0)

    def _evaluate(self, X, out, *args, **kwargs):
        n = X.shape[0]
        objs = np.zeros((n, 4))
        cons = np.zeros((n, self.n_clients + self.m_providers))
        for idx in range(n):
            alloc = repair_solution(X[idx].reshape(self.n_clients, self.m_providers),
                                    self.demands, self.capacities)
            RT, E, Profit, E_ng = compute_metrics_for_solution(alloc)
            objs[idx] = [RT, -Profit, E, E_ng]
            g1 = self.demands - alloc.sum(axis=1)
            g2 = alloc.sum(axis=0) - self.capacities
            cons[idx] = np.concatenate([g1, g2])
        cons[np.abs(cons) < 1e-2] = 0.0
        out["F"] = objs
        out["G"] = cons


class CloudBrokerUnconstrained(CloudBrokerProblem):
    """Meme probleme, sans declarer les contraintes : uniquement pour MOEA/D,
    que pymoo n'accepte pas avec contraintes. Les objectifs sont identiques,
    car _evaluate repare deja chaque allocation avant de calculer F."""

    def __init__(self):
        super().__init__()
        self.n_ieq_constr = 0

    def _evaluate(self, X, out, *args, **kwargs):
        super()._evaluate(X, out, *args, **kwargs)
        out.pop("G", None)


class CloudRepair(Repair):
    def _do(self, problem, X, **kwargs):
        for i in range(X.shape[0]):
            alloc = repair_solution(X[i].reshape(problem.n_clients, problem.m_providers),
                                    problem.demands, problem.capacities)
            X[i] = alloc.flatten()
        return np.clip(X, 0.0, 1.0)


# ============================================================================
# Fabrique d'instances
# ============================================================================
def make_problem(name, n_obj, unconstrained=False):
    """Retourne (problem, ref_dirs, pop_size, fe_max, repair)."""
    from cloud_v2 import parse_cb, ScaledCloudBroker, ScaledCloudRepair
    nm = parse_cb(name)
    if nm is not None:                              # instances cb_N.._M.. (modele P(u))
        problem = ScaledCloudBroker(nm[0], nm[1], unconstrained=unconstrained, kind=nm[2])
        ref_dirs = get_ref_dirs(4)
        return problem, ref_dirs, pop_size_for(ref_dirs), CLOUDBROKER_FE_MAX, ScaledCloudRepair()
    if name == "cloudbroker":
        problem = CloudBrokerUnconstrained() if unconstrained else CloudBrokerProblem()
        ref_dirs = get_ref_dirs(4)
        N = pop_size_for(ref_dirs)
        return problem, ref_dirs, N, CLOUDBROKER_FE_MAX, CloudRepair()
    if name.startswith("wfg"):                      # [WFG-1]
        k = 2 * (n_obj - 1)
        problem = get_problem(name, n_var=k + WFG_L, n_obj=n_obj, k=k)
        ref_dirs = get_ref_dirs(n_obj)
        N = pop_size_for(ref_dirs)
        return problem, ref_dirs, N, N * G_REF_WFG[n_obj], None
    k = DTLZ_K[name]
    problem = get_problem(name, n_var=n_obj + k - 1, n_obj=n_obj)
    ref_dirs = get_ref_dirs(n_obj)
    N = pop_size_for(ref_dirs)
    fe_max = N * G_REF[name][n_obj]
    return problem, ref_dirs, N, fe_max, None


def _dtlz7_front(n_obj, n_samples=40_000, chunk=4_000, seed=0):
    """Front vrai de DTLZ7 calcule analytiquement (Deb et al. 2005) :
    sur le front g = 1, f_i = x_i (i < M) et
    f_M = (1+g) * (M - sum_i f_i/(1+g) * (1 + sin(3*pi*f_i))).
    Necessaire car certaines versions de pymoo ne l'implementent pas.
    Le tri non domine est fait par paquets pour limiter la memoire
    (un tri sur 200 000 points provoquait un MemoryError sous Windows)."""
    from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
    nds = NonDominatedSorting()
    rng = np.random.default_rng(seed)
    X = rng.random((n_samples, n_obj - 1))
    h = n_obj - np.sum(X / 2.0 * (1 + np.sin(3 * np.pi * X)), axis=1)
    F = np.column_stack([X, 2.0 * h])
    keep = [F[k:k + chunk][nds.do(F[k:k + chunk], only_non_dominated_front=True)]
            for k in range(0, len(F), chunk)]
    F = np.vstack(keep)
    if len(F) > 8_000:                                   # sous-echantillon fixe
        F = F[rng.choice(len(F), 8_000, replace=False)]
    F = F[nds.do(F, only_non_dominated_front=True)]
    return F


def _wfg_front(name, n_obj):
    """[WFG-1] Front de reference des WFG.
    WFG4 a WFG9 ont le meme front concave : f_i = 2i * z_i avec ||z|| = 1
    (verifie : les points de l'ensemble de Pareto de WFG4 sont sur cette
    sphere a 1e-16 pres) -> calcul analytique sur des directions Das-Dennis.
    WFG1 et WFG2 : methode de pymoo (echantillonnage de l'ensemble de Pareto
    et point le plus proche de chaque direction), puis filtre non domine
    (WFG2 a un front deconnecte)."""
    from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
    parts = {3: 99, 5: 12, 8: 6}[n_obj]
    S = 2.0 * np.arange(1, n_obj + 1)
    if name in ("wfg4", "wfg5", "wfg6", "wfg7", "wfg8", "wfg9"):
        Z = get_reference_directions("das-dennis", n_obj, n_partitions=parts)
        return Z / np.linalg.norm(Z, axis=1, keepdims=True) * S
    parts = {3: 40, 5: 10, 8: 6}[n_obj]
    problem, *_ = make_problem(name, n_obj)
    np.random.seed(0)
    F = problem.pareto_front(get_reference_directions("das-dennis", n_obj, n_partitions=parts))
    return F[NonDominatedSorting().do(F, only_non_dominated_front=True)]


def true_pareto_front(name, n_obj):
    """[CHG-4] Front vrai dense pour IGD+ ; None pour CloudBroker."""
    if name == "cloudbroker" or name.startswith(("cb_", "az_")):
        return None
    if name == "dtlz7":
        return _dtlz7_front(n_obj)
    if name.startswith("wfg"):                      # [WFG-1]
        return _wfg_front(name, n_obj)
    problem, *_ = make_problem(name, n_obj)
    parts = {3: 99, 5: 12, 8: 6, 10: 5}[n_obj]
    dense = get_reference_directions("das-dennis", n_obj, n_partitions=parts)
    return problem.pareto_front(dense)
