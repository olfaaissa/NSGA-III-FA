"""
cloud_v2.py -- Instances CloudBroker a plusieurs tailles et modele energetique
P(u) fonde sur des mesures SPECpower.

Noms d'instances : "cb_N<clients>_M<fournisseurs>", par ex. cb_N50_M5.
  - cb_N10_M3 reprend EXACTEMENT les donnees de l'instance de reference (Samti
    et al. 2025), seul le modele energetique change.
  - les autres tailles sont generees avec une graine fixe a partir des trois
    types de fournisseurs de l'instance de base (Small, Medium, XL).

Modele energetique (remplace le modele lineaire E = sum x_ij E_j) :
  u_j   = charge_j / A_j                 utilisation du fournisseur j (0..1)
  P_j   = S_j * P_spec(u_j)              puissance (W), S_j = A_j serveurs,
                                          P_spec interpolee lineairement entre
                                          les 11 mesures SPECpower (0 %..100 %)
  E_j   = PUE_j * P_j * dt / 1000        energie (kWh) sur l'epoque dt = 1 h
  E     = sum_j E_j ;  E_ng = sum_j (1 - r_j) * E_j
Hypothese : charge repartie uniformement sur les S_j serveurs du fournisseur
(pas de consolidation), serveurs allumes pendant l'epoque.

SOURCES (verifiees le 9 oct. 2026) :
  - Courbes SPECpower : valeurs identiques au code source de CloudSim
    (github.com/Cloudslab/cloudsim, package org.cloudbus.cloudsim.power.models),
    tirees des rapports SPECpower_ssj2008 :
      HP ML110 G4 : spec.org/power_ssj2008/results/res2011q1/power_ssj2008-20110127-00342.html
      HP ML110 G5 : spec.org/power_ssj2008/results/res2011q1/power_ssj2008-20110124-00339.html
      IBM x3550 M3 (2 x X5670) : spec.org/power_ssj2008/results/res2010q2/power_ssj2008-20100315-00239.html
    Citer aussi Beloglazov & Buyya, CCPE 24(13):1397-1420, 2012, doi:10.1002/cpe.1867.
  - PUE (sources primaires) : 1,09 = PUE TTM de la flotte Google (T2 2026,
    datacenters.google/efficiency) ; 1,47 = PUE moyen pondere par la capacite
    IT et 1,58 = PUE moyen par repondant, Uptime Institute, enquete 2023
    (journal.uptimeinstitute.com/large-data-centers-are-mostly-more-efficient-analysis-confirms/, 7 fev. 2024).
  - r_j : part d'energie decarbonee horaire (CFE%) 2025 publiee par Google Cloud
    (cloud.google.com/sustainability/region-carbon) : 0,90 southamerica-west1
    (Santiago), 0,57 us-east4 (Virginie du Nord), 0,20 asia-south1 (Mumbai).
    Attention : CFE inclut le nucleaire -> parler d'energie "non decarbonee"
    (non carbon-free) plutot que "non renouvelable".
"""

import os
import re

import numpy as np
import pandas as pd
from pymoo.core.problem import Problem
from pymoo.core.repair import Repair

# Puissance (W) a 0 %, 10 %, ..., 100 % d'utilisation -- verifiee (CloudSim / SPEC)
SPEC_POWER = {
    "HP_ML110_G4_Xeon3040": [86, 89.4, 92.6, 96, 99.5, 102, 106, 108, 112, 114, 117],
    "HP_ML110_G5_Xeon3075": [93.7, 97, 101, 105, 110, 116, 121, 125, 129, 133, 135],
    "IBM_x3550_XeonX5670":  [66, 107, 120, 131, 143, 156, 173, 191, 211, 229, 247],
}
SERVER_OF_TYPE = ["HP_ML110_G4_Xeon3040", "HP_ML110_G5_Xeon3075", "IBM_x3550_XeonX5670"]
PUE_OF_TYPE = [1.09, 1.47, 1.58]                  # sources : voir en-tete
EPOCH_H = 1.0

# Types de base (instance de reference) : Small, Medium, Extra-Large
BASE_T = np.array([0.2525, 0.1541, 0.19])
BASE_C = np.array([0.1, 0.125, 0.143])
BASE_A = np.array([5, 8, 12], dtype=float)
BASE_R = np.array([0.90, 0.57, 0.20])            # CFE% Google Cloud 2025 (voir en-tete)

_U = np.linspace(0.0, 1.0, 11)

# --------------------------------------------------------------------------
# [AZ-1] Instances pilotees par des donnees reelles : "az_N<clients>_M10"
#   - demande R_i (vCPU) : trace Azure Public Dataset V2 (2019), VM creees
#     pendant l'heure 215 (heure dont le nombre d'abonnements actifs est la
#     mediane des 719 heures completes) ; client = abonnement ; R_i = somme
#     des vCPU demandes (bucket ">24" compte 30). Echantillon aleatoire
#     (graine 2026) de N abonnements -> data/azure2019_h215_N<N>.csv
#     (genere par make_azure_instance.py).
#   - 10 sites = 10 regions Google Cloud : r_j = CFE% horaire 2025 publie
#     (cloud.google.com/sustainability/region-carbon) ; PUE_j = PUE TTM T2 2026
#     du campus Google de la region (datacenters.google/efficiency).
#   - type de serveur, T_j, C_j : les trois types de base, en alternance
#     (pas de bruit) ; capacite totale = 125 % de la demande, repartie
#     comme A = 5/8/12 ; x_ij dans [0, R_i].
#   - vCPU par serveur (fils materiels) : ML110 G4 et G5 = 2, x3550 M3 = 24.
#   - L_ij et P_i : memes lois que les autres instances (non observables
#     dans la trace).
AZ_SITES = [  # (region, campus, r_j CFE 2025, PUE TTM T2 2026)
    ("us-east1",           "Berkeley County SC", 0.31, 1.10),
    ("us-east4",           "Loudoun County VA",  0.57, 1.08),
    ("us-central1",        "Council Bluffs IA",  0.88, 1.11),
    ("us-west1",           "The Dalles OR",      0.83, 1.10),
    ("europe-west1",       "St. Ghislain BE",    0.79, 1.08),
    ("europe-north1",      "Hamina FI",          0.98, 1.09),
    ("asia-southeast1",    "Singapore",          0.05, 1.12),
    ("asia-east1",         "Changhua TW",        0.15, 1.13),
    ("asia-northeast1",    "Inzai JP",           0.23, 1.10),
    ("southamerica-west1", "Quilicura CL",       0.90, 1.08),
]
VCPU_PER_SERVER = {"HP_ML110_G4_Xeon3040": 2, "HP_ML110_G5_Xeon3075": 2, "IBM_x3550_XeonX5670": 24}
AZ_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")



def spec_power(server, u):
    return np.interp(np.clip(u, 0.0, 1.0), _U, SPEC_POWER[server])


# ---------------------------------------------------------------------------
def repair_vectorized(X, demands, capacities, tol=1e-8, max_iter=200):
    """Meme projection iterative que repair_solution (common_v2), vectorisee :
    les boucles sur clients et fournisseurs sont independantes, donc le
    resultat est identique, mais 50 a 200 fois plus rapide sur les grandes
    instances."""
    X = np.maximum(np.array(X, dtype=float), 0.0)
    m = X.shape[1]
    empty = (X.sum(axis=1) <= tol) & (demands > 0)
    X[empty] = (demands[empty] / float(m))[:, None]
    for _ in range(max_iter):
        rs = X.sum(axis=1)
        low = rs < demands - tol
        fac = np.where(low & (rs > tol), demands / (rs + 1e-12), 1.0)
        X *= fac[:, None]
        cs = X.sum(axis=0)
        over = (cs > capacities + tol) & (cs > 0)
        X *= np.where(over, capacities / (cs + 1e-12), 1.0)[None, :]
        rs, cs = X.sum(axis=1), X.sum(axis=0)
        if max(np.max(np.maximum(0.0, demands - rs)), np.max(np.maximum(0.0, cs - capacities))) < 1e-6:
            break
    return np.clip(X, 0.0, None)


class ScaledCloudBroker(Problem):
    """Objectifs [RT, -Profit, E (kWh), E_ng (kWh)], contraintes demande/capacite."""

    def __init__(self, N, M, seed=None, unconstrained=False, kind="cb"):
        self.n_clients, self.m_providers = N, M
        self.kind = kind
        self.units_per_server = np.ones(M)            # unites abstraites : 1 unite = 1 serveur
        xu = 1.0
        if kind == "az":                              # [AZ-1] donnees reelles
            if M != len(AZ_SITES):
                raise ValueError("az_ : M doit valoir 10")
            dem = pd.read_csv(os.path.join(AZ_DATA_DIR, f"azure2019_h215_N{N}.csv"))
            rng = np.random.default_rng(5000 + N)
            self.P = rng.uniform(0.2, 0.5, size=N)
            self.demands = dem["R"].to_numpy(float)
            self.L = rng.uniform(0.05, 0.5, size=(N, M))
            types = np.arange(M) % 3
            self.T, self.C = BASE_T[types].copy(), BASE_C[types].copy()
            self.r = np.array([s[2] for s in AZ_SITES])
            cap = BASE_A[types]
            self.capacities = cap / cap.sum() * 1.25 * self.demands.sum()
            self.servers = [SERVER_OF_TYPE[t] for t in types]
            self.pue = np.array([s[3] for s in AZ_SITES])
            self.units_per_server = np.array([VCPU_PER_SERVER[sv] for sv in self.servers], float)
            xu = np.repeat(self.demands, M)               # x_ij dans [0, R_i]
        elif N == 10 and M == 3:                       # donnees exactes de l'instance de reference
            rng = np.random.default_rng(42)
            self.P = rng.uniform(0.2, 0.5, size=N)
            self.demands = rng.integers(1, 3, size=N).astype(float)
            self.L = rng.uniform(0.05, 0.5, size=(N, M))
            types = np.arange(3)
            self.T, self.C, self.capacities, self.r = BASE_T.copy(), BASE_C.copy(), BASE_A.copy(), BASE_R.copy()
        else:
            rng = np.random.default_rng(1000 + 31 * N + M if seed is None else seed)
            self.P = rng.uniform(0.2, 0.5, size=N)
            self.demands = rng.integers(1, 3, size=N).astype(float)
            self.L = rng.uniform(0.05, 0.5, size=(N, M))
            types = np.arange(M) % 3
            jit = lambda: rng.uniform(0.9, 1.1, size=M)
            self.T = BASE_T[types] * jit()
            self.C = BASE_C[types] * jit()
            self.r = BASE_R[types]
            cap = BASE_A[types] * jit()
            self.capacities = cap / cap.sum() * 1.25 * self.demands.sum()   # 25 % de marge
        if kind != "az":
            self.servers = [SERVER_OF_TYPE[t] for t in types]
            self.pue = np.array([PUE_OF_TYPE[t] for t in types])
        n_con = 0 if unconstrained else N + M
        super().__init__(n_var=N * M, n_obj=4, n_ieq_constr=n_con, xl=0.0, xu=xu)
        self._unc = unconstrained

    def metrics(self, alloc):
        RT = float(np.sum(alloc * (self.L + self.T[None, :])))
        Profit = float(np.sum(alloc * (self.P[:, None] - self.C[None, :])))
        u = alloc.sum(axis=0) / self.capacities
        Ej = np.array([self.pue[j] * (self.capacities[j] / self.units_per_server[j]) * spec_power(self.servers[j], u[j])
                       for j in range(self.m_providers)]) * EPOCH_H / 1000.0
        return RT, Profit, float(Ej.sum()), float(np.sum((1.0 - self.r) * Ej))

    def _evaluate(self, X, out, *args, **kwargs):
        n = X.shape[0]
        F = np.zeros((n, 4))
        G = np.zeros((n, self.n_clients + self.m_providers))
        for k in range(n):
            a = repair_vectorized(X[k].reshape(self.n_clients, self.m_providers), self.demands, self.capacities)
            RT, Profit, E, Eng = self.metrics(a)
            F[k] = [RT, -Profit, E, Eng]
            G[k] = np.concatenate([self.demands - a.sum(axis=1), a.sum(axis=0) - self.capacities])
        G[np.abs(G) < 1e-2] = 0.0
        out["F"] = F
        if not self._unc:
            out["G"] = G


class ScaledCloudRepair(Repair):
    def _do(self, problem, X, **kwargs):
        for i in range(X.shape[0]):
            X[i] = repair_vectorized(X[i].reshape(problem.n_clients, problem.m_providers),
                                     problem.demands, problem.capacities).flatten()
        return np.clip(X, problem.xl, problem.xu)      # [AZ-1] bornes [0, R_i] (=[0,1] ailleurs)


def parse_cb(name):
    """cb_N50_M5 -> (50, 5, "cb") ; az_N100_M10 -> (100, 10, "az")."""
    m = re.fullmatch(r"(cb|az)_N(\d+)_M(\d+)", name)
    return (int(m.group(2)), int(m.group(3)), m.group(1)) if m else None
