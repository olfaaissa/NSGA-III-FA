"""
run_v2.py -- Lance les campagnes d'experiences, a budget d'evaluations egal.

Exemples (depuis le dossier noyau) :

    # Essai preliminaire (environ 1 a 3 h selon la machine)
    python run_v2.py --preset pilot --workers 6

    # Ablation seule sur CloudBroker
    python run_v2.py --problems cloudbroker --algos A0_randomwalk A1_FC A2_decay A3_switch A4_rawdisp A5_FA2 A6_nofallback --runs 30

    # Campagne complete (long : a lancer sur la nuit / plusieurs jours)
    python run_v2.py --preset full --workers 8

Le script reprend la ou il s'est arrete : un run deja termine (fichier
_meta.json present) est saute. Vous pouvez donc l'interrompre et le relancer.

Protocole (etiquettes reprises dans le code) :

  [RUN-1] Arret par termination("n_eval", FE_max) pour TOUS les algorithmes.
  [RUN-2] n_eval mesure enregistre pour chaque run dans _meta.json (colonne
          "mesure" du tableau des FE). FireflyMutationV2 ne fait aucun appel
          cache au probleme.
  [RUN-3] NSGA-III-PM au reglage standard : prob = 1 par individu,
          prob_var = 1/n, eta = 20. Ajout de PM-matched (prob = 0,7,
          prob_var = 1, c'est-a-dire la meme empreinte que Firefly) : controle
          de l'effet de la probabilite de mutation.
  [RUN-4] Baselines : RVEA, MOEA/D-PBI, AGE-MOEA-II (si numba est installe),
          MOPSO-CD.
  [RUN-5] SBX : prob = 1,0 et eta = 30, valeurs de Deb & Jain 2014 pour
          NSGA-III. Modifiable avec --sbx_prob / --sbx_eta.
  [RUN-6] Historique de convergence : le front non domine est sauvegarde
          tous les 5 % du budget (courbes HV/IGD+ en fonction des FE).
  [RUN-7] Journal d'activation de l'operateur sauvegarde par run.
  [RUN-8] copy_algorithm=False : on garde l'etat de l'operateur et son
          journal.
  [RUN-9] Execution parallele (un processus par run) et reprise apres
          interruption.
"""

import argparse
import json
import os
import time
import warnings
from multiprocessing import Pool

import numpy as np
import pandas as pd
from pymoo.core.callback import Callback
from pymoo.core.sampling import Sampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

from common_v2 import SEED, make_problem
from operators_v2 import ABLATIONS, FireflyMutationV2

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
OUT_ROOT = os.path.join(HERE, "results_v2")

PRESETS = {
    # Essai preliminaire : 5 runs, 2 problemes x 2 M + CloudBroker, 4 methodes cles
    "pilot": dict(
        instances=[("dtlz2", 5), ("dtlz2", 8), ("dtlz4", 5), ("dtlz4", 8), ("cloudbroker", 4)],
        algos=["NSGA3-PM", "NSGA3-PMm", "A0_randomwalk", "A5_FA2", "A7_fullvector", "RVEA"], runs=5),
    # Campagne complete
    "full": dict(
        instances=[(p, m) for p in ["dtlz1", "dtlz2", "dtlz3", "dtlz4", "dtlz7"] for m in [3, 5, 8, 10]]
        + [("cloudbroker", 4)],
        algos=["NSGA3-PM", "NSGA3-PMm", "A1_FC", "A5_FA2", "RVEA", "MOEAD", "AGEMOEA2", "MOPSO"],
        runs=30),
    # Sensibilite a var_prob (1/n, 0.1, 0.3, 0.5, 1.0)
    "varprob": dict(
        instances=[("dtlz2", 5), ("dtlz4", 8), ("dtlz7", 5), ("cloudbroker", 4)],
        algos=["A5_FA2", "S_vp0.1", "S_vp0.3", "S_vp0.5", "A7_fullvector"], runs=30),
    # Test de la dispersion adaptative [OP-9] : 4 instances de l'ablation
    # + 2 instances jamais utilisees pour le reglage (dtlz1 M8, dtlz3 M5)
    "adaptive": dict(
        instances=[("dtlz2", 5), ("dtlz4", 8), ("dtlz7", 5), ("cloudbroker", 4), ("dtlz1", 8), ("dtlz3", 5)],
        algos=["A5_FA2", "S_vp0.3", "A8_adaptfoot", "A9_adaptfoot_nodecay", "A1_FC", "NSGA3-PM"], runs=30),
    # 2e test : beta0 = 1 [OP-11] (le plancher [OP-10] n'aidait pas). Meme dossier que "adaptive" :
    # les runs deja faits sont reutilises. dtlz1 M5 et dtlz3 M8 = jamais vus.
    "adaptive2": dict(
        instances=[("dtlz2", 5), ("dtlz4", 8), ("dtlz7", 5), ("cloudbroker", 4), ("dtlz1", 8), ("dtlz3", 5),
                   ("dtlz1", 5), ("dtlz3", 8)],
        algos=["A5_FA2", "A8_adaptfoot", "A12_beta1", "A13_beta1_nodecay", "A1_FC", "NSGA3-PM"], runs=30),
    # CAMPAGNE PRINCIPALE : courtage cloud, modele energetique P(u),
    # 4 tailles d'instances, methode proposee = A8_adaptfoot
    "cloud": dict(
        instances=[("cb_N10_M3", 4), ("cb_N50_M5", 4), ("cb_N100_M10", 4), ("cb_N200_M10", 4)],
        algos=["A8_adaptfoot", "A5_FA2", "A1_FC", "NSGA3-PM", "NSGA3-PMm", "RVEA", "MOEAD", "AGEMOEA2", "MOPSO"],
        runs=30),
    # [AZ-1] Instances pilotees par la trace Azure 2019 + 10 regions Google Cloud
    "azure": dict(
        instances=[("az_N100_M10", 4), ("az_N200_M10", 4)],
        algos=["A8_adaptfoot", "A5_FA2", "A1_FC", "NSGA3-PM", "NSGA3-PMm", "RVEA", "MOEAD", "AGEMOEA2", "MOPSO"],
        runs=30),
    # Etude de generalisation DTLZ (M = 3, 5, 8, 10)
    "dtlz_gen": dict(
        instances=[(p, m) for p in ["dtlz1", "dtlz2", "dtlz3", "dtlz4", "dtlz7"] for m in [3, 5, 8, 10]],
        algos=["A8_adaptfoot", "A1_FC", "NSGA3-PM", "RVEA", "MOEAD", "AGEMOEA2"], runs=30),
    # [WFG-1] Generalisation WFG : WFG1-2 et WFG4-9, M = 3, 5, 8
    "wfg_gen": dict(
        instances=[(f"wfg{i}", m) for i in [1, 2, 4, 5, 6, 7, 8, 9] for m in [3, 5, 8]],
        algos=["A8_adaptfoot", "A1_FC", "NSGA3-PM", "RVEA", "MOEAD", "AGEMOEA2"], runs=30),
    # Ablation par composants (A0-A7)
    "ablation": dict(
        instances=[("dtlz2", 5), ("dtlz4", 8), ("dtlz7", 5), ("cloudbroker", 4)],
        algos=[k for k in ABLATIONS if k[:2] in ("A0", "A1", "A2", "A3", "A4", "A5", "A6", "A7")], runs=30),
}


# ---------------------------------------------------------------------------
class FixedSampling(Sampling):
    """MOPSO_CD exige un objet Sampling (pas un tableau)."""

    def __init__(self, X):
        super().__init__()
        self.X = X

    def _do(self, problem, n_samples, **kwargs):
        return self.X


class SnapshotCallback(Callback):
    """[RUN-6] Sauvegarde le front non domine (faisable) tous les 5 % de FE."""

    def __init__(self, fe_max, step=0.05):
        super().__init__()
        self.checkpoints = list(np.round(np.arange(step, 1.0 + 1e-9, step) * fe_max).astype(int))
        self.snaps = {}

    def notify(self, algorithm):
        n_eval = algorithm.evaluator.n_eval
        while self.checkpoints and n_eval >= self.checkpoints[0]:
            cp = self.checkpoints.pop(0)
            self.snaps[f"fe_{cp}"] = nondominated_feasible(algorithm.pop)


def nondominated_feasible(pop):
    F = pop.get("F")
    CV = pop.get("CV")
    if CV is not None:
        feas = (np.asarray(CV, float).reshape(-1) <= 1e-6)
        if feas.any():
            F = F[feas]
    idx = NonDominatedSorting().do(F, only_non_dominated_front=True)
    return F[idx]


def build_algorithm(name, ref_dirs, N, X_init, repair, seed, sbx_prob, sbx_eta, var_prob=None):
    from pymoo.algorithms.moo.nsga3 import NSGA3
    common = dict(sampling=X_init, crossover=SBX(prob=sbx_prob, eta=sbx_eta))
    if repair is not None:
        common["repair"] = repair
    pm_std = PM(prob=1.0, eta=20)                       # [RUN-3] prob_var = 1/n
    if name == "NSGA3-PM":
        return NSGA3(ref_dirs=ref_dirs, pop_size=N, mutation=pm_std, **common)
    if name == "NSGA3-PMm":
        return NSGA3(ref_dirs=ref_dirs, pop_size=N,
                     mutation=PM(prob=0.7, prob_var=1.0, eta=20), **common)
    if name in ABLATIONS:
        cfg = dict(ABLATIONS[name])
        if var_prob is not None and "var_prob" not in cfg:   # --var_prob (A7/S_vp gardent le leur)
            cfg["var_prob"] = var_prob
        return NSGA3(ref_dirs=ref_dirs, pop_size=N,
                     mutation=FireflyMutationV2(seed=seed, **cfg), **common)
    if name == "RVEA":
        from pymoo.algorithms.moo.rvea import RVEA
        return RVEA(ref_dirs=ref_dirs, pop_size=N, mutation=pm_std, **common)
    if name == "MOEAD":
        from pymoo.algorithms.moo.moead import MOEAD
        from pymoo.decomposition.pbi import PBI
        common.pop("repair", None)
        common["sampling"] = X_init[:len(ref_dirs)]      # MOEA/D : 1 individu par direction
        alg = MOEAD(ref_dirs=ref_dirs, n_neighbors=20, decomposition=PBI(),
                    prob_neighbor_mating=0.9, mutation=pm_std, **common)
        if repair is not None:
            alg.repair = repair
        return alg
    if name == "AGEMOEA2":
        from pymoo.algorithms.moo.age2 import AGEMOEA2   # necessite numba
        return AGEMOEA2(pop_size=N, mutation=pm_std, **common)
    if name == "MOPSO":
        from pymoo.algorithms.moo.mopso_cd import MOPSO_CD
        kw = dict(sampling=FixedSampling(X_init))
        if repair is not None:
            kw["repair"] = repair
        return MOPSO_CD(pop_size=N, w=0.6, c1=2.0, c2=2.0, max_velocity_rate=0.5,
                        archive_size=N, **kw)
    raise ValueError(name)


def run_one(job):
    try:
        return _run_one(job)
    except Exception as e:                                # un echec n'arrete pas la campagne
        return f"ERREUR {job[:4]}: {type(e).__name__}: {e}"


def _run_one(job):
    pname, m, algo, run, sbx_prob, sbx_eta, fe_scale, out_root, var_prob = job
    out_dir = os.path.join(out_root, f"{pname}_M{m}", algo)
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.join(out_dir, f"run{run + 1:02d}")
    if os.path.exists(stem + "_meta.json"):
        return f"skip {pname} M{m} {algo} run{run + 1}"

    problem, ref_dirs, N, fe_max, repair = make_problem(pname, m, unconstrained=(algo == "MOEAD"))
    fe_max = int(fe_max * fe_scale)          # fe_scale < 1 : test rapide uniquement
    seed = SEED + run
    # Population initiale partagee par toutes les methodes (appariement
    # Wilcoxon), dans les bornes reelles du probleme.
    rs = np.random.RandomState(seed)
    X_init = problem.xl + rs.rand(N, problem.n_var) * (problem.xu - problem.xl)

    try:
        algorithm = build_algorithm(algo, ref_dirs, N, X_init, repair, seed, sbx_prob, sbx_eta, var_prob)
    except Exception as e:                               # ex. numba absent
        return f"SKIP {algo}: {e}"

    cb = SnapshotCallback(fe_max)
    t0 = time.time()
    res = minimize(problem, algorithm, get_termination("n_eval", fe_max),   # [RUN-1]
                   seed=seed, callback=cb, copy_algorithm=False, verbose=False)  # [RUN-8]
    cpu = time.time() - t0

    F_final = nondominated_feasible(res.pop) if res.pop is not None else res.F
    np.savetxt(stem + "_F.csv", F_final, delimiter=",")
    np.savez_compressed(stem + "_hist.npz", **cb.snaps)
    mut = getattr(getattr(algorithm, "mating", None), "mutation", None)
    if isinstance(mut, FireflyMutationV2) and mut.log:
        pd.DataFrame(mut.log).to_csv(stem + "_activation.csv", index=False)   # [RUN-7]
    meta = dict(problem=pname, n_obj=m, algo=algo, run=run + 1, seed=seed,
                pop_size=N, fe_max=fe_max, n_eval_measured=int(res.algorithm.evaluator.n_eval),
                n_gen=int(res.algorithm.n_gen), cpu_seconds=round(cpu, 2),
                n_front=int(len(F_final)), sbx=[sbx_prob, sbx_eta], var_prob=var_prob)
    with open(stem + "_meta.json", "w") as f:                                # [RUN-2]
        json.dump(meta, f, indent=1)
    return f"ok   {pname} M{m} {algo} run{run + 1}  FE={meta['n_eval_measured']}  {cpu:.0f}s"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", choices=list(PRESETS.keys()))
    ap.add_argument("--problems", nargs="*", help="ex. dtlz2 cloudbroker")
    ap.add_argument("--n_obj", nargs="*", type=int, default=[5, 8])
    ap.add_argument("--algos", nargs="*")
    ap.add_argument("--runs", type=int, default=None)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--sbx_prob", type=float, default=1.0)
    ap.add_argument("--sbx_eta", type=float, default=30.0)
    ap.add_argument("--var_prob", type=float, default=None,
                    help="part des variables perturbees par Firefly (defaut 1/n) ; ex. 0.3")
    ap.add_argument("--fe_scale", type=float, default=1.0,
                    help="test rapide : ex. 0.02 = 2%% du budget (NE PAS utiliser pour les resultats)")
    ap.add_argument("--out", default=OUT_ROOT)
    a = ap.parse_args()

    if a.preset:
        cfg = PRESETS[a.preset]
        instances, algos, runs = cfg["instances"], cfg["algos"], cfg["runs"]
    else:
        is_cb = lambda p: p == "cloudbroker" or p.startswith(("cb_", "az_"))
        instances = [(p, 4) if is_cb(p) else (p, m)
                     for p in (a.problems or ["dtlz2"]) for m in ([4] if is_cb(p) else a.n_obj)]
        instances = list(dict.fromkeys(instances))
        algos, runs = a.algos or ["NSGA3-PM", "A5_FA2"], 30
    if a.algos:
        algos = a.algos
    if a.runs:
        runs = a.runs

    jobs = [(p, m, al, r, a.sbx_prob, a.sbx_eta, a.fe_scale, a.out, a.var_prob) for (p, m) in instances for al in algos for r in range(runs)]
    print(f"{len(jobs)} runs -> {a.out}  ({a.workers} processus)")
    if a.workers <= 1:
        for j in jobs:
            print(run_one(j), flush=True)
    else:
        with Pool(a.workers) as pool:
            for msg in pool.imap_unordered(run_one, jobs):
                print(msg, flush=True)


if __name__ == "__main__":   # obligatoire sous Windows pour multiprocessing
    main()
