"""
comparaison_gkm_grandes.py -- Comparaison avec NSGA-III-GKM++ sur des instances
plus grandes que l'instance 10x3 de son code public (suite de l'etape 08).

But : ne plus limiter la comparaison avec GKM++ a une seule petite instance.

Formulation harmonisee (la meme qu'a l'etape 08) :
  - 3 objectifs de NSGA-III-GKM++ : RT = sum x (L+T), E = sum x E_j (coefficient
    lineaire par type de fournisseur : 190, 220, 240, comme dans son code public),
    -Profit = -sum x (P_i - C_j) ;
  - contraintes de Kumrai et al. (2017) : sum_j x_ij >= R_i, sum_i x_ij <= A_j ;
  - meme reparation pour toutes les methodes.
Instances : clients et fournisseurs des instances CloudBroker 50x5, 100x10 et
200x10 de l'article (memes graines que cloud_v2.py), types de fournisseurs
Small / Medium / XL en alternance.

Methodes (reglages de l'etape 08) :
  GKM++                      : initialisation GKM++ (4-5 centres), SBX 0.9/15, PM 0.1/15
  GKM++(completed)           : idem, centres completes a 100 solutions aleatoires
  NSGA-III-PM(GKM settings)  : reglages de GKM++, initialisation aleatoire
  NSGA-III-FA                : operateur propose (reglages du Tableau 8)
Population 100, 91 directions de reference, 125 000 evaluations (budget des
instances de courtage de l'article), 30 runs apparies (graines 42 a 71).

Usage :
  python comparaison_gkm_grandes.py --workers 7
  python comparaison_gkm_grandes.py --analyze
"""
# --- chemins du dossier organise : le noyau partage et le dossier resultats/ ---
import os as _os, sys as _sys
_ICI = _os.path.dirname(_os.path.abspath(__file__))
_RACINE = _os.path.dirname(_ICI)
_sys.path.insert(0, _os.path.join(_RACINE, "noyau"))
_sys.path.insert(0, _os.path.join(_RACINE, "08_comparaison_gkm"))
RESULTATS = _os.path.join(_RACINE, "resultats")
# ---------------------------------------------------------------------------
import argparse, glob, json, os, random, time, warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
OUT = os.path.join(RESULTATS, "results_gkm_grandes")
INSTANCES = [(50, 5), (100, 10), (200, 10)]
ALGOS = ["NSGA-III-FA", "GKM++", "GKM++(completed)", "NSGA-III-PM(GKM settings)"]
REF = "NSGA-III-FA"
E_TYPE = np.array([190.0, 220.0, 240.0])      # coefficients lineaires du code public GKM++


def make_instance(N, M):
    from cloud_v2 import ScaledCloudBroker, SERVER_OF_TYPE
    cb = ScaledCloudBroker(N, M)
    types = np.array([SERVER_OF_TYPE.index(s) for s in cb.servers])
    return dict(N=N, M=M, L=cb.L, T=cb.T, C=cb.C, P=cb.P, R=cb.demands, A=cb.capacities, E=E_TYPE[types])


def build_problem(d):
    from pymoo.core.problem import Problem
    from pymoo.core.repair import Repair
    from cloud_v2 import repair_vectorized
    N, M = d["N"], d["M"]

    class GKMBrokerLarge(Problem):
        def __init__(self):
            super().__init__(n_var=N * M, n_obj=3, n_ieq_constr=N + M, xl=0.0, xu=1.0)

        def _evaluate(self, X, out, *args, **kwargs):
            F = np.zeros((len(X), 3)); G = np.zeros((len(X), N + M))
            for k, x in enumerate(X):
                a = repair_vectorized(x.reshape(N, M), d["R"], d["A"])
                F[k] = [np.sum(a * (d["L"] + d["T"])), np.sum(a * d["E"]),
                        -np.sum(a * (d["P"][:, None] - d["C"][None, :]))]
                G[k] = np.concatenate([d["R"] - a.sum(1), a.sum(0) - d["A"]])
            G[np.abs(G) < 1e-2] = 0.0
            out["F"], out["G"] = F, G

    class RepairLarge(Repair):
        def _do(self, problem, X, **kwargs):
            for i in range(len(X)):
                X[i] = repair_vectorized(X[i].reshape(N, M), d["R"], d["A"]).flatten()
            return np.clip(X, 0.0, 1.0)

    return GKMBrokerLarge(), RepairLarge()


def genetic_kmeans_plus_plus(n_var, pop_size_gkm=5, n_gen_gkm=10, a=2, b=1.2):
    """Fonction du code public GKM++ (memes appels aleatoires), dimension n_var."""
    from sklearn.cluster import KMeans
    population = np.random.rand(pop_size_gkm, n_var)
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
    try:
        return _run(job)
    except Exception as e:
        return f"ERREUR {job[:3]}: {type(e).__name__}: {e}"


def _run(job):
    from pymoo.algorithms.moo.nsga3 import NSGA3
    from pymoo.operators.crossover.sbx import SBX
    from pymoo.operators.mutation.pm import PM
    from pymoo.optimize import minimize
    from pymoo.termination import get_termination
    from pymoo.util.ref_dirs import get_reference_directions
    from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
    from operators_v2 import ABLATIONS, FireflyMutationV2

    (N, M), algo, run_id, fe_max, out = job
    d_out = os.path.join(out, f"gkm_N{N}_M{M}", algo); os.makedirs(d_out, exist_ok=True)
    stem = os.path.join(d_out, f"run{run_id + 1:02d}")
    if os.path.exists(stem + "_meta.json"):
        return f"skip {N}x{M} {algo} {run_id + 1}"
    seed = 42 + run_id
    np.random.seed(seed); random.seed(seed)
    d = make_instance(N, M)
    prob, rep = build_problem(d)
    n = N * M
    ref = get_reference_directions("das-dennis", 3, n_partitions=12)
    if algo in ("GKM++", "GKM++(completed)"):
        init = genetic_kmeans_plus_plus(n)
        if algo == "GKM++(completed)":
            init = np.vstack([init, np.random.rand(100 - len(init), n)])
        alg = NSGA3(pop_size=100, ref_dirs=ref, sampling=init, crossover=SBX(prob=0.9, eta=15),
                    mutation=PM(prob=0.1, eta=15), repair=rep)
    elif algo == "NSGA-III-PM(GKM settings)":
        alg = NSGA3(pop_size=100, ref_dirs=ref, sampling=np.random.rand(100, n), crossover=SBX(prob=0.9, eta=15),
                    mutation=PM(prob=0.1, eta=15), repair=rep)
    elif algo == "NSGA-III-FA":
        alg = NSGA3(pop_size=100, ref_dirs=ref, sampling=np.random.rand(100, n), crossover=SBX(prob=1.0, eta=30),
                    mutation=FireflyMutationV2(seed=seed, **ABLATIONS["A8_adaptfoot"]), repair=rep)
    else:
        raise ValueError(algo)
    t = time.time()
    res = minimize(prob, alg, get_termination("n_eval", fe_max), seed=seed, copy_algorithm=False, verbose=False)
    F, CV = res.pop.get("F"), res.pop.get("CV").ravel()
    F = F[CV <= 1e-6] if (CV <= 1e-6).any() else F
    F = F[NonDominatedSorting().do(F, only_non_dominated_front=True)]
    np.savetxt(stem + "_F.csv", F, delimiter=",")
    json.dump(dict(instance=f"{N}x{M}", algo=algo, run=run_id + 1, seed=seed, fe=int(res.algorithm.evaluator.n_eval),
                   cpu=round(time.time() - t, 2), feasible=int((CV <= 1e-6).sum())), open(stem + "_meta.json", "w"))
    return f"ok   {N}x{M} {algo} run{run_id + 1} FE={res.algorithm.evaluator.n_eval} {time.time() - t:.0f}s"


def run_all(workers, runs, fe, out):
    from multiprocessing import Pool
    os.makedirs(out, exist_ok=True)
    jobs = [(nm, a, r, fe, out) for r in range(runs) for nm in INSTANCES for a in ALGOS]
    print(f"{len(jobs)} jobs (deja faits = sautes) -> {out}  ({workers} processus)")
    with Pool(workers) as p:
        for k, m in enumerate(p.imap_unordered(run, jobs), 1):
            print(f"[{k}/{len(jobs)}] {m}", flush=True)


def analyze(out, alpha=0.05):
    from scipy import stats
    from analyze_v2 import Indicators, holm
    from analyse_gkm import load
    adir = os.path.join(out, "_analysis"); os.makedirs(adir, exist_ok=True)
    rows, summ, txt = [], [], []
    for N, M in INSTANCES:
        root = os.path.join(out, f"gkm_N{N}_M{M}")
        if not os.path.isdir(root):
            continue
        data = load(root)
        common = set.intersection(*[set(v) for v in data.values()])
        data = {a: {r: v[r] for r in common} for a, v in data.items()}
        ind = Indicators("cloudbroker", 3, data, hv_samples=0)       # union des fronts
        for a, runs in data.items():
            for r, x in runs.items():
                rows.append(dict(instance=f"{N}x{M}", algo=a, run=r, HV=ind.hv(x["F"]), IGDp=ind.igdp(x["F"]),
                                 cpu=x["meta"]["cpu"]))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(adir, "gkm_grandes_runs.csv"), index=False)
    for inst, g in df.groupby("instance", sort=False):
        piv = {m: g.pivot(index="run", columns="algo", values=m) for m in ["HV", "IGDp"]}
        others = [a for a in ALGOS if a != REF and a in piv["HV"]]
        rec = {"instance": inst, "n_runs": len(piv["HV"])}
        for a in ALGOS:
            if a in piv["HV"]:
                x = g[g.algo == a]
                rec[f"HV_{a}"] = x.HV.mean(); rec[f"HVsd_{a}"] = x.HV.std()
                rec[f"HVmed_{a}"] = x.HV.median(); rec[f"HVq1_{a}"] = x.HV.quantile(.25); rec[f"HVq3_{a}"] = x.HV.quantile(.75)
                rec[f"IGDp_{a}"] = x.IGDp.mean(); rec[f"IGDpsd_{a}"] = x.IGDp.std(); rec[f"cpu_{a}"] = x.cpu.mean()
        for m in ["HV", "IGDp"]:
            ph = holm([stats.wilcoxon(piv[m][REF], piv[m][a]).pvalue for a in others])
            for a, pv in zip(others, ph):
                better = piv[m][REF].mean() > piv[m][a].mean() if m == "HV" else piv[m][REF].mean() < piv[m][a].mean()
                rec[f"{m}_vs_{a}"] = ("+" if better else "-") if pv < alpha else "="
                rec[f"pHolm_{m}_vs_{a}"] = pv
                rec[f"wins_{m}_vs_{a}"] = int(((piv[m][REF] > piv[m][a]) if m == "HV" else (piv[m][REF] < piv[m][a])).sum())
        # hypothese de l'etape 08 : completer les centres supprime-t-il l'ecart avec l'init. aleatoire ?
        if {"GKM++", "GKM++(completed)", "NSGA-III-PM(GKM settings)"} <= set(piv["HV"].columns):
            rec["p_completed_vs_GKM"] = stats.wilcoxon(piv["HV"]["GKM++(completed)"], piv["HV"]["GKM++"]).pvalue
            rec["p_completed_vs_random"] = stats.wilcoxon(piv["HV"]["GKM++(completed)"],
                                                          piv["HV"]["NSGA-III-PM(GKM settings)"]).pvalue
        summ.append(rec)
        txt.append(f"=== {inst} ({rec['n_runs']} runs)")
        for a in ALGOS:
            if f"HV_{a}" in rec:
                tag = "" if a == REF else f"   FA {rec['HV_vs_' + a]} (HV, p={rec['pHolm_HV_vs_' + a]:.1e}, FA meilleur {rec['wins_HV_vs_' + a]}/{rec['n_runs']})" \
                                               f"  {rec['IGDp_vs_' + a]} (IGD+)"
                txt.append(f"  {a:28s} HV {rec['HV_' + a]:.3f}  IGD+ {rec['IGDp_' + a]:.4f}  cpu {rec['cpu_' + a]:.1f}s{tag}")
        if "p_completed_vs_GKM" in rec:
            txt.append(f"  GKM++(completed) vs GKM++ : p={rec['p_completed_vs_GKM']:.1e} ; vs init. aleatoire : p={rec['p_completed_vs_random']:.2f}")
        txt.append("")
    pd.DataFrame(summ).to_csv(os.path.join(adir, "gkm_grandes_summary.csv"), index=False)
    open(os.path.join(adir, "gkm_grandes_resume.txt"), "w", encoding="utf-8").write("\n".join(txt))
    print("\n".join(txt))


if __name__ == "__main__":   # obligatoire sous Windows
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--runs", type=int, default=30)
    ap.add_argument("--fe", type=int, default=125000)
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    if a.analyze:
        analyze(a.out)
    else:
        run_all(a.workers, a.runs, a.fe, a.out)
