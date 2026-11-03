"""
consolidation_serveurs_recents.py -- Deux hypotheses du modele energetique
mises a l'epreuve (suite de l'etape 10).

1) CONSOLIDATION. Dans l'article, la charge d'un fournisseur est repartie sur
   tous ses serveurs, tous allumes. Ici, les serveurs inutiles sont ETEINTS :
   la charge s_j = (sum_i x_ij) / kappa_j (en serveurs) remplit floor(s_j)
   serveurs a 100 %, le reste s_j - floor(s_j) occupe un dernier serveur, et les
   autres serveurs consomment 0 W (modele de consolidation de Beloglazov et
   Buyya 2012) :
       E_j = PUE_j * [floor(s_j) * P(1) + 1{frac>0} * P(frac)] * dt / 1000

2) SERVEURS RECENTS (2024). Les trois courbes SPECpower de CloudSim (2007-2010)
   sont remplacees par trois serveurs testes en 2024 (rapports SPECpower_ssj2008
   officiels, valeurs verifiees sur spec.org le 18 oct. 2026) :
     Small  : Dell PowerEdge T160 (1 x Xeon E-2488, 8 coeurs, 16 fils)
              power_ssj2008-20240406-01395 : 23.3 W (repos) ... 85.5 W (100 %)
     Medium : HPE ProLiant DL320 Gen11 (1 x Xeon Platinum 8581V, 60 coeurs, 120 fils)
              power_ssj2008-20240401-01392 : 94.4 W ... 349 W
     XL     : Fujitsu PRIMERGY RX2530 M7 (2 x Xeon Platinum 8592+, 128 coeurs, 256 fils)
              power_ssj2008-20240312-01380 : 171 W ... 930 W
   Instances Azure : kappa_j = nombre de fils du nouveau serveur (16, 120, 256).

Scenarios : base (copie de l'etape 10), consolidation, serveurs_recents,
consolidation_recents (les deux a la fois).
Instances : CloudBroker 50x5 et Azure-trace 100x10. Methodes : NSGA-III-FA,
NSGA-III-PM, NSGA-III-FC. 30 runs, 125 000 evaluations, memes graines.

Usage :
  python consolidation_serveurs_recents.py --workers 7
  python consolidation_serveurs_recents.py --analyze
"""
# --- chemins du dossier organise : le noyau partage et le dossier resultats/ ---
import os as _os, sys as _sys
_ICI = _os.path.dirname(_os.path.abspath(__file__))
_RACINE = _os.path.dirname(_ICI)
_sys.path.insert(0, _os.path.join(_RACINE, "noyau"))
_sys.path.insert(0, _os.path.join(_RACINE, "10_sensibilite_modele_energie"))
RESULTATS = _os.path.join(_RACINE, "resultats")
# ---------------------------------------------------------------------------
import argparse, glob, json, os, shutil, time, types, warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

OUT = os.path.join(RESULTATS, "results_consolidation_recents")
INSTANCES = ["cb_N50_M5", "az_N100_M10"]
ALGOS = ["A8_adaptfoot", "NSGA3-PM", "A1_FC"]
SCENARIOS = ["base", "consolidation", "serveurs_recents", "consolidation_recents"]
_U = np.linspace(0.0, 1.0, 11)

# Puissance (W) a 0 % (repos actif), 10 %, ..., 100 % -- rapports SPECpower 2024
SPEC_2024 = {
    "Dell_T160_XeonE2488":        [23.3, 32.3, 39.3, 45.5, 51.5, 57.4, 63.2, 68.8, 75.1, 80.3, 85.5],
    "HPE_DL320Gen11_Xeon8581V":   [94.4, 123, 140, 158, 177, 198, 219, 243, 265, 299, 349],
    "Fujitsu_RX2530M7_2xXeon8592": [171, 329, 376, 422, 468, 517, 570, 630, 711, 829, 930],
}
NEW_OF_OLD = {"HP_ML110_G4_Xeon3040": "Dell_T160_XeonE2488",
              "HP_ML110_G5_Xeon3075": "HPE_DL320Gen11_Xeon8581V",
              "IBM_x3550_XeonX5670": "Fujitsu_RX2530M7_2xXeon8592"}
THREADS_2024 = {"Dell_T160_XeonE2488": 16, "HPE_DL320Gen11_Xeon8581V": 120, "Fujitsu_RX2530M7_2xXeon8592": 256}


def apply_scenario(problem, scen):
    from cloud_v2 import SPEC_POWER, EPOCH_H
    recent = scen in ("serveurs_recents", "consolidation_recents")
    consol = scen in ("consolidation", "consolidation_recents")
    if scen == "base":
        return problem
    if recent:
        problem.servers = [NEW_OF_OLD[s] for s in problem.servers]
        if problem.kind == "az":                        # 1 unite = 1 vCPU
            problem.units_per_server = np.array([THREADS_2024[s] for s in problem.servers], float)
    curves = {**SPEC_POWER, **SPEC_2024}

    def power(j, u):
        return np.interp(np.clip(u, 0.0, 1.0), _U, curves[problem.servers[j]])

    def metrics(self, alloc):
        RT = float(np.sum(alloc * (self.L + self.T[None, :])))
        Profit = float(np.sum(alloc * (self.P[:, None] - self.C[None, :])))
        load = alloc.sum(axis=0)
        Ej = np.zeros(self.m_providers)
        for j in range(self.m_providers):
            n_srv = self.capacities[j] / self.units_per_server[j]
            if consol:
                s = min(load[j] / self.units_per_server[j], n_srv)
                full = np.floor(s); frac = s - full
                P = full * power(j, 1.0) + (power(j, frac) if frac > 1e-9 else 0.0)
            else:
                P = n_srv * power(j, load[j] / self.capacities[j])
            Ej[j] = self.pue[j] * P * EPOCH_H / 1000.0
        return RT, Profit, float(Ej.sum()), float(np.sum((1.0 - self.r) * Ej))

    problem.metrics = types.MethodType(metrics, problem)
    return problem


def run_one(job):
    try:
        return _run_one(job)
    except Exception as e:
        return f"ERREUR {job[:4]}: {type(e).__name__}: {e}"


def _run_one(job):
    from pymoo.optimize import minimize
    from pymoo.termination import get_termination
    from common_v2 import SEED, make_problem
    from operators_v2 import FireflyMutationV2
    from run_v2 import SnapshotCallback, build_algorithm, nondominated_feasible

    scen, pname, algo, run, fe_scale, out_root = job
    out_dir = os.path.join(out_root, scen, f"{pname}_M4", algo)
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.join(out_dir, f"run{run + 1:02d}")
    if os.path.exists(stem + "_meta.json"):
        return f"skip {scen} {pname} {algo} run{run + 1}"
    problem, ref_dirs, N, fe_max, repair = make_problem(pname, 4)
    apply_scenario(problem, scen)
    fe_max = int(fe_max * fe_scale)
    seed = SEED + run
    rs = np.random.RandomState(seed)
    X_init = problem.xl + rs.rand(N, problem.n_var) * (problem.xu - problem.xl)
    algorithm = build_algorithm(algo, ref_dirs, N, X_init, repair, seed, 1.0, 30.0)
    cb = SnapshotCallback(fe_max)
    t0 = time.time()
    res = minimize(problem, algorithm, get_termination("n_eval", fe_max), seed=seed,
                   callback=cb, copy_algorithm=False, verbose=False)
    cpu = time.time() - t0
    F = nondominated_feasible(res.pop)
    np.savetxt(stem + "_F.csv", F, delimiter=",")
    meta = dict(scenario=scen, problem=pname, n_obj=4, algo=algo, run=run + 1, seed=seed, fe_max=fe_max,
                n_eval_measured=int(res.algorithm.evaluator.n_eval), cpu_seconds=round(cpu, 2),
                servers=list(problem.servers), units_per_server=list(map(float, problem.units_per_server)))
    json.dump(meta, open(stem + "_meta.json", "w"), indent=1)
    return f"ok   {scen} {pname} {algo} run{run + 1}  FE={meta['n_eval_measured']}  {cpu:.0f}s"


def copy_base(out):
    """Scenario de base : copie des runs de l'etape 10 (ou des campagnes principales)."""
    srcs = [os.path.join(RESULTATS, "results_sens_energie", "base")]
    srcs += [os.path.join(RESULTATS, "results_cloud"), os.path.join(RESULTATS, "results_azure")]
    for p in INSTANCES:
        for a in ALGOS:
            dst = os.path.join(out, "base", f"{p}_M4", a)
            for root in srcs:
                src = os.path.join(root, f"{p}_M4", a)
                if os.path.isdir(src):
                    os.makedirs(dst, exist_ok=True)
                    n = 0
                    for f in glob.glob(os.path.join(src, "run*_F.csv")) + glob.glob(os.path.join(src, "run*_meta.json")):
                        if not os.path.exists(os.path.join(dst, os.path.basename(f))):
                            shutil.copy2(f, dst); n += 1
                    print(f"  base/{p}/{a} : {n} fichiers copies")
                    break


def run_all(workers, runs, fe_scale, out):
    from multiprocessing import Pool
    os.makedirs(out, exist_ok=True)
    if fe_scale == 1.0:
        copy_base(out)
    jobs = [(s, p, a, r, fe_scale, out) for r in range(runs) for s in SCENARIOS for p in INSTANCES for a in ALGOS]
    print(f"{len(jobs)} jobs (deja faits = sautes) -> {out}  ({workers} processus)")
    with Pool(workers) as pool:
        for k, msg in enumerate(pool.imap_unordered(run_one, jobs), 1):
            print(f"[{k}/{len(jobs)}] {msg}", flush=True)


def analyze(out):
    import sensibilite_energie as se
    se.SCENARIOS = SCENARIOS
    se.INSTANCES = [(p, None) for p in INSTANCES]
    se.analyze(out)
    # le tableau LaTeX de l'etape 10 utilise ses propres libelles : on renomme
    adir = os.path.join(out, "_analysis")
    os.replace(os.path.join(adir, "tab_sens_energie.tex"), os.path.join(adir, "tab_consolidation_recents.tex"))
    for f in ["sens_energie_runs.csv", "sens_energie_summary.csv", "sens_energie_resume.txt"]:
        os.replace(os.path.join(adir, f), os.path.join(adir, f.replace("sens_energie", "consolidation_recents")))


if __name__ == "__main__":   # obligatoire sous Windows
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--runs", type=int, default=30)
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--fe_scale", type=float, default=1.0, help="test rapide seulement (ex. 0.01)")
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    if a.analyze:
        analyze(a.out)
    else:
        run_all(a.workers, a.runs, a.fe_scale, a.out)
