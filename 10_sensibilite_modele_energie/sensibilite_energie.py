"""
sensibilite_energie.py -- Sensibilite des conclusions aux parametres du modele
energetique (PUE et part d'energie decarbonee r_j).

Question : le classement des algorithmes change-t-il si l'on modifie les PUE
et les parts decarbonees ? Les memes algorithmes sont relances avec, a chaque
fois, un jeu de parametres energetiques different :

  base          : parametres de l'article (Tableaux 5 et A4)
  PUE_uniforme  : PUE = 1,58 pour tous les fournisseurs (moyenne du secteur,
                  Uptime Institute) -> plus aucune difference de PUE
  PUE_inverse   : PUE redistribues dans l'ordre inverse (le fournisseur le plus
                  efficace recoit le PUE le plus eleve, et ainsi de suite)
  CFE_inverse   : parts decarbonees redistribuees dans l'ordre inverse
  CFE_aleatoire : parts decarbonees tirees uniformement dans [0,05 ; 0,98]
                  (bornes des regions du Tableau A4), graine fixe

Instances : CloudBroker 50x5 (generee) et Azure-trace 100x10 (donnees reelles).
Methodes  : NSGA-III-FA (A8_adaptfoot), NSGA-III-PM, NSGA-III-FC (A1_FC)
            = la methode, la reference standard et le concurrent le plus proche.
30 runs, 125 000 evaluations, memes graines et memes populations initiales
que la campagne principale.

  - Scenario "base" : les runs sont COPIES depuis results_cloud et
    results_azure (memes graines, meme code -> memes resultats) ; s'ils sont
    absents, ils sont recalcules.
  - Ordre run-major : run 1 de tout, puis run 2... Si on arrete avant la fin,
    chaque configuration a le meme nombre de runs.
  - Reprise : relancer le meme .bat saute les runs deja faits.

Analyse (--analyze) : pour chaque scenario et chaque instance, HV et IGD+
normalises sur l'union des fronts des trois methodes DANS CE SCENARIO
(les valeurs d'energie changent d'un scenario a l'autre), Wilcoxon apparie
avec correction de Holm (NSGA-III-FA contre chacune des deux autres).

Usage :
  python sensibilite_energie.py --workers 7
  python sensibilite_energie.py --analyze
"""
# --- chemins du dossier organise : le noyau partage et le dossier resultats/ ---
import os as _os, sys as _sys
_ICI = _os.path.dirname(_os.path.abspath(__file__))
_RACINE = _os.path.dirname(_ICI)
_sys.path.insert(0, _os.path.join(_RACINE, "noyau"))
RESULTATS = _os.path.join(_RACINE, "resultats")
# ---------------------------------------------------------------------------
import argparse, glob, json, os, shutil, time, warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

OUT = os.path.join(RESULTATS, "results_sens_energie")
INSTANCES = [("cb_N50_M5", "results_cloud"), ("az_N100_M10", "results_azure")]
ALGOS = ["A8_adaptfoot", "NSGA3-PM", "A1_FC"]
REF = "A8_adaptfoot"
SCENARIOS = ["base", "PUE_uniforme", "PUE_inverse", "CFE_inverse", "CFE_aleatoire"]
LABEL = {"A8_adaptfoot": "NSGA-III-FA", "NSGA3-PM": "NSGA-III-PM", "A1_FC": "NSGA-III-FC"}


def _reverse_ranks(v):
    """Ordre inverse sur les valeurs distinctes : la plus petite valeur devient
    la plus grande, la 2e plus petite la 2e plus grande, etc. Deux fournisseurs
    qui avaient la meme valeur (meme type) gardent la meme valeur."""
    v = np.asarray(v, float)
    u = np.unique(v)
    return u[::-1][np.searchsorted(u, v)]


def apply_scenario(problem, scen):
    if scen == "base":
        pass
    elif scen == "PUE_uniforme":
        problem.pue = np.full(problem.m_providers, 1.58)
    elif scen == "PUE_inverse":
        problem.pue = _reverse_ranks(problem.pue)
    elif scen == "CFE_inverse":
        problem.r = _reverse_ranks(problem.r)
    elif scen == "CFE_aleatoire":
        rng = np.random.default_rng(700 + problem.n_clients + problem.m_providers)
        problem.r = rng.uniform(0.05, 0.98, size=problem.m_providers)
    else:
        raise ValueError(scen)
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
    rs = np.random.RandomState(seed)                       # meme population initiale
    X_init = problem.xl + rs.rand(N, problem.n_var) * (problem.xu - problem.xl)
    algorithm = build_algorithm(algo, ref_dirs, N, X_init, repair, seed, 1.0, 30.0)
    cb = SnapshotCallback(fe_max)
    t0 = time.time()
    res = minimize(problem, algorithm, get_termination("n_eval", fe_max), seed=seed,
                   callback=cb, copy_algorithm=False, verbose=False)
    cpu = time.time() - t0
    F = nondominated_feasible(res.pop)
    np.savetxt(stem + "_F.csv", F, delimiter=",")
    mut = getattr(getattr(algorithm, "mating", None), "mutation", None)
    if isinstance(mut, FireflyMutationV2) and mut.log:
        pd.DataFrame(mut.log).to_csv(stem + "_activation.csv", index=False)
    meta = dict(scenario=scen, problem=pname, n_obj=4, algo=algo, run=run + 1, seed=seed,
                fe_max=fe_max, n_eval_measured=int(res.algorithm.evaluator.n_eval),
                cpu_seconds=round(cpu, 2), pue=list(map(float, problem.pue)), r=list(map(float, problem.r)))
    json.dump(meta, open(stem + "_meta.json", "w"), indent=1)
    return f"ok   {scen} {pname} {algo} run{run + 1}  FE={meta['n_eval_measured']}  {cpu:.0f}s"


def copy_base(out):
    for pname, src_root in INSTANCES:
        for algo in ALGOS:
            src = os.path.join(RESULTATS, src_root, f"{pname}_M4", algo)
            dst = os.path.join(out, "base", f"{pname}_M4", algo)
            if not os.path.isdir(src):
                print(f"  ({src_root}/{pname}_M4/{algo} absent : ces runs seront recalcules)")
                continue
            os.makedirs(dst, exist_ok=True)
            n = 0
            for f in glob.glob(os.path.join(src, "run*_F.csv")) + glob.glob(os.path.join(src, "run*_meta.json")):
                if not os.path.exists(os.path.join(dst, os.path.basename(f))):
                    shutil.copy2(f, dst); n += 1
            print(f"  base/{pname}/{algo} : {n} fichiers copies depuis {src_root}")


def run_all(workers, runs, fe_scale, out):
    from multiprocessing import Pool
    os.makedirs(out, exist_ok=True)
    if fe_scale == 1.0:
        copy_base(out)
    jobs = [(s, p, a, r, fe_scale, out) for r in range(runs) for s in SCENARIOS
            for p, _ in INSTANCES for a in ALGOS]
    print(f"{len(jobs)} jobs (deja faits = sautes) -> {out}  ({workers} processus)")
    with Pool(workers) as pool:
        for k, msg in enumerate(pool.imap_unordered(run_one, jobs), 1):
            print(f"[{k}/{len(jobs)}] {msg}", flush=True)


def analyze(out, alpha=0.05):
    from scipy import stats
    from analyze_v2 import Indicators, holm, load_instance

    rows = []
    for scen in SCENARIOS:
        for pname, _ in INSTANCES:
            d = os.path.join(out, scen, f"{pname}_M4")
            if not os.path.isdir(d):
                continue
            data = {a: v for a, v in load_instance(d).items() if a in ALGOS}
            if REF not in data or len(data) < 2:
                continue
            common = set.intersection(*[set(v) for v in data.values()])
            data = {a: {r: v[r] for r in common} for a, v in data.items()}
            ind = Indicators(pname, 4, data, hv_samples=0)
            for a, runs in data.items():
                for r, x in runs.items():
                    rows.append(dict(scenario=scen, instance=pname, algo=a, run=r,
                                     HV=ind.hv(x["F"]), IGDp=ind.igdp(x["F"])))
    df = pd.DataFrame(rows)
    adir = os.path.join(out, "_analysis"); os.makedirs(adir, exist_ok=True)
    df.to_csv(os.path.join(adir, "sens_energie_runs.csv"), index=False)

    summ = []
    for (scen, inst), g in df.groupby(["scenario", "instance"], sort=False):
        piv = {m: g.pivot(index="run", columns="algo", values=m) for m in ["HV", "IGDp"]}
        rec = dict(scenario=scen, instance=inst, n_runs=len(piv["HV"]))
        for a in ALGOS:
            if a in piv["HV"]:
                rec[f"HV_{LABEL[a]}"] = piv["HV"][a].mean()
                rec[f"IGDp_{LABEL[a]}"] = piv["IGDp"][a].mean()
        others = [a for a in ALGOS if a != REF and a in piv["HV"]]
        for m in ["HV", "IGDp"]:
            p = [stats.wilcoxon(piv[m][REF], piv[m][a]).pvalue for a in others]
            ph = holm(p)
            for a, pv in zip(others, ph):
                better = (piv[m][REF].mean() > piv[m][a].mean()) if m == "HV" else (piv[m][REF].mean() < piv[m][a].mean())
                rec[f"{m}_vs_{LABEL[a]}"] = ("+" if better else "-") if pv < alpha else "="
                rec[f"pHolm_{m}_vs_{LABEL[a]}"] = pv
        rec["gainHV_vs_PM_pct"] = 100 * (piv["HV"][REF].mean() / piv["HV"]["NSGA3-PM"].mean() - 1)
        summ.append(rec)
    S = pd.DataFrame(summ)
    S.to_csv(os.path.join(adir, "sens_energie_summary.csv"), index=False)

    # Tableau LaTeX (pret a coller dans l'article)
    inst_lab = {"cb_N50_M5": r"CloudBroker $50\times5$", "az_N100_M10": r"Azure-trace $100\times10$"}
    scen_lab = {"base": "Baseline", "PUE_uniforme": "Uniform PUE (1.58)", "PUE_inverse": "Reversed PUE",
                "CFE_inverse": r"Reversed $r_j$", "CFE_aleatoire": r"Random $r_j$",
                "consolidation": "Consolidation", "serveurs_recents": "2024 servers",
                "consolidation_recents": "Consolidation + 2024 servers"}
    lines = [r"\begin{tabular}{llcccccc}", r"\toprule",
             r" & & \multicolumn{3}{c}{HV} & \multicolumn{3}{c}{IGD$^{+}$} \\",
             r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}",
             r"Instance & Scenario & FA & PM & FC & FA & PM & FC \\", r"\midrule"]
    mark = {"+": r"$^{+}$", "-": r"$^{-}$", "=": r"$^{\approx}$"}
    for _, r in S.iterrows():
        cells = []
        for m, fmt in [("HV", "{:.3f}"), ("IGDp", "{:.3f}")]:
            for a in ["NSGA-III-FA", "NSGA-III-PM", "NSGA-III-FC"]:
                v = fmt.format(r[f"{m}_{a}"])
                if a != "NSGA-III-FA":
                    v += mark[r[f"{m}_vs_{a}"]]
                cells.append(v)
        lines.append(f"{inst_lab[r.instance]} & {scen_lab.get(r.scenario, r.scenario)} & " + " & ".join(cells) + r" \\")
    lines += [r"\botrule", r"\end{tabular}"]
    open(os.path.join(adir, "tab_sens_energie.tex"), "w").write("\n".join(lines))

    n = len(S)
    txt = [f"Configurations (scenario x instance) : {n}",
           f"HV : NSGA-III-FA significativement meilleur que NSGA-III-PM dans {int((S['HV_vs_NSGA-III-PM'] == '+').sum())}/{n}, "
           f"pire dans {int((S['HV_vs_NSGA-III-PM'] == '-').sum())}/{n}",
           f"HV : NSGA-III-FA contre NSGA-III-FC : meilleur {int((S['HV_vs_NSGA-III-FC'] == '+').sum())}, "
           f"egal {int((S['HV_vs_NSGA-III-FC'] == '=').sum())}, pire {int((S['HV_vs_NSGA-III-FC'] == '-').sum())}",
           f"IGD+ : FA contre PM : meilleur {int((S['IGDp_vs_NSGA-III-PM'] == '+').sum())}, "
           f"egal {int((S['IGDp_vs_NSGA-III-PM'] == '=').sum())}, pire {int((S['IGDp_vs_NSGA-III-PM'] == '-').sum())}",
           f"IGD+ : FA contre FC : meilleur {int((S['IGDp_vs_NSGA-III-FC'] == '+').sum())}, "
           f"egal {int((S['IGDp_vs_NSGA-III-FC'] == '=').sum())}, pire {int((S['IGDp_vs_NSGA-III-FC'] == '-').sum())}",
           f"Gain HV de FA sur PM : de {S.gainHV_vs_PM_pct.min():+.1f} % a {S.gainHV_vs_PM_pct.max():+.1f} %",
           "", S.round(4).to_string()]
    open(os.path.join(adir, "sens_energie_resume.txt"), "w", encoding="utf-8").write("\n".join(txt))
    print("\n".join(txt))
    print(f"\nFichiers dans {adir}")


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
