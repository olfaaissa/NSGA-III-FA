"""
sensibilite_q_sigma_autres.py -- Sensibilite du seuil (q, sigma) sur des
instances qui n'ont servi a regler AUCUN parametre.

L'etude de l'etape 07 n'utilise que CloudBroker 50x5, alors que les valeurs par
defaut (q = 0,15 ; sigma = 0,50) avaient ete fixees sur CloudBroker.
On refait donc la meme etude sur trois instances jamais utilisees pour un
reglage (ni pour fixer (q, sigma), ni pour choisir la methode finale) :

  az_N100_M10 : Azure-trace 100x10 (donnees reelles)   grille 3x3 (la plus longue)
                q in {0.05, 0.15, 0.30} x sigma in {0.25, 0.50, 1.00}
  dtlz2, M=8  : front concave regulier, 8 objectifs     grille 5x5 (comme l'etape 07)
  dtlz4, M=5  : densite biaisee, 5 objectifs            grille 5x5

  - 30 runs par case, budget de la campagne principale, tous les autres
    parametres = ceux de NSGA-III-FA (A8_adaptfoot).
  - La case par defaut (0.15, 0.50) = A8 exactement : ses runs sont COPIES
    depuis results_azure et results_dtlz (memes graines, meme code).
  - Normalisation : comme dans l'article (union des fronts de results_azure
    pour Azure ; front de Pareto exact pour DTLZ).
  - Ordre run-major, reprise automatique (relancer saute les runs faits).

Usage :
  python sensibilite_q_sigma_autres.py --workers 7
  python sensibilite_q_sigma_autres.py --analyze
"""
# --- chemins du dossier organise : le noyau partage et le dossier resultats/ ---
import os as _os, sys as _sys
_ICI = _os.path.dirname(_os.path.abspath(__file__))
_RACINE = _os.path.dirname(_ICI)
_sys.path.insert(0, _os.path.join(_RACINE, "noyau"))
RESULTATS = _os.path.join(_RACINE, "resultats")
# ---------------------------------------------------------------------------
import argparse, glob, os, shutil

import numpy as np
import pandas as pd
from scipy import stats

from operators_v2 import ABLATIONS

OUT = os.path.join(RESULTATS, "results_sens_qsigma_autres")
GRID5 = ([0.05, 0.10, 0.15, 0.20, 0.30], [0.10, 0.25, 0.50, 0.75, 1.00])
GRID3 = ([0.05, 0.15, 0.30], [0.25, 0.50, 1.00])
INSTANCES = [  # (probleme, M, dossier de la campagne principale, grille)
    ("az_N100_M10", 4, "results_azure", GRID3),
    ("dtlz2", 8, "results_dtlz", GRID5),
    ("dtlz4", 5, "results_dtlz", GRID5),
]
DEFAULT = (0.15, 0.50)
TITLE = {"az_N100_M10": "Azure-trace 100x10", "dtlz2": "DTLZ2", "dtlz4": "DTLZ4"}


def cname(q, s):
    return f"Q{q:.2f}_S{s:.2f}"


# Enregistrement de toutes les configurations (au niveau du module : Windows = spawn)
for _qs, _ss in (GRID5, GRID3):
    for _q in _qs:
        for _s in _ss:
            ABLATIONS[cname(_q, _s)] = dict(ABLATIONS["A8_adaptfoot"], q=_q, sigma=_s)


def copy_default(out):
    for p, m, src_root, _ in INSTANCES:
        src = os.path.join(RESULTATS, src_root, f"{p}_M{m}", "A8_adaptfoot")
        dst = os.path.join(out, f"{p}_M{m}", cname(*DEFAULT))
        if not os.path.isdir(src):
            print(f"  ({src_root}/{p}_M{m} absent : le point par defaut sera recalcule)")
            continue
        os.makedirs(dst, exist_ok=True)
        n = 0
        for f in glob.glob(os.path.join(src, "run*")):
            if not os.path.exists(os.path.join(dst, os.path.basename(f))):
                shutil.copy2(f, dst); n += 1
        print(f"  {p}_M{m} : point par defaut, {n} fichiers copies depuis {src_root}")


def run_all(workers, runs, fe_scale, out):
    from multiprocessing import Pool
    from run_v2 import run_one
    os.makedirs(out, exist_ok=True)
    if fe_scale == 1.0:
        copy_default(out)
    jobs = []
    for r in range(runs):                                   # run-major
        for p, m, _, (qs, ss) in INSTANCES:
            for q in qs:
                for s in ss:
                    jobs.append((p, m, cname(q, s), r, 1.0, 30.0, fe_scale, out, None))
    print(f"{len(jobs)} jobs (deja faits = sautes) -> {out}  ({workers} processus)")
    with Pool(workers) as pool:
        for k, msg in enumerate(pool.imap_unordered(run_one, jobs), 1):
            print(f"[{k}/{len(jobs)}] {msg}", flush=True)


def analyze(out, alpha=0.05):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from analyze_v2 import Indicators, cliffs_delta, holm, load_instance

    adir = os.path.join(out, "_analysis"); os.makedirs(adir, exist_ok=True)
    allS, txt = [], []
    fig, axes = plt.subplots(1, len(INSTANCES), figsize=(4.4 * len(INSTANCES), 3.6))
    for ax, (p, m, src_root, (qs, ss)) in zip(np.atleast_1d(axes), INSTANCES):
        d = os.path.join(out, f"{p}_M{m}")
        if not os.path.isdir(d):
            continue
        sens = load_instance(d)
        refd = os.path.join(RESULTATS, src_root, f"{p}_M{m}")
        ref_data = load_instance(refd) if (p.startswith("az_") and os.path.isdir(refd)) else sens
        ind = Indicators(p, m, ref_data, hv_samples=100000)
        rows = []
        for name, runs in sens.items():
            q, s = float(name[1:5]), float(name[7:11])
            for r, x in runs.items():
                rows.append(dict(q=q, sigma=s, run=r, HV=ind.hv(x["F"]), IGDp=ind.igdp(x["F"])))
        df = pd.DataFrame(rows)
        common = set.intersection(*[set(g.run) for _, g in df.groupby(["q", "sigma"])])
        df = df[df.run.isin(common)]
        base = df[(df.q == DEFAULT[0]) & (df.sigma == DEFAULT[1])].sort_values("run")
        summ = []
        for (q, s), g in df.groupby(["q", "sigma"]):
            g = g.sort_values("run")
            rec = dict(instance=f"{p}_M{m}", q=q, sigma=s, HV_mean=g.HV.mean(), IGDp_mean=g.IGDp.mean())
            if (q, s) != DEFAULT:
                rec["p_HV"] = stats.wilcoxon(g.HV.values, base.HV.values).pvalue
                rec["p_IGDp"] = stats.wilcoxon(g.IGDp.values, base.IGDp.values).pvalue
                rec["cliff_HV"] = cliffs_delta(g.HV.values, base.HV.values)
                rec["cliff_IGDp"] = cliffs_delta(g.IGDp.values, base.IGDp.values)
            summ.append(rec)
        S = pd.DataFrame(summ)
        mk = S.p_HV.notna()
        S.loc[mk, "pHolm_HV"] = holm(S.loc[mk, "p_HV"].values)
        S.loc[mk, "pHolm_IGDp"] = holm(S.loc[mk, "p_IGDp"].values)
        S["dHV_pct"] = 100 * (S.HV_mean / base.HV.mean() - 1)
        S["dIGDp_pct"] = 100 * (S.IGDp_mean / base.IGDp.mean() - 1)
        allS.append(S)
        piv = df.pivot_table(index="run", columns=["q", "sigma"], values="HV")
        fr = stats.friedmanchisquare(*[piv[c].values for c in piv.columns])
        hv_w = int(((S.pHolm_HV < alpha) & (S.cliff_HV < 0)).sum())
        hv_b = int(((S.pHolm_HV < alpha) & (S.cliff_HV > 0)).sum())
        ig_w = int(((S.pHolm_IGDp < alpha) & (S.cliff_IGDp > 0)).sum())   # IGD+ plus grand = pire
        ig_b = int(((S.pHolm_IGDp < alpha) & (S.cliff_IGDp < 0)).sum())
        nc = len(S) - 1
        txt += [f"=== {TITLE[p]} M={m} ({len(common)} runs apparies, grille {len(qs)}x{len(ss)})",
                f"HV defaut : {base.HV.mean():.4f} ; ecart des autres cases : {S.dHV_pct.min():+.1f} % a {S.dHV_pct.max():+.1f} %",
                f"IGD+ defaut : {base.IGDp.mean():.4f} ; ecart : {S.dIGDp_pct.min():+.1f} % a {S.dIGDp_pct.max():+.1f} %",
                f"HV  : cases significativement pires {hv_w}/{nc}, meilleures {hv_b}/{nc}",
                f"IGD+: cases significativement pires {ig_w}/{nc}, meilleures {ig_b}/{nc}",
                f"Friedman (HV) : chi2={fr.statistic:.1f}, p={fr.pvalue:.2e}", ""]
        # heatmap HV de l'instance
        Z = S.pivot(index="q", columns="sigma", values="HV_mean").loc[qs, ss].values
        im = ax.imshow(Z, cmap="Blues", aspect="auto", origin="lower")
        for i, q in enumerate(qs):
            for j, s in enumerate(ss):
                row = S[(S.q == q) & (S.sigma == s)].iloc[0]
                star = "" if pd.isna(row.get("pHolm_HV")) or row["pHolm_HV"] >= alpha else "*"
                lum = (Z[i, j] - np.nanmin(Z)) / max(np.nanmax(Z) - np.nanmin(Z), 1e-12)
                ax.text(j, i, f"{Z[i, j]:.3f}{star}", ha="center", va="center", fontsize=7.5,
                        color="white" if lum > 0.55 else "#1a1a1a")
        di, dj = qs.index(DEFAULT[0]), ss.index(DEFAULT[1])
        ax.add_patch(plt.Rectangle((dj - .5, di - .5), 1, 1, fill=False, ec="#d62728", lw=2))
        ax.set_xticks(range(len(ss)), [f"{s:.2f}" for s in ss])
        ax.set_yticks(range(len(qs)), [f"{q:.2f}" for q in qs])
        ax.set_xlabel(r"$\sigma$"); ax.set_ylabel(r"$q$")
        ax.set_title(f"Mean HV, {TITLE[p]}" + ("" if p.startswith("az_") else f" ($m$ = {m})"), fontsize=9)
        fig.colorbar(im, ax=ax, shrink=0.85)
    fig.tight_layout()
    fig.savefig(os.path.join(adir, "sens_qsigma_autres_hv.pdf"))
    fig.savefig(os.path.join(adir, "sens_qsigma_autres_hv.png"), dpi=200)
    pd.concat(allS).to_csv(os.path.join(adir, "sens_qsigma_autres_summary.csv"), index=False)
    open(os.path.join(adir, "sens_qsigma_autres_resume.txt"), "w", encoding="utf-8").write("\n".join(txt))
    print("\n".join(txt))
    print(f"Fichiers dans {adir}")


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
