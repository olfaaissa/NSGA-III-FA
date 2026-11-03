"""
sens_qsigma.py -- Analyse de sensibilite (q, sigma) du seuil de dispersion
pour la methode finale A8_adaptfoot.

Principe :
  - La grille (q, sigma) est evaluee avec l'operateur FINAL (A8 : seuil normalise + empreinte adaptative),
    a budget d'evaluations egal (125 000 FE), sur CloudBroker 50 x 5.
  - Grille : q in {0.05, 0.10, 0.15, 0.20, 0.30} x sigma in {0.10, 0.25, 0.50, 0.75, 1.00}
    = 25 configurations x 30 runs. Tous les autres parametres = ceux de A8.
  - Point par defaut (q = 0.15, sigma = 0.50) = A8 exactement : ses 30 runs sont
    COPIES depuis results_cloud (memes graines, meme code -> memes resultats),
    donc 24 x 30 = 720 runs a calculer.
  - Ordre des runs : run 1 de toutes les configs, puis run 2, etc. Si on arrete
    la nuit avant la fin, chaque configuration a le meme nombre de runs et
    l'analyse reste valable (avec moins de runs).
  - Reprise : relancer le meme .bat saute les runs deja faits.
  - Normalisation HV / IGD+ : la MEME que la campagne principale (union des
    fronts finaux de toutes les methodes de results_cloud pour 50 x 5). Ainsi
    la valeur au point par defaut est identique a celle du Tableau 9 (HV, 0,465).

Usage :
  python sens_qsigma.py --workers 7            # calcule les runs
  python sens_qsigma.py --analyze              # heatmaps + tableau LaTeX
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

HERE = RESULTATS   # results_cloud et results_sens_qsigma sont dans resultats/
PNAME, M = "cb_N50_M5", 4
INST = f"{PNAME}_M{M}"
Q_GRID = [0.05, 0.10, 0.15, 0.20, 0.30]
S_GRID = [0.10, 0.25, 0.50, 0.75, 1.00]
DEFAULT = (0.15, 0.50)
OUT = os.path.join(HERE, "results_sens_qsigma")
CLOUD = os.path.join(HERE, "results_cloud")


def cname(q, s):
    return f"Q{q:.2f}_S{s:.2f}"


# Enregistrement des 25 configurations : A8 + (q, sigma). Ce code est au niveau
# du module pour etre re-execute dans chaque processus (Windows = spawn).
for _q in Q_GRID:
    for _s in S_GRID:
        ABLATIONS[cname(_q, _s)] = dict(ABLATIONS["A8_adaptfoot"], q=_q, sigma=_s)


def copy_default():
    """Copie les 30 runs A8 de la campagne principale dans la case par defaut."""
    src = os.path.join(CLOUD, INST, "A8_adaptfoot")
    dst = os.path.join(OUT, INST, cname(*DEFAULT))
    if not os.path.isdir(src):
        print("  (results_cloud absent : le point par defaut sera recalcule)")
        return
    os.makedirs(dst, exist_ok=True)
    n = 0
    for f in glob.glob(os.path.join(src, "run*")):
        if not os.path.exists(os.path.join(dst, os.path.basename(f))):
            shutil.copy2(f, dst); n += 1
    print(f"  point par defaut {cname(*DEFAULT)} : {n} fichiers copies depuis results_cloud")


def run_all(workers, runs, fe_scale=1.0):
    from multiprocessing import Pool
    from run_v2 import run_one
    os.makedirs(OUT, exist_ok=True)
    copy_default()
    names = [cname(q, s) for q in Q_GRID for s in S_GRID]
    # run-major : run 1 de toutes les configs, puis run 2, ...
    jobs = [(PNAME, M, n, r, 1.0, 30.0, fe_scale, OUT, None) for r in range(runs) for n in names]
    print(f"{len(jobs)} jobs (dont deja faits = sautes) -> {OUT}  ({workers} processus)")
    with Pool(workers) as pool:
        for k, msg in enumerate(pool.imap_unordered(run_one, jobs), 1):
            print(f"[{k}/{len(jobs)}] {msg}", flush=True)


def analyze(alpha=0.05):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from analyze_v2 import load_instance, Indicators, holm, cliffs_delta

    sens = load_instance(os.path.join(OUT, INST))
    ref_data = load_instance(os.path.join(CLOUD, INST)) if os.path.isdir(os.path.join(CLOUD, INST)) else sens
    if ref_data is sens:
        print("  (results_cloud absent : normalisation sur les fronts de la sensibilite)")
    ind = Indicators(PNAME, M, ref_data, hv_samples=100000)

    rows = []
    for name, runs in sens.items():
        q, s = float(name[1:5]), float(name[7:11])
        for r, d in runs.items():
            rows.append(dict(q=q, sigma=s, run=r, HV=ind.hv(d["F"]), IGDp=ind.igdp(d["F"])))
    df = pd.DataFrame(rows)
    adir = os.path.join(OUT, "_analysis"); os.makedirs(adir, exist_ok=True)
    df.to_csv(os.path.join(adir, "sens_runs.csv"), index=False)

    # On ne garde que les runs presents dans TOUTES les configs (appariement)
    common = set.intersection(*[set(g.run) for _, g in df.groupby(["q", "sigma"])])
    df = df[df.run.isin(common)]
    n_runs = len(common)
    dq, ds = DEFAULT
    base = df[(df.q == dq) & (df.sigma == ds)].sort_values("run")

    summ = []
    for (q, s), g in df.groupby(["q", "sigma"]):
        g = g.sort_values("run")
        rec = dict(q=q, sigma=s, HV_mean=g.HV.mean(), HV_std=g.HV.std(),
                   IGDp_mean=g.IGDp.mean(), IGDp_std=g.IGDp.std())
        if (q, s) != DEFAULT:
            rec["p_HV"] = stats.wilcoxon(g.HV.values, base.HV.values).pvalue
            rec["p_IGDp"] = stats.wilcoxon(g.IGDp.values, base.IGDp.values).pvalue
            rec["cliff_HV"] = cliffs_delta(g.HV.values, base.HV.values)
        summ.append(rec)
    S = pd.DataFrame(summ)
    mask = S.p_HV.notna()
    S.loc[mask, "pHolm_HV"] = holm(S.loc[mask, "p_HV"].values)
    S.loc[mask, "pHolm_IGDp"] = holm(S.loc[mask, "p_IGDp"].values)
    S["dHV_pct"] = 100 * (S.HV_mean / base.HV.mean() - 1)
    S.to_csv(os.path.join(adir, "sens_summary.csv"), index=False)

    # Friedman sur les 25 configurations (blocs = runs apparies, memes graines)
    piv = df.pivot_table(index="run", columns=["q", "sigma"], values="HV")
    fr = stats.friedmanchisquare(*[piv[c].values for c in piv.columns])

    # ---- heatmaps (HV et IGD+) ; case par defaut encadree
    for ind_name, col, cmap, fmt in [("HV", "HV_mean", "Blues", "{:.3f}"), ("IGD+", "IGDp_mean", "Blues_r", "{:.3f}")]:
        Z = S.pivot(index="q", columns="sigma", values=col).loc[Q_GRID, S_GRID].values
        fig, ax = plt.subplots(figsize=(4.6, 3.6))
        im = ax.imshow(Z, cmap=cmap, aspect="auto", origin="lower")
        for i in range(len(Q_GRID)):
            for j in range(len(S_GRID)):
                row = S[(S.q == Q_GRID[i]) & (S.sigma == S_GRID[j])].iloc[0]
                pcol = "pHolm_HV" if ind_name == "HV" else "pHolm_IGDp"
                star = "" if pd.isna(row.get(pcol)) or row[pcol] >= alpha else "*"
                v = Z[i, j]
                lum = (v - np.nanmin(Z)) / max(np.nanmax(Z) - np.nanmin(Z), 1e-12)
                dark = lum > 0.55 if cmap == "Blues" else lum < 0.45
                ax.text(j, i, fmt.format(v) + star, ha="center", va="center", fontsize=7.5,
                        color="white" if dark else "#1a1a1a")
        di, dj = Q_GRID.index(dq), S_GRID.index(ds)
        ax.add_patch(plt.Rectangle((dj - .5, di - .5), 1, 1, fill=False, ec="#d62728", lw=2))
        ax.set_xticks(range(len(S_GRID)), [f"{s:.2f}" for s in S_GRID])
        ax.set_yticks(range(len(Q_GRID)), [f"{q:.2f}" for q in Q_GRID])
        ax.set_xlabel(r"$\sigma$"); ax.set_ylabel(r"$q$")
        ax.set_title(f"Mean {ind_name}, CloudBroker 50x5 ({n_runs} runs)", fontsize=9)
        fig.colorbar(im, ax=ax, shrink=0.85)
        fig.tight_layout()
        tag = "hv" if ind_name == "HV" else "igdp"
        fig.savefig(os.path.join(adir, f"sens_qsigma_{tag}.pdf"))
        fig.savefig(os.path.join(adir, f"sens_qsigma_{tag}.png"), dpi=200)
        plt.close(fig)

    # ---- resume texte
    nsig_w = int(((S.pHolm_HV < alpha) & (S.cliff_HV < 0)).sum())
    nsig_b = int(((S.pHolm_HV < alpha) & (S.cliff_HV > 0)).sum())
    best = S.loc[S.HV_mean.idxmax()]
    txt = [f"Runs apparies par configuration : {n_runs}",
           f"HV au point par defaut (q={dq}, sigma={ds}) : {base.HV.mean():.4f}",
           f"HV moyen : min {S.HV_mean.min():.4f}, max {S.HV_mean.max():.4f} "
           f"(ecart au defaut : {S.dHV_pct.min():+.1f} % a {S.dHV_pct.max():+.1f} %)",
           f"Meilleure case : q={best.q}, sigma={best.sigma}, HV={best.HV_mean:.4f}",
           f"Cases significativement PIRES que le defaut (Wilcoxon-Holm) : {nsig_w}/24",
           f"Cases significativement MEILLEURES que le defaut : {nsig_b}/24",
           f"Friedman sur les 25 cases (HV) : chi2={fr.statistic:.1f}, p={fr.pvalue:.2e}"]
    open(os.path.join(adir, "sens_resume.txt"), "w", encoding="utf-8").write("\n".join(txt))
    print("\n".join(txt))
    print(f"\nFichiers dans {adir} : sens_qsigma_hv.pdf/png, sens_qsigma_igdp.pdf/png, "
          f"sens_summary.csv, sens_runs.csv, sens_resume.txt")


if __name__ == "__main__":   # obligatoire sous Windows
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--runs", type=int, default=30)
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--fe_scale", type=float, default=1.0, help="test rapide seulement (ex. 0.01)")
    ap.add_argument("--out", default=None, help="dossier de sortie (test)")
    a = ap.parse_args()
    if a.out:
        OUT = a.out
    if a.analyze:
        analyze()
    else:
        run_all(a.workers, a.runs, a.fe_scale)
