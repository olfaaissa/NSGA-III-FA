"""
gkm_analyze.py -- Indicateurs de la comparaison NSGA-III-GKM++ (Tableau 18 de l'article).

Lit results_gkm/<algo>/runXX_F.csv (produits par gkm_compare.py) et calcule,
avec la meme methode que analyze_v2.py :
  - normalisation par l'ideal et le nadir de l'union de tous les fronts finaux ;
  - front de reference IGD+ = union non dominee de tous les fronts ;
  - HV exact (point de reference 1.1) et IGD+ ;
  - Wilcoxon apparie (meme graine = meme run) avec correction de Holm.

Usage : python gkm_analyze.py --root results_gkm
"""
# --- chemins du dossier organise : le noyau partage et le dossier resultats/ ---
import os as _os, sys as _sys
_ICI = _os.path.dirname(_os.path.abspath(__file__))
_RACINE = _os.path.dirname(_ICI)
_sys.path.insert(0, _os.path.join(_RACINE, "noyau"))
RESULTATS = _os.path.join(_RACINE, "resultats")
# ---------------------------------------------------------------------------
import argparse, glob, json, os

import numpy as np
import pandas as pd
from scipy import stats

from analyze_v2 import Indicators, holm

REF = "NSGA-III-FA"


def load(root):
    data = {}
    for d in sorted(glob.glob(os.path.join(root, "*"))):
        if not os.path.isdir(d) or os.path.basename(d).startswith("_"):
            continue
        runs = {}
        for m in sorted(glob.glob(os.path.join(d, "run*_meta.json"))):
            meta = json.load(open(m))
            F = np.atleast_2d(np.loadtxt(m.replace("_meta.json", "_F.csv"), delimiter=","))
            runs[meta["run"]] = dict(meta=meta, F=F)
        if runs:
            data[os.path.basename(d)] = runs
    return data


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.join(RESULTATS, "results_gkm"))
    a = ap.parse_args()
    data = load(a.root)
    # pf=None -> union des fronts (comme pour CloudBroker dans analyze_v2).
    # La normalisation utilise les fronts des trois methodes du Tableau 18 ;
    # la variante "GKM++(completed)" (ajoutee ensuite) est evaluee avec ces
    # memes bornes, pour que les valeurs du Tableau 18 restent inchangees.
    base3 = {k: v for k, v in data.items() if k != "GKM++(completed)"}
    ind = Indicators("cloudbroker", 3, base3, hv_samples=0)
    rows = [dict(algo=al, run=r, HV=ind.hv(d["F"]), IGDp=ind.igdp(d["F"]), cpu=d["meta"]["cpu"])
            for al, runs in data.items() for r, d in runs.items()]
    df = pd.DataFrame(rows)
    out = os.path.join(a.root, "_analysis"); os.makedirs(out, exist_ok=True)
    df.to_csv(os.path.join(out, "gkm_runs.csv"), index=False)
    s = df.groupby("algo").agg(HV_mean=("HV", "mean"), HV_std=("HV", "std"),
                               IGDp_mean=("IGDp", "mean"), IGDp_std=("IGDp", "std"), cpu=("cpu", "mean"))
    base = df[df.algo == REF].sort_values("run")
    others = [al for al in s.index if al != REF]
    for met in ["HV", "IGDp"]:
        p = []
        for al in others:
            o = df[df.algo == al].sort_values("run")
            common = sorted(set(o.run) & set(base.run))
            p.append(stats.wilcoxon(base.set_index("run").loc[common, met],
                                    o.set_index("run").loc[common, met]).pvalue)
        s.loc[others, f"pHolm_{met}"] = holm(p)
    s.round(4).to_csv(os.path.join(out, "gkm_summary.csv"))
    print(s.round(4).to_string())
    # Test de l'hypothese de diversite initiale : GKM++(completed) contre GKM++
    if "GKM++(completed)" in s.index and "GKM++" in s.index:
        a_ = df[df.algo == "GKM++(completed)"].set_index("run"); b_ = df[df.algo == "GKM++"].set_index("run")
        cm = sorted(set(a_.index) & set(b_.index))
        lines = []
        for met in ["HV", "IGDp"]:
            w = stats.wilcoxon(a_.loc[cm, met], b_.loc[cm, met])
            lines.append(f"{met}: completed {a_.loc[cm, met].mean():.4f} vs GKM++ {b_.loc[cm, met].mean():.4f} "
                         f"(W={w.statistic:.0f}, p={w.pvalue:.2e}, n={len(cm)})")
        open(os.path.join(out, "gkm_completed_test.txt"), "w").write("\n".join(lines))
        print("\n".join(lines))
