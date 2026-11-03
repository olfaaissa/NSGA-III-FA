"""
ablation_final.py -- Ablation "un mecanisme retire a la fois" de la methode
finale NSGA-III-FA (A8_adaptfoot) : variantes L1 a L8 definies dans
operators_v2.py, [OP-12].

Principe :
  - L'ablation par composants (A0-A7, etape 02) portait sur la variante A5 et
    l'instance 10x3. Celle-ci porte sur la methode FINALE, sur les instances de
    l'article : CloudBroker 50x5 et 200x10, DTLZ2 (M=5), DTLZ4 (M=8), DTLZ7 (M=5).
  - 8 variantes x 5 instances x 30 runs, budget d'evaluations identique
    (le meme que dans les campagnes principales).
  - Les runs de A8 et de NSGA-III-PM deja faits (results_cloud, results_dtlz)
    sont COPIES : memes graines, meme code -> memes resultats.
  - Ordre run-major (run 1 de tout, puis run 2...) : si on arrete avant la fin,
    toutes les variantes ont le meme nombre de runs et l'analyse reste valable.
  - Reprise automatique : relancer saute les runs deja faits.

Usage : python ablation_final.py --workers 7
        python ../noyau/analyze_v2.py --root ../resultats/results_ablation_final --ref A8_adaptfoot
"""
# --- chemins du dossier organise : le noyau partage et le dossier resultats/ ---
import os as _os, sys as _sys
_ICI = _os.path.dirname(_os.path.abspath(__file__))
_RACINE = _os.path.dirname(_ICI)
_sys.path.insert(0, _os.path.join(_RACINE, "noyau"))
RESULTATS = _os.path.join(_RACINE, "resultats")
# ---------------------------------------------------------------------------
import argparse, glob, os, shutil

from operators_v2 import ABLATIONS  # noqa: F401  (enregistre L1-L8)

HERE = RESULTATS   # results_cloud, results_dtlz et results_ablation_final sont dans resultats/
OUT = os.path.join(HERE, "results_ablation_final")
INSTANCES = [("cb_N50_M5", 4, "results_cloud"), ("cb_N200_M10", 4, "results_cloud"),
             ("dtlz2", 5, "results_dtlz"), ("dtlz4", 8, "results_dtlz"), ("dtlz7", 5, "results_dtlz")]
VARIANTS = ["L1_noattraction", "L2_nodecay", "L3_noswitch", "L4_fixedparams",
            "L5_fixedfoot", "L6_nofallback", "L7_rawdisp", "L8_firstfound"]
REUSED = ["A8_adaptfoot", "NSGA3-PM"]


def copy_existing(out):
    for p, m, src_root in INSTANCES:
        for algo in REUSED:
            src = os.path.join(HERE, src_root, f"{p}_M{m}", algo)
            dst = os.path.join(out, f"{p}_M{m}", algo)
            if not os.path.isdir(src):
                print(f"  ({src_root}/{p}_M{m}/{algo} absent : ces runs seront recalcules)")
                continue
            os.makedirs(dst, exist_ok=True)
            n = 0
            for f in glob.glob(os.path.join(src, "run*")):
                if not os.path.exists(os.path.join(dst, os.path.basename(f))):
                    shutil.copy2(f, dst); n += 1
            print(f"  {p}_M{m}/{algo} : {n} fichiers copies")


if __name__ == "__main__":   # obligatoire sous Windows
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--runs", type=int, default=30)
    ap.add_argument("--fe_scale", type=float, default=1.0, help="test rapide seulement")
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    from multiprocessing import Pool
    from run_v2 import run_one
    os.makedirs(a.out, exist_ok=True)
    if a.fe_scale == 1.0:
        copy_existing(a.out)
    algos = VARIANTS + REUSED            # A8 et PM copies => sautes
    jobs = [(p, m, al, r, 1.0, 30.0, a.fe_scale, a.out, None)
            for r in range(a.runs) for (p, m, _) in INSTANCES for al in algos]
    print(f"{len(jobs)} jobs (deja faits = sautes) -> {a.out}  ({a.workers} processus)")
    with Pool(a.workers) as pool:
        for k, msg in enumerate(pool.imap_unordered(run_one, jobs), 1):
            print(f"[{k}/{len(jobs)}] {msg}", flush=True)
