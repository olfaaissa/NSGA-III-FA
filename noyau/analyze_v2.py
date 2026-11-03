"""
analyze_v2.py -- Analyse statistique et figures des campagnes.

    python analyze_v2.py                       # analyse results_v2/
    python analyze_v2.py --ref A5_FA2 --hv_samples 200000

Sorties dans results_v2/_analysis/ :
    final_metrics.csv          HV et IGD+ finaux, par instance, algorithme et run
    summary_<HV|IGDp>.csv/.tex moyenne +- ecart-type + symbole Wilcoxon-Holm
    fe_table.csv/.tex          FE theorique / mesure / temps CPU
    activation_summary.csv     taux d'activation et repartition
    friedman.txt               Friedman + rangs moyens
    cd_<HV|IGDp>.pdf           diagrammes de difference critique
    conv_<instance>.pdf        IGD+ (et HV si M <= 5) en fonction des FE,
                               mediane + IC 95 % bootstrap
    activation_<instance>.pdf  taux d'activation au fil des FE, A5 vs A6
    dispersion_cloudbroker.pdf delta brut vs normalise

Methode (etiquettes reprises dans le code) :
  [AN-1] DTLZ : normalisation et IGD+ calcules avec le FRONT VRAI.
         CloudBroker : union des fronts obtenus (front inconnu).
  [AN-2] HV exact pour M <= 5 ; Monte-Carlo a echantillons FIXES pour
         M > 5 (memes points pour tous les algorithmes, donc l'appariement
         est conserve).
  [AN-3] Wilcoxon apparie : methode de reference contre chaque autre, avec
         correction de Holm PAR INSTANCE, et delta de Cliff.
  [AN-4] Diagramme CD Nemenyi (Demsar 2006), courbes de convergence avec
         IC 95 %, tableau des FE, taux d'activation.
"""

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
from scipy import stats
from pymoo.indicators.hv import HV
import gc
from pymoo.indicators.igd_plus import IGDPlus   # utilise seulement pour le test de coherence
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common_v2 import true_pareto_front

HERE = os.path.dirname(os.path.abspath(__file__))

# Valeurs critiques q_alpha (alpha = 0,05) du test de Nemenyi, Demsar 2006
# NOUVEAU (9 oct.) : noms des methodes tels qu'ils apparaissent dans l'article,
# utilises dans les legendes des figures (avant : noms internes du code).
LABELS = {"A8_adaptfoot": "NSGA-III-FA", "A1_FC": "NSGA-III-FC", "A5_FA2": "FA-fixed",
          "NSGA3-PM": "NSGA-III-PM", "NSGA3-PMm": "NSGA-III-PM*", "RVEA": "RVEA",
          "MOEAD": "MOEA/D", "AGEMOEA2": "AGE-MOEA-II", "MOPSO": "MOPSO-CD"}
# Couleur fixe par methode (la meme dans toutes les figures)
COLORS = {"A8_adaptfoot": "#d62728", "A1_FC": "#ff7f0e", "A5_FA2": "#9467bd",
          "NSGA3-PM": "#1f77b4", "NSGA3-PMm": "#17becf", "RVEA": "#2ca02c",
          "MOEAD": "#8c564b", "AGEMOEA2": "#e377c2", "MOPSO": "#7f7f7f"}


def lab(algo):
    return LABELS.get(algo, algo)


def inst_title(pname, m):
    """cb_N50_M5 -> 'CloudBroker 50x5' ; dtlz7, 10 -> 'DTLZ7 (m=10)' (m = nombre d'objectifs)."""
    if pname.startswith("cb_N"):
        n, mp = pname[4:].split("_M")
        return f"CloudBroker {n}\u00d7{mp}"
    if pname.startswith("az_N"):
        n, mp = pname[4:].split("_M")
        return f"Azure-trace {n}\u00d7{mp}"
    return f"{pname.upper()} (m={m})"


Q_ALPHA = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949,
           8: 3.031, 9: 3.102, 10: 3.164}


# ------------------------------------------------------------------ helpers
def load_instance(inst_dir):
    data = {}
    for algo_dir in sorted(glob.glob(os.path.join(inst_dir, "*"))):
        if not os.path.isdir(algo_dir):
            continue
        algo = os.path.basename(algo_dir)
        runs = {}
        for meta_f in sorted(glob.glob(os.path.join(algo_dir, "run*_meta.json"))):
            stem = meta_f[:-len("_meta.json")]
            meta = json.load(open(meta_f))
            F = np.atleast_2d(np.loadtxt(stem + "_F.csv", delimiter=","))
            hist = dict(np.load(stem + "_hist.npz")) if os.path.exists(stem + "_hist.npz") else {}
            act = pd.read_csv(stem + "_activation.csv") if os.path.exists(stem + "_activation.csv") else None
            runs[meta["run"]] = dict(meta=meta, F=F, hist=hist, act=act)
        if runs:
            data[algo] = runs
    return data


def nd(F):
    return F[NonDominatedSorting().do(F, only_non_dominated_front=True)]


class Indicators:
    """[AN-1] [AN-2] Normalisation et indicateurs pour une instance."""

    def __init__(self, pname, m, data, hv_samples, seed=0):
        pf = true_pareto_front(pname, m)
        if pf is None:                                  # CloudBroker
            allF = np.vstack([r["F"] for runs in data.values() for r in runs.values()])
            self.ideal, self.nadir = allF.min(axis=0), allF.max(axis=0)
            self.pf_n = nd(self._n(allF))
        else:
            self.ideal, self.nadir = pf.min(axis=0), pf.max(axis=0)
            self.pf_n = self._n(pf)
        self.m = m
        self.pf_n = np.ascontiguousarray(self.pf_n, dtype=np.float64)
        self.ref = np.full(m, 1.1)
        if m <= 5:
            self._hv = HV(ref_point=self.ref)
            self.samples = None
        else:
            rng = np.random.default_rng(seed)
            self.samples = rng.random((hv_samples, m)) * 1.1

    def _n(self, F):
        span = np.where(self.nadir - self.ideal > 1e-12, self.nadir - self.ideal, 1.0)
        return (F - self.ideal) / span

    def igdp(self, F, chunk=256):
        """IGD+ (Ishibuchi et al. 2015) calcule par paquets du front de
        reference : memoire bornee, meme resultat que pymoo.IGDPlus."""
        A = self._n(np.atleast_2d(F)).astype(np.float64)
        total = 0.0
        for k in range(0, len(self.pf_n), chunk):
            Z = self.pf_n[k:k + chunk]
            d = np.maximum(A[None, :, :] - Z[:, None, :], 0.0)
            total += np.sqrt((d * d).sum(axis=2)).min(axis=1).sum()
        return float(total / len(self.pf_n))

    def hv(self, F):
        Fn = self._n(F)
        Fn = Fn[np.all(Fn < self.ref, axis=1)]
        if len(Fn) == 0:
            return 0.0
        if self.samples is None:
            return float(self._hv(Fn))
        dom = np.zeros(len(self.samples), dtype=bool)
        for k in range(0, len(self.samples), 2000):
            S = self.samples[k:k + 2000]
            dom[k:k + 2000] = np.any(np.all(Fn[None, :, :] <= S[:, None, :], axis=2), axis=1)
        return float(dom.mean() * 1.1 ** self.m)


def safe_latex(df, path, **kw):
    """Export LaTeX facultatif : pandas exige le paquet jinja2. S'il manque,
    on continue sans le .tex (pip install jinja2 pour l'avoir)."""
    try:
        df.to_latex(path, **kw)
    except ImportError:
        print(f"  (jinja2 absent : {os.path.basename(path)} non genere)")


def cliffs_delta(a, b):
    a, b = np.asarray(a), np.asarray(b)
    gt = (a[:, None] > b[None, :]).sum()
    lt = (a[:, None] < b[None, :]).sum()
    return (gt - lt) / float(len(a) * len(b))


def holm(pvals):
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (len(p) - rank) * p[idx])
        adj[idx] = min(1.0, running)
    return adj


def boot_median_ci(x, n_boot=1000, seed=0):
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    meds = np.median(rng.choice(x, size=(n_boot, len(x)), replace=True), axis=1)
    return np.median(x), np.percentile(meds, 2.5), np.percentile(meds, 97.5)


def cd_diagram(avg_ranks, n_datasets, fname, title):
    """[AN-4] Diagramme de difference critique (Nemenyi, alpha = 0,05),
    disposition de Demsar 2006 : rang 1 a droite, barres = groupes non
    significativement differents."""
    r = avg_ranks.sort_values()
    names, vals = list(r.index), r.values
    k = len(names)
    cd = Q_ALPHA.get(k, 3.164) * np.sqrt(k * (k + 1) / (6.0 * n_datasets))
    X = lambda rank: k + 1 - rank                     # rang 1 a droite
    half = (k + 1) // 2                               # meilleurs a droite
    n_rows = max(half, k - half)
    fig, ax = plt.subplots(figsize=(8, 2.2 + 0.35 * n_rows))
    ax.set_xlim(-2.6, k + 3.6)
    ax.set_ylim(-1.4 - 0.45 * n_rows, 1.6)
    ax.axis("off")
    ax.hlines(0, X(k), X(1), color="k", lw=1)
    for t in range(1, k + 1):
        ax.vlines(X(t), 0, 0.12, color="k", lw=1)
        ax.text(X(t), 0.22, str(t), ha="center", va="bottom", fontsize=9)
    ax.hlines(0.85, X(k), X(k) + cd, color="tab:red", lw=2)
    ax.vlines([X(k), X(k) + cd], 0.78, 0.92, color="tab:red", lw=1)
    ax.text(X(k) + cd / 2, 0.98, f"CD = {cd:.2f}", ha="center", va="bottom", fontsize=9, color="tab:red")
    # groupes maximaux non significativement differents
    cliques = []
    for i in range(k):
        j = i
        while j + 1 < k and vals[j + 1] - vals[i] < cd:
            j += 1
        if j > i and not any(a0 <= i and j <= b0 for a0, b0 in cliques):
            cliques.append((i, j))
    for c, (i, j) in enumerate(cliques):
        y = -0.25 - 0.14 * c
        ax.hlines(y, X(vals[i]) + 0.04, X(vals[j]) - 0.04, color="tab:blue", lw=3)
    y_start = -0.45 - 0.14 * max(1, len(cliques))
    for i, (name, v) in enumerate(zip(names, vals)):
        if i < half:                                  # cote droit
            y = y_start - 0.45 * i
            edge, ha, xt = X(1) + 0.3, "left", X(1) + 0.4
        else:                                         # cote gauche
            y = y_start - 0.45 * (k - 1 - i)
            edge, ha, xt = X(k) - 0.3, "right", X(k) - 0.4
        ax.plot([X(v), X(v), edge], [0, y, y], color="k", lw=0.8)
        ax.text(xt, y, f"{name} ({v:.2f})", ha=ha, va="center", fontsize=9)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(fname)
    plt.close(fig)
    return cd


# --------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.join(HERE, "results_v2"))
    ap.add_argument("--ref", default="A5_FA2", help="methode proposee")
    ap.add_argument("--hv_samples", type=int, default=100000)
    ap.add_argument("--alpha", type=float, default=0.05)
    # NOUVEAU (9 oct.) : ne traiter que certaines instances et ecrire ailleurs
    # (pour regenerer les figures de l'article sans toucher a _analysis)
    ap.add_argument("--only", nargs="*", default=None, help="ex. cb_N50_M5_M4 dtlz7_M10")
    ap.add_argument("--out", default=None, help="dossier de sortie (defaut : <root>/_analysis)")
    a = ap.parse_args()
    out = a.out or os.path.join(a.root, "_analysis")
    os.makedirs(out, exist_ok=True)

    rows, fe_rows, act_rows = [], [], []
    for inst_dir in sorted(glob.glob(os.path.join(a.root, "*_M*"))):
        inst = os.path.basename(inst_dir)
        if a.only and inst not in a.only:
            continue
        pname, m = inst.rsplit("_M", 1)
        m = int(m)
        data = load_instance(inst_dir)
        if not data:
            continue
        print(f"== {inst}: {', '.join(f'{k}({len(v)})' for k, v in data.items())}")
        ind = Indicators(pname, m, data, a.hv_samples)

        conv = {}
        for algo, runs in data.items():
            for run, r in runs.items():
                rows.append(dict(instance=inst, algo=algo, run=run,
                                 HV=ind.hv(r["F"]), IGDp=ind.igdp(r["F"])))
                me = r["meta"]
                fe_rows.append(dict(instance=inst, algo=algo, pop_size=me["pop_size"],
                                    fe_max=me["fe_max"], fe_measured=me["n_eval_measured"],
                                    n_gen=me["n_gen"], cpu_s=me["cpu_seconds"]))
                for key, Fh in r["hist"].items():
                    fe = int(key.split("_")[1])
                    v = dict(IGDp=ind.igdp(Fh))
                    if m <= 5:
                        v["HV"] = ind.hv(Fh)
                    conv.setdefault(algo, {}).setdefault(fe, []).append(v)
                if r["act"] is not None:
                    d = r["act"]
                    tot = d["n_off"].sum()
                    act_rows.append(dict(instance=inst, algo=algo, run=run,
                                         activation_rate=d["moved"].sum() / tot,
                                         share_dom=d["dom"].sum() / tot, share_niche=d["niche"].sum() / tot,
                                         share_elite=d["elite"].sum() / tot, share_inactive=d["inactive"].sum() / tot,
                                         share_exploit_gens=(d["regime"] == "exploit").mean()))

        # ---- courbes de convergence [AN-4]
        metrics = ["IGDp"] + (["HV"] if m <= 5 else [])
        fig, axes = plt.subplots(1, len(metrics), figsize=(5.2 * len(metrics), 3.6), squeeze=False)
        for ax, met in zip(axes[0], metrics):
            # la methode proposee est tracee en dernier (au-dessus des autres)
            for algo, by_fe in sorted(conv.items(), key=lambda kv: (kv[0] == a.ref, kv[0])):
                fes = sorted(by_fe)
                stats_ = [boot_median_ci([v[met] for v in by_fe[fe]]) for fe in fes]
                med = [s[0] for s in stats_]
                c = COLORS.get(algo)
                ax.plot(fes, med, lw=2.2 if algo == a.ref else 1.2, label=lab(algo), color=c,
                        zorder=3 if algo == a.ref else 2)
                ax.fill_between(fes, [s[1] for s in stats_], [s[2] for s in stats_], alpha=0.15, color=c)
            ax.set_xlabel("Function evaluations")
            ax.set_ylabel("IGD+" if met == "IGDp" else "HV")
            if met == "IGDp":
                ax.set_yscale("log")
            ax.set_title(f"{inst_title(pname, m)}: median, 95% CI", fontsize=10)
            ax.grid(alpha=0.3)
        axes[0][-1].legend(fontsize=7, loc="best")
        fig.tight_layout()
        fig.savefig(os.path.join(out, f"conv_{inst}.pdf"))
        plt.close(fig)

        # ---- activation au fil des FE (A5 contre A6 si present)
        acts = {al: [r["act"] for r in data[al].values() if r["act"] is not None] for al in data}
        acts = {k: v for k, v in acts.items() if v}
        if acts:
            fig, ax = plt.subplots(figsize=(5.2, 3.4))
            for al, lst in acts.items():
                df = pd.concat(lst).groupby("n_eval")["activation_rate"].mean()
                ax.plot(df.index, df.values, label=lab(al), color=COLORS.get(al))
            ax.set_xlabel("Function evaluations")
            ax.set_ylabel("Activation rate")
            ax.set_ylim(0, 1.05)
            ax.set_title(inst_title(pname, m), fontsize=10)
            ax.legend(fontsize=7)
            ax.grid(alpha=0.3)
            fig.tight_layout()
            fig.savefig(os.path.join(out, f"activation_{inst}.pdf"))
            plt.close(fig)
            if pname == "cloudbroker" and a.ref in acts:
                d = acts[a.ref][0]
                fig, ax1 = plt.subplots(figsize=(5.2, 3.4))
                ax1.plot(d["n_eval"], d["delta_raw"], color="tab:gray", label="raw δ")
                ax1.set_ylabel("raw dispersion")
                ax2 = ax1.twinx()
                ax2.plot(d["n_eval"], d["delta_norm"], color="tab:blue", label="normalized δ")
                ax2.plot(d["n_eval"], d["theta"], color="tab:blue", ls="--", label="θ_t")
                ax2.set_ylabel("normalized dispersion")
                ax1.set_xlabel("Function evaluations")
                fig.legend(fontsize=7, loc="upper right")
                fig.tight_layout()
                fig.savefig(os.path.join(out, "dispersion_cloudbroker.pdf"))
                plt.close(fig)

        del data, conv, ind                         # libere la memoire entre instances
        gc.collect()

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out, "final_metrics.csv"), index=False)

    # ---- tableaux resumes + Wilcoxon-Holm [AN-3]
    for met, better_high in [("HV", True), ("IGDp", False)]:
        lines = []
        for inst, g in df.groupby("instance"):
            piv = g.pivot(index="run", columns="algo", values=met)
            others = [c for c in piv.columns if c != a.ref]
            pv, deltas = [], []
            for o in others:
                pair = piv[[a.ref, o]].dropna()
                if len(pair) >= 5 and not np.allclose(pair[a.ref], pair[o]):
                    pv.append(stats.wilcoxon(pair[a.ref], pair[o]).pvalue)
                else:
                    pv.append(1.0)
                deltas.append(cliffs_delta(pair[a.ref], pair[o]) if len(pair) else 0.0)
            padj = holm(pv) if pv else []
            rec = dict(instance=inst)
            for c in piv.columns:
                rec[c] = f"{piv[c].mean():.4e} ± {piv[c].std():.1e}"
            for o, p, dlt in zip(others, padj, deltas):
                ref_better = (piv[a.ref].mean() > piv[o].mean()) == better_high
                sym = "≈" if p >= a.alpha else ("▲" if ref_better else "▽")   # ▲ = ref meilleur
                rec[o] += f" {sym}"
                rec[f"p_holm[{o}]"] = p
                rec[f"cliff[{o}]"] = round(dlt, 3)
            lines.append(rec)
        s = pd.DataFrame(lines)
        s.to_csv(os.path.join(out, f"summary_{met}.csv"), index=False)
        cols = ["instance"] + [c for c in s.columns if not c.startswith(("p_holm", "cliff")) and c != "instance"]
        safe_latex(s[cols], os.path.join(out, f"summary_{met}.tex"), index=False, escape=True)

    # ---- Friedman + CD [AN-4]
    with open(os.path.join(out, "friedman.txt"), "w", encoding="utf-8") as f:
        for met, better_high in [("HV", True), ("IGDp", False)]:
            means = df.groupby(["instance", "algo"])[met].mean().unstack().dropna(axis=1)
            if means.shape[0] < 2 or means.shape[1] < 3:
                f.write(f"{met}: pas assez d'instances/algorithmes communs\n")
                continue
            ranks = (-means if better_high else means).rank(axis=1)
            chi2, p = stats.friedmanchisquare(*[means[c] for c in means.columns])
            avg = ranks.mean().sort_values()
            f.write(f"{met}: Friedman chi2={chi2:.3f}, p={p:.3g}, {means.shape[0]} instances\n")
            f.write(avg.to_string() + "\n\n")
            cd = cd_diagram(avg, means.shape[0], os.path.join(out, f"cd_{met}.pdf"),
                            f"Critical difference — {'HV' if met == 'HV' else 'IGD+'} (Nemenyi, α=0.05)")
            f.write(f"CD = {cd:.3f}\n\n")

    # ---- FE + activation
    fe = pd.DataFrame(fe_rows).groupby(["instance", "algo"]).agg(
        pop_size=("pop_size", "first"), fe_max=("fe_max", "first"),
        fe_measured_mean=("fe_measured", "mean"), n_gen_mean=("n_gen", "mean"),
        cpu_s_mean=("cpu_s", "mean")).reset_index()
    fe.to_csv(os.path.join(out, "fe_table.csv"), index=False)
    safe_latex(fe, os.path.join(out, "fe_table.tex"), index=False, float_format="%.0f")
    if act_rows:
        act = pd.DataFrame(act_rows).groupby(["instance", "algo"]).mean(numeric_only=True).drop(columns="run")
        act.round(3).to_csv(os.path.join(out, "activation_summary.csv"))
    print(f"\nAnalyse ecrite dans {out}")


if __name__ == "__main__":
    main()
