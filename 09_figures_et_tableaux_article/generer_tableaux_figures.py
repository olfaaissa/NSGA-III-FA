"""
generer_tableaux_figures.py -- Produit les tableaux LaTeX et deux figures de l'article
a partir des resultats deja analyses (dossier resultats/).

Tableaux : cloud_HV, cloud_IGDp, fe, activation, dtlz_HV, dtlz_IGDp, ablation,
footprint, footprint_sens. Figures : dispersion_cb_N50_M5.pdf, spec_power.pdf.
Sortie : ../article_tableaux_figures/tables et .../figures.
Prerequis : avoir lance les etapes 02, 03 et 05 (dossiers results_* dans resultats/).
Les tableaux gkm, azure, ablation_loo et activation_dtlz de l'article ont ete
ecrits a partir des fichiers CSV produits par les etapes 04, 06 et 08.
"""
import numpy as np, pandas as pd, os
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_ICI = os.path.dirname(os.path.abspath(__file__)); _RACINE = os.path.dirname(_ICI)
V = os.path.join(_RACINE, "resultats") + os.sep
OUT = os.path.join(_RACINE, "article_tableaux_figures") + os.sep
os.makedirs(OUT + "tables", exist_ok=True); os.makedirs(OUT + "figures", exist_ok=True)
T, F = OUT + "tables/", OUT + "figures/"

NAME = {"A8_adaptfoot": "NSGA-III-FA", "A1_FC": "NSGA-III-FC", "A5_FA2": "FA-fixed",
        "NSGA3-PM": "NSGA-III-PM", "NSGA3-PMm": "NSGA-III-PM$^{*}$", "RVEA": "RVEA",
        "MOEAD": "MOEA/D", "AGEMOEA2": "AGE-MOEA-II", "MOPSO": "MOPSO-CD"}
ABL = {"A0_randomwalk": "A0 no attraction", "A1_FC": "A1 attraction only (FC)",
       "A2_decay": "A2 A1 + step decay", "A3_switch": "A3 A1 + regime switch",
       "A4_rawdisp": "A4 A5, raw dispersion", "A5_FA2": "A5 full, $\\rho=1/n$",
       "A6_nofallback": "A6 A5, no fallback", "A7_fullvector": "A7 A5, $\\rho=1$"}


def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); a = np.empty_like(p); r = 0
    for i, idx in enumerate(o):
        r = max(r, (len(p) - i) * p[idx]); a[idx] = min(1, r)
    return a


def sym_table(df, met, hi, ref, instances, algos, inst_label, fmt="{:.3f}", sdfmt="{:.3f}", sd=True):
    """cell = mean (sd) + symbol vs ref (Wilcoxon-Holm). Returns dict[(inst,algo)] -> str and best."""
    cells, best = {}, {}
    for inst in instances:
        g = df[df.instance == inst]
        piv = g.pivot(index="run", columns="algo", values=met)
        means = piv.mean()
        best[inst] = means[algos].idxmax() if hi else means[algos].idxmin()
        oth = [a for a in algos if a != ref]
        ps = []
        for o in oth:
            pr = piv[[ref, o]].dropna()
            ps.append(stats.wilcoxon(pr[ref], pr[o]).pvalue if len(pr) > 5 and not np.allclose(pr[ref], pr[o]) else 1.0)
        pa = dict(zip(oth, holm(ps)))
        for a in algos:
            s = fmt.format(means[a]) + ((" (" + sdfmt.format(piv[a].std()).lstrip("0") + ")") if sd else "")
            if a != ref:
                refbetter = (means[ref] > means[a]) == hi
                s += r" $\approx$" if pa[a] >= 0.05 else (r" $\blacktriangle$" if refbetter else r" $\blacktriangledown$")
            if a == best[inst]:
                s = r"\textbf{" + s + "}"
            cells[(inst, a)] = s
    return cells


def write_table(path, cells, instances, algos, names, inst_label, caption, label, size=r"\footnotesize", rows="algos"):
    L = [r"\begin{table}[htbp]", r"\centering", r"\caption{" + caption + "}\\label{" + label + "}", size,
         r"\setlength{\tabcolsep}{2pt}"]
    if rows == "algos":
        L.append(r"\begin{tabular}{l" + "c" * len(instances) + "}")
        L.append(r"\toprule")
        L.append("Algorithm & " + " & ".join(inst_label[i] for i in instances) + r" \\")
        L.append(r"\midrule")
        for a in algos:
            L.append(names[a] + " & " + " & ".join(cells[(i, a)] for i in instances) + r" \\")
    else:
        L.append(r"\begin{tabular}{l" + "c" * len(algos) + "}")
        L.append(r"\toprule")
        L.append("Instance & " + " & ".join(names[a] for a in algos) + r" \\")
        L.append(r"\midrule")
        for i in instances:
            L.append(inst_label[i] + " & " + " & ".join(cells[(i, a)] for a in algos) + r" \\")
    L += [r"\botrule", r"\end{tabular}", r"\end{table}"]
    open(path, "w").write("\n".join(L) + "\n")


def friedman(df, met, hi, algos):
    m = df[df.algo.isin(algos)].groupby(["instance", "algo"])[met].mean().unstack()[algos].dropna()
    r = (-m if hi else m).rank(axis=1)
    chi, p = stats.friedmanchisquare(*[m[c] for c in m.columns])
    return r.mean().sort_values(), chi, p, len(m)


Q = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164}


def cd_plot(avg, n, fname, title):
    names, vals = list(avg.index), avg.values; k = len(names)
    cd = Q[k] * np.sqrt(k * (k + 1) / (6.0 * n))
    X = lambda r: k + 1 - r; half = (k + 1) // 2; rows = max(half, k - half)
    fig, ax = plt.subplots(figsize=(7.2, 1.9 + 0.32 * rows))
    ax.set_xlim(-2.8, k + 3.8); ax.set_ylim(-1.4 - 0.45 * rows, 1.6); ax.axis("off")
    ax.hlines(0, X(k), X(1), color="k", lw=1)
    for t in range(1, k + 1):
        ax.vlines(X(t), 0, 0.12, color="k", lw=1); ax.text(X(t), 0.22, str(t), ha="center", va="bottom", fontsize=9)
    ax.hlines(0.85, X(k), X(k) + cd, color="tab:red", lw=2); ax.vlines([X(k), X(k) + cd], 0.78, 0.92, color="tab:red", lw=1)
    ax.text(X(k) + cd / 2, 0.98, f"CD = {cd:.2f}", ha="center", va="bottom", fontsize=9, color="tab:red")
    cl = []
    for i in range(k):
        j = i
        while j + 1 < k and vals[j + 1] - vals[i] < cd: j += 1
        if j > i and not any(a <= i and j <= b for a, b in cl): cl.append((i, j))
    for c, (i, j) in enumerate(cl):
        ax.hlines(-0.25 - 0.14 * c, X(vals[i]) + 0.04, X(vals[j]) - 0.04, color="tab:blue", lw=3)
    y0 = -0.45 - 0.14 * max(1, len(cl))
    for i, (nm, v) in enumerate(zip(names, vals)):
        if i < half: y, e, ha, xt = y0 - 0.45 * i, X(1) + 0.3, "left", X(1) + 0.4
        else: y, e, ha, xt = y0 - 0.45 * (k - 1 - i), X(k) - 0.3, "right", X(k) - 0.4
        ax.plot([X(v), X(v), e], [0, y, y], color="k", lw=0.8)
        ax.text(xt, y, f"{nm} ({v:.2f})", ha=ha, va="center", fontsize=9)
    fig.suptitle(title, fontsize=10); fig.tight_layout(); fig.savefig(fname); plt.close(fig)
    return cd


plain = {k: v.replace("$^{*}$", "*") for k, v in NAME.items()}
SHORT = {"A8_adaptfoot": "FA (prop.)", "A1_FC": "FC", "A5_FA2": "FA-fixed", "NSGA3-PM": "PM", "RVEA": "RVEA", "MOEAD": "MOEA/D", "AGEMOEA2": "AGE-II"}
summary = {}

# ---------------- CloudBroker main campaign ----------------
d = pd.read_csv(V + "results_cloud/_analysis/final_metrics.csv")
inst = ["cb_N10_M3_M4", "cb_N50_M5_M4", "cb_N100_M10_M4", "cb_N200_M10_M4"]
il = {"cb_N10_M3_M4": r"$10\times3$", "cb_N50_M5_M4": r"$50\times5$", "cb_N100_M10_M4": r"$100\times10$", "cb_N200_M10_M4": r"$200\times10$"}
algos = ["A8_adaptfoot", "A1_FC", "A5_FA2", "NSGA3-PM", "NSGA3-PMm", "RVEA", "MOEAD", "AGEMOEA2", "MOPSO"]
for met, hi, lab in [("HV", True, "HV"), ("IGDp", False, "IGD$^{+}$")]:
    c = sym_table(d, met, hi, "A8_adaptfoot", inst, algos, il)
    write_table(T + f"cloud_{met}.tex", c, inst, algos, NAME, il,
                f"Mean {lab} (standard deviation) over 30 paired runs on the four CloudBroker instances ($N$ clients $\\times$ $M$ providers), equal budget of 125\\,000 evaluations. "
                r"$\blacktriangle$/$\blacktriangledown$/$\approx$: NSGA-III-FA significantly better/worse/not different (paired Wilcoxon, Holm-adjusted, $\alpha=0.05$). Best mean in bold.",
                f"tab:cloud_{met}", size=r"\scriptsize")
    avg, chi, p, n = friedman(d, met, hi, algos)
    avg.index = [plain[a] for a in avg.index]
    cd = cd_plot(avg, n, F + f"cd_cloud_{met}.pdf", f"Critical difference, CloudBroker, {'HV' if met=='HV' else 'IGD+'}")
    summary[f"cloud_{met}"] = (avg.round(2).to_dict(), round(chi, 2), p, cd)

# FE / CPU table
fe = pd.read_csv(V + "results_cloud/_analysis/fe_table.csv")
L = [r"\begin{table}[htbp]", r"\centering",
     r"\caption{Evaluation budget and run time on CloudBroker: population size $N_p$, prescribed budget FE$_{\max}$, measured number of evaluations (mean over 30 runs) and mean CPU time per run (s) on the $200\times10$ instance. The proposed operator performs no hidden evaluation.}\label{tab:fe}",
     r"\footnotesize", r"\begin{tabular}{lcccc}", r"\toprule", r"Algorithm & $N_p$ & FE$_{\max}$ & Measured FE & CPU (s) \\", r"\midrule"]
for a in algos:
    r = fe[(fe.instance == "cb_N200_M10_M4") & (fe.algo == a)].iloc[0]
    L.append(f"{NAME[a]} & {int(r.pop_size)} & {int(r.fe_max):,} & {r.fe_measured_mean:,.0f} & {r.cpu_s_mean:.0f}".replace(",", "\\,") + r" \\")
L += [r"\botrule", r"\end{tabular}", r"\end{table}"]
open(T + "fe.tex", "w").write("\n".join(L) + "\n")

# activation table
act = pd.read_csv(V + "results_cloud/_analysis/activation_summary.csv")
L = [r"\begin{table}[htbp]", r"\centering",
     r"\caption{Activation of the Firefly operator on CloudBroker (mean over 30 runs): share of offspring actually moved, split by attractor type (dominating parent, same-niche parent, first-front parent), and share of generations spent in the exploitation regime. No offspring is ever left unchanged by lack of an attractor.}\label{tab:activation}",
     r"\footnotesize", r"\begin{tabular}{llccccc}", r"\toprule",
     r"Instance & Operator & Activation & Dominating & Niche & Elite & Exploit. gens \\", r"\midrule"]
for i in inst:
    for a in ["A8_adaptfoot", "A5_FA2", "A1_FC"]:
        r = act[(act.instance == i) & (act.algo == a)].iloc[0]
        L.append(f"{il[i]} & {NAME[a]} & {r.activation_rate:.2f} & {r.share_dom:.3f} & {r.share_niche:.2f} & {r.share_elite:.2f} & {r.share_exploit_gens:.2f}" + r" \\")
    if i != inst[-1]: L.append(r"\addlinespace")
L += [r"\botrule", r"\end{tabular}", r"\end{table}"]
open(T + "activation.tex", "w").write("\n".join(L) + "\n")

# ---------------- DTLZ generalisation ----------------
d2 = pd.read_csv(V + "results_dtlz/_analysis/final_metrics.csv")
ins2 = [f"{p}_M{m}" for p in ["dtlz1", "dtlz2", "dtlz3", "dtlz4", "dtlz7"] for m in [3, 5, 8, 10]]
il2 = {i: i.split("_")[0].upper() + f", {i.split('_M')[1]}" for i in ins2}
al2 = ["A8_adaptfoot", "A1_FC", "NSGA3-PM", "RVEA", "MOEAD", "AGEMOEA2"]
for met, hi, lab in [("IGDp", False, "IGD$^{+}$"), ("HV", True, "HV")]:
    c = sym_table(d2, met, hi, "A8_adaptfoot", ins2, al2, il2, fmt="{:.4g}", sd=False)
    write_table(T + f"dtlz_{met}.tex", c, ins2, al2, SHORT, il2,
                f"Mean {lab} on DTLZ1--4 and DTLZ7 with $m\\in\\{{3,5,8,10\\}}$ objectives (rows: problem, $m$; 30 paired runs, equal budgets). FA (prop.) = NSGA-III-FA, FC = NSGA-III-FC, PM = NSGA-III-PM, AGE-II = AGE-MOEA-II. Symbols as in Table~\\ref{{tab:cloud_HV}}. AGE-MOEA-II failed on 3 runs of DTLZ4 ($m$=3) because of a division by zero in its pymoo implementation; statistics use the remaining 27 runs. Standard deviations are given in the published raw results.",
                f"tab:dtlz_{met}", size=r"\scriptsize", rows="inst")
    avg, chi, p, n = friedman(d2, met, hi, al2)
    avg.index = [plain[a] for a in avg.index]
    cd = cd_plot(avg, n, F + f"cd_dtlz_{met}.pdf", f"Critical difference, DTLZ (20 instances), {'HV' if met=='HV' else 'IGD+'}")
    summary[f"dtlz_{met}"] = (avg.round(2).to_dict(), round(chi, 2), p, cd)

# ---------------- Ablation (A0-A7) ----------------
d3 = pd.read_csv(V + "results_ablation/_analysis/final_metrics.csv")
ins3 = ["cloudbroker_M4", "dtlz2_M5", "dtlz4_M8", "dtlz7_M5"]
il3 = {"cloudbroker_M4": "CloudBroker$^{\\dagger}$", "dtlz2_M5": "DTLZ2 ($m$=5)", "dtlz4_M8": "DTLZ4 ($m$=8)", "dtlz7_M5": "DTLZ7 ($m$=5)"}
al3 = list(ABL.keys())
c = sym_table(d3, "IGDp", False, "A5_FA2", ins3, al3, il3, fmt="{:.4f}", sd=False)
write_table(T + "ablation.tex", c, ins3, al3, ABL, il3,
            r"Component-wise ablation (mean IGD$^{+}$, 30 paired runs, equal budgets). Symbols compare the reference variant A5 with each other variant ($\blacktriangle$: A5 significantly better). $^{\dagger}$Reference $10\times3$ instance~\cite{ref44} with its linear energy model.",
            "tab:ablation", size=r"\scriptsize")

# ---------------- adaptive footprint test ----------------
d4 = pd.read_csv(V + "results_adaptive/_analysis/final_metrics.csv")
ins4 = ["cloudbroker_M4", "dtlz2_M5", "dtlz4_M8", "dtlz7_M5", "dtlz1_M5", "dtlz1_M8", "dtlz3_M5", "dtlz3_M8"]
al4 = ["A8_adaptfoot", "A5_FA2", "A1_FC", "NSGA3-PM"]
il4 = {**il3, "dtlz1_M5": "DTLZ1 ($m$=5)$^{\\ddagger}$", "dtlz1_M8": "DTLZ1 ($m$=8)$^{\\ddagger}$", "dtlz3_M5": "DTLZ3 ($m$=5)$^{\\ddagger}$", "dtlz3_M8": "DTLZ3 ($m$=8)$^{\\ddagger}$"}
c = sym_table(d4, "IGDp", False, "A8_adaptfoot", ins4, al4, il4, fmt="{:.4g}", sd=False)
write_table(T + "footprint.tex", c, ins4, al4, SHORT, il4,
            r"Effect of the dispersion-driven mutation footprint (mean IGD$^{+}$, 30 paired runs). NSGA-III-FA lets the dispersion regime choose the footprint; FA-fixed uses a fixed $1/n$ footprint. $^{\ddagger}$Instances never used during design. Symbols as in Table~\ref{tab:cloud_HV}.",
            "tab:footprint", size=r"\scriptsize", rows="inst")

# ---------------- var_prob sensitivity ----------------
d5 = pd.read_csv(V + "results_varprob/_analysis/final_metrics.csv")
vp = {"A5_FA2": "$1/n$", "S_vp0.1": "0.1", "S_vp0.3": "0.3", "S_vp0.5": "0.5", "A7_fullvector": "1.0"}
m5 = d5.groupby(["algo", "instance"]).IGDp.mean().unstack()[ins3]
L = [r"\begin{table}[htbp]", r"\centering",
     r"\caption{Sensitivity to a fixed mutation footprint $\rho$ (share of perturbed variables; mean IGD$^{+}$, 30 runs). No single value is best on all problems, which motivates letting the dispersion regime select $\rho$.}\label{tab:footprint_sens}",
     r"\footnotesize", r"\begin{tabular}{lcccc}", r"\toprule", r"$\rho$ & " + " & ".join(il3[i] for i in ins3) + r" \\", r"\midrule"]
for a, lab in vp.items():
    row = []
    for i in ins3:
        v = m5.loc[a, i]; s = f"{v:.4f}"
        if v == m5[i].min(): s = r"\textbf{" + s + "}"
        row.append(s)
    L.append(lab + " & " + " & ".join(row) + r" \\")
L += [r"\botrule", r"\end{tabular}", r"\end{table}"]
open(T + "footprint_sens.tex", "w").write("\n".join(L) + "\n")

# ---------------- figures: dispersion raw vs normalized, SPEC curves ----------------
a = pd.read_csv(V + "results_cloud/cb_N50_M5_M4/A8_adaptfoot/run01_activation.csv")
fig, ax1 = plt.subplots(figsize=(6.4, 3.0))
ax1.plot(a.n_eval, a.delta_raw, color="tab:gray", lw=1, label=r"raw $\delta_t$ (unnormalized)")
ax1.set_ylabel("raw dispersion"); ax1.set_xlabel("Function evaluations")
ax2 = ax1.twinx()
ax2.plot(a.n_eval, a.delta_norm, color="tab:blue", lw=1.2, label=r"normalized $\tilde\delta_t$")
ax2.plot(a.n_eval, a.theta, color="tab:red", lw=1, ls="--", label=r"threshold $\theta_t$")
ex = a.regime.eq("exploit").values
ax2.fill_between(a.n_eval, 0, 1, where=ex, transform=ax2.get_xaxis_transform(), color="tab:orange", alpha=0.12, lw=0, label="exploitation regime")
ax2.set_ylabel("normalized dispersion")
h1, l1 = ax1.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
ax1.legend(h1 + h2, l1 + l2, fontsize=7, loc="upper right"); ax1.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(F + "dispersion_cb_N50_M5.pdf"); plt.close(fig)

import sys; sys.path.insert(0, os.path.join(_RACINE, "noyau"))
from cloud_v2 import SPEC_POWER
u = np.linspace(0, 100, 11)
fig, ax = plt.subplots(figsize=(4.6, 3.0))
lab = {"HP_ML110_G4_Xeon3040": "Small: HP ML110 G4", "HP_ML110_G5_Xeon3075": "Medium: HP ML110 G5", "IBM_x3550_XeonX5670": "XL: IBM x3550 (2$\\times$X5670)"}
for k, v in SPEC_POWER.items():
    ax.plot(u, v, marker="o", ms=3, label=lab[k])
ax.set_xlabel("CPU utilization (%)"); ax.set_ylabel("Power (W)"); ax.grid(alpha=0.3); ax.legend(fontsize=7)
fig.tight_layout(); fig.savefig(F + "spec_power.pdf"); plt.close(fig)

for k, v in summary.items(): print(k, v)
