"""
figure_article.py -- Figure 6 de l article : difference de HV (%) de chaque case (q, sigma)
par rapport au reglage par defaut (0.15, 0.50), a partir de sens_qsigma_autres_summary.csv
(produit par sensibilite_q_sigma_autres.py --analyze). + / - : significativement meilleur / moins bon
(Wilcoxon apparie, Holm). Usage : python figure_article.py
"""
import os
_ICI = os.path.dirname(os.path.abspath(__file__))
RESULTATS = os.path.join(os.path.dirname(_ICI), "resultats")
import pandas as pd, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
d=pd.read_csv(os.path.join(RESULTATS,'results_sens_qsigma_autres','_analysis','sens_qsigma_autres_summary.csv'))
inst=[('az_N100_M10_M4','Azure-trace 100×10'),('dtlz2_M8','DTLZ2 ($m$ = 8)'),('dtlz4_M5','DTLZ4 ($m$ = 5)')]
fig,axes=plt.subplots(1,3,figsize=(10.6,3.6),gridspec_kw={'width_ratios':[3,5,5]},layout='constrained')
vmax=2.5
for ax,(k,t) in zip(axes,inst):
    s=d[d.instance==k]; qs=sorted(s.q.unique()); ss=sorted(s.sigma.unique())
    Z=np.zeros((len(qs),len(ss)))
    for i,q in enumerate(qs):
        for j,sg in enumerate(ss):
            r=s[(s.q==q)&(s.sigma==sg)]
            Z[i,j]=0 if r.empty else r.dHV_pct.values[0]
    im=ax.imshow(Z,cmap='RdBu',vmin=-vmax,vmax=vmax,origin='lower',aspect='auto')
    for i,q in enumerate(qs):
        for j,sg in enumerate(ss):
            r=s[(s.q==q)&(s.sigma==sg)]
            if abs(q-0.15)<1e-9 and abs(sg-0.5)<1e-9:
                ax.add_patch(Rectangle((j-.5+.08,i-.5+.08),.84,.84,fill=False,lw=1.8,ec='#222'))
                ax.text(j,i,'default',ha='center',va='center',fontsize=7.5); continue
            v=r.dHV_pct.values[0]; p=r.pHolm_HV.values[0]
            lab=f"{v:+.1f}" if abs(v)>=0.05 else "0.0"
            if p<0.05: lab+=('$^{+}$' if v>0 else '$^{-}$')
            ax.text(j,i,lab,ha='center',va='center',fontsize=7.5,color='white' if abs(v)>1.5 else 'black')
    ax.set_xticks(range(len(ss))); ax.set_xticklabels([f"{x:.2f}" for x in ss])
    ax.set_yticks(range(len(qs))); ax.set_yticklabels([f"{x:.2f}" for x in qs])
    ax.set_xlabel(r'$\sigma$'); ax.set_ylabel(r'$q$'); ax.set_title(t,fontsize=10)
cb=fig.colorbar(im,ax=axes,shrink=0.9); cb.set_label('HV difference from default (%)')
fig.savefig(os.path.join(RESULTATS,'results_sens_qsigma_autres','_analysis','sens_qsigma_others.pdf'),bbox_inches='tight')
