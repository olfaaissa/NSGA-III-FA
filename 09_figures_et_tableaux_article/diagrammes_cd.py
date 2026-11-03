"""
diagrammes_cd.py -- Figures 3 (HV) et A4 (IGD+) de l'article : diagrammes de difference
critique (rangs moyens de Friedman sur les moyennes par instance, test de Nemenyi, alpha=0.05)
pour les 4 instances CloudBroker, les 20 instances DTLZ et les 24 instances WFG.
Entree : resultats/results_{cloud,dtlz,wfg}/_analysis/final_metrics.csv
Sortie : resultats/figures_cd/cd_HV_all.pdf et cd_IGDp_all.pdf
Usage  : python diagrammes_cd.py
"""
import os
_ICI = os.path.dirname(os.path.abspath(__file__))
RESULTATS = os.path.join(os.path.dirname(_ICI), "resultats")
OUT = os.path.join(RESULTATS, "figures_cd"); os.makedirs(OUT, exist_ok=True)
import pandas as pd, numpy as np
from scipy import stats
srcs={k: os.path.join(RESULTATS, f'results_{k}', '_analysis', 'final_metrics.csv') for k in ('cloud','dtlz','wfg')}
NAME={'A8_adaptfoot':'NSGA-III-FA','A1_FC':'NSGA-III-FC','A5_FA2':'FA-fixed','NSGA3-PM':'NSGA-III-PM','NSGA3-PMm':'NSGA-III-PM*','RVEA':'RVEA','MOEAD':'MOEA/D','AGEMOEA2':'AGE-MOEA-II','MOPSO':'MOPSO-CD'}
QA={2:1.960,3:2.343,4:2.569,5:2.728,6:2.850,7:2.949,8:3.031,9:3.102,10:3.164}
def ranks(key,met):
    d=pd.read_csv(srcs[key]); m=d.groupby(['instance','algo'])[met].mean().unstack()
    m=m.dropna(axis=1)
    r=m.rank(axis=1,ascending=(met!='HV'))
    avg=r.mean().sort_values(); N,k=m.shape
    chi=stats.friedmanchisquare(*[m[c] for c in m.columns])
    cd=QA[k]*np.sqrt(k*(k+1)/(6*N))
    return {NAME[a]:v for a,v in avg.items()},cd,N,chi

import numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from decimal import Decimal, ROUND_HALF_UP
plt.rcParams.update({'font.size':8})
def r2(v): return str(Decimal(str(v)).quantize(Decimal('0.01'),rounding=ROUND_HALF_UP))
def cliques(items,cd):
    vals=[v for _,v in items]; out=[]
    for i in range(len(vals)):
        j=i
        while j+1<len(vals) and vals[j+1]-vals[i]<cd-1e-12: j+=1
        if j>i and not any(a<=i and j<=b for a,b in out): out.append((i,j))
    return out
def draw(ax,key,met,title):
    rk,cd,N,_=ranks(key,met); items=sorted(rk.items(),key=lambda t:t[1]); k=len(items)
    lo,hi=1,k
    S=hi-lo; ax.set_xlim(lo-0.80*S, hi+0.80*S); 
    nleft=(k+1)//2; nrows=max(nleft,k-nleft)
    ax.set_ylim(-(nrows+0.5)*0.42, 1.5); ax.axis('off')
    ax.plot([lo,hi],[0,0],'k',lw=0.9)
    for t in range(lo,hi+1):
        ax.plot([t,t],[0,0.08],'k',lw=0.9); ax.text(t,0.14,str(t),ha='center',va='bottom',fontsize=7.5)
    # CD bar
    ax.plot([lo,lo+cd],[0.62,0.62],color='#c0392b',lw=1.2)
    for x in (lo,lo+cd): ax.plot([x,x],[0.56,0.68],color='#c0392b',lw=1.2)
    ax.text(lo+cd/2,0.72,f'CD = {cd:.2f}',ha='center',va='bottom',color='#c0392b',fontsize=7.5)
    ax.text((lo+hi)/2,1.22,title,ha='center',va='bottom',fontsize=8.5,fontweight='bold')
    # labels
    for i,(name,v) in enumerate(items):
        if i<nleft:
            y=-(i+1)*0.42; xe=lo-0.10*S; ha='right'
        else:
            y=-(k-i)*0.42; xe=hi+0.10*S; ha='left'
        col='#1f4e9c' if name=='NSGA-III-FA' else 'black'; fw='bold' if name=='NSGA-III-FA' else 'normal'
        ax.plot([v,v],[0,y],color=col,lw=0.8); ax.plot([v,xe],[y,y],color=col,lw=0.8)
        ax.text(xe+(-0.02*S if ha=='right' else 0.02*S),y,f'{name} ({r2(v)})',ha=ha,va='center',color=col,fontweight=fw,fontsize=7.5)
    for n,(a,b) in enumerate(cliques(items,cd)):
        y=-0.12-0.12*n
        ax.plot([items[a][1]-0.03,items[b][1]+0.03],[y,y],color='#555',lw=1.8,solid_capstyle='butt')
W=372/72
def fig(met,fname,lab):
    hs=[1.3,1.0,1.0]
    f,axes=plt.subplots(3,1,figsize=(W,sum(hs)),gridspec_kw={'height_ratios':hs})
    for ax,(key,t) in zip(axes,[('cloud','CloudBroker (4 instances)'),('dtlz','DTLZ (20 instances)'),('wfg','WFG (24 instances)')]):
        draw(ax,key,met,f'{lab}, {t}')
    f.subplots_adjust(left=0.01,right=0.99,top=0.98,bottom=0.01,hspace=0.25)
    f.savefig(os.path.join(OUT,fname))
fig('HV','cd_HV_all.pdf','HV'); fig('IGDp','cd_IGDp_all.pdf','IGD$^{+}$')
