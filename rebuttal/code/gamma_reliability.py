#!/usr/bin/env python3
"""Measurement reliability of gamma on the 82-way benchmark: (1) split-half (random halves of the 984 cases; 200 splits) Spearman-Brown-corrected correlation across models, for gamma_e, gamma_p and accuracy;
(2) case-bootstrap 95% CIs of each model's gamma and the share of model pairs whose gamma intervals separate."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
from common_pl import *
from scipy.stats import spearmanr
b=Bench();models=[m for m in b.models() if b.accuracy(m)>=.05];E=b.E0;rng=np.random.default_rng(8)
ch={m:b.choices(m,E) for m in models};idx={m:[i for i,c in enumerate(b.predictions(m)) if c] for m in models};first={m:{i:b.predictions(m)[i] for i in idx[m]} for m in models}
n=len(b.v);corr=[]
for s in range(200):
    perm=rng.permutation(n);A=set(perm[:n//2]);G={'ge':([],[]),'gp':([],[]),'acc':([],[])}
    for m in models:
        for h,S in enumerate([A,set(range(n))-A]):
            cs=[Choice(b.predictions(m)[i],b.P,E[i]) for i in idx[m] if i in S]
            f=fit_plackett_luce(cs,n_starts=1);G['ge'][h].append(f['gamma_evidence']);G['gp'][h].append(f['gamma_prior'])
            tgt=lambda i:b.di[b.target[i]];G['acc'][h].append(np.mean([ (b.predictions(m)[i][0]==tgt(i)) if b.predictions(m)[i] else False for i in S]))
    corr.append({k:spearmanr(v[0],v[1]).statistic for k,v in G.items()})
R=pd.DataFrame(corr);sb=lambda r:2*r/(1+r);out=pd.DataFrame({'quantity':['gamma evidence','gamma prior','accuracy'],'split_half_spearman_mean':[R['ge'].mean(),R['gp'].mean(),R['acc'].mean()],'spearman_brown':[sb(R['ge'].mean()),sb(R['gp'].mean()),sb(R['acc'].mean())],'ci_lo':[R['ge'].quantile(.025),R['gp'].quantile(.025),R['acc'].quantile(.025)],'ci_hi':[R['ge'].quantile(.975),R['gp'].quantile(.975),R['acc'].quantile(.975)],'n_models':len(models)});out.to_csv(OUT/'gamma_split_half_reliability.csv',index=False);print(out.round(3).to_string())
rows=[]
for m in models:
    cs=ch[m];bs=[]
    for _ in range(200):
        ix=rng.integers(0,len(cs),len(cs));f=fit_plackett_luce([cs[i] for i in ix],n_starts=1);bs.append((f['gamma_evidence'],f['gamma_prior']))
    bs=np.array(bs);f0=fit_plackett_luce(cs,n_starts=1);rows.append(dict(model=m,gamma_e=f0['gamma_evidence'],ge_lo=np.quantile(bs[:,0],.025),ge_hi=np.quantile(bs[:,0],.975),gamma_p=f0['gamma_prior'],gp_lo=np.quantile(bs[:,1],.025),gp_hi=np.quantile(bs[:,1],.975)))
B=pd.DataFrame(rows);B.to_csv(OUT/'gamma_bootstrap_ci.csv',index=False)
def sep(lo,hi):
    k=0;t=0
    for i in range(len(lo)):
        for j in range(i+1,len(lo)):t+=1;k+=(lo[i]>hi[j]) or (lo[j]>hi[i])
    return k/t
print('share of model pairs with non-overlapping 95%% CIs: gamma_e %.2f, gamma_p %.2f'%(sep(B.ge_lo.to_numpy(),B.ge_hi.to_numpy()),sep(B.gp_lo.to_numpy(),B.gp_hi.to_numpy())))
pd.Series({'sep_gamma_e':sep(B.ge_lo.to_numpy(),B.ge_hi.to_numpy()),'sep_gamma_p':sep(B.gp_lo.to_numpy(),B.gp_hi.to_numpy())}).to_json(OUT/'gamma_ci_separation.json')
