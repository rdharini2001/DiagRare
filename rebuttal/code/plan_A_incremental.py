#!/usr/bin/env python3
"""Plan workstream A: gamma-only vs accuracy-only vs combined for the two independently measured outcomes (randomized evidence effect delta_E; sequential recovery).
LOO and leave-one-family-out R^2 (1 - SSE/SST_loo), paired model-bootstrap CI of the difference in LOO squared error, accuracy-matched pairs."""
import sys
from pathlib import Path
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'rebuttal/results'
mm=pd.read_csv(OUT/'model_metrics.csv');ce=pd.read_csv(ROOT/'results_final_v2/main/causal_response_validation.csv')[['model','delta_E']];sq=pd.read_csv(ROOT/'results_final_v2/main/sequential_anchoring_validation.csv')[['model','recovery_rate','n_used_for_recovery','order_effect_at_step2']]
D=mm.merge(ce,on='model',how='left').merge(sq,on='model',how='left');print(len(D),D.delta_E.notna().sum(),D.recovery_rate.notna().sum());meta=pd.read_csv(ROOT/'data_final_v2/metadata/model_metadata.csv').set_index('model_tag')
D['family']=[meta.loc[m,'family'] if m in meta.index else m for m in D.model]
def design(X,tr,te):
    mu=X[tr].mean(0);sd=X[tr].std(0);sd[sd<1e-9]=1;return np.c_[np.ones(len(tr)),(X[tr]-mu)/sd],np.c_[np.ones(len(te)),(X[te]-mu)/sd]
def cv_pred(X,y,groups):
    n=len(y);p=np.zeros(n)
    for g in np.unique(groups):
        te=np.flatnonzero(groups==g);tr=np.flatnonzero(groups!=g);a,t=design(X,tr,te);p[te]=t@np.linalg.lstsq(a,y[tr],rcond=None)[0]
    return p
rng=np.random.default_rng(2);rows=[];specs={'accuracy':['accuracy'],'gamma_evidence':['gamma_evidence'],'accuracy+gamma_evidence':['accuracy','gamma_evidence'],'gamma_evidence+gamma_prior':['gamma_evidence','gamma_prior'],'mean_only':[]}
for outcome in ['delta_E','recovery_rate']:
    for minfrac in [0.0,0.5]:
        d=D.dropna(subset=[outcome,'accuracy','gamma_evidence']);d=d[d.used_frac>=minfrac].reset_index(drop=True);n=len(d);y=d[outcome].to_numpy()
        for cv_name,groups in [('leave-one-model-out',np.arange(n)),('leave-one-family-out',pd.factorize(d.family)[0])]:
            preds={};
            for s,c in specs.items():
                X=d[c].to_numpy() if c else np.zeros((n,1));preds[s]=cv_pred(X,y,groups) if c else np.array([y[groups!=g].mean() for g in groups])
            sst=np.sum((y-y.mean())**2)
            for s,p in preds.items():
                rows.append(dict(outcome=outcome,min_used_frac=minfrac,cv=cv_name,n=n,spec=s,cv_R2=1-np.sum((y-p)**2)/sst,cv_rmse=np.sqrt(np.mean((y-p)**2))))
            for a,b in [('gamma_evidence','accuracy'),('accuracy+gamma_evidence','accuracy'),('accuracy+gamma_evidence','gamma_evidence')]:
                ea=(y-preds[a])**2;eb=(y-preds[b])**2;diff=eb-ea;bs=np.mean(diff[rng.integers(0,n,(5000,n))],axis=1)
                rows.append(dict(outcome=outcome,min_used_frac=minfrac,cv=cv_name,n=n,spec=f'COMPARE {a} vs {b}',cv_R2=np.nan,cv_rmse=np.nan,mean_sq_error_reduction=diff.mean(),ci_lo=np.quantile(bs,.025),ci_hi=np.quantile(bs,.975)))
R=pd.DataFrame(rows);R.to_csv(OUT/'plan_A_incremental.csv',index=False);pd.set_option('display.width',250)
print(R[~R.spec.str.startswith('COMPARE')].pivot_table(index=['outcome','min_used_frac','cv','n'],columns='spec',values='cv_R2').round(3).to_string())
print(R[R.spec.str.startswith('COMPARE')][['outcome','min_used_frac','cv','n','spec','mean_sq_error_reduction','ci_lo','ci_hi']].round(4).to_string())
# accuracy-matched pairs: within +-5pp accuracy, largest gamma_evidence gap
for outcome in ['delta_E','recovery_rate']:
    d=D.dropna(subset=[outcome]);d=d[d.used_frac>=.5].reset_index(drop=True);pr=[]
    for i in range(len(d)):
        for j in range(i+1,len(d)):
            if abs(d.accuracy[i]-d.accuracy[j])<=.05:pr.append(dict(a=d.model[i],b=d.model[j],acc_a=d.accuracy[i],acc_b=d.accuracy[j],gap_gamma=d.gamma_evidence[i]-d.gamma_evidence[j],outcome_a=d[outcome][i],outcome_b=d[outcome][j]))
    P=pd.DataFrame(pr);
    if len(P):
        P['outcome_gap']=P.outcome_a-P.outcome_b;P['same_sign']=np.sign(P.gap_gamma)==np.sign(P.outcome_gap);P=P.reindex(P.gap_gamma.abs().sort_values(ascending=False).index);P.to_csv(OUT/f'plan_A_matched_pairs_{outcome}.csv',index=False);print(outcome,'accuracy-matched pairs',len(P),'gamma gap predicts outcome sign in',int(P.same_sign.sum()));print(P.head(5).round(3).to_string())
