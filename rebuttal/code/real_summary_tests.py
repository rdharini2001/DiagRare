import numpy as np,pandas as pd
from pathlib import Path
from scipy.stats import spearmanr
OUT=Path('rebuttal/results/real');R=pd.read_csv(OUT/'real_model_summary.csv');D=pd.read_csv(OUT/'real_scored.csv')
rng=np.random.default_rng(4)
def auc(c,y):
    p=c[y==1];n=c[y==0]
    return float((p[:,None]>n[None,:]).mean()+.5*(p[:,None]==n[None,:]).mean()) if len(p)>=5 and len(n)>=5 else np.nan
rows=[]
for m,g in D[(D.exp=='real_closed')&D.parse&D.conf.notna()].groupby('model'):
    c=g.conf.to_numpy();y=g.correct.to_numpy();o=np.argsort(-c);cov50=o[:len(o)//2]
    rows.append(dict(model=m,conf_auroc=auc(c,y),mean_conf=c.mean(),acc=y.mean(),brier=np.mean((c-y)**2),acc_top50pct_conf=y[cov50].mean(),acc_bottom50pct_conf=y[o[len(o)//2:]].mean()))
C=pd.DataFrame(rows);R=R.merge(C,on='model',how='left');R.to_csv(OUT/'real_model_summary.csv',index=False)
ok=R[(R.accuracy_82way>=.05)&(R.parse_closed>=.8)];print('validity panel',len(ok),ok.model.tolist())
out=[]
def perm(a,b,df,n=5000):
    x=df[a].to_numpy();y=df[b].to_numpy();m=~(np.isnan(x)|np.isnan(y));x,y=x[m],y[m]
    if len(x)<6:return np.nan,np.nan,len(x)
    o=spearmanr(x,y).statistic;pp=np.array([spearmanr(rng.permutation(x),y).statistic for _ in range(n)]);return o,(np.sum(abs(pp)>=abs(o))+1)/(n+1),len(x)
for a,b in [('gamma_e_82way','real_gamma_e'),('accuracy_82way','real_gamma_e'),('acc_closed_parseable','real_gamma_e'),('gamma_p_82way','real_gamma_p'),('accuracy_82way','acc_closed_parseable'),('gamma_e_82way','acc_closed_parseable'),('gamma_e_82way','correct_but_insensitive_to_key_removal'),('acc_closed_parseable','correct_but_insensitive_to_key_removal'),('gamma_e_82way','order_flip'),('acc_closed_parseable','order_flip'),('gamma_e_82way','conf_auroc')]:
    r,p,n=perm(a,b,ok);out.append(dict(x=a,y=b,spearman=r,perm_p=p,n_models=n))
T=pd.DataFrame(out);T.to_csv(OUT/'real_validity_tests.csv',index=False);print(T.round(3).to_string())
print(R[['model','conf_auroc','mean_conf','acc','acc_top50pct_conf','acc_bottom50pct_conf']].round(3).to_string())
print(R[['model','parse_closed','acc_closed_parseable','real_gamma_e','real_gamma_e_lo','real_gamma_e_hi','real_gamma_p','abstain_when_true_missing','abstain_when_true_present','forced_wrong_when_missing','evidence_dependence_index','recovery_given_early_wrong','harm_given_early_right']].round(3).to_string())
