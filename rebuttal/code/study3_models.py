import pandas as pd,numpy as np
from pathlib import Path
from scipy.stats import spearmanr
R=Path(__file__).resolve().parents[1]/'results';A=R/'study3A';rng=np.random.default_rng(20261118)
P=pd.read_csv(R/'study3_predictors.csv');P=P[(P.accuracy>=.05)&(P.used_frac>=.5)]
P=P.merge(pd.read_csv(R/'model_params.csv')[['model','params_b']] if (R/'model_params.csv').exists() else pd.read_csv(R/'study2A/sct_by_model.csv')[['model','params_b']],on='model',how='left')
E1=pd.read_csv(A/'e1_by_model.csv');E2=pd.read_csv(A/'e2_by_model.csv');E3=pd.read_csv(A/'e3_by_model.csv')
T=[('Y1 staged rr gain (E1) PRIMARY',E1,'Y1_rr_gain',.9),('Y2 paired reversal (E2)',E2,'Y2_reversal',.9),('Y3 selective deletion effect (E3)',E3,'Y3_deletion_effect',.9),('S1 stage-1 reciprocal rank (E1)',E1,'rr_s1',.9),('S2 trap accuracy (E2)',E2,'acc_trap',.9)]
def lomo(X,y):
    X=np.column_stack([np.ones(len(y)),X]);e=[]
    for i in range(len(y)):
        m=np.ones(len(y),bool);m[i]=False;b=np.linalg.lstsq(X[m],y[m],rcond=None)[0];e.append(y[i]-X[i]@b)
    return np.array(e)
def bd(a,b,B=4000):
    d=a**2-b**2;n=len(d);bs=np.array([d[rng.integers(0,n,n)].mean() for _ in range(B)]);return d.mean(),*np.percentile(bs,[2.5,97.5]),2*min((bs<=0).mean(),(bs>=0).mean())
rows=[]
for name,df,col,thr in T:
    V=P.merge(df[['model','parse',col]],on='model');V=V[V.parse>=thr].dropna(subset=[col]);n=len(V)
    if n<8:print(name,'n=',n,'too few');continue
    y=V[col].values;X=lambda c:V[c].values;e={}
    e['M1 acc']=lomo(X(['accuracy']),y);e['M2 gamma_e']=lomo(X(['gamma_evidence']),y);e['M3 acc+gamma_e']=lomo(X(['accuracy','gamma_evidence']),y);e['M4 acc+top3']=lomo(X(['accuracy','top3']),y);e['M5 acc+mrr3']=lomo(X(['accuracy','mrr3']),y);e['M6 acc+gamma_p']=lomo(X(['accuracy','gamma_prior']),y);e['M7 params']=lomo(V[['params_b']].fillna(V.params_b.median()).values,y)
    d31=bd(e['M1 acc'],e['M3 acc+gamma_e']);d34=bd(e['M4 acc+top3'],e['M3 acc+gamma_e']);d35=bd(e['M5 acc+mrr3'],e['M3 acc+gamma_e'])
    rm={k:np.sqrt((v**2).mean()) for k,v in e.items()};rel=1-rm['M3 acc+gamma_e']/rm['M1 acc']
    ok=(rel>=.10) and d31[1]>0 and d34[1]>0 and d35[1]>0
    rows.append(dict(outcome=name,n=n,rho_acc=spearmanr(V.accuracy,y)[0],rho_gamma_e=spearmanr(V.gamma_evidence,y)[0],rho_top3=spearmanr(V.top3,y)[0],**{'rmse '+k:v for k,v in rm.items()},rel_reduction_M3_vs_M1=rel,d_sq_M1_M3=d31[0],lo=d31[1],hi=d31[2],p_boot=d31[3],M3_beats_M4_lo=d34[1],M3_beats_M5_lo=d35[1],criterion_met=ok))
O=pd.DataFrame(rows)
# Holm across tests
p=O.p_boot.values;o=np.argsort(p);h=np.empty(len(p));m=len(p);run=0
for i,j in enumerate(o):run=max(run,(m-i)*p[j]);h[j]=min(run,1)
O['p_holm']=h;O.to_csv(A/'study3_model_level.csv',index=False);pd.set_option('display.width',300);print(O.round(3).T.to_string())
