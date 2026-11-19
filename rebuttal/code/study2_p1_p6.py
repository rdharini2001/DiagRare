import pandas as pd,numpy as np
from pathlib import Path
from scipy.stats import spearmanr
R=Path(__file__).resolve().parents[1]/'results';A=R/'study2A';rng=np.random.default_rng(20261118)
S=pd.read_csv(A/'sct_by_model.csv')[['model','parse','sct_score','params_b','acc82','gamma_e','gamma_p','used82']].rename(columns={'parse':'sct_parse'})
G=pd.read_csv(A/'grounded_by_model.csv')[['model','parse','agree_reference','agree_reference_conflict']].rename(columns={'parse':'g_parse'})
Ag=pd.read_csv(A/'age_prior_by_model.csv')[['model','parse_age','age_sensitivity']]
E=pd.read_csv(A/'ehr_by_model.csv')[['model','gain_true_vs_none']]
C=pd.read_csv(A/'cupcase_scored.csv');C=C[C.kind=='closed'].groupby('model').agg(cup_parse=('parsed','mean'),cup_acc=('correct','mean') if 'correct' in C else ('parsed','mean')).reset_index()
M=pd.read_csv(A/'mm_by_model.csv')[['model','acc_text_only']]
D=S.merge(G,on='model',how='left').merge(Ag,on='model',how='left').merge(E,on='model',how='left').merge(C,on='model',how='left').merge(M,on='model',how='left')
U=D[(D.acc82>=.05)&(D.used82>=.5)&(D.sct_parse>=.9)].copy();print('panel n =',len(U))
def boot_rho(x,y,B=4000):
    n=len(x);r=[]
    for _ in range(B):
        i=rng.integers(0,n,n)
        if np.std(x.values[i])>0 and np.std(y.values[i])>0:r.append(spearmanr(x.values[i],y.values[i])[0])
    return np.percentile(r,[2.5,97.5])
r=spearmanr(U.gamma_e,U.acc82)[0];lo,hi=boot_rho(U.gamma_e,U.acc82);print('P1 rho(gamma_e,acc82)=%.3f CI %.3f-%.3f; criterion >=0.90 -> %s'%(r,lo,hi,'MET' if r>=.9 else 'not met'))
L=U[U.params_b>=10];print('  models >=10B: n=%d rho=%.3f'%(len(L),spearmanr(L.gamma_e,L.acc82)[0]))
def lomo(X,y):
    X=np.column_stack([np.ones(len(y)),X]);e=[]
    for i in range(len(y)):
        m=np.ones(len(y),bool);m[i]=False;b=np.linalg.lstsq(X[m],y[m],rcond=None)[0];e.append(y[i]-X[i]@b)
    return np.array(e)
out=[]
for name,col in [('SCT panel-referenced score','sct_score'),('Grounded crossover agreement','agree_reference'),('Grounded conflict-cell agreement','agree_reference_conflict'),('Age-prior sensitivity','age_sensitivity'),('EHR gain from true history','gain_true_vs_none'),('MM text-only accuracy','acc_text_only')]:
    V=U.dropna(subset=[col]);
    if col in('agree_reference','agree_reference_conflict'):V=V[V.g_parse>=.9]
    if col=='age_sensitivity':V=V[V.parse_age>=.9]
    y=V[col].values;n=len(V)
    if n<8:continue
    e0=lomo(V[['acc82']].values,y);e1=lomo(V[['acc82','gamma_e']].values,y);e2=lomo(V[['acc82','gamma_p']].values,y);eg=lomo(V[['gamma_e']].values,y);ep=lomo(V[['params_b']].values,y)
    def dif(a,b):
        d=a**2-b**2;bs=[d[rng.integers(0,n,n)].mean() for _ in range(4000)];return d.mean(),*np.percentile(bs,[2.5,97.5])
    d1=dif(e0,e1);d2=dif(e0,e2)
    out.append(dict(outcome=name,n=n,rho_acc=spearmanr(V.acc82,y)[0],rho_gamma_e=spearmanr(V.gamma_e,y)[0],rho_gamma_p=spearmanr(V.gamma_p,y)[0],rho_params=spearmanr(V.params_b,y)[0],rmse_acc=np.sqrt((e0**2).mean()),rmse_acc_ge=np.sqrt((e1**2).mean()),gain_from_ge=d1[0],ge_lo=d1[1],ge_hi=d1[2],gain_from_gp=d2[0],gp_lo=d2[1],gp_hi=d2[2],rmse_ge_only=np.sqrt((eg**2).mean()),rmse_params_only=np.sqrt((ep**2).mean())))
O=pd.DataFrame(out);O.to_csv(A/'p1_p2_incremental_value.csv',index=False);pd.set_option('display.width',250);print(O.round(3).to_string())
