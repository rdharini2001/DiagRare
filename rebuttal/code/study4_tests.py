import sys,numpy as np,pandas as pd,itertools
from scipy.stats import spearmanr,binomtest,chi2
import statsmodels.api as sm
sys.path.insert(0,'rebuttal/scripts')
from common_pl import Bench,ROOT,OUT
S4=OUT/'study4';rng=np.random.default_rng(20261118);B=Bench();E=B.E0;n=len(B.v);pd.set_option('display.width',260);pd.set_option('display.max_columns',40)
D=pd.read_csv(S4/'coefs_raw.csv');SGi=pd.read_csv(S4/'subgroup_indices.csv');SG={k:SGi[k].dropna().astype(int).values for k in SGi}
P=pd.read_csv(OUT/'study3_predictors.csv');P=P[(P.accuracy>=.05)&(P.used_frac>=.5)];MODELS=[m for m in P.model if m in set(D.model)];P=P.set_index('model').loc[MODELS]
params=pd.read_csv(OUT/'study2A/sct_by_model.csv').set_index('model').params_b
def tab(g,e,col='ge'):return D[(D.group==g)&(D.est==e)].set_index('model')[col].reindex(MODELS)
def partial(x,y,z):
    rx,ry,rz=[pd.Series(a).rank().values for a in (x,y,z)];ex=rx-np.polyval(np.polyfit(rz,rx,1),rz);ey=ry-np.polyval(np.polyfit(rz,ry,1),rz);return spearmanr(ex,ey)[0]
# ---- per-case correctness per model
top1={m:np.array([bool(c) and B.diseases[c[0]]==t for c,t in zip(B.predictions(m),B.target)]) for m in MODELS}
sub_acc=lambda g:pd.Series({m:top1[m][SG[g]].mean() for m in MODELS})
# ---- 1 reliability, stability, descriptive
VARS=[('all','top1'),('all','tail'),('all','full'),('A','full'),('B','full'),('C','full'),('D_rare','full'),('D_common_matched','full'),('unambiguous','full'),('A','tail'),('B','tail'),('C','tail')]
rel=[];desc=[]
for g,e in VARS:
    x=tab(g,e);ok=x.notna();r=np.nan;rr=np.nan
    if g in('all','A','B','C'):
        sp=D[(D.est==e)&D.group.str.startswith(g+'|split')];rs=[]
        for _,h in sp.groupby('group'):
            h=h.set_index('model').reindex(MODELS).dropna(subset=['ge','gp'])
            if len(h)>=8:rs.append(spearmanr(h['ge'],h['gp'])[0])
        rr=np.nanmean(rs);r=2*rr/(1+rr)
        h1=tab(g+'|H1',e);h2=tab(g+'|H2',e);j=h1.notna()&h2.notna();hs=spearmanr(h1[j],h2[j])[0] if j.sum()>=8 else np.nan
    else:hs=np.nan
    acc=P.accuracy;sa=sub_acc(g) if g in SG else acc
    row=dict(group=g,est=e,n_models=int(ok.sum()),reliability_SB=r,disease_half_rho=hs)
    if ok.sum()>=8:
        row.update(rho_acc=spearmanr(x[ok],acc[ok])[0],partial_rho_acc=partial(x[ok].values,P.top3[ok].values if False else acc[ok].values,acc[ok].values) if False else np.nan)
        row['partial_rho_vs_top3|acc']=partial(x[ok].values,P.top3[ok].values,acc[ok].values) # coef vs top3 net of accuracy
        row['rho_subgroup_acc']=spearmanr(x[ok],sa[ok])[0];row['rho_top3']=spearmanr(x[ok],P.top3[ok])[0];row['rho_mrr3']=spearmanr(x[ok],P.mrr3[ok])[0]
    desc.append(row)
Dd=pd.DataFrame(desc).drop(columns=['partial_rho_acc'],errors='ignore');Dd.to_csv(S4/'descriptive.csv',index=False);print('=== DESCRIPTIVE (n models = analysis panel)');print(Dd.round(3).to_string(index=False))
# subgroup accuracy dispersion
print('\nsubgroup accuracy (mean over models):',{g:round(sub_acc(g).mean(),3) for g in ['all','A','B','C','D_rare','D_common_matched','unambiguous']})
# ---- 2 independent outcomes
A2=OUT/'study2A';S=pd.read_csv(A2/'sct_by_model.csv').set_index('model');G=pd.read_csv(A2/'grounded_by_model.csv').set_index('model');Ag=pd.read_csv(A2/'age_prior_by_model.csv').set_index('model');Eh=pd.read_csv(A2/'ehr_by_model.csv').set_index('model');Mm=pd.read_csv(A2/'mm_by_model.csv').set_index('model');L=pd.read_csv(OUT/'incremental_v2_model_outcomes.csv').set_index('model')
OUTC={'SCT panel score (PRIMARY)':S.sct_score,'grounded agreement':G.agree_reference.where(G.parse>=.9),'age-prior sensitivity':Ag.age_sensitivity.where(Ag.parse_age>=.9),'EHR gain':Eh.gain_true_vs_none,'MM text-only acc':Mm.acc_text_only}
for c in ['verified_minus_fabricated_flip','recovery_after_error','profile_card_gain','relation_probe_accuracy','four_candidate_accuracy']:OUTC['legacy: '+c]=L[c]
def lomo(X,y):
    X=np.column_stack([np.ones(len(y)),X]);e=[]
    for i in range(len(y)):
        m=np.ones(len(y),bool);m[i]=False;b=np.linalg.lstsq(X[m],y[m],rcond=None)[0];e.append(y[i]-X[i]@b)
    return np.array(e)
def bd(a,b,B_=4000):
    d=a**2-b**2;k=len(d);bs=np.array([d[rng.integers(0,k,k)].mean() for _ in range(B_)]);return d.mean(),np.percentile(bs,2.5),np.percentile(bs,97.5),2*min((bs<=0).mean(),(bs>=0).mean())
TESTS=[('gamma_top1','all','top1'),('gamma_tail','all','tail'),('gamma_full','all','full'),('gamma_e A near-tie','A','full'),('gamma_e B ambiguous','B','full'),('gamma_e C exclusionary','C','full'),('gamma_e D rare','D_rare','full')]
rows=[]
for oname,oser in OUTC.items():
    for tname,g,e in TESTS:
        x=tab(g,e);V=pd.concat([oser.reindex(MODELS).rename('y'),x.rename('x'),P.accuracy,P.top3,P.mrr3],axis=1).dropna();k=len(V)
        if k<10:continue
        y=V.y.values;e0=lomo(V[['accuracy']].values,y);e3=lomo(V[['top3']].values if False else V[['accuracy','top3']].values,y);e5=lomo(V[['accuracy','mrr3']].values,y);ec=lomo(V[['accuracy','x']].values,y)
        best=min([e0,e3,e5],key=lambda z:(z**2).mean());rb=np.sqrt((best**2).mean());rc=np.sqrt((ec**2).mean());rel_=1-rc/rb
        d0=bd(e0,ec);d3=bd(e3,ec);d5=bd(e5,ec)
        rows.append(dict(outcome=oname,test=tname,n=k,rho_coef_outcome=spearmanr(V.x,V.y)[0],rho_acc_outcome=spearmanr(V.accuracy,V.y)[0],rmse_acc=np.sqrt((e0**2).mean()),rmse_best_baseline=rb,rmse_with_coef=rc,rel_reduction_vs_best=rel_,sqdiff_vs_acc=d0[0],lo_vs_acc=d0[1],hi_vs_acc=d0[2],p_vs_acc=d0[3],lo_vs_top3=d3[1],lo_vs_mrr=d5[1],success=bool(rel_>=.10 and d0[1]>0 and d3[1]>0 and d5[1]>0)))
I=pd.DataFrame(rows)
def holm(p):
    p=np.asarray(p);o=np.argsort(p);h=np.empty(len(p));run=0
    for i,j in enumerate(o):run=max(run,(len(p)-i)*p[j]);h[j]=min(run,1)
    return h
def bh(p):
    p=np.asarray(p);o=np.argsort(p);m=len(p);q=np.empty(m);prev=1
    for rnk,j in list(enumerate(o))[::-1]:prev=min(prev,p[j]*m/(rnk+1));q[j]=prev
    return q
I['p_adj']=np.nan
pm=I.outcome.str.contains('PRIMARY')
if pm.any():I.loc[pm,'p_adj']=holm(I.loc[pm,'p_vs_acc'])
for o,g in I[~pm].groupby('outcome'):I.loc[g.index,'p_adj']=bh(g.p_vs_acc)
I.to_csv(S4/'incremental_tests.csv',index=False);print('\n=== INCREMENTAL VALUE (LOMO). Primary outcome = SCT; Holm over 7 tests; others BH within outcome');print(I[['outcome','test','n','rho_coef_outcome','rho_acc_outcome','rmse_acc','rmse_best_baseline','rmse_with_coef','rel_reduction_vs_best','lo_vs_acc','hi_vs_acc','p_adj','success']].round(3).to_string(index=False))
print('\nTESTS MEETING CRITERION:',I[I.success][['outcome','test']].values.tolist())
