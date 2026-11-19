#!/usr/bin/env python3
"""Hausman-McFadden IIA test on observed top-1 choices from the 82-way benchmark (conditional logit, 2 covariates).
Under IIA, estimates from the full choice set and from a restricted choice set A (using only cases whose first choice lies in A,
with alternatives outside A deleted) are both consistent; a significant difference is evidence against IIA for that alternative group.
H=(b_A-b_F)' (V_A-V_F)^+ (b_A-b_F) ~ chi2(df). This is a test of the choice-model assumption for the *observed* rankings,
unlike the earlier refit sensitivity, and uses the actual displayed candidate set."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
from common_pl import *
from scipy.optimize import minimize
from scipy.special import logsumexp
from scipy.stats import chi2,spearmanr
b=Bench();models=b.models()
GROUPS={'organ:'+o:[d for d,og in zip(b.diseases,b.organ) if og==o] for o in sorted(set(b.organ))}
GROUPS.update({'pair:Sarcoidosis|Sarcoidosis_Pulmo':['Sarcoidosis','Sarcoidosis_Pulmo'],'pair:UC|Crohn':['Ulcerative_Colitis','Crohn_Disease'],
 'pair:T1D|T2D':['T1_Diabetes','T2_Diabetes'],'cluster:primary headaches':['Migraine','Tension_Headache','Cluster_Headache','Trigeminal_Neuralgia'],
 'cluster:inflammatory arthritides':['Rheumatoid_Arthritis','Psoriatic_Arthritis','Ankylosing_Spondylitis','Reactive_Arthritis','Systemic_Lupus','Sjogren_Syndrome','Gout','Osteoarthritis'],
 'cluster:thyroid/adrenal':['Hyperthyroidism','Hypothyroidism','Addison_Disease','Cushing_Syndrome','Primary_Hyperaldosteronism','Pheochromocytoma'],
 'cluster:microangiopathies':['TTP','HUS','ITP','Disseminated_Intravascular_Coagulation'],
 'cluster:acute chest':['Acute_MI','Aortic_Dissection','Pulmonary_Embolism','Pericarditis','Pneumothorax','Cardiac_Tamponade']})
def cl(X,y,ridge=1e-3):
    n,k,p=X.shape
    def f(beta):
        u=X@beta;lse=logsumexp(u,axis=1);pr=np.exp(u-lse[:,None]);ll=(u[np.arange(n),y]-lse).sum()
        g=(X[np.arange(n),y]-(pr[:,:,None]*X).sum(1)).sum(0)-2*ridge*beta
        return -ll+ridge*beta@beta,-g
    r=minimize(f,np.zeros(p),jac=True,method='BFGS');beta=r.x;u=X@beta;pr=np.exp(u-logsumexp(u,axis=1)[:,None])
    m=(pr[:,:,None]*X).sum(1);H=np.einsum('nk,nkp,nkq->pq',pr,X,X)-m.T@m+2*ridge*np.eye(p)
    return beta,np.linalg.inv(H),-r.fun
rows=[]
E=b.E0
for model in models:
    pc=b.predictions(model);keep=[i for i,c in enumerate(pc) if c];y=np.array([pc[i][0] for i in keep])
    X=np.stack([np.broadcast_to(b.P,(len(keep),len(b.diseases))),E[keep]],-1);mu=X.mean((0,1));sd=X.std((0,1));X=(X-mu)/sd
    bF,VF,_=cl(X,y);accm=b.accuracy(model)
    for g,members in GROUPS.items():
        idx=[b.di[d] for d in members];drop=np.array(idx);alt=np.array([j for j in range(len(b.diseases)) if j not in set(idx)])
        sel=np.flatnonzero(~np.isin(y,drop))
        if len(sel)<100 or len(set(y[sel]))<8:continue
        remap={j:i for i,j in enumerate(alt)};yA=np.array([remap[v] for v in y[sel]])
        bA,VA,_=cl(X[sel][:,alt],yA);d=bA-bF;D=VA-VF
        w,Q=np.linalg.eigh(D);pos=w>1e-10*max(abs(w).max(),1e-12);Hs=float(d@(Q[:,pos]/w[pos])@Q[:,pos].T@d) if pos.sum()>0 else np.nan;df=int(pos.sum())
        rows.append(dict(model=model,accuracy=accm,group=g,n_all=len(y),n_restricted=len(sel),frac_dropped=1-len(sel)/len(y),gamma_evidence_full=bF[1]/sd[1],gamma_evidence_restricted=bA[1]/sd[1],
            delta_gamma_evidence=(bA[1]-bF[1])/sd[1],rel_change=(bA[1]-bF[1])/bF[1],gamma_prior_full=bF[0]/sd[0],gamma_prior_restricted=bA[0]/sd[0],hausman_H=Hs,df=df,p_value=chi2.sf(Hs,df) if df>0 else np.nan,V_diff_psd=bool((w>-1e-8).all())))
    print(model,flush=True)
df=pd.DataFrame(rows);df.to_csv(OUT/'iia_hausman_tests.csv',index=False)
# multiplicity: Benjamini-Hochberg within the (model, group) family
p=df.p_value.fillna(1).to_numpy();order=np.argsort(p);m=len(p);bh=np.empty(m);bh[order]=np.minimum.accumulate((p[order]*m/(np.arange(m)+1))[::-1])[::-1];df['p_bh']=np.minimum(bh,1);df.to_csv(OUT/'iia_hausman_tests.csv',index=False)
good=df[df.accuracy>=0.05]
summ=good.groupby('group').agg(n_models=('model','nunique'),median_rel_change=('rel_change','median'),max_abs_rel_change=('rel_change',lambda x:x.abs().max()),frac_p_lt_05=('p_value',lambda x:(x<.05).mean()),frac_bh_lt_05=('p_bh',lambda x:(x<.05).mean())).reset_index()
rho=[]
for g,gg in good.groupby('group'):
    if len(gg)>=6:rho.append(dict(group=g,spearman_full_vs_restricted=spearmanr(gg.gamma_evidence_full,gg.gamma_evidence_restricted).statistic))
summ=summ.merge(pd.DataFrame(rho),on='group');summ.to_csv(OUT/'iia_hausman_summary.csv',index=False);print(summ.round(3).to_string())
