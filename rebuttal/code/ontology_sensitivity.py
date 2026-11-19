#!/usr/bin/env python3
"""How sensitive are gamma_evidence and its model ordering to the stipulated ontology parameters and to relation errors?
Rankings are fixed (no new inference); only the evidence feature is recomputed. Case text is unchanged, so this tests the
*measurement* (feature) side of validity, not the models' behavior under corrected prompts."""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
from common_pl import *
from scipy.stats import spearmanr,pearsonr
b=Bench();rel0=b.relations();models=b.models();acc={m:b.accuracy(m) for m in models}
base_params={'req':.90,'sup':.50,'exc':.02,'bg':.03}
def fit_all(E):
    out={}
    for m in models:
        f=fit_plackett_luce(b.choices(m,E),n_starts=1);out[m]=(f['gamma_evidence'],f['gamma_prior'],f['converged'])
    return out
def summarize(name,fits,ref,rows,note=''):
    ms=[m for m in models if acc[m]>=0.05]
    g=np.array([fits[m][0] for m in ms]);g0=np.array([ref[m][0] for m in ms]);a=np.array([acc[m] for m in ms])
    gp=np.array([fits[m][1] for m in ms])
    rows.append(dict(scheme=name,n_models=len(ms),spearman_vs_baseline=spearmanr(g,g0).statistic,pearson_vs_baseline=pearsonr(g,g0)[0],
        spearman_gamma_accuracy=spearmanr(g,a).statistic,pearson_gamma_accuracy=pearsonr(g,a)[0],spearman_gammaprior_accuracy=spearmanr(gp,a).statistic,
        all_converged=all(fits[m][2] for m in models),note=note))
rows=[];per_model=[]
ref=fit_all(b.evidence(rel0,base_params))
for m in models:per_model.append(dict(scheme='baseline',model=m,accuracy=acc[m],gamma_evidence=ref[m][0],gamma_prior=ref[m][1]))
summarize('baseline (replication)',ref,ref,rows,'recomputed evidence reproduces stored feature to 1e-14')
schemes={'softer_probabilities req.70/sup.40/exc.10/bg.05':{'req':.70,'sup':.40,'exc':.10,'bg':.05},
 'sharper_probabilities req.95/sup.60/exc.01/bg.02':{'req':.95,'sup':.60,'exc':.01,'bg':.02},
 'weak_exclusion exc=.20 (exclusions barely informative)':{'req':.90,'sup':.50,'exc':.20,'bg':.03},
 'required=supporting (no hard necessity) req=sup=.50':{'req':.50,'sup':.50,'exc':.02,'bg':.03},
 'exclusions removed (treated as background)':{'req':.90,'sup':.50,'exc':.03,'bg':.03}}
for name,p in schemes.items():
    fits=fit_all(b.evidence(rel0,p));summarize(name,fits,ref,rows)
    for m in models:per_model.append(dict(scheme=name,model=m,accuracy=acc[m],gamma_evidence=fits[m][0],gamma_prior=fits[m][1]))
# random relation-label corruption: each relation independently reassigned with prob q to another class (req/sup/exc/bg)
rng=np.random.default_rng(20261109)
for q in [0.1,0.25]:
    sp=[];pc=[];ga=[]
    for rep in range(20):
        rel={}
        for d,mp in rel0.items():
            new={}
            for f,c in mp.items():
                if rng.random()<q:
                    alt=[x for x in ['req','sup','exc','bg'] if x!=c];c=alt[rng.integers(0,3)]
                if c!='bg':new[f]=c
            rel[d]=new
        fits=fit_all(b.evidence(rel,base_params));rr=[]
        summarize('tmp',fits,ref,rr);sp.append(rr[0]['spearman_vs_baseline']);pc.append(rr[0]['pearson_vs_baseline']);ga.append(rr[0]['spearman_gamma_accuracy'])
    rows.append(dict(scheme=f'random relation corruption q={q} (20 draws): mean [min,max]',n_models=sum(acc[m]>=.05 for m in models),spearman_vs_baseline=np.mean(sp),pearson_vs_baseline=np.mean(pc),spearman_gamma_accuracy=np.mean(ga),note=f'spearman_vs_baseline range [{min(sp):.3f},{max(sp):.3f}]; gamma-accuracy rho range [{min(ga):.3f},{max(ga):.3f}]'))
    print('corruption',q,np.mean(sp),min(sp),flush=True)
# audit-corrected variant, if the audit exists
corr=ROOT/'rebuttal/data/profile_corrections.csv'
if corr.exists():
    c=pd.read_csv(corr);variants={'audit: contradicted/overstated-exclusion relations -> background':['exclusion_overstated','contradicted'],
        'audit: + overstated-required relations downgraded to supporting':['exclusion_overstated','contradicted','required_overstated']}
    for name,kinds in variants.items():
        rel={d:dict(mp) for d,mp in rel0.items()}
        for _,r in c.iterrows():
            if r.issue_class in kinds and r.finding in rel[r.disease]:
                if r.issue_class=='required_overstated':rel[r.disease][r.finding]='sup'
                else:rel[r.disease].pop(r.finding)
        fits=fit_all(b.evidence(rel,base_params));summarize(name,fits,ref,rows,f'{int(c.issue_class.isin(kinds).sum())} relations changed')
        for m in models:per_model.append(dict(scheme=name,model=m,accuracy=acc[m],gamma_evidence=fits[m][0],gamma_prior=fits[m][1]))
pd.DataFrame(rows).to_csv(OUT/'ontology_sensitivity_summary.csv',index=False);pd.DataFrame(per_model).to_csv(OUT/'ontology_sensitivity_per_model.csv',index=False)
print(pd.DataFrame(rows).round(3).to_string())
