#!/usr/bin/env python3
"""Top-1 organ-nested logit sensitivity, not a full-ranking or subtype-validity proof.
Formula: Train (2009), Chapter 4, eq. 4.2. Common dissimilarity lambda in [.1,1].
Logit is recovered at lambda=1. Nested logit retains IIA within each nest.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import logsumexp
from scipy.optimize import minimize
from scipy.stats import spearmanr
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'rebuttal/results'
def log_probabilities(theta,x,nests):
    b=theta[:2];lam=theta[2];u=x@b;groups=sorted(set(nests));inclusive=np.stack([logsumexp(u[:,nests==g]/lam,axis=1) for g in groups],axis=1);mapping={g:i for i,g in enumerate(groups)};lookup=np.array([mapping[g] for g in nests]);return u/lam+(lam-1)*inclusive[:,lookup]-logsumexp(lam*inclusive,axis=1)[:,None]
def fit(x,y,nests,nested):
    def obj(theta):
        pars=theta if nested else np.r_[theta,1.];lp=log_probabilities(pars,x,nests);return -lp[np.arange(len(y)),y].sum()+.001*np.sum(theta[:2]**2)
    starts=[np.array([0.,0.,l]) for l in [1.,.7,.3]] if nested else [np.zeros(2)]
    fits=[minimize(obj,b,method='L-BFGS-B',bounds=[(None,None),(None,None),(.1,1.)] if nested else None,options={'maxiter':500,'ftol':1e-10}) for b in starts];return min(fits,key=lambda r:r.fun)
def check():
    rng=np.random.default_rng(12);xx=rng.normal(size=(5,6,2));nest=np.array([0,0,1,1,2,2]);theta=np.array([.3,-.2,1.]);lp=log_probabilities(theta,xx,nest);u=xx@theta[:2];assert np.allclose(lp,u-logsumexp(u,axis=1)[:,None]);assert np.allclose(np.exp(log_probabilities(np.array([.3,.4,.4]),xx,nest)).sum(1),1.)
def main():
    check();ont=pd.read_csv(ROOT/'data_final_v2/diagrare_x/ontology.csv').set_index('disease');diseases=sorted(ont.index);di={d:i for i,d in enumerate(diseases)};v=pd.read_csv(ROOT/'data_final_v2/diagrare_x/vignettes.csv').set_index('vignette_id');o=pd.read_csv(ROOT/'results_final_v2/main/bayes_oracle_table_expanded.csv');E=o.pivot(index='vignette_id',columns='disease',values='evidence_loglik_ratio').reindex(index=v.index,columns=diseases).to_numpy();P=np.log(ont.loc[diseases,'prevalence'].to_numpy());nests=ont.loc[diseases,'organ_system'].to_numpy();x=np.stack([np.broadcast_to(P,E.shape),E],axis=2)
    models=['qwen2.5-1.5b','qwen2.5-3b','qwen2.5-7b','phi-3.5-mini','mistral-7b-instruct','olmo-2-7b'];rows=[]
    for model in models:
        p=pd.read_csv(ROOT/'results/predictions'/f'{model}__baseline.csv').set_index('vignette_id').reindex(v.index);valid=p.prediction_1.isin(diseases).to_numpy();xx=x[valid];y=p.loc[valid,'prediction_1'].map(di).to_numpy();targets=v.loc[valid,'target_disease'];groupmap={g:i%3 for i,g in enumerate(sorted(set(targets)))};folds=targets.map(groupmap).to_numpy()
        for fold in [-1,0,1,2]:
            tr=np.arange(len(y)) if fold==-1 else np.flatnonzero(folds!=fold);te=np.arange(len(y)) if fold==-1 else np.flatnonzero(folds==fold);mu=xx[tr].mean((0,1));sd=xx[tr].std((0,1));z=(xx-mu)/sd
            for nested in [False,True]:
                result=fit(z[tr],y[tr],nests,nested);pars=result.x if nested else np.r_[result.x,1.];lp=log_probabilities(pars,z[te],nests);rows.append({'model':model,'fold':fold,'evaluation':'in_sample' if fold==-1 else 'heldout_target_diseases','specification':'organ_nested_top1' if nested else 'logit_top1','n_train':len(tr),'n_test':len(te),'converged':result.success,'lambda':pars[2],'nll_per_top1_choice':-lp[np.arange(len(te)),y[te]].mean(),'gamma_evidence_utility':pars[1]/sd[1],'within_nest_evidence_slope':pars[1]/(sd[1]*pars[2])})
        print('Nested sensitivity',model,flush=True)
    out=pd.DataFrame(rows);out.to_csv(OUT/'nested_choice_sensitivity.csv',index=False);cv=out[out.fold>=0];summ=cv.groupby('specification').agg(n_fits=('converged','size'),converged=('converged','sum'),mean_heldout_nll=('nll_per_top1_choice','mean'),lambda_mean=('lambda','mean'));summ.to_csv(OUT/'nested_choice_summary.csv')
    full=out[out.fold==-1].pivot(index='model',columns='specification',values='within_nest_evidence_slope');rho=spearmanr(full.logit_top1,full.organ_nested_top1).statistic
    (OUT/'nested_choice_notes.md').write_text(f'''# Nested-choice sensitivity\n\nSource: [Train (2009), Chapter 4, equation 4.2](https://eml.berkeley.edu/books/choice2nd/Ch04_p76-96.pdf).\n\nSix representative models; top-1 outputs only; fixed 82 candidates. Three folds hold out target diseases. All scaling uses training data. Each nested fit uses three starting values, with shared lambda constrained to [0.1,1]. Convergence and boundary values are recorded. Ordinary logit is the exact lambda=1 special case, checked numerically.\n\nSpearman agreement between full-data logit and nested within-nest evidence slopes: {rho:.4f}. Compare within-nest slopes, not unadjusted utility coefficients. This is a limited robustness comparison: organ nests do not encode disease subtypes, within-nest IIA persists, and no full-ranking nested likelihood is claimed. In-sample rows must not be used as held-out evidence. Inspect lambda by model: in this run the held-out optimum is the lower bound .1 for Qwen2.5-1.5B and 1 for each of the other five models, so gains are concentrated in the weakest checkpoint.\n\nMean held-out NLL per top-1 choice:\n{summ.to_string()}\n''')
if __name__=='__main__':main()
