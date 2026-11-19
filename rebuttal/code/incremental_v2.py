#!/usr/bin/env python3
"""Added value of gamma beyond accuracy on NEW, independently generated outcomes (new suite, leading-object scoring) across the model panel.
Predictors come from the original 82-way benchmark (accuracy, gamma_evidence, gamma_prior); outcomes come from different cases/prompts.
Leave-one-model-out RMSE; paired bootstrap over models (models resampled; the whole LOO is repeated within each resample)."""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
from common_pl import *
from scipy.stats import spearmanr
SC=OUT/'scoring_leading';mm=pd.read_csv(OUT/'model_metrics.csv').set_index('model')
df=pd.read_csv(SC/'new_inference_scored.csv');st=json.loads((SC/'new_inference_status.json').read_text());full=st['complete_models']
def flips(g,exp,a,b,key='top'):
    A=g[(g.experiment==exp)&(g.condition==a)].set_index('pair_id');B=g[(g.experiment==exp)&(g.condition==b)].set_index('pair_id');i=A.index.intersection(B.index);ok=A.loc[i].parseable&B.loc[i].parseable
    return (A.loc[i][ok][key]!=B.loc[i][ok][key]).astype(float),ok
rows=[]
for m in full:
    if m not in mm.index:continue
    g=df[df.model==m];r=dict(model=m,rank_correlation=mm.loc[m,'rank_correlation'],evidence_margin=mm.loc[m,'evidence_margin'],accuracy=mm.loc[m,'accuracy'],gamma_evidence=mm.loc[m,'gamma_evidence'],gamma_prior=mm.loc[m,'gamma_prior'],used_frac=mm.loc[m,'used_frac'])
    fv,_=flips(g,'source_reliability','valid_verified','clean');ff,_=flips(g,'source_reliability','fabricated','clean');fi,_=flips(g,'source_reliability','irrelevant','clean');fo,_=flips(g,'candidate_context','related_order0','related_order1')
    common=fv.index.intersection(ff.index)
    r['verified_minus_fabricated_flip']=(fv.loc[common]-ff.loc[common]).mean() if len(common) else np.nan
    r['fabricated_flip']=ff.mean();r['irrelevant_flip']=fi.mean();r['order_flip_related']=fo.mean();r['verified_flip']=fv.mean()
    st2=g[(g.experiment=='stateful_revision')&(g.condition=='confounder_first')];el=st2.parseable&st2.step2_parseable.eq(True)&(st2.top!=st2.target_id);r['recovery_after_error']=st2.loc[el,'step2_correct'].mean() if el.sum()>=10 else np.nan
    kp=g[(g.experiment=='knowledge_provided')];cc=g[(g.experiment=='candidate_context')&(g.condition=='related_order0')];j=kp.set_index('pair_id').index.intersection(cc.set_index('pair_id').index)
    r['profile_card_gain']=(kp.set_index('pair_id').loc[j].correct.mean()-cc.set_index('pair_id').loc[j].correct.mean())
    r['relation_probe_accuracy']=g[(g.experiment=='relation_knowledge')&g.parseable].correct.mean();r['four_candidate_accuracy']=cc.correct.mean()
    r['n_parseable_main']=g[(g.experiment=='candidate_context')].parseable.mean()
    rows.append(r)
P=pd.DataFrame(rows);P.to_csv(OUT/'incremental_v2_model_outcomes.csv',index=False)
def design(X,tr,te):
    mu=X[tr].mean(0);sd=X[tr].std(0);sd[sd<1e-9]=1;a=np.c_[np.ones(len(tr)),(X[tr]-mu)/sd];return a,np.c_[np.ones(len(te)),(X[te]-mu)/sd]
def loo_rmse(X,y):
    n=len(y);e=[]
    for i in range(n):
        tr=np.array([j for j in range(n) if j!=i]);a,t=design(X,tr,np.array([i]));beta=np.linalg.lstsq(a,y[tr],rcond=None)[0];e.append((t@beta)[0]-y[i])
    return float(np.sqrt(np.mean(np.square(e))))
specs={'accuracy':['accuracy'],'accuracy+gamma_e':['accuracy','gamma_evidence'],'accuracy+gamma_e+gamma_p':['accuracy','gamma_evidence','gamma_prior'],'gamma_e':['gamma_evidence'],'accuracy+gamma_p':['accuracy','gamma_prior'],'accuracy+rank_corr':['accuracy','rank_correlation'],'accuracy+evidence_margin':['accuracy','evidence_margin'],'rank_corr+evidence_margin (no accuracy)':['rank_correlation','evidence_margin']}
outs=['verified_minus_fabricated_flip','fabricated_flip','irrelevant_flip','order_flip_related','recovery_after_error','profile_card_gain','relation_probe_accuracy']
rng=np.random.default_rng(99);res=[]
for panel_name,cond in [('all complete models',lambda d:d.accuracy>=0),('accuracy>=5% and usable>=50%',lambda d:(d.accuracy>=.05)&(d.used_frac>=.5))]:
    base=P[cond(P)].reset_index(drop=True)
    for o in outs:
        d=base.dropna(subset=[o]).reset_index(drop=True);n=len(d)
        if n<8:continue
        y=d[o].to_numpy();r={s:loo_rmse(d[c].to_numpy(),y) for s,c in specs.items()}
        base_rmse=r['accuracy']
        boots=[]
        for _ in range(300):
            ix=rng.integers(0,n,n)
            if len(set(ix))<6:continue
            yy=y[ix];dd=d.iloc[ix]
            try:boots.append(loo_rmse(dd[specs['accuracy']].to_numpy(),yy)-loo_rmse(dd[specs['accuracy+gamma_e']].to_numpy(),yy))
            except Exception:pass
        boots=np.array(boots)
        pr=lambda x,z,w:np.corrcoef(x-np.polyval(np.polyfit(w,x,1),w),z-np.polyval(np.polyfit(w,z,1),w))[0,1]
        res.append(dict(panel=panel_name,outcome=o,n_models=n,outcome_sd=y.std(),rmse_mean_only=float(np.sqrt(np.mean((y-y.mean())**2))),**{f'rmse_{k}':v for k,v in r.items()},
            improvement_gamma_e_over_accuracy=base_rmse-r['accuracy+gamma_e'],boot_lo=np.quantile(boots,.025),boot_hi=np.quantile(boots,.975),
            spearman_outcome_accuracy=spearmanr(d.accuracy,y).statistic,spearman_outcome_gamma_e=spearmanr(d.gamma_evidence,y).statistic,spearman_outcome_gamma_p=spearmanr(d.gamma_prior,y).statistic,
            partial_r_gamma_e_given_acc=pr(d.gamma_evidence.to_numpy(),y,d.accuracy.to_numpy()),partial_r_gamma_p_given_acc=pr(d.gamma_prior.to_numpy(),y,d.accuracy.to_numpy())))
R=pd.DataFrame(res);R.to_csv(OUT/'incremental_v2.csv',index=False);print(R.round(3).to_string())
