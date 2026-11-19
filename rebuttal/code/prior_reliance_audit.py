#!/usr/bin/env python3
"""Does gamma_prior / gamma_evidence expose anything accuracy hides? (o8eS: counterexamples; practical necessity of the decomposition.)
(1) rank discordance between accuracy and gamma_evidence; (2) rare-vs-common accuracy gap vs gamma_prior controlling for accuracy;
(3) prior-driven errors: among wrong first choices, share that are more prevalent than the target (chance-adjusted)."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
from common_pl import *
from scipy.stats import spearmanr,rankdata
b=Bench();mm=pd.read_csv(OUT/'model_metrics.csv').set_index('model');ms=[m for m in b.models() if mm.loc[m,'accuracy']>=.05]
rare=b.v.is_rare.to_numpy().astype(bool);tgt=np.array([b.di[t] for t in b.target]);prev=np.exp(b.P);rows=[]
for m in ms:
    pc=b.predictions(m);first=np.array([c[0] if c else -1 for c in pc]);ok=first>=0;corr=(first==tgt)
    a_rare=corr[rare&ok].mean();a_com=corr[~rare&ok].mean()
    wrong=ok&~corr;more=(prev[first[wrong]]>prev[tgt[wrong]]).mean()
    # chance reference: probability that a uniformly random non-target candidate is more prevalent than the target
    ref=np.mean([ (prev>prev[tgt[i]]).sum()/81 for i in np.flatnonzero(wrong)])
    rows.append(dict(model=m,accuracy=mm.loc[m,'accuracy'],gamma_evidence=mm.loc[m,'gamma_evidence'],gamma_prior=mm.loc[m,'gamma_prior'],used_frac=mm.loc[m,'used_frac'],acc_rare=a_rare,acc_common=a_com,rare_minus_common=a_rare-a_com,
        wrong_more_prevalent_than_target=more,chance_more_prevalent=ref,excess_prior_error=more-ref,n_wrong=int(wrong.sum())))
d=pd.DataFrame(rows);d['rank_acc']=rankdata(-d.accuracy);d['rank_gamma_e']=rankdata(-d.gamma_evidence);d['rank_gap']=d.rank_acc-d.rank_gamma_e;d.to_csv(OUT/'prior_reliance_audit.csv',index=False)
print(d.sort_values('rank_gap').round(3)[['model','accuracy','gamma_evidence','gamma_prior','rank_acc','rank_gamma_e','rank_gap','rare_minus_common','excess_prior_error']].to_string())
def partial(x,y,z):
    rx=x-np.polyval(np.polyfit(z,x,1),z);ry=y-np.polyval(np.polyfit(z,y,1),z);return np.corrcoef(rx,ry)[0,1]
for col in ['rare_minus_common','excess_prior_error']:
    print(col,'rho(gamma_prior)=',round(spearmanr(d.gamma_prior,d[col]).statistic,3),'partial r | accuracy=',round(partial(d.gamma_prior.to_numpy(),d[col].to_numpy(),d.accuracy.to_numpy()),3),
      '| rho(gamma_evidence)=',round(spearmanr(d.gamma_evidence,d[col]).statistic,3),'| rho(accuracy)=',round(spearmanr(d.accuracy,d[col]).statistic,3))
rng=np.random.default_rng(11);out=[]
for col,pred in [('excess_prior_error','gamma_prior'),('excess_prior_error','accuracy'),('excess_prior_error','gamma_evidence'),('rare_minus_common','gamma_prior')]:
    obs=spearmanr(d[pred],d[col]).statistic;perm=np.array([spearmanr(rng.permutation(d[pred].to_numpy()),d[col]).statistic for _ in range(5000)])
    out.append(dict(outcome=col,predictor=pred,n_models=len(d),spearman=obs,perm_p_two_sided=(np.sum(np.abs(perm)>=abs(obs))+1)/(5001),
        partial_r_given_accuracy=partial(d[pred].to_numpy(),d[col].to_numpy(),d.accuracy.to_numpy()) if pred!='accuracy' else np.nan))
pd.DataFrame(out).to_csv(OUT/'prior_reliance_tests.csv',index=False);print(pd.DataFrame(out).round(4).to_string())
