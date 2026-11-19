#!/usr/bin/env python3
"""Replication on clinically sourced benchmarks: DDXPlus (synthetic generator driven by a proprietary medical knowledge base and rule-based expert system, expert-system differential) and an Orphanet-grounded differential benchmark (expert-curated phenotype frequencies, validated prevalence)."""
import sys,json,re
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
from common_pl import *
from scipy.special import logsumexp
from scipy.stats import spearmanr
RES=OUT/'clinical';OUTC=OUT/'clin';OUTC.mkdir(exist_ok=True);mm=pd.read_csv(OUT/'model_metrics.csv').set_index('model')
ddx={f['case']:f for f in json.load(open(ROOT/'rebuttal/data/clin_ddxplus_features.json'))};orph={f['case']:f for f in json.load(open(ROOT/'rebuttal/data/clin_orphanet_features.json'))}
def lead(raw):
    t=re.sub(r'^```(?:json)?\s*','',str(raw).strip(),flags=re.I)
    if not t.startswith('{'):return {}
    try:v,_=json.JSONDecoder().raw_decode(t);return v if isinstance(v,dict) else {}
    except Exception:return {}
rows=[]
for f in sorted(RES.glob('*__clinical.jsonl')):
    L=f.read_text().splitlines()
    if len(L)<1198:continue
    for l in L:
        r=json.loads(l);o=lead(r['raw_response']);allowed=r['candidate_ids'];rk=o.get('ranking',[]);pre=[]
        for x in (rk if isinstance(rk,list) else []):
            if not isinstance(x,str) or x not in allowed or x in pre:break
            pre.append(x)
        rows.append(dict(model=r['model'],suite=r['experiment'],item=r['item_id'],cond=r['condition'],pred=json.dumps([allowed.index(x) for x in pre]),parse=bool(pre),truth_idx=allowed.index(r['target_id']),conf=o.get('confidence')))
D=pd.DataFrame(rows);D.to_csv(OUTC/'clin_scored.csv',index=False);models=sorted(D.model.unique());print('models',len(models))
def feats(r):
    if r.suite=='clin_ddxplus':
        c=int(r['item'].split('/')[1]);f=ddx[str(c)] if str(c) in ddx else ddx[c];return np.array(f['logprior']),np.array(f['E']),f
    f=orph[r['item']];return np.array(f['logprior']),np.array(f['E']),f
def evaluate(suite):
    out=[];sub=D[D.suite==suite]
    for m in models:
        g=sub[sub.model==m];ch=[];recs=[]
        for _,r in g.iterrows():
            pred=json.loads(r.pred)
            if not pred:continue
            lp,E,f=feats(r);ch.append(Choice(pred,lp,E));recs.append((pred,lp,E,f,r))
        if len(ch)<50:continue
        fit=fit_plackett_luce(ch,n_starts=1);acc=np.mean([p[0]==r.truth_idx for p,_,_,_,r in recs]);kb=np.mean([p[0]==(int(np.argmax(f['post'])) if 'post' in f else int(np.argmax(E))) for p,lp,E,f,_ in recs])
        # prior-favouring errors
        w=[(p,lp,r.truth_idx) for p,lp,E,_,r in recs if p[0]!=r.truth_idx];more=np.mean([lp[p[0]]>lp[t] for p,lp,t in w]) if w else np.nan;ref=np.mean([(lp>lp[t]).sum()/(len(lp)-1) for p,lp,t in w]) if w else np.nan
        out.append(dict(model=m,suite=suite,parse=g.parse.mean(),n=len(recs),accuracy=acc,agree_with_expert_top1=kb,gamma_p=fit['gamma_prior'],gamma_e=fit['gamma_evidence'],excess_prior_error=more-ref,n_wrong=len(w),acc_82=mm.loc[m,'accuracy'] if m in mm.index else np.nan,gamma_e_82=mm.loc[m,'gamma_evidence'] if m in mm.index else np.nan,gamma_p_82=mm.loc[m,'gamma_prior'] if m in mm.index else np.nan,used82=mm.loc[m,'used_frac'] if m in mm.index else np.nan))
    return pd.DataFrame(out)
rng=np.random.default_rng(3)
def perm(x,y,n=10000):
    x=np.asarray(x,float);y=np.asarray(y,float);ok=~(np.isnan(x)|np.isnan(y));x,y=x[ok],y[ok]
    if len(x)<6:return np.nan,np.nan,len(x)
    o=spearmanr(x,y).statistic;pp=np.array([spearmanr(rng.permutation(x),y).statistic for _ in range(n)]);return o,(np.sum(abs(pp)>=abs(o))+1)/(n+1),len(x)
allres=[];tests=[]
for suite in ['clin_ddxplus','clin_orphanet']:
    R=evaluate(suite);allres.append(R)
    if R.empty:continue
    use=R[(R.parse>=.9)&(R.acc_82>=.05)&(R.used82>=.5)]
    print(suite,'usable',len(use),'of',len(R));print(R[['model','parse','accuracy','agree_with_expert_top1','gamma_p','gamma_e','excess_prior_error']].round(3).to_string())
    for name,a,b in [('gamma_e vs accuracy (same benchmark)','gamma_e','accuracy'),('gamma_e vs agreement with expert top-1','gamma_e','agree_with_expert_top1'),('accuracy vs agreement with expert top-1','accuracy','agree_with_expert_top1'),('gamma_p vs prior-favouring error excess','gamma_p','excess_prior_error'),('accuracy vs prior-favouring error excess','accuracy','excess_prior_error'),('gamma_e (here) vs gamma_e (82-way synthetic)','gamma_e','gamma_e_82'),('accuracy (here) vs accuracy (82-way synthetic)','accuracy','acc_82'),('gamma_p (here) vs gamma_p (82-way synthetic)','gamma_p','gamma_p_82')]:
        r,p,n=perm(use[a],use[b]);tests.append(dict(suite=suite,test=name,spearman=r,perm_p=p,n_models=n))
pd.concat(allres).to_csv(OUTC/'clin_model_metrics.csv',index=False);T=pd.DataFrame(tests);T.to_csv(OUTC/'clin_tests.csv',index=False);print(T.round(3).to_string())
# ---- conflict-slice prediction on DDXPlus: fit gamma on expert-consistent cases, predict choices on expert-inconsistent cases
sub=D[D.suite=='clin_ddxplus'];rows=[]
for m in models:
    g=sub[(sub.model==m)&sub.parse];cons=[];inc=[]
    for _,r in g.iterrows():
        lp,E,f=feats(r);pred=json.loads(r.pred);item=(pred,lp,E,r.truth_idx,f)
        (cons if f['truth']==f['kb_top'] else inc).append(item)
    if len(cons)<80 or len(inc)<30:continue
    fit=fit_plackett_luce([Choice(p,lp,E) for p,lp,E,_,_ in cons],n_starts=1);gp,ge=fit['gamma_prior'],fit['gamma_evidence']
    def ll1(pred,lp,E,a,b):
        u=a*lp+b*E;return u[pred[0]]-logsumexp(u)
    acc_c=np.mean([p[0]==t for p,_,_,t,_ in cons])
    def ll_acc(pred,lp,E,t):  # accuracy-only baseline: truth with prob acc_c (from consistent slice), others uniform
        n=len(lp);return np.log(acc_c) if pred[0]==t else np.log((1-acc_c)/(n-1))
    L={'gamma_from_consistent_slice':np.mean([ll1(p,lp,E,gp,ge) for p,lp,E,t,_ in inc]),'accuracy_only_(consistent_slice)':np.mean([ll_acc(p,lp,E,t) for p,lp,E,t,_ in inc]),'expert_posterior_(gamma=1,1)':np.mean([ll1(p,lp,E,1,1) for p,lp,E,t,_ in inc]),'prior_only_(gamma_p=1)':np.mean([ll1(p,lp,E,1,0) for p,lp,E,t,_ in inc]),'evidence_only_(gamma_e=1)':np.mean([ll1(p,lp,E,0,1) for p,lp,E,t,_ in inc]),'uniform':np.mean([-np.log(len(lp)) for p,lp,E,t,_ in inc]),'gamma_e_only_fitted':np.mean([ll1(p,lp,E,0,ge) for p,lp,E,t,_ in inc])}
    rows.append(dict(model=m,n_consistent=len(cons),n_conflict=len(inc),acc_consistent=acc_c,acc_conflict=np.mean([p[0]==t for p,_,_,t,_ in inc]),agree_expert_conflict=np.mean([p[0]==int(np.argmax(f['post'])) for p,_,E,_,f in inc]),gamma_p=gp,gamma_e=ge,**L))
C=pd.DataFrame(rows);C.to_csv(OUTC/'clin_conflict_slice.csv',index=False)
if len(C):
    print(C.round(3).to_string());cols=[c for c in C.columns if c not in['model','n_consistent','n_conflict','acc_consistent','acc_conflict','agree_expert_conflict','gamma_p','gamma_e']];base='gamma_from_consistent_slice';S=[]
    for c in cols:
        if c==base:continue
        d=(C[base]-C[c]).to_numpy();bs=np.mean(d[rng.integers(0,len(d),(5000,len(d)))],axis=1);S.append(dict(competitor=c,mean_ll_gain_of_gamma=d.mean(),lo=np.quantile(bs,.025),hi=np.quantile(bs,.975),models_better=int((d>0).sum()),n_models=len(d)))
    S=pd.DataFrame(S);S.to_csv(OUTC/'clin_conflict_slice_summary.csv',index=False);print(S.round(3).to_string())
