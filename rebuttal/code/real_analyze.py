#!/usr/bin/env python3
"""Analyses on 300 real published case reports (MedCaseReasoning test split). Leading-object scoring, valid-prefix rankings.
Evidence = MedCPT similarity of case text to a PubMed literature profile of each candidate; prior = log PubMed publication count (both z-scored); NOT clinical ground truth."""
import json,re,sys
from pathlib import Path
import numpy as np,pandas as pd
from scipy.stats import spearmanr
ROOT=Path(__file__).resolve().parents[2];RD=ROOT/'rebuttal/real_data';OUT=ROOT/'rebuttal/results/real';OUT.mkdir(parents=True,exist_ok=True)
sys.path.insert(0,str(ROOT/'code_final_v2/analysis'));from plackett_luce import Choice,fit_plackett_luce
cases=pd.read_csv(RD/'real_cases.csv').set_index('case_id');feat={r['case_id']:r for r in json.load(open(RD/'real_features.json'))}
mm=pd.read_csv(ROOT/'rebuttal/results/model_metrics.csv').set_index('model')
def lead(raw):
    t=re.sub(r'^```(?:json)?\s*','',str(raw).strip(),flags=re.I)
    if not t.startswith('{'):return {}
    try:v,_=json.JSONDecoder().raw_decode(t);return v if isinstance(v,dict) else {}
    except Exception:return {}
def prefix(obj,allowed):
    rr=obj.get('ranking',[]);out=[]
    for x in (rr if isinstance(rr,list) else []):
        if not isinstance(x,str) or x not in allowed or x in out:break
        out.append(x)
    return out
def conf(obj):
    c=obj.get('confidence',np.nan);return float(c) if isinstance(c,(int,float)) and 0<=c<=1 else np.nan
# annotation (comorbidity) from Qwen2.5-7B
ann={}
af=RD/'results/qwen2.5-7b__real_annot.jsonl'
if af.exists():
    for l in af.read_text().splitlines():
        r=json.loads(l);o=lead(r['raw_response']);p=o.get('preexisting_conditions',[]);ann[r['pair_id']]=len([x for x in p if isinstance(x,str)]) if isinstance(p,list) else np.nan
cases['n_preexisting_llm']=pd.Series(ann);cases['multimorbid_llm']=cases.n_preexisting_llm>=2
rows=[]
for f in sorted((RD/'results').glob('*__real.jsonl')):
    if len(f.read_text().splitlines())<2398:continue  # only fully completed models
    for l in f.read_text().splitlines():
        r=json.loads(l);ex=r['experiment'];cid=r['pair_id'];o=lead(r['raw_response']);allowed=r['candidate_ids']+(['OTHER','ABSTAIN'] if ex=='real_missing' else [])
        if ex=='real_open':
            a=o.get('diagnoses',[]);a=[x for x in a if isinstance(x,str)] if isinstance(a,list) else [];rows.append(dict(model=r['model'],exp=ex,cond=r['condition'],case=cid,top=a[0] if a else None,answers=' | '.join(a[:3]),parse=bool(a),conf=conf(o)));continue
        pre=prefix(o,allowed);d=dict(model=r['model'],exp=ex,cond=r['condition'],case=cid,rank=json.dumps(pre),top=pre[0] if pre else None,parse=bool(pre),conf=conf(o),finish=r['finish_reason'])
        d['correct']=float(d['top']=='R0') if ex!='real_missing' or r['condition']=='present' else np.nan
        if ex=='real_revise':
            o2=lead(r.get('step2_response',''));p2=prefix(o2,allowed);d['top2']=p2[0] if p2 else None;d['parse2']=bool(p2);d['correct2']=float(bool(p2) and p2[0]=='R0');d['rank2']=json.dumps(p2);d['conf2']=conf(o2)
        rows.append(d)
D=pd.DataFrame(rows);D.to_csv(OUT/'real_scored.csv',index=False)
models=sorted(D.model.unique());res=[]
# z-scored features
allE=np.array([e for f in feat.values() for e in f['E']]);allP=np.array([p for f in feat.values() for p in f['logP']]);mE,sE,mP,sP=allE.mean(),allE.std(),allP.mean(),allP.std()
def choices(sub):
    cs=[]
    for _,r in sub.iterrows():
        rk=json.loads(r['rank']);
        if not rk:continue
        f=feat[r['case']];ids=f['candidate_ids'];cs.append(Choice([ids.index(x) for x in rk],(np.array(f['logP'])-mP)/sP,(np.array(f['E'])-mE)/sE))
    return cs
def fit(sub):
    cs=choices(sub)
    if len(cs)<40:return (np.nan,np.nan,len(cs))
    r=fit_plackett_luce(cs,n_starts=1);return (r['gamma_evidence'],r['gamma_prior'],len(cs))
def boot_fit(sub,B=200,seed=1):
    rng=np.random.default_rng(seed);cs=choices(sub);vals=[]
    if len(cs)<40:return (np.nan,np.nan)
    for _ in range(B):
        ix=rng.integers(0,len(cs),len(cs));vals.append(fit_plackett_luce([cs[i] for i in ix],n_starts=1)['gamma_evidence'])
    return tuple(np.quantile(vals,[.025,.975]))
closed=D[(D.exp=='real_closed')]
for m in models:
    g=closed[closed.model==m];r=dict(model=m,accuracy_82way=mm.loc[m,'accuracy'] if m in mm.index else np.nan,gamma_e_82way=mm.loc[m,'gamma_evidence'] if m in mm.index else np.nan,gamma_p_82way=mm.loc[m,'gamma_prior'] if m in mm.index else np.nan)
    r['parse_closed']=g.parse.mean();r['acc_closed']=g.correct.mean();r['acc_closed_parseable']=g[g.parse].correct.mean()
    ge,gp,n=fit(g[g.parse]);lo,hi=boot_fit(g[g.parse]);r.update(real_gamma_e=ge,real_gamma_p=gp,real_gamma_e_lo=lo,real_gamma_e_hi=hi,n_rankings=n)
    # position effect
    o0=g[g.cond=='order0'].set_index('case');o1=g[g.cond=='order1'].set_index('case');ix=o0.index.intersection(o1.index);ok=o0.loc[ix].parse&o1.loc[ix].parse
    r['order_flip']=(o0.loc[ix][ok].top!=o1.loc[ix][ok].top).mean()
    # missing true
    ms=D[(D.model==m)&(D.exp=='real_missing')];mi=ms[ms.cond=='missing'];pr=ms[ms.cond=='present']
    r['parse_missing']=mi.parse.mean();r['abstain_when_true_missing']=mi.top.isin(['OTHER','ABSTAIN']).mean();r['abstain_when_true_present']=pr.top.isin(['OTHER','ABSTAIN']).mean();r['acc_present_with_other_option']=pr.correct.mean()
    r['forced_wrong_when_missing']=(mi.parse&~mi.top.isin(['OTHER','ABSTAIN'])).mean()
    # ablation
    a=D[(D.model==m)&(D.exp=='real_ablate')];base=closed[(closed.model==m)&(closed.cond=='order0')].set_index('case')
    def flips(c):
        x=a[a.cond==c].set_index('case');ix=x.index.intersection(base.index);ok=x.loc[ix].parse&base.loc[ix].parse;return (x.loc[ix][ok].top!=base.loc[ix][ok].top).astype(float),ix[ok.values]
    fk,ik=flips('key_removed');fr,ir=flips('random_removed');r['flip_key_removed']=fk.mean();r['flip_random_removed']=fr.mean();r['evidence_dependence_index']=fk.mean()-fr.mean()
    cor=base[(base.correct==1)].index;kk=fk[fk.index.isin(cor)];rr=fr[fr.index.isin(cor)];r['correct_but_insensitive_to_key_removal']=1-kk.mean() if len(kk) else np.nan;r['correct_flip_key']=kk.mean() if len(kk) else np.nan;r['correct_flip_random']=rr.mean() if len(rr) else np.nan
    # revision
    rv=D[(D.model==m)&(D.exp=='real_revise')];ok=rv.parse&rv.parse2.eq(True);rv=rv[ok];r['revise_acc_early']=rv.correct.mean();r['revise_acc_full']=rv.correct2.mean();w=rv[rv.correct==0];r['recovery_given_early_wrong']=w.correct2.mean() if len(w)>=10 else np.nan;rt=rv[rv.correct==1];r['harm_given_early_right']=(1-rt.correct2).mean() if len(rt)>=10 else np.nan;r['revise_n']=len(rv)
    res.append(r)
R=pd.DataFrame(res);R.to_csv(OUT/'real_model_summary.csv',index=False);pd.set_option('display.width',250);print(R.round(3).to_string())
# strata
st=[]
for m in models:
    g=closed[(closed.model==m)&closed.parse].merge(cases[['multimorbid_llm','n_preexisting_llm','long_timeline','age_group','multimorbid','n_sentences']],left_on='case',right_index=True)
    for name,col in [('multimorbid_llm','multimorbid_llm'),('long_timeline','long_timeline'),('age_group','age_group'),('multimorbid_regex','multimorbid')]:
        for v,gg in g.groupby(col):
            ge,gp,n=fit(gg);st.append(dict(model=m,stratum=name,value=str(v),n_cases=gg.case.nunique(),n_rankings=n,accuracy=gg.correct.mean(),gamma_e=ge,gamma_p=gp))
pd.DataFrame(st).to_csv(OUT/'real_strata.csv',index=False)
# cross-dataset validity
ok=R[(R.accuracy_82way>=.05)&(R.parse_closed>=.8)]
def rho(a,b):
    return spearmanr(ok[a],ok[b]).statistic if len(ok)>=6 else np.nan
print('\nmodels in validity panel:',len(ok));
for a,b in [('gamma_e_82way','real_gamma_e'),('accuracy_82way','real_gamma_e'),('acc_closed','real_gamma_e'),('gamma_p_82way','real_gamma_p'),('gamma_e_82way','evidence_dependence_index'),('acc_closed','evidence_dependence_index'),('gamma_p_82way','correct_but_insensitive_to_key_removal'),('acc_closed','correct_but_insensitive_to_key_removal'),('gamma_e_82way','recovery_given_early_wrong'),('acc_closed','recovery_given_early_wrong')]:
    print(a,'vs',b,round(rho(a,b),3))
