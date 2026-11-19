#!/usr/bin/env python3
"""Study 2 analyses (see docs/STUDY2_PREREGISTRATION.md). Usage: python study2_analyze.py [sct grounded age ledger cot cupcase ehr mm]. Writes results/study2A/*.csv"""
import json,re,sys,warnings
from pathlib import Path
import numpy as np,pandas as pd
import statsmodels.api as sm
from scipy.stats import spearmanr
warnings.filterwarnings('ignore')
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'rebuttal/results';S2=OUT/'study2';A=OUT/'study2A';A.mkdir(exist_ok=True)
mm=pd.read_csv(OUT/'model_metrics.csv').set_index('model');params=pd.read_csv(OUT/'model_params.csv').set_index('model').params_b
rng=np.random.default_rng(11)
def files(suffix):return sorted(S2.glob(f'*__{suffix}.jsonl'))
def load(suffix):
    rows=[]
    for f in files(suffix):
        for l in f.read_text().splitlines():
            if l.strip():rows.append(json.loads(l))
    return rows
def lead(raw):
    t=re.sub(r'^```(?:json)?\s*','',str(raw).strip(),flags=re.I)
    if not t.startswith('{'):return {}
    try:v,_=json.JSONDecoder().raw_decode(t);return v if isinstance(v,dict) else {}
    except Exception:return {}
def last_json(raw):
    best={}
    for m in re.finditer(r'\{[^{}]*\}',str(raw)):
        try:
            v=json.loads(m.group(0))
            if isinstance(v,dict) and ('ranking' in v or 'diagnoses' in v):best=v
        except Exception:pass
    return best
def first_choice(obj,allowed):
    rk=obj.get('ranking',[]);return rk[0] if isinstance(rk,list) and rk and isinstance(rk[0],str) and rk[0] in allowed else None
def usable(m,parse,thr=.9):
    return m in mm.index and mm.loc[m,'accuracy']>=.05 and mm.loc[m,'used_frac']>=.5 and parse>=thr
def boot(v,B=3000):
    v=np.asarray(v,float);v=v[~np.isnan(v)]
    if len(v)<5:return (np.nan,np.nan)
    b=np.mean(v[rng.integers(0,len(v),(B,len(v)))],axis=1);return (np.quantile(b,.025),np.quantile(b,.975))
def perm(x,y,n=10000):
    x=np.asarray(x,float);y=np.asarray(y,float);ok=~(np.isnan(x)|np.isnan(y));x,y=x[ok],y[ok]
    if len(x)<6:return np.nan,np.nan,len(x)
    o=spearmanr(x,y).statistic;pp=np.array([spearmanr(rng.permutation(x),y).statistic for _ in range(n)]);return round(o,3),round((np.sum(abs(pp)>=abs(o))+1)/(n+1),4),len(x)
def sct():
    R=load('sct');rows=[]
    for r in R:
        m=re.search(r'Rating:\s*\**\s*([+-]?\s*[0-2])',r['raw_response']);rate=int(m.group(1).replace(' ','')) if m else None
        ps=r['panel_score'];cnt=r['panel_counts'];idx=None if rate is None else rate+2
        rows.append(dict(model=r['model'],q=r['pair_id'],source=r['source'],parsed=rate is not None,rating=rate,score=ps[idx] if idx is not None else np.nan,modal=int(np.argmax(cnt))-2,extreme=abs(rate)==2 if rate is not None else np.nan))
    D=pd.DataFrame(rows);D.to_csv(A/'sct_scored.csv',index=False);res=[]
    for m,g in D.groupby('model'):
        lo,hi=boot(g.score.fillna(0).values);sc=g.score.fillna(0)
        res.append(dict(model=m,parse=g.parsed.mean(),sct_score=sc.mean(),lo=lo,hi=hi,sct_open_medical=sc[g.source=='Open Medical SCT'].mean(),sct_adelaide=sc[g.source=='Adelaide SCT'].mean(),sct_score_parsed_only=g.score.mean(),direction_agree=((np.sign(g.rating)==np.sign(g.modal))[g.parsed]).mean(),extreme_rate=g.extreme.mean(),params_b=params.get(m,np.nan),acc82=mm.loc[m,'accuracy'] if m in mm.index else np.nan,gamma_e=mm.loc[m,'gamma_evidence'] if m in mm.index else np.nan,gamma_p=mm.loc[m,'gamma_prior'] if m in mm.index else np.nan,used82=mm.loc[m,'used_frac'] if m in mm.index else np.nan))
    R2=pd.DataFrame(res);R2.to_csv(A/'sct_by_model.csv',index=False);print(R2.round(3).to_string());U=R2[(R2.parse>=.9)&(R2.acc82>=.05)&(R2.used82>=.5)];t=[]
    for a,b in [('acc82','sct_score'),('gamma_e','sct_score'),('gamma_p','sct_score'),('params_b','sct_score')]:r,p,n=perm(U[a],U[b]);t.append(dict(x=a,y=b,spearman=r,perm_p=p,n_models=n))
    pd.DataFrame(t).to_csv(A/'sct_tests.csv',index=False);print(pd.DataFrame(t).to_string())
def grounded():
    des=pd.read_csv(ROOT/'rebuttal/data/grounded_design.csv').set_index('item_id');R=load('grounded');rows=[]
    for r in R:
        d=des.loc[r['item_id']];top=first_choice(lead(r['raw_response']),r['candidate_ids']);rows.append(dict(model=r['model'],item=r['item_id'],kind=d.kind,pair=d.pair,level=d.level,r_index=d.r_index,order=d.order,age=d.age if d.kind=='age' else None,log_lr=d.log_lr,lpo=d.log_prior_odds,lpost=d.log_post_odds,ref_A=d.ref_A,chose_A=float(top=='P0') if top else np.nan))
    D=pd.DataFrame(rows);D.to_csv(A/'grounded_scored.csv',index=False);res=[];ages=[]
    for m,g in D.groupby('model'):
        s=g[g.kind=='grounded'];parse=s.chose_A.notna().mean();v=s[s.chose_A.notna()&s.ref_A.notna()].copy();r={'model':m,'parse':parse,'n':len(v)}
        if len(v)>=200:
            v['ok']=(v.chose_A==v.ref_A.astype(float)).astype(float);r['agree_reference']=v.ok.mean();c=v[(np.sign(v.lpo)!=np.sign(v.log_lr))&(v.lpo!=0)];r['agree_reference_conflict']=c.ok.mean() if len(c) else np.nan;r['agree_L1']=v[v.level==1].ok.mean();r['agree_L4']=v[v.level==4].ok.mean()
            try:
                f=sm.Logit(v.chose_A,v[['lpo','log_lr']]).fit(disp=0,maxiter=300);r['beta_prior']=f.params['lpo'];r['beta_evidence']=f.params['log_lr'];r['weight_ratio']=f.params['lpo']/f.params['log_lr'] if f.params['log_lr']>0 else np.nan
            except Exception:pass
        res.append(r);a=g[g.kind=='age']
        if len(a):
            pv=a.pivot_table(index=['pair','order'],columns='age',values='chose_A',dropna=False).reindex(columns=['child','adult','none']);pv=pv.dropna(subset=['child','adult'])
            if len(pv):ages.append(dict(model=m,parse_age=a.chose_A.notna().mean(),p_pediatric_if_child=pv.child.mean(),p_pediatric_if_adult=pv.adult.mean(),p_pediatric_if_none=pv['none'].mean(),age_sensitivity=pv.child.mean()-pv.adult.mean(),agree_age_reference=((pv.child==1).mean()+(pv.adult==0).mean())/2,n=len(pv)))
    R2=pd.DataFrame(res);R2['acc82']=[mm.loc[m,'accuracy'] if m in mm.index else np.nan for m in R2.model];R2['gamma_e']=[mm.loc[m,'gamma_evidence'] if m in mm.index else np.nan for m in R2.model];R2['gamma_p']=[mm.loc[m,'gamma_prior'] if m in mm.index else np.nan for m in R2.model];R2['used82']=[mm.loc[m,'used_frac'] if m in mm.index else np.nan for m in R2.model];R2['params_b']=[params.get(m,np.nan) for m in R2.model]
    R2.to_csv(A/'grounded_by_model.csv',index=False);print(R2.round(3).to_string());pd.DataFrame(ages).to_csv(A/'age_prior_by_model.csv',index=False);print(pd.DataFrame(ages).round(3).to_string())
    # P4: explicit-number (aliased) pilot agreement vs grounded agreement
    try:
        X=pd.read_csv(OUT/'crossoverA/crossover_by_model.csv')[['model','decision_accuracy_vs_reference','conflict_cells_accuracy']];J=R2.merge(X,on='model');J=J[(J.parse>=.9)];print('P4 aliased vs grounded agreement',perm(J.decision_accuracy_vs_reference,J.agree_reference),'; params vs grounded agreement',perm(J.params_b,J.agree_reference))
    except Exception as e:print('P4 skipped',e)
def handling():
    des=pd.read_csv(ROOT/'rebuttal/data/crossover_design.csv').set_index('item_id');rows=[]
    plain=pd.read_csv(OUT/'crossoverA/crossover_scored.csv');plain=plain[plain.kind.isin(['dup'])&(plain.order==0)];
    for _,r in plain.iterrows():rows.append(dict(model=r.model,variant='plain',tag=r.tag,form=r.form,chose_A=r.chose_A,ref=r.ref_A))
    for suf,var in [('ledger','ledger'),('cot','cot')]:
        for r in load(suf):
            iid=r['item_id'].replace('xoverLedger','xover').replace('xoverCoT','xover')
            if iid not in des.index:continue
            d=des.loc[iid]
            if d.kind!='dup':continue
            obj=lead(r['raw_response']) if var=='ledger' else last_json(r['raw_response']);top=first_choice(obj,r['candidate_ids']);rows.append(dict(model=r['model'],variant=var,tag=d.tag,form=d.form,chose_A=float(top=='K1') if top else np.nan,ref=d.ref_choice_A))
    D=pd.DataFrame(rows);D=D[D.chose_A.notna()&D.ref.notna()].copy();D['ok']=(D.chose_A==D.ref.astype(float)).astype(float);T=D.groupby(['model','variant','form']).ok.mean().unstack('form');T['handling']=T[['duplicate','independent']].mean(axis=1);T=T.reset_index();T.to_csv(A/'handling_by_model.csv',index=False);print(T.pivot(index='model',columns='variant',values='handling').round(3).to_string())
    P=T.pivot(index='model',columns='variant',values='handling').dropna(subset=['plain']);
    for v in ['ledger','cot']:
        if v in P:d=(P[v]-P.plain).dropna();print(v,'minus plain handling accuracy: mean',round(d.mean(),3),'models improved',int((d>0).sum()),'of',len(d))
def cupcase():
    R=load('cupcase');rows=[]
    for r in R:
        if r['experiment']=='cupcase_closed':
            top=first_choice(lead(r['raw_response']),r['candidate_ids']);rows.append(dict(model=r['model'],kind='closed',case=r['pair_id'],ok=float(top==r['target_id']) if top else np.nan,parsed=top is not None))
        else:
            a=lead(r['raw_response']).get('diagnoses',[]);a=[x for x in a if isinstance(x,str)] if isinstance(a,list) else [];rows.append(dict(model=r['model'],kind='open',case=r['pair_id'],answers=' | '.join(a[:3]),target=r['target'],parsed=bool(a)))
    D=pd.DataFrame(rows);D.to_csv(A/'cupcase_scored.csv',index=False);c=D[D.kind=='closed'].groupby('model').agg(parse=('parsed','mean'),closed_acc_parsed=('ok','mean'));print(c.round(3).to_string())
def ehr():
    R=load('ehr');des=pd.read_csv(ROOT/'rebuttal/data/ehr_design.csv').set_index('item_id');rows=[]
    for r in R:
        if r['experiment']!='ehr_closed':continue
        d=des.loc[r['item_id']];top=first_choice(lead(r['raw_response']),r['candidate_ids']);rows.append(dict(model=r['model'],case=d.case,subject=d.subject,cond=d.cond,ok=float(top==r['target_id']) if top else np.nan))
    D=pd.DataFrame(rows);D.to_csv(A/'ehr_scored.csv',index=False);res=[]
    for m,g in D.groupby('model'):
        p=g.pivot_table(index=['subject','case'],columns='cond',values='ok').dropna()
        if len(p)<50:continue
        subj=p.index.get_level_values(0).values;us=np.unique(subj);d1=(p.true_history-p.no_history).values;d2=(p.true_history-p.mismatched_history).values;b1=[];b2=[]
        for _ in range(2000):
            pick=rng.choice(us,len(us));idx=np.concatenate([np.flatnonzero(subj==u) for u in pick]);b1.append(d1[idx].mean());b2.append(d2[idx].mean())
        res.append(dict(model=m,n=len(p),acc_no_history=p.no_history.mean(),acc_true_history=p.true_history.mean(),acc_mismatched=p.mismatched_history.mean(),gain_true_vs_none=d1.mean(),gain_lo=np.quantile(b1,.025),gain_hi=np.quantile(b1,.975),gain_true_vs_mismatched=d2.mean(),gm_lo=np.quantile(b2,.025),gm_hi=np.quantile(b2,.975)))
    R2=pd.DataFrame(res);R2.to_csv(A/'ehr_by_model.csv',index=False);print(R2.round(3).to_string())
def mm_():
    rows=[]
    for suf in ['mmvlm','mmtext']:
        for r in load(suf):
            if r['experiment']!='mm_closed':continue
            top=first_choice(lead(r['raw_response']),r['candidate_ids']);rows.append(dict(model=r['model'],case=r['pair_id'],cond=r['condition'],ok=float(top==r['target_id']) if top else np.nan))
    D=pd.DataFrame(rows).drop_duplicates(['model','case','cond']);D.to_csv(A/'mm_closed_scored.csv',index=False);res=[]
    for m,g in D.groupby('model'):
        p=g.pivot_table(index='case',columns='cond',values='ok');r=dict(model=m,n=len(p));
        for c in p.columns:r['acc_'+c]=p[c].mean()
        if 'image_text' in p and 'text_only' in p:
            q=p[['image_text','text_only']].dropna();d=(q.image_text-q.text_only).values;lo,hi=boot(d);r.update(gain_image_vs_text=d.mean(),gi_lo=lo,gi_hi=hi)
        if 'image_text' in p and 'mismatched_image' in p:
            q=p[['image_text','mismatched_image']].dropna();d=(q.image_text-q.mismatched_image).values;lo,hi=boot(d);r.update(gain_true_vs_mismatched_image=d.mean(),gm_lo=lo,gm_hi=hi)
        res.append(r)
    R2=pd.DataFrame(res);R2.to_csv(A/'mm_by_model.csv',index=False);print(R2.round(3).to_string())
if __name__=='__main__':
    which=sys.argv[1:] or ['sct','grounded','ledger','cot','cupcase','ehr','mm']
    for w in which:
        print('\n=====',w)
        try:{'sct':sct,'grounded':grounded,'age':grounded,'ledger':handling,'cot':handling,'cupcase':cupcase,'ehr':ehr,'mm':mm_}[w]()
        except Exception as e:print('FAILED',w,repr(e))
