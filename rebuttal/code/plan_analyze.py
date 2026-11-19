#!/usr/bin/env python3
"""Plan items C4 (behavioural IIA), D (retraction/correction selectivity), B3/B4 (aliased-name knowledge controls).
Predeclared: IIA failure = |shift in P(target | target or confounder chosen)| >= 10 percentage points between the pair-only context and a context with a third candidate."""
import json,re,sys,warnings
warnings.filterwarnings('ignore')
from pathlib import Path
import numpy as np,pandas as pd
from scipy.stats import spearmanr
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'rebuttal/results';RES=OUT/'plan';OUTP=OUT/'planA';OUTP.mkdir(exist_ok=True)
des=pd.read_csv(ROOT/'rebuttal/data/iia_v2_design.csv').set_index('item_id');mm=pd.read_csv(OUT/'model_metrics.csv').set_index('model')
def lead(raw):
    t=re.sub(r'^```(?:json)?\s*','',str(raw).strip(),flags=re.I)
    if not t.startswith('{'):return {}
    try:v,_=json.JSONDecoder().raw_decode(t);return v if isinstance(v,dict) else {}
    except Exception:return {}
def first(raw,allowed):
    rk=lead(raw).get('ranking',[]);
    return rk[0] if isinstance(rk,list) and rk and isinstance(rk[0],str) and rk[0] in allowed else None
rows=[]
for f in sorted(RES.glob('*__plan.jsonl')):
    L=f.read_text().splitlines()
    if len(L)<1484:continue
    for l in L:
        r=json.loads(l);a=r['candidate_ids'];top=first(r['raw_response'],a);d=dict(model=r['model'],exp=r['experiment'],item=r['item_id'],pair=r['pair_id'],cond=r['condition'],top=top,target=r.get('target_id'),conf=r.get('confounder_id'),correct=float(top==r.get('target_id')) if top else np.nan)
        if r['experiment']=='correction':d['top2']=first(r.get('step2_response',''),a)
        rows.append(d)
D=pd.DataFrame(rows);D.to_csv(OUTP/'plan_scored.csv',index=False);models=sorted(D.model.unique());print('models',len(models))
usable=[m for m in models if m in mm.index and mm.loc[m,'accuracy']>=.05 and mm.loc[m,'used_frac']>=.5]
rng=np.random.default_rng(5)
# ---- IIA v2
I=D[D.exp=='iia_v2'].copy();I['ctx']=I.item.map(lambda x:x.split('/')[2]);I['para']=I.item.map(lambda x:x.split('/')[3]);I['order']=I.item.map(lambda x:x.split('/')[4])
res=[];pair_tab=[]
for m in models:
    g=I[I.model==m];parse=g.top.notna().mean()
    def pt(ctx):
        s=g[g.ctx==ctx];s=s[s.top.isin(s.target.unique().tolist()+s.conf.unique().tolist())&((s.top==s.target)|(s.top==s.conf))];return s.groupby('pair').correct.mean()
    base=pt('base')
    for ctx in ['unrelated','near_subtype','common']:
        c=pt(ctx);idx=base.index.intersection(c.index)
        if len(idx)<30:continue
        d=(c.loc[idx]-base.loc[idx]).to_numpy();bs=np.mean(d[rng.integers(0,len(d),(5000,len(d)))],axis=1)
        share3=np.mean([(g[(g.ctx==ctx)&(g.pair==p)].top.notna()&~g[(g.ctx==ctx)&(g.pair==p)].top.isin(list(g[g.pair==p].target.unique())+list(g[g.pair==p].conf.unique()))).mean() for p in idx])
        res.append(dict(model=m,ctx=ctx,n_pairs=len(idx),parse=parse,p_target_base=base.loc[idx].mean(),p_target_ctx=c.loc[idx].mean(),shift=d.mean(),lo=np.quantile(bs,.025),hi=np.quantile(bs,.975),mean_abs_pair_shift=np.abs(d).mean(),share_pairs_shift_ge10pp=(np.abs(d)>=.10).mean(),mass_on_third=share3,fail=bool(abs(d.mean())>=.10 and (np.quantile(bs,.025)>0 or np.quantile(bs,.975)<0))))
R=pd.DataFrame(res);R.to_csv(OUTP/'iia_v2_by_model.csv',index=False)
for tag,thr in [('strict (parse>=0.9)',.9),('relaxed (parse>=0.5)',.5)]:
    U=R[R.model.isin(usable)&(R.parse>=thr)]
    if not len(U):continue
    print('IIA panel',tag,'models',U.model.nunique());
    S=U.groupby('ctx').agg(models=('model','nunique'),mean_shift=('shift','mean'),median_abs_shift=('shift',lambda x:x.abs().median()),mean_abs_pair_shift=('mean_abs_pair_shift','mean'),models_failing_10pp=('fail','sum'),mean_mass_on_third=('mass_on_third','mean')).reset_index();S['panel']=tag;S.to_csv(OUTP/f"iia_v2_summary_{'strict' if thr==.9 else 'relaxed'}.csv",index=False);print(S.round(3).to_string())
U=R[R.model.isin(usable)&(R.parse>=.5)]
if len(U):
        acc=U.groupby('model').agg(abs_shift=('shift',lambda x:x.abs().mean())).join(mm.accuracy);print('spearman(accuracy, mean |shift|) =',round(spearmanr(acc.accuracy,acc.abs_shift).statistic,3),'n',len(acc))
# ---- correction
C=D[D.exp=='correction'].copy();C['var']=C.cond.str.split('_').str[0];res=[]
for m in models:
    g=C[C.model==m];g=g[g.top.notna()&g.top2.notna()]
    mis=g[(g.top==g.conf)]  # misled at step 1 (chose the confounder supported by the injected finding)
    r={}
    for v in ['retracted','control','confirmed']:
        s=mis[mis['var']==v];r[f'revert_to_target_{v}']=(s.top2==s.target).mean() if len(s)>=10 else np.nan;r[f'n_misled_{v}']=len(s)
    allv=g.groupby('var').apply(lambda s:(s.top2==s.target).mean());
    res.append(dict(model=m,parse1=C[C.model==m].top.notna().mean(),parse2=C[C.model==m].top2.notna().mean(),p_target_step1=(g.top==g.target).mean(),**{f'p_target_after_{k}':v for k,v in allv.items()},**r,
        retraction_selectivity=r['revert_to_target_retracted']-r['revert_to_target_control'],confirmed_stays_wrong=1-r['revert_to_target_confirmed'] if not np.isnan(r['revert_to_target_confirmed']) else np.nan))
CR=pd.DataFrame(res);CR.to_csv(OUTP/'correction_by_model.csv',index=False)
cu=CR[CR.model.isin(usable)&(CR.parse1>=.9)&(CR.parse2>=.8)];print('correction usable',len(cu));print(cu[['model','p_target_step1','revert_to_target_retracted','revert_to_target_control','revert_to_target_confirmed','retraction_selectivity','n_misled_retracted']].round(3).to_string())
if len(cu):print('mean retraction selectivity',cu.retraction_selectivity.mean().round(3),'; models with selectivity>0.2:',int((cu.retraction_selectivity>.2).sum()),'of',cu.retraction_selectivity.notna().sum())
# ---- alias controls
sc=pd.read_csv(OUT/'scoring_leading/new_inference_scored.csv');named_nc=sc[(sc.experiment=='candidate_context')&(sc.condition=='related_order0')].groupby('model').correct.mean().rename('named_no_card');named_c=sc[sc.experiment=='knowledge_provided'].groupby('model').correct.mean().rename('named_card')
A=D[D.exp.isin(['alias_card','alias_nocard'])];al=A.groupby(['model','exp']).correct.mean().unstack().rename(columns={'alias_card':'aliased_card','alias_nocard':'aliased_no_card'});parse=A.groupby('model').top.apply(lambda x:x.notna().mean()).rename('parse')
T=al.join([named_nc,named_c,parse]);T['name_knowledge_effect(named−aliased, no card)']=T.named_no_card-T.aliased_no_card;T['card_use(aliased card − aliased no card)']=T.aliased_card-T.aliased_no_card;T['names_beyond_card(named card − aliased card)']=T.named_card-T.aliased_card;T=T.reset_index();T.to_csv(OUTP/'alias_controls.csv',index=False)
tu=T[T.model.isin(usable)&(T.parse>=.9)];print(tu.round(3).to_string());print(tu.drop(columns=['model','parse']).mean().round(3).to_dict())
