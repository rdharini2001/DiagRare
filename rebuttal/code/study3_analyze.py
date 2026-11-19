import pandas as pd,numpy as np,json,re,glob,sys
from pathlib import Path
from scipy.stats import spearmanr
ROOT=Path(__file__).resolve().parents[2];R=ROOT/'rebuttal/results';S3=R/'study3';D=ROOT/'rebuttal/data';rng=np.random.default_rng(20261118);A=R/'study3A';A.mkdir(exist_ok=True)
def lead(raw):
    t=re.sub(r'^```(?:json)?\s*','',str(raw).strip(),flags=re.I)
    if not t.startswith('{'):return {}
    try:v,_=json.JSONDecoder().raw_decode(t);return v if isinstance(v,dict) else {}
    except Exception:return {}
def ranks(raw,cands):
    rk=lead(raw).get('ranking',[]);pre=[]
    if isinstance(rk,list):
        for x in rk:
            if isinstance(x,str) and x in cands and x not in pre:pre.append(x)
            else:break
    if not pre:return None
    n=len(cands);out={c:i+1 for i,c in enumerate(pre)};rest=[c for c in cands if c not in out];avg=len(pre)+(n-len(pre)+1)/2
    for c in rest:out[c]=avg
    return out
def load(suf):
    rows=[]
    for f in sorted(S3.glob(f'*__{suf}.jsonl')):
        for l in open(f):
            r=json.loads(l)
            if r['finish_reason']=='prompt_too_long':rk=None
            else:rk=ranks(r['raw_response'],r['candidate_ids'])
            rows.append((r['model'],r['item_id'],rk))
    return rows
def boot(x,B=2000):
    x=np.asarray(x,float);x=x[~np.isnan(x)];
    if len(x)<10:return np.nan,np.nan,np.nan
    b=[x[rng.integers(0,len(x),len(x))].mean() for _ in range(B)];return x.mean(),*np.percentile(b,[2.5,97.5])
res={}
# ---- E1
des=pd.read_csv(D/'study3_arena_design.csv').set_index('item_id');rows=load('arena');E1=[]
for m,iid,rk in rows:
    d=des.loc[iid];E1.append(dict(model=m,case=d.case,stage=d.stage,order=d.order,parsed=rk is not None,rank=rk[d.correct_id] if rk else np.nan,top1=(min(rk,key=rk.get)==d.correct_id) if rk else np.nan))
E1=pd.DataFrame(E1);E1.to_csv(A/'e1_scored.csv',index=False);out=[]
for m,g in E1.groupby('model'):
    w=g.pivot_table(index=['case','order'],columns='stage',values=['rank','top1'],aggfunc='first');
    try:w=w.dropna();rr1=1/w[('rank','s1')];rr3=1/w[('rank','s3')]
    except Exception:continue
    dd=(rr3-rr1);same=(w[('top1','s1')]==w[('top1','s3')])
    cm=dd.groupby(level='case').mean();ci=dd[same].groupby(level='case').mean()
    r=dict(model=m,parse=g.parsed.mean(),n_case=len(cm),acc_s1=w[('top1','s1')].mean(),acc_s2=g[g.stage=='s2'].top1.astype(float).mean(),acc_s3=w[('top1','s3')].mean(),rr_s1=rr1.mean(),rr_s3=rr3.mean())
    r['Y1_rr_gain'],r['Y1_lo'],r['Y1_hi']=boot(cm.values);r['inv_gain'],r['inv_lo'],r['inv_hi']=boot(ci.values);r['inv_share']=same.mean();out.append(r)
E1m=pd.DataFrame(out);E1m.to_csv(A/'e1_by_model.csv',index=False);res['E1']=E1m
# ---- E2
des=pd.read_csv(D/'study3_einst_design.csv').set_index('item_id');rows=load('einst');E2=[]
for m,iid,rk in rows:
    d=des.loc[iid];
    if rk is None:E2.append(dict(model=m,case=d.case,ct=d.case_type,ok=np.nan,gt_above=np.nan,top1_ok=np.nan,pred_gt=np.nan));continue
    top=min(rk,key=rk.get);E2.append(dict(model=m,case=d.case,ct=d.case_type,top1_ok=float(top==d.truth_id),pred_gt=float(top==d.id_gt),gt_above=float(rk[d.id_gt]<rk[d.id_bias])))
E2=pd.DataFrame(E2);out=[]
for m,g in E2.groupby('model'):
    c=g[g.ct=='control'].set_index('case');t=g[g.ct=='trap'].set_index('case');j=c.join(t,lsuffix='_c',rsuffix='_t',how='inner').dropna(subset=['gt_above_c','gt_above_t'])
    rev=((j.gt_above_c==1)&(j.gt_above_t==0)).astype(float);cc=j[j.top1_ok_c==1]
    r=dict(model=m,parse=g.top1_ok.notna().mean(),n_pairs=len(j),acc_control=j.top1_ok_c.mean(),acc_trap=j.top1_ok_t.mean(),btr=(cc.pred_gt_t==1).mean() if len(cc) else np.nan)
    r['Y2_reversal'],r['Y2_lo'],r['Y2_hi']=boot(rev.values);out.append(r)
E2m=pd.DataFrame(out);E2m.to_csv(A/'e2_by_model.csv',index=False);res['E2']=E2m
# ---- E3
des=pd.read_csv(D/'study3_xdx_design.csv').set_index('item_id');rows=load('xdx');rk_d={}
for m,iid,rk in rows:rk_d[(m,iid)]=rk
dd=des.reset_index();E3=[];out=[]
base=dd[dd.variant=='base'].set_index(['case','order']).item_id
for m in sorted({k[0] for k in rk_d}):
    vals=[];n_all=n_ok=0
    for (case,order),bid in base.items():
        rb=rk_d.get((m,bid));sub=dd[(dd.case==case)&(dd.order==order)&(dd.variant!='base')]
        for tgt,g in sub.groupby('target'):
            n_all+=1;dl=g[g.variant=='del'];cl=g[g.variant=='ctl']
            if rb is None or dl.empty or cl.empty:continue
            rd=rk_d.get((m,dl.item_id.iloc[0]));rc=rk_d.get((m,cl.item_id.iloc[0]))
            if rd is None or rc is None:continue
            tid=dl.target_id.iloc[0];n_ok+=1;vals.append(((rd[tid]-rb[tid])-(rc[tid]-rb[tid]),case))
    if not vals:continue
    v=pd.DataFrame(vals,columns=['y','case']);cm=v.groupby('case').y.mean();r=dict(model=m,parse=n_ok/max(n_all,1),n_obs=len(v),n_case=len(cm));r['Y3_deletion_effect'],r['Y3_lo'],r['Y3_hi']=boot(cm.values);out.append(r)
E3m=pd.DataFrame(out);E3m.to_csv(A/'e3_by_model.csv',index=False);res['E3']=E3m
pd.set_option('display.width',250)
for k,v in res.items():print('\n=====',k);print(v.round(3).to_string())
