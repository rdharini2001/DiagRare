#!/usr/bin/env python3
"""MIMIC-IV demo (open access, 100 patients, no free text). Task: infer the principal diagnosis of an admission from demographics, first-24h abnormal labs and (varying) history.
Conditions: no_history, true_history (diagnoses from the patient's EARLIER admissions only), mismatched_history (another patient's earlier diagnoses). Closed set: truth + 3 distractor principal diagnoses (same ICD chapter letter when possible)."""
import json
from pathlib import Path
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[2];H=ROOT/'rebuttal/clinical_sources/mimic/mimic-iv-clinical-database-demo-2.2/hosp';D=ROOT/'rebuttal/data'
a=pd.read_csv(H/'admissions.csv.gz',parse_dates=['admittime','dischtime']);dx=pd.read_csv(H/'diagnoses_icd.csv.gz');dd=pd.read_csv(H/'d_icd_diagnoses.csv.gz');p=pd.read_csv(H/'patients.csv.gz');lab=pd.read_csv(H/'labevents.csv.gz',parse_dates=['charttime']);dl=pd.read_csv(H/'d_labitems.csv.gz')
dx=dx.merge(dd,on=['icd_code','icd_version']);princ=dx[dx.seq_num==1].set_index('hadm_id');rng=np.random.default_rng(20261118)
a=a.merge(p[['subject_id','gender','anchor_age','anchor_year']],on='subject_id');a['age']=a.anchor_age+(a.admittime.dt.year-a.anchor_year);a=a.sort_values(['subject_id','admittime'])
lab=lab.merge(dl[['itemid','label','fluid']],on='itemid');lab=lab[lab.flag.astype(str).str.lower().eq('abnormal')]
def labs(hid,t0):
    x=lab[(lab.hadm_id==hid)&(lab.charttime<=t0+pd.Timedelta(hours=24))&lab.valuenum.notna()].copy()
    if x.empty:return []
    x['dev']=np.where(x.ref_range_upper.notna()&(x.valuenum>x.ref_range_upper),(x.valuenum-x.ref_range_upper)/x.ref_range_upper.clip(lower=1e-6),np.where(x.ref_range_lower.notna()&(x.valuenum<x.ref_range_lower),(x.ref_range_lower-x.valuenum)/x.ref_range_lower.clip(lower=1e-6),0))
    x=x.sort_values('dev',ascending=False).drop_duplicates('label').head(12);return [f"{r.label} ({r.fluid}): {r.valuenum:g} {r.valueuom if isinstance(r.valueuom,str) else ''} {'(high)' if (pd.notna(r.ref_range_upper) and r.valuenum>r.ref_range_upper) else '(low)'}".replace('  ',' ') for r in x.itertuples()]
hist={};cases=[]
for sid,g in a.groupby('subject_id'):
    g=g.sort_values('admittime');seen=[]
    for _,r in g.iterrows():
        if seen:cases.append(r)
        hist[r.hadm_id]=list(seen);seen=seen+[r.hadm_id]
titles=lambda hs:sorted({t for h in hs for t in dx[dx.hadm_id==h].long_title.tolist()})
prin_all=princ.long_title;chap=lambda t:dx[dx.long_title==t].icd_code.iloc[0][0]
items=[];design=[];HEAD="You are assisting with a diagnostic exercise using de-identified hospital records."
RET="Return only a JSON object with two keys: ranking (an array containing ALL option letters once each, best first) and confidence (probability 0-1 that your first choice is correct)."
sample=[c for c in cases if c.hadm_id in princ.index];print('admissions with earlier admissions & principal dx',len(sample))
for ci,r in enumerate(sample):
    truth=princ.loc[r.hadm_id,'long_title'];pool=[t for t in prin_all.unique() if t!=truth];same=[t for t in pool if chap(t)==chap(truth)]
    dis=list(rng.choice(same,min(3,len(same)),replace=False)) if len(same)>=3 else list(rng.choice(same,len(same),replace=False))+list(rng.choice([t for t in pool if t not in same],3-len(same),replace=False))
    opts=[truth]+dis;order=list(rng.permutation(4));L=['A','B','C','D'];tid=L[order.index(0)]
    L_=labs(r.hadm_id,r.admittime);base=f"Patient: {r.age}-year-old {'woman' if r.gender=='F' else 'man'}; admission type: {r.admission_type.lower()}; admission location: {str(r.admission_location).lower()}.\nAbnormal laboratory results in the first 24 hours: "+('; '.join(L_) if L_ else 'none recorded')+'.'
    others=[h for h in hist if h!=r.hadm_id and a[a.hadm_id==h].subject_id.iloc[0]!=r.subject_id and hist[h]];wrong=others[int(rng.integers(len(others)))]
    for cond,hs in [('no_history',None),('true_history',hist[r.hadm_id]),('mismatched_history',hist[wrong])]:
        htxt='' if hs is None else ("\nPrevious hospital admissions: "+str(len(hs))+". Diagnoses recorded at those earlier admissions: "+('; '.join(titles(hs)[:15]) or 'none')+'.')
        prompt=f"{HEAD}\n{base}{htxt}\n\nWhich was the principal diagnosis of this admission?\nOptions:\n"+'\n'.join(f'{L[k]}: {opts[order[k]]}' for k in range(4))+'\n'+RET
        iid=f'ehr/{ci}/{cond}';items.append(dict(item_id=iid,experiment='ehr_closed',pair_id=ci,condition=cond,prompt=prompt,target=truth,target_id=tid,candidate_ids=L))
        design.append(dict(item_id=iid,case=ci,cond=cond,subject=int(r.subject_id),n_prior_adm=len(hist[r.hadm_id]),n_hist_dx=len(titles(hist[r.hadm_id])),truth=truth,n_labs=len(L_)))
    for cond,hs in [('no_history',None),('true_history',hist[r.hadm_id])]:
        htxt='' if hs is None else ("\nPrevious hospital admissions: "+str(len(hs))+". Diagnoses recorded at those earlier admissions: "+('; '.join(titles(hs)[:15]) or 'none')+'.')
        items.append(dict(item_id=f'ehr_open/{ci}/{cond}',experiment='ehr_open',pair_id=ci,condition=cond,prompt=f"{HEAD}\n{base}{htxt}\n\nGive your three most likely principal diagnoses for this admission, most likely first, using standard ICD-style disease names. Return only a JSON object with keys diagnoses (array of up to three strings) and confidence (probability 0-1 that the first is correct).",target=truth,target_id='',candidate_ids=[]))
(D/'ehr_suite.jsonl').write_text('\n'.join(json.dumps(x) for x in items));pd.DataFrame(design).to_csv(D/'ehr_design.csv',index=False)
d=pd.DataFrame(design);print(len(items),'prompts;',d.case.nunique(),'cases; patients',d.subject.nunique(),'; median prior admissions',d.n_prior_adm.median(),'; median history dx',d.n_hist_dx.median());print(items[1]['prompt'][:1500])
