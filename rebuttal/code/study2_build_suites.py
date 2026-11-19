#!/usr/bin/env python3
"""Study 2 suites: SCT-Bench (clinician-panel-scored updating), Orphanet-grounded crossover (real findings, curated frequencies, stated prevalences) + age-of-onset patient-specific priors,
ledger and chain-of-thought variants of the aliased explicit-number tests, CUPCase open-ended and closed."""
import json,re,sys
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[2];D=ROOT/'rebuttal/data';CS=ROOT/'rebuttal/clinical_sources'
rng=np.random.default_rng(20261118);out={}
# ---------------- SCT-Bench
sct=pd.read_csv(CS/'sctbench/data/sct_cleaned_full.csv',encoding_errors='ignore');G=(CS/'sctbench/data/templates/guideline.md').read_text()
items=[]
for i,r in sct.iterrows():
    cnt=np.array([r[c] for c in ['-2','-1','0','1','2']],float);score=cnt/cnt.max() if cnt.max()>0 else cnt
    q=str(r.question).replace('If you were thinking of:','').strip();a=str(r.additional_info).replace('And then you find:','').strip()
    prompt=f"{G}\n\n## Scenario\n\n{r.sct_stem}\n\n## Hypothesis\n\n{q}\n\n## Additional Information\n\n{a}\n\n## Response"
    items.append(dict(item_id=f'sct/{r.question_id}',experiment='sct',pair_id=int(i),condition=str(r.source),prompt=prompt,target='',target_id='',candidate_ids=[],panel_counts=cnt.tolist(),panel_score=score.tolist(),source=str(r.source)))
out['sct']=items
# ---------------- Orphanet-grounded crossover + ages
FREQ={'Obligate (100%)':.99,'Very frequent (99-80%)':.9,'Frequent (79-30%)':.55,'Occasional (29-5%)':.17,'Very rare (<4-1%)':.025,'Excluded (0%)':.002}
CLASS={'>1 / 1000':2e-3,'1-5 / 10 000':3e-4,'6-9 / 10 000':7.5e-4,'1-9 / 100 000':5e-5,'1-9 / 1 000 000':5e-6,'<1 / 1 000 000':5e-7}
prev={}
for d in ET.parse(CS/'orphanet/prev.xml').getroot().iter('Disorder'):
    vals=[]
    for p in d.iter('Prevalence'):
        if p.findtext('PrevalenceType/Name')!='Point prevalence' or p.findtext('PrevalenceValidationStatus/Name')!='Validated':continue
        v=float(p.findtext('ValMoy') or 0);cl=p.findtext('PrevalenceClass/Name');x=v/1e5 if v>0 else CLASS.get(cl)
        if x:vals.append(x)
    if vals:prev[d.findtext('OrphaCode')]=(d.findtext('Name'),float(np.median(vals)))
ages={}
for d in ET.parse(CS/'orphanet/ages.xml').getroot().iter('Disorder'):
    ages[d.findtext('OrphaCode')]=dict(onset=[a.findtext('Name') for a in d.iter('AverageAgeOfOnset')],inh=[a.findtext('Name') for a in d.iter('TypeOfInheritance')])
assoc={}
for d in ET.parse(CS/'orphanet/hpo_assoc.xml').getroot().iter('Disorder'):
    c=d.findtext('OrphaCode')
    if d.findtext('DisorderType/Name')!='Disease' or c not in prev:continue
    L=[(a.findtext('HPO/HPOId'),a.findtext('HPO/HPOTerm'),FREQ.get(a.findtext('HPOFrequency/Name'))) for a in d.iter('HPODisorderAssociation')];L=[x for x in L if x[2] is not None]
    if len(L)>=12 and sum(p>=.3 for _,_,p in L)>=4 and 1e-5<=prev[c][1]<=5e-3:assoc[c]=L
codes=sorted(assoc);N=len(codes);names=[prev[c][0] for c in codes];pf=[{h:p for h,_,p in assoc[c]} for c in codes];term={h:t for c in codes for h,t,_ in assoc[c]}
vocab={h:i for i,h in enumerate(sorted(term))};X=np.zeros((N,len(vocab)))
for j in range(N):
    for h,p in pf[j].items():X[j,vocab[h]]=p
Xn=X/np.linalg.norm(X,axis=1,keepdims=True);SIM=Xn@Xn.T;np.fill_diagonal(SIM,-1)
def syn(a,b):
    ta=set(re.sub(r'[^a-z0-9 ]','',a.lower()).split());tb=set(re.sub(r'[^a-z0-9 ]','',b.lower()).split());return len(ta&tb)/max(1,len(ta|tb))>=.5 or a.lower() in b.lower() or b.lower() in a.lower()
HEAD="You are assisting with a diagnostic exercise about two diagnoses in a referral clinic."
TAIL="Return only a JSON object with two keys: ranking (an array containing both candidate IDs once each, most probable first) and confidence (your numerical probability between 0 and 1 that your first choice is correct). Do not copy an example; choose from this case."
items=[];design=[];BG=.03;R_LEVELS=[1e-3,1/30,1,30,1e3];pairs=[]
for j in rng.permutation(N):
    nn=[k for k in np.argsort(-SIM[j])[:15] if not syn(names[j],names[k])]
    if len(nn)<3:continue
    k=int(rng.choice(nn[:8]));A,B=j,k
    fa=[(h,p,pf[B].get(h,BG)) for h,_,p in assoc[codes[A]] if p>=.3 and pf[B].get(h,BG)<=.1]
    if len(fa)<4:continue
    rng.shuffle(fa);pairs.append((A,B,fa[:4]))
    if len(pairs)==100:break
for pi,(A,B,fa) in enumerate(pairs):
    for lv in (1,2,3,4):
        sel=fa[:lv];loglr=sum(np.log(p/q) for _,p,q in sel)
        for ri,r in enumerate(R_LEVELS):
            pa=50*r;pb=50.0;lpo=np.log(r)+loglr
            for o in (0,1):
                cand=[('P0',names[A]),('P1',names[B])] if o==0 else [('P1',names[B]),('P0',names[A])]
                prompt=f"{HEAD}\nStated prevalence in this clinic's patient population (cases per 100,000 patients with this type of presentation): {names[A]}: {pa:g}; {names[B]}: {pb:g}.\nPresent findings: {', '.join(term[h].lower() for h,_,_ in sel)}.\nExplicitly absent findings: none reported.\n\nCandidate IDs:\n"+'\n'.join(f'{i}: {n}' for i,n in cand)+'\n'+TAIL
                iid=f'gx/{pi}/L{lv}/R{ri}/o{o}';items.append(dict(item_id=iid,experiment='grounded_crossover',pair_id=pi,condition=f'L{lv}_R{ri}_o{o}',prompt=prompt,target=names[A],target_id='P0',candidate_ids=['P0','P1']))
                design.append(dict(item_id=iid,kind='grounded',pair=pi,level=lv,r_index=ri,order=o,log_lr=loglr,log_prior_odds=np.log(r),log_post_odds=lpo,ref_A=bool(lpo>0) if abs(lpo)>=.5 else np.nan,A=names[A],B=names[B],onset_A='',onset_B=''))
# age-of-onset patient-specific prior
PED={'Antenatal','Neonatal','Infancy','Childhood'};ADU={'Adult','Elderly'}
ped=[j for j in range(N) if ages.get(codes[j],{}).get('onset') and set(ages[codes[j]]['onset'])<=PED];adu=[j for j in range(N) if ages.get(codes[j],{}).get('onset') and set(ages[codes[j]]['onset'])<=ADU]
rng.shuffle(ped);rng.shuffle(adu);ap=[(ped[i],adu[i]) for i in range(min(60,len(ped),len(adu)))]
for pi,(P,Ad) in enumerate(ap):
    fp=[(h,p) for h,_,p in assoc[codes[P]] if p>=.3 and pf[Ad].get(h,BG)<=.1];fa2=[(h,p) for h,_,p in assoc[codes[Ad]] if p>=.3 and pf[P].get(h,BG)<=.1]
    if not fp or not fa2:continue
    f1=fp[int(rng.integers(len(fp)))];f2=fa2[int(rng.integers(len(fa2)))]
    for age,txt in [('child','The patient is a 6-year-old child.'),('adult','The patient is a 58-year-old adult.'),('none','The patient age is not recorded.')]:
        for o in (0,1):
            cand=[('P0',names[P]),('P1',names[Ad])] if o==0 else [('P1',names[Ad]),('P0',names[P])]
            prompt=f"{HEAD}\n{txt}\nPresent findings: {term[f1[0]].lower()}, {term[f2[0]].lower()}.\nExplicitly absent findings: none reported.\nUse the patient's age in your reasoning where it is relevant.\n\nCandidate IDs:\n"+'\n'.join(f'{i}: {n}' for i,n in cand)+'\n'+TAIL
            iid=f'age/{pi}/{age}/o{o}';items.append(dict(item_id=iid,experiment='age_prior',pair_id=pi,condition=f'{age}_o{o}',prompt=prompt,target=names[P],target_id='P0',candidate_ids=['P0','P1']))
            design.append(dict(item_id=iid,kind='age',pair=pi,age=age,order=o,A=names[P],B=names[Ad],onset_A='/'.join(ages[codes[P]]['onset']),onset_B='/'.join(ages[codes[Ad]]['onset']),ref_A={'child':True,'adult':False,'none':np.nan}[age],log_lr=np.nan,log_prior_odds=np.nan,log_post_odds=np.nan,level=0,r_index=-1))
out['grounded']=items;pd.DataFrame(design).to_csv(D/'grounded_design.csv',index=False)
# ---------------- ledger and CoT variants of the aliased explicit-number families (reuse crossover design)
lt=[];ct=[]
cd=pd.read_csv(D/'crossover_design.csv').set_index('item_id');cs={json.loads(l)['item_id']:json.loads(l) for l in open(D/'crossover_suite.jsonl')}
COT="First reason step by step, stating the posterior odds you compute from the numbers, then finish with the JSON object (ranking and confidence) on its own line as the final text of your answer."
KEYS="\nThe JSON keys are: ranking (array of both candidate IDs, most probable first) and confidence (probability 0-1 that the first choice is correct)."
HEAD2="This is a quantitative reasoning exercise about two diagnoses. The disease names are arbitrary labels; use only the numbers given and the chart records. Assume the two diagnoses are the only possibilities and that wrongly choosing either one is equally costly. Combine only evidence that is genuinely independent."
for iid,it in cs.items():
    d=cd.loc[iid]
    if d['order']!=0:continue
    body=it['prompt'].split('Return only a JSON object')[0]
    if d.kind in('surface','dup'):ct.append(dict(it,item_id=iid.replace('xover','xoverCoT'),experiment='crossover_cot',condition=it['condition']+'_cot',prompt=body+COT+KEYS))
    if d.kind=='dup' and d.form in('duplicate','independent','single'):
        lr=d.base_lr;pa=d.prior_A;aN=re.search(r'with (\w+ syndrome) as in',it['prompt']).group(1);bN=re.search(r'as in patients with (\w+ syndrome)',it['prompt']).group(1)
        if d.form=='single':led=f"#1 | source: laboratory assay A | report R-1 | specimen S-1 | collected 08:00 | result: likelihood ratio {lr:g} favouring {aN}"
        elif d.form=='duplicate':led=f"#1 | source: laboratory assay A | report R-1 | specimen S-1 | collected 08:00 | result: likelihood ratio {lr:g} favouring {aN}\n#2 | source: chart printout | report R-1 | specimen S-1 | collected 08:00 | result: likelihood ratio {lr:g} favouring {aN}"
        else:led=f"#1 | source: laboratory assay A | report R-1 | specimen S-1 | collected 08:00 | result: likelihood ratio {lr:g} favouring {aN}\n#2 | source: laboratory assay B | report R-2 | specimen S-2 (new sample) | collected 14:00 | result: likelihood ratio {lr:g} favouring {aN}"
        tail=it['prompt'].split('Candidate IDs:')[1]
        lt.append(dict(it,item_id=iid.replace('xover','xoverLedger'),experiment='crossover_ledger',condition=it['condition']+'_ledger',prompt=f"{HEAD2}\nPrior: {aN} accounts for {pa*100:g}% of patients with this presentation and {bN} for the rest.\nEvidence ledger (each entry is one chart record):\n{led}\n\nCandidate IDs:"+tail))
out['ledger']=lt;out['cot']=ct
# ---------------- CUPCase: closed (MCQ) and open-ended
cc=pd.read_parquet(CS/'cupcase/test.parquet');cc=cc.iloc[rng.permutation(len(cc))[:400]].reset_index(drop=True);items=[]
for i,r in cc.iterrows():
    opts=[r.correct_diagnosis,r.distractor1,r.distractor2,r.distractor3];order=list(rng.permutation(4));ids=['A','B','C','D'];tid=ids[order.index(0)]
    prompt=f"You are assisting with a diagnostic exercise on a published case report.\nCase:\n{r.clean_case_presentation}\n\nOptions:\n"+'\n'.join(f'{ids[k]}: {opts[order[k]]}' for k in range(4))+"\nReturn only a JSON object with two keys: ranking (an array containing ALL option letters once each, best first) and confidence (probability 0-1 that your first choice is correct)."
    items.append(dict(item_id=f'cup/closed/{i}',experiment='cupcase_closed',pair_id=i,condition='closed',prompt=prompt,target=r.correct_diagnosis,target_id=tid,candidate_ids=ids))
    items.append(dict(item_id=f'cup/open/{i}',experiment='cupcase_open',pair_id=i,condition='open',prompt=f"You are assisting with a diagnostic exercise on a published case report.\nCase:\n{r.clean_case_presentation}\n\nGive your three most likely diagnoses, most likely first, using standard disease names. Return only a JSON object with two keys: diagnoses (array of up to three strings) and confidence (probability 0-1 that the first diagnosis is correct).",target=r.correct_diagnosis,target_id='',candidate_ids=[]))
out['cupcase']=items
for k,v in out.items():
    (D/f'study2_{k}_suite.jsonl').write_text('\n'.join(json.dumps(x) for x in v));print(k,len(v))
print(len(ap),'age pairs;',len(pairs),'grounded pairs;eligible diseases',N,'; ped-only',len(ped),'adult-only',len(adu))
print(out['grounded'][0]['prompt']);print(out['ledger'][2]['prompt'][:900])
