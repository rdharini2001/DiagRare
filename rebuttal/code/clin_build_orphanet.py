#!/usr/bin/env python3
"""Orphanet-grounded benchmark: disease profiles = Orphanet expert-curated HPO phenotype associations with literature frequency classes;
prior = Orphanet validated point prevalence (PMID-sourced). Vignettes are sampled from the curated frequencies; evidence = exact naive-Bayes log-likelihood ratio under those frequencies."""
import re,json
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[2];S=ROOT/'rebuttal/clinical_sources/orphanet';OUTD=ROOT/'rebuttal/data'
FREQ={'Obligate (100%)':.99,'Very frequent (99-80%)':.9,'Frequent (79-30%)':.55,'Occasional (29-5%)':.17,'Very rare (<4-1%)':.025,'Excluded (0%)':.002}
CLASS={'>1 / 1000':2e-3,'1-5 / 10 000':3e-4,'6-9 / 10 000':7.5e-4,'1-9 / 100 000':5e-5,'1-9 / 1 000 000':5e-6,'<1 / 1 000 000':5e-7}
prev={}
for d in ET.parse(S/'prev.xml').getroot().iter('Disorder'):
    code=d.findtext('OrphaCode');name=d.findtext('Name');vals=[]
    for p in d.iter('Prevalence'):
        if p.findtext('PrevalenceType/Name')!='Point prevalence' or p.findtext('PrevalenceValidationStatus/Name')!='Validated':continue
        v=float(p.findtext('ValMoy') or 0);cl=p.findtext('PrevalenceClass/Name')
        x=v/1e5 if v>0 else CLASS.get(cl)
        if x:vals.append(x)
    if vals:prev[code]=(name,float(np.median(vals)))
assoc={}
for d in ET.parse(S/'hpo_assoc.xml').getroot().iter('Disorder'):
    code=d.findtext('OrphaCode')
    if d.findtext('DisorderType/Name')!='Disease' or code not in prev:continue
    L=[(a.findtext('HPO/HPOId'),a.findtext('HPO/HPOTerm'),FREQ.get(a.findtext('HPOFrequency/Name'))) for a in d.iter('HPODisorderAssociation')]
    L=[x for x in L if x[2] is not None]
    if len(L)>=12 and sum(p>=.3 for _,_,p in L)>=4 and 1e-5<=prev[code][1]<=5e-3:assoc[code]=L
print('eligible diseases',len(assoc))

codes=sorted(assoc);N=len(codes);names=[prev[c][0] for c in codes];logprior=np.log([prev[c][1] for c in codes]);p_f=[{h:p for h,_,p in assoc[c]} for c in codes];term={h:t for c in codes for h,t,_ in assoc[c]}
vocab={h:i for i,h in enumerate(sorted(term))};X=np.zeros((N,len(vocab)))
for j,c in enumerate(codes):
    for h,p in p_f[j].items():X[j,vocab[h]]=p
Xn=X/np.linalg.norm(X,axis=1,keepdims=True);SIM=Xn@Xn.T;np.fill_diagonal(SIM,-1)
rng=np.random.default_rng(20261116);BG=.03;K=10
def synonymish(a,b):
    ta=set(re.sub(r'[^a-z0-9 ]','',a.lower()).split());tb=set(re.sub(r'[^a-z0-9 ]','',b.lower()).split());return len(ta&tb)/max(1,len(ta|tb))>=.6 or a.lower() in b.lower() or b.lower() in a.lower()
truths=list(rng.permutation(N)[:120]);items=[];feat=[];HEAD='You are assisting with a diagnostic exercise on rare and uncommon diseases.'
for j in truths:
    nn=[k for k in np.argsort(-SIM[j])[:60] if not synonymish(names[j],names[k])]
    for v in range(5):
        hp=assoc[codes[j]];pres=[h for h,_,p in hp if rng.random()<p and p>.01]
        if len(pres)<3:continue
        rng.shuffle(pres);pres=pres[:int(rng.integers(3,7))];absent=[h for h,_,p in hp if h not in pres and p>=.55];rng.shuffle(absent);absent=absent[:int(rng.integers(0,3))]
        cand=[j]+list(rng.choice(nn[:25],K-1,replace=False));rng.shuffle(cand)
        E=np.zeros(K)
        for a,k in enumerate(cand):
            for h in pres:E[a]+=np.log(p_f[k].get(h,BG)/BG)
            for h in absent:E[a]+=np.log((1-p_f[k].get(h,BG))/(1-BG))
        lp=logprior[cand];ids=[f'O{a:02d}' for a in range(K)];t=cand.index(j);cid=f'orph/{j}/{v}'
        feat.append(dict(case=cid,cand=[int(x) for x in cand],names=[names[x] for x in cand],ids=ids,truth_pos=t,E=list(E),logprior=list(lp),n_present=len(pres),n_absent=len(absent)))
        lines='\n'.join(f'{ids[a]}: {names[k]}' for a,k in enumerate(cand))
        prompt=f"{HEAD}\nPresent findings: {', '.join(term[h].lower() for h in pres)}.\nExplicitly absent findings: {', '.join(term[h].lower() for h in absent) if absent else 'none reported'}.\n\nCandidate IDs:\n{lines}\nReturn only a JSON object with two keys: ranking (an array of your top 3 candidate IDs, best first) and confidence (your numerical probability between 0 and 1 that your first diagnosis is correct). Do not copy an example; choose from this case."
        items.append(dict(item_id=cid,experiment='clin_orphanet',pair_id=len(items),condition='differential10',prompt=prompt,target=names[j],target_id=ids[t],candidate_ids=ids))
json.dump(feat,open(OUTD/'clin_orphanet_features.json','w'));open(OUTD/'clin_orphanet_suite.jsonl','w').write('\n'.join(json.dumps(x) for x in items))
f=pd.DataFrame(feat);top_e=np.array([int(np.argmax(r)) for r in f.E]);top_o=np.array([int(np.argmax(np.array(r)+np.array(l))) for r,l in zip(f.E,f.logprior)]);tp=f.truth_pos.to_numpy();top_p=np.array([int(np.argmax(l)) for l in f.logprior])
print(len(items),'vignettes; evidence-top1==truth',(top_e==tp).mean().round(3),'; prior-top1==truth',(top_p==tp).mean().round(3),'; oracle(prior+evidence)==truth',(top_o==tp).mean().round(3))
print(items[7]['prompt'][:1200])
