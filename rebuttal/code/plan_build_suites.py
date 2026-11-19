#!/usr/bin/env python3
"""Plan items C4 (behavioural IIA v2), D (retraction/correction), B3/B4 (aliased-name knowledge controls). One combined suite per model."""
import sys,json,re
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
from common_pl import *
b=Bench();rel=b.relations();ont=b.ont;rng=np.random.default_rng(20261117);ids=lambda d:f'D{b.di[d]:03d}'
def fset(d):return split_tokens(ont.loc[d,'required'])|split_tokens(ont.loc[d,'supporting'])
def jac(a,c):
    A,C=fset(a),fset(c);return len(A&C)/max(1,len(A|C))
# ---------- pairs
cand=[d for d in b.diseases if len(split_tokens(ont.loc[d,'required']))>=3];rng.shuffle(cand);pairs=[]
for t in cand:
    same=[d for d in b.diseases if d!=t and ont.loc[d,'organ_system']==ont.loc[t,'organ_system']]
    if len(same)<3:continue
    c=str(rng.choice(same));others=[d for d in same if d!=c]
    near=max(others,key=lambda d:jac(t,d)+jac(c,d));common=max([d for d in others if d!=near] or others,key=lambda d:ont.loc[d,'prevalence'])
    unrel=str(rng.choice([d for d in b.diseases if ont.loc[d,'organ_system']!=ont.loc[t,'organ_system']]))
    pairs.append(dict(t=t,c=c,near=near,common=common,unrel=unrel))
    if len(pairs)==60:break
def findings(t,c):
    req=sorted(split_tokens(ont.loc[t,'required'])-fset(c));sup=sorted(split_tokens(ont.loc[t,'supporting'])-fset(c));rng.shuffle(req)
    pos=(req[:2]+sup[:1]) if len(req)>=2 else (req+sup)[:3]
    neg=[f for f in sorted(split_tokens(ont.loc[c,'excluding'])) if f not in fset(t)][:2]
    return pos,neg
WORD=[("Rank the candidate diagnoses using the reported patient findings.","Return only a JSON object with two keys: ranking (an array containing ALL listed candidate IDs once each, best first) and confidence (your numerical probability between 0 and 1 that your first diagnosis is correct). Do not copy an example; choose from this case."),
      ("Order the following possible diagnoses from most to least likely for this patient.","Answer with only a JSON object containing ranking (every listed candidate ID exactly once, most likely first) and confidence (a probability between 0 and 1 that your first choice is right). Choose from this case; do not copy an example.")]
items=[];design=[]
for pi,p in enumerate(pairs):
    pos,neg=findings(p['t'],p['c']);fnd='Present findings: '+', '.join(pos)+'.\nExplicitly absent findings: '+(', '.join(neg) if neg else 'none reported')+'.'
    for ctx in ['base','unrelated','near_subtype','common']:
        cs=[p['t'],p['c']]+([] if ctx=='base' else [p[{'unrelated':'unrel','near_subtype':'near','common':'common'}[ctx]]])
        for w in (0,1):
            for o in (0,1):
                order=list(rng.permutation(len(cs)));order=order if o==0 else order[::-1]
                lines='\n'.join(f'{ids(cs[k])}: {cs[k]}' for k in order);head,tail=WORD[w]
                iid=f'iia2/{pi}/{ctx}/w{w}/o{o}';items.append(dict(item_id=iid,experiment='iia_v2',pair_id=pi,condition=f'{ctx}_w{w}_o{o}',prompt=f"{head}\n{fnd}\n\nCandidate IDs (IDs stay fixed when display order changes):\n{lines}\n{tail}",target=p['t'],target_id=ids(p['t']),confounder=p['c'],confounder_id=ids(p['c']),candidate_ids=[ids(x) for x in cs]))
                design.append(dict(item_id=iid,pair=pi,ctx=ctx,para=w,order=o,target=p['t'],confounder=p['c'],third=(cs[2] if len(cs)>2 else ''),jac_third_target=jac(p['t'],cs[2]) if len(cs)>2 else np.nan))
# ---------- correction / retraction (two-turn)
for pi,p in enumerate(pairs):
    t,c=p['t'],p['c'];pos=sorted(split_tokens(ont.loc[t,'required'])-fset(c));rng.shuffle(pos);pos=pos[:2];mis=sorted(split_tokens(ont.loc[c,'required'])-fset(t));
    if not mis or len(pos)<1:continue
    mf=str(rng.choice(mis));base=f"{WORD[0][0]}\nPresent findings: {', '.join(pos+[mf])}.\nExplicitly absent findings: none reported.\n\nCandidate IDs:\n"
    for o in (0,1):
        order=[t,c] if o==0 else [c,t];lines='\n'.join(f'{ids(x)}: {x}' for x in order);prompt=base+lines+'\n'+WORD[0][1]
        for var,txt in [('retracted',f"Update: the finding '{mf}' was a laboratory/transcription error and has been formally retracted; it must be disregarded.\nRe-rank the same candidates."),('control',"Update: the patient's follow-up appointment was rescheduled to next week; no new clinical information.\nRe-rank the same candidates."),('confirmed',f"Update: the finding '{mf}' was independently confirmed by repeat testing.\nRe-rank the same candidates.")]:
            items.append(dict(item_id=f'corr/{pi}/{var}/o{o}',experiment='correction',pair_id=pi,condition=f'{var}_o{o}',prompt=prompt,followup=txt+'\n'+WORD[0][1],target=t,target_id=ids(t),confounder=c,confounder_id=ids(c),candidate_ids=[ids(t),ids(c)],misleading_finding=mf))
# ---------- aliased-name knowledge controls (from the existing synthetic suite)
src=[json.loads(l) for l in open(ROOT/'rebuttal/data/inference_suite.jsonl')];alias=lambda n:'Condition '+re.sub(r'[^A-Za-z]','',n.title())[:0]
used={}
def aliasing(prompt):
    out=[]
    for l in prompt.split('\n'):
        m=re.match(r'^(D\d+): ([A-Za-z_\-]+)$',l)
        if m:
            used.setdefault(m.group(1),'Condition '+m.group(1).replace('D','Q'));out.append(f'{m.group(1)}: {used[m.group(1)]}')
        else:out.append(l)
    return '\n'.join(out)
for i in src:
    if i['experiment']=='knowledge_provided':items.append(dict(i,item_id=i['item_id'].replace('knowledge_provided','alias_card'),experiment='alias_card',condition='aliased_with_card',prompt=aliasing(i['prompt'])))
    if i['experiment']=='candidate_context' and i['condition']=='related_order0':items.append(dict(i,item_id=i['item_id'].replace('candidate_context','alias_nocard'),experiment='alias_nocard',condition='aliased_no_card',prompt=aliasing(i['prompt'])))
open(ROOT/'rebuttal/data/plan_suite.jsonl','w').write('\n'.join(json.dumps(x) for x in items));pd.DataFrame(design).to_csv(ROOT/'rebuttal/data/iia_v2_design.csv',index=False)
import collections;print(len(items),collections.Counter(i['experiment'] for i in items));print(items[5]['prompt']);print([i for i in items if i['experiment']=='correction'][0]['followup'])
print('pairs',len(pairs),'near-subtype Jaccard mean',np.mean([jac(p['t'],p['near']) for p in pairs]).round(3),'unrelated',np.mean([jac(p['t'],p['unrel']) for p in pairs]).round(3))
