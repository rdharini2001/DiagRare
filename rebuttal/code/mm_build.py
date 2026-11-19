#!/usr/bin/env python3
"""MedReaMM (625 expert-validated case reports with real images, MIT). 300 cases. Conditions: text_only (basic info), image_text (+ the case's images), caption_text (+ expert captions as text),
mismatched_image (basic info + images from a case of a different category). Closed set (truth + 3 same-category distractor diagnoses) and open-ended top-3."""
import json,os
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];M=ROOT/'rebuttal/clinical_sources/medreamm';D=ROOT/'rebuttal/data';rng=np.random.default_rng(20261118)
L=[json.loads(l) for l in open(M/'benchmark.jsonl')];cases=[]
for r in L:
    p=r['patient_info'];p=eval(p) if isinstance(p,str) else p;imgs=[];caps=[]
    for s in p['supplementary_info']:
        pa=s['path'];pa=eval(pa) if isinstance(pa,str) else pa
        for q in (pa or []):
            f=M/q
            if f.exists():imgs.append(str(f))
        if s.get('caption'):caps.append(s['caption'])
    dx=str(p.get('diagnosis') or '').strip()
    if imgs and 4<=len(dx)<=90 and len(p['basic_info'])>80:cases.append(dict(pmid=r['pmid'],cls=r['classification'],basic=p['basic_info'].strip(),imgs=imgs[:3],caps=caps,dx=dx))
print('usable cases',len(cases));rng.shuffle(cases);cases=cases[:300]
HEAD="You are assisting with a diagnostic exercise on a published clinical case report."
RET="Return only a JSON object with two keys: ranking (an array containing ALL option letters once each, best first) and confidence (probability 0-1 that your first choice is correct)."
OPEN="Give your three most likely diagnoses, most likely first, using standard disease names. Return only a JSON object with two keys: diagnoses (array of up to three strings) and confidence (probability 0-1 that the first diagnosis is correct)."
items=[];bycls={}
for c in cases:bycls.setdefault(c['cls'],[]).append(c)
for i,c in enumerate(cases):
    pool=[x['dx'] for x in bycls[c['cls']] if x['dx']!=c['dx']];
    if len(pool)<3:pool=[x['dx'] for x in cases if x['dx']!=c['dx']]
    dis=list(rng.choice(pool,3,replace=False));opts=[c['dx']]+dis;order=list(rng.permutation(4));L_=['A','B','C','D'];tid=L_[order.index(0)];optstr='\n'.join(f'{L_[k]}: {opts[order[k]]}' for k in range(4))
    other=[x for x in cases if x['cls']!=c['cls']];wrong=other[int(rng.integers(len(other)))]
    for cond in ('text_only','image_text','caption_text','mismatched_image'):
        txt=f"{HEAD}\nPatient information:\n{c['basic']}\n"
        if cond=='caption_text':txt+="\nDescriptions of the clinical images (written by the case authors):\n"+'\n'.join('- '+x for x in c['caps'])+'\n'
        if cond in('image_text','mismatched_image'):txt+="\nThe case images are attached.\n"
        imgs=c['imgs'] if cond=='image_text' else (wrong['imgs'] if cond=='mismatched_image' else [])
        items.append(dict(item_id=f'mm/closed/{i}/{cond}',experiment='mm_closed',pair_id=i,condition=cond,prompt=txt+"\nOptions:\n"+optstr+"\n"+RET,target=c['dx'],target_id=tid,candidate_ids=L_,images=imgs))
        if cond in('text_only','image_text','caption_text'):items.append(dict(item_id=f'mm/open/{i}/{cond}',experiment='mm_open',pair_id=i,condition=cond,prompt=txt+"\n"+OPEN,target=c['dx'],target_id='',candidate_ids=[],images=imgs))
(D/'mm_vlm_suite.jsonl').write_text('\n'.join(json.dumps(x) for x in items))
txt=[x for x in items if x['condition'] in('text_only','caption_text')];(D/'study2_mmtext_suite.jsonl').write_text('\n'.join(json.dumps({k:v for k,v in x.items() if k!='images'}) for x in txt))
import collections;print(len(items),'VLM prompts;',len(txt),'text-only prompts;',collections.Counter(x['condition'] for x in items))
cap_leak=np.mean([any(w in ' '.join(c['caps']).lower() for w in c['dx'].lower().split() if len(w)>7) for c in cases]);print('captions contain a long diagnosis word in',round(float(cap_leak),3),'of cases')
print(items[1]['prompt'][:800]);print(items[1]['images'])
