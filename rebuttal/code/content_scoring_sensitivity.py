#!/usr/bin/env python3
"""Separate leading-JSON content scoring; strict scores remain unchanged.
Read only an initial complete JSON object, optionally following a JSON fence.
Ignore trailing prose; never search later text for a preferred answer.
"""
from pathlib import Path
import json,re
import pandas as pd
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'rebuttal/results'
def leading_object(raw):
    text=raw.strip();text=re.sub(r'^```(?:json)?\s*','',text,flags=re.I)
    if not text.startswith('{'):return {}
    try:value,_=json.JSONDecoder().raw_decode(text)
    except (ValueError,TypeError):return {}
    return value if isinstance(value,dict) else {}
def norm(x):return re.sub('[^a-z0-9]','',str(x).lower())
def main(output=None):
    destination=Path(output) if output else OUT
    rows=[]
    for path in sorted(destination.glob('*__new_inference.jsonl')):
        for line in path.read_text().splitlines():
            try:r=json.loads(line)
            except ValueError:continue
            obj=leading_object(r['raw_response']);ex=r['experiment'];allowed=r.get('candidate_ids',[])+(['OTHER','ABSTAIN'] if ex=='missing_candidate' else []);rr=obj.get('ranking',[]);prefix=[]
            if isinstance(rr,list):
                for val in rr:
                    if not isinstance(val,str) or val not in allowed or val in prefix:break
                    prefix.append(val)
            top=prefix[0] if prefix else None;valid=bool(prefix);correct=top==r.get('target_id')
            if ex=='missing_candidate' and r['condition']=='missing':correct=top=='OTHER'
            if ex=='relation_knowledge':valid=obj.get('answer') in ['A','B'];correct=obj.get('answer')==r['gold_answer']
            if ex=='open_generation':
                answers=obj.get('diagnoses',[]);valid=isinstance(answers,list) and bool(answers) and all(isinstance(a,str) for a in answers);top=answers[0] if valid else None;correct=valid and norm(top)==norm(r['target'])
            row={'model':r['model'],'item_id':r['item_id'],'experiment':ex,'condition':r['condition'],'pair_id':r['pair_id'],'content_parseable':valid,'content_top':top,'content_correct':int(correct),'finish_reason':r['finish_reason'],'other_or_abstain':int(top in ['OTHER','ABSTAIN'])}
            if ex=='stateful_revision':
                sec=leading_object(r.get('step2_response',''));ss=sec.get('ranking',[]);valid2=isinstance(ss,list) and bool(ss) and isinstance(ss[0],str) and ss[0] in allowed;row['step2_content_parseable']=valid2;row['step2_content_correct']=int(valid2 and ss[0]==r['target_id'])
            rows.append(row)
    if not rows:return
    df=pd.DataFrame(rows);df.to_csv(destination/'content_scoring_per_item.csv',index=False)
    summary=df.groupby(['model','experiment','condition']).agg(n=('item_id','size'),content_parseable_rate=('content_parseable','mean'),content_accuracy_all=('content_correct','mean'),other_or_abstain_all=('other_or_abstain','mean')).reset_index();summary.to_csv(destination/'content_scoring_summary.csv',index=False)
    (destination/'content_scoring_notes.md').write_text('''# Leading-object answer-content sensitivity\n\nStrict output-contract scoring remains unchanged. This separate analysis reads the first complete JSON object only when it begins the response, optionally after a JSON Markdown fence, and ignores trailing prose. A ranking prefix stops at the first invalid or duplicated ID. It never guesses diagnoses or extracts a later preferred answer. Free diagnosis uses strict lexical matching and still requires synonym/granularity adjudication. Length-truncated outputs may have a complete answer object and incomplete trailing explanation; finish_reason is retained. Do not equate output-format failure with lack of medical knowledge. Compare completed disease sets and report both content coverage and strict compliance.\n''')
if __name__=='__main__':main()
