import json,re,sys
from pathlib import Path
import numpy as np,pandas as pd
ROOT=Path('.');OUT=ROOT/'rebuttal/results';sys.path.insert(0,'rebuttal/scripts')
def lead(raw):
    t=re.sub(r'^```(?:json)?\s*','',str(raw).strip(),flags=re.I)
    if not t.startswith('{'):return {}
    try:v,_=json.JSONDecoder().raw_decode(t);return v if isinstance(v,dict) else {}
    except Exception:return {}
rows=[]
for f in sorted((OUT/'knowledge_controls').glob('*__kcontrols.jsonl')):
    L=f.read_text().splitlines()
    if len(L)<164:continue
    for l in L:
        r=json.loads(l);o=lead(r['raw_response']);rk=o.get('ranking',[]);top=rk[0] if isinstance(rk,list) and rk and isinstance(rk[0],str) and rk[0] in r['candidate_ids'] else None
        rows.append(dict(model=r['model'],exp=r['experiment'],pair=r['pair_id'],parse=top is not None,correct=float(top==r['target_id']) if top else np.nan,follows_card=float(top==r.get('card_following_id')) if top and r['experiment']=='knowledge_swapped' else np.nan))
D=pd.DataFrame(rows);sc=pd.read_csv(OUT/'scoring_leading/new_inference_scored.csv')
base=sc[(sc.experiment=='candidate_context')&(sc.condition=='related_order0')][['model','pair_id','parseable','correct']].rename(columns={'pair_id':'pair','correct':'no_card','parseable':'p0'})
card=sc[sc.experiment=='knowledge_provided'][['model','pair_id','parseable','correct']].rename(columns={'pair_id':'pair','correct':'card','parseable':'p1'})
neu=D[D.exp=='knowledge_neutral'][['model','pair','parse','correct']].rename(columns={'correct':'neutral','parse':'p2'});swp=D[D.exp=='knowledge_swapped'][['model','pair','parse','correct','follows_card']].rename(columns={'correct':'swapped_own_knowledge','parse':'p3'})
M=base.merge(card,on=['model','pair']).merge(neu,on=['model','pair']).merge(swp,on=['model','pair']);ok=M[M.p0&M.p1&M.p2&M.p3]
S=ok.groupby('model').agg(n=('pair','size'),no_card=('no_card','mean'),neutral_control=('neutral','mean'),correct_card=('card','mean'),swapped_card_answers_by_own_knowledge=('swapped_own_knowledge','mean'),swapped_card_follows_card=('follows_card','mean')).reset_index()
S['card_gain_vs_no_card']=S.correct_card-S.no_card;S['card_gain_vs_neutral']=S.correct_card-S.neutral_control;S['neutral_minus_no_card']=S.neutral_control-S.no_card
S.to_csv(OUT/'knowledge_controls_summary.csv',index=False);pd.set_option('display.width',250);print(S.round(3).to_string());print(S.drop(columns='model').mean().round(3).to_dict())
