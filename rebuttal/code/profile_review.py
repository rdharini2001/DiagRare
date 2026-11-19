#!/usr/bin/env python3
"""AI-assisted literature audit of all 888 required/supporting/excluding relations.
NOT independent clinician validation. Three evidence layers per relation:
 (1) lexical_mention: the finding (with synonyms) co-occurs in a short window in retrieved source text for that disease;
 (2) reviewer_flag: category judgment from data/profile_exceptions.txt (class + rationale + cited source key);
 (3) rationale_verified: a regex for the rationale's key term is found in the cited source's retrieved text.
Sources are PubMed/PMC records resolved by resolve_sources.py. Text availability differs (PMC full text vs abstract); a missing
lexical mention in abstract-only sources is uninformative and is labelled as such."""
import re,sys
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[2];LIT=ROOT/'rebuttal/literature';TXT=LIT/'source_text';DATA=ROOT/'rebuttal/data'
src=pd.read_csv(LIT/'resolved_sources.csv').set_index('key')
text={k:(TXT/f'{k}.txt').read_text() for k in src.index}
prof=pd.read_csv(DATA/'full_profile_validation.csv')
# disease -> sources
d2s={}
for k,r in src.iterrows():
    for d in str(r.diseases).split(';'):d2s.setdefault(d,[]).append(k)
SYN={'RLQ':'right lower quadrant|RLQ','LLQ':'left lower quadrant|LLQ','RUQ':'right upper quadrant|RUQ','LUQ':'left upper quadrant|LUQ','CXR':'chest radiograph|x-ray|CXR','PND':'paroxysmal nocturnal','BNP':'natriuretic|BNP','MCV':'mean corpuscular|MCV|microcyt|macrocyt','RF':'rheumatoid factor|RF','AFB':'acid.fast|AFB|smear','PPD':'tuberculin|PPD|IGRA','ACE':'angiotensin.converting|ACE','DIP':'distal interphalangeal|DIP','ROM':'range of motion','ESR':'sedimentation|ESR','TSH':'TSH|thyrotropin|thyroid.stimulating','MSU':'monosodium urate|MSU|urate','ANA':'antinuclear|ANA','HDL':'HDL','LDH':'lactate dehydrogenase|LDH','D_dimer':'D.dimer','CSF':'cerebrospinal|CSF','CT':'computed tomograph|CT','EEG':'EEG|electroencephal','HbA1c':'HbA1c|glycated|A1c','HSV':'HSV|herpes simplex','aPTT':'aPTT|activated partial','JAK2':'JAK2','ST_elevation':'ST.elevation|ST.segment elevation|STEMI','hypotension':'hypotens','dyspnea':'dyspn|breathless|shortness of breath','diaphoresis':'diaphore|sweat','itching':'itch|prurit','edema':'edema|oedema','wheezing':'wheez','nausea':'nause','hematuria':'haematuria|hematuria','hemoptysis':'haemoptysis|hemoptysis','anemia':'an[a]?emia','hemolytic':'h[a]?emolytic','thrombocytopenia':'thrombocytopeni','fever':'fever|febrile|pyrexia','tachycardia':'tachycardi','pallor':'pallor|pale','fatigue':'fatigue|tired','headache':'headache','seizures':'seizure','ptosis':'ptosis','diplopia':'diplopia|double vision','proteinuria':'proteinuria','hypokalemia':'hypokal[a]?emia','hyperkalemia':'hyperkal[a]?emia','hyponatremia':'hyponatr[a]?emia','hypernatremia':'hypernatr[a]?emia','hypercalcemia':'hypercalc[a]?emia','glycosuria':'glycosuria|glucosuria','polyuria':'polyuria','polydipsia':'polydipsia|thirst','weight':'weight'}
STOP={'elevated','high','low','positive','negative','normal','present','absent','acute','chronic','recent','sudden','severe','mild','with','or','and','of','on','to','in','the','history','no','not','over','under','after','before','without','pattern','symptoms','findings','markedly','early','onset','new','preceding','involvement','distribution','signs','sign','test','case','during','between','from','at','by','as','rapid','persistent','typical','atypical','present','free'}
def words(tok):return [w for w in tok.split('_') if w.lower() not in STOP and len(w)>1]
def pattern_for(tok):
    if tok in SYN:return SYN[tok]
    ws=words(tok)
    if not ws:ws=tok.split('_')
    return ws
def mention(tok,txt):
    pat=pattern_for(tok)
    if isinstance(pat,str):
        m=re.search(pat,txt,flags=re.I)
        return (txt[max(0,m.start()-100):m.end()+100],True) if m else ('',False)
    regs=[SYN.get(w,re.escape(w[:max(5,min(len(w),7))])) for w in pat]
    first=re.compile(regs[0],re.I)
    for m in first.finditer(txt):
        win=txt[max(0,m.start()-250):m.end()+250]
        if all(re.search(r,win,re.I) for r in regs[1:]):return win[150:450] if len(win)>450 else win,True
    return '',False
exc={}
for ln in (DATA/'profile_exceptions.txt').read_text().splitlines():
    if not ln.strip():continue
    p=ln.split('|');exc[(p[0],p[1])]=dict(cls=p[2],why=p[3],src=p[4],regex='|'.join(p[5:]))
rows=[];unused=set(exc)
for _,r in prof.iterrows():
    d,f=r.disease,r.finding;keys=d2s.get(d,[]);found=False;snip='';fsrc=''
    for k in keys:
        s,ok=mention(f,text[k])
        if ok:found=True;snip=s;fsrc=k;break
    full=any(src.loc[k,'text_available']=='PMC full text' for k in keys)
    e=exc.get((d,f));unused.discard((d,f))
    row=dict(relation_id=r.relation_id,disease=d,organ_system=r.organ_system,finding=f,original_relation=r.original_relation,
        sources_retrieved=';'.join(keys),full_text_source_available=full,lexical_mention_in_source=found,mention_source=fsrc,mention_snippet=re.sub(r'\s+',' ',snip)[:300])
    if e:
        ks=e['src'];ok=bool(re.search(e['regex'],text.get(ks,''),flags=re.I)) if ks in text else False
        row.update(reviewer_flag=e['cls'],reviewer_rationale=e['why'],rationale_source=ks,rationale_source_title=src.loc[ks,'title'] if ks in src.index else '',rationale_term_found_in_source=ok)
    else:
        row.update(reviewer_flag='none',reviewer_rationale='',rationale_source='',rationale_source_title='',rationale_term_found_in_source='')
    row['independent_clinician_validation']='not performed'
    rows.append(row)
out=pd.DataFrame(rows)
def status(r):
    if r.reviewer_flag!='none':return 'flagged_'+r.reviewer_flag
    if r.lexical_mention_in_source:return 'unflagged_mentioned_in_source'
    return 'unflagged_no_mention_in_retrieved_text' if r.full_text_source_available else 'unflagged_unverifiable_abstract_only'
out['audit_status']=out.apply(status,axis=1)
out.to_csv(DATA/'full_profile_audit.csv',index=False)
corr=out[out.reviewer_flag!='none'][['disease','finding','original_relation','reviewer_flag','reviewer_rationale','rationale_source','rationale_term_found_in_source']].rename(columns={'reviewer_flag':'issue_class'});corr.to_csv(DATA/'profile_corrections.csv',index=False)
print('rows',len(out),'unused exceptions',sorted(unused))
print(out.audit_status.value_counts());print(out.groupby('original_relation').reviewer_flag.value_counts().unstack().fillna(0))
fl=out[out.reviewer_flag!='none'];print('rationale term verified in cited source:',fl.rationale_term_found_in_source.mean().round(3),'n',len(fl));print('diseases with ≥1 flag',fl.disease.nunique(),'/',out.disease.nunique())
print('diseases without any resolved source:',sorted(set(prof.disease)-set(d2s)))
