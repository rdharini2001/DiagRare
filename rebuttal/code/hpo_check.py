#!/usr/bin/env python3
"""Cross-check ontology relations against HPO disease-phenotype annotations (phenotype.hpoa, release 2026-09-02) for diseases with OMIM/ORPHA entries.
Evidence of presence (and frequency) for 'required'/'supporting' relations, and *contradictions* where HPO annotates an 'excluding' finding in the disease.
Token->HPO term mapping is lexical (term names + exact synonyms, descendants included) so coverage is partial; absence of annotation is not evidence of absence."""
import re,collections
import numpy as np
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[2];H=ROOT/'rebuttal/literature/hpo';DATA=ROOT/'rebuttal/data'
# --- parse hp.obo
terms={};cur=None
for ln in open(H/'hp.obo'):
    ln=ln.rstrip('\n')
    if ln=='[Term]':cur={'id':None,'name':'','syn':[],'is_a':[]};terms_list=cur;continue
    if cur is None:continue
    if ln.startswith('id: '):cur['id']=ln[4:];terms[cur['id']]=cur
    elif ln.startswith('name: '):cur['name']=ln[6:]
    elif ln.startswith('synonym: ') and 'EXACT' in ln:
        m=re.match(r'synonym: "(.*)" EXACT',ln)
        if m:cur['syn'].append(m.group(1))
    elif ln.startswith('is_a: '):cur['is_a'].append(ln[6:].split(' ')[0])
    elif ln.startswith('is_obsolete: true'):cur['obs']=True
children=collections.defaultdict(set)
for t in terms.values():
    for p in t['is_a']:children[p].add(t['id'])
def descendants(t):
    out={t};st=[t]
    while st:
        x=st.pop()
        for c in children.get(x,()):
            if c not in out:out.add(c);st.append(c)
    return out
hpoa=pd.read_csv(H/'phenotype.hpoa',sep='\t',comment='#',dtype=str).fillna('')
hpoa=hpoa[(hpoa.aspect=='P')&(hpoa.qualifier!='NOT')]
FREQ={'HP:0040280':1.0,'HP:0040281':.9,'HP:0040282':.55,'HP:0040283':.17,'HP:0040284':.025,'HP:0040285':0.}
def fval(s):
    if not s:return None
    if s in FREQ:return FREQ[s]
    m=re.match(r'(\d+)/(\d+)$',s)
    if m and int(m.group(2))>0:return int(m.group(1))/int(m.group(2))
    m=re.match(r'([\d.]+)%$',s)
    return float(m.group(1))/100 if m else None
# disease -> HPOA disease-name patterns (searched case-insensitively in disease_name; ORPHANET and OMIM entries)
PAT={'Addison_Disease':['^addison disease','autoimmune.*adrenal','primary adrenal insufficiency'],'Aplastic_Anemia':['aplastic anemia'],'Cushing_Syndrome':["cushing"],'Hemophilia_A':['hemophilia a'],'TTP':['thrombotic thrombocytopenic purpura'],'HUS':['hemolytic.uremic'],'Sickle_Cell_Disease':['sickle cell (anemia|disease)'],'Guillain_Barre':['guillain'],'Myasthenia_Gravis':['myasthenia gravis'],'Sarcoidosis':['^sarcoidosis'],'Sjogren_Syndrome':['sj.gren'],'Chronic_Lymphocytic_Leukemia':['chronic lymphocytic leukemia'],'Hodgkin_Lymphoma':['hodgkin'],'Polycythemia_Vera':['polycythemia vera'],'ITP':['immune thrombocytop'],'Pheochromocytoma':['pheochromocytoma'],'Primary_Hyperaldosteronism':['primary aldosteronism|hyperaldosteronism'],'Insulinoma':['insulinoma'],'Diabetes_Insipidus':['diabetes insipidus'],'Polycystic_Kidney_Disease':['polycystic kidney disease.*dominant|autosomal dominant polycystic'],'Stevens_Johnson_Syndrome':['stevens.johnson'],'Erythema_Multiforme':['erythema multiforme'],'Anaphylaxis':['anaphylaxis'],'Disseminated_Intravascular_Coagulation':['disseminated intravascular'],'Trigeminal_Neuralgia':['trigeminal neuralgia'],'Idiopathic_Intracranial_HTN':['intracranial hypertension'],'Pulmonary_Fibrosis':['idiopathic pulmonary fibrosis|pulmonary fibrosis'],'Ankylosing_Spondylitis':['ankylosing spondylitis'],'Psoriatic_Arthritis':['psoriatic arthritis'],'Rheumatoid_Arthritis':['rheumatoid arthritis'],'Systemic_Lupus':['systemic lupus'],'Multiple_Sclerosis':['multiple sclerosis'],'Parkinsons_Disease':['^parkinson disease|parkinson disease, late'],'Epilepsy':['^epilepsy'],'Thalassemia_Trait':['thalassemia'],'B12_Deficiency':['cobalamin|pernicious'],'Iron_Deficiency_Anemia':['iron.deficiency anemia'],'Nephrotic_Syndrome':['nephrotic syndrome'],'Hyperthyroidism':['graves'],'Hypothyroidism':['hypothyroidism'],'Gout':['^gout'],'Infective_Endocarditis':['endocarditis'],'Cluster_Headache':['cluster headache'],'Migraine':['^migraine'],'Pericarditis':['pericarditis'],'Cardiac_Tamponade':['tamponade'],'Aortic_Dissection':['aortic dissection'],'Pneumothorax':['pneumothorax'],'Crohn_Disease':['crohn'],'Ulcerative_Colitis':['ulcerative colitis'],'Psoriasis':['^psoriasis'],'Herpes_Zoster':['zoster'],'Renal_Cell_Carcinoma':['renal cell carcinoma'],'Asthma':['^asthma'],'Glomerulonephritis':['glomerulonephritis'],'Lung_Cancer':['lung cancer|small cell lung|non-small cell lung'],'Tuberculosis':['tuberculosis'],'T1_Diabetes':['type 1 diabetes|diabetes mellitus, type 1|insulin-dependent'],'T2_Diabetes':['diabetes mellitus, type 2|type 2 diabetes'],'Acute_MI':['myocardial infarction'],'Heart_Failure':['^heart failure|cardiac failure'],'COPD':['chronic obstructive'],'Allergic_Rhinitis':['allergic rhinitis'],'Osteoarthritis':['^osteoarthritis'],'Reactive_Arthritis':['reactive arthritis'],'Metabolic_Syndrome':['metabolic syndrome']}
d2ids={}
names=hpoa[['database_id','disease_name']].drop_duplicates()
for d,pats in PAT.items():
    ids=set()
    for p in pats:
        ids|=set(names[names.disease_name.str.contains(p,case=False,regex=True)].database_id)
    d2ids[d]=ids
# token -> HPO terms
GEN={'elevated','high','low','positive','negative','normal','present','absent','acute','chronic','recent','sudden','severe','mild','with','or','and','of','on','to','in','the','history','no','not','over','under','after','before','without','pattern','symptoms','findings','markedly','early','onset','new','preceding','involvement','distribution'}
MANUAL={'low_TSH':['Decreased circulating thyroid-stimulating hormone concentration','Low TSH'],'elevated_TSH':['Elevated circulating thyroid-stimulating hormone concentration'],'high_MCV':['Macrocytic anemia','Macrocytosis','Elevated mean corpuscular volume'],'low_MCV':['Microcytic anemia','Microcytosis','Decreased mean corpuscular volume'],'weight_loss_rapid':['Weight loss'],'dyspnea':['Dyspnea'],'diaphoresis':['Hyperhidrosis'],'itching':['Pruritus'],'pallor':['Pallor'],'fever':['Fever'],'tachycardia':['Tachycardia'],'bradycardia':['Bradycardia'],'hypotension':['Hypotension'],'hypertension':['Hypertension'],'headache':['Headache'],'seizures':['Seizure'],'fatigue':['Fatigue'],'nausea':['Nausea'],'vomiting':['Vomiting'],'hematuria':['Hematuria'],'edema':['Edema'],'hyperpigmentation':['Hyperpigmentation of the skin'],'low_cortisol':['Decreased circulating cortisol level','Adrenal insufficiency'],'elevated_cortisol':['Increased circulating cortisol level'],'central_obesity':['Truncal obesity'],'moon_face':['Moon facies'],'purple_striae':['Abdominal striae','Striae distensae'],'splenomegaly':['Splenomegaly'],'thrombocytopenia':['Thrombocytopenia'],'pancytopenia':['Pancytopenia'],'petechiae':['Petechiae'],'bleeding':['Abnormal bleeding'],'recurrent_infections':['Recurrent infections'],'hemarthrosis':['Hemarthrosis'],'prolonged_aPTT':['Prolonged partial thromboplastin time'],'jaundice':['Jaundice'],'ptosis':['Ptosis'],'diplopia':['Diplopia'],'polyuria':['Polyuria'],'polydipsia':['Polydipsia'],'hypokalemia':['Hypokalemia'],'hyperkalemia':['Hyperkalemia'],'hyponatremia':['Hyponatremia'],'hypernatremia':['Hypernatremia'],'hypercalcemia':['Hypercalcemia'],'hypoglycemia':['Hypoglycemia'],'fasting_hypoglycemia':['Hypoglycemia'],'metabolic_acidosis':['Metabolic acidosis'],'proteinuria_heavy':['Proteinuria','Nephrotic-range proteinuria'],'hypoalbuminemia':['Hypoalbuminemia'],'hyperlipidemia':['Hyperlipidemia'],'peripheral_edema':['Peripheral edema'],'uveitis':['Uveitis'],'lymphocytosis':['Lymphocytosis'],'dry_eyes':['Dry eye','Keratoconjunctivitis sicca'],'dry_mouth':['Xerostomia'],'wheezing':['Wheezing'],'cough':['Cough'],'chronic_cough':['Chronic cough','Cough'],'hemoptysis':['Hemoptysis'],'clubbing':['Clubbing','Finger clubbing'],'hypertension_history':['Hypertension'],'obesity':['Obesity'],'weight_gain':['Weight gain'],'constipation':['Constipation'],'diarrhea':['Diarrhea'],'hair_loss':['Alopecia'],'dry_skin':['Dry skin'],'cold_intolerance':['Cold intolerance'],'heat_intolerance':['Heat intolerance'],'tremor':['Tremor'],'resting_tremor':['Rest tremor'],'rigidity':['Rigidity'],'bradykinesia':['Bradykinesia'],'anxiety':['Anxiety'],'exophthalmos':['Exophthalmos'],'papilledema':['Papilledema'],'vision_changes':['Visual impairment','Blurred vision'],'neck_stiffness':['Nuchal rigidity'],'photophobia':['Photophobia'],'aura':['Aura'],'peripheral_neuropathy':['Peripheral neuropathy'],'numbness':['Numbness','Hypoesthesia'],'areflexia':['Areflexia'],'hyperreflexia':['Hyperreflexia'],'ascending_weakness':[],'muscle_weakness':['Muscle weakness'],'proximal_muscle_weakness':['Proximal muscle weakness'],'dysphagia':['Dysphagia'],'joint_pain':['Arthralgia'],'psoriasis':['Psoriasiform dermatitis','Psoriasis'],'dactylitis':['Dactylitis'],'enthesitis':['Enthesitis'],'malar_rash':['Malar rash'],'oral_ulcers':['Oral ulcer'],'photosensitivity':['Photosensitivity'],'serositis':['Serositis'],'cytopenias':['Cytopenia'],'renal_failure':['Renal insufficiency','Renal failure'],'schistocytes':['Schistocytosis'],'microangiopathic_hemolytic_anemia':['Microangiopathic hemolytic anemia'],'hemolytic_anemia':['Hemolytic anemia'],'chronic_hemolytic_anemia':['Hemolytic anemia'],'sickle_cells':['Sickle cell'],'target_cells':['Target cells'],'pica':['Pica'],'glossitis':['Glossitis'],'koilonychia':['Koilonychia'],'brittle_nails':['Brittle nails'],'nail_pitting':['Nail pits','Pitted nails'],'pruritus':['Pruritus'],'night_sweats':['Night sweats'],'B_symptoms':['Fever','Night sweats','Weight loss'],'painless_lymphadenopathy':['Lymphadenopathy'],'erythema_nodosum':['Erythema nodosum'],'conjunctival_injection':['Conjunctivitis','Conjunctival hyperemia'],'urticaria_acute':['Urticaria'],'angioedema':['Angioedema'],'oliguria':['Oliguria'],'hyperuricemia':['Hyperuricemia'],'high_uric_acid':['Hyperuricemia'],'proteinuria_mild':['Proteinuria'],'hypoxemia':['Hypoxemia'],'plethora':['Plethora'],'glycosuria':['Glycosuria'],'ketoacidosis':['Ketoacidosis'],'high_fasting_glucose':['Hyperglycemia'],'hyperglycemia':['Hyperglycemia'],'acanthosis_nigricans':['Acanthosis nigricans'],'fatty_liver':['Hepatic steatosis'],'easy_bruising':['Easy bruisability'],'bruising':['Easy bruisability'],'family_history':[],'male':[],'female':[],'young_age':[],'age_over_50':[]}
by_label=collections.defaultdict(set)
for t in terms.values():
    if t.get('obs'):continue
    by_label[t['name'].lower()].add(t['id'])
    for s in t['syn']:by_label[s.lower()].add(t['id'])
REJECT={('ITP','splenomegaly'):'HPO record ORPHA:444463 is not the classic ITP entity; mapping unverified',('Migraine','fever'):'HPO fever annotations come from familial hemiplegic migraine subtype records'}
def token_terms(tok):
    if re.match(r'(no|normal|absent|negative)_',tok):return set()
    labels=MANUAL.get(tok)
    out=set()
    if labels is not None:
        for l in labels:out|=by_label.get(l.lower(),set())
        return out
    ws=[w for w in tok.split('_') if w.lower() not in GEN]
    for lab in [' '.join(ws).lower(),' '.join(ws).lower().replace('ia','ia')]:
        out|=by_label.get(lab,set())
    return out
prof=pd.read_csv(DATA/'full_profile_audit.csv').drop(columns=[c for c in ['hpo_disease_records','hpo_term_mapped','hpo_annotated','hpo_max_frequency','hpo_records_with_annotation','hpo_evidence','hpo_signal','flag_origin'] if c in pd.read_csv(DATA/'full_profile_audit.csv',nrows=1).columns]);rows=[]
desc_cache={}
def anc(ts):
    s=set()
    for t in ts:
        if t not in desc_cache:desc_cache[t]=descendants(t)
        s|=desc_cache[t]
    return s
ann=hpoa.groupby('database_id')
for _,r in prof.iterrows():
    ids=d2ids.get(r.disease,set());tt=token_terms(r.finding);res=dict(relation_id=r.relation_id,hpo_disease_records=len(ids),hpo_term_mapped=bool(tt),hpo_annotated=False,hpo_max_frequency=None,hpo_records_with_annotation=0,hpo_evidence='')
    if (r.disease,r.finding) in REJECT:tt=set()
    if ids and tt:
        want=anc(tt);hits=[];fr=[]
        for i in ids:
            g=ann.get_group(i) if i in ann.groups else None
            if g is None:continue
            m=g[g.hpo_id.isin(want)]
            if len(m):
                hits.append(i);fs=[fval(x) for x in m.frequency if fval(x) is not None]
                if fs:fr.append(max(fs))
        if hits:
            res.update(hpo_annotated=True,hpo_records_with_annotation=len(hits),hpo_max_frequency=max(fr) if fr else None,hpo_evidence=';'.join(hits[:4]))
    rows.append(res)
x=pd.DataFrame(rows);m=prof.merge(x,on='relation_id')
def sig(r):
    if not r.hpo_annotated or pd.isna(r.hpo_max_frequency):return 'none'
    f=r.hpo_max_frequency
    if r.original_relation=='excluding':return 'excluding_hpo_absent_0pct' if f==0 else ('excluding_hpo_rare_lt10pct' if f<.10 else 'excluding_hpo_present_ge10pct')
    return 'hpo_very_frequent_ge80pct' if f>=.80 else 'hpo_lt80pct'
m['hpo_signal']=m.apply(sig,axis=1)
m['flag_origin']=np.where((m.reviewer_flag!='none')&m.hpo_signal.isin(['hpo_lt80pct','excluding_hpo_present_ge10pct']),'reviewer+HPO',np.where(m.reviewer_flag!='none','reviewer',np.where(m.hpo_signal.isin(['hpo_lt80pct','excluding_hpo_present_ge10pct']),'HPO_only','none')))
hp=m.flag_origin=='HPO_only'
m.loc[hp,'reviewer_flag']=np.where(m.loc[hp,'original_relation']=='excluding','exclusion_overstated','required_overstated')
m.loc[hp,'reviewer_rationale']='HPO disease-phenotype annotation gives frequency '+m.loc[hp,'hpo_max_frequency'].round(2).astype(str)+' (category midpoint or reported ratio); not independently clinician-reviewed'
m.loc[hp,'rationale_source']='HPO'
m.loc[(m.original_relation=='supporting')&hp,'reviewer_flag']='none'; m.loc[(m.original_relation=='supporting')&hp,'flag_origin']='none'
m.loc[m.flag_origin!='none','audit_status']='flagged_'+m.loc[m.flag_origin!='none','reviewer_flag']
m.to_csv(DATA/'full_profile_audit.csv',index=False)
fl=m[m.flag_origin!='none'][['disease','finding','original_relation','reviewer_flag','reviewer_rationale','rationale_source','rationale_term_found_in_source','flag_origin','hpo_signal','hpo_max_frequency']].rename(columns={'reviewer_flag':'issue_class'});fl.to_csv(DATA/'profile_corrections.csv',index=False)
print('FLAG ORIGIN',m.flag_origin.value_counts().to_dict())
print('HPO agreement among reviewer flags that have HPO signal:',pd.crosstab(m[(m.flag_origin.isin(['reviewer','reviewer+HPO']))&(m.hpo_signal!='none')].original_relation,m[(m.flag_origin.isin(['reviewer','reviewer+HPO']))&(m.hpo_signal!='none')].hpo_signal))
m.to_csv(DATA/'full_profile_audit_hpo.csv',index=False)
cov=m[m.hpo_disease_records>0]
print('diseases with HPOA record:',cov.disease.nunique(),'/ 82; relations on those diseases:',len(cov))
print('mapped tokens among them:',cov.hpo_term_mapped.mean().round(3))
print(cov.groupby('original_relation').agg(n=('relation_id','size'),mapped=('hpo_term_mapped','sum'),annotated=('hpo_annotated','sum')))
ex=cov[(cov.original_relation=='excluding')&cov.hpo_annotated];print('EXCLUDING relations annotated in HPO (contradiction candidates):',len(ex));print(ex[['disease','finding','hpo_max_frequency','hpo_evidence']].to_string())
