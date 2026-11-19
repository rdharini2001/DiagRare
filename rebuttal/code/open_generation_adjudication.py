#!/usr/bin/env python3
"""Rule-based, alias-aware adjudication of open-ended (candidate-free) diagnoses. NOT clinician adjudication.
Each ontology disease has (a) EXACT aliases/synonyms/subtypes counted correct and (b) BROAD umbrella terms counted 'partial' (right
syndrome, insufficient specificity). Matching is whole-phrase, case-insensitive, on the model's listed diagnoses.
A 10% random sample of decisions is dumped for human spot-checking."""
import json,re,sys
from pathlib import Path
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'rebuttal/results'
A={ # disease: (exact aliases regex, broad regex)
'Acute_MI':(r'myocardial infarction|heart attack|\bSTEMI\b|\bNSTEMI\b|\bMI\b|acute coronary syndrome|\bACS\b',r'coronary artery disease|angina|ischemic heart|cardiac ischemia'),
'Addison_Disease':(r'addison|primary adrenal insufficiency|adrenal insufficiency|adrenal crisis|hypoadrenal|adrenocortical insufficiency',r'adrenal'),
'Allergic_Rhinitis':(r'allergic rhinitis|hay fever|seasonal allergies|pollinosis',r'rhinitis|allerg'),
'Ankylosing_Spondylitis':(r'ankylosing spondylitis|axial spondyloarthritis|spondyloarthr',r'spondyl|sacroiliitis|inflammatory back'),
'Aortic_Dissection':(r'aortic dissection|dissecting aort',r'aortic aneurysm|aortic'),
'Aplastic_Anemia':(r'aplastic anemia|aplastic anaemia|bone marrow failure',r'pancytopenia|myelodysplas'),
'Asthma':(r'asthma',r'bronchospasm|reactive airway|bronchoconstriction'),
'B12_Deficiency':(r'b12|cobalamin|pernicious anemia|megaloblastic|vitamin b',r'macrocytic|folate'),
'Bacterial_Meningitis':(r'bacterial meningitis|meningococc|pyogenic meningitis|meningitis',r'meningoencephalitis|encephalitis'),
'COPD':(r'copd|chronic obstructive|emphysema|chronic bronchitis',r'obstructive lung|airway disease'),
'Cardiac_Tamponade':(r'tamponade',r'pericardial effusion|pericarditis'),
'Chronic_Lymphocytic_Leukemia':(r'chronic lymphocytic leuk|\bCLL\b',r'leukemia|leukaemia|lymphoproliferative|lymphoma'),
'Cluster_Headache':(r'cluster headache|trigeminal autonomic|autonomic cephalalgia|hemicrania',r'cephalalgia'),
'Community_Pneumonia':(r'pneumonia',r'lower respiratory tract infection|chest infection|bronchopneumonia|pulmonary infection'),
'Cushing_Syndrome':(r'cushing|hypercortisol',r'adrenal (adenoma|tumor)|steroid excess'),
'Diabetes_Insipidus':(r'diabetes insipidus|\bDI\b',r'polyuria|polydipsia|psychogenic polydipsia'),
'Epilepsy':(r'epilep|seizure disorder|convulsive disorder|status epilepticus',r'seizure|syncope'),
'Gout':(r'\bgout|gouty|crystal arthropathy|urate arthropathy|hyperuricemia',r'pseudogout|septic arthritis|crystal'),
'Guillain_Barre':(r'guillain|acute inflammatory demyelinating|\bAIDP\b|\bCIDP\b|miller fisher',r'polyneuropathy|demyelinating|ascending paralysis'),
'HUS':(r'hemolytic.uremic|\bHUS\b|thrombotic microangiopathy|\bTMA\b|\bSTEC\b',r'hemolytic anemia|microangiopathic'),
'Heart_Failure':(r'heart failure|cardiac failure|\bCHF\b|\bHFrEF\b|\bHFpEF\b|cardiomyopathy',r'cardiac|cardiogenic'),
'Hemophilia_A':(r'hemophilia|haemophilia|factor viii',r'bleeding disorder|coagulopathy|clotting disorder|von willebrand'),
'Hodgkin_Lymphoma':(r'hodgkin|\bHL\b',r'lymphoma|lymphoproliferative'),
'Hyperthyroidism':(r'hyperthyroid|graves|thyrotoxic|toxic (multinodular )?goiter|toxic adenoma',r'thyroid'),
'Hypothyroidism':(r'hypothyroid|hashimoto|myxedema|myxoedema',r'thyroid'),
'ITP':(r'immune thrombocytopenic|idiopathic thrombocytopenic|immune thrombocytopenia|\bITP\b',r'thrombocytopeni'),
'Idiopathic_Intracranial_HTN':(r'idiopathic intracranial hypertension|pseudotumor cerebri|benign intracranial hypertension|\bIIH\b',r'intracranial hypertension|raised intracranial|increased intracranial'),
'Insulinoma':(r'insulinoma|insulin.secreting|islet cell tumou?r',r'hypoglyc|pancreatic (neuroendocrine|tumou?r)|neuroendocrine tumou?r'),
'Iron_Deficiency_Anemia':(r'iron.deficiency|\bIDA\b',r'microcytic anemia|microcytic anaemia|anemia|anaemia'),
'Lung_Cancer':(r'lung (cancer|carcinoma|adenocarcinoma|neoplasm|malignancy)|bronchogenic|pulmonary (carcinoma|malignancy)|small cell|non.small cell',r'lung (mass|nodule|tumou?r)|pulmonary (mass|nodule)|malignancy|carcinoma|cancer'),
'Metabolic_Syndrome':(r'metabolic syndrome|syndrome x|insulin resistance syndrome',r'insulin resistance|obesity'),
'Migraine':(r'migraine',r'headache'),
'Multiple_Sclerosis':(r'multiple sclerosis|\bMS\b|optic neuritis|demyelinating',r'demyelinat'),
'Myasthenia_Gravis':(r'myasthenia|lambert.eaton',r'neuromuscular|ocular myasthenia'),
'Osteoarthritis':(r'osteoarthritis|\bOA\b|degenerative joint|degenerative arthritis|osteoarthrosis',r'arthritis|arthrosis'),
'Parkinsons_Disease':(r'parkinson',r'parkinsonism|movement disorder|essential tremor'),
'Pericarditis':(r'pericarditis|myopericarditis',r'pericardial|myocarditis'),
'Pheochromocytoma':(r'pheochromocytoma|phaeochromocytoma|paraganglioma|catecholamine',r'adrenal (tumou?r|mass)|hypertensive crisis'),
'Polycythemia_Vera':(r'polycythemia|polycythaemia|myeloproliferative',r'erythrocytosis|\bMPN\b'),
'Primary_Hyperaldosteronism':(r'hyperaldosteron|conn.s|aldosteronism|aldosterone.producing',r'adrenal adenoma|secondary hypertension|hypokalemia'),
'Psoriatic_Arthritis':(r'psoriatic arthritis|psoriatic arthropathy|\bPsA\b',r'spondyloarthr|arthritis'),
'Pulmonary_Embolism':(r'pulmonary embol|\bPE\b|venous thromboembol|\bVTE\b|deep vein thrombosis with',r'thromboembol|pulmonary infarct|\bDVT\b'),
'Pulmonary_Fibrosis':(r'pulmonary fibrosis|\bIPF\b|interstitial lung disease|\bILD\b|usual interstitial',r'interstitial|fibrosing|lung fibrosis'),
'Reactive_Arthritis':(r'reactive arthritis|reiter|post.infectious arthritis|postinfectious arthritis',r'spondyloarthr|arthritis'),
'Rheumatoid_Arthritis':(r'rheumatoid arthritis|\bRA\b',r'inflammatory arthritis|polyarthritis|arthritis'),
'Sarcoidosis':(r'sarcoid|loefgren|lofgren',r'granulomatous'),
'Sarcoidosis_Pulmo':(r'sarcoid|loefgren|lofgren',r'granulomatous|interstitial lung|hilar lymphadenopathy'),
'Sickle_Cell_Disease':(r'sickle',r'hemoglobinopathy|haemoglobinopathy|hemolytic anemia|vaso.occlusive'),
'Sjogren_Syndrome':(r'sj.gren|sicca',r'dry eye|xerostomia|keratoconjunctivitis'),
'Stroke_Ischemic':(r'ischemic stroke|ischaemic stroke|cerebral infarct|cerebrovascular accident|\bCVA\b|\bstroke\b|\bTIA\b|transient ischemic',r'cerebrovascular|intracranial hemorrhage'),
'Subarachnoid_Hemorrhage':(r'subarachnoid|ruptured (cerebral )?aneurysm|berry aneurysm|aneurysmal',r'intracranial (hemorrhage|bleed)|cerebral hemorrhage|thunderclap'),
'Systemic_Lupus':(r'lupus|\bSLE\b',r'connective tissue|autoimmune'),
'T1_Diabetes':(r'type 1 diabetes|type i diabetes|\bT1DM?\b|insulin.dependent|diabetic ketoacidosis|\bDKA\b|\bLADA\b',r'diabetes|ketoacidosis|hyperglyc'),
'T2_Diabetes':(r'type 2 diabetes|type ii diabetes|\bT2DM?\b|adult.onset diabetes|diabetes mellitus',r'diabetes|hyperglyc|prediabetes|insulin resistance'),
'TTP':(r'thrombotic thrombocytopenic|\bTTP\b|thrombotic microangiopathy|\bTMA\b|pentad',r'microangiopathic|hemolytic anemia|thrombocytopeni'),
'Tension_Headache':(r'tension.type|tension headache|tension\-type',r'headache|cephalalgia'),
'Thalassemia_Trait':(r'thalass',r'microcytic|hemoglobinopathy|haemoglobinopathy'),
'Tuberculosis':(r'tubercul|\bTB\b|pott|mycobacter',r'granulomatous|fungal infection'),
'Infective_Endocarditis':(r'endocarditis',r'bacteremia|sepsis|valv'),
'Pneumothorax':(r'pneumothorax|collapsed lung|tension pneumo',r'pleural|lung collapse|spontaneous'),
'Trigeminal_Neuralgia':(r'trigeminal neuralgia|tic douloureux',r'neuralgia|facial pain'),
'Diabetic_Ketoacidosis':(r'ketoacidosis|\bDKA\b|hyperglycemic crisis',r'diabetic|metabolic acidosis'),
'Anaphylaxis':(r'anaphyla',r'allergic reaction|angioedema|urticaria|hypersensitivity'),
'Disseminated_Intravascular_Coagulation':(r'disseminated intravascular|\bDIC\b|consumptive coagulopathy',r'coagulopathy|bleeding disorder'),
'Acute_Kidney_Injury':(r'acute kidney injury|\bAKI\b|acute renal failure|acute tubular necrosis|\bATN\b|prerenal',r'renal failure|renal insufficiency|kidney injury|kidney failure'),
'Nephrotic_Syndrome':(r'nephrotic|minimal change|membranous|focal segmental',r'proteinuria|glomerul'),
'Polycystic_Kidney_Disease':(r'polycystic kidney|\bADPKD\b|\bPKD\b',r'renal cyst|kidney cyst|cystic'),
'Renal_Cell_Carcinoma':(r'renal cell|kidney (cancer|carcinoma|tumou?r)|renal (carcinoma|mass|neoplasm|tumou?r)|hypernephroma',r'renal|kidney|carcinoma|malignancy'),
'Glomerulonephritis':(r'glomerulonephritis|nephritic|iga nephropathy|post.streptococcal|\bPSGN\b|lupus nephritis',r'nephritis|glomerul|kidney disease'),
'Nephrolithiasis':(r'nephrolith|kidney stone|renal (stone|calcul|colic)|urolith|ureteral (stone|calcul)|ureterolith|renal calculi',r'urinary tract|ureteric|obstruct|colic'),
'Acute_Appendicitis':(r'appendicitis|appendiceal',r'appendix|right lower quadrant|peritonitis|acute abdomen'),
'Peptic_Ulcer_Disease':(r'peptic ulcer|duodenal ulcer|gastric ulcer|\bPUD\b|h\.? pylori|helicobacter',r'gastritis|dyspepsia|gerd|gastroesophageal|ulcer'),
'Ulcerative_Colitis':(r'ulcerative colitis|\bUC\b',r'inflammatory bowel|\bIBD\b|colitis'),
'Crohn_Disease':(r'crohn',r'inflammatory bowel|\bIBD\b|enteritis|ileitis'),
'Acute_Pancreatitis':(r'pancreatitis',r'pancrea'),
'Acute_Cholecystitis':(r'cholecystitis|biliary colic|cholelithiasis|gallstone|cholangitis|choledocholithiasis',r'biliary|gallbladder|hepatobiliary'),
'Psoriasis':(r'psoria',r'dermatitis|eczema|parapsoriasis|plaque'),
'Cellulitis':(r'cellulitis|erysipelas|skin and soft tissue|soft tissue infection|necrotizing fasciitis',r'abscess|skin infection|dermatitis|infection'),
'Allergic_Contact_Dermatitis':(r'contact dermatitis|allergic dermatitis|rhus|poison ivy',r'dermatitis|eczema|rash'),
'Herpes_Zoster':(r'zoster|shingles|herpes',r'varicella|neuralgia|vesicular'),
'Stevens_Johnson_Syndrome':(r'stevens.johnson|\bSJS\b|toxic epidermal|\bTEN\b',r'drug (eruption|reaction)|erythema multiforme|bullous'),
'Erythema_Multiforme':(r'erythema multiforme',r'drug (eruption|reaction|rash)|target lesion|urticaria'),
'Metabolic_Syndrome ':(r'$^',r'$^')}
def clean(x):return re.sub(r'\s*\(\s*[\d.]+\s*\)\s*$','',str(x)).strip()
def judge(target,ans):
    e,b=A[target];a=clean(ans)
    if re.search(e,a,flags=re.I):return 'exact'
    grp={'Sarcoidosis','Sarcoidosis_Pulmo'} if target in('Sarcoidosis','Sarcoidosis_Pulmo') else {target}
    if any(re.search(A[d][0],a,flags=re.I) for d in A if d not in grp and A[d][0]!='$^'):return 'other'  # names a different ontology disease
    if re.search(b,a,flags=re.I):return 'broad'
    return 'other'
def leading(raw):
    t=re.sub(r'^```(?:json)?\s*','',raw.strip(),flags=re.I)
    try:v,_=json.JSONDecoder().raw_decode(t);return v if isinstance(v,dict) else {}
    except Exception:return {}
rows=[]
for f in sorted(OUT.glob('*__new_inference.jsonl')):
    for line in f.read_text().splitlines():
        try:r=json.loads(line)
        except ValueError:continue
        if r['experiment']!='open_generation':continue
        obj=leading(r['raw_response']);ans=obj.get('diagnoses',[]);ans=[x for x in ans if isinstance(x,str)] if isinstance(ans,list) else []
        js=[judge(r['target'],a) for a in ans[:3]]
        rows.append(dict(model=r['model'],pair_id=r['pair_id'],target=r['target'],answers=' | '.join(ans[:3]),n_answers=len(ans),top1=js[0] if js else 'none',any_top3_exact=int('exact' in js),any_top3_exact_or_broad=int(any(j in('exact','broad') for j in js)),top1_exact=int(bool(js) and js[0]=='exact'),top1_exact_or_broad=int(bool(js) and js[0] in('exact','broad'))))
d=pd.DataFrame(rows);d.to_csv(OUT/'open_generation_rule_adjudicated.csv',index=False)
s=d.groupby('model').agg(n=('pair_id','size'),answered=('n_answers',lambda x:(x>0).mean()),top1_exact=('top1_exact','mean'),top1_exact_or_broad=('top1_exact_or_broad','mean'),top3_exact=('any_top3_exact','mean'),top3_exact_or_broad=('any_top3_exact_or_broad','mean')).reset_index();s.to_csv(OUT/'open_generation_rule_summary.csv',index=False);print(s.round(3).to_string())
d.sample(frac=.1,random_state=5).sort_values(['model','pair_id']).to_csv(OUT/'open_generation_spotcheck_sample.csv',index=False)
