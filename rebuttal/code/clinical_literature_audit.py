#!/usr/bin/env python3
"""Source-backed audit flags and a target-case exclusion sensitivity.
This is AI-assisted literature checking, NOT independent clinician validation.
"""
from pathlib import Path
import sys,json
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'rebuttal/results';sys.path.insert(0,str(ROOT/'code_final_v2/analysis'))
from plackett_luce import Choice,fit_plackett_luce
refs=pd.read_csv(ROOT/'rebuttal/literature/verified_sources.csv').set_index('key')
audit=[]
def add(d,f,relation,source,association,category,action,section):
    s=refs.loc[source];audit.append({'disease':d,'finding':f,'original_relation':relation,'source_key':source,'source_title':s.title,'source_url':s.url,'section':section,'association_assessment':association,'category_assessment':category,'proposed_action_for_review':action,'audit_type':'AI-assisted literature audit','independent_clinician_validation':'not performed','audit_date':'2026-11-08'})
add('Acute_MI','ST_elevation','required','thygesen2018','supports an ST-elevation presentation','not universal across broad MI label','Qualify as an ST-elevation prototype or relax required interpretation.','MI definition and ECG categories')
add('Acute_MI','troponin_elevated','required','thygesen2018','supports myocardial injury assessment','static token omits dynamics, timing and ischemia context','Specify intended MI subtype and measurement context; do not infer universal token probability.','Definition of myocardial injury and MI')
add('Hypothyroidism','elevated_TSH','required','ata_tests','supports primary hypothyroidism with appropriate FT4','not universal across primary and secondary hypothyroidism','Qualify label as primary hypothyroidism or separate subtypes.','TSH TESTS; T4 TESTS')
add('Hypothyroidism','low_TSH','excluding','ata_tests','can coexist with pituitary-related hypothyroidism','broad exclusion needs subtype qualification','Do not generalize exclusion across hypothyroidism subtypes.','T4 TESTS')
add('Asthma','reversible_airflow_obstruction','required','gina_diagnosis','objective variable airflow supports diagnosis','one test and one presentation are not universal requirements','Specify test conditions; describe as characteristic prototype evidence.','How is asthma diagnosed')
add('Asthma','wheezing','required','gina_diagnosis','listed among variable respiratory symptoms','not necessarily present at every evaluation','Separate characteristic features from mandatory findings.','What is asthma; diagnosis')
add('Asthma','dyspnea','required','gina_diagnosis','listed among variable respiratory symptoms','not necessarily present at every evaluation','Avoid clinical necessity claim for an individual symptom.','What is asthma; diagnosis')
add('Asthma','episodic_symptoms','required','gina_diagnosis','variable symptoms align with description','fixed .90 probability not supported by this reference','Retain as synthetic parameter unless calibrated.','What is asthma')
add('Diabetic_Ketoacidosis','hyperglycemia_severe','required','umpierrez2024','supports a common hyperglycemic presentation','euglycemic presentations are recognized','Qualify as hyperglycemic prototype; review threshold and subtype.','Diagnostic criteria and euglycemic DKA')
add('Diabetic_Ketoacidosis','normal_blood_glucose','excluding','umpierrez2024','non-markedly elevated glucose does not rule out DKA','blanket exclusion or fixed .02 probability is unsupported','Separate euglycemic presentation; avoid broad exclusion.','Diagnostic criteria and euglycemic DKA')
add('Diabetic_Ketoacidosis','ketonemia','required','umpierrez2024','supports ketosis assessment','exact token threshold and fixed probability unvalidated','Specify intended ketone measurement and context.','Diagnostic criteria')
add('Diabetic_Ketoacidosis','metabolic_acidosis','required','umpierrez2024','supports acidosis assessment','fixed probability unvalidated','Preserve synthetic/empirical distinction.','Diagnostic criteria')
add('Infective_Endocarditis','positive_blood_cultures','required','fowler2023','positive microbiology can support a major criterion','culture-negative disease is explicitly recognized','Separate culture-positive prototype or relax required interpretation.','Microbiologic criteria')
add('Infective_Endocarditis','negative_blood_cultures','excluding','fowler2023','negative cultures do not generally exclude IE','blanket exclusion or fixed .02 probability is unsupported','Record antibiotic exposure, organisms and alternate microbiology; review exclusion.','Culture-negative endocarditis')
for finding in ['hypotension','hyponatremia','hyperpigmentation']:
    add('Addison_Disease',finding,'required','bornstein2016','listed among findings prompting diagnostic testing','guideline does not establish universal necessity or .90 probability','Association supported; clinical category and frequency remain unvalidated.','Recommendation 1.1')
add('Addison_Disease','hyperkalemia','supporting','bornstein2016','listed among findings prompting diagnostic testing','fixed .50 probability is not established','Association supported; seek empirical frequency if calibration claimed.','Recommendation 1.1')
a=pd.DataFrame(audit);a.to_csv(ROOT/'rebuttal/data/literature_profile_audit.csv',index=False)
# Exclude target cases for the five profiles with broad-label/necessity concerns.
flagged=['Acute_MI','Hypothyroidism','Asthma','Diabetic_Ketoacidosis','Infective_Endocarditis']
v=pd.read_csv(ROOT/'data_final_v2/diagrare_x/vignettes.csv').set_index('vignette_id');o=pd.read_csv(ROOT/'results_final_v2/main/bayes_oracle_table_expanded.csv');ont=pd.read_csv(ROOT/'data_final_v2/diagrare_x/ontology.csv');diseases=sorted(ont.disease);di={d:i for i,d in enumerate(diseases)};E=o.pivot(index='vignette_id',columns='disease',values='evidence_loglik_ratio').reindex(index=v.index,columns=diseases).to_numpy();P=np.log(ont.set_index('disease').loc[diseases,'prevalence'].to_numpy());master=pd.read_csv(ROOT/'rebuttal/results/model_metrics.csv');rows=[]
for _,m in master.iterrows():
    p=pd.read_csv(ROOT/'results/predictions'/f'{m.model}__baseline.csv').set_index('vignette_id').reindex(v.index);cs=[];n_total=0
    for j,(vid,r) in enumerate(p.iterrows()):
        if v.loc[vid,'target_disease'] in flagged:continue
        n_total+=1;chosen=[]
        for col in ['prediction_1','prediction_2','prediction_3']:
            name=str(r[col]).strip()
            if name not in di or di[name] in chosen:break
            chosen.append(di[name])
        if chosen:cs.append(Choice(chosen,P,E[j]))
    if len(cs)<30:continue
    fit=fit_plackett_luce(cs,n_starts=1);rows.append({'model':m.model,'n_target_cases_retained':n_total,'n_used':len(cs),'used_frac':len(cs)/n_total,'original_gamma_evidence':m.gamma_evidence,'refit_gamma_evidence':fit['gamma_evidence'],'converged':fit['converged']})
x=pd.DataFrame(rows);x.to_csv(OUT/'clinical_flag_target_exclusion_sensitivity.csv',index=False);good=x[x.used_frac>=.5];rho=spearmanr(good.original_gamma_evidence,good.refit_gamma_evidence).statistic
(OUT/'clinical_literature_audit_status.json').write_text(json.dumps({'relationships_in_ontology':888,'relationships_with_initial_literature_assessment':len(a),'diseases_with_initial_assessment':int(a.disease.nunique()),'clinical_reviewer_count':0,'profiles_flagged_for_target_exclusion_sensitivity':flagged,'target_cases_excluded':int(v.target_disease.isin(flagged).sum()),'n_models_sensitivity':len(x),'n_models_at_least_50_percent_coverage':len(good),'spearman_original_vs_refit_good_coverage':rho,'limitations':['Target-case exclusion retains the original 82 candidate features, including flagged profiles as alternatives.','This is not a complete ontology correction or validation.','Flags were chosen after reading sources and are exploratory; no prespecified clinical-validation claim.','Original ontology and inference suite remain unchanged; known issues are disclosed.']},indent=2))
print('Audited',len(a),'relationships; coverage-restricted exclusion rank agreement',rho)
