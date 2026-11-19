import pandas as pd,numpy as np,itertools,sys
from scipy.stats import binomtest
from pathlib import Path
R=Path(__file__).resolve().parents[1]/'results';A=R/'study2A'
P=pd.read_csv(R/'study3_predictors.csv');U=P[(P.accuracy>=.05)&(P.used_frac>=.5)].copy()
S=pd.read_csv(A/'sct_by_model.csv')[['model','sct_score']];G=pd.read_csv(A/'grounded_by_model.csv')[['model','parse','agree_reference']];G=G[G.parse>=.9][['model','agree_reference']]
Ag=pd.read_csv(A/'age_prior_by_model.csv');Ag=Ag[Ag.parse_age>=.9][['model','age_sensitivity']];E=pd.read_csv(A/'ehr_by_model.csv')[['model','gain_true_vs_none']];M=pd.read_csv(A/'mm_by_model.csv')[['model','acc_text_only']]
O=U[['model','accuracy','gamma_evidence','gamma_prior','top3','mrr3']]
for t in [S,G,Ag,E,M]:O=O.merge(t,on='model',how='left')
extra=sys.argv[1:]  # optional extra outcome csvs (model + columns)
for f in extra:O=O.merge(pd.read_csv(f),on='model',how='left')
outs=[c for c in O.columns if c not in('model','accuracy','gamma_evidence','gamma_prior','top3','mrr3')]
pairs=[(a,b) for a,b in itertools.combinations(O.index,2) if abs(O.accuracy[a]-O.accuracy[b])<=0.02 and O.accuracy[a]>=.05 and O.accuracy[b]>=.05]
print('analysis panel',len(O),'accuracy-matched pairs (|dAcc|<=0.02, acc>=.05):',len(pairs));
for a,b in pairs:print('  ',O.model[a],O.model[b],'acc %.3f/%.3f gamma_e %.3f/%.3f'%(O.accuracy[a],O.accuracy[b],O.gamma_evidence[a],O.gamma_evidence[b]))
rows=[]
for c in outs:
    k=n=0
    for a,b in pairs:
        if np.isnan(O[c][a]) or np.isnan(O[c][b]) or O.gamma_evidence[a]==O.gamma_evidence[b] or O[c][a]==O[c][b]:continue
        n+=1;k+=(O.gamma_evidence[a]>O.gamma_evidence[b])==(O[c][a]>O[c][b])
    p=binomtest(k,n,.5).pvalue if n else np.nan;rows.append(dict(outcome=c,pairs=n,gamma_concordant=k,share=k/n if n else np.nan,p_binom=p))
T=pd.DataFrame(rows);T.to_csv(R/'study3_matched_pairs.csv',index=False);print(T.round(3).to_string())
