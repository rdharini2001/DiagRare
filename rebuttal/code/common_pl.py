"""Shared loaders for rebuttal analyses on the primary 82-way benchmark."""
from pathlib import Path
import sys
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'rebuttal/results'
sys.path.insert(0,str(ROOT/'code_final_v2/analysis'));sys.path.insert(0,str(ROOT/'code_final_v2/theory'))
from plackett_luce import Choice,fit_plackett_luce
def split_tokens(x):
    return set() if pd.isna(x) or str(x).strip()=='' else {t.strip() for t in str(x).split(';') if t.strip()}
class Bench:
    def __init__(self):
        self.v=pd.read_csv(ROOT/'data_final_v2/diagrare_x/vignettes.csv').set_index('vignette_id')
        self.ont=pd.read_csv(ROOT/'data_final_v2/diagrare_x/ontology.csv').set_index('disease').sort_index()
        self.diseases=list(self.ont.index);self.di={d:i for i,d in enumerate(self.diseases)}
        self.P=np.log(self.ont.prevalence.to_numpy())
        o=pd.read_csv(ROOT/'results_final_v2/main/bayes_oracle_table_expanded.csv')
        self.E0=o.pivot(index='vignette_id',columns='disease',values='evidence_loglik_ratio').reindex(index=self.v.index,columns=self.diseases).to_numpy()
        self.pos=[split_tokens(x) for x in self.v.positive_findings];self.neg=[split_tokens(x) for x in self.v.negative_findings]
        self.target=self.v.target_disease.to_numpy();self.organ=self.ont.organ_system.to_numpy()
        m=pd.read_csv(ROOT/'results_final_v2/main/per_master_table.csv').query("source=='open_weight' and condition=='baseline' and split=='ALL'").dropna(subset=['gamma_evidence']).drop_duplicates('model')
        self.master=m.set_index('model')
        self._pred={}
    def relations(self):
        """dict disease -> {finding: 'req'|'sup'|'exc'}"""
        out={}
        for d,r in self.ont.iterrows():
            m={}
            for cls,col in [('req','required'),('sup','supporting'),('exc','excluding')]:
                for f in split_tokens(r[col]):m[f]=cls
            out[d]=m
        return out
    def evidence(self,rel,p=None):
        p=p or {'req':.90,'sup':.50,'exc':.02,'bg':.03}
        E=np.zeros((len(self.v),len(self.diseases)))
        for i in range(len(self.v)):
            pos,neg=self.pos[i],self.neg[i];blind=len(pos)*np.log(p['bg'])+len(neg)*np.log(1-p['bg'])
            for j,d in enumerate(self.diseases):
                m=rel[d];ll=0.
                for f in pos:ll+=np.log(p[m.get(f,'bg')])
                for f in neg:ll+=np.log(1-p[m.get(f,'bg')])
                E[i,j]=ll-blind
        return E
    def predictions(self,model):
        if model not in self._pred:
            pr=pd.read_csv(ROOT/'results/predictions'/f'{model}__baseline.csv').set_index('vignette_id').reindex(self.v.index)
            names=pr[['prediction_1','prediction_2','prediction_3']].fillna('').astype(str).to_numpy();chosen=[]
            for ns in names:
                c=[]
                for n in ns:
                    if n not in self.di or self.di[n] in c:break
                    c.append(self.di[n])
                chosen.append(c)
            self._pred[model]=chosen
        return self._pred[model]
    def choices(self,model,E):
        return [Choice(c,self.P,E[i]) for i,c in enumerate(self.predictions(model)) if c]
    def models(self,min_used=0.5):
        out=[]
        for m in self.master.index:
            if not (ROOT/'results/predictions'/f'{m}__baseline.csv').exists():continue
            if self.master.loc[m,'used_frac']>=min_used:out.append(m)
        return out
    def accuracy(self,model):
        ch=self.predictions(model);return float(np.mean([bool(c) and self.diseases[c[0]]==t for c,t in zip(ch,self.target)]))
