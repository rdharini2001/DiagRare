#!/usr/bin/env python3
"""Refresh only disease-heldout specification comparisons."""
from pathlib import Path
import sys
import pandas as pd
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(Path(__file__).parent));sys.path.insert(0,str(ROOT/'code_final_v2/analysis'))
from analyze_existing import flexible
from plackett_luce import build_choices_from_predictions
v=pd.read_csv(ROOT/'data_final_v2/diagrare_x/vignettes.csv').set_index('vignette_id');o=pd.read_csv(ROOT/'results_final_v2/main/bayes_oracle_table_expanded.csv');ont=pd.read_csv(ROOT/'data_final_v2/diagrare_x/ontology.csv').set_index('disease');diseases=sorted(ont.index)
o=o.sort_values(['vignette_id','disease']);rows=[]
for model in ['qwen2.5-1.5b','qwen2.5-3b','qwen2.5-7b','phi-3.5-mini','mistral-7b-instruct','olmo-2-7b']:
    p=pd.read_csv(ROOT/'results/predictions'/f'{model}__baseline.csv').set_index('vignette_id').reindex(v.index).reset_index()
    valid=p.prediction_1.isin(diseases).to_numpy();cs,_,_=build_choices_from_predictions(p,o);assert len(cs)==sum(valid)
    rows.extend(flexible(model,cs,valid.nonzero()[0],v,diseases,ont));print('Refit',model,flush=True)
pd.DataFrame(rows).to_csv(ROOT/'rebuttal/results/grouped_heldout_specifications.csv',index=False)
