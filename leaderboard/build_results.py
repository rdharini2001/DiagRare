#!/usr/bin/env python3
"""Build leaderboard/results.json and leaderboard/leaderboard.csv from the released result tables.

Run from anywhere:  python leaderboard/build_results.py
Inputs  : tables/, tables/supplementary/external_validation/, rebuttal/results/, data/metadata/model_metadata.csv
Nothing is imputed: a value that was not estimated for a checkpoint is written as null and shown as a dash.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[1]
T = ROOT / 'tables'
RB = ROOT / 'rebuttal/results'
OUT = Path(__file__).resolve().parent

DISPLAY = {
    'qwen2.5-0.5b': 'Qwen2.5-0.5B', 'qwen2.5-1.5b': 'Qwen2.5-1.5B', 'qwen2.5-3b': 'Qwen2.5-3B', 'qwen2.5-7b': 'Qwen2.5-7B', 'qwen2.5-14b': 'Qwen2.5-14B',
    'qwen2.5-32b': 'Qwen2.5-32B', 'qwen2.5-72b': 'Qwen2.5-72B', 'qwen3-32b': 'Qwen3-32B', 'yi-1.5-6b': 'Yi-1.5-6B', 'yi-1.5-9b': 'Yi-1.5-9B',
    'olmo-2-7b': 'OLMo-2-7B', 'olmo-2-32b': 'OLMo-2-32B', 'deepseek-llm-7b': 'DeepSeek-LLM-7B', 'mistral-7b-instruct': 'Mistral-7B-Instruct',
    'mistral-small-24b': 'Mistral-Small-24B', 'phi-3.5-mini': 'Phi-3.5-mini', 'phi-4': 'Phi-4', 'falcon-7b': 'Falcon-7B', 'stablelm-zephyr-3b': 'StableLM-Zephyr-3B',
    'granite-3.1-8b': 'Granite-3.1-8B', 'granite-3.1-2b': 'Granite-3.1-2B', 'zephyr-7b': 'Zephyr-7B', 'openchat-3.5': 'OpenChat-3.5', 'vicuna-7b': 'Vicuna-7B',
    'tinyllama-1.1b': 'TinyLlama-1.1B', 'glm-4-9b': 'GLM-4-9B', 'nous-hermes-2-7b': 'Nous-Hermes-2-7B', 'medalpaca-7b': 'MedAlpaca-7B', 'openbiollm-8b': 'OpenBioLLM-8B',
    'asclepius-7b': 'Asclepius-7B', 'med42-8b': 'Med42-8B', 'biomistral-7b': 'BioMistral-7B', 'biomistral-7b-retry': 'BioMistral-7B', 'smollm2-1.7b': 'SmolLM2-1.7B',
}
ADDED = {  # checkpoints added during review (same protocol, 82-way baseline); organisation / family / hf repo from the released run scripts and docs
    'qwen2.5-14b': ('Alibaba', 'Qwen', 14.0, 'Qwen/Qwen2.5-14B-Instruct', False),
    'qwen2.5-32b': ('Alibaba', 'Qwen', 32.0, 'Qwen/Qwen2.5-32B-Instruct', False),
    'qwen2.5-72b': ('Alibaba', 'Qwen', 72.0, 'Qwen/Qwen2.5-72B-Instruct', False),
    'qwen3-32b': ('Alibaba', 'Qwen', 32.0, 'Qwen/Qwen3-32B', False),
    'mistral-small-24b': ('Mistral AI', 'Mistral', 24.0, 'mistralai/Mistral-Small-24B-Instruct-2501', False),
    'phi-4': ('Microsoft', 'Phi', 14.0, 'microsoft/phi-4', False),
    'olmo-2-32b': ('Allen AI', 'OLMo', 32.0, 'allenai/OLMo-2-0325-32B-Instruct', False),
}


def nn(x):
    return None if x is None or (isinstance(x, float) and not np.isfinite(x)) else float(x)


per = pd.read_csv(T / 'per_master_table.csv')
base = per[(per.source == 'open_weight') & (per.condition == 'baseline') & (per.split == 'ALL')].copy()
base['model'] = base.model.replace({'biomistral-7b-retry': 'biomistral-7b'})
base = base.sort_values('used_frac', ascending=False).drop_duplicates('model').set_index('model')
acc = pd.read_csv(T / 'open_weight_summary.csv')
acc = acc[(acc.condition == 'baseline') & (acc.split == 'ALL')].set_index('model')
meta = pd.read_csv(ROOT / 'data/metadata/model_metadata.csv')
meta['model_tag'] = meta.model_tag.replace({'biomistral-7b-retry': 'biomistral-7b'})
meta = meta.drop_duplicates('model_tag').set_index('model_tag')
surf = pd.read_csv(T / 'causal_response_surface.csv').assign(model=lambda d: d.model.replace({'biomistral-7b-retry': 'biomistral-7b'})).drop_duplicates('model').set_index('model')
grid = pd.read_csv(T / 'causal_grid_analysis.csv').assign(model=lambda d: d.model.replace({'biomistral-7b-retry': 'biomistral-7b'})).drop_duplicates('model').set_index('model')
seq = pd.read_csv(T / 'sequential_anchoring_analysis.csv').set_index('model')
cup = pd.read_csv(T / 'supplementary/external_validation/cupcase_full_analysis.csv').assign(model=lambda d: d.model.replace({'biomistral-7b-retry': 'biomistral-7b'})).drop_duplicates('model').set_index('model')
rare = pd.read_csv(T / 'supplementary/external_validation/rarearena_full_analysis.csv').assign(model=lambda d: d.model.replace({'biomistral-7b-retry': 'biomistral-7b'})).drop_duplicates('model').set_index('model')
big = pd.read_csv(RB / 'study2A/sct_by_model.csv').set_index('model')       # same baseline protocol, includes the added checkpoints
sct = big['sct_score'].to_dict()

rows = []
tags = list(meta.index) + [m for m in ADDED if m not in meta.index]
for m in tags:
    r = dict(id=m, name=DISPLAY.get(m, m), added=m in ADDED)
    if m in ADDED:
        org, fam, pb, repo, med = ADDED[m]
        r.update(org=org, family=fam, params=pb, hf=repo, medical=med)
        b = big.loc[m]
        r.update(acc1=nn(b.acc82), acc3=None, rare1=None, unparsed=None, ge=nn(b.gamma_e), ge_lo=None, ge_hi=None, gp=nn(b.gamma_p), gp_lo=None, gp_hi=None, used=nn(b.used82))
    else:
        mm = meta.loc[m]
        r.update(org=mm.organization, family=mm.family, params=nn(mm.param_count_b), hf=mm.hf_repo, medical=bool(mm.is_medical_specialized))
        b = base.loc[m]; a = acc.loc[m]
        r.update(acc1=nn(a.top1_accuracy), acc3=nn(a.top3_accuracy), rare1=nn(a.rare_top1_accuracy), unparsed=nn(a.unparsed_rate),
                 ge=nn(b.gamma_evidence), ge_lo=nn(b.gamma_evidence_lo), ge_hi=nn(b.gamma_evidence_hi), gp=nn(b.gamma_prior), gp_lo=nn(b.gamma_prior_lo), gp_hi=nn(b.gamma_prior_hi), used=nn(b.used_frac))
    # randomized 3x3 experiment, sequential revision, transfer
    s = surf.loc[m] if m in surf.index else None
    r['dE'] = nn(s.delta_E) if s is not None else None
    r['dE_lo'] = nn(s.delta_E_ci_lo) if s is not None else None
    r['dE_hi'] = nn(s.delta_E_ci_hi) if s is not None else None
    r['dE_n'] = int(s.n_pairs) if s is not None else None
    q = seq.loc[m] if m in seq.index else None
    ok = q is not None and nn(q.recovery_rate) is not None and q.n_used_for_recovery >= 5
    r['rec'] = nn(q.recovery_rate) if ok else None
    r['rec_lo'] = nn(q.recovery_ci_lo) if ok else None
    r['rec_hi'] = nn(q.recovery_ci_hi) if ok else None
    r['rec_n'] = int(q.n_used_for_recovery) if ok else None
    r['cup'] = nn(cup.loc[m].gamma_evidence_cupcase) if m in cup.index else None
    r['cup_cov'] = nn(cup.loc[m].n_used / cup.loc[m].n_total) if m in cup.index else None
    r['rare_ext'] = nn(rare.loc[m].gamma_evidence_rarearena) if m in rare.index else None
    r['rare_cov'] = nn(rare.loc[m].n_used / rare.loc[m].n_total) if m in rare.index else None
    r['sct'] = nn(sct.get(m))
    # tier
    ident = r['ge'] is not None
    cover = r['used'] is not None and r['used'] >= 0.5
    good_acc = r['acc1'] is not None and r['acc1'] >= 0.05
    r['tier'] = 'core' if ident and cover and good_acc else ('low' if ident else 'none')
    r['reason'] = '' if r['tier'] == 'core' else ('fewer than 50% of outputs usable' if ident and not cover else ('top-1 accuracy below 5%' if ident and cover else 'coefficient not identifiable'))
    rows.append(r)

df = pd.DataFrame(rows)
core = df[df.tier == 'core']
stats = {
    'n_checkpoints': int(len(df)), 'n_original': int((~df.added).sum()), 'n_added': int(df.added.sum()),
    'n_identifiable': int(df['ge'].notna().sum()), 'n_core': int(len(core)),
    'r_acc_ge_core': float(pearsonr(core.acc1, core['ge'])[0]), 'rho_acc_ge_core': float(spearmanr(core.acc1, core['ge'])[0]),
}
orig_ident = df[(~df.added) & df['ge'].notna() & df.acc1.notna()]
stats['r_acc_ge_original21'] = float(pearsonr(orig_ident.acc1, orig_ident['ge'])[0]); stats['n_original_identifiable'] = int(len(orig_ident))

# referral-context experiment (six well-covered models; exploratory)
cis = pd.read_csv(RB / 'study5/cis.csv'); pt = pd.read_csv(RB / 'study5/point.csv').set_index('model')
ctx = []
for m in pt.index:
    if pt.loc[m, 'panel'] != 'well': continue
    g = lambda s: cis[(cis.model == m) & (cis.stat == s)].iloc[0]
    ctx.append(dict(id=m, name=DISPLAY.get(m, m), gp_prim=nn(pt.loc[m, 'gp_prim']), gp_rare=nn(pt.loc[m, 'gp_rare']),
                    dgp=nn(g('dgp').est), dgp_lo=nn(g('dgp').lo), dgp_hi=nn(g('dgp').hi), dacc=nn(g('dacc_all').est), dacc_lo=nn(g('dacc_all').lo), dacc_hi=nn(g('dacc_all').hi),
                    switch=nn(g('switch_rate').est), lower=nn(g('sw_share_lower_prev').est), n_both=int(pt.loc[m, 'n_both'])))
ctx.sort(key=lambda d: d['dgp'])

out = dict(meta=dict(title='DiagRare-Bench', stats=stats), models=rows, context=ctx)
(OUT / 'results.json').write_text(json.dumps(out, indent=1, allow_nan=False))
cols = ['id', 'name', 'org', 'family', 'medical', 'params', 'hf', 'tier', 'added', 'used', 'acc1', 'acc3', 'rare1', 'ge', 'ge_lo', 'ge_hi', 'gp', 'gp_lo', 'gp_hi', 'dE', 'dE_lo', 'dE_hi', 'rec', 'rec_lo', 'rec_hi', 'cup', 'rare_ext', 'sct']
sort = df.assign(_k=df.tier.map({'core': 0, 'low': 1, 'none': 2}), _g=-df['ge'].fillna(-9)).sort_values(['_k', '_g'])
sort[cols].rename(columns={'id': 'model_tag', 'name': 'model', 'org': 'organization', 'medical': 'medical_specialized', 'params': 'param_count_b', 'hf': 'hf_repo', 'used': 'usable_fraction',
                           'acc1': 'top1_accuracy', 'acc3': 'top3_accuracy', 'rare1': 'rare_top1_accuracy', 'ge': 'gamma_evidence', 'ge_lo': 'gamma_evidence_lo', 'ge_hi': 'gamma_evidence_hi',
                           'gp': 'gamma_prior', 'gp_lo': 'gamma_prior_lo', 'gp_hi': 'gamma_prior_hi', 'dE': 'delta_E', 'dE_lo': 'delta_E_ci_lo', 'dE_hi': 'delta_E_ci_hi',
                           'rec': 'recovery_rate', 'rec_lo': 'recovery_ci_lo', 'rec_hi': 'recovery_ci_hi', 'cup': 'gamma_evidence_cupcase', 'rare_ext': 'gamma_evidence_rarearena',
                           'sct': 'sct_score'}).to_csv(OUT / 'leaderboard.csv', index=False, float_format='%.6g')
print(json.dumps(stats, indent=1)); print(df.groupby('tier').size())
print(core[['name', 'acc1', 'ge', 'used']].sort_values('ge', ascending=False).round(3).to_string(index=False))

# ---- consistency checks against the manuscript (printed; also stored in results.json) ----
def corr(x, y, mask):
    d = df[mask & df[x].notna() & df[y].notna()]
    return dict(n=int(len(d)), r=float(pearsonr(d[x], d[y])[0]), rho=float(spearmanr(d[x], d[y])[0]))
orig = ~df.added
chk = {'dE_vs_ge': corr('ge', 'dE', orig), 'rec_vs_ge': corr('ge', 'rec', orig), 'cup_vs_ge': corr('ge', 'cup', orig), 'rare_vs_ge': corr('ge', 'rare_ext', orig)}
print(chk)
stats.update(chk)
out['meta']['stats'] = stats
(OUT / 'results.json').write_text(json.dumps(out, indent=1, allow_nan=False))

# results.js lets the page work when opened from disk or inside an iframe without fetch()
(OUT / 'results.js').write_text('window.DIAGRARE = ' + json.dumps(out, allow_nan=False) + ';\n')
