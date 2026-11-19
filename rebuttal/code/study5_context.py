"""Study 5: context-conditioned diagnostic reclassification (inference-free).
Compares stored predictions under the primary-care and rare-clinic prompts. See docs/STUDY5_PREREGISTRATION.md."""
import sys, time, numpy as np, pandas as pd
from multiprocessing import Pool
from scipy.stats import spearmanr
sys.path.insert(0, 'rebuttal/scripts')
from common_pl import Bench, ROOT, OUT
from plackett_luce import Choice, fit_plackett_luce

S5 = OUT / 'study5'; S5.mkdir(exist_ok=True)
B = Bench(); E = B.E0; P = B.P; N = len(B.diseases); n = len(B.v)
PRED = ROOT / 'results/predictions'
CTX = {'prim': 'counterfactual_primary_care', 'rare': 'counterfactual_rare_clinic', 'base': 'baseline'}
WELL = ['mistral-7b-instruct', 'olmo-2-7b', 'phi-3.5-mini', 'qwen2.5-1.5b', 'qwen2.5-3b', 'qwen2.5-7b']
LOWER = ['qwen2.5-0.5b', 'yi-1.5-6b']
NB = 2000; NG = 400

tgt = np.array([B.di[t] for t in B.target])
dis_of_case = pd.factorize(B.target)[0]; ND = dis_of_case.max() + 1
ont_rare_disease = B.ont.is_rare.to_numpy().astype(bool)  # label of a picked disease
r_ont = B.v.is_rare.to_numpy().astype(bool)
aud = pd.read_csv(ROOT / 'rebuttal/data/ontology_metadata_audit_v1.csv').set_index('disease')
r_eu = np.array([bool(aud.loc[t, 'is_rare_eu_definition']) for t in B.target])
prev = B.v.prevalence.to_numpy(); lp10 = np.log10(prev)
q1, q2 = np.quantile(lp10, [1 / 3, 2 / 3]); terc_lo = lp10 <= q1; terc_hi = lp10 >= q2


def load(model, ctx):
    pr = pd.read_csv(PRED / f'{model}__{CTX[ctx]}.csv').set_index('vignette_id').reindex(B.v.index)
    names = pr[['prediction_1', 'prediction_2', 'prediction_3']].fillna('').astype(str).to_numpy(); out = []
    for ns in names:
        c = []
        for nm in ns:
            if nm not in B.di or B.di[nm] in c: break
            c.append(B.di[nm])
        out.append(c)
    return out


def arrays(ch):
    valid = np.array([bool(c) for c in ch]); top1 = np.array([c[0] if c else -1 for c in ch])
    rank = np.zeros(n, int)
    for i, c in enumerate(ch):
        if tgt[i] in c: rank[i] = c.index(tgt[i]) + 1
    return valid, top1, rank


def wmean(x, w, m): d = (w * m).sum(); return np.nan if d == 0 else (w * m * x).sum() / d


def fitg(chs, w=None):
    cl = []
    for i, c in enumerate(chs):
        if not c: continue
        k = 1 if w is None else int(w[i])
        cl += [Choice(c, P, E[i])] * k
    if len(cl) < 40: return (np.nan, np.nan)
    r = fit_plackett_luce(cl, n_starts=1); return (r['gamma_prior'], r['gamma_evidence'])


def point_stats(a_p, a_r, w, mask):
    """accuracy stats on the case set `mask` (bool), weights w; returns dict"""
    vp, t1p, rkp = a_p; vr, t1r, rkr = a_r
    cp = (rkp == 1).astype(float); cr = (rkr == 1).astype(float); d = cr - cp
    o = {}
    for nm, g in [('all', np.ones(n, bool)), ('rare', r_ont), ('common', ~r_ont), ('rare_eu', r_eu), ('common_eu', ~r_eu), ('terc_lo', terc_lo), ('terc_hi', terc_hi)]:
        m = mask & g
        o[f'dacc_{nm}'] = wmean(d, w, m); o[f'acc_prim_{nm}'] = wmean(cp, w, m); o[f'acc_rare_{nm}'] = wmean(cr, w, m)
    o['did_ont'] = o['dacc_rare'] - o['dacc_common']; o['did_eu'] = o['dacc_rare_eu'] - o['dacc_common_eu']; o['did_terc'] = o['dacc_terc_lo'] - o['dacc_terc_hi']
    # slope of d on log10 prevalence (weighted OLS over mask)
    ww = w * mask
    if ww.sum() > 0:
        xm = (ww * lp10).sum() / ww.sum(); ym = (ww * d).sum() / ww.sum(); vx = (ww * (lp10 - xm) ** 2).sum()
        o['slope_dacc_per_log10prev'] = (ww * (lp10 - xm) * (d - ym)).sum() / vx if vx > 0 else np.nan
    # switch stats
    sw = mask & (t1p != t1r) & (t1p >= 0) & (t1r >= 0)
    dlp = P[np.clip(t1r, 0, None)] - P[np.clip(t1p, 0, None)]
    de = np.array([E[i, t1r[i]] - E[i, t1p[i]] if t1r[i] >= 0 and t1p[i] >= 0 else 0 for i in range(n)])
    o['switch_rate'] = wmean(1.0, w, mask) and (w * sw).sum() / (w * mask).sum()
    o['mean_top1_logprev_prim'] = wmean(P[np.clip(t1p, 0, None)], w, mask); o['mean_top1_logprev_rare'] = wmean(P[np.clip(t1r, 0, None)], w, mask)
    o['d_mean_top1_logprev'] = o['mean_top1_logprev_rare'] - o['mean_top1_logprev_prim']
    o['d_share_rare_label_top1'] = wmean(ont_rare_disease[np.clip(t1r, 0, None)].astype(float), w, mask) - wmean(ont_rare_disease[np.clip(t1p, 0, None)].astype(float), w, mask)
    if (w * sw).sum() > 0:
        o['sw_mean_dlogprev'] = wmean(dlp, w, sw); o['sw_share_lower_prev'] = wmean((dlp < 0).astype(float), w, sw)
        o['sw_mean_dE'] = wmean(de, w, sw); o['sw_share_higher_E'] = wmean((de > 0).astype(float), w, sw)
        for nm, g in [('rare', r_ont), ('common', ~r_ont)]:
            s2 = sw & g
            if (w * s2).sum() > 0: o[f'sw_{nm}_share_lower_prev'] = wmean((dlp < 0).astype(float), w, s2); o[f'sw_{nm}_mean_dE'] = wmean(de, w, s2)
        same = sw & (cp == cr)
        if (w * same).sum() > 0: o['sw_samecorrect_share_lower_prev'] = wmean((dlp < 0).astype(float), w, same); o['sw_samecorrect_frac_of_switches'] = (w * same).sum() / (w * sw).sum()
    return o


def work(m):
    t0 = time.time(); chp = load(m, 'prim'); chr_ = load(m, 'rare'); chb = load(m, 'base')
    ap = arrays(chp); ar = arrays(chr_); ab = arrays(chb)
    both = ap[0] & ar[0]; allm = np.ones(n, bool); one = np.ones(n)
    rows = []
    # original unpaired fits (all valid in each context) and paired fits (both-valid)
    gp_p0, ge_p0 = fitg(chp); gp_r0, ge_r0 = fitg(chr_)
    chp_b = [c if both[i] else [] for i, c in enumerate(chp)]; chr_b = [c if both[i] else [] for i, c in enumerate(chr_)]
    gp_p, ge_p = fitg(chp_b); gp_r, ge_r = fitg(chr_b)
    pt = {'model': m, 'n_valid_prim': int(ap[0].sum()), 'n_valid_rare': int(ar[0].sum()), 'n_both': int(both.sum()),
          'orig_gp_prim': gp_p0, 'orig_gp_rare': gp_r0, 'orig_dgp': gp_r0 - gp_p0, 'orig_ge_prim': ge_p0, 'orig_ge_rare': ge_r0, 'orig_dge': ge_r0 - ge_p0,
          'gp_prim': gp_p, 'gp_rare': gp_r, 'dgp': gp_r - gp_p, 'ge_prim': ge_p, 'ge_rare': ge_r, 'dge': ge_r - ge_p}
    sb = point_stats(ap, ar, one, both)
    # original definition (all 984; unparseable = wrong)
    so = point_stats(ap, ar, one, allm)
    for k, v in sb.items(): pt['b_' + k] = v
    for k in ['acc_prim_all', 'acc_rare_all', 'dacc_all', 'acc_prim_rare', 'acc_rare_rare', 'dacc_rare', 'acc_prim_common', 'acc_rare_common', 'dacc_common', 'did_ont']: pt['o_' + k] = so[k]
    # neutral baseline reference: mean top-1 log prevalence on cases valid in all three
    all3 = both & ab[0]
    pt['n_all3'] = int(all3.sum())
    for nm, a in [('base', ab), ('prim', ap), ('rare', ar)]:
        pt[f'ref_mean_logprev_{nm}'] = wmean(P[np.clip(a[1], 0, None)], one, all3)
        pt[f'ref_share_rare_{nm}'] = wmean(ont_rare_disease[np.clip(a[1], 0, None)].astype(float), one, all3)
        pt[f'ref_acc_{nm}'] = wmean((a[2] == 1).astype(float), one, all3)
    rows.append(('point', pt))
    # reclassification tables (both-valid)
    rec = []
    for gname, g in [('rare', r_ont), ('common', ~r_ont), ('rare_eu', r_eu), ('common_eu', ~r_eu)]:
        mk = both & g; cp = ap[2][mk] == 1; cr = ar[2][mk] == 1
        rec.append(dict(model=m, group=gname, n=int(mk.sum()), retained_correct=int((cp & cr).sum()), newly_correct=int((~cp & cr).sum()), newly_incorrect=int((cp & ~cr).sum()), remained_wrong=int((~cp & ~cr).sum()),
                        top1_identity_changed=int((ap[1][mk] != ar[1][mk]).sum()), **{f'rank{a}to{b}': int(((ap[2][mk] == a) & (ar[2][mk] == b)).sum()) for a in [0, 1, 2, 3] for b in [0, 1, 2, 3]}))
    rows.append(('rec', rec))
    # disease-cluster bootstrap
    rng = np.random.default_rng(20261118); boots = []; gb = []
    dcases = [np.where(dis_of_case == d)[0] for d in range(ND)]
    for b in range(NB):
        cnt = np.bincount(rng.integers(0, ND, ND), minlength=ND); w = cnt[dis_of_case].astype(float)
        s = point_stats(ap, ar, w, both); s['b'] = b; boots.append(s)
        if b < NG:
            wi = w.astype(int); g1 = fitg(chp_b, wi); g2 = fitg(chr_b, wi); gb.append(dict(b=b, dgp=g2[0] - g1[0], dge=g2[1] - g1[1]))
    rows.append(('boot', pd.DataFrame(boots).assign(model=m))); rows.append(('gboot', pd.DataFrame(gb).assign(model=m)))
    print(m, 'done %.0fs' % (time.time() - t0), flush=True)
    return rows


def pair_illustration():
    """Mistral-7B vs Nous-Hermes-2 on baseline, paired by vignette; disease-cluster bootstrap."""
    def lb(m):
        pr = pd.read_csv(PRED / f'{m}__baseline.csv').set_index('vignette_id').reindex(B.v.index)
        names = pr[['prediction_1', 'prediction_2', 'prediction_3']].fillna('').astype(str).to_numpy(); out = []
        for ns in names:
            c = []
            for nm in ns:
                if nm not in B.di or B.di[nm] in c: break
                c.append(B.di[nm])
            out.append(c)
        return out
    A = lb('mistral-7b-instruct'); Bm = lb('nous-hermes-2-7b'); rng = np.random.default_rng(20261118)
    def stats(w):
        o = {}
        for nm, ch in [('mis', A), ('nh', Bm)]:
            a = arrays(ch); gp, ge = fitg(ch, w.astype(int)) if w is not None else fitg(ch)
            top1 = (a[2] == 1).astype(float); top3 = (a[2] >= 1).astype(float); ww = np.ones(n) if w is None else w
            o[f'{nm}_gp'] = gp; o[f'{nm}_ge'] = ge; o[f'{nm}_acc'] = wmean(top1, ww, np.ones(n, bool)); o[f'{nm}_rare_acc'] = wmean(top1, ww, r_ont); o[f'{nm}_top3'] = wmean(top3, ww, np.ones(n, bool))
        for k in ['gp', 'ge', 'acc', 'rare_acc', 'top3']: o[f'd_{k}'] = o[f'mis_{k}'] - o[f'nh_{k}']
        return o
    pt = stats(None); bs = []
    for b in range(NG):
        cnt = np.bincount(rng.integers(0, ND, ND), minlength=ND); bs.append(stats(cnt[dis_of_case].astype(float)))
    bs = pd.DataFrame(bs); out = []
    for k in ['gp', 'ge', 'acc', 'rare_acc', 'top3']:
        out.append(dict(metric=k, mistral=pt[f'mis_{k}'], nous_hermes=pt[f'nh_{k}'], diff=pt[f'd_{k}'], lo=bs[f'd_{k}'].quantile(.025), hi=bs[f'd_{k}'].quantile(.975)))
    return pd.DataFrame(out)


if __name__ == '__main__':
    MODELS = WELL + LOWER
    with Pool(6) as p: R = p.map(work, MODELS)
    pts = pd.DataFrame([dict(r[1]) for r in R if False] + [x[1] for rr in R for x in rr if x[0] == 'point'])
    pts['panel'] = pts.model.map(lambda m: 'well' if m in WELL else 'lower'); pts.to_csv(S5 / 'point.csv', index=False)
    pd.DataFrame([x for rr in R for k, v in rr if k == 'rec' for x in v]).to_csv(S5 / 'reclassification.csv', index=False)
    bt = pd.concat([v for rr in R for k, v in rr if k == 'boot']); bt.to_csv(S5 / 'boot_acc.csv', index=False)
    gbt = pd.concat([v for rr in R for k, v in rr if k == 'gboot']); gbt.to_csv(S5 / 'boot_gamma.csv', index=False)
    # CIs
    rows = []
    for m in MODELS:
        pm = pts[pts.model == m].iloc[0]; b = bt[bt.model == m]; g = gbt[gbt.model == m]
        for k in [c for c in bt.columns if c not in ('b', 'model')]:
            bv = b[k].dropna()
            if len(bv) > 100: rows.append(dict(model=m, stat=k, est=pm.get('b_' + k, np.nan), lo=bv.quantile(.025), hi=bv.quantile(.975), n_boot=len(bv)))
        for k in ['dgp', 'dge']: rows.append(dict(model=m, stat=k, est=pm[k], lo=g[k].quantile(.025), hi=g[k].quantile(.975), n_boot=len(g)))
    pd.DataFrame(rows).to_csv(S5 / 'cis.csv', index=False)
    pair_illustration().to_csv(S5 / 'pair_mistral_nous.csv', index=False)
    print('done')
