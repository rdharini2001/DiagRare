"""Study 6: diagnostic substitution matrix, held-out substitution-direction test, equivalence test, compliance sensitivity.
See docs/STUDY6_PREREGISTRATION.md. Inference-free."""
import sys, numpy as np, pandas as pd
from multiprocessing import Pool
from scipy.optimize import minimize
from scipy.special import expit, log_expit
sys.path.insert(0, 'rebuttal/scripts')
import study5_context as s5
from study5_context import B, E, P, n, tgt, dis_of_case, ND, WELL, PRED, CTX, load, arrays, fitg, wmean
from plackett_luce import Choice, fit_plackett_luce
S6 = s5.OUT / 'study6'; S6.mkdir(exist_ok=True)
NB = 2000; rng0 = 20261118


def sgn(x, tol=1e-9): return np.where(x > tol, 1, np.where(x < -tol, -1, 0))


def pl_fit(ch, idx, feat):
    """PL on cases idx; feat in {'both','prev','evi'} (zeroes the other feature)."""
    cl = []
    for i in idx:
        if not ch[i]: continue
        cl.append(Choice(ch[i], P if feat != 'evi' else np.zeros_like(P), E[i] if feat != 'prev' else np.zeros(len(P))))
    r = fit_plackett_luce(cl, n_starts=1)
    return (r['gamma_prior'] if feat != 'evi' else 0.0, r['gamma_evidence'] if feat != 'prev' else 0.0)


def fit_dir(X, y_dummy=None):
    """logistic with no intercept: P(b chosen)=sigmoid(X theta); observed outcome always 1; ridge 1e-3"""
    f = lambda th: -log_expit(X @ th).sum() + 1e-3 * (th ** 2).sum()
    return minimize(f, np.zeros(X.shape[1]), method='BFGS').x


def work(m):
    chp = load(m, 'prim'); chr_ = load(m, 'rare'); ap = arrays(chp); ar = arrays(chr_)
    both = ap[0] & ar[0]; sw = np.where(both & (ap[1] != ar[1]))[0]
    a_, b_ = ap[1][sw], ar[1][sw]
    dprev = P[b_] - P[a_]; dE = np.array([E[i, b] - E[i, a] for i, a, b in zip(sw, a_, b_)])
    cp = (ap[2][sw] == 1); cr = (ar[2][sw] == 1)
    trans = np.where(~cp & cr, 'wrong->correct', np.where(cp & ~cr, 'correct->wrong', 'wrong->different wrong'))
    quad = np.array([('less' if p < 0 else 'more' if p > 0 else 'eqprev') + '_prev|' + ('higher' if e > 0 else 'lower' if e < 0 else 'eq') + '_evid' for p, e in zip(sgn(dprev), sgn(dE))])
    sub = pd.DataFrame(dict(model=m, case=sw, disease=dis_of_case[sw], quad=quad, trans=trans, dprev=dprev, dE=dE))
    # --- held-out direction test: 5 folds by disease
    r = np.random.default_rng(rng0); perm = r.permutation(ND); fold_of_dis = np.empty(ND, int); fold_of_dis[perm] = np.arange(ND) % 5
    fold = fold_of_dis[dis_of_case]; ll = {k: np.full(len(sw), np.nan) for k in ['M_r', 'M_p', 'M_prev', 'M_evi', 'M_0', 'D_prev', 'D_both']}
    for k in range(5):
        tr = np.where(fold != k)[0]; te_mask = fold[sw] == k
        if te_mask.sum() == 0: continue
        gr = pl_fit(chr_, tr[both[tr]], 'both'); gp = pl_fit(chp, tr[both[tr]], 'both')
        gpr = pl_fit(chr_, tr[both[tr]], 'prev'); ger = pl_fit(chr_, tr[both[tr]], 'evi')
        trsw = ~te_mask; Xtr_prev = dprev[trsw][:, None]; Xtr_both = np.c_[dprev[trsw], dE[trsw]]
        th1 = fit_dir(Xtr_prev); th2 = fit_dir(Xtr_both)
        for nm, g in [('M_r', gr), ('M_p', gp), ('M_prev', gpr), ('M_evi', ger)]:
            ll[nm][te_mask] = log_expit(g[0] * dprev[te_mask] + g[1] * dE[te_mask])
        ll['M_0'][te_mask] = np.log(.5)
        ll['D_prev'][te_mask] = log_expit(th1[0] * dprev[te_mask]); ll['D_both'][te_mask] = log_expit(th2[0] * dprev[te_mask] + th2[1] * dE[te_mask])
    LL = pd.DataFrame(ll); LL['model'] = m; LL['disease'] = dis_of_case[sw]
    # --- bootstrap
    rg = np.random.default_rng(rng0 + 1); dis_idx = [np.where(sub.disease.to_numpy() == d)[0] for d in range(ND)]
    qshare = []; cont = []
    qs = ['less_prev|higher_evid', 'less_prev|lower_evid', 'more_prev|higher_evid', 'more_prev|lower_evid']
    for b in range(NB):
        pick = rg.integers(0, ND, ND); idx = np.concatenate([dis_idx[d] for d in pick]) if len(pick) else np.array([], int)
        if len(idx) == 0: continue
        q = sub.quad.to_numpy()[idx]; row = {k: np.mean(q == k) for k in qs}; row['b'] = b; row['less_prev'] = np.mean(sgn(dprev[idx]) < 0); row['higher_evid'] = np.mean(sgn(dE[idx]) > 0)
        qshare.append(row)
        L = LL.iloc[idx]; cont.append(dict(b=b, Mr_minus_Mp=(L.M_r - L.M_p).mean(), Dboth_minus_Dprev=(L.D_both - L.D_prev).mean(), Mr_minus_Dprev=(L.M_r - L.D_prev).mean(), Mr_minus_M0=(L.M_r - L.M_0).mean(), Dprev_minus_M0=(L.D_prev - L.M_0).mean(), Mr_minus_Mprev=(L.M_r - L.M_prev).mean()))
    return dict(model=m, sub=sub, LL=LL, qshare=pd.DataFrame(qshare).assign(model=m), cont=pd.DataFrame(cont).assign(model=m))


def compliance(m):
    out = []
    for ctx in ['prim', 'rare']: pass
    dfp = pd.read_csv(PRED / f'{m}__counterfactual_primary_care.csv').set_index('vignette_id').reindex(B.v.index); dfr = pd.read_csv(PRED / f'{m}__counterfactual_rare_clinic.csv').set_index('vignette_id').reindex(B.v.index)
    clean = (dfp.parsed_cleanly.fillna(False).astype(bool) & dfr.parsed_cleanly.fillna(False).astype(bool)).to_numpy()
    chp = load(m, 'prim'); chr_ = load(m, 'rare'); ap = arrays(chp); ar = arrays(chr_)
    rows = []
    for nm, mask in [('both_valid', ap[0] & ar[0]), ('clean_both', ap[0] & ar[0] & clean)]:
        cpm = [c if mask[i] else [] for i, c in enumerate(chp)]; crm = [c if mask[i] else [] for i, c in enumerate(chr_)]
        g1 = fitg(cpm); g2 = fitg(crm); swm = mask & (ap[1] != ar[1])
        dl = P[ar[1][swm]] - P[ap[1][swm]]; cp = ap[2] == 1; cr = ar[2] == 1
        rows.append(dict(model=m, set=nm, n=int(mask.sum()), dgp=g2[0] - g1[0], dge=g2[1] - g1[1], switch_rate=swm.sum() / mask.sum(), share_less_prev=np.mean(dl < 0) if swm.sum() else np.nan,
                         gross=int((mask & (cp != cr)).sum()), net=int((mask & cr).sum() - (mask & cp).sum()), dacc=float(((mask & cr).sum() - (mask & cp).sum()) / mask.sum())))
    return rows


if __name__ == '__main__':
    with Pool(6) as p: R = p.map(work, WELL)
    sub = pd.concat([r['sub'] for r in R]); sub.to_csv(S6 / 'switches.csv', index=False)
    pd.concat([r['LL'] for r in R]).to_csv(S6 / 'heldout_ll.csv', index=False)
    qs = pd.concat([r['qshare'] for r in R]); qs.to_csv(S6 / 'boot_quadrants.csv', index=False)
    ct = pd.concat([r['cont'] for r in R]); ct.to_csv(S6 / 'boot_contrasts.csv', index=False)
    # tables
    sub['quad2'] = sub.quad; M = sub.groupby(['model', 'quad2', 'trans']).size().rename('n').reset_index(); M.to_csv(S6 / 'substitution_matrix_by_transition.csv', index=False)
    rows = []
    for m in WELL:
        s_ = sub[sub.model == m]; b = qs[qs.model == m]
        for k in ['less_prev|higher_evid', 'less_prev|lower_evid', 'more_prev|higher_evid', 'more_prev|lower_evid']:
            rows.append(dict(model=m, quad=k, n=int((s_.quad == k).sum()), share=(s_.quad == k).mean(), lo=b[k].quantile(.025), hi=b[k].quantile(.975)))
        rows.append(dict(model=m, quad='ALL_switches', n=len(s_), share=1.0, lo=np.nan, hi=np.nan))
    pd.DataFrame(rows).to_csv(S6 / 'quadrants.csv', index=False)
    rows = []
    LLall = pd.concat([r['LL'] for r in R])
    for m in WELL:
        L = LLall[LLall.model == m]; c = ct[ct.model == m]
        d = dict(model=m, n_switch=len(L), **{f'll_{k}': L[k].mean() for k in ['M_r', 'M_p', 'M_prev', 'M_evi', 'M_0', 'D_prev', 'D_both']})
        for k in ['Mr_minus_Mp', 'Dboth_minus_Dprev', 'Mr_minus_Dprev', 'Mr_minus_M0', 'Dprev_minus_M0', 'Mr_minus_Mprev']:
            d[k] = c[k].mean(); d[k + '_lo'] = c[k].quantile(.025); d[k + '_hi'] = c[k].quantile(.975)
        rows.append(d)
    pd.DataFrame(rows).to_csv(S6 / 'heldout_summary.csv', index=False)
    # equivalence on dge (Study 5 bootstrap)
    g = pd.read_csv(s5.S5 / 'boot_gamma.csv'); pt = pd.read_csv(s5.S5 / 'point.csv').set_index('model'); rows = []
    for m in WELL:
        x = g[g.model == m].dge; lo, hi = x.quantile(.05), x.quantile(.95)
        rows.append(dict(model=m, dge=pt.loc[m, 'dge'], ci90_lo=lo, ci90_hi=hi, equiv_0p03=bool(lo > -.03 and hi < .03), equiv_0p02=bool(lo > -.02 and hi < .02), ci95_lo=x.quantile(.025), ci95_hi=x.quantile(.975)))
    pd.DataFrame(rows).to_csv(S6 / 'equivalence_dge.csv', index=False)
    with Pool(6) as p: Rc = p.map(compliance, WELL)
    pd.DataFrame([x for r in Rc for x in r]).to_csv(S6 / 'compliance.csv', index=False)
    print('done')
