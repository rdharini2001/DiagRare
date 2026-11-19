import sys,numpy as np,pandas as pd,itertools,warnings
warnings.filterwarnings('ignore')
from scipy.stats import spearmanr,chi2,binomtest
import statsmodels.api as sm
sys.path.insert(0,'rebuttal/scripts')
from common_pl import Bench,ROOT,OUT
from plackett_luce import Choice,fit_plackett_luce
S4=OUT/'study4';B=Bench();E=B.E0;P_=B.P;n=len(B.v);N=len(B.diseases);rng=np.random.default_rng(20261118);pd.set_option('display.width',250);pd.set_option('display.max_columns',40)
D=pd.read_csv(S4/'coefs_raw.csv');SGi=pd.read_csv(S4/'subgroup_indices.csv');SG={k:SGi[k].dropna().astype(int).values for k in SGi}
P=pd.read_csv(OUT/'study3_predictors.csv');P=P[(P.accuracy>=.05)&(P.used_frac>=.5)];MODELS=[m for m in P.model if m in set(D.model)];P=P.set_index('model').loc[MODELS]
tab=lambda g,e:D[(D.group==g)&(D.est==e)].set_index('model')['ge'].reindex(MODELS)
pred={m:B.predictions(m) for m in MODELS};tgt=B.target;di=B.di
def pick(m,i,j):
    c=pred[m][i];return c[j] if len(c)>j else -1
top1={m:np.array([bool(c) and B.diseases[c[0]]==t for c,t in zip(pred[m],tgt)]) for m in MODELS}
in3={m:np.array([any(B.diseases[x]==t for x in c[:3]) for c,t in zip(pred[m],tgt)]) for m in MODELS}
def tailsup(m):
    v=[np.mean([E[i,pick(m,i,1)],E[i,pick(m,i,2)]]) for i in range(n) if len(pred[m][i])>=3];return np.mean(v)
TS={m:tailsup(m) for m in MODELS}
pairs=[(a,b) for a,b in itertools.combinations(MODELS,2) if abs(P.accuracy[a]-P.accuracy[b])<=0.02]
print('analysis panel',len(MODELS),'| accuracy-matched pairs (|dAcc|<=0.02):',len(pairs))
rows=[]
for a,b in pairs:
    both=[i for i in range(n) if pred[a][i] and pred[b][i]];d1=np.array([pick(a,i,0)!=pick(b,i,0) for i in both]);same=~d1
    d23=np.array([(pick(a,i,1),pick(a,i,2))!=(pick(b,i,1),pick(b,i,2)) for i in both])
    bi=np.array(both);r=dict(a=a,b=b,acc_a=P.accuracy[a],acc_b=P.accuracy[b],n=len(both),top1_disagree=d1.mean(),tail_disagree_given_same_top1=d23[same].mean() if same.any() else np.nan)
    for g in ['A','B','C','D_rare','unambiguous']:
        msk=np.isin(bi,SG[g]);r[f'top1_dis_{g}']=d1[msk].mean() if msk.any() else np.nan;r[f'tail_dis_{g}']=d23[msk&same].mean() if (msk&same).any() else np.nan
    # gamma gap vs sampling noise (vignette bootstrap, 100 draws)
    dif=[]
    for _ in range(100):
        idx=rng.integers(0,n,n);fa=fit_plackett_luce([Choice(pred[a][i],P_,E[i]) for i in idx if pred[a][i]],n_starts=1)['gamma_evidence'];fb=fit_plackett_luce([Choice(pred[b][i],P_,E[i]) for i in idx if pred[b][i]],n_starts=1)['gamma_evidence'];dif.append(fa-fb)
    lo,hi=np.percentile(dif,[2.5,97.5]);r.update(ge_a=tab('all','full')[a],ge_b=tab('all','full')[b],gap=tab('all','full')[a]-tab('all','full')[b],gap_lo=lo,gap_hi=hi,gap_distinguishable=bool(lo>0 or hi<0))
    r.update(tail_a=tab('all','tail')[a],tail_b=tab('all','tail')[b],top3_a=in3[a][both].mean(),top3_b=in3[b][both].mean(),tailsup_a=TS[a],tailsup_b=TS[b])
    rows.append(r)
PR=pd.DataFrame(rows);PR.to_csv(S4/'matched_pairs_full.csv',index=False)
print('\n=== ALL accuracy-matched pairs (not selected by gamma)');print(PR[['a','b','acc_a','acc_b','top1_disagree','tail_disagree_given_same_top1','gap','gap_lo','gap_hi','gap_distinguishable','top3_a','top3_b','tailsup_a','tailsup_b']].round(3).to_string(index=False))
print('\nmean disagreement: top1 %.3f; tail(2nd/3rd) given same top1 %.3f'%(PR.top1_disagree.mean(),PR.tail_disagree_given_same_top1.mean()))
for g in ['A','B','C','D_rare']:print(' concentration %s: top1 dis in S %.3f vs unambiguous %.3f | tail dis in S %.3f vs unamb %.3f'%(g,PR[f'top1_dis_{g}'].mean(),PR['top1_dis_unambiguous'].mean(),PR[f'tail_dis_{g}'].mean(),PR['tail_dis_unambiguous'].mean()))
print('pairs with distinguishable gamma_e gap (95%% CI excludes 0): %d of %d'%(PR.gap_distinguishable.sum(),len(PR)))
# concordance of higher-gamma model with better tail support / top3
for nm,ca,cb in [('top3 inclusion','top3_a','top3_b'),('tail evidence support','tailsup_a','tailsup_b')]:
    k=nn=0
    for _,r in PR.iterrows():
        if r.gap==0 or r[ca]==r[cb]:continue
        nn+=1;k+=(r.gap>0)==(r[ca]>r[cb])
    print('higher gamma_e model has higher %s: %d/%d'%(nm,k,nn),'p=%.2f'%binomtest(k,nn,.5).pvalue if nn else '')
# ---- item-level eta analysis
C=np.vstack([top1[m] for m in MODELS]).astype(float);res=[]
for gname,idxs in [('A',SG['A']),('B',SG['B']),('C',SG['C']),('D_rare',SG['D_rare'])]:
    ind=np.zeros(n);ind[idxs]=1
    for k,m in enumerate(MODELS):
        oth=np.delete(C,k,axis=0).mean(0);x=np.log(np.clip(oth,.02,.98)/(1-np.clip(oth,.02,.98)))
        X=sm.add_constant(np.column_stack([x,ind]));
        try:f=sm.GLM(C[k],X,family=sm.families.Binomial()).fit();res.append(dict(model=m,group=gname,eta=f.params[2],se=f.bse[2],acc=P.accuracy[m]))
        except Exception:pass
H=pd.DataFrame(res);H.to_csv(S4/'item_level_eta.csv',index=False);print('\n=== ITEM-LEVEL eta_m (logit(correct) ~ theta_m + a_m*leave-one-out easiness + eta_m*subgroup)')
for g,h in H.groupby('group'):
    h=h[(h.se<3)&h.eta.notna()];w=1/h.se**2;eb=(w*h.eta).sum()/w.sum();Q=(w*(h.eta-eb)**2).sum();dfq=len(h)-1
    ge=tab({'D_rare':'D_rare'}.get(g,g),'full').reindex(h.model);hh=h.set_index('model').acc
    ok=ge.notna()
    print(f' {g}: models {len(h)}, heterogeneity Q={Q:.1f} df={dfq} p={chi2.sf(Q,dfq):.4f}; SD(eta)={h.eta.std():.2f} vs median SE={h.se.median():.2f}; rho(gamma_e^S,eta)={spearmanr(ge[ok],h.set_index("model").eta[ok])[0]:.2f}, rho(acc,eta)={spearmanr(hh[ok],h.set_index("model").eta[ok])[0]:.2f}')
    mp=[(a,b) for a,b in pairs if a in set(h.model) and b in set(h.model)];hs=h.set_index('model')
    zs=[abs(hs.eta[a]-hs.eta[b])/np.sqrt(hs.se[a]**2+hs.se[b]**2) for a,b in mp];print('   accuracy-matched pairs with |z|>1.96 on eta: %d of %d'%(sum(z>1.96 for z in zs),len(zs)))
