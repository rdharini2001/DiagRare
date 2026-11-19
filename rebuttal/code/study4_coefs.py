import sys,numpy as np,pandas as pd,itertools,time
from multiprocessing import Pool
from scipy.stats import spearmanr
sys.path.insert(0,'rebuttal/scripts')
from common_pl import Bench,ROOT,OUT
from plackett_luce import Choice,fit_plackett_luce
S4=OUT/'study4';S4.mkdir(exist_ok=True)
B=Bench();E=B.E0;P=B.P;N=len(B.diseases);n=len(B.v)
mm=pd.read_csv(OUT/'model_metrics.csv');mm=mm[(mm.accuracy>=.05)&(mm.used_frac>=.5)];MODELS=[m for m in mm.model if (ROOT/'results/predictions'/f'{m}__baseline.csv').exists()]
# ---- subgroups (frozen definitions)
cs=pd.read_csv(OUT/'conflict_regimes_cases.csv');assert (cs.disease.to_numpy()==B.target).all()
post=P[None,:]+E;s=-np.sort(-post,axis=1);gap2=s[:,0]-s[:,1]
rng=np.random.default_rng(20261118);rare=B.v.is_rare.to_numpy().astype(bool);org=B.v.organ_system.to_numpy();dif=B.v.difficulty.to_numpy()
rare_idx=np.where(rare)[0];match=[]
for (o,d),g in pd.DataFrame({'o':org,'d':dif,'r':rare,'i':np.arange(n)}).groupby(['o','d']):
    r=g[g.r].i.values;c=g[~g.r].i.values
    if len(r) and len(c):match+=list(rng.choice(c,size=min(len(r),len(c)),replace=False))
SG={'all':np.arange(n),'A':np.where(cs.min_abs_margin.to_numpy()<=4)[0],'A2':np.where(cs.min_abs_margin.to_numpy()<=2)[0],'B':np.where(gap2<=2)[0],'C':np.where(B.v.used_confounder_distractor.to_numpy().astype(bool))[0],'D_rare':rare_idx,'D_common_matched':np.array(sorted(match))}
SG['unambiguous']=np.setdiff1d(np.arange(n),np.union1d(SG['A'],SG['B']))
print({k:len(v) for k,v in SG.items()},flush=True)
pd.DataFrame({k:pd.Series(v) for k,v in SG.items()}).to_csv(S4/'subgroup_indices.csv',index=False)
alt_cache={}
def mk(pred,idx,mode):
    out=[]
    for i in idx:
        c=pred[i]
        if not c:continue
        if mode=='full':out.append(Choice(c,P,E[i]))
        elif mode=='top1':out.append(Choice(c[:1],P,E[i]))
        else:
            if len(c)<2:continue
            alt=[k for k in range(N) if k!=c[0]];mp={k:j for j,k in enumerate(alt)};out.append(Choice([mp[x] for x in c[1:]],P[alt],E[i][alt]))
    return out
def fit(pred,idx,mode,minn=40):
    ch=mk(pred,idx,mode)
    if len(ch)<minn:return (np.nan,np.nan,len(ch))
    r=fit_plackett_luce(ch,n_starts=2);return (r['gamma_evidence'],r['gamma_prior'],len(ch))
def work(m):
    pred=B.predictions(m);rows=[];t=time.time()
    for g,idx in SG.items():
        for mode in ['top1','tail','full']:
            ge,gp,k=fit(pred,idx,mode);rows.append(dict(model=m,group=g,est=mode,ge=ge,gp=gp,n=k))
    # disease-level halves
    dis=sorted(set(B.target));r2=np.random.default_rng(20261118);r2.shuffle(dis);h1=set(dis[:len(dis)//2]);tg=np.array(B.target)
    for hn,hs in [('H1',h1),('H2',set(dis)-h1)]:
        sel=np.where(np.isin(tg,list(hs)))[0]
        for g in ['all','A','B','C']:
            idx=np.intersect1d(SG[g],sel)
            for mode in ['top1','tail','full']:
                ge,gp,k=fit(pred,idx,mode,minn=25);rows.append(dict(model=m,group=f'{g}|{hn}',est=mode,ge=ge,gp=gp,n=k))
    # split-half reliability draws (vignette-level), 60 splits
    rs=np.random.default_rng(7)
    for g in ['all','A','B','C']:
        for mode in ['top1','tail','full']:
            for b in range(60):
                idx=SG[g];perm=rs.permutation(idx);a,c=perm[:len(perm)//2],perm[len(perm)//2:]
                ga=fit(pred,a,mode,minn=20)[0];gc=fit(pred,c,mode,minn=20)[0];rows.append(dict(model=m,group=f'{g}|split{b}',est=mode,ge=ga,gp=gc,n=0))
    print(m,'done %.0fs'%(time.time()-t),flush=True);return rows
if __name__=='__main__':
    with Pool(6) as p:R=p.map(work,MODELS)
    D=pd.DataFrame([r for rr in R for r in rr]);D.to_csv(S4/'coefs_raw.csv',index=False);print(D.shape)
