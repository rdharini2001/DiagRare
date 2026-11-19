"""Simulation check: Hausman test rejection rate when data are generated from a true conditional logit (IIA holds)."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
import importlib.util
from common_pl import *
from scipy.special import logsumexp
from scipy.stats import chi2
src=open(Path(__file__).parent/'iia_hausman.py').read().split("rows=[]\nE=b.E0")[0]
exec(src)
rng=np.random.default_rng(7);res=[]
for model in ['qwen2.5-7b','mistral-7b-instruct','granite-3.1-8b']:
    pc=b.predictions(model);keep=[i for i,c in enumerate(pc) if c];y=np.array([pc[i][0] for i in keep])
    X=np.stack([np.broadcast_to(b.P,(len(keep),82)),b.E0[keep]],-1);X=(X-X.mean((0,1)))/X.std((0,1));bF,VF,_=cl(X,y)
    pr=np.exp(X@bF-logsumexp(X@bF,axis=1)[:,None])
    for g in ['organ:GI','cluster:inflammatory arthritides','pair:UC|Crohn','organ:Cardio']:
        idx=np.array([b.di[d] for d in GROUPS[g]]);alt=np.array([j for j in range(82) if j not in set(idx)]);rej=0;N=40
        for s in range(N):
            ys=np.array([rng.choice(82,p=p/p.sum()) for p in pr]);b1,V1,_=cl(X,ys);sel=np.flatnonzero(~np.isin(ys,idx));remap={j:i for i,j in enumerate(alt)}
            b2,V2,_=cl(X[sel][:,alt],np.array([remap[v] for v in ys[sel]]));d=b2-b1;w,Q=np.linalg.eigh(V2-V1);pos=w>1e-10*abs(w).max()
            H=float(d@(Q[:,pos]/w[pos])@Q[:,pos].T@d);rej+=chi2.sf(H,int(pos.sum()))<.05
        res.append(dict(model=model,group=g,sims=N,rejection_rate_at_05=rej/N));print(res[-1],flush=True)
pd.DataFrame(res).to_csv(OUT/'iia_hausman_null_calibration.csv',index=False)
