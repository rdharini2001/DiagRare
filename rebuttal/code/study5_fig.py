import pandas as pd, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
S='rebuttal/results/study5/'; c=pd.read_csv(S+'cis.csv'); p=pd.read_csv(S+'point.csv')
W=['mistral-7b-instruct','olmo-2-7b','phi-3.5-mini','qwen2.5-1.5b','qwen2.5-3b','qwen2.5-7b']; lab=['Mistral-7B','OLMo-2-7B','Phi-3.5-mini','Qwen2.5-1.5B','Qwen2.5-3B','Qwen2.5-7B']
def get(s): x=c[c.stat==s].set_index('model').loc[W]; return x
fig,ax=plt.subplots(2,2,figsize=(11,8)); y=np.arange(6)[::-1]
def err(a,s,col,off=0,l=None):
    x=get(s); a.errorbar(x.est,y+off,xerr=[x.est-x.lo,x.hi-x.est],fmt='o',color=col,capsize=3,label=l)
a=ax[0,0]; err(a,'dgp','#1f77b4',0.12,'Δγ prior'); err(a,'dge','#d95f02',-0.12,'Δγ evidence'); a.axvline(0,color='k',lw=.6); a.set_yticks(y); a.set_yticklabels(lab); a.set_title('A. Coefficient shift, rare clinic − primary care'); a.legend(fontsize=8,loc='upper center',bbox_to_anchor=(0.5,-0.14),ncol=2); a.set_xlabel('Δ coefficient (disease-cluster 95% CI)')
a=ax[0,1]; err(a,'dacc_rare','#1b9e77',0.2,'rare targets'); err(a,'dacc_common','#7570b3',0,'common targets'); err(a,'dacc_all','#666',-0.2,'overall'); a.axvline(0,color='k',lw=.6); a.set_yticks(y); a.set_yticklabels(lab); a.set_title('B. Accuracy change (both-valid cases)'); a.legend(fontsize=8); a.set_xlabel('Δ top-1 accuracy')
a=ax[1,0]; err(a,'did_ont','#1b9e77',0.2,'ontology rare label'); err(a,'did_eu','#d95f02',0,'< 5/10,000 definition'); err(a,'did_terc','#7570b3',-0.2,'prevalence terciles'); a.axvline(0,color='k',lw=.6); a.set_yticks(y); a.set_yticklabels(lab); a.set_title('C. Rare minus common accuracy change, three rarity definitions'); a.legend(fontsize=8); a.set_xlabel('difference-in-differences')
a=ax[1,1]; err(a,'sw_share_lower_prev','#1f77b4',0.12,'share of switches to LOWER prevalence'); err(a,'sw_share_higher_E','#d95f02',-0.12,'share of switches to HIGHER evidence'); a.axvline(.5,color='k',lw=.6); a.set_yticks(y); a.set_yticklabels(lab); a.set_xlim(0,1); a.set_title('D. Direction of top-1 switches'); a.legend(fontsize=8,loc='upper center',bbox_to_anchor=(0.5,-0.14),ncol=1); a.set_xlabel('share (0.5 = no preference)')
plt.tight_layout(); plt.savefig('rebuttal/figures/fig14_context_reclassification.png',dpi=160); plt.savefig('rebuttal/figures/fig14_context_reclassification.pdf')
