from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPO=Path(__file__).resolve().parents[2]
RES=REPO/'results'; OUT=REPO/'figures'; OUT.mkdir(parents=True,exist_ok=True)
NAVY='#173F5F'; BLUE='#2C6CB0'; TEAL='#4AA999'; GOLD='#D7A63D'; CORAL='#D96A4C'; WINE='#8A1538'; INK='#1E2937'; GRID='#D8E0E8'
FAMILY_COL={'Qwen':BLUE,'Mistral':'#2F63B8','Phi':'#143E87','Yi':TEAL,'OLMo':'#596DD2','Falcon':GOLD,'GLM':'#3E91B7','Granite':'#7AA6C2','StableLM':CORAL,'DeepSeek':WINE,'Llama2':'#8E63A9','Llama3':'#2A9D8F','Llama':'#B08968'}
SHORT={'qwen2.5-0.5b':'Qwen2.5-0.5B','qwen2.5-1.5b':'Qwen2.5-1.5B','qwen2.5-3b':'Qwen2.5-3B','qwen2.5-7b':'Qwen2.5-7B','mistral-7b-instruct':'Mistral-7B','phi-3.5-mini':'Phi-3.5-mini','yi-1.5-6b':'Yi-1.5-6B','yi-1.5-9b':'Yi-1.5-9B','olmo-2-7b':'OLMo-2-7B','deepseek-llm-7b':'DeepSeek-LLM-7B','falcon-7b':'Falcon-7B','stablelm-zephyr-3b':'StableLM-Zephyr-3B','granite-3.1-2b':'Granite-3.1-2B','granite-3.1-8b':'Granite-3.1-8B','zephyr-7b':'Zephyr-7B','openchat-3.5':'OpenChat-3.5','glm-4-9b':'GLM-4-9B','nous-hermes-2-7b':'Nous-Hermes-2-7B','medalpaca-7b':'MedAlpaca-7B','asclepius-7b':'Asclepius-7B','med42-8b':'Med42-8B'}
plt.rcParams.update({'font.family':'DejaVu Serif','font.size':7.6,'axes.titlesize':9.2,'axes.labelsize':8.2,'xtick.labelsize':6.8,'ytick.labelsize':6.8,'figure.facecolor':'white','savefig.facecolor':'white','axes.facecolor':'white','axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#AAB6C3','axes.linewidth':.8,'grid.color':GRID,'grid.linestyle':'--','grid.linewidth':.7,'axes.axisbelow':True,'pdf.fonttype':42,'ps.fonttype':42})

def main():
    d=pd.read_csv(RES/'causal_grid_analysis.csv').dropna(subset=['gamma_evidence_revealed','beta_evidence_causal']).copy()
    meta=pd.read_csv(REPO/'data/model_metadata.csv')[['model_tag','family']]
    d=d.merge(meta,left_on='model',right_on='model_tag',how='left').sort_values('gamma_evidence_revealed').reset_index(drop=True)
    d['idx']=np.arange(1,len(d)+1)
    r,p=pearsonr(d.gamma_evidence_revealed,d.beta_evidence_causal)
    fig=plt.figure(figsize=(7.2,4.0)); gs=fig.add_gridspec(1,2,width_ratios=[2.15,1.05],wspace=.18)
    ax=fig.add_subplot(gs[0]); ax.grid(True)
    for _,row in d.iterrows():
        c=FAMILY_COL.get(row.family,BLUE)
        ax.scatter(row.gamma_evidence_revealed,row.beta_evidence_causal,s=62,color=c,edgecolor='white',linewidth=.8,zorder=3)
        ax.text(row.gamma_evidence_revealed,row.beta_evidence_causal,str(int(row.idx)),ha='center',va='center',fontsize=5.0,color='white',fontweight='bold',zorder=4)
    z=np.polyfit(d.gamma_evidence_revealed,d.beta_evidence_causal,1); xs=np.linspace(d.gamma_evidence_revealed.min(),d.gamma_evidence_revealed.max(),100)
    ax.plot(xs,np.polyval(z,xs),color=NAVY,lw=1.8,zorder=2)
    ax.set_xlabel(r'Baseline $\gamma_{evidence}$')
    ax.set_ylabel('Evidence coefficient in randomized experiment')
    ax.set_title('a  Rank-derived coefficient predicts response to assigned evidence',loc='left',fontweight='bold',color=INK,pad=6)
    ax.text(.03,.95,fr'$r={r:.3f},\ p={p:.1e}$',transform=ax.transAxes,ha='left',va='top',fontsize=7.5,bbox=dict(boxstyle='round,pad=.25',fc='white',ec='#CFD8E3',alpha=.95))
    ax2=fig.add_subplot(gs[1]); ax2.axis('off')
    ax2.text(0,.99,'b  Model index',ha='left',va='top',fontsize=9.2,fontweight='bold',color=INK)
    lines=[f"{int(row.idx):>2}. {SHORT.get(row.model,row.model)}" for _,row in d.iterrows()]
    ax2.text(0,.90,'\n'.join(lines),ha='left',va='top',fontsize=5.6,family='monospace',color=INK,linespacing=1.12)
    fig.suptitle('Randomized evidence validates the rank-derived coefficient',fontsize=10.1,fontweight='bold',color=INK,y=.995)
    fig.subplots_adjust(top=.84)
    for ext in ('svg','pdf'):
        fig.savefig(OUT/f'fig2b_causal_validation.{ext}',bbox_inches='tight')
    fig.savefig(OUT/'fig2b_causal_validation.png',dpi=240,bbox_inches='tight')
    plt.close(fig)

if __name__=='__main__': main()
