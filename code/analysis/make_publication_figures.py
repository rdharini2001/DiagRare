from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

REPO = Path(__file__).resolve().parents[2]
RES = REPO / 'results'
OUT = REPO / 'figures'
OUT.mkdir(parents=True, exist_ok=True)

NAVY='#173F5F'; BLUE='#2C6CB0'; SKY='#6CB7D4'; TEAL='#4AA999'; GOLD='#D7A63D'; CORAL='#D96A4C'; WINE='#8A1538'; INK='#1E2937'; SLATE='#6B7785'; GRID='#D8E0E8'
plt.rcParams.update({
    'font.family':'DejaVu Serif','font.size':7.5,'axes.titlesize':9.2,'axes.labelsize':8.0,
    'xtick.labelsize':6.7,'ytick.labelsize':6.7,'legend.fontsize':6.5,'figure.facecolor':'white',
    'savefig.facecolor':'white','axes.facecolor':'white','axes.spines.top':False,'axes.spines.right':False,
    'axes.edgecolor':'#AAB6C3','axes.linewidth':0.8,'grid.color':GRID,'grid.linestyle':'--','grid.linewidth':0.7,
    'axes.axisbelow':True,'pdf.fonttype':42,'ps.fonttype':42
})
SHORT={
'qwen2.5-0.5b':'Qwen2.5-0.5B','qwen2.5-1.5b':'Qwen2.5-1.5B','qwen2.5-3b':'Qwen2.5-3B','qwen2.5-7b':'Qwen2.5-7B',
'yi-1.5-6b':'Yi-1.5-6B','yi-1.5-9b':'Yi-1.5-9B','olmo-2-7b':'OLMo-2-7B','mistral-7b-instruct':'Mistral-7B',
'phi-3.5-mini':'Phi-3.5-mini','glm-4-9b':'GLM-4-9B','granite-3.1-2b':'Granite-3.1-2B','granite-3.1-8b':'Granite-3.1-8B',
'nous-hermes-2-7b':'Nous-Hermes-2-7B','openchat-3.5':'OpenChat-3.5','stablelm-zephyr-3b':'StableLM-Zephyr-3B',
'zephyr-7b':'Zephyr-7B','falcon-7b':'Falcon-7B','deepseek-llm-7b':'DeepSeek-LLM-7B','vicuna-7b':'Vicuna-7B',
'tinyllama-1.1b':'TinyLlama-1.1B','med42-8b':'Med42-8B','medalpaca-7b':'MedAlpaca-7B','asclepius-7b':'Asclepius-7B',
'openbiollm-8b':'OpenBioLLM-8B','biomistral-7b':'BioMistral-7B'
}
FAMILY={'Qwen':BLUE,'Mistral':'#2F63B8','Phi':'#143E87','Yi':TEAL,'OLMo':'#596DD2','Falcon':GOLD,'GLM':'#3E91B7','Granite':'#7AA6C2','StableLM':CORAL,'DeepSeek':WINE,'Llama2':SKY,'Llama3':'#2A9D8F','Llama':'#B08968'}


def base(ax):
    ax.grid(True,zorder=0)
    ax.tick_params(color='#97A5B3',labelcolor=INK)
    ax.spines['left'].set_color('#AAB6C3'); ax.spines['bottom'].set_color('#AAB6C3')

def save(fig, stem):
    fig.savefig(OUT/f'{stem}.svg',bbox_inches='tight')
    fig.savefig(OUT/f'{stem}.pdf',bbox_inches='tight')
    fig.savefig(OUT/f'{stem}.png',dpi=240,bbox_inches='tight')
    plt.close(fig)

def family_map():
    p=REPO / 'data' / 'model_metadata.csv'
    m=pd.read_csv(p)
    return dict(zip(m.model_tag,m.family))


def fig2_landscape():
    d=pd.read_csv(RES/'per_master_table.csv')
    d=d[(d.source=='open_weight')&(d.condition=='baseline')].dropna(subset=['gamma_prior','gamma_evidence']).copy()
    d=d.sort_values('gamma_evidence').reset_index(drop=True)
    d['idx']=np.arange(1,len(d)+1)
    fam=family_map()
    fig=plt.figure(figsize=(7.2,4.15))
    gs=fig.add_gridspec(1,2,width_ratios=[2.15,1],wspace=.16)
    ax=fig.add_subplot(gs[0]); base(ax)
    ax.axhline(0,color='#BAC4D0',lw=.8); ax.axvline(0,color='#BAC4D0',lw=.8)
    for _,r in d.iterrows():
        c=FAMILY.get(fam.get(r.model,''),BLUE)
        ax.scatter(r.gamma_prior,r.gamma_evidence,s=64,color=c,edgecolor='white',linewidth=.8,zorder=3)
        ax.text(r.gamma_prior,r.gamma_evidence,str(int(r.idx)),ha='center',va='center',fontsize=5.1,color='white',fontweight='bold',zorder=4)
    ax.set_xlabel(r'Disease-prior coefficient, $\gamma_{prior}$')
    ax.set_ylabel(r'Evidence coefficient, $\gamma_{evidence}$')
    ax.set_title('Prior and evidence coefficients across open-weight checkpoints',loc='left',fontweight='bold',color=INK,pad=6)
    ax2=fig.add_subplot(gs[1]); ax2.axis('off')
    lines=[f"{int(r.idx):>2}. {SHORT.get(r.model,r.model)}" for _,r in d.iterrows()]
    h=(len(lines)+1)//2
    ax2.text(.0,.98,'Model index',va='top',fontsize=7.5,fontweight='bold',color=INK)
    ax2.text(.0,.92,'\n'.join(lines[:h]),va='top',fontsize=5.9,color=INK,family='monospace')
    ax2.text(.54,.92,'\n'.join(lines[h:]),va='top',fontsize=5.9,color=INK,family='monospace')
    save(fig,'fig2_prior_evidence_landscape')


def fig_a1_sensitivity():
    d=pd.read_csv(RES/'supplementary/per_sensitivity_grid_summary.csv')
    d=d[(d.source=='open_weight')].dropna(subset=['gamma_evidence_min','gamma_evidence_max']).sort_values('gamma_evidence_max')
    y=np.arange(len(d))
    fig,ax=plt.subplots(figsize=(5.6,3.5)); base(ax)
    ax.hlines(y,d.gamma_evidence_min,d.gamma_evidence_max,color=BLUE,lw=4,alpha=.9)
    ax.scatter(d.gamma_evidence_min,y,color=NAVY,s=20,zorder=3)
    ax.scatter(d.gamma_evidence_max,y,color=TEAL,s=20,zorder=3)
    ax.set_yticks(y,[SHORT.get(m,m) for m in d.model])
    ax.set_xlabel(r'Range of $\gamma_{evidence}$ across 16 evidence-score settings')
    ax.set_title('Evidence responsiveness is stable to evidence-score parameterization',loc='left',fontweight='bold',color=INK,pad=6)
    ax.legend(handles=[Patch(color=NAVY,label='Minimum'),Patch(color=TEAL,label='Maximum')],frameon=False,loc='lower right')
    save(fig,'fig_a1_sensitivity_grid')


def fig_a2_difficulty():
    d=pd.read_csv(RES/'supplementary/per_by_difficulty.csv')
    models=['qwen2.5-0.5b','qwen2.5-1.5b','qwen2.5-3b','qwen2.5-7b','mistral-7b-instruct','phi-3.5-mini']
    tiers=['easy','medium','hard']; colors=[SKY,BLUE,GOLD]
    fig,ax=plt.subplots(figsize=(6.6,3.5)); base(ax)
    x=np.arange(len(models)); w=.24
    for i,t in enumerate(tiers):
        vals=[]
        for m in models:
            s=d[(d.source=='open_weight')&(d.model==m)&(d.difficulty==t)]
            vals.append(float(s.gamma_evidence.iloc[0]) if len(s) else np.nan)
        ax.bar(x+(i-1)*w,vals,w,color=colors[i],label=t.capitalize())
    ax.axhline(0,color='#AAB6C3',lw=.8)
    ax.set_xticks(x,[SHORT[m] for m in models],rotation=15,ha='right')
    ax.set_ylabel(r'$\gamma_{evidence}$')
    ax.set_title('Evidence responsiveness across case difficulty',loc='left',fontweight='bold',color=INK,pad=6)
    ax.legend(frameon=False,ncol=3,loc='upper left')
    save(fig,'fig_a2_difficulty_stratified')


def fig3_interventions():
    per=pd.read_csv(RES/'per_master_table.csv')
    ow=pd.read_csv(RES/'open_weight_summary.csv')
    sc=pd.read_csv(RES/'open_weight_selfconsistency_summary.csv')
    lora=pd.read_csv(RES/'lora_summary.csv')
    fig,axs=plt.subplots(1,2,figsize=(7.2,3.4),gridspec_kw={'wspace':.33})

    ax=axs[0]; base(ax)
    splits=[('original','Original cases'),('expanded_heldout','Held-out organ systems')]
    before=per[per.source=='lora_pilot_before'].set_index('split')
    after=per[per.source=='lora_pilot'].set_index('split')
    for i,(sp,lab) in enumerate(splits):
        bacc=float(ow[(ow.model=='qwen2.5-7b')&(ow.condition=='baseline')&(ow.split==sp)].top1_accuracy.iloc[0])
        aacc=float(lora[(lora.condition==sp)&(lora.split==sp)].top1_accuracy.iloc[0])
        bg=float(before.loc[sp,'gamma_evidence']); ag=float(after.loc[sp,'gamma_evidence'])
        ax.annotate('',xy=(100*aacc,ag),xytext=(100*bacc,bg),arrowprops=dict(arrowstyle='->',lw=1.8,color=[BLUE,TEAL][i]))
        ax.scatter([100*bacc],[bg],s=58,color='white',edgecolor=[BLUE,TEAL][i],linewidth=1.7,zorder=3)
        ax.scatter([100*aacc],[ag],s=58,color=[BLUE,TEAL][i],edgecolor='white',linewidth=.7,zorder=3,label=lab)
    ax.set_xlabel('Top-1 accuracy (%)'); ax.set_ylabel(r'$\gamma_{evidence}$')
    ax.set_title('a  QLoRA adaptation',loc='left',fontweight='bold',color=INK,pad=6)
    ax.legend(frameon=False,loc='lower right')

    ax=axs[1]; base(ax)
    models=[('qwen2.5-3b','Qwen2.5-3B'),('qwen2.5-7b','Qwen2.5-7B')]
    scper=per[per.source=='self_consistency'].set_index('model')
    for i,(m,lab) in enumerate(models):
        bacc=float(ow[(ow.model==m)&(ow.condition=='baseline')&(ow.split=='ALL')].top1_accuracy.iloc[0])
        sacc=float(sc[(sc.model==m+'-selfcons5')&(sc.split=='ALL')].top1_accuracy.iloc[0])
        bg=float(per[(per.source=='open_weight')&(per.model==m)&(per.condition=='baseline')&(per.split=='ALL')].gamma_evidence.iloc[0])
        sg=float(scper.loc[m+'-selfcons5','gamma_evidence'])
        c=[GOLD,WINE][i]
        ax.annotate('',xy=(100*sacc,sg),xytext=(100*bacc,bg),arrowprops=dict(arrowstyle='->',lw=1.8,color=c))
        ax.scatter([100*bacc],[bg],s=58,color='white',edgecolor=c,linewidth=1.7,zorder=3)
        ax.scatter([100*sacc],[sg],s=58,color=c,edgecolor='white',linewidth=.7,zorder=3,label=lab)
    ax.set_xlabel('Top-1 accuracy (%)'); ax.set_ylabel(r'$\gamma_{evidence}$')
    ax.set_title('b  Self-consistency',loc='left',fontweight='bold',color=INK,pad=6)
    ax.legend(frameon=False,loc='lower right')
    fig.suptitle('Accuracy and evidence responsiveness respond differently to model interventions',fontsize=9.7,fontweight='bold',color=INK,y=.99)
    fig.subplots_adjust(top=.84)
    save(fig,'fig3_interventions')

if __name__=='__main__':
    fig2_landscape(); fig_a1_sensitivity(); fig_a2_difficulty(); fig3_interventions()
    print('regenerated open-weight-only figures')


def main():
    fig2_landscape(); fig_a1_sensitivity(); fig_a2_difficulty(); fig3_interventions()

