from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
RES = REPO / 'results'
OUT = REPO / 'figures'
OUT.mkdir(parents=True, exist_ok=True)

NAVY='#173F5F'; BLUE='#2C6CB0'; TEAL='#4AA999'; GOLD='#D7A63D'; CORAL='#D96A4C'; WINE='#8A1538'; INK='#1E2937'; SLATE='#6B7785'; GRID='#D8E0E8'
plt.rcParams.update({
    'font.family':'DejaVu Serif','font.size':8.0,'axes.titlesize':10,'axes.labelsize':8.5,
    'xtick.labelsize':7.0,'ytick.labelsize':7.2,'legend.fontsize':6.6,
    'figure.facecolor':'white','savefig.facecolor':'white','axes.facecolor':'white',
    'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#AAB6C3',
    'axes.linewidth':0.8,'grid.color':GRID,'grid.linestyle':'--','grid.linewidth':0.7,
    'axes.axisbelow':True,'pdf.fonttype':42,'ps.fonttype':42
})

LABELS={
    'mistral-7b-instruct':'Mistral-7B', 'olmo-2-7b':'OLMo-2-7B',
    'phi-3.5-mini':'Phi-3.5-mini','qwen2.5-0.5b':'Qwen2.5-0.5B',
    'qwen2.5-1.5b':'Qwen2.5-1.5B','qwen2.5-3b':'Qwen2.5-3B',
    'qwen2.5-7b':'Qwen2.5-7B','yi-1.5-6b':'Yi-1.5-6B'
}
CONDITIONS=[
    ('baseline','Baseline',NAVY,'o'),
    ('cot','Step-by-step',BLUE,'s'),
    ('debias','Evidence-focused',TEAL,'D'),
    ('counterfactual_rare_clinic','Rare-disease referral',GOLD,'^'),
    ('counterfactual_primary_care','Primary care',CORAL,'v'),
]

def main():
    df=pd.read_csv(RES/'open_weight_summary.csv')
    d=df[(df['split']=='ALL') & (df['model'].isin(LABELS))].copy()
    piv=d.pivot_table(index='model',columns='condition',values='top1_accuracy',aggfunc='first')
    # Order by baseline accuracy, so rows can be read without crossing lines.
    order=piv['baseline'].sort_values().index.tolist()
    y=np.arange(len(order))
    fig,ax=plt.subplots(figsize=(7.1,3.8))
    ax.grid(True,axis='x'); ax.grid(False,axis='y')
    for cond,label,color,marker in CONDITIONS:
        xs=[piv.loc[m,cond] if cond in piv.columns else np.nan for m in order]
        ax.scatter(xs,y,s=42,color=color,marker=marker,edgecolor='white',linewidth=.7,label=label,zorder=3)
    ax.set_yticks(y,[LABELS[m] for m in order])
    ax.set_xlim(0,0.75)
    ax.set_xlabel('Top-1 diagnostic accuracy')
    ax.set_title('Prompt framing changes accuracy differently across checkpoints',loc='left',fontweight='bold',color=INK,pad=7)
    ax.legend(frameon=False,ncol=3,loc='lower right',columnspacing=1.0,handletextpad=.4)
    ax.tick_params(color='#97A5B3',labelcolor=INK)
    fig.tight_layout()
    for ext in ('svg','pdf'):
        fig.savefig(OUT/f'prompting_ablation.{ext}',bbox_inches='tight')
    fig.savefig(OUT/'prompting_ablation.png',dpi=240,bbox_inches='tight')
    plt.close(fig)

if __name__=='__main__':
    main()
