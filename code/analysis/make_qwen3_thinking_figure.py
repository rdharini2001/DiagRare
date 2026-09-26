from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPO=Path(__file__).resolve().parents[2]
RES=REPO/'results'; OUT=REPO/'figures'; OUT.mkdir(parents=True,exist_ok=True)
NAVY='#173F5F'; BLUE='#2C6CB0'; TEAL='#4AA999'; GOLD='#D7A63D'; CORAL='#D96A4C'; INK='#1E2937'; SLATE='#6B7785'; GRID='#D8E0E8'
plt.rcParams.update({'font.family':'DejaVu Serif','font.size':7.5,'axes.titlesize':8.8,'axes.labelsize':7.8,'xtick.labelsize':6.8,'ytick.labelsize':6.8,'figure.facecolor':'white','savefig.facecolor':'white','axes.facecolor':'white','axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#AAB6C3','axes.linewidth':.8,'grid.color':GRID,'grid.linestyle':'--','grid.linewidth':.7,'axes.axisbelow':True,'pdf.fonttype':42,'ps.fonttype':42})
CONDS=['no_think','think_low','think_med','think_high']; LABELS=['No think','Think\n512','Think\n2048','Think\n8192']; COLORS=[NAVY,BLUE,TEAL,GOLD]

def panel(ax,df,col,title,ylabel,lowcov=False):
    vals=[]
    for c in CONDS:
        s=df[df.condition==c]
        vals.append(float(s[col].iloc[0]) if len(s) and pd.notna(s[col].iloc[0]) else np.nan)
    x=np.arange(4)
    present=np.isfinite(vals)
    ax.plot(x[present],np.asarray(vals)[present],color='#9BA8B5',lw=1.1,ls='--',zorder=1)
    for i,v in enumerate(vals):
        if np.isfinite(v):
            if lowcov and i in (1,2):
                ax.scatter(i,v,s=58,facecolor='white',edgecolor=COLORS[i],linewidth=1.6,zorder=3)
            else:
                ax.scatter(i,v,s=58,color=COLORS[i],edgecolor='white',linewidth=.7,zorder=3)
    ax.set_xticks(x,LABELS)
    ax.set_ylabel(ylabel)
    ax.set_title(title,loc='left',fontweight='bold',color=INK,pad=5)
    ax.grid(True,axis='y'); ax.grid(False,axis='x')
    ax.tick_params(color='#97A5B3',labelcolor=INK)
    if not np.isfinite(vals[-1]):
        lo,hi=ax.get_ylim(); yy=lo+.05*(hi-lo)
        ax.scatter(3,yy,marker='x',s=34,color=SLATE,zorder=4)
        ax.text(3,yy+.05*(hi-lo),'no completed run',ha='center',va='bottom',fontsize=5.7,color=SLATE)

def main():
    df=pd.read_csv(RES/'qwen3_thinking_budget_analysis.csv')
    fig,axs=plt.subplots(2,2,figsize=(7.2,5.2),gridspec_kw={'wspace':.30,'hspace':.42})
    panel(axs[0,0],df,'top1_accuracy','a  Primary diagnostic accuracy','Accuracy')
    panel(axs[0,1],df,'gamma_evidence','b  Evidence coefficient','Evidence responsiveness',lowcov=True)
    panel(axs[1,0],df,'delta_E','c  Randomized evidence effect',r'$\Delta_E$')
    panel(axs[1,1],df,'recovery_rate','d  Sequential recovery','Recovery rate')
    fig.suptitle('Qwen3-8B under fixed reasoning budgets',fontsize=10.2,fontweight='bold',color=INK,y=.995)
    fig.text(.5,.012,'Hollow markers in panel b denote low-coverage coefficient estimates.',ha='center',fontsize=6.3,color=SLATE)
    fig.subplots_adjust(bottom=.10,top=.91)
    for ext in ('svg','pdf'):
        fig.savefig(OUT/f'fig6_qwen3_thinking_budget.{ext}',bbox_inches='tight')
    fig.savefig(OUT/'fig6_qwen3_thinking_budget.png',dpi=240,bbox_inches='tight')
    plt.close(fig)

if __name__=='__main__': main()
