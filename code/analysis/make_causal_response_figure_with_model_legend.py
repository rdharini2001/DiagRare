from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[2]
TABLES = ROOT / "tables"
OUT = ROOT / "figures"
OUT.mkdir(parents=True, exist_ok=True)

SHORT = {
    "qwen2.5-0.5b":"Qwen2.5-0.5B", "qwen2.5-1.5b":"Qwen2.5-1.5B", "qwen2.5-3b":"Qwen2.5-3B", "qwen2.5-7b":"Qwen2.5-7B",
    "yi-1.5-6b":"Yi-1.5-6B", "yi-1.5-9b":"Yi-1.5-9B", "olmo-2-7b":"OLMo-2-7B", "mistral-7b-instruct":"Mistral-7B",
    "phi-3.5-mini":"Phi-3.5-mini", "glm-4-9b":"GLM-4-9B", "granite-3.1-2b":"Granite-3.1-2B", "granite-3.1-8b":"Granite-3.1-8B",
    "nous-hermes-2-7b":"Nous-Hermes-2-7B", "openchat-3.5":"OpenChat-3.5", "stablelm-zephyr-3b":"StableLM-Zephyr-3B",
    "zephyr-7b":"Zephyr-7B", "falcon-7b":"Falcon-7B", "deepseek-llm-7b":"DeepSeek-LLM-7B", "vicuna-7b":"Vicuna-7B",
    "tinyllama-1.1b":"TinyLlama-1.1B", "med42-8b":"Med42-8B", "medalpaca-7b":"MedAlpaca-7B", "asclepius-7b":"Asclepius-7B",
    "openbiollm-8b":"OpenBioLLM-8B", "biomistral-7b":"BioMistral-7B"
}

plt.rcParams.update({
    "font.family": "DejaVu Serif",
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 6.5,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "axes.facecolor": "white",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#AAB6C3",
    "axes.linewidth": 0.8,
    "grid.color": "#D8E0E8",
    "grid.linestyle": "--",
    "grid.linewidth": 0.7,
    "axes.axisbelow": True,
    "pdf.fonttype": 42,
    "ps.fonttype": 42
})

EVS = ["weak", "medium", "strong"]
PRIORS = ["target_rare", "equal", "target_common"]
PRIOR_LABELS = ["Target rare", "Equal", "Target common"]
EVID_LABELS = ["Weak", "Medium", "Strong"]
REP_MODELS = [("med42-8b", "high response"), ("olmo-2-7b", "medium response"), ("falcon-7b", "low response")]

def draw_surface(ax, cells, model, subtitle):
    grid = np.full((3, 3), np.nan)
    dd = cells[cells.model == model]
    for _, row in dd.iterrows():
        i = EVS.index(row["evidence_level"])
        j = PRIORS.index(row["prior_level"])
        grid[i, j] = row["Pr_target"]
    im = ax.imshow(grid, cmap="RdBu", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(3), PRIOR_LABELS, rotation=35, ha='right')
    ax.set_yticks(range(3), EVID_LABELS)
    ax.set_title(f"{SHORT.get(model, model)}\n{subtitle}", fontsize=8.5)
    for i in range(3):
        for j in range(3):
            v = grid[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.5,
                        color="white" if v <= 0.30 or v >= 0.70 else "#1E2937")
    return im

def main():
    cells = pd.read_csv(TABLES / "causal_response_surface_cells.csv")
    valid = pd.read_csv(TABLES / "causal_response_validation.csv").dropna(subset=["gamma_evidence", "delta_E"]).copy()
    valid["label"] = valid["model"].map(lambda x: SHORT.get(x, x))
    valid = valid.sort_values(["gamma_evidence", "label"]).reset_index(drop=True)
    valid["idx"] = np.arange(1, len(valid) + 1)

    colors = list(plt.get_cmap("tab20").colors) + list(plt.get_cmap("tab20b").colors)
    color_map = {m: colors[i] for i, m in enumerate(valid["model"].tolist())}

    fig = plt.figure(figsize=(10.8, 3.9))
    gs = fig.add_gridspec(1, 5, width_ratios=[1, 1, 1, 1.5, 1.45], wspace=0.45)

    ims = []
    for k, (model, subtitle) in enumerate(REP_MODELS):
        ax = fig.add_subplot(gs[0, k])
        im = draw_surface(ax, cells, model, subtitle)
        ax.set_xlabel("Stated prevalence")
        if k == 0:
            ax.set_ylabel("Target-supportive evidence")
        else:
            ax.set_ylabel("")
        ax.text(-0.18, 1.03, chr(ord('a')+k), transform=ax.transAxes, fontsize=10, fontweight='bold', va='bottom')
        ims.append(im)

    cbar = fig.colorbar(ims[0], ax=[fig.axes[0], fig.axes[1], fig.axes[2]], fraction=0.025, pad=0.01)
    cbar.set_label("Pr(target choice)")

    ax = fig.add_subplot(gs[0, 3])
    ax.grid(True, zorder=0)
    for _, row in valid.iterrows():
        c = color_map[row.model]
        ax.scatter(row.gamma_evidence, row.delta_E, s=70, color=c, edgecolor='white', linewidth=0.9, zorder=3)
        ax.text(row.gamma_evidence, row.delta_E, str(int(row.idx)), ha='center', va='center', fontsize=5.2, color='white', fontweight='bold', zorder=4)
    z = np.polyfit(valid.gamma_evidence, valid.delta_E, 1)
    xs = np.linspace(valid.gamma_evidence.min(), valid.gamma_evidence.max(), 100)
    ax.plot(xs, np.polyval(z, xs), color='#173F5F', lw=1.8, zorder=2)
    r, p = pearsonr(valid.gamma_evidence, valid.delta_E)
    ax.text(0.03, 0.97, fr"$r = {r:.3f},\ p = {p:.1e}$", transform=ax.transAxes, ha='left', va='top', fontsize=7.2,
            bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='#CFD8E3', alpha=0.95))
    ax.set_xlabel(r"Baseline $\gamma_{evidence}$")
    ax.set_ylabel(r"Assigned-evidence effect $\Delta_E$")
    ax.set_title("Cross-model validation", pad=8)
    ax.text(-0.16, 1.03, 'd', transform=ax.transAxes, fontsize=10, fontweight='bold', va='bottom')

    axl = fig.add_subplot(gs[0, 4])
    axl.axis('off')
    axl.text(0.0, 1.02, 'Model legend for panel d', fontsize=8.8, fontweight='bold', va='bottom')
    y_positions = np.linspace(0.96, 0.04, len(valid))
    for y, (_, row) in zip(y_positions, valid.iterrows()):
        c = color_map[row.model]
        axl.scatter([0.04], [y], s=50, color=c, edgecolor='white', linewidth=0.7)
        axl.text(0.04, y, str(int(row.idx)), ha='center', va='center', fontsize=4.8, color='white', fontweight='bold')
        axl.text(0.10, y, row.label, va='center', fontsize=6.5, color='#1E2937')
    axl.set_xlim(0, 1)
    axl.set_ylim(0, 1)

    fig.suptitle('How model choices change when evidence and stated prior are manipulated independently',
                 fontsize=10.5, fontweight='bold', y=0.99)
    out = OUT / 'figure3_causal_response_with_panel_d_model_legend'
    fig.savefig(out.with_suffix('.svg'), bbox_inches='tight')
    fig.savefig(out.with_suffix('.pdf'), bbox_inches='tight')
    fig.savefig(out.with_suffix('.png'), dpi=240, bbox_inches='tight')

if __name__ == '__main__':
    main()
