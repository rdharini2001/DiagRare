"""The paper's 13 figures, same layouts as the submission, one shared palette (figtheme). Inputs: tables/. Run: python figures/src/paper_figures.py"""
import sys, warnings
import numpy as np, pandas as pd
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent))
from figtheme import *
from scipy.stats import pearsonr
warnings.filterwarnings('ignore')
setup()
MAIN = ROOT / 'tables'; SUP = MAIN / 'supplementary'
per = pd.read_csv(MAIN / 'per_master_table.csv'); acc = pd.read_csv(MAIN / 'open_weight_summary.csv'); meta = pd.read_csv(ROOT / 'data/metadata/model_metadata.csv').set_index('model_tag')
NAMES = {'qwen2.5-0.5b': 'Qwen2.5-0.5B', 'qwen2.5-1.5b': 'Qwen2.5-1.5B', 'qwen2.5-3b': 'Qwen2.5-3B', 'qwen2.5-7b': 'Qwen2.5-7B', 'yi-1.5-6b': 'Yi-1.5-6B', 'yi-1.5-9b': 'Yi-1.5-9B',
         'olmo-2-7b': 'OLMo-2-7B', 'mistral-7b-instruct': 'Mistral-7B', 'phi-3.5-mini': 'Phi-3.5-mini', 'falcon-7b': 'Falcon-7B', 'stablelm-zephyr-3b': 'StableLM-Zephyr-3B',
         'zephyr-7b': 'Zephyr-7B', 'granite-3.1-2b': 'Granite-3.1-2B', 'granite-3.1-8b': 'Granite-3.1-8B', 'openchat-3.5': 'OpenChat-3.5', 'glm-4-9b': 'GLM-4-9B',
         'nous-hermes-2-7b': 'Nous-Hermes-2-7B', 'deepseek-llm-7b': 'DeepSeek-LLM-7B', 'med42-8b': 'Med42-8B', 'medalpaca-7b': 'MedAlpaca-7B', 'asclepius-7b': 'Asclepius-7B'}
nm = lambda m: NAMES.get(m, m)
medcol = lambda ms: [ORANGE if (m in meta.index and bool(meta.loc[m, 'is_medical_specialized'])) else BLUE for m in ms]
base_open = per[(per.source == 'open_weight') & (per.condition == 'baseline')]
CURRENT = 'paper'
def sv(fig, name): save(fig, name, CURRENT)


def scatter_with_index(ax_s, ax_i, d, xcol, ycol, xlabel, ylabel, title, hollow=None, line=True, stat=True, ylim=None, xlim=None, err=None, cols=2):
    x = d[xcol].to_numpy(); y = d[ycol].to_numpy()
    if err is not None: ax_s.errorbar(x, y, yerr=err, fmt='none', ecolor='#cfcdc4', elinewidth=1, zorder=2)
    if line:
        b = np.polyfit(x, y, 1); xs = np.linspace(x.min(), x.max(), 50); ax_s.plot(xs, np.polyval(b, xs), color=NAVY, lw=1.6, zorder=3)
    numbered(ax_s, x, y, medcol(d.model), hollow=hollow)
    if stat:
        r, p = pearsonr(x, y); ax_s.text(0.03, 0.97, p_text(r, p), transform=ax_s.transAxes, va='top', fontsize=8.5, color=INK2)
    ax_s.set_xlabel(xlabel); ax_s.set_ylabel(ylabel); ax_s.set_title(title)
    if ylim: ax_s.set_ylim(*ylim)
    if xlim: ax_s.set_xlim(*xlim)
    model_index(ax_i, [nm(m) for m in d.model], cols=cols)


def fig1():
    a = acc[(acc.condition == 'baseline') & (acc.split == 'ALL')][['model', 'top1_accuracy']]
    d = base_open.merge(a, on='model').dropna(subset=['gamma_evidence', 'top1_accuracy']).sort_values('top1_accuracy').reset_index(drop=True); d['acc'] = 100 * d.top1_accuracy
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.6, 4.8), gridspec_kw={'width_ratios': [2.3, 1.25], 'wspace': 0.04})
    scatter_with_index(a1, a2, d, 'acc', 'gamma_evidence', 'Top-1 diagnostic accuracy (%)', r'Evidence responsiveness, $\gamma_{\rm evidence}$', 'Evidence responsiveness is associated with diagnostic accuracy',
                       hollow=(d.used_frac < 0.5).to_numpy(), err=[d.gamma_evidence - d.gamma_evidence_lo, d.gamma_evidence_hi - d.gamma_evidence], xlim=(-3, 75), ylim=(-0.12, 0.43))
    legend_markers(a1, [('General', BLUE, 'o', False), ('Medical', ORANGE, 'o', False), ('< 50% usable outputs', MUTED, 'o', True)], loc='lower right')
    sv(fig, 'paper_fig01_evidence_vs_accuracy')


def fig2():
    d = pd.read_csv(MAIN / 'causal_grid_analysis.csv').dropna(subset=['beta_evidence_causal', 'gamma_evidence_revealed']).sort_values('gamma_evidence_revealed').reset_index(drop=True); d['model'] = d['model']
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.6, 4.8), gridspec_kw={'width_ratios': [2.3, 1.25], 'wspace': 0.04})
    scatter_with_index(a1, a2, d, 'gamma_evidence_revealed', 'beta_evidence_causal', r'Baseline $\gamma_{\rm evidence}$', 'Evidence coefficient in randomized experiment', 'Rank-derived coefficient predicts response to assigned evidence',
                       err=1.96 * d.beta_evidence_causal_se.to_numpy())
    sv(fig, 'paper_fig02_independent_validation')


def fig3():
    cells = pd.read_csv(MAIN / 'causal_response_surface_cells.csv'); ates = pd.read_csv(MAIN / 'causal_response_surface.csv'); val = pd.read_csv(MAIN / 'causal_response_validation.csv')
    rk = ates.dropna(subset=['delta_E']).sort_values('delta_E', ascending=False); reps = [(rk.iloc[0].model, 'high response'), (rk.iloc[len(rk) // 2].model, 'median response'), (rk.iloc[-1].model, 'low response')]
    EV, PR = ['strong', 'medium', 'weak'], ['target_rare', 'equal', 'target_common']
    fig = plt.figure(figsize=(14, 4.2)); gs = fig.add_gridspec(1, 7, width_ratios=[1, 1, 1, 0.05, 0.3, 1.7, 0.75], wspace=0.45)
    for k, (m, lab) in enumerate(reps):
        ax = fig.add_subplot(gs[0, k]); g = np.full((3, 3), np.nan)
        for _, r_ in cells[cells.model == m].iterrows(): g[EV.index(r_.evidence_level), PR.index(r_.prior_level)] = r_.Pr_target
        im = heatmap(ax, g, SEQ)
        for i in range(3):
            for j in range(3):
                if np.isfinite(g[i, j]): ax.text(j, i, f'{g[i, j]:.2f}', ha='center', va='center', fontsize=8.8, color='white' if g[i, j] > 0.55 else INK)
        ax.set_xticks(range(3)); ax.set_xticklabels(['rare', 'equal', 'common'], fontsize=8.5); ax.set_yticks(range(3)); ax.set_yticklabels(['strong', 'medium', 'weak'] if k == 0 else [])
        ax.set_title(f'{nm(m)}\n{lab}', fontsize=9.5, pad=6, loc='center'); [s.set_visible(False) for s in ax.spines.values()]; ax.tick_params(length=0)
        if k == 0: ax.set_ylabel('Evidence strength');
        if k == 1: ax.set_xlabel('Stated prior of target')
        letter(ax, 'abc'[k], dx=0.0, dy=1.2)
    cb = fig.colorbar(im, cax=fig.add_subplot(gs[0, 3])); cb.outline.set_visible(False); cb.set_label('Pr(choose target)', fontsize=8.5); cb.ax.tick_params(length=0, labelsize=8)
    v = val.dropna(subset=['delta_E', 'gamma_evidence']).sort_values('gamma_evidence').reset_index(drop=True); ax = fig.add_subplot(gs[0, 5]); ai = fig.add_subplot(gs[0, 6])
    scatter_with_index(ax, ai, v, 'gamma_evidence', 'delta_E', r'Baseline $\gamma_{\rm evidence}$', r'Finite-difference effect, $\Delta_E$', 'Cross-model validation', err=[(v.delta_E - v.delta_E_ci_lo).clip(lower=0), (v.delta_E_ci_hi - v.delta_E).clip(lower=0)], cols=1); letter(ax, 'd', dx=0.0, dy=1.2)
    sv(fig, 'paper_fig03_causal_response_surface')


def fig4():
    an = pd.read_csv(MAIN / 'sequential_anchoring_analysis.csv').dropna(subset=['recovery_rate']); an = an[an.n_anchored_at_step1 >= 5].sort_values('recovery_rate')
    va = pd.read_csv(MAIN / 'sequential_anchoring_validation.csv').sort_values('gamma_evidence').reset_index(drop=True)
    fig = plt.figure(figsize=(12.6, 4.8)); gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1.2, 0.5], wspace=0.5); ax = fig.add_subplot(gs[0, 0]); y = np.arange(len(an))
    ax.barh(y, an.recovery_rate, height=0.68, color='#8db8ee', zorder=3); ax.hlines(y, an.recovery_ci_lo, an.recovery_ci_hi, color=NAVY, lw=1.1, zorder=4)
    ax.set_yticks(y); ax.set_yticklabels([nm(m) for m in an.model], fontsize=8.3); ax.set_xlim(0, 1.03); ax.set_xlabel('Recovery rate'); ax.set_title('Revision after an initial confounder diagnosis', fontsize=10); ax.grid(axis='y', visible=False); letter(ax, 'a', dx=-0.01, dy=1.04)
    ax2 = fig.add_subplot(gs[0, 1]); ai = fig.add_subplot(gs[0, 2])
    scatter_with_index(ax2, ai, va, 'gamma_evidence', 'recovery_rate', r'Baseline $\gamma_{\rm evidence}$', 'Sequential recovery rate', 'Baseline coefficient predicts later revision', err=[(va.recovery_rate - va.recovery_ci_lo).clip(lower=0), (va.recovery_ci_hi - va.recovery_rate).clip(lower=0)], stat=True, cols=1)
    ax2.text(0.97, 0.05, '', transform=ax2.transAxes); letter(ax2, 'b', dx=0.0)
    ax2.texts[0].set_position((0.03, 0.10)) if False else None
    sv(fig, 'paper_fig04_sequential_revision')


def grouped_bars(ax, groups, series, colors, labels, width=0.26):
    x = np.arange(len(groups)); k = len(series)
    for i, (vals, c, l) in enumerate(zip(series, colors, labels)): ax.bar(x + (i - (k - 1) / 2) * width, vals, width=width * 0.94, color=c, label=l, zorder=3)
    ax.set_xticks(x); ax.set_xticklabels(groups, rotation=22, ha='right', fontsize=8.6)


def fig5():
    d = pd.read_csv(SUP / 'per_by_difficulty.csv'); models = ['qwen2.5-0.5b', 'qwen2.5-1.5b', 'qwen2.5-3b', 'qwen2.5-7b', 'mistral-7b-instruct', 'phi-3.5-mini']
    fig, ax = plt.subplots(figsize=(7.4, 4.0)); ser = [[d[(d.model == m) & (d.difficulty == t)].gamma_evidence.iloc[0] for m in models] for t in ['easy', 'medium', 'hard']]
    grouped_bars(ax, [nm(m) for m in models], ser, DIFF3, ['Easy', 'Medium', 'Hard']); ax.axhline(0, color='#b9b8b0', lw=0.8); ax.set_ylabel(r'$\gamma_{\rm evidence}$'); ax.set_title('Evidence responsiveness across case difficulty'); ax.legend(ncol=3, loc='upper left'); ax.grid(axis='x', visible=False)
    sv(fig, 'paper_fig05_difficulty_stratified')


def fig6():
    d = pd.read_csv(SUP / 'robustness/rare_common_matched_summary.csv'); models = ['qwen2.5-0.5b', 'qwen2.5-1.5b', 'qwen2.5-3b', 'qwen2.5-7b', 'yi-1.5-6b', 'olmo-2-7b', 'mistral-7b-instruct', 'phi-3.5-mini']; d = d.set_index('model').loc[models]
    fig, ax = plt.subplots(figsize=(7.4, 4.0)); grouped_bars(ax, [nm(m) for m in models], [d.rare_accuracy, d.common_accuracy], [ORANGE, BLUE], ['Rare targets', 'Common targets'], width=0.36)
    ax.set_ylabel('Top-1 accuracy'); ax.set_title('Rare and common accuracy after matching case characteristics'); ax.legend(loc='upper left'); ax.grid(axis='x', visible=False)
    sv(fig, 'paper_fig06_rare_common_matched')


def fig7():
    g = pd.read_csv(SUP / 'per_sensitivity_grid_summary.csv'); models = ['qwen2.5-0.5b', 'qwen2.5-1.5b', 'qwen2.5-3b', 'qwen2.5-7b', 'mistral-7b-instruct', 'phi-3.5-mini']; g = g.set_index('model').loc[models]
    fig, ax = plt.subplots(figsize=(7.2, 3.8)); y = np.arange(len(g))
    ax.hlines(y, g.gamma_evidence_min, g.gamma_evidence_max, color=BLUE, lw=3, zorder=3, alpha=0.55); dot(ax, g.gamma_evidence_min, y, NAVY, s=44); dot(ax, g.gamma_evidence_max, y, AQUA, s=44)
    ax.set_yticks(y); ax.set_yticklabels([nm(m) for m in g.index]); ax.grid(axis='y', visible=False); ax.set_xlabel(r'Range of $\gamma_{\rm evidence}$ across 16 evidence-score settings'); ax.set_title('Evidence responsiveness is stable to evidence-score parameterization')
    legend_markers(ax, [('Minimum', NAVY, 'o', False), ('Maximum', AQUA, 'o', False)], loc='lower right'); sv(fig, 'paper_fig07_parameterization_sensitivity')


def fig8():
    d = pd.read_csv(SUP / 'robustness/hierarchical_organ_estimates.csv'); mt = pd.read_csv(SUP / 'robustness/hierarchical_organ_meta.csv'); m = 'mistral-7b-instruct'
    s = d[(d.model == m) & d.gamma_evidence.notna() & (d.se_gamma_evidence > 0)].sort_values('organ_system').reset_index(drop=True); pooled = mt.loc[mt.model == m, 'pooled_gamma_evidence_main_paper'].iloc[0]; mu = mt.loc[mt.model == m, 'mu_random_effects'].iloc[0]
    fig, ax = plt.subplots(figsize=(7.6, 4.3)); y = np.arange(len(s))[::-1]
    ax.errorbar(s.gamma_evidence, y, xerr=1.96 * s.se_gamma_evidence, fmt='o', color=MUTED, ecolor='#c3c2ba', elinewidth=1.4, ms=6.5, mec='white', zorder=3, label='Per-organ estimate'); dot(ax, s.gamma_evidence_shrunken, y, ORANGE, s=55, label='Shrunken estimate')
    ax.axvline(pooled, color=NAVY, ls='--', lw=1.4, label='Pooled fit'); ax.axvline(mu, color=AQUA, ls=':', lw=1.8, label='Random-effects mean')
    ax.set_yticks(y); ax.set_yticklabels(s.organ_system); ax.grid(axis='y', visible=False); ax.set_xlabel(r'$\gamma_{\rm evidence}$'); ax.set_title('Organ-level estimates and hierarchical shrinkage'); ax.legend(loc='lower right')
    sv(fig, 'paper_fig08_organ_shrinkage')


def fig9():
    b = acc[(acc.condition == 'baseline') & (acc.split == 'ALL')].set_index('model'); order = ['qwen2.5-0.5b', 'qwen2.5-1.5b', 'qwen2.5-3b', 'qwen2.5-7b']
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.4, 3.9), gridspec_kw={'wspace': 0.28}); x = np.arange(4)
    a1.plot(x, 100 * b.loc[order, 'top1_accuracy'], '-o', color=BLUE, ms=7, mec='white', label='Top-1 accuracy'); a1.plot(x, 100 * b.loc[order, 'logical_violation_rate'], '-s', color=ORANGE, ms=7, mec='white', label='Logical violation')
    a1.set_xticks(x); a1.set_xticklabels(['0.5B', '1.5B', '3B', '7B']); a1.set_ylabel('%'); a1.set_title('Qwen2.5 scale'); a1.legend(loc='upper left')
    ms = ['mistral-7b-instruct', 'biomistral-7b']; mets = ['top1_accuracy', 'rare_top1_accuracy', 'logical_violation_rate']
    for k, (m, c, l) in enumerate(zip(ms, [BLUE, AQUA], ['Mistral-7B', 'BioMistral-7B'])): a2.bar(np.arange(3) + (k - 0.5) * 0.38, [100 * b.loc[m, q] for q in mets], width=0.36, color=c, label=l, zorder=3)
    a2.set_xticks(range(3)); a2.set_xticklabels(['Top-1', 'Rare Top-1', 'Violation']); a2.set_ylabel('%'); a2.set_title('Medical specialization'); a2.legend(loc='upper right'); a2.grid(axis='x', visible=False)
    sv(fig, 'paper_fig09_scale_and_specialization')


def fig10():
    bins = pd.read_csv(SUP / 'robustness/self_consistency_calibration_bins.csv'); sm = pd.read_csv(SUP / 'robustness/self_consistency_calibration_summary.csv').set_index('model')
    fig, ax = plt.subplots(figsize=(5.6, 4.7)); ax.plot([0, 1], [0, 1], color='#c3c2ba', ls='--', lw=1.3, zorder=1)
    for m, c, mk in [('qwen2.5-0.5b', MUTED, 'o'), ('qwen2.5-3b', YELLOW, 's'), ('qwen2.5-7b', BLUE, '^')]:
        s = bins[bins.model == m].sort_values('mean_agreement'); ax.plot(s.mean_agreement, s.empirical_accuracy, '-', color=c, lw=1.8, marker=mk, ms=7, mec='white', label=f'{nm(m)} (ECE = {sm.loc[m, "ECE"]:.2f})')
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02); ax.set_xlabel('Self-consistency agreement'); ax.set_ylabel('Empirical accuracy'); ax.set_title('Agreement as a confidence signal'); ax.legend(loc='upper left'); sv(fig, 'paper_fig10_selfconsistency_calibration')


def fig11():
    d = acc[acc.split == 'ALL']; models = ['phi-3.5-mini', 'mistral-7b-instruct', 'olmo-2-7b', 'qwen2.5-7b', 'qwen2.5-3b', 'yi-1.5-6b', 'qwen2.5-1.5b', 'qwen2.5-0.5b']
    conds = [('baseline', 'Baseline', BLUE, 'o'), ('debias', 'Evidence-focused', AQUA, 'D'), ('cot', 'Step-by-step', VIOLET, 's'), ('counterfactual_primary_care', 'Primary care', ORANGE, 'v'), ('counterfactual_rare_clinic', 'Rare-disease referral', YELLOW, '^')]
    fig, ax = plt.subplots(figsize=(7.6, 4.6)); y = np.arange(len(models))[::-1]
    for i, m in enumerate(models): ax.axhline(y[i], color=GRID, lw=0.8, zorder=1)
    for c, l, col, mk in conds:
        v = [d[(d.model == m) & (d.condition == c)].top1_accuracy for m in models]; xs = [x.iloc[0] if len(x) else np.nan for x in v]; ax.scatter(xs, y, s=48, color=col, marker=mk, edgecolor='white', linewidth=0.7, zorder=4, label=l)
    ax.set_yticks(y); ax.set_yticklabels([nm(m) for m in models]); ax.set_xlabel('Top-1 diagnostic accuracy'); ax.set_xlim(-0.02, 0.76); ax.grid(axis='y', visible=False); ax.set_title('Prompt framing changes accuracy differently across checkpoints')
    ax.legend(ncol=3, loc='upper center', bbox_to_anchor=(0.5, -0.16), fontsize=8.2); sv(fig, 'paper_fig11_prompt_decoding_sensitivity')


def fig12():
    d = pd.read_csv(MAIN / 'qwen3_thinking_budget_analysis.csv').set_index('condition'); C = ['no_think', 'think_low', 'think_med', 'think_high']; XL = ['No think', 'Think\n512', 'Think\n2048', 'Think\n8192']
    P_ = [('top1_accuracy', 'Primary diagnostic accuracy', 'Accuracy'), ('gamma_evidence', 'Evidence coefficient', 'Evidence responsiveness'), ('delta_E', 'Randomized evidence effect', r'$\Delta_E$'), ('recovery_rate', 'Sequential recovery', 'Recovery rate')]
    fig, axs = plt.subplots(2, 2, figsize=(8.2, 6.2), gridspec_kw={'hspace': 0.5, 'wspace': 0.3})
    for ax, (k, t, yl), L in zip(axs.ravel(), P_, 'abcd'):
        xs = [i for i, c in enumerate(C) if c in d.index and pd.notna(d.loc[c, k])]; ys = [d.loc[C[i], k] for i in xs]; ax.plot(xs, ys, '--', color='#b9b8b0', lw=1.3, zorder=1)
        for i, yv in zip(xs, ys):
            low = k == 'gamma_evidence' and C[i] in ('think_low', 'think_med'); dot(ax, [i], [yv], c=NAVY if i == 0 else (AQUA if i == 2 else BLUE), s=70, hollow=low)
        lo = ax.get_ylim()[0]; ax.scatter([3], [lo + 0.02 * np.ptp(ax.get_ylim())], marker='x', color=MUTED, s=50, zorder=4)
        ax.set_xticks(range(4)); ax.set_xticklabels(XL, fontsize=8.4); ax.set_xlim(-0.4, 3.4); ax.set_ylabel(yl); ax.set_title(t, fontsize=10); letter(ax, L, dx=-0.01, dy=1.13); ax.grid(axis='x', visible=False)
    legend_markers(axs[0, 1], [('Low-coverage estimate', MUTED, 'o', True), ('No completed run', MUTED, 'x', False)], loc='upper right', fontsize=7.6)
    sv(fig, 'paper_fig12_qwen3_reasoning_budget')


def fig13():
    d = base_open.dropna(subset=['gamma_prior', 'gamma_evidence']).sort_values('gamma_evidence').reset_index(drop=True)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.6, 4.8), gridspec_kw={'width_ratios': [2.3, 1.25], 'wspace': 0.04})
    scatter_with_index(a1, a2, d, 'gamma_prior', 'gamma_evidence', r'Disease-prior coefficient, $\gamma_{\rm prior}$', r'Evidence coefficient, $\gamma_{\rm evidence}$', 'Prior and evidence coefficients across open-weight checkpoints', hollow=(d.used_frac < 0.5).to_numpy(), line=False, stat=False,
                       err=[d.gamma_evidence - d.gamma_evidence_lo, d.gamma_evidence_hi - d.gamma_evidence])
    a1.axhline(0, color='#c3c2ba', lw=0.9, zorder=1); a1.axvline(0, color='#c3c2ba', lw=0.9, zorder=1); sv(fig, 'paper_fig13_prior_evidence_landscape')


if __name__ == '__main__':
    only = set(sys.argv[1:])
    for k, f in [('1', fig1), ('2', fig2), ('3', fig3), ('4', fig4), ('5', fig5), ('6', fig6), ('7', fig7), ('8', fig8), ('9', fig9), ('10', fig10), ('11', fig11), ('12', fig12), ('13', fig13)]:
        if not only or k in only:
            try: f()
            except Exception as e:
                import traceback; print('FAILED fig', k, e); traceback.print_exc()
