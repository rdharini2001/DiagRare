#!/usr/bin/env python3
"""Appendix figures for the new robustness suite (paper Sec 7.10, 7.13, 7.14).
Same style conventions as make_theory_figures.py: BLUE/ORANGE two-series
palette for before/after or paired comparisons, direct-labeled neutral
markers instead of an 8-color legend for >3-entity plots.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
FIGS = ROOT / "figures"
FIGS.mkdir(parents=True, exist_ok=True)

BLUE = "#2a78d6"
ORANGE = "#eb6834"
INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#dedcd3"

plt.rcParams.update({
    "font.size": 11, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "text.color": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "figure.facecolor": "white", "axes.facecolor": "white",
})

LABELS = {"qwen2.5-0.5b": "Qwen2.5-0.5B", "qwen2.5-1.5b": "Qwen2.5-1.5B",
          "qwen2.5-3b": "Qwen2.5-3B", "qwen2.5-7b": "Qwen2.5-7B",
          "yi-1.5-6b": "Yi-1.5-6B", "yi-1.5-9b": "Yi-1.5-9B", "olmo-2-7b": "OLMo-2-7B",
          "mistral-7b-instruct": "Mistral-7B", "phi-3.5-mini": "Phi-3.5-mini",
          "falcon-7b": "Falcon-7B", "stablelm-zephyr-3b": "StableLM-Zephyr-3B",
          "zephyr-7b": "Zephyr-7B", "granite-3.1-2b": "Granite-3.1-2B",
          "granite-3.1-8b": "Granite-3.1-8B", "openchat-3.5": "OpenChat-3.5",
          "glm-4-9b": "GLM-4-9B", "nous-hermes-2-7b": "Nous-Hermes-2-7B",
          "deepseek-llm-7b": "DeepSeek-LLM-7B", "vicuna-7b": "Vicuna-7B",
          "tinyllama-1.1b": "TinyLlama-1.1B"}


def fig_a3_rare_common_matched():
    """Rare vs. common accuracy, matched on organ/difficulty/findings/evidence."""
    d = pd.read_csv(ROOT / "results" / "theory" / "robustness" / "rare_common_matched_summary.csv")
    models = ["qwen2.5-0.5b", "qwen2.5-1.5b", "qwen2.5-3b", "qwen2.5-7b",
              "yi-1.5-6b", "olmo-2-7b", "mistral-7b-instruct", "phi-3.5-mini"]
    d = d.set_index("model").loc[models].reset_index()

    fig, ax = plt.subplots(figsize=(8.5, 5))
    x = np.arange(len(models))
    w = 0.35
    ax.bar(x - w / 2, d["rare_accuracy"], width=w, color=ORANGE, label="Rare (matched)")
    ax.bar(x + w / 2, d["common_accuracy"], width=w, color=BLUE, label="Common (matched)")
    sig = d["wilcoxon_p"] < 0.05
    for i, s in enumerate(sig):
        if s:
            ax.annotate("*", (x[i], max(d["rare_accuracy"][i], d["common_accuracy"][i]) + 0.02),
                        ha="center", fontsize=13, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[m] for m in models], fontsize=9, rotation=15, ha="right")
    ax.set_ylabel("Top-1 accuracy")
    ax.set_title("Common-disease accuracy exceeds rare-disease accuracy even after\n"
                  "matching organ system, difficulty, finding-count, and evidence strength\n"
                  "(* = Wilcoxon p<0.05, n=432 matched pairs)", fontsize=10.5)
    ax.legend(fontsize=9, frameon=False)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGS / "fig_a3_rare_common_matched.png", dpi=200)
    plt.close(fig)
    print("Wrote fig_a3_rare_common_matched.png")


def fig_a4_hierarchical_shrinkage():
    """Forest-style plot: per-organ raw gamma_evidence estimates (with SE) vs.
    the random-effects shrunken estimate and the main-text pooled fit, for
    one representative model."""
    model_tag = "mistral-7b-instruct"
    d = pd.read_csv(ROOT / "results" / "theory" / "robustness" / "hierarchical_organ_estimates.csv")
    meta = pd.read_csv(ROOT / "results" / "theory" / "robustness" / "hierarchical_organ_meta.csv")
    all_sub = d[(d.model == model_tag) & d.gamma_evidence.notna()].sort_values("organ_system").reset_index(drop=True)
    n_dropped = int((all_sub["se_gamma_evidence"].isna() | (all_sub["se_gamma_evidence"] <= 0)).sum())
    sub = all_sub[all_sub["se_gamma_evidence"] > 0].reset_index(drop=True)
    pooled = meta.loc[meta.model == model_tag, "pooled_gamma_evidence_main_paper"].iloc[0]
    mu_re = meta.loc[meta.model == model_tag, "mu_random_effects"].iloc[0]

    fig, ax = plt.subplots(figsize=(7.5, 5))
    y = np.arange(len(sub))
    ax.errorbar(sub["gamma_evidence"], y, xerr=1.96 * sub["se_gamma_evidence"],
                fmt="o", color=MUTED, ecolor=MUTED, capsize=3, label="Per-organ MLE (95% Wald CI)")
    ax.scatter(sub["gamma_evidence_shrunken"], y, color=ORANGE, zorder=5, s=55,
               label="Random-effects shrunken estimate")
    ax.axvline(pooled, color=BLUE, linestyle="--", linewidth=1.5,
               label=f"Main-text pooled fit ({pooled:.3f})")
    ax.axvline(mu_re, color=ORANGE, linestyle=":", linewidth=1.5,
               label=f"Random-effects mean ({mu_re:.3f})")
    ax.set_yticks(y)
    ax.set_yticklabels(sub["organ_system"])
    ax.set_xlabel(r"$\gamma_{\mathrm{evidence}}$" +
                  (f"   ({n_dropped} organ(s) omitted: degenerate Hessian)" if n_dropped else ""), fontsize=9.5)
    ax.set_title(f"Organ-level estimates shrink toward the random-effects mean,\n"
                 f"which closely tracks the main-text pooled fit ({model_tag})", fontsize=10.5)
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGS / "fig_a4_hierarchical_shrinkage.png", dpi=200)
    plt.close(fig)
    print("Wrote fig_a4_hierarchical_shrinkage.png")


def fig_a5_selfconsistency_calibration():
    """Reliability diagram: self-consistency agreement rate vs. empirical
    accuracy, one panel per model, plus the gamma_evidence-by-confidence bars."""
    bins = pd.read_csv(ROOT / "results" / "theory" / "robustness" / "self_consistency_calibration_bins.csv")
    summary = pd.read_csv(ROOT / "results" / "theory" / "robustness" / "self_consistency_calibration_summary.csv")
    models = ["qwen2.5-0.5b", "qwen2.5-3b", "qwen2.5-7b"]

    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.plot([0, 1], [0, 1], color=GRID, linewidth=1.5, linestyle="--", zorder=1)
    markers = ["o", "s", "^"]
    for m, mk in zip(models, markers):
        sub = bins[bins.model == m].sort_values("mean_agreement")
        color = BLUE if m == "qwen2.5-7b" else (ORANGE if m == "qwen2.5-3b" else MUTED)
        ax.plot(sub["mean_agreement"], sub["empirical_accuracy"], marker=mk, color=color,
                label=f"{LABELS[m]} (ECE={summary.loc[summary.model==m,'ECE'].iloc[0]:.2f})", markersize=7)
    ax.set_xlabel("Self-consistency agreement rate (confidence proxy)")
    ax.set_ylabel("Empirical accuracy")
    ax.set_title("Self-consistency agreement is a well-calibrated confidence signal\n"
                 "for capable models, badly overconfident for the near-floor model", fontsize=10.5)
    ax.legend(fontsize=9, frameon=False, loc="upper left")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(-0.02, 1.02)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGS / "fig_a5_selfconsistency_calibration.png", dpi=200)
    plt.close(fig)
    print("Wrote fig_a5_selfconsistency_calibration.png")


if __name__ == "__main__":
    fig_a3_rare_common_matched()
    fig_a4_hierarchical_shrinkage()
    fig_a5_selfconsistency_calibration()


def fig2b_causal_validation():
    """decisive result: revealed-preference gamma_evidence predicts
    experimentally-manipulated behavioral sensitivity on an unrelated task.
    Uses every model with a valid fit on both axes (expanded panel, not
    just the original 8) -- numbered markers + side legend, same scheme as
    fig1, since direct text labels collide badly at this n."""
    d = pd.read_csv(ROOT / "results" / "theory" / "causal_grid_analysis.csv")
    d = d.dropna(subset=["beta_evidence_causal", "gamma_evidence_revealed"])
    d = d.sort_values("gamma_evidence_revealed").reset_index(drop=True)
    d["idx"] = np.arange(1, len(d) + 1)

    from scipy.stats import pearsonr
    r, p = pearsonr(d.gamma_evidence_revealed, d.beta_evidence_causal)

    fig, (ax, ax_legend) = plt.subplots(1, 2, figsize=(11, 6),
                                          gridspec_kw={"width_ratios": [2.1, 1]})
    ax.scatter(d.gamma_evidence_revealed, d.beta_evidence_causal, s=110, color=BLUE,
               edgecolor=INK, linewidth=0.8, zorder=3)
    for _, row in d.iterrows():
        ax.annotate(str(row.idx), (row["gamma_evidence_revealed"], row["beta_evidence_causal"]),
                    ha="center", va="center", fontsize=7.5, color="white", fontweight="bold", zorder=4)
    # linear fit line
    z = np.polyfit(d.gamma_evidence_revealed, d.beta_evidence_causal, 1)
    xs = np.linspace(d.gamma_evidence_revealed.min(), d.gamma_evidence_revealed.max(), 50)
    ax.plot(xs, np.polyval(z, xs), color=ORANGE, linestyle="--", linewidth=1.5, zorder=2)
    ax.set_xlabel(r"$\gamma_{\mathrm{evidence}}$ (revealed preference, main ranking task)")
    ax.set_ylabel(r"$\beta_{\mathrm{evidence}}$ (experimental manipulation, causal grid)")
    ax.set_title(f"A coefficient estimated from free-text ranking predicts\n"
                 f"behavioral response to experimentally manipulated evidence\n"
                 f"(r={r:.3f}, p={p:.4f}, n={len(d)} models)", fontsize=10.5)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    ax_legend.axis("off")
    legend_lines = [f"{row.idx:>2}.  {LABELS.get(row.model, row.model)}" for _, row in d.iterrows()]
    half = (len(legend_lines) + 1) // 2
    ax_legend.text(0.0, 0.98, "\n".join(legend_lines[:half]), transform=ax_legend.transAxes,
                   fontsize=9, va="top", ha="left", color=INK, family="monospace")
    ax_legend.text(0.62, 0.98, "\n".join(legend_lines[half:]), transform=ax_legend.transAxes,
                   fontsize=9, va="top", ha="left", color=INK, family="monospace")
    ax_legend.set_title("Model index\n(sorted by $\\gamma_{\\mathrm{evidence}}$)", fontsize=10,
                        color=MUTED, loc="left")
    fig.tight_layout()
    fig.savefig(FIGS / "fig2b_causal_validation.png", dpi=200)
    plt.close(fig)
    print("Wrote fig2b_causal_validation.png")


if __name__ == "__main__":
    fig2b_causal_validation()
