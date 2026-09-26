#!/usr/bin/env python3
"""Figure for the causal response surface (theory/causal_response_surface.py):
a 3x3 evidence x prior heatmap of Pr(target) for a few representative
models, plus the headline population-level scatter (baseline
gamma_evidence vs. measured causal Delta_E). Same style conventions as
analysis/make_theory_figures.py (see that file's module docstring for the
palette rationale)."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

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
    "axes.grid": False, "figure.facecolor": "white", "axes.facecolor": "white",
})

EVS = ["weak", "medium", "strong"]
PRIORS = ["target_rare", "equal", "target_common"]
PRIOR_LABELS = ["rare", "equal", "common"]


def draw_surface(ax, surf_model: pd.DataFrame, title: str) -> None:
    grid = np.full((3, 3), np.nan)
    for _, row in surf_model.iterrows():
        i = EVS.index(row["evidence_level"])
        j = PRIORS.index(row["prior_level"])
        grid[i, j] = row["Pr_target"]
    im = ax.imshow(grid, cmap="RdYlBu_r", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(3)); ax.set_xticklabels(PRIOR_LABELS, fontsize=10)
    ax.set_yticks(range(3)); ax.set_yticklabels(EVS, fontsize=10)
    ax.set_xlabel("prior level (target vs. confounder)", fontsize=9)
    ax.set_ylabel("evidence level")
    ax.set_title(title, fontsize=11)
    for i in range(3):
        for j in range(3):
            v = grid[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color="white" if abs(v - 0.5) > 0.3 else INK, fontsize=10)
    return im


def main() -> None:
    out_dir = ROOT / "results" / "theory"
    ates = pd.read_csv(out_dir / "causal_response_surface.csv")
    cells = pd.read_csv(out_dir / "causal_response_surface_cells.csv")
    valid = pd.read_csv(out_dir / "causal_response_validation.csv")

    ates_ranked = ates.dropna(subset=["delta_E"]).sort_values("delta_E", ascending=False)
    top_model = ates_ranked.iloc[0]["model"]
    mid_model = ates_ranked.iloc[len(ates_ranked) // 2]["model"]
    bottom_model = ates_ranked.iloc[-1]["model"]
    reps = [(top_model, "most evidence-responsive"),
            (mid_model, "median"),
            (bottom_model, "least evidence-responsive")]

    fig = plt.figure(figsize=(15.5, 5.0))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 1.3], wspace=0.6)

    im = None
    for k, (model, label) in enumerate(reps):
        ax = fig.add_subplot(gs[0, k])
        surf_model = cells[cells.model == model]
        im = draw_surface(ax, surf_model, f"{model}\n({label})")

    cax = fig.colorbar(im, ax=fig.get_axes()[:3], shrink=0.85, pad=0.02)
    cax.set_label("Pr(choose target)", fontsize=10)

    ax4 = fig.add_subplot(gs[0, 3])
    ax4.scatter(valid["gamma_evidence"], valid["delta_E"], color=BLUE, s=55, zorder=3, edgecolor="white", linewidth=0.6)
    r, p = pearsonr(valid["gamma_evidence"], valid["delta_E"])
    z = np.polyfit(valid["gamma_evidence"], valid["delta_E"], 1)
    xs = np.linspace(valid["gamma_evidence"].min(), valid["gamma_evidence"].max(), 50)
    ax4.plot(xs, np.polyval(z, xs), color=ORANGE, linewidth=1.8, zorder=2)
    ax4.set_xlabel("gamma_evidence (baseline rank fit)", fontsize=10)
    ax4.set_ylabel("Delta_E (measured causal effect)", fontsize=10)
    ax4.set_title(f"r={r:.2f}, p={p:.4g}, n={len(valid)}", fontsize=11)
    ax4.grid(True, color=GRID, linewidth=0.6)
    ax4.set_axisbelow(True)

    fig.suptitle("Causal response surface: Pr(target) under randomized evidence x prior intervention", y=1.04, fontsize=12)
    fig.savefig(FIGS / "fig4_causal_response_surface.png", dpi=200, bbox_inches="tight")
    print(f"Wrote {FIGS / 'fig4_causal_response_surface.png'}")


if __name__ == "__main__":
    main()
