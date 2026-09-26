#!/usr/bin/env python3
"""Figure for the sequential anchoring/recovery experiment
(theory/analyze_sequential_anchoring.py): the cross-model scatter (from the baseline ranking
-fitted gamma_evidence vs. measured recovery_rate) plus a bar chart of
per-model recovery rates with bootstrap 95% CIs, sorted low to high, to
show the spread from Falcon-7B's total failure to recover (0%) up to
several models near-ceiling. Same style conventions as
analysis/make_theory_figures.py."""
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
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "figure.facecolor": "white", "axes.facecolor": "white",
})


def main() -> None:
    out_dir = ROOT / "results" / "theory"
    ana = pd.read_csv(out_dir / "sequential_anchoring_analysis.csv").dropna(subset=["recovery_rate"])
    ana = ana[ana["n_anchored_at_step1"] >= 5].sort_values("recovery_rate")
    valid = pd.read_csv(out_dir / "sequential_anchoring_validation.csv")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2))

    y = np.arange(len(ana))
    err_lo = (ana["recovery_rate"] - ana["recovery_ci_lo"]).clip(lower=0)
    err_hi = (ana["recovery_ci_hi"] - ana["recovery_rate"]).clip(lower=0)
    ax1.barh(y, ana["recovery_rate"], xerr=[err_lo, err_hi], color=BLUE, height=0.65,
              error_kw={"ecolor": MUTED, "elinewidth": 1.2, "capsize": 3})
    ax1.set_yticks(y)
    ax1.set_yticklabels(ana["model"], fontsize=9)
    ax1.set_xlabel("recovery rate (switches to target once decisive\nfollow-up evidence arrives, among initially-anchored cases)")
    ax1.set_xlim(0, 1.05)
    ax1.set_title("Recovery from an initial misdiagnosis\n(95% bootstrap CI over disease pairs)", fontsize=11)

    ax2.scatter(valid["gamma_evidence"], valid["recovery_rate"], color=BLUE, s=60, zorder=3,
                edgecolor="white", linewidth=0.6)
    r, p = pearsonr(valid["gamma_evidence"], valid["recovery_rate"])
    z = np.polyfit(valid["gamma_evidence"], valid["recovery_rate"], 1)
    xs = np.linspace(valid["gamma_evidence"].min(), valid["gamma_evidence"].max(), 50)
    ax2.plot(xs, np.polyval(z, xs), color=ORANGE, linewidth=1.8, zorder=2)
    ax2.set_xlabel("gamma_evidence (baseline rank fit)")
    ax2.set_ylabel("recovery rate (sequential, interactive setting)")
    ax2.set_title(f"r={r:.3f}, p={p:.2g}, n={len(valid)}", fontsize=11)

    fig.suptitle("Does ordinary evidence-following behavior predict recovery from an anchored misdiagnosis?", y=1.02, fontsize=12)
    fig.tight_layout()
    fig.savefig(FIGS / "fig5_sequential_anchoring_recovery.png", dpi=200, bbox_inches="tight")
    print(f"Wrote {FIGS / 'fig5_sequential_anchoring_recovery.png'}")


if __name__ == "__main__":
    main()
