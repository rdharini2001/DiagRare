#!/usr/bin/env python3
"""Sensitivity check on the analysis single most important number: Sec 6.2's
r=0.900 (p=0.002) correlation between gamma_evidence (revealed preference)
and beta_evidence_causal (experimental manipulation), computed on only 8
models. A correlation at n=8 can be dominated by one or two points --
leave-one-model-out jackknife refits the correlation 8 times, each time
excluding one model, to check whether the result depends on any single
ask for at small n, done proactively.

Will be re-run once the expanded (~23-model) panel's causal-grid data
lands, which is the real fix for the small-n concern -- this jackknife is
the honest interim answer for the panel we have now.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[3]


def main() -> None:
    d = pd.read_csv(ROOT / "results" / "theory" / "causal_grid_analysis.csv")
    d = d.dropna(subset=["beta_evidence_causal", "gamma_evidence_revealed",
                          "beta_prior_causal", "gamma_prior_revealed"])
    models = d["model"].tolist()
    n = len(models)

    full_r, full_p = pearsonr(d.gamma_evidence_revealed, d.beta_evidence_causal)
    full_rho, _ = spearmanr(d.gamma_evidence_revealed, d.beta_evidence_causal)
    print(f"Full sample (n={n}): r={full_r:.3f} (p={full_p:.4f}), rho={full_rho:.3f}\n")

    rows = []
    for held_out in models:
        sub = d[d.model != held_out]
        r, p = pearsonr(sub.gamma_evidence_revealed, sub.beta_evidence_causal)
        rho, _ = spearmanr(sub.gamma_evidence_revealed, sub.beta_evidence_causal)
        rows.append({"held_out_model": held_out, "n": len(sub), "r": r, "p": p, "rho": rho})
        print(f"  excluding {held_out:22s}: r={r:.3f} (p={p:.4f}), rho={rho:.3f}  "
              f"[delta from full: {r - full_r:+.3f}]")

    out = pd.DataFrame(rows)
    out_path = ROOT / "results" / "theory" / "robustness" / "causal_validation_jackknife.csv"
    out.to_csv(out_path, index=False)

    r_range = (out.r.min(), out.r.max())
    max_p = out.p.max()
    print(f"\nJackknife r range: [{r_range[0]:.3f}, {r_range[1]:.3f}] (full-sample r={full_r:.3f})")
    print(f"Worst-case (highest) jackknife p-value: {max_p:.4f}")
    print(f"Result is {'ROBUST' if max_p < 0.05 else 'SENSITIVE'} to single-model exclusion "
          f"(all {n} leave-one-out refits remain significant at p<0.05: {(out.p < 0.05).all()})")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
