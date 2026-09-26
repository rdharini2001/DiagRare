#!/usr/bin/env python3
"""Evaluate sensitivity to the parameters used in the evidence score."""
from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from bayes_oracle import OracleParams, compute_oracle_table, oracle_accuracy

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    vignettes = pd.read_csv(ROOT / "data" / "expanded" / "vignettes_combined.csv")
    ontology = pd.read_csv(ROOT / "data" / "expanded" / "ontology_diseases_expanded.csv")

    grid = {
        "p_req": [0.80, 0.90, 0.97],
        "p_sup": [0.35, 0.50, 0.65],
        "p_bg": [0.01, 0.03, 0.06],
        "p_exc": [0.01, 0.02, 0.08],
    }
    # Full factorial (3^4=81) is expensive; sample a Latin-hypercube-ish subset
    # plus the two extremes to cover the space without 81 full passes.
    combos = list(itertools.product(*grid.values()))
    rng = np.random.default_rng(0)
    sampled = [combos[0], combos[-1]] + list(rng.choice(len(combos), size=14, replace=False))
    sampled = [combos[i] if isinstance(i, (int, np.integer)) else i for i in sampled]

    reference_table = None
    rows = []
    evidence_by_combo = {}
    for combo in sampled:
        params = OracleParams(*combo)
        table = compute_oracle_table(vignettes, ontology, params)
        acc1 = oracle_accuracy(table, vignettes, k=1)
        acc3 = oracle_accuracy(table, vignettes, k=3)
        rows.append({"p_req": combo[0], "p_sup": combo[1], "p_bg": combo[2], "p_exc": combo[3],
                     "bayes_top1": acc1, "bayes_top3": acc3})
        key = tuple(combo)
        evidence_by_combo[key] = table.set_index(["vignette_id", "disease"])["evidence_loglik_ratio"]
        if reference_table is None:
            reference_table = key

    summary = pd.DataFrame(rows)
    out_dir = ROOT / "results" / "theory"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_dir / "oracle_sensitivity_grid.csv", index=False)

    print("Oracle sensitivity grid (16 sampled param combos out of 81 full-factorial):")
    print(summary.to_string(index=False))
    print(f"\nBayes top-1 ceiling range: {100*summary.bayes_top1.min():.1f}% - {100*summary.bayes_top1.max():.1f}%")

    # Global pooled Spearman is dominated by the huge mass of candidates that
    # share ZERO finding vocabulary with a given vignette (evidence_loglik_ratio
    # exactly 0 under every parameter choice -- floating-point noise among ties
    # swamps any real signal). What actually matters for the analysis's claims is
    # the WITHIN-VIGNETTE ranking of candidates (which one looks best-supported
    # for this patient), averaged across vignettes.
    ref_evidence = evidence_by_combo[reference_table]
    corrs = []
    top1_agreement = []
    for key, ev in evidence_by_combo.items():
        aligned = pd.concat([ref_evidence, ev], axis=1, keys=["ref", "this"]).dropna()
        per_vignette_rho = []
        per_vignette_top1_match = []
        for vid, grp in aligned.groupby(level=0):
            if grp["ref"].nunique() < 2:
                continue
            rho, _ = spearmanr(grp["ref"], grp["this"])
            if not np.isnan(rho):
                per_vignette_rho.append(rho)
            per_vignette_top1_match.append(grp["ref"].idxmax() == grp["this"].idxmax())
        corrs.append(np.mean(per_vignette_rho))
        top1_agreement.append(np.mean(per_vignette_top1_match))

    print(f"\nWithin-vignette Spearman rank-correlation of evidence_loglik_ratio vs. reference "
          f"params, averaged over {vignettes['vignette_id'].nunique()} vignettes, across "
          f"{len(corrs)} grid points: min={min(corrs):.3f} mean={np.mean(corrs):.3f}")
    print(f"Within-vignette top-evidence-candidate agreement vs. reference: "
          f"min={min(top1_agreement):.3f} mean={np.mean(top1_agreement):.3f}")
    print("(near 1.0 means which candidate looks best-supported FOR A GIVEN PATIENT is essentially "
          "invariant to the exact emission-probability choice -- the PER estimates inherit this robustness.)")

    pd.DataFrame({"within_vignette_spearman": corrs, "top1_agreement": top1_agreement}).to_csv(
        out_dir / "oracle_sensitivity_spearman.csv", index=False)


if __name__ == "__main__":
    main()
