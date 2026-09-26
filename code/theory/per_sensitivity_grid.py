#!/usr/bin/env python3
"""Does the PER HEADLINE NUMBER (not just the oracle's covariate ranking,
already checked in sensitivity_analysis.py) survive varying the oracle's 4
fixed finding-emission probabilities? Re-fits Plackett-Luce (point estimates
only -- no bootstrap, to keep this fast) for the released open-weight
models's baseline condition, across the same parameter grid used in
sensitivity_analysis.py, and reports the range of PER and gamma_evidence
each model takes across that grid.
"""
from __future__ import annotations

import itertools
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
from bayes_oracle import OracleParams, compute_oracle_table
from plackett_luce import build_choices_from_predictions, fit_plackett_luce

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    vignettes = pd.read_csv(ROOT / "data" / "expanded" / "vignettes_combined.csv")
    ontology = pd.read_csv(ROOT / "data" / "expanded" / "ontology_diseases_expanded.csv")

    grid_params = {
        "p_req": [0.80, 0.90, 0.97], "p_sup": [0.35, 0.50, 0.65],
        "p_bg": [0.01, 0.03, 0.06], "p_exc": [0.01, 0.02, 0.08],
    }
    combos = list(itertools.product(*grid_params.values()))
    rng = np.random.default_rng(0)
    sampled = [combos[0], combos[-1]] + [combos[i] for i in rng.choice(len(combos), size=14, replace=False)]

    models = [f.stem.split("__baseline")[0] for f in (ROOT / "results" / "predictions").glob("*__baseline.csv")]
    pred_files = {m: ROOT / "results" / "predictions" / f"{m}__baseline.csv" for m in models}


    rows = []
    t0 = time.time()
    for gi, combo in enumerate(sampled):
        params = OracleParams(*combo)
        oracle82 = compute_oracle_table(vignettes, ontology, params)
        oracle58 = compute_oracle_table(orig_vignettes, orig_ontology, params)

        for model_tag, f in pred_files.items():
            choices, n_total, n_used = build_choices_from_predictions(pd.read_csv(f), oracle82)
            fit = fit_plackett_luce(choices, n_starts=1)
            rows.append({"grid_idx": gi, **dict(zip(grid_params.keys(), combo)), "source": "open_weight",
                         "model": model_tag, "gamma_prior": fit["gamma_prior"],
                         "gamma_evidence": fit["gamma_evidence"], "PER": fit["PER"]})



    out = pd.DataFrame(rows)
    out_path = ROOT / "results" / "theory" / "per_sensitivity_grid.csv"
    out.to_csv(out_path, index=False)

    print("\nRange of gamma_evidence and PER across the parameter grid, per model:")
    summary = out.groupby(["source", "model"]).agg(
        gamma_evidence_min=("gamma_evidence", "min"), gamma_evidence_max=("gamma_evidence", "max"),
        PER_min=("PER", "min"), PER_max=("PER", "max"),
    ).reset_index()
    print(summary.to_string(index=False))
    summary.to_csv(ROOT / "results" / "theory" / "per_sensitivity_grid_summary.csv", index=False)
    print(f"\nWrote {out_path} and per_sensitivity_grid_summary.csv")


if __name__ == "__main__":
    main()
