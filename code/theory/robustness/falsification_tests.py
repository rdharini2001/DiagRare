#!/usr/bin/env python3
"""Falsification tests: does the estimator actually read the structure it
claims to, or would it report similar coefficients on scrambled data?

1. Prior-shuffle: randomly permute prevalence VALUES across diseases (so the
   numeric prevalence attached to each candidate no longer matches its real
   epidemiology) while leaving all evidence/finding structure untouched.
   gamma_prior should lose its meaning -- specifically, its magnitude should
   collapse toward 0 relative to the un-shuffled fit, since log_prevalence(d)
   is now uncorrelated with which candidate the model actually favors.
2. Evidence-shuffle: independently permute which candidate each evidence
   score belongs to WITHIN each vignette (so "this candidate is well
   supported by the stated findings" becomes essentially random) while
   leaving prevalence untouched. gamma_evidence should collapse toward 0.

These are cheap (reuse existing predictions + oracle table, no new GPU
inference) and diagnostic: if shuffling a variable barely changes its own
coefficient, the estimator isn't actually sensitive to that variable, which
would undermine the whole framework.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "analysis"))
from plackett_luce import Choice, build_choices_from_predictions, fit_plackett_luce

ROOT = Path(__file__).resolve().parents[3]


def shuffle_prior(oracle_table: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Permute log_prevalence values across candidate diseases GLOBALLY (same
    permutation applied consistently across all vignettes, since prevalence is
    a per-disease constant, not per-vignette) -- breaks the true
    disease-to-prevalence mapping while preserving the marginal distribution
    of prevalence values."""
    rng = np.random.default_rng(seed)
    diseases = oracle_table["disease"].unique()
    shuffled = diseases.copy()
    rng.shuffle(shuffled)
    mapping = dict(zip(diseases, shuffled))
    # look up each disease's TRUE log_prevalence once, then reassign via the shuffled mapping
    true_prev = oracle_table.drop_duplicates("disease").set_index("disease")["log_prevalence"]
    out = oracle_table.copy()
    out["log_prevalence"] = out["disease"].map(mapping).map(true_prev)
    return out


def shuffle_evidence(oracle_table: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Permute evidence_loglik_ratio values WITHIN each vignette across its
    own candidate set (breaks which candidate the evidence actually supports,
    while preserving the per-vignette distribution of evidence scores)."""
    rng = np.random.default_rng(seed)
    out = oracle_table.copy()
    out["evidence_loglik_ratio"] = out.groupby("vignette_id")["evidence_loglik_ratio"].transform(
        lambda s: rng.permutation(s.values))
    return out


def main() -> None:
    oracle = pd.read_csv(ROOT / "results" / "theory" / "bayes_oracle_table_expanded.csv")
    models = ["qwen2.5-7b", "phi-3.5-mini", "mistral-7b-instruct", "qwen2.5-3b"]

    rows = []
    for model_tag in models:
        preds = pd.read_csv(ROOT / "results" / "predictions" / f"{model_tag}__baseline.csv")

        choices_real, _, n_used = build_choices_from_predictions(preds, oracle)
        fit_real = fit_plackett_luce(choices_real)

        prior_shuffled = shuffle_prior(oracle, seed=1)
        choices_ps, _, _ = build_choices_from_predictions(preds, prior_shuffled)
        fit_ps = fit_plackett_luce(choices_ps)

        evidence_shuffled = shuffle_evidence(oracle, seed=1)
        choices_es, _, _ = build_choices_from_predictions(preds, evidence_shuffled)
        fit_es = fit_plackett_luce(choices_es)

        rows.append({
            "model": model_tag, "n_used": n_used,
            "gamma_prior_real": fit_real["gamma_prior"], "gamma_evidence_real": fit_real["gamma_evidence"],
            "gamma_prior_after_prior_shuffle": fit_ps["gamma_prior"],
            "gamma_evidence_after_prior_shuffle": fit_ps["gamma_evidence"],
            "gamma_prior_after_evidence_shuffle": fit_es["gamma_prior"],
            "gamma_evidence_after_evidence_shuffle": fit_es["gamma_evidence"],
        })
        print(f"{model_tag}: real gp={fit_real['gamma_prior']:.3f} ge={fit_real['gamma_evidence']:.3f} | "
              f"prior-shuffled gp={fit_ps['gamma_prior']:.3f} ge={fit_ps['gamma_evidence']:.3f} | "
              f"evidence-shuffled gp={fit_es['gamma_prior']:.3f} ge={fit_es['gamma_evidence']:.3f}")

    out = pd.DataFrame(rows)
    out_path = ROOT / "results" / "theory" / "robustness" / "falsification_tests.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    print(f"\nExpected pattern: |gamma_prior| should shrink toward 0 after prior-shuffle "
          f"(gamma_evidence relatively unaffected); |gamma_evidence| should shrink toward 0 "
          f"after evidence-shuffle (gamma_prior relatively unaffected).")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
