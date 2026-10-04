#!/usr/bin/env python3
"""Does PER depend on using the Bayes-oracle's continuous evidence_loglik_ratio
specifically, or would a much simpler, assumption-free "evidence" covariate
give the same qualitative story? Rebuilds the choice covariate table using
the integer overlap-counting support_score from the symbolic verifier
(analysis/symbolic_verifier.py -- no finding-emission probabilities, no
Bayesian assumptions at all, just "how many required/supporting findings does
this candidate share with what's stated") instead of the oracle's
evidence_loglik_ratio, and re-fits Plackett-Luce for every open-weight
baseline model. Agreement in sign/ranking with the oracle-based PER numbers
means the finding is not an artifact of the specific Naive-Bayes assumptions.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
from symbolic_verifier import build_constraints, support_score, split_tokens
from plackett_luce import Choice, build_choices_from_predictions, fit_plackett_luce

ROOT = Path(__file__).resolve().parents[2]


def build_support_score_table(vignettes: pd.DataFrame, ontology: pd.DataFrame) -> pd.DataFrame:
    constraints = build_constraints(ontology)
    diseases = ontology["disease"].tolist()
    log_prev = {d: np.log(max(p, 1e-8)) for d, p in zip(ontology["disease"], ontology["prevalence"])}

    rows = []
    for _, v in vignettes.iterrows():
        pos = split_tokens(v["positive_findings"])
        neg = split_tokens(v["negative_findings"])
        for d in diseases:
            rows.append({
                "vignette_id": v["vignette_id"], "disease": d,
                "is_true_disease": d == v["target_disease"],
                "log_prevalence": log_prev[d],
                "evidence_loglik_ratio": float(support_score(d, pos, neg, constraints)),  # simple integer overlap count
            })
    return pd.DataFrame(rows)


def main() -> None:
    vignettes = pd.read_csv(ROOT / "data" / "expanded" / "vignettes_combined.csv")
    ontology = pd.read_csv(ROOT / "data" / "expanded" / "ontology_diseases_expanded.csv")
    alt_table = build_support_score_table(vignettes, ontology)

    oracle_table = pd.read_csv(ROOT / "results" / "theory" / "bayes_oracle_table_expanded.csv")
    per_master = pd.read_csv(ROOT / "results" / "theory" / "per_master_table.csv")

    rows = []
    for f in sorted((ROOT / "results" / "predictions").glob("*__baseline.csv")):
        model_tag = f.stem.split("__baseline")[0]
        preds = pd.read_csv(f)

        choices_alt, n_total, n_used = build_choices_from_predictions(preds, alt_table)
        fit_alt = fit_plackett_luce(choices_alt)

        oracle_row = per_master[(per_master.source == "open_weight") & (per_master.model == model_tag)
                                 & (per_master.condition == "baseline")]
        row = {"model": model_tag, "n_used": n_used,
               "gamma_prior_supportscore": fit_alt["gamma_prior"], "gamma_evidence_supportscore": fit_alt["gamma_evidence"],
               "PER_supportscore": fit_alt["PER"]}
        if len(oracle_row):
            r = oracle_row.iloc[0]
            row.update({"gamma_prior_oracle": r["gamma_prior"], "gamma_evidence_oracle": r["gamma_evidence"],
                        "PER_oracle": r["PER"]})
        rows.append(row)

    out = pd.DataFrame(rows)
    out_path = ROOT / "results" / "theory" / "per_covariate_robustness.csv"
    out.to_csv(out_path, index=False)

    pd.set_option("display.width", 200)
    print(out.to_string(index=False))
    if "gamma_evidence_oracle" in out.columns:
        same_sign = np.sign(out["gamma_evidence_supportscore"]) == np.sign(out["gamma_evidence_oracle"])
        rank_corr = out[["gamma_evidence_supportscore", "gamma_evidence_oracle"]].corr(method="spearman").iloc[0, 1]
        print(f"\ngamma_evidence sign agreement (support_score vs oracle): {same_sign.mean():.2f} "
              f"({same_sign.sum()}/{len(same_sign)})")
        print(f"gamma_evidence Spearman rank correlation across models: {rank_corr:.3f}")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
