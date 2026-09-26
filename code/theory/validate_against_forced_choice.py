#!/usr/bin/env python3
"""Compares the revealed-preference Plackett-Luce PER estimates (fit on
free-text top-3 rankings) against DIRECT model-internal probabilities from
teacher-forced log-likelihood scoring (run_forced_choice_scoring.py). If the
two methods agree in sign and rough magnitude, that's strong evidence the
revealed-preference approach is measuring something real, not a parsing
artifact -- the two use completely different mechanisms (rank of a generated
answer vs. raw next-token probability of a forced continuation) so agreement
is not a foregone conclusion.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.special import logsumexp

ROOT = Path(__file__).resolve().parents[2]


def fit_direct(model_tag: str) -> dict:
    fc_path = ROOT / "results" / "theory" / "forced_choice" / f"{model_tag}__baseline.csv"
    fc = pd.read_csv(fc_path)
    oracle = pd.read_csv(ROOT / "results" / "theory" / "bayes_oracle_table_expanded.csv")

    merged = fc.merge(oracle, left_on=["vignette_id", "candidate"], right_on=["vignette_id", "disease"])

    # within-vignette normalization: log q_model(d) = log_prob(d) - logsumexp(log_prob | same vignette)
    merged["log_q_model"] = merged.groupby("vignette_id")["log_prob"].transform(
        lambda x: x - logsumexp(x.to_numpy())
    )

    X = sm.add_constant(merged[["log_prevalence", "evidence_loglik_ratio"]])
    y = merged["log_q_model"]
    model = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": merged["vignette_id"]})

    return {
        "model": model_tag,
        "n_rows": len(merged),
        "n_vignettes": merged["vignette_id"].nunique(),
        "gamma_prior_direct": model.params["log_prevalence"],
        "gamma_prior_direct_se": model.bse["log_prevalence"],
        "gamma_prior_direct_p": model.pvalues["log_prevalence"],
        "gamma_evidence_direct": model.params["evidence_loglik_ratio"],
        "gamma_evidence_direct_se": model.bse["evidence_loglik_ratio"],
        "gamma_evidence_direct_p": model.pvalues["evidence_loglik_ratio"],
        "PER_direct": model.params["log_prevalence"] / model.params["evidence_loglik_ratio"],
        "r_squared": model.rsquared,
    }


def main() -> None:
    per_master = pd.read_csv(ROOT / "results" / "theory" / "per_master_table.csv")
    rows = []
    for model_tag in ["qwen2.5-1.5b", "mistral-7b-instruct", "phi-3.5-mini"]:
        direct = fit_direct(model_tag)
        rp = per_master[(per_master.source == "open_weight") & (per_master.model == model_tag)
                         & (per_master.condition == "baseline")]
        if len(rp):
            rp = rp.iloc[0]
            direct.update({
                "gamma_prior_revealed": rp["gamma_prior"], "gamma_evidence_revealed": rp["gamma_evidence"],
                "PER_revealed": rp["PER"], "PER_revealed_lo": rp["PER_lo"], "PER_revealed_hi": rp["PER_hi"],
            })
        rows.append(direct)
        print(f"\n=== {model_tag} ===")
        print(f"  DIRECT (forced-choice logprob):    gamma_prior={direct['gamma_prior_direct']:+.4f} "
              f"(p={direct['gamma_prior_direct_p']:.3g})  gamma_evidence={direct['gamma_evidence_direct']:+.4f} "
              f"(p={direct['gamma_evidence_direct_p']:.3g})  PER={direct['PER_direct']:.3f}  "
              f"R2={direct['r_squared']:.3f}  n={direct['n_rows']} rows / {direct['n_vignettes']} vignettes")
        if "gamma_prior_revealed" in direct:
            print(f"  REVEALED (Plackett-Luce top-3):    gamma_prior={direct['gamma_prior_revealed']:+.4f}  "
                  f"gamma_evidence={direct['gamma_evidence_revealed']:+.4f}  PER={direct['PER_revealed']:.3f} "
                  f"[{direct['PER_revealed_lo']:.3f},{direct['PER_revealed_hi']:.3f}]")
            same_sign_prior = np.sign(direct['gamma_prior_direct']) == np.sign(direct['gamma_prior_revealed'])
            same_sign_evidence = np.sign(direct['gamma_evidence_direct']) == np.sign(direct['gamma_evidence_revealed'])
            print(f"  Agreement: prior sign match={same_sign_prior}  evidence sign match={same_sign_evidence}")

    out = pd.DataFrame(rows)
    out_path = ROOT / "results" / "theory" / "forced_choice_validation.csv"
    out.to_csv(out_path, index=False)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
