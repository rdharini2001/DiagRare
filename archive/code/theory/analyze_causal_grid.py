#!/usr/bin/env python3
"""Analyze the randomized evidence and stated-prior experiment.

For each model, logistic regression relates binary target choice to the stated log
prior ratio and the assigned evidence level. The resulting intervention coefficients
are compared with the coefficients estimated from the separate ranked-diagnosis task.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[2]

EVIDENCE_NUMERIC = {"weak": 0.0, "medium": 1.0, "strong": 2.0}


def fit_causal_sensitivity(preds: pd.DataFrame) -> dict:
    d = preds[preds["choice"].isin(["target", "confounder"])].copy()
    d["y"] = (d["choice"] == "target").astype(int)
    d["log_prior_ratio"] = np.log(d["target_prevalence_per_100k"] / d["confounder_prevalence_per_100k"])
    d["evidence_numeric"] = d["evidence_level"].map(EVIDENCE_NUMERIC)

    X = sm.add_constant(d[["log_prior_ratio", "evidence_numeric"]])
    y = d["y"]
    n_used = len(d)
    n_total = len(preds)

    if y.nunique() < 2 or n_used < 10:
        return {"n_used": n_used, "n_total": n_total, "beta_prior_causal": np.nan,
                "beta_evidence_causal": np.nan, "converged": False}

    try:
        model = sm.Logit(y, X).fit(disp=0)
        return {
            "n_used": n_used, "n_total": n_total,
            "beta_prior_causal": model.params["log_prior_ratio"],
            "beta_prior_causal_se": model.bse["log_prior_ratio"],
            "beta_prior_causal_p": model.pvalues["log_prior_ratio"],
            "beta_evidence_causal": model.params["evidence_numeric"],
            "beta_evidence_causal_se": model.bse["evidence_numeric"],
            "beta_evidence_causal_p": model.pvalues["evidence_numeric"],
            "pseudo_r2": model.prsquared, "converged": True,
        }
    except Exception as e:
        return {"n_used": n_used, "n_total": n_total, "beta_prior_causal": np.nan,
                "beta_evidence_causal": np.nan, "converged": False, "error": str(e)}


def main() -> None:
    per_master = pd.read_csv(ROOT / "results" / "theory" / "per_master_table.csv")
    grid_dir = ROOT / "results" / "theory" / "causal_grid"

    rows = []
    for f in sorted(grid_dir.glob("*.csv")):
        model_tag = f.stem
        preds = pd.read_csv(f)
        fit = fit_causal_sensitivity(preds)
        gamma_row = per_master[(per_master.source == "open_weight") & (per_master.model == model_tag)
                                & (per_master.condition == "baseline")]
        row = {"model": model_tag, **fit}
        if len(gamma_row):
            g = gamma_row.iloc[0]
            row.update({"gamma_prior_revealed": g["gamma_prior"], "gamma_evidence_revealed": g["gamma_evidence"]})
        rows.append(row)
        frac_target = (preds["choice"] == "target").mean()
        frac_unparsed = (preds["choice"] == "unparsed").mean()
        print(f"{model_tag}: n_used={fit['n_used']}/{fit['n_total']} (unparsed={frac_unparsed:.1%}) "
              f"frac_target={frac_target:.2f} "
              f"beta_prior_causal={fit.get('beta_prior_causal', float('nan')):+.4f} "
              f"beta_evidence_causal={fit.get('beta_evidence_causal', float('nan')):+.4f}")

    out = pd.DataFrame(rows)
    out_path = ROOT / "results" / "theory" / "causal_grid_analysis.csv"
    out.to_csv(out_path, index=False)

    valid = out.dropna(subset=["beta_prior_causal", "gamma_prior_revealed"])
    print(f"\n=== critical behavioral-validation result (n={len(valid)} models) ===")
    if len(valid) > 2:
        r_p, p_p = pearsonr(valid.beta_prior_causal, valid.gamma_prior_revealed)
        rho_p, _ = spearmanr(valid.beta_prior_causal, valid.gamma_prior_revealed)
        print(f"gamma_prior (revealed, main task) vs. beta_prior_causal (experimental manipulation): "
              f"r={r_p:.3f} (p={p_p:.4f}), rho={rho_p:.3f}")

        r_e, p_e = pearsonr(valid.beta_evidence_causal, valid.gamma_evidence_revealed)
        rho_e, _ = spearmanr(valid.beta_evidence_causal, valid.gamma_evidence_revealed)
        print(f"gamma_evidence (revealed, main task) vs. beta_evidence_causal (experimental manipulation): "
              f"r={r_e:.3f} (p={p_e:.4f}), rho={rho_e:.3f}")

        # cross terms -- should be WEAKER than the matched pairs above
        r_cross1, p_cross1 = pearsonr(valid.beta_prior_causal, valid.gamma_evidence_revealed)
        r_cross2, p_cross2 = pearsonr(valid.beta_evidence_causal, valid.gamma_prior_revealed)
        print(f"\n(cross-check, should be weaker than matched pairs above:)")
        print(f"  beta_prior_causal vs. gamma_evidence_revealed: r={r_cross1:.3f} (p={p_cross1:.4f})")
        print(f"  beta_evidence_causal vs. gamma_prior_revealed: r={r_cross2:.3f} (p={p_cross2:.4f})")

    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
