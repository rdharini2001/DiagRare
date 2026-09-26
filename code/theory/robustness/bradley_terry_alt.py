#!/usr/bin/env python3
"""Bradley-Terry pairwise-logistic estimator, as an alternative to
Plackett-Luce, fit on the SAME data. If the scientific conclusion (which
model has higher evidence-sensitivity, in what rank order) is an artifact of
choosing Plackett-Luce specifically, it should NOT survive under this
completely different (pairwise, not sequential-softmax) estimation approach.

Reduction: a top-3-of-N ranking (c1 > c2 > c3 > {N-3 unranked}) implies the
pairwise wins c1-beats-everyone-else, c2-beats-everyone-except-c1,
c3-beats-everyone-except-c1,c2 (unranked items' relative order among
themselves is NOT implied and is not used). Each pair (winner, loser)
contributes a logistic-regression observation on (delta_log_prevalence,
delta_evidence) = (x_winner - x_loser), fit via
P(winner beats loser) = sigmoid(gamma_prior*delta_prior + gamma_evidence*delta_evidence).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "analysis"))
from plackett_luce import build_choices_from_predictions

ROOT = Path(__file__).resolve().parents[3]


def build_pairwise_frame(choices) -> pd.DataFrame:
    rows = []
    for c in choices:
        remaining = list(range(len(c.log_prevalence)))
        for winner in c.chosen:
            for loser in remaining:
                if loser == winner:
                    continue
                rows.append({
                    "delta_prior": c.log_prevalence[winner] - c.log_prevalence[loser],
                    "delta_evidence": c.evidence[winner] - c.evidence[loser],
                })
            remaining.remove(winner)
    return pd.DataFrame(rows)


def fit_bradley_terry(choices) -> dict:
    pairs = build_pairwise_frame(choices)
    if len(pairs) < 20:
        return {"gamma_prior_bt": np.nan, "gamma_evidence_bt": np.nan, "PER_bt": np.nan, "n_pairs": len(pairs)}
    # Every row is a "win" for the first-listed candidate by construction, so this is
    # equivalent to logistic regression of a constant-1 outcome on the DIFFERENCE
    # features with no intercept -- the standard trick for fitting Bradley-Terry via
    # symmetric augmentation: duplicate each pair with the covariates negated and
    # outcome 0, which is mathematically identical to the conditional MLE.
    X_pos = pairs[["delta_prior", "delta_evidence"]].to_numpy()
    X = np.vstack([X_pos, -X_pos])
    y = np.concatenate([np.ones(len(X_pos)), np.zeros(len(X_pos))])
    model = sm.Logit(y, X)
    try:
        res = model.fit(disp=0, maxiter=200)
        gp, ge = res.params
        per = gp / ge if abs(ge) > 1e-8 else np.inf
        return {"gamma_prior_bt": gp, "gamma_evidence_bt": ge, "PER_bt": per, "n_pairs": len(pairs),
                "se_gamma_prior_bt": res.bse[0], "se_gamma_evidence_bt": res.bse[1],
                "p_gamma_prior_bt": res.pvalues[0], "p_gamma_evidence_bt": res.pvalues[1]}
    except Exception as e:
        return {"gamma_prior_bt": np.nan, "gamma_evidence_bt": np.nan, "PER_bt": np.nan,
                "n_pairs": len(pairs), "error": str(e)}


def main() -> None:
    oracle = pd.read_csv(ROOT / "results" / "theory" / "bayes_oracle_table_expanded.csv")
    per_master = pd.read_csv(ROOT / "results" / "theory" / "per_master_table.csv")
    models = ["qwen2.5-0.5b", "qwen2.5-1.5b", "qwen2.5-3b", "qwen2.5-7b",
              "yi-1.5-6b", "olmo-2-7b", "mistral-7b-instruct", "phi-3.5-mini"]

    rows = []
    for model_tag in models:
        preds = pd.read_csv(ROOT / "results" / "predictions" / f"{model_tag}__baseline.csv")
        choices, n_total, n_used = build_choices_from_predictions(preds, oracle)
        bt = fit_bradley_terry(choices)

        pl_row = per_master[(per_master.source == "open_weight") & (per_master.model == model_tag)
                             & (per_master.condition == "baseline")]
        row = {"model": model_tag, "n_used": n_used, **bt}
        if len(pl_row):
            r = pl_row.iloc[0]
            row.update({"gamma_prior_pl": r["gamma_prior"], "gamma_evidence_pl": r["gamma_evidence"], "PER_pl": r["PER"]})
        rows.append(row)
        print(f"{model_tag}: BT gp={bt.get('gamma_prior_bt'):.4f} ge={bt.get('gamma_evidence_bt'):.4f} "
              f"(n_pairs={bt.get('n_pairs')})" if np.isfinite(bt.get("gamma_evidence_bt", np.nan))
              else f"{model_tag}: BT fit failed/insufficient data")

    out = pd.DataFrame(rows)
    out_path = ROOT / "results" / "theory" / "robustness" / "bradley_terry_comparison.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    from scipy.stats import spearmanr
    valid = out.dropna(subset=["gamma_evidence_bt", "gamma_evidence_pl"])
    if len(valid) > 2:
        rho, p = spearmanr(valid.gamma_evidence_bt, valid.gamma_evidence_pl)
        print(f"\nSpearman rank correlation between Bradley-Terry and Plackett-Luce "
              f"gamma_evidence across {len(valid)} models: rho={rho:.3f} (p={p:.4f})")
    pd.set_option("display.width", 200)
    print(out[["model", "gamma_prior_pl", "gamma_prior_bt", "gamma_evidence_pl", "gamma_evidence_bt"]].to_string(index=False))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
