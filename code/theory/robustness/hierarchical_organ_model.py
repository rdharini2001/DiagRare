#!/usr/bin/env python3
"""Estimate organ-specific evidence coefficients with hierarchical shrinkage."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "analysis"))
from plackett_luce import (  # noqa: E402
    ChoiceBatch, build_choices_from_predictions, fit_plackett_luce, neg_log_likelihood_fast,
)


def numerical_hessian(f, x0, eps=1e-3):
    n = len(x0)
    H = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            xpp, xpm, xmp, xmm = x0.copy(), x0.copy(), x0.copy(), x0.copy()
            xpp[i] += eps; xpp[j] += eps
            xpm[i] += eps; xpm[j] -= eps
            xmp[i] -= eps; xmp[j] += eps
            xmm[i] -= eps; xmm[j] -= eps
            H[i, j] = (f(xpp) - f(xpm) - f(xmp) + f(xmm)) / (4 * eps * eps)
    return H


def fit_with_se(choices) -> dict:
    fit = fit_plackett_luce(choices, n_starts=2)
    batch = ChoiceBatch(choices)
    theta_hat = np.array([fit["gamma0"], fit["gamma_prior"], fit["gamma_evidence"]])
    try:
        H = numerical_hessian(lambda th: neg_log_likelihood_fast(th, batch), theta_hat)
        cov = np.linalg.inv(H)
        diag = np.diag(cov)
        with np.errstate(invalid="ignore"):
            se = np.sqrt(np.where(diag >= 0, diag, np.nan))
    except np.linalg.LinAlgError:
        se = np.array([np.nan, np.nan, np.nan])
    return {**fit, "se_gamma_evidence": se[2], "n_choices": len(choices)}


def der_simonian_laird(estimates: np.ndarray, ses: np.ndarray) -> dict:
    """Standard random-effects meta-analysis: pooled mean + between-study
    variance tau^2 via the method-of-moments (DerSimonian-Laird) estimator,
    with empirical-Bayes shrunken per-study estimates."""
    valid = ~(np.isnan(estimates) | np.isnan(ses) | (ses <= 0))
    est, se = estimates[valid], ses[valid]
    k = len(est)
    if k < 2:
        return {"mu": np.nan, "tau2": np.nan, "shrunken": np.full(len(estimates), np.nan), "valid_mask": valid}

    w_fixed = 1.0 / se**2
    mu_fixed = np.sum(w_fixed * est) / np.sum(w_fixed)
    Q = np.sum(w_fixed * (est - mu_fixed) ** 2)
    df = k - 1
    c = np.sum(w_fixed) - np.sum(w_fixed**2) / np.sum(w_fixed)
    tau2 = max(0.0, (Q - df) / c) if c > 0 else 0.0

    w_re = 1.0 / (se**2 + tau2)
    mu_re = np.sum(w_re * est) / np.sum(w_re)

    shrunken_valid = (w_re * est + (1.0 / tau2 if tau2 > 0 else 0) * mu_re) / (w_re + (1.0 / tau2 if tau2 > 0 else 0)) \
        if tau2 > 0 else np.full(k, mu_re)
    shrunken = np.full(len(estimates), np.nan)
    shrunken[np.where(valid)[0]] = shrunken_valid

    return {"mu": mu_re, "mu_fixed_effect": mu_fixed, "tau2": tau2, "Q": Q, "df": df,
            "shrunken": shrunken, "valid_mask": valid}


def main() -> None:
    vignettes = pd.read_csv(ROOT / "data" / "expanded" / "vignettes_combined.csv")
    oracle = pd.read_csv(ROOT / "results" / "theory" / "bayes_oracle_table_expanded.csv")
    per_master = pd.read_csv(ROOT / "results" / "theory" / "per_master_table.csv")
    organs = sorted(vignettes["organ_system"].unique())

    models = ["qwen2.5-0.5b", "qwen2.5-1.5b", "qwen2.5-3b", "qwen2.5-7b",
              "yi-1.5-6b", "olmo-2-7b", "mistral-7b-instruct", "phi-3.5-mini"]

    organ_rows, meta_rows = [], []
    for m in models:
        f = ROOT / "results" / "predictions" / f"{m}__baseline.csv"
        if not f.exists():
            continue
        preds = pd.read_csv(f)

        per_organ_ge, per_organ_se = [], []
        for organ in organs:
            vids = set(vignettes.loc[vignettes.organ_system == organ, "vignette_id"])
            choices, n_total, n_used = build_choices_from_predictions(preds, oracle, vignette_ids=vids)
            if n_used < 8:
                per_organ_ge.append(np.nan)
                per_organ_se.append(np.nan)
                organ_rows.append({"model": m, "organ_system": organ, "n_used": n_used,
                                    "gamma_evidence": np.nan, "se_gamma_evidence": np.nan})
                continue
            fit = fit_with_se(choices)
            per_organ_ge.append(fit["gamma_evidence"])
            per_organ_se.append(fit["se_gamma_evidence"])
            organ_rows.append({"model": m, "organ_system": organ, "n_used": n_used,
                                "gamma_evidence": fit["gamma_evidence"], "se_gamma_evidence": fit["se_gamma_evidence"]})

        re = der_simonian_laird(np.array(per_organ_ge), np.array(per_organ_se))
        for i, organ in enumerate(organs):
            for row in organ_rows[-len(organs):]:
                if row["organ_system"] == organ and row["model"] == m:
                    row["gamma_evidence_shrunken"] = re["shrunken"][i]

        pooled_row = per_master[(per_master.source == "open_weight") & (per_master.model == m)
                                 & (per_master.condition == "baseline")]
        pooled_ge = pooled_row.iloc[0]["gamma_evidence"] if len(pooled_row) else np.nan

        meta_rows.append({
            "model": m, "mu_random_effects": re["mu"], "mu_fixed_effect": re.get("mu_fixed_effect", np.nan),
            "tau2_between_organ": re["tau2"], "Q_statistic": re.get("Q", np.nan), "df": re.get("df", np.nan),
            "pooled_gamma_evidence_main_paper": pooled_ge,
            "n_organs_used": int(np.sum(re["valid_mask"])) if isinstance(re["valid_mask"], np.ndarray) else 0,
        })
        print(f"{m}: pooled(main-paper)={pooled_ge:.4f}  random-effects mu={re['mu']:.4f}  "
              f"tau2(between-organ)={re['tau2']:.5f}  ({int(np.sum(re['valid_mask']))} organs)")

    out_dir = ROOT / "results" / "theory" / "robustness"
    out_dir.mkdir(parents=True, exist_ok=True)
    organ_df = pd.DataFrame(organ_rows)
    organ_df.to_csv(out_dir / "hierarchical_organ_estimates.csv", index=False)
    meta_df = pd.DataFrame(meta_rows)
    meta_df.to_csv(out_dir / "hierarchical_organ_meta.csv", index=False)

    print("\n=== Consistency check: does random-effects mu match the main paper's pooled fit? ===")
    meta_df["abs_diff"] = (meta_df.mu_random_effects - meta_df.pooled_gamma_evidence_main_paper).abs()
    print(meta_df[["model", "pooled_gamma_evidence_main_paper", "mu_random_effects", "abs_diff", "tau2_between_organ"]]
          .to_string(index=False))

    print(f"\nWrote {out_dir / 'hierarchical_organ_estimates.csv'} and {out_dir / 'hierarchical_organ_meta.csv'}")


if __name__ == "__main__":
    main()
