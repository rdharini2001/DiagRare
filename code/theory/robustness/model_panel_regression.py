#!/usr/bin/env python3
"""Model-level regressions for evidence responsiveness, scale, family, and output coverage.

The analysis asks whether the evidence coefficient explains variation in diagnostic
accuracy beyond simple model-level covariates.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parents[3]


def main() -> None:
    meta = pd.read_csv(ROOT / "data" / "model_metadata.csv")
    per_master = pd.read_csv(ROOT / "results" / "theory" / "per_master_table.csv")
    acc = pd.read_csv(ROOT / "results" / "open_weight_summary.csv")

    per_base = per_master[(per_master.source == "open_weight") & (per_master.condition == "baseline")]
    acc_base = acc[(acc.condition == "baseline") & (acc.split == "ALL")]

    d = meta.merge(per_base[["model", "gamma_evidence", "gamma_prior", "n_used", "n_total"]],
                    left_on="model_tag", right_on="model", how="inner")
    d = d.merge(acc_base[["model", "top1_accuracy", "unparsed_rate"]], on="model", how="inner")
    d["log_params"] = np.log(d["param_count_b"])
    d["unparsed_rate"] = d["unparsed_rate"].fillna(1.0 - d["n_used"] / d["n_total"])
    d["is_medical_specialized"] = d["is_medical_specialized"].astype(bool)

    out_path = ROOT / "results" / "theory" / "robustness" / "model_panel_regression_data.csv"
    d.to_csv(out_path, index=False)
    print(f"Panel: {len(d)} models across {d['family'].nunique()} families, "
          f"{d['organization'].nunique()} organizations")
    print(d[["model_tag", "family", "param_count_b", "top1_accuracy", "gamma_evidence", "unparsed_rate"]]
          .sort_values("top1_accuracy", ascending=False).to_string(index=False))

    results = {}

    # Model 1: accuracy ~ gamma_evidence alone (baseline correlation, matches Sec 6.1 but on the full panel)
    m1 = smf.ols("top1_accuracy ~ gamma_evidence", data=d).fit()
    results["gamma_evidence_only"] = m1

    # Model 2: accuracy ~ log_params alone (does scale alone explain it?)
    m2 = smf.ols("top1_accuracy ~ log_params", data=d).fit()
    results["log_params_only"] = m2

    m3 = smf.ols("top1_accuracy ~ gamma_evidence + log_params + unparsed_rate", data=d).fit()
    results["full_no_family_fe"] = m3

    # Model 4: + family fixed effects (only if enough within-family variation exists)
    family_counts = d["family"].value_counts()
    multi_model_families = family_counts[family_counts >= 2].index.tolist()
    if len(multi_model_families) >= 2:
        d["family_grp"] = d["family"].apply(lambda f: f if f in multi_model_families else "Other")
        m4 = smf.ols("top1_accuracy ~ gamma_evidence + log_params + unparsed_rate + C(family_grp)", data=d).fit()
        results["full_with_family_fe"] = m4

    # medical fine-tune predict accuracy or gamma_evidence beyond scale/family,
    # once we only count models with genuinely usable output (unparsed_rate < 0.5)?
    d_usable = d[d["unparsed_rate"] < 0.5]
    if d_usable["is_medical_specialized"].nunique() == 2:
        m5 = smf.ols("top1_accuracy ~ gamma_evidence + log_params + C(is_medical_specialized)", data=d_usable).fit()
        results["medical_specialization_effect"] = m5

    all_terms = ["gamma_evidence", "log_params", "unparsed_rate", "C(is_medical_specialized)[T.True]"]

    print("\n" + "=" * 70)
    for name, m in results.items():
        print(f"\n--- {name} (n={int(m.nobs)}, R2={m.rsquared:.3f}, adj-R2={m.rsquared_adj:.3f}) ---")
        for term in all_terms:
            if term in m.params.index:
                print(f"  {term:35s} coef={m.params[term]:+.4f}  se={m.bse[term]:.4f}  p={m.pvalues[term]:.4g}")

    summary_rows = []
    for name, m in results.items():
        row = {"model_spec": name, "n": int(m.nobs), "r2": m.rsquared, "adj_r2": m.rsquared_adj}
        for term in all_terms:
            if term in m.params.index:
                row[f"{term}_coef"] = m.params[term]
                row[f"{term}_p"] = m.pvalues[term]
        summary_rows.append(row)
    out = pd.DataFrame(summary_rows)
    out_path2 = ROOT / "results" / "theory" / "robustness" / "model_panel_regression.csv"
    out.to_csv(out_path2, index=False)

    print(f"\n{'='*70}")
    if "full_with_family_fe" in results:
        m = results["full_with_family_fe"]
        p_ge = m.pvalues["gamma_evidence"]
        print(f"KEY RESULT: gamma_evidence remains {'SIGNIFICANT' if p_ge < 0.05 else 'not significant'} "
              f"(p={p_ge:.4g}) after controlling for log(param count), parseability, AND family fixed effects.")
    if "medical_specialization_effect" in results:
        m = results["medical_specialization_effect"]
        term = "C(is_medical_specialized)[T.True]"
        if term in m.params.index:
            coef, p = m.params[term], m.pvalues[term]
            print(f"MEDICAL-SPECIALIZATION RESULT (n={int(m.nobs)} models with usable output): "
                  f"being a domain-specialized medical fine-tune is associated with "
                  f"{coef:+.4f} accuracy (p={p:.4g}) after controlling for gamma_evidence and log(param count).")
    print(f"\nWrote {out_path} and {out_path2}")


if __name__ == "__main__":
    main()
