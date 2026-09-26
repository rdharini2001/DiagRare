#!/usr/bin/env python3
"""Translates the abstract evidence-elasticity metric into a concrete
clinical number: given each rare disease's REAL population prevalence
(data/expanded/ontology_diseases_expanded.csv's `prevalence` column, a real
fraction of the population, not the equal-weighted vignette-sampling rate
used elsewhere in the benchmark) and each model's OWN measured per-disease
accuracy, Monte-Carlo-free direct expectation:

    expected missed rare-disease diagnoses per 10,000 patients (model m)
        = 10,000 * sum over rare diseases d of [ prevalence(d) * (1 - accuracy(m, d)) ]

This is not a new experiment -- it is a re-weighting of results we already
have (existing baseline predictions + the ontology's own real prevalence
values) by the population mix an equal-weighted benchmark accuracy number
obscures: a benchmark with 12 vignettes per disease treats a
1-in-500,000-prevalence disease the same as a 1-in-5 one, which is exactly
right for measuring a model's diagnostic competence per disease but exactly
wrong for asking "how many real patients would this model misdiagnose."
Reweighting by real prevalence answers the second, clinically load-bearing
question, and lets us correlate the resulting concrete number against
gamma_evidence -- turning an abstract coefficient into "N more missed rare
diagnoses per 10,000 patients."
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[2]


def per_disease_accuracy(preds: pd.DataFrame, vignettes: pd.DataFrame) -> pd.Series:
    m = vignettes.merge(preds, on="vignette_id", how="inner")
    m["correct"] = m["prediction_1"].astype(str).str.strip() == m["target_disease"].astype(str).str.strip()
    return m.groupby("target_disease")["correct"].mean()


def main() -> None:
    ontology = pd.read_csv(ROOT / "data" / "expanded" / "ontology_diseases_expanded.csv")
    vignettes = pd.read_csv(ROOT / "data" / "expanded" / "vignettes_combined.csv")
    per_master = pd.read_csv(ROOT / "results" / "theory" / "per_master_table.csv")

    rare = ontology[ontology["is_rare"]].set_index("disease")["prevalence"]
    print(f"{len(rare)} rare diseases, prevalence range [{rare.min():.6f}, {rare.max():.5f}] "
          f"(fraction of population)")

    rows = []
    for f in sorted((ROOT / "results" / "predictions").glob("*__baseline.csv")):
        model_tag = f.stem.replace("__baseline", "")
        preds = pd.read_csv(f)
        acc_by_disease = per_disease_accuracy(preds, vignettes)

        # only score diseases we actually have both prevalence and accuracy for
        common = rare.index.intersection(acc_by_disease.index)
        miss_rate = 1.0 - acc_by_disease.loc[common]
        expected_missed_per_10k = 10_000 * (rare.loc[common] * miss_rate).sum()
        # also compute the ALL-disease equal-weighted accuracy for reference (matches Sec 6.1)
        overall_acc = (vignettes.merge(preds, on="vignette_id")
                       .pipe(lambda m: (m["prediction_1"].astype(str).str.strip()
                                         == m["target_disease"].astype(str).str.strip()).mean()))

        ge_row = per_master[(per_master.source == "open_weight") & (per_master.model == model_tag)
                             & (per_master.condition == "baseline")]
        row = {"model": model_tag, "top1_accuracy": overall_acc,
               "rare_disease_miss_rate_mean": miss_rate.mean(),
               "expected_missed_rare_dx_per_10k": expected_missed_per_10k,
               "n_rare_diseases_scored": len(common)}
        if len(ge_row):
            row["gamma_evidence"] = ge_row.iloc[0]["gamma_evidence"]
        rows.append(row)
        print(f"{model_tag}: acc={overall_acc:.1%}  expected_missed_rare_dx_per_10k={expected_missed_per_10k:.1f}  "
              f"(mean rare-disease miss rate={miss_rate.mean():.1%})")

    out = pd.DataFrame(rows).sort_values("expected_missed_rare_dx_per_10k")
    out_path = ROOT / "results" / "theory" / "clinical_stakes_simulation.csv"
    out.to_csv(out_path, index=False)

    valid = out.dropna(subset=["gamma_evidence", "expected_missed_rare_dx_per_10k"])
    if len(valid) > 2:
        r, p = pearsonr(valid.gamma_evidence, valid.expected_missed_rare_dx_per_10k)
        rho, _ = spearmanr(valid.gamma_evidence, valid.expected_missed_rare_dx_per_10k)
        print(f"\ngamma_evidence vs. expected missed rare diagnoses per 10k: r={r:.3f} (p={p:.4f}), rho={rho:.3f} "
              f"(expect strongly NEGATIVE: higher evidence-elasticity -> fewer missed rare diagnoses)")

        best, worst = out.iloc[0], out.iloc[-1]
        print(f"\nBest model ({best['model']}): {best['expected_missed_rare_dx_per_10k']:.1f} expected missed "
              f"rare diagnoses per 10,000 patients")
        print(f"Worst model ({worst['model']}): {worst['expected_missed_rare_dx_per_10k']:.1f} expected missed "
              f"rare diagnoses per 10,000 patients")
        print(f"Difference: {worst['expected_missed_rare_dx_per_10k'] - best['expected_missed_rare_dx_per_10k']:.1f} "
              f"more missed diagnoses per 10,000 patients with the worst- vs best-evidence-elasticity model")

    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
