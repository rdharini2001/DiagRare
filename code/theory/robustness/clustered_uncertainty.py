#!/usr/bin/env python3
"""The 984 vignettes are not 984 independent clinical concepts: there are 12
vignettes per disease, and diseases share organ systems. The main-paper
bootstrap (analysis/plackett_luce.bootstrap_ci) resamples VIGNETTES, silently
treating them as independent. This re-estimates uncertainty three more
conservative ways:

  1. Disease-cluster bootstrap: resample DISEASES with replacement, keep all
     ~12 vignettes for each resampled disease intact (respects the true
     clustering).
  2. Leave-one-disease-out jackknife: refit with each disease held out in
     turn; a coefficient driven by a handful of unusual diseases will show
     high jackknife variance even if the (wrong) vignette-level bootstrap
     looked tight.
  3. Organ-system cluster bootstrap: resample ORGAN SYSTEMS with replacement
     (the coarsest, most conservative clustering available here).

If significance survives disease- and organ-level resampling, not just
vignette-level, that is a much stronger claim.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "analysis"))
from plackett_luce import Choice, ChoiceBatch, fit_plackett_luce, build_choices_from_predictions

ROOT = Path(__file__).resolve().parents[3]


def build_choices_with_labels(preds: pd.DataFrame, oracle: pd.DataFrame, vignettes: pd.DataFrame):
    """Like build_choices_from_predictions but also returns each choice's
    (disease, organ_system) label for cluster resampling."""
    choices, n_total, n_used = build_choices_from_predictions(preds, oracle)
    # re-derive labels by re-walking the same predictions in the same order build_choices_from_predictions uses
    oracle_by_vid = {vid: grp.reset_index(drop=True) for vid, grp in oracle.groupby("vignette_id")}
    vign_lookup = vignettes.set_index("vignette_id")[["target_disease", "organ_system"]]
    labels = []
    for _, row in preds.iterrows():
        vid = row["vignette_id"]
        if vid not in oracle_by_vid or vid not in vign_lookup.index:
            continue
        grp = oracle_by_vid[vid]
        disease_to_idx = {d: i for i, d in enumerate(grp["disease"])}
        chosen = []
        seen = set()
        for col in ("prediction_1", "prediction_2", "prediction_3"):
            val = str(row.get(col, "")).strip()
            if val in disease_to_idx and val not in seen:
                chosen.append(disease_to_idx[val]); seen.add(val)
            else:
                break
        if not chosen:
            continue
        labels.append((vign_lookup.loc[vid, "target_disease"], vign_lookup.loc[vid, "organ_system"]))
    assert len(labels) == len(choices), f"{len(labels)} vs {len(choices)}"
    return choices, labels


def cluster_bootstrap(choices: list[Choice], cluster_ids: list, n_boot: int = 500, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    unique_clusters = sorted(set(cluster_ids))
    by_cluster = {c: [ch for ch, cid in zip(choices, cluster_ids) if cid == c] for c in unique_clusters}
    gps, ges, pers = [], [], []
    for b in range(n_boot):
        sampled_clusters = rng.choice(unique_clusters, size=len(unique_clusters), replace=True)
        sample = [ch for c in sampled_clusters for ch in by_cluster[c]]
        fit = fit_plackett_luce(sample, n_starts=1, seed=seed + b)
        if fit["converged"] and np.isfinite(fit["PER"]):
            gps.append(fit["gamma_prior"]); ges.append(fit["gamma_evidence"]); pers.append(fit["PER"])
    if not gps:
        return {"gp_lo": np.nan, "gp_hi": np.nan, "ge_lo": np.nan, "ge_hi": np.nan}
    return {"gp_lo": np.percentile(gps, 2.5), "gp_hi": np.percentile(gps, 97.5),
            "ge_lo": np.percentile(ges, 2.5), "ge_hi": np.percentile(ges, 97.5),
            "gp_mean": np.mean(gps), "ge_mean": np.mean(ges)}


def leave_one_disease_out(choices: list[Choice], diseases: list[str]) -> pd.DataFrame:
    rows = []
    for d in sorted(set(diseases)):
        kept = [ch for ch, dd in zip(choices, diseases) if dd != d]
        fit = fit_plackett_luce(kept, n_starts=1)
        rows.append({"held_out_disease": d, "gamma_prior": fit["gamma_prior"], "gamma_evidence": fit["gamma_evidence"]})
    return pd.DataFrame(rows)


def main() -> None:
    oracle = pd.read_csv(ROOT / "results" / "theory" / "bayes_oracle_table_expanded.csv")
    vignettes = pd.read_csv(ROOT / "data" / "expanded" / "vignettes_combined.csv")
    models = ["qwen2.5-7b", "phi-3.5-mini"]  # jackknife over ~82 diseases x model is the expensive part; 2 representative models

    summary_rows = []
    for model_tag in models:
        preds = pd.read_csv(ROOT / "results" / "predictions" / f"{model_tag}__baseline.csv")
        choices, labels = build_choices_with_labels(preds, oracle, vignettes)
        diseases = [l[0] for l in labels]
        organs = [l[1] for l in labels]

        fit_point = fit_plackett_luce(choices)
        disease_boot = cluster_bootstrap(choices, diseases, n_boot=500, seed=1)
        organ_boot = cluster_bootstrap(choices, organs, n_boot=500, seed=2)
        loo = leave_one_disease_out(choices, diseases)
        loo_path = ROOT / "results" / "theory" / "robustness" / f"loo_disease_{model_tag}.csv"
        loo.to_csv(loo_path, index=False)

        summary_rows.append({
            "model": model_tag, "gamma_prior_point": fit_point["gamma_prior"], "gamma_evidence_point": fit_point["gamma_evidence"],
            "gp_disease_boot_lo": disease_boot["gp_lo"], "gp_disease_boot_hi": disease_boot["gp_hi"],
            "ge_disease_boot_lo": disease_boot["ge_lo"], "ge_disease_boot_hi": disease_boot["ge_hi"],
            "gp_organ_boot_lo": organ_boot["gp_lo"], "gp_organ_boot_hi": organ_boot["gp_hi"],
            "ge_organ_boot_lo": organ_boot["ge_lo"], "ge_organ_boot_hi": organ_boot["ge_hi"],
            "loo_ge_min": loo.gamma_evidence.min(), "loo_ge_max": loo.gamma_evidence.max(),
            "loo_ge_std": loo.gamma_evidence.std(),
        })
        print(f"{model_tag}: point ge={fit_point['gamma_evidence']:.3f} | "
              f"disease-cluster-boot 95% CI=[{disease_boot['ge_lo']:.3f},{disease_boot['ge_hi']:.3f}] | "
              f"organ-cluster-boot 95% CI=[{organ_boot['ge_lo']:.3f},{organ_boot['ge_hi']:.3f}] | "
              f"LOO-disease range=[{loo.gamma_evidence.min():.3f},{loo.gamma_evidence.max():.3f}] "
              f"(std={loo.gamma_evidence.std():.4f})")

    out = pd.DataFrame(summary_rows)
    out_path = ROOT / "results" / "theory" / "robustness" / "clustered_uncertainty.csv"
    out.to_csv(out_path, index=False)
    print(f"\nWrote {out_path} and per-model loo_disease_*.csv files")


if __name__ == "__main__":
    main()
