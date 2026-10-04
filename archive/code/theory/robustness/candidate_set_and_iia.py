#!/usr/bin/env python3
"""Candidate-set robustness and IIA (independence of irrelevant alternatives)
diagnostics for the Plackett-Luce fit.

Plackett-Luce (like multinomial/conditional logit) assumes IIA: the relative
odds between two leading candidates shouldn't change just because irrelevant
alternatives are added or removed. We test this directly rather than waiting

1. Refit on THREE candidate-set definitions per vignette:
   (a) full 82-disease list used in the primary evaluation
   (b) same-organ-system diseases only (a smaller, more clinically realistic
       "differential" -- removes obviously-irrelevant alternatives)
   (c) a "hard-negative" set: same-organ-system diseases + the single globally
       most-prevalent disease (tests sensitivity to adding one very salient
       irrelevant-ish alternative)
2. Report how much gamma_prior/gamma_evidence move across these three -- large
   swings would indicate a real IIA problem; small ones mean the concern,
   while real in principle, isn't practically material here.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "analysis"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "theory"))
from plackett_luce import build_choices_from_predictions, fit_plackett_luce
from candidate_sets import build_candidate_sets

ROOT = Path(__file__).resolve().parents[3]


def restrict_oracle_table(oracle: pd.DataFrame, candidate_sets: dict[int, list[str]]) -> pd.DataFrame:
    keep_mask = oracle.apply(lambda r: r["disease"] in candidate_sets.get(r["vignette_id"], []), axis=1)
    return oracle[keep_mask].reset_index(drop=True)


def main() -> None:
    oracle_full = pd.read_csv(ROOT / "results" / "theory" / "bayes_oracle_table_expanded.csv")
    vignettes = pd.read_csv(ROOT / "data" / "expanded" / "vignettes_combined.csv")
    ontology = pd.read_csv(ROOT / "data" / "expanded" / "ontology_diseases_expanded.csv")

    same_organ_sets = build_candidate_sets(vignettes, ontology)  # same-organ + 1 global common (our existing helper)
    # a stricter same-organ-ONLY set (no added global-common disease) for a cleaner IIA probe:
    by_system = {sys_: grp["disease"].tolist() for sys_, grp in ontology.groupby("organ_system")}
    organ_only_sets = {}
    for _, v in vignettes.iterrows():
        cands = list(by_system.get(v["organ_system"], [v["target_disease"]]))
        if v["target_disease"] not in cands:
            cands.append(v["target_disease"])
        organ_only_sets[v["vignette_id"]] = cands

    oracle_organ_only = restrict_oracle_table(oracle_full, organ_only_sets)
    oracle_hard_negative = restrict_oracle_table(oracle_full, same_organ_sets)

    models = ["qwen2.5-7b", "phi-3.5-mini", "mistral-7b-instruct", "qwen2.5-3b", "olmo-2-7b", "yi-1.5-6b"]
    rows = []
    for model_tag in models:
        preds = pd.read_csv(ROOT / "results" / "predictions" / f"{model_tag}__baseline.csv")

        for label, otab in [("full_82", oracle_full), ("same_organ_only", oracle_organ_only),
                             ("same_organ_plus_global_common", oracle_hard_negative)]:
            choices, n_total, n_used = build_choices_from_predictions(preds, otab)
            fit = fit_plackett_luce(choices)
            rows.append({"model": model_tag, "candidate_set": label, "n_used": n_used,
                         "gamma_prior": fit["gamma_prior"], "gamma_evidence": fit["gamma_evidence"], "PER": fit["PER"]})
        print(f"{model_tag} done")

    out = pd.DataFrame(rows)
    out_path = ROOT / "results" / "theory" / "robustness" / "candidate_set_robustness.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    pd.set_option("display.width", 200)
    piv_ge = out.pivot_table(index="model", columns="candidate_set", values="gamma_evidence")
    piv_gp = out.pivot_table(index="model", columns="candidate_set", values="gamma_prior")
    print("\ngamma_evidence across candidate-set definitions:")
    print(piv_ge.to_string())
    print("\ngamma_prior across candidate-set definitions:")
    print(piv_gp.to_string())

    # simple IIA severity index: max relative change in gamma_evidence across the 3 definitions
    rel_change = (piv_ge.max(axis=1) - piv_ge.min(axis=1)) / piv_ge["full_82"].abs()
    print("\nMax relative gamma_evidence swing across candidate-set definitions (IIA severity):")
    print(rel_change.sort_values(ascending=False).to_string())
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
