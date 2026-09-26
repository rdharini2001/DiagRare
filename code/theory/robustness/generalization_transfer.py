#!/usr/bin/env python3
"""Does the (gamma_prior, gamma_evidence) MEASUREMENT itself transfer across
distributions, not just describe whichever data it was fit on?

1. Fit on original-58-disease vignettes only, predict (evaluate held-out PL
   log-likelihood) on the 3 held-out organ systems (never seen in any
   training/tuning) -- and the reverse.
2. Fit on easy+medium difficulty, evaluate on hard (and the reverse).
3. Leave-one-organ-system-out: fit on 8 of 9 organ systems (6 original + 3
   new = 9 total), evaluate held-out log-likelihood on the withheld one, for
   every fold.

All of this reuses EXISTING baseline predictions -- no new GPU inference.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "analysis"))
from plackett_luce import fit_plackett_luce, build_choices_from_predictions, neg_log_likelihood

ROOT = Path(__file__).resolve().parents[3]


def held_out_nll_per_choice(fit: dict, choices: list) -> float:
    if not choices:
        return np.nan
    theta = np.array([fit["gamma0"], fit["gamma_prior"], fit["gamma_evidence"]])
    return neg_log_likelihood(theta, choices) / len(choices)


def main() -> None:
    oracle = pd.read_csv(ROOT / "results" / "theory" / "bayes_oracle_table_expanded.csv")
    vignettes = pd.read_csv(ROOT / "data" / "expanded" / "vignettes_combined.csv")
    models = ["qwen2.5-7b", "phi-3.5-mini", "mistral-7b-instruct", "qwen2.5-3b"]

    rows = []
    for model_tag in models:
        preds = pd.read_csv(ROOT / "results" / "predictions" / f"{model_tag}__baseline.csv")

        # (1) original <-> expanded_heldout
        orig_vids = set(vignettes.loc[vignettes.split == "original", "vignette_id"])
        held_vids = set(vignettes.loc[vignettes.split == "expanded_heldout", "vignette_id"])
        c_orig, _, _ = build_choices_from_predictions(preds, oracle, vignette_ids=orig_vids)
        c_held, _, _ = build_choices_from_predictions(preds, oracle, vignette_ids=held_vids)
        fit_orig, fit_held = fit_plackett_luce(c_orig), fit_plackett_luce(c_held)
        rows.append({"model": model_tag, "test": "fit_original_predict_heldout_organs",
                     "gamma_prior_fit": fit_orig["gamma_prior"], "gamma_evidence_fit": fit_orig["gamma_evidence"],
                     "held_out_nll_per_choice": held_out_nll_per_choice(fit_orig, c_held)})
        rows.append({"model": model_tag, "test": "fit_heldout_organs_predict_original",
                     "gamma_prior_fit": fit_held["gamma_prior"], "gamma_evidence_fit": fit_held["gamma_evidence"],
                     "held_out_nll_per_choice": held_out_nll_per_choice(fit_held, c_orig)})

        # (2) easy+medium <-> hard
        em_vids = set(vignettes.loc[vignettes.difficulty.isin(["easy", "medium"]), "vignette_id"])
        hard_vids = set(vignettes.loc[vignettes.difficulty == "hard", "vignette_id"])
        c_em, _, _ = build_choices_from_predictions(preds, oracle, vignette_ids=em_vids)
        c_hard, _, _ = build_choices_from_predictions(preds, oracle, vignette_ids=hard_vids)
        fit_em, fit_hard = fit_plackett_luce(c_em), fit_plackett_luce(c_hard)
        rows.append({"model": model_tag, "test": "fit_easymed_predict_hard",
                     "gamma_prior_fit": fit_em["gamma_prior"], "gamma_evidence_fit": fit_em["gamma_evidence"],
                     "held_out_nll_per_choice": held_out_nll_per_choice(fit_em, c_hard)})
        rows.append({"model": model_tag, "test": "fit_hard_predict_easymed",
                     "gamma_prior_fit": fit_hard["gamma_prior"], "gamma_evidence_fit": fit_hard["gamma_evidence"],
                     "held_out_nll_per_choice": held_out_nll_per_choice(fit_hard, c_em)})

        # (3) leave-one-organ-system-out
        for organ in sorted(vignettes["organ_system"].unique()):
            train_vids = set(vignettes.loc[vignettes.organ_system != organ, "vignette_id"])
            test_vids = set(vignettes.loc[vignettes.organ_system == organ, "vignette_id"])
            c_train, _, _ = build_choices_from_predictions(preds, oracle, vignette_ids=train_vids)
            c_test, _, _ = build_choices_from_predictions(preds, oracle, vignette_ids=test_vids)
            if len(c_test) < 5:
                continue
            fit_train = fit_plackett_luce(c_train)
            rows.append({"model": model_tag, "test": f"leave_out_{organ}",
                         "gamma_prior_fit": fit_train["gamma_prior"], "gamma_evidence_fit": fit_train["gamma_evidence"],
                         "held_out_nll_per_choice": held_out_nll_per_choice(fit_train, c_test)})
        print(f"{model_tag} done")

    out = pd.DataFrame(rows)
    out_path = ROOT / "results" / "theory" / "robustness" / "generalization_transfer.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    pd.set_option("display.width", 200)
    print("\nCross-distribution transfer (original <-> held-out organs, easy/medium <-> hard):")
    print(out[~out.test.str.startswith("leave_out_")].to_string(index=False))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
