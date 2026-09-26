#!/usr/bin/env python3
"""Held-out predictive comparison of nested utility specifications:
  prior-only:      U = g0 + gp*prior
  evidence-only:    U = g0 + ge*evidence
  additive (main):  U = g0 + gp*prior + ge*evidence
  interaction:      U = g0 + gp*prior + ge*evidence + gpe*prior*evidence

Fit each on a random 70% of vignettes, evaluate Plackett-Luce NLL on the
held-out 30% (5 repeated random splits). If the additive model isn't
reliably beaten by the interaction model, the simple two-axis (gamma_prior,
gamma_evidence) interpretation used throughout the analysis is well supported
rather than an oversimplification.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "analysis"))
from plackett_luce import Choice, build_choices_from_predictions

ROOT = Path(__file__).resolve().parents[3]
L2 = 1e-3


def nll_generic(theta, choices, use_prior, use_evidence, use_interaction):
    total = 0.0
    idx = 0
    g0 = theta[idx]; idx += 1
    gp = theta[idx] if use_prior else 0.0
    if use_prior:
        idx += 1
    ge = theta[idx] if use_evidence else 0.0
    if use_evidence:
        idx += 1
    gpe = theta[idx] if use_interaction else 0.0

    for c in choices:
        u = g0 + gp * c.log_prevalence + ge * c.evidence
        if use_interaction:
            u = u + gpe * c.log_prevalence * c.evidence
        remaining = list(range(len(u)))
        for picked in c.chosen:
            total += u[picked] - logsumexp(u[remaining])
            remaining.remove(picked)
    penalty = L2 * (gp ** 2 + ge ** 2 + gpe ** 2)
    return -total + penalty


def n_params(use_prior, use_evidence, use_interaction):
    return 1 + int(use_prior) + int(use_evidence) + int(use_interaction)


def fit_variant(choices, use_prior, use_evidence, use_interaction):
    k = n_params(use_prior, use_evidence, use_interaction)
    res = minimize(nll_generic, np.zeros(k), args=(choices, use_prior, use_evidence, use_interaction),
                    method="L-BFGS-B")
    return res.x, res.fun


def held_out_nll(theta, choices, use_prior, use_evidence, use_interaction):
    return nll_generic(theta, choices, use_prior, use_evidence, use_interaction)


def main() -> None:
    oracle = pd.read_csv(ROOT / "results" / "theory" / "bayes_oracle_table_expanded.csv")
    models = ["qwen2.5-7b", "phi-3.5-mini", "mistral-7b-instruct", "qwen2.5-3b"]
    specs = [("prior_only", True, False, False), ("evidence_only", False, True, False),
             ("additive", True, True, False), ("interaction", True, True, True)]

    rows = []
    rng = np.random.default_rng(0)
    for model_tag in models:
        preds = pd.read_csv(ROOT / "results" / "predictions" / f"{model_tag}__baseline.csv")
        choices, _, _ = build_choices_from_predictions(preds, oracle)
        n = len(choices)

        for split in range(5):
            idx = rng.permutation(n)
            n_train = int(0.7 * n)
            train = [choices[i] for i in idx[:n_train]]
            test = [choices[i] for i in idx[n_train:]]

            for spec_name, up, ue, ui in specs:
                theta, train_nll = fit_variant(train, up, ue, ui)
                test_nll = held_out_nll(theta, test, up, ue, ui)
                rows.append({"model": model_tag, "split": split, "spec": spec_name,
                             "test_nll_per_choice": test_nll / len(test), "n_test": len(test)})
        print(f"{model_tag} done")

    out = pd.DataFrame(rows)
    out_path = ROOT / "results" / "theory" / "robustness" / "model_comparison.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    pd.set_option("display.width", 200)
    summary = out.groupby(["model", "spec"])["test_nll_per_choice"].agg(["mean", "std"]).reset_index()
    piv = summary.pivot(index="model", columns="spec", values="mean")
    piv = piv[["prior_only", "evidence_only", "additive", "interaction"]]
    print("\nMean held-out NLL per choice occasion (lower = better fit to unseen rankings):")
    print(piv.to_string())
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
