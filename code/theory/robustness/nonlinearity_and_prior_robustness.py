#!/usr/bin/env python3
"""Robustness checks for the linear evidence term and the disease-prior representation.

The script compares the linear evidence specification with a binned alternative and
refits the rank model under several representations of disease prevalence.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "analysis"))
from plackett_luce import Choice, build_choices_from_predictions, fit_plackett_luce

ROOT = Path(__file__).resolve().parents[3]
L2 = 1e-3


def nll_binned_evidence(theta, choices, bin_edges):
    """utility = g0 + gp*prior + sum_b gb*1[evidence in bin b] -- a step-function
    (binned) evidence term instead of the linear gamma_evidence*evidence term."""
    n_bins = len(bin_edges) - 1
    g0, gp = theta[0], theta[1]
    bin_coefs = theta[2:2 + n_bins]
    total = 0.0
    for c in choices:
        bin_idx = np.clip(np.digitize(c.evidence, bin_edges) - 1, 0, n_bins - 1)
        u = g0 + gp * c.log_prevalence + bin_coefs[bin_idx]
        remaining = list(range(len(u)))
        for picked in c.chosen:
            total += u[picked] - logsumexp(u[remaining])
            remaining.remove(picked)
    penalty = L2 * (gp ** 2 + np.sum(bin_coefs ** 2))
    return -total + penalty


def held_out_nll_linear(theta, choices):
    g0, gp, ge = theta
    total = 0.0
    for c in choices:
        u = g0 + gp * c.log_prevalence + ge * c.evidence
        remaining = list(range(len(u)))
        for picked in c.chosen:
            total += u[picked] - logsumexp(u[remaining])
            remaining.remove(picked)
    return -total


def held_out_nll_binned(theta, choices, bin_edges):
    n_bins = len(bin_edges) - 1
    g0, gp = theta[0], theta[1]
    bin_coefs = theta[2:2 + n_bins]
    total = 0.0
    for c in choices:
        bin_idx = np.clip(np.digitize(c.evidence, bin_edges) - 1, 0, n_bins - 1)
        u = g0 + gp * c.log_prevalence + bin_coefs[bin_idx]
        remaining = list(range(len(u)))
        for picked in c.chosen:
            total += u[picked] - logsumexp(u[remaining])
            remaining.remove(picked)
    return -total


def nonlinearity_test(oracle: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    all_evidence = oracle["evidence_loglik_ratio"].to_numpy()
    bin_edges = np.quantile(all_evidence, [0, 0.2, 0.4, 0.6, 0.8, 1.0])
    bin_edges[0] -= 1e-6
    bin_edges[-1] += 1e-6
    n_bins = len(bin_edges) - 1

    rows = []
    rng = np.random.default_rng(0)
    for model_tag in models:
        preds = pd.read_csv(ROOT / "results" / "predictions" / f"{model_tag}__baseline.csv")
        choices, _, _ = build_choices_from_predictions(preds, oracle)
        n = len(choices)
        idx = rng.permutation(n)
        n_train = int(0.7 * n)
        train = [choices[i] for i in idx[:n_train]]
        test = [choices[i] for i in idx[n_train:]]

        res_lin = minimize(held_out_nll_linear, np.zeros(3), args=(train,), method="L-BFGS-B")
        test_nll_lin = held_out_nll_linear(res_lin.x, test) / len(test)

        res_bin = minimize(nll_binned_evidence, np.zeros(2 + n_bins), args=(train, bin_edges), method="L-BFGS-B")
        test_nll_bin = held_out_nll_binned(res_bin.x, test, bin_edges) / len(test)

        rows.append({"model": model_tag, "test_nll_linear": test_nll_lin, "test_nll_binned": test_nll_bin,
                     "binned_better_by": test_nll_lin - test_nll_bin})
        print(f"{model_tag}: linear={test_nll_lin:.3f} binned(5-bin)={test_nll_bin:.3f} "
              f"(binned better by {test_nll_lin - test_nll_bin:+.3f} nats/choice)")
    return pd.DataFrame(rows)


def prevalence_representation_robustness(oracle: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    ontology = oracle.drop_duplicates("disease")[["disease", "log_prevalence"]].copy()
    ontology["prevalence_rank"] = ontology["log_prevalence"].rank()
    # normalize rank to a comparable scale (z-score) so gamma magnitudes are interpretable
    ontology["prevalence_rank_z"] = (ontology["prevalence_rank"] - ontology["prevalence_rank"].mean()) / ontology["prevalence_rank"].std()
    tertiles = ontology["log_prevalence"].quantile([1 / 3, 2 / 3]).to_numpy()
    def bin3(x):
        if x <= tertiles[0]:
            return -1.0  # rare
        elif x <= tertiles[1]:
            return 0.0   # uncommon
        return 1.0        # common
    ontology["prevalence_bin3"] = ontology["log_prevalence"].apply(bin3)

    rank_map = dict(zip(ontology.disease, ontology.prevalence_rank_z))
    bin_map = dict(zip(ontology.disease, ontology.prevalence_bin3))

    rows = []
    for model_tag in models:
        preds = pd.read_csv(ROOT / "results" / "predictions" / f"{model_tag}__baseline.csv")

        oracle_rank = oracle.copy()
        oracle_rank["log_prevalence"] = oracle_rank["disease"].map(rank_map)
        oracle_bin = oracle.copy()
        oracle_bin["log_prevalence"] = oracle_bin["disease"].map(bin_map)

        fits = {}
        for label, otab in [("continuous", oracle), ("rank", oracle_rank), ("bin3", oracle_bin)]:
            choices, _, n_used = build_choices_from_predictions(preds, otab)
            fit = fit_plackett_luce(choices)
            fits[label] = fit
        rows.append({
            "model": model_tag,
            "gamma_prior_continuous": fits["continuous"]["gamma_prior"], "gamma_evidence_continuous": fits["continuous"]["gamma_evidence"],
            "gamma_prior_rank": fits["rank"]["gamma_prior"], "gamma_evidence_rank": fits["rank"]["gamma_evidence"],
            "gamma_prior_bin3": fits["bin3"]["gamma_prior"], "gamma_evidence_bin3": fits["bin3"]["gamma_evidence"],
        })
        print(f"{model_tag}: gamma_evidence stays {fits['continuous']['gamma_evidence']:.3f} / "
              f"{fits['rank']['gamma_evidence']:.3f} / {fits['bin3']['gamma_evidence']:.3f} "
              f"across continuous/rank/bin3 prior representations")
    return pd.DataFrame(rows)


def main() -> None:
    oracle = pd.read_csv(ROOT / "results" / "theory" / "bayes_oracle_table_expanded.csv")
    models = ["qwen2.5-7b", "phi-3.5-mini", "mistral-7b-instruct", "qwen2.5-3b"]

    print("=== Nonlinearity test (linear vs. 5-bin evidence term, held-out NLL) ===")
    nl = nonlinearity_test(oracle, models)
    nl.to_csv(ROOT / "results" / "theory" / "robustness" / "nonlinearity_test.csv", index=False)

    print("\n=== Prevalence-representation robustness (continuous / rank / 3-bin) ===")
    models8 = ["qwen2.5-0.5b", "qwen2.5-1.5b", "qwen2.5-3b", "qwen2.5-7b",
               "yi-1.5-6b", "olmo-2-7b", "mistral-7b-instruct", "phi-3.5-mini"]
    pr = prevalence_representation_robustness(oracle, models8)
    pr.to_csv(ROOT / "results" / "theory" / "robustness" / "prevalence_representation_robustness.csv", index=False)

    from scipy.stats import spearmanr
    rho_cont_rank, _ = spearmanr(pr.gamma_evidence_continuous, pr.gamma_evidence_rank)
    rho_cont_bin, _ = spearmanr(pr.gamma_evidence_continuous, pr.gamma_evidence_bin3)
    print(f"\nSpearman rank correlation of gamma_evidence across prior representations: "
          f"continuous-vs-rank={rho_cont_rank:.3f}, continuous-vs-bin3={rho_cont_bin:.3f}")
    print("\nWrote nonlinearity_test.csv and prevalence_representation_robustness.csv")


if __name__ == "__main__":
    main()
