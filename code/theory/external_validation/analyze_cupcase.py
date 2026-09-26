#!/usr/bin/env python3
"""Fits the SAME Plackett-Luce-style discrete-choice decomposition used
throughout the main paper, but on CUPCase -- a real, independently-authored
external dataset with no relationship to DiagRare-X. This is the analysis
external-validity test: does gamma_evidence's model ranking (and its
correlation with accuracy) survive on data we did not design?

Prior axis: log(PubMed article count + 1) for each candidate -- a real,
externally-sourced signal, independent of anything in DiagRare-X.
Evidence axis: TF-IDF cosine similarity between the case presentation and
each candidate's text (score_cupcase_evidence.py) -- deliberately the same
kind of simple, transparent lexical-overlap proxy that our own robustness
check (theory/robustness/per_covariate_robustness.py) already showed gives
IDENTICAL model rankings to a much more sophisticated oracle on DiagRare-X
itself, so using it here is principled, not a shortcut.

Since each case has EXACTLY 4 candidates (no variable-N ranking machinery
needed), this fits a straightforward multinomial logit via statsmodels
rather than reusing the Plackett-Luce partial-ranking code -- mathematically
equivalent to Plackett-Luce for a single, complete top-1 choice among a
fixed candidate set.
"""
from __future__ import annotations

import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[3]


def nll_choice(theta, log_prior, evidence, chosen_idx):
    g0, gp, ge = theta
    total = 0.0
    for lp, ev, c in zip(log_prior, evidence, chosen_idx):
        if c is None:
            continue
        u = g0 + gp * np.array(lp) + ge * np.array(ev)
        total += u[c] - logsumexp(u)
    penalty = 1e-3 * (gp ** 2 + ge ** 2)
    return -total + penalty


def fit_one_model(preds: pd.DataFrame, data: pd.DataFrame) -> dict:
    merged = preds.merge(data, on="case_id")
    cand_order = ["correct_diagnosis", "distractor1", "distractor2", "distractor3"]
    key_order = ["correct", "distractor1", "distractor2", "distractor3"]

    log_prior, evidence, chosen_idx = [], [], []
    for _, row in merged.iterrows():
        lp = [np.log(row[f"pubmed_count_{'correct' if k=='correct' else 'd'+k[-1]}"] + 1) for k in key_order]
        ev = [row[f"evidence_tfidf_{c}"] for c in cand_order]
        log_prior.append(lp)
        evidence.append(ev)
        chosen_idx.append(key_order.index(row["chosen_key"]) if row["chosen_key"] in key_order else None)

    n_used = sum(1 for c in chosen_idx if c is not None)
    res = minimize(nll_choice, x0=[0, 0, 0], args=(log_prior, evidence, chosen_idx), method="L-BFGS-B")
    g0, gp, ge = res.x
    per = gp / ge if abs(ge) > 1e-8 else np.inf
    return {"n_used": n_used, "n_total": len(merged), "gamma0": g0, "gamma_prior_cupcase": gp,
            "gamma_evidence_cupcase": ge, "PER_cupcase": per, "converged": bool(res.success),
            "top1_accuracy": merged["is_correct"].mean()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true",
                     help="analyze the full-scale 3562-case CUPCase results instead of the "
                          "250-case subset")
    args = ap.parse_args()

    data_name = "cupcase_full_with_evidence.csv" if args.full else "cupcase_with_evidence.csv"
    data = pd.read_csv(ROOT / "data" / "external" / data_name)
    per_master = pd.read_csv(ROOT / "results" / "theory" / "per_master_table.csv")

    ext_subdir = "external_validation_full" if args.full else "external_validation"
    pred_dir = ROOT / "results" / "theory" / ext_subdir / "cupcase"
    rows = []
    for f in sorted(pred_dir.glob("*.csv")):
        model_tag = f.stem
        preds = pd.read_csv(f)
        fit = fit_one_model(preds, data)
        gamma_row = per_master[(per_master.source == "open_weight") & (per_master.model == model_tag)
                                & (per_master.condition == "baseline")]
        row = {"model": model_tag, **fit}
        if len(gamma_row):
            g = gamma_row.iloc[0]
            row.update({"gamma_prior_diagrare": g["gamma_prior"], "gamma_evidence_diagrare": g["gamma_evidence"]})
        rows.append(row)
        print(f"{model_tag}: cupcase_acc={fit['top1_accuracy']:.1%} n_used={fit['n_used']}/{fit['n_total']} "
              f"gamma_prior_cupcase={fit['gamma_prior_cupcase']:.4f} gamma_evidence_cupcase={fit['gamma_evidence_cupcase']:.4f}")

    out = pd.DataFrame(rows)
    out_name = "cupcase_full_analysis.csv" if args.full else "cupcase_analysis.csv"
    out_path = ROOT / "results" / "theory" / ext_subdir / out_name
    out.to_csv(out_path, index=False)

    valid = out.dropna(subset=["gamma_evidence_cupcase", "gamma_evidence_diagrare"])
    if len(valid) > 2:
        r_matched, p_matched = pearsonr(valid.gamma_evidence_cupcase, valid.gamma_evidence_diagrare)
        rho_matched, _ = spearmanr(valid.gamma_evidence_cupcase, valid.gamma_evidence_diagrare)
        r_cross, p_cross = pearsonr(valid.gamma_prior_cupcase, valid.gamma_evidence_diagrare)
        print(f"\n=== CROSS-DATASET VALIDATION (DiagRare-X gamma_evidence vs. CUPCase gamma_evidence) ===")
        print(f"Pearson r={r_matched:.3f} (p={p_matched:.4f}), Spearman rho={rho_matched:.3f} across {len(valid)} models")
        print(f"(cross check: gamma_prior_cupcase vs gamma_evidence_diagrare: r={r_cross:.3f}, should be weaker)")

    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
