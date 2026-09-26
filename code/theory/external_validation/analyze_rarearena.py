#!/usr/bin/env python3
"""Same discrete-choice decomposition as analyze_cupcase.py, applied to
RareArena. Also cross-correlates gamma_evidence_rarearena against
gamma_evidence_cupcase (both externally-sourced, independently-distractored)
in addition to gamma_evidence_diagrare -- three-way agreement across a
synthetic benchmark and two independently authored real-case datasets with
DIFFERENT distractor-generation processes (curated hard-negatives vs. random
sampling) would be a considerably stronger external-validity claim than any
one of the three alone.
"""
from __future__ import annotations

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
    return {"n_used": n_used, "n_total": len(merged), "gamma0": g0, "gamma_prior_rarearena": gp,
            "gamma_evidence_rarearena": ge, "PER_rarearena": per, "converged": bool(res.success),
            "top1_accuracy": merged["is_correct"].mean()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true",
                     help="analyze the full-scale 22901-case RareArena results instead of the "
                          "250-case subset")
    args = ap.parse_args()

    data_name = "rarearena_full_with_evidence.csv" if args.full else "rarearena_with_evidence.csv"
    data = pd.read_csv(ROOT / "data" / "external" / data_name)
    per_master = pd.read_csv(ROOT / "results" / "theory" / "per_master_table.csv")

    ext_subdir = "external_validation_full" if args.full else "external_validation"
    cupcase_name = "cupcase_full_analysis.csv" if args.full else "cupcase_analysis.csv"
    cupcase_path = ROOT / "results" / "theory" / ext_subdir / cupcase_name
    cupcase = pd.read_csv(cupcase_path) if cupcase_path.exists() else None

    pred_dir = ROOT / "results" / "theory" / ext_subdir / "rarearena"
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
        if cupcase is not None:
            crow = cupcase[cupcase.model == model_tag]
            if len(crow):
                row["gamma_evidence_cupcase"] = crow.iloc[0]["gamma_evidence_cupcase"]
        rows.append(row)
        print(f"{model_tag}: rarearena_acc={fit['top1_accuracy']:.1%} n_used={fit['n_used']}/{fit['n_total']} "
              f"gamma_prior_rarearena={fit['gamma_prior_rarearena']:.4f} "
              f"gamma_evidence_rarearena={fit['gamma_evidence_rarearena']:.4f}")

    out = pd.DataFrame(rows)
    out_name = "rarearena_full_analysis.csv" if args.full else "rarearena_analysis.csv"
    out_path = ROOT / "results" / "theory" / ext_subdir / out_name
    out.to_csv(out_path, index=False)

    valid = out.dropna(subset=["gamma_evidence_rarearena", "gamma_evidence_diagrare"])
    if len(valid) > 2:
        r, p = pearsonr(valid.gamma_evidence_rarearena, valid.gamma_evidence_diagrare)
        rho, _ = spearmanr(valid.gamma_evidence_rarearena, valid.gamma_evidence_diagrare)
        print(f"\n=== RareArena vs. DiagRare-X: r={r:.3f} (p={p:.4f}), rho={rho:.3f} across {len(valid)} models ===")

    if "gamma_evidence_cupcase" in out.columns:
        valid3 = out.dropna(subset=["gamma_evidence_rarearena", "gamma_evidence_cupcase"])
        if len(valid3) > 2:
            r2, p2 = pearsonr(valid3.gamma_evidence_rarearena, valid3.gamma_evidence_cupcase)
            rho2, _ = spearmanr(valid3.gamma_evidence_rarearena, valid3.gamma_evidence_cupcase)
            print(f"=== RareArena vs. CUPCase (two independent external datasets, different "
                  f"distractor processes): r={r2:.3f} (p={p2:.4f}), rho={rho2:.3f} across {len(valid3)} models ===")

    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
