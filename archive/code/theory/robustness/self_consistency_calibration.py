#!/usr/bin/env python3
"""Evaluate self-consistency agreement as a confidence signal.

The analysis compares agreement among repeated samples with empirical diagnostic
accuracy for the Qwen2.5 checkpoints.
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "inference"))
sys.path.insert(0, str(ROOT / "analysis"))
from run_inference_vllm import parse_predictions  # noqa: E402
from plackett_luce import build_choices_from_predictions, fit_plackett_luce  # noqa: E402


def agreement_rate_for_row(raw_response: str, disease_list: list[str], majority_p1: str) -> tuple[float, int]:
    segments = [s for s in raw_response.split(" ||| ") if s.strip()]
    votes = []
    for seg in segments:
        (p1, _, _), _ = parse_predictions(seg, disease_list)
        if p1:
            votes.append(p1)
    if not votes:
        return np.nan, 0
    counts = Counter(votes)
    # agreement with the model's OWN majority-vote prediction_1 (already stored)
    agree = counts.get(majority_p1, 0) / len(votes)
    return agree, len(votes)


def calibration_table(df: pd.DataFrame, n_bins: int = 5) -> pd.DataFrame:
    d = df.dropna(subset=["agreement_rate"]).copy()
    d["bin"] = pd.qcut(d["agreement_rate"], n_bins, duplicates="drop")
    g = d.groupby("bin", observed=True).agg(
        mean_agreement=("agreement_rate", "mean"),
        empirical_accuracy=("is_correct", "mean"),
        n=("is_correct", "size"),
    ).reset_index()
    g["abs_gap"] = (g["mean_agreement"] - g["empirical_accuracy"]).abs()
    return g


def ece(g: pd.DataFrame) -> float:
    w = g["n"] / g["n"].sum()
    return float((w * g["abs_gap"]).sum())


def gamma_evidence_by_confidence_stratum(preds: pd.DataFrame, oracle: pd.DataFrame, agreement: pd.Series) -> dict:
    median = agreement.median()
    high_vids = set(agreement[agreement >= median].index)
    low_vids = set(agreement[agreement < median].index)

    out = {}
    for label, vid_set in [("high_confidence", high_vids), ("low_confidence", low_vids)]:
        sub = preds[preds.vignette_id.isin(vid_set)]
        choices, _, n_used = build_choices_from_predictions(sub, oracle)
        fit = fit_plackett_luce(choices)
        out[f"gamma_evidence_{label}"] = fit["gamma_evidence"]
        out[f"gamma_prior_{label}"] = fit["gamma_prior"]
        out[f"n_{label}"] = n_used
    return out


def main() -> None:
    ontology = pd.read_csv(ROOT / "data" / "expanded" / "ontology_diseases_expanded.csv")
    disease_list = sorted(ontology["disease"].astype(str).tolist())
    vignettes = pd.read_csv(ROOT / "data" / "expanded" / "vignettes_combined.csv")
    oracle = pd.read_csv(ROOT / "results" / "theory" / "bayes_oracle_table_expanded.csv")

    models = ["qwen2.5-0.5b", "qwen2.5-3b", "qwen2.5-7b"]
    cal_rows, dist_rows = [], []
    all_binned = []

    for m in models:
        f = ROOT / "results" / "predictions_selfconsistency" / f"{m}-selfcons5__baseline.csv"
        preds = pd.read_csv(f)
        merged = preds.merge(vignettes[["vignette_id", "target_disease"]], on="vignette_id", how="inner")
        merged["is_correct"] = merged["prediction_1"] == merged["target_disease"]

        agree_list, nvotes_list = [], []
        for _, row in merged.iterrows():
            a, nv = agreement_rate_for_row(row["raw_response"], disease_list, row["prediction_1"])
            agree_list.append(a)
            nvotes_list.append(nv)
        merged["agreement_rate"] = agree_list
        merged["n_votes_recovered"] = nvotes_list

        g = calibration_table(merged)
        g["model"] = m
        all_binned.append(g)
        e = ece(g)
        overall_acc = merged["is_correct"].mean()
        overall_agree = merged["agreement_rate"].mean()
        cal_rows.append({"model": m, "ECE": e, "mean_agreement": overall_agree,
                          "top1_accuracy": overall_acc, "n": len(merged),
                          "median_votes_recovered": float(np.median(nvotes_list))})
        print(f"{m}: mean_agreement={overall_agree:.3f} top1_acc={overall_acc:.3f} ECE={e:.4f} "
              f"(median votes recovered/5 = {np.median(nvotes_list):.0f})")

        agreement_by_vid = merged.set_index("vignette_id")["agreement_rate"].dropna()
        strat = gamma_evidence_by_confidence_stratum(preds, oracle, agreement_by_vid)
        strat["model"] = m
        dist_rows.append(strat)
        print(f"    gamma_evidence: high_confidence={strat['gamma_evidence_high_confidence']:.4f} "
              f"low_confidence={strat['gamma_evidence_low_confidence']:.4f}")

    out_dir = ROOT / "results" / "theory" / "robustness"
    out_dir.mkdir(parents=True, exist_ok=True)

    cal_df = pd.DataFrame(cal_rows)
    cal_df.to_csv(out_dir / "self_consistency_calibration_summary.csv", index=False)

    binned_df = pd.concat(all_binned, ignore_index=True)
    binned_df.to_csv(out_dir / "self_consistency_calibration_bins.csv", index=False)

    dist_df = pd.DataFrame(dist_rows)
    dist_df.to_csv(out_dir / "self_consistency_gamma_by_confidence.csv", index=False)

    print("\n=== Summary: is gamma_evidence just relabeled confidence? ===")
    for _, r in dist_df.iterrows():
        ratio = r["gamma_evidence_high_confidence"] / r["gamma_evidence_low_confidence"] if r["gamma_evidence_low_confidence"] else np.nan
        print(f"  {r['model']}: gamma_evidence high/low-confidence ratio = {ratio:.2f} "
              f"(1.0 = fully confidence-independent; large deviation = confounded with confidence)")

    print(f"\nWrote {out_dir / 'self_consistency_calibration_summary.csv'}, "
          f"{out_dir / 'self_consistency_calibration_bins.csv'}, "
          f"{out_dir / 'self_consistency_gamma_by_confidence.csv'}")


if __name__ == "__main__":
    main()
