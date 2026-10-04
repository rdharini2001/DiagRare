#!/usr/bin/env python3
"""Does gamma_evidence add information beyond cheap, off-the-shelf ranking
statistics one could compute without any discrete-choice machinery at all?
For each model (baseline condition), computes:
  - MRR: mean reciprocal rank of the true disease in the model's top-3
    (1/1, 1/2, 1/3, or 0 if absent).
  - NDCG@3 against the oracle's own ranking (treats the oracle's
    bayes_log_posterior-ranked list as the "ideal" order, scores the model's
    top-3 against it).
  - prevalence_overrule_rate: fraction of vignettes where the model's Top-1
    is NOT the highest-prevalence candidate among ones it could plausibly
    have meant (i.e. the model picks something other than the "obvious"
    common answer) -- a crude, one-number version of "does this model ever
    override its prior".
  - evidence_margin: mean(evidence_score(model's Top-1) - evidence_score(the
    globally most-evidence-supported candidate)) -- how far the model's pick
    is, in oracle evidence terms, from the best-supported option.

Then correlates each of these against gamma_evidence AND against raw
accuracy, to show gamma_evidence is not just a relabeling of any one of them.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "analysis"))
from symbolic_verifier import split_tokens

ROOT = Path(__file__).resolve().parents[3]


def compute_simple_stats(preds: pd.DataFrame, vignettes: pd.DataFrame, oracle: pd.DataFrame) -> dict:
    d = vignettes.merge(preds, on="vignette_id", how="inner")
    oracle_by_vid = {vid: grp.set_index("disease") for vid, grp in oracle.groupby("vignette_id")}
    ontology_prev = oracle.drop_duplicates("disease").set_index("disease")["log_prevalence"]

    reciprocal_ranks, ndcgs, overrule_flags, margins = [], [], [], []
    for _, row in d.iterrows():
        truth = str(row["target_disease"]).strip()
        cands = [str(row["prediction_1"]).strip(), str(row["prediction_2"]).strip(), str(row["prediction_3"]).strip()]
        rr = 0.0
        for rank, c in enumerate(cands, start=1):
            if c == truth:
                rr = 1.0 / rank
                break
        reciprocal_ranks.append(rr)

        vid = row["vignette_id"]
        og = oracle_by_vid.get(vid)
        if og is None:
            continue
        ideal_order = og["bayes_log_posterior"].sort_values(ascending=False).index.tolist()
        ideal_rank = {dsx: i for i, dsx in enumerate(ideal_order)}
        dcg = sum((1.0 / np.log2(i + 2)) * (1.0 / (ideal_rank.get(c, 1000) + 1)) for i, c in enumerate(cands))
        idcg = sum(1.0 / np.log2(i + 2) for i in range(3))
        ndcgs.append(dcg / idcg if idcg > 0 else 0.0)

        valid_prevs = [(c, ontology_prev.get(c, -np.inf)) for c in cands if c in ontology_prev.index]
        if valid_prevs:
            most_common = max(valid_prevs, key=lambda x: x[1])[0]
            overrule_flags.append(cands[0] != most_common)

        if og is not None and cands[0] in og.index:
            top1_evidence = og.loc[cands[0], "evidence_loglik_ratio"]
            best_evidence = og["evidence_loglik_ratio"].max()
            margins.append(top1_evidence - best_evidence)

    return {
        "MRR": np.mean(reciprocal_ranks), "NDCG@3": np.mean(ndcgs) if ndcgs else np.nan,
        "prevalence_overrule_rate": np.mean(overrule_flags) if overrule_flags else np.nan,
        "evidence_margin": np.mean(margins) if margins else np.nan,
    }


def main() -> None:
    vignettes = pd.read_csv(ROOT / "data" / "expanded" / "vignettes_combined.csv")
    oracle = pd.read_csv(ROOT / "results" / "theory" / "bayes_oracle_table_expanded.csv")
    per_master = pd.read_csv(ROOT / "results" / "theory" / "per_master_table.csv")
    acc_summary = pd.read_csv(ROOT / "results" / "open_weight_summary.csv")

    models = ["qwen2.5-0.5b", "qwen2.5-1.5b", "qwen2.5-3b", "qwen2.5-7b",
              "yi-1.5-6b", "olmo-2-7b", "mistral-7b-instruct", "phi-3.5-mini"]
    rows = []
    for model_tag in models:
        preds = pd.read_csv(ROOT / "results" / "predictions" / f"{model_tag}__baseline.csv")
        stats = compute_simple_stats(preds, vignettes, oracle)
        ge_row = per_master[(per_master.source == "open_weight") & (per_master.model == model_tag)
                             & (per_master.condition == "baseline")]
        acc_row = acc_summary[(acc_summary.model == model_tag) & (acc_summary.condition == "baseline")
                               & (acc_summary.split == "ALL")]
        row = {"model": model_tag, **stats}
        if len(ge_row):
            row["gamma_evidence"] = ge_row.iloc[0]["gamma_evidence"]
        if len(acc_row):
            row["top1_accuracy"] = acc_row.iloc[0]["top1_accuracy"]
        rows.append(row)
        print(f"{model_tag}: MRR={stats['MRR']:.3f} NDCG@3={stats['NDCG@3']:.3f} "
              f"overrule_rate={stats['prevalence_overrule_rate']:.3f} margin={stats['evidence_margin']:.3f}")

    out = pd.DataFrame(rows)
    out_path = ROOT / "results" / "theory" / "robustness" / "simple_stats_comparison.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    print("\nSpearman correlation of each simple statistic with gamma_evidence and with top1_accuracy:")
    for col in ["MRR", "NDCG@3", "prevalence_overrule_rate", "evidence_margin"]:
        valid = out.dropna(subset=[col, "gamma_evidence", "top1_accuracy"])
        rho_ge, p_ge = spearmanr(valid[col], valid["gamma_evidence"])
        rho_acc, p_acc = spearmanr(valid[col], valid["top1_accuracy"])
        print(f"  {col:28s} vs gamma_evidence: rho={rho_ge:.3f} (p={p_ge:.3f})  "
              f"vs top1_accuracy: rho={rho_acc:.3f} (p={p_acc:.3f})")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
