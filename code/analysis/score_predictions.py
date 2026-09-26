#!/usr/bin/env python3
"""Score one or more prediction CSVs (original schema: vignette_id,
prediction_1, prediction_2, prediction_3[, raw_response]) against the
DiagRare vignette/ontology definitions.

Prediction filenames are expected as "<model_tag>__<condition>.csv" so
model and condition can be recovered automatically; pass --model_tag/
--condition to override for a single file.

Computes, per (model, condition) and broken out by benchmark split
(original / expanded_indist / expanded_heldout):
  - top1_accuracy, top3_accuracy, rare_top1_accuracy
  - logical_violation_rate  (symbolic verifier: does prediction_1
    contradict the stated evidence?)
  - verified_top1_accuracy  (top-1 after symbolic reranking of the
    model's own top-3, no new information)

Usage:
  python score_predictions.py --pred_dir ../results/predictions \
      --vignettes ../data/expanded/vignettes_combined.csv \
      --ontology ../data/expanded/ontology_diseases_expanded.csv \
      --out ../results/open_weight_summary.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from symbolic_verifier import build_constraints, contradictions, rerank, split_tokens


def score_one(pred_path: Path, vignettes: pd.DataFrame, constraints: dict, model_tag: str, condition: str) -> dict:
    pred = pd.read_csv(pred_path)
    d = vignettes.merge(pred, on="vignette_id", how="inner", validate="one_to_one")
    if len(d) == 0:
        raise ValueError(f"No overlapping vignette_ids between {pred_path} and the vignette set")

    truth = d["target_disease"].astype(str).str.strip()
    p1 = d["prediction_1"].astype(str).str.strip()
    p2 = d["prediction_2"].astype(str).str.strip()
    p3 = d["prediction_3"].astype(str).str.strip()
    d["correct_top1"] = p1.eq(truth)
    d["correct_top3"] = p1.eq(truth) | p2.eq(truth) | p3.eq(truth)

    violations, verified_correct = [], []
    for _, row in d.iterrows():
        pos = split_tokens(row["positive_findings"])
        neg = split_tokens(row["negative_findings"])
        cands = [str(row["prediction_1"]).strip(), str(row["prediction_2"]).strip(), str(row["prediction_3"]).strip()]
        violations.append(len(contradictions(cands[0], pos, neg, constraints)) > 0)
        reranked = rerank(cands, pos, neg, constraints)
        verified_correct.append(reranked[0] == str(row["target_disease"]).strip())
    d["is_logical_violation"] = violations
    d["verified_top1_correct"] = verified_correct

    rows = []
    for split_name, sub in [("ALL", d)] + [(s, d[d["split"] == s]) for s in d["split"].dropna().unique()]:
        if len(sub) == 0:
            continue
        rare = sub["is_rare"].astype(bool)
        row = {
            "model": model_tag, "condition": condition, "split": split_name, "n": len(sub),
            "top1_accuracy": sub["correct_top1"].mean(),
            "top3_accuracy": sub["correct_top3"].mean(),
            "rare_top1_accuracy": sub.loc[rare, "correct_top1"].mean() if rare.any() else float("nan"),
            "logical_violation_rate": sub["is_logical_violation"].mean(),
            "verified_top1_accuracy": sub["verified_top1_correct"].mean(),
        }
        if "parsed_cleanly" in sub.columns:
            row["unparsed_rate"] = 1.0 - sub["parsed_cleanly"].astype(bool).mean()
        rows.append(row)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_dir", required=True)
    ap.add_argument("--vignettes", required=True)
    ap.add_argument("--ontology", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    vignettes = pd.read_csv(args.vignettes)
    ontology = pd.read_csv(args.ontology)
    constraints = build_constraints(ontology)

    all_rows = []
    pred_files = sorted(Path(args.pred_dir).glob("*.csv"))
    if not pred_files:
        raise SystemExit(f"No prediction CSVs found in {args.pred_dir}")

    for f in pred_files:
        stem = f.stem
        if "__" in stem:
            model_tag, condition = stem.split("__", 1)
        else:
            model_tag, condition = stem, "unknown"
        try:
            all_rows.extend(score_one(f, vignettes, constraints, model_tag, condition))
        except Exception as e:
            print(f"WARN: failed to score {f}: {e}", file=sys.stderr)

    summary = pd.DataFrame(all_rows).sort_values(["model", "condition", "split"])
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_path, index=False)

    print(summary.to_string(index=False))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
