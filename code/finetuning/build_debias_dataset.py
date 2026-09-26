#!/usr/bin/env python3
"""Build a small supervised debiasing fine-tune set.

Training source: ONLY the 6 "expanded_indist" diseases' vignettes (72
vignettes total, all newly generated, disjoint from the original 696
and from the 3 held-out organ systems). Each example pairs the baseline
prompt with the CORRECT top-3 ranking derived directly from the ontology
(ground truth, not a model's guess), so the model is taught to follow
evidence-based ranking rather than population-prevalence ranking. The
correct top-3 is: the true disease first, then the two other diseases
(from the full ontology) with the highest symbolic support score against
the stated evidence -- i.e. supervision comes from the SAME logical
constraints the verifier uses at test time, not from a larger teacher LLM.

Evaluation of this fine-tune must ONLY use:
  - the original 696-vignette benchmark (unseen at fine-tune time), and
  - the expanded_heldout split (3 new organ systems, unseen diseases),
so any accuracy gain reflects real generalization, not memorization.

Usage: python build_debias_dataset.py --out ../data/expanded/debias_sft.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "inference"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
from prompts import baseline as baseline_prompt  # noqa: E402
from symbolic_verifier import build_constraints, split_tokens, support_score  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vignettes", default=str(ROOT / "data" / "expanded" / "vignettes_combined.csv"))
    ap.add_argument("--ontology", default=str(ROOT / "data" / "expanded" / "ontology_diseases_expanded.csv"))
    ap.add_argument("--out", default=str(ROOT / "data" / "expanded" / "debias_sft.jsonl"))
    args = ap.parse_args()

    vignettes = pd.read_csv(args.vignettes)
    ontology = pd.read_csv(args.ontology)
    disease_list = sorted(ontology["disease"].astype(str).tolist())
    disease_str = ", ".join(disease_list)
    constraints = build_constraints(ontology)

    train = vignettes[vignettes["split"] == "expanded_indist"].reset_index(drop=True)
    if len(train) == 0:
        raise SystemExit("No expanded_indist vignettes found -- run generate_vignettes.py first")

    examples = []
    for _, row in train.iterrows():
        pos = split_tokens(row["positive_findings"])
        neg = split_tokens(row["negative_findings"])
        truth = str(row["target_disease"])

        others = [d for d in disease_list if d != truth]
        others_scored = sorted(others, key=lambda d: -support_score(d, pos, neg, constraints))
        top3 = [truth] + others_scored[:2]

        prompt = baseline_prompt(
            disease_str,
            str(row["positive_findings"]).replace(";", ", "),
            (str(row["negative_findings"]).replace(";", ", ") or "none"),
        )
        completion = "1. [{}]\n2. [{}]\n3. [{}]".format(*top3)
        examples.append({"prompt": prompt, "completion": completion, "vignette_id": int(row["vignette_id"])})

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")

    print(f"Wrote {len(examples)} debiasing SFT examples to {out_path}")
    print("(source: expanded_indist split only -- original 696 vignettes and expanded_heldout "
          "organ systems are NEVER included, so eval on those measures true generalization)")


if __name__ == "__main__":
    main()
