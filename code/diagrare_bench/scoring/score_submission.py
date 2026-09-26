#!/usr/bin/env python3
"""Score predictions for the public DiagRare-Bench tasks.

The primary benchmark is included in this repository. CUPCase and RareArena
remain external datasets, so their reference CSV must be supplied explicitly.

Primary prediction format:
  vignette_id,prediction_1,prediction_2,prediction_3

External prediction format:
  case_id,chosen_key
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]

def score_primary(pred_path: Path) -> dict:
    ref = pd.read_csv(REPO_ROOT / "data" / "expanded" / "vignettes_combined.csv")
    pred = pd.read_csv(pred_path)
    m = ref.merge(pred, on="vignette_id", how="inner")
    target = m["target_disease"].astype(str).str.strip()
    p1 = m["prediction_1"].astype(str).str.strip()
    p2 = m["prediction_2"].astype(str).str.strip()
    p3 = m["prediction_3"].astype(str).str.strip()
    top1 = p1.eq(target)
    top3 = top1 | p2.eq(target) | p3.eq(target)
    common = top1[~m["is_rare"].astype(bool)].mean()
    rare = top1[m["is_rare"].astype(bool)].mean()
    discordance = (common - rare) / (common + rare + 1e-6)
    return {"n": len(m), "top1_accuracy": top1.mean(), "top3_accuracy": top3.mean(),
            "common_accuracy": common, "rare_accuracy": rare, "discordance_index": discordance}

def score_external(pred_path: Path, reference_path: Path) -> dict:
    ref = pd.read_csv(reference_path)
    pred = pd.read_csv(pred_path)
    m = ref.merge(pred, on="case_id", how="inner")
    correct = m["chosen_key"].astype(str).eq("correct")
    out = {"n": len(m), "top1_accuracy": correct.mean()}
    if "pubmed_count_correct" in m.columns:
        logp = np.log(m["pubmed_count_correct"] + 1)
        q1, q2 = logp.quantile([1/3, 2/3])
        rare = correct[logp <= q1].mean()
        common = correct[logp > q2].mean()
        out.update({"common_accuracy": common, "rare_accuracy": rare,
                    "discordance_index": (common-rare)/(common+rare+1e-6)})
    return out

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["primary", "cupcase", "rarearena"])
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--reference", default=None,
                    help="Reference CSV for CUPCase/RareArena; not needed for primary")
    args = ap.parse_args()
    if args.dataset == "primary":
        result = score_primary(Path(args.predictions))
    else:
        if not args.reference:
            raise SystemExit("--reference is required for external datasets")
        result = score_external(Path(args.predictions), Path(args.reference))
    for k,v in result.items():
        print(f"{k}: {v:.4f}" if isinstance(v,float) else f"{k}: {v}")

if __name__ == "__main__":
    main()
