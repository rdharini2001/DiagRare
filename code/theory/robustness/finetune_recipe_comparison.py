#!/usr/bin/env python3
"""A natural experiment the expanded panel makes possible for free: four
models in the panel are ALL fine-tunes of the identical Mistral-7B-v0.1/v0.3
base checkpoint, differing only in alignment recipe:
  - mistral-7b-instruct: Mistral AI's own instruction-tuning (SFT)
  - zephyr-7b: HuggingFace H4's DPO recipe (UltraFeedback)
  - openchat-3.5: OpenChat's C-RLFT recipe
  - nous-hermes-2-7b: Nous Research's SFT+DPO recipe

If gamma_evidence varies substantially across these four despite an
IDENTICAL base architecture and pretraining corpus, that is evidence that
evidence-elasticity is substantially shaped by the alignment/fine-tuning
recipe, not fixed at pretraining -- directly relevant to alignment
literature (RLHF/DPO/SFT effects on calibration and evidence-use) and a
genuinely novel angle no other part of this paper's analysis surfaces,
since the main panel varies base architecture AND recipe simultaneously
and so cannot separate the two.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]

MISTRAL_FAMILY_FINETUNES = {
    "mistral-7b-instruct": "Mistral AI (official SFT)",
    "zephyr-7b": "HuggingFace H4 (DPO, UltraFeedback)",
    "openchat-3.5": "OpenChat (C-RLFT)",
    "nous-hermes-2-7b": "Nous Research (SFT+DPO)",
}


def main() -> None:
    per_master = pd.read_csv(ROOT / "results" / "theory" / "per_master_table.csv")
    acc = pd.read_csv(ROOT / "results" / "open_weight_summary.csv")
    base = per_master[(per_master.source == "open_weight") & (per_master.condition == "baseline")]
    acc_base = acc[(acc.condition == "baseline") & (acc.split == "ALL")]

    rows = []
    for tag, recipe in MISTRAL_FAMILY_FINETUNES.items():
        ge_row = base[base.model == tag]
        acc_row = acc_base[acc_base.model == tag]
        if not len(ge_row):
            print(f"  {tag}: no data yet")
            continue
        g = ge_row.iloc[0]
        row = {"model": tag, "recipe": recipe, "gamma_prior": g["gamma_prior"],
               "gamma_evidence": g["gamma_evidence"], "PER": g["PER"]}
        if len(acc_row):
            row["top1_accuracy"] = acc_row.iloc[0]["top1_accuracy"]
        rows.append(row)

    if len(rows) < 2:
        print("Not enough of the four Mistral-family fine-tunes have completed yet -- rerun later.")
        return

    out = pd.DataFrame(rows).sort_values("gamma_evidence", ascending=False)
    out_path = ROOT / "results" / "theory" / "robustness" / "finetune_recipe_comparison.csv"
    out.to_csv(out_path, index=False)

    print("Same base architecture (Mistral-7B), four different alignment recipes:")
    print(out.to_string(index=False))
    spread = out["gamma_evidence"].max() - out["gamma_evidence"].min()
    print(f"\ngamma_evidence range across recipes (identical base model): "
          f"{out['gamma_evidence'].min():.3f} to {out['gamma_evidence'].max():.3f} (spread={spread:.3f})")
    print("If this spread is comparable to the spread seen across DIFFERENT base architectures "
          "in the main panel, that is evidence evidence-elasticity is substantially a property "
          "of the alignment recipe, not fixed by pretraining.")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
