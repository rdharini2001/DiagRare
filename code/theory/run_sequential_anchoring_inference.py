#!/usr/bin/env python3
"""Runs the sequential anchoring/recovery experiment
(build_sequential_anchoring_cases.py) against one model: independent
single-turn forced-choice generations for each (pair, order, step) row --
Step 2's prompt explicitly frames its findings as "initial presentation"
(the Step-1 findings) plus "follow-up workup" (the newly revealed
findings), so the model sees a plausible clinical narrative rather than an
undifferentiated finding dump, while both orders' Step 2 shows the exact
same total evidence.

Usage:
  python run_sequential_anchoring_inference.py --model Qwen/Qwen2.5-7B-Instruct \
      --model_tag qwen2.5-7b --out ../results/theory/sequential_anchoring/qwen2.5-7b.csv
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_causal_grid_inference import parse_binary_choice  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]

STEP1_TEMPLATE = """You are a diagnostic reasoning system. A patient presents with findings that could indicate {target} or {confounder}.

Population base rates in this clinical setting (stated explicitly -- use these numbers, not general knowledge): {target} occurs in approximately {target_prev} per 100,000 patients. {confounder} occurs in approximately {comp_prev} per 100,000 patients.

INITIAL PRESENTATION: {findings}

Which is the more likely diagnosis: {target} or {confounder}?
Respond with ONLY the exact disease name, nothing else."""

STEP2_TEMPLATE = """You are a diagnostic reasoning system, continuing to evaluate the same patient. A patient presents with findings that could indicate {target} or {confounder}.

Population base rates in this clinical setting (stated explicitly -- use these numbers, not general knowledge): {target} occurs in approximately {target_prev} per 100,000 patients. {confounder} occurs in approximately {comp_prev} per 100,000 patients.

INITIAL PRESENTATION: {initial_findings}

FOLLOW-UP WORKUP REVEALS ADDITIONAL FINDINGS: {new_findings}

Given the initial presentation AND the follow-up findings together, which is the more likely diagnosis: {target} or {confounder}?
Respond with ONLY the exact disease name, nothing else."""


def build_prompt(row: pd.Series) -> str:
    if row["step"] == 1:
        return STEP1_TEMPLATE.format(
            target=row["target_disease"], confounder=row["confounder_disease"],
            target_prev=row["target_prevalence_per_100k"], comp_prev=row["confounder_prevalence_per_100k"],
            findings=str(row["cumulative_findings"]).replace(";", ","))
    return STEP2_TEMPLATE.format(
        target=row["target_disease"], confounder=row["confounder_disease"],
        target_prev=row["target_prevalence_per_100k"], comp_prev=row["confounder_prevalence_per_100k"],
        initial_findings=str(row["initial_findings"]).replace(";", ","),
        new_findings=str(row["new_findings"]).replace(";", ","))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--model_tag", required=True)
    ap.add_argument("--data_csv", default=str(ROOT / "data" / "causal_grid" / "sequential_anchoring_cases.csv"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    ap.add_argument("--max_model_len", type=int, default=4096)
    args = ap.parse_args()

    from vllm import LLM, SamplingParams

    df = pd.read_csv(args.data_csv)
    prompts = [build_prompt(row) for _, row in df.iterrows()]

    llm = LLM(model=args.model, max_model_len=args.max_model_len,
              gpu_memory_utilization=args.gpu_memory_utilization, trust_remote_code=True)
    sp = SamplingParams(temperature=0, max_tokens=20)

    t0 = time.time()
    outs = llm.generate(prompts, sp)
    elapsed = time.time() - t0

    rows = []
    for (_, row), out in zip(df.iterrows(), outs):
        text = out.outputs[0].text
        choice = parse_binary_choice(text, row["target_disease"], row["confounder_disease"])
        rows.append({**row.to_dict(), "raw_response": text[:200], "choice": choice})

    out_df = pd.DataFrame(rows)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    n_unparsed = (out_df.choice == "unparsed").sum()
    print(f"model={args.model_tag} n={len(out_df)} elapsed={elapsed:.1f}s unparsed={n_unparsed}/{len(out_df)} -> {out_path}")


if __name__ == "__main__":
    main()
