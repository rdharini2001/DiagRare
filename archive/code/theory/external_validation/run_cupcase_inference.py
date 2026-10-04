#!/usr/bin/env python3
"""Runs each model on the prepared CUPCase external-validation subset: a
4-way forced choice (correct diagnosis + 3 real hard-negative distractors,
all authored independently of DiagRare-Bench) per case. Candidates are shuffled
per case and presented as lettered options (A-D) since candidate texts can
be full sentences, making a letter-based answer far more reliably parseable
than asking the model to reproduce the exact text.

Usage:
  python run_cupcase_inference.py --model Qwen/Qwen2.5-7B-Instruct \
      --model_tag qwen2.5-7b --out ../../results/theory/external_validation/cupcase/qwen2.5-7b.csv
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]

PROMPT_TEMPLATE = """You are a diagnostic reasoning system. Read the clinical case below and choose the single most likely diagnosis from the options given.

CASE PRESENTATION:
{case}

OPTIONS:
{options}

Respond with ONLY the letter of the correct option (A, B, C, or D), nothing else."""


def build_prompt_and_key(row: pd.Series, rng: np.random.Generator) -> tuple[str, dict[str, str], str]:
    candidates = {
        "correct": row["correct_diagnosis"], "distractor1": row["distractor1"],
        "distractor2": row["distractor2"], "distractor3": row["distractor3"],
    }
    keys = list(candidates.keys())
    rng.shuffle(keys)
    letters = ["A", "B", "C", "D"]
    letter_to_key = dict(zip(letters, keys))
    options_text = "\n".join(f"{letter}. {candidates[key]}" for letter, key in letter_to_key.items())
    # Truncate long case reports (same pattern as run_rarearena_inference.py's 4000-char cap).
    # Never triggered at the original 250-case subset by sampling luck -- the full 3562-case
    # CUPCase set has a handful of outlier reports up to 17,810 chars (~5100+ tokens), which
    # overflowed max_model_len=4096 and crashed inference for several models.
    case_text = str(row["case_presentation"])[:4000]
    prompt = PROMPT_TEMPLATE.format(case=case_text, options=options_text)
    correct_letter = next(letter for letter, key in letter_to_key.items() if key == "correct")
    return prompt, letter_to_key, correct_letter


def parse_letter(text: str) -> str:
    m = re.search(r"\b([ABCD])\b", text.strip())
    return m.group(1) if m else "unparsed"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--model_tag", required=True)
    ap.add_argument("--data_csv", default=str(ROOT / "data" / "external" / "cupcase_with_evidence.csv"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    ap.add_argument("--max_model_len", type=int, default=3072)
    ap.add_argument("--seed", type=int, default=20260912)
    args = ap.parse_args()

    from vllm import LLM, SamplingParams

    df = pd.read_csv(args.data_csv)
    rng = np.random.default_rng(args.seed)

    prompts, letter_maps, correct_letters = [], [], []
    for _, row in df.iterrows():
        prompt, letter_to_key, correct_letter = build_prompt_and_key(row, rng)
        prompts.append(prompt)
        letter_maps.append(letter_to_key)
        correct_letters.append(correct_letter)

    llm = LLM(model=args.model, max_model_len=args.max_model_len,
              gpu_memory_utilization=args.gpu_memory_utilization, trust_remote_code=True)
    sp = SamplingParams(temperature=0, max_tokens=10)

    t0 = time.time()
    outs = llm.generate(prompts, sp)
    elapsed = time.time() - t0

    rows = []
    for (_, row), out, letter_map, correct_letter in zip(df.iterrows(), outs, letter_maps, correct_letters):
        text = out.outputs[0].text
        chosen_letter = parse_letter(text)
        chosen_key = letter_map.get(chosen_letter, "unparsed")
        rows.append({
            "case_id": row["case_id"], "chosen_key": chosen_key, "correct_letter": correct_letter,
            "chosen_letter": chosen_letter, "is_correct": chosen_key == "correct",
            "letter_map": str(letter_map), "raw_response": text[:100],
        })

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    out_df = pd.DataFrame(rows)
    n_unparsed = (out_df.chosen_key == "unparsed").sum()
    acc = out_df.is_correct.mean()
    print(f"model={args.model_tag} n={len(rows)} elapsed={elapsed:.1f}s unparsed={n_unparsed}/{len(rows)} "
          f"top1_accuracy={acc:.1%} -> {out_path}")


if __name__ == "__main__":
    main()
