#!/usr/bin/env python3
"""Runs the causal prior x evidence intervention grid (theory/causal_intervention_grid.py)
against one model: a clean 2-candidate forced choice per combination, with
explicit numeric prevalence stated in-prompt (decoupled from real
epidemiology) and evidence strength independently varied.

Usage:
  python run_causal_grid_inference.py --model Qwen/Qwen2.5-7B-Instruct \
      --model_tag qwen2.5-7b --out ../results/theory/causal_grid/qwen2.5-7b.csv
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
ROOT = Path(__file__).resolve().parents[2]

PROMPT_TEMPLATE = """You are a diagnostic reasoning system. A patient presents with findings that could indicate {target} or {confounder}.

Population base rates in this clinical setting (stated explicitly -- use these numbers, not general knowledge): {target} occurs in approximately {target_prev} per 100,000 patients. {confounder} occurs in approximately {comp_prev} per 100,000 patients.

POSITIVE FINDINGS (present): {positive}
NEGATIVE/ABSENT FINDINGS (explicitly absent): {negative}

Which is the more likely diagnosis: {target} or {confounder}?
Respond with ONLY the exact disease name, nothing else."""


def build_prompt(row: pd.Series) -> str:
    pos = str(row["positive_findings"]).replace(";", ", ") if pd.notna(row["positive_findings"]) else "none"
    neg = str(row["negative_findings"]).replace(";", ", ") if pd.notna(row["negative_findings"]) else "none"
    return PROMPT_TEMPLATE.format(
        target=row["target_disease"], confounder=row["confounder_disease"],
        target_prev=row["target_prevalence_per_100k"], comp_prev=row["confounder_prevalence_per_100k"],
        positive=pos, negative=neg)


def parse_binary_choice(text: str, target: str, confounder: str) -> str:
    """Three-tier parser, since verbose responses can defeat a naive
    last-mention heuristic (e.g. "this is X rather than Y" textually ends
    with Y but MEANS X):
      1. Exact match after stripping punctuation/whitespace (the compliant
         case -- the prompt asks for ONLY the disease name).
      2. Whichever name occurs MORE times (the model's actual answer is
         usually restated/emphasized more than the rejected alternative).
      3. Last-mention, as a final tiebreak.
    """
    stripped = re.sub(r"[^a-z0-9_ ]", "", text.strip().lower())
    target_norm = target.lower().replace("_", " ")
    comp_norm = confounder.lower().replace("_", " ")

    if stripped == target_norm or stripped == target.lower():
        return "target"
    if stripped == comp_norm or stripped == confounder.lower():
        return "confounder"

    text_norm = text.lower()
    # "A rather than B" / "A, not B" contrastive pattern: A is the answer regardless
    # of which name appears textually last.
    for name_a, name_b, label in [(target_norm, comp_norm, "target"), (comp_norm, target_norm, "confounder")]:
        if re.search(rf"{re.escape(name_a)}\s*,?\s*(rather than|not|instead of)\s+{re.escape(name_b)}", text_norm):
            return label
    t_count = text_norm.count(target_norm) + text_norm.count(target.lower())
    c_count = text_norm.count(comp_norm) + text_norm.count(confounder.lower())
    if t_count == 0 and c_count == 0:
        return "unparsed"
    if t_count != c_count:
        return "target" if t_count > c_count else "confounder"

    t_pos = text_norm.rfind(target_norm) if target_norm in text_norm else text_norm.rfind(target.lower())
    c_pos = text_norm.rfind(comp_norm) if comp_norm in text_norm else text_norm.rfind(confounder.lower())
    if t_pos > c_pos:
        return "target"
    if c_pos > t_pos:
        return "confounder"
    return "unparsed"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--model_tag", required=True)
    ap.add_argument("--grid_csv", default=str(ROOT / "data" / "causal_grid" / "causal_grid.csv"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    ap.add_argument("--max_model_len", type=int, default=2048)
    args = ap.parse_args()

    from vllm import LLM, SamplingParams

    grid = pd.read_csv(args.grid_csv)
    prompts = [build_prompt(row) for _, row in grid.iterrows()]

    llm = LLM(model=args.model, max_model_len=args.max_model_len,
              gpu_memory_utilization=args.gpu_memory_utilization, trust_remote_code=True)
    sp = SamplingParams(temperature=0, max_tokens=20)

    t0 = time.time()
    outs = llm.generate(prompts, sp)
    elapsed = time.time() - t0

    rows = []
    for (_, row), out in zip(grid.iterrows(), outs):
        text = out.outputs[0].text
        choice = parse_binary_choice(text, row["target_disease"], row["confounder_disease"])
        rows.append({**row.to_dict(), "raw_response": text[:300], "choice": choice})

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    n_unparsed = sum(1 for r in rows if r["choice"] == "unparsed")
    print(f"model={args.model_tag} n={len(rows)} elapsed={elapsed:.1f}s unparsed={n_unparsed}/{len(rows)} -> {out_path}")


if __name__ == "__main__":
    main()
