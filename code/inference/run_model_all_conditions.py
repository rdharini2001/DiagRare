#!/usr/bin/env python3
"""Load one model ONCE via vLLM and run every prompting condition against it,
to amortize model-load time across conditions (used by the SLURM array job,
one array task per model).

Usage:
  python run_model_all_conditions.py \
      --model Qwen/Qwen2.5-7B-Instruct --model_tag qwen2.5-7b \
      --vignettes ../data/expanded/vignettes_combined.csv \
      --ontology ../data/expanded/ontology_diseases_expanded.csv \
      --out_dir ../results/predictions \
      --conditions baseline,debias,cot,counterfactual_rare_clinic,counterfactual_primary_care
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prompts import CONDITIONS, MAX_TOKENS_BY_CONDITION
from run_inference_vllm import build_prompts, load_disease_list, parse_predictions


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--model_tag", required=True)
    ap.add_argument("--vignettes", required=True)
    ap.add_argument("--ontology", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--conditions", default=",".join(CONDITIONS.keys()))
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max_tokens", type=int, default=300)
    ap.add_argument("--max_model_len", type=int, default=4096)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    ap.add_argument("--lora_path", default=None)
    ap.add_argument("--use_chat_template", action="store_true",
                     help="wrap each prompt with the model's tokenizer chat template "
                          "(as a single user turn) before generation, instead of passing "
                          "raw text straight to vLLM. Off by default to keep the main "
                          "7-model/5-condition results as originally run; some models "
                          "(observed: BioMistral-7B) produce empty generations without it.")
    args = ap.parse_args()

    from vllm import LLM, SamplingParams

    vignettes = pd.read_csv(args.vignettes)
    ontology = pd.read_csv(args.ontology)
    disease_list = load_disease_list(ontology)

    llm = LLM(model=args.model, max_model_len=args.max_model_len,
              gpu_memory_utilization=args.gpu_memory_utilization,
              enable_lora=args.lora_path is not None, trust_remote_code=True)

    chat_tokenizer = None
    if args.use_chat_template:
        from transformers import AutoTokenizer
        chat_tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    lora_request = None
    if args.lora_path:
        from vllm.lora.request import LoRARequest
        lora_request = LoRARequest("debias_adapter", 1, args.lora_path)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for condition in args.conditions.split(","):
        condition = condition.strip()
        prompts = build_prompts(vignettes, disease_list, condition)
        if chat_tokenizer is not None:
            prompts = [chat_tokenizer.apply_chat_template(
                [{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
                for p in prompts]
        max_tokens = MAX_TOKENS_BY_CONDITION.get(condition, args.max_tokens)
        sampling = SamplingParams(temperature=args.temperature, max_tokens=max_tokens)
        t0 = time.time()
        outputs = llm.generate(prompts, sampling, lora_request=lora_request)
        elapsed = time.time() - t0

        rows = []
        n_unparsed = 0
        for vrow, out in zip(vignettes.to_dict("records"), outputs):
            (p1, p2, p3), parsed_cleanly = parse_predictions(out.outputs[0].text, disease_list)
            n_unparsed += int(not parsed_cleanly)
            rows.append({"vignette_id": vrow["vignette_id"], "prediction_1": p1,
                         "prediction_2": p2, "prediction_3": p3,
                         "parsed_cleanly": parsed_cleanly,
                         "raw_response": out.outputs[0].text[:2000]})

        out_path = out_dir / f"{args.model_tag}__{condition}.csv"
        pd.DataFrame(rows).to_csv(out_path, index=False)
        print(f"[{args.model_tag}] condition={condition} n={len(rows)} elapsed={elapsed:.1f}s "
              f"unparsed={n_unparsed}/{len(rows)} -> {out_path}")


if __name__ == "__main__":
    main()
