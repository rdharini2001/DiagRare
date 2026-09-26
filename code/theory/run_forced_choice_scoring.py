#!/usr/bin/env python3
"""Supplementary validation experiment: directly score each model's log
P(candidate disease | vignette) via teacher-forced continuation scoring
(no free-text generation/parsing at all), for a small clinically-realistic
differential per vignette (theory/candidate_sets.py). This sidesteps every
parsing issue in the main study and gives real model-internal probabilities
to validate the Plackett-Luce revealed-preference PER estimates against.

Usage:
  python run_forced_choice_scoring.py --model Qwen/Qwen2.5-7B-Instruct \
      --model_tag qwen2.5-7b --condition baseline \
      --out ../results/theory/forced_choice/qwen2.5-7b__baseline.csv
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "inference"))
from candidate_sets import build_candidate_sets
from prompts import CONDITIONS

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--model_tag", required=True)
    ap.add_argument("--condition", default="baseline", choices=list(CONDITIONS.keys()))
    ap.add_argument("--vignettes", default=str(ROOT / "data" / "expanded" / "vignettes_combined.csv"))
    ap.add_argument("--ontology", default=str(ROOT / "data" / "expanded" / "ontology_diseases_expanded.csv"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    ap.add_argument("--max_model_len", type=int, default=4096)
    ap.add_argument("--chunk", type=int, default=None,
                     help="override the auto vocab-scaled CHUNK size (see comment below)")
    args = ap.parse_args()

    from vllm import LLM, SamplingParams

    vignettes = pd.read_csv(args.vignettes)
    ontology = pd.read_csv(args.ontology)
    disease_list = sorted(ontology["disease"].astype(str).tolist())
    disease_str = ", ".join(disease_list)
    candidate_sets = build_candidate_sets(vignettes, ontology)

    prompt_fn = CONDITIONS[args.condition]
    ANSWER_PREFIX = "\nThe single most likely diagnosis is: "

    # Build one (vignette, candidate) scoring request per pair.
    requests = []  # (vignette_id, candidate, prompt_text, completion_text)
    for _, v in vignettes.iterrows():
        pos = str(v["positive_findings"]).replace(";", ", ") if pd.notna(v["positive_findings"]) else ""
        neg = (str(v["negative_findings"]).replace(";", ", ") if pd.notna(v["negative_findings"]) else "") or "none"
        base_prompt = prompt_fn(disease_str, pos, neg) + ANSWER_PREFIX
        for cand in candidate_sets[v["vignette_id"]]:
            requests.append((v["vignette_id"], cand, base_prompt, cand))

    # Prefix caching is disabled because the tested vLLM/XFORMERS configuration
    # was unstable on the evaluation GPUs. Eager execution avoids CUDA-graph
    # memory growth for large-vocabulary checkpoints, and max_num_seqs bounds
    # concurrent prompt-logprob requests.
    llm = LLM(model=args.model, max_model_len=args.max_model_len,
              gpu_memory_utilization=args.gpu_memory_utilization, enable_prefix_caching=False,
              max_num_seqs=32, trust_remote_code=True, enforce_eager=True)
    tok = llm.get_tokenizer()
    sp = SamplingParams(temperature=0, max_tokens=1, prompt_logprobs=0)

    full_texts = [p + c for _, _, p, c in requests]
    prompt_ntoks_cache: dict[str, int] = {}

    # Prompt-logprob scoring materializes vocabulary-sized tensors for in-flight
    # requests. Scale the request chunk size inversely with vocabulary size to
    # keep peak memory comparable across model families.
    vocab_size = len(tok)
    CHUNK = args.chunk if args.chunk is not None else max(8, int(60 * 32000 / vocab_size))
    print(f"vocab_size={vocab_size} -> CHUNK={CHUNK}", flush=True)
    import torch
    rows = []
    t0 = time.time()
    for start in range(0, len(full_texts), CHUNK):
        chunk_texts = full_texts[start:start + CHUNK]
        chunk_meta = requests[start:start + CHUNK]
        outs = llm.generate(chunk_texts, sp, use_tqdm=False)
        for (vid, cand, prompt_text, completion_text), out in zip(chunk_meta, outs):
            if prompt_text not in prompt_ntoks_cache:
                prompt_ntoks_cache[prompt_text] = len(tok(prompt_text)["input_ids"])
            prompt_ntoks = prompt_ntoks_cache[prompt_text]
            comp_lps = out.prompt_logprobs[prompt_ntoks:]
            total_lp = sum(list(d.values())[0].logprob for d in comp_lps if d is not None)
            rows.append({"vignette_id": vid, "candidate": cand, "n_completion_tokens": len(comp_lps),
                         "log_prob": total_lp})
        # Empirically, large-vocab models (Qwen's 152k) OOM partway through
        # the full request list even with a small per-chunk CHUNK size --
        # peak memory grows cumulatively across chunks (allocator
        # fragmentation from repeatedly allocating/freeing large
        # vocab-sized tensors), not just within one chunk. Force a cache
        # release between chunks to keep peak bounded independent of how
        # many chunks have already run.
        torch.cuda.empty_cache()
        if (start // CHUNK) % 4 == 0:
            print(f"  ...{start + len(chunk_texts)}/{len(full_texts)} scored "
                  f"({time.time()-t0:.0f}s elapsed)", flush=True)
    elapsed = time.time() - t0

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"model={args.model_tag} condition={args.condition} n_requests={len(requests)} "
          f"elapsed={elapsed:.1f}s -> {out_path}")


if __name__ == "__main__":
    main()
