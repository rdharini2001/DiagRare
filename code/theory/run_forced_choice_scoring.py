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

    # enable_prefix_caching=True hangs indefinitely on this cluster's GPUs: it
    # forces vLLM's V1 engine, which is incompatible with the XFORMERS backend
    # this project requires for Turing-generation GPUs (confirmed via a
    # dedicated-node test -- V1+XFORMERS logs "falling back to V0" and then
    # hangs). Without it, each candidate's shared prompt prefix is simply
    # recomputed per request instead of cached across the ~11 candidates per
    # vignette -- slower, but small enough here (~11k short requests total)
    # to be a non-issue.
    # max_num_seqs caps how many sequences vLLM schedules CONCURRENTLY,
    # independent of how many requests are handed to one generate() call --
    # the CHUNK-based batching below does NOT bound this. qwen2.5-7b OOM'd
    # identically at CHUNK=500 and CHUNK=150 (same byte-for-byte allocation
    # failure), which only makes sense if vLLM's scheduler was packing far
    # more concurrent sequences than either chunk size in each case; each
    # sequence with prompt_logprobs=0 buffers a logprob record per PROMPT
    # token (~500-700 tokens here), so concurrency, not request count, is
    # what blew up memory.
    # enforce_eager=True: CUDA-graph capture (35 shapes x LM-head-sized logit
    # buffers) turned out to be the actual memory hog for large-vocab models
    # (Qwen's 152k vocab vs Mistral's 32k) -- vLLM's own profiling report
    # only showed ~0.07-0.16GiB for "CUDAGraph memory," but the real captured
    # graphs pin far more than that for a 152k-wide LM head, which is
    # invisible to the profiler step but consistently OOM'd the qwen family
    # (not mistral/phi) at the identical ~2.13GiB allocation regardless of
    # CHUNK size -- eager mode skips graph capture entirely.
    llm = LLM(model=args.model, max_model_len=args.max_model_len,
              gpu_memory_utilization=args.gpu_memory_utilization, enable_prefix_caching=False,
              max_num_seqs=32, trust_remote_code=True, enforce_eager=True)
    tok = llm.get_tokenizer()
    sp = SamplingParams(temperature=0, max_tokens=1, prompt_logprobs=0)

    full_texts = [p + c for _, _, p, c in requests]
    prompt_ntoks_cache: dict[str, int] = {}

    # Requesting prompt_logprobs=0 makes vLLM buffer a logprob object for EVERY
    # prompt token of EVERY in-flight request -- with ~11k requests submitted
    # at once this OOM'd even a 24GB GPU. Chunk the requests so peak memory is
    # bounded regardless of total count or which GPU this lands on.
    # CHUNK=150 was tuned on a 44GB L40S with Mistral's ~32k vocab. RTX6000
    # has only 23.46GB, AND the real driver of the OOM turned out to be
    # VOCAB SIZE, not GPU size: get_logprobs materializes a
    # (chunk_tokens x vocab_size) tensor internally regardless of how few
    # logprobs are actually requested per token (prompt_logprobs=0). At
    # gpu_memory_utilization=0.65/CHUNK=60, mistral-7b-instruct (32k vocab)
    # and phi-3.5-mini succeed, but qwen2.5-1.5b/3b/7b (152k vocab, ~4.7x
    # larger) OOM'd identically regardless of CHUNK/max_num_seqs tuning --
    # confirming the failure scales with vocab_size, not batch size alone.
    # Scale CHUNK down by vocab size (calibrated against the 60-worked-at-32k
    # data point) so every model family gets a comparable, safe memory
    # footprint instead of the same fixed constant.
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
