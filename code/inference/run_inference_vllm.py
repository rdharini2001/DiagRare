#!/usr/bin/env python3
"""Run one (model, prompting-condition) combination over the DiagRare
vignette set using vLLM, and save predictions in the same schema as the
primary evaluation's model_predictions/*.csv files.

Usage:
  python run_inference_vllm.py \
      --model Qwen/Qwen2.5-7B-Instruct \
      --model_tag qwen2.5-7b \
      --condition baseline \
      --vignettes ../data/expanded/vignettes_combined.csv \
      --ontology ../data/expanded/ontology_diseases_expanded.csv \
      --out ../results/predictions/qwen2.5-7b__baseline.csv \
      --n_samples 1 --temperature 0.0
"""
from __future__ import annotations

import argparse
import difflib
import re
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prompts import CONDITIONS, MAX_TOKENS_BY_CONDITION


def load_disease_list(ontology_df: pd.DataFrame) -> list[str]:
    return sorted(ontology_df["disease"].astype(str).tolist())


_ITEM = r"\[?([A-Za-z0-9_\-\/\s,]+?)\]?(?:\s*\([^\n]*?\))?\s*"  # tolerates a trailing "(...)" aside
# Item "1." need not start a fresh line -- CoT models often finish a sentence
# and continue straight into "...from the list. 1. [X]" on the same line --
# but items 2/3 must each start their own line (else we'd match "1." occurring
# incidentally inside prose).
NUMBERED_LINE = re.compile(rf"(?:^|[.\s])1[\.\)]\s*{_ITEM}\n\s*2[\.\)]\s*{_ITEM}\n\s*3[\.\)]\s*{_ITEM}$",
                            re.MULTILINE)


def parse_predictions(raw_text: str, disease_list: list[str]) -> tuple[list[str], bool]:
    """Extract up to 3 diagnosis strings from free text, then map each to
    the closest known disease name (exact normalized match first, then
    fuzzy). Returns (predictions, parsed_cleanly) where predictions always
    has exactly 3 entries (padded with "" if fewer were found) and
    parsed_cleanly is False whenever we had to fall back to a heuristic
    line-split instead of finding a well-formed "1./2./3." block -- this
    itself is a useful signal (some models fail to converge on a clean
    final answer, especially under the debias/CoT prompts).

    CoT/debias outputs often restate the answer multiple times (a rough
    draft, then a "however, considering..." aside, then a final answer);
    we take the LAST well-formed "1./2./3." block, matching how these
    models actually place their final answer, rather than the first."""
    blocks = NUMBERED_LINE.findall(raw_text)
    parsed_cleanly = bool(blocks)
    if blocks:
        matches = list(blocks[-1])
    else:
        # fallback: split on newlines, take the LAST 3 non-empty lines
        # (final-answer position for verbose responses), strip bullets/numbering
        lines = [ln.strip() for ln in raw_text.splitlines() if ln.strip()]
        cleaned = []
        for ln in lines:
            ln = re.sub(r"^[\-\*\d\.\)]+\s*", "", ln).strip("[]. ")
            if ln:
                cleaned.append(ln)
        matches = cleaned[-3:] if cleaned else []

    norm_lookup = {d.lower().replace("_", " ").replace("-", " "): d for d in disease_list}

    out = []
    for m in matches[:3]:
        key = m.lower().replace("_", " ").replace("-", " ").strip()
        if key in norm_lookup:
            out.append(norm_lookup[key])
            continue
        close = difflib.get_close_matches(key, norm_lookup.keys(), n=1, cutoff=0.6)
        out.append(norm_lookup[close[0]] if close else m.strip())
    while len(out) < 3:
        out.append("")
    return out[:3], parsed_cleanly


def build_prompts(vignettes: pd.DataFrame, disease_list: list[str], condition: str) -> list[str]:
    fn = CONDITIONS[condition]
    disease_str = ", ".join(disease_list)
    prompts = []
    for _, row in vignettes.iterrows():
        pos = str(row["positive_findings"]) if pd.notna(row["positive_findings"]) else ""
        neg = str(row["negative_findings"]) if pd.notna(row["negative_findings"]) else ""
        prompts.append(fn(disease_str, pos.replace(";", ", "), neg.replace(";", ", ") or "none"))
    return prompts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF model id or local path")
    ap.add_argument("--model_tag", required=True, help="short name used in output filenames")
    ap.add_argument("--condition", required=True, choices=list(CONDITIONS.keys()))
    ap.add_argument("--vignettes", required=True)
    ap.add_argument("--ontology", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split_filter", default=None,
                     help="optional: only run vignettes whose 'split' column equals this value")
    ap.add_argument("--n_samples", type=int, default=1, help=">1 enables self-consistency sampling")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max_tokens", type=int, default=None,
                     help="override the per-condition default in prompts.MAX_TOKENS_BY_CONDITION")
    ap.add_argument("--max_model_len", type=int, default=4096)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    ap.add_argument("--lora_path", default=None, help="optional LoRA adapter path to apply")
    args = ap.parse_args()

    from vllm import LLM, SamplingParams

    vignettes = pd.read_csv(args.vignettes)
    if args.split_filter:
        vignettes = vignettes[vignettes["split"] == args.split_filter].reset_index(drop=True)
    ontology = pd.read_csv(args.ontology)
    disease_list = load_disease_list(ontology)

    prompts = build_prompts(vignettes, disease_list, args.condition)

    llm_kwargs = dict(model=args.model, max_model_len=args.max_model_len,
                       gpu_memory_utilization=args.gpu_memory_utilization,
                       enable_lora=args.lora_path is not None)
    llm = LLM(**llm_kwargs)

    max_tokens = args.max_tokens if args.max_tokens is not None else MAX_TOKENS_BY_CONDITION.get(args.condition, 300)
    sampling = SamplingParams(
        temperature=args.temperature if args.n_samples == 1 else max(args.temperature, 0.7),
        max_tokens=max_tokens,
        n=args.n_samples,
    )

    lora_request = None
    if args.lora_path:
        from vllm.lora.request import LoRARequest
        lora_request = LoRARequest("debias_adapter", 1, args.lora_path)

    t0 = time.time()
    outputs = llm.generate(prompts, sampling, lora_request=lora_request)
    elapsed = time.time() - t0

    rows = []
    for vrow, out in zip(vignettes.to_dict("records"), outputs):
        sample_preds = [parse_predictions(o.text, disease_list) for o in out.outputs]  # list of (preds3, parsed_cleanly)
        if args.n_samples == 1:
            (p1, p2, p3), parsed_cleanly = sample_preds[0]
            raw = out.outputs[0].text
        else:
            # self-consistency: majority vote over each sample's prediction_1
            from collections import Counter
            top1_votes = Counter(preds[0] for preds, _ in sample_preds if preds[0])
            p1 = top1_votes.most_common(1)[0][0] if top1_votes else sample_preds[0][0][0]
            # fill 2/3 from the highest-scoring single sample that agrees with the vote
            agreeing = next((s for s in sample_preds if s[0][0] == p1), sample_preds[0])
            p2, p3 = agreeing[0][1], agreeing[0][2]
            parsed_cleanly = agreeing[1]
            raw = " ||| ".join(o.text for o in out.outputs)

        rows.append({
            "vignette_id": vrow["vignette_id"],
            "prediction_1": p1, "prediction_2": p2, "prediction_3": p3,
            "parsed_cleanly": parsed_cleanly,
            "raw_response": raw[:2000],
        })

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)

    print(f"model={args.model_tag} condition={args.condition} n={len(rows)} "
          f"elapsed={elapsed:.1f}s ({elapsed/max(len(rows),1):.3f}s/vignette) -> {out_path}")


if __name__ == "__main__":
    main()
