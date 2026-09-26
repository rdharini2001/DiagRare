#!/usr/bin/env python3
"""Diagnostic (not part of the main uniform panel): med42-8b, openbiollm-8b,
and biomistral-7b produced 100% unparseable/empty output under this
project's established raw-completion prompting protocol (no chat template --
used uniformly across all 24 panel models for comparability). Tokenizer
inspection showed why: Med42-8B and BioMistral-7B ship real chat templates
and eos tokens (<|eot_id|>, </s>) tuned to fire the instant a raw,
non-chat-formatted prompt looks "done" to the RLHF'd model; OpenBioLLM-8B
ships NO chat_template at all and has pad_token==eos_token, a known failure
pattern for exactly this kind of silent all-EOS collapse.

This script re-runs baseline + causal_grid for those 3 models WITH each
model's native chat template applied (Llama-3's official template
hardcoded as a fallback for OpenBioLLM-8B, whose tokenizer_config.json
does not carry one despite being fine-tuned from Llama-3-8B-Instruct), to
determine whether they are genuinely weak or merely prompt-format-brittle.
Writes to a separate results/theory/chattemplate_diagnostic/ tree -- this
does NOT feed back into the main uniform-protocol panel/gamma tables, to
keep that comparison's methodology identical across all models. It is
reported as its own robustness finding: some domain-specialized medical
LLMs are far more sensitive to exact prompt formatting than general-purpose
models, a real deployment-robustness dimension in its own right.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "inference"))

sys.path.insert(0, str(Path(__file__).resolve().parent / "external_validation"))

from run_causal_grid_inference import build_prompt as cg_build_prompt  # noqa: E402
from run_causal_grid_inference import parse_binary_choice  # noqa: E402
from run_inference_vllm import load_disease_list, parse_predictions, build_prompts  # noqa: E402
from prompts import MAX_TOKENS_BY_CONDITION  # noqa: E402
from run_cupcase_inference import build_prompt_and_key as cc_build_prompt_and_key  # noqa: E402
from run_cupcase_inference import parse_letter  # noqa: E402
from run_rarearena_inference import build_prompt_and_key as ra_build_prompt_and_key  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]

LLAMA3_FALLBACK_TEMPLATE = (
    "{% set loop_messages = messages %}"
    "{% for message in loop_messages %}"
    "{{ '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n' + message['content'] | trim + '<|eot_id|>' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}"
)

# MedAlpaca ships no tokenizer chat_template, but its model card documents
# the Stanford Alpaca instruction format it was fine-tuned on.
ALPACA_TEMPLATE = (
    "Below is an instruction that describes a task. Write a response that "
    "appropriately completes the request.\n\n### Instruction:\n{prompt}\n\n### Response:\n"
)

# Model tags whose tokenizer has NO chat_template and need a hardcoded,
# model-card-documented instruction format instead. Asclepius-7B is
# deliberately NOT included here: its native format expects a separate
# discharge-summary + instruction split that our single raw-prompt string
# does not cleanly decompose into, so we leave its raw-completion result as
# the reported (documented-limitation) number rather than guessing a format.
MANUAL_TEMPLATES = {
    "medalpaca-7b": ALPACA_TEMPLATE,
    "openbiollm-8b": None,  # uses LLAMA3_FALLBACK_TEMPLATE below
}


def wrap_chat(tokenizer, raw_prompt: str, model_tag: str) -> str:
    if model_tag in MANUAL_TEMPLATES:
        if MANUAL_TEMPLATES[model_tag] is not None:
            return MANUAL_TEMPLATES[model_tag].format(prompt=raw_prompt)
        tokenizer.chat_template = LLAMA3_FALLBACK_TEMPLATE
        return "<|begin_of_text|>" + tokenizer.apply_chat_template(
            [{"role": "user", "content": raw_prompt}], tokenize=False, add_generation_prompt=True)
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": raw_prompt}], tokenize=False, add_generation_prompt=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--model_tag", required=True)
    ap.add_argument("--max_model_len", type=int, default=4096)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.80)
    ap.add_argument("--full", action="store_true",
                     help="use the FULL-scale external-validation datasets instead of the "
                          "250-case subsets, writing to a *_full suffixed output filename so "
                          "the original chattemplate_diagnostic/ results are not overwritten")
    ap.add_argument("--only_external", action="store_true",
                     help="skip baseline/causal_grid entirely (they don't depend on external "
                          "dataset scale and already have results) and run only "
                          "cupcase/rarearena")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tok = AutoTokenizer.from_pretrained(args.model)
    print(f"=== Loading {args.model} ===", flush=True)
    llm = LLM(model=args.model, max_model_len=args.max_model_len,
              gpu_memory_utilization=args.gpu_memory_utilization, trust_remote_code=True)

    out_root = ROOT / "results" / "theory" / "chattemplate_diagnostic"
    out_root.mkdir(parents=True, exist_ok=True)
    out_suffix = "_full" if args.full else ""

    if not args.only_external:
        # --- baseline ---
        vignettes = pd.read_csv(ROOT / "data" / "expanded" / "vignettes_combined.csv")
        ontology = pd.read_csv(ROOT / "data" / "expanded" / "ontology_diseases_expanded.csv")
        disease_list = load_disease_list(ontology)
        raw_prompts = build_prompts(vignettes, disease_list, "baseline")
        chat_prompts = [wrap_chat(tok, p, args.model_tag) for p in raw_prompts]
        sp = SamplingParams(temperature=0.0, max_tokens=MAX_TOKENS_BY_CONDITION["baseline"])
        t0 = time.time()
        outs = llm.generate(chat_prompts, sp)
        elapsed = time.time() - t0
        rows = []
        for vrow, out in zip(vignettes.to_dict("records"), outs):
            (p1, p2, p3), parsed_cleanly = parse_predictions(out.outputs[0].text, disease_list)
            rows.append({"vignette_id": vrow["vignette_id"], "prediction_1": p1, "prediction_2": p2,
                         "prediction_3": p3, "parsed_cleanly": parsed_cleanly,
                         "raw_response": out.outputs[0].text[:2000]})
        df = pd.DataFrame(rows)
        unparsed_rate = 1.0 - df["parsed_cleanly"].astype(bool).mean()
        print(f"  [baseline, chat-template] n={len(df)} elapsed={elapsed:.1f}s unparsed_rate={unparsed_rate:.1%}")
        p = out_root / f"{args.model_tag}__baseline.csv"
        df.to_csv(p, index=False)
        print(f"  wrote {p}")

        # --- causal_grid ---
        grid = pd.read_csv(ROOT / "data" / "causal_grid" / "causal_grid.csv")
        raw_cg_prompts = [cg_build_prompt(row) for _, row in grid.iterrows()]
        chat_cg_prompts = [wrap_chat(tok, p, args.model_tag) for p in raw_cg_prompts]
        sp2 = SamplingParams(temperature=0, max_tokens=20)
        t0 = time.time()
        outs2 = llm.generate(chat_cg_prompts, sp2)
        elapsed2 = time.time() - t0
        rows2 = []
        for (_, row), out in zip(grid.iterrows(), outs2):
            text = out.outputs[0].text
            choice = parse_binary_choice(text, row["target_disease"], row["confounder_disease"])
            rows2.append({**row.to_dict(), "raw_response": text[:300], "choice": choice})
        df2 = pd.DataFrame(rows2)
        n_unparsed2 = (df2.choice == "unparsed").sum()
        print(f"  [causal_grid, chat-template] n={len(df2)} elapsed={elapsed2:.1f}s unparsed={n_unparsed2}/{len(df2)}")
        p2 = out_root / f"{args.model_tag}__causal_grid.csv"
        df2.to_csv(p2, index=False)
        print(f"  wrote {p2}")

    # --- cupcase / rarearena (4-way lettered forced choice) ---
    ds_files = {
        "cupcase": "cupcase_full_with_evidence.csv" if args.full else "cupcase_with_evidence.csv",
        "rarearena": "rarearena_full_with_evidence.csv" if args.full else "rarearena_with_evidence.csv",
    }
    for ds_name, data_file, build_fn, seed in [
        ("cupcase", ds_files["cupcase"], cc_build_prompt_and_key, 20260912),
        ("rarearena", ds_files["rarearena"], ra_build_prompt_and_key, 20260913),
    ]:
        data_csv = ROOT / "data" / "external" / data_file
        if not data_csv.exists():
            print(f"  [{ds_name}] SKIPPED: {data_file} not found")
            continue
        ds_df = pd.read_csv(data_csv)
        rng = np.random.default_rng(seed)
        raw_ds_prompts, letter_maps, correct_letters = [], [], []
        for _, row in ds_df.iterrows():
            prompt, letter_to_key, correct_letter = build_fn(row, rng)
            raw_ds_prompts.append(prompt)
            letter_maps.append(letter_to_key)
            correct_letters.append(correct_letter)
        chat_ds_prompts = [wrap_chat(tok, p, args.model_tag) for p in raw_ds_prompts]
        sp3 = SamplingParams(temperature=0, max_tokens=10)
        t0 = time.time()
        try:
            outs3 = llm.generate(chat_ds_prompts, sp3)
        except ValueError as e:
            print(f"  [{ds_name}] SKIPPED: {e}")
            continue
        elapsed3 = time.time() - t0
        rows3 = []
        for (_, row), out, letter_map, correct_letter in zip(ds_df.iterrows(), outs3, letter_maps, correct_letters):
            text = out.outputs[0].text
            chosen_letter = parse_letter(text)
            chosen_key = letter_map.get(chosen_letter, "unparsed")
            rows3.append({"case_id": row["case_id"], "chosen_key": chosen_key, "correct_letter": correct_letter,
                          "chosen_letter": chosen_letter, "is_correct": chosen_key == "correct",
                          "letter_map": str(letter_map), "raw_response": text[:100]})
        df3 = pd.DataFrame(rows3)
        n_unparsed3 = (df3.chosen_key == "unparsed").sum()
        acc3 = df3.is_correct.mean() if len(df3) else float("nan")
        print(f"  [{ds_name}, chat-template] n={len(df3)} elapsed={elapsed3:.1f}s "
              f"unparsed={n_unparsed3}/{len(df3)} top1_accuracy={acc3:.1%}")
        p3 = out_root / f"{args.model_tag}__{ds_name}{out_suffix}.csv"
        df3.to_csv(p3, index=False)
        print(f"  wrote {p3}")

    print(f"=== ALL DONE for {args.model_tag} ===")


if __name__ == "__main__":
    main()
