#!/usr/bin/env python3
"""Runs ALL THREE remaining GPU-dependent experiments (causal-intervention
grid, CUPCase external validation, RareArena external validation) for ONE
model in a single vLLM engine load. Model loading is the exact operation
that was stalling for 45+ minutes under the Lustre-mmap issue (see
slurm/env.sh for the diagnosis and fix); combining three separate
inference jobs that each pay that cost once into a single job that pays it
ONCE for all three is the difference between roughly 3x and 1x of whatever
the (now much-reduced, but not zero) loading overhead remains -- worth
doing regardless of how well the staging fix works.

Usage:
  python run_combined_gpu_experiments.py --model Qwen/Qwen2.5-7B-Instruct \
      --model_tag qwen2.5-7b --out_dir ../results/theory
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "external_validation"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "inference"))
ROOT = Path(__file__).resolve().parents[2]

from run_causal_grid_inference import build_prompt as cg_build_prompt  # noqa: E402
from run_causal_grid_inference import parse_binary_choice  # noqa: E402
from run_cupcase_inference import build_prompt_and_key as cc_build_prompt_and_key  # noqa: E402
from run_cupcase_inference import parse_letter  # noqa: E402
from run_rarearena_inference import build_prompt_and_key as ra_build_prompt_and_key  # noqa: E402
from run_inference_vllm import load_disease_list, parse_predictions, build_prompts  # noqa: E402
from prompts import MAX_TOKENS_BY_CONDITION  # noqa: E402

import numpy as np


def run_baseline(llm, model_tag: str) -> pd.DataFrame:
    """Main DiagRare-X baseline-condition inference (984 vignettes, 82-way
    top-3 free-text ranking) -- needed to compute this model's accuracy and
    gamma_prior/gamma_evidence alongside the new-panel external-validation
    results. Writes in the exact schema score_predictions.py and
    fit_per_all_models.py already expect, so no downstream code changes."""
    from vllm import SamplingParams
    vignettes = pd.read_csv(ROOT / "data" / "expanded" / "vignettes_combined.csv")
    ontology = pd.read_csv(ROOT / "data" / "expanded" / "ontology_diseases_expanded.csv")
    disease_list = load_disease_list(ontology)
    prompts = build_prompts(vignettes, disease_list, "baseline")
    sp = SamplingParams(temperature=0.0, max_tokens=MAX_TOKENS_BY_CONDITION["baseline"])
    t0 = time.time()
    outs = llm.generate(prompts, sp)
    elapsed = time.time() - t0
    rows = []
    for vrow, out in zip(vignettes.to_dict("records"), outs):
        (p1, p2, p3), parsed_cleanly = parse_predictions(out.outputs[0].text, disease_list)
        rows.append({"vignette_id": vrow["vignette_id"], "prediction_1": p1, "prediction_2": p2,
                     "prediction_3": p3, "parsed_cleanly": parsed_cleanly,
                     "raw_response": out.outputs[0].text[:2000]})
    df = pd.DataFrame(rows)
    unparsed_rate = 1.0 - df["parsed_cleanly"].astype(bool).mean()
    print(f"  [baseline] n={len(df)} elapsed={elapsed:.1f}s unparsed_rate={unparsed_rate:.1%}")
    return df


def run_causal_grid(llm, sp_short) -> pd.DataFrame:
    from vllm import SamplingParams
    grid = pd.read_csv(ROOT / "data" / "causal_grid" / "causal_grid.csv")
    prompts = [cg_build_prompt(row) for _, row in grid.iterrows()]
    sp = SamplingParams(temperature=0, max_tokens=20)
    t0 = time.time()
    outs = llm.generate(prompts, sp)
    elapsed = time.time() - t0
    rows = []
    for (_, row), out in zip(grid.iterrows(), outs):
        text = out.outputs[0].text
        choice = parse_binary_choice(text, row["target_disease"], row["confounder_disease"])
        rows.append({**row.to_dict(), "raw_response": text[:300], "choice": choice})
    df = pd.DataFrame(rows)
    n_unparsed = (df.choice == "unparsed").sum()
    print(f"  [causal_grid] n={len(df)} elapsed={elapsed:.1f}s unparsed={n_unparsed}/{len(df)}")
    return df


def run_cupcase(llm, model_tag: str, seed: int = 20260912, full: bool = False) -> pd.DataFrame:
    from vllm import SamplingParams
    fname = "cupcase_full_with_evidence.csv" if full else "cupcase_with_evidence.csv"
    data_csv = ROOT / "data" / "external" / fname
    if not data_csv.exists():
        print(f"  [cupcase] SKIPPED: {fname} not found")
        return pd.DataFrame()
    df = pd.read_csv(data_csv)
    rng = np.random.default_rng(seed)
    prompts, letter_maps, correct_letters = [], [], []
    for _, row in df.iterrows():
        prompt, letter_to_key, correct_letter = cc_build_prompt_and_key(row, rng)
        prompts.append(prompt)
        letter_maps.append(letter_to_key)
        correct_letters.append(correct_letter)
    sp = SamplingParams(temperature=0, max_tokens=10)
    t0 = time.time()
    outs = llm.generate(prompts, sp)
    elapsed = time.time() - t0
    rows = []
    for (_, row), out, letter_map, correct_letter in zip(df.iterrows(), outs, letter_maps, correct_letters):
        text = out.outputs[0].text
        chosen_letter = parse_letter(text)
        chosen_key = letter_map.get(chosen_letter, "unparsed")
        rows.append({"case_id": row["case_id"], "chosen_key": chosen_key, "correct_letter": correct_letter,
                     "chosen_letter": chosen_letter, "is_correct": chosen_key == "correct",
                     "letter_map": str(letter_map), "raw_response": text[:100]})
    out_df = pd.DataFrame(rows)
    acc = out_df.is_correct.mean() if len(out_df) else float("nan")
    print(f"  [cupcase] n={len(out_df)} elapsed={elapsed:.1f}s top1_accuracy={acc:.1%}")
    return out_df


def run_rarearena(llm, model_tag: str, seed: int = 20260913, full: bool = False) -> pd.DataFrame:
    from vllm import SamplingParams
    fname = "rarearena_full_with_evidence.csv" if full else "rarearena_with_evidence.csv"
    data_csv = ROOT / "data" / "external" / fname
    if not data_csv.exists():
        print(f"  [rarearena] SKIPPED: {fname} not found")
        return pd.DataFrame()
    df = pd.read_csv(data_csv)
    rng = np.random.default_rng(seed)
    prompts, letter_maps, correct_letters = [], [], []
    for _, row in df.iterrows():
        prompt, letter_to_key, correct_letter = ra_build_prompt_and_key(row, rng)
        prompts.append(prompt)
        letter_maps.append(letter_to_key)
        correct_letters.append(correct_letter)
    sp = SamplingParams(temperature=0, max_tokens=10)
    t0 = time.time()
    outs = llm.generate(prompts, sp)
    elapsed = time.time() - t0
    rows = []
    for (_, row), out, letter_map, correct_letter in zip(df.iterrows(), outs, letter_maps, correct_letters):
        text = out.outputs[0].text
        chosen_letter = parse_letter(text)
        chosen_key = letter_map.get(chosen_letter, "unparsed")
        rows.append({"case_id": row["case_id"], "chosen_key": chosen_key, "correct_letter": correct_letter,
                     "chosen_letter": chosen_letter, "is_correct": chosen_key == "correct",
                     "letter_map": str(letter_map), "raw_response": text[:100]})
    out_df = pd.DataFrame(rows)
    acc = out_df.is_correct.mean() if len(out_df) else float("nan")
    print(f"  [rarearena] n={len(out_df)} elapsed={elapsed:.1f}s top1_accuracy={acc:.1%}")
    return out_df



def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--model_tag", required=True)
    ap.add_argument("--out_dir", default=str(ROOT / "results" / "theory"))
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    ap.add_argument("--max_model_len", type=int, default=4096)
    ap.add_argument("--tokenizer_mode", default="auto", choices=["auto", "slow"],
                     help="use 'slow' for models whose fast-tokenizer conversion fails "
                          "(e.g. InternLM2's tiktoken-based fast tokenizer)")
    ap.add_argument("--skip", nargs="*", default=[],
                     choices=["baseline", "causal_grid", "cupcase", "rarearena"],
                     help="experiments to skip (e.g. if already completed for this model)")
    ap.add_argument("--full", action="store_true",
                     help="use the FULL-scale external-validation datasets "
                          "(cupcase_full_with_evidence.csv etc.) instead of the 250-case subsets, "
                          "and write to results/theory/external_validation_full/ instead of "
                          "external_validation/ so the original results are not overwritten")
    args = ap.parse_args()

    from vllm import LLM

    print(f"=== Loading {args.model} (tokenizer_mode={args.tokenizer_mode}) ===", flush=True)
    t0 = time.time()
    llm = LLM(model=args.model, max_model_len=args.max_model_len,
              gpu_memory_utilization=args.gpu_memory_utilization, trust_remote_code=True,
              tokenizer_mode=args.tokenizer_mode)
    print(f"=== Model loaded in {time.time()-t0:.1f}s ===", flush=True)

    out_dir = Path(args.out_dir)
    ext_subdir = "external_validation_full" if args.full else "external_validation"

    if "baseline" not in args.skip:
        print(f"--- Running main baseline condition for {args.model_tag} ---", flush=True)
        df = run_baseline(llm, args.model_tag)
        p = ROOT / "results" / "predictions" / f"{args.model_tag}__baseline.csv"
        p.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(p, index=False)
        print(f"  wrote {p}")

    if "causal_grid" not in args.skip:
        print(f"--- Running causal_grid for {args.model_tag} ---", flush=True)
        df = run_causal_grid(llm, None)
        p = out_dir / "causal_grid" / f"{args.model_tag}.csv"
        p.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(p, index=False)
        print(f"  wrote {p}")

    if "cupcase" not in args.skip:
        print(f"--- Running cupcase for {args.model_tag} ---", flush=True)
        df = run_cupcase(llm, args.model_tag, full=args.full)
        if len(df):
            p = out_dir / ext_subdir / "cupcase" / f"{args.model_tag}.csv"
            p.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(p, index=False)
            print(f"  wrote {p}")

    if "rarearena" not in args.skip:
        print(f"--- Running rarearena for {args.model_tag} ---", flush=True)
        df = run_rarearena(llm, args.model_tag, full=args.full)
        if len(df):
            p = out_dir / ext_subdir / "rarearena" / f"{args.model_tag}.csv"
            p.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(p, index=False)
            print(f"  wrote {p}")


    print(f"=== ALL DONE for {args.model_tag} (total {time.time()-t0:.1f}s) ===")


if __name__ == "__main__":
    main()
