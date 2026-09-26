#!/usr/bin/env python3
"""Run the Qwen3-8B thinking-budget experiment.

The same checkpoint is evaluated with thinking disabled and with fixed reasoning-token
budgets so changes in accuracy and evidence responsiveness can be compared directly.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "inference"))

from run_causal_grid_inference import build_prompt as cg_build_prompt  # noqa: E402
from run_causal_grid_inference import parse_binary_choice  # noqa: E402
from run_sequential_anchoring_inference import build_prompt as seq_build_prompt  # noqa: E402
from run_inference_vllm import load_disease_list, parse_predictions, build_prompts  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]

CONDITIONS = {
    "no_think": {"enable_thinking": False, "budget": None},
    "think_low": {"enable_thinking": True, "budget": 512},
    "think_med": {"enable_thinking": True, "budget": 2048},
    "think_high": {"enable_thinking": True, "budget": 8192},
}
NO_THINK_MAXTOK = {"baseline": 300, "causal_grid": 20, "sequential_anchoring": 20}


def strip_think(text: str) -> str:
    if "</think>" in text:
        return text.split("</think>", 1)[1].strip()
    return text.strip()


def wrap(tok, raw_prompt: str, enable_thinking: bool) -> str:
    return tok.apply_chat_template(
        [{"role": "user", "content": raw_prompt}], tokenize=False,
        add_generation_prompt=True, enable_thinking=enable_thinking)


def run_condition(llm, tok, cond_name: str, cond: dict, vignettes, disease_list, grid, seq_cases, out_dir):
    from vllm import SamplingParams
    enable_thinking = cond["enable_thinking"]
    budget = cond["budget"]

    # --- (a) baseline ---
    max_tok = budget if budget else NO_THINK_MAXTOK["baseline"]
    raw_prompts = build_prompts(vignettes, disease_list, "baseline")
    chat_prompts = [wrap(tok, p, enable_thinking) for p in raw_prompts]
    sp = SamplingParams(temperature=0.0, max_tokens=max_tok)
    t0 = time.time()
    outs = llm.generate(chat_prompts, sp)
    elapsed = time.time() - t0
    rows = []
    for vrow, out in zip(vignettes.to_dict("records"), outs):
        raw_text = out.outputs[0].text
        n_tok = len(out.outputs[0].token_ids)
        answer_text = strip_think(raw_text)
        (p1, p2, p3), parsed_cleanly = parse_predictions(answer_text, disease_list)
        ran_out_of_budget = enable_thinking and "</think>" not in raw_text
        rows.append({"vignette_id": vrow["vignette_id"], "prediction_1": p1, "prediction_2": p2,
                     "prediction_3": p3, "parsed_cleanly": parsed_cleanly and not ran_out_of_budget,
                     "n_tokens": n_tok, "ran_out_of_budget": ran_out_of_budget,
                     "raw_response": raw_text[:2500]})
    df = pd.DataFrame(rows)
    unparsed_rate = 1.0 - df["parsed_cleanly"].astype(bool).mean()
    print(f"  [{cond_name}][baseline] n={len(df)} elapsed={elapsed:.1f}s unparsed_rate={unparsed_rate:.1%} "
          f"mean_tokens={df['n_tokens'].mean():.0f}", flush=True)
    p = out_dir / "baseline" / f"qwen3-8b__{cond_name}.csv"
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False)

    # --- (b) causal_intervention_grid ---
    max_tok_cg = budget if budget else NO_THINK_MAXTOK["causal_grid"]
    raw_cg_prompts = [cg_build_prompt(row) for _, row in grid.iterrows()]
    chat_cg_prompts = [wrap(tok, p, enable_thinking) for p in raw_cg_prompts]
    sp2 = SamplingParams(temperature=0, max_tokens=max_tok_cg)
    t0 = time.time()
    outs2 = llm.generate(chat_cg_prompts, sp2)
    elapsed2 = time.time() - t0
    rows2 = []
    for (_, row), out in zip(grid.iterrows(), outs2):
        raw_text = out.outputs[0].text
        n_tok = len(out.outputs[0].token_ids)
        answer_text = strip_think(raw_text)
        ran_out_of_budget = enable_thinking and "</think>" not in raw_text
        choice = "unparsed" if ran_out_of_budget else parse_binary_choice(
            answer_text, row["target_disease"], row["confounder_disease"])
        rows2.append({**row.to_dict(), "raw_response": raw_text[:500], "choice": choice,
                      "n_tokens": n_tok, "ran_out_of_budget": ran_out_of_budget})
    df2 = pd.DataFrame(rows2)
    n_unparsed2 = (df2.choice == "unparsed").sum()
    print(f"  [{cond_name}][causal_grid] n={len(df2)} elapsed={elapsed2:.1f}s unparsed={n_unparsed2}/{len(df2)} "
          f"mean_tokens={df2['n_tokens'].mean():.0f}", flush=True)
    p2 = out_dir / "causal_grid" / f"qwen3-8b__{cond_name}.csv"
    p2.parent.mkdir(parents=True, exist_ok=True)
    df2.to_csv(p2, index=False)

    # --- (c) sequential_anchoring ---
    max_tok_seq = budget if budget else NO_THINK_MAXTOK["sequential_anchoring"]
    raw_seq_prompts = [seq_build_prompt(row) for _, row in seq_cases.iterrows()]
    chat_seq_prompts = [wrap(tok, p, enable_thinking) for p in raw_seq_prompts]
    sp3 = SamplingParams(temperature=0, max_tokens=max_tok_seq)
    t0 = time.time()
    outs3 = llm.generate(chat_seq_prompts, sp3)
    elapsed3 = time.time() - t0
    rows3 = []
    for (_, row), out in zip(seq_cases.iterrows(), outs3):
        raw_text = out.outputs[0].text
        n_tok = len(out.outputs[0].token_ids)
        answer_text = strip_think(raw_text)
        ran_out_of_budget = enable_thinking and "</think>" not in raw_text
        choice = "unparsed" if ran_out_of_budget else parse_binary_choice(
            answer_text, row["target_disease"], row["confounder_disease"])
        rows3.append({**row.to_dict(), "raw_response": raw_text[:500], "choice": choice,
                      "n_tokens": n_tok, "ran_out_of_budget": ran_out_of_budget})
    df3 = pd.DataFrame(rows3)
    n_unparsed3 = (df3.choice == "unparsed").sum()
    print(f"  [{cond_name}][sequential_anchoring] n={len(df3)} elapsed={elapsed3:.1f}s "
          f"unparsed={n_unparsed3}/{len(df3)} mean_tokens={df3['n_tokens'].mean():.0f}", flush=True)
    p3 = out_dir / "sequential_anchoring" / f"qwen3-8b__{cond_name}.csv"
    p3.parent.mkdir(parents=True, exist_ok=True)
    df3.to_csv(p3, index=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--max_model_len", type=int, default=16384)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.88)
    ap.add_argument("--conditions", nargs="*", default=list(CONDITIONS.keys()))
    ap.add_argument("--baseline_n", type=int, default=None,
                     help="subsample the 984-vignette baseline to this many rows (fixed seed) -- "
                          "a safety valve for high-budget conditions where the model's reasoning "
                          "traces are long enough that the full baseline set is too slow to run")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM

    tok = AutoTokenizer.from_pretrained(args.model)
    print(f"=== Loading {args.model} ===", flush=True)
    llm = LLM(model=args.model, max_model_len=args.max_model_len,
              gpu_memory_utilization=args.gpu_memory_utilization, trust_remote_code=True)

    vignettes = pd.read_csv(ROOT / "data" / "expanded" / "vignettes_combined.csv")
    if args.baseline_n is not None:
        vignettes = vignettes.sample(n=min(args.baseline_n, len(vignettes)), random_state=20260920) \
            .reset_index(drop=True)
    ontology = pd.read_csv(ROOT / "data" / "expanded" / "ontology_diseases_expanded.csv")
    disease_list = load_disease_list(ontology)
    grid = pd.read_csv(ROOT / "data" / "causal_grid" / "causal_grid.csv")
    seq_cases = pd.read_csv(ROOT / "data" / "causal_grid" / "sequential_anchoring_cases.csv")

    out_dir = ROOT / "results" / "theory" / "qwen3_thinking_budget"
    for cond_name in args.conditions:
        cond = CONDITIONS[cond_name]
        print(f"=== Condition: {cond_name} (enable_thinking={cond['enable_thinking']}, budget={cond['budget']}) ===",
              flush=True)
        run_condition(llm, tok, cond_name, cond, vignettes, disease_list, grid, seq_cases, out_dir)

    print("=== ALL DONE ===")


if __name__ == "__main__":
    main()
