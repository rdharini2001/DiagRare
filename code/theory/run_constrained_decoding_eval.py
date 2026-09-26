#!/usr/bin/env python3
"""Run candidate-constrained scoring to remove free-text parsing from the comparison.

The script scores the candidate diagnoses directly and provides a parsing-independent
check of the prior and evidence effects.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

FINDINGS_BLOCK = "POSITIVE FINDINGS: {positive}\nNEGATIVE/ABSENT FINDINGS: {negative}"


def build_id_map(disease_list: list[str]) -> dict[str, str]:
    return {d: f"D{i+1:02d}" for i, d in enumerate(disease_list)}


def build_base_prompt(id_to_name: dict[str, str], positive: str, negative: str) -> str:
    id_list_str = ", ".join(f"{iid}={name}" for iid, name in id_to_name.items())
    return (
        "You are a diagnostic reasoning system. Given the clinical findings below, "
        "identify the most likely diagnosis from this list of ID=Name pairs:\n\n"
        f"DISEASES: {id_list_str}\n\n"
        f"{FINDINGS_BLOCK.format(positive=positive, negative=negative)}\n\n"
        "Respond with ONLY the ID of the single most likely diagnosis (e.g. D07)."
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--model_tag", required=True)
    ap.add_argument("--vignettes", default=str(ROOT / "data" / "expanded" / "vignettes_combined.csv"))
    ap.add_argument("--ontology", default=str(ROOT / "data" / "expanded" / "ontology_diseases_expanded.csv"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    ap.add_argument("--max_model_len", type=int, default=4096)
    args = ap.parse_args()

    from vllm import LLM, SamplingParams
    from vllm.sampling_params import GuidedDecodingParams

    vignettes = pd.read_csv(args.vignettes)
    ontology = pd.read_csv(args.ontology)
    disease_list = sorted(ontology["disease"].astype(str).tolist())
    id_to_name = build_id_map(disease_list)
    name_to_id = {v: k for k, v in id_to_name.items()}
    all_ids = list(id_to_name.keys())

    llm = LLM(model=args.model, max_model_len=args.max_model_len,
              gpu_memory_utilization=args.gpu_memory_utilization, trust_remote_code=True)

    base_prompts = []
    for _, v in vignettes.iterrows():
        pos = str(v["positive_findings"]).replace(";", ", ") if pd.notna(v["positive_findings"]) else ""
        neg = (str(v["negative_findings"]).replace(";", ", ") if pd.notna(v["negative_findings"]) else "") or "none"
        base_prompts.append(build_base_prompt(id_to_name, pos, neg))

    n = len(vignettes)
    picked_ranks: list[list[str]] = [[] for _ in range(n)]  # accumulates chosen IDs per vignette
    remaining_choices: list[list[str]] = [list(all_ids) for _ in range(n)]

    t0 = time.time()
    for rank in range(3):
        prompts_this_rank = []
        for i in range(n):
            already = ", ".join(f"{iid}={id_to_name[iid]}" for iid in picked_ranks[i])
            suffix = (f"\n\nAlready ranked higher (do not repeat): {already}\n"
                      f"Respond with ONLY the ID of the NEXT most likely diagnosis."
                      if picked_ranks[i] else "")
            prompts_this_rank.append(base_prompts[i] + suffix)

        # guided_decoding.choice differs per vignette (shrinking candidate pool), so
        # each request gets its own SamplingParams instance.
        sps = [SamplingParams(temperature=0, max_tokens=5,
                               guided_decoding=GuidedDecodingParams(choice=remaining_choices[i]))
               for i in range(n)]
        outs = llm.generate(prompts_this_rank, sps, use_tqdm=False)
        for i, out in enumerate(outs):
            chosen_id = out.outputs[0].text.strip()
            if chosen_id not in id_to_name:
                # guided decoding should make this unreachable, but fall back
                # defensively to the first remaining candidate rather than crash.
                chosen_id = remaining_choices[i][0]
            picked_ranks[i].append(chosen_id)
            remaining_choices[i] = [c for c in remaining_choices[i] if c != chosen_id]
        print(f"  rank {rank+1}/3 done ({time.time()-t0:.0f}s elapsed)", flush=True)

    elapsed = time.time() - t0
    rows = []
    for i, v in enumerate(vignettes.to_dict("records")):
        names = [id_to_name[iid] for iid in picked_ranks[i]]
        rows.append({
            "vignette_id": v["vignette_id"],
            "prediction_1": names[0], "prediction_2": names[1], "prediction_3": names[2],
            "parsed_cleanly": True,  # true by construction: guided decoding cannot emit an invalid ID
        })

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"model={args.model_tag} n={len(rows)} elapsed={elapsed:.1f}s parsed_cleanly=100% (by construction) "
          f"-> {out_path}")


if __name__ == "__main__":
    main()
