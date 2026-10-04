#!/usr/bin/env python3
"""Prepares a RareArena (THUMedInfo/RareArena, real de-identified rare-disease
case reports linked to Orphanet diagnoses) subset for a SECOND, independent
external validation of the DiagRare-Bench Plackett-Luce framework -- a deliberate
complement to CUPCase (prepare_cupcase.py), not a duplicate of it.

The two external datasets differ in a methodologically useful way:
  - CUPCase ships CURATED hard-negative distractors (authored by its own
    creators to be plausible confusions) -- tests the framework under
    adversarially hard candidate sets.
  - RareArena ships only a single correct diagnosis per case with no
    distractor structure at all, so here we construct the candidate set
    OURSELVES via uniform random sampling from the full ~11,500-diagnosis
    vocabulary observed in the dataset (excluding the true diagnosis, fixed
    seed for reproducibility) -- tests the framework under a completely
    different, non-adversarial distractor-generation process. Whether
    gamma_evidence's cross-dataset behavior (Sec 7.15) is consistent under
    BOTH distractor regimes is itself a robustness statement neither dataset
    alone could make.

Prior axis: log(PubMed article count) per candidate diagnosis, identical
methodology to prepare_cupcase.py (NCBI E-utilities, real external signal).
Evidence axis: TF-IDF cosine similarity, computed separately in
score_rarearena_evidence.py (pure CPU, no GPU needed).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_cupcase import pubmed_count  # noqa: E402

N_SAMPLE = 250
N_DISTRACTORS = 3
RNG_SEED = 20260913  # distinct from CUPCase's seed -- independent draw


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true",
                     help="use the ENTIRE RareArena dataset instead of the 250-case subset; "
                          "writes to rarearena_full_prepared.csv, leaving the original untouched")
    args = ap.parse_args()

    rdc_path = list((ROOT / "data" / "external" / "hf_cache").rglob("RDC.json"))[0]
    records = []
    with open(rdc_path) as f:
        for line in f:
            d = json.loads(line)
            if d.get("case_report") and d.get("diagnosis"):
                records.append(d)
    print(f"Loaded {len(records)} RareArena records with a case_report + diagnosis from {rdc_path}")

    df = pd.DataFrame(records)
    df["diagnosis_norm"] = df["diagnosis"].str.strip()
    all_diagnoses = df["diagnosis_norm"].unique().tolist()
    print(f"{len(all_diagnoses)} unique diagnosis strings available as the distractor pool")

    rng = np.random.default_rng(RNG_SEED)
    n = len(df) if args.full else N_SAMPLE
    sample = df.sample(n=min(n, len(df)), random_state=RNG_SEED).reset_index(drop=True)
    sample["case_id"] = range(len(sample))

    pubmed_cache: dict[str, int] = {}
    rows = []
    t0 = time.time()
    for i, row in sample.iterrows():
        correct = row["diagnosis_norm"]
        pool = [d for d in all_diagnoses if d.lower() != correct.lower()]
        distractors = rng.choice(pool, size=N_DISTRACTORS, replace=False).tolist()
        candidates = [correct] + distractors

        counts = []
        for c in candidates:
            key = c.strip()
            if key not in pubmed_cache:
                pubmed_cache[key] = pubmed_count(key)
                time.sleep(0.34)  # ~3 req/sec, matches prepare_cupcase.py's rate limit
            counts.append(pubmed_cache[key])

        case_text = str(row["case_report"])
        if pd.notna(row.get("test_results")):
            case_text = case_text + " " + str(row["test_results"])

        rows.append({
            "case_id": row["case_id"], "orig_id": row["_id"], "case_presentation": case_text,
            "correct_diagnosis": candidates[0], "distractor1": candidates[1],
            "distractor2": candidates[2], "distractor3": candidates[3],
            "pubmed_count_correct": counts[0], "pubmed_count_d1": counts[1],
            "pubmed_count_d2": counts[2], "pubmed_count_d3": counts[3],
            "orpha_id": row.get("Orpha_id"), "pub_date": row.get("pub_date"),
        })
        if (i + 1) % 25 == 0:
            print(f"  [{time.time()-t0:.0f}s] {i+1}/{len(sample)} cases done", flush=True)

    out = pd.DataFrame(rows)
    out_dir = ROOT / "data" / "external"
    out_name = "rarearena_full_prepared.csv" if args.full else "rarearena_prepared.csv"
    out.to_csv(out_dir / out_name, index=False)
    print(f"\nWrote {out_dir / out_name} ({len(out)} cases)")
    print(f"Unique diagnosis names queried for PubMed: {len(pubmed_cache)}")
    print(f"PubMed count distribution (correct diagnosis): "
          f"min={out.pubmed_count_correct.min()} median={out.pubmed_count_correct.median()} "
          f"max={out.pubmed_count_correct.max()}")


if __name__ == "__main__":
    main()
