#!/usr/bin/env python3
"""Prepares a CUPCase (Perets et al., AAAI 2025 -- real case reports from BMC,
formulated as 4-way MCQ with hard-negative distractors) subset for external
validation of the DiagRare-X Plackett-Luce framework.

This is the "genuinely external, independently-sourced" test the analysis needs:
DiagRare-X's ontology and generator are OUR contribution; CUPCase's cases,
diagnoses, and distractors were authored by someone else entirely, from real
clinical literature, with no knowledge of our benchmark.

Prior axis: log(PubMed article count) for each candidate diagnosis name,
fetched via the NCBI E-utilities API -- a REAL, externally-sourced proxy for
"how well-known/common this diagnosis is" that has nothing to do with any
prevalence number we assigned ourselves in DiagRare-X.

Evidence axis: computed separately in score_cupcase_evidence.py once a
working torch/embedding environment is available (this script only prepares
the case/candidate/prior data, which needs no GPU).
"""
from __future__ import annotations

import argparse
import time
import urllib.parse
import urllib.request
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
N_SAMPLE = 250
RNG_SEED = 20260912


def pubmed_count(term: str, retries: int = 3) -> int:
    url = ("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&retmode=json&term="
           + urllib.parse.quote(f'"{term}"'))
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=10) as resp:
                data = json.load(resp)
            return int(data["esearchresult"]["count"])
        except Exception:
            time.sleep(1.0)
    return 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true",
                     help="use the ENTIRE CUPCase dataset instead of the 250-case subset; "
                          "writes to cupcase_full_prepared.csv, leaving the original untouched")
    args = ap.parse_args()

    parquet_path = list((ROOT / "data" / "external" / "hf_cache").rglob("*.parquet"))[0]
    df = pd.read_parquet(parquet_path)
    print(f"Loaded {len(df)} CUPCase cases from {parquet_path}")

    n = len(df) if args.full else N_SAMPLE
    sample = df.sample(n=min(n, len(df)), random_state=RNG_SEED).reset_index(drop=True)
    sample["case_id"] = range(len(sample))

    print(f"Fetching PubMed counts for {len(sample) * 4} candidate names (rate-limited to ~3/sec)...")
    pubmed_cache: dict[str, int] = {}
    rows = []
    t0 = time.time()
    for i, row in sample.iterrows():
        candidates = [row["correct_diagnosis"], row["distractor1"], row["distractor2"], row["distractor3"]]
        counts = []
        for c in candidates:
            key = c.strip()
            if key not in pubmed_cache:
                pubmed_cache[key] = pubmed_count(key)
                time.sleep(0.34)  # ~3 req/sec, safe without an API key
            counts.append(pubmed_cache[key])
        rows.append({
            "case_id": row["case_id"], "case_presentation": row["clean_case_presentation"],
            "correct_diagnosis": candidates[0], "distractor1": candidates[1],
            "distractor2": candidates[2], "distractor3": candidates[3],
            "pubmed_count_correct": counts[0], "pubmed_count_d1": counts[1],
            "pubmed_count_d2": counts[2], "pubmed_count_d3": counts[3],
        })
        if (i + 1) % 25 == 0:
            print(f"  [{time.time()-t0:.0f}s] {i+1}/{len(sample)} cases done", flush=True)

    out = pd.DataFrame(rows)
    out_dir = ROOT / "data" / "external"
    out_name = "cupcase_full_prepared.csv" if args.full else "cupcase_prepared.csv"
    out.to_csv(out_dir / out_name, index=False)
    print(f"\nWrote {out_dir / out_name} ({len(out)} cases)")
    print(f"Unique diagnosis names queried: {len(pubmed_cache)}")
    print(f"PubMed count distribution (correct diagnosis): "
          f"min={out.pubmed_count_correct.min()} median={out.pubmed_count_correct.median()} "
          f"max={out.pubmed_count_correct.max()}")


if __name__ == "__main__":
    main()
