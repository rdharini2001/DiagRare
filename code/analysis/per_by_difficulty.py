#!/usr/bin/env python3
"""Stratifies the Plackett-Luce PER fit by vignette difficulty tier
(easy/medium/hard). Directly tests the mechanism the benchmark was designed
around: hard-tier vignettes deliberately inject a same-organ-system
higher-prevalence "confounder" finding (benchmark_generation/generate_vignettes.py),
so if prevalence-vs-evidence conflict is really what's driving gamma_prior
up / gamma_evidence down for weak models, that effect should be concentrated
in the hard tier and much weaker (or absent) on easy vignettes.

Restricted to the baseline condition for the released open-weight checkpoints. The analysis is
run separately for easy, medium, and hard cases.
the full 5-condition version would be a straightforward extension if needed.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plackett_luce import build_choices_from_predictions, fit_plackett_luce, bootstrap_ci

ROOT = Path(__file__).resolve().parents[2]
N_BOOT = 200  # smaller than the main 1000 -- this is a secondary/appendix analysis


def main() -> None:
    oracle82 = pd.read_csv(ROOT / "results" / "theory" / "bayes_oracle_table_expanded.csv")
    vignettes = pd.read_csv(ROOT / "data" / "expanded" / "vignettes_combined.csv")

    rows = []
    t0 = time.time()

    for f in sorted((ROOT / "results" / "predictions").glob("*__baseline.csv")):
        model_tag = f.stem.split("__baseline")[0]
        preds = pd.read_csv(f)
        for tier in ["easy", "medium", "hard"]:
            vids = set(vignettes.loc[vignettes["difficulty"] == tier, "vignette_id"])
            choices, n_total, n_used = build_choices_from_predictions(preds, oracle82, vignette_ids=vids)
            fit = fit_plackett_luce(choices)
            ci = bootstrap_ci(choices, n_boot=N_BOOT)
            rows.append({"source": "open_weight", "model": model_tag, "difficulty": tier,
                         "n_used": n_used, "n_total": n_total, **fit, **ci})
        print(f"[{time.time()-t0:.0f}s] {model_tag} done", flush=True)

    out = pd.DataFrame(rows)
    out_path = ROOT / "results" / "theory" / "per_by_difficulty.csv"
    out.to_csv(out_path, index=False)

    pd.set_option("display.width", 200)
    print("\nPER by difficulty tier (baseline condition):")
    print(out[["source", "model", "difficulty", "n_used", "gamma_prior", "gamma_evidence", "PER"]].to_string(index=False))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
