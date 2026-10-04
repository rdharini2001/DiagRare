#!/usr/bin/env python3
"""Pairwise significance tests between models' gamma_evidence, using the raw
bootstrap draws saved by fit_per_all_models.py (results/theory/bootstrap_samples/*.npz)
rather than just eyeballing overlapping CIs. For two independently-bootstrapped
models A and B, the two-sided bootstrap p-value for H0: gamma_evidence_A ==
gamma_evidence_B is 2 * min(mean(boot_A > boot_B), mean(boot_A < boot_B))
(resampling A and B independently, pairing draws index-by-index -- a standard
approximate two-sample bootstrap test).
"""
from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SAMPLES_DIR = ROOT / "results" / "theory" / "bootstrap_samples"


def load_samples(source: str, model: str, condition: str, split: str) -> dict[str, np.ndarray] | None:
    key = f"{source}__{model}__{condition}__{split}".replace("/", "-")
    path = SAMPLES_DIR / f"{key}.npz"
    if not path.exists():
        return None
    return dict(np.load(path))


def bootstrap_pvalue(a: np.ndarray, b: np.ndarray, n: int = 5000, seed: int = 0) -> float:
    if len(a) == 0 or len(b) == 0:
        return np.nan
    rng = np.random.default_rng(seed)
    da = rng.choice(a, size=n, replace=True)
    db = rng.choice(b, size=n, replace=True)
    diff = da - db
    p_gt = np.mean(diff > 0)
    p_lt = np.mean(diff < 0)
    return float(2 * min(p_gt, p_lt))


def main() -> None:
    # Qwen2.5 scaling ladder: pairwise tests on gamma_evidence
    qwen_models = ["qwen2.5-0.5b", "qwen2.5-1.5b", "qwen2.5-3b", "qwen2.5-7b"]
    rows = []
    for a, b in itertools.combinations(qwen_models, 2):
        sa = load_samples("open_weight", a, "baseline", "ALL")
        sb = load_samples("open_weight", b, "baseline", "ALL")
        if sa is None or sb is None:
            continue
        p = bootstrap_pvalue(sa["gamma_evidence"], sb["gamma_evidence"])
        rows.append({"comparison": f"{a} vs {b}", "metric": "gamma_evidence", "p_value": p,
                     "significant_at_0.05": p < 0.05})


    # LoRA before/after: is the fine-tune's gamma shift significant?
    for split in ["original", "expanded_heldout"]:
        s_before = load_samples("lora_pilot_before", "qwen2.5-7b", "baseline", split)
        s_after = load_samples("lora_pilot", "qwen2.5-7b-lora-debias", "baseline", split)
        if s_before is not None and s_after is not None:
            for metric in ["gamma_prior", "gamma_evidence"]:
                p = bootstrap_pvalue(s_before[metric], s_after[metric])
                rows.append({"comparison": f"qwen2.5-7b before vs after LoRA [{split}]",
                             "metric": metric, "p_value": p, "significant_at_0.05": p < 0.05})

    out = pd.DataFrame(rows)
    out_path = ROOT / "results" / "theory" / "per_significance_tests.csv"
    out.to_csv(out_path, index=False)

    pd.set_option("display.width", 200)
    print(out.to_string(index=False))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
