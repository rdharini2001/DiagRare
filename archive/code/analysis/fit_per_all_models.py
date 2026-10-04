#!/usr/bin/env python3
"""Fit the Plackett-Luce rank model for every open-weight model and condition
with released predictions, using the Bayes-oracle covariates computed
in theory/bayes_oracle.py. Produces one master table:

    results/theory/per_master_table.csv

with columns: source, model, condition, split, n_total, n_used, gamma0,
gamma_prior, gamma_evidence, PER, PER_lo, PER_hi, converged.

`n_used / n_total` matters: a model that mostly fails to produce a parseable
disease name (BioMistral, Qwen2.5-0.5B) yields a PER estimated from very few
choice occasions -- reported through n_used and bootstrap confidence intervals
rather than hidden.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plackett_luce import build_choices_from_predictions, fit_plackett_luce, bootstrap_samples

ROOT = Path(__file__).resolve().parents[2]
N_BOOT = 1000
BOOT_SAMPLES_DIR = ROOT / "results" / "theory" / "bootstrap_samples"


def fit_one(name_parts: dict, pred_csv: Path, oracle_table: pd.DataFrame, vignette_ids: set | None = None) -> dict:
    preds = pd.read_csv(pred_csv)
    choices, n_total, n_used = build_choices_from_predictions(preds, oracle_table, vignette_ids)
    fit = fit_plackett_luce(choices)
    samples = bootstrap_samples(choices, n_boot=N_BOOT)
    ci = {
        "PER_lo": np.percentile(samples["PER"], 2.5) if len(samples["PER"]) else np.nan,
        "PER_hi": np.percentile(samples["PER"], 97.5) if len(samples["PER"]) else np.nan,
        "gamma_prior_lo": np.percentile(samples["gamma_prior"], 2.5) if len(samples["gamma_prior"]) else np.nan,
        "gamma_prior_hi": np.percentile(samples["gamma_prior"], 97.5) if len(samples["gamma_prior"]) else np.nan,
        "gamma_evidence_lo": np.percentile(samples["gamma_evidence"], 2.5) if len(samples["gamma_evidence"]) else np.nan,
        "gamma_evidence_hi": np.percentile(samples["gamma_evidence"], 97.5) if len(samples["gamma_evidence"]) else np.nan,
    }
    BOOT_SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    key = f"{name_parts['source']}__{name_parts['model']}__{name_parts['condition']}__{name_parts['split']}"
    key = key.replace("/", "-")
    np.savez(BOOT_SAMPLES_DIR / f"{key}.npz", **samples)
    return {**name_parts, "n_total": n_total, "n_used": n_used, "used_frac": n_used / max(n_total, 1), **fit, **ci}


def main() -> None:
    oracle82 = pd.read_csv(ROOT / "results" / "theory" / "bayes_oracle_table_expanded.csv")

    rows = []
    t0 = time.time()

    # 1. Main 7-model x 5-condition open-weight study (984 vignettes, 82-disease candidate set)
    pred_dir = ROOT / "results" / "predictions"
    for f in sorted(pred_dir.glob("*.csv")):
        model_tag, condition = f.stem.split("__", 1)
        r = fit_one({"source": "open_weight", "model": model_tag, "condition": condition, "split": "ALL"},
                    f, oracle82)
        rows.append(r)
        print(f"[{time.time()-t0:6.1f}s] open_weight {model_tag:22s} {condition:28s} "
              f"n_used={r['n_used']:4d}/{r['n_total']:4d} PER={r['PER']:.3f} "
              f"[{r['PER_lo']:.3f},{r['PER_hi']:.3f}]")

    # 2. Chat-template-corrected Mistral vs BioMistral ablation
    pred_dir = ROOT / "results" / "predictions_chattemplate"
    for f in sorted(pred_dir.glob("*.csv")):
        model_tag, condition = f.stem.split("__", 1)
        r = fit_one({"source": "chattemplate_ablation", "model": model_tag, "condition": condition, "split": "ALL"},
                    f, oracle82)
        rows.append(r)
        print(f"[{time.time()-t0:6.1f}s] chattemplate {model_tag:22s} {condition:28s} "
              f"n_used={r['n_used']:4d}/{r['n_total']:4d} PER={r['PER']:.3f}")

    # 3. Self-consistency (k=5 majority vote)
    pred_dir = ROOT / "results" / "predictions_selfconsistency"
    for f in sorted(pred_dir.glob("*.csv")):
        model_tag, condition = f.stem.split("__", 1)
        r = fit_one({"source": "self_consistency", "model": model_tag, "condition": condition, "split": "ALL"},
                    f, oracle82)
        rows.append(r)
        print(f"[{time.time()-t0:6.1f}s] self_consistency {model_tag:22s} PER={r['PER']:.3f}")

    # 4. LoRA debiasing pilot (before/after, on its two eval splits)
    vignettes = pd.read_csv(ROOT / "data" / "expanded" / "vignettes_combined.csv")
    for split_name in ["original", "expanded_heldout"]:
        vids = set(vignettes.loc[vignettes["split"] == split_name, "vignette_id"])

        f = ROOT / "results" / "predictions_lora" / f"qwen2.5-7b-lora-debias__{split_name}.csv"
        r = fit_one({"source": "lora_pilot", "model": "qwen2.5-7b-lora-debias", "condition": "baseline",
                     "split": split_name}, f, oracle82, vignette_ids=vids)
        rows.append(r)
        print(f"[{time.time()-t0:6.1f}s] lora_pilot qwen2.5-7b-lora-debias [{split_name}] PER={r['PER']:.3f}")

        # matched "before" baseline on the SAME vignette subset for apples-to-apples comparison
        f_before = ROOT / "results" / "predictions" / "qwen2.5-7b__baseline.csv"
        r_before = fit_one({"source": "lora_pilot_before", "model": "qwen2.5-7b", "condition": "baseline",
                            "split": split_name}, f_before, oracle82, vignette_ids=vids)
        rows.append(r_before)
        print(f"[{time.time()-t0:6.1f}s] lora_pilot_before qwen2.5-7b [{split_name}] PER={r_before['PER']:.3f}")

    # 5. Bayes-oracle self-validation (own top-3 ranking) -- sanity anchor, expect PER approx 1
    oracle_choices_rows = []
    for vid, grp in oracle82.groupby("vignette_id"):
        grp = grp.reset_index(drop=True)
        top3_idx = grp["bayes_log_posterior"].nlargest(3).index.tolist()
        oracle_choices_rows.append((top3_idx, grp["log_prevalence"].to_numpy(), grp["evidence_loglik_ratio"].to_numpy()))
    from plackett_luce import Choice
    oracle_choices = [Choice(chosen=c, log_prevalence=lp, evidence=ev) for c, lp, ev in oracle_choices_rows]
    fit = fit_plackett_luce(oracle_choices)
    oracle_samples = bootstrap_samples(oracle_choices, n_boot=N_BOOT)
    ci = {
        "PER_lo": np.percentile(oracle_samples["PER"], 2.5), "PER_hi": np.percentile(oracle_samples["PER"], 97.5),
        "gamma_prior_lo": np.percentile(oracle_samples["gamma_prior"], 2.5),
        "gamma_prior_hi": np.percentile(oracle_samples["gamma_prior"], 97.5),
        "gamma_evidence_lo": np.percentile(oracle_samples["gamma_evidence"], 2.5),
        "gamma_evidence_hi": np.percentile(oracle_samples["gamma_evidence"], 97.5),
    }
    np.savez(BOOT_SAMPLES_DIR / "oracle_selfcheck__bayes_oracle__baseline__ALL.npz", **oracle_samples)
    rows.append({"source": "oracle_selfcheck", "model": "bayes_oracle", "condition": "baseline", "split": "ALL",
                 "n_total": len(oracle_choices), "n_used": len(oracle_choices), "used_frac": 1.0, **fit, **ci})
    print(f"[{time.time()-t0:6.1f}s] oracle_selfcheck PER={fit['PER']:.4f} (expect approx 1.0)")

    out = pd.DataFrame(rows)
    out_path = ROOT / "results" / "theory" / "per_master_table.csv"
    out.to_csv(out_path, index=False)
    print(f"\nWrote {out_path} ({len(out)} rows) in {time.time()-t0:.1f}s total")


if __name__ == "__main__":
    main()
