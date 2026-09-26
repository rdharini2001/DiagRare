#!/usr/bin/env python3
"""Compare general-purpose and medically specialized open-weight checkpoints.

The analysis summarizes performance across the primary benchmark, the randomized
intervention, CUPCase, and RareArena. Where a medically specialized checkpoint
requires its native chat template to produce the requested ranking, the lower-failure
format is used and the source is recorded explicitly.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
RAW_PRED = ROOT / "results" / "predictions"
RAW_CG = ROOT / "results" / "theory" / "causal_grid"
RAW_EXT = ROOT / "results" / "theory" / "external_validation"
CHAT_DIR = ROOT / "results" / "theory" / "chattemplate_diagnostic"


def _baseline_stats(df: pd.DataFrame, vignettes: pd.DataFrame) -> tuple[float, float]:
    m = vignettes.merge(df, on="vignette_id", how="inner")
    unparsed = 1.0 - df["parsed_cleanly"].astype(bool).mean()
    acc = (m["prediction_1"].astype(str).str.strip() == m["target_disease"].astype(str).str.strip()).mean()
    return unparsed, acc


def _causal_grid_stats(df: pd.DataFrame) -> tuple[float, float]:
    unparsed = (df["choice"] == "unparsed").mean()
    used = df[df["choice"] != "unparsed"]
    acc = (used["choice"] == "target").mean() if len(used) else float("nan")
    return unparsed, acc


def _forced_choice_stats(df: pd.DataFrame) -> tuple[float, float]:
    unparsed = (df["chosen_key"] == "unparsed").mean()
    acc = df["is_correct"].mean()
    return unparsed, acc


def best_available(model_tag: str, dataset: str, vignettes: pd.DataFrame, full: bool = False) -> dict:
    """Returns the lower-unparsed-rate result for (model_tag, dataset) across
    the raw protocol and (if it exists) the chat-template diagnostic run."""
    candidates = []

    if dataset == "baseline":
        raw_f = RAW_PRED / f"{model_tag}__baseline.csv"
        chat_f = CHAT_DIR / f"{model_tag}__baseline.csv"
        for f, src in [(raw_f, "raw"), (chat_f, "chat_template")]:
            if f.exists():
                unparsed, acc = _baseline_stats(pd.read_csv(f), vignettes)
                candidates.append({"source": src, "unparsed_rate": unparsed, "accuracy": acc})
    elif dataset == "causal_grid":
        raw_f = RAW_CG / f"{model_tag}.csv"
        chat_f = CHAT_DIR / f"{model_tag}__causal_grid.csv"
        for f, src in [(raw_f, "raw"), (chat_f, "chat_template")]:
            if f.exists():
                unparsed, acc = _causal_grid_stats(pd.read_csv(f))
                candidates.append({"source": src, "unparsed_rate": unparsed, "accuracy": acc})
    else:
        ext_subdir = "external_validation_full" if full else "external_validation"
        chat_suffix = "_full" if full else ""
        raw_f = ROOT / "results" / "theory" / ext_subdir / dataset / f"{model_tag}.csv"
        chat_f = CHAT_DIR / f"{model_tag}__{dataset}{chat_suffix}.csv"
        for f, src in [(raw_f, "raw"), (chat_f, "chat_template")]:
            if f.exists():
                unparsed, acc = _forced_choice_stats(pd.read_csv(f))
                candidates.append({"source": src, "unparsed_rate": unparsed, "accuracy": acc})

    if not candidates:
        return {"source": None, "unparsed_rate": np.nan, "accuracy": np.nan}
    return min(candidates, key=lambda c: c["unparsed_rate"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true",
                     help="use the full-scale CUPCase/RareArena external validation results")
    args = ap.parse_args()

    meta = pd.read_csv(ROOT / "data" / "model_metadata.csv")
    vignettes = pd.read_csv(ROOT / "data" / "expanded" / "vignettes_combined.csv")

    datasets = ["baseline", "causal_grid", "cupcase", "rarearena"]
    rows = []
    for _, mrow in meta.iterrows():
        tag = mrow["model_tag"]
        row = {"model": tag, "is_medical_specialized": mrow["is_medical_specialized"],
               "family": mrow["family"], "param_count_b": mrow["param_count_b"]}
        any_usable = False
        for ds in datasets:
            r = best_available(tag, ds, vignettes, full=args.full)
            row[f"{ds}_source"] = r["source"]
            row[f"{ds}_unparsed_rate"] = r["unparsed_rate"]
            row[f"{ds}_accuracy"] = r["accuracy"]
            if r["source"] is not None and r["unparsed_rate"] < 0.5:
                any_usable = True
        row["usable"] = any_usable
        rows.append(row)

    out = pd.DataFrame(rows)
    out_name = "specialization_comparison_full.csv" if args.full else "specialization_comparison.csv"
    out_path = ROOT / "results" / "theory" / out_name
    out.to_csv(out_path, index=False)

    print("=== Per-model best-available results (source = raw protocol vs. native chat template) ===")
    display_cols = ["model", "is_medical_specialized"] + [f"{ds}_accuracy" for ds in datasets] + \
                    [f"{ds}_source" for ds in datasets]
    print(out[display_cols].to_string(index=False))

    print("\n=== General-purpose vs. medical-specialized: head-to-head (usable models only) ===")
    usable = out[out["usable"]]
    for ds in datasets:
        col = f"{ds}_accuracy"
        sub = usable[usable[f"{ds}_unparsed_rate"] < 0.5]
        gen = sub.loc[~sub["is_medical_specialized"], col].dropna()
        med = sub.loc[sub["is_medical_specialized"], col].dropna()
        if len(gen) < 2 or len(med) < 2:
            print(f"{ds:12s}: insufficient usable medical models for a comparison (n_medical={len(med)})")
            continue
        u_stat, p = stats.mannwhitneyu(gen, med, alternative="two-sided")
        print(f"{ds:12s}: general n={len(gen)} mean_acc={gen.mean():.1%} | "
              f"medical n={len(med)} mean_acc={med.mean():.1%} | "
              f"Mann-Whitney p={p:.4g}")

    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
