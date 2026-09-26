#!/usr/bin/env python3
"""Analyzes the Qwen3-8B thinking-budget experiment
(run_qwen3_thinking_budget.py). For each of the 4 conditions (no_think,
think_low=512, think_med=2048, think_high=8192), computes -- from the SAME
model, the only thing varied is whether/how much it was allowed to reason:

  - baseline top-1 accuracy and gamma_evidence/gamma_prior (Plackett-Luce
    fit, reusing analysis/plackett_luce.py exactly as the rest of the panel
    does)
  - Delta_E, Delta_P, and the interaction (causal_response_surface.py's
    compute_deltas, applied to this condition's causal_grid predictions)
  - sequential-anchoring recovery_rate and order_effect
    (analyze_sequential_anchoring.py's analyze_one, applied to this
    condition's sequential_anchoring predictions)
  - mean tokens used and the fraction of items that ran out of budget
    before producing an answer (think_* conditions only)

RESPONSIVE TO EVIDENCE (higher gamma_evidence / Delta_E / recovery_rate),
or does it only improve final-answer ACCURACY (or neither / both)? These
can dissociate -- accuracy improving without evidence-responsiveness
improving would itself be a reportable, non-obvious finding.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
from causal_response_surface import load_model_grid, pair_level_table, compute_deltas, bootstrap_ci  # noqa: E402
from analyze_sequential_anchoring import analyze_one as analyze_seq_one  # noqa: E402
from plackett_luce import build_choices_from_predictions, fit_plackett_luce  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
PRED_DIR = ROOT / "results" / "theory" / "qwen3_thinking_budget"
CONDITIONS = ["no_think", "think_low", "think_med", "think_high"]


def fit_baseline(cond: str, oracle: pd.DataFrame, vignettes: pd.DataFrame) -> dict:
    f = PRED_DIR / "baseline" / f"qwen3-8b__{cond}.csv"
    if not f.exists():
        return {}
    preds = pd.read_csv(f)
    m = vignettes.merge(preds, on="vignette_id", how="inner")
    acc = (m["prediction_1"].astype(str).str.strip() == m["target_disease"].astype(str).str.strip()).mean()
    choices, n_total, n_used = build_choices_from_predictions(preds, oracle)
    fit = fit_plackett_luce(choices)
    out = {"top1_accuracy": acc, "gamma_prior": fit["gamma_prior"], "gamma_evidence": fit["gamma_evidence"],
           "n_used": n_used, "n_total": n_total}
    if "n_tokens" in preds.columns:
        out["mean_tokens"] = preds["n_tokens"].mean()
        out["frac_ran_out_of_budget"] = preds.get("ran_out_of_budget", pd.Series([False])).mean()
    return out


def fit_causal_grid(cond: str) -> dict:
    f = PRED_DIR / "causal_grid" / f"qwen3-8b__{cond}.csv"
    if not f.exists():
        return {}
    df = load_model_grid(f)
    if df["pair_id"].nunique() < 5:
        return {}
    wide = pair_level_table(df)
    d = compute_deltas(wide)
    ci_e = bootstrap_ci(d["delta_e_by_pair"])
    out = {"delta_E": d["delta_e"], "delta_E_ci_lo": ci_e[0], "delta_E_ci_hi": ci_e[1], "delta_P": d["delta_p"]}
    raw = pd.read_csv(f)
    if "n_tokens" in raw.columns:
        out["cg_mean_tokens"] = raw["n_tokens"].mean()
    return out


def fit_sequential(cond: str) -> dict:
    f = PRED_DIR / "sequential_anchoring" / f"qwen3-8b__{cond}.csv"
    if not f.exists():
        return {}
    df = pd.read_csv(f)
    stats = analyze_seq_one(df)
    out = {"recovery_rate": stats["recovery_rate"], "recovery_ci_lo": stats["recovery_ci_lo"],
           "recovery_ci_hi": stats["recovery_ci_hi"], "order_effect_at_step2": stats["order_effect_at_step2"],
           "n_anchored_at_step1": stats["n_anchored_at_step1"]}
    if "n_tokens" in df.columns:
        out["seq_mean_tokens"] = df["n_tokens"].mean()
    return out


def main() -> None:
    oracle = pd.read_csv(ROOT / "results" / "theory" / "bayes_oracle_table_expanded.csv")
    vignettes = pd.read_csv(ROOT / "data" / "expanded" / "vignettes_combined.csv")

    rows = []
    for cond in CONDITIONS:
        row = {"condition": cond}
        row.update(fit_baseline(cond, oracle, vignettes))
        row.update(fit_causal_grid(cond))
        row.update(fit_sequential(cond))
        rows.append(row)

    out = pd.DataFrame(rows)
    out_path = ROOT / "results" / "theory" / "qwen3_thinking_budget_analysis.csv"
    out.to_csv(out_path, index=False)

    print("=== Qwen3-8B: accuracy, evidence-elasticity, causal effect, and recovery vs. reasoning budget ===")
    cols = ["condition", "top1_accuracy", "gamma_evidence", "delta_E", "recovery_rate", "mean_tokens",
            "frac_ran_out_of_budget"]
    cols = [c for c in cols if c in out.columns]
    print(out[cols].to_string(index=False))

    if len(out) >= 2 and "top1_accuracy" in out.columns:
        acc_trend = out["top1_accuracy"].diff().dropna()
        ge_trend = out["gamma_evidence"].diff().dropna() if "gamma_evidence" in out.columns else pd.Series()
        print("\n=== Does more budget help accuracy vs. evidence-responsiveness? ===")
        print(f"Accuracy: no_think={out.iloc[0]['top1_accuracy']:.1%} -> "
              f"think_high={out.iloc[-1]['top1_accuracy']:.1%} "
              f"(monotonic increase: {(acc_trend >= -1e-9).all()})")
        if "gamma_evidence" in out.columns:
            print(f"gamma_evidence: no_think={out.iloc[0]['gamma_evidence']:.4f} -> "
                  f"think_high={out.iloc[-1]['gamma_evidence']:.4f} "
                  f"(monotonic increase: {(ge_trend >= -1e-9).all()})")
        if "recovery_rate" in out.columns:
            print(f"recovery_rate: no_think={out.iloc[0]['recovery_rate']} -> "
                  f"think_high={out.iloc[-1]['recovery_rate']}")

    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
