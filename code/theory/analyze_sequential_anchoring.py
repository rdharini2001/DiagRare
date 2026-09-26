#!/usr/bin/env python3
"""Analyzes the sequential anchoring/recovery experiment
(run_sequential_anchoring_inference.py). Per model:

  - order_effect_at_step2 = Pr(target | confounder_first, step2)
                           - Pr(target | target_first, step2)
    Step 2's total evidence is IDENTICAL between orders by construction, so
    a nonzero order_effect is diagnostic anchoring / path dependence, not a
    difference in what evidence the model has seen.

  - recovery_rate = among confounder_first pairs where Step 1 chose the
    confounder (the model started "anchored" on the wrong diagnosis),
    the fraction that switch to the target once the decisive follow-up
    findings arrive at Step 2.

  - anchoring_rate (the mirror-image failure mode) = among confounder_first
    pairs where Step 1 chose target_first... no -- among confounder_first
    pairs where Step 2 STILL fails to choose the target despite it now
    being fully evidenced, i.e. 1 - recovery_rate restricted to the
    initially-anchored subset. Reported directly for clarity.

  - Bootstrap 95% CIs over the 82 disease pairs (the independent unit).

VALIDATION: does the baseline gamma_evidence predict
recovery_rate -- i.e., does a model's ordinary, non-interactive
evidence-following behavior predict whether it can revise an initial
misdiagnosis once decisive contradictory evidence arrives in a SEQUENTIAL,
interactive setting? This connects the static coefficient to a genuinely
interactive clinical behavior.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[2]
PRED_DIR = ROOT / "results" / "theory" / "sequential_anchoring"
N_BOOT = 5000
RNG_SEED = 20260920


def bootstrap_ci(vals: np.ndarray, n_boot: int = N_BOOT, seed: int = RNG_SEED) -> tuple[float, float]:
    vals = vals[~np.isnan(vals)]
    if len(vals) < 3:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    boot = np.array([rng.choice(vals, size=len(vals), replace=True).mean() for _ in range(n_boot)])
    return (np.percentile(boot, 2.5), np.percentile(boot, 97.5))


def analyze_one(df: pd.DataFrame) -> dict:
    df = df[df["choice"] != "unparsed"].copy()
    df["is_target"] = (df["choice"] == "target").astype(int)

    step2 = df[df.step == 2]
    cf2 = step2[step2.order == "confounder_first"].set_index("pair_id")["is_target"]
    tf2 = step2[step2.order == "target_first"].set_index("pair_id")["is_target"]
    common_pairs = cf2.index.intersection(tf2.index)
    order_effect_by_pair = cf2.loc[common_pairs] - tf2.loc[common_pairs]

    cf1 = df[(df.order == "confounder_first") & (df.step == 1)].set_index("pair_id")["is_target"]
    cf2_full = df[(df.order == "confounder_first") & (df.step == 2)].set_index("pair_id")["is_target"]
    anchored_pairs = cf1[cf1 == 0].index  # step1 chose confounder (anchored)
    recovery_by_pair = cf2_full.loc[cf2_full.index.intersection(anchored_pairs)]

    ci_order = bootstrap_ci(order_effect_by_pair.to_numpy())
    ci_recovery = bootstrap_ci(recovery_by_pair.to_numpy())

    return {
        "n_pairs_total": df.pair_id.nunique(),
        "order_effect_at_step2": order_effect_by_pair.mean(),
        "order_effect_ci_lo": ci_order[0], "order_effect_ci_hi": ci_order[1],
        "n_anchored_at_step1": len(anchored_pairs),
        "recovery_rate": recovery_by_pair.mean() if len(recovery_by_pair) else np.nan,
        "recovery_ci_lo": ci_recovery[0], "recovery_ci_hi": ci_recovery[1],
        "n_used_for_recovery": len(recovery_by_pair),
    }


def main() -> None:
    out_dir = ROOT / "results" / "theory"
    rows = []
    for f in sorted(PRED_DIR.glob("*.csv")):
        model_tag = f.stem
        df = pd.read_csv(f)
        stats = analyze_one(df)
        stats["model"] = model_tag
        rows.append(stats)

    out = pd.DataFrame(rows)
    out_path = out_dir / "sequential_anchoring_analysis.csv"
    out.to_csv(out_path, index=False)

    print("=== Sequential anchoring/recovery: per-model results ===")
    cols = ["model", "order_effect_at_step2", "order_effect_ci_lo", "order_effect_ci_hi",
            "recovery_rate", "recovery_ci_lo", "recovery_ci_hi", "n_anchored_at_step1"]
    print(out[cols].sort_values("recovery_rate", ascending=False).to_string(index=False))

    per_master = pd.read_csv(out_dir / "per_master_table.csv")
    base = per_master[(per_master.source == "open_weight") & (per_master.condition == "baseline")]
    merged = out.merge(base[["model", "gamma_evidence"]], on="model", how="inner").dropna(
        subset=["gamma_evidence", "recovery_rate"])
    merged = merged[merged["n_used_for_recovery"] >= 5]

    print(f"\n=== VALIDATION: gamma_evidence vs. recovery_rate (n={len(merged)}, "
          f"restricted to models with >=5 anchored pairs) ===")
    if len(merged) > 2:
        r, p = pearsonr(merged.gamma_evidence, merged.recovery_rate)
        rho, _ = spearmanr(merged.gamma_evidence, merged.recovery_rate)
        print(f"Pearson r={r:.3f} (p={p:.4g}), Spearman rho={rho:.3f}")
        merged.to_csv(out_dir / "sequential_anchoring_validation.csv", index=False)

    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
