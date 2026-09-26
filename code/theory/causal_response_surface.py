#!/usr/bin/env python3
"""Estimate response surfaces for the randomized evidence and stated-prior experiment.

The analysis reports target-choice probabilities across the 3 by 3 design and tests
whether the baseline rank coefficient predicts the finite-difference response to
assigned evidence.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[2]
PRED_DIR = ROOT / "results" / "theory" / "causal_grid"
N_BOOT = 5000
RNG_SEED = 20260920


def load_model_grid(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df["choice"] != "unparsed"].copy()
    df["Y"] = (df["choice"] == "target").astype(int)
    df["pair_id"] = df["target_disease"] + "::" + df["confounder_disease"]
    return df


def pair_level_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (pair, evidence_level, prior_level) -- Y is already
    binary per pair since each pair has exactly one cell per (e,p)."""
    return df.pivot_table(index="pair_id", columns=["evidence_level", "prior_level"], values="Y")


def compute_deltas(wide: pd.DataFrame) -> dict:
    """wide: pairs x (evidence_level, prior_level) -> Y in {0,1,nan}."""
    priors = ["target_rare", "equal", "target_common"]
    evs = ["weak", "medium", "strong"]

    def cell(e, p):
        return wide[(e, p)] if (e, p) in wide.columns else pd.Series(np.nan, index=wide.index)

    delta_e_by_pair = pd.concat([cell("strong", p) - cell("weak", p) for p in priors], axis=1).mean(axis=1)
    delta_p_by_pair = pd.concat([cell(e, "target_common") - cell(e, "target_rare") for e in evs], axis=1).mean(axis=1)
    delta_e_rare_by_pair = cell("strong", "target_rare") - cell("weak", "target_rare")
    delta_e_common_by_pair = cell("strong", "target_common") - cell("weak", "target_common")
    interaction_by_pair = delta_e_common_by_pair - delta_e_rare_by_pair
    prior_override_by_pair = cell("strong", "target_rare") - cell("weak", "target_rare")

    surface = {(e, p): cell(e, p).mean() for e in evs for p in priors}

    return {
        "delta_e": delta_e_by_pair.mean(), "delta_e_by_pair": delta_e_by_pair,
        "delta_p": delta_p_by_pair.mean(), "delta_p_by_pair": delta_p_by_pair,
        "interaction": interaction_by_pair.mean(), "interaction_by_pair": interaction_by_pair,
        "prior_override": prior_override_by_pair.mean(), "prior_override_by_pair": prior_override_by_pair,
        "surface": surface,
    }


def bootstrap_ci(by_pair: pd.Series, n_boot: int = N_BOOT, seed: int = RNG_SEED) -> tuple[float, float]:
    by_pair = by_pair.dropna()
    if len(by_pair) < 3:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    pairs = by_pair.index.to_numpy()
    vals = by_pair.to_numpy()
    boot_means = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, len(vals), size=len(vals))
        boot_means[i] = vals[idx].mean()
    return (np.percentile(boot_means, 2.5), np.percentile(boot_means, 97.5))


def main() -> None:
    out_dir = ROOT / "results" / "theory"
    rows = []
    surfaces = []
    for f in sorted(PRED_DIR.glob("*.csv")):
        model_tag = f.stem
        df = load_model_grid(f)
        if df["pair_id"].nunique() < 5:
            continue
        wide = pair_level_table(df)
        d = compute_deltas(wide)
        ci_e = bootstrap_ci(d["delta_e_by_pair"])
        ci_p = bootstrap_ci(d["delta_p_by_pair"])
        ci_int = bootstrap_ci(d["interaction_by_pair"])
        ci_override = bootstrap_ci(d["prior_override_by_pair"])
        rows.append({
            "model": model_tag, "n_pairs": wide.shape[0],
            "delta_E": d["delta_e"], "delta_E_ci_lo": ci_e[0], "delta_E_ci_hi": ci_e[1],
            "delta_P": d["delta_p"], "delta_P_ci_lo": ci_p[0], "delta_P_ci_hi": ci_p[1],
            "interaction_common_minus_rare": d["interaction"],
            "interaction_ci_lo": ci_int[0], "interaction_ci_hi": ci_int[1],
            "prior_override_at_target_rare": d["prior_override"],
            "prior_override_ci_lo": ci_override[0], "prior_override_ci_hi": ci_override[1],
        })
        for (e, p), val in d["surface"].items():
            surfaces.append({"model": model_tag, "evidence_level": e, "prior_level": p, "Pr_target": val})

    out = pd.DataFrame(rows)
    surf = pd.DataFrame(surfaces)
    out_path = out_dir / "causal_response_surface.csv"
    surf_path = out_dir / "causal_response_surface_cells.csv"
    out.to_csv(out_path, index=False)
    surf.to_csv(surf_path, index=False)

    print("=== Causal response surface: per-model ATEs (bootstrap 95% CI over 27 disease pairs) ===")
    print(out.sort_values("delta_E", ascending=False).to_string(index=False))

    # --- headline validation: does baseline gamma_evidence/gamma_prior
    # predict the SAME model's measured causal response? ---
    per_master = pd.read_csv(out_dir / "per_master_table.csv")
    base = per_master[(per_master.source == "open_weight") & (per_master.condition == "baseline")]
    merged = out.merge(base[["model", "gamma_prior", "gamma_evidence"]], on="model", how="inner").dropna(
        subset=["gamma_evidence", "delta_E"])

    print(f"\n=== VALIDATION: rank-derived coefficient vs. measured causal response (n={len(merged)}) ===")
    if len(merged) > 2:
        r_e, p_e = pearsonr(merged.gamma_evidence, merged.delta_E)
        rho_e, _ = spearmanr(merged.gamma_evidence, merged.delta_E)
        print(f"gamma_evidence vs. Delta_E (causal evidence effect): Pearson r={r_e:.3f} (p={p_e:.4g}), "
              f"Spearman rho={rho_e:.3f}")
        merged_p = merged.dropna(subset=["gamma_prior", "delta_P"])
        if len(merged_p) > 2:
            r_p, p_p = pearsonr(merged_p.gamma_prior, merged_p.delta_P)
            rho_p, _ = spearmanr(merged_p.gamma_prior, merged_p.delta_P)
            print(f"gamma_prior vs. Delta_P (causal prior effect): Pearson r={r_p:.3f} (p={p_p:.4g}), "
                  f"Spearman rho={rho_p:.3f}")
        merged.to_csv(out_dir / "causal_response_validation.csv", index=False)

    print(f"\nWrote {out_path}, {surf_path}")


if __name__ == "__main__":
    main()
