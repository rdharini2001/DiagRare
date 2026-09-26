#!/usr/bin/env python3
"""Compare rare and common targets after matching case characteristics."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "analysis"))
from plackett_luce import build_choices_from_predictions, fit_plackett_luce  # noqa: E402


def build_matched_pairs(vignettes: pd.DataFrame, oracle: pd.DataFrame) -> pd.DataFrame:
    ev_by_vid_disease = oracle.set_index(["vignette_id", "disease"])["evidence_loglik_ratio"]
    v = vignettes.copy()
    v["total_findings"] = v["n_positive"] + v["n_negative"]

    def target_evidence(row):
        key = (row["vignette_id"], row["target_disease"])
        return ev_by_vid_disease.get(key, np.nan)

    v["target_evidence_strength"] = v.apply(target_evidence, axis=1)
    v = v.dropna(subset=["target_evidence_strength"])

    pairs = []
    for (organ, diff), grp in v.groupby(["organ_system", "difficulty"]):
        rare_pool = grp[grp.is_rare].copy()
        common_pool = grp[~grp.is_rare].copy().reset_index(drop=True)
        used = set()
        for _, rrow in rare_pool.iterrows():
            if len(used) >= len(common_pool):
                break
            candidates = common_pool[~common_pool.index.isin(used)]
            if candidates.empty:
                break
            dist = (
                (candidates["total_findings"] - rrow["total_findings"]).abs() * 2.0
                + (candidates["target_evidence_strength"] - rrow["target_evidence_strength"]).abs()
            )
            best_idx = dist.idxmin()
            crow = candidates.loc[best_idx]
            used.add(best_idx)
            pairs.append({
                "organ_system": organ, "difficulty": diff,
                "rare_vignette_id": rrow["vignette_id"], "rare_disease": rrow["target_disease"],
                "rare_findings": rrow["total_findings"], "rare_evidence": rrow["target_evidence_strength"],
                "common_vignette_id": crow["vignette_id"], "common_disease": crow["target_disease"],
                "common_findings": crow["total_findings"], "common_evidence": crow["target_evidence_strength"],
            })
    return pd.DataFrame(pairs)


def analyze_model(model_tag: str, pairs: pd.DataFrame, oracle: pd.DataFrame) -> dict:
    preds = pd.read_csv(ROOT / "results" / "predictions" / f"{model_tag}__baseline.csv")
    pred_by_vid = preds.set_index("vignette_id")

    rare_correct, common_correct = [], []
    for _, r in pairs.iterrows():
        if r["rare_vignette_id"] not in pred_by_vid.index or r["common_vignette_id"] not in pred_by_vid.index:
            continue
        rare_correct.append(pred_by_vid.loc[r["rare_vignette_id"], "prediction_1"] == r["rare_disease"])
        common_correct.append(pred_by_vid.loc[r["common_vignette_id"], "prediction_1"] == r["common_disease"])

    rare_vids = set(pairs["rare_vignette_id"])
    common_vids = set(pairs["common_vignette_id"])
    preds_rare = preds[preds.vignette_id.isin(rare_vids)]
    preds_common = preds[preds.vignette_id.isin(common_vids)]

    ge = {}
    for label, sub in [("rare", preds_rare), ("common", preds_common)]:
        choices, _, n_used = build_choices_from_predictions(sub, oracle)
        fit = fit_plackett_luce(choices)
        ge[f"gamma_evidence_{label}"] = fit["gamma_evidence"]
        ge[f"gamma_prior_{label}"] = fit["gamma_prior"]
        ge[f"n_{label}"] = n_used

    diffs = np.array(common_correct, dtype=float) - np.array(rare_correct, dtype=float)
    try:
        stat, p = wilcoxon(diffs)
    except ValueError:
        p = np.nan  # all-zero differences

    return {
        "model": model_tag,
        "n_pairs": len(pairs),
        "n_pairs_scored": len(diffs),
        "rare_accuracy": np.mean(rare_correct), "common_accuracy": np.mean(common_correct),
        "accuracy_gap": np.mean(common_correct) - np.mean(rare_correct),
        "wilcoxon_p": p,
        **ge,
        "mean_rare_findings": pairs["rare_findings"].mean(), "mean_common_findings": pairs["common_findings"].mean(),
        "mean_rare_evidence": pairs["rare_evidence"].mean(), "mean_common_evidence": pairs["common_evidence"].mean(),
    }


def main() -> None:
    vignettes = pd.read_csv(ROOT / "data" / "expanded" / "vignettes_combined.csv")
    oracle = pd.read_csv(ROOT / "results" / "theory" / "bayes_oracle_table_expanded.csv")

    pairs = build_matched_pairs(vignettes, oracle)
    out_dir = ROOT / "results" / "theory" / "robustness"
    out_dir.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(out_dir / "rare_common_matched_pairs.csv", index=False)
    print(f"Built {len(pairs)} matched rare/common vignette pairs "
          f"(within organ_system x difficulty, nearest-neighbor on finding-count + evidence-strength)")
    print(f"Balance check -- mean finding count: rare={pairs.rare_findings.mean():.2f} "
          f"common={pairs.common_findings.mean():.2f}  "
          f"mean evidence strength: rare={pairs.rare_evidence.mean():.3f} common={pairs.common_evidence.mean():.3f}")

    models = ["qwen2.5-0.5b", "qwen2.5-1.5b", "qwen2.5-3b", "qwen2.5-7b",
              "yi-1.5-6b", "olmo-2-7b", "mistral-7b-instruct", "phi-3.5-mini"]
    rows = []
    for m in models:
        f = ROOT / "results" / "predictions" / f"{m}__baseline.csv"
        if not f.exists():
            continue
        res = analyze_model(m, pairs, oracle)
        rows.append(res)
        print(f"{m}: rare_acc={res['rare_accuracy']:.3f} common_acc={res['common_accuracy']:.3f} "
              f"gap={res['accuracy_gap']:+.3f}  gamma_evidence rare={res['gamma_evidence_rare']:.3f} "
              f"common={res['gamma_evidence_common']:.3f}")

    out = pd.DataFrame(rows)
    out.to_csv(out_dir / "rare_common_matched_summary.csv", index=False)

    print(f"\nMean accuracy gap (common - rare) across models, holding organ/difficulty/findings/evidence fixed: "
          f"{out['accuracy_gap'].mean():+.3f} (range {out['accuracy_gap'].min():+.3f} to {out['accuracy_gap'].max():+.3f})")
    print(f"Wrote {out_dir / 'rare_common_matched_pairs.csv'} and {out_dir / 'rare_common_matched_summary.csv'}")


if __name__ == "__main__":
    main()
