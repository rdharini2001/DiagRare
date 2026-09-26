#!/usr/bin/env python3
"""Analyze sensitivity to clinically irrelevant text.

The negative control measures how often an unrelated sentence changes the top-ranked
diagnosis and compares that rate with baseline evidence responsiveness.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    per_master = pd.read_csv(ROOT / "results" / "theory" / "per_master_table.csv")
    negctrl_dir = ROOT / "results" / "predictions_negcontrol"

    rows = []
    for f in sorted(negctrl_dir.glob("*__baseline.csv")):
        model_tag = f.stem.replace("__baseline", "")
        orig_path = ROOT / "results" / "predictions" / f"{model_tag}__baseline.csv"
        if not orig_path.exists():
            print(f"  SKIP {model_tag}: no original baseline predictions found")
            continue
        orig = pd.read_csv(orig_path)[["vignette_id", "prediction_1"]].rename(
            columns={"prediction_1": "prediction_1_orig"})
        negctrl = pd.read_csv(f)[["vignette_id", "prediction_1"]].rename(
            columns={"prediction_1": "prediction_1_negctrl"})
        m = orig.merge(negctrl, on="vignette_id", how="inner")
        flip_rate = (m["prediction_1_orig"] != m["prediction_1_negctrl"]).mean()

        ge_row = per_master[(per_master.source == "open_weight") & (per_master.model == model_tag)
                             & (per_master.condition == "baseline")]
        row = {"model": model_tag, "n": len(m), "flip_rate": flip_rate}
        if len(ge_row):
            row["gamma_evidence"] = ge_row.iloc[0]["gamma_evidence"]
        rows.append(row)
        print(f"{model_tag}: n={len(m)} flip_rate={flip_rate:.1%} (answer changed due to an "
              f"irrelevant, diagnostically-null detail)")

    out = pd.DataFrame(rows)
    out_path = ROOT / "results" / "theory" / "robustness" / "negative_control_analysis.csv"
    out.to_csv(out_path, index=False)

    valid = out.dropna(subset=["gamma_evidence", "flip_rate"])
    if len(valid) > 2:
        r, p = pearsonr(valid.gamma_evidence, valid.flip_rate)
        rho, _ = spearmanr(valid.gamma_evidence, valid.flip_rate)
        print(f"\ngamma_evidence vs. negative-control flip rate: r={r:.3f} (p={p:.4f}), rho={rho:.3f}")
        print("Interpretation: " + (
            "flip rate is essentially uncorrelated with (or negatively correlated with) "
            "gamma_evidence -- evidence-elastic models are selectively responsive to REAL "
            "evidence (Sec 6.2), not simply more reactive to any text change (this check)."
            if r <= 0.3 else
            "flip rate correlates positively with gamma_evidence; this is treated as a "
            "genuine caveat: some of what looks like evidence-elasticity may include general "
            "sensitivity to added text, not only to diagnostically relevant content."
        ))
    print(f"\nMean flip rate across models: {out['flip_rate'].mean():.1%}")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
