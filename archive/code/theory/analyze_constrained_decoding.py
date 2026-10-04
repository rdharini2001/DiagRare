#!/usr/bin/env python3
""""eliminate free-text parsing as a confound" via
constrained/ID-based decoding (run_constrained_decoding_eval.py). The
HONEST result, reported here rather than suppressed: replacing disease
NAMES with abstract IDs (D01-D82) collapses essentially every model's
accuracy to near the D01-always baseline, regardless of model capability --
this is a genuine, verified (not a parsing/script bug -- see
slurm/debug_guided.sbatch, which confirmed the SAME guided-decoding
mechanism differentiates correctly on a 3-option toy task) finding about
what these models' apparent free-text diagnostic competence actually rests
on: the semantic content of disease NAMES triggering memorized medical
associations, not robust reasoning that survives an abstract symbolic
indirection layer at realistic (82-way) scale.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    vignettes = pd.read_csv(ROOT / "data" / "expanded" / "vignettes_combined.csv")[["vignette_id", "target_disease"]]
    ontology = pd.read_csv(ROOT / "data" / "expanded" / "ontology_diseases_expanded.csv")
    disease_list = sorted(ontology["disease"].astype(str).tolist())
    id_to_name = {f"D{i+1:02d}": d for i, d in enumerate(disease_list)}

    acc_summary = pd.read_csv(ROOT / "results" / "open_weight_summary.csv")

    rows = []
    for f in sorted((ROOT / "results" / "theory" / "constrained").glob("*.csv")):
        model_tag = f.stem
        d = pd.read_csv(f)
        d["pred_name"] = d["prediction_1"].map(id_to_name)
        m = d.merge(vignettes, on="vignette_id")
        constrained_acc = (m.pred_name == m.target_disease).mean()
        n_unique = d["prediction_1"].nunique()
        top_id = d["prediction_1"].value_counts().idxmax()
        top_id_frac = d["prediction_1"].value_counts(normalize=True).max()

        baseline_row = acc_summary[(acc_summary.model == model_tag) & (acc_summary.condition == "baseline")
                                    & (acc_summary.split == "ALL")]
        baseline_acc = baseline_row.iloc[0]["top1_accuracy"] if len(baseline_row) else float("nan")

        rows.append({
            "model": model_tag, "constrained_id_accuracy": constrained_acc,
            "freetext_baseline_accuracy": baseline_acc,
            "n_unique_ids_used": n_unique, "modal_id": top_id,
            "modal_id_name": id_to_name[top_id], "modal_id_fraction": top_id_frac,
        })
        print(f"{model_tag}: constrained_id_acc={constrained_acc:.3f} "
              f"(vs. freetext baseline={baseline_acc:.3f})  "
              f"n_unique_ids_used={n_unique}/82  modal_id={top_id}={id_to_name[top_id]} "
              f"({top_id_frac:.1%} of all vignettes)")

    out = pd.DataFrame(rows)
    out_path = ROOT / "results" / "theory" / "constrained_decoding_analysis.csv"
    out.to_csv(out_path, index=False)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
