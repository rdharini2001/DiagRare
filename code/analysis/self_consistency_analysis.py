#!/usr/bin/env python3
"""Does self-consistency (k=5 sampling + majority vote) improve accuracy by
increasing evidence-elasticity (gamma_evidence), or just by averaging out
noise around the same underlying gamma's (i.e. a variance reduction with no
change in the model's revealed prior/evidence weighting)? Pulls the
already-fit PER rows for the 3 self-consistency runs and their matched
single-sample baseline counterparts out of results/theory/per_master_table.csv
and results/open_weight_summary.csv / results/open_weight_selfconsistency_summary.csv,
and tabulates the comparison -- no new fitting, this is pure bookkeeping over
already-computed results.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    per = pd.read_csv(ROOT / "results" / "theory" / "per_master_table.csv")
    acc_single = pd.read_csv(ROOT / "results" / "open_weight_summary.csv")
    acc_sc = pd.read_csv(ROOT / "results" / "open_weight_selfconsistency_summary.csv")

    models = ["qwen2.5-0.5b", "qwen2.5-3b", "qwen2.5-7b"]
    rows = []
    for m in models:
        single_per = per[(per.source == "open_weight") & (per.model == m) & (per.condition == "baseline")]
        sc_per = per[(per.source == "self_consistency") & (per.model == f"{m}-selfcons5")]
        single_acc = acc_single[(acc_single.model == m) & (acc_single.condition == "baseline")
                                 & (acc_single.split == "ALL")]
        sc_acc = acc_sc[(acc_sc.model == f"{m}-selfcons5") & (acc_sc.split == "ALL")]

        if len(single_per) == 0 or len(sc_per) == 0:
            continue
        sp, scp = single_per.iloc[0], sc_per.iloc[0]
        sa = single_acc.iloc[0] if len(single_acc) else None
        sca = sc_acc.iloc[0] if len(sc_acc) else None

        rows.append({
            "model": m,
            "top1_acc_single": sa["top1_accuracy"] if sa is not None else None,
            "top1_acc_selfcons5": sca["top1_accuracy"] if sca is not None else None,
            "gamma_prior_single": sp["gamma_prior"], "gamma_prior_selfcons5": scp["gamma_prior"],
            "gamma_evidence_single": sp["gamma_evidence"], "gamma_evidence_selfcons5": scp["gamma_evidence"],
            "PER_single": sp["PER"], "PER_selfcons5": scp["PER"],
            "gamma_evidence_ci_width_single": sp["gamma_evidence_hi"] - sp["gamma_evidence_lo"],
            "gamma_evidence_ci_width_selfcons5": scp["gamma_evidence_hi"] - scp["gamma_evidence_lo"],
        })

    out = pd.DataFrame(rows)
    out_path = ROOT / "results" / "theory" / "self_consistency_per_comparison.csv"
    out.to_csv(out_path, index=False)

    pd.set_option("display.width", 200)
    print("Self-consistency (k=5 vote) vs. single-sample: does it shift gamma_evidence or just reduce CI width?")
    print(out.to_string(index=False))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
