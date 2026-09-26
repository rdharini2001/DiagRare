#!/usr/bin/env python3
"""Construct a negative-control set by adding clinically irrelevant information."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

IRRELEVANT_DETAILS = [
    "patient_works_as_accountant", "patient_arrived_by_personal_vehicle",
    "patient_owns_a_pet_cat", "patient_has_private_PPO_insurance",
    "patient_prefers_tea_over_coffee", "patient_is_married",
    "patient_lives_in_a_two_story_house", "patient_blood_type_O_positive",
    "patient_has_brown_hair", "patient_shoe_size_9",
    "patient_commutes_by_bicycle", "patient_is_right_handed",
]


def verify_no_overlap(ontology: pd.DataFrame) -> None:
    all_findings = set()
    for col in ("required", "supporting", "excluding"):
        for cell in ontology[col].dropna():
            all_findings.update(str(cell).split(";"))
    overlap = set(IRRELEVANT_DETAILS) & all_findings
    if overlap:
        raise ValueError(f"Irrelevant details overlap with real ontology findings: {overlap}")
    print(f"Verified: none of {len(IRRELEVANT_DETAILS)} irrelevant details appear among "
          f"{len(all_findings)} real ontology finding tokens.")


def main() -> None:
    vignettes = pd.read_csv(ROOT / "data" / "expanded" / "vignettes_combined.csv")
    ontology = pd.read_csv(ROOT / "data" / "expanded" / "ontology_diseases_expanded.csv")
    verify_no_overlap(ontology)

    rng = np.random.default_rng(20260914)
    out = vignettes.copy()
    assigned = rng.choice(IRRELEVANT_DETAILS, size=len(out))
    out["positive_findings"] = out.apply(
        lambda r: (str(r["positive_findings"]) + ";" + assigned[r.name])
        if pd.notna(r["positive_findings"]) else assigned[r.name], axis=1)
    out["injected_irrelevant_detail"] = assigned

    out_path = ROOT / "data" / "expanded" / "vignettes_negative_control.csv"
    out.to_csv(out_path, index=False)
    print(f"Wrote {out_path} ({len(out)} vignettes, one irrelevant detail injected each, "
          f"{len(set(assigned))} distinct detail types used)")


if __name__ == "__main__":
    main()
