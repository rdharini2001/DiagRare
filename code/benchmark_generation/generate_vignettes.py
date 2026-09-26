#!/usr/bin/env python3
"""Build the expanded DiagRare ontology + vignette set.

Extends the original 58-disease / 696-vignette benchmark with 24 new
diseases:
  - 6 "expanded_indist" diseases in the original 6 organ systems
    (usable for fine-tuning and evaluation)
  - 18 "expanded_heldout" diseases in 3 NEW organ systems (Renal, GI,
    Dermato) never seen in the original benchmark, held out from
    fine-tuning and used only to test generalization.

Vignette generation logic mirrors the observed structure of the
original vignettes.csv (12 vignettes/disease, difficulty tiers evenly
split, positive findings = subset of required + supporting findings,
negative findings = subset of this disease's excluding findings plus,
for harder tiers, one supporting finding borrowed from the nearest
common confounder disease in the same organ system, to create genuine
prevalence-vs-logic conflicts).
"""
from __future__ import annotations

import random
from pathlib import Path

import pandas as pd

from new_diseases import NEW_DISEASES

ROOT = Path(__file__).resolve().parents[2]
ORIG = ROOT / "data" / "original"
OUT = ROOT / "data" / "expanded"
OUT.mkdir(parents=True, exist_ok=True)

RNG = random.Random(20260908)

DIFFICULTIES = ["easy", "medium", "hard"]
N_PER_DISEASE = 12  # matches original benchmark density


def split_tokens(s: str) -> list[str]:
    return [t.strip() for t in s.split(";") if t.strip()]


def build_expanded_ontology() -> pd.DataFrame:
    orig = pd.read_csv(ORIG / "ontology_diseases.csv")
    orig["split"] = "original"
    orig = orig.rename(columns={
        "required_findings": "required", "supporting_findings": "supporting",
        "excluding_findings": "excluding",
    })

    new_rows = []
    for d in NEW_DISEASES:
        new_rows.append({
            "disease": d["disease"], "organ_system": d["organ_system"],
            "prevalence": d["prevalence"], "is_rare": d["is_rare"],
            "required": d["required"], "supporting": d["supporting"], "excluding": d["excluding"],
            "n_required": d["n_required"], "n_supporting": d["n_supporting"], "n_excluding": d["n_excluding"],
            "split": d["split"],
        })
    new_df = pd.DataFrame(new_rows)

    combined = pd.concat([orig, new_df], ignore_index=True)
    if combined["disease"].duplicated().any():
        dupes = combined.loc[combined["disease"].duplicated(), "disease"].tolist()
        raise ValueError(f"Duplicate disease names introduced: {dupes}")
    return combined


def nearest_confounder(disease_row, ontology: pd.DataFrame) -> pd.Series | None:
    """Pick another disease in the same organ system with the highest prevalence
    (the 'obvious' prevalence-driven alternative diagnosis) to source a
    plausible-but-wrong supporting finding as a distractor for hard vignettes."""
    same_system = ontology[
        (ontology["organ_system"] == disease_row["organ_system"])
        & (ontology["disease"] != disease_row["disease"])
    ]
    if same_system.empty:
        return None
    return same_system.sort_values("prevalence", ascending=False).iloc[0]


def generate_vignettes_for_disease(disease_row: pd.Series, ontology: pd.DataFrame, start_id: int) -> list[dict]:
    required = split_tokens(disease_row["required"])
    supporting = split_tokens(disease_row["supporting"])
    excluding = split_tokens(disease_row["excluding"])
    confounder = nearest_confounder(disease_row, ontology)
    confounder_finding = None
    if confounder is not None:
        conf_supporting = split_tokens(confounder["supporting"])
        if conf_supporting:
            confounder_finding = RNG.choice(conf_supporting)

    rows = []
    n_total_required = len(required)
    for i in range(N_PER_DISEASE):
        difficulty = DIFFICULTIES[i % 3]
        vid = start_id + i

        if difficulty == "easy":
            n_req_given = n_total_required
            n_supp = RNG.randint(min(2, len(supporting)), len(supporting)) if supporting else 0
            n_neg = RNG.randint(2, max(2, len(excluding)))
            use_confounder = False
        elif difficulty == "medium":
            n_req_given = max(2, n_total_required - 1) if n_total_required > 2 else n_total_required
            n_supp = RNG.randint(1, max(1, len(supporting) - 1)) if supporting else 0
            n_neg = RNG.randint(1, max(1, len(excluding) - 1)) if excluding else 0
            use_confounder = RNG.random() < 0.5
        else:  # hard
            n_req_given = max(2, n_total_required - 2)
            n_supp = RNG.randint(0, 1) if supporting else 0
            n_neg = RNG.randint(0, 1) if excluding else 0
            use_confounder = True

        pos_required = RNG.sample(required, k=min(n_req_given, len(required)))
        pos_supporting = RNG.sample(supporting, k=min(n_supp, len(supporting))) if supporting else []
        positive = pos_required + pos_supporting
        if use_confounder and confounder_finding and confounder_finding not in positive:
            positive.append(confounder_finding)
        RNG.shuffle(positive)

        negative = RNG.sample(excluding, k=min(n_neg, len(excluding))) if excluding else []

        rows.append({
            "vignette_id": vid,
            "target_disease": disease_row["disease"],
            "organ_system": disease_row["organ_system"],
            "is_rare": bool(disease_row["is_rare"]),
            "prevalence": disease_row["prevalence"],
            "difficulty": difficulty,
            "positive_findings": ";".join(positive),
            "negative_findings": ";".join(negative),
            "n_positive": len(positive),
            "n_negative": len(negative),
            "n_required_given": len(pos_required),
            "n_required_total": n_total_required,
            "split": disease_row["split"],
            "used_confounder_distractor": use_confounder and confounder_finding is not None,
        })
    return rows


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20260908,
                     help="RNG seed for vignette sampling -- used to check that PER conclusions "
                          "aren't an artifact of one particular random benchmark instantiation")
    ap.add_argument("--out_dir", type=str, default=str(OUT))
    args = ap.parse_args()

    global RNG
    RNG = random.Random(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ontology = build_expanded_ontology()
    ontology.to_csv(out_dir / "ontology_diseases_expanded.csv", index=False)

    new_diseases = ontology[ontology["split"] != "original"]
    all_rows = []
    next_id = 1001
    for _, drow in new_diseases.iterrows():
        vs = generate_vignettes_for_disease(drow, ontology, next_id)
        all_rows.extend(vs)
        next_id += N_PER_DISEASE

    new_vignettes = pd.DataFrame(all_rows)
    new_vignettes.to_csv(out_dir / "vignettes_new.csv", index=False)

    orig_vignettes = pd.read_csv(ORIG / "vignettes.csv")
    orig_vignettes["split"] = "original"
    orig_vignettes["used_confounder_distractor"] = False
    combined_vignettes = pd.concat([orig_vignettes, new_vignettes], ignore_index=True)
    combined_vignettes.to_csv(out_dir / "vignettes_combined.csv", index=False)

    print("Expanded ontology:", len(ontology), "diseases "
          f"({(ontology['split']=='original').sum()} original, "
          f"{(ontology['split']=='expanded_indist').sum()} expanded_indist, "
          f"{(ontology['split']=='expanded_heldout').sum()} expanded_heldout)")
    print("New vignettes generated:", len(new_vignettes))
    print("Combined vignette set:", len(combined_vignettes))
    print("\nBy split:")
    print(combined_vignettes["split"].value_counts())
    print("\nBy difficulty (new only):")
    print(new_vignettes["difficulty"].value_counts())
    print("\nOrgan systems in expanded ontology:")
    print(ontology.groupby(["organ_system", "split"]).size())


if __name__ == "__main__":
    main()
